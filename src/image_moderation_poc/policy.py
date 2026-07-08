from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass

from image_moderation_poc.schemas import EvidenceMatch


@dataclass(frozen=True)
class PolicyTerm:
    family: str
    term: str
    weight: float = 1.0
    min_score: float = 0.88
    strong_signal: bool = True


POLICY_TERMS: tuple[PolicyTerm, ...] = (
    PolicyTerm("campaign_event", "hot sale", 1.0),
    PolicyTerm("campaign_event", "black friday", 1.0),
    PolicyTerm("campaign_event", "black days", 1.0),
    PolicyTerm("campaign_event", "ofertas relampago", 0.95),
    PolicyTerm("shipping_promise", "envio inmediato", 1.0),
    PolicyTerm("shipping_promise", "entrega inmediata", 1.0),
    PolicyTerm("shipping_promise", "inmediata", 0.88, 0.92),
    PolicyTerm("shipping_promise", "al instante", 0.95),
    PolicyTerm("shipping_promise", "descarga inmediata", 0.95),
    PolicyTerm("shipping_promise", "pronta entrega", 0.95),
    PolicyTerm("shipping_promise", "mismo dia", 1.0),
    PolicyTerm("shipping_promise", "llega hoy", 1.0),
    PolicyTerm("shipping_promise", "llega manana", 0.95),
    PolicyTerm("price_promotion", "gratis", 0.90),
    PolicyTerm("price_promotion", "envio gratis", 1.0),
    PolicyTerm("price_promotion", "oferta", 0.90),
    PolicyTerm("price_promotion", "descuento", 0.95),
    PolicyTerm("price_promotion", "desconto", 0.90),
    PolicyTerm("price_promotion", "descontos", 0.90),
    PolicyTerm("price_promotion", "super descontos", 0.95),
    PolicyTerm("price_promotion", "promocion", 0.95),
    PolicyTerm("price_promotion", "12 cuotas fijas", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "recomendado", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "mas vendido", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "#1 en ventas", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "mercado lider", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "official store", 1.0, 0.80),
    PolicyTerm("marketplace_badge_social_proof", "oficial store", 1.0, 0.80),
    PolicyTerm("marketplace_badge_social_proof", "ofiatal sore", 0.95, 0.80),
    PolicyTerm("marketplace_badge_social_proof", "officialstone", 0.95, 0.80),
    PolicyTerm("marketplace_badge_social_proof", "amazon exclusive", 0.95, 0.82),
    PolicyTerm("marketplace_badge_social_proof", "best seller", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "bestseller", 0.95),
    PolicyTerm("marketplace_badge_social_proof", "new york times bestseller", 0.95),
    PolicyTerm("trust_payment_platform_claim", "compra segura", 0.95),
    PolicyTerm("trust_payment_platform_claim", "mercado pago", 0.95),
    PolicyTerm("trust_payment_platform_claim", "mercado envios", 0.95),
    PolicyTerm("trust_payment_platform_claim", "metodos de pago", 0.90),
    PolicyTerm("trust_payment_platform_claim", "entregamos factura", 0.90),
    PolicyTerm("quality_originality_claim", "mejor calidad", 0.85),
    PolicyTerm("quality_originality_claim", "nuevas y originales", 0.90),
)

WEAK_POLICY_CUE_RE = re.compile(
    r"sale|descont|ofert|gratis|promoc|mercado|amazon|exclusive|official|ofi|best|seller|"
    r"inmedi|black|hot|destacad|envio|envios|calidad|recomend|limitado|tempo",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[_\-]+", " ", ascii_text)
    ascii_text = re.sub(r"[^a-z0-9#+% ]+", " ", ascii_text)
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    return ascii_text


def weak_policy_cues(text: str) -> set[str]:
    return set(WEAK_POLICY_CUE_RE.findall(normalize_text(text)))


def _windows(tokens: list[str], size: int) -> list[str]:
    if size <= 0 or len(tokens) < size:
        return []
    return [" ".join(tokens[i : i + size]) for i in range(0, len(tokens) - size + 1)]


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_policy_matches(
    text: str,
    fuzzy_threshold: float = 0.88,
    max_fuzzy_tokens: int = 80,
) -> list[EvidenceMatch]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    tokens = normalized.split()
    matches: list[EvidenceMatch] = []
    seen: set[tuple[str, str, str]] = set()

    for policy in POLICY_TERMS:
        term = normalize_text(policy.term)
        if not term:
            continue

        if term in normalized:
            score = 1.0 * policy.weight
            key = (policy.family, policy.term, term)
            if key not in seen:
                matches.append(
                    EvidenceMatch(policy.family, policy.term, term, min(score, 1.0), "substring")
                )
                seen.add(key)
            continue

        if len(tokens) > max_fuzzy_tokens:
            continue

        term_tokens = term.split()
        sizes = {len(term_tokens)}
        if len(term_tokens) > 1:
            sizes.add(len(term_tokens) + 1)
            sizes.add(max(1, len(term_tokens) - 1))

        min_score = max(fuzzy_threshold, policy.min_score)
        for size in sorted(sizes):
            best_window = ""
            best_ratio = 0.0
            for window in _windows(tokens, size):
                ratio = _similarity(term, window)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_window = window
            weighted = best_ratio * policy.weight
            if best_ratio >= min_score:
                key = (policy.family, policy.term, best_window)
                if key not in seen:
                    matches.append(
                        EvidenceMatch(
                            policy.family,
                            policy.term,
                            best_window,
                            min(weighted, 1.0),
                            "fuzzy",
                        )
                    )
                    seen.add(key)
                break

    return sorted(matches, key=lambda m: m.score, reverse=True)


def rule_score(matches: list[EvidenceMatch]) -> float:
    if not matches:
        return 0.0
    score = 1.0
    for match in matches[:6]:
        score *= 1.0 - min(match.score, 0.99)
    return 1.0 - score


def explain_matches(matches: list[EvidenceMatch]) -> str:
    if not matches:
        return "No se detectaron terminos de politica en el texto extraido."
    rendered = []
    for match in matches[:4]:
        rendered.append(
            f"{match.family}: '{match.matched_text}' similar a '{match.policy_term}' "
            f"({match.strategy}, score={match.score:.2f})"
        )
    return "Se detectaron senales: " + "; ".join(rendered)
