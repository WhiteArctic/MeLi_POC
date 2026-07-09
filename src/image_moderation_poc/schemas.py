from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceMatch:
    family: str
    policy_term: str
    matched_text: str
    score: float
    strategy: str


@dataclass(frozen=True)
class Diagnosis:
    has_infraction: bool
    evidence: str
    score: float
    rule_score: float
    model_score: float | None = None
    semantic_score: float | None = None
    visual_score: float = 0.0
    visual_model_score: float | None = None
    ocr_text: str = ""
    matched_terms: tuple[EvidenceMatch, ...] = field(default_factory=tuple)

    def public_dict(self) -> dict[str, object]:
        signals: dict[str, float] = {
            "rules": round(self.rule_score, 4),
        }
        if self.model_score is not None:
            signals["text_model"] = round(self.model_score, 4)
        if self.semantic_score is not None:
            signals["semantic_text"] = round(self.semantic_score, 4)
        if self.visual_score:
            signals["visual_heuristics"] = round(self.visual_score, 4)
        if self.visual_model_score is not None:
            signals["visual_model"] = round(self.visual_model_score, 4)

        return {
            "has_infraction": self.has_infraction,
            "decision": "infraction_detected" if self.has_infraction else "no_infraction_detected",
            "score": round(self.score, 4),
            "evidence": self.evidence,
            "ocr_text": self.ocr_text.strip(),
            "signals": signals,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "has_infraction": self.has_infraction,
            "evidence": self.evidence,
            "score": self.score,
            "rule_score": self.rule_score,
            "model_score": self.model_score,
            "semantic_score": self.semantic_score,
            "visual_score": self.visual_score,
            "visual_model_score": self.visual_model_score,
            "ocr_text": self.ocr_text,
            "matched_terms": [m.__dict__ for m in self.matched_terms],
        }
