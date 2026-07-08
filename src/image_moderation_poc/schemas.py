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
