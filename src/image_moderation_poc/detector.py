from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from image_moderation_poc import config
from image_moderation_poc.image_io import download_image, load_or_download_image
from image_moderation_poc.ocr import EasyOCRBackend, OCRBackend, TesseractCliOCR
from image_moderation_poc.policy import explain_matches, find_policy_matches, rule_score, weak_policy_cues
from image_moderation_poc.schemas import Diagnosis, EvidenceMatch
from image_moderation_poc.text_model import TextNaiveBayes
from image_moderation_poc.visual import VisualHeuristicDetector
from image_moderation_poc.visual_model import MobileNetV3VisualClassifier


@dataclass
class ImageModerationService:
    model: TextNaiveBayes | None = None
    ocr_backend: OCRBackend | None = None
    visual_classifier: object | None = None
    rule_threshold: float = config.DEFAULT_RULE_THRESHOLD
    hybrid_threshold: float = config.DEFAULT_HYBRID_THRESHOLD
    model_threshold: float = config.DEFAULT_MODEL_THRESHOLD
    semantic_threshold: float = 0.99
    semantic_min_cues: int = 3
    visual_semantic_threshold: float = 0.99
    visual_model_threshold: float = config.CANDIDATE_VISUAL_FUSION_THRESHOLD
    verify_ssl: bool = False
    image_cache_dir: Path | None = config.IMAGE_CACHE_DIR

    @classmethod
    def from_model_path(
        cls,
        model_path: Path = config.MODEL_PATH,
        ocr_backend: OCRBackend | None = None,
        visual_classifier: object | None = None,
        verify_ssl: bool = False,
        image_cache_dir: Path | None = config.IMAGE_CACHE_DIR,
    ) -> "ImageModerationService":
        model = TextNaiveBayes.load(model_path) if model_path.exists() else None
        return cls(
            model=model,
            ocr_backend=ocr_backend or TesseractCliOCR(),
            visual_classifier=visual_classifier,
            verify_ssl=verify_ssl,
            image_cache_dir=image_cache_dir,
        )

    @classmethod
    def from_candidate(
        cls,
        model_path: Path = config.MODEL_PATH,
        visual_model_path: Path = config.PRETRAINED_VISUAL_MODEL_PATH,
        verify_ssl: bool = False,
        image_cache_dir: Path | None = config.IMAGE_CACHE_DIR,
    ) -> "ImageModerationService":
        easyocr = EasyOCRBackend()
        ocr_backend: OCRBackend = easyocr if easyocr.is_available() else TesseractCliOCR()
        visual_classifier = None
        if visual_model_path.exists() and MobileNetV3VisualClassifier.is_available():
            visual_classifier = MobileNetV3VisualClassifier.load(
                visual_model_path,
                threshold=config.CANDIDATE_VISUAL_FUSION_THRESHOLD,
            )
        return cls.from_model_path(
            model_path=model_path,
            ocr_backend=ocr_backend,
            visual_classifier=visual_classifier,
            verify_ssl=verify_ssl,
            image_cache_dir=image_cache_dir,
        )

    def diagnose_text(
        self,
        text: str,
        visual_matches: list[EvidenceMatch] | None = None,
        visual_model_score: float | None = None,
    ) -> Diagnosis:
        text_matches = find_policy_matches(text)
        visual_matches = visual_matches or []
        visual_model_match = self._visual_model_match(visual_model_score)
        matches = text_matches + visual_matches + ([visual_model_match] if visual_model_match else [])
        rules = rule_score(text_matches)
        model_score = self.model.predict_proba_one(text) if self.model else None
        semantic_score, semantic_chunk = self._semantic_score(text)
        visual_score = rule_score(visual_matches)
        visual_classifier_score = visual_model_score or 0.0
        visual_model_candidate = (
            visual_model_score is not None
            and visual_model_score >= self.visual_model_threshold
        )
        visual_decision_score = visual_classifier_score if visual_model_candidate else 0.0
        cue_count = len(weak_policy_cues(text))

        if model_score is None:
            final_score = max(rules, visual_decision_score)
            has_infraction = rules >= self.rule_threshold or visual_model_candidate
        else:
            model_has_support = cue_count >= 1 or rules > 0.0 or visual_score >= 0.35
            semantic_candidate = (
                semantic_score is not None
                and semantic_score >= self.semantic_threshold
                and cue_count >= self.semantic_min_cues
            )
            visual_semantic_candidate = (
                semantic_score is not None
                and semantic_score >= self.visual_semantic_threshold
                and cue_count >= 1
                and visual_score >= 0.35
            )
            semantic_blend = (
                0.45 * rules + 0.45 * semantic_score + 0.10 * visual_score
                if semantic_score is not None and cue_count >= 1
                else 0.0
            )
            model_blend = 0.35 * rules + 0.65 * model_score if model_has_support else 0.0
            final_score = max(rules, model_blend, semantic_blend, visual_decision_score)
            has_infraction = (
                rules >= self.rule_threshold
                or (model_has_support and model_score >= self.model_threshold)
                or (model_has_support and final_score >= self.hybrid_threshold)
                or semantic_candidate
                or visual_semantic_candidate
                or visual_model_candidate
            )

        if has_infraction:
            evidence = explain_matches(matches)
            if model_score is not None:
                evidence += f" Modelo textual score={model_score:.2f}."
            if semantic_score is not None:
                evidence += f" Clasificador semantico score={semantic_score:.2f}."
                if semantic_chunk:
                    evidence += f" Fragmento='{semantic_chunk[:80]}'."
            if visual_score:
                evidence += f" Visual score={visual_score:.2f}."
            if visual_model_score is not None:
                evidence += f" Clasificador visual score={visual_model_score:.2f}."
        else:
            evidence = "No se encontraron senales suficientes de infraccion."
            if model_score is not None:
                evidence += f" Modelo textual score={model_score:.2f}."
            if semantic_score is not None:
                evidence += f" Clasificador semantico score={semantic_score:.2f}."
            if visual_score:
                evidence += f" Visual score={visual_score:.2f}."
            if visual_model_score is not None:
                evidence += f" Clasificador visual score={visual_model_score:.2f}."

        return Diagnosis(
            has_infraction=has_infraction,
            evidence=evidence,
            score=final_score,
            rule_score=rules,
            model_score=model_score,
            semantic_score=semantic_score,
            visual_score=visual_score,
            visual_model_score=visual_model_score,
            ocr_text=text,
            matched_terms=tuple(matches),
        )

    def diagnose_image(self, picture_url: str) -> Diagnosis:
        downloaded = (
            load_or_download_image(picture_url, self.image_cache_dir, verify_ssl=self.verify_ssl)
            if self.image_cache_dir is not None
            else download_image(picture_url, verify_ssl=self.verify_ssl)
        )
        ocr = self.ocr_backend or TesseractCliOCR()
        text = ocr.extract_text(downloaded.image)
        visual_matches = VisualHeuristicDetector().detect(downloaded.image)
        visual_model_score = self._score_visual_model(downloaded.image)
        return self.diagnose_text(
            text,
            visual_matches=visual_matches,
            visual_model_score=visual_model_score,
        )

    def _score_visual_model(self, image) -> float | None:
        if not self.visual_classifier:
            return None
        predict = getattr(self.visual_classifier, "predict_proba_image", None)
        if predict is None:
            return None
        try:
            return float(predict(image))
        except Exception:
            return None

    def _visual_model_match(self, score: float | None) -> EvidenceMatch | None:
        if score is None or score < self.visual_model_threshold:
            return None
        return EvidenceMatch(
            family="visual_classifier",
            policy_term="visual moderation pattern",
            matched_text=f"mobilenet_v3_score={score:.2f}",
            score=min(1.0, score),
            strategy="mobilenet_v3_linear",
        )

    def _semantic_score(self, text: str) -> tuple[float | None, str]:
        if not self.model:
            return None, ""

        chunks = self._semantic_chunks(text)
        if not chunks:
            return self.model.predict_proba_one(text), text

        best_chunk = ""
        best_score = 0.0
        for chunk in chunks:
            score = self.model.predict_proba_one(chunk)
            if score > best_score:
                best_score = score
                best_chunk = chunk
        return best_score, best_chunk

    def _semantic_chunks(self, text: str) -> list[str]:
        lines = [" ".join(line.split()) for line in text.splitlines()]
        lines = [line for line in lines if len(line) >= 6]
        chunks = list(lines)
        chunks.extend(f"{lines[i]} {lines[i + 1]}" for i in range(0, len(lines) - 1))
        if text.strip():
            chunks.append(text)
        return chunks


def diagnose_image(picture_url: str) -> dict[str, object]:
    service = ImageModerationService.from_candidate(verify_ssl=False)
    diagnosis = service.diagnose_image(picture_url)
    return diagnosis.public_dict()
