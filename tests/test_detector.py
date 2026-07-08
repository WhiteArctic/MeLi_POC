import unittest

from image_moderation_poc.detector import ImageModerationService
from image_moderation_poc.schemas import EvidenceMatch
from image_moderation_poc.text_model import TextNaiveBayes


class DetectorTest(unittest.TestCase):
    def test_detector_explains_rule_based_infraction(self):
        service = ImageModerationService()
        diagnosis = service.diagnose_text("Oferta especial con envio gratis")

        self.assertIs(diagnosis.has_infraction, True)
        self.assertIn("price_promotion", diagnosis.evidence)

    def test_detector_uses_model_when_rules_are_weak(self):
        model = TextNaiveBayes.train(
            ["best seller recomendado mercado lider", "texto de producto normal"],
            [True, False],
            feature_count=2048,
        )
        service = ImageModerationService(model=model, model_threshold=0.5)
        diagnosis = service.diagnose_text("best seller recomendado")

        self.assertIs(diagnosis.has_infraction, True)
        self.assertIsNotNone(diagnosis.model_score)

    def test_detector_does_not_use_unsupported_model_score_alone(self):
        model = TextNaiveBayes.train(
            ["manual tecnico avanzado", "texto de producto normal"],
            [True, False],
            feature_count=2048,
        )
        service = ImageModerationService(model=model, model_threshold=0.5)
        diagnosis = service.diagnose_text("manual tecnico avanzado")

        self.assertIs(diagnosis.has_infraction, False)
        self.assertGreaterEqual(diagnosis.model_score or 0.0, 0.5)

    def test_detector_uses_visual_classifier_score_as_candidate_signal(self):
        service = ImageModerationService(visual_model_threshold=0.91)
        diagnosis = service.diagnose_text("", visual_model_score=0.93)

        self.assertIs(diagnosis.has_infraction, True)
        self.assertEqual(diagnosis.visual_model_score, 0.93)
        self.assertIn("visual_classifier", diagnosis.evidence)

    def test_detector_ignores_visual_classifier_score_below_threshold(self):
        visual_match = EvidenceMatch(
            "visual_overlay_signal",
            "high contrast text or badge overlay",
            "edge_density=0.20",
            0.35,
            "visual_heuristic",
        )
        service = ImageModerationService(
            visual_model_threshold=0.91,
        )
        diagnosis = service.diagnose_text(
            "",
            visual_matches=[visual_match],
            visual_model_score=0.78,
        )

        self.assertIs(diagnosis.has_infraction, False)
        self.assertEqual(diagnosis.visual_model_score, 0.78)


if __name__ == "__main__":
    unittest.main()
