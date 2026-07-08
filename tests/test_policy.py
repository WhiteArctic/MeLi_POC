import unittest

from image_moderation_poc.policy import find_policy_matches, normalize_text


class PolicyTest(unittest.TestCase):
    def test_normalize_text_removes_accents_and_noise(self):
        self.assertEqual(normalize_text("Envio INMEDIATO!!!"), "envio inmediato")
        self.assertEqual(normalize_text("Envío-gratis"), "envio gratis")

    def test_fuzzy_matching_accepts_small_ocr_errors(self):
        matches = find_policy_matches("Envio inmediat0 para este producto", fuzzy_threshold=0.84)
        self.assertTrue(matches)
        self.assertEqual(matches[0].family, "shipping_promise")

    def test_best_seller_is_a_marketplace_badge_signal(self):
        matches = find_policy_matches("Autor best seller del New York Times")
        self.assertTrue(matches)
        self.assertEqual(matches[0].family, "marketplace_badge_social_proof")
        self.assertGreaterEqual(matches[0].score, 0.88)

    def test_marketplace_badge_terms_are_detected(self):
        families = {
            match.family
            for match in find_policy_matches("K-POP Official Store Amazon Exclusive")
        }

        self.assertIn("marketplace_badge_social_proof", families)


if __name__ == "__main__":
    unittest.main()
