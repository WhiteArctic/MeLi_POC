import unittest

from image_moderation_poc.text_model import TextNaiveBayes


class TextModelTest(unittest.TestCase):
    def test_text_model_learns_simple_policy_language(self):
        texts = [
            "envio gratis descuento hot sale",
            "oferta 50 off envio inmediato",
            "zapato cuero talle 42",
            "manual de usuario color negro",
        ]
        labels = [True, True, False, False]
        model = TextNaiveBayes.train(texts, labels, feature_count=2048)

        self.assertGreater(model.predict_proba_one("descuento y envio gratis"), 0.5)
        self.assertLess(model.predict_proba_one("producto color negro"), 0.5)


if __name__ == "__main__":
    unittest.main()

