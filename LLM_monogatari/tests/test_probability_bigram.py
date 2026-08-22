import math
import unittest

from llm_monogatari.bigram import BigramLanguageModel
from llm_monogatari.probability import cross_entropy, perplexity, sample_index, softmax


class ProbabilityTests(unittest.TestCase):
    def test_softmax_sums_to_one(self) -> None:
        probabilities = softmax([1001.0, 1000.0, 999.0])
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_cross_entropy_and_perplexity(self) -> None:
        loss = cross_entropy([0.25, 0.75], 1)
        self.assertAlmostEqual(loss, -math.log(0.75))
        self.assertAlmostEqual(perplexity(loss), 1 / 0.75)

    def test_sample_index_boundaries(self) -> None:
        self.assertEqual(sample_index([0.2, 0.3, 0.5], 0.0), 0)
        self.assertEqual(sample_index([0.2, 0.3, 0.5], 0.2), 1)
        self.assertEqual(sample_index([0.2, 0.3, 0.5], 0.99), 2)


class BigramTests(unittest.TestCase):
    def test_distribution_sums_to_one(self) -> None:
        model = BigramLanguageModel()
        model.train(["葵は海へ。", "蓮は山へ。"])
        self.assertAlmostEqual(sum(prob for _, prob in model.distribution("は")), 1.0)

    def test_generation_is_reproducible(self) -> None:
        model = BigramLanguageModel()
        model.train(["葵は海へ。", "蓮は山へ。"])
        self.assertEqual(model.generate(seed=9), model.generate(seed=9))


if __name__ == "__main__":
    unittest.main()
