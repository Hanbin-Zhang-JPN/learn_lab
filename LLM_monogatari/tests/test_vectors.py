import math
import unittest

from llm_monogatari.vectors import bag_of_words, cosine_similarity, dot, one_hot


class VectorTests(unittest.TestCase):
    def test_one_hot(self) -> None:
        self.assertEqual(one_hot("猫", ["犬", "猫", "鳥"]), [0.0, 1.0, 0.0])

    def test_bag_of_words_counts_repetition(self) -> None:
        self.assertEqual(bag_of_words(["猫", "猫", "犬"], ["犬", "猫"]), [1.0, 2.0])

    def test_dot_product(self) -> None:
        self.assertEqual(dot([1, 2, 3], [4, 5, 6]), 32)

    def test_cosine(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 1], [1, 0]), 1 / math.sqrt(2))

    def test_zero_vector_has_no_cosine(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])


if __name__ == "__main__":
    unittest.main()
