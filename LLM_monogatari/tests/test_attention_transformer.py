import unittest

from llm_monogatari.attention import scaled_dot_product_attention
from llm_monogatari.tiny_transformer import TinyDecoder


class AttentionTests(unittest.TestCase):
    def test_causal_mask(self) -> None:
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        result = scaled_dot_product_attention(vectors, vectors, vectors, causal=True)
        self.assertEqual(result.weights[0][1:], [0.0, 0.0])
        self.assertEqual(result.weights[1][2], 0.0)
        for row in result.weights:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_weighted_values(self) -> None:
        result = scaled_dot_product_attention(
            queries=[[1.0]],
            keys=[[1.0], [1.0]],
            values=[[2.0, 0.0], [0.0, 4.0]],
        )
        self.assertEqual(result.outputs[0], [1.0, 2.0])


class TinyTransformerTests(unittest.TestCase):
    def test_trace_shapes(self) -> None:
        model = TinyDecoder(vocabulary_size=7, dimension=4)
        trace = model.forward([1, 2, 3])
        self.assertEqual(len(trace["embeddings"]), 3)
        self.assertEqual(len(trace["embeddings"][0]), 4)
        self.assertEqual(len(trace["attention_weights"][0]), 3)
        self.assertEqual(len(trace["logits"][0]), 7)


if __name__ == "__main__":
    unittest.main()
