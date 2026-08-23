import unittest

import torch

from monogatari_llm.model import ModelConfig, StoryTransformer


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig(vocab_size=20, block_size=12, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
        self.model = StoryTransformer(self.config).eval()

    def test_forward_shape_and_loss(self) -> None:
        x = torch.randint(1, 20, (2, 8))
        logits, loss = self.model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 8, 20))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_future_token_cannot_change_past_logits(self) -> None:
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 9, 8]])
        logits_a, _ = self.model(first)
        logits_b, _ = self.model(second)
        self.assertTrue(torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

