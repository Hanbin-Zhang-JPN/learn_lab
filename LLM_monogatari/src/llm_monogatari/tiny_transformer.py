"""只有一次前向传播的迷你 Transformer 解码器。

它没有训练代码，也不会写出好故事。用途是把 embedding、位置、Q/K/V、
残差、前馈层和 logits 串在同一条可观察的数据流上。
"""

from __future__ import annotations

import math
import random

from llm_monogatari.attention import scaled_dot_product_attention


Matrix = list[list[float]]


def _random_matrix(rows: int, columns: int, rng: random.Random) -> Matrix:
    scale = 1.0 / math.sqrt(columns)
    return [[rng.uniform(-scale, scale) for _ in range(columns)] for _ in range(rows)]


def linear(vector: list[float], weights: Matrix) -> list[float]:
    """weights 的形状为 output × input。"""

    return [sum(value * weight for value, weight in zip(vector, row)) for row in weights]


def layer_norm(vector: list[float], epsilon: float = 1e-5) -> list[float]:
    mean = sum(vector) / len(vector)
    variance = sum((value - mean) ** 2 for value in vector) / len(vector)
    return [(value - mean) / math.sqrt(variance + epsilon) for value in vector]


def add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


class TinyDecoder:
    def __init__(self, vocabulary_size: int, dimension: int = 4, seed: int = 7) -> None:
        if vocabulary_size < 2 or dimension < 2:
            raise ValueError("词表和隐藏维度都太小")
        rng = random.Random(seed)
        self.dimension = dimension
        self.token_embeddings = _random_matrix(vocabulary_size, dimension, rng)
        self.position_embeddings = _random_matrix(64, dimension, rng)
        self.wq = _random_matrix(dimension, dimension, rng)
        self.wk = _random_matrix(dimension, dimension, rng)
        self.wv = _random_matrix(dimension, dimension, rng)
        self.ffn_up = _random_matrix(dimension * 2, dimension, rng)
        self.ffn_down = _random_matrix(dimension, dimension * 2, rng)
        self.output = _random_matrix(vocabulary_size, dimension, rng)

    def forward(self, token_ids: list[int]) -> dict[str, Matrix]:
        if not token_ids:
            raise ValueError("至少输入一个 token id")
        if len(token_ids) > len(self.position_embeddings):
            raise ValueError("序列超过这个教学模型的 64 token 上限")

        hidden = [
            add(self.token_embeddings[token_id], self.position_embeddings[position])
            for position, token_id in enumerate(token_ids)
        ]
        queries = [linear(vector, self.wq) for vector in hidden]
        keys = [linear(vector, self.wk) for vector in hidden]
        values = [linear(vector, self.wv) for vector in hidden]
        attention = scaled_dot_product_attention(queries, keys, values, causal=True)

        after_attention = [
            layer_norm(add(original, attended))
            for original, attended in zip(hidden, attention.outputs)
        ]
        feed_forward = []
        for vector in after_attention:
            widened = [max(0.0, value) for value in linear(vector, self.ffn_up)]
            feed_forward.append(linear(widened, self.ffn_down))
        final = [
            layer_norm(add(original, update))
            for original, update in zip(after_attention, feed_forward)
        ]
        logits = [linear(vector, self.output) for vector in final]
        return {
            "embeddings": hidden,
            "attention_weights": attention.weights,
            "hidden_states": final,
            "logits": logits,
        }
