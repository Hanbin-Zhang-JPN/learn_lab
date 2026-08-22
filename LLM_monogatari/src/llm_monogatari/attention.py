"""纯 Python 的 scaled dot-product attention。"""

from __future__ import annotations

from dataclasses import dataclass
import math

from llm_monogatari.probability import softmax
from llm_monogatari.vectors import dot


@dataclass
class AttentionResult:
    outputs: list[list[float]]
    weights: list[list[float]]


def scaled_dot_product_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
    causal: bool = False,
) -> AttentionResult:
    """计算 Attention(Q, K, V) = softmax(QK^T / sqrt(d))V。"""

    if not queries or not keys or not values:
        raise ValueError("Q、K、V 都不能为空")
    if len(keys) != len(values):
        raise ValueError("每个 key 必须有对应的 value")
    key_dimension = len(keys[0])
    if key_dimension == 0:
        raise ValueError("key 不能是零维向量")
    if any(len(query) != key_dimension for query in queries):
        raise ValueError("query 与 key 的维数必须相同")
    if any(len(key) != key_dimension for key in keys):
        raise ValueError("所有 key 的维数必须相同")
    value_dimension = len(values[0])
    if any(len(value) != value_dimension for value in values):
        raise ValueError("所有 value 的维数必须相同")

    scale = math.sqrt(key_dimension)
    all_weights: list[list[float]] = []
    outputs: list[list[float]] = []

    for query_index, query in enumerate(queries):
        visible_count = min(query_index + 1, len(keys)) if causal else len(keys)
        logits = [dot(query, key) / scale for key in keys[:visible_count]]
        visible_weights = softmax(logits)
        weights = visible_weights + [0.0] * (len(keys) - visible_count)
        output = [
            sum(weight * value[column] for weight, value in zip(weights, values))
            for column in range(value_dimension)
        ]
        all_weights.append(weights)
        outputs.append(output)

    return AttentionResult(outputs=outputs, weights=all_weights)
