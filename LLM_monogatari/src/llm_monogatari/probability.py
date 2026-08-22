"""语言模型中最常见的几个概率运算。"""

from __future__ import annotations

import math


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        raise ValueError("softmax 至少需要一个数")
    largest = max(logits)
    exponentials = [math.exp(value - largest) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def cross_entropy(probabilities: list[float], target_index: int) -> float:
    if target_index < 0 or target_index >= len(probabilities):
        raise ValueError("目标下标超出概率列表")
    probability = probabilities[target_index]
    if probability <= 0.0 or probability > 1.0:
        raise ValueError("目标概率必须在 (0, 1] 内")
    return -math.log(probability)


def perplexity(mean_cross_entropy: float) -> float:
    return math.exp(mean_cross_entropy)


def sample_index(probabilities: list[float], random_number: float) -> int:
    """用 [0, 1) 的一个随机数从离散分布中抽样。"""

    if not 0.0 <= random_number < 1.0:
        raise ValueError("random_number 必须在 [0, 1) 内")
    total = sum(probabilities)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("概率之和必须为 1")
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if random_number < cumulative:
            return index
    return len(probabilities) - 1
