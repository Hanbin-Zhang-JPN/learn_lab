"""不依赖 NumPy 的小向量工具，用于理解向量与相似度。"""

from __future__ import annotations

import math
from collections import Counter


def make_vocabulary(documents: list[list[str]]) -> list[str]:
    """用稳定的字典序建立词表。"""

    return sorted({token for document in documents for token in document})


def one_hot(token: str, vocabulary: list[str]) -> list[float]:
    if token not in vocabulary:
        raise ValueError(f"词表里没有 token: {token}")
    return [1.0 if item == token else 0.0 for item in vocabulary]


def bag_of_words(tokens: list[str], vocabulary: list[str]) -> list[float]:
    counts = Counter(tokens)
    return [float(counts[token]) for token in vocabulary]


def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("点积要求两个向量维数相同")
    return sum(a * b for a, b in zip(left, right))


def magnitude(vector: list[float]) -> float:
    return math.sqrt(dot(vector, vector))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    denominator = magnitude(left) * magnitude(right)
    if denominator == 0.0:
        raise ValueError("零向量没有方向，不能计算余弦相似度")
    return dot(left, right) / denominator


def nearest(
    query: list[float], candidates: dict[str, list[float]]
) -> list[tuple[str, float]]:
    """按相似度从高到低返回候选项。"""

    scored = [(name, cosine_similarity(query, vector)) for name, vector in candidates.items()]
    return sorted(scored, key=lambda item: (-item[1], item[0]))
