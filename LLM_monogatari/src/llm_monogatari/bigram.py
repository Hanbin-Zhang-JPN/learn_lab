"""根据“上一个字符”预测“下一个字符”的计数语言模型。"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random


class BigramLanguageModel:
    START = "<S>"
    END = "<E>"

    def __init__(self, smoothing: float = 0.1) -> None:
        if smoothing <= 0.0:
            raise ValueError("smoothing 必须大于 0")
        self.smoothing = smoothing
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: list[str] = []

    def train(self, texts: list[str]) -> None:
        vocabulary = {self.END}
        for text in texts:
            sequence = [self.START, *text, self.END]
            vocabulary.update(text)
            for left, right in zip(sequence, sequence[1:]):
                self.counts[left][right] += 1
        self.vocabulary = sorted(vocabulary)

    def distribution(self, previous: str, temperature: float = 1.0) -> list[tuple[str, float]]:
        if not self.vocabulary:
            raise RuntimeError("请先训练模型")
        if temperature <= 0.0:
            raise ValueError("temperature 必须大于 0")
        raw = [self.counts[previous][token] + self.smoothing for token in self.vocabulary]
        adjusted = [value ** (1.0 / temperature) for value in raw]
        total = sum(adjusted)
        return list(zip(self.vocabulary, [value / total for value in adjusted]))

    def generate(self, seed: int = 0, max_characters: int = 60, temperature: float = 1.0) -> str:
        rng = random.Random(seed)
        previous = self.START
        output: list[str] = []
        for _ in range(max_characters):
            choices = self.distribution(previous, temperature)
            tokens = [token for token, _ in choices]
            weights = [weight for _, weight in choices]
            token = rng.choices(tokens, weights=weights, k=1)[0]
            if token == self.END:
                break
            output.append(token)
            previous = token
        return "".join(output)

    def negative_log_likelihood(self, text: str) -> float:
        sequence = [self.START, *text, self.END]
        losses: list[float] = []
        for previous, target in zip(sequence, sequence[1:]):
            distribution = dict(self.distribution(previous))
            losses.append(-math.log(distribution[target]))
        return sum(losses) / len(losses)
