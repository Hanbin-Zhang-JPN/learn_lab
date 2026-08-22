"""两种可以完整读懂的分词器。

这里故意不用现成 tokenizer。字符分词让“文字变成编号”一览无余；
SimpleBPE 则展示现代子词分词器背后的核心合并思想，但不是生产级实现。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import unicodedata


def normalize(text: str) -> str:
    """把全角拉丁字母等兼容字符统一，但不改变日文的词序。"""

    return unicodedata.normalize("NFKC", text)


@dataclass
class CharacterTokenizer:
    """一个字符对应一个 token，保留可逆的 id 映射。"""

    id_to_token: list[str] = field(default_factory=lambda: ["<pad>", "<unk>"])

    @classmethod
    def train(cls, texts: list[str]) -> "CharacterTokenizer":
        chars = sorted(set("".join(normalize(text) for text in texts)))
        return cls(["<pad>", "<unk>", *chars])

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.id_to_token)}

    def encode(self, text: str) -> list[int]:
        mapping = self.token_to_id
        unknown = mapping["<unk>"]
        return [mapping.get(char, unknown) for char in normalize(text)]

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                raise ValueError(f"token id 超出词表范围: {token_id}")
            token = self.id_to_token[token_id]
            tokens.append("□" if token == "<unk>" else "" if token == "<pad>" else token)
        return "".join(tokens)


@dataclass
class SimpleBPE:
    """用相邻 token 频次训练的极小 BPE。

    与工业分词器相比，它没有字节回退、特殊 token 规则和预分词；优点是
    `train` 与 `encode` 的全部逻辑都在本文件里。
    """

    merges: list[tuple[str, str]] = field(default_factory=list)
    learned_vocabulary: set[str] = field(default_factory=set)

    @staticmethod
    def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                result.append(tokens[index] + tokens[index + 1])
                index += 2
            else:
                result.append(tokens[index])
                index += 1
        return result

    @classmethod
    def train(
        cls,
        texts: list[str],
        num_merges: int = 20,
        min_frequency: int = 2,
    ) -> "SimpleBPE":
        sequences = [list(normalize(text)) for text in texts if text]
        if not sequences:
            raise ValueError("至少需要一段非空训练文字")

        merges: list[tuple[str, str]] = []
        vocabulary = set(token for sequence in sequences for token in sequence)

        for _ in range(num_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break

            # 频次相同时按文字排序，保证每台电脑得到相同结果。
            best_pair, count = sorted(
                pair_counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            if count < min_frequency:
                break

            sequences = [cls._merge_pair(sequence, best_pair) for sequence in sequences]
            merges.append(best_pair)
            vocabulary.add("".join(best_pair))

        return cls(merges=merges, learned_vocabulary=vocabulary)

    def encode(self, text: str) -> list[str]:
        tokens = list(normalize(text))
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return "".join(tokens)
