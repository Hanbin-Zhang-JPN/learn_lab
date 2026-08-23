"""A deliberately small character tokenizer.

Japanese text does not put spaces between every word.  A production tokenizer
would normally use subwords, but one Unicode character = one token makes the
first learning pass visible and predictable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass
class CharTokenizer:
    tokens: list[str]

    def __post_init__(self) -> None:
        if self.tokens[: len(SPECIAL_TOKENS)] != list(SPECIAL_TOKENS):
            raise ValueError(f"词表必须以 {SPECIAL_TOKENS} 开头")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        if len(self.token_to_id) != len(self.tokens):
            raise ValueError("词表里出现了重复 token")

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        characters: set[str] = set()
        for text in texts:
            characters.update(text)
        return cls(list(SPECIAL_TOKENS) + sorted(characters))

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        result: list[str] = []
        special = set(SPECIAL_TOKENS)
        for token_id in ids:
            if not 0 <= int(token_id) < len(self.tokens):
                token = "<unk>"
            else:
                token = self.tokens[int(token_id)]
            if skip_special and token in special:
                continue
            result.append(token)
        return "".join(result)

    def to_dict(self) -> dict[str, list[str]]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> "CharTokenizer":
        return cls(list(payload["tokens"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

