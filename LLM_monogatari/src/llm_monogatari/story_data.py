"""训练数据与网页共同使用的输入规则，避免两套格式悄悄分叉。"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


SYSTEM_PROMPT = (
    "あなたは短編作家です。与えられた人物名と場所を必ず一度ずつ自然に使い、"
    "120〜260字ほどの穏やかな日本語の物語を一つだけ書いてください。"
    "説明、見出し、箇条書き、創作過程は書かないでください。"
)

_JAPANESE_LABEL = re.compile(r"^[\u3040-\u30ff\u3400-\u9fff々〆ヵヶー・ 　]+$")


def validate_label(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}必须是文字")
    if any(character in value for character in "\r\n\t"):
        raise ValueError(f"{field_name}不能包含换行或制表符")
    clean = " ".join(value.strip().split())
    if not 1 <= len(clean) <= 20:
        raise ValueError(f"{field_name}长度必须在 1 到 20 个字符之间")
    if not _JAPANESE_LABEL.fullmatch(clean):
        raise ValueError(f"{field_name}只能使用日文汉字、平假名、片假名和间隔号")
    return clean


def user_prompt(name: str, place: str) -> str:
    return f"人物名：{name}\n場所：{place}"


def card_to_chat(card: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    name = validate_label(card.get("name"), "name")
    place = validate_label(card.get("place"), "place")
    story = card.get("story")
    if not isinstance(story, str) or not 40 <= len(story.strip()) <= 500:
        raise ValueError("story 长度必须在 40 到 500 个字符之间")
    if name not in story or place not in story:
        raise ValueError(f"故事必须包含人物名与地名: {name}, {place}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(name, place)},
            {"role": "assistant", "content": story.strip()},
        ]
    }


def load_cards(path: Path) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                card = json.loads(line)
                card_to_chat(card)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            cards.append(card)
    if not cards:
        raise ValueError(f"{path} 没有故事卡")
    return cards
