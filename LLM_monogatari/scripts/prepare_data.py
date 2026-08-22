#!/usr/bin/env python3
"""校验人工故事卡并生成 MLX-LM 所需的三个 JSONL 文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from llm_monogatari.story_data import card_to_chat, load_cards


PROJECT_DIR = Path(__file__).resolve().parents[1]


def split_cards(cards: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    if len(cards) < 20:
        raise ValueError("至少准备 20 张故事卡，才有意义地划分 train/valid/test")
    pairs = [(card["name"], card["place"]) for card in cards]
    if len(pairs) != len(set(pairs)):
        raise ValueError("发现重复的人名/地名组合，请先去重")

    shuffled = cards[:]
    random.Random(seed).shuffle(shuffled)
    test_size = max(2, round(len(shuffled) * 0.1))
    valid_size = max(2, round(len(shuffled) * 0.1))
    return {
        "train": shuffled[: -(valid_size + test_size)],
        "valid": shuffled[-(valid_size + test_size) : -test_size],
        "test": shuffled[-test_size:],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(card_to_chat(record), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备本地故事微调数据")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_DIR / "data" / "source" / "story_cards.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "data" / "processed",
    )
    args = parser.parse_args()

    cards = load_cards(args.source)
    splits = split_cards(cards)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", records)
        print(f"{name:>5}: {len(records):>3} 篇 -> {args.output / f'{name}.jsonl'}")
    print("已固定随机种子 42；重复运行会得到相同划分。")


if __name__ == "__main__":
    main()
