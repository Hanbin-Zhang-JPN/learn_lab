#!/usr/bin/env python3
"""用相同题目生成基座与 LoRA 结果，保存供人工盲评的记录。"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
from pathlib import Path

from llm_monogatari.web import DEFAULT_ADAPTER, DEFAULT_MODEL, StoryEngine


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROMPTS = [
    ("灯", "函館"),
    ("海斗", "京都"),
    ("すみれ", "長崎"),
    ("凪", "東京・浅草"),
    ("千秋", "富士山"),
    ("結", "小樽"),
]


def generate_set(model: str, adapter: Path | None) -> list[str]:
    engine = StoryEngine(model, adapter)
    outputs = []
    for index, (name, place) in enumerate(PROMPTS, start=1):
        print(f"生成 {index}/{len(PROMPTS)}: {name} / {place}")
        outputs.append(engine.generate(name, place))
    del engine
    gc.collect()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="比较基座与微调后的故事")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--adapter-only", action="store_true", help="只生成微调结果，节省时间")
    args = parser.parse_args()
    if not args.adapter.exists():
        raise SystemExit(f"没有找到 {args.adapter}。请先完成 make train。")

    base = [] if args.adapter_only else generate_set(args.model, None)
    tuned = generate_set(args.model, args.adapter)
    output_dir = PROJECT_DIR / "runs"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M')}.md"

    lines = [
        "# 故事人工评测",
        "",
        "评分：人名与地名是否自然出现（0/1）；日文自然度（1–5）；完整性（1–5）；是否模板化（是/否）。",
        "不要只凭一篇决定模型优劣。先隐藏 A/B 标记，请另一位读者盲评。",
        "",
    ]
    for index, ((name, place), tuned_story) in enumerate(zip(PROMPTS, tuned), start=1):
        lines.extend([f"## {index}. {name} / {place}", ""])
        if base:
            lines.extend(["### A", "", base[index - 1], ""])
        lines.extend(["### B", "", tuned_story, "", "评分与备注：", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"评测表已保存：{path}")


if __name__ == "__main__":
    main()
