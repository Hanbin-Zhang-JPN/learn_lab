"""Generate a story from a local checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import clean_condition, normalize_style, story_prompt
from .model import ModelConfig, StoryTransformer
from .tokenizer import CharTokenizer
from .train import choose_device


def load_model(path: str | Path, device: torch.device) -> tuple[StoryTransformer, CharTokenizer]:
    # Only open a checkpoint you trained yourself. Pickle-based checkpoints are not a file-sharing format.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = ModelConfig(**checkpoint["model_config"])
    model = StoryTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def generate_story(
    model: StoryTransformer,
    tokenizer: CharTokenizer,
    name: str,
    place: str,
    style: str,
    *,
    max_new_tokens: int = 110,
    temperature: float = 0.85,
    top_k: int = 24,
) -> str:
    name = clean_condition(name, "姓名")
    place = clean_condition(place, "地点")
    style = normalize_style(style)
    prompt = story_prompt(name, place, style)
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    continuation = tokenizer.decode(output[len(prompt_ids) :])
    return f"{place}で、{name}は{continuation}".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="用姓名、地点和作风生成日文小故事")
    parser.add_argument("--checkpoint", default="artifacts/monogatari.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--place", required=True)
    parser.add_argument(
        "--style",
        required=True,
        help="恋愛 / 文芸 / ユーモア / 恐怖（也接受爱情、文艺、幽默）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=110)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=24)
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise SystemExit("还没有检查点。请先运行：python -m monogatari_llm.train")
    device = choose_device()
    model, tokenizer = load_model(args.checkpoint, device)
    print(generate_story(
        model,
        tokenizer,
        args.name,
        args.place,
        args.style,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
