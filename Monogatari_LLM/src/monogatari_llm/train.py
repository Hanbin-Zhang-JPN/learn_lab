"""Train Monogatari_LLM from random weights on the local synthetic corpus."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import build_dataset, load_records, training_text
from .model import ModelConfig, StoryTransformer
from .tokenizer import CharTokenizer


def choose_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def padded_examples(texts: list[str], tokenizer: CharTokenizer, block_size: int) -> torch.Tensor:
    rows: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        ids = ids[: block_size + 1]
        ids += [tokenizer.pad_id] * (block_size + 1 - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


def sample_batch(examples: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, examples.size(0), (batch_size,))
    batch = examples[indices].to(device)
    return batch[:, :-1], batch[:, 1:]


@torch.no_grad()
def estimate_loss(
    model: StoryTransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    device: torch.device,
    batches: int = 12,
) -> tuple[float, float]:
    model.eval()
    values: list[float] = []
    for dataset in (train_data, val_data):
        losses = []
        for _ in range(batches):
            x, y = sample_batch(dataset, batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
        values.append(sum(losses) / len(losses))
    model.train()
    return values[0], values[1]


def save_checkpoint(
    path: Path,
    model: StoryTransformer,
    tokenizer: CharTokenizer,
    config: ModelConfig,
    step: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(
        {
            "model_state": state,
            "model_config": asdict(config),
            "tokenizer": tokenizer.to_dict(),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="从随机权重开始训练 Monogatari_LLM")
    parser.add_argument("--data", default="data/stories.jsonl")
    parser.add_argument("--output", default="artifacts/monogatari.pt")
    parser.add_argument("--records", type=int, default=8_000)
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--time-limit-minutes", type=float, default=85.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--tiny", action="store_true", help="只做快速冒烟训练，不追求故事质量")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    if not data_path.exists():
        print("没有发现训练集，正在从可见的句子部件生成……")
        build_dataset(data_path, args.records, args.seed)
    records = load_records(data_path)
    if not records or any("style" not in record for record in records):
        print("发现旧版训练集，正在按四种作风重新生成……")
        build_dataset(data_path, args.records, args.seed)
        records = load_records(data_path)
    texts = [training_text(record) for record in records]
    tokenizer = CharTokenizer.build(texts)

    if args.tiny:
        config = ModelConfig(tokenizer.vocab_size, block_size=96, n_embd=32, n_head=4, n_layer=1, dropout=0.0)
        args.steps = min(args.steps, 8)
        args.batch_size = min(args.batch_size, 4)
    else:
        config = ModelConfig(tokenizer.vocab_size)

    all_examples = padded_examples(texts, tokenizer, config.block_size)
    split = max(1, int(len(all_examples) * 0.9))
    train_data, val_data = all_examples[:split], all_examples[split:]
    device = choose_device()
    if device.type == "cpu" and not args.tiny:
        args.batch_size = min(args.batch_size, 24)

    model = StoryTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    print(f"设备：{device.type}")
    print(f"训练故事：{len(train_data):,}，验证故事：{len(val_data):,}")
    print(f"词表：{tokenizer.vocab_size} 个字符；参数：{model.parameter_count():,}")
    print(f"最多 {args.steps:,} 步或 {args.time_limit_minutes:.0f} 分钟，以先到者为准。")

    started = time.monotonic()
    best_val = math.inf
    last_step = 0
    try:
        for step in range(1, args.steps + 1):
            last_step = step
            x, y = sample_batch(train_data, args.batch_size, device)
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % (2 if args.tiny else 100) == 0:
                elapsed = (time.monotonic() - started) / 60
                print(f"step {step:4d} | loss {loss.item():.3f} | {elapsed:5.1f} min")

            should_evaluate = step == args.steps or step % (4 if args.tiny else 250) == 0
            if should_evaluate:
                train_loss, val_loss = estimate_loss(
                    model, train_data, val_data, args.batch_size, device, batches=2 if args.tiny else 12
                )
                print(f"          train {train_loss:.3f} | val {val_loss:.3f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(Path(args.output), model, tokenizer, config, step, val_loss)
                    print(f"          已保存当前最佳模型 → {args.output}")

            if (time.monotonic() - started) / 60 >= args.time_limit_minutes:
                print("已到时间上限，停止训练。")
                break
    except KeyboardInterrupt:
        print("\n收到 Control+C；若已有最佳检查点，它会保留在输出路径。")
    finally:
        if not Path(args.output).exists() or math.isinf(best_val):
            save_checkpoint(Path(args.output), model, tokenizer, config, last_step, float("nan"))

    elapsed = (time.monotonic() - started) / 60
    print(f"完成，用时 {elapsed:.1f} 分钟。下一步：")
    print("python -m monogatari_llm.generate --name 葵 --place 鎌倉 --style 文芸")


if __name__ == "__main__":
    main()
