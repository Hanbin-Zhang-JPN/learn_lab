"""Export a local PyTorch checkpoint to transparent browser files.

model.bin is simply little-endian float32 tensors laid one after another.
model-config.json records every tensor name, shape and byte offset.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="导出浏览器可读的模型权重")
    parser.add_argument("--checkpoint", default="artifacts/monogatari.pt")
    parser.add_argument("--output-dir", default="public")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit("还没有检查点。请先运行：python -m monogatari_llm.train")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "model.bin"
    config_path = output_dir / "model-config.json"

    manifest: list[dict[str, object]] = []
    offset = 0
    with binary_path.open("wb") as binary:
        for name, source in checkpoint["model_state"].items():
            tensor = source.detach().to(dtype=torch.float32).contiguous().view(-1)
            values = tensor.tolist()
            raw = struct.pack(f"<{len(values)}f", *values)
            binary.write(raw)
            manifest.append({
                "name": name,
                "shape": list(source.shape),
                "offset": offset,
                "length": len(values),
            })
            offset += len(raw)

    payload = {
        "format": "monogatari-f32-v1",
        "model": checkpoint["model_config"],
        "tokenizer": checkpoint["tokenizer"],
        "checkpoint": {"step": checkpoint.get("step"), "val_loss": checkpoint.get("val_loss")},
        "weights": manifest,
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"已导出 {binary_path}（{offset / 1024 / 1024:.2f} MB）")
    print(f"已导出 {config_path}")
    print("下一步：npm run dev")


if __name__ == "__main__":
    main()
