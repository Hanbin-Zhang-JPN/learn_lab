#!/bin/zsh
set -eu

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if ! .venv/bin/python -c "import mlx_lm" >/dev/null 2>&1; then
  echo "尚未安装训练依赖。请先运行：make train-install"
  exit 1
fi

if [[ ! -f data/processed/train.jsonl ]]; then
  echo "训练数据不存在。请先运行：make data"
  exit 1
fi

echo "开始训练。按 Control + C 可以安全停止；已保存的 checkpoint 不会消失。"
.venv/bin/python -m mlx_lm.lora --config config/lora.yaml
