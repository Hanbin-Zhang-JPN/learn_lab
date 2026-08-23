#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.12)"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
else
  echo "没有找到 Python 3.11 或 3.12。"
  echo "请先运行：brew install python@3.12"
  exit 1
fi

echo "使用 $($PYTHON_BIN --version)"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

echo
echo "安装完成。接下来运行："
echo "  source .venv/bin/activate"
echo "  python lessons/00_check.py"

