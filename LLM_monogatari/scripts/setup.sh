#!/bin/zsh
set -eu

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  echo "正在创建 .venv（只影响当前项目）..."
  python3 -m venv .venv
else
  echo ".venv 已存在，不重复创建。"
fi

echo "Python: $(.venv/bin/python --version)"
echo "基础课程只用标准库，无需下载第三方包。"
echo "环境准备完成。"
