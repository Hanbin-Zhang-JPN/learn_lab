#!/bin/zsh
set -eu

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "没有找到 cloudflared。macOS 可运行：brew install cloudflared"
  exit 1
fi

if ! curl --silent --fail --max-time 2 http://127.0.0.1:8000/api/health >/dev/null; then
  echo "本地网页尚未运行。先在另一个终端执行：make web"
  exit 1
fi

echo "即将生成临时公网地址。Control + C 会立即停止分享。"
echo "不要把它当作长期、私密或带身份认证的服务。"
cloudflared tunnel --url http://127.0.0.1:8000
