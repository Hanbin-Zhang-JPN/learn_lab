"""一个只靠 Python 标准库提供网页与 JSON API 的小服务器。"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

from llm_monogatari.story_data import SYSTEM_PROMPT, user_prompt, validate_label


PROJECT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_DIR / "web"
DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_ADAPTER = PROJECT_DIR / "adapters" / "LLM_monogatari-lora"
PUBLIC_INPUT_ERROR = (
    "名前と場所は、20文字以内の漢字・ひらがな・カタカナで入力してください。"
)


def remove_thinking(text: str) -> str:
    """Qwen3 有时会输出思考标签；公开页面只显示故事正文。"""

    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    clean = clean.replace("<|im_end|>", "").strip()
    # 服务器端再做一道长度限制，防止页面被异常长输出占满。
    return clean[:900]


class StoryEngine:
    def __init__(self, model_id: str, adapter_path: Path | None, demo: bool = False) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.demo = demo
        self._model: Any = None
        self._tokenizer: Any = None
        self._generation_lock = threading.Lock()

    @property
    def mode(self) -> str:
        if self.demo:
            return "demo"
        return "adapter" if self.adapter_path is not None else "base"

    def _load(self) -> None:
        if self._model is not None or self.demo:
            return
        if self.adapter_path is not None and not self.adapter_path.exists():
            raise RuntimeError(
                f"没有找到微调结果 {self.adapter_path}。先运行 make train，"
                "或使用 make demo 查看无模型演示。"
            )
        try:
            from mlx_lm import load
        except ImportError as error:
            raise RuntimeError("没有安装 MLX-LM。先运行 make train-install。") from error

        adapter = str(self.adapter_path) if self.adapter_path is not None else None
        self._model, self._tokenizer = load(self.model_id, adapter_path=adapter)

    @staticmethod
    def _demo_story(name: str, place: str) -> str:
        # 演示模式故意使用固定模板，避免让人误以为已经运行了语言模型。
        templates = [
            f"雨上がりの{place}で、{name}は古い切符を一枚拾った。駅員に届けると、それは昨日まで走っていた小さな路線の最終切符だという。帰り道、雲の切れ間から細い光が差し、濡れた線路だけがしばらく金色に残った。",
            f"{name}が{place}を訪れた朝、商店街の時計は五分だけ遅れていた。誰も困ってはいなかったが、パン屋の主人だけが毎日そっと針を直していた。その理由を尋ねると、主人は「急がない朝も町には必要です」と笑った。",
            f"夕暮れの{place}で、{name}は名前のない小さな橋を渡った。欄干には旅人が置いた白い石が一つあり、川音に混じって鈴のような響きがした。翌朝、石は消えていたが、靴の底には淡い砂が残っていた。",
        ]
        return templates[sum(ord(char) for char in name + place) % len(templates)]

    def generate(self, name: str, place: str) -> str:
        name = validate_label(name, "人物名")
        place = validate_label(place, "地名")
        if self.demo:
            return self._demo_story(name, place)

        with self._generation_lock:
            self._load()
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(name, place)},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            sampler = make_sampler(temp=0.75, top_p=0.90, top_k=40)
            text = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=320,
                sampler=sampler,
                verbose=False,
            )
            return remove_thinking(text)


class RateLimiter:
    def __init__(self, limit: int = 6, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._visits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            visits = self._visits[client]
            while visits and now - visits[0] > self.window_seconds:
                visits.popleft()
            if len(visits) >= self.limit:
                return False
            visits.append(now)
            return True


def make_handler(engine: StoryEngine, limiter: RateLimiter) -> type[BaseHTTPRequestHandler]:
    class StoryHandler(BaseHTTPRequestHandler):
        server_version = "LLMMonogatari/0.1"

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, filename: str, content_type: str) -> None:
            path = WEB_DIR / filename
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 规定的名字
            path = self.path.split("?", maxsplit=1)[0]
            if path == "/":
                self._send_file("index.html", "text/html; charset=utf-8")
            elif path == "/styles.css":
                self._send_file("styles.css", "text/css; charset=utf-8")
            elif path == "/app.js":
                self._send_file("app.js", "text/javascript; charset=utf-8")
            elif path == "/api/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "mode": engine.mode})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/story":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            client = self.headers.get("CF-Connecting-IP", self.client_address[0])
            if not limiter.allow(client):
                self._send_json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "少し時間をおいてから、もう一度お試しください。"},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2048:
                    raise ValueError("请求内容为空或过长")
                payload = json.loads(self.rfile.read(length))
                story = engine.generate(payload.get("name"), payload.get("place"))
                self._send_json(HTTPStatus.OK, {"story": story, "mode": engine.mode})
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": PUBLIC_INPUT_ERROR},
                )
            except json.JSONDecodeError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "入力を読み取れませんでした。もう一度お試しください。"},
                )
            except Exception as error:  # 保持服务存活，但不把内部路径发给公网用户
                print(f"生成失败: {error}")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "物語を作れませんでした。Mac側の画面をご確認ください。"},
                )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}")

    return StoryHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM_monogatari 本地网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default=os.environ.get("LLM_MONOGATARI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--demo", action="store_true", help="不加载模型，使用固定模板")
    parser.add_argument("--base-only", action="store_true", help="不用 adapter，仅运行基座")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapter = None if args.base_only else args.adapter
    engine = StoryEngine(args.model, adapter, demo=args.demo)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, RateLimiter()))
    print(f"LLM_monogatari 已启动：http://{args.host}:{args.port}")
    print(f"模式：{engine.mode}。按 Control + C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
