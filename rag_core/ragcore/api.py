"""標準ライブラリだけの小さな JSON HTTP API を提供する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈の評価を遅らせる。

import json  # HTTP の要求と応答を JSON で扱う。
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # 並行要求を標準機能で処理する。
from typing import Any  # JSON 値の型を簡潔に示す。

from .answer import ExtractiveAnswerer  # 外部不要の回答器を使う。
from .model import SearchHit  # 検索結果の変換型を使う。
from .search import SearchIndex  # 共有する読み取り索引を使う。


class RagHandler(BaseHTTPRequestHandler):  # /health、/search、/ask を処理する。
    index: SearchIndex | None = None  # 起動時に読み取り索引を一つ設定する。
    answerer = ExtractiveAnswerer()  # 状態を持たない回答器を共有する。
    max_body = 64 * 1024  # 過大な要求を 64 KiB で拒否する。

    def do_GET(self) -> None:  # GET 要求を処理する。
        if self.path == "/health":  # 生存確認の経路か確認する。
            if self.index is None:  # 起動設定の不足を確認する。
                self._send(503, {"ok": False, "error": "索引が未設定です。"})  # 利用不可を返す。
                return  # 以降の処理を止める。
            self._send(200, {"ok": True, "index": self.index.info()})  # 索引情報付きで正常を返す。
            return  # 他の経路判定を止める。
        self._send(404, {"ok": False, "error": "経路が見つかりません。"})  # 未定義経路を返す。

    def do_POST(self) -> None:  # POST 要求を処理する。
        if self.path not in {"/search", "/ask"}:  # 対応経路か確認する。
            self._send(404, {"ok": False, "error": "経路が見つかりません。"})  # 未定義経路を返す。
            return  # 本文を読まず終了する。
        if self.index is None:  # 起動設定の不足を確認する。
            self._send(503, {"ok": False, "error": "索引が未設定です。"})  # 利用不可を返す。
            return  # 検索を行わない。
        try:  # 入力エラーを JSON へ変換する。
            payload = self._read_json()  # 制限付きで要求 JSON を読む。
            query = str(payload.get("query", "")).strip()  # 質問を文字列として得る。
            top_k = int(payload.get("top_k", 5))  # 返却件数を整数として得る。
            source_prefix = str(payload.get("source_prefix", ""))  # 任意の出典範囲を得る。
            hits = self.index.search(query, top_k=top_k, source_prefix=source_prefix)  # 通常と同じ検索を行う。
            result: dict[str, Any] = {"ok": True, "hits": [_hit_dict(number, hit) for number, hit in enumerate(hits, start=1)], "stats": self.index.last_stats}  # 検索結果と診断値を作る。
            if self.path == "/ask":  # 回答生成経路か確認する。
                result["answer"] = self.answerer.answer(query, hits)  # 外部不要の引用回答を加える。
            self._send(200, result)  # UTF-8 JSON を返す。
        except (ValueError, json.JSONDecodeError) as error:  # 利用者が直せる入力失敗を捕捉する。
            self._send(400, {"ok": False, "error": str(error)})  # 追跡情報なしで原因を返す。

    def _read_json(self) -> dict[str, Any]:  # 制限付きで JSON 本文を読む。
        length = int(self.headers.get("Content-Length", "0"))  # 宣言された本文長を得る。
        if length < 1 or length > self.max_body:  # 空または過大か確認する。
            raise ValueError("要求本文は 1 byte 以上 64 KiB 以下にしてください。")  # 受け付ける範囲を示す。
        value = json.loads(self.rfile.read(length).decode("utf-8"))  # 指定長だけを UTF-8 JSON として読む。
        if not isinstance(value, dict):  # 最上位がオブジェクトか確認する。
            raise ValueError("JSON の最上位はオブジェクトにしてください。")  # 必要形式を示す。
        return value  # 検証済み辞書を返す。

    def _send(self, status: int, payload: dict[str, Any]) -> None:  # 一貫した JSON 応答を送る。
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")  # 日本語を保った UTF-8 JSON を作る。
        self.send_response(status)  # HTTP 状態番号を送る。
        self.send_header("Content-Type", "application/json; charset=utf-8")  # 応答形式を示す。
        self.send_header("Content-Length", str(len(body)))  # 本文長を byte 単位で示す。
        self.send_header("X-Content-Type-Options", "nosniff")  # ブラウザーの形式推測を抑える。
        self.end_headers()  # ヘッダー部を確定する。
        self.wfile.write(body)  # JSON 本文を送る。

    def log_message(self, format_string: str, *args: object) -> None:  # 標準アクセスログを短く保つ。
        print(f"API {self.address_string()} {format_string % args}")  # 要求元と結果だけを表示する。


def _hit_dict(number: int, hit: SearchHit) -> dict[str, Any]:  # 検索結果を JSON 可能な辞書へ変える。
    return {"citation": f"S{number}", "source": hit.chunk.source, "title": hit.chunk.title, "heading": hit.chunk.heading, "start": hit.chunk.start, "end": hit.chunk.end, "text": hit.chunk.text, "score": hit.score, "bm25": hit.bm25, "cosine": hit.cosine, "phrase": hit.phrase}  # 原文と説明可能な内訳を返す。


def serve(index: SearchIndex, host: str = "127.0.0.1", port: int = 8080) -> None:  # 読み取り API を起動する。
    RagHandler.index = index  # 全要求が参照する索引を設定する。
    server = ThreadingHTTPServer((host, port), RagHandler)  # 並行処理できる標準サーバーを作る。
    print(f"RAG API: http://{host}:{port}  終了: Ctrl+C")  # 接続先を利用者へ示す。
    try:  # キーボード終了を正常に扱う。
        server.serve_forever()  # 停止要求まで HTTP を処理する。
    except KeyboardInterrupt:  # Ctrl+C を捕捉する。
        print("\nAPI を停止します。")  # 正常な停止を示す。
    finally:  # どの終了経路でもソケットを閉じる。
        server.server_close()  # 待ち受け資源を解放する。
