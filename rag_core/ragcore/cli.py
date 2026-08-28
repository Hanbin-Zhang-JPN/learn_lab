"""索引、検索、回答、評価、API を一つの CLI から操作する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈の評価を遅らせる。

import argparse  # 標準ライブラリでコマンド引数を扱う。
import json  # JSON 表示と入力例外を扱う。
import os  # LLM 設定を環境変数から読む。
import sqlite3  # 索引ファイルの読み取り失敗を捕捉する。
import sys  # 標準エラーと終了コードを扱う。
import tempfile  # デモ用索引を自動削除する。
import urllib.error  # 任意 LLM の接続失敗を捕捉する。
from collections import Counter  # inspect で特徴頻度を表示する。
from pathlib import Path  # 文書、索引、評価パスを扱う。
from typing import Any  # JSON 表示辞書の型を示す。

from .answer import ExtractiveAnswerer, LlmAnswerer  # 回答方式を読み込む。
from .api import serve  # ローカル HTTP API を読み込む。
from .evaluate import evaluate, load_cases  # 検索品質評価を読み込む。
from .model import SearchHit  # 表示対象の型を読み込む。
from .search import SearchIndex  # SQLite 検索索引を読み込む。
from .terms import analyze, normalize  # 特徴観察用の処理を読み込む。
from .text import TextChunker, load_documents  # 文書読み込みと分割を読み込む。


def build_index(documents_path: Path, output: Path, chunk_size: int, overlap: int) -> SearchIndex:  # 文書フォルダから永続索引を作る。
    documents = load_documents(documents_path)  # 対応文書を安定順で読む。
    chunker = TextChunker(chunk_size, overlap)  # 指定幅の分割器を作る。
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]  # 全文書をチャンク列へ展開する。
    return SearchIndex.build(output, chunks)  # 原子的に索引を構築して返す。


def retrieve(index: SearchIndex, args: argparse.Namespace) -> list[SearchHit]:  # CLI の共通設定で検索する。
    return index.search(args.query, top_k=args.top_k, candidate_limit=args.candidates, source_prefix=args.source_prefix, mmr_lambda=args.mmr_lambda, min_signal_ratio=args.min_signal_ratio)  # 値を検索器へそのまま渡す。


def hit_record(number: int, hit: SearchHit) -> dict[str, Any]:  # 検索結果を JSON 可能な辞書へ変える。
    return {"citation": f"S{number}", "source": hit.chunk.source, "title": hit.chunk.title, "heading": hit.chunk.heading, "start": hit.chunk.start, "end": hit.chunk.end, "text": hit.chunk.text, "score": hit.score, "bm25": hit.bm25, "cosine": hit.cosine, "phrase": hit.phrase}  # 引用、原文、内訳をまとめる。


def print_hits(hits: list[SearchHit]) -> None:  # 人が確認しやすい検索結果を表示する。
    if not hits:  # 空結果を先に扱う。
        print("一致する根拠はありません。")  # 推測なしを明確に示す。
        return  # 通常表示を止める。
    for number, hit in enumerate(hits, start=1):  # 順位付きで各根拠を表示する。
        place = f"{hit.chunk.source} / {hit.chunk.heading} / chars {hit.chunk.start}:{hit.chunk.end}"  # 出典位置を一行にする。
        scores = f"score={hit.score:.4f} bm25={hit.bm25:.4f} cosine={hit.cosine:.4f} phrase={hit.phrase:.2f}"  # 順位の内訳を一行にする。
        print(f"\n[S{number}] {place}\n{scores}\n{hit.chunk.text}")  # 引用番号、内訳、原文を表示する。


def add_search_options(parser: argparse.ArgumentParser) -> None:  # search と ask の共通引数を作る。
    parser.add_argument("-q", "--query", required=True, help="検索または質問する文")  # 質問を必須にする。
    parser.add_argument("-k", "--top-k", type=int, default=5, help="返す根拠数（既定: 5）")  # 最終件数を設定可能にする。
    parser.add_argument("--candidates", type=int, default=1200, help="精密採点する候補上限")  # 速度と再現率を調整可能にする。
    parser.add_argument("--source-prefix", default="", help="検索を限定する出典パスの先頭")  # 部門やテナントの範囲を絞れるようにする。
    parser.add_argument("--mmr-lambda", type=float, default=0.78, help="関連性の重み（0〜1）")  # 重複抑制との比率を設定可能にする。
    parser.add_argument("--min-signal-ratio", type=float, default=0.12, help="最上位信号に対する足切り（0〜1）")  # 偶然の弱い一致を除けるようにする。
    parser.add_argument("--json", action="store_true", help="機械処理しやすい JSON で表示")  # CLI を他処理へ接続しやすくする。


def create_parser() -> argparse.ArgumentParser:  # CLI 全体の引数定義を作る。
    parser = argparse.ArgumentParser(prog="ragcore", description="標準ライブラリ中心の説明可能な実用 RAG")  # 最上位パーサーを作る。
    commands = parser.add_subparsers(dest="command", required=True)  # サブコマンドを必須にする。
    index_parser = commands.add_parser("index", help=".md/.txt 文書から索引を作る")  # 索引作成を定義する。
    index_parser.add_argument("--docs", type=Path, default=Path("docs"), help="文書ファイルまたはフォルダ")  # 入力場所を設定可能にする。
    index_parser.add_argument("--out", type=Path, default=Path("store/rag.db"), help="保存する SQLite 索引")  # 出力場所を設定可能にする。
    index_parser.add_argument("--chunk-size", type=int, default=700, help="一チャンクの最大文字数")  # 文脈幅を設定可能にする。
    index_parser.add_argument("--overlap", type=int, default=120, help="隣接チャンクへ残す文字数")  # 境界重複を設定可能にする。
    search_parser = commands.add_parser("search", help="関連する根拠を検索する")  # 検索を定義する。
    search_parser.add_argument("--index", type=Path, default=Path("store/rag.db"), help="読み込む索引")  # 索引場所を設定可能にする。
    add_search_options(search_parser)  # 共通検索設定を追加する。
    ask_parser = commands.add_parser("ask", help="検索結果から引用付き回答を作る")  # 質問回答を定義する。
    ask_parser.add_argument("--index", type=Path, default=Path("store/rag.db"), help="読み込む索引")  # 索引場所を設定可能にする。
    add_search_options(ask_parser)  # 共通検索設定を追加する。
    ask_parser.add_argument("--provider", choices=("extractive", "llm"), default="extractive", help="回答方式（既定: 外部不要）")  # 任意 LLM を明示選択にする。
    ask_parser.add_argument("--endpoint", default=os.getenv("RAG_LLM_URL", "http://localhost:11434/v1/chat/completions"), help="OpenAI 互換 API URL")  # ローカル LLM などの URL を設定する。
    ask_parser.add_argument("--model", default=os.getenv("RAG_LLM_MODEL", ""), help="モデル名または RAG_LLM_MODEL")  # モデル名を履歴外からも受け取る。
    eval_parser = commands.add_parser("eval", help="JSONL 正解セットで検索品質を測る")  # 品質評価を定義する。
    eval_parser.add_argument("--index", type=Path, default=Path("store/rag.db"), help="評価する索引")  # 索引場所を設定可能にする。
    eval_parser.add_argument("--cases", type=Path, default=Path("eval/cases.jsonl"), help="評価ケース JSONL")  # 正解セット場所を設定可能にする。
    eval_parser.add_argument("-k", "--top-k", type=int, default=5, help="Recall を測る上位件数")  # 評価順位幅を設定可能にする。
    serve_parser = commands.add_parser("serve", help="検索と回答の JSON API を起動する")  # HTTP API を定義する。
    serve_parser.add_argument("--index", type=Path, default=Path("store/rag.db"), help="公開する索引")  # 索引場所を設定可能にする。
    serve_parser.add_argument("--host", default="127.0.0.1", help="待ち受け先（既定: ローカルのみ）")  # 安全な既定値で公開範囲を決める。
    serve_parser.add_argument("--port", type=int, default=8080, help="待ち受けポート")  # ポートを設定可能にする。
    info_parser = commands.add_parser("info", help="索引件数、作成時刻、指紋を見る")  # 索引診断を定義する。
    info_parser.add_argument("--index", type=Path, default=Path("store/rag.db"), help="確認する索引")  # 索引場所を設定可能にする。
    inspect_parser = commands.add_parser("inspect", help="文字列から生まれる検索特徴を見る")  # 特徴観察を定義する。
    inspect_parser.add_argument("-t", "--text", required=True, help="観察する文字列")  # 対象文字列を必須にする。
    inspect_parser.add_argument("--limit", type=int, default=40, help="表示する特徴数")  # 表示量を調整可能にする。
    demo_parser = commands.add_parser("demo", help="同梱文書で構築から回答まで試す")  # 一手実行を定義する。
    demo_parser.add_argument("-q", "--query", default="返品できる期限と条件は？", help="デモ文書への質問")  # 分かりやすい既定質問を用意する。
    return parser  # 完成したパーサーを返す。


def run(args: argparse.Namespace) -> int:  # 選ばれたサブコマンドを実行する。
    if args.command == "index":  # 索引作成を処理する。
        index = build_index(args.docs, args.out, args.chunk_size, args.overlap)  # 文書から新しい索引を作る。
        info = index.info()  # 作成結果の規模を読む。
        print(f"索引を保存しました: {index.path}（{info['chunk_count']} chunks）")  # 保存場所と件数を示す。
        return 0  # 正常終了する。
    if args.command == "demo":  # 同梱デモを処理する。
        root = Path(__file__).resolve().parent.parent  # インストール前でもプロジェクトを見つける。
        with tempfile.TemporaryDirectory() as directory:  # 実行後に自動削除される場所を作る。
            index = build_index(root / "docs", Path(directory) / "demo.db", 700, 120)  # サンプル文書を一時索引にする。
            hits = index.search(args.query, top_k=5)  # デモ質問を検索する。
            print(ExtractiveAnswerer().answer(args.query, hits))  # 外部不要の回答を表示する。
            print("\n--- 検索詳細 ---")  # 回答と内部結果を分ける。
            print_hits(hits)  # 原文とスコアを表示する。
        return 0  # 正常終了する。
    if args.command == "inspect":  # 特徴観察を処理する。
        counts = Counter(analyze(args.text))  # 実際の解析器で特徴頻度を数える。
        print(f"normalized: {normalize(args.text)}")  # 正規化後の文字列を示す。
        for term, count in counts.most_common(args.limit):  # 頻度順に指定件数だけ表示する。
            print(f"{term:<18} count={count}")  # 特徴名と回数を示す。
        return 0  # 正常終了する。
    index = SearchIndex(args.index)  # 残りの処理で使う索引を参照する。
    if args.command == "info":  # 索引診断を処理する。
        print(json.dumps(index.info(), ensure_ascii=False, indent=2))  # 全メタデータを JSON 表示する。
        return 0  # 正常終了する。
    if args.command == "eval":  # 品質評価を処理する。
        report = evaluate(index, load_cases(args.cases), args.top_k)  # 正解セットで通常検索を測る。
        print(json.dumps(report, ensure_ascii=False, indent=2))  # 比較しやすい JSON で表示する。
        return 0  # 正常終了する。
    if args.command == "serve":  # HTTP API を処理する。
        index.info()  # 起動前に索引形式を検証する。
        serve(index, args.host, args.port)  # 停止要求まで API を動かす。
        return 0  # 正常停止後に終了する。
    hits = retrieve(index, args)  # search または ask の共通検索を行う。
    if args.command == "search":  # 検索だけを処理する。
        if args.json:  # JSON 表示が指定されたか確認する。
            payload = {"hits": [hit_record(number, hit) for number, hit in enumerate(hits, start=1)], "stats": index.last_stats}  # 結果と処理量をまとめる。
            print(json.dumps(payload, ensure_ascii=False, indent=2))  # UTF-8 JSON を表示する。
        else:  # 人向け表示を処理する。
            print_hits(hits)  # 根拠原文と内訳を表示する。
            print(f"\n検索統計: {index.last_stats}")  # 全件数と採点候補数を示す。
        return 0  # 正常終了する。
    if args.provider == "extractive":  # 外部不要方式か確認する。
        answer = ExtractiveAnswerer().answer(args.query, hits)  # 根拠文を抽出する。
    else:  # 任意 LLM 方式を処理する。
        api_key = os.getenv("RAG_LLM_API_KEY", "")  # 秘密をコマンド履歴へ残さず読む。
        answer = LlmAnswerer(args.endpoint, args.model, api_key).answer(args.query, hits)  # 根拠限定の自然文回答を作る。
    if args.json:  # JSON 表示が指定されたか確認する。
        payload = {"answer": answer, "hits": [hit_record(number, hit) for number, hit in enumerate(hits, start=1)], "stats": index.last_stats}  # 回答、根拠、処理量をまとめる。
        print(json.dumps(payload, ensure_ascii=False, indent=2))  # UTF-8 JSON を表示する。
    else:  # 人向け表示を処理する。
        print(answer)  # 引用付き回答を表示する。
        print("\n出典:")  # 回答と出典一覧を分ける。
        for number, hit in enumerate(hits, start=1):  # 引用番号と位置を対応付ける。
            print(f"[S{number}] {hit.chunk.source} / {hit.chunk.heading} / chars {hit.chunk.start}:{hit.chunk.end}")  # 検証可能な出典位置を示す。
    return 0  # 正常終了する。


def main() -> None:  # コンソールから呼ばれる入口を定義する。
    parser = create_parser()  # 引数定義を作る。
    try:  # 利用者向けの短いエラーへ変換する。
        exit_code = run(parser.parse_args())  # 引数を解析して処理する。
    except (FileNotFoundError, ValueError, sqlite3.Error, json.JSONDecodeError, urllib.error.URLError) as error:  # 入力、索引、接続の代表的失敗を捕捉する。
        print(f"エラー: {error}", file=sys.stderr)  # 追跡表示なしで原因を示す。
        exit_code = 2  # CLI 利用エラーとして終了する。
    raise SystemExit(exit_code)  # OS へ終了コードを返す。
