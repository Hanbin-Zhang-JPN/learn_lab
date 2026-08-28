"""索引作成、検索、回答、デモを一つの CLI から操作できるようにする。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈を実行時評価から切り離す。

import argparse  # 標準ライブラリだけでコマンド引数を処理する。
import json  # 壊れた索引 JSON の例外型を参照する。
import os  # LLM の接続情報を環境変数から安全に読む。
import sys  # 終了コードと標準エラー出力を扱う。
import urllib.error  # LLM API の接続失敗を説明可能にする。
from collections import Counter  # inspect 表示で特徴量の回数をまとめる。
from pathlib import Path  # 文書と索引のパスを扱う。

from .answer import ExtractiveAnswerer, OpenAICompatibleAnswerer  # 二種類の回答器を読み込む。
from .index import RagIndex  # 検索索引を読み込む。
from .model import SearchHit  # 表示関数の型を読み込む。
from .text import TextChunker, load_documents  # 文書読み込みとチャンク化を読み込む。
from .vector import analyze, normalize_text  # 文字特徴を観察する機能へ使う。


def build_index(documents_path: Path, chunk_size: int, overlap: int) -> RagIndex:  # 文書フォルダから検索索引を作る。
    documents = load_documents(documents_path)  # 対応する文書を安定順で読む。
    chunker = TextChunker(max_chars=chunk_size, overlap_chars=overlap)  # 指定サイズの分割器を作る。
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]  # 全文書を一つのチャンク列へ展開する。
    index = RagIndex()  # 空の索引を用意する。
    index.build(chunks)  # TF-IDF と BM25 の統計を構築する。
    return index  # 検索可能な索引を返す。


def retrieve(index: RagIndex, args: argparse.Namespace) -> list[SearchHit]:  # 共通設定でハイブリッド検索を行う。
    return index.search(query=args.query, top_k=args.top_k, vector_weight=args.vector_weight, bm25_weight=args.bm25_weight, mmr_lambda=args.mmr_lambda, min_signal_ratio=args.min_signal_ratio)  # CLI 設定をそのまま検索器へ渡す。


def print_hits(hits: list[SearchHit]) -> None:  # 検索結果とスコア内訳を読みやすく表示する。
    if not hits:  # 検索結果がない場合を扱う。
        print("一致するチャンクはありません。")  # 空結果を明確に伝える。
        return  # 通常表示を行わず終了する。
    for number, hit in enumerate(hits, start=1):  # 順位付きで各結果を表示する。
        location = f"{hit.chunk.source} / {hit.chunk.heading} / chars {hit.chunk.start}:{hit.chunk.end}"  # 出典位置を一行にまとめる。
        scores = f"hybrid={hit.score:.4f} cosine={hit.cosine:.4f} bm25={hit.bm25:.4f}"  # 三種類のスコアを比較可能にする。
        print(f"\n[S{number}] {location}\n{scores}\n{hit.chunk.text}")  # 根拠ラベル、内訳、本文を表示する。


def add_retrieval_options(parser: argparse.ArgumentParser) -> None:  # search と ask の共通引数を定義する。
    parser.add_argument("-q", "--query", required=True, help="検索または質問する文")  # 質問文字列を必須にする。
    parser.add_argument("-k", "--top-k", type=int, default=4, help="返すチャンク数（既定: 4）")  # 最終候補数を設定する。
    parser.add_argument("--vector-weight", type=float, default=1.0, help="RRF 内の TF-IDF 順位重み")  # ベクトル検索の寄与を調整可能にする。
    parser.add_argument("--bm25-weight", type=float, default=1.0, help="RRF 内の BM25 順位重み")  # 字面検索の寄与を調整可能にする。
    parser.add_argument("--mmr-lambda", type=float, default=0.75, help="MMR の関連性重み（0〜1）")  # 多様性とのバランスを調整可能にする。
    parser.add_argument("--min-signal-ratio", type=float, default=0.15, help="最上位信号に対する相対足切り（0〜1）")  # 弱すぎる文字一致を回答から除けるようにする。


def create_parser() -> argparse.ArgumentParser:  # CLI 全体の引数定義を作る。
    parser = argparse.ArgumentParser(prog="raglab", description="中身を読める日本語 RAG ミニ実装")  # 最上位パーサーを作る。
    subparsers = parser.add_subparsers(dest="command", required=True)  # サブコマンドを必須にする。
    index_parser = subparsers.add_parser("index", help="文書をチャンク化して索引を作る")  # 索引作成コマンドを定義する。
    index_parser.add_argument("--docs", type=Path, default=Path("docs"), help=".md/.txt のファイルまたはフォルダ")  # 入力文書場所を設定する。
    index_parser.add_argument("--out", type=Path, default=Path("store/index.json"), help="保存する索引 JSON")  # 索引の保存先を設定する。
    index_parser.add_argument("--chunk-size", type=int, default=500, help="一チャンクの最大文字数")  # チャンク上限を設定する。
    index_parser.add_argument("--overlap", type=int, default=100, help="隣接チャンクへ残す目標文字数")  # 文脈重複幅を設定する。
    search_parser = subparsers.add_parser("search", help="索引から関連チャンクを検索する")  # 検索コマンドを定義する。
    search_parser.add_argument("--index", type=Path, default=Path("store/index.json"), help="読み込む索引 JSON")  # 検索索引の場所を設定する。
    add_retrieval_options(search_parser)  # 共通の検索設定を追加する。
    ask_parser = subparsers.add_parser("ask", help="検索結果から根拠付き回答を作る")  # 質問回答コマンドを定義する。
    ask_parser.add_argument("--index", type=Path, default=Path("store/index.json"), help="読み込む索引 JSON")  # 回答用索引の場所を設定する。
    add_retrieval_options(ask_parser)  # 共通の検索設定を追加する。
    ask_parser.add_argument("--provider", choices=("extractive", "llm"), default="extractive", help="回答器（既定: 外部不要の抽出型）")  # 回答生成方式を選べるようにする。
    ask_parser.add_argument("--endpoint", default=os.getenv("RAG_LLM_URL", "http://localhost:11434/v1/chat/completions"), help="OpenAI 互換 chat/completions URL")  # ローカル LLM などの URL を設定する。
    ask_parser.add_argument("--model", default=os.getenv("RAG_LLM_MODEL", ""), help="LLM モデル名、または RAG_LLM_MODEL")  # モデル名を安全に受け取る。
    demo_parser = subparsers.add_parser("demo", help="同梱文書で構築から回答まで試す")  # 一手で試せるデモを定義する。
    demo_parser.add_argument("-q", "--query", default="返品できる期間と条件は何ですか？", help="デモ文書への質問")  # 既定質問を用意する。
    inspect_parser = subparsers.add_parser("inspect", help="テキストから生まれる特徴量とベクトルを見る")  # ベクトル化の観察コマンドを定義する。
    inspect_parser.add_argument("-t", "--text", required=True, help="特徴量を確認するテキスト")  # 観察対象の文字列を必須にする。
    inspect_parser.add_argument("--index", type=Path, help="IDF 重みも見る場合の索引 JSON")  # 学習済み座標の任意入力を受け取る。
    inspect_parser.add_argument("--limit", type=int, default=30, help="表示する特徴数")  # 長い表示を制限できるようにする。
    return parser  # 完成した引数定義を返す。


def run(args: argparse.Namespace) -> int:  # 解析済み引数に対応する処理を実行する。
    if args.command == "index":  # 索引作成が選ばれた場合を扱う。
        index = build_index(args.docs, args.chunk_size, args.overlap)  # 文書から索引を構築する。
        index.save(args.out)  # 指定場所へ索引を保存する。
        print(f"索引を保存しました: {args.out}（{len(index.chunks)} chunks / {len(index.vectorizer.idf)} features）")  # 構築規模を表示する。
        return 0  # 正常終了コードを返す。
    if args.command == "demo":  # 同梱デモが選ばれた場合を扱う。
        project_root = Path(__file__).resolve().parent.parent  # インストール前でも同梱文書を見つける。
        index = build_index(project_root / "docs", 500, 100)  # 既定設定で一時索引を作る。
        hits = index.search(args.query, top_k=4)  # デモ質問を検索する。
        print(ExtractiveAnswerer().answer(args.query, hits))  # 外部不要の回答を表示する。
        print("\n--- 検索詳細 ---")  # 回答と検索内部を区切る。
        print_hits(hits)  # スコア内訳と原文を表示する。
        return 0  # 正常終了コードを返す。
    if args.command == "inspect":  # 特徴量観察が選ばれた場合を扱う。
        counts = Counter(analyze(args.text))  # 正規化と n-gram 化を実行する。
        print(f"normalized: {normalize_text(args.text)}")  # 検索時の正規化結果を表示する。
        print("features:")  # 生の特徴量一覧を始める。
        for term, count in counts.most_common(args.limit):  # 頻度順に指定件数だけ表示する。
            print(f"  {term:<16} count={count}")  # 特徴名と出現回数を表示する。
        if args.index:  # 学習済み索引も指定されたか確認する。
            inspect_index = RagIndex.load(args.index)  # 索引から IDF 空間を復元する。
            vector = inspect_index.vectorizer.transform_one(args.text)  # 対象文を学習済み空間へ写す。
            print("tf-idf vector:")  # 重み付き疎ベクトル表示を始める。
            for term, value in sorted(vector.items(), key=lambda item: item[1], reverse=True)[:args.limit]:  # 大きい座標から指定件数だけ表示する。
                print(f"  {term:<16} value={value:.6f} idf={inspect_index.vectorizer.idf[term]:.6f}")  # 正規化値と IDF を表示する。
        return 0  # 正常終了コードを返す。
    index = RagIndex.load(args.index)  # search または ask 用の索引を読む。
    hits = retrieve(index, args)  # 共通ハイブリッド検索を実行する。
    if args.command == "search":  # 検索だけが選ばれた場合を扱う。
        print_hits(hits)  # スコア内訳と原文を表示する。
        return 0  # 正常終了コードを返す。
    if args.provider == "extractive":  # 外部不要の回答器が選ばれた場合を扱う。
        answer = ExtractiveAnswerer().answer(args.query, hits)  # 関連文を抽出して回答にする。
    else:  # OpenAI 互換 LLM が選ばれた場合を扱う。
        if not args.model:  # 必須のモデル名があるか確認する。
            raise ValueError("--provider llm では --model または RAG_LLM_MODEL が必要です。")  # 設定方法を明確に伝える。
        api_key = os.getenv("RAG_LLM_API_KEY", "")  # キーをコマンド履歴へ残さず環境変数から読む。
        answerer = OpenAICompatibleAnswerer(args.endpoint, args.model, api_key)  # LLM 回答器を接続設定付きで作る。
        answer = answerer.answer(args.query, hits)  # 検索根拠を使って回答を生成する。
    print(answer)  # 最終回答を表示する。
    print("\n出典:")  # 回答の後に出典一覧を示す。
    for number, hit in enumerate(hits, start=1):  # 根拠番号とファイル位置を対応付ける。
        print(f"[S{number}] {hit.chunk.source} / {hit.chunk.heading} / chars {hit.chunk.start}:{hit.chunk.end}")  # 検証可能な位置を表示する。
    return 0  # 正常終了コードを返す。


def main() -> None:  # コンソールから呼ばれる入口を定義する。
    parser = create_parser()  # CLI の引数定義を作る。
    try:  # 利用者向けの短いエラー表示へ変換する。
        exit_code = run(parser.parse_args())  # 引数を解析して処理を実行する。
    except (FileNotFoundError, ValueError, json.JSONDecodeError, urllib.error.URLError) as error:  # 入力、索引、接続の代表的な失敗を捕捉する。
        print(f"エラー: {error}", file=sys.stderr)  # 詳細な追跡表示なしで原因を示す。
        exit_code = 2  # CLI の利用エラーとして終了する。
    raise SystemExit(exit_code)  # OS へ終了コードを返す。
