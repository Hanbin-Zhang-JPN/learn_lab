"""合成文書で索引構築時間と検索遅延を再現可能に測る。"""  # このスクリプトの目的を示す。

from __future__ import annotations  # 型注釈の評価を遅らせる。

import argparse  # 文書数と質問数を引数で受け取る。
import statistics  # 検索時間の中央値を求める。
import sys  # インストール前のパッケージ位置を登録する。
import tempfile  # 計測後に索引を自動削除する。
import time  # 構築と検索の経過時間を測る。
from pathlib import Path  # 一時索引の場所を扱う。

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # このリポジトリの最上位を求める。
sys.path.insert(0, str(PROJECT_ROOT))  # pip install 前でも同梱パッケージを読めるようにする。

from ragcore.model import Chunk  # 合成チャンク型を読み込む。
from ragcore.search import SearchIndex  # 計測対象の索引を読み込む。


def make_chunks(count: int) -> list[Chunk]:  # 指定件数の識別可能な合成文書を作る。
    chunks: list[Chunk] = []  # 作成したチャンクをためる。
    for number in range(count):  # 一件ずつ安定した内容を作る。
        code = f"ITEM-{number:06d}"  # 文書ごとに異なる製品番号を作る。
        days = 30 + number % 180  # 答えも文書ごとに変える。
        text = f"製品番号 {code} の保守期間は購入日から {days} 日です。点検申請には製品番号と購入証明が必要です。"  # 検索可能な本文を作る。
        chunks.append(Chunk(code, code, f"manual/{code}.md", f"製品 {code}", "保守", 0, 0, len(text), text))  # 完成チャンクを保存する。
    return chunks  # 合成チャンク列を返す。


def percentile(values: list[float], ratio: float) -> float:  # nearest-rank 法の値を求める。
    ordered = sorted(values)  # 小さい順に並べる。
    index = max(0, min(len(ordered) - 1, int(len(ordered) * ratio + 0.999999) - 1))  # 範囲内の切り上げ位置を求める。
    return ordered[index]  # 指定位置の値を返す。


def main() -> None:  # コマンド実行の入口を定義する。
    parser = argparse.ArgumentParser(description="rag_core のローカル性能を測る")  # 引数定義を作る。
    parser.add_argument("--chunks", type=int, default=5000, help="合成チャンク数")  # 索引規模を設定可能にする。
    parser.add_argument("--queries", type=int, default=100, help="検索回数")  # 反復数を設定可能にする。
    args = parser.parse_args()  # コマンド引数を解析する。
    if args.chunks < 1 or args.queries < 1:  # 正の件数か確認する。
        raise SystemExit("--chunks と --queries は 1 以上にしてください。")  # 不正値を短く伝える。
    chunks = make_chunks(args.chunks)  # 計測対象の合成データを作る。
    with tempfile.TemporaryDirectory() as directory:  # 計測後に消える場所を作る。
        started = time.perf_counter()  # 構築開始時刻を取る。
        index = SearchIndex.build(Path(directory) / "bench.db", chunks)  # 通常と同じ索引を構築する。
        build_seconds = time.perf_counter() - started  # 構築時間を秒で求める。
        latencies: list[float] = []  # 各検索のミリ秒をためる。
        correct = 0  # 最上位正解数を数える。
        for turn in range(args.queries):  # 指定回数だけ検索する。
            number = turn % args.chunks  # 存在する製品番号を順に選ぶ。
            code = f"ITEM-{number:06d}"  # 期待する製品番号を作る。
            query = f"製品番号 {code} の保守期間は？"  # 識別子を含む質問を作る。
            query_started = time.perf_counter()  # 検索開始時刻を取る。
            hits = index.search(query, top_k=3)  # 通常と同じ検索を実行する。
            latencies.append((time.perf_counter() - query_started) * 1000.0)  # 経過時間をミリ秒で保存する。
            correct += int(bool(hits) and hits[0].chunk.source == f"manual/{code}.md")  # 最上位が正解なら加算する。
        size_mb = index.path.stat().st_size / (1024 * 1024)  # 索引ファイル容量を MiB で求める。
    print(f"chunks       : {args.chunks}")  # 文書規模を表示する。
    print(f"index seconds: {build_seconds:.3f}")  # 構築時間を表示する。
    print(f"index MiB    : {size_mb:.2f}")  # 保存容量を表示する。
    print(f"top1 accuracy: {correct / args.queries:.3f}")  # 合成質問の正解率を表示する。
    print(f"latency p50  : {statistics.median(latencies):.3f} ms")  # 中央検索時間を表示する。
    print(f"latency p95  : {percentile(latencies, 0.95):.3f} ms")  # 95 パーセンタイルを表示する。
    print(f"last stats   : {index.last_stats}")  # 全件と採点候補の差を表示する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    main()  # 性能計測を開始する。
