"""小さな正解セットで検索品質と応答時間を測る。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈の評価を遅らせる。

import json  # JSON Lines の評価ケースを読む。
import statistics  # 中央値を標準ライブラリで求める。
import time  # 一回ごとの検索時間を測る。
from dataclasses import dataclass  # 評価ケースを簡潔に表す。
from pathlib import Path  # 評価ファイルを扱う。

from .search import SearchIndex  # 評価対象の検索索引を読み込む。


@dataclass(frozen=True)  # 読み込み後の正解ケースを不変にする。
class EvalCase:  # 一つの質問と期待出典を表す。
    query: str  # 評価する質問を持つ。
    expected_sources: tuple[str, ...]  # 正解とする出典名を持つ。


def load_cases(path: Path) -> list[EvalCase]:  # JSONL から評価ケースを読む。
    cases: list[EvalCase] = []  # 有効なケースをためる。
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):  # 行番号付きで読む。
        if not line.strip():  # 空行か確認する。
            continue  # 空行を無視する。
        try:  # JSON の誤りへ行番号を付ける。
            record = json.loads(line)  # 一行を辞書へ変える。
        except json.JSONDecodeError as error:  # 壊れた JSON を捕捉する。
            raise ValueError(f"評価ファイル {line_number} 行目の JSON が不正です。") from error  # 修正場所を示す。
        query = str(record.get("query", "")).strip()  # 質問を文字列として得る。
        sources = tuple(str(item).strip() for item in record.get("expected_sources", []) if str(item).strip())  # 空でない期待出典を集める。
        if not query or not sources:  # 必須値を確認する。
            raise ValueError(f"評価ファイル {line_number} 行目には query と expected_sources が必要です。")  # 不足箇所を示す。
        cases.append(EvalCase(query, sources))  # 型付きケースへ変換する。
    if not cases:  # ケースが一件もないか確認する。
        raise ValueError("評価ケースがありません。")  # 空評価を防ぐ。
    return cases  # 読み込み順で返す。


def evaluate(index: SearchIndex, cases: list[EvalCase], top_k: int = 5) -> dict[str, float | int]:  # Recall、MRR、時間を集計する。
    recalls: list[float] = []  # ケースごとの Recall をためる。
    reciprocal_ranks: list[float] = []  # ケースごとの逆順位をためる。
    latencies: list[float] = []  # ケースごとのミリ秒をためる。
    for case in cases:  # 各質問を独立に評価する。
        started = time.perf_counter()  # 高精度時計で開始時刻を取る。
        hits = index.search(case.query, top_k=top_k)  # 通常と同じ検索経路を実行する。
        latencies.append((time.perf_counter() - started) * 1000.0)  # 経過時間をミリ秒で保存する。
        returned = [hit.chunk.source for hit in hits]  # 順位順の出典名を得る。
        expected = set(case.expected_sources)  # 重複しない正解集合を作る。
        found = expected & set(returned)  # 上位 k 内で見つかった正解を得る。
        recalls.append(len(found) / len(expected))  # 正解の回収率を保存する。
        first = next((rank for rank, source in enumerate(returned, start=1) if source in expected), None)  # 最初の正解順位を探す。
        reciprocal_ranks.append(1.0 / first if first else 0.0)  # 見つからない場合をゼロにする。
    ordered_latency = sorted(latencies)  # パーセンタイル用に時間を並べる。
    p95_index = min(len(ordered_latency) - 1, math_index(len(ordered_latency), 0.95))  # 小標本でも範囲内の位置を得る。
    return {"cases": len(cases), "top_k": top_k, "recall_at_k": sum(recalls) / len(recalls), "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks), "latency_p50_ms": statistics.median(latencies), "latency_p95_ms": ordered_latency[p95_index]}  # 主要指標を一つの辞書で返す。


def math_index(count: int, ratio: float) -> int:  # nearest-rank 法のゼロ始まり位置を求める。
    return max(0, int((count * ratio + 0.999999)) - 1)  # 切り上げ相当の位置を返す。
