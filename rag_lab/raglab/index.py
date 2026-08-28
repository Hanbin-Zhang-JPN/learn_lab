"""TF-IDF、BM25、RRF、MMR を組み合わせた検索索引を実装する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈を実行時評価から切り離す。

import json  # 索引を人が読める形式で保存する。
import math  # BM25 の IDF を計算する。
from collections import Counter  # BM25 用の語頻度を数える。
from dataclasses import asdict  # チャンクを JSON 化可能な辞書へ変える。
from pathlib import Path  # 索引ファイルを安全に扱う。

from .model import Chunk, SearchHit  # 共通データ構造を読み込む。
from .vector import SparseVector, TfidfVectorizer, analyze, cosine_similarity  # 自前のベクトル処理を読み込む。


class RagIndex:  # チャンク、ベクトル、統計量を一つにまとめる。
    FORMAT_VERSION = 1  # 保存形式の互換性番号を定義する。

    def __init__(self) -> None:  # 空の索引を初期化する。
        self.chunks: list[Chunk] = []  # 文書チャンクを文書順で保持する。
        self.vectorizer = TfidfVectorizer()  # TF-IDF 変換器を用意する。
        self.vectors: list[SparseVector] = []  # 各チャンクの疎ベクトルを保持する。
        self.term_counts: list[dict[str, int]] = []  # BM25 用の特徴頻度を保持する。
        self.document_frequency: dict[str, int] = {}  # BM25 用の文書頻度を保持する。
        self.lengths: list[int] = []  # 各チャンクの特徴数を保持する。
        self.average_length = 0.0  # BM25 の平均文書長を保持する。

    def build(self, chunks: list[Chunk]) -> None:  # チャンク列から索引を構築する。
        if not chunks:  # 空の索引構築を防ぐ。
            raise ValueError("索引の構築には一件以上のチャンクが必要です。")  # 原因を明確に伝える。
        self.chunks = list(chunks)  # 呼び出し側のリストから独立させる。
        texts = [chunk.text for chunk in self.chunks]  # ベクトル化対象の本文を取り出す。
        self.vectors = self.vectorizer.fit_transform(texts)  # 共通 IDF で全チャンクをベクトル化する。
        counters = [Counter(analyze(text)) for text in texts]  # BM25 用の特徴頻度を数える。
        self.term_counts = [dict(counter) for counter in counters]  # JSON 保存しやすい辞書へ変える。
        self.lengths = [sum(counter.values()) for counter in counters]  # 各チャンクの特徴総数を求める。
        self.average_length = sum(self.lengths) / len(self.lengths)  # 文書長の平均を求める。
        frequencies: Counter[str] = Counter()  # 文書頻度の格納先を用意する。
        for counter in counters:  # 各チャンクの特徴集合を調べる。
            frequencies.update(counter.keys())  # 一チャンクにつき一回だけ文書頻度へ加える。
        self.document_frequency = dict(frequencies)  # 通常辞書として保持する。

    def search(self, query: str, top_k: int = 4, vector_weight: float = 1.0, bm25_weight: float = 1.0, mmr_lambda: float = 0.75, min_signal_ratio: float = 0.15) -> list[SearchHit]:  # ハイブリッド検索を行う。
        if not query.strip():  # 空の質問を防ぐ。
            raise ValueError("検索質問を入力してください。")  # 原因を明確に伝える。
        if top_k < 1:  # 不正な件数を防ぐ。
            raise ValueError("top_k は 1 以上にしてください。")  # 利用条件を明確に伝える。
        if not 0.0 <= mmr_lambda <= 1.0:  # MMR の補間範囲を検証する。
            raise ValueError("mmr_lambda は 0.0 から 1.0 の範囲にしてください。")  # 利用条件を明確に伝える。
        if not 0.0 <= min_signal_ratio <= 1.0:  # 相対しきい値の範囲を検証する。
            raise ValueError("min_signal_ratio は 0.0 から 1.0 の範囲にしてください。")  # 利用条件を明確に伝える。
        if not self.chunks:  # 未構築索引の利用を防ぐ。
            raise ValueError("索引が空です。先に build または load を実行してください。")  # 次の行動を示す。
        query_vector = self.vectorizer.transform_one(query)  # 質問を文書と同じ TF-IDF 空間へ写す。
        cosine_scores = [cosine_similarity(query_vector, vector) for vector in self.vectors]  # 全チャンクとのコサイン類似度を求める。
        query_terms = analyze(query)  # BM25 で使う質問特徴を作る。
        bm25_scores = [self._bm25(query_terms, index) for index in range(len(self.chunks))]  # 全チャンクの BM25 を求める。
        cosine_order = [index for index, score in sorted(enumerate(cosine_scores), key=lambda pair: pair[1], reverse=True) if score > 0.0]  # ベクトル順位を作る。
        bm25_order = [index for index, score in sorted(enumerate(bm25_scores), key=lambda pair: pair[1], reverse=True) if score > 0.0]  # BM25 順位を作る。
        if not cosine_order and not bm25_order:  # どの特徴も一致しない場合を扱う。
            return []  # 根拠なしとして空の結果を返す。
        cosine_ranks = {index: rank for rank, index in enumerate(cosine_order, start=1)}  # チャンク番号からベクトル順位を引けるようにする。
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}  # チャンク番号から BM25 順位を引けるようにする。
        all_candidates = set(cosine_ranks) | set(bm25_ranks)  # 二つの検索器の候補を合流する。
        best_cosine = max(cosine_scores, default=0.0)  # ベクトル検索の最上位信号を得る。
        best_bm25 = max(bm25_scores, default=0.0)  # BM25 検索の最上位信号を得る。
        candidate_indexes = {index for index in all_candidates if cosine_scores[index] >= best_cosine * min_signal_ratio or bm25_scores[index] >= best_bm25 * min_signal_ratio}  # 最上位に比べ極端に弱い一致を除く。
        rrf_scores: dict[int, float] = {}  # RRF スコアの格納先を用意する。
        for index in candidate_indexes:  # 各候補の順位を融合する。
            vector_part = vector_weight / (60.0 + cosine_ranks[index]) if index in cosine_ranks else 0.0  # ベクトル順位の寄与を求める。
            bm25_part = bm25_weight / (60.0 + bm25_ranks[index]) if index in bm25_ranks else 0.0  # BM25 順位の寄与を求める。
            rrf_scores[index] = vector_part + bm25_part  # 尺度に依存しない順位融合値を保存する。
        ordered = sorted(candidate_indexes, key=lambda index: rrf_scores[index], reverse=True)  # 融合順位を作る。
        pool = ordered[: max(20, top_k * 8)]  # MMR に渡す候補数を制限する。
        maximum = max(rrf_scores[index] for index in pool)  # 表示用相対値の基準を得る。
        relevance = {index: rrf_scores[index] / maximum for index in pool}  # 最上位を 1 とする関連度へ直す。
        selected = self._mmr(pool, relevance, top_k, mmr_lambda)  # 内容の重複を抑えて最終候補を選ぶ。
        return [SearchHit(self.chunks[index], relevance[index], cosine_scores[index], bm25_scores[index]) for index in selected]  # 内訳付き結果を返す。

    def _bm25(self, query_terms: list[str], document_index: int, k1: float = 1.5, b: float = 0.75) -> float:  # 一チャンクの BM25 を計算する。
        counts = self.term_counts[document_index]  # 対象チャンクの特徴頻度を得る。
        length = self.lengths[document_index]  # 対象チャンクの特徴総数を得る。
        total = len(self.chunks)  # 全チャンク数を得る。
        safe_average = self.average_length or 1.0  # ゼロ除算を避ける平均長を用意する。
        score = 0.0  # スコアをゼロから加算する。
        for term, query_frequency in Counter(query_terms).items():  # 質問内の重複も穏やかに反映する。
            frequency = counts.get(term, 0)  # 対象チャンク内の頻度を得る。
            if frequency == 0:  # 特徴が存在しない場合を除く。
                continue  # 次の質問特徴へ進む。
            document_frequency = self.document_frequency.get(term, 0)  # 何チャンクに出たかを得る。
            idf = math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))  # BM25 の IDF を求める。
            denominator = frequency + k1 * (1.0 - b + b * length / safe_average)  # 文書長補正を含む分母を求める。
            score += idf * (frequency * (k1 + 1.0) / denominator) * (1.0 + math.log(query_frequency))  # 飽和 TF と質問頻度を加える。
        return score  # 対象チャンクの BM25 スコアを返す。

    def _mmr(self, pool: list[int], relevance: dict[int, float], top_k: int, strength: float) -> list[int]:  # 関連性と多様性を両立させる。
        remaining = list(pool)  # 未選択候補を融合順で保持する。
        selected: list[int] = []  # 選択済み候補の格納先を用意する。
        while remaining and len(selected) < top_k:  # 必要件数または候補切れまで繰り返す。
            best_index = remaining[0]  # 同点時に融合順位が高い候補を優先する。
            best_score = float("-inf")  # 比較開始用の最小値を置く。
            for index in remaining:  # 未選択候補を一つずつ評価する。
                redundancy = max((cosine_similarity(self.vectors[index], self.vectors[chosen]) for chosen in selected), default=0.0)  # 選択済みとの最大類似度を求める。
                mmr_score = strength * relevance[index] - (1.0 - strength) * redundancy  # 関連性から重複罰を引く。
                if mmr_score > best_score:  # 現在までの最良候補か確認する。
                    best_index = index  # 最良候補番号を更新する。
                    best_score = mmr_score  # 最良 MMR 値を更新する。
            selected.append(best_index)  # 最良候補を結果へ加える。
            remaining.remove(best_index)  # 同じ候補の再選択を防ぐ。
        return selected  # 多様化された候補番号を返す。

    def save(self, path: Path) -> None:  # 索引全体を JSON ファイルへ保存する。
        if not self.chunks:  # 空索引の保存を防ぐ。
            raise ValueError("空の索引は保存できません。")  # 原因を明確に伝える。
        payload = {"format_version": self.FORMAT_VERSION, "chunks": [asdict(chunk) for chunk in self.chunks], "idf": self.vectorizer.idf, "vectors": self.vectors, "term_counts": self.term_counts, "document_frequency": self.document_frequency, "lengths": self.lengths, "average_length": self.average_length}  # 復元に必要な値をまとめる。
        path.parent.mkdir(parents=True, exist_ok=True)  # 保存先フォルダを必要時だけ作る。
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")  # 日本語を保った圧縮 JSON として書く。

    @classmethod  # ファイルから新しいインスタンスを作る入口にする。
    def load(cls, path: Path) -> RagIndex:  # 保存済み JSON 索引を復元する。
        payload = json.loads(path.read_text(encoding="utf-8"))  # UTF-8 JSON を辞書へ読み込む。
        if payload.get("format_version") != cls.FORMAT_VERSION:  # 保存形式が対応版か確認する。
            raise ValueError("索引形式の版が対応していません。文書から再構築してください。")  # 安全な復旧方法を示す。
        index = cls()  # 空の索引インスタンスを作る。
        index.chunks = [Chunk(**item) for item in payload["chunks"]]  # チャンクの型を復元する。
        index.vectorizer = TfidfVectorizer({term: float(value) for term, value in payload["idf"].items()})  # IDF を浮動小数として復元する。
        index.vectors = [{term: float(value) for term, value in vector.items()} for vector in payload["vectors"]]  # 疎ベクトルを復元する。
        index.term_counts = [{term: int(value) for term, value in counts.items()} for counts in payload["term_counts"]]  # BM25 頻度を整数として復元する。
        index.document_frequency = {term: int(value) for term, value in payload["document_frequency"].items()}  # 文書頻度を整数として復元する。
        index.lengths = [int(value) for value in payload["lengths"]]  # 文書長を整数として復元する。
        index.average_length = float(payload["average_length"])  # 平均長を浮動小数として復元する。
        return index  # 検索可能な索引を返す。
