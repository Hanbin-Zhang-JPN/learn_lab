"""SQLite 転置索引と手書きのハイブリッド順位付けを実装する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈の評価を遅らせる。

import hashlib  # 索引内容の指紋を作る。
import math  # IDF、対数 TF、ベクトル長を計算する。
import os  # 完成した索引を原子的に置き換える。
import secrets  # 衝突しにくい一時ファイル名を作る。
import sqlite3  # 標準ライブラリで永続的な表を扱う。
import threading  # 並行 API 要求ごとに検索統計を分離する。
from collections import Counter  # 特徴頻度と候補信号を数える。
from contextlib import closing  # SQLite 接続を確実に閉じる。
from datetime import datetime, timezone  # 索引作成時刻を UTC で保存する。
from pathlib import Path  # 索引ファイルの場所を扱う。
from typing import Iterator  # 小分け処理の戻り型を示す。

from .model import Chunk, SearchHit  # 共通データ型を読み込む。
from .terms import analyze, keyword_text, normalize, term_counts  # 透明な文字特徴処理を読み込む。


class SearchIndex:  # SQLite ファイル一つを検索索引として扱う。
    SCHEMA_VERSION = "1"  # 保存表の互換性番号を定義する。

    def __init__(self, path: Path) -> None:  # 索引ファイルの場所を受け取る。
        self.path = path.resolve()  # 呼び出し位置に依存しない絶対パスを保持する。
        self._local = threading.local()  # スレッドごとの小さな診断領域を作る。

    @property  # 読み取り属性として公開する。
    def last_stats(self) -> dict[str, int | float]:  # 現在要求の検索処理量を返す。
        return getattr(self._local, "last_stats", {})  # 未検索時は空辞書を返す。

    @last_stats.setter  # 内部検索から更新できるようにする。
    def last_stats(self, value: dict[str, int | float]) -> None:  # 現在要求の診断値を保存する。
        self._local.last_stats = value  # 他スレッドと共有せず保存する。

    @classmethod  # 構築済みインスタンスを返す入口にする。
    def build(cls, path: Path, chunks: list[Chunk]) -> SearchIndex:  # チャンク列から新しい索引を原子的に作る。
        if not chunks:  # 空の索引作成を防ぐ。
            raise ValueError("索引には一件以上のチャンクが必要です。")  # 原因を明確に伝える。
        output = path.resolve()  # 保存先を絶対パスにする。
        output.parent.mkdir(parents=True, exist_ok=True)  # 必要な保存フォルダだけを作る。
        temporary = output.with_name(f".{output.name}.{secrets.token_hex(6)}.tmp")  # 同じ場所に一時 DB を作る。
        try:  # 失敗時に一時 DB を残さないよう囲む。
            cls._write_database(temporary, chunks)  # 完全な新規 DB を先に構築する。
            os.replace(temporary, output)  # 完成後だけ旧索引と原子的に入れ替える。
        finally:  # 成功と失敗の両方で後始末する。
            if temporary.exists():  # 一時 DB が残っているか確認する。
                temporary.unlink()  # この処理が作った一時 DB だけを削除する。
        return cls(output)  # 読み取り用の索引を返す。

    @classmethod  # DB 書き込みを構築入口から分離する。
    def _write_database(cls, path: Path, chunks: list[Chunk]) -> None:  # SQLite の表と転置索引を作る。
        body_counts = [term_counts(chunk.text) for chunk in chunks]  # 本文の特徴頻度を一度だけ数える。
        head_counts = [term_counts(f"{chunk.title} {chunk.heading}") for chunk in chunks]  # タイトルと見出しの頻度を数える。
        document_frequency: Counter[str] = Counter()  # 各特徴が何チャンクに出たか数える。
        for body, head in zip(body_counts, head_counts):  # 本文と見出しを同じチャンクごとに扱う。
            document_frequency.update(set(body) | set(head))  # 一チャンク内の重複を一回として数える。
        total = len(chunks)  # 全チャンク数を保持する。
        idf = {term: math.log((1.0 + total) / (1.0 + frequency)) + 1.0 for term, frequency in document_frequency.items()}  # 平滑化 TF-IDF を作る。
        lengths = [sum(counter.values()) or 1 for counter in body_counts]  # BM25 用の本文長を求める。
        average_length = sum(lengths) / total  # 全チャンクの平均長を求める。
        corpus_seed = "\n".join(chunk.chunk_id for chunk in chunks)  # チャンク順を含む指紋材料を作る。
        corpus_hash = hashlib.sha256(corpus_seed.encode("utf-8")).hexdigest()  # 索引内容の指紋を作る。
        with closing(sqlite3.connect(path)) as connection:  # 新しい DB 接続を確実に閉じる。
            connection.execute("PRAGMA journal_mode=OFF")  # 一時 DB なので構築中の二重書きを省く。
            connection.execute("PRAGMA synchronous=OFF")  # 最後の原子置換で安全性を確保し構築を速める。
            connection.executescript(cls._schema())  # 小さく明示した表定義を作る。
            metadata = {"schema_version": cls.SCHEMA_VERSION, "chunk_count": str(total), "average_length": repr(average_length), "created_utc": datetime.now(timezone.utc).isoformat(), "corpus_hash": corpus_hash}  # 復元と診断に必要な値をまとめる。
            connection.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", metadata.items())  # メタデータを保存する。
            term_rows = ((term, frequency, idf[term]) for term, frequency in document_frequency.items())  # 特徴統計を遅延生成する。
            connection.executemany("INSERT INTO terms(term, df, idf) VALUES (?, ?, ?)", term_rows)  # 特徴表を一括保存する。
            for chunk, body, head, length in zip(chunks, body_counts, head_counts, lengths):  # チャンク単位で保存する。
                merged = set(body) | set(head)  # 本文または見出しにある特徴を集める。
                norm_square = 0.0  # 文書ベクトル長の二乗を初期化する。
                for term in merged:  # 各座標の重みを計算する。
                    frequency = body.get(term, 0) + 2.0 * head.get(term, 0)  # 見出し一致を二倍にする。
                    weight = (1.0 + math.log(frequency)) * idf[term]  # 対数 TF と IDF を掛ける。
                    norm_square += weight * weight  # ベクトル長へ座標の二乗を足す。
                vector_norm = math.sqrt(norm_square) or 1.0  # ゼロ除算を避けた長さを得る。
                chunk_row = (chunk.chunk_id, chunk.doc_id, chunk.source, chunk.title, chunk.heading, chunk.ordinal, chunk.start, chunk.end, chunk.text, length, vector_norm)  # チャンクの保存値を並べる。
                connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", chunk_row)  # 本文とメタデータを保存する。
                posting_rows = ((term, chunk.chunk_id, body.get(term, 0), head.get(term, 0)) for term in merged)  # 特徴の出現先を遅延生成する。
                connection.executemany("INSERT INTO postings(term, chunk_id, body_tf, head_tf) VALUES (?, ?, ?, ?)", posting_rows)  # 転置リストを保存する。
            connection.commit()  # 完全な索引を一度に確定する。

    @staticmethod  # 固定の表定義であることを示す。
    def _schema() -> str:  # 人が追える SQLite の表定義を返す。
        return """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE terms(term TEXT PRIMARY KEY, df INTEGER NOT NULL, idf REAL NOT NULL) WITHOUT ROWID;
        CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, source TEXT NOT NULL, title TEXT NOT NULL, heading TEXT NOT NULL, ordinal INTEGER NOT NULL, start_pos INTEGER NOT NULL, end_pos INTEGER NOT NULL, text TEXT NOT NULL, body_len INTEGER NOT NULL, vector_norm REAL NOT NULL) WITHOUT ROWID;
        CREATE TABLE postings(term TEXT NOT NULL, chunk_id TEXT NOT NULL, body_tf INTEGER NOT NULL, head_tf INTEGER NOT NULL, PRIMARY KEY(term, chunk_id), FOREIGN KEY(term) REFERENCES terms(term), FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)) WITHOUT ROWID;
        CREATE INDEX postings_chunk ON postings(chunk_id);
        CREATE INDEX chunks_source ON chunks(source);
        """  # 特徴、本文、出現先を分離した単純な構造にする。

    def info(self) -> dict[str, str]:  # 索引の基本情報を返す。
        with closing(self._connect()) as connection:  # 読み取り接続を確実に閉じる。
            rows = connection.execute("SELECT key, value FROM meta").fetchall()  # 全メタデータを読む。
        return {str(row["key"]): str(row["value"]) for row in rows}  # 通常辞書へ変えて返す。

    def search(self, query: str, top_k: int = 5, candidate_limit: int = 1200, source_prefix: str = "", mmr_lambda: float = 0.78, min_signal_ratio: float = 0.12) -> list[SearchHit]:  # 転置索引で候補を絞りハイブリッド検索する。
        if not query.strip():  # 空の質問を防ぐ。
            raise ValueError("検索質問を入力してください。")  # 修正方法を示す。
        if top_k < 1:  # 不正な返却件数を防ぐ。
            raise ValueError("top_k は 1 以上にしてください。")  # 設定範囲を示す。
        if candidate_limit < top_k:  # 候補上限が小さすぎないか確認する。
            raise ValueError("candidate_limit は top_k 以上にしてください。")  # 候補不足を防ぐ。
        if not 0.0 <= mmr_lambda <= 1.0:  # 多様化の範囲を確認する。
            raise ValueError("mmr_lambda は 0.0 から 1.0 にしてください。")  # 設定範囲を示す。
        if not 0.0 <= min_signal_ratio <= 1.0:  # 弱信号の足切り範囲を確認する。
            raise ValueError("min_signal_ratio は 0.0 から 1.0 にしてください。")  # 設定範囲を示す。
        query_counter = Counter(analyze(query))  # 質問特徴の回数を数える。
        with closing(self._connect()) as connection:  # 一回の検索で接続を確実に閉じる。
            meta = {str(row["key"]): str(row["value"]) for row in connection.execute("SELECT key, value FROM meta")}  # 計算に必要な統計を読む。
            total = int(meta["chunk_count"])  # 全チャンク数を整数へ戻す。
            average_length = float(meta["average_length"])  # 平均長を浮動小数へ戻す。
            known = self._known_terms(connection, query_counter)  # 索引に存在する質問特徴だけを得る。
            if not known:  # 既知特徴がない場合を扱う。
                self.last_stats = {"total_chunks": total, "matched_chunks": 0, "scored_chunks": 0}  # 空検索の処理量を記録する。
                return []  # 推測せず空結果を返す。
            selected_terms = sorted(known, key=lambda term: (-known[term][1], term))[:240]  # 希少度の高い特徴を SQL 上限内で選ぶ。
            rare_limit = max(8, int(total * 0.20))  # 全体の二割以下に現れる特徴を候補生成の目安にする。
            seed_terms = [term for term in selected_terms if known[term][0] <= rare_limit] or selected_terms  # 希少特徴がある質問では共通語の大量一致を避ける。
            candidate_ids = self._candidate_ids(connection, seed_terms, candidate_limit, source_prefix)  # 希少特徴中心の転置索引で候補 ID を得る。
            if not candidate_ids:  # 出典絞り込み後に候補がない場合を扱う。
                self.last_stats = {"total_chunks": total, "matched_chunks": 0, "scored_chunks": 0}  # 空検索の処理量を記録する。
                return []  # 空結果を返す。
            chunk_rows = self._chunk_rows(connection, candidate_ids)  # 候補の本文とメタデータだけを読む。
            posting_map = self._posting_map(connection, selected_terms, candidate_ids)  # 候補と質問特徴の頻度だけを読む。
            matched = self._matched_count(connection, seed_terms, source_prefix)  # 候補上限前の一致数を数える。
        self.last_stats = {"total_chunks": total, "matched_chunks": matched, "scored_chunks": len(candidate_ids)}  # 実際の削減量を記録する。
        raw: dict[str, tuple[float, float, float]] = {}  # 候補ごとの三スコアをためる。
        query_norm = math.sqrt(sum(((1.0 + math.log(query_counter[term])) * known[term][1]) ** 2 for term in selected_terms)) or 1.0  # 質問 TF-IDF ベクトル長を求める。
        for chunk_id in candidate_ids:  # 候補だけを精密採点する。
            row = chunk_rows[chunk_id]  # チャンクの保存値を得る。
            postings = posting_map.get(chunk_id, {})  # 質問と一致した特徴頻度を得る。
            bm25 = self._bm25(query_counter, postings, known, int(row["body_len"]), average_length, total)  # BM25 を計算する。
            cosine = self._cosine(query_counter, postings, known, query_norm, float(row["vector_norm"]))  # TF-IDF コサインを計算する。
            phrase = self._phrase_score(query, str(row["title"]), str(row["heading"]), str(row["text"]))  # 直接語句一致を計算する。
            raw[chunk_id] = (bm25, cosine, phrase)  # 三種類の独立信号を保存する。
        best_bm25 = max((values[0] for values in raw.values()), default=0.0)  # 最も強い BM25 信号を得る。
        best_cosine = max((values[1] for values in raw.values()), default=0.0)  # 最も強い cosine 信号を得る。
        candidate_ids = [item for item in candidate_ids if raw[item][0] >= best_bm25 * min_signal_ratio or raw[item][1] >= best_cosine * min_signal_ratio or raw[item][2] > 0.0]  # 最上位と比べ極端に弱い偶然一致を除く。
        self.last_stats["kept_chunks"] = len(candidate_ids)  # 足切り後の件数を診断値へ加える。
        kept_raw = {item: raw[item] for item in candidate_ids}  # 足切り後の候補だけを順位融合へ渡す。
        relevance = self._rrf(kept_raw)  # 尺度の違う順位を RRF で融合する。
        pool = sorted(candidate_ids, key=lambda item: relevance[item], reverse=True)[: max(30, top_k * 8)]  # 多様化へ渡す上位候補を制限する。
        selected = self._mmr(pool, relevance, chunk_rows, top_k, mmr_lambda)  # 似すぎる根拠を抑える。
        return [self._to_hit(chunk_id, chunk_rows[chunk_id], relevance[chunk_id], raw[chunk_id]) for chunk_id in selected]  # 型付き検索結果へ変える。

    def _connect(self) -> sqlite3.Connection:  # 読み取り専用の SQLite 接続を作る。
        if not self.path.exists():  # 索引ファイルの存在を確認する。
            raise FileNotFoundError(f"索引が見つかりません: {self.path}")  # 次に確認する場所を示す。
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)  # 誤書き込みを防ぐ読み取り接続を開く。
        connection.row_factory = sqlite3.Row  # 列名で値を読めるようにする。
        version = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()  # 保存形式番号を読む。
        if version is None or str(version[0]) != self.SCHEMA_VERSION:  # 対応形式か確認する。
            connection.close()  # 不適合な接続を閉じる。
            raise ValueError("索引形式が対応していません。index コマンドで再構築してください。")  # 安全な復旧方法を示す。
        return connection  # 検証済み接続を返す。

    @staticmethod  # 接続と質問以外の状態を使わない処理と示す。
    def _known_terms(connection: sqlite3.Connection, query_counter: Counter[str]) -> dict[str, tuple[int, float]]:  # 既知特徴の DF と IDF を読む。
        known: dict[str, tuple[int, float]] = {}  # 特徴統計をためる。
        for batch in _batches(list(query_counter), 800):  # SQLite の変数上限より小さく分ける。
            marks = ",".join("?" for _ in batch)  # 値埋め込み用の場所を作る。
            rows = connection.execute(f"SELECT term, df, idf FROM terms WHERE term IN ({marks})", batch).fetchall()  # この小分けの統計を読む。
            known.update({str(row["term"]): (int(row["df"]), float(row["idf"])) for row in rows})  # 型を明確にして保存する。
        return known  # 既知特徴だけを返す。

    @staticmethod  # 候補生成だけを分離する。
    def _candidate_ids(connection: sqlite3.Connection, terms: list[str], limit: int, source_prefix: str) -> list[str]:  # 希少特徴の一致から候補を選ぶ。
        marks = ",".join("?" for _ in terms)  # 質問特徴の場所を作る。
        source_sql = " AND c.source LIKE ? ESCAPE '\\'" if source_prefix else ""  # 必要時だけ出典条件を加える。
        sql = f"SELECT p.chunk_id, SUM((1.0 / (t.df + 1.0)) * (1.0 + 2.0 * p.head_tf)) AS seed FROM postings p JOIN terms t ON t.term=p.term JOIN chunks c ON c.chunk_id=p.chunk_id WHERE p.term IN ({marks}){source_sql} GROUP BY p.chunk_id ORDER BY seed DESC, p.chunk_id LIMIT ?"  # 転置表だけで軽い候補順位を作る。
        parameters: list[object] = list(terms)  # 質問特徴を安全な値引数にする。
        if source_prefix:  # 出典絞り込みが指定されたか確認する。
            parameters.append(_like_prefix(source_prefix))  # LIKE の記号を無害化して前方一致にする。
        parameters.append(limit)  # 最後に候補上限を加える。
        return [str(row["chunk_id"]) for row in connection.execute(sql, parameters)]  # 候補 ID を順位順で返す。

    @staticmethod  # チャンク読み込みを分離する。
    def _chunk_rows(connection: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, sqlite3.Row]:  # 候補本文を小分けで読む。
        result: dict[str, sqlite3.Row] = {}  # ID から行を引く辞書を作る。
        for batch in _batches(chunk_ids, 800):  # SQLite の変数上限より小さく分ける。
            marks = ",".join("?" for _ in batch)  # ID 用の場所を作る。
            rows = connection.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", batch).fetchall()  # 候補行だけを読む。
            result.update({str(row["chunk_id"]): row for row in rows})  # ID ごとに保存する。
        return result  # 候補行辞書を返す。

    @staticmethod  # 転置頻度読み込みを分離する。
    def _posting_map(connection: sqlite3.Connection, terms: list[str], chunk_ids: list[str]) -> dict[str, dict[str, tuple[int, int]]]:  # 質問特徴の頻度だけを読む。
        result: dict[str, dict[str, tuple[int, int]]] = {}  # チャンク別の特徴頻度をためる。
        for chunk_batch in _batches(chunk_ids, 500):  # ID と特徴の合計を変数上限内にする。
            term_marks = ",".join("?" for _ in terms)  # 特徴用の場所を作る。
            chunk_marks = ",".join("?" for _ in chunk_batch)  # チャンク用の場所を作る。
            parameters = [*terms, *chunk_batch]  # 二種類の安全な値引数を並べる。
            rows = connection.execute(f"SELECT term, chunk_id, body_tf, head_tf FROM postings WHERE term IN ({term_marks}) AND chunk_id IN ({chunk_marks})", parameters).fetchall()  # 必要な交点だけを読む。
            for row in rows:  # 交点をチャンクごとにまとめる。
                result.setdefault(str(row["chunk_id"]), {})[str(row["term"])] = (int(row["body_tf"]), int(row["head_tf"]))  # 本文と見出し頻度を保存する。
        return result  # 採点用の頻度辞書を返す。

    @staticmethod  # 診断値の計算を分離する。
    def _matched_count(connection: sqlite3.Connection, terms: list[str], source_prefix: str) -> int:  # 候補上限前の一致件数を数える。
        marks = ",".join("?" for _ in terms)  # 特徴用の場所を作る。
        source_sql = " AND c.source LIKE ? ESCAPE '\\'" if source_prefix else ""  # 必要時だけ出典条件を加える。
        parameters: list[object] = list(terms)  # 質問特徴を値引数にする。
        if source_prefix:  # 出典絞り込みがあるか確認する。
            parameters.append(_like_prefix(source_prefix))  # 安全な前方一致値を加える。
        row = connection.execute(f"SELECT COUNT(DISTINCT p.chunk_id) AS count FROM postings p JOIN chunks c ON c.chunk_id=p.chunk_id WHERE p.term IN ({marks}){source_sql}", parameters).fetchone()  # 一致チャンク数だけを求める。
        return int(row["count"]) if row else 0  # 整数の診断値を返す。

    @staticmethod  # 数式を独立して読めるようにする。
    def _bm25(query: Counter[str], postings: dict[str, tuple[int, int]], known: dict[str, tuple[int, float]], length: int, average: float, total: int, k1: float = 1.5, b: float = 0.75) -> float:  # 一候補の BM25 を計算する。
        score = 0.0  # 加算前の値を用意する。
        safe_average = average or 1.0  # ゼロ除算を防ぐ。
        for term, query_frequency in query.items():  # 質問の各特徴を処理する。
            body_tf, head_tf = postings.get(term, (0, 0))  # 本文と見出しの頻度を得る。
            frequency = body_tf + 2.0 * head_tf  # 見出し一致を二倍にする。
            if frequency == 0.0 or term not in known:  # 一致のない特徴を除く。
                continue  # 次の特徴へ進む。
            document_frequency = known[term][0]  # 特徴を含むチャンク数を得る。
            idf = math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))  # BM25 の IDF を求める。
            denominator = frequency + k1 * (1.0 - b + b * length / safe_average)  # 文書長補正を含む分母を求める。
            score += idf * frequency * (k1 + 1.0) / denominator * (1.0 + math.log(query_frequency))  # 飽和 TF を加算する。
        return score  # BM25 スコアを返す。

    @staticmethod  # 数式を独立して読めるようにする。
    def _cosine(query: Counter[str], postings: dict[str, tuple[int, int]], known: dict[str, tuple[int, float]], query_norm: float, document_norm: float) -> float:  # TF-IDF コサインを計算する。
        dot = 0.0  # 内積をゼロから始める。
        for term, query_frequency in query.items():  # 質問の各座標を処理する。
            if term not in known:  # 索引にない座標を除く。
                continue  # 次の座標へ進む。
            body_tf, head_tf = postings.get(term, (0, 0))  # 文書側の頻度を得る。
            document_frequency = body_tf + 2.0 * head_tf  # 見出し一致を二倍にする。
            if document_frequency == 0.0:  # 文書にない座標を除く。
                continue  # 次の座標へ進む。
            idf = known[term][1]  # 構築時の平滑化 IDF を得る。
            query_weight = (1.0 + math.log(query_frequency)) * idf  # 質問側の TF-IDF を求める。
            document_weight = (1.0 + math.log(document_frequency)) * idf  # 文書側の TF-IDF を求める。
            dot += query_weight * document_weight  # 共通座標の積を足す。
        return dot / (query_norm * document_norm)  # 長さで割った角度類似度を返す。

    @staticmethod  # 直接一致規則を独立して読めるようにする。
    def _phrase_score(query: str, title: str, heading: str, text: str) -> float:  # 質問語句の直接一致を測る。
        needle = keyword_text(query)  # 記号を除いた質問文字列を作る。
        if len(needle) < 2:  # 短すぎる直接一致を除く。
            return 0.0  # 偶然一致へ加点しない。
        title_text = keyword_text(f"{title}{heading}")  # タイトル領域を同じ形にする。
        body_text = keyword_text(text)  # 本文を同じ形にする。
        if needle in title_text:  # 質問全体がタイトル領域にあるか確認する。
            return 1.0  # 最も強い直接一致を返す。
        if needle in body_text:  # 質問全体が本文にあるか確認する。
            return 0.7  # 本文の直接一致を返す。
        return 0.0  # 全体一致なしを返す。

    @staticmethod  # 順位融合を独立して読めるようにする。
    def _rrf(raw: dict[str, tuple[float, float, float]]) -> dict[str, float]:  # 三尺度を Reciprocal Rank Fusion で統合する。
        ids = list(raw)  # 全候補 ID を得る。
        bm25_order = sorted(ids, key=lambda item: raw[item][0], reverse=True)  # BM25 順位を作る。
        cosine_order = sorted(ids, key=lambda item: raw[item][1], reverse=True)  # コサイン順位を作る。
        phrase_order = [item for item in sorted(ids, key=lambda item: raw[item][2], reverse=True) if raw[item][2] > 0.0]  # 直接一致のある順位を作る。
        bm25_rank = {item: rank for rank, item in enumerate(bm25_order, start=1)}  # ID から BM25 順位を引けるようにする。
        cosine_rank = {item: rank for rank, item in enumerate(cosine_order, start=1)}  # ID からコサイン順位を引けるようにする。
        phrase_rank = {item: rank for rank, item in enumerate(phrase_order, start=1)}  # ID から直接一致順位を引けるようにする。
        fused = {item: 1.0 / (60.0 + bm25_rank[item]) + 1.0 / (60.0 + cosine_rank[item]) + (0.35 / (60.0 + phrase_rank[item]) if item in phrase_rank else 0.0) for item in ids}  # 生尺度を混ぜず順位だけを足す。
        maximum = max(fused.values()) or 1.0  # 最上位を正規化基準にする。
        return {item: value / maximum for item, value in fused.items()}  # 最上位が一になる相対値を返す。

    @staticmethod  # 多様化を独立して読めるようにする。
    def _mmr(pool: list[str], relevance: dict[str, float], rows: dict[str, sqlite3.Row], top_k: int, strength: float) -> list[str]:  # 関連性を保ちながら重複を抑える。
        feature_sets = {item: set(analyze(str(rows[item]["text"]))) for item in pool}  # 小さな上位集合だけ特徴集合を作る。
        remaining = list(pool)  # 未選択候補を順位順で持つ。
        selected: list[str] = []  # 選んだ候補をためる。
        while remaining and len(selected) < top_k:  # 必要件数まで繰り返す。
            best = remaining[0]  # 同点時は元順位を優先する。
            best_score = float("-inf")  # 比較開始用の最小値を置く。
            for item in remaining:  # 未選択候補を一つずつ評価する。
                redundancy = max((_jaccard(feature_sets[item], feature_sets[chosen]) for chosen in selected), default=0.0)  # 選択済みとの最大重複を求める。
                score = strength * relevance[item] - (1.0 - strength) * redundancy  # 関連度から重複罰を引く。
                if score > best_score:  # 現在までの最良か確認する。
                    best = item  # 最良 ID を更新する。
                    best_score = score  # 最良値を更新する。
            selected.append(best)  # 最良候補を結果へ加える。
            remaining.remove(best)  # 再選択を防ぐ。
        return selected  # 多様化した ID 列を返す。

    @staticmethod  # DB 行から公開型への変換と示す。
    def _to_hit(chunk_id: str, row: sqlite3.Row, score: float, raw: tuple[float, float, float]) -> SearchHit:  # 検索結果型を作る。
        chunk = Chunk(chunk_id, str(row["doc_id"]), str(row["source"]), str(row["title"]), str(row["heading"]), int(row["ordinal"]), int(row["start_pos"]), int(row["end_pos"]), str(row["text"]))  # 保存値をチャンクへ戻す。
        return SearchHit(chunk, score, raw[0], raw[1], raw[2])  # スコア内訳付きで返す。


def _batches(items: list[str], size: int) -> Iterator[list[str]]:  # 長い値列を SQLite 上限内へ分ける。
    for start in range(0, len(items), size):  # 固定幅で開始位置を進める。
        yield items[start:start + size]  # 現在の小分けを返す。


def _like_prefix(value: str) -> str:  # LIKE の記号を文字として扱う前方一致値を作る。
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")  # 特殊な三文字を無害化する。
    return escaped + "%"  # 安全な前方一致記号を末尾だけに付ける。


def _jaccard(left: set[str], right: set[str]) -> float:  # 二つの特徴集合の重複率を求める。
    union = left | right  # 少なくとも一方にある特徴を集める。
    return len(left & right) / len(union) if union else 0.0  # 共通数を全体数で割って返す。
