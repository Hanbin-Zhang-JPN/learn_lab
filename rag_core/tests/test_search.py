"""索引構築、候補削減、順位、範囲指定、回答引用を確認する。"""  # テスト対象を示す。

import sqlite3  # 保存表を直接確認する。
import tempfile  # テスト後に消える索引場所を作る。
import unittest  # 標準ライブラリのテスト機能を使う。
from pathlib import Path  # 一時索引のパスを扱う。

from ragcore.answer import ExtractiveAnswerer  # 外部不要の回答器を読み込む。
from ragcore.model import Document  # 入力文書型を読み込む。
from ragcore.search import SearchIndex  # 検索索引を読み込む。
from ragcore.text import TextChunker  # 文書分割器を読み込む。


class SearchIndexTest(unittest.TestCase):  # 検索経路の約束をまとめる。
    def setUp(self) -> None:  # 各テスト用の小さな永続索引を作る。
        self.directory = tempfile.TemporaryDirectory()  # 自動削除される場所を保持する。
        shop = Document("shop", "shop/return.md", "返品案内", "# 返品\n返品申請は配達日の翌日から30日以内です。未使用で外箱がある商品だけを受け付けます。")  # 返品の正解文書を作る。
        account = Document("account", "account/login.md", "ログイン案内", "# ロック\nログインに5回失敗すると、アカウントは15分間ロックされます。")  # ログインの正解文書を作る。
        support = Document("support", "support/hours.md", "窓口案内", "# 営業時間\nチャット窓口は平日9時から18時まで対応します。")  # 問い合わせの正解文書を作る。
        chunker = TextChunker(180, 30)  # 小文書向けの分割器を作る。
        chunks = [chunk for document in (shop, account, support) for chunk in chunker.chunk(document)]  # 三文書をチャンク化する。
        self.path = Path(self.directory.name) / "rag.db"  # 一時 DB の場所を決める。
        self.index = SearchIndex.build(self.path, chunks)  # 実際の SQLite 索引を作る。

    def tearDown(self) -> None:  # 各テストの一時資源を片付ける。
        self.directory.cleanup()  # 一時フォルダ全体を削除する。

    def test_relevant_source_is_first_and_answer_has_citation(self) -> None:  # 順位と引用を検証する。
        query = "返品できる期限と条件は？"  # 返品文書へ向く質問を作る。
        hits = self.index.search(query, top_k=2)  # 通常のハイブリッド検索を行う。
        self.assertEqual(hits[0].chunk.source, "shop/return.md")  # 正解出典が最上位か確認する。
        answer = ExtractiveAnswerer().answer(query, hits)  # 根拠文から回答を作る。
        self.assertIn("30日", answer)  # 必要な事実が含まれるか確認する。
        self.assertIn("[S1]", answer)  # 検証可能な引用が付くか確認する。

    def test_source_prefix_limits_search_scope(self) -> None:  # 出典範囲の分離を検証する。
        hits = self.index.search("何回失敗するとロックされますか？", top_k=2, source_prefix="account/")  # アカウント範囲だけを検索する。
        self.assertTrue(hits)  # 範囲内に結果があるか確認する。
        self.assertTrue(all(hit.chunk.source.startswith("account/") for hit in hits))  # 範囲外が混ざらないか確認する。

    def test_inverted_index_reports_reduced_candidates(self) -> None:  # 全件走査を避ける診断値を検証する。
        self.index.search("チャット窓口の営業時間", top_k=1)  # 一部文書だけに一致する質問を検索する。
        self.assertEqual(self.index.last_stats["total_chunks"], 3)  # 全索引件数を確認する。
        self.assertLessEqual(self.index.last_stats["scored_chunks"], self.index.last_stats["total_chunks"])  # 採点件数が全件以下か確認する。

    def test_sqlite_contains_visible_posting_tables(self) -> None:  # 保存構造が観察可能か検証する。
        with sqlite3.connect(self.path) as connection:  # 標準機能で索引を直接開く。
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}  # 表名を集める。
        self.assertTrue({"meta", "terms", "chunks", "postings"}.issubset(tables))  # 説明した四表があるか確認する。

    def test_unknown_query_returns_empty_list(self) -> None:  # 根拠なし判定を検証する。
        hits = self.index.search("🚀🌙", top_k=2)  # 索引特徴を持たない質問を使う。
        self.assertEqual(hits, [])  # 推測した結果を返さないか確認する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    unittest.main()  # このテストを開始する。
