"""構築、検索、保存、回答までの一連の経路を確認する。"""  # テスト対象を示す。

import tempfile  # テスト後に自動削除される保存場所を作る。
import unittest  # 標準ライブラリのテスト機能を使う。
from pathlib import Path  # 一時索引のパスを扱う。

from raglab.answer import ExtractiveAnswerer  # 外部不要の回答器を読み込む。
from raglab.index import RagIndex  # 検索索引を読み込む。
from raglab.model import Document  # テスト文書型を読み込む。
from raglab.text import TextChunker  # 文書分割器を読み込む。


class PipelineTest(unittest.TestCase):  # RAG 全体の振る舞いをまとめる。
    def setUp(self) -> None:  # 各テスト用の小さな索引を作る。
        returns = Document("returns", "returns.md", "# 返品\n返品申請は配達日の翌日から30日以内です。未使用の商品だけを受け付けます。")  # 期待根拠を含む文書を作る。
        security = Document("security", "security.md", "# 安全\nログインに5回失敗すると15分間ロックされます。")  # 無関係な比較文書を作る。
        chunker = TextChunker(max_chars=120, overlap_chars=20)  # 小文書向けの分割器を作る。
        chunks = chunker.chunk(returns) + chunker.chunk(security)  # 二文書をチャンク化する。
        self.index = RagIndex()  # 空の索引を作る。
        self.index.build(chunks)  # 検索統計を構築する。

    def test_search_and_extractive_answer_have_citation(self) -> None:  # 検索順位と引用を検証する。
        query = "返品できる期限は？"  # 期待根拠へ一致する質問を作る。
        hits = self.index.search(query, top_k=2)  # ハイブリッド検索を実行する。
        self.assertEqual(hits[0].chunk.source, "returns.md")  # 返品文書が最上位か確認する。
        answer = ExtractiveAnswerer().answer(query, hits)  # 検索結果から回答を作る。
        self.assertIn("30日", answer)  # 必要な事実が回答に含まれるか確認する。
        self.assertIn("[S1]", answer)  # 根拠番号が回答に含まれるか確認する。

    def test_saved_index_keeps_ranking(self) -> None:  # JSON 往復後も検索結果が同じか検証する。
        with tempfile.TemporaryDirectory() as directory:  # 自動削除されるフォルダを作る。
            path = Path(directory) / "index.json"  # 一時索引パスを決める。
            self.index.save(path)  # 構築済み索引を保存する。
            loaded = RagIndex.load(path)  # JSON から索引を復元する。
            hit = loaded.search("ログイン失敗時のロック時間", top_k=1)[0]  # 復元索引を検索する。
            self.assertEqual(hit.chunk.source, "security.md")  # 正しい出典が最上位か確認する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    unittest.main()  # このファイルのテストを実行する。
