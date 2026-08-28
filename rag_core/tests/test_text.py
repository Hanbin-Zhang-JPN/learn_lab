"""見出し境界、重複窓、位置情報を確認する。"""  # テスト対象を示す。

import unittest  # 標準ライブラリのテスト機能を使う。

from ragcore.model import Document  # 入力文書型を読み込む。
from ragcore.text import TextChunker  # 分割器を読み込む。


class TextChunkerTest(unittest.TestCase):  # 文書分割の約束をまとめる。
    def test_heading_size_and_position_are_preserved(self) -> None:  # 見出し、上限、位置を検証する。
        text = "# 操作手順\n" + "最初の説明です。" * 12 + "次の説明です。" * 12  # 上限を越える日本語文書を作る。
        document = Document("doc-1", "guide.md", "操作手順", text)  # テスト文書を作る。
        chunks = TextChunker(100, 20).chunk(document)  # 小さい上限で分割する。
        self.assertGreater(len(chunks), 1)  # 複数チャンクになったか確認する。
        self.assertTrue(all(chunk.heading == "操作手順" for chunk in chunks))  # 見出しが全件に付くか確認する。
        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))  # 上限を守るか確認する。
        self.assertTrue(all(text[chunk.start:chunk.end].strip().startswith(chunk.text.splitlines()[0]) for chunk in chunks))  # 開始位置が原文と対応するか確認する。

    def test_long_sentence_uses_overlapping_windows(self) -> None:  # 文末のない長文を検証する。
        document = Document("doc-2", "long.txt", "長文", "あ" * 230)  # 一文だけの長い文書を作る。
        chunks = TextChunker(100, 20).chunk(document)  # 20 文字重複で分割する。
        self.assertEqual([len(chunk.text) for chunk in chunks], [100, 100, 70])  # 固定窓の長さを確認する。
        self.assertEqual(chunks[0].text[-20:], chunks[1].text[:20])  # 隣接窓の重複を確認する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    unittest.main()  # このテストを開始する。
