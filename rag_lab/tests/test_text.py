"""チャンク化の境界とメタデータを確認する。"""  # テスト対象を示す。

import unittest  # 標準ライブラリのテスト機能を使う。

from raglab.model import Document  # テスト文書型を読み込む。
from raglab.text import TextChunker  # テスト対象の分割器を読み込む。


class TextChunkerTest(unittest.TestCase):  # チャンク化の振る舞いをまとめる。
    def test_heading_size_and_position_are_preserved(self) -> None:  # 見出し、上限、位置を検証する。
        text = "# 手順\n" + "最初の説明です。" * 8 + "次の説明です。" * 8  # 上限を超える日本語文書を作る。
        document = Document(doc_id="doc-1", source="sample.md", text=text)  # 位置確認用の文書を作る。
        chunks = TextChunker(max_chars=80, overlap_chars=20).chunk(document)  # 小さい上限で分割する。
        self.assertGreater(len(chunks), 1)  # 複数チャンクになったことを確認する。
        self.assertTrue(all(chunk.heading == "手順" for chunk in chunks))  # 見出しが全件へ付いたことを確認する。
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))  # 各チャンクが上限内か確認する。
        self.assertTrue(all(text[chunk.start:chunk.end].strip().startswith(chunk.text.splitlines()[0]) for chunk in chunks))  # 開始位置が元文書と対応するか確認する。

    def test_long_sentence_uses_overlapping_windows(self) -> None:  # 長い一文の固定窓処理を検証する。
        document = Document(doc_id="doc-2", source="long.txt", text="あ" * 170)  # 文末のない長文を作る。
        chunks = TextChunker(max_chars=80, overlap_chars=20).chunk(document)  # 重なり付きで分割する。
        self.assertEqual([len(chunk.text) for chunk in chunks], [80, 80, 50])  # 期待する窓長を確認する。
        self.assertEqual(chunks[0].text[-20:], chunks[1].text[:20])  # 隣接窓の重複を確認する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    unittest.main()  # このファイルのテストを実行する。
