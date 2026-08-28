"""評価ケース読み込みと Recall・MRR 集計を確認する。"""  # テスト対象を示す。

import tempfile  # テスト後に消える保存場所を作る。
import unittest  # 標準ライブラリのテスト機能を使う。
from pathlib import Path  # 一時ファイルのパスを扱う。

from ragcore.evaluate import EvalCase, evaluate  # 評価型と集計器を読み込む。
from ragcore.model import Chunk  # 小さな索引用のチャンク型を読み込む。
from ragcore.search import SearchIndex  # 評価対象の索引を読み込む。


class EvaluationTest(unittest.TestCase):  # 品質指標の約束をまとめる。
    def test_perfect_cases_return_full_recall_and_mrr(self) -> None:  # 全問正解の集計を検証する。
        first_text = "青い申請書の提出期限は10月1日です。"  # 一問目の根拠を作る。
        second_text = "赤い申請書の提出期限は11月1日です。"  # 二問目の根拠を作る。
        chunks = [Chunk("blue", "blue", "blue.md", "青い申請", "期限", 0, 0, len(first_text), first_text), Chunk("red", "red", "red.md", "赤い申請", "期限", 0, 0, len(second_text), second_text)]  # 二つの検索単位を作る。
        cases = [EvalCase("青い申請書の期限", ("blue.md",)), EvalCase("赤い申請書の期限", ("red.md",))]  # 対応する正解セットを作る。
        with tempfile.TemporaryDirectory() as directory:  # 自動削除される場所を作る。
            index = SearchIndex.build(Path(directory) / "eval.db", chunks)  # 実際の索引を構築する。
            report = evaluate(index, cases, top_k=1)  # 上位一件で評価する。
        self.assertEqual(report["recall_at_k"], 1.0)  # 全正解を回収できたか確認する。
        self.assertEqual(report["mrr"], 1.0)  # 全正解が最上位か確認する。
        self.assertEqual(report["cases"], 2)  # ケース数が保存されるか確認する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    unittest.main()  # このテストを開始する。
