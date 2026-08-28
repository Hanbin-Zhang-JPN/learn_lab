"""TF-IDF ベクトルとコサイン類似度を確認する。"""  # テスト対象を示す。

import unittest  # 標準ライブラリのテスト機能を使う。

from raglab.vector import TfidfVectorizer, analyze, cosine_similarity  # ベクトル処理を読み込む。


class VectorTest(unittest.TestCase):  # ベクトル化の振る舞いをまとめる。
    def test_japanese_bigrams_are_visible(self) -> None:  # 日本語の部分特徴を検証する。
        features = analyze("返品条件")  # 短い日本語を特徴化する。
        self.assertIn("c2:返品", features)  # 先頭 bigram が存在するか確認する。
        self.assertIn("c3:品条件", features)  # 末尾 trigram が存在するか確認する。

    def test_related_text_has_higher_cosine(self) -> None:  # 関連文が上位になることを検証する。
        texts = ["返品の申請期限は配達後30日です。", "深煎りコーヒーの香りを説明します。"]  # 異なる二文書を用意する。
        vectorizer = TfidfVectorizer()  # 空の変換器を作る。
        vectors = vectorizer.fit_transform(texts)  # 二文書を同じ空間へ写す。
        query = vectorizer.transform_one("返品期限を知りたい")  # 関連質問を同じ空間へ写す。
        self.assertGreater(cosine_similarity(query, vectors[0]), cosine_similarity(query, vectors[1]))  # 関連文の類似度が高いか確認する。


if __name__ == "__main__":  # ファイル単体実行か確認する。
    unittest.main()  # このファイルのテストを実行する。
