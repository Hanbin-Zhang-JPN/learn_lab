"""依存ライブラリなしの日本語対応 TF-IDF ベクトル化を実装する。"""  # このファイルの責務を示す。

from __future__ import annotations  # 型注釈を実行時評価から切り離す。

import math  # IDF、対数 TF、ベクトル長を計算する。
import re  # 日本語文字列と英数字語を抽出する。
import unicodedata  # 全角・半角などの表記揺れを整える。
from collections import Counter  # 特徴量の出現回数を数える。


SparseVector = dict[str, float]  # 特徴名と重みだけを持つ疎ベクトルを表す。


def normalize_text(text: str) -> str:  # 検索用に文字表現をそろえる。
    normalized = unicodedata.normalize("NFKC", text).lower()  # Unicode と英字の大小を正規化する。
    return re.sub(r"\s+", " ", normalized).strip()  # 連続空白を一つへまとめる。


def analyze(text: str) -> list[str]:  # 日本語辞書なしで検索特徴を作る。
    normalized = normalize_text(text)  # 先に表記揺れを抑える。
    runs = re.findall(r"[a-z0-9_]+|[ぁ-んァ-ヶー一-龯々〆ヵヶ]+", normalized)  # 英数字語と日本語文字列を抜き出す。
    features: list[str] = []  # 特徴量の格納先を用意する。
    for run in runs:  # 各文字列を種類に応じて処理する。
        if re.fullmatch(r"[a-z0-9_]+", run):  # 英数字語か確認する。
            features.append(f"w:{run}")  # 英数字は単語全体を特徴にする。
            if len(run) >= 4:  # 長い英数字語だけ部分一致も許す。
                features.extend(f"c3:{run[index:index + 3]}" for index in range(len(run) - 2))  # 3-gram を追加する。
            continue  # 日本語用の処理を飛ばす。
        features.append(f"j:{run}")  # 完全一致時に効く日本語列も保持する。
        features.extend(f"c2:{run[index:index + 2]}" for index in range(max(0, len(run) - 1)))  # 日本語 bigram を追加する。
        features.extend(f"c3:{run[index:index + 3]}" for index in range(max(0, len(run) - 2)))  # 日本語 trigram を追加する。
        if len(run) == 1:  # 一文字だけの検索語を救済する。
            features.append(f"c1:{run}")  # 一文字特徴を限定的に追加する。
    return features  # 順序と重複を保った特徴列を返す。


class TfidfVectorizer:  # TF-IDF の学習と変換を目に見える形で行う。
    def __init__(self, idf: dict[str, float] | None = None) -> None:  # 保存済み IDF も受け取れるようにする。
        self.idf = dict(idf or {})  # 呼び出し側の辞書を複製して保持する。

    def fit(self, texts: list[str]) -> TfidfVectorizer:  # 文書集合から IDF を学習する。
        if not texts:  # 空の学習データを防ぐ。
            raise ValueError("TF-IDF の学習には一件以上のテキストが必要です。")  # 原因を明確に伝える。
        document_frequency: Counter[str] = Counter()  # 各特徴が何文書に出たか数える。
        for text in texts:  # 全チャンクを一度ずつ調べる。
            document_frequency.update(set(analyze(text)))  # 同じ文書内の重複を一回として数える。
        count = len(texts)  # 全文書数を保存する。
        self.idf = {term: math.log((1.0 + count) / (1.0 + frequency)) + 1.0 for term, frequency in document_frequency.items()}  # 平滑化 IDF を計算する。
        return self  # メソッド連結に使えるよう自分自身を返す。

    def transform_one(self, text: str) -> SparseVector:  # 一つのテキストを疎ベクトルへ変換する。
        counts = Counter(analyze(text))  # 特徴量ごとの出現回数を数える。
        weighted: SparseVector = {}  # 未正規化ベクトルの格納先を用意する。
        for term, frequency in counts.items():  # 出現した特徴だけを処理する。
            if term not in self.idf:  # 学習時に存在しない特徴か確認する。
                continue  # 未知語は座標がないため無視する。
            tf = 1.0 + math.log(frequency)  # 頻出語の影響を抑える対数 TF を求める。
            weighted[term] = tf * self.idf[term]  # TF と IDF を掛けて座標値にする。
        norm = math.sqrt(sum(value * value for value in weighted.values()))  # L2 ベクトル長を求める。
        if norm == 0.0:  # 既知特徴がない場合を扱う。
            return {}  # ゼロベクトルを空辞書で返す。
        return {term: value / norm for term, value in weighted.items()}  # 単位長へ正規化して返す。

    def fit_transform(self, texts: list[str]) -> list[SparseVector]:  # 学習と全文書変換を連続して行う。
        self.fit(texts)  # 文書集合から IDF を学習する。
        return [self.transform_one(text) for text in texts]  # 同じ尺度ですべてをベクトル化する。


def cosine_similarity(left: SparseVector, right: SparseVector) -> float:  # 二つの正規化疎ベクトルの類似度を求める。
    if not left or not right:  # どちらかがゼロベクトルか確認する。
        return 0.0  # 比較できる特徴がなければ類似なしとする。
    smaller, larger = (left, right) if len(left) <= len(right) else (right, left)  # 少ない側だけを走査する。
    return sum(value * larger.get(term, 0.0) for term, value in smaller.items())  # 共通座標の内積を返す。
