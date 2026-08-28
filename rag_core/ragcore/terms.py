"""日本語辞書なしで検索用の見える特徴量を作る。"""  # このファイルの責務を示す。

import re  # 英数字列と日本語列を取り出す。
import unicodedata  # 全角と半角などの表記差を整える。
from collections import Counter  # 特徴の出現回数を数える。


def normalize(text: str) -> str:  # 検索前の文字表現をそろえる。
    normalized = unicodedata.normalize("NFKC", text).lower()  # Unicode と英字大小を正規化する。
    return re.sub(r"\s+", " ", normalized).strip()  # 連続空白を一つへまとめる。


def analyze(text: str) -> list[str]:  # 日本語と英数字から疎な検索特徴を作る。
    normalized = normalize(text)  # 最初に表記揺れを抑える。
    runs = re.findall(r"[a-z0-9_]+|[ぁ-んァ-ヶー一-龯々〆ヵヶ]+", normalized)  # 文字種ごとの連続列を抜き出す。
    terms: list[str] = []  # 特徴を出現順にためる。
    for run in runs:  # 各連続列を種類別に処理する。
        if re.fullmatch(r"[a-z0-9_]+", run):  # 英数字語か確認する。
            terms.append(f"w:{run}")  # 単語全体を完全一致特徴にする。
            if len(run) >= 5:  # 長い識別子だけ部分一致を許す。
                terms.extend(f"a3:{run[index:index + 3]}" for index in range(len(run) - 2))  # 英数字 3-gram を追加する。
            continue  # 日本語用の処理を飛ばす。
        if len(run) <= 12:  # 短い日本語列だけ完全一致特徴にする。
            terms.append(f"j:{run}")  # まとまった語句の一致を強く拾う。
        terms.extend(f"c2:{run[index:index + 2]}" for index in range(max(0, len(run) - 1)))  # 日本語 bigram を追加する。
        terms.extend(f"c3:{run[index:index + 3]}" for index in range(max(0, len(run) - 2)))  # 日本語 trigram を追加する。
        if len(run) == 1:  # 一文字の型番や略語を救う。
            terms.append(f"c1:{run}")  # 限定的な一文字特徴を追加する。
    return terms  # 重複を保った特徴列を返す。


def term_counts(text: str) -> Counter[str]:  # 一つの文字列の特徴頻度を返す。
    return Counter(analyze(text))  # 共通解析器の結果を数える。


def keyword_text(text: str) -> str:  # 直接一致判定用の空白なし文字列を作る。
    normalized = normalize(text)  # 表記を検索時と同じ形へそろえる。
    return re.sub(r"[^a-z0-9ぁ-んァ-ヶー一-龯々〆ヵヶ]+", "", normalized)  # 記号と空白だけを除く。
