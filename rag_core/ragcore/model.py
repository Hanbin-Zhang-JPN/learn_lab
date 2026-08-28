"""RAG 内で共有する小さな不変データ型を定義する。"""  # このファイルの責務を示す。

from dataclasses import dataclass  # 標準のデータクラスだけを使う。


@dataclass(frozen=True)  # 読み込み後の文書を不変にする。
class Document:  # 一つの入力ファイルを表す。
    doc_id: str  # 内容とパスから作る安定 ID を持つ。
    source: str  # 利用者へ示す相対パスを持つ。
    title: str  # 文書の表示名を持つ。
    text: str  # UTF-8 で読んだ本文を持つ。


@dataclass(frozen=True)  # 検索単位を不変にする。
class Chunk:  # 文書から切り出した一つの根拠を表す。
    chunk_id: str  # 位置と内容から作る安定 ID を持つ。
    doc_id: str  # 元文書の ID を持つ。
    source: str  # 出典表示用の相対パスを持つ。
    title: str  # 文書タイトルを持つ。
    heading: str  # 最も近い Markdown 見出しを持つ。
    ordinal: int  # 文書内の並び順を持つ。
    start: int  # 元文書内の開始文字位置を持つ。
    end: int  # 元文書内の終了文字位置を持つ。
    text: str  # 検索と回答に使う本文を持つ。


@dataclass(frozen=True)  # 検索結果を安全に受け渡す。
class SearchHit:  # 一件の根拠とスコア内訳を表す。
    chunk: Chunk  # ヒットした本文と出典を持つ。
    score: float  # 融合後の相対スコアを持つ。
    bm25: float  # 単語頻度に基づくスコアを持つ。
    cosine: float  # TF-IDF の角度類似度を持つ。
    phrase: float  # 質問語句の直接一致度を持つ。
