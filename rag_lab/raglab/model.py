"""RAG 内部で共有する小さなデータ構造を定義する。"""  # このファイルの責務を示す。

from dataclasses import dataclass  # 標準ライブラリの軽量なデータクラスを使う。


@dataclass(frozen=True)  # 読み込み後の文書を変更できない値として扱う。
class Document:  # 一つの入力ファイルを表す。
    doc_id: str  # 文書を安定して識別する短い ID を持つ。
    source: str  # 利用者に示す元ファイル名を持つ。
    text: str  # UTF-8 で読み込んだ本文を持つ。


@dataclass(frozen=True)  # 索引に入るチャンクも不変の値として扱う。
class Chunk:  # 文書から切り出した検索単位を表す。
    chunk_id: str  # チャンクを安定して識別する ID を持つ。
    doc_id: str  # 元文書の ID を保持する。
    source: str  # 回答の出典表示に使うパスを保持する。
    heading: str  # Markdown 見出しをメタデータとして保持する。
    ordinal: int  # 文書内でのチャンク順を保持する。
    start: int  # 元文書内の開始文字位置を保持する。
    end: int  # 元文書内の終了文字位置を保持する。
    text: str  # 検索と回答生成に使う本文を保持する。


@dataclass(frozen=True)  # 検索結果を安全に受け渡す。
class SearchHit:  # 一件の検索結果と内訳を表す。
    chunk: Chunk  # ヒットしたチャンク本体を持つ。
    score: float  # RRF と MMR を反映した相対スコアを持つ。
    cosine: float  # TF-IDF コサイン類似度を説明用に持つ。
    bm25: float  # BM25 スコアを説明用に持つ。
