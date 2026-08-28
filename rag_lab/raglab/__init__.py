"""仕組みを追える、標準ライブラリだけの小さな RAG パッケージ。"""  # パッケージの目的を示す。

from .answer import ExtractiveAnswerer, OpenAICompatibleAnswerer  # 公開する回答器を読み込む。
from .index import RagIndex  # 公開する索引型を読み込む。
from .model import Chunk, Document, SearchHit  # 公開するデータ型を読み込む。
from .text import TextChunker, load_documents  # 公開する文書処理を読み込む。

__all__ = ["Chunk", "Document", "ExtractiveAnswerer", "OpenAICompatibleAnswerer", "RagIndex", "SearchHit", "TextChunker", "load_documents"]  # 意図した公開 API を明示する。
