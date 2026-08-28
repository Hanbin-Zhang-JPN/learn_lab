"""中身を追える実用 RAG の公開 API。"""  # パッケージの役割を示す。

from .answer import ExtractiveAnswerer, LlmAnswerer  # 二種類の回答器を公開する。
from .model import Chunk, Document, SearchHit  # 共通データ型を公開する。
from .search import SearchIndex  # 永続検索索引を公開する。
from .text import TextChunker, load_documents  # 文書処理機能を公開する。

__all__ = ["Chunk", "Document", "ExtractiveAnswerer", "LlmAnswerer", "SearchHit", "SearchIndex", "TextChunker", "load_documents"]  # 公開対象を明示する。
