"""索引层：chunk 真源、向量索引、BM25 索引。"""

from fastrag.index.bm25 import build_index, load_index, search as search_bm25
from fastrag.index.dense import (
    build_embeddings,
    embed_texts,
    ensure_vectors,
    search_dense,
)
from fastrag.index.store import ChunkStore, file_sha256

__all__ = [
    "ChunkStore",
    "build_embeddings",
    "build_index",
    "embed_texts",
    "ensure_vectors",
    "file_sha256",
    "load_index",
    "search_bm25",
    "search_dense",
]
