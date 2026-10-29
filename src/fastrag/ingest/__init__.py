"""数据准备层：下载、切块、产物工具。"""

from fastrag.ingest.artifacts import (
    ArtifactKey,
    matches,
    read_manifest,
    write_json_atomic,
    write_manifest,
    write_npy_atomic,
)
from fastrag.ingest.chunker import (
    ChunkStats,
    build_chunks,
    chunk_id,
    iter_docs,
    stats,
    strip_boilerplate,
    write_chunks,
)
from fastrag.ingest.download import DownloadSummary, collect_urls, download_docs

__all__ = [
    "ArtifactKey",
    "ChunkStats",
    "DownloadSummary",
    "build_chunks",
    "chunk_id",
    "collect_urls",
    "download_docs",
    "iter_docs",
    "matches",
    "read_manifest",
    "stats",
    "strip_boilerplate",
    "write_chunks",
    "write_json_atomic",
    "write_manifest",
    "write_npy_atomic",
]
