"""数据产物刷新：切块 -> 向量 -> BM25 索引。

把三个阶段串成库函数，CLI 与脚本共用同一条路径，避免 CLI 互相调用。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastrag.config import Config
from fastrag.errors import FastragError
from fastrag.index import ChunkStore, build_index, ensure_vectors
from fastrag.ingest import ChunkStats, build_chunks, stats, write_chunks

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PrepareReport:
    """一次 `prepare` 的结果摘要。"""

    chunks: ChunkStats
    vector_shape: tuple[int, int]
    elapsed: float


def prepare(config: Config) -> PrepareReport:
    """按依赖顺序刷新 chunks、向量与 BM25 索引。"""
    started = time.perf_counter()

    logger.info("[1/3] 切块")
    chunks = build_chunks(config)
    if not chunks:
        raise FastragError(f"没有可切分的文档 -> {config.paths.docs_dir}")
    write_chunks(config, chunks)

    logger.info("[2/3] 嵌入向量")
    store = ChunkStore.load(config)
    vectors = ensure_vectors(config, store)

    logger.info("[3/3] BM25 索引")
    build_index(config, store)

    return PrepareReport(
        chunks=stats(chunks),
        vector_shape=(int(vectors.shape[0]), int(vectors.shape[1])),
        elapsed=time.perf_counter() - started,
    )
