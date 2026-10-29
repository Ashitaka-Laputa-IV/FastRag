"""向量索引：构建、增量复用与余弦检索。

- 向量以 L2 归一化后的 float32 矩阵落盘，查询时只做内积；
- manifest 记录 `chunks.jsonl` 指纹、模型、维度与 chunk id 列表；
- chunk id 内容寻址，因此语料变化时可以按 id 复用未变化的向量，只嵌入新增块。
"""

from __future__ import annotations

import logging
import pathlib
import time

import numpy as np
from langchain_openai import OpenAIEmbeddings

from fastrag.config import Config, endpoint_api_key
from fastrag.domain import ScoredChunk
from fastrag.errors import FastragError
from fastrag.index.store import ChunkStore
from fastrag.ingest.artifacts import (
    ArtifactKey,
    read_manifest,
    write_manifest,
    write_npy_atomic,
)

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
MANIFEST_FILE = "manifest.json"


def build_embeddings(config: Config) -> OpenAIEmbeddings:
    """按配置建 OpenAI 兼容的 embeddings 客户端。"""
    embedding = config.embedding
    return OpenAIEmbeddings(
        model=embedding.model,
        base_url=config.endpoints[embedding.endpoint].base_url,
        api_key=endpoint_api_key(config, embedding.endpoint),
        chunk_size=embedding.batch_size,
        max_retries=embedding.max_retries,
        request_timeout=embedding.timeout,
        # 非 OpenAI 模型不做 tiktoken 预切分，交给服务端处理
        check_embedding_ctx_length=False,
    )


def embed_texts(
    embeddings: OpenAIEmbeddings, config: Config, texts: list[str]
) -> np.ndarray:
    """分批嵌入文本，返回 (N, dimensions) 的 float32 矩阵。"""
    dimensions = config.embedding.dimensions
    batch_size = config.embedding.batch_size
    if not texts:
        return np.zeros((0, dimensions), dtype=np.float32)
    vectors = np.zeros((len(texts), dimensions), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        batch_vectors = embeddings.embed_documents(batch)
        if len(batch_vectors) != len(batch):
            raise FastragError(
                f"嵌入服务返回 {len(batch_vectors)} 条向量，期望 {len(batch)} 条"
            )
        if len(batch_vectors[0]) != dimensions:
            raise FastragError(
                f"返回维度 {len(batch_vectors[0])} 与配置 dimensions={dimensions} 不一致"
            )
        vectors[start : start + len(batch)] = batch_vectors
        logger.info("已嵌入 %d/%d", min(start + batch_size, len(texts)), len(texts))
    return vectors


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _manifest_key(config: Config, store: ChunkStore) -> ArtifactKey:
    return ArtifactKey(
        kind="embeddings",
        source_digest=store.digest,
        params={
            "model": config.embedding.model,
            "dimensions": config.embedding.dimensions,
            "normalized": True,
        },
    )


def _load_previous(
    config: Config, store: ChunkStore
) -> tuple[np.ndarray, list[str]] | None:
    """读取可复用的旧向量；模型/维度不一致或形状不匹配时返回 None。"""
    directory = config.paths.embeddings_dir
    vectors_path = directory / VECTORS_FILE
    manifest = read_manifest(directory, MANIFEST_FILE)
    if manifest is None or not vectors_path.exists():
        return None
    params = manifest.get("params") or {}
    if (
        params.get("model") != config.embedding.model
        or params.get("dimensions") != config.embedding.dimensions
    ):
        return None
    ids = manifest.get("ids")
    if not isinstance(ids, list):
        return None
    try:
        vectors = np.load(vectors_path)
    except (OSError, ValueError):
        return None
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        return None
    return vectors, [str(item) for item in ids]


def ensure_vectors(
    config: Config,
    store: ChunkStore,
    embeddings: OpenAIEmbeddings | None = None,
) -> np.ndarray:
    """加载可用向量；不可用时按 id 增量补齐或全量重建，并原子落盘。"""
    directory = config.paths.embeddings_dir
    key = _manifest_key(config, store)
    current_ids = [chunk.id for chunk in store.chunks]
    previous = _load_previous(config, store)

    if previous is not None:
        vectors, old_ids = previous
        if old_ids == current_ids:
            logger.info("复用已嵌入向量：%d 条", len(current_ids))
            return vectors
        index_by_id = {chunk_id: row for row, chunk_id in enumerate(old_ids)}
        matrix = np.zeros(
            (len(store.chunks), config.embedding.dimensions), dtype=np.float32
        )
        missing: list[int] = []
        for row, chunk in enumerate(store.chunks):
            old_row = index_by_id.get(chunk.id)
            if old_row is None:
                missing.append(row)
            else:
                matrix[row] = vectors[old_row]
        if missing:
            logger.info(
                "增量嵌入：复用 %d 条，新增 %d 条",
                len(store.chunks) - len(missing),
                len(missing),
            )
            embedded = embed_texts(
                embeddings or build_embeddings(config),
                config,
                [store.chunks[row].text for row in missing],
            )
            matrix[missing] = embedded
        else:
            logger.info("向量按 id 全部复用：%d 条", len(store.chunks))
        matrix = _normalize(matrix)
    else:
        logger.info("未找到可复用向量，开始完整嵌入 ...")
        started = time.perf_counter()
        matrix = _normalize(
            embed_texts(embeddings or build_embeddings(config), config,
                        [chunk.text for chunk in store.chunks])
        )
        logger.info("嵌入完成，耗时 %.1fs", time.perf_counter() - started)

    write_npy_atomic(directory / VECTORS_FILE, matrix)
    write_manifest(
        directory,
        key,
        {
            "count": len(current_ids),
            "ids": current_ids,
            "chunks_file": str(store.path),
        },
        MANIFEST_FILE,
    )
    logger.info("向量已落盘：%s", directory / VECTORS_FILE)
    return matrix


def search_dense(
    embeddings: OpenAIEmbeddings,
    vectors: np.ndarray,
    store: ChunkStore,
    text: str,
    k: int,
) -> list[ScoredChunk]:
    """返回 top-k 余弦命中。向量已归一化，查询向量归一化后直接内积。"""
    if len(store) == 0 or vectors.shape[0] == 0 or k <= 0:
        return []
    query = np.asarray(embeddings.embed_query(text), dtype=np.float32)
    if query.ndim != 1 or query.shape[0] != vectors.shape[1]:
        raise FastragError(
            f"查询向量维度 {query.shape} 与语料向量 {vectors.shape} 不一致"
        )
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    scores = vectors @ query
    top = np.argsort(-scores)[: min(k, len(store))]
    return [
        ScoredChunk(
            chunk=store.chunks[int(index)],
            score=float(scores[int(index)]),
            channel="vector",
            rank=rank,
        )
        for rank, index in enumerate(top, start=1)
    ]


def vectors_path(config: Config) -> pathlib.Path:
    return config.paths.embeddings_dir / VECTORS_FILE
