"""向量索引：构建、增量复用与余弦检索。

- 向量由 `InMemoryVectorStore` 管理并落盘为 `vector_store.json`；
- manifest 记录 `chunks.jsonl` 指纹、模型、维度与 chunk id 列表；
- chunk id 内容寻址，因此语料变化时可以按 id 复用未变化的向量，只嵌入新增块。
"""

from __future__ import annotations

import logging
import time

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from fastrag.config import Config, endpoint_api_key
from fastrag.domain import ScoredChunk
from fastrag.index.store import ChunkStore
from fastrag.ingest.artifacts import (
    ArtifactKey,
    atomic_path,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)

STORE_FILE = "vector_store.json"
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


def _manifest_key(config: Config, store: ChunkStore) -> ArtifactKey:
    return ArtifactKey(
        kind="embeddings",
        source_digest=store.digest,
        params={
            "model": config.embedding.model,
            "dimensions": config.embedding.dimensions,
        },
    )


def _previous_ids(config: Config) -> list[str] | None:
    """读取可复用的 chunk id；模型/维度不一致时返回 None。"""
    manifest = read_manifest(config.paths.embeddings_dir, MANIFEST_FILE)
    if manifest is None:
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
    return [str(item) for item in ids]


def _load_store(
    config: Config, embeddings: OpenAIEmbeddings
) -> InMemoryVectorStore | None:
    path = config.paths.embeddings_dir / STORE_FILE
    if not path.exists():
        return None
    try:
        return InMemoryVectorStore.load(str(path), embeddings)
    except (OSError, ValueError):
        return None


def ensure_vector_store(
    config: Config,
    store: ChunkStore,
    embeddings: OpenAIEmbeddings | None = None,
) -> InMemoryVectorStore:
    """加载可用向量库；不可用时按 id 增量补齐或全量重建，并原子落盘。"""
    embeddings = embeddings or build_embeddings(config)
    directory = config.paths.embeddings_dir
    current_ids = [chunk.id for chunk in store.chunks]
    previous_ids = _previous_ids(config)
    vector_store = _load_store(config, embeddings) if previous_ids is not None else None
    rebuild = vector_store is None

    if rebuild:
        logger.info("未找到可复用向量，开始完整嵌入 ...")
        started = time.perf_counter()
        vector_store = InMemoryVectorStore(embeddings)
        missing = list(store.chunks)
    else:
        present = {doc.id for doc in vector_store.get_by_ids(previous_ids or [])}
        stale = present - set(current_ids)
        if stale:
            vector_store.delete(sorted(stale))
        reused = present - stale
        missing = [chunk for chunk in store.chunks if chunk.id not in reused]
        if missing:
            logger.info(
                "增量嵌入：复用 %d 条，新增 %d 条", len(reused), len(missing)
            )
        else:
            logger.info("向量按 id 全部复用：%d 条", len(store.chunks))

    if missing:
        vector_store.add_texts(
            [chunk.text for chunk in missing],
            ids=[chunk.id for chunk in missing],
        )
    if rebuild:
        logger.info("嵌入完成，耗时 %.1fs", time.perf_counter() - started)

    with atomic_path(directory / STORE_FILE, suffix=".json") as temp:
        vector_store.dump(str(temp))
    write_manifest(
        directory,
        _manifest_key(config, store),
        {
            "count": len(current_ids),
            "ids": current_ids,
            "chunks_file": str(store.path),
        },
        MANIFEST_FILE,
    )
    logger.info("向量库已落盘：%s", directory / STORE_FILE)
    return vector_store


def search_dense(
    vector_store: InMemoryVectorStore,
    store: ChunkStore,
    text: str,
    k: int,
) -> list[ScoredChunk]:
    """返回 top-k 余弦命中；相似度与排序由 `InMemoryVectorStore` 负责。"""
    if len(store) == 0 or k <= 0:
        return []
    hits: list[ScoredChunk] = []
    for rank, (document, score) in enumerate(
        vector_store.similarity_search_with_score(text, k=min(k, len(store))),
        start=1,
    ):
        chunk = store.get(document.id or "")
        if chunk is None:
            logger.warning("向量库中的 chunk id 不在语料里: %s", document.id)
            continue
        hits.append(
            ScoredChunk(
                chunk=chunk, score=float(score), channel="vector", rank=rank
            )
        )
    return hits
