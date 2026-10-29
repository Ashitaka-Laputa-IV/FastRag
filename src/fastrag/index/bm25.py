"""BM25 索引：构建、校验与关键词检索。

索引里只保存 chunk id，不再复制整份语料；检索后通过 `ChunkStore` 回表，
因此 `chunks.jsonl` 与索引不会出现两份内容不一致的问题。
"""

from __future__ import annotations

import logging
import pathlib

import bm25s

from fastrag.config import Config
from fastrag.domain import ScoredChunk
from fastrag.errors import FastragError
from fastrag.index.store import ChunkStore
from fastrag.ingest.artifacts import ArtifactKey, matches, write_manifest

logger = logging.getLogger(__name__)

META_FILE = "meta.json"
REBUILD_HINT = "fastrag index（或 uv run python -m fastrag prepare）"


def index_params(config: Config) -> dict:
    """记录影响索引的构建参数。"""
    return {"method": config.bm25.method, "stopwords": config.bm25.stopwords}


def _key(config: Config, store: ChunkStore) -> ArtifactKey:
    return ArtifactKey("bm25", store.digest, index_params(config))


def build_index(config: Config, store: ChunkStore | None = None) -> None:
    """对全部 chunk 建立 BM25 索引，并把 id 语料与元信息落盘。"""
    store = store or ChunkStore.load(config)
    if not len(store):
        raise FastragError("chunk 语料为空，无法建立 BM25 索引")
    tokenized = bm25s.tokenize(
        [chunk.text for chunk in store.chunks],
        stopwords=config.bm25.stopwords,
        show_progress=False,
    )
    retriever = bm25s.BM25(method=config.bm25.method)
    retriever.index(tokenized, show_progress=False)

    index_dir = config.paths.bm25_index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(
        index_dir, corpus=[chunk.id for chunk in store.chunks], show_progress=False
    )
    write_manifest(
        index_dir,
        _key(config, store),
        {"count": len(store), "chunks_file": str(store.path)},
        META_FILE,
    )
    logger.info("%d 个 chunk 已建索引 -> %s", len(store), index_dir)


def check_index(config: Config, store: ChunkStore) -> None:
    """校验索引与当前语料、BM25 参数一致；不一致时要求重建。"""
    meta_path = config.paths.bm25_index_dir / META_FILE
    if not meta_path.exists():
        raise FastragError(
            f"BM25 索引缺少 {META_FILE}（旧版索引无法校验），请重新运行: {REBUILD_HINT}"
        )
    key = _key(config, store)
    if not matches(config.paths.bm25_index_dir, key, META_FILE):
        raise FastragError(
            "chunks.jsonl 或 bm25 参数已变更，BM25 索引与语料不一致，"
            f"请重新运行: {REBUILD_HINT}"
        )


def load_index(config: Config, store: ChunkStore | None = None) -> bm25s.BM25:
    """加载 BM25 索引（含 id corpus），加载前校验语料与构建参数。"""
    store = store or ChunkStore.load(config)
    index_dir = config.paths.bm25_index_dir
    if not index_dir.exists():
        raise FastragError(f"索引不存在，请先运行: {REBUILD_HINT}")
    check_index(config, store)
    return bm25s.BM25.load(index_dir, load_corpus=True, show_progress=False)


def search(
    retriever: bm25s.BM25,
    store: ChunkStore,
    config: Config,
    text: str,
    k: int,
    min_score: float | None = None,
) -> list[ScoredChunk]:
    """返回 top-k 关键词命中。"""
    if retriever.corpus is None or len(retriever.corpus) == 0 or k <= 0:
        return []
    query_tokens = bm25s.tokenize(
        text, stopwords=config.bm25.stopwords, show_progress=False
    )
    results, scores = retriever.retrieve(
        query_tokens, k=min(k, len(retriever.corpus)), show_progress=False
    )
    if results.size == 0:
        return []
    threshold = config.bm25.min_score if min_score is None else min_score
    hits: list[ScoredChunk] = []
    for index, (score, raw) in enumerate(zip(scores[0], results[0]), start=1):
        if float(score) <= threshold:
            continue
        if isinstance(raw, dict):
            # bm25s 会把字符串 corpus 包装成 {"id": i, "text": 原字符串}
            raw_id = raw.get("text", raw.get("id"))
        else:
            raw_id = raw
        chunk = store.get(str(raw_id))
        if chunk is None:
            logger.warning("索引中的 chunk id 不在语料里: %s", raw_id)
            continue
        hits.append(
            ScoredChunk(
                chunk=chunk, score=float(score), channel="bm25", rank=index
            )
        )
    return hits


def index_path(config: Config) -> pathlib.Path:
    return config.paths.bm25_index_dir
