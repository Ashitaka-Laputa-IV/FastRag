"""混合检索编排：一次跑 BM25 与向量两路。

图里的召回是并行节点（`retrieve_bm25` / `retrieve_vector`）；CLI 的
`fastrag rrf` 需要一次拿到两路结果，所以收在这里。`collect_rankings`
同时供图节点的 `fuse` 使用，通道名只在一处定义。
"""

from __future__ import annotations

from langchain_core.vectorstores import InMemoryVectorStore

from fastrag.config import Config
from fastrag.domain import ScoredChunk
from fastrag.index import ChunkStore, search_bm25, search_dense

CHANNELS = ("bm25", "vector")


def collect_rankings(
    bm25: list[ScoredChunk], vector: list[ScoredChunk]
) -> dict[str, list[ScoredChunk]]:
    """把两路排名装成融合输入，通道名与配置权重保持一致。"""
    return {"bm25": list(bm25), "vector": list(vector)}


def hybrid_search(
    config: Config,
    store: ChunkStore,
    bm25_retriever,
    vector_store: InMemoryVectorStore,
    bm25_query: str,
    vector_query: str | None = None,
) -> dict[str, list[ScoredChunk]]:
    """按配置的候选条数跑两路召回，返回按通道名归类的排名。"""
    retrieval = config.retrieval
    return collect_rankings(
        search_bm25(
            bm25_retriever, store, config, bm25_query, retrieval.bm25_candidates
        ),
        search_dense(
            vector_store,
            store,
            vector_query or bm25_query,
            retrieval.vector_candidates,
        ),
    )
