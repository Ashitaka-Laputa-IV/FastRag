"""Cross-encoder 重排：把 `ScoredChunk` 交给 Cohere 风格 `/rerank` 接口。

LangChain 的 `Document` 只在本模块内部使用，不越过模块边界。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_reranker.rerankers import init_reranker

from fastrag.config import Config, endpoint_api_key
from fastrag.domain import ScoredChunk


def build_reranker(config: Config) -> BaseDocumentCompressor:
    """按配置创建重排器。"""
    reranker = config.reranker
    return init_reranker(
        f"custom:{reranker.model}",
        base_url=config.endpoints[reranker.endpoint].base_url,
        api_key=endpoint_api_key(config, reranker.endpoint),
        timeout=reranker.timeout,
        top_n=config.retrieval.top_n,
    )


def rerank_chunks(
    reranker: BaseDocumentCompressor,
    query: str,
    candidates: Sequence[ScoredChunk],
    top_n: int,
) -> list[ScoredChunk]:
    """重排候选，返回按 relevance_score 降序的前 `top_n` 条。"""
    if not candidates or top_n <= 0:
        return []
    documents = [
        Document(
            page_content=item.chunk.text,
            metadata={"fastrag_index": index},
        )
        for index, item in enumerate(candidates)
    ]
    ranked = list(reranker.compress_documents(documents, query=query))
    results: list[ScoredChunk] = []
    for rank, document in enumerate(ranked[:top_n], start=1):
        index = document.metadata.get("fastrag_index")
        if not isinstance(index, int) or not 0 <= index < len(candidates):
            continue
        results.append(
            ScoredChunk(
                chunk=candidates[index].chunk,
                score=float(document.metadata.get("relevance_score", 0.0)),
                channel="rerank",
                rank=rank,
            )
        )
    return results
