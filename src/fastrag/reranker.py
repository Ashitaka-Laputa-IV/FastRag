"""用 langchain-reranker 对召回结果做重排。

- 通过 `init_reranker("custom:<model>", base_url=..., api_key=...)` 创建重排器，
  调用任意 Cohere 风格的 `/rerank` 接口（交叉编码器，默认配置指向 SiliconFlow）
- 排序、top_n 与 relevance_score 写回 metadata 均由 `compress_documents` 提供
- 参数来自 config/fastrag.toml 的 [reranker]、[endpoints]、[retrieval]
- 演示流程：BM25 召回 bm25_candidates 条 -> 重排 -> 输出 top_n 条
  （`rerank_chunks` 只依赖 chunk 列表，可接任意检索器的候选）

用法:
    uv run python -m fastrag.reranker "human in the loop approval"

输出示例:
    $ uv run python -m fastrag.reranker "human in the loop approval"
    查询: human in the loop approval
    1. 0.9964  data/langchain-docs/oss/python/langchain/frontend/human-in-the-loop.md#0-2280  (bm25 #11)
       > ## Documentation Index > Fetch the complete documentation index at: https://docs...
    ...
"""

import argparse
import os

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_reranker.rerankers import init_reranker

from fastrag.config import Config, load_config, load_env


def build_reranker(config: Config) -> BaseDocumentCompressor:
    """按配置创建重排器，缺 key 时给出明确提示。

    endpoint 的 api_key_env 留空视为本地服务，此时用占位 key。
    """
    api_key = os.environ.get(config.reranker.endpoint.api_key_env)
    if not api_key:
        if config.reranker.endpoint.api_key_env:
            raise SystemExit(
                f"缺少 {config.reranker.endpoint.api_key_env}，"
                "请先 cp .env.example .env 并填写"
            )
        api_key = "local"
    return init_reranker(
        f"custom:{config.reranker.model}",
        base_url=config.reranker.endpoint.base_url,
        api_key=api_key,
        timeout=config.reranker.timeout,
        top_n=config.retrieval.top_n,
    )


def rerank_chunks(
    reranker: BaseDocumentCompressor, query: str, chunks: list[dict]
) -> list[Document]:
    """把候选 chunk 交给重排器，返回带 relevance_score 的文档（降序）。"""
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "id": chunk["id"],
                "source": chunk["source"],
                "start": chunk["start"],
                "end": chunk["end"],
                "retrieval_rank": rank,
            },
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]
    return list(reranker.compress_documents(documents, query=query))


def main() -> None:
    # 演示只用到关键词检索，导入放在 main 里，避免模块级依赖 bm25s
    from fastrag.keyword_search import (
        format_snippet,
        load_index,
        search as search_keywords,
    )

    parser = argparse.ArgumentParser(description="对检索候选做重排")
    parser.add_argument("text", help="查询语句")
    parser.add_argument("-n", type=int, default=None, help="输出条数，默认取配置值")
    args = parser.parse_args()

    load_env()
    config = load_config()
    candidates = search_keywords(
        load_index(config), config, args.text, config.retrieval.bm25_candidates
    )
    documents = rerank_chunks(
        build_reranker(config), args.text, [chunk for _, chunk in candidates]
    )

    print(f"查询: {args.text}")
    for rank, document in enumerate(
        documents[: args.n or config.retrieval.top_n], start=1
    ):
        metadata = document.metadata
        print(
            f"{rank}. {metadata['relevance_score']:.4f}  "
            f"{metadata['source']}#{metadata['start']}-{metadata['end']}  "
            f"(bm25 #{metadata['retrieval_rank']})"
        )
        print(f"   {format_snippet(document.page_content)}")


if __name__ == "__main__":
    main()
