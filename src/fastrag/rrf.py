"""Reciprocal Rank Fusion（RRF）：融合 BM25 与向量检索的排名。

公式:
    score(d) = Σ_i w_i / (k + rank_i(d))

其中 rank_i(d) 是文档 d 在第 i 路排名中的名次（从 1 开始），k 是平滑常数
（默认 60），w_i 是各路权重；只出现在单路的文档同样按该式计分。

- 参数来自 config/rrf.toml（k、权重、每路候选数、输出条数）
- 两路候选：keyword_search（BM25）与 embed_chunks（向量余弦相似度）
- 用法:
    uv run python -m fastrag.rrf "human in the loop approval"

输出示例:
    $ uv run python -m fastrag.rrf "human in the loop approval"
    查询: human in the loop approval
    1. 0.0313  data/langchain-docs/oss/python/deepagents/human-in-the-loop.md#0-2111  (bm25 #7 / 向量 #1)
       > ## Documentation Index > Fetch the complete documentation index at: https://docs...
    ...
"""

import argparse
import pathlib
import tomllib
from dataclasses import dataclass
from typing import Sequence

from fastrag.embed_chunks import (
    ensure_vectors,
    load_chunks,
    load_config as load_embedding_config,
    load_env,
    search as search_vectors,
)
from fastrag.keyword_search import (
    format_snippet,
    load_config as load_keyword_config,
    search as search_keywords,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "rrf.toml"


@dataclass(frozen=True)
class FusionConfig:
    """融合配置，对应 config/rrf.toml。"""

    k: int
    weights: tuple[float, ...]
    candidates: int
    top_n: int


def load_config(path: pathlib.Path = CONFIG_PATH) -> FusionConfig:
    """读取 TOML 配置。"""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))["fusion"]
    return FusionConfig(
        k=raw["k"],
        weights=tuple(raw["weights"]),
        candidates=raw["candidates"],
        top_n=raw["top_n"],
    )


def rrf_fuse(
    rankings: Sequence[Sequence[dict]],
    k: int,
    weights: Sequence[float] | None = None,
) -> list[tuple[float, dict]]:
    """按 RRF 融合多路排名，返回 (分数, chunk) 的降序列表。

    rankings 中每一路都是按相关度降序排列的 chunk 列表（需含 id 字段）；
    同一个 id 在多路命中时分数累加，只有单路命中的也计入最终排名。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights 数量必须与 rankings 一致")

    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, chunk in enumerate(ranking, start=1):
            key = chunk["id"]
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            chunks.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [(scores[key], chunks[key]) for key in ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description="RRF 融合 BM25 与向量检索")
    parser.add_argument("text", help="查询语句")
    parser.add_argument("-n", type=int, default=None, help="输出条数，默认取配置值")
    args = parser.parse_args()

    fusion = load_config()

    bm25_hits = search_keywords(load_keyword_config(), args.text, fusion.candidates)

    load_env()
    embedding_config = load_embedding_config()
    chunks = load_chunks(embedding_config)
    vectors = ensure_vectors(embedding_config, chunks)
    vector_hits = search_vectors(
        embedding_config, vectors, chunks, args.text, fusion.candidates
    )

    rankings = [[chunk for _, chunk in bm25_hits], [chunk for _, chunk in vector_hits]]
    fused = rrf_fuse(rankings, fusion.k, fusion.weights)

    names = ("bm25", "向量")
    ranks = [
        {chunk["id"]: rank for rank, chunk in enumerate(ranking, start=1)}
        for ranking in rankings
    ]
    print(f"查询: {args.text}")
    for rank, (score, chunk) in enumerate(fused[: args.n or fusion.top_n], start=1):
        positions = " / ".join(
            f"{name} #{rank_map[chunk['id']]}" if chunk["id"] in rank_map else f"{name} -"
            for name, rank_map in zip(names, ranks)
        )
        print(
            f"{rank}. {score:.4f}  {chunk['source']}#{chunk['start']}-{chunk['end']}"
            f"  ({positions})"
        )
        print(f"   {format_snippet(chunk['text'])}")


if __name__ == "__main__":
    main()
