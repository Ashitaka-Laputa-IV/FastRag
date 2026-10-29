"""多路排名融合：Reciprocal Rank Fusion。

    score(d) = Σ_channel w_channel / (k + rank_channel(d))

权重按通道名配置（`[fusion.weights] bm25 = 1.1, vector = 1.0`），
新增检索通道时不需要修改融合代码。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fastrag.domain import Chunk, ScoredChunk


def rrf_fuse(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    k: int,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredChunk]:
    """按 RRF 融合多路排名，返回按分数降序的 `ScoredChunk`。"""
    if k <= 0:
        raise ValueError("rrf_k 必须大于 0")
    weights = weights or {}

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for channel, ranking in rankings.items():
        weight = float(weights.get(channel, 1.0))
        for rank, item in enumerate(ranking, start=1):
            key = item.chunk.id
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            chunks.setdefault(key, item.chunk)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        ScoredChunk(chunk=chunks[key], score=scores[key], channel="rrf", rank=rank)
        for rank, key in enumerate(ordered, start=1)
    ]
