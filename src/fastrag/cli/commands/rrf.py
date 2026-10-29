"""`fastrag rrf`：融合 BM25 与向量两路排名。"""

from __future__ import annotations

import argparse

from fastrag.cli.format import snippet
from fastrag.config import Config
from fastrag.index import (
    ChunkStore,
    build_embeddings,
    ensure_vectors,
    load_index,
    search_bm25,
    search_dense,
)
from fastrag.retrieve import rrf_fuse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("rrf", help="融合 BM25 与向量两路排名")
    parser.add_argument("text", help="查询语句")
    parser.add_argument("-n", type=int, default=None, help="输出条数，默认取配置值")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    n = max(config.retrieval.top_n if args.n is None else args.n, 0)
    print(f"查询: {args.text}")
    if n == 0:
        return 0

    store = ChunkStore.load(config)
    embeddings = build_embeddings(config)
    vectors = ensure_vectors(config, store, embeddings)
    retriever = load_index(config, store)

    bm25 = search_bm25(
        retriever, store, config, args.text, config.retrieval.bm25_candidates
    )
    vector = search_dense(
        embeddings, vectors, store, args.text, config.retrieval.vector_candidates
    )
    rankings = {"bm25": bm25, "vector": vector}
    fused = rrf_fuse(rankings, config.fusion.rrf_k, config.fusion.weights)

    positions = {
        channel: {item.chunk.id: item.rank for item in ranking}
        for channel, ranking in rankings.items()
    }
    for item in fused[:n]:
        chunk = item.chunk
        where = " / ".join(
            f"{name} #{rank_map[chunk.id]}"
            if chunk.id in rank_map
            else f"{name} -"
            for name, rank_map in positions.items()
        )
        print(
            f"{item.rank}. {item.score:.4f}  "
            f"{chunk.source}#{chunk.start}-{chunk.end}  ({where})"
        )
        print(f"   {snippet(chunk.text)}")
    return 0
