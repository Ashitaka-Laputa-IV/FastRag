"""`fastrag embed`：生成或复用向量。"""

from __future__ import annotations

import argparse

from fastrag.cli.format import format_hits
from fastrag.config import Config
from fastrag.index import (
    ChunkStore,
    build_embeddings,
    ensure_vector_store,
    search_dense,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("embed", help="生成并复用向量")
    parser.add_argument(
        "--query", help="构建后可选地跑一次向量检索演示", default=None
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    store = ChunkStore.load(config)
    embeddings = build_embeddings(config)
    vector_store = ensure_vector_store(config, store, embeddings)
    print(f"{len(store)} 个 chunk -> {config.embedding.dimensions} 维")
    if args.query:
        hits = search_dense(vector_store, store, args.query, config.retrieval.top_n)
        for line in format_hits(hits):
            print(line)
    return 0
