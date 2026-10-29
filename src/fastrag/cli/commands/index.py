"""`fastrag index`：构建 BM25 索引。"""

from __future__ import annotations

import argparse

from fastrag.config import Config
from fastrag.index import ChunkStore, build_index


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("index", help="构建 BM25 索引")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    store = ChunkStore.load(config)
    build_index(config, store)
    print(f"{len(store)} 个 chunk 已建索引 -> {config.paths.bm25_index_dir}")
    return 0
