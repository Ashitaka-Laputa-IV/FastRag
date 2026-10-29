"""`fastrag search`：BM25 关键词检索。"""

from __future__ import annotations

import argparse

from fastrag.cli.format import format_hits
from fastrag.config import Config
from fastrag.index import ChunkStore, load_index, search_bm25


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("search", help="BM25 关键词检索")
    parser.add_argument("text", help="查询语句")
    parser.add_argument("-k", type=int, default=None, help="返回条数，默认取配置值")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    store = ChunkStore.load(config)
    retriever = load_index(config, store)
    k = config.retrieval.top_n if args.k is None else max(args.k, 0)
    hits = search_bm25(retriever, store, config, args.text, k)
    for line in format_hits(hits):
        print(line)
    return 0
