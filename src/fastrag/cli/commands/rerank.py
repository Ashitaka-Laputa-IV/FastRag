"""`fastrag rerank`：对 BM25 候选做重排。"""

from __future__ import annotations

import argparse

from fastrag.cli.format import format_hits
from fastrag.config import Config
from fastrag.index import ChunkStore, load_index, search_bm25
from fastrag.retrieve import build_reranker, rerank_chunks


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("rerank", help="对 BM25 候选做重排")
    parser.add_argument("text", help="查询语句")
    parser.add_argument("-n", type=int, default=None, help="输出条数，默认取配置值")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    n = max(config.retrieval.top_n if args.n is None else args.n, 0)
    print(f"查询: {args.text}")
    if n == 0:
        return 0
    store = ChunkStore.load(config)
    retriever = load_index(config, store)
    candidates = search_bm25(
        retriever, store, config, args.text, config.retrieval.bm25_candidates
    )
    hits = rerank_chunks(build_reranker(config), args.text, candidates, n)
    for line in format_hits(hits, digits=4):
        print(line)
    return 0
