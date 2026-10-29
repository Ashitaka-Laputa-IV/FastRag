"""`fastrag prepare`：一条命令刷新全部数据产物。"""

from __future__ import annotations

import argparse
import time

from fastrag.cli.commands import chunk as chunk_command
from fastrag.config import Config
from fastrag.index import ChunkStore, build_index, ensure_vectors


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prepare", help="切块 -> 向量 -> BM25 索引"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    started = time.perf_counter()
    print("[1/3] 切块")
    if chunk_command.run(args, config) != 0:
        return 1

    print("[2/3] 嵌入向量")
    store = ChunkStore.load(config)
    ensure_vectors(config, store)

    print("[3/3] BM25 索引")
    build_index(config, store)

    print(f"全部产物已刷新，耗时 {time.perf_counter() - started:.1f}s")
    return 0
