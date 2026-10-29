"""`fastrag prepare`：一条命令刷新全部数据产物。"""

from __future__ import annotations

import argparse

from fastrag.build import prepare
from fastrag.config import Config


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare", help="切块 -> 向量 -> BM25 索引")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    report = prepare(config)
    info = report.chunks
    print(
        f"{info.documents} 篇文档 -> {info.chunks} 个 chunk，"
        f"向量 {report.vector_shape[0]}x{report.vector_shape[1]}，"
        f"耗时 {report.elapsed:.1f}s"
    )
    return 0
