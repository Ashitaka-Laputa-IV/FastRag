"""CLI 命令实现：每个命令只负责参数到库函数的映射与终端输出。"""

from __future__ import annotations

import argparse

from fastrag.cli.commands import (
    ask,
    chunk,
    download,
    embed,
    index,
    prepare,
    rerank,
    rrf,
    search,
)

COMMANDS = (download, chunk, embed, index, search, rerank, rrf, prepare, ask)


def register(subparsers: argparse._SubParsersAction) -> None:
    for module in COMMANDS:
        module.add_parser(subparsers)
