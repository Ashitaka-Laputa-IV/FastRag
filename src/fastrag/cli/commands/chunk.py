"""`fastrag chunk`：切块并写入 `chunks.jsonl`。"""

from __future__ import annotations

import argparse

from fastrag.config import Config
from fastrag.ingest import build_chunks, stats, write_chunks


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("chunk", help="按配置切块")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    chunks = build_chunks(config)
    if not chunks:
        print(f"没有可切分的文档 -> {config.paths.docs_dir}，未生成 chunk 文件")
        return 1
    write_chunks(config, chunks)
    info = stats(chunks)
    print(
        f"{info.documents} 篇文档 -> {info.chunks} 个 chunk，"
        f"token 数 {info.min_tokens}~{info.max_tokens}，"
        f"平均 {info.avg_tokens:.0f} token -> {config.paths.chunks_file}"
    )
    return 0
