"""`fastrag download`：下载示例文档。"""

from __future__ import annotations

import argparse

from fastrag.config import Config
from fastrag.ingest import download_docs


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("download", help="下载示例文档")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    summary = download_docs(config)
    print(
        f"成功 {summary.succeeded}/{summary.attempted}，"
        f"共 {summary.total_bytes / 1024 / 1024:.2f} MB -> {config.paths.docs_dir}"
    )
    for url, error in summary.failed:
        print(f"  失败 {url} {error}")
    return 0 if summary.succeeded else 1
