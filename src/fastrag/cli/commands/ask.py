"""`fastrag ask`：交互式问答。"""

from __future__ import annotations

import argparse
import sys

from fastrag.config import Config
from fastrag.pipeline import RagPipeline


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ask", help="交互式问答")
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    pipeline = RagPipeline.from_config(config)

    while True:
        try:
            question = input("问题（回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0

        try:
            printed = False
            for event in pipeline.stream(question):
                if event.kind == "token":
                    if event.text:
                        printed = True
                        sys.stdout.write(event.text)
                        sys.stdout.flush()
                    continue
                result = event.result
                if result is None:
                    continue
                if not printed:
                    print(result.answer or "（本轮没有生成回答）")
                else:
                    print()
                if result.metadata.get("need_search"):
                    queries = result.metadata.get("queries") or {}
                    print(f"BM25 查询: {queries.get('bm25', '')}")
                    print(f"向量查询: {queries.get('vector', '')}")
                else:
                    print("未检索本地文档")
                for item in result.sources:
                    chunk = item.chunk
                    print(f"来源: {chunk.source}#{chunk.start}-{chunk.end}")
        except KeyboardInterrupt:
            print()
            return 0
        except Exception as exc:  # noqa: BLE001 - 单轮外部故障不终止交互
            print()
            print(
                f"本轮问答失败（{type(exc).__name__}: {exc}），可以重新提问或回车退出",
                file=sys.stderr,
            )
