"""统一命令行入口：`fastrag <子命令> [参数...]`。

子命令与原来的模块入口一一对应，参数解析仍由各模块自己的 argparse 负责：
这里只负责全局选项（--config）与转发，因此
`fastrag search "hello" -k 3` 等价于
`uv run python -m fastrag.keyword_search query "hello" -k 3`。

用法:
    fastrag --help
    fastrag download                 # 下载示例文档
    fastrag chunk                    # 切块
    fastrag embed                    # 向量化（语料未变时复用）
    fastrag index                    # 建 BM25 索引
    fastrag prepare                  # 依次执行 chunk -> embed -> index
    fastrag search "human in the loop" -k 5
    fastrag rerank "human in the loop" -n 5
    fastrag rrf "human in the loop" -n 5
    fastrag ask                      # 交互式问答

`--config` 是可选的全局参数，既接受文件路径，也接受包内配置名（default / local），
例如 `fastrag --config local ask`。
"""

import argparse
import os
import pathlib
import sys

from fastrag import (
    chunk_docs,
    download_example_docs,
    embed_chunks,
    keyword_search,
    prepare,
    rag_graph,
    reranker,
    rrf,
)
from fastrag.config import PACKAGE_RESOURCES
from fastrag.errors import FastragError, run_cli

# 子命令 -> (模块入口, 固定前置参数, 说明)
COMMANDS = {
    "download": (download_example_docs.main, [], "下载示例文档"),
    "chunk": (chunk_docs.main, [], "按配置切块"),
    "embed": (embed_chunks.main, [], "生成并复用向量"),
    "index": (keyword_search.main, ["build"], "构建 BM25 索引"),
    "search": (keyword_search.main, ["query"], "BM25 关键词检索"),
    "rerank": (reranker.main, [], "对 BM25 候选做重排"),
    "rrf": (rrf.main, [], "融合 BM25 与向量两路排名"),
    "prepare": (prepare.main, [], "切块 → 向量 → 索引"),
    "ask": (rag_graph.main, [], "交互式问答"),
}


def resolve_config(value: str) -> str:
    """把 --config 的值解析成配置文件路径。

    优先当作路径；不存在时按包内配置名解析（`local` -> fastrag.local.toml）。
    """
    candidate = pathlib.Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    filename = "fastrag.toml" if value == "default" else f"fastrag.{value}.toml"
    packaged = pathlib.Path(str(PACKAGE_RESOURCES / "config" / filename))
    if packaged.exists():
        return str(packaged)
    raise FastragError(f"配置文件不存在: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastrag",
        description="本地文档混合检索问答（BM25 + 向量）",
        epilog="子命令的参数会透传给对应模块，例如 fastrag search --help。",
    )
    parser.add_argument(
        "--config",
        metavar="PATH|NAME",
        help="配置文件路径，或包内配置名（default / local）；等价于 FASTRAG_CONFIG",
    )
    sub = parser.add_subparsers(dest="command", metavar="<子命令>")
    for name, (_, _, help_text) in COMMANDS.items():
        # 参数由目标模块解析，子命令本身不注册选项
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def _pop_config(argv: list[str]) -> tuple[str | None, list[str]]:
    """取出子命令之前的全局 --config，支持 `--config X` 与 `--config=X`。"""
    config = None
    while argv:
        if argv[0] == "--config":
            if len(argv) < 2:
                raise FastragError("--config 需要一个值")
            config, argv = argv[1], argv[2:]
            continue
        if argv[0].startswith("--config="):
            config, argv = argv[0].split("=", 1)[1], argv[1:]
            continue
        break
    return config, argv


def main() -> None:
    run_cli(_main)


def _main() -> None:
    parser = _build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    config, argv = _pop_config(argv)
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        parser.error(f"未知子命令: {command}")
    if config:
        os.environ["FASTRAG_CONFIG"] = resolve_config(config)

    entry, fixed_args, _ = COMMANDS[command]
    sys.argv = [f"fastrag {command}", *fixed_args, *rest]
    entry()


if __name__ == "__main__":
    main()
