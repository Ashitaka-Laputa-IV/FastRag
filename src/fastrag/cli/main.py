"""统一命令行入口。

- `--config` / `-v` / `--version` 可放在子命令前后任意位置；
- 参数解析只在 `fastrag.cli` 发生，库模块不再各自维护 argparse 入口；
- 不修改 `sys.argv`，不向进程环境写入配置。
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from fastrag.cli.commands import register
from fastrag.config import Config, load_config, load_env, resolve_config
from fastrag.errors import FastragError
from fastrag.version import VERSION


class _Globals:
    def __init__(self) -> None:
        self.config: str | None = None
        self.verbose = False
        self.version = False


def _extract_globals(argv: list[str]) -> tuple[_Globals, list[str]]:
    """取出全局选项，返回剩余参数；支持 `--config X` 与 `--config=X`。"""
    globals_ = _Globals()
    rest: list[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--config":
            if index + 1 >= len(argv):
                raise FastragError("--config 需要一个值")
            globals_.config = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--config="):
            globals_.config = arg.split("=", 1)[1]
            index += 1
            continue
        if arg in ("-v", "--verbose"):
            globals_.verbose = True
            index += 1
            continue
        if arg in ("--version", "-V"):
            globals_.version = True
            index += 1
            continue
        rest.append(arg)
        index += 1
    return globals_, rest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastrag",
        description="本地文档混合检索问答（BM25 + 向量）",
        epilog="示例: fastrag --config local prepare && fastrag --config local ask",
    )
    sub = parser.add_subparsers(dest="command", metavar="<子命令>")
    register(sub)
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s",
        stream=sys.stderr,
    )


def _load(config_value: str | None) -> Config:
    if config_value is None:
        return load_config()
    return load_config(ref=resolve_config(config_value))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：返回退出码，不直接退出进程。"""
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        globals_, rest = _extract_globals(raw)
    except FastragError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if globals_.version:
        print(f"fastrag {VERSION}")
        return 0

    _configure_logging(globals_.verbose)

    parser = _build_parser()
    args = parser.parse_args(rest)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        load_env()
        config = _load(globals_.config)
    except FastragError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    try:
        return int(args.handler(args, config))
    except FastragError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
