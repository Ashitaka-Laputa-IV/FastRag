"""库层统一异常，以及把异常转成退出码的 CLI 入口包装。

约定：

- `fastrag` 里的库函数只抛 `FastragError`（配置错误、数据产物缺失/过期、
  外部服务返回异常等），不直接调用 `SystemExit`。这样测试、服务或 notebook
  可以捕获并自行处理，而不是整个进程退出。
- 是否以非零码退出进程由 CLI 入口决定：各模块的 `__main__` 统一调用
  `run_cli(main)`，捕获 `FastragError` 后打印消息并以退出码 1 结束。
"""

from collections.abc import Callable


class FastragError(Exception):
    """fastrag 库层的统一异常。"""


def run_cli(main: Callable[[], None]) -> None:
    """执行 CLI 入口：把 `FastragError` 转成「错误: ...」+ 退出码 1。"""
    try:
        main()
    except FastragError as exc:
        raise SystemExit(f"错误: {exc}") from None
