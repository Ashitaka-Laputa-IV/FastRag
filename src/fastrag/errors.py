"""库层统一异常。

`fastrag` 里的库函数只抛 `FastragError`（配置错误、数据产物缺失/过期、
外部服务返回异常等），不直接调用 `SystemExit`。是否以非零码退出进程
由 CLI 入口决定。
"""


class FastragError(Exception):
    """fastrag 库层的统一异常。"""
