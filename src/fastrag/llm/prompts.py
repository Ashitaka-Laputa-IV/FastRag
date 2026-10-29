"""提示词读取与渲染。

模板使用 `string.Template` 的 `$name` 占位符，避免 `str.format` 遇到
JSON 示例或代码片段里的字面量花括号时报错。用户可用
`$FASTRAG_HOME/prompts/<name>.txt` 覆盖包内模板。
"""

from __future__ import annotations

import hashlib
import importlib.resources
from string import Template

from fastrag.config import home
from fastrag.errors import FastragError

PACKAGE_RESOURCES = importlib.resources.files("fastrag.resources")
REQUIRED = ("search_plan", "answer", "chat")


def load_prompt(name: str) -> str:
    """读取提示词模板：优先工作目录下的 `prompts/`，否则用包内默认模板。"""
    override = home() / "prompts" / f"{name}.txt"
    if override.exists():
        try:
            return override.read_text(encoding="utf-8")
        except OSError as exc:
            raise FastragError(f"读取提示词失败: {override}: {exc}") from None
    resource = PACKAGE_RESOURCES / "prompts" / f"{name}.txt"
    if not resource.is_file():
        raise FastragError(f"提示词不存在: {name}")
    return resource.read_text(encoding="utf-8")


def render(name: str, **values: str) -> str:
    """用 `$name` 占位符渲染模板；未知占位符原样保留，避免静默丢内容。"""
    return Template(load_prompt(name)).safe_substitute(values)


def prompt_sha(name: str) -> str:
    """提示词内容指纹，用于回答元数据与实验复现。"""
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()[:12]
