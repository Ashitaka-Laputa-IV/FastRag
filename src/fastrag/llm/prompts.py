"""提示词读取与渲染。

模板使用 langchain-core 的 `PromptTemplate`（mustache：`{{name}}`），
`PromptTemplate` 负责占位符校验与渲染；mustache 不会把 JSON 示例或代码
片段里的字面量花括号当成占位符。用户可用
`$FASTRAG_HOME/prompts/<name>.txt` 覆盖包内模板。
"""

from __future__ import annotations

import hashlib
import importlib.resources

from langchain_core.prompts import PromptTemplate

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


def build_template(name: str) -> PromptTemplate:
    """按名称构造提示词模板。"""
    return PromptTemplate.from_template(load_prompt(name), template_format="mustache")


def render_prompt(template: PromptTemplate, name: str, **values: str) -> str:
    """渲染已加载的模板；缺少占位符取值时直接报错，避免静默丢内容。"""
    missing = [var for var in template.input_variables if var not in values]
    if missing:
        raise FastragError(f"提示词 {name} 缺少取值: {', '.join(missing)}")
    return template.format(**values)


def render(name: str, **values: str) -> str:
    """按名称加载并渲染模板。"""
    return render_prompt(build_template(name), name, **values)


def prompt_sha(name: str) -> str:
    """提示词内容指纹，用于回答元数据与实验复现。"""
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()[:12]
