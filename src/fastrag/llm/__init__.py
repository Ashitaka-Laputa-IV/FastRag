"""LLM 层：客户端、兜底链与提示词。"""

from fastrag.llm.client import (
    EmptyResponseError,
    build_llm,
    build_with_fallbacks,
    extract_text,
)
from fastrag.llm.prompts import load_prompt, render, render_text

__all__ = [
    "EmptyResponseError",
    "build_llm",
    "build_with_fallbacks",
    "extract_text",
    "load_prompt",
    "render",
    "render_text",
]
