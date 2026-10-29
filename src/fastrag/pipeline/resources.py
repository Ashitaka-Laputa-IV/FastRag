"""资源容器：懒加载语料、向量、BM25、重排器与 LLM 链。

构造 `ResourceBundle` 不产生 I/O；第一次访问某个 `cached_property` 时才加载。
这样寒暄类问题不会触发嵌入或索引加载，测试也可以注入替身。
"""

from __future__ import annotations

import hashlib
from functools import cached_property

import bm25s
import numpy as np
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.runnables import Runnable
from langchain_openai import OpenAIEmbeddings

from fastrag.config import Config
from fastrag.domain import SearchPlan
from fastrag.index import ChunkStore, build_embeddings, ensure_vectors, load_index
from fastrag.llm import build_llm, build_with_fallbacks, extract_text, load_prompt
from fastrag.retrieve import build_reranker

PROMPT_NAMES = ("search_plan", "answer", "chat")


class ResourceBundle:
    """一次问答所需资源的懒加载容器。"""

    def __init__(self, config: Config) -> None:
        self.config = config

    @cached_property
    def store(self) -> ChunkStore:
        return ChunkStore.load(self.config)

    @cached_property
    def embeddings(self) -> OpenAIEmbeddings:
        return build_embeddings(self.config)

    @cached_property
    def vectors(self) -> np.ndarray:
        return ensure_vectors(self.config, self.store, self.embeddings)

    @cached_property
    def bm25(self) -> bm25s.BM25:
        return load_index(self.config, self.store)

    @cached_property
    def reranker(self) -> BaseDocumentCompressor | None:
        if self.config.fusion.method != "rerank":
            return None
        return build_reranker(self.config)

    @cached_property
    def plan_chain(self) -> Runnable:
        return build_with_fallbacks(
            lambda role: build_llm(role, self.config).with_structured_output(SearchPlan),
            self.config.llm.route,
        )

    @cached_property
    def answer_chain(self) -> Runnable:
        return build_with_fallbacks(
            lambda role: build_llm(role, self.config) | extract_text,
            self.config.llm.answer,
        )

    @cached_property
    def prompts(self) -> dict[str, str]:
        return {name: load_prompt(name) for name in PROMPT_NAMES}

    @cached_property
    def prompt_hashes(self) -> dict[str, str]:
        """提示词内容指纹，直接复用已加载的模板，避免二次读盘。"""
        return {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            for name, text in self.prompts.items()
        }

    def warmup(self) -> None:
        """显式预热全部资源（服务启动时可选调用）。"""
        _ = self.store, self.vectors, self.bm25
        if self.config.fusion.method == "rerank":
            _ = self.reranker
