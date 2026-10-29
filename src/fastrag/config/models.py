"""配置模型：用 pydantic 强制 schema，未知键与非法值在加载期失败。

与旧的「手写 dict 映射 + 手写校验」相比：

- `extra="forbid"`：`[retrival]`、`chunk_sise` 这类拼写错误不会再被静默忽略；
- 字段级 `Field(gt=..., le=...)`：范围校验与字段定义写在一起；
- 跨字段规则（overlap < size、endpoint 引用存在）用 `model_validator`。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FUSION_METHODS = ("rrf", "rerank")
BM25_METHODS = ("lucene", "robertson", "atire", "bm25l", "bm25+")
BM25_STOPWORDS = (
    "english",
    "en",
    "english_plus",
    "en_plus",
    "german",
    "de",
    "dutch",
    "nl",
    "french",
    "fr",
    "spanish",
    "es",
    "portuguese",
    "pt",
    "italian",
    "it",
    "russian",
    "ru",
    "swedish",
    "sv",
    "norwegian",
    "no",
    "chinese",
    "zh",
    "turkish",
    "tr",
    "korean",
    "ko",
    "danish",
    "da",
)


class _Strict(BaseModel):
    """所有配置节的基类：禁止未知键、不可变。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    """文件与目录，加载后均为绝对路径。"""

    docs_dir: Path
    chunks_file: Path
    embeddings_dir: Path
    bm25_index_dir: Path


class Endpoint(_Strict):
    """一家服务商的接入信息。"""

    base_url: str
    api_key_env: str = ""


class Chunking(_Strict):
    tokenizer: str = "cl100k_base"
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=128, ge=0)
    include_delim: str = "prev"
    boilerplate_pattern: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Chunking":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) 必须小于 chunk_size ({self.chunk_size})"
            )
        if self.boilerplate_pattern:
            try:
                re.compile(self.boilerplate_pattern)
            except re.error as exc:
                raise ValueError(f"boilerplate_pattern 不是合法正则: {exc}") from None
        return self


class Embedding(_Strict):
    endpoint: str
    model: str
    batch_size: int = Field(default=32, gt=0)
    dimensions: int = Field(default=2560, gt=0)
    timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class LLMConfig(_Strict):
    """单个 LLM 角色。`fallback` 可继续嵌套，形成多级兜底。"""

    endpoint: str
    model: str
    provider: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    thinking: bool = True
    timeout: float = Field(default=120.0, gt=0)
    retries: int = Field(default=0, ge=0)
    fallback: "LLMConfig | None" = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class LLMRoles(_Strict):
    route: LLMConfig
    answer: LLMConfig


class Reranker(_Strict):
    endpoint: str
    model: str
    timeout: float = Field(default=60.0, gt=0)


class BM25(_Strict):
    stopwords: str | list[str] = "english"
    method: Literal["lucene", "robertson", "atire", "bm25l", "bm25+"] = "lucene"
    min_score: float = 0.0

    @model_validator(mode="after")
    def _check_stopwords(self) -> "BM25":
        if isinstance(self.stopwords, str) and self.stopwords not in BM25_STOPWORDS:
            raise ValueError(
                f"stopwords 必须是内置语言名或字符串列表，当前为 {self.stopwords!r}"
            )
        return self


class Retrieval(_Strict):
    bm25_candidates: int = Field(default=20, gt=0)
    vector_candidates: int = Field(default=20, gt=0)
    top_n: int = Field(default=5, gt=0)
    # 送入 LLM 的片段数；不与 CLI 默认返回条数（top_n）混用
    context_chunks: int | None = Field(default=None, gt=0)


class Fusion(_Strict):
    method: Literal["rrf", "rerank"] = "rrf"
    rrf_k: int = Field(default=60, gt=0)
    # 按通道名给权重，新增检索通道时不需要改融合代码
    weights: dict[str, float] = Field(
        default_factory=lambda: {"bm25": 1.1, "vector": 1.0}
    )

    @model_validator(mode="after")
    def _check_weights(self) -> "Fusion":
        if not self.weights:
            raise ValueError("weights 不能为空")
        for channel in ("bm25", "vector"):
            if channel not in self.weights:
                raise ValueError(f"weights 缺少通道 {channel!r}")
        for name, weight in self.weights.items():
            if weight <= 0:
                raise ValueError(f"weights[{name!r}] 必须大于 0，当前为 {weight!r}")
        return self


class Download(_Strict):
    index_url: str = (
        "https://docs.langchain.com/_llms/agent-development-lifecycle/build/python.txt"
    )
    base_url: str = "https://docs.langchain.com/"
    concurrency: int = Field(default=12, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    retries: int = Field(default=3, ge=0)
    user_agent: str = "fastrag-docs-downloader/0.1"
    exclude: list[str] = Field(default_factory=lambda: ["/contributing/"])


class Config(_Strict):
    """配置文件的完整内容。"""

    paths: Paths
    endpoints: dict[str, Endpoint]
    chunking: Chunking
    embedding: Embedding
    llm: LLMRoles
    reranker: Reranker
    bm25: BM25
    retrieval: Retrieval
    fusion: Fusion
    download: Download = Field(default_factory=Download)

    @model_validator(mode="after")
    def _check_endpoints(self) -> "Config":
        known = set(self.endpoints)
        references: list[tuple[str, str]] = [
            ("embedding", self.embedding.endpoint),
            ("reranker", self.reranker.endpoint),
        ]
        role: LLMConfig | None = self.llm.route
        name = "llm.route"
        while role is not None:
            references.append((name, role.endpoint))
            role = role.fallback
            name = f"{name}.fallback"
        role = self.llm.answer
        name = "llm.answer"
        while role is not None:
            references.append((name, role.endpoint))
            role = role.fallback
            name = f"{name}.fallback"
        for owner, endpoint in references:
            if endpoint not in known:
                raise ValueError(
                    f"{owner}.endpoint = {endpoint!r} 未在 [endpoints] 中定义"
                )
        return self


LLMConfig.model_rebuild()
