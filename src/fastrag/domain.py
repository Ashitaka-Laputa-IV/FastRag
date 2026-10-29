"""领域类型：跨模块传递的统一数据结构。

这些类型是各层之间唯一的契约：

- `Chunk` / `ScoredChunk` 取代裸 dict 与 `(score, dict)` 元组；
- `SearchPlan` 是路由模型的结构化输出；
- `AnswerResult` / `AnswerEvent` 是库层对外的返回形态。

字段全部不可变（`frozen=True`），可以安全地在并发召回与图节点之间共享。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class Chunk:
    """文档切块的最小单元。"""

    id: str
    source: str
    start: int
    end: int
    text: str
    token_count: int = 0

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Chunk:
        """从 JSONL 记录还原；缺字段时抛 `KeyError`，由调用方转成配置/数据错误。"""
        return cls(
            id=str(record["id"]),
            source=str(record["source"]),
            start=int(record["start"]),
            end=int(record["end"]),
            text=str(record["text"]),
            token_count=int(record.get("token_count", 0)),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "token_count": self.token_count,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """一路检索结果：chunk + 分数 + 来源通道 + 通道内名次。"""

    chunk: Chunk
    score: float
    channel: str
    rank: int


class SearchPlan(BaseModel):
    """`plan_search` 节点的结构化输出：是否检索 + 两路各自的查询。"""

    need_search: bool = Field(description="回答该问题是否需要检索本地文档库")
    bm25_query: str = Field(
        default="",
        description="给 BM25 的英文关键词查询，空格分隔的术语；不需要检索时为空字符串",
    )
    vector_query: str = Field(
        default="",
        description="给向量检索的英文自然语言问句；不需要检索时为空字符串",
    )


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """一次问答的完整结果，供库调用方直接消费。"""

    answer: str
    sources: tuple[ScoredChunk, ...] = ()
    plan: SearchPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnswerEvent:
    """流式问答事件：`token` 增量文本，`done` 表示结束并携带最终结果。"""

    kind: Literal["token", "done"]
    text: str = ""
    result: AnswerResult | None = None
