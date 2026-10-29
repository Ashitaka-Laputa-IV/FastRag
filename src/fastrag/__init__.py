"""FastRag：本地文档的混合检索（BM25 + 向量）问答。

公共 API：

    from fastrag import ask, RagPipeline, load_config

    result = ask("human in the loop 怎么实现？")
    print(result.answer, result.sources)
"""

from fastrag.api import (
    AnswerEvent,
    AnswerResult,
    Chunk,
    Config,
    FastragError,
    RagPipeline,
    ScoredChunk,
    SearchPlan,
    ask,
    load_config,
    load_env,
    prepare,
    stream,
)
from fastrag.version import VERSION

__all__ = [
    "AnswerEvent",
    "AnswerResult",
    "Chunk",
    "Config",
    "FastragError",
    "RagPipeline",
    "ScoredChunk",
    "SearchPlan",
    "VERSION",
    "ask",
    "load_config",
    "load_env",
    "prepare",
    "stream",
]
