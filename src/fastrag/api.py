"""库门面：只导出稳定的公共 API。"""

from fastrag.build import prepare
from fastrag.config import Config, load_config, load_env
from fastrag.domain import AnswerEvent, AnswerResult, Chunk, ScoredChunk, SearchPlan
from fastrag.errors import FastragError
from fastrag.pipeline import RagPipeline, ask, stream

__all__ = [
    "AnswerEvent",
    "AnswerResult",
    "Chunk",
    "Config",
    "FastragError",
    "RagPipeline",
    "ScoredChunk",
    "SearchPlan",
    "ask",
    "load_config",
    "load_env",
    "prepare",
    "stream",
]
