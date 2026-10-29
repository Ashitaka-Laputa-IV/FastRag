"""对外问答门面：`RagPipeline` / `ask` / `stream`。

构造 `RagPipeline` 不读盘、不联网；资源在第一次检索时懒加载。
同一进程可以为不同 `Config` 各建一个实例，互不干扰。
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Iterator

from fastrag.config import Config, load_config, load_env
from fastrag.domain import AnswerEvent, AnswerResult
from fastrag.pipeline.graph import build_graph
from fastrag.pipeline.nodes import RagNodes, RagState, result_from_state
from fastrag.pipeline.resources import ResourceBundle


class RagPipeline:
    """一次配置对应一个问答流水线。"""

    def __init__(self, config: Config, resources: ResourceBundle | None = None) -> None:
        self.config = config
        self.resources = resources or ResourceBundle(config)
        self.nodes = RagNodes(self.resources)
        self.graph = build_graph(self.nodes)

    @classmethod
    def from_config(cls, config: Config | None = None) -> "RagPipeline":
        return cls(config or load_config())

    def warmup(self) -> None:
        """显式加载全部资源（服务启动时可调用）。"""
        self.resources.warmup()

    def _metadata(self) -> dict:
        answer = self.config.llm.answer
        return {
            "model": answer.model,
            "provider": answer.provider,
            "fusion": self.config.fusion.method,
            "prompt_sha": self.resources.prompt_hashes,
        }

    def ask(self, question: str) -> AnswerResult:
        """同步问答，返回回答、来源与元数据。"""
        started = time.perf_counter()
        state: RagState = self.graph.invoke({"question": question})
        return result_from_state(
            state, self._metadata(), time.perf_counter() - started
        )

    def stream(self, question: str) -> Iterator[AnswerEvent]:
        """流式问答：逐个 token 产出，最后产出带 `AnswerResult` 的 `done`。"""
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
            warnings.filterwarnings(
                "ignore",
                message="The v3 streaming protocol on Pregel is experimental.",
            )
            run = self.graph.stream_events({"question": question}, version="v3")
            for stream in run.messages:
                if stream.node != "answer":
                    continue
                for delta in stream.text:
                    yield AnswerEvent(kind="token", text=delta)
            state: RagState = run.output or {}
        yield AnswerEvent(
            kind="done",
            result=result_from_state(
                state, self._metadata(), time.perf_counter() - started
            ),
        )


def ask(question: str, config: Config | None = None) -> AnswerResult:
    """便捷入口：加载 `.env` 与配置，跑完一次问答。"""
    load_env()
    return RagPipeline.from_config(config or load_config()).ask(question)


def stream(question: str, config: Config | None = None) -> Iterator[AnswerEvent]:
    """便捷入口：流式问答。"""
    load_env()
    return RagPipeline.from_config(config or load_config()).stream(question)
