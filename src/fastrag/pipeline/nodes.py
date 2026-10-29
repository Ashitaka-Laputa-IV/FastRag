"""图节点：plan_search / retrieve / fuse / answer。

节点只读取 `ResourceBundle`，不负责资源创建；`Pipeline` 负责装配。
外部服务失败在离失败点最近的地方降级：

- 路由模型全部失败 → 用原始问题继续检索；
- 重排失败 → 退回 RRF；
- 回答模型全部失败 → 返回空回答，由上层提示。
"""

from __future__ import annotations

import logging
from string import Template
from typing import TypedDict

from fastrag.domain import AnswerResult, ScoredChunk
from fastrag.index import search_bm25, search_dense
from fastrag.retrieve import rerank_chunks, rrf_fuse

logger = logging.getLogger(__name__)


class RagState(TypedDict, total=False):
    """图状态：问题、检索计划、两路召回、融合结果、回答。"""

    question: str
    plan: object
    need_search: bool
    bm25_query: str
    vector_query: str
    bm25: list[ScoredChunk]
    vector: list[ScoredChunk]
    fused: list[ScoredChunk]
    answer: str


def _render(prompt: str, **values: str) -> str:
    return Template(prompt).safe_substitute(values)


class RagNodes:
    """RAG 图节点的实现。"""

    def __init__(self, resources) -> None:
        self.resources = resources
        self.config = resources.config

    def plan_search(self, state: RagState) -> dict:
        """一次调用同时判断是否检索，并给出两路各自的查询。"""
        prompt = _render(
            self.resources.prompts["search_plan"], question=state["question"]
        )
        try:
            plan = self.resources.plan_chain.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - 路由不可用时仍要保证可检索
            return self._degrade_route(
                state, f"路由模型全部失败（{type(exc).__name__}: {exc}）"
            )
        if plan is None:
            return self._degrade_route(state, "路由模型返回空结果")
        if not plan.need_search:
            return {
                "plan": plan,
                "need_search": False,
                "bm25_query": "",
                "vector_query": "",
            }
        question = state["question"]
        return {
            "plan": plan,
            "need_search": True,
            "bm25_query": plan.bm25_query.strip() or question,
            "vector_query": plan.vector_query.strip() or question,
        }

    @staticmethod
    def _degrade_route(state: RagState, reason: str) -> dict:
        logger.warning("%s，使用原始问题继续检索", reason)
        question = state["question"]
        return {
            "plan": None,
            "need_search": True,
            "bm25_query": question,
            "vector_query": question,
        }

    def retrieve_bm25(self, state: RagState) -> dict:
        hits = search_bm25(
            self.resources.bm25,
            self.resources.store,
            self.config,
            state["bm25_query"],
            self.config.retrieval.bm25_candidates,
        )
        return {"bm25": hits}

    def retrieve_vector(self, state: RagState) -> dict:
        hits = search_dense(
            self.resources.embeddings,
            self.resources.vectors,
            self.resources.store,
            state["vector_query"],
            self.config.retrieval.vector_candidates,
        )
        return {"vector": hits}

    def fuse(self, state: RagState) -> dict:
        """按配置融合；rerank 失败时自动降级为 RRF。"""
        bm25 = state.get("bm25", [])
        vector = state.get("vector", [])
        if not bm25 and not vector:
            return {"fused": []}
        rankings = {"bm25": bm25, "vector": vector}

        if self.config.fusion.method == "rerank":
            merged = self._merge(bm25, vector)
            query = state.get("vector_query") or state["question"]
            try:
                fused = rerank_chunks(
                    self.resources.reranker,
                    query,
                    merged,
                    self.config.retrieval.top_n,
                )
            except Exception as exc:  # noqa: BLE001 - 外部服务失败时保可用性
                logger.warning(
                    "重排失败（%s: %s），降级为 RRF", type(exc).__name__, exc
                )
                return {"fused": self._fuse_rrf(rankings)}
            return {"fused": fused}

        return {"fused": self._fuse_rrf(rankings)}

    def _fuse_rrf(self, rankings: dict[str, list[ScoredChunk]]) -> list[ScoredChunk]:
        fused = rrf_fuse(rankings, self.config.fusion.rrf_k, self.config.fusion.weights)
        return fused[: self.config.retrieval.top_n]

    def answer(self, state: RagState) -> dict:
        """按是否需要检索选择提示词，再让 LLM 作答。"""
        if state.get("need_search", True):
            limit = (
                self.config.retrieval.context_chunks
                or self.config.retrieval.top_n
            )
            chunks = state.get("fused", [])[:limit]
            context = self._context(chunks)
            prompt = _render(
                self.resources.prompts["answer"],
                context=context,
                question=state["question"],
            )
        else:
            prompt = _render(
                self.resources.prompts["chat"], question=state["question"]
            )
        return {"answer": self._answer_text(prompt)}

    @staticmethod
    def _context(chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "（无检索结果）"
        return "\n\n".join(
            f"[{index}] {item.chunk.source}\n{item.chunk.text}"
            for index, item in enumerate(chunks, start=1)
        )

    def _answer_text(self, prompt: str) -> str:
        try:
            return self.resources.answer_chain.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - 单轮失败不退出交互
            logger.error(
                "回答模型全部失败（%s: %s），本轮没有回答",
                type(exc).__name__,
                exc,
            )
            return ""

    @staticmethod
    def _merge(*rankings: list[ScoredChunk]) -> list[ScoredChunk]:
        """按 chunk id 去重合并多路候选，保留先出现的顺序。"""
        seen: set[str] = set()
        merged: list[ScoredChunk] = []
        for ranking in rankings:
            for item in ranking:
                if item.chunk.id in seen:
                    continue
                seen.add(item.chunk.id)
                merged.append(item)
        return merged

    @staticmethod
    def route_after_plan(state: RagState) -> list[str]:
        """需要检索就同时走 BM25 与向量两路，否则直接回答。"""
        if not state.get("need_search", True):
            return ["answer"]
        return ["retrieve_bm25", "retrieve_vector"]


def result_from_state(
    state: RagState, metadata: dict, elapsed: float
) -> AnswerResult:
    """把图状态转成库层返回类型。"""
    sources = tuple(state.get("fused") or ())
    plan = state.get("plan")
    merged_metadata = dict(metadata)
    merged_metadata["elapsed_seconds"] = round(elapsed, 3)
    merged_metadata["need_search"] = bool(state.get("need_search", True))
    merged_metadata["queries"] = {
        "bm25": state.get("bm25_query", ""),
        "vector": state.get("vector_query", ""),
    }
    return AnswerResult(
        answer=state.get("answer") or "",
        sources=sources,
        plan=plan if plan is not None else None,
        metadata=merged_metadata,
    )
