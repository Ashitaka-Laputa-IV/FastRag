"""用 langgraph 编排 RAG 流程。

节点链路：decide_search(LLM) -> requery(LLM) -> bm25 / vector 并行召回
          -> fuse(RRF 或 rerank) -> answer(LLM)

- 流程图见 README.md
- 提示词单独放在 prompts/，便于迭代
- 参数见 config/rag.toml；LLM 与重排参数分别见 config/llm.toml、config/reranker.toml

用法:
    uv run python -m fastrag.rag_graph     # 交互式问答，回答流式输出

输出示例:
    $ uv run python -m fastrag.rag_graph
    问题（回车退出）: LangGraph 中如何实现 human-in-the-loop？
    检索查询: LangGraph human-in-the-loop 实现方法
    来源:
      - data/langchain-docs/oss/python/langgraph/interrupts.md#0-1234
    回答: 依据资料，...（逐 token 输出）
"""

import asyncio
import pathlib
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import TypedDict
import warnings

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from fastrag.embed_chunks import (
    ensure_vectors,
    load_chunks as load_embedding_chunks,
    load_config as load_embedding_config,
    load_env,
    search as search_vectors,
)
from fastrag.keyword_search import (
    load_config as load_keyword_config,
    search as search_keywords,
)
from fastrag.llm import build_llm, load_config as load_llm_config
from fastrag.reranker import (
    build_reranker,
    load_config as load_reranker_config,
    rerank_chunks,
)
from fastrag.rrf import rrf_fuse

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "rag.toml"
PROMPTS_DIR = ROOT / "prompts"
FUSION_METHODS = ("rrf", "rerank")


class RAGState(TypedDict, total=False):
    """图状态：问题、是否需要检索、改写后的查询、两路召回、融合结果、回答。"""

    question: str
    need_search: bool
    search_query: str
    bm25: list[dict]
    vector: list[dict]
    fused: list[dict]
    answer: str


class SearchDecision(BaseModel):
    """decide_search 节点的结构化输出。"""

    need_search: bool = Field(description="回答该问题是否需要检索本地文档库")


class SearchQuery(BaseModel):
    """requery 节点的结构化输出。"""

    query: str = Field(description="改写后、适合文档检索的查询")


@dataclass(frozen=True)
class RAGConfig:
    """RAG 配置，对应 config/rag.toml。"""

    candidates: int
    method: str
    k: int
    weights: tuple[float, ...]
    context_n: int


def load_config(path: pathlib.Path = CONFIG_PATH) -> RAGConfig:
    """读取 TOML 配置，并校验融合方式。"""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    retrieval, fusion, answer = raw["retrieval"], raw["fusion"], raw["answer"]
    if fusion["method"] not in FUSION_METHODS:
        raise SystemExit(f"fusion.method 必须是 {' / '.join(FUSION_METHODS)}")
    return RAGConfig(
        candidates=retrieval["candidates"],
        method=fusion["method"],
        k=fusion["k"],
        weights=tuple(fusion["weights"]),
        context_n=answer["context_n"],
    )


def load_prompt(name: str) -> str:
    """读取 prompts/ 下的提示词模板。"""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


class Pipeline:
    """把图需要的配置与检索资源准备好，节点函数通过闭包使用。"""

    def __init__(self, config: RAGConfig):
        load_env()
        self.config = config
        self.llm: BaseChatModel = build_llm(load_llm_config())
        self.decide_chain = self.llm.with_structured_output(SearchDecision)
        self.requery_chain = self.llm.with_structured_output(SearchQuery)
        self.prompts = {
            name: load_prompt(name) for name in ("decide_search", "requery", "answer")
        }
        self.keyword_config = load_keyword_config()
        self.embedding_config = load_embedding_config()
        self.chunks = load_embedding_chunks(self.embedding_config)
        self.vectors = ensure_vectors(self.embedding_config, self.chunks)
        self.reranker = (
            build_reranker(load_reranker_config())
            if config.method == "rerank"
            else None
        )
        self.graph = self._build_graph()

    def decide_search(self, state: RAGState) -> dict:
        """用 LLM 判断是否需要检索本地文档。"""
        decision = self.decide_chain.invoke(
            self.prompts["decide_search"].format(question=state["question"])
        )
        return {"need_search": decision.need_search}

    def requery(self, state: RAGState) -> dict:
        """用 LLM 把问题改写成更适合检索的查询。"""
        result = self.requery_chain.invoke(
            self.prompts["requery"].format(question=state["question"])
        )
        return {"search_query": result.query}

    def retrieve_bm25(self, state: RAGState) -> dict:
        """BM25 关键词召回。"""
        hits = search_keywords(
            self.keyword_config, state["search_query"], self.config.candidates
        )
        return {"bm25": [chunk for _, chunk in hits]}

    def retrieve_vector(self, state: RAGState) -> dict:
        """向量语义召回。"""
        hits = search_vectors(
            self.embedding_config,
            self.vectors,
            self.chunks,
            state["search_query"],
            self.config.candidates,
        )
        return {"vector": [chunk for _, chunk in hits]}

    def fuse(self, state: RAGState) -> dict:
        """按配置的融合方式（rrf / rerank）排序两路召回结果。"""
        bm25 = state.get("bm25", [])
        vector = state.get("vector", [])
        if self.config.method == "rerank":
            merged = self._merge(bm25, vector)
            documents = rerank_chunks(self.reranker, state["question"], merged)
            return {"fused": [self._to_chunk(document) for document in documents]}

        fused = rrf_fuse([bm25, vector], self.config.k, self.config.weights)
        return {"fused": [chunk for _, chunk in fused][: self.config.context_n]}

    def answer(self, state: RAGState) -> dict:
        """把检索片段拼进 prompt，让 LLM 依据资料作答。"""
        chunks = state.get("fused", [])[: self.config.context_n]
        context = (
            "\n\n".join(
                f"[{index}] {chunk['source']}\n{chunk['text']}"
                for index, chunk in enumerate(chunks, start=1)
            )
            or "（无检索结果）"
        )
        prompt = self.prompts["answer"].format(
            context=context, question=state["question"]
        )
        return {"answer": self.llm.invoke(prompt).text.strip()}

    @staticmethod
    def _merge(*rankings: list[dict]) -> list[dict]:
        """按 id 去重合并多路候选，保留先出现的顺序。"""
        seen: set[str] = set()
        merged = []
        for ranking in rankings:
            for chunk in ranking:
                if chunk["id"] not in seen:
                    seen.add(chunk["id"])
                    merged.append(chunk)
        return merged

    @staticmethod
    def _to_chunk(document: Document) -> dict:
        """把重排后的 Document 还原成 chunk 字典。"""
        metadata = document.metadata
        return {
            "id": metadata["id"],
            "source": metadata["source"],
            "start": metadata["start"],
            "end": metadata["end"],
            "text": document.page_content,
        }

    @staticmethod
    def route_after_decide(state: RAGState) -> str:
        """需要检索就改写查询，否则直接回答。"""
        return "requery" if state.get("need_search", True) else "answer"

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("decide_search", self.decide_search)
        graph.add_node("requery", self.requery)
        graph.add_node("retrieve_bm25", self.retrieve_bm25)
        graph.add_node("retrieve_vector", self.retrieve_vector)
        graph.add_node("fuse", self.fuse)
        graph.add_node("answer", self.answer)

        graph.add_edge(START, "decide_search")
        graph.add_conditional_edges(
            "decide_search",
            self.route_after_decide,
            {"requery": "requery", "answer": "answer"},
        )
        graph.add_edge("requery", "retrieve_bm25")
        graph.add_edge("requery", "retrieve_vector")
        graph.add_edge("retrieve_bm25", "fuse")
        graph.add_edge("retrieve_vector", "fuse")
        graph.add_edge("fuse", "answer")
        graph.add_edge("answer", END)
        return graph.compile()


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """进程内复用 Pipeline（向量、重排器等只初始化一次）。"""
    return Pipeline(load_config())


def run(question: str) -> RAGState:
    """跑一次完整的 RAG 流程。"""
    return get_pipeline().graph.invoke({"question": question})


async def stream_answer(question: str) -> None:
    """跑一次问答，用 astream_events 流式输出回答。"""
    graph = get_pipeline().graph
    answer_started = False
    async for event in graph.astream_events({"question": question}, version="v2"):
        kind = event["event"]
        node = (event.get("metadata") or {}).get("langgraph_node")
        output = event.get("data", {}).get("output")
        if kind == "on_chain_end" and event.get("name") == "requery":
            if isinstance(output, dict) and output.get("search_query"):
                print(f"检索查询: {output['search_query']}")
        elif kind == "on_chain_end" and event.get("name") == "fuse":
            fused = (output or {}).get("fused", [])
            if fused:
                print("来源:")
                for chunk in fused:
                    print(f"  - {chunk['source']}#{chunk['start']}-{chunk['end']}")
            print("回答: ", end="", flush=True)
            answer_started = True
        elif kind == "on_chat_model_stream" and node == "answer":
            if not answer_started:
                print("回答: ", end="", flush=True)
                answer_started = True
            print(event["data"]["chunk"].text, end="", flush=True)
    print("\n")


async def chat() -> None:
    """交互式问答：每行一个问题，空行退出。"""
    load_env()
    get_pipeline()  # 先完成初始化，避免第一次输入时卡顿
    while True:
        try:
            question = input("问题（回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        await stream_answer(question)


def main() -> None:
    # structured output 的事件序列化会触发无害的 pydantic 警告
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    asyncio.run(chat())


if __name__ == "__main__":
    main()
