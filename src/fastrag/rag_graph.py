"""用 langgraph 编排 RAG 流程。

流程（与设计图一致）:

    START -> decide_search --(需要检索)--> requery -> bm25 / vector
                         \\--(不需要)----------------------------\\
                                                     bm25、vector -> fuse(RRF/RERANK)
                                                                  -> answer(LLM) -> END

- decide_search：用 LLM 判断问题是否需要检索本地文档
- requery：用 LLM 把问题改写成更适合检索的查询
- bm25 / vector：并行调用 keyword_search 与 embed_chunks（余弦相似度）
- fuse：RRF 融合，可选再用 reranker 交叉编码器重排
- answer：把 top-n 片段拼进 prompt，让 LLM 依据资料作答

参数来自 config/rag.toml（LLM、重排分别见 config/llm.toml、config/reranker.toml）。

用法:
    uv run python -m fastrag.rag_graph

输出示例:
    $ uv run python -m fastrag.rag_graph
    问题: LangGraph 中如何实现 human-in-the-loop？
    检索查询: LangGraph human-in-the-loop interrupt 实现
    来源:
      - data/langchain-docs/oss/python/langgraph/interrupts.md#0-1234
    回答: ...
"""

import json
import pathlib
import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

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
DEMO_QUESTION = "LangGraph 中如何实现 human-in-the-loop？"


class RAGState(TypedDict, total=False):
    """图状态：问题、改写后的查询、两路召回、融合结果、回答。"""

    question: str
    need_search: bool
    search_query: str
    bm25: list[dict]
    vector: list[dict]
    fused: list[dict]
    answer: str


@dataclass(frozen=True)
class RAGConfig:
    """RAG 配置，对应 config/rag.toml。"""

    candidates: int
    k: int
    weights: tuple[float, ...]
    context_n: int
    use_rerank: bool


def load_config(path: pathlib.Path = CONFIG_PATH) -> RAGConfig:
    """读取 TOML 配置。"""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    retrieval, fusion, answer = raw["retrieval"], raw["fusion"], raw["answer"]
    return RAGConfig(
        candidates=retrieval["candidates"],
        k=fusion["k"],
        weights=tuple(fusion["weights"]),
        context_n=answer["context_n"],
        use_rerank=answer["use_rerank"],
    )


def parse_json(text: str) -> dict:
    """从模型输出里抠出第一个 JSON 对象，解析失败时返回空字典。"""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class Pipeline:
    """把图需要的配置与检索资源准备好，节点函数通过闭包使用。"""

    def __init__(self, config: RAGConfig):
        load_env()
        self.config = config
        self.llm: BaseChatModel = build_llm(load_llm_config())
        self.keyword_config = load_keyword_config()
        self.embedding_config = load_embedding_config()
        self.chunks = load_embedding_chunks(self.embedding_config)
        self.vectors = ensure_vectors(self.embedding_config, self.chunks)
        self.reranker = (
            build_reranker(load_reranker_config()) if config.use_rerank else None
        )
        self.graph = self._build_graph()

    def decide_search(self, state: RAGState) -> dict:
        """用 LLM 判断是否需要检索本地文档。"""
        prompt = (
            "判断回答下面的问题是否需要检索本地技术文档库。"
            '只输出 JSON，格式为 {"need_search": true/false}，不要输出其他内容。\n'
            f"问题: {state['question']}"
        )
        decision = parse_json(self.llm.invoke(prompt).text)
        return {"need_search": bool(decision.get("need_search", True))}

    def requery(self, state: RAGState) -> dict:
        """用 LLM 把问题改写成更适合检索的查询。"""
        prompt = (
            "把下面的问题改写成一个适合文档检索的查询，保留关键实体与术语，"
            "只输出一行查询，不要解释。\n"
            f"问题: {state['question']}"
        )
        return {"search_query": self.llm.invoke(prompt).text.strip()}

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
        """RRF 融合两路结果，可选重排，截取送入 LLM 的片段。"""
        fused = rrf_fuse(
            [state.get("bm25", []), state.get("vector", [])],
            self.config.k,
            self.config.weights,
        )
        chunks = [chunk for _, chunk in fused]
        if self.reranker is not None:
            documents = rerank_chunks(self.reranker, state["question"], chunks)
            return {"fused": [self._to_chunk(document) for document in documents]}
        return {"fused": chunks[: self.config.context_n]}

    def answer(self, state: RAGState) -> dict:
        """把检索片段拼进 prompt，让 LLM 依据资料作答。"""
        chunks = state.get("fused", [])[: self.config.context_n]
        if chunks:
            context = "\n\n".join(
                f"[{index}] {chunk['source']}#{chunk['start']}-{chunk['end']}\n{chunk['text']}"
                for index, chunk in enumerate(chunks, start=1)
            )
        else:
            context = "（无检索结果）"
        prompt = (
            "请依据下面的资料回答问题，资料不足时直接说明，不要编造。\n\n"
            f"资料:\n{context}\n\n"
            f"问题: {state['question']}\n"
            "回答:"
        )
        return {"answer": self.llm.invoke(prompt).text.strip()}

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
    pipeline = get_pipeline()
    return pipeline.graph.invoke({"question": question})


def main() -> None:
    load_env()
    result = run(DEMO_QUESTION)
    print(f"问题: {DEMO_QUESTION}")
    print(f"检索查询: {result.get('search_query') or '（未检索）'}")
    print("来源:")
    for chunk in result.get("fused", []):
        print(f"  - {chunk['source']}#{chunk['start']}-{chunk['end']}")
    print(f"回答: {result.get('answer', '')}")


if __name__ == "__main__":
    main()
