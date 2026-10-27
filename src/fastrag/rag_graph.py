"""用 langgraph 编排 RAG 流程。

节点链路：plan_search(LLM：判断是否检索 + 生成两路查询) -> bm25 / vector 并行召回
          -> fuse(RRF 或 rerank) -> answer(LLM)

- 流程图见 README.md
- 提示词默认随包分发（fastrag.resources/prompts），可用 $FASTRAG_HOME/prompts 覆盖
- 参数统一见配置文件；判断检索与生成回答分别在
  [llm.route]、[llm.answer] 配置，可使用不同模型与厂家
- 两路查询都用英文：BM25 用术语叠加，向量检索用自然语言问句，plan_search 一次输出两条
- 融合选 rerank 时，cross-encoder 与向量召回共用 vector_query（改写后的英文问句）
- 不需要检索时 answer 节点改用 chat 提示词直接回答，不套「依据资料」的约束

用法:
    uv run python -m fastrag.rag_graph     # 交互式问答，回答通过 stream_events 流式输出

输出示例:
    $ uv run python -m fastrag.rag_graph
    问题（回车退出）: LangGraph 中如何实现 human-in-the-loop？
    BM25 查询: human-in-the-loop interrupt approval
    向量查询: How to implement human-in-the-loop human approval in LangGraph?
    来源:
      - data/langchain-docs/oss/python/langgraph/interrupts.md#0-1234
    回答: 依据资料，...（逐 token 输出）
"""

import warnings
from functools import lru_cache
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from fastrag.chunks import load_chunks
from fastrag.config import Config, load_config, load_env, prompt_path
from fastrag.embed_chunks import (
    build_embeddings,
    ensure_vectors,
    search as search_vectors,
)
from fastrag.errors import run_cli
from fastrag.keyword_search import load_index, search as search_keywords
from fastrag.llm import build_llm
from fastrag.reranker import build_reranker, rerank_chunks
from fastrag.rrf import rrf_fuse



class RAGState(TypedDict, total=False):
    """图状态：问题、是否需要检索、两路查询、两路召回、融合结果、回答。"""

    question: str
    need_search: bool
    bm25_query: str
    vector_query: str
    bm25: list[dict]
    vector: list[dict]
    fused: list[dict]
    answer: str


class SearchPlan(BaseModel):
    """plan_search 节点的结构化输出：是否检索 + 两路各自的查询。"""

    need_search: bool = Field(description="回答该问题是否需要检索本地文档库")
    bm25_query: str = Field(
        default="",
        description="给 BM25 的英文关键词查询，空格分隔的术语；不需要检索时为空字符串",
    )
    vector_query: str = Field(
        default="",
        description="给向量检索的英文自然语言问句；不需要检索时为空字符串",
    )


def load_prompt(name: str) -> str:
    """读取提示词模板：优先工作目录下的 prompts/，否则用包内默认模板。"""
    return prompt_path(name).read_text(encoding="utf-8")


class Pipeline:
    """把图需要的配置与检索资源准备好，节点函数通过闭包使用。

    语料、向量、BM25 索引与嵌入客户端都只初始化一次：
    召回节点并发执行时不会重复读索引或新建 HTTP 客户端。
    """

    def __init__(self, config: Config):
        load_env()
        self.config = config
        self.route_llm: BaseChatModel = build_llm(config.llm.route)
        self.answer_llm: BaseChatModel = build_llm(config.llm.answer)
        self.plan_chain = self.route_llm.with_structured_output(SearchPlan)
        self.prompts = {
            name: load_prompt(name) for name in ("search_plan", "answer", "chat")
        }
        self.chunks = load_chunks(config)
        self.embeddings = build_embeddings(config)
        self.vectors = ensure_vectors(config, self.chunks, self.embeddings)
        self.retriever = load_index(config)
        self.reranker = (
            build_reranker(config) if config.fusion.method == "rerank" else None
        )
        self.graph = self._build_graph()

    def plan_search(self, state: RAGState) -> dict:
        """一次调用同时判断是否检索，并给出两路各自的查询。

        不需要检索时不生成查询，直接返回空字符串；需要检索时若模型漏填某一路，
        就退回原始问题，保证召回节点总能拿到可用的查询。
        """
        plan = self.plan_chain.invoke(
            self.prompts["search_plan"].format(question=state["question"])
        )
        if not plan.need_search:
            return {"need_search": False, "bm25_query": "", "vector_query": ""}

        question = state["question"]
        return {
            "need_search": True,
            "bm25_query": plan.bm25_query.strip() or question,
            "vector_query": plan.vector_query.strip() or question,
        }

    def retrieve_bm25(self, state: RAGState) -> dict:
        """BM25 关键词召回，使用术语叠加的 bm25_query 与复用的索引。"""
        hits = search_keywords(
            self.retriever,
            self.config,
            state["bm25_query"],
            self.config.retrieval.bm25_candidates,
        )
        return {"bm25": [chunk for _, chunk in hits]}

    def retrieve_vector(self, state: RAGState) -> dict:
        """向量语义召回，使用自然语言问句 vector_query 与复用的嵌入客户端。"""
        hits = search_vectors(
            self.embeddings,
            self.vectors,
            self.chunks,
            state["vector_query"],
            self.config.retrieval.vector_candidates,
        )
        return {"vector": [chunk for _, chunk in hits]}

    def fuse(self, state: RAGState) -> dict:
        """按配置的融合方式（rrf / rerank）排序两路召回结果。

        rerank 分支用改写后的英文 vector_query 做 cross-encoder 输入，
        与向量召回使用同一个查询，保证打分口径与召回一致；
        RRF 只看名次，两种方式都按 retrieval.top_n 截断。
        """
        bm25 = state.get("bm25", [])
        vector = state.get("vector", [])
        if not bm25 and not vector:
            return {"fused": []}

        top_n = self.config.retrieval.top_n
        if self.config.fusion.method == "rerank":
            merged = self._merge(bm25, vector)
            query = state.get("vector_query") or state["question"]
            documents = rerank_chunks(self.reranker, query, merged)
            return {
                "fused": [self._to_chunk(document) for document in documents[:top_n]]
            }

        fused = rrf_fuse(
            [bm25, vector], self.config.fusion.rrf_k, self.config.fusion.weights
        )
        return {"fused": [chunk for _, chunk in fused][:top_n]}

    def answer(self, state: RAGState) -> dict:
        """按是否需要检索选择提示词，再让 LLM 作答。

        需要检索时把片段拼进 answer 模板，要求只依据资料；
        不需要检索时没有资料可依据，改用 chat 模板直接回答，
        否则模型会把「你好」这类问题也答成「资料不足」。
        """
        if state.get("need_search", True):
            chunks = state.get("fused", [])[: self.config.retrieval.top_n]
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
        else:
            prompt = self.prompts["chat"].format(question=state["question"])
        return {"answer": self.answer_llm.invoke(prompt).text.strip()}

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
    def route_after_plan(state: RAGState) -> list[str]:
        """需要检索就同时走 BM25 与向量两路，否则直接回答。"""
        if not state.get("need_search", True):
            return ["answer"]
        return ["retrieve_bm25", "retrieve_vector"]

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("plan_search", self.plan_search)
        graph.add_node("retrieve_bm25", self.retrieve_bm25)
        graph.add_node("retrieve_vector", self.retrieve_vector)
        graph.add_node("fuse", self.fuse)
        graph.add_node("answer", self.answer)

        graph.add_edge(START, "plan_search")
        graph.add_conditional_edges(
            "plan_search",
            self.route_after_plan,
            ["retrieve_bm25", "retrieve_vector", "answer"],
        )
        graph.add_edge("retrieve_bm25", "fuse")
        graph.add_edge("retrieve_vector", "fuse")
        graph.add_edge("fuse", "answer")
        graph.add_edge("answer", END)
        return graph.compile()


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    """进程内复用 Pipeline（向量、重排器等只初始化一次）。"""
    return Pipeline(load_config())


def main() -> None:
    """交互式问答：每行一个问题，空行退出；回答通过 stream_events 流式输出。"""
    load_env()
    # structured output 的事件序列化会触发无害的 pydantic 警告
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    warnings.filterwarnings(
        "ignore", message="The v3 streaming protocol on Pregel is experimental."
    )
    graph = get_pipeline().graph

    while True:
        try:
            question = input("问题（回车退出）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        run = graph.stream_events({"question": question}, version="v3")
        print("回答: ", end="", flush=True)
        for stream in run.messages:
            if stream.node != "answer":
                continue
            for delta in stream.text:
                print(delta, end="", flush=True)
        print()

        state = run.output or {}
        if state.get("need_search"):
            print(f"BM25 查询: {state.get('bm25_query', '')}")
            print(f"向量查询: {state.get('vector_query', '')}")
        else:
            print("未检索本地文档")
        for chunk in state.get("fused", []):
            print(f"来源: {chunk['source']}#{chunk['start']}-{chunk['end']}")


if __name__ == "__main__":
    run_cli(main)
