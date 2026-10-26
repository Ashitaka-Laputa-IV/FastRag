"""用 langgraph 编排 RAG 流程。

节点链路：plan_search(LLM：判断是否检索 + 生成两路查询) -> bm25 / vector 并行召回
          -> fuse(RRF 或 rerank) -> answer(LLM)

- 流程图见 README.md
- 提示词单独放在 prompts/，便于迭代
- 参数统一见 config/fastrag.toml；判断检索与生成回答分别在
  [llm.route]、[llm.answer] 配置，可使用不同模型与厂家
- 两路查询都用英文：BM25 用术语叠加，向量检索用自然语言问句，plan_search 一次输出两条
- 每个节点用 timed 装饰器把耗时写进 state["timings"]，供 bench.py 统计
- 基准默认以「首 token 延迟」计量生成：读到第一个 chunk 就断开，不等待整段回答

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
    耗时(ms): 检索规划 4213.5  召回 164.1  融合 0.1  生成 20932.7
"""

import time
import warnings
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from fastrag.config import ROOT, Config, load_config, load_env
from fastrag.embed_chunks import ensure_vectors, load_chunks, search as search_vectors
from fastrag.keyword_search import search as search_keywords
from fastrag.llm import build_llm
from fastrag.reranker import build_reranker, rerank_chunks
from fastrag.rrf import rrf_fuse

PROMPTS_DIR = ROOT / "prompts"
STAGE_LABELS = {
    "plan": "检索规划",
    "retrieve": "召回",
    "retrieve_bm25": "BM25 召回",
    "retrieve_vector": "向量召回",
    "rerank": "重排",
    "fusion": "融合",
    "generate": "生成",
}


def merge_timings(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    """合并各节点上报的耗时；并行节点写不同键，不能简单覆盖。"""
    return {**left, **right}


class RAGState(TypedDict, total=False):
    """图状态：问题、是否需要检索、两路查询、两路召回、融合结果、回答与耗时。"""

    question: str
    need_search: bool
    bm25_query: str
    vector_query: str
    bm25: list[dict]
    vector: list[dict]
    fused: list[dict]
    answer: str
    timings: Annotated[dict[str, float], merge_timings]


def timed(stage: str | Callable[["Pipeline"], str]):
    """记录节点耗时（毫秒），写进 state["timings"]。

    stage 可以传固定名字，也可以传 `lambda pipeline: name`，按配置决定阶段名
    （融合节点在 method=rerank 时记作「重排」，否则记作「融合」）。
    """

    def decorate(func: Callable[["Pipeline", RAGState], dict]):
        @wraps(func)
        def wrapper(self: "Pipeline", state: RAGState) -> dict:
            started = time.perf_counter()
            update = func(self, state)
            name = stage(self) if callable(stage) else stage
            update["timings"] = {
                name: round((time.perf_counter() - started) * 1000, 1)
            }
            return update

        return wrapper

    return decorate


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
    """读取 prompts/ 下的提示词模板。"""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


class Pipeline:
    """把图需要的配置与检索资源准备好，节点函数通过闭包使用。

    first_token_only=True 时，「生成」阶段读到首个 token 就断开流，
    节点耗时即首 token 延迟（TTFT），bench 默认使用该口径；
    answer_thinking=False 会关掉回答模型的思考过程，让首 token 就是正文。
    """

    def __init__(
        self,
        config: Config,
        first_token_only: bool = False,
        answer_thinking: bool = True,
    ):
        load_env()
        self.config = config
        self.first_token_only = first_token_only
        self.answer_thinking = answer_thinking
        self.route_llm: BaseChatModel = build_llm(config.llm.route)
        self.answer_llm: BaseChatModel = build_llm(
            config.llm.answer, thinking=answer_thinking
        )
        self.plan_chain = self.route_llm.with_structured_output(SearchPlan)
        self.prompts = {
            name: load_prompt(name) for name in ("search_plan", "answer")
        }
        self.chunks = load_chunks(config)
        self.vectors = ensure_vectors(config, self.chunks)
        self.reranker = (
            build_reranker(config) if config.fusion.method == "rerank" else None
        )
        self.graph = self._build_graph()

    @timed("plan")
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

    @timed("retrieve_bm25")
    def retrieve_bm25(self, state: RAGState) -> dict:
        """BM25 关键词召回，使用术语叠加的 bm25_query。"""
        hits = search_keywords(
            self.config,
            state["bm25_query"],
            self.config.retrieval.bm25_candidates,
        )
        return {"bm25": [chunk for _, chunk in hits]}

    @timed("retrieve_vector")
    def retrieve_vector(self, state: RAGState) -> dict:
        """向量语义召回，使用自然语言问句 vector_query。"""
        hits = search_vectors(
            self.config,
            self.vectors,
            self.chunks,
            state["vector_query"],
            self.config.retrieval.vector_candidates,
        )
        return {"vector": [chunk for _, chunk in hits]}

    @timed(lambda pipeline: "rerank" if pipeline.config.fusion.method == "rerank" else "fusion")
    def fuse(self, state: RAGState) -> dict:
        """按配置的融合方式（rrf / rerank）排序两路召回结果。

        rerank 分支用原始问题做 cross-encoder 输入（保留完整语义），
        与 RRF 只看名次的口径不同，两种方式都按 retrieval.top_n 截断。
        """
        bm25 = state.get("bm25", [])
        vector = state.get("vector", [])
        if not bm25 and not vector:
            return {"fused": []}

        top_n = self.config.retrieval.top_n
        if self.config.fusion.method == "rerank":
            merged = self._merge(bm25, vector)
            documents = rerank_chunks(self.reranker, state["question"], merged)
            return {
                "fused": [self._to_chunk(document) for document in documents[:top_n]]
            }

        fused = rrf_fuse(
            [bm25, vector], self.config.fusion.rrf_k, self.config.fusion.weights
        )
        return {"fused": [chunk for _, chunk in fused][:top_n]}

    @timed("generate")
    def answer(self, state: RAGState) -> dict:
        """把检索片段拼进 prompt，让 LLM 依据资料作答。

        first_token_only=True 时只读到第一个有内容的 chunk 就断开连接，
        节点耗时因此等于首 token 延迟（TTFT），不包含剩余生成时间。
        """
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
        if self.first_token_only:
            return {"answer": self._first_token(prompt)}
        return {"answer": self.answer_llm.invoke(prompt).text.strip()}

    def _first_token(self, prompt: str) -> str:
        """读取首个有内容的 chunk 后立即断开流，返回这段文本。

        断流可以取消后续请求，避免继续为完整回答付出时间和 token；
        首个 chunk 可能只有角色信息、文本为空，需要跳过。
        """
        stream = self.answer_llm.stream(prompt)
        try:
            for chunk in stream:
                if chunk.text:
                    return chunk.text
            return ""
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

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


@lru_cache(maxsize=4)
def get_pipeline(
    first_token_only: bool = False, answer_thinking: bool = True
) -> Pipeline:
    """进程内复用 Pipeline（向量、重排器等只初始化一次），按生成口径区分缓存。"""
    return Pipeline(
        load_config(),
        first_token_only=first_token_only,
        answer_thinking=answer_thinking,
    )


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
        timings = state.get("timings", {})
        if timings:
            print(
                "耗时(ms): "
                + "  ".join(
                    f"{STAGE_LABELS.get(name, name)} {ms:.1f}"
                    for name, ms in timings.items()
                )
            )


if __name__ == "__main__":
    main()
