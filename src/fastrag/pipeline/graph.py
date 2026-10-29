"""图编排：把节点装配成 LangGraph StateGraph。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from fastrag.pipeline.nodes import RagNodes, RagState


def build_graph(nodes: RagNodes):
    """构造并编译问答图。"""
    graph = StateGraph(RagState)
    graph.add_node("plan_search", nodes.plan_search)
    graph.add_node("retrieve_bm25", nodes.retrieve_bm25)
    graph.add_node("retrieve_vector", nodes.retrieve_vector)
    graph.add_node("fuse", nodes.fuse)
    graph.add_node("answer", nodes.answer)

    graph.add_edge(START, "plan_search")
    graph.add_conditional_edges(
        "plan_search",
        nodes.route_after_plan,
        ["retrieve_bm25", "retrieve_vector", "answer"],
    )
    graph.add_edge("retrieve_bm25", "fuse")
    graph.add_edge("retrieve_vector", "fuse")
    graph.add_edge("fuse", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
