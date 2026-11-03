"""混合检索编排：两路查询与通道名只在一处拼装。"""

from __future__ import annotations

from fastrag.config import load_config
from fastrag.domain import Chunk, ScoredChunk
from fastrag.retrieve import CHANNELS, hybrid_search
from fastrag.retrieve import hybrid as hybrid_module


def _hit(chunk_id: str, channel: str) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id, source="a.md", start=0, end=1, text=chunk_id, token_count=1
    )
    return ScoredChunk(chunk=chunk, score=1.0, channel=channel, rank=1)


def test_hybrid_search_wires_both_channels(monkeypatch):
    calls: dict[str, object] = {}

    def fake_bm25(retriever, store, config, text, k):
        calls["bm25"] = (text, k)
        return [_hit("a", "bm25")]

    def fake_dense(store_vector, store, text, k):
        calls["vector"] = (text, k)
        return [_hit("b", "vector")]

    monkeypatch.setattr(hybrid_module, "search_bm25", fake_bm25)
    monkeypatch.setattr(hybrid_module, "search_dense", fake_dense)

    config = load_config()
    rankings = hybrid_search(
        config, object(), object(), object(), "关键词", "semantic question"
    )

    assert tuple(rankings) == CHANNELS
    assert calls["bm25"] == ("关键词", config.retrieval.bm25_candidates)
    assert calls["vector"] == ("semantic question", config.retrieval.vector_candidates)
    assert rankings["bm25"][0].chunk.id == "a"
    assert rankings["vector"][0].chunk.id == "b"


def test_hybrid_search_falls_back_to_bm25_query(monkeypatch):
    calls: dict[str, str] = {}

    monkeypatch.setattr(
        hybrid_module, "search_bm25", lambda *args: calls.setdefault("b", args[3]) or []
    )
    monkeypatch.setattr(
        hybrid_module,
        "search_dense",
        lambda *args: calls.setdefault("v", args[2]) or [],
    )

    hybrid_search(load_config(), object(), object(), object(), "同一个查询")
    assert calls["v"] == "同一个查询"
