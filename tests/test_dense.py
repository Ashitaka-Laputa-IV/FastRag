"""向量索引：用 InMemoryVectorStore 落盘，并按 chunk id 增量复用。"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

from langchain_core.embeddings import Embeddings

from fastrag.domain import Chunk
from fastrag.index.dense import ensure_vector_store, search_dense


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [[1.0, 0.0] if text == "alpha" else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def make_config(tmp_path, model: str = "fake") -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(embeddings_dir=tmp_path / "embeddings"),
        embedding=SimpleNamespace(model=model, dimensions=2),
    )


def make_store(*texts: str) -> object:
    chunks = tuple(
        Chunk(
            id=text,
            source=f"{text}.md",
            start=0,
            end=len(text),
            text=text,
            token_count=1,
        )
        for text in texts
    )
    return type(
        "Store",
        (),
        {
            "chunks": chunks,
            "digest": "digest",
            "path": pathlib.Path("chunks.jsonl"),
            "get": lambda self, chunk_id: next(
                (chunk for chunk in chunks if chunk.id == chunk_id), None
            ),
            "__len__": lambda self: len(chunks),
        },
    )()


def test_first_build_embeds_all_and_searches(tmp_path):
    embeddings = FakeEmbeddings()
    store = make_store("alpha", "beta")

    vector_store = ensure_vector_store(make_config(tmp_path), store, embeddings)

    assert embeddings.embedded == ["alpha", "beta"]
    hits = search_dense(vector_store, store, "alpha?", k=2)
    assert [hit.chunk.id for hit in hits] == ["alpha", "beta"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score > hits[1].score
    assert (tmp_path / "embeddings" / "vector_store.json").exists()
    assert (tmp_path / "embeddings" / "manifest.json").exists()


def test_reload_reuses_all_vectors(tmp_path):
    config = make_config(tmp_path)
    ensure_vector_store(config, make_store("alpha", "beta"), FakeEmbeddings())

    embeddings = FakeEmbeddings()
    vector_store = ensure_vector_store(
        config, make_store("alpha", "beta"), embeddings
    )

    assert embeddings.embedded == []
    assert len(search_dense(vector_store, make_store("alpha", "beta"), "x", 5)) == 2


def test_incremental_embeds_only_new_chunks(tmp_path):
    config = make_config(tmp_path)
    ensure_vector_store(config, make_store("alpha", "beta"), FakeEmbeddings())

    embeddings = FakeEmbeddings()
    store = make_store("alpha", "beta", "gamma")
    vector_store = ensure_vector_store(config, store, embeddings)

    assert embeddings.embedded == ["gamma"]
    assert {doc.id for doc in vector_store.get_by_ids(["alpha", "beta", "gamma"])} == {
        "alpha",
        "beta",
        "gamma",
    }


def test_model_change_forces_full_rebuild(tmp_path):
    ensure_vector_store(make_config(tmp_path), make_store("alpha"), FakeEmbeddings())

    embeddings = FakeEmbeddings()
    ensure_vector_store(
        make_config(tmp_path, model="other"), make_store("alpha"), embeddings
    )

    assert embeddings.embedded == ["alpha"]


def test_search_dense_empty_store(tmp_path):
    store = type(
        "Store",
        (),
        {
            "chunks": (),
            "digest": "digest",
            "path": pathlib.Path("chunks.jsonl"),
            "get": lambda self, chunk_id: None,
            "__len__": lambda self: 0,
        },
    )()
    vector_store = ensure_vector_store(make_config(tmp_path), store, FakeEmbeddings())
    assert search_dense(vector_store, store, "x", k=3) == []
