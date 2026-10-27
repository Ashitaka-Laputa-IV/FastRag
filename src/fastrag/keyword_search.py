"""用 bm25s 对切块后的 chunk 做 BM25 关键词检索。

- 索引与检索参数来自 config/fastrag.toml（[paths]、[bm25]、[retrieval]）
- 语料是 chunk_docs 产出的 data/chunks.jsonl（每行含 id/source/start/end/token_count/text）
- 索引连同 corpus 落盘到 data/bm25-index/，检索时无需重新切块
- 索引同时写入 meta.json（语料 sha256 + method/stopwords + 切块参数）；
  检索前校验，语料或参数变化时要求重建，避免配置静默失效
- `load_index` 负责加载并校验，返回的 retriever 可复用（Pipeline 只加载一次）

用法:
    uv run python -m fastrag.keyword_search build
    uv run python -m fastrag.keyword_search query "how to stream tokens"
    uv run python -m fastrag.keyword_search query "human in the loop" -k 3
"""

import argparse
import datetime
import json

import bm25s

from fastrag.chunks import chunking_params, chunks_sha256, load_chunks
from fastrag.config import Config, load_config
from fastrag.errors import FastragError, run_cli

SNIPPET_CHARS = 160
REBUILD_HINT = "uv run python -m fastrag.keyword_search build"


def index_params(config: Config) -> dict:
    """记录影响 BM25 索引的构建参数，写入 meta.json。"""
    return {"method": config.bm25.method, "stopwords": config.bm25.stopwords}


def build_index(config: Config) -> None:
    """对全部 chunk 建立 BM25 索引，并把 corpus 与元信息一并落盘。"""
    chunks = load_chunks(config)
    tokenized = bm25s.tokenize(
        [chunk["text"] for chunk in chunks],
        stopwords=config.bm25.stopwords,
        show_progress=False,
    )
    retriever = bm25s.BM25(method=config.bm25.method)
    retriever.index(tokenized, show_progress=False)

    index_dir = config.paths.bm25_index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(index_dir, corpus=chunks, show_progress=False)
    meta = {
        "chunks_sha256": chunks_sha256(config.paths.chunks_file),
        "count": len(chunks),
        "bm25": index_params(config),
        "chunking": chunking_params(config),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    (index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(chunks)} 个 chunk 已建索引 -> {index_dir}")


def check_index(config: Config) -> None:
    """校验索引与当前语料、BM25 参数一致；不一致时要求重建而不是静默使用。"""
    meta_path = config.paths.bm25_index_dir / "meta.json"
    if not meta_path.exists():
        raise FastragError(
            f"BM25 索引缺少 meta.json（旧版索引无法校验），请重新运行: {REBUILD_HINT}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("chunks_sha256") != chunks_sha256(config.paths.chunks_file):
        raise FastragError(
            f"chunks.jsonl 已变更，BM25 索引与语料不一致，请重新运行: {REBUILD_HINT}"
        )
    params = index_params(config)
    if meta.get("bm25") != params:
        raise FastragError(
            f"bm25 参数已变更（索引: {meta.get('bm25')}，当前: {params}），"
            f"请重新运行: {REBUILD_HINT}"
        )


def load_index(config: Config) -> bm25s.BM25:
    """加载 BM25 索引（含 corpus），加载前校验语料与构建参数。

    返回的 retriever 可在进程内复用（Pipeline 只加载一次），避免每轮提问
    都重读索引与整份 corpus；校验也只在加载时做一次。
    """
    index_dir = config.paths.bm25_index_dir
    if not index_dir.exists():
        raise FastragError(
            f"索引不存在，请先运行: {REBUILD_HINT}"
        )
    check_index(config)
    return bm25s.BM25.load(index_dir, load_corpus=True, show_progress=False)


def search(
    retriever: bm25s.BM25,
    config: Config,
    text: str,
    k: int,
    min_score: float | None = None,
) -> list[tuple[float, dict]]:
    """返回 top-k 命中 chunk 及其分数（降序）。

    retriever 由 `load_index` 提供，便于调用方复用；min_score 为 None 时取
    config.bm25.min_score，只保留 score > min_score 的命中，默认过滤掉零分文档，
    避免无关键词命中的 chunk 进入融合。
    """
    query_tokens = bm25s.tokenize(
        text, stopwords=config.bm25.stopwords, show_progress=False
    )
    results, scores = retriever.retrieve(
        query_tokens, k=min(k, len(retriever.corpus)), show_progress=False
    )
    threshold = config.bm25.min_score if min_score is None else min_score
    hits = [(float(score), chunk) for score, chunk in zip(scores[0], results[0])]
    return [hit for hit in hits if hit[0] > threshold]


def format_snippet(text: str) -> str:
    """压平空白并截断，便于终端查看。"""
    flat = " ".join(text.split())
    return flat[:SNIPPET_CHARS] + ("..." if len(flat) > SNIPPET_CHARS else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="bm25s 关键词检索（基于 chunk 语料）")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="构建或重建索引")
    query = sub.add_parser("query", help="关键词检索")
    query.add_argument("text", help="查询语句")
    query.add_argument("-k", type=int, default=None, help="返回条数，默认取配置值")
    args = parser.parse_args()

    config = load_config()
    if args.command == "build":
        build_index(config)
        return

    hits = search(
        load_index(config), config, args.text, args.k or config.retrieval.top_n
    )
    for rank, (score, chunk) in enumerate(hits, start=1):
        print(f"{rank}. {score:.3f}  {chunk['source']}#{chunk['start']}-{chunk['end']}")
        print(f"   {format_snippet(chunk['text'])}")


if __name__ == "__main__":
    run_cli(main)
