"""用 bm25s 对切块后的 chunk 做 BM25 关键词检索。

- 索引与检索参数来自 config/fastrag.toml（[paths]、[bm25]、[retrieval]）
- 语料是 chunk_docs 产出的 data/chunks.jsonl（每行含 id/source/start/end/token_count/text）
- 索引连同 corpus 落盘到 data/bm25-index/，检索时无需重新切块

用法:
    uv run python -m fastrag.keyword_search build
    uv run python -m fastrag.keyword_search query "how to stream tokens"
    uv run python -m fastrag.keyword_search query "human in the loop" -k 3
"""

import argparse
import json

import bm25s

from fastrag.config import Config, load_config

SNIPPET_CHARS = 160


def load_chunks(config: Config) -> list[dict]:
    """读取 JSONL 形式的 chunk 语料。"""
    with config.paths.chunks_file.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_index(config: Config) -> None:
    """对全部 chunk 建立 BM25 索引，并把 corpus 一并落盘。"""
    chunks = load_chunks(config)
    tokenized = bm25s.tokenize(
        [chunk["text"] for chunk in chunks],
        stopwords=config.bm25.stopwords,
        show_progress=False,
    )
    retriever = bm25s.BM25(method=config.bm25.method)
    retriever.index(tokenized, show_progress=False)

    config.paths.bm25_index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(config.paths.bm25_index_dir, corpus=chunks, show_progress=False)
    print(f"{len(chunks)} 个 chunk 已建索引 -> {config.paths.bm25_index_dir}")


def search(config: Config, text: str, k: int) -> list[tuple[float, dict]]:
    """返回 top-k 命中 chunk 及其分数（降序）。"""
    index_dir = config.paths.bm25_index_dir
    if not index_dir.exists():
        raise SystemExit("索引不存在，请先运行: uv run python -m fastrag.keyword_search build")

    retriever = bm25s.BM25.load(index_dir, load_corpus=True, show_progress=False)
    query_tokens = bm25s.tokenize(
        text, stopwords=config.bm25.stopwords, show_progress=False
    )
    results, scores = retriever.retrieve(
        query_tokens, k=min(k, len(retriever.corpus)), show_progress=False
    )
    return [(float(score), chunk) for score, chunk in zip(scores[0], results[0])]


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

    hits = search(config, args.text, args.k or config.retrieval.top_n)
    for rank, (score, chunk) in enumerate(hits, start=1):
        print(f"{rank}. {score:.3f}  {chunk['source']}#{chunk['start']}-{chunk['end']}")
        print(f"   {format_snippet(chunk['text'])}")


if __name__ == "__main__":
    main()
