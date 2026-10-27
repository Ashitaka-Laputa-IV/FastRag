"""一条命令刷新全部数据产物：切块 -> 向量 -> BM25 索引。

手工依次跑三个阶段容易漏步，而三份产物靠 chunks.jsonl 的内容指纹互相校验，
漏步会在检索时报错或（更糟）静默混入两套语料。这里按依赖顺序串起来：

1. `fastrag.chunk_docs`：生成 data/chunks.jsonl
2. `fastrag.embed_chunks`：生成 data/embeddings/vectors.npy + manifest.json
3. `fastrag.keyword_search`：生成 data/bm25-index/（含 meta.json）

用法:
    uv run python -m fastrag.prepare

输出示例:
    $ uv run python -m fastrag.prepare
    [1/3] 切块
    180 篇文档 -> 2608 个 chunk，...
    [2/3] 嵌入向量
    chunks.jsonl 内容已变更（或清单缺少指纹），重新嵌入 ...
    ...
    [3/3] BM25 索引
    2608 个 chunk 已建索引 -> .../data/bm25-index
    全部产物已刷新，耗时 623.4s
"""

import time

from fastrag import chunk_docs
from fastrag.chunks import load_chunks
from fastrag.config import load_config, load_env
from fastrag.embed_chunks import ensure_vectors
from fastrag.errors import run_cli
from fastrag.keyword_search import build_index


def main() -> None:
    load_env()
    config = load_config()
    started = time.perf_counter()

    print("[1/3] 切块")
    chunk_docs.main()

    print("[2/3] 嵌入向量")
    chunks = load_chunks(config)
    ensure_vectors(config, chunks)

    print("[3/3] BM25 索引")
    build_index(config)

    print(f"全部产物已刷新，耗时 {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    run_cli(main)
