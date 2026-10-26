"""为切块后的 chunk 生成向量，并用固定问题演示一次向量检索。

模型与接口由 config/fastrag.toml 的 [embedding]、[endpoints]、[paths] 指定，
代码与具体模型无关。

- API key 只从 .env / 环境变量读取
- 嵌入走 langchain 的 OpenAIEmbeddings（OpenAI 兼容接口）
- 输入是 chunk_docs 产出的 data/chunks.jsonl，输出 data/embeddings/ 下的
  vectors.npy（按 chunks.jsonl 行序排列的 float32 矩阵）与 manifest.json
- 已嵌入且模型/语料未变时直接复用 vectors.npy，不重复请求接口
- 运行结束用固定问题检索一次（余弦相似度），默认返回 retrieval.top_n 条

用法:
    uv run python -m fastrag.embed_chunks

输出示例:
    $ uv run python -m fastrag.embed_chunks
    复用已嵌入向量：2625 条 -> .../data/embeddings/vectors.npy
    查询: human in the loop approval
    1. 0.684  data/langchain-docs/oss/python/deepagents/human-in-the-loop.md#0-2111
       > ## Documentation Index > Fetch the complete documentation index at: https://docs...
    2. 0.678  data/langchain-docs/oss/python/concepts/human-in-the-loop.md#0-2152
       > ## Documentation Index > Fetch the complete documentation index at: https://docs...
    ...
"""

import datetime
import json
import os
import time

import numpy as np
from langchain_openai import OpenAIEmbeddings

from fastrag.config import ROOT, Config, load_config, load_env

SNIPPET_CHARS = 160
DEMO_QUERY = "human in the loop approval"


def load_chunks(config: Config) -> list[dict]:
    """读取 JSONL 形式的 chunk 语料。"""
    with config.paths.chunks_file.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_embeddings(config: Config) -> OpenAIEmbeddings:
    """按配置建 OpenAI 兼容的 embeddings 客户端，缺 key 时给出明确提示。"""
    api_key = os.environ.get(config.embedding.endpoint.api_key_env)
    if not api_key:
        if config.embedding.endpoint.api_key_env:
            raise SystemExit(
                f"缺少 {config.embedding.endpoint.api_key_env}，"
                "请先 cp .env.example .env 并填写"
            )
        # 本地服务通常不校验 key，SDK 又要求非空，给一个占位值
        api_key = "local"
    return OpenAIEmbeddings(
        model=config.embedding.model,
        base_url=config.embedding.endpoint.base_url,
        api_key=api_key,
        chunk_size=config.embedding.batch_size,
        max_retries=config.embedding.max_retries,
        request_timeout=config.embedding.timeout,
        # 非 OpenAI 模型不做 tiktoken 预切分，交给服务端处理
        check_embedding_ctx_length=False,
    )


def embed_chunks(
    embeddings: OpenAIEmbeddings,
    config: Config,
    chunks: list[dict],
) -> np.ndarray:
    """分批嵌入全部 chunk，返回 (N, dimensions) 的 float32 矩阵。"""
    dimensions = config.embedding.dimensions
    batch_size = config.embedding.batch_size
    vectors = np.zeros((len(chunks), dimensions), dtype=np.float32)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = embeddings.embed_documents([chunk["text"] for chunk in batch])
        if len(batch_vectors[0]) != dimensions:
            raise SystemExit(
                f"返回维度 {len(batch_vectors[0])} 与配置 dimensions={dimensions} 不一致"
            )
        vectors[start : start + len(batch)] = batch_vectors
        print(
            f"\r已嵌入 {min(start + batch_size, len(chunks))}/{len(chunks)}",
            end="",
            flush=True,
        )
    print()
    return vectors


def save_vectors(config: Config, vectors: np.ndarray) -> None:
    """落盘向量矩阵与元信息（对齐 chunks.jsonl 行序）。"""
    out_dir = config.paths.embeddings_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vectors.npy", vectors)
    manifest = {
        "model": config.embedding.model,
        "dimensions": config.embedding.dimensions,
        "count": int(vectors.shape[0]),
        "chunks_file": config.paths.chunks_file.relative_to(ROOT).as_posix(),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def ensure_vectors(config: Config, chunks: list[dict]) -> np.ndarray:
    """已有可用向量则直接加载，否则嵌入并落盘。"""
    vectors_path = config.paths.embeddings_dir / "vectors.npy"
    manifest_path = config.paths.embeddings_dir / "manifest.json"
    if vectors_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["model"] == config.embedding.model
            and manifest["dimensions"] == config.embedding.dimensions
            and manifest["count"] == len(chunks)
        ):
            print(f"复用已嵌入向量：{manifest['count']} 条 -> {vectors_path}")
            return np.load(vectors_path)
        print("已有向量与当前模型/语料不一致，重新嵌入 ...")
    else:
        print("未找到已嵌入向量，开始嵌入 ...")

    started = time.perf_counter()
    vectors = embed_chunks(build_embeddings(config), config, chunks)
    save_vectors(config, vectors)
    print(
        f"{len(chunks)} 个 chunk -> {vectors.shape} float32，"
        f"耗时 {time.perf_counter() - started:.1f}s -> {config.paths.embeddings_dir}"
    )
    return vectors


def search(
    config: Config,
    vectors: np.ndarray,
    chunks: list[dict],
    text: str,
    k: int,
) -> list[tuple[float, dict]]:
    """返回 top-k 命中 chunk 及其余弦相似度（降序）。

    度量：余弦相似度 —— 查询与语料向量都先做 L2 归一化，再取内积，
    取值 [-1, 1]，越大越相近；归一化可消除向量长度对打分的影响。
    """
    query = np.asarray(build_embeddings(config).embed_query(text), dtype=np.float32)
    # 余弦相似度 = 归一化后的内积
    query /= np.linalg.norm(query)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = (vectors / np.maximum(norms, 1e-12)) @ query
    top = np.argsort(-scores)[: min(k, len(chunks))]
    return [(float(scores[index]), chunks[index]) for index in top]


def format_snippet(text: str) -> str:
    """压平空白并截断，便于终端查看。"""
    flat = " ".join(text.split())
    return flat[:SNIPPET_CHARS] + ("..." if len(flat) > SNIPPET_CHARS else "")


def print_hits(hits: list[tuple[float, dict]]) -> None:
    """按「排名 分数 来源#偏移」+ 文本片段的形式打印检索结果。"""
    for rank, (score, chunk) in enumerate(hits, start=1):
        print(f"{rank}. {score:.3f}  {chunk['source']}#{chunk['start']}-{chunk['end']}")
        print(f"   {format_snippet(chunk['text'])}")


def main() -> None:
    load_env()
    config = load_config()
    chunks = load_chunks(config)
    vectors = ensure_vectors(config, chunks)

    print(f"查询: {DEMO_QUERY}")
    print_hits(search(config, vectors, chunks, DEMO_QUERY, config.retrieval.top_n))


if __name__ == "__main__":
    main()
