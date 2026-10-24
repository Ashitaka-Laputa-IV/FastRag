"""为切块后的 chunk 生成向量，并做向量检索。

模型与接口由 config/embedding.toml 指定，代码与具体模型无关。

- 参数来自 config/embedding.toml；API key 只从 .env / 环境变量读取
- 嵌入走 langchain 的 OpenAIEmbeddings（OpenAI 兼容接口）
- 输入是 chunk_docs 产出的 data/chunks.jsonl，输出 data/embeddings/ 下的
  vectors.npy（按 chunks.jsonl 行序排列的 float32 矩阵）与 manifest.json
- search 子命令：嵌入查询文本，与已落盘向量做余弦相似度，返回 top-k

用法:
    uv run python -m fastrag.embed_chunks                     # 全量嵌入
    uv run python -m fastrag.embed_chunks embed --limit 8     # 冒烟测试
    uv run python -m fastrag.embed_chunks search "human in the loop" -k 3

检索示例:
    $ uv run python -m fastrag.embed_chunks search "human in the loop approval" -k 2
    1. 0.685  data/langchain-docs/oss/python/deepagents/human-in-the-loop.md#0-2111
       > ## Documentation Index > Fetch the complete documentation index at: https://docs...
    2. 0.679  data/langchain-docs/oss/python/concepts/human-in-the-loop.md#0-2152
       > ## Documentation Index > Fetch the complete documentation index at: https://docs...
"""

import argparse
import datetime
import json
import os
import pathlib
import time
import tomllib
from dataclasses import dataclass

import numpy as np
from langchain_openai import OpenAIEmbeddings

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "embedding.toml"
ENV_PATH = ROOT / ".env"
SNIPPET_CHARS = 160


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入配置，对应 config/embedding.toml。"""

    model: str
    batch_size: int
    dimensions: int
    chunks_file: pathlib.Path
    out_dir: pathlib.Path
    base_url: str
    api_key_env: str
    timeout: float
    max_retries: int
    k: int


def load_env(path: pathlib.Path = ENV_PATH) -> None:
    """把 .env 里的键值写进环境变量（不覆盖已存在的值）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: pathlib.Path = CONFIG_PATH) -> EmbeddingConfig:
    """读取 TOML 配置，并把相对路径解析到仓库根目录下。"""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    model, paths, api, search = raw["model"], raw["paths"], raw["api"], raw["search"]
    return EmbeddingConfig(
        model=model["name"],
        batch_size=model["batch_size"],
        dimensions=model["dimensions"],
        chunks_file=(ROOT / paths["chunks_file"]).resolve(),
        out_dir=(ROOT / paths["out_dir"]).resolve(),
        base_url=api["base_url"],
        api_key_env=api["api_key_env"],
        timeout=api["timeout"],
        max_retries=api["max_retries"],
        k=search["k"],
    )


def load_chunks(config: EmbeddingConfig) -> list[dict]:
    """读取 JSONL 形式的 chunk 语料。"""
    with config.chunks_file.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_embeddings(config: EmbeddingConfig) -> OpenAIEmbeddings:
    """按配置建 OpenAI 兼容的 embeddings 客户端，缺 key 时给出明确提示。"""
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise SystemExit(
            f"缺少 {config.api_key_env}，请先 cp .env.example .env 并填写"
        )
    return OpenAIEmbeddings(
        model=config.model,
        base_url=config.base_url,
        api_key=api_key,
        chunk_size=config.batch_size,
        max_retries=config.max_retries,
        request_timeout=config.timeout,
        # 非 OpenAI 模型不做 tiktoken 预切分，交给服务端处理
        check_embedding_ctx_length=False,
    )


def embed_chunks(
    embeddings: OpenAIEmbeddings,
    config: EmbeddingConfig,
    chunks: list[dict],
    batch_size: int,
) -> np.ndarray:
    """分批嵌入全部 chunk，返回 (N, dimensions) 的 float32 矩阵。"""
    vectors = np.zeros((len(chunks), config.dimensions), dtype=np.float32)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = embeddings.embed_documents([chunk["text"] for chunk in batch])
        if len(batch_vectors[0]) != config.dimensions:
            raise SystemExit(
                f"返回维度 {len(batch_vectors[0])} 与配置 dimensions={config.dimensions} 不一致"
            )
        vectors[start : start + len(batch)] = batch_vectors
        print(
            f"\r已嵌入 {min(start + batch_size, len(chunks))}/{len(chunks)}",
            end="",
            flush=True,
        )
    print()
    return vectors


def save_vectors(config: EmbeddingConfig, vectors: np.ndarray) -> None:
    """落盘向量矩阵与元信息（对齐 chunks.jsonl 行序）。"""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.out_dir / "vectors.npy", vectors)
    manifest = {
        "model": config.model,
        "dimensions": config.dimensions,
        "count": int(vectors.shape[0]),
        "chunks_file": config.chunks_file.relative_to(ROOT).as_posix(),
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    (config.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_vectors(config: EmbeddingConfig) -> tuple[np.ndarray, dict]:
    """读取已落盘的向量与元信息。"""
    vectors_path = config.out_dir / "vectors.npy"
    manifest_path = config.out_dir / "manifest.json"
    if not vectors_path.exists() or not manifest_path.exists():
        raise SystemExit(
            "向量不存在，请先运行: uv run python -m fastrag.embed_chunks embed"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return np.load(vectors_path), manifest


def search(
    config: EmbeddingConfig, text: str, k: int
) -> list[tuple[float, dict]]:
    """返回 top-k 命中 chunk 及其余弦相似度（降序）。

    度量：余弦相似度 —— 查询与语料向量都先做 L2 归一化，再取内积，
    取值 [-1, 1]，越大越相近；归一化可消除向量长度对打分的影响。
    """
    chunks = load_chunks(config)
    vectors, manifest = load_vectors(config)
    if manifest["model"] != config.model or manifest["count"] != len(chunks):
        raise SystemExit("向量与当前模型/语料不一致，请重新运行 embed")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="chunk 向量化与检索")
    sub = parser.add_subparsers(dest="command")
    embed = sub.add_parser("embed", help="嵌入全部 chunk（默认）")
    embed.add_argument("--limit", type=int, default=None, help="只嵌入前 N 条")
    embed.add_argument("--batch-size", type=int, default=None, help="覆盖 batch_size")
    query = sub.add_parser("search", help="向量检索")
    query.add_argument("text", help="查询语句")
    query.add_argument("-k", type=int, default=None, help="返回条数，默认取配置值")
    args = parser.parse_args()

    load_env()
    config = load_config()

    if args.command == "search":
        for rank, (score, chunk) in enumerate(
            search(config, args.text, args.k or config.k), start=1
        ):
            print(f"{rank}. {score:.3f}  {chunk['source']}#{chunk['start']}-{chunk['end']}")
            print(f"   {format_snippet(chunk['text'])}")
        return

    chunks = load_chunks(config)
    if getattr(args, "limit", None):
        chunks = chunks[: args.limit]
    started = time.perf_counter()
    vectors = embed_chunks(
        build_embeddings(config),
        config,
        chunks,
        getattr(args, "batch_size", None) or config.batch_size,
    )
    save_vectors(config, vectors)
    print(
        f"{len(chunks)} 个 chunk -> {vectors.shape} float32，"
        f"耗时 {time.perf_counter() - started:.1f}s -> {config.out_dir}"
    )


if __name__ == "__main__":
    main()
