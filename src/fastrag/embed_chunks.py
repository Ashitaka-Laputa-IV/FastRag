"""用 Qwen3-Embedding 为 chunk 生成向量。

- 参数来自 config/embedding.toml；api key 与 base_url 只从 .env / 环境变量读取
- 输入是 chunk_docs 产出的 data/chunks.jsonl，输出 data/embeddings/ 下的
  vectors.npy（按 chunks.jsonl 行序排列的 float32 矩阵）与 manifest.json
- 走 OpenAI 兼容接口（SiliconFlow），模型默认 Qwen/Qwen3-Embedding-4B

用法:
    cp .env.example .env          # 填 SILICONFLOW_API_KEY
    uv run python -m fastrag.embed_chunks            # 全量
    uv run python -m fastrag.embed_chunks --limit 8  # 冒烟测试
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
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "embedding.toml"
ENV_PATH = ROOT / ".env"


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入配置，对应 config/embedding.toml。"""

    model: str
    batch_size: int
    dimensions: int
    chunks_file: pathlib.Path
    out_dir: pathlib.Path
    api_key_env: str
    base_url_env: str
    timeout: float
    max_retries: int


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
    model, paths, api = raw["model"], raw["paths"], raw["api"]
    return EmbeddingConfig(
        model=model["name"],
        batch_size=model["batch_size"],
        dimensions=model["dimensions"],
        chunks_file=(ROOT / paths["chunks_file"]).resolve(),
        out_dir=(ROOT / paths["out_dir"]).resolve(),
        api_key_env=api["api_key_env"],
        base_url_env=api["base_url_env"],
        timeout=api["timeout"],
        max_retries=api["max_retries"],
    )


def load_chunks(config: EmbeddingConfig) -> list[dict]:
    """读取 JSONL 形式的 chunk 语料。"""
    with config.chunks_file.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_client(config: EmbeddingConfig) -> OpenAI:
    """按 .env 里的凭据建 OpenAI 兼容客户端，缺凭据时给出明确提示。"""
    api_key = os.environ.get(config.api_key_env)
    base_url = os.environ.get(config.base_url_env)
    if not api_key or not base_url:
        raise SystemExit(
            f"缺少 {config.api_key_env} / {config.base_url_env}，"
            "请先 cp .env.example .env 并填写"
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=config.timeout)


def embed_batch(
    client: OpenAI, config: EmbeddingConfig, texts: list[str]
) -> list[list[float]]:
    """嵌入一批文本，失败按指数退避重试。"""
    last_error: Exception | None = None
    for attempt in range(config.max_retries):
        try:
            response = client.embeddings.create(
                model=config.model, input=texts, encoding_format="float"
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"嵌入请求连续失败 {config.max_retries} 次: {last_error!r}")


def embed_chunks(
    client: OpenAI, config: EmbeddingConfig, chunks: list[dict], batch_size: int
) -> np.ndarray:
    """分批嵌入全部 chunk，返回 (N, dimensions) 的 float32 矩阵。"""
    vectors = np.zeros((len(chunks), config.dimensions), dtype=np.float32)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embed_batch(client, config, [chunk["text"] for chunk in batch])
        if len(embeddings[0]) != config.dimensions:
            raise SystemExit(
                f"返回维度 {len(embeddings[0])} 与配置 dimensions={config.dimensions} 不一致"
            )
        vectors[start : start + len(batch)] = embeddings
        print(f"\r已嵌入 {min(start + batch_size, len(chunks))}/{len(chunks)}", end="", flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Qwen3-Embedding 为 chunk 生成向量")
    parser.add_argument("--limit", type=int, default=None, help="只嵌入前 N 条（冒烟测试）")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖配置里的 batch_size")
    args = parser.parse_args()

    load_env()
    config = load_config()
    chunks = load_chunks(config)
    if args.limit:
        chunks = chunks[: args.limit]

    started = time.perf_counter()
    vectors = embed_chunks(
        build_client(config), config, chunks, args.batch_size or config.batch_size
    )
    save_vectors(config, vectors)
    print(
        f"{len(chunks)} 个 chunk -> {vectors.shape} float32，"
        f"耗时 {time.perf_counter() - started:.1f}s -> {config.out_dir}"
    )


if __name__ == "__main__":
    main()
