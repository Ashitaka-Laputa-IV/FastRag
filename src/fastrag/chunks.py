"""读取 chunk 语料并计算内容指纹，供各阶段校验产物版本。

`chunks.jsonl` 是向量与 BM25 索引共同的上游：只要内容变化，下游就必须重建。
各阶段在元信息里记录同一个 sha256，加载时比对，避免「count 没变但内容变了」
导致的静默错位。
"""

import functools
import hashlib
import json
import pathlib

from fastrag.config import Config

BLOCK_SIZE = 1 << 20


def load_chunks(config: Config) -> list[dict]:
    """读取 JSONL 形式的 chunk 语料；缺失时提示先切块。"""
    path = config.paths.chunks_file
    if not path.exists():
        raise SystemExit(
            f"chunk 语料不存在: {path}\n"
            "请先运行: uv run python -m fastrag.chunk_docs"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@functools.lru_cache(maxsize=8)
def _digest(path: str, mtime_ns: int, size: int) -> str:
    """按 (路径, mtime, 大小) 缓存指纹，避免每次检索都重读整份语料。"""
    sha = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(BLOCK_SIZE), b""):
            sha.update(block)
    return sha.hexdigest()


def chunks_sha256(path: pathlib.Path) -> str:
    """返回 chunks.jsonl 的 sha256；mtime 或大小变化后自动重算。"""
    if not path.exists():
        raise SystemExit(
            f"chunk 语料不存在: {path}\n"
            "请先运行: uv run python -m fastrag.chunk_docs"
        )
    stat = path.stat()
    return _digest(str(path), stat.st_mtime_ns, stat.st_size)


def chunking_params(config: Config) -> dict:
    """记录影响 chunk 内容的切块参数，随下游元信息一起落盘。"""
    chunking = config.chunking
    return {
        "tokenizer": chunking.tokenizer,
        "chunk_size": chunking.chunk_size,
        "chunk_overlap": chunking.chunk_overlap,
        "include_delim": chunking.include_delim,
        "boilerplate_pattern": chunking.boilerplate_pattern,
    }
