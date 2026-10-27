"""读取 chunk 语料并计算内容指纹，供各阶段校验产物版本。

`chunks.jsonl` 是向量与 BM25 索引共同的上游：只要内容变化，下游就必须重建。
各阶段在元信息里记录同一个 sha256，加载时比对，避免「count 没变但内容变了」
导致的静默错位。
"""

import hashlib
import json
import pathlib

from fastrag.config import Config
from fastrag.errors import FastragError

BLOCK_SIZE = 1 << 20


def load_chunks(config: Config) -> list[dict]:
    """读取 JSONL 形式的 chunk 语料；缺失时提示先切块。"""
    path = config.paths.chunks_file
    if not path.exists():
        raise FastragError(
            f"chunk 语料不存在: {path}\n"
            "请先运行: uv run python -m fastrag.chunk_docs"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def chunks_sha256(path: pathlib.Path) -> str:
    """返回 chunks.jsonl 的 sha256 指纹。

    范围是文件的**全部字节**（含换行与分隔符），不解析 JSON、不按 chunk
    逐条累加，因此：
    - 同样的字节内容永远得到同一个指纹，与文件路径、是否复制/移动无关；
    - 只有字节内容变化才会换指纹，时间戳或大小变化本身不影响结果；
    - 行序、空白、字段顺序等任何改动都会让指纹变化，而 `count` 不会。

    每次调用都完整读一遍（约 5 MB / 几毫秒），不做按 mtime 的缓存，
    避免「内容变了但 mtime 没变」时拿到旧指纹。
    """
    if not path.exists():
        raise FastragError(
            f"chunk 语料不存在: {path}\n"
            "请先运行: uv run python -m fastrag.chunk_docs"
        )
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(BLOCK_SIZE), b""):
            sha.update(block)
    return sha.hexdigest()


def chunking_params(config: Config) -> dict:
    """记录影响 chunk 内容的切块参数，随下游元信息一起落盘。

    仅作排查用的快照，**不参与**向量/索引的复用与校验判断：
    `chunks.jsonl` 的全字节 sha256 已经覆盖了切块结果，改切块参数但没重切块时，
    旧向量/索引与现有语料仍然一致，继续复用是正确的。
    """
    chunking = config.chunking
    return {
        "tokenizer": chunking.tokenizer,
        "chunk_size": chunking.chunk_size,
        "chunk_overlap": chunking.chunk_overlap,
        "include_delim": chunking.include_delim,
        "boilerplate_pattern": chunking.boilerplate_pattern,
    }
