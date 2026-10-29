"""chunk 语料的唯一真源。

`ChunkStore` 从 `chunks.jsonl` 一次性加载全部 `Chunk`，并对外提供：

- `digest`：文件全字节 sha256，供向量 / BM25 的 manifest 校验；
- `by_id`：检索结果回表，BM25 索引只需存 id，不再复制整份语料。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterator, Mapping

from fastrag.config import Config
from fastrag.domain import Chunk
from fastrag.errors import FastragError

BLOCK_SIZE = 1 << 20


def file_sha256(path: pathlib.Path) -> str:
    """返回文件全部字节的 sha256。"""
    if not path.exists():
        raise FastragError(
            f"chunk 语料不存在: {path}\n"
            "请先运行: fastrag chunk（或 uv run python -m fastrag prepare）"
        )
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ChunkStore:
    """已加载的 chunk 集合。"""

    path: pathlib.Path
    chunks: tuple[Chunk, ...]
    digest: str
    by_id: Mapping[str, Chunk] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "by_id", MappingProxyType({chunk.id: chunk for chunk in self.chunks})
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self.by_id.get(chunk_id)

    @classmethod
    def load(cls, config: Config) -> "ChunkStore":
        path = config.paths.chunks_file
        digest = file_sha256(path)
        chunks: list[Chunk] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        chunks.append(Chunk.from_record(json.loads(line)))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        raise FastragError(
                            f"chunk 语料第 {number} 行格式错误: {path}: {exc}"
                        ) from None
        except OSError as exc:
            raise FastragError(f"读取 chunk 语料失败: {path}: {exc}") from None
        return cls(path=path, chunks=tuple(chunks), digest=digest)
