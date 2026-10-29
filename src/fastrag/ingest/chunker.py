"""把文档切成句子级 chunk，并落盘为 JSONL。

- 参数全部来自 `[paths]`、`[chunking]`；
- chunk id 采用内容寻址（`sha1(source + text)`），文档局部修改时只影响变化的块；
- `start` / `end` 仍是原文偏移，保证 `text == 原文[start:end]`（模板头剥离后加回偏移）；
- 输出先写临时文件再原子替换。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import tempfile

import tiktoken
from chonkie import SentenceChunker

from fastrag.config import Config, display_path
from fastrag.domain import Chunk
from fastrag.errors import FastragError


def iter_docs(config: Config) -> list[pathlib.Path]:
    """返回待切分的 markdown 文档（按路径排序，保证输出稳定）。"""
    docs_dir = config.paths.docs_dir
    if not docs_dir.is_dir():
        raise FastragError(
            f"文档目录不存在: {docs_dir}\n"
            "请先运行: fastrag download（或 uv run python -m fastrag download）"
        )
    return sorted(docs_dir.rglob("*.md"))


def strip_boilerplate(
    text: str, pattern: re.Pattern[str] | None
) -> tuple[str, int]:
    """去掉文档开头的站点模板头，返回（正文, 前缀字符数）。"""
    if pattern is None:
        return text, 0
    match = pattern.match(text)
    if match is None:
        return text, 0
    return text[match.end() :], match.end()


def chunk_id(source: str, text: str) -> str:
    """内容寻址 id：同一文档内文本不变则 id 不变，与位置无关。"""
    digest = hashlib.sha1(f"{source}\0{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def build_chunks(config: Config) -> list[Chunk]:
    """切分所有文档，返回带元信息的 `Chunk` 列表。"""
    encoding = tiktoken.get_encoding(config.chunking.tokenizer)
    chunker = SentenceChunker(
        tokenizer=encoding,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        include_delim=config.chunking.include_delim,
    )
    pattern = (
        re.compile(config.chunking.boilerplate_pattern)
        if config.chunking.boilerplate_pattern
        else None
    )

    chunks: list[Chunk] = []
    for path in iter_docs(config):
        source = display_path(path)
        text, offset = strip_boilerplate(
            path.read_text(encoding="utf-8"), pattern
        )
        if not text.strip():
            continue
        for piece in chunker.chunk(text):
            start = piece.start_index + offset
            end = piece.end_index + offset
            chunks.append(
                Chunk(
                    id=chunk_id(source, piece.text),
                    source=source,
                    start=start,
                    end=end,
                    text=piece.text,
                    token_count=piece.token_count,
                )
            )
    return chunks


def write_chunks(config: Config, chunks: list[Chunk]) -> None:
    """原子写入 JSONL。"""
    path = config.paths.chunks_file
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".chunks.", suffix=".jsonl", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")
        os.replace(temp, path)
    except BaseException:
        pathlib.Path(temp).unlink(missing_ok=True)
        raise


def stats(chunks: list[Chunk]) -> dict[str, float]:
    tokens = [chunk.token_count for chunk in chunks]
    sources = {chunk.source for chunk in chunks}
    return {
        "documents": len(sources),
        "chunks": len(chunks),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "avg_tokens": sum(tokens) / len(tokens),
    }
