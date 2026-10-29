"""CLI 共用输出格式。"""

from __future__ import annotations

from collections.abc import Sequence

from fastrag.domain import ScoredChunk

SNIPPET_CHARS = 160


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """压平空白并截断，便于终端查看。"""
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


def format_hits(hits: Sequence[ScoredChunk], digits: int = 3) -> list[str]:
    """把命中结果格式化成终端行。"""
    lines: list[str] = []
    for item in hits:
        chunk = item.chunk
        lines.append(
            f"{item.rank}. {item.score:.{digits}f}  "
            f"{chunk.source}#{chunk.start}-{chunk.end}"
        )
        lines.append(f"   {snippet(chunk.text)}")
    return lines
