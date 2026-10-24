"""用 chonkie 把 data/langchain-docs/ 下的示例文档切成句子级 chunk。

- 切分器：chonkie.SentenceChunker（按句末标点与换行切句）
- 长度：约 500 字符（tokenizer="character"，即以字符数计量）
- overlap：SentenceChunker 原生支持 chunk_overlap，取 100 字符（约 20%）；
  重叠量按句子边界取整，句边界大于目标值时可能不足 100 字符
- 元信息：source（相对仓库根目录的文档路径）、start / end（文档内字符偏移，
  满足 text == 原文[start:end]）、token_count

每个 chunk 输出为 JSONL 的一行，便于后续灌向量库。

用法:
    uv run python -m fastrag.chunk_docs
"""

import json
import pathlib

from chonkie import SentenceChunker

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DOCS_DIR = ROOT / "data" / "langchain-docs"
OUT_PATH = ROOT / "data" / "chunks.jsonl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def iter_docs() -> list[pathlib.Path]:
    """返回待切分的 markdown 文档（按路径排序，保证输出稳定）。"""
    return sorted(DOCS_DIR.rglob("*.md"))


def build_records() -> list[dict]:
    """切分所有文档，返回带元信息的 chunk 记录。"""
    chunker = SentenceChunker(
        tokenizer="character",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        include_delim="prev",
    )

    records = []
    for path in iter_docs():
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        source = path.relative_to(ROOT).as_posix()
        for chunk in chunker.chunk(text):
            records.append(
                {
                    "id": f"{source}#{chunk.start_index}-{chunk.end_index}",
                    "source": source,
                    "start": chunk.start_index,
                    "end": chunk.end_index,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                }
            )
    return records


def main() -> None:
    records = build_records()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [record["end"] - record["start"] for record in records]
    sources = {record["source"] for record in records}
    print(
        f"{len(sources)} 篇文档 -> {len(records)} 个 chunk，"
        f"长度 {min(lengths)}~{max(lengths)}，"
        f"平均 {sum(lengths) / len(lengths):.0f} 字符 -> {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
