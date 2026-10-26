"""用 chonkie 把 data/langchain-docs/ 下的示例文档切成句子级 chunk。

- 参数全部来自 config/fastrag.toml（[paths]、[chunking]）
- 切分器：chonkie.SentenceChunker（按句末标点与换行切句）
- 长度：按 token 计量（tiktoken 编码），默认 512 token（约 500），
  便于对齐后续嵌入模型的上下文上限
- overlap：SentenceChunker 原生支持 chunk_overlap，默认 128 token（约 25%）；
  重叠量按句子边界取整，句边界大于目标值时可能不足 128 token
- 元信息：source（相对仓库根目录的文档路径）、start / end（文档内字符偏移，
  满足 text == 原文[start:end]）、token_count（token 数）

每个 chunk 输出为 JSONL 的一行，便于后续灌向量库。

用法:
    uv run python -m fastrag.chunk_docs
"""

import json
import pathlib

import tiktoken
from chonkie import SentenceChunker

from fastrag.config import ROOT, Config, load_config


def iter_docs(config: Config) -> list[pathlib.Path]:
    """返回待切分的 markdown 文档（按路径排序，保证输出稳定）。"""
    return sorted(config.paths.docs_dir.rglob("*.md"))


def build_records(config: Config) -> list[dict]:
    """切分所有文档，返回带元信息的 chunk 记录。"""
    encoding = tiktoken.get_encoding(config.chunking.tokenizer)
    chunker = SentenceChunker(
        tokenizer=encoding,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        include_delim=config.chunking.include_delim,
    )

    records = []
    for path in iter_docs(config):
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
    config = load_config()
    records = build_records(config)
    config.paths.chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with config.paths.chunks_file.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    tokens = [record["token_count"] for record in records]
    sources = {record["source"] for record in records}
    print(
        f"{len(sources)} 篇文档 -> {len(records)} 个 chunk，"
        f"token 数 {min(tokens)}~{max(tokens)}，"
        f"平均 {sum(tokens) / len(tokens):.0f} token -> {config.paths.chunks_file}"
    )


if __name__ == "__main__":
    main()
