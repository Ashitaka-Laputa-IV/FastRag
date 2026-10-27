"""用 chonkie 把 data/langchain-docs/ 下的示例文档切成句子级 chunk。

- 参数全部来自 config/fastrag.toml（[paths]、[chunking]）
- 切分器：chonkie.SentenceChunker（按句末标点与换行切句）
- 长度：按 token 计量（tiktoken 近似），默认 512 token（约 500），
  便于对齐后续嵌入模型的上下文上限；tiktoken 不是嵌入模型的分词器，
  块长与真实 token 数存在偏差
- overlap：SentenceChunker 原生支持 chunk_overlap，默认 128 token（约 25%）；
  重叠量按句子边界取整，句边界大于目标值时可能不足 128 token
- 模板头：`[chunking].boilerplate_pattern` 给出一条从文档开头剥离的模板头正则
  （默认匹配示例语料的导航块，留空表示不剥离）；偏移仍按原文记录，
  `text == 原文[start:end]` 不受影响
- 元信息：source（相对仓库根目录的文档路径）、start / end（文档内字符偏移）、
  token_count（token 数）

每个 chunk 输出为 JSONL 的一行，便于后续灌向量库。

用法:
    uv run python -m fastrag.chunk_docs
"""

import json
import pathlib
import re

import tiktoken
from chonkie import SentenceChunker

from fastrag.config import ROOT, Config, load_config
from fastrag.errors import FastragError, run_cli


def iter_docs(config: Config) -> list[pathlib.Path]:
    """返回待切分的 markdown 文档（按路径排序，保证输出稳定）。"""
    docs_dir = config.paths.docs_dir
    if not docs_dir.is_dir():
        raise FastragError(
            f"文档目录不存在: {docs_dir}\n"
            "请先运行: uv run python -m fastrag.download_example_docs"
        )
    return sorted(docs_dir.rglob("*.md"))


def strip_boilerplate(
    text: str, pattern: re.Pattern[str] | None
) -> tuple[str, int]:
    """去掉文档开头的站点模板头，返回（正文, 前缀字符数）。

    模板头正则来自 `[chunking].boilerplate_pattern`，pattern 为 None 时原样返回。
    前缀长度会加回 chunk 偏移，因此 `text == 原文[start:end]` 依然成立。
    """
    if pattern is None:
        return text, 0
    match = pattern.match(text)
    if match is None:
        return text, 0
    return text[match.end() :], match.end()


def build_records(config: Config) -> list[dict]:
    """切分所有文档，返回带元信息的 chunk 记录。"""
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

    records = []
    for path in iter_docs(config):
        source = path.relative_to(ROOT).as_posix()
        text, offset = strip_boilerplate(
            path.read_text(encoding="utf-8"), pattern
        )
        if not text.strip():
            continue
        for chunk in chunker.chunk(text):
            start = chunk.start_index + offset
            end = chunk.end_index + offset
            records.append(
                {
                    "id": f"{source}#{start}-{end}",
                    "source": source,
                    "start": start,
                    "end": end,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                }
            )
    return records


def main() -> None:
    config = load_config()
    records = build_records(config)
    if not records:
        print(f"没有可切分的文档 -> {config.paths.docs_dir}，未生成 chunk 文件")
        return

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
    run_cli(main)
