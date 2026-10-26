"""对 RAG 各阶段做耗时基准测试。

固定 3 个问题，每个问题跑 runs 次（默认 3），统计路由、问题重写、召回、重排、
生成的耗时；跑完输出 Markdown 表格，并把明细写入 JSON 方便复查。

- 耗时由 rag_graph 的节点装饰器记录在 state["timings"]（毫秒）
- 召回是 BM25 与向量两路并行，阶段耗时取较慢的一路
- 统计前先用第一个问题做一次预热，避免首次建连、懒加载影响结果

用法:
    uv run python -m fastrag.bench
    uv run python -m fastrag.bench --runs 5
    uv run python -m fastrag.bench --json data/bench/results.json

输出示例:
    $ uv run python -m fastrag.bench
    预热完成（不计入统计）
    [1/9] LangGraph 中如何实现 human-in-the-loop？ 4213 ms
    ...
    ## 全部 9 次
    | 阶段 | 平均 (ms) | 最小 (ms) | 最大 (ms) | 标准差 (ms) |
    ...
"""

import argparse
import datetime
import json
import pathlib
import statistics
import time

from fastrag.rag_graph import STAGE_LABELS, Pipeline, get_pipeline, load_config

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON = ROOT / "data" / "bench" / "results.json"
QUESTIONS = (
    "LangGraph 中如何实现 human-in-the-loop？",
    "如何让 LangGraph 的图流式输出 token？",
    "LangChain 的 Agent 如何调用工具？",
)
RUNS = 3
STAGES = ("route", "requery", "retrieve", "rerank", "generate")


def measure(timings: dict[str, float]) -> dict[str, float]:
    """把节点耗时整理成 5 个阶段；召回取 BM25 与向量中较慢的一路。"""
    return {
        "route": timings.get("route", 0.0),
        "requery": timings.get("requery", 0.0),
        "retrieve": max(
            timings.get("retrieve_bm25", 0.0), timings.get("retrieve_vector", 0.0)
        ),
        "rerank": timings.get("rerank", timings.get("fusion", 0.0)),
        "generate": timings.get("generate", 0.0),
    }


def describe(values: list[float]) -> dict[str, float]:
    """均值、中位数、最小、最大、标准差（样本数小于 2 时标准差记 0）。"""
    return {
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
    }


def summarize(records: list[dict]) -> dict:
    """按问题分组统计各阶段与端到端耗时，再给出全部运行的汇总。"""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["question"], []).append(record)

    per_question = []
    for question, rows in groups.items():
        per_question.append(
            {
                "question": question,
                "runs": len(rows),
                "stages": {
                    stage: describe([row[stage] for row in rows]) for stage in STAGES
                },
                "wall": describe([row["wall_ms"] for row in rows]),
            }
        )
    overall = {
        "runs": len(records),
        "stages": {
            stage: describe([row[stage] for row in records]) for stage in STAGES
        },
        "wall": describe([row["wall_ms"] for row in records]),
    }
    return {"per_question": per_question, "overall": overall}


def markdown_table(stages: dict[str, dict[str, float]], wall: dict[str, float]) -> str:
    """把一组统计结果排成 Markdown 表格。"""
    lines = [
        "| 阶段 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) | 标准差 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        item = stages[stage]
        lines.append(
            f"| {STAGE_LABELS[stage]} | {item['mean']:.1f} | {item['median']:.1f} "
            f"| {item['min']:.1f} | {item['max']:.1f} | {item['stdev']:.1f} |"
        )
    lines.append(
        f"| **端到端** | **{wall['mean']:.1f}** | **{wall['median']:.1f}** "
        f"| **{wall['min']:.1f}** | **{wall['max']:.1f}** | **{wall['stdev']:.1f}** |"
    )
    return "\n".join(lines)


def print_report(summary: dict) -> None:
    """按问题输出表格，最后输出全部运行的汇总。"""
    for index, item in enumerate(summary["per_question"], start=1):
        print(f"\n### 问题 {index}：{item['question']}（{item['runs']} 次）")
        print(markdown_table(item["stages"], item["wall"]))
    overall = summary["overall"]
    print(f"\n### 全部 {overall['runs']} 次运行")
    print(markdown_table(overall["stages"], overall["wall"]))


def run(
    pipeline: Pipeline, questions: tuple[str, ...], runs: int, init_ms: float
) -> dict:
    """预热后逐个问题重复运行，返回可落盘的明细与统计。"""
    pipeline.graph.invoke({"question": questions[0]})
    print("预热完成（不计入统计）")

    records = []
    total = len(questions) * runs
    for question in questions:
        for index in range(1, runs + 1):
            started = time.perf_counter()
            state = pipeline.graph.invoke({"question": question})
            wall_ms = (time.perf_counter() - started) * 1000
            stages = measure(state.get("timings", {}))
            record = {
                "question": question,
                "run": index,
                "need_search": bool(state.get("need_search")),
                "search_query": state.get("search_query", ""),
                "wall_ms": round(wall_ms, 1),
                **stages,
                "nodes": state.get("timings", {}),
                "answer_chars": len(state.get("answer", "")),
            }
            records.append(record)
            print(
                f"[{len(records)}/{total}] {question} {wall_ms:.0f} ms",
                flush=True,
            )

    summary = summarize(records)
    return {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "init_ms": round(init_ms, 1),
        "runs_per_question": runs,
        "records": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 各阶段耗时基准测试")
    parser.add_argument("--runs", type=int, default=RUNS, help="每个问题运行次数")
    parser.add_argument(
        "--json", type=pathlib.Path, default=DEFAULT_JSON, help="明细输出路径"
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs 必须大于 0")

    config = load_config()
    started = time.perf_counter()
    pipeline = get_pipeline()
    init_ms = (time.perf_counter() - started) * 1000
    print(f"初始化完成 {init_ms:.0f} ms（模型、向量、索引、重排器，不计入统计）")

    payload = run(pipeline, QUESTIONS, args.runs, init_ms)
    payload["config"] = {
        "method": config.method,
        "candidates": config.candidates,
        "context_n": config.context_n,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n明细已写入 {args.json}")
    print_report(payload["summary"])


if __name__ == "__main__":
    main()
