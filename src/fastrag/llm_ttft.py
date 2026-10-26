"""直接测量各对话模型的首 token 时延（TTFT），不经过 RAG 链路。

对每个候选模型发送同一条短请求，流式读取，记录「发出请求 -> 第一个非空文本
chunk」的时间。和 bench.py 的区别：这里只测模型本身，不含检索、规划、融合。

- 模型与 endpoint 来自 config/fastrag.toml 的 [endpoints.*]
- 推理模型会先输出 reasoning_content，langchain-openai 不解析它；本脚本默认
  关闭思考（enable_thinking=false），保证首 token 就是回答正文，另留一条开启
  思考的 DeepSeek 作为对照
- 每个模型先跑一次预热（不计入统计），再跑 --runs 次
- 失败（例如厂家 503、不支持 enable_thinking）会自动去掉思考参数重试一次

用法:
    uv run python -m fastrag.llm_ttft
    uv run python -m fastrag.llm_ttft --runs 5
    uv run python -m fastrag.llm_ttft --json data/bench/llm-ttft.json

输出示例:
    $ uv run python -m fastrag.llm_ttft
    | 模型 | endpoint | 思考 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) |
    | --- | --- | --- | ---: | ---: | ---: | ---: |
    | deepseek-ai/DeepSeek-V4.1-Flash | siliconflow | 关 | 431.2 | 428.0 | 402.1 | 463.5 |
    ...
"""

import argparse
import dataclasses
import datetime
import json
import pathlib
import statistics
import time

from langchain_core.language_models import BaseChatModel

from fastrag.config import ROOT, LLMConfig, load_config, load_env
from fastrag.llm import build_llm

DEFAULT_JSON = ROOT / "data" / "bench" / "llm-ttft.json"
PROMPT = "用一句话解释什么是 RAG。"
RUNS = 3
TIMEOUT = 30
# (显示名, endpoint, model, thinking)
CASES = (
    ("DeepSeek-V4.1-Flash（开思考）", "siliconflow", "deepseek-ai/DeepSeek-V4.1-Flash", True),
    ("DeepSeek-V4.1-Flash（关思考）", "siliconflow", "deepseek-ai/DeepSeek-V4.1-Flash", False),
    ("DeepSeek-V3.2", "siliconflow", "deepseek-ai/DeepSeek-V3.2", False),
    ("Qwen3.5-35B-A3B", "siliconflow", "Qwen/Qwen3.5-35B-A3B", False),
    ("Qwen3.5-9B", "siliconflow", "Qwen/Qwen3.5-9B", False),
    ("Qwen3-30B-A3B-Instruct", "siliconflow", "Qwen/Qwen3-30B-A3B-Instruct-2507", False),
    ("Qwen3-8B", "siliconflow", "Qwen/Qwen3-8B", False),
    ("Hunyuan-A13B-Instruct", "siliconflow", "tencent/Hunyuan-A13B-Instruct", False),
    ("GLM-4.5-Air", "siliconflow", "zai-org/GLM-4.5-Air", False),
    ("Step-3.5-Flash", "siliconflow", "stepfun-ai/Step-3.5-Flash", False),
    ("Qwen3-4B-Instruct（本地）", "local", "qwen3-4b-instruct", True),
)


def first_token_ms(llm: BaseChatModel, prompt: str) -> tuple[float, str]:
    """发出请求并读到第一个非空文本 chunk，返回耗时（毫秒）与该段文本。"""
    started = time.perf_counter()
    stream = llm.stream(prompt)
    try:
        for chunk in stream:
            if chunk.text:
                return (time.perf_counter() - started) * 1000, chunk.text
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    raise RuntimeError("流结束但没有收到文本 chunk")


def describe(values: list[float]) -> dict[str, float]:
    """均值、中位数、最小、最大（样本数小于 2 时标准差记 0）。"""
    return {
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def _is_bad_request(error: Exception) -> bool:
    """判断异常是否为 400（参数不被支持），只有这种才值得去掉思考参数重试。"""
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status == 400


def measure(
    llm_config: LLMConfig, thinking: bool, runs: int
) -> dict:
    """预热一次后测 runs 次首 token；思考参数被拒时退回厂家默认再试一次。

    注意：每次都用同一条提示词，llama.cpp 会命中 prompt cache，本地的首 token
    会明显偏快；需要冷启动口径时请自行改变提问内容。
    """
    fallback = False
    try:
        llm = build_llm(llm_config, thinking=thinking, timeout=TIMEOUT)
        first_token_ms(llm, PROMPT)
        samples = [first_token_ms(llm, PROMPT)[0] for _ in range(runs)]
    except Exception as error:  # noqa: BLE001
        if thinking or not _is_bad_request(error):
            raise
        # 厂家不接受 enable_thinking 参数时，退回不下发参数（厂家默认）
        fallback = True
        llm = build_llm(llm_config, thinking=True, timeout=TIMEOUT)
        first_token_ms(llm, PROMPT)
        samples = [first_token_ms(llm, PROMPT)[0] for _ in range(runs)]
    return {
        "samples": [round(value, 1) for value in samples],
        **describe(samples),
        "fallback": fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="直接测量各模型的首 token 时延")
    parser.add_argument("--runs", type=int, default=RUNS, help="每个模型重复次数")
    parser.add_argument(
        "--json", type=pathlib.Path, default=DEFAULT_JSON, help="明细输出路径"
    )
    parser.add_argument(
        "--base-url",
        help="改用这个 OpenAI 兼容地址（例如本地 llama.cpp），只测 --model 指定的模型",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="配合 --base-url 使用，可重复；模型名原样作为请求里的 model",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="配合 --base-url：保留思考（默认关闭，即下发 enable_thinking=false）",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs 必须大于 0")
    if args.base_url and not args.model:
        raise SystemExit("--base-url 需要至少一个 --model")

    load_env()
    config = load_config()
    cases = CASES
    if args.base_url:
        cases = tuple((name, "custom", name, args.thinking) for name in args.model)
        config = dataclasses.replace(
            config,
            endpoints={
                **config.endpoints,
                "custom": dataclasses.replace(
                    config.endpoints["local"], base_url=args.base_url
                ),
            },
        )
    results = []
    for label, endpoint_name, model, thinking in cases:
        try:
            endpoint = config.endpoints[endpoint_name]
        except KeyError:
            raise SystemExit(f"endpoint {endpoint_name!r} 未在 [endpoints] 中定义") from None
        llm_config = dataclasses.replace(
            config.llm.answer,
            endpoint=endpoint,
            model=model,
            thinking=thinking,
        )
        try:
            stats = measure(llm_config, thinking, args.runs)
        except Exception as error:  # noqa: BLE001
            print(f"{label}: ERROR {type(error).__name__}: {str(error)[:160]}", flush=True)
            results.append(
                {
                    "label": label,
                    "endpoint": endpoint_name,
                    "model": model,
                    "thinking": thinking,
                    "error": f"{type(error).__name__}: {str(error)[:300]}",
                }
            )
            continue
        print(
            f"{label}: 中位数 {stats['median']:.0f} ms "
            f"({stats['min']:.0f}–{stats['max']:.0f})",
            flush=True,
        )
        results.append(
            {
                "label": label,
                "endpoint": endpoint_name,
                "base_url": endpoint.base_url,
                "model": model,
                "thinking": thinking,
                **stats,
            }
        )

    payload = {
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "prompt": PROMPT,
        "runs_per_model": args.runs,
        "timeout": TIMEOUT,
        "results": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n| 模型 | endpoint | 思考 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for item in results:
        if "error" in item:
            print(f"| {item['label']} | {item['endpoint']} | - | 失败 | - | - | - |")
            continue
        thinking = "开" if item["thinking"] else "关"
        if item.get("fallback"):
            thinking = "默认"
        print(
            f"| {item['label']} | {item['endpoint']} | {thinking} "
            f"| {item['mean']:.1f} | {item['median']:.1f} "
            f"| {item['min']:.1f} | {item['max']:.1f} |"
        )
    print(f"\n明细已写入 {args.json}")


if __name__ == "__main__":
    main()
