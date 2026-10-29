"""用 init_chat_model 创建一个通用对话模型。

- 模型、provider、temperature、thinking、超时来自配置文件的 [llm.*]
- 接口地址与 key 环境变量由该角色引用的 [endpoints.*] 决定；
  api_key_env 留空表示本地服务，客户端用一个占位 key
- 路由（判断检索 + 生成两路查询）/ 生成在 rag_graph 中分别创建，互不影响
- API key 只从 .env 读取
- 重试交给客户端 SDK（max_retries）：它只对超时、限流、5xx 等瞬时错误
  做指数退避，并遵守 Retry-After，不需要在应用层再包一层
- 重试耗尽后按 [llm.*.fallback] 用 with_fallbacks 逐级切换备用模型
  （fallback 自身也可再带 fallback）
- 空回答按失败处理，直接触发备用模型
- main 用生成角色做一次最简单的问答，无交互

用法:
    uv run python -m fastrag.llm

输出示例:
    $ uv run python -m fastrag.llm
    问题: 用一句话解释什么是 RAG。
    回答: RAG（检索增强生成）是指先检索外部知识库中的相关信息，再让大模型基于这些信息生成更准确、可追溯的回答。
"""

import os
from collections.abc import Callable

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from fastrag.config import LLMConfig, load_config, load_env
from fastrag.errors import FastragError, run_cli

DEMO_QUESTION = "用一句话解释什么是 RAG。"


class EmptyResponseError(RuntimeError):
    """模型返回空文本：按失败处理，切换备用模型。"""


def build_llm(config: LLMConfig, thinking: bool | None = None) -> BaseChatModel:
    """按角色配置创建对话模型，缺 key 时给出明确提示。

    thinking 覆盖配置里的 llm.*.thinking；关闭时下发
    `extra_body={"enable_thinking": False}`，开启时沿用厂家默认行为。
    endpoint 的 api_key_env 留空视为本地服务：SDK 要求非空 key，
    这里用占位值 "local"。

    retries 与 timeout 直接交给客户端：重试由 SDK 对瞬时错误做退避，
    应用层只负责重试耗尽后的备用模型，避免两层重试叠加。
    """
    api_key = os.environ.get(config.endpoint.api_key_env)
    if not api_key:
        if config.endpoint.api_key_env:
            raise FastragError(
                f"缺少 {config.endpoint.api_key_env}，请先 cp .env.example .env 并填写"
            )
        api_key = "local"
    enabled = config.thinking if thinking is None else thinking
    extra = {} if enabled else {"extra_body": {"enable_thinking": False}}
    return init_chat_model(
        config.model,
        model_provider=config.provider,
        base_url=config.endpoint.base_url,
        api_key=api_key,
        temperature=config.temperature,
        timeout=config.timeout,
        max_retries=config.retries,
        **extra,
    )


def extract_text(message: BaseMessage) -> str:
    """取回答正文；空回答抛异常，交给 with_fallbacks 切换备用模型。"""
    text = message.text.strip()
    if not text:
        raise EmptyResponseError("模型返回空回答")
    return text


def build_with_fallbacks(
    build: Callable[[LLMConfig], Runnable],
    config: LLMConfig,
) -> Runnable:
    """按 fallback 链逐级兜底；重试由各模型客户端的 max_retries 负责。

    `build` 负责把单个 LLMConfig 变成一条链（例如再套 with_structured_output）。
    遍历 fallback 的 fallback，因此任意层级都能生效；全部失败时抛出最后一个异常。
    """
    chain = build(config)
    fallbacks = []
    fallback = config.fallback
    while fallback is not None:
        fallbacks.append(build(fallback))
        fallback = fallback.fallback
    return chain.with_fallbacks(fallbacks) if fallbacks else chain


def ask(config: LLMConfig, question: str) -> str:
    """发一次单轮提问，返回文本回答；失败时按配置切换备用模型。"""
    chain = build_with_fallbacks(
        lambda role: build_llm(role) | extract_text,
        config,
    )
    try:
        return chain.invoke(question)
    except Exception as exc:  # noqa: BLE001 - 库层统一转成 FastragError
        raise FastragError(
            f"调用模型失败（{type(exc).__name__}: {exc}）"
        ) from None


def main() -> None:
    load_env()
    config = load_config()
    print(f"问题: {DEMO_QUESTION}")
    print(f"回答: {ask(config.llm.answer, DEMO_QUESTION)}")


if __name__ == "__main__":
    run_cli(main)
