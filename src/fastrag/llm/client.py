"""LLM 客户端构建与多级兜底。

- 模型、provider、temperature、thinking、超时来自 `[llm.*]`；
- 接口地址与 key 由角色引用的 `[endpoints.*]` 决定；
- 重试交给客户端 SDK（`max_retries`，只对瞬时错误退避）；
- 重试耗尽后按 `fallback` 用 `with_fallbacks` 逐级切换；
- 空回答按失败处理，直接触发备用模型。
"""

from __future__ import annotations

from collections.abc import Callable

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from fastrag.config import Config, LLMConfig, endpoint_api_key
from fastrag.errors import FastragError


class EmptyResponseError(RuntimeError):
    """模型返回空文本：按失败处理，切换备用模型。"""


def build_llm(role: LLMConfig, config: Config, thinking: bool | None = None) -> BaseChatModel:
    """按角色配置创建对话模型。"""
    enabled = role.thinking if thinking is None else thinking
    extra_body = dict(role.extra_body)
    if not enabled:
        extra_body.setdefault("enable_thinking", False)
    kwargs = {"extra_body": extra_body} if extra_body else {}
    return init_chat_model(
        role.model,
        model_provider=role.provider,
        base_url=config.endpoints[role.endpoint].base_url,
        api_key=endpoint_api_key(config, role.endpoint),
        temperature=role.temperature,
        timeout=role.timeout,
        max_retries=role.retries,
        **kwargs,
    )


def extract_text(message: BaseMessage) -> str:
    """取回答正文；空回答抛异常，交给 `with_fallbacks` 切换备用模型。"""
    text = message.text.strip()
    if not text:
        raise EmptyResponseError("模型返回空回答")
    return text


def build_with_fallbacks(
    build: Callable[[LLMConfig], Runnable],
    config: Config,
    role: LLMConfig,
) -> Runnable:
    """按 fallback 链逐级兜底；重试由各模型客户端的 `max_retries` 负责。"""
    chain = build(role)
    fallbacks = []
    fallback = role.fallback
    while fallback is not None:
        fallbacks.append(build(fallback))
        fallback = fallback.fallback
    return chain.with_fallbacks(fallbacks) if fallbacks else chain


def ask(config: Config, role: LLMConfig, question: str) -> str:
    """发一次单轮提问，返回文本回答。"""
    chain = build_with_fallbacks(
        lambda item: build_llm(item, config) | extract_text,
        config,
        role,
    )
    try:
        return chain.invoke(question)
    except Exception as exc:  # noqa: BLE001 - 库层统一转成 FastragError
        raise FastragError(
            f"调用模型失败（{type(exc).__name__}: {exc}）"
        ) from None
