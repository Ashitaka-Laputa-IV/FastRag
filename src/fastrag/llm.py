"""用 init_chat_model 创建一个通用对话模型。

- 模型、provider、temperature 来自 config/fastrag.toml 的 [llm.*]
- 接口地址与 key 环境变量由该角色引用的 [endpoints.*] 决定
- 路由（判断检索 + 生成两路查询）/ 生成在 rag_graph 中分别创建，互不影响
- API key 只从 .env 读取
- main 用生成角色做一次最简单的问答，无交互

用法:
    uv run python -m fastrag.llm

输出示例:
    $ uv run python -m fastrag.llm
    问题: 用一句话解释什么是 RAG。
    回答: RAG（检索增强生成）是指先检索外部知识库中的相关信息，再让大模型基于这些信息生成更准确、可追溯的回答。
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from fastrag.config import LLMConfig, load_config, load_env

DEMO_QUESTION = "用一句话解释什么是 RAG。"


def build_llm(config: LLMConfig, thinking: bool | None = None) -> BaseChatModel:
    """按角色配置创建对话模型，缺 key 时给出明确提示。

    thinking 覆盖配置里的 llm.*.thinking；关掉思考时传
    `extra_body={"enable_thinking": False}`（SiliconFlow 等 OpenAI 兼容接口）。
    开启时不下发该参数，沿用厂家默认行为。
    """
    api_key = os.environ.get(config.endpoint.api_key_env)
    if not api_key:
        raise SystemExit(
            f"缺少 {config.endpoint.api_key_env}，请先 cp .env.example .env 并填写"
        )
    enabled = config.thinking if thinking is None else thinking
    extra = {} if enabled else {"extra_body": {"enable_thinking": False}}
    return init_chat_model(
        config.model,
        model_provider=config.provider,
        base_url=config.endpoint.base_url,
        api_key=api_key,
        temperature=config.temperature,
        **extra,
    )


def ask(llm: BaseChatModel, question: str) -> str:
    """发一次单轮提问，返回文本回答。"""
    return llm.invoke(question).text


def main() -> None:
    load_env()
    config = load_config()
    print(f"问题: {DEMO_QUESTION}")
    print(f"回答: {ask(build_llm(config.llm.answer), DEMO_QUESTION)}")


if __name__ == "__main__":
    main()
