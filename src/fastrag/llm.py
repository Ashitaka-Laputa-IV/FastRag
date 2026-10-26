"""用 init_chat_model 创建一个通用对话模型。

- 模型、provider、temperature 来自 config/fastrag.toml 的 [llm]
- 接口地址与 key 环境变量来自 [api]；API key 只从 .env 读取
- provider 交给 langchain 的 `init_chat_model`，默认走 OpenAI 兼容接口
- main 用一个固定问题做一次最简单的问答，无交互

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

from fastrag.config import Config, load_config, load_env

DEMO_QUESTION = "用一句话解释什么是 RAG。"


def build_llm(config: Config) -> BaseChatModel:
    """按配置创建对话模型，缺 key 时给出明确提示。"""
    api_key = os.environ.get(config.api.api_key_env)
    if not api_key:
        raise SystemExit(
            f"缺少 {config.api.api_key_env}，请先 cp .env.example .env 并填写"
        )
    return init_chat_model(
        config.llm.model,
        model_provider=config.llm.provider,
        base_url=config.api.base_url,
        api_key=api_key,
        temperature=config.llm.temperature,
    )


def ask(llm: BaseChatModel, question: str) -> str:
    """发一次单轮提问，返回文本回答。"""
    return llm.invoke(question).text


def main() -> None:
    load_env()
    config = load_config()
    print(f"问题: {DEMO_QUESTION}")
    print(f"回答: {ask(build_llm(config), DEMO_QUESTION)}")


if __name__ == "__main__":
    main()
