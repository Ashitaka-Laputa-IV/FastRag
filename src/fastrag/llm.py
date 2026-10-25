"""用 init_chat_model 创建一个通用对话模型。

- 模型、provider、base_url 等由 config/llm.toml 指定，代码与具体模型无关
- provider 交给 langchain 的 `init_chat_model`，默认走 OpenAI 兼容接口
- API key 只从 .env 读取
- main 用一个固定问题做一次最简单的问答，无交互

用法:
    uv run python -m fastrag.llm

输出示例:
    $ uv run python -m fastrag.llm
    问题: 用一句话解释什么是 RAG。
    回答: RAG（检索增强生成）是指先检索外部知识库中的相关信息，再让大模型基于这些信息生成更准确、可追溯的回答。
"""

import os
import pathlib
import tomllib
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "llm.toml"
ENV_PATH = ROOT / ".env"
DEMO_QUESTION = "用一句话解释什么是 RAG。"


@dataclass(frozen=True)
class LLMConfig:
    """LLM 配置，对应 config/llm.toml。"""

    model: str
    provider: str
    base_url: str
    api_key_env: str
    temperature: float


def load_env(path: pathlib.Path = ENV_PATH) -> None:
    """把 .env 里的键值写进环境变量（不覆盖已存在的值）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: pathlib.Path = CONFIG_PATH) -> LLMConfig:
    """读取 TOML 配置。"""
    llm = tomllib.loads(path.read_text(encoding="utf-8"))["llm"]
    return LLMConfig(
        model=llm["model"],
        provider=llm["provider"],
        base_url=llm["base_url"],
        api_key_env=llm["api_key_env"],
        temperature=llm["temperature"],
    )


def build_llm(config: LLMConfig) -> BaseChatModel:
    """按配置创建对话模型，缺 key 时给出明确提示。"""
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise SystemExit(f"缺少 {config.api_key_env}，请先 cp .env.example .env 并填写")
    return init_chat_model(
        config.model,
        model_provider=config.provider,
        base_url=config.base_url,
        api_key=api_key,
        temperature=config.temperature,
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
