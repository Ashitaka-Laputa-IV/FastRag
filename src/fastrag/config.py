"""统一读取配置，并加载 .env 里的 API key。

所有模块共用这一份配置，不再各自维护加载逻辑：

- 默认配置与提示词随包分发（`fastrag.resources`），wheel 安装后依然可用；
- `FASTRAG_HOME` 指定工作目录（默认：源码树根目录，安装 wheel 后为当前目录），
  配置里的相对路径、`.env`、可选的 `prompts/` 覆盖都在该目录下解析；
- `FASTRAG_CONFIG` 可直接指向某份配置（优先于 `FASTRAG_HOME/config/fastrag.toml`）；
- [endpoints.*] 描述厂家（base_url + key 环境变量），各模型按需引用
- 嵌入、重排、判断检索（含两路查询）、生成可以分别使用不同的厂家与模型
- 召回条数（BM25 / 向量分开配置）与输出条数（top_n）只保留一份
"""

import importlib.resources
import os
import pathlib
import re
import tomllib
from dataclasses import dataclass

from fastrag.errors import FastragError

PACKAGE_RESOURCES = importlib.resources.files("fastrag.resources")
FUSION_METHODS = ("rrf", "rerank")
BM25_METHODS = ("lucene", "robertson", "atire", "bm25l", "bm25+")
# 与 bm25s._infer_stopwords 支持的语言名/别名保持一致
BM25_STOPWORDS = (
    "english",
    "en",
    "english_plus",
    "en_plus",
    "german",
    "de",
    "dutch",
    "nl",
    "french",
    "fr",
    "spanish",
    "es",
    "portuguese",
    "pt",
    "italian",
    "it",
    "russian",
    "ru",
    "swedish",
    "sv",
    "norwegian",
    "no",
    "chinese",
    "zh",
    "turkish",
    "tr",
    "korean",
    "ko",
    "danish",
    "da",
)

# editable 安装时 __file__ 位于 <repo>/src/fastrag/，据此识别源码树；
# 装成 wheel 后这里指向 site-packages，不再是项目根目录。
_SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def home() -> pathlib.Path:
    """工作目录：数据、`.env`、自定义配置与提示词的根。

    优先级：`FASTRAG_HOME` > 源码树根目录（editable 运行）> 当前目录。
    """
    override = os.environ.get("FASTRAG_HOME")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    if (_SOURCE_ROOT / "pyproject.toml").exists():
        return _SOURCE_ROOT
    return pathlib.Path.cwd()


def default_config_path() -> pathlib.Path:
    """默认配置：`FASTRAG_HOME/config/fastrag.toml`，不存在时用包内模板。"""
    user_config = home() / "config" / "fastrag.toml"
    if user_config.exists():
        return user_config
    return pathlib.Path(str(PACKAGE_RESOURCES / "config" / "fastrag.toml"))


def prompt_path(name: str) -> pathlib.Path:
    """提示词文件：优先 `FASTRAG_HOME/prompts/<name>.txt`，否则用包内模板。"""
    override = home() / "prompts" / f"{name}.txt"
    if override.exists():
        return override
    return pathlib.Path(str(PACKAGE_RESOURCES / "prompts" / f"{name}.txt"))


def display_path(path: pathlib.Path) -> str:
    """相对工作目录的展示路径；在目录外时退回绝对路径。"""
    try:
        return path.resolve().relative_to(home()).as_posix()
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class Paths:
    """文件与目录，均已解析为绝对路径。"""

    docs_dir: pathlib.Path
    chunks_file: pathlib.Path
    embeddings_dir: pathlib.Path
    bm25_index_dir: pathlib.Path


@dataclass(frozen=True)
class EndpointConfig:
    """一家服务商的接入信息。"""

    base_url: str
    api_key_env: str


@dataclass(frozen=True)
class ChunkingConfig:
    """切块参数。"""

    tokenizer: str
    chunk_size: int
    chunk_overlap: int
    include_delim: str
    boilerplate_pattern: str


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入模型参数。"""

    endpoint: EndpointConfig
    model: str
    batch_size: int
    dimensions: int
    timeout: float
    max_retries: int


@dataclass(frozen=True)
class LLMConfig:
    """单个 LLM 角色的模型参数。

    thinking 只对推理模型有意义：True 用厂家默认（通常开启），
    False 时下发 enable_thinking=false 关掉思考，首 token 直接是回答正文。
    retries 是主模型失败后的重试次数；重试耗尽后若配置了 fallback，
    则改用备用模型（fallback 自身也可以带 retries）。
    """

    endpoint: EndpointConfig
    model: str
    provider: str
    temperature: float
    thinking: bool = True
    retries: int = 0
    fallback: "LLMConfig | None" = None


@dataclass(frozen=True)
class LLMRoles:
    """判断检索并生成查询、生成回答两个独立的 LLM 角色。"""

    route: LLMConfig
    answer: LLMConfig


@dataclass(frozen=True)
class RerankerConfig:
    """重排模型参数。"""

    endpoint: EndpointConfig
    model: str
    timeout: float


@dataclass(frozen=True)
class BM25Config:
    """BM25 索引与打分参数。"""

    stopwords: str
    method: str
    min_score: float


@dataclass(frozen=True)
class RetrievalConfig:
    """召回与输出条数，流水线和各命令行工具共用。"""

    bm25_candidates: int
    vector_candidates: int
    top_n: int


@dataclass(frozen=True)
class FusionConfig:
    """两路召回的融合参数。"""

    method: str
    rrf_k: int
    weights: tuple[float, ...]


@dataclass(frozen=True)
class Config:
    """配置文件的完整内容。"""

    paths: Paths
    endpoints: dict[str, EndpointConfig]
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMRoles
    reranker: RerankerConfig
    bm25: BM25Config
    retrieval: RetrievalConfig
    fusion: FusionConfig


def _env_value(raw: str) -> str:
    """解析 `.env` 的取值：去掉行尾注释，再去掉一层成对引号。

    - `KEY=value  # 注释` -> `value`
    - `KEY="a # b"` -> `a # b`（引号内的 # 不当注释）
    - `KEY='x'` -> `x`（只剥一层，不会误删值里的引号）
    """
    quote = ""
    for index, char in enumerate(raw):
        if quote:
            if char == quote:
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == "#" and (index == 0 or raw[index - 1].isspace()):
            raw = raw[:index]
            break
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def load_env(path: pathlib.Path | None = None) -> None:
    """把 .env 里的键值写进环境变量（不覆盖已存在的值），默认读工作目录。"""
    path = home() / ".env" if path is None else path
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), _env_value(value))


def _resolve(path: str) -> pathlib.Path:
    """把配置里的相对路径解析到工作目录下。"""
    return (home() / path).resolve()


def _endpoint(
    endpoints: dict[str, EndpointConfig], name: str, owner: str
) -> EndpointConfig:
    """按名字取 endpoint，缺失时指出是哪一个配置引用了它。"""
    try:
        return endpoints[name]
    except KeyError:
        raise FastragError(
            f"{owner}.endpoint = {name!r} 未在 [endpoints] 中定义"
        ) from None


def _llm(
    role: dict, endpoints: dict[str, EndpointConfig], name: str
) -> LLMConfig:
    """读取单个 LLM 角色，并解析它引用的 endpoint 与可选备用模型。"""
    retries = role.get("retries", 0)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise FastragError(
            f"llm.{name}.retries 必须是非负整数，当前为 {retries!r}"
        )
    fallback = role.get("fallback")
    return LLMConfig(
        endpoint=_endpoint(endpoints, role["endpoint"], f"llm.{name}"),
        model=role["model"],
        provider=role["provider"],
        temperature=role["temperature"],
        thinking=role.get("thinking", True),
        retries=retries,
        fallback=(
            _llm(fallback, endpoints, f"{name}.fallback") if fallback else None
        ),
    )


def _number(owner: str, value: object) -> float:
    """确认配置值是数字（排除 bool 与字符串），否则抛 FastragError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FastragError(f"{owner} 必须是数字，当前为 {value!r}")
    return value


def _validate_llm_role(role: dict, name: str) -> None:
    """校验一个 LLM 角色（含递归的 fallback）的温度等参数。"""
    temperature = _number(f"llm.{name}.temperature", role["temperature"])
    if not 0 <= temperature <= 2:
        raise FastragError(
            f"llm.{name}.temperature 必须在 0~2 之间，当前为 {temperature!r}"
        )
    fallback = role.get("fallback")
    if fallback:
        _validate_llm_role(fallback, f"{name}.fallback")


def _validate(
    chunking: dict,
    embedding: dict,
    bm25: dict,
    retrieval: dict,
    fusion: dict,
    llm: dict,
    reranker: dict,
) -> str:
    """集中校验数值、枚举与正则参数，返回 boilerplate_pattern。

    这里只做「进入运行期前就能发现」的检查，错误统一抛 FastragError：
    避免 chunk_overlap >= chunk_size、batch_size=0、rrf_k<=0 之类的配置
    在切块/嵌入/检索阶段才以晦涩的异常暴露。
    """
    if fusion["method"] not in FUSION_METHODS:
        raise FastragError(f"fusion.method 必须是 {' / '.join(FUSION_METHODS)}")
    raw_weights = fusion["weights"]
    if not isinstance(raw_weights, list):
        raise FastragError(f"fusion.weights 必须是数组，当前为 {raw_weights!r}")
    weights = [
        _number(f"fusion.weights[{index}]", weight)
        for index, weight in enumerate(raw_weights)
    ]
    if len(weights) != 2:
        raise FastragError("fusion.weights 必须包含两路权重（BM25、向量）")
    if any(weight <= 0 for weight in weights):
        raise FastragError(f"fusion.weights 必须全为正数，当前为 {weights}")
    rrf_k = _number("fusion.rrf_k", fusion["rrf_k"])
    if rrf_k <= 0:
        raise FastragError(f"fusion.rrf_k 必须大于 0，当前为 {rrf_k!r}")

    for key in ("bm25_candidates", "vector_candidates", "top_n"):
        value = _number(f"retrieval.{key}", retrieval[key])
        if value <= 0:
            raise FastragError(f"retrieval.{key} 必须大于 0，当前为 {value!r}")

    chunk_size = _number("chunking.chunk_size", chunking["chunk_size"])
    chunk_overlap = _number("chunking.chunk_overlap", chunking["chunk_overlap"])
    if chunk_size <= 0:
        raise FastragError(f"chunking.chunk_size 必须大于 0，当前为 {chunk_size!r}")
    if chunk_overlap < 0:
        raise FastragError(
            f"chunking.chunk_overlap 不能为负数，当前为 {chunk_overlap!r}"
        )
    if chunk_overlap >= chunk_size:
        raise FastragError(
            f"chunking.chunk_overlap ({chunk_overlap}) 必须小于 "
            f"chunk_size ({chunk_size})"
        )

    batch_size = _number("embedding.batch_size", embedding["batch_size"])
    if batch_size <= 0:
        raise FastragError(f"embedding.batch_size 必须大于 0，当前为 {batch_size!r}")
    dimensions = _number("embedding.dimensions", embedding["dimensions"])
    if dimensions <= 0:
        raise FastragError(f"embedding.dimensions 必须大于 0，当前为 {dimensions!r}")
    timeout = _number("embedding.timeout", embedding["timeout"])
    if timeout <= 0:
        raise FastragError(f"embedding.timeout 必须大于 0，当前为 {timeout!r}")
    max_retries = _number("embedding.max_retries", embedding["max_retries"])
    if max_retries < 0:
        raise FastragError(
            f"embedding.max_retries 不能为负数，当前为 {max_retries!r}"
        )

    _number("bm25.min_score", bm25["min_score"])
    if bm25["method"] not in BM25_METHODS:
        raise FastragError(
            f"bm25.method 必须是 {' / '.join(BM25_METHODS)} 之一，"
            f"当前为 {bm25['method']!r}"
        )
    stopwords = bm25["stopwords"]
    if isinstance(stopwords, list):
        if not all(isinstance(word, str) for word in stopwords):
            raise FastragError(
                f"bm25.stopwords 列表元素必须是字符串，当前为 {stopwords!r}"
            )
    elif not isinstance(stopwords, str) or stopwords not in BM25_STOPWORDS:
        raise FastragError(
            f"bm25.stopwords 必须是内置语言名或字符串列表，当前为 {stopwords!r}"
        )

    reranker_timeout = _number("reranker.timeout", reranker["timeout"])
    if reranker_timeout <= 0:
        raise FastragError(
            f"reranker.timeout 必须大于 0，当前为 {reranker_timeout!r}"
        )

    _validate_llm_role(llm["route"], "route")
    _validate_llm_role(llm["answer"], "answer")

    pattern = chunking.get("boilerplate_pattern", "")
    if pattern:
        if not isinstance(pattern, str):
            raise FastragError(
                f"chunking.boilerplate_pattern 必须是字符串，当前为 {pattern!r}"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise FastragError(
                f"chunking.boilerplate_pattern 不是合法正则: {exc}"
            ) from None
    return pattern


def load_config(path: pathlib.Path | None = None) -> Config:
    """读取并校验统一配置。

    解析顺序：显式 path > `FASTRAG_CONFIG` > `FASTRAG_HOME/config/fastrag.toml`
    > 包内默认配置（`fastrag.resources/config/fastrag.toml`）。
    """
    if path is None:
        override = os.environ.get("FASTRAG_CONFIG")
        if override:
            path = pathlib.Path(override).expanduser().resolve()
        else:
            path = default_config_path()
    if not path.exists():
        raise FastragError(f"配置文件不存在: {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = raw["paths"]
    endpoints = {
        name: EndpointConfig(
            base_url=item["base_url"],
            api_key_env=item["api_key_env"],
        )
        for name, item in raw["endpoints"].items()
    }
    chunking, embedding = raw["chunking"], raw["embedding"]
    llm, reranker = raw["llm"], raw["reranker"]
    bm25, retrieval, fusion = raw["bm25"], raw["retrieval"], raw["fusion"]

    pattern = _validate(
        chunking, embedding, bm25, retrieval, fusion, llm, reranker
    )

    return Config(
        paths=Paths(
            docs_dir=_resolve(paths["docs_dir"]),
            chunks_file=_resolve(paths["chunks_file"]),
            embeddings_dir=_resolve(paths["embeddings_dir"]),
            bm25_index_dir=_resolve(paths["bm25_index_dir"]),
        ),
        endpoints=endpoints,
        chunking=ChunkingConfig(
            tokenizer=chunking["tokenizer"],
            chunk_size=chunking["chunk_size"],
            chunk_overlap=chunking["chunk_overlap"],
            include_delim=chunking["include_delim"],
            boilerplate_pattern=pattern,
        ),
        embedding=EmbeddingConfig(
            endpoint=_endpoint(endpoints, embedding["endpoint"], "embedding"),
            model=embedding["model"],
            batch_size=embedding["batch_size"],
            dimensions=embedding["dimensions"],
            timeout=embedding["timeout"],
            max_retries=embedding["max_retries"],
        ),
        llm=LLMRoles(
            route=_llm(llm["route"], endpoints, "route"),
            answer=_llm(llm["answer"], endpoints, "answer"),
        ),
        reranker=RerankerConfig(
            endpoint=_endpoint(endpoints, reranker["endpoint"], "reranker"),
            model=reranker["model"],
            timeout=reranker["timeout"],
        ),
        bm25=BM25Config(
            stopwords=bm25["stopwords"],
            method=bm25["method"],
            min_score=bm25["min_score"],
        ),
        retrieval=RetrievalConfig(
            bm25_candidates=retrieval["bm25_candidates"],
            vector_candidates=retrieval["vector_candidates"],
            top_n=retrieval["top_n"],
        ),
        fusion=FusionConfig(
            method=fusion["method"],
            rrf_k=fusion["rrf_k"],
            weights=tuple(fusion["weights"]),
        ),
    )
