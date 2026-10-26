"""统一读取 config/fastrag.toml，并加载 .env 里的 API key。

所有模块共用这一份配置，不再各自维护 TOML 文件与加载逻辑：

- 路径统一解析到仓库根目录
- [endpoints.*] 描述厂家（base_url + key 环境变量），各模型按需引用
- 嵌入、重排、路由、问题重写、生成可以分别使用不同的厂家与模型
- 召回条数（BM25 / 向量分开配置）与输出条数（top_n）只保留一份
"""

import os
import pathlib
import tomllib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "fastrag.toml"
ENV_PATH = ROOT / ".env"
FUSION_METHODS = ("rrf", "rerank")


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
    """单个 LLM 角色的模型参数。"""

    endpoint: EndpointConfig
    model: str
    provider: str
    temperature: float


@dataclass(frozen=True)
class LLMRoles:
    """路由、问题重写、生成三个独立的 LLM 角色。"""

    route: LLMConfig
    requery: LLMConfig
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
    """config/fastrag.toml 的完整内容。"""

    paths: Paths
    endpoints: dict[str, EndpointConfig]
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMRoles
    reranker: RerankerConfig
    bm25: BM25Config
    retrieval: RetrievalConfig
    fusion: FusionConfig


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


def _resolve(path: str) -> pathlib.Path:
    """把配置里的相对路径解析到仓库根目录下。"""
    return (ROOT / path).resolve()


def _endpoint(
    endpoints: dict[str, EndpointConfig], name: str, owner: str
) -> EndpointConfig:
    """按名字取 endpoint，缺失时指出是哪一个配置引用了它。"""
    try:
        return endpoints[name]
    except KeyError:
        raise SystemExit(f"{owner}.endpoint = {name!r} 未在 [endpoints] 中定义") from None


def _llm(
    roles: dict, endpoints: dict[str, EndpointConfig], name: str
) -> LLMConfig:
    """读取一个 LLM 角色，并解析它引用的 endpoint。"""
    role = roles[name]
    return LLMConfig(
        endpoint=_endpoint(endpoints, role["endpoint"], f"llm.{name}"),
        model=role["model"],
        provider=role["provider"],
        temperature=role["temperature"],
    )


def load_config(path: pathlib.Path = CONFIG_PATH) -> Config:
    """读取并校验统一配置。"""
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

    if fusion["method"] not in FUSION_METHODS:
        raise SystemExit(f"fusion.method 必须是 {' / '.join(FUSION_METHODS)}")
    if len(fusion["weights"]) != 2:
        raise SystemExit("fusion.weights 必须包含两路权重（BM25、向量）")
    for key in ("bm25_candidates", "vector_candidates", "top_n"):
        if retrieval[key] < 1:
            raise SystemExit(f"retrieval.{key} 必须大于 0")

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
            route=_llm(llm, endpoints, "route"),
            requery=_llm(llm, endpoints, "requery"),
            answer=_llm(llm, endpoints, "answer"),
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
