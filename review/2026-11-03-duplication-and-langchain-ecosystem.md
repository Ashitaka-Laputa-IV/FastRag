# FastRag 代码重复与 LangChain 生态重复实现审阅

- 审阅对象：`dev@90d3673` 的 `src/fastrag/`（42 个 Python 文件，约 2.6k 行）
- 审阅日期：2026-11-03
- 运行环境：CPython 3.13.18、`langchain-core 1.7.0`、`langchain 1.4.7`、`langgraph 1.2.17`、`bm25s 0.3.15`
- 审阅方法：全量静态阅读 + `jscpd` 重复检测 + 运行时 `inspect.signature/getsource` 核对 + LangChain v1 文档核对
- 未覆盖：未调用付费 LLM / 嵌入 / 重排接口，未重跑 `prepare`；因此本文只评价“实现方式与职责归属”，不评价检索质量

---

## 1. 总体结论

先给结论，避免把“重复造轮子”说成一句空话：

1. **机械重复率不高**。`jscpd` 在“40 token / 5 行”阈值下只测到 **0.42%**（1 处克隆）；放宽到“20 token / 3 行”也只有 **4.33%**（其中 TOML 占 2 处）。说明项目里没有大面积复制粘贴。
2. **真正的问题不是“复制了多少行”，而是“同一职责存在两套实现”**。集中在这几处：
   - 向量检索：自己写归一化、内积、`argsort`、id→行号映射，而 `langchain-core` 已有 `InMemoryVectorStore` / `VectorStore` / `VectorStoreRetriever`；
   - 提示词：自己写 `string.Template.safe_substitute`，而 `PromptTemplate` / `ChatPromptTemplate` 已是标准做法；
   - `.env`：自己写 28 行解析器，等价于 `python-dotenv` 一行 `load_dotenv()`；
   - HTTP 重试：自己写 `urllib + sleep`，而 `httpx`、`tenacity` 已经是当前依赖树里的传递依赖；
   - 原子写：同一个“临时文件 + `os.replace` + 失败清理”写了 **4 遍**；
   - 检索编排：`pipeline/nodes.py` 和 `cli/commands/rrf.py` 各拼一遍 `{"bm25": ..., "vector": ...}`。
3. **也有几处看起来像重复、其实不该替换**：
   - RRF 融合（v1 的 `EnsembleRetriever` 已不在 `langchain-core`，且引入 `langchain-classic` 不划算）；
   - `create_retrieval_chain` / `create_stuff_documents_chain`（v1 已移出主包，当前 LangGraph 才是官方推荐路径）；
   - `bm25s`（`langchain-community` 的 `BM25Retriever` 不持久化，反而更弱）；
   - `langchain-reranker`（本身就是作者维护的生态包，仓库侧只是薄适配）。

一句话：**这不是一个“重复造了很多轮子”的项目，而是一个边界意识不错、但在几处生态边界和内部边界上多写了一套实现的项目。** 可安全合并/删除的代码约 **120–150 行**（约占 `src/` 的 5%），其中 `.env`、原子写、提示词渲染的收益最直接。

### 1.1 结论速览

| # | 重复对象 | 仓库位置 | LangChain 生态现状 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | 向量检索 | `index/dense.py` | `core.vectorstores.InMemoryVectorStore` / `VectorStore` / `VectorStoreRetriever` | **复用算法与接口，保留 npy 持久化** |
| 2 | 提示词渲染 | `llm/prompts.py`、`pipeline/nodes.py` | `PromptTemplate` / `ChatPromptTemplate` | **建议替换** |
| 3 | `.env` 解析 | `config/loader.py` | `python-dotenv` / `pydantic-settings` | **建议替换** |
| 4 | 下载重试 | `ingest/download.py` | `httpx` + `tenacity`（已在依赖树） | **建议替换** |
| 5 | Retriever 接口 | `index/*.search`、`pipeline/nodes.py` | `core.retrievers.BaseRetriever` | **可选薄适配** |
| 6 | `Chunk` / `Document` | `domain.py`、`ingest/chunker.py` | `core.documents.Document` | **保留领域类型 + 加适配器** |
| 7 | RRF | `retrieve/fusion.py` | v1 core 无 `EnsembleRetriever` | **保留** |
| 8 | RAG 链 | `pipeline/nodes.py` | v1 已移除 legacy chains | **保留 LangGraph，正确选择** |
| 9 | BM25 | `index/bm25.py` + `bm25s` | `langchain-community.BM25Retriever` | **保留，当前更合适** |
| 10 | 重排 | `retrieve/rerank.py` + `langchain-reranker` | `BaseDocumentCompressor` | **保留** |
| 11 | 配置 schema | `config/models.py` | 无对应物（pydantic 已足够） | **保留** |
| 12 | CLI 全局参数 | `cli/main.py` | argparse 自身 | **低优先级，可不动** |

---

## 2. 重复率数据

### 2.1 工具结果

对 `src/fastrag` 运行 `jscpd`（Python / TOML / TXT）：

| 阈值 | 克隆数 | 重复行 | 总行 | 重复率 | 重复 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 token / 5 行 | 1 | 12 | 2840 | **0.42%** | 0.66% |
| 20 token / 3 行 | 21 | 124 | 2864 | **4.33%** | 4.15% |

说明：

- 第一组阈值接近“有意义的重复代码块”，结果只有 `cli/commands/rerank.py` 与 `rrf.py` 的 12 行参数/准备逻辑；
- 第二组阈值会把 argparse 样板、TOML 中重复的 `[llm.route]` / `[llm.answer]` 也算进来，因此 4.33% 是“上限口径”；只看 Python 文件分别是 **0.46%** 与 **4.07%**；
- `jscpd` 只能看**字面克隆**，看不到“同一个职责、两套命名”的语义重复。本项目的主要问题恰好属于后者，所以下面的清单比百分比更重要。

### 2.2 内部重复清单

| # | 重复模式 | 位置 | 规模 | 处理建议 |
| --- | --- | --- | ---: | --- |
| D1 | 原子写：`mkstemp` → 写入 → `os.replace` → 失败 `unlink` | `ingest/artifacts.py:57-68,83-94`、`ingest/chunker.py:94-106`、`ingest/download.py:107-116` | 4 份 / ~60 行 | 抽一个 `atomic_write()` 上下文管理器 |
| D2 | 提示词渲染 | `llm/prompts.py:35-37`、`pipeline/nodes.py:39-40` | 2 份 / 2 行 | 只保留一个 `render_template(text, **values)` |
| D3 | 提示词名单 | `llm/prompts.py:18`（`REQUIRED`）、`pipeline/resources.py:25`（`PROMPT_NAMES`） | 2 份 / 1 行 | 单一来源；`REQUIRED` 目前完全未使用 |
| D4 | 检索编排：加载 store / embeddings / vectors / bm25 → 两路 search → rankings | `pipeline/nodes.py:89-138`、`cli/commands/rrf.py:33-50` | 2 份 / ~40 行 | 抽 `retrieve_all()` 或 `Retriever` 协议 |
| D5 | manifest 参数知识：写 key 时列一遍、复用校验时再列一遍 | `index/dense.py:79-88` 与 `91-115` | 2 份 / ~10 行 | 参数 dict 只构造一次 |
| D6 | fallback 链遍历：`route` 与 `answer` 各写一遍 | `config/models.py:207-218` | 2 份 / ~10 行 | 抽 `_iter_roles()` |
| D7 | 提示词读取两次：`prompts` 与 `prompt_hashes` 各 `load_prompt` 一次 | `pipeline/resources.py:72-78`、`llm/prompts.py:40-42` | 每次提问 6 次读文件 | 哈希从已加载文本算 |
| D8 | CLI 子命令样板：`add_parser` + `set_defaults` | `cli/commands/*.py`（9 个文件） | ~45 行 | 可选；argparse 风格如此，不强求 |
| D9 | 去重逻辑：`_merge` 按 id 去重、`rrf_fuse` 按 id 累加 | `pipeline/nodes.py:180-191`、`retrieve/fusion.py:26-35` | 2 处 / ~15 行 | 抽 `dedupe_by_id()`；顺带修掉“同一通道重复 id 会重复计分”的隐患 |
| D10 | 死代码 | `errors.py:19`（`run_cli`）、`llm/prompts.py:18`（`REQUIRED`）、`index/bm25.py:128`（`index_path`）、`index/dense.py:213`（`vectors_path`） | ~15 行 | 删除或用起来 |

> 注：`REQUIRED`、`run_cli`、`index_path`、`vectors_path` 在 `src/` 内只有定义、没有调用；其中 `run_cli` 的 docstring 还写着“兼容旧入口”，属于历史残留。

---

## 3. LangChain 生态逐项比对

本节只讨论“生态里有没有现成的”，以及“该不该换”。判断标准是：**换掉后代码更少、行为更正确、依赖不增加或增加可控**，而不是“框架里有就必须用”。

### 3.1 向量检索：最值得复用的一块

**现状**（`index/dense.py`）：

- `_normalize` 自己算 L2 归一化（74-76）；
- `search_dense` 自己算内积、`np.argsort(-scores)`、按行号回表（184-210）；
- `_load_previous` 自己按 id 复用旧向量（91-115）；
- `ensure_vectors` 自己维护 `vectors.npy` + `manifest.json`（118-181）。

**生态现状**（已核对 `langchain-core 1.7.0`）：

- `langchain_core.vectorstores.InMemoryVectorStore`：
  - 使用 cosine similarity（源码里调用 `cosine_similarity`），支持 `filter`；
  - 支持 `add_texts(ids=...)`、`delete()`、`get_by_ids()`；
  - 支持 `max_marginal_relevance_search()`，可直接用 `langchain_core.vectorstores.utils.maximal_marginal_relevance`；
  - 有 `dump()` / `load()`，但**是 JSON**，不适合 2.6k × 2560 的 float32 矩阵；
- `VectorStore` / `VectorStoreRetriever`：提供 `similarity_search_with_score`、`as_retriever()`、`invoke/batch/astream`；
- `BaseRetriever`：只要求实现 `_get_relevant_documents`。

**判断**：**部分复用，不要整体替换。**

- 不值得替换：`vectors.npy` 的持久化、按 chunk id 增量嵌入、`ArtifactKey` 指纹校验。这些是本项目相对通用 vector store 的**真实增量**，`InMemoryVectorStore.dump/load` 换成 JSON 反而会让 26MB 矩阵体积膨胀、加载变慢；
- 值得替换：
  1. 用 `maximal_marginal_relevance` 直接获得 MMR（上一轮审阅提过“相邻块挤占 top-k”，这是低成本解法）；
  2. 把 `search_dense` 的返回改成 `VectorStore` 或 `BaseRetriever` 适配层，至少让 `similarity_search_with_score` / `as_retriever` 成为统一入口；
  3. 去掉重复的归一化/内积/排序手写逻辑，保留 npy 加载与 manifest。

**建议的落点**（不必一次做完）：

```python
# index/dense.py：保留矩阵与 manifest，只把“检索语义”接到 ecosystem 接口
from langchain_core.vectorstores.utils import maximal_marginal_relevance

def search_dense_mmr(..., k: int, fetch_k: int, lambda_mult: float) -> list[ScoredChunk]:
    query = _normalized_query(embeddings, text)
    scores = vectors @ query
    candidates = np.argsort(-scores)[:fetch_k]
    chosen = maximal_marginal_relevance(
        query, [vectors[i] for i in candidates], k=k, lambda_mult=lambda_mult
    )
    return [
        ScoredChunk(chunk=store.chunks[int(candidates[i])], score=float(scores[candidates[i]]),
                    channel="vector", rank=rank)
        for rank, i in enumerate(chosen, start=1)
    ]
```

如果后续要做成 `BaseRetriever`，建议只在 `index/` 边界把 `ScoredChunk` 转成 `Document`，不要让 `Document` 穿过 `pipeline` 与 `domain`：

```python
class DenseRetriever(BaseRetriever):
    store: ChunkStore
    vectors: np.ndarray
    embeddings: OpenAIEmbeddings
    k: int

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return [item.chunk.to_document() for item in search_dense(...)]
```

**风险**：`InMemoryVectorStore` 每次查询都会重算 cosine，和当前“预归一化 + 内积”相比略慢；2.6k 条规模可以忽略，但如果语料上到十万级，应保留预归一化矩阵。

另外，`langchain_core.indexing.index()` + `RecordManager` 也覆盖“增量索引/删除/跳过未变文档”，概念上与 `ensure_vectors` 的 id 复用重叠；但 core 只提供 `InMemoryRecordManager`（docstring 明确写的是 testing purposes），没有持久化实现，而本项目的“内容寻址 id + 全字节 sha256 + manifest”更严格，因此**不建议替换**这套增量逻辑。

---

### 3.2 提示词：建议换成 `PromptTemplate` / `ChatPromptTemplate`

**现状**（`llm/prompts.py:35-37`、`pipeline/nodes.py:39-40`）：

```python
Template(prompt).safe_substitute(values)
```

问题有三层：

1. **重复**：`nodes._render` 与 `prompts.render` 是同一行实现，`nodes` 却绕过了 `llm.prompts`；
2. **静默失败**：`safe_substitute` 对未知占位符原样保留。`$questoin` 这类拼写错误不会报错，而是把字面量 `$questoin` 发给模型。docstring 说“避免静默丢内容”，但实际制造了另一种静默错误；
3. **能力缺失**：三份提示词都是单条 user 文本，没有 system / human 角色区分；对 `thinking` 模型、后续多轮和 few-shot 都不友好。

**生态现状**：

- `PromptTemplate`：支持 `f-string` / `mustache` / `jinja2`，`format()` 缺变量直接 `KeyError`（已实测）；
- `ChatPromptTemplate.from_messages([...])`：system / human / placeholder 角色化；
- `FewShotPromptTemplate`、`MessagesPlaceholder` 等按需可用。

**判断**：**建议替换**，但分两步，避免一次改动过大：

1. 先把 `safe_substitute` 改成 `Template(...).substitute(values)`，让缺变量立即报错；同时删除 `nodes._render`，统一走 `llm.prompts`；
2. 再把 3 份 `.txt` 的 `$context` / `$question` 改成 `{context}` / `{question}`，用 `ChatPromptTemplate` 承载 system + human：

```python
# llm/prompts.py
from langchain_core.prompts import ChatPromptTemplate

def answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "你是文档问答助手。只依据给定资料回答；资料不足时明确说明。"),
        ("human", load_prompt("answer")),  # 模板里用 {context} / {question}
    ])
```

注意：如果采用 system 消息，应把 `answer.txt` 第一行的同类指令删掉或下移，避免 system 与 human 里出现两遍“资料不足不要编造”。

`plan_chain` 也可以复用同一套提示词对象：

```python
prompt = ChatPromptTemplate.from_messages([("human", load_prompt("search_plan"))])
plan_chain = build_with_fallbacks(
    lambda role: prompt | build_llm(role, config).with_structured_output(SearchPlan),
    config.llm.route,
)
```

**注意**：`PromptTemplate` 默认是 f-string 语法，当前模板是 `$name`。要么改模板文件（3 个文件、4 个占位符，成本很低），要么显式用 `template_format="mustache"`，但 mustache 对未知变量同样是原样保留，无法解决第 2 点。

---

### 3.3 `.env`：28 行解析器换成 `python-dotenv`

**现状**（`config/loader.py:95-123`）：

- `_env_value` 自己处理行尾注释、成对引号；
- `load_env` 自己按 `=` 切分、`os.environ.setdefault`。

**生态现状**：

- `python-dotenv`：支持 `export KEY=...`、引号、注释、多行值、变量插值、`override` 语义；
- `pydantic-settings`：`SettingsConfigDict(env_file=...)`，与现有 pydantic 同族；
- 两者当前都**未安装**，但都是标准、轻量的依赖。

**判断**：**建议替换**。这是投入产出比最高的一处：删掉 28 行手写解析，换来完整的 `.env` 语义。

**当前实现的具体缺口**：

| 输入 | 当前行为 | 期望 |
| --- | --- | --- |
| `export FOO=1` | key 变成 `export FOO`，写入一个错误变量 | 识别 `export` 前缀 |
| `FOO=` | 写入空字符串 | 一致，保留 |
| `FOO`（无 `=`） | 静默跳过 | dotenv 同样跳过，可接受 |
| 多行值 `FOO="a\nb"` | 按行切分，第二行被丢弃 | 支持引号内换行 |
| `FOO=${BAR}` | 原样保留 | dotenv 支持插值（可选） |

**建议**：

```python
from dotenv import load_dotenv

def load_env(path: pathlib.Path | None = None) -> None:
    """把 .env 写进环境变量；不覆盖已存在的值。"""
    load_dotenv(home() / ".env" if path is None else path, override=False)
```

如果不想加依赖，至少补上 `export ` 前缀与“空值”测试；但 28 行解析器长期看不如直接用标准实现。

---

### 3.4 下载重试：`urllib + sleep` 换成 `httpx + tenacity`

**现状**（`ingest/download.py:44-64`）：

```python
for attempt in range(attempts):
    try: ...
    except HTTPError as exc:
        if exc.code not in RETRY_STATUS: raise
    ...
    time.sleep(1 + attempt)
```

**生态现状**：

- `httpx 0.28.1` 已是 `langchain-core` 的传递依赖；
- `tenacity 9.2.1` 也已在依赖树里（`langchain-core` 直接依赖）；
- 用 `httpx.Client(follow_redirects=True)` + `tenacity.retry` 可以把重试策略、退避、异常分类写成声明式。

**判断**：**建议替换**，保留现有业务逻辑（URL 白名单、`_safe_destination`、HTML 检测、原子写）。

```python
import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))

@retry(retry=retry_if_exception(_retryable), stop=stop_after_attempt(...),
       wait=wait_exponential(multiplier=0.5, max=8), reraise=True)
def fetch(url: str, client: httpx.Client) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content
```

**收益**：少 15–20 行手写循环，退避带指数与上限，异常分类更清楚；**风险**：`httpx` 的异常层次与 `urllib` 不同，`404` 必须显式不重试，测试要覆盖。

---

### 3.5 Retriever 接口：可选，但能一次解决两处编排重复

**现状**：

- `index/bm25.py::search`、`index/dense.py::search_dense` 都是普通函数；
- `pipeline/nodes.py` 的 `retrieve_bm25` / `retrieve_vector` 各包一层；
- `cli/commands/rrf.py` 又手工拼一遍 `{"bm25": ..., "vector": ...}`。

**生态现状**：

- `BaseRetriever` 只要求 `_get_relevant_documents`，自带 `invoke/batch/astream`；
- `RunnableParallel` 可以并行组合多个 retriever；
- 但 **`EnsembleRetriever` 不在 v1 的 `langchain-core`**（`langchain.retrievers` 已不存在，`langchain-classic` 未安装）。

**判断**：**内部抽 `Protocol` 即可，是否实现 `BaseRetriever` 取决于是否需要异步/批处理。**

如果只为本项目服务，最小改动是：

```python
class Retriever(Protocol):
    channel: str
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...

@dataclass
class HybridRetriever:
    bm25: Retriever
    vector: Retriever

    def retrieve_all(self, bm25_query: str, vector_query: str) -> dict[str, list[ScoredChunk]]:
        return {
            "bm25": self.bm25.retrieve(bm25_query, ...),
            "vector": self.vector.retrieve(vector_query, ...),
        }
```

这样 `nodes.fuse` 与 `cli/rrf.py` 都只依赖 `retrieve_all()`，不用再各自知道“有哪些通道、候选数取哪个配置”。

**不建议**为了 `EnsembleRetriever` 引入 `langchain-classic`：它会把 v1 已经拆走的 legacy chains 再装回来，依赖面和维护面都变大。

---

### 3.6 `Chunk` / `Document`：保留领域类型，补适配器

**现状**：

- `domain.Chunk` 是 `id/source/start/end/text/token_count` 的 frozen dataclass；
- `rerank.py` 手工构造 `Document(page_content=..., metadata={"fastrag_index": ...})`，再按 index 回表；
- `_context` 手工拼 `[i] source\ntext`。

**生态现状**：`Document` 是 `page_content + metadata + id`，本身没有 `start/end/token_count` 这些字段；`Document` 的 `id` 可以承载 chunk id，其余放 metadata。

**判断**：**保留 `Chunk` 是合理的**。`Chunk` 明确表达“原文偏移 + token 数”这个领域约束，比把一切塞进 `metadata: dict` 更安全；而且 README 的“检索层不传裸 dict / Document”是清晰的设计取舍。

**可改进**：加一对适配器，消灭 `rerank.py` 里的手工映射和 `_context` 的重复格式化：

```python
# domain.py
def to_document(self) -> Document:
    return Document(
        id=self.id,
        page_content=self.text,
        metadata={"source": self.source, "start": self.start, "end": self.end,
                  "token_count": self.token_count},
    )
```

`rerank.py` 仍需要“按原候选位置回表”，因为 `BaseDocumentCompressor` 只返回排序后的 `Document`；用 `Document.id` 做映射比 `metadata["fastrag_index"]` 更稳，也不依赖 int 类型校验。

---

### 3.7 RRF 融合：保留，不要换

`retrieve/fusion.py` 只有 25 行左右的纯函数，公式正确、可测试。`langchain-core 1.7.0` 没有 `EnsembleRetriever`；v1 的 legacy 版本在 `langchain-classic` 里，当前项目没装。

**判断**：**保留**。这不算重复造轮子，属于“框架已拆走、项目自留的合理原语”。

唯一值得加的是 MMR / 去重（见 D9、3.1），以及把 `channel="rrf"` 改成保留来源通道信息，便于 CLI 复盘。

---

### 3.8 RAG 链：保留 LangGraph，这是正确选择

v1 的 `langchain` 主包已经不再导出 `create_retrieval_chain` / `create_stuff_documents_chain`（实测 `No module named 'langchain.chains'`），这些在 `langchain-classic`。当前项目用 `StateGraph` 编排 `plan_search -> retrieve_* -> fuse -> answer`，与 v1 官方推荐路径一致。

**判断**：**保留，不要为了“少写代码”换回 legacy chain。** 手写的 `_context` 只有 7 行，换成 `create_stuff_documents_chain` 需要额外依赖，且会失去现有 `SearchPlan` 分支与降级逻辑。

---

### 3.9 BM25：保留 `bm25s`

`langchain-community` 的 `BM25Retriever`：

- 需要额外安装 `langchain-community`；
- 不提供本项目需要的持久化 + 指纹校验；
- 属于 v1 的 legacy 生态。

当前 `bm25s` + `ArtifactKey` + `ChunkStore` 回表的组合更贴合“`chunks.jsonl` 是唯一真源”的设计。

**判断**：**保留**。

---

### 3.10 重排：`langchain-reranker` 是作者自己的生态包

`pyproject.toml` 依赖 `langchain-reranker>=0.1.4`，其 `init_reranker("custom:...")` 返回 `BaseDocumentCompressor`，支持任意 Cohere 风格 `/rerank` 端点。查看包元数据，作者是 `Ashitaka-Laputa-IV`，与本仓库同一作者。

**判断**：**保留**。这不是“仓库重复造轮子”，而是“把轮子拆成独立包”。仓库侧 `rerank.py` 只做 `ScoredChunk <-> Document` 适配，职责正确。

**注意**：`langchain-reranker` 依赖 `langchain-core>=1.0`，但没有上限；`BaseDocumentCompressor` 属于 `langchain_core.documents.compressor`，升级时需要跟 `langchain-core` 一起验证。建议在 `pyproject.toml` 里给主依赖加上限（例如 `langchain-core<2`），或在 CI 里跑一次 `init_reranker("custom:...")` 的构造测试。

---

### 3.11 配置与 CLI：不属于 LangChain 的职责

- `config/models.py` 用 pydantic + `extra="forbid"` + `model_validator`，已经比手写 dict 校验好很多；`pydantic-settings` 只在“想顺带解决 `.env`”时才有价值，不必为了“用生态”而迁移。
- `cli/main._extract_globals` 手写了“全局参数可出现在子命令前后”的解析（28 行）。argparse 的 `parents=` + 两次 `parse_known_args` 也能做，但当前实现可读、可测，**低优先级**，不建议为了减少 20 行而引入更绕的 argparse 技巧。

---

## 4. 内部重复的具体改法

以下改动都不依赖新框架，纯粹是“同一份逻辑只留一处”。

### 4.1 D1：原子写统一成 `atomic_write()`

当前 4 处实现几乎是同一段代码：

- `artifacts.write_json_atomic`（57-68）
- `artifacts.write_npy_atomic`（83-94）
- `chunker.write_chunks`（94-106）
- `download.download` 尾部（107-116）

建议：

```python
from contextlib import contextmanager

@contextmanager
def atomic_path(path: pathlib.Path, suffix: str = "") -> Iterator[BinaryIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            yield fh
        os.replace(temp, path)
    except BaseException:
        pathlib.Path(temp).unlink(missing_ok=True)
        raise
```

调用侧：

```python
with atomic_path(directory / "manifest.json") as fh:
    fh.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
    fh.write(b"\n")

with atomic_path(directory / "vectors.npy", suffix=".npy") as fh:
    np.save(fh, matrix)
```

`download.download` 只需把 `fh.write(content)` 放进同一个上下文。预计净减 **35–40 行**，且以后新增产物不会再复制第五份。

### 4.2 D2 + D3：提示词渲染与名单单一来源

- 删除 `nodes._render`，改为 `from fastrag.llm.prompts import render_template`；
- 在 `llm/prompts.py` 里保留唯一的 `PROMPT_NAMES`（或 `REQUIRED`），`resources.py` 直接 import；
- 用 `substitute` 替代 `safe_substitute`，让拼写错误在开发期暴露。

```python
# llm/prompts.py
PROMPT_NAMES = ("search_plan", "answer", "chat")

def render_template(template: str, **values: str) -> str:
    return Template(template).substitute(values)  # 缺键 -> KeyError
```

> 如果确实希望“未知占位符原样保留”，至少记录一行 `logger.warning`，不要让模型悄悄收到 `$questoin`。

### 4.3 D4：检索编排只留一处

把 `nodes.retrieve_bm25` / `nodes.retrieve_vector` / `nodes.fuse` 的前半段，与 `cli/rrf.py` 的 33-50 行合并成 `HybridRetriever`（见 3.5）。`nodes.fuse` 只负责“选 RRF 还是 rerank”，CLI 只负责“打印”，两边都不再知道候选数来自哪个配置项。

### 4.4 D5：manifest 参数只构造一次

```python
def _embedding_params(config: Config) -> dict:
    return {
        "model": config.embedding.model,
        "dimensions": config.embedding.dimensions,
        "normalized": True,
    }
```

`_manifest_key` 与 `_load_previous` 都调用它。顺手删掉 `_load_previous(config, store)` 里未使用的 `store` 参数。

### 4.5 D6：fallback 链遍历只写一次

`config/models.py` 的 `_check_endpoints` 对 `route` 与 `answer` 各走一遍 while 循环。抽：

```python
def _iter_roles(role: LLMConfig | None, prefix: str):
    while role is not None:
        yield prefix, role
        role = role.fallback
        prefix = f"{prefix}.fallback"
```

### 4.6 D7：提示词只读一次

`resources.prompts` 已经缓存了文本；`prompt_hashes` 不必再 `load_prompt` 一遍：

```python
@cached_property
def prompt_hashes(self) -> dict[str, str]:
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        for name, text in self.prompts.items()
    }
```

每次提问少 3 次文件读取（当前 `prompts` + `prompt_hashes` 合计 6 次，每次 ask 都发生）。

### 4.7 D10：删除死代码

| 符号 | 位置 | 处理 |
| --- | --- | --- |
| `run_cli` | `errors.py:19-24` | 删除；`cli/main.py` 已自己捕获 `FastragError` |
| `REQUIRED` | `llm/prompts.py:18` | 与 `PROMPT_NAMES` 合并后删除 |
| `index_path` | `index/bm25.py:128` | 删除或让 CLI 使用 |
| `vectors_path` | `index/dense.py:213` | 删除或让 CLI 使用 |

---

## 5. 建议的改动顺序

### P0：半天内可完成，风险低

1. **D1 原子写**：4 → 1，净减约 35–40 行；
2. **D2/D3/D7**：提示词渲染与名单单一来源，`substitute` 替代 `safe_substitute`；
3. **D10**：删 4 个死符号；
4. **D6**：fallback 链遍历抽函数。

预期：`jscpd` 的 4.33% 口径会降到 3% 左右，且不再有“改一处忘另一处”的入口。

### P1：1–2 天，行为有可见改善

5. **`.env` → `python-dotenv`**：删 28 行，补 `export`/空值/多行测试；
6. **提示词 → `ChatPromptTemplate`**：system / human 分离，模板从 `$x` 改 `{x}`；
7. **向量检索 → 复用 `maximal_marginal_relevance`**：解决相邻块挤占 top-k，约 20 行；
8. **下载重试 → `tenacity` + `httpx`**：退避与异常分类更标准。

### P2：按需，不建议现在做

9. `BaseRetriever` / `VectorStore` 适配层：只有出现异步、批处理、第三方 retriever 混用时才值得；
10. `HybridRetriever` 统一编排：如果 CLI 的 `rrf` 只是调试工具，可以先保留现状；
11. 配置迁移到 `pydantic-settings`：只有确实需要环境变量覆盖嵌套字段时再做。

---

## 6. 明确不建议做的事

| 反模式 | 原因 |
| --- | --- |
| 为了 `EnsembleRetriever` 引入 `langchain-classic` | v1 已把 legacy 链拆走；引入后依赖面变大，收益只是省 25 行 RRF |
| 把 LangGraph 换成 `create_retrieval_chain` | v1 主包已移除；会丢失现有 `SearchPlan` 分支与降级逻辑，且需要额外依赖 |
| 用 `InMemoryVectorStore.dump/load` 替代 `vectors.npy` | dump 是 JSON，2.6k × 2560 float32 会显著膨胀；npy 更适合数值矩阵 |
| 用 `langchain-community.BM25Retriever` 替代 `bm25s` | 不持久化、无指纹校验，且属于 legacy 生态 |
| 强制 `Document` 贯穿 `domain/pipeline` | 会破坏“检索层只交换 `ScoredChunk`”的清晰边界，收益只是少一个 dataclass |
| 用 `StrOutputParser` 替代 `extract_text` | `StrOutputParser` 对空字符串不报错，而“空回答必须触发 fallback”是本项目有意为之的语义；保留自定义异常更正确 |
| 用 `WebBaseLoader` / `RecursiveUrlLoader` 替代自定义下载器 | 当前直接抓官方 `.md` 源码；loader 走 HTML 再抽文本，会引入 `bs4` 且丢掉原文偏移 |
| 用 `RecursiveCharacterTextSplitter` 替代 `chonkie` | 现有切块按 token + 句子边界，且保留 `text == original[start:end]` 偏移；通用 splitter 反而更难保证 |
| 为了“用生态”把固定 RAG 流程改成 agent | 本任务是确定的 plan → retrieve → answer，agent 只会增加不确定性与成本 |

---

## 7. 验证建议

改动前后建议补 3 个很小的测试（项目目前无测试，见 `AGENTS.md`）：

1. **原子写**：写入过程中抛异常，断言目标文件不存在、临时文件被清理；
2. **提示词缺变量**：`render_template("$a", b="x")` 必须抛 `KeyError`，而不是把 `$a` 发给模型；
3. **向量检索对照**：同一批向量分别走 `search_dense` 与 `InMemoryVectorStore.similarity_search_with_score`，断言 top-k 的 chunk id 与分数在 1e-5 内一致（用来证明“换实现不改行为”）。

另外建议加一条 CI 检查：

```bash
npx jscpd src/fastrag --min-tokens 20 --min-lines 3 --threshold 5
```

阈值设为 5%，防止内部复制重新增长；同时保留人工审阅语义重复（工具测不到）。

---

## 8. 附：核查记录

### 8.1 版本与可用性

| 包 | 版本 | 与本审阅相关 |
| --- | --- | --- |
| `langchain-core` | 1.7.0 | `InMemoryVectorStore`、`VectorStore`、`BaseRetriever`、`PromptTemplate`、`ChatPromptTemplate`、`maximal_marginal_relevance` 均存在 |
| `langchain` | 1.4.7 | 仅 `agents/chat_models/embeddings/mcp/messages/rate_limiters/tools`；无 `langchain.chains` / `langchain.retrievers` |
| `langgraph` | 1.2.17 | 当前图编排依赖 |
| `bm25s` | 0.3.15 | BM25 实现 |
| `langchain-reranker` | 0.1.4 | 作者自维护的重排包 |
| `tenacity` | 9.2.1 | 已在依赖树，可直接用于重试 |
| `httpx` | 0.28.1 | 已在依赖树，可替代 `urllib` |
| `python-dotenv` | 未安装 | 需新增，或用 `pydantic-settings` |
| `pydantic-settings` | 未安装 | 可选 |
| `langchain-community` | 未安装 | 不需要 |
| `langchain-classic` | 未安装 | 不建议引入 |
| `langchain-text-splitters` | 未安装 | 当前用 `chonkie`，无需替换 |

### 8.2 关键 API 实测

- `PromptTemplate.from_template("hello {name}")` 缺 `name` 时抛 `KeyError`；`template_format="mustache"` 对未知变量原样保留（与 `safe_substitute` 类似）；
- `InMemoryVectorStore._similarity_search_with_score_by_vector` 使用 cosine similarity，并支持 `filter`；
- `InMemoryVectorStore` 有 `dump/load`，但 dump 为 JSON；
- `maximal_marginal_relevance(query_embedding, embedding_list, lambda_mult, k)` 是 `langchain_core.vectorstores.utils` 的公开函数；
- `BaseRetriever` 的抽象方法只有 `_get_relevant_documents`；
- `VectorStore` 的抽象方法是 `from_texts` 与 `similarity_search`；`add_texts` 未覆盖时会走 `add_documents` 或抛 `NotImplementedError`，`delete` / `get_by_ids` 有默认实现。

### 8.3 行号索引

| 主题 | 位置 |
| --- | --- |
| 向量归一化 / 检索 | `index/dense.py:74-76,184-210` |
| 向量增量复用 / manifest | `index/dense.py:79-115,118-181` |
| 原子写 | `ingest/artifacts.py:57-94`、`ingest/chunker.py:94-106`、`ingest/download.py:107-116` |
| `.env` | `config/loader.py:95-123` |
| 提示词 | `llm/prompts.py:18-42`、`pipeline/nodes.py:39-40` |
| 资源缓存 | `pipeline/resources.py:25,72-78` |
| 检索编排 | `pipeline/nodes.py:89-138`、`cli/commands/rrf.py:33-50` |
| 重排适配 | `retrieve/rerank.py:29-59` |
| fallback | `llm/client.py:53-64`、`config/models.py:207-218` |
| BM25 | `index/bm25.py:35-125` |

---

## 9. 最终判断

- **值得马上做**：D1（原子写）、D2/D3/D7（提示词单一来源 + `substitute`）、D10（死代码）、`.env` 替换。这几项不改变检索行为，能直接减少约 **80–100 行**并把“两套实现”收敛成一套。
- **值得排期**：`ChatPromptTemplate`、MMR、`tenacity`。这三项会改变少量行为（消息角色、结果多样性、重试节奏），需要测试兜底。
- **明确不做**：引入 `langchain-classic`、替换 LangGraph、替换 `bm25s`、强制 `Document` 贯穿全项目。现有的这些选择是对的，不要为了“看起来更像 LangChain 项目”而改。

总体评价：**代码重复率不高（字面克隆 0.42%–4.33%），主要问题是“生态已有能力被局部重写”和“同一职责的多处内部复制”。按 P0/P1 做完后，项目会更短、更一致，但架构方向不需要推翻。**
