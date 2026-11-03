# FastRag 第三方库替代审阅

- 前置阅读：
  - `review/2026-11-03-duplication-and-langchain-ecosystem.md`（重复率与 LangChain 生态比对）
  - `review/2026-11-03-minimal-code-plan.md`（最小代码量改造）
- 本文目标：找出**可以交给第三方库**的部分，同时判断它是否真的降低“管理复杂度”，而不是把一个 bug 换成一次依赖升级
- 现状依赖：9 个运行时依赖；`42` 个 Python 文件、`2662` 行（`dev@90d3673`）
- 建议结果：**只新增 1 个运行时依赖**（`python-dotenv`），**移除 1–3 个**（视功能取舍），**明确不引入 5 类重框架**

---

## 1. 结论速览

### 1.1 推荐做

| 替换 | 现在 | 换成 | 新增依赖 | 代码收益 |
| --- | --- | --- | --- | ---: |
| `.env` 解析 | 手写 29 行 | `python-dotenv` | +1（轻量） | **-28 行** |
| 下载重试 | 手写 `for + sleep` | `tenacity` | 0（已在树里） | -12 行 + 更稳的退避 |
| HTTP 请求 | `urllib` | `httpx` | 0（已在树里） | -10 行 + 连接池/超时 |
| 提示词渲染 | `string.Template` | `PromptTemplate` / `ChatPromptTemplate` | 0（core 自带） | -15 行 + 缺变量报错 |
| 模型构建 | `init_chat_model` | `langchain_openai.ChatOpenAI` | **-1**（去掉 `langchain` 元包） | -10 行 + 少一层 provider 间接 |
| 测试 / lint | 无 | `pytest` / `ruff` | dev 依赖 | 防止再次失控 |

### 1.2 可以做，但只在特定条件下

| 替换 | 条件 | 代价 |
| --- | --- | --- |
| CLI → `Typer` / `Click` | 命令数继续增长（>6 个）或需要 shell 补全 | +4 个左右传递依赖；当前 4 个命令不值得 |
| 配置 → `pydantic-settings` | 确实需要环境变量覆盖嵌套配置 | +2 个传递依赖；TOML + 嵌套 `endpoints` 的写法会更绕 |
| 切块 → `langchain-text-splitters` | 想统一 LangChain 体系 | 与已有的 `chonkie` 二选一，不减少依赖数 |
| 向量检索 → `FAISS` | 语料到十万级，或需要 ANN | 数十 MB 二进制 wheel；当前 2.6k 条用不上 |
| 向量检索 → `Chroma` | 需要持久化 + 元数据过滤 + 多进程 | 大量传递依赖 + 本地数据库，管理面明显变大 |
| BM25 → SQLite FTS5 | 想把文档、索引、元数据收进一个 `.db` | 无新依赖，但要重写分词/参数，搜索质量需回归 |
| 重排 → `flashrank` | 需要完全本地、CPU 重排 | 引入 onnxruntime 等本地推理栈 |
| 可观测性 → LangSmith | 需要追踪，且接受外部服务 | 零代码（环境变量开启），但数据出本机 |

### 1.3 明确不建议

| 替换 | 原因 |
| --- | --- |
| `atomicwrites` | 已不维护；15 行 `tempfile + os.replace` 更透明 |
| `TinyDB` / `sqlite-utils` 替代 JSONL | 代码更多，查询需求还没出现 |
| `rank_bm25` 替代 `bm25s` | 更慢、无持久化，纯倒退 |
| `EnsembleRetriever` / `langchain-classic` | v1 已把 legacy 拆走；为一个 25 行 RRF 引入整个 legacy 包 |
| `orjson` / `msgspec` | 2.6k 条 JSONL，性能收益可忽略，只增加依赖 |
| `loguru` / `structlog` | 当前 stdlib logging 调用不到 10 处，换库不省代码 |
| `rich` | 输出只有几行，换库不省代码 |
| `dynaconf` / `hydra` / `omegaconf` | 配置只有一份 TOML；引入“配置框架”会显著增加理解成本 |
| `keyring` | 本地 demo 的 `.env` 足够；OS keychain 是新的排障面 |
| `requests` / `aiohttp` | `httpx` 已在依赖树，不必再加一个 HTTP 客户端 |
| `pandas` / `polars` / `duckdb` | 没有表格分析需求 |

### 1.4 明确避免

| 方案 | 为什么不适合这个项目 |
| --- | --- |
| Haystack / LlamaIndex / txtai / RAGFlow | 它们会**替换整个项目**，而不是替换某个模块；依赖、概念、抽象都成倍增加 |
| LangChain agents / deepagents | 当前是确定的 `plan → retrieve → answer`；改成 agent 会增加不可预测性、成本和调试难度 |
| Elasticsearch / Meilisearch | 为 2.6k 个 chunk 引入外部服务，运维面远大于收益 |
| Chroma 作为常驻服务 | 单机 demo 不需要服务化；本地持久化用文件即可 |
| `unstructured` 解析 markdown | 语料本来就是 `.md`；引入解析器栈属于杀鸡用牛刀 |
| PromptLayer / Langfuse 自托管 | 提示词只有 3 份、无多人协作；外部服务会新增账号、网络和隐私面 |

### 1.5 决策矩阵

| | 低管理成本 | 高管理成本 |
| --- | --- | --- |
| **高代码收益** | `python-dotenv`、`tenacity`、`httpx`、`PromptTemplate`、`pytest/ruff` | `FAISS`、`pydantic-settings`、`Typer` |
| **低代码收益** | `orjson`、`rich`、`loguru`、`tqdm` | `Chroma`、`unstructured`、Haystack/LlamaIndex、外部向量数据库 |

优先做左上角；右上角只在对应需求出现时做；右下角不要碰。左下角属于“加了也没坏处，但没必要”，默认不加。

---

## 2. 判断标准：什么才算“降低管理复杂度”

不是“有库就用”。一个第三方库只有同时满足下面几条，才值得引入：

1. **已经在依赖树里**：`httpx`、`tenacity`、`langchain-core` 都是现成的，零新增成本；
2. **替换的是标准、易错、非业务代码**：`.env`、重试、HTTP、提示词渲染；
3. **节省 ≥ 30 行，或消除一类 bug**：例如 `safe_substitute` 的静默拼写错误；
4. **无重量级传递依赖、无外部服务、无二进制**；
5. **API 面小且稳定**：一个函数/一个类能解决，不需要学习一套框架；
6. **业务核心不外包**：RRF、检索编排、降级策略、产物校验仍然自己写。

反过来，只要出现下面任意一条，就应该拒绝：

- 为了省 5–10 行引入一个新包；
- 新包会带入 5 个以上传递依赖；
- 新包需要常驻服务、数据库或云账号；
- 新包的核心抽象会渗透到 `pipeline` / `domain`；
- 新包的版本节奏比项目快，且没有上限约束。

---

## 3. 推荐替换（逐项）

### 3.1 `.env`：手写 29 行 → `python-dotenv`

**位置**：`config/loader.py:95-123`

**现状**：

- `_env_value` 处理行尾注释、成对引号；
- `load_env` 按 `=` 切分、`os.environ.setdefault`；
- 不支持 `export FOO=1`，多行值会被截断。

**替换**：

```python
from dotenv import load_dotenv

def load_env(path: pathlib.Path | None = None) -> None:
    load_dotenv(home() / ".env" if path is None else path, override=False)
```

**收益**：-28 行；`export`、引号、注释、多行、插值都由库处理。
**成本**：+1 个轻量运行时依赖。
**结论**：**强推荐**。这是全文唯一建议新增的运行时依赖。

---

### 3.2 重试：手写循环 → `tenacity`

**位置**：`ingest/download.py:44-64`

**现状**：`for attempt in range(...)` + `time.sleep(1 + attempt)`，线性退避，异常分类靠手写。

**替换**：

```python
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

@retry(retry=retry_if_exception(_retryable),
       stop=stop_after_attempt(4),
       wait=wait_exponential(multiplier=0.5, max=8),
       reraise=True)
def fetch(url: str, client: httpx.Client) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content
```

**收益**：-12 行；指数退避带上限；`404` 与 `5xx` 的分类写在 `_retryable` 里，可测试。
**成本**：0（`tenacity 9.2.1` 已是 `langchain-core` 的直接依赖）。
**结论**：**推荐**。

---

### 3.3 HTTP：`urllib` → `httpx`

**位置**：`ingest/download.py` 全文

**现状**：`urllib.request` + 手写 `Request`/`urlopen`/`HTTPError`。

**替换**：`httpx.Client(timeout=..., follow_redirects=True)`。

**收益**：-10 行左右；连接池、超时、重定向、异常层次更清楚；与 `tenacity` 组合自然。
**成本**：0（`httpx 0.28.1` 已是 `langchain-core` 的直接依赖）。
**结论**：**推荐**。如果保留 `urllib`，至少把重试换成 `tenacity`。

---

### 3.4 提示词：`string.Template` → `PromptTemplate` / `ChatPromptTemplate`

**位置**：`llm/prompts.py:35-37`、`pipeline/nodes.py:39-40`

**现状**：`Template(...).safe_substitute(...)`，未知占位符原样保留。

**替换**：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    ("human", load_prompt("answer")),   # {context} / {question}
])
```

**收益**：-15 行；缺变量直接 `KeyError`（已实测）；system/human 角色分开；以后加 few-shot 不用自己实现。
**成本**：0（`langchain-core` 自带）。
**结论**：**推荐**。至少把 `safe_substitute` 换成 `substitute`。

---

### 3.5 模型构建：`init_chat_model` → `ChatOpenAI`（去掉 `langchain` 元包）

**位置**：`llm/client.py:14,33`

**现状**：`from langchain.chat_models import init_chat_model`，只为了按 `provider` 字符串创建一个 OpenAI 兼容模型。

**替换**：

```python
from langchain_openai import ChatOpenAI

ChatOpenAI(
    model=role.model,
    base_url=config.endpoints[role.endpoint].base_url,
    api_key=endpoint_api_key(config, role.endpoint),
    temperature=role.temperature,
    timeout=role.timeout,
    max_retries=role.retries,
    extra_body={"enable_thinking": False},   # 需要时
)
```

**收益**：-10 行；去掉 `langchain` 元包这一层 provider 间接；`extra_body` 是 `ChatOpenAI` 的正式字段（已实测）。
**成本**：直接依赖 -1；失去“任意 provider 字符串”能力。当前所有 endpoint 都是 OpenAI 兼容，不需要这个能力。
**结论**：**推荐**。如果以后要接 Anthropic/Gemini 原生 SDK，再考虑 `init_chat_model`。

---

### 3.6 测试与 lint：`pytest` / `ruff`

**现状**：无测试、无 lint、无 CI；`AGENTS.md` 明确说“没有 test suite”。

**建议**：

- dev 依赖加 `pytest`、`ruff`；
- 3 个冒烟测试（切块偏移、RRF、配置加载）；
- `ruff` 只开 `E,F,I,UP,B,SIM`，行宽与现状一致；
- CI 跑 `ruff check` + `pytest` + 代码量闸门。

**收益**：不是省代码，而是**防止删代码时改坏**。代码越少，测试越重要。
**成本**：dev 依赖，不进运行时。
**结论**：**推荐**。

---

## 4. 可选替换（有明确条件）

| 区域 | 候选库 | 能省多少 | 新增/传递依赖 | 什么时候值得 | 结论 |
| --- | --- | ---: | --- | --- | --- |
| CLI | `Typer` / `Click` | -50~80 行 | +4 左右（click/rich/…） | 命令数 >6，或需要补全/子命令嵌套 | 暂不 |
| 配置 | `pydantic-settings` | -60 行 | +2 左右（含 dotenv） | 需要 `FASTRAG_*` 环境变量覆盖嵌套配置 | 暂不；dotenv + pydantic 已够 |
| 切块 | `langchain-text-splitters` | -20 行 | +1（替换 `chonkie`，依赖数不变） | 想统一到 LangChain，且接受切块质量变化 | 二选一 |
| 向量 | `faiss-cpu` | -80 行 | +1 二进制（数十 MB） | 语料到十万级，或需要 ANN | 暂不 |
| 向量 | `Chroma` | -120 行 | 大量传递依赖 + 本地 DB | 需要持久化、元数据过滤、多进程 | 暂不 |
| BM25 | SQLite FTS5（stdlib） | -60 行 | 0 | 想把文档/索引/元数据收进一个 `.db` | 实验性 |
| 重排 | `flashrank` | -20 行 | 本地推理栈 | 需要完全离线、CPU 重排 | 暂不 |
| 并发下载 | `anyio` / `asyncio` | 0~10 行 | 0（httpx 已带） | 需要更高并发或异步 API | 暂不 |
| 并发写 | `filelock` | -5 行 | +1 | 允许多人同时跑 `prepare` | 可选 |
| 进度 | `tqdm` | -3 行 | +1 | 只影响体验，不影响代码量 | 可选 |
| 观测 | LangSmith | 0 行 | 0（已随 core） | 需要追踪且接受外部服务 | 可选 |
| 缓存 | `diskcache` / `joblib.Memory` | -20 行 | +1 | API 调用成本高，且能接受缓存失效复杂度 | 可选 |

### 4.1 关于 `pydantic-settings`

它确实能把 `loader.py` 的 TOML + `.env` + 路径解析收掉一部分，但有两个坑：

- `endpoints` 是 `dict[str, Endpoint]`，用环境变量覆盖嵌套键需要 `env_nested_delimiter`，写法比直接读 TOML 更绕；
- TOML 源需要 `TomlConfigSettingsSource`，仍然是 pydantic-settings 的 API，不是“零配置”。

如果只是想要 `.env`，`python-dotenv` 更小、更直白。**除非确实需要环境变量覆盖，否则不引入。**

### 4.2 关于 `Typer`

`Typer` 能把 9 个命令模块压成 1 个文件里的 4 个函数，但：

- 当前 CLI 只有 4 条命令（如果按最小方案砍掉调试命令）；
- 引入 `Typer` 会带 `click`、`rich`、`shellingham` 等；
- 团队要同时理解 argparse 和 Typer 两套（如果旧代码还没删干净）。

**命令少于 6 个时，argparse 更划算。** 只有 CLI 继续长，才考虑 Typer。

### 4.3 关于向量库

按“管理复杂度”排序：

```text
numpy（当前） < FAISS < Chroma / LanceDB < 常驻向量数据库
```

- 2.6k × 2560 的矩阵，numpy 一次内积约 10–20ms，完全够用；
- `FAISS` 是单二进制、成熟、API 小，是“要换就换它”的第一选择；
- `Chroma` 功能多，但依赖和运维面大；单机 demo 不值得；
- `InMemoryVectorStore.dump/load` 是 JSON，26MB 矩阵会膨胀，**不要**拿它替代 `vectors.npy`；
- 如果只需要 MMR，直接调用 `langchain_core.vectorstores.utils.maximal_marginal_relevance`，零新依赖。

### 4.4 关于 SQLite FTS5

这是一个“零依赖”的诱人选项：`sqlite3` 是标准库，当前环境（SQLite 3.53.1）已启用 FTS5，可以：

- 用一张 `documents` 表存 `id/content/metadata`；
- 用 `documents_fts` 虚表做 BM25 排序；
- 用一条 SQL 完成“关键词检索 + 回表”。

但它不是“免费”的：

- FTS5 的分词、停用词、BM25 参数与 `bm25s` 不同，检索质量需要重新评估；
- 英文可以用 `unicode61`，中文仍然需要额外分词；
- 会引入 schema、迁移、SQL 拼接，代码不一定更少。

**结论**：如果你想把“文档 + 索引 + 元数据”收进一个文件，可以做一个实验分支；否则保留 `bm25s`。

---

## 5. 不建议替换（自定义更合适）

| 部分 | 为什么保留自定义 |
| --- | --- |
| 原子写（`tempfile + os.replace`） | 15 行、标准库、无依赖；`atomicwrites` 已停止维护 |
| JSONL 文档存储 | 2.6k 条、按行读写、可直接 `head`/`wc`；换成 TinyDB/SQLite 只会增加 schema 代码 |
| RRF 融合 | 25 行纯函数，可测试；`EnsembleRetriever` 已移出 core |
| 产物指纹 / `meta.json` | 15 行买“防止静默错位”，这是项目最值得保留的防护 |
| 文档的 `source/start/end` metadata | 语义明确；即使改用 `Document`，也应保留这几个 metadata 键 |
| 检索编排与降级 | “路由失败用原问题、重排失败退 RRF”是业务策略，不应外包给框架 |
| CLI 参数校验 | 4 个命令、少量参数；argparse 足够，不需要 Typer 的抽象 |
| 日志 | stdlib logging 不到 10 处；loguru/structlog 只会增加依赖 |
| JSON 序列化 | 2.6k 条，`json` 足够；orjson/msgspec 的收益在微秒级 |
| 路径解析 | `home()` + `FASTRAG_HOME` 只有 10 行；`platformdirs` 会改变现有语义 |

---

## 6. 明确避免：会显著增加管理复杂度的方案

### 6.1 换 RAG 框架

| 候选 | 会发生什么 |
| --- | --- |
| Haystack | 用 Pipeline/Component 重写全部逻辑；依赖数量翻倍；现有 `SearchPlan`、降级策略要重新表达 |
| LlamaIndex | 以 `VectorStoreIndex`/`QueryEngine` 为中心；自定义双路检索和 RRF 会变成“框架里的插件” |
| txtai | 自带 embeddings + vector + pipeline；等于把项目交给一个更大的抽象 |
| RAGFlow | 偏服务化/平台化；本地 demo 不需要 |

这些不是“替换一个模块”，而是“替换整个项目的控制权”。在已经失去掌控的情况下，引入它们只会让排查链路更长。

### 6.2 换成 agent

当前流程是确定的四步。改成 `create_agent` / deepagents 后：

- 每次调用多若干次 LLM 决策；
- 检索时机、查询改写、是否重试都变得不确定；
- 成本和延迟上升；
- 出问题时无法用一条固定路径复现。

**保留 LangGraph 的四节点 DAG**，或者在最小方案里用 30–40 行函数编排替代；不要换成 agent。

### 6.3 引入外部服务

- Elasticsearch / Meilisearch：为 2.6k chunk 起一个服务，运维成本远超收益；
- Chroma server / Qdrant / Milvus：需要部署、备份、网络配置；
- Langfuse / PromptLayer：提示词只有 3 份，没有协作需求；
- Redis / Celery：没有队列需求。

---

## 7. 依赖预算

### 7.1 当前运行时依赖（9 个）

| 依赖 | 作用 | 建议 |
| --- | --- | --- |
| `bm25s` | BM25 | 保留 |
| `chonkie` | 切块 | 保留（或与 `langchain-text-splitters` 二选一） |
| `langchain` | 仅用 `init_chat_model` | **移除**，改用 `ChatOpenAI` |
| `langchain-openai` | LLM / embeddings | 保留 |
| `langchain-reranker` | 重排 | 保留；若砍掉重排则移除 |
| `langgraph` | 图编排 | 保留；若不需要 checkpoint/streaming，可用 30–40 行函数替代 |
| `numpy` | 向量矩阵 | 保留 |
| `pydantic` | 配置 | 保留 |
| `tiktoken` | token 计数 | 保留 |

### 7.2 建议的目标依赖

```text
运行时（推荐档 B）：
  bm25s, chonkie, langchain-core, langchain-openai, langgraph,
  langchain-reranker, numpy, pydantic, tiktoken, python-dotenv
  → 10 个（去掉 langchain 元包，加入 dotenv）

最小 MVP（档 C）：
  bm25s, chonkie, langchain-core, langchain-openai, numpy,
  pydantic, tiktoken, python-dotenv
  → 8 个（再去掉 rerank 与 langgraph）

开发：
  pytest, ruff
```

### 7.3 “一进一出”规则

> 新增一个运行时依赖，必须同时满足：删掉一个旧依赖，或减少 ≥50 行代码，或消除一类已知 bug。

否则不加。把这条和“代码量闸门”一起写进 `AGENTS.md`。

---

## 8. 与最小代码方案的对应关系

| 最小方案档位 | 第三方替换 |
| --- | --- |
| A 只合并 | `python-dotenv`、`tenacity`、`httpx`、`PromptTemplate`；其余不动 |
| **B 推荐** | A + 去掉 `langchain` 元包（`ChatOpenAI`）+ `Document`/`InMemoryVectorStore` 的检索接口 |
| C 最小 MVP | B + 去掉 `langchain-reranker`（砍重排）+ 去掉 `langgraph`（函数编排）；可选 SQLite FTS5 |

换句话说：

- **B 档仍然保留重排和 LangGraph**，只是把能交给库的标准件交出去；
- **C 档把“框架”也交出去**，但这会改变项目形态，需单独评估。

---

## 9. 建议的行动顺序

### 今天就能做（低风险）

1. `.env` → `python-dotenv`（-28 行，+1 依赖）；
2. 重试 → `tenacity`（0 新依赖）；
3. HTTP → `httpx`（0 新依赖）；
4. 提示词 → `PromptTemplate` / `ChatPromptTemplate`（0 新依赖）；
5. 模型构建 → `ChatOpenAI`，去掉 `langchain` 元包（-1 依赖）。

### 下一轮（需要测试兜底）

6. `pytest` + `ruff` + 3 个冒烟测试；
7. 把 `InMemoryVectorStore` / MMR 作为可选能力接入，但保留 `vectors.npy`；
8. 评估是否砍掉重排（顺带移除 `langchain-reranker`）。

### 只在规模/需求变化时做

9. 语料 > 10 万：评估 `FAISS`；
10. 需要多进程/持久化：评估 Chroma/LanceDB；
11. 需要完全离线重排：评估 `flashrank`；
12. 需要 HTML/PDF 语料：评估 `trafilatura` / `markitdown` / `unstructured`。

---

## 10. 附：核查记录

### 10.1 环境

| 包 | 版本 |
| --- | --- |
| Python | 3.13.18 |
| `langchain-core` | 1.7.0 |
| `langchain` | 1.4.7 |
| `langchain-openai` | 1.7.2 |
| `langgraph` | 1.2.17 |
| `bm25s` | 0.3.15 |
| `chonkie` | 1.7.0 |
| `tiktoken` | 0.14.0 |
| `numpy` | 2.5.4 |
| `pydantic` | 2.14.1 |
| `tenacity` | 9.2.1（已在依赖树） |
| `httpx` | 0.28.1（已在依赖树） |
| `python-dotenv` | 未安装 |
| `pydantic-settings` | 未安装 |
| `pytest` / `ruff` | 未安装 |
| `faiss-cpu` / `chromadb` / `flashrank` | 未安装 |

### 10.2 关键 API 实测

- `ChatOpenAI.model_fields` 包含 `extra_body`，可直接传 `enable_thinking` 等字段；
- `PromptTemplate.from_template("hello {name}")` 缺 `name` 时抛 `KeyError`；
- `InMemoryVectorStore` 使用 cosine similarity，支持 `filter`、MMR、`dump/load`（JSON）；
- `maximal_marginal_relevance` 是 `langchain_core.vectorstores.utils` 的公开函数；
- 当前 SQLite 3.53.1 已启用 FTS5（`pragma compile_options` 含 `ENABLE_FTS5`）；
- `langchain` 1.4.7 顶层只有 `agents/chat_models/embeddings/mcp/messages/rate_limiters/tools`，无 `langchain.chains` / `langchain.retrievers`。

### 10.3 结论

- **新增运行时依赖：只建议 `python-dotenv` 一个。**
- **可移除：`langchain` 元包（推荐）；`langchain-reranker`、`langgraph`（视功能取舍）。**
- **不建议为了“少写代码”引入 Chroma、Haystack、LlamaIndex、Typer、dynaconf 等重框架。**
- **最划算的三件事：dotenv、tenacity/httpx、PromptTemplate；它们都不增加（或只增加一个）依赖，却能去掉一批手写代码和一类静默错误。**
