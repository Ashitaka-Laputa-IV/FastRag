# FastRag 最小代码量改造建议

- 前置阅读：
  - `review/2026-11-03-duplication-and-langchain-ecosystem.md`（重复率与生态比对）
  - `review/2026-11-03-third-party-replacement-review.md`（第三方库替代与依赖预算）
- 本文目标：在**功能可用**的前提下，把代码量压到尽可能小，并建立“防止再次膨胀”的约束
- 现状：`42` 个 Python 文件、`2662` 行（`dev@90d3673`）
- 建议目标：**10–11 个文件、约 1,000 行**（推荐档）；如果愿意砍功能，可到 **600–800 行**

---

## 1. 先说结论

### 1.1 失控的来源不是“文件多”，是“概念多”

只合并文件，最多降 20%：

| 动作 | 预计变化 |
| --- | ---: |
| 42 个文件合并成约 10 个 | 2662 → ~2,400 |
| 9 个 CLI 命令文件合成 1 个 | ~2,220 |
| 删死代码 / 合并重复的原子写 | ~2,160 |
| 配置去掉冗余校验与常量 | ~2,080 |

也就是说：**纯搬家救不了代码量**。真正的大头是这些“看起来都应该有”的东西：

- 5 个独立调试命令（`chunk` / `embed` / `index` / `rerank` / `rrf`）
- 流式问答（`AnswerEvent` + `stream_events(version="v3")` + 两段 `warnings.filterwarnings`）
- 增量嵌入 + 两套 manifest + 指纹校验
- 可嵌套的 fallback 链
- 按角色拆分的 `[llm.route]` / `[llm.answer]`
- 几乎每个参数都做成配置项
- 自定义 `Chunk` / `ScoredChunk` / `ChunkStore` / `ResourceBundle`

**建议的取舍顺序：先砍功能，再合并文件，最后换实现。** 顺序反过来会反复改同一段代码。

### 1.2 三档方案

| 档位 | 代码量 | 文件数 | 保留功能 | 工期 | 风险 |
| --- | ---: | ---: | --- | --- | --- |
| A 只合并 | ~2,100 行 | ~12 | 全部保留 | 1 天 | 低 |
| **B 推荐** | **~1,000–1,200 行** | **~11** | 混合检索 + RRF + 可选重排 + 单轮问答 | 3–4 天 | 中 |
| C 最小 MVP | ~600–800 行 | ~9 | 混合检索 + RRF + 单轮问答 | 1 周 | 中高 |

推荐 **B**：功能基本不变，代码量减半以上；如果“掌控感”比“功能完整”更重要，直接做 C。

### 1.3 一句话原则

> **每个功能都要有代码预算；预算外的东西不加，超预算就先删一个旧功能。**

建议把这条写进 `AGENTS.md`，否则半年后还会回到 2.6k 行。

---

## 2. 目标目录（推荐档 B）

```text
src/fastrag/
  __init__.py        # 公开 API 再导出（~20 行）
  __main__.py        # python -m fastrag（~5 行）
  errors.py          # FastragError（~5 行）
  config.py          # pydantic 模型 + TOML + .env（~120 行）
  domain.py          # SearchPlan / Hit / AnswerResult（~30 行）
  ingest.py          # 下载 + 切块 + documents.jsonl（~110 行）
  index.py           # BM25 + 向量 + 检索（~160 行）
  retrieve.py        # RRF + 可选重排（~60 行）
  llm.py             # 模型构建 + fallback + 提示词（~70 行）
  pipeline.py        # 资源 + 节点 + 图 + RagPipeline（~180 行）
  cli.py             # 4 个子命令（~150 行）
  resources/         # 配置模板 + 3 份提示词（不增加代码）
```

代码行合计约 **910 行**；按当前注释密度，实际约 **1,000–1,100 行**。

### 2.1 现状 → 目标

| 目标文件 | 现状（文件/行） | 目标行数 | 来源 |
| --- | ---: | ---: | --- |
| `config.py` | 3 / 472 | 120 | `config/models.py` + `loader.py` + `__init__.py` |
| `domain.py` | 1 / 94 | 30 | 去掉 `Chunk` / `ScoredChunk` |
| `ingest.py` | 4 / 383 | 110 | `download` + `chunker` + `artifacts` |
| `index.py` | 4 / 445 | 160 | `store` + `bm25` + `dense` |
| `retrieve.py` | 3 / 106 | 60 | `fusion` + `rerank` |
| `llm.py` | 3 / 125 | 70 | `client` + `prompts` |
| `pipeline.py` | 5 / 435 | 180 | `resources` + `nodes` + `graph` + `runner` |
| `cli.py` | 13 / 502 | 150 | `main` + `format` + 9 个命令 |
| `__init__.py` | 1 / 41 | 20 | `api.py` 合并进来 |
| `__main__.py` | 1 / 6 | 5 | 不变 |
| `errors.py` | 1 / 24 | 5 | 只留异常类 |
| **合计** | **42 / 2662** | **~910** | **11 个文件** |

---

## 3. 先砍什么：按“每行代码的价值”排序

### 3.1 建议直接砍掉

| 功能 | 当前成本 | 砍掉后的影响 | 建议 |
| --- | ---: | --- | --- |
| 5 个调试子命令（`chunk`/`embed`/`index`/`rerank`/`rrf`） | ~200 行 + 5 个文件（整个命令层 9 个文件共 313 行） | 用 `prepare` 和 `search --mode` 覆盖 | **砍** |
| 流式输出（`AnswerEvent` / `stream` / v3 API） | ~60 行 + 实验性 API 风险 | CLI 直接打印最终回答；以后要恢复约 +45 行 | **砍** |
| 增量嵌入 + 两套 manifest + 指纹校验 | ~150 行 | 语料变化后 `prepare` 全量重嵌；保留一个 15 行的 `meta.json` 防错位 | **砍增量，保留一个指纹** |
| `[llm.route]` / `[llm.answer]` 双角色 | ~30 行 + 一份配置 | 路由与回答共用同一个模型；需要时加 `route_model` | **砍** |
| 可嵌套 fallback 链 | ~25 行 | 只保留一个 `fallback_endpoint`，或完全不配 | **砍嵌套** |
| `[download]` 配置节 | ~25 行 | 索引 URL、并发、超时写死为常量 | **砍** |
| `[fusion.weights]` / `rrf_k` | ~20 行 | 固定 `k=60`、等权 | **砍** |
| `[bm25].method` / `stopwords` / `min_score` | ~30 行 | 固定 `lucene` / `english` / `>0` | **砍** |
| 提示词 `$FASTRAG_HOME/prompts` 覆盖 | ~10 行 | 只用包内模板 | **砍** |
| `--config local` 这类资源名解析 | ~30 行 | `--config` 只接受文件路径 | **砍** |
| 交互式 `ask` 循环 | ~20 行 | `fastrag ask "问题"` 单次问答 | **砍** |
| `cli/format.py` 的截断与排版 | ~28 行 | 直接 `print(f"{score:.4f} {doc.id}")` | **砍** |

合计可削减约 **600–700 行**，且不碰核心检索逻辑。

### 3.2 建议保留

- **BM25 + 向量 + RRF**：这是项目本身，不要动；
- **`SearchPlan` 结构化输出**：一次调用同时决定“是否检索 + 两路查询”，是质量核心；
- **`prepare` 持久化产物**：否则每次 `ask` 都要重新嵌入，既慢又花钱；
- **一个 `meta.json` 指纹**：防止 `documents.jsonl` 与向量/BM25 错位，15 行买一个安全；
- **CLI 边界的一层 `try/except`**：库层抛 `FastragError`，CLI 打印并返回 1；
- **3 个冒烟测试**：见第 7 节。

### 3.3 功能预算表（加功能前先看）

| 功能 | 允许增量 | 说明 |
| --- | ---: | --- |
| 核心（混合检索 + RRF + plan + answer） | ~650 行 | 不可超 |
| 重排 | +55 行 | `retrieve.py` 内一个函数；不单独做命令 |
| 流式 | +45 行 | 只在 `ask` 内加，不引入 `AnswerEvent` 类型 |
| fallback | +20 行 | 单级，不嵌套 |
| 增量嵌入 | +40 行 | 只按 id 复用，不做 manifest 双写 |
| 交互式问答 | +20 行 | 可选 |
| 每个额外配置项 | +2~5 行 | 超过 20 个配置项就停下来 |

---

## 4. 怎么合并：11 个文件的职责

### 4.1 `config.py`（~120 行）

只做三件事：读 TOML、校验、解析路径。不要 `ConfigRef`、包内配置名、endpoint 引用遍历。

```python
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    paths: Paths
    endpoints: dict[str, Endpoint]
    embedding: Embedding
    llm: LLM
    reranker: Reranker | None = None
    retrieval: Retrieval = Retrieval()
    chunking: Chunking = Chunking()

def load_config(path: str | Path | None = None) -> Config:
    load_dotenv(home() / ".env", override=False)
    p = Path(path or os.environ.get("FASTRAG_CONFIG") or home() / "config/fastrag.toml")
    if not p.exists():
        p = PACKAGE / "resources/config/fastrag.toml"
    raw = tomllib.loads(p.read_text("utf-8"))
    raw["paths"] = {
        key: str((home() / value).resolve()) for key, value in raw["paths"].items()
    }
    return Config.model_validate(raw)
```

配置模型可以直接 `frozen=True`；路径在 `model_validate` 之前展开，避免对 frozen 实例赋值。

建议的配置面（从 ~50 个键压到 ~18 个）：

```toml
[paths]
data = "data"                 # docs/documents/vectors/bm25 都由它派生

[endpoints.main]
base_url = "https://api.siliconflow.cn/v1"
api_key_env = "SILICONFLOW_API_KEY"

[endpoints.local]
base_url = "http://127.0.0.1:9938/v1"

[embedding]
endpoint = "main"
model = "Qwen/Qwen3-Embedding-4B"
dimensions = 2560

[llm]
endpoint = "main"
model = "deepseek-ai/DeepSeek-V4.1-Flash"
fallback_endpoint = "local"   # 可删

[reranker]                     # 可删
endpoint = "main"
model = "Qwen/Qwen3-Reranker-4B"

[retrieval]
candidates = 20
top_n = 5

[chunking]
size = 512
overlap = 128
```

### 4.2 `domain.py`（~30 行）

**建议放弃 `Chunk` / `ScoredChunk`，直接用 `Document`。**

这与上一份报告“保留领域类型”的建议相反，理由是目标变了：现在优先“代码少、概念少”，而不是“类型最纯”。`Document` 已经有 `id` + `page_content` + `metadata`，足够承载 `source/start/end/score`。

```python
class SearchPlan(BaseModel):
    need_search: bool
    bm25_query: str = ""
    vector_query: str = ""

class Hit(NamedTuple):
    doc: Document
    score: float
    channel: str
    rank: int

@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: tuple[Hit, ...] = ()
    plan: SearchPlan | None = None
```

收益：删掉 `Chunk`、`ScoredChunk`、`ChunkStore`、`Chunk.to_record/from_record`，以及 `rerank.py` 里 Document ↔ Chunk 的映射，合计约 **120 行**。

代价：`source/start/end` 从字段变成 metadata 键，失去静态类型。用 `Hit` 和 3 行注释约束即可，不值得为它保留一层类型。

### 4.3 `ingest.py`（~110 行）

把下载、切块、写盘合成一个文件：

- 下载：`httpx.Client` + `tenacity`，只保留 `download` / `collect_urls`；
- 切块：`RecursiveCharacterTextSplitter.from_tiktoken_encoder(..., add_start_index=True)`，直接产出 `Document`，`metadata={"source": ..., "start_index": ...}`；
- 写盘：`documents.jsonl`，每行 `doc.model_dump_json()`；读回 `Document.model_validate_json`；
- 原子写：一个 15 行的 `atomic_write(path, writer)`，不再维护 `artifacts.py`。

```python
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=cfg.chunking.size,
    chunk_overlap=cfg.chunking.overlap,
    add_start_index=True,   # 保留原文偏移，替代自定义 chunker
)
docs = splitter.split_documents([
    Document(page_content=strip_boilerplate(p.read_text("utf-8")),
             metadata={"source": display_path(p)})
    for p in sorted(cfg.paths.docs.rglob("*.md"))
])
```

注意：`add_start_index` 的偏移是相对“剥掉模板头之后”的文本；如果必须精确到原文件，再在 `strip_boilerplate` 里加回前缀长度（3 行），不要为此恢复整个自定义 chunker。

### 4.4 `index.py`（~160 行）

一个文件里只放两类东西：BM25 和向量。

```python
# BM25
def build_bm25(docs, cfg) -> None: ...
def load_bm25(cfg): ...
def search_bm25(index, by_id, query, k) -> list[Hit]: ...

# 向量
def build_vectors(docs, embeddings, cfg) -> None:
    vectors = embed_all(...)
    np.savez(cfg.paths.vectors, ids=np.array([d.id for d in docs]),
             vectors=vectors.astype(np.float32))

def load_vectors(cfg, ids) -> np.ndarray | None:
    data = np.load(cfg.paths.vectors)
    return data["vectors"] if data["ids"].tolist() == ids else None

def search_vectors(query, vectors, docs, k) -> list[Hit]: ...
```

建议：

- **保留一个 `meta.json`**：`{"docs_sha": ..., "model": ..., "dimensions": ...}`，加载时比一次，15 行；不要两套 manifest；
- **增量嵌入可选**：如果 embedding 成本不敏感，直接全量重嵌，再省 40 行；
- 排序/归一化保留 10 行手写即可，`InMemoryVectorStore` 的 `dump/load` 是 JSON，不适合 26MB 矩阵；
- MMR 只有需要时再加，直接调用 `maximal_marginal_relevance`，约 15 行。

### 4.5 `retrieve.py`（~60 行）

- `rrf(rankings, k=60)`：20 行；
- `rerank(query, hits, top_n)`：25 行；
- 不要 `weights` 配置，先等权；不要 `EnsembleRetriever`（v1 已移出 core）。

### 4.6 `llm.py`（~70 行）

- `build_llm(cfg, model=None)`：15 行；
- `with_fallback(model, cfg)`：10 行，只支持单级；
- `load_prompt(name)`：10 行，只读包内；
- `plan_chain` / `answer_chain`：各 10 行，用 `ChatPromptTemplate` + `StrOutputParser`；
- 不需要 `llm/prompts.py` 单独成文件。

### 4.7 `pipeline.py`（~180 行）

`resources.py` + `nodes.py` + `graph.py` + `runner.py` 合成一个：

```python
class RagState(TypedDict, total=False):
    question: str
    plan: SearchPlan
    bm25: list[Hit]
    vector: list[Hit]
    fused: list[Hit]
    answer: str

class RagPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graph = self._build_graph()

    @cached_property
    def docs(self) -> list[Document]: ...
    @cached_property
    def by_id(self) -> dict[str, Document]: ...
    @cached_property
    def bm25(self): ...
    @cached_property
    def vectors(self): ...
    @cached_property
    def plan_chain(self): ...
    @cached_property
    def answer_chain(self): ...

    def ask(self, question: str) -> AnswerResult:
        state = self.graph.invoke({"question": question})
        return AnswerResult(...)
```

节点仍然是 4 个：`plan` → `retrieve_bm25` / `retrieve_vector` → `fuse` → `answer`。`StateGraph` 的装配 15 行，不需要单独的 `graph.py`。

### 4.8 `cli.py`（~150 行）

只留 4 个命令：

```text
fastrag download              # 可选：抓示例文档
fastrag prepare               # 切块 + 嵌入 + BM25（一条命令）
fastrag search "..." --mode bm25|vector|rrf [-k N]
fastrag ask "问题"             # 单次问答
```

- 不再有 `chunk` / `embed` / `index` / `rerank` / `rrf` 五个入口；
- `--mode` 覆盖原来的调试需求；
- `--config` 只接受路径；`-v` 控制日志；`--version` 用 `importlib.metadata.version("fastrag")`；
- 全局参数不再支持“子命令前后任意位置”，直接 `fastrag --config x search ...`；这能删掉 `_extract_globals` 的 28 行。

### 4.9 `__init__.py` / `errors.py`

- `api.py` 合并进 `__init__.py`，只导出 `ask / RagPipeline / load_config / AnswerResult / SearchPlan`；
- `errors.py` 只留 `class FastragError(Exception): pass`（5 行）；
- `version.py` 删除，改用 `importlib.metadata`。

---

## 5. 迁移路线（不要大爆炸重写）

### Phase 0：冻结现状（0.5 天）

- 打 tag（例如 `v0.2.0-before-min`）；
- 用当前版本跑 3 条命令并保存输出：`search`、`rrf`、`ask`（没有 key 就 mock）；
- 记录当前 CLI 行为，作为回归基线。

### Phase 1：只合并，不改行为（1 天）

按第 2 节的目标文件合并，保留所有现有函数与配置项。此阶段**只允许移动代码**，不允许改逻辑、改提示词、改参数。

目标：42 → 11 个文件，2662 → ~2,200 行。合并后先跑一次 `fastrag --help`、`prepare`、`search`、`ask`。

### Phase 2：砍功能（1–2 天）

按 3.1 的顺序删：

1. 5 个调试子命令 → `prepare` + `search --mode`；
2. 流式输出 → 只打印最终回答；
3. 增量嵌入 + 两套 manifest → 全量重嵌 + 一个 `meta.json`；
4. 双角色 LLM + 嵌套 fallback → 单模型 + 单级 fallback；
5. 配置项从 ~50 压到 ~18；
6. `--config local`、提示词覆盖、交互循环。

每删一项跑一次 `prepare` + `search` + `ask`。目标：~1,500–1,600 行。

### Phase 3：换实现（1 天）

- `.env` → `python-dotenv`；
- 提示词 → `ChatPromptTemplate` + `StrOutputParser`；
- 切块 → `RecursiveCharacterTextSplitter`；
- 下载 → `httpx` + `tenacity`；
- 原子写 → 一个 helper；
- 删除 `api.py` / `version.py` / `format.py` / `artifacts.py` / `store.py`。

目标：~1,000 行。

### Phase 4：加约束（0.5 天）

- 加 `ruff`（一个 `[tool.ruff]` 配置）；
- 加 3 个冒烟测试（见第 7 节）；
- 加代码量闸门：

```bash
# scripts/loc.sh
find src/fastrag -name '*.py' -print0 | xargs -0 wc -l | tail -1
```

CI 里超过 1,200 行直接失败；想加功能，先删旧代码。

---

## 6. 明确的删改清单

| 删除/合并 | 当前行数 | 替代 |
| --- | ---: | --- |
| `api.py` | 21 | `__init__.py` 直接导出 |
| `version.py` | 3 | `importlib.metadata.version` |
| `errors.py` 的 `run_cli` | 6 | 删除 |
| `cli/format.py` | 28 | 3 行 `print` |
| `cli/commands/*.py`（9 个） | 313 | 1 个 `cli.py` |
| `pipeline/graph.py` | 29 | `RagPipeline._build_graph()` |
| `pipeline/resources.py` | 84 | `@cached_property` |
| `llm/prompts.py` | 42 | 并入 `llm.py` |
| `ingest/artifacts.py` | 94 | 15 行 `atomic_write` + `meta.json` |
| `index/store.py` | 80 | `dict[str, Document]` |
| `domain.Chunk` / `ScoredChunk` | ~50 | `Document` + `Hit` |
| `config` 的常量与交叉校验 | ~80 | pydantic 类型 + 默认值 |
| 双 manifest / 增量嵌入 | ~120 | 一个 `meta.json`，全量重嵌 |

这些删除/合并约 **950 行**；再叠加第 3.1 节的功能裁剪（约 600–700 行），`2662 - 950 - 650 ≈ 1,060`，正好落在 1,000 行左右。

---

## 7. 测试：只留 3 个，但必须有

代码越少，越需要“防止改坏”的底线。建议 `tests/test_smoke.py`（约 50 行）：

1. **切块偏移**：`doc.metadata["start_index"]` 与 `page_content` 在原文中可定位；
2. **RRF**：两路重复 id、空输入、权重固定时的排序稳定；
3. **配置**：最小 TOML 能加载，缺字段/拼错键会报错。

不要引入完整测试框架矩阵；`pytest` + 这 3 条就够。没有它们，删代码会变成新一轮失控。

---

## 8. 建议的“最小配置”

从当前 2 份 TOML、约 50 个配置项压到 **1 份、约 18 个键**：

```toml
[paths]
data = "data"

[endpoints.main]
base_url = "https://api.siliconflow.cn/v1"
api_key_env = "SILICONFLOW_API_KEY"

[endpoints.local]
base_url = "http://127.0.0.1:9938/v1"

[embedding]
endpoint = "main"
model = "Qwen/Qwen3-Embedding-4B"
dimensions = 2560

[llm]
endpoint = "main"
model = "deepseek-ai/DeepSeek-V4.1-Flash"
fallback_endpoint = "local"

[reranker]
endpoint = "main"
model = "Qwen/Qwen3-Reranker-4B"

[retrieval]
candidates = 20
top_n = 5

[chunking]
size = 512
overlap = 128
```

如果连重排也不要，删掉 `[reranker]` 和 `retrieve.py` 的重排分支；如果连 fallback 也不要，删掉 `fallback_endpoint`。每删一项，代码和配置同时变小。

---

## 9. 如果只有一天：止血清单

按顺序做，全部是“低风险、可见收益”：

- [ ] 删除 `run_cli`、`REQUIRED`、`index_path`、`vectors_path`（约 15 行）
- [ ] 4 处原子写合并成 1 个 helper（约 -35 行）
- [ ] 9 个 CLI 命令文件合并成 1 个（约 -180 行）
- [ ] `api.py` / `version.py` / `format.py` 合并或删除（约 -40 行）
- [ ] `pipeline` 的 5 个文件合成 1 个（约 -80 行）
- [ ] `llm` 的 3 个文件合成 1 个（约 -25 行）
- [ ] 配置的 3 个文件合成 1 个，先不删校验（约 -80 行）
- [ ] `index` 的 4 个文件合成 1 个，先不删 manifest（约 -60 行）

一天后大约：**42 → 11 个文件，2662 → 2,100 行**，代码仍然全部可运行。第二天再决定要不要继续砍功能。

---

## 10. 最终建议

1. **不要大爆炸重写**。2.6k 行不算大，按 Phase 1→4 分四步走，每步都能运行、能回滚。
2. **先砍功能，再合并文件**。文件数只是表象，功能与配置项才是代码量的来源。
3. **推荐档 B**：11 个文件、约 1,000 行，保留混合检索、RRF、可选的 rerank 与单轮问答。
4. **如果只想要掌控感**：直接做档 C（600–800 行），砍掉流式、重排、fallback、增量嵌入和所有调试子命令；核心检索仍然完整。
5. **最后一定要加代码量闸门**。没有闸门，这次收缩只是下一次膨胀的起点。
