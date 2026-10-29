# FastRag 工程化审阅报告

- 审阅对象：`FastRag` 仓库（`src/fastrag/`、配置模板、提示词、打包与仓库配套）
- 审阅分支：`review`（基于 `dev` 的 `ed0c439` 创建）
- 审阅时间：2026-10-29
- 审阅环境：Windows / CPython 3.13.18 / uv 0.13.6 / `git core.autocrlf=true`
- 审阅方式：静态阅读 + 可执行验证（安装、构建、wheel 内容检查、CLI、配置非法输入、临时目录探针）
- 审阅边界：仓库未附带 `data/`，也没有 API key，因此**没有执行端到端检索与真实 LLM 调用**；涉及运行期的结论来自代码路径、CLI 行为与临时探针，不包含真实回答质量评估。

> 与前几轮的关系：`bot-reviews/1.md`、`2.md`、`3.md` 偏代码正确性，`4.md` 偏架构设计。本报告只讨论**工程化能力**：构建、打包、依赖、测试、静态检查、CI、发布、配置治理、数据产物治理、可观测性、开发者体验与安全。前几轮已修复的问题不重复展开，仍未关闭的会在第 4 节统一列出。

---

## 0. 结论摘要

### 0.1 一句话结论

**这是一个“可安装、可运行、文档质量高于平均水平”的 RAG 示例项目，但还不是一个“可维护、可发布”的工程**：核心链路设计扎实，构建产物可用，但测试、CI、静态检查、发布流程、产物一致性与失败处理仍有明显缺口，尤其是**零自动化测试**和**跨平台指纹不稳定**两项，会让后续任何改动都缺少安全网。

工程化成熟度可以定为：

| 等级 | 定义 | 当前 |
| --- | --- | --- |
| L1 | 能跑的脚本 | 已超过 |
| L2 | 可安装、可复现的示例 | **当前在这里** |
| L3 | 有测试/CI/发布门禁的可维护工程 | 目标 |
| L4 | 可运营的生产系统 | 不适用（示例项目） |

### 0.2 评分卡

| 维度 | 权重 | 得分（1–5） | 主要依据 |
| --- | ---: | ---: | --- |
| 构建与打包 | 10% | 4.0 | `uv sync --frozen` 可用；wheel/sdist 构建成功；配置与提示词正确随包分发 |
| 依赖与可复现性 | 15% | 2.0 | 依赖只有下限、wheel 安装不看 `uv.lock`；实测新解析拉到 `openai 3.41.0`，而锁文件是 `3.37.0`；构建后端版本范围已落后于本机 uv |
| 测试 | 20% | 0.5 | **0 个测试**；未安装 pytest；核心纯函数全部无回归保护 |
| 静态检查与格式化 | 5% | 1.0 | 无 ruff/mypy/pyright/pre-commit 配置；代码风格靠人工约定 |
| CI/CD 与发布 | 10% | 0.0 | 无 `.github/`、无 PR 检查、无 tag、无发布流程；版本固定 `0.1.0` |
| 配置管理 | 10% | 3.0 | 单一 TOML、frozen dataclass、校验在补强；但非法结构仍是裸 `KeyError`/`TypeError`，整型/枚举仍有缺口 |
| 错误处理 | 5% | 3.0 | `FastragError` + `run_cli` 约定好、CLI 退出码为 1；但配置期异常与下载失败路径不统一 |
| 数据产物治理 | 10% | 3.0 | 全字节 sha256 指纹设计优秀；但跨平台换行不稳定、空语料保留旧产物、写入非原子、缺 provenance |
| 可观测性 | 5% | 1.5 | 全部 `print`；无日志级别、无耗时/成本/token 指标、CLI 无 `--json`/`--quiet` |
| 文档与 DX | 5% | 3.5 | README 覆盖用户路径、docstring 解释“为什么”；缺开发/测试/排错/发布文档，`bot-reviews` 未归档到 `docs/` |
| 安全与合规 | 5% | 3.5 | key 只走环境变量、`.env` 已忽略、未发现硬编码 key；缺 secret 扫描、依赖审计与提示注入约束 |
| **加权总分** | **100%** | **2.0 / 5** | 示例级完成度高，工程级门禁缺失 |

### 0.3 最该先做的五件事

1. **建立测试与 CI**：先覆盖 `chunks_sha256`、`rrf_fuse`、`strip_boilerplate`、`load_config`、空语料分支、CLI 退出码（约 20 个用例，0.5–1 天）。
2. **修复产物的跨平台确定性**：所有机器生成的 JSONL/manifest 固定 `\n`，并加跨平台断言；否则 Windows 上构建的向量/索引在 Linux CI 上会被判为“语料已变更”。
3. **让失败显性化**：空文档目录时不要继续用旧 `chunks.jsonl`；下载失败或 0 条 URL 时返回非零退出码，并清理上游已删除的文档。
4. **收敛配置错误出口**：把缺 section/缺字段/类型错误的裸 `KeyError`/`TypeError` 全部转成 `FastragError`，并补整型、枚举、URL、路径校验；增加 `fastrag doctor`。
5. **补齐发布基础设施**：填 `description`、classifiers、`[project.urls]`、`py.typed`；CI 中 `uv sync --frozen`；更新 `uv_build` 版本范围；打第一个 tag 前先跑通测试门禁。

---

## 1. 工程化现状（实测）

### 1.1 仓库结构

```text
FastRag/
├── .env.example                 # 只演示 SILICONFLOW_API_KEY
├── .gitignore                   # .venv/ data/ dist/ __pycache__/ .env
├── .python-version              # 3.13
├── LICENSE                      # MIT
├── README.md                    # 用户使用说明
├── pyproject.toml               # 打包元信息 + 8 个运行时依赖
├── uv.lock                      # 53 个包的解析结果（317 KB）
├── bot-reviews/                 # 4 份历史审阅文档（1.md ~ 4.md）
└── src/fastrag/
    ├── __init__.py
    ├── chunk_docs.py            # 文档 -> chunks.jsonl
    ├── chunks.py                # 读取语料 + 内容指纹
    ├── cli.py                   # 统一 CLI 转发
    ├── config.py                # TOML 配置 + 校验 + .env
    ├── download_example_docs.py # 抓取示例文档
    ├── embed_chunks.py          # 向量化 + 向量检索
    ├── errors.py                # FastragError + run_cli
    ├── keyword_search.py       # BM25 建索引 / 检索
    ├── llm.py                   # init_chat_model 封装
    ├── prepare.py               # chunk -> embed -> index
    ├── rag_graph.py             # LangGraph 编排
    ├── reranker.py              # cross-encoder 重排
    ├── rrf.py                   # Reciprocal Rank Fusion
    └── resources/               # 配置模板 + 提示词（随包分发）
```

规模：15 个 Python 文件、2,110 行（含资源 `__init__.py`），其中 `config.py` 502 行、`rag_graph.py` 420 行，其余模块多在 100–230 行。这个体量下**完全有条件做高覆盖率单测**，缺少测试更多是流程问题而不是成本问题。

缺失的工程目录：

```text
tests/                # 不存在
.github/workflows/    # 不存在
docs/                 # 本次审阅前不存在
.pre-commit-config.yaml  # 不存在
.gitattributes        # 未跟踪（本机 core.autocrlf=true）
CHANGELOG.md / CONTRIBUTING.md / SECURITY.md  # 不存在
```

### 1.2 构建与打包（实测通过，但有警告）

执行 `uv sync --frozen` 成功安装 53 个包；`uv build` 成功产出：

```text
dist/fastrag-0.1.0-py3-none-any.whl
dist/fastrag-0.1.0.tar.gz
```

wheel 内容检查确认：

- `fastrag/resources/config/fastrag.toml` 与 `fastrag.local.toml` 在包内；
- `fastrag/resources/prompts/*.txt` 在包内；
- `entry_points.txt` 注册了 `fastrag = fastrag.cli:main`；
- **没有** `py.typed`，也**没有**测试（本来就没有测试可打）；
- sdist 只包含 `src/`、`README.md`、`LICENSE`、`pyproject.toml`，没有误带 `data/` 或 `.env`，这一点是好的。

在一个干净 venv 中只安装 wheel（`--no-deps`）后，从临时工作目录运行：

```text
fastrag/config.py -> <venv>/Lib/site-packages/fastrag/config.py
home()            -> <临时工作目录>
默认配置          -> <venv>/Lib/site-packages/fastrag/resources/config/fastrag.toml
提示词            -> <venv>/Lib/site-packages/fastrag/resources/prompts/answer.txt
```

结论：README 中“安装 wheel 后不依赖源码树、配置与提示词随包分发”的说法**成立**。`home()` 在 wheel 场景回退到当前目录的行为也符合文档。

但构建过程出现警告：

```text
warning: build_system.requires = ["uv-build>=0.12.23,<0.13.0"]
         does not contain the current uv version 0.13.6
```

这说明构建后端约束已经落后于开发者本机工具链。当前能构建是“碰巧兼容”，不是被约束保证的。

### 1.3 依赖与可复现性

`uv.lock` 被提交，且 `uv sync --frozen` 可以精确复现 53 个包，这是加分项。但存在三个可复现性缺口：

1. **运行时依赖全部只有下限**（`pyproject.toml:12-21`）。wheel 用户安装时不会读取仓库的 `uv.lock`，而是重新解析。实测在干净 venv 中安装 wheel 时，解析器选择了 `openai 3.41.0`、`langchain-openai 1.7.3`，而锁文件里是 `openai 3.37.0`、`langchain-openai 1.7.2`。示例项目可以接受一定漂移，但至少要明确“以 lock 为准”，并在 CI/文档里使用 `uv sync --frozen`。
2. **README 的安装命令是 `uv sync` 而不是 `uv sync --frozen`**。对已有 lock 的仓库，前者仍可能更新 lock，导致开发者本地与 CI 解析结果不同。
3. **构建后端版本范围**与本机 uv 不一致（见 1.2），且 `requires-python >=3.13` 比较激进：代码实际使用的能力（`tomllib`、`datetime.UTC`、`importlib.resources.files`）在 3.11+ 已具备，若面向更广用户，建议评估放宽到 `>=3.11` 并加 3.11/3.12/3.13 矩阵。

另一个隐藏问题：锁文件里没有 dev 依赖（pytest/ruff/pyright），因为 `pyproject.toml` 没有 `[dependency-groups]`。这意味着**开发工具本身也无法用 lock 复现**。

### 1.4 测试：当前为 0

实测：

```text
$ uv run python -m pytest
... No module named pytest
exit code: 1
```

仓库没有 `tests/`、没有 pytest 配置、没有测试依赖、没有覆盖率门槛。而这套代码恰恰有很多“低成本、高价值”的纯函数：

- `chunks_sha256`：指纹是否只依赖字节；缺失文件是否给友好错误；
- `rrf_fuse`：权重、并列、重复 id、非法 `k`；
- `strip_boilerplate`：匹配/不匹配/偏移回加不变式；
- `_env_value`：引号、行尾注释、未闭合引号；
- `load_config`：缺 section、缺字段、边界值、非法枚举；
- `search`（BM25/向量）：空语料、`k <= 0`；
- `plan_search`/`answer`：重试、fallback、空检索分支（可用假 LLM 对象）；
- CLI：`--config` 位置、退出码、子命令转发；
- `download`：URL 收集、HTML 探测、失败聚合（可用本地 HTTP server 或 mock）。

没有这些测试，前几轮审阅反复修复的边界（`-k 0`、空语料、偏移、指纹复用）只能靠人肉记忆维持。

### 1.5 静态检查与格式化

`pyproject.toml` 中没有 `[tool.ruff]`、`[tool.mypy]`、`[tool.pyright]`、`[tool.pytest.ini_options]`、`[tool.coverage.*]`。代码里出现了 `# noqa: BLE001`，说明作者有 lint 意识，但仓库并没有真正执行这条规则的配置。结果是：

- 未使用导入、过长行、裸 `except`、类型不一致等问题只能靠审阅发现；
- 跨模块的 `list[dict]`、`dict` 形态没有类型检查兜底；
- 新贡献者无法在提交前自动格式化。

代码表面风格是统一的（大致 88 列、docstring 完整），但“统一”目前是约定，不是门禁。

### 1.6 CI/CD 与发布

仓库有 77 个提交、多个特性分支和 PR 合并记录（如 #21–#25），说明有分支/PR 工作流；但：

- 没有 `.github/workflows/`，所有 PR 都没有自动检查；
- 没有 tag，`pyproject.toml` 的 `0.1.0` 从未通过发布流程验证；
- 没有 `--version`，用户无法从 CLI 查询版本；
- 版本号是静态字符串，没有与 tag/CHANGELOG 联动；
- 没有发布说明、没有 PyPI/Release 流程、没有产物签名/哈希。

对示例项目来说可以理解，但既然已经做到 wheel 可安装，**最少应有一个“构建 + 冒烟”CI**，避免打包资源再次被漏掉。

### 1.7 配置与运行环境

做得好的部分：

- 单一 TOML、`frozen` dataclass、集中校验入口；
- `.env` 解析已支持行尾注释与成对引号（`config.py:222-242`）；
- `--config` 支持路径与包内名称（`default`/`local`）；
- 本地 endpoint 用空 `api_key_env` + 占位 key，云端/本地一套代码；
- 缺失 key 的错误信息具体到变量名。

但仍存在“配置错误进入运行期才爆炸”的路径。实测结果（临时 TOML）：

| 非法配置 | 实际表现 | 期望 |
| --- | --- | --- |
| 删除 `[reranker]` section | `KeyError: 'reranker'` + traceback | `FastragError: 缺少 [reranker]` |
| 删除 `[paths]` section | `KeyError: 'paths'` | 同上 |
| endpoint 缺 `base_url` | `KeyError: 'base_url'` | 指出 endpoint 名与字段 |
| `[llm.route]` 缺 `model` | `KeyError: 'model'` | 指出 `llm.route.model` |
| `docs_dir = 1` | `TypeError: unsupported operand ...` | `FastragError: paths.docs_dir 必须是字符串` |
| `tokenizer = "nope"` | **加载成功**，切块时才失败 | 加载期用 tiktoken 校验或给出白名单 |
| `base_url = 123` | **加载成功**，运行期才失败 | 校验为字符串且以 http(s):// 开头 |
| 未知配置键 | **静默忽略** | 警告或报错 |
| `top_n = 2.5` | **加载成功**，后续切片时 `TypeError` | 校验为正整数 |
| `fusion.method = "bogus"` | `FastragError`（正确） | 保持 |
| `stopwords = ["english", 1]` | `FastragError`（正确） | 保持 |
| 非法正则 | `FastragError`（正确） | 保持 |

根因是 `load_config` 直接做 `raw["paths"]`、`raw["reranker"]` 等索引（`config.py:438-449`），且 `_number` 只检查“是数字”，不检查整型/枚举/URL（`config.py:298-302`）。

另外，`paths.*` 没有 `expanduser()`（`config.py:258-260`），写 `~/data` 会解析成 `<home>/~/data`；而 `--config` 与 `FASTRAG_HOME` 是支持 `~` 的，口径不一致。

### 1.8 数据产物与生命周期

现有设计是项目最扎实的部分：

- `chunks.jsonl` 是唯一上游；
- 向量 manifest 与 BM25 meta 都记录其全字节 sha256；
- 条数相同但内容变化也能识别；
- `save_vectors` 复用调用方算好的 digest，避免重复读取（`embed_chunks.py:98-130`）；
- BM25 在加载时校验语料与参数，不一致直接提示重建（`keyword_search.py:62-98`）。

但工程化上仍有四个关键缺口：

**(1) 指纹是“字节指纹”，不是“内容指纹”，跨平台不稳定。**

`chunk_docs.py:107-109` 用文本模式写 `chunks.jsonl`：

```python
with config.paths.chunks_file.open("w", encoding="utf-8") as fh:
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
```

在 Windows 上，文本模式会把 `\n` 翻译成 `\r\n`。实测：

```text
写入一行 JSON 后的字节: b'{"id": "a"}\r\n'
LF 文件 sha256:   f04e3940ef65049c76c521e2a173b54a16e599fde7e2edd0b76c3a100f1ddb65
CRLF 文件 sha256: a9e0cd4a8f523a76cea35af8a6f6050dc2515e1c52f34216e92b2b6263855960
same? False
```

后果：Windows 上生成的向量/索引，在 Linux CI 或另一台机器上会被判定为“语料已变更”，从而触发不必要的全量重建；如果 CI 缓存或共享产物，则会出现“同一份语料、不同指纹”的不可解释失败。`git core.autocrlf=true` 还意味着源码检出的换行也可能随平台变化（虽然 `.env`/TOML 的读取会做换行归一，但 `chunks.jsonl` 的指纹不会）。

**修复方向**：所有机器生成的产物显式 `newline="\n"` 或直接写 bytes；把“规范化序列化”作为产物契约；增加跨平台测试，断言同一逻辑语料在 Windows/Linux 得到同一指纹。

**(2) 空文档目录时，旧 `chunks.jsonl` 不会被清理。**

`chunk_docs.py:99-109` 在 `records` 为空时打印“未生成 chunk 文件”并直接 `return`，不会删除已有文件。实测：

```text
$ FASTRAG_CONFIG=<空 docs 目录 + 已存在旧 chunks.jsonl> python -m fastrag.chunk_docs
没有可切分的文档 -> ...，未生成 chunk 文件
exit: 0
chunks still exists: True
chunks content: {"id": "stale", "text": "old"}
```

而 `prepare.main` 在 `chunk_docs.main()` 之后继续 `load_chunks` → `ensure_vectors` → `build_index`（`prepare.py:40-48`）。也就是说，用户删空/移走文档后执行 `fastrag prepare`，会**继续用旧语料重建向量与索引**，并正常退出。这直接违背“一条命令刷新全部产物”的承诺。

**修复方向**：无记录时明确失败（推荐）或写入空文件并让下游显式处理 0 条；`prepare` 应拿到本次切块的记录/指纹并与磁盘比对，而不是无条件继续。

**(3) 产物写入非原子，且没有并发保护。**

`chunks.jsonl`、`vectors.npy`、`manifest.json`、BM25 索引都是就地写：

- 写到一半 Ctrl-C / 断电 → 截断文件；
- 两个 `prepare` 并发 → 互相覆盖；
- `ask` 与 `prepare` 并发 → 可能读到半个 JSONL，抛 `JSONDecodeError`；
- BM25 索引写失败时，旧索引可能已被部分覆盖。

现有指纹能发现“不一致”，但发现时用户拿到的是一句“请重新运行”，而不是一次原子提交后的可用状态。

**修复方向**：临时文件/临时目录 + `os.replace`；manifest 最后写，作为提交点；用 `filelock` 包住 `prepare`；读取端对截断文件给出明确提示。

**(4) 缺少 provenance（产物来源信息）。**

manifest/meta 目前记录了模型、维度、count、chunks 指纹、切块参数和时间，但没有记录：

- 包版本、git commit；
- 配置文件的 sha256；
- 提示词的 sha256/版本；
- `bm25s`、`numpy` 等关键库版本；
- manifest schema 版本。

一旦结果异常，很难回答“这份向量是哪次提交、哪份配置、哪版提示词生成的”。

### 1.9 可观测性

当前所有输出都是 `print`：

- 库函数直接向 stdout 打印进度（`embed_chunks.py:89-94`），作为库复用时无法关闭；
- `\r` 进度条不判断是否是 TTY，重定向到日志会产生超长行；
- 没有日志级别、时间戳、模块名、耗时；
- 没有 token/费用统计，也没有 rerank 降级、重试次数的结构化记录；
- CLI 只有人类可读输出，没有 `--json`，不利于脚本、评估与 CI 断言；
- `warnings.filterwarnings(...)` 全局屏蔽了 pydantic 与 LangGraph v3 的实验性警告（`rag_graph.py:367-371`），可能连真实告警一起吞掉。

### 1.10 文档与开发者体验

优点：

- README 覆盖安装、CLI、构建数据、本地模型、配置项与流程图；
- 每个模块 docstring 都在解释“为什么”，质量明显高于同类示例；
- 错误提示大多给出下一步命令（如“请先运行 fastrag download”）。

缺口：

- 没有“开发者路径”：如何跑测试、如何加 lint、如何构建 wheel、如何发版；
- 没有项目结构说明与模块职责表；
- 没有排错章节（缺 key、缺索引、维度不一致、SOCKS 代理、Windows 路径）；
- README 使用 `cp .env.example .env`，对 Windows PowerShell 用户不友好；
- `bot-reviews/` 直接堆在根目录、以数字命名，既没有索引，也没有在 README 中链接；随着审阅轮次增加会越来越难发现；
- 没有 `docs/` 结构，用户文档与审阅记录混在一起；
- `fastrag ask` 只支持交互模式，不能 `fastrag ask "问题"`，也不支持从 stdin 读问题——这让自动化评估与 CI 冒烟都很难做；
- `fastrag search --help` 的 usage 显示为 `fastrag search query ...`（`cli.py:128` 改写 `sys.argv[0]` 的副作用），属于小但明显的 DX 瑕疵；
- `--config` 只能放在子命令前；放到后面会得到 `unrecognized arguments: --config ...`（`cli.py:89-102`），不够友好；
- 没有 `fastrag --version`、`fastrag doctor`、`fastrag clean` 等运维命令。

### 1.11 安全与合规

做得好的部分：

- key 只从环境变量/`.env` 读取，`.env` 已被 `.gitignore` 忽略；
- 未在仓库中发现硬编码的 `sk-...` 密钥（`git grep` 验证）；
- BM25 索引以 JSON/NumPy 格式加载，`bm25s.BM25.load` 默认 `allow_pickle=False`，未引入 pickle 反序列化风险；
- TLS 校验使用默认行为，没有关闭校验的开关。

需要补的部分：

- 没有 secret 扫描（gitleaks/trufflehog）与依赖审计（Dependabot/pip-audit）；
- 从 `docs.langchain.com` 抓取的文档没有完整性校验与来源清单，属于外部供应链输入；
- `answer.txt` 没有明确要求“把资料中的指令当作数据”，也没有引用标注要求，面对非可信语料存在提示注入风险；
- 没有 `SECURITY.md`，漏洞报告路径不明确；
- 本地 endpoint 用占位 key 的约定清晰，但如果误把本地地址配成公网地址，key 会以占位值发出——风险低，但可在文档中强调。

### 1.12 性能与可扩展性（工程视角）

- `ask` 启动即加载语料、构建嵌入客户端、确保向量、加载 BM25 索引与重排器（`rag_graph.py:97-123`），即使第一轮只是寒暄；冷启动成本没有在 README 中说明。
- 向量检索每次查询都对整个矩阵做归一化与矩阵乘（`embed_chunks.py:200-203`），当前示例规模可接受，但没有 ANN 接口，扩展到十万级以上会明显退化。
- 任何 chunk 变化都会触发全量重嵌入（整份 `chunks.jsonl` 的 digest 变化），没有按 chunk 内容哈希的增量缓存；这是正确性优先的合理取舍，但应在文档中说明成本。
- BM25 每次都从磁盘加载，`ask` 内部只加载一次，但 CLI 子命令每次进程都重新加载；对交互式没问题，对批量脚本不友好。
- `download` 固定 12 线程、固定 30s 超时、固定 3 次重试，没有整体限速/退避/缓存；对示例语料规模足够，但不适合频繁刷新。
- 没有 benchmark、没有评估集、没有 token/成本统计；“调参”和“换模型”的收益无法量化。

---

## 2. 问题清单（按严重程度）

> 编号仅用于本报告追踪。P0 = 阻碍工程化/会造成静默错误；P1 = 应尽快补齐；P2 = 可选改进。

### P0（建议在下一轮迭代前完成）

#### P0-1　没有测试：核心不变式完全无保护

**证据**

- 仓库无 `tests/`；
- `uv run python -m pytest` → `No module named pytest`，退出码 1；
- 无 pytest/coverage 配置、无 dev 依赖组。

**影响**

指纹、RRF、偏移、配置校验、空语料等边界全靠人工审阅维持。任何重构都可能悄悄破坏“条数不变但内容变了”“`-k 0` 语义”“模板头偏移”等已经修过的问题。

**建议**

- 先建 `tests/`，目标不是追求覆盖率数字，而是锁住 20 个高风险行为（见 5.6 的用例清单）；
- 纯函数直接测；外部服务用 fake/mock，禁止单测访问网络；
- 集成测试用 2–3 篇微型文档跑通 `chunk → embed(fake) → index → search`；
- CI 中跑 `pytest --cov=fastrag`，核心模块覆盖率门槛先定 70%。

**工作量**：0.5–1 天（含测试夹具）。

#### P0-2　跨平台换行导致产物指纹不稳定

**证据**

- `chunk_docs.py:107-109` 文本模式写 JSONL；Windows 实测写入 `\r\n`；
- `chunks_sha256` 对字节做 sha256（`chunks.py:47-51`）；
- 同一逻辑内容，LF 与 CRLF 的 sha256 分别为 `f04e...` 与 `a9e0...`；
- 本机 `core.autocrlf=true`。

**影响**

Windows 构建的向量/索引在 Linux 上会被判定为“语料已变更”，触发重建或直接报错；跨平台 CI 缓存、共享产物、团队协作都会受影响。这是对“内容指纹”承诺的实质性破坏。

**建议**

- 机器生成的 JSONL/JSON/manifest 统一 `newline="\n"`（或写 bytes）；
- 为 `chunks.jsonl` 定义规范化序列化契约（字段顺序、分隔符、换行）；
- 新增跨平台测试：同一 fixture 在 Windows/Linux 计算出的 digest 必须一致；
- 加 `.gitattributes`（`* text=auto eol=lf` 或按文件类型明确），减少源码层的换行漂移。

**工作量**：1–2 小时 + 1 个测试。

#### P0-3　空文档目录时 `prepare` 继续使用旧语料

**证据**

- `chunk_docs.py:102-104`：无记录时打印后直接返回，不删除旧 `chunks.jsonl`；
- 实测旧文件保留、退出码 0；
- `prepare.py:40-48` 随后继续 `load_chunks` → `ensure_vectors` → `build_index`。

**影响**

用户删除/迁移文档后执行“刷新全部产物”，实际得到的是旧语料的向量与索引，且命令成功退出。最坏情况下回答引用已不存在的文档，且没有任何告警。

**建议**

- 无记录时抛 `FastragError`（明确告知“没有可切分文档，已保留/未修改旧产物”），或删除旧产物并让下游显式处理空语料；
- `prepare` 用本次切块结果与磁盘指纹比对，不一致就失败；
- 增加测试：空 docs + 旧 chunks → 必须失败且旧文件不被静默复用。

**工作量**：1–2 小时。

#### P0-4　下载失败不反映到退出码，且不清理过期文档

**证据**

- `download_example_docs.py:80-96`：失败项只收集并打印，`main` 不抛错；
- 只写新文件，不删除上游已移除的文档；
- `collect_urls` 可能返回空列表，此时打印“成功 0/0”并正常退出；
- 上游索引 URL 与正则硬编码（`:23-31`、`:49-62`）。

**影响**

CI/脚本无法判断语料是否完整；用户可能在“部分下载 + 旧文件残留”的语料上构建索引；上游改版后可能悄无声息地得到 0 篇新文档。

**建议**

- 任意失败或 `len(urls) == 0` 时非零退出（提供 `--allow-partial` 显式降级）；
- 写 `docs-manifest.json`（URL、sha256、抓取时间），支持 `--prune` 删除不在清单中的文件；
- 增加 `--dry-run`/`--check`，先打印将下载/删除的列表；
- 对索引页面结构变化做显式校验与友好报错。

**工作量**：半天。

#### P0-5　产物写入非原子，缺少并发保护

**证据**

- `chunk_docs.py:106-109` 直接覆盖 `chunks.jsonl`；
- `embed_chunks.py:117` 直接 `np.save`；
- `keyword_search.py:46-58` 直接写入索引目录；
- 无临时文件、无 `os.replace`、无文件锁；
- 指纹校验只在加载时发生，无法阻止“写到一半”的文件被读到。

**影响**

中断/并发/磁盘满会留下截断文件或半新半旧产物；`ask` 可能在 `prepare` 运行时读到不完整 JSONL，抛裸 `JSONDecodeError`。

**建议**

- 所有产物先写临时路径，再 `os.replace` 原子替换；
- manifest/meta 最后写，作为“提交点”；
- `prepare` 用 `filelock` 串行化；
- 读取端捕获 `JSONDecodeError`，提示“产物可能正在写入/已损坏，请重跑 prepare”。

**工作量**：3–4 小时。

### P1（建议 1–2 周内完成）

#### P1-1　配置错误仍以裸 `KeyError`/`TypeError` 暴露

**证据**：见 1.7 实测表；`config.py:438-449` 直接索引 section；`_resolve` 在遇到非字符串时抛 `TypeError`。

**影响**：用户看到的是 Python traceback，而不是“哪个配置项错了、应该怎么改”；与 `FastragError` 的统一约定不一致。

**建议**：

- 增加 `_section(raw, name)` / `_field(section, key, owner)` 辅助函数，统一抛 `FastragError`；
- 或在加载期用 pydantic/msgspec/dataclass `__post_init__` 做 schema 校验；
- 对未知键给出警告（或 `--strict-config` 下报错）；
- 增加 `fastrag doctor`：加载配置 + 检查产物 + 打印各 endpoint 的 key 是否存在（不打印值）。

**工作量**：半天到 1 天。

#### P1-2　数值校验不区分整型，部分字段完全未校验

**证据**：

- `top_n = 2.5` 通过校验（实测 `type == float`），后续 `list[:2.5]` 会抛 `TypeError`；
- `tokenizer = "nope"` 加载成功，切块时才由 tiktoken 报错；
- `base_url = 123` 加载成功；
- `include_delim`、`provider`、`api_key_env` 类型未校验；
- `paths.*` 未校验字符串/未展开 `~`。

**建议**：

- 增加 `_int(owner, value, minimum=...)`，拒绝 bool/float/字符串；
- 对 `tokenizer` 在加载期调用 `tiktoken.get_encoding` 验证（或维护白名单）；
- 校验 `base_url` 为 `http(s)://` 字符串、`provider` 非空、`include_delim ∈ {"prev", "next", None}`；
- 校验 `retrieval.candidates >= top_n`（不满足时警告或报错）；
- 路径统一 `expanduser()`；
- 为每个字段补一个非法值测试。

**工作量**：半天。

#### P1-3　依赖漂移与构建工具链未锁定

**证据**：

- `pyproject.toml:12-21` 全部 `>=`；
- 干净 venv 安装 wheel 时解析到 `openai 3.41.0` / `langchain-openai 1.7.3`，与 lock 的 `3.37.0` / `1.7.2` 不同；
- README 使用 `uv sync`（非 frozen）；
- `uv-build>=0.12.23,<0.13.0` 不包含本机 uv 0.13.6，构建时告警；
- 无 dev 依赖组。

**影响**：不同时间/不同机器安装同一版本 wheel 可能得到不同依赖行为；构建后端升级后可能突然无法构建。

**建议**：

- 文档与 CI 统一 `uv sync --frozen`；
- 把 `uv_build` 约束更新到当前工具链（例如 `>=0.13.6` 或 `~=0.13`），并在 CI 中锁定 uv 版本；
- 增加 `[dependency-groups] dev = [...]`；
- 对关键依赖（`langchain*`、`langgraph`）考虑上限或至少在 README 声明“以 uv.lock 为准”；
- 在 CI 加 `uv lock --check`（或 `uv sync --frozen` 间接验证）。

**工作量**：1–2 小时。

#### P1-4　缺少静态检查、格式化与 pre-commit

**证据**：`pyproject.toml` 无 ruff/mypy/pyright/pytest 配置；仓库无 `.pre-commit-config.yaml`；`# noqa: BLE001` 没有对应规则执行。

**建议**：

- 引入 ruff（lint + format），先选 `E/F/I/B/UP/SIM/RUF`，对现有代码跑一遍基线；
- 引入 pyright（`basic` 起步）或 mypy；先只检查 `src/fastrag`，逐步收紧；
- pre-commit 运行 ruff + 末尾换行/文件大小检查；
- CI 强制 `ruff check`、`ruff format --check`、pyright；
- 在 README 写清“提交前跑什么”。

**工作量**：半天（含修基线）。

#### P1-5　打包元信息与发布流程不完整

**证据**：

- wheel `METADATA` 的 `Summary` 仍是 `Add your description here`（`pyproject.toml:4`）；
- 无 classifiers、keywords、`[project.urls]`；
- 无 `py.typed`，类型提示对下游不可见；
- 版本静态 `0.1.0`，无 tag、无 CHANGELOG、无 `--version`；
- 无 release workflow。

**建议**：

- 填写真实描述、classifiers、项目主页/仓库/Issue 链接；
- 增加 `src/fastrag/py.typed` 并在 wheel 内容测试中断言存在；
- 用 `importlib.metadata.version("fastrag")` 实现 `fastrag --version`；
- 用 tag 驱动版本（或至少 release 时同步 pyproject/CHANGELOG）；
- 建立 GitHub Release：CI 构建 wheel/sdist 并附产物哈希。

**工作量**：2–3 小时。

#### P1-6　LLM 调用没有显式超时，重试策略可能叠加

**证据**：

- `build_llm`（`llm.py:47-54`）只传 `base_url/api_key/temperature/extra_body`，没有 `timeout`/`max_retries`；
- `_call_with_retries`（`rag_graph.py:281-304`）在应用层重试；
- embedding/reranker 都有显式 timeout 配置，唯独 LLM 没有。

**影响**：外部 LLM 卡住时可能长时间挂起；SDK 自身重试与应用层重试叠加，放大延迟与费用；与“一次外部故障不阻断交互”的目标不一致。

**建议**：

- 在 `[llm.*]` 增加 `timeout`（秒）与可选的 `max_retries`，明确“应用层重试 vs SDK 重试”只保留一层；
- 把重试/超时/备用模型策略抽成公共函数，route 与 answer 共用；
- 记录每次调用的耗时与重试次数，纳入日志/指标。

**工作量**：半天。

#### P1-7　可观测性不足：print 驱动、无日志级别、无结构化输出

**证据**：库函数直接 `print`（`embed_chunks.py:89`、`rag_graph.py:210` 等）；`\r` 进度；无 `--quiet`/`--log-level`/`--json`；warnings 全局屏蔽（`rag_graph.py:367-371`）。

**建议**：

- 库层改用 `logging.getLogger("fastrag.*")`，CLI 层决定 handler/level；
- 进度输出到 stderr，且 TTY 判断后再用 `\r`；
- 增加 `--quiet`、`--log-level`、`--json`；
- 为 `ask` 输出结构化摘要：`elapsed_ms`、`need_search`、`retrieved`、`rerank_degraded`、`retries`；
- 不要全局屏蔽警告；如必须屏蔽，限定到具体调用点并记录原因。

**工作量**：1 天。

#### P1-8　可测试性/可复用性受全局状态影响

**证据**：

- `get_pipeline` 使用无 key 的 `@lru_cache(maxsize=1)`（`rag_graph.py:358-361`），不同配置连续调用会拿到第一次的 Pipeline；
- `load_env` 直接修改 `os.environ`（`config.py:245-255`），测试之间会互相污染；
- `cli._main` 改写 `sys.argv` 并写 `os.environ["FASTRAG_CONFIG"]`（`cli.py:124-128`），多次调用会泄漏状态；
- 各阶段 `main()` 内部自行 `load_config()`（`prepare.py:36-48`），无法注入配置；
- `ask` 没有“一次问答”的库函数，只有交互循环。

**建议**：

- `get_pipeline(config_path)` 以配置路径为 key，或提供 `reset_pipeline()`；
- `load_env(path) -> dict` 返回解析结果，由调用方决定是否注入环境；
- `main(argv: list[str] | None = None)`，不在库层改写全局 `sys.argv`；
- 阶段函数接受 `Config`，`main()` 只负责 I/O；
- 抽出 `answer_once(question, pipeline)`，供 CLI、HTTP、评估复用。

**工作量**：1 天。

#### P1-9　产物 provenance 与 schema 版本缺失

**证据**：manifest/meta 字段见 `embed_chunks.py:118-127`、`keyword_search.py:49-55`，没有包版本、git sha、配置/提示词哈希、库版本、schema 版本。

**建议**：

```json
{
  "schema_version": 1,
  "artifact": "embeddings",
  "package_version": "0.1.0",
  "git_sha": "ed0c439",
  "config_sha256": "...",
  "prompts": {"answer.txt": "...", "search_plan.txt": "..."},
  "chunks_sha256": "...",
  "model": "...",
  "dimensions": 2560,
  "count": 2625
}
```

加载时对 `schema_version` 做兼容处理；排错时能回答“这份产物从哪来”。

**工作量**：半天。

#### P1-10　向量复用只校验条数，不校验加载后的形状/类型

**证据**：`ensure_vectors`（`embed_chunks.py:152-161`）在 manifest 匹配后直接 `np.load` 返回；只比较 `manifest["count"] == len(chunks)`，没有校验 `array.shape == (count, dimensions)`、`dtype == float32`；`np.load` 也没有显式 `allow_pickle=False`。

**影响**：`.npy` 被截断/替换/手工修改但 manifest 未变时，`search` 会用错误的行数与 `chunks` 对齐，产生静默错配。

**建议**：

- 加载后校验 `shape`、`dtype`、有限值（NaN/Inf）；
- 显式 `allow_pickle=False`；
- 失败时提示“向量文件损坏，请重跑 prepare”。

**工作量**：1–2 小时。

#### P1-11　文档/DX 缺少开发与排错路径

**证据**：README 无测试/lint/发布说明；无项目结构；无排错章节；`cp` 对 Windows 不友好；`bot-reviews/` 无索引；`fastrag ask` 只能交互；`--config` 位置限制。

**建议**：

- README 增加“开发”章节：`uv sync --frozen --group dev`、`pytest`、`ruff`、`pyright`、`uv build`、跨平台提示；
- 增加 `docs/architecture.md`、`docs/troubleshooting.md`、`docs/reviews/`（把 `bot-reviews/*.md` 迁入并加索引）；
- `fastrag ask "问题"` 支持一次性问答，`--json` 支持脚本；
- `--config` 在所有位置都能解析，或在错误信息中提示“请放到子命令之前”；
- 修正 `fastrag search --help` 的 usage/prog。

**工作量**：半天到 1 天。

#### P1-12　没有评估/基准，无法量化检索与生成改动

**证据**：无 `eval/`、无问题集、无 `recall@k`/MRR、无 token/成本统计；默认 `temperature=0.7`（`fastrag.toml` 的 `[llm.route]`/`[llm.answer]`）。

**建议**：

- 建 20–50 条 `eval/questions.jsonl`（问题 + 期望 source），跑 `recall@5`/MRR；
- 对 `rrf` vs `rerank`、`top_n`、`candidates`、`chunk_size` 做基线；
- 评估时把 `temperature` 设为 0（或单独 profile），并固定随机种子（如模型支持）；
- CI 只跑离线检索评估，不依赖真实 LLM；LLM 评估作为手动/定时任务。

**工作量**：1–2 天。

### P2（可选改进）

| 编号 | 问题 | 证据 | 建议 |
| --- | --- | --- | --- |
| P2-1 | `.gitattributes` 缺失 | 仓库无该文件，本机 `core.autocrlf=true` | 明确文本/二进制规则，配合 P0-2 |
| P2-2 | `.gitignore` 覆盖不全 | 只有 `.venv/ data/ dist/ __pycache__/ .env` | 补 `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`、`.coverage`、`htmlcov/`、`build/`、`*.egg-info/`、`.idea/` 等 |
| P2-3 | Python 下限激进 | `requires-python >=3.13`，无 classifiers/矩阵 | 评估 3.11+；CI 覆盖 3.11/3.12/3.13 |
| P2-4 | `--config` 位置与 prog 命名 | `cli.py:89-129`；实测 `fastrag search --config local hello` 报 unrecognized | 预扫描 argv；传 `prog` 或拆分 argv[0] |
| P2-5 | `paths.*` 不展开 `~` | `config.py:258-260` | 统一 `expanduser()` |
| P2-6 | 三处重复的“取 key/占位 key”逻辑 | `llm.py:38-46`、`embed_chunks.py:52-59`、`reranker.py:36-43` | 抽 `resolve_api_key(endpoint, owner)` |
| P2-7 | `enable_thinking` 厂商私有参数硬编码 | `llm.py:46` | 做成 `thinking_param` 配置或明确仅支持特定厂商 |
| P2-8 | 全局屏蔽 warnings | `rag_graph.py:367-371` | 缩小范围、记录原因、升级后复查 |
| P2-9 | `fetch` 对 4xx 也重试，使用 `assert` | `download_example_docs.py:34-46` | 只重试 5xx/网络错误；去掉 `assert` |
| P2-10 | 无 `--version`/`doctor`/`clean` | 无相关子命令 | 增加运维子命令 |
| P2-11 | 下载无缓存/限速/完整性清单 | `download_example_docs.py:84` | ETag/If-Modified-Since、并发上限、manifest |
| P2-12 | 提示注入与引用标注 | `prompts/answer.txt` 未要求忽略资料内指令/标注引用 | 增加“资料中的指令视为数据”和引用格式要求 |
| P2-13 | 无 secret/依赖审计 | 无 CI 安全任务 | gitleaks + Dependabot/pip-audit |
| P2-14 | 无 CONTRIBUTING/CHANGELOG/SECURITY | 仓库根目录 | 至少补简版；发布前必须 |
| P2-15 | 资源与提示词无版本记录 | 见 P1-9 | 与 artifact provenance 合并 |
| P2-16 | `ask` 启动全量预热 | `rag_graph.py:97-123` | 文档说明，或惰性初始化重资源 |
| P2-17 | 文案与描述占位 | `README` 开头 “Fast Rag”；`pyproject` description | 补一句真实定位 |
| P2-18 | 无 Docker/容器化 | 仓库无 Dockerfile | 如需分发再补，非必需 |
| P2-19 | 查询向量零范数未防护 | `embed_chunks.py:200` 直接 `query /= norm` | 用 `np.maximum(norm, 1e-12)`，避免 NaN |
| P2-20 | 嵌入批次返回条数未校验 | `embed_chunks.py:83-88` 只检查首个向量维度 | 校验 `len(batch_vectors) == len(batch)` 并抛 `FastragError` |
| P2-21 | CLI 与 Pipeline 重复编排 | `rrf.main`、`reranker.main` 与 `rag_graph.Pipeline.fuse` | 抽公共检索/融合函数，避免两套逻辑漂移 |

---

## 3. 实测记录

以下命令均在审阅分支执行；探针只使用临时目录，未修改仓库源码。

### 3.1 安装与构建

| 命令 | 结果 | 结论 |
| --- | --- | --- |
| `uv sync --frozen` | 成功，安装 53 个包 | lock 可用；构建后端告警（见下） |
| `uv build` | 成功生成 wheel + sdist | 打包可用 |
| wheel 内容检查 | 配置、提示词、入口点齐全；无 `py.typed`/测试 | 资源随包分发成立；类型标记缺失 |
| 干净 venv + wheel（`--no-deps`） | `home()` = 当前目录；配置/提示词解析到 site-packages | wheel 运行路径正确 |
| 干净 venv 完整安装 wheel | 解析到 `openai 3.41.0` / `langchain-openai 1.7.3`，与 lock 不同 | 依赖漂移，见 P1-3 |

构建告警原文：

```text
warning: build_system.requires = ["uv-build>=0.12.23,<0.13.0"]
         does not contain the current uv version 0.13.6
```

### 3.2 CLI 行为

| 命令 | 结果 |
| --- | --- |
| `uv run fastrag --help` | 正常列出 9 个子命令 |
| `uv run fastrag search --help` | usage 显示 `fastrag search query [-h] [-k K] text`（prog 瑕疵） |
| `uv run fastrag chunk`（无 docs） | `错误: 文档目录不存在...`，退出码 1 |
| `uv run fastrag search hello`（无索引） | `错误: 索引不存在...`，退出码 1 |
| `uv run fastrag --config local chunk` | 正确解析包内 `local` 配置 |
| `uv run fastrag --config does-not-exist chunk` | `错误: 配置文件不存在`，退出码 1 |
| `uv run fastrag search --config local hello` | `unrecognized arguments: --config hello`（位置限制） |
| `uv run python -m pytest` | 退出码 1：`No module named pytest` |
| `uv run python -m compileall -q src` | 退出码 0（语法层面可编译） |

### 3.3 配置非法输入探针

| 输入 | 结果 |
| --- | --- |
| 删除 `[reranker]` | `KeyError: 'reranker'` |
| 删除 `[paths]` | `KeyError: 'paths'` |
| endpoint 缺 `base_url` | `KeyError: 'base_url'` |
| route 缺 `model` | `KeyError: 'model'` |
| `docs_dir = 1` | `TypeError: unsupported operand type(s) for /: 'WindowsPath' and 'int'` |
| `tokenizer = "nope"` | 加载成功（未校验） |
| `base_url = 123` | 加载成功（未校验） |
| 未知键 `unknown_option = 1` | 加载成功，静默忽略 |
| `top_n = 2.5` | 加载成功，类型为 `float` |
| `fusion.method = "bogus"` | `FastragError`（正确） |
| 非法正则 | `FastragError`（正确） |
| stopwords 列表含非字符串 | `FastragError`（正确） |

### 3.4 指纹与换行

| 实验 | 结果 |
| --- | --- |
| Windows 文本模式写入一行 JSON | 字节为 `b'{"id": "a"}\r\n'` |
| LF 文件 sha256 | `f04e3940ef65049c76c521e2a173b54a16e599fde7e2edd0b76c3a100f1ddb65` |
| CRLF 文件 sha256 | `a9e0cd4a8f523a76cea35af8a6f6050dc2515e1c52f34216e92b2b6263855960` |
| 两者相同？ | 否 |

### 3.5 空语料与旧产物

| 实验 | 结果 |
| --- | --- |
| 空 docs 目录 + 已存在旧 `chunks.jsonl`，运行 `python -m fastrag.chunk_docs` | 打印“未生成 chunk 文件”，退出码 0，旧文件保留且内容不变 |
| 随后 `prepare` 的代码路径 | 继续 `load_chunks` → 向量 → 索引（旧语料） |

### 3.6 仓库状态

| 项目 | 结果 |
| --- | --- |
| 当前提交 | `ed0c439`（2026-10-27） |
| 提交数 | 77 |
| tag | 无 |
| 远程分支 | `main`、`dev`、多个 `feat/*`、`fix/*`、`perf/*`、`bench` |
| `core.autocrlf` | `true` |
| 硬编码密钥扫描 | 未发现 `sk-...` |
| `git status` | 审阅前干净；审阅过程中生成的 `.venv/`、`dist/` 均被 `.gitignore` 忽略 |

### 3.7 审阅限制

- 未提供 `SILICONFLOW_API_KEY`，未执行真实嵌入、重排、LLM 与端到端问答；
- 仓库不包含 `data/`，未验证真实语料上的 recall/延迟；
- 网络较慢，`uvx ruff` 下载超时，静态检查结论来自人工阅读与配置缺失事实，而非工具输出；
- wheel 完整依赖安装因网络超时未完成；依赖漂移结论来自解析器已给出的版本信息。

---

## 4. 与历史审阅的关系

### 4.1 已修复（本轮确认）

| 历史问题 | 当前状态 | 证据 |
| --- | --- | --- |
| 配置数值边界校验缺失（第 2 轮 H1） | 已补大部分 | `config.py:317-421` 覆盖 chunking/embedding/retrieval/fusion/bm25/reranker/temperature |
| `-k 0` / `-n 0` 回退默认值（第 3 轮 M1） | 已修 | `keyword_search.py:152` 用 `is None`；`rrf.py:71`、`reranker.py:88` 用 `max(..., 0)` |
| 空语料 IndexError（第 2 轮 M2） | 已修 | `keyword_search.py:117-118`、`embed_chunks.py:196-197` |
| `answer` 无重试/备用模型（第 2 轮 M4） | 已修 | `rag_graph.py:252-304` + `[llm.answer.fallback]` |
| `route` 无重试（第 3 轮 M3） | 已修 | `rag_graph.py:125-158` |
| `load_env` 解析不健壮（第 3 轮 M4） | 已修 | `config.py:222-242` |
| `reranker.timeout` / `temperature` / `bm25.method` 未校验（第 3 轮 M5） | 已修 | `config.py:384-404` |
| `chunking` 元数据语义模糊（第 2 轮 M3） | 已明确 | `chunks.py:54-60`、`embed_chunks.py:124-125` |
| `ensure_vectors` 重复计算 sha256（第 2 轮 L7） | 已修 | `save_vectors(..., digest)` |
| 重排失败降级 RRF（第 2 轮） | 已修 | `rag_graph.py:204-218` |
| 资源随包分发 / CLI 入口（第 1 轮 P0-1） | 已修 | wheel 内容实测 |

### 4.2 仍未关闭

| 历史问题 | 当前状态 | 本轮编号 |
| --- | --- | --- |
| 无 tests / lint / CI（第 2、3 轮 L9） | 仍未关闭 | P0-1、P1-4 |
| `get_pipeline` 的 `lru_cache` 无 key（第 3 轮 L6） | 仍在 | P1-8 |
| `enable_thinking` 厂商耦合（第 3 轮 L3） | 仍在 | P2-7 |
| `sys.argv[0]` 改写 / `--config` 位置（第 3 轮 L4/L5） | 仍在 | P2-4 |
| `download` 无整体限速（第 3 轮 L8） | 仍在 | P2-11 |
| `requires-python >=3.13`（第 3 轮 L9） | 仍在 | P2-3 |
| `prepare` 重复加载配置（第 3 轮 L7） | 仍在 | P1-8 |
| 查询向量除零（第 3 轮 L1） | 仍在 | P2-19 |
| 每批嵌入返回条数未校验（第 3 轮 L2） | 仍在 | P2-20 |
| `description` 占位符（第 3 轮 M2） | 仍在 | P1-5 |

---

## 5. 建议的工程化门禁

### 5.1 PR 必须通过的检查（Definition of Done）

每个 PR 至少满足：

- [ ] `uv run ruff check .` 通过；
- [ ] `uv run ruff format --check .` 通过；
- [ ] `uv run pyright`（或 mypy）通过；
- [ ] `uv run pytest` 通过，核心模块覆盖率 ≥ 70%；
- [ ] `uv build` 成功；
- [ ] wheel 冒烟：安装后 `fastrag --help` 成功，且资源文件存在；
- [ ] 新增/修改的行为有测试；修复 bug 时先写失败用例；
- [ ] 用户可见错误是 `FastragError`，不是 traceback；
- [ ] 库层不新增裸 `print`（进度/日志走 `logging`）；
- [ ] 涉及产物格式时，更新 schema/provenance 并加兼容测试；
- [ ] 涉及配置项时，同步 README 与非法值测试。

### 5.2 CI 建议

```yaml
name: ci

on:
  push:
    branches: [main, dev]
  pull_request:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
        python: ["3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv python install ${{ matrix.python }}
      - run: uv sync --frozen --group dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright
      - run: uv run pytest --cov=fastrag --cov-report=term-missing
      - run: uv build
      - run: uv run fastrag --help
```

要点：

- **Windows + Linux 双平台**：直接暴露 P0-2 这类换行问题；
- 使用 `--frozen`：保证 CI 与 lock 一致；
- 构建与 CLI 冒烟：防止资源再次漏包；
- 真实 API 调用不进入 CI；需要时用 `@pytest.mark.integration` + 环境变量守卫。

### 5.3 `pyproject.toml` 建议补丁（示意）

```toml
[project]
name = "fastrag"
version = "0.1.0"
description = "本地文档混合检索（BM25 + 向量）问答"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
license-files = ["LICENSE"]
keywords = ["rag", "bm25", "vector-search", "langgraph"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.urls]
Repository = "https://github.com/Ashitaka-Laputa-IV/FastRag"
Issues = "https://github.com/Ashitaka-Laputa-IV/FastRag/issues"

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-cov>=6",
    "ruff>=0.9",
    "pyright>=1.1.400",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.pyright]
include = ["src", "tests"]
typeCheckingMode = "basic"
```

注意：`uv_build` 的版本范围要单独更新（见 P1-3），不要直接照抄。

### 5.4 原子写与规范化换行（示意）

```python
import os
import pathlib


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_bytes_atomic(path: pathlib.Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
```

`chunks.jsonl`、`manifest.json`、`meta.json` 都走这条路径；BM25 索引先写临时目录，再整体替换。

### 5.5 配置校验（示意）

```python
def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise FastragError(f"缺少配置段 [{name}]")
    return value


def _int(owner: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FastragError(f"{owner} 必须是整数，当前为 {value!r}")
    if value < minimum:
        raise FastragError(f"{owner} 必须 >= {minimum}，当前为 {value!r}")
    return value
```

所有 `raw["..."]` 改成 `_section`/`_field`；`top_n`、`candidates`、`batch_size`、`chunk_size`、`rrf_k` 等一律用 `_int`。

### 5.6 测试用例清单（建议首批）

| 层 | 对象 | 用例 |
| --- | --- | --- |
| 单元 | `chunks_sha256` | 相同字节同指纹；LF/CRLF 在规范化后一致；缺失文件友好报错 |
| 单元 | `strip_boilerplate` | 匹配剥离；不匹配原样；偏移回加后 `text == src[start:end]` |
| 单元 | `rrf_fuse` | 单路/双路；权重；并列；重复 id（当前会重复计分，明确期望）；`k <= 0`（当前 `k=-1` 抛 `ZeroDivisionError`） |
| 单元 | `_merge` | 去重且保序；空输入 |
| 单元 | `_env_value` | 行尾注释；引号内 `#`；成对引号；未闭合引号；空值 |
| 单元 | `load_config` | 缺 section/字段；`bool` 冒充数字；`top_n=2.5`；非法枚举；非法正则；未知键；`~` 路径 |
| 单元 | `search`（BM25） | 空语料；`k=0`；`min_score` 过滤；全零分 |
| 单元 | `search`（向量） | 空向量；`k=0`；零查询向量（应无 NaN）；维度不一致 |
| 单元 | `embed_chunks` | 每批返回条数不符；维度不符；复用/重建分支 |
| 单元 | `plan_search` | 正常 JSON；模型漏字段；重试耗尽；fallback；结构化输出异常 |
| 单元 | `_answer_text` | 主模型成功；重试后成功；fallback 成功；全部失败返回空串 |
| 集成 | 小语料 | 2–3 篇 fixture 跑 chunk → fake embedding → index → search |
| 集成 | 空语料 | `prepare` 必须失败/不得复用旧产物 |
| 集成 | 重排降级 | 假 reranker 抛错 → RRF 结果 |
| CLI | 命令 | `--help`；`--config` 前置/后置；退出码 0/1；`ask` 一次性模式（新增后） |
| 跨平台 | 指纹 | 同一 fixture 在 Windows/Linux 得到同一 digest |

---

## 6. 整改路线图

### 第 1 批：半天内（低成本、立刻降低风险）

- [ ] `pyproject.toml`：填 `description`、classifiers、`[project.urls]`、dev 依赖组；
- [ ] 更新 `uv_build` 版本范围，消除构建告警；
- [ ] README：改 `uv sync --frozen`；补 Windows 的 `Copy-Item`/`copy` 写法；
- [ ] `.gitattributes` + 补全 `.gitignore`；
- [ ] 新增 `py.typed`（确认 wheel 中包含）；
- [ ] 把 `bot-reviews/` 迁到 `docs/reviews/` 并加索引；
- [ ] `fastrag --version`。

### 第 2 批：1–2 天（建立安全网）

- [ ] `tests/` 骨架 + 上表前 10 组用例；
- [ ] `[tool.ruff]`、`[tool.pyright]`、`[tool.pytest.ini_options]`；
- [ ] `.pre-commit-config.yaml`；
- [ ] `.github/workflows/ci.yml`（Linux + Windows，lint/type/test/build/smoke）；
- [ ] 覆盖率门槛（先 60%–70%，只统计核心模块）。

### 第 3 批：2–4 天（数据与错误治理）

- [ ] 统一产物换行 + 原子写 + `filelock`；
- [ ] 空文档目录 fail-fast；`prepare` 校验本次切块结果；
- [ ] 下载失败/0 URL 非零退出；`docs-manifest.json` + `--prune` + `--dry-run`；
- [ ] 配置错误全部转 `FastragError`；补整型/枚举/URL/路径校验；
- [ ] `fastrag doctor`；
- [ ] 向量加载后 shape/dtype/NaN 校验；
- [ ] manifest schema + provenance（包版本、git sha、config/prompt 哈希）。

### 第 4 批：1 周（可维护与可发布）

- [ ] `logging` 改造 + `--quiet`/`--log-level`/`--json`；
- [ ] `ask "问题"` 一次性模式 + `answer_once()` 库函数；
- [ ] `get_pipeline` 缓存 key / 可重置；`load_env` 返回 dict；阶段函数接受 `Config`；
- [ ] LLM 超时/重试统一；
- [ ] `eval/` 问题集 + `recall@5`/MRR 基线；
- [ ] GitHub Release 流程 + CHANGELOG；
- [ ] gitleaks / Dependabot / pip-audit；
- [ ] 文档：架构、排错、开发、发布。

---

## 7. 附录

### 7.1 模块工程化关注点

| 模块 | 行数 | 工程化关注点 |
| --- | ---: | --- |
| `chunk_docs.py` | 121 | 文本模式写 JSONL（P0-2）；空记录保留旧文件（P0-3）；非原子写（P0-5） |
| `chunks.py` | 68 | 指纹设计正确；需跨平台测试（P0-2） |
| `cli.py` | 133 | `sys.argv` 改写、`--config` 位置、prog 命名（P1-11/P2-4） |
| `config.py` | 502 | 裸索引异常（P1-1）；整型/枚举/URL/路径校验（P1-2）；配置获取与校验耦合，可考虑 schema 模型 |
| `download_example_docs.py` | 100 | 失败不反映退出码、无清理、无清单、无限速（P0-4/P2-11）；`assert`（P2-9） |
| `embed_chunks.py` | 232 | 非原子写（P0-5）；复用不校验 shape/dtype（P1-10）；批次条数未校验（P2-20）；查询零向量（P2-19）；库层 print（P1-7） |
| `errors.py` | 24 | 约定好；可扩展退出码与 `--debug` |
| `keyword_search.py` | 160 | 索引校验严谨；非原子写（P0-5）；CLI 输出无 JSON（P1-7） |
| `llm.py` | 70 | 无 timeout/max_retries（P1-6）；`enable_thinking` 厂商耦合（P2-7） |
| `prepare.py` | 54 | 重复加载配置（P1-8）；不校验切块是否真的发生（P0-3） |
| `rag_graph.py` | 420 | 缓存无 key（P1-8）；全局 warnings 屏蔽（P2-8）；交互 I/O 与图耦合；无一次性问答（P1-11） |
| `reranker.py` | 111 | 结构干净；CLI 与 Pipeline 重复编排（P2-21） |
| `rrf.py` | 109 | 实现正确；`k <= 0` 未防御；CLI 重复编排（P2-21） |
| `resources/` | — | 配置/提示词分发正确；提示词无版本记录（P1-9） |

### 7.2 配置项工程化检查表

| 配置项 | 当前校验 | 缺口 |
| --- | --- | --- |
| `paths.*` | 无 | 非字符串、`~` 未展开 |
| `endpoints.*.base_url` | 无 | 类型、协议 |
| `endpoints.*.api_key_env` | 无 | 类型（空=本地是约定，需说明） |
| `chunking.tokenizer` | 无 | 非法编码名到切块才报错 |
| `chunking.chunk_size/overlap` | 数值 + 边界 | 未限制整型 |
| `chunking.include_delim` | 无 | 枚举 |
| `chunking.boilerplate_pattern` | 正则合法性 | 正确 |
| `embedding.batch_size/dimensions/timeout/max_retries` | 数值 + 边界 | 未限制整型 |
| `embedding.model` | 无 | 空字符串 |
| `llm.*.model/provider` | 无 | 空字符串 |
| `llm.*.temperature` | 0–2 | 正确 |
| `llm.*.thinking` | 无 | 类型 |
| `llm.*.retries` | 非负整数 | 正确 |
| `llm.*.fallback` | 递归校验 | 正确 |
| `reranker.model/timeout` | timeout 已校验 | model 空字符串 |
| `bm25.stopwords` | 语言名/字符串列表 | 正确 |
| `bm25.method` | 枚举 | 正确 |
| `bm25.min_score` | 数值 | 可接受 |
| `retrieval.*` | 数值 > 0 | 未限制整型；candidates 与 top_n 关系 |
| `fusion.method` | 枚举 | 正确 |
| `fusion.weights` | 数组长度/正数 | 正确（rerank 模式下权重无效，需文档说明） |
| `fusion.rrf_k` | 数值 > 0 | 未限制整型 |

### 7.3 审阅执行记录

```text
uv sync --frozen
uv build
wheel/sdist 内容检查（zipfile/tarfile）
干净 venv wheel 安装与 home()/resources 探针
uv run fastrag --help / search --help
uv run fastrag chunk / search / --config ...
配置非法输入探针（12 组）
LF/CRLF 指纹探针
空 docs + 旧 chunks 探针
float top_n 探针
git remote/tag/branch/core.autocrlf 检查
git grep 硬编码密钥
```

本轮未修改源码；新增文件仅 `docs/engineering.md`。
