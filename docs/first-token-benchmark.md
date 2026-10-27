# 首 token 延迟基准：全云 / 全本地 / 混合

> 本报告只统计**未被限流、未被拥塞拖慢**的样本；被厂家限流（429）和明显拥塞的请求已剔除，剔除规则与数量见第 2 节。

- 日期：2026-10-27
- 分支：`bench`
- 主指标：**用户首字** —— 从发出问题到看见第一个回答字符（token）的时间（规划 + 召回 + 融合 + 回答首 token）
- 样本：8 个问题 × 每个问题 5 次 = 每方案最多 40 次

## 1. 三种方案

| 环节 | 全云 | 全本地 | 混合 |
| --- | --- | --- | --- |
| 检索规划 | SiliconFlow `DeepSeek-V4.1-Flash`（关思考） | 本地 `qwen3-4b-instruct` | 本地 `qwen3-4b-instruct` |
| 回答 | SiliconFlow `DeepSeek-V4.1-Flash`（关思考） | 本地 `qwen3-4b-instruct` | SiliconFlow `DeepSeek-V4.1-Flash`（关思考） |
| 嵌入 / 召回 | SiliconFlow `Qwen3-Embedding-4B`（2560 维） | 本地 `bge-m3`（1024 维） | 本地 `bge-m3`（1024 维） |
| 融合 | RRF（k=60，权重 1.1 / 1.0） | 同左 | 同左 |
| 配置 | `config/fastrag.cloud.toml` | `config/fastrag.local.toml` | `config/fastrag.hybrid.toml` |

- 三个方案的语料（2625 个 chunk）、切块、BM25 索引、`top_n = 5` 完全一致，只替换模型与嵌入。
- 云端模型全部关闭思考（`enable_thinking = false`），避免推理 token 干扰首字延迟。
- 本地服务：`bge-m3`（`127.0.0.1:9931`）、`qwen3-4b-instruct`（`127.0.0.1:9938`），llama.cpp。

**问题集**（LangChain / LangGraph 官方文档）：

1. LangGraph 中如何实现 human-in-the-loop？
2. 如何让 LangGraph 的图流式输出 token？
3. LangChain 的 Agent 如何调用工具？
4. LangGraph 如何用 checkpointer 持久化会话状态？
5. LangChain 如何搭建 RAG 检索链？
6. LangGraph 的条件边（conditional edges）怎么用？
7. LangChain 的文本分割器怎么选？
8. LangGraph 如何实现子图与并行分支？

整轮开始前用第一个问题预热 1 次（不计入统计），之后每个问题跑 5 次。

## 2. 数据清洗

本次测量窗口内 SiliconFlow 出现限流与拥塞，因此先剔除异常样本再统计：

| 方案 | 原始尝试 | 429 失败 | 拥塞样本 | **有效样本** |
| --- | ---: | ---: | ---: | ---: |
| 全本地 | 40 | 0 | 0 | **40** |
| 混合 | 40 | 4 | 8 | **28** |
| 全云 | 40 | 3 | 18 | **19** |

剔除规则（毫秒）：

- 请求失败（`OpenAIRateLimitError`，厂家返回 429「System is too busy」）。
- 远端阶段出现明显尖峰：检索规划 > 4000、召回 > 1000、回答首 token > 4000。

阈值取在本次数据的自然断点上。例如全云召回的正常样本是 132–329 ms，之后直接跳到 1129 ms；混合的回答首 token 正常到 3479 ms，之后直接跳到 5714 ms。本地三个阶段最大只有 1256 / 56 / 1222 ms，所以一条都没被剔除。

> 为了透明，原始未过滤数据的中位数也保留：全云 4170 ms、混合 2405 ms、全本地 1918 ms。清洗后全云从 4170 降到 2360，说明它的原始数字主要被拥塞拉高。

## 3. 总体结果

| 方案 | 有效样本 | 中位数 | 平均 | P90 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **全本地** | 40 | **1918 ms** | 1896 ms | **2237 ms** | 891 ms | 2370 ms |
| 混合 | 28 | 2198 ms | 2366 ms | 3274 ms | 1199 ms | 4641 ms |
| 全云 | 19 | 2360 ms | 2556 ms | 3021 ms | 1865 ms | 4236 ms |

- 三种方案在「正常服务」下的中位数都落在 **1.9–2.4 秒**，差距不到 0.5 秒。
- 全本地的 P90 最低（2237 ms），且最大值只有 2370 ms，没有长尾。
- 混合的 P90（3274 ms）略高于全云，因为远端回答偶尔仍会慢到 3.5 秒级。
- 全云只有 19 个有效样本，P90 的参考价值低于另外两组。

### 3.1 阶段拆解（中位数）

| 阶段 | 全云 | 全本地 | 混合 |
| --- | ---: | ---: | ---: |
| 检索规划 | 1628 ms | 931 ms | 895 ms |
| 召回 | 168 ms | 37 ms | 34 ms |
| 融合 | 0.1 ms | 0.1 ms | 0.1 ms |
| 回答首 token | 504 ms | 1047 ms | 1255 ms |
| **用户首字** | **2360 ms** | **1918 ms** | **2198 ms** |

> 各阶段中位数不能直接相加：尖峰落在不同运行上，用户首字是逐次计时后再取中位数。

- 全云的前端最慢（规划 1.63 s + 召回 0.17 s），但回答最快（0.50 s）。
- 全本地的回答最慢（1.05 s），但前端只要 0.97 s，整体最短。
- 混合把前端压到 0.93 s，回答交给云端；同一个 DeepSeek 在混合里是 1.26 s、在全云里是 0.50 s，差异来自送入的上下文不同与测量窗口波动，不是模型变了。

### 3.2 分问题（用户首字中位数）

| 问题 | 全云 | 全本地 | 混合 |
| --- | ---: | ---: | ---: |
| Q1 human-in-the-loop | 2174 ms | 2226 ms | 3453 ms |
| Q2 流式输出 token | 2768 ms | 1824 ms | 2134 ms |
| Q3 Agent 调用工具 | 2465 ms | 1920 ms | 2330 ms |
| Q4 checkpointer 持久化 | 4236 ms | 925 ms | 2118 ms |
| Q5 搭建 RAG 检索链 | 2582 ms | 1795 ms | 2460 ms |
| Q6 条件边 | 2249 ms | 1916 ms | 1828 ms |
| Q7 文本分割器 | 2483 ms | 1992 ms | 2560 ms |
| Q8 子图与并行分支 | 2217 ms | 2001 ms | 1801 ms |

- 全本地每题 0.9–2.2 秒，问题之间差异最小。
- 全云在 Q4 出现 4.2 秒（规划与召回都偏慢的那次），其余问题在 2.2–2.8 秒。
- 混合在 Q1 出现 3.5 秒，是远端回答首 token 偏慢的一次；其余与全本地接近。

## 4. 解读

1. **正常服务下，三种方案的用户首字都在 2 秒左右**：本地 1.92 s、混合 2.20 s、全云 2.36 s。差距主要来自前端（规划与召回），不是模型本身。
2. **全本地的优势是稳定性**：没有长尾，P90 与中位数只差 0.3 秒；代价是 4B 回答质量未评测。
3. **全云的代价是依赖**：40 次尝试里有 3 次 429、18 次被拥塞拖慢，接近一半的请求不达标。即使只看正常样本，它也没有明显快于混合或本地。
4. **嵌入本地化收益最大**：全云召回 168 ms、混合/本地 34–37 ms；更重要的是远端嵌入一旦拥塞就是数秒级，而本地始终 25–56 ms。
5. **回答模型不是瓶颈**：三者回答首 token 在 0.50–1.26 秒之间，链路前端的规划（0.9–1.6 秒）才是主要成本。

## 5. 建议

| 场景 | 选择 | 理由 |
| --- | --- | --- |
| 追求稳定与速度 | **全本地** | 中位数 1.92 s、P90 2.24 s、无长尾；需要接受 4B 的回答质量 |
| 需要云端回答质量 | **混合** | 中位数 2.20 s，回答沿用 DeepSeek；建议给回答加 3 秒超时并回退本地 |
| 全云作为默认 | 不推荐 | 正常时 2.36 s，但本次有近一半请求被限流/拥塞；除非能接受这种波动 |

- 下一步优化顺序：**本地化嵌入 → 本地化规划 → 回答侧超时/降级**。
- 若要继续用全云，至少把嵌入换到本地，并把 `max_retries` 提高到 4–5，同时给回答加超时回退。
- 所有关思考的结论只针对延迟，回答质量需要单独评测。

## 6. 复现

```bash
# 前置：本地服务在线
esl start bge-m3
esl start qwen3-4b-instruct

# 三个方案各跑 8 问题 × 5 次
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  uv run python -m fastrag.bench --config config/fastrag.local.toml  --runs 5
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  uv run python -m fastrag.bench --config config/fastrag.hybrid.toml --runs 5
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  uv run python -m fastrag.bench --config config/fastrag.cloud.toml  --runs 5
```

- 失败与拥塞样本的过滤规则：`plan > 4000`、`retrieve > 1000`、`generate > 4000`（毫秒），并剔除 429 记录。
- 清洗后的数据单独保存为 `*-clean.json`，可直接复算本报告的所有数字。

## 7. 数据留档

| 文件 | 内容 | sha256 |
| --- | --- | --- |
| `data/bench/ttft-2026-10-27-local-clean.json` | 全本地，40 条有效样本 | `93753007ae6b8477b55e8559271300bc66dc2bf675582039a998f4b8cea6bd3e` |
| `data/bench/ttft-2026-10-27-hybrid-clean.json` | 混合，28 条有效样本 | `0d1f17d5f5f97d212222ffdbda9df6b89bc4eedea1bc3cc019e8e6137c2256f8` |
| `data/bench/ttft-2026-10-27-cloud-clean.json` | 全云，19 条有效样本 | `827f7a1ad1ca0f3fb5c57ba6f73bd49049a45b89ce01027ea4c5673f29ef93ac` |
| `data/bench/ttft-2026-10-27-local.json` | 全本地原始 40 条 | `bf790f87580ad8ad2e3743185a44c8bcd8c2e1351dc85e3a4ec47d1513f76a5f` |
| `data/bench/ttft-2026-10-27-hybrid.json` | 混合原始 40 条（含 4 次 429） | `5e7c27684b591bf63ffc3606a6a24c2a224133ba8d49e2bf91af35b66a3a4410` |
| `data/bench/ttft-2026-10-27-cloud.json` | 全云原始 40 条（含 3 次 429） | `090b4bbc8fee209ce245036e5edd500a115e60f315460160df66b9e5638e6843` |

- 日志：`ttft-2026-10-27-{cloud,local,hybrid}.log`，与原始 JSON 同目录。
- `data/` 已被 `.gitignore` 忽略，明细只在本机；clean 文件由原始 JSON 按第 2 节规则过滤生成。

## 8. 局限

- 全云的有效样本只有 19 个（原始 37 个成功样本中被拥塞污染 18 个），P90 与均值不如另外两组可靠。
- 剔除阈值是事后根据本次数据的断点确定的，换一个测量窗口可能需要重新标定；因此报告同时给出了原始中位数。
- 每题 5 次，问题级结论只适合看量级。
- 只测到首 token，没有评估回答质量、完整回答耗时与幻觉率。
- 本地回答在重复同一问题时可能命中 llama.cpp 的 prompt cache（40 次里 3 次首 token < 100 ms），中位数影响很小。
- 本地向量（bge-m3 1024 维）与云端（2560 维）的检索质量未做量化对比。
- 同一台机器（Intel i7-11800H / 22 GB）；换机器、换量化都会变化。
