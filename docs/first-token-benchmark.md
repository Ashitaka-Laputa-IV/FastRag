# 首 token 延迟基准：全云 / 全本地 / 混合

> 只统计**未被限流、未被拥塞拖慢**的样本；剔除规则与数量见第 2 节。
> 全云方案的检索规划已从 DeepSeek 换成更快的 `Hunyuan-A13B-Instruct`，回答仍用 DeepSeek。

- 日期：2026-10-27
- 分支：`bench`
- 主指标：**首字输出** —— 从用户发出问题到看见第一个回答字符的时间。用户只感知这一个时间；规划、召回等阶段只用于解释它，不作为体验指标。
- 样本：8 个问题 × 每个问题 5 次 = 每方案最多 40 次

## 1. 三种方案

| 环节 | 全云 | 全本地 | 混合 |
| --- | --- | --- | --- |
| 检索规划 | SiliconFlow `Hunyuan-A13B-Instruct`（关思考） | 本地 `qwen3-4b-instruct` | 本地 `qwen3-4b-instruct` |
| 回答 | SiliconFlow `DeepSeek-V4.1-Flash`（关思考） | 本地 `qwen3-4b-instruct` | SiliconFlow `DeepSeek-V4.1-Flash`（关思考） |
| 嵌入 / 召回 | SiliconFlow `Qwen3-Embedding-4B`（2560 维） | 本地 `bge-m3`（1024 维） | 本地 `bge-m3`（1024 维） |
| 融合 | RRF（k=60，权重 1.1 / 1.0） | 同左 | 同左 |
| 配置 | `config/fastrag.cloud.toml` | `config/fastrag.local.toml` | `config/fastrag.hybrid.toml` |

- 为什么选 Hunyuan 做规划：直测模型首 token 为 230 ms，是云端最快档（DeepSeek 为 411 ms）。
- 三个方案的语料（2625 个 chunk）、切块、BM25 索引、`top_n = 5` 完全一致，只替换模型与嵌入。
- 云端模型全部关闭思考（`enable_thinking = false`）。
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

### 1.1 指标说明

- **体验指标只有「首字输出」**：用户从提问到看见第一个回答字符。`bench.py` 里的 `wall_ms` 就是它。
- **规划模型的 TTFT 不属于用户体验**：用户看不到规划何时开口，只会感受到它让首字变晚。
- 「检索规划」「召回」「回答首 token」是**内部诊断指标**，只回答「首字为什么慢」，不能单独拿来评价体验好坏。

## 2. 数据清洗

本次测量窗口内 SiliconFlow 仍有限流与拥塞，先剔除异常样本再统计：

| 方案 | 原始尝试 | 429 失败 | 拥塞样本 | **有效样本** |
| --- | ---: | ---: | ---: | ---: |
| 全本地 | 40 | 0 | 0 | **40** |
| 混合 | 40 | 4 | 8 | **28** |
| 全云（Hunyuan 规划） | 40 | 0 | 10 | **30** |

剔除规则（毫秒）：

- 请求失败（`OpenAIRateLimitError`，厂家 429「System is too busy」）。
- 远端阶段明显尖峰：检索规划 > 4000、召回 > 1000、回答首 token > 4000。

阈值取在本次数据的自然断点上：全云召回正常为 132–196 ms，拥塞样本是 1.1 s 以上；混合的回答首 token 正常到 3479 ms，之后直接跳到 5714 ms。本地三个阶段最大只有 1256 / 56 / 1222 ms，一条都没被剔除。

> 原始未过滤数据的中位数：全云 2591 ms、混合 2405 ms、全本地 1918 ms。

## 3. 总体结果

| 方案 | 有效样本 | 中位数 | 平均 | P90 | 最小 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **全本地** | 40 | **1918 ms** | 1896 ms | **2237 ms** | 891 ms | 2370 ms |
| 混合 | 28 | 2198 ms | 2366 ms | 3274 ms | 1199 ms | 4641 ms |
| 全云 | 30 | 2466 ms | 2550 ms | 2840 ms | 1967 ms | 4356 ms |

- 三种方案在「正常服务」下的中位数是 **1.9–2.5 秒**，全本地最快，全云最慢。
- 全本地的 P90 最低（2237 ms），最大值只有 2370 ms，没有长尾。
- 全云的 P90（2840 ms）反而低于混合（3274 ms）：它的抖动主要来自嵌入，剔除后较整齐；混合的抖动来自远端回答。

### 3.1 阶段拆解（中位数）

| 阶段 | 全云 | 全本地 | 混合 |
| --- | ---: | ---: | ---: |
| 检索规划 | 1702 ms | 931 ms | 895 ms |
| 召回 | 159 ms | 37 ms | 34 ms |
| 融合 | 0.1 ms | 0.1 ms | 0.1 ms |
| 回答首 token | 571 ms | 1047 ms | 1255 ms |
| **首字输出（体验）** | **2466 ms** | **1918 ms** | **2198 ms** |

> 各阶段中位数不能直接相加：尖峰落在不同运行上，首字输出是逐次计时后再取中位数。阶段耗时仅用于定位瓶颈，不单独代表体验。

- 全云的规划比本地慢约 0.8 秒，召回慢约 0.12 秒，回答快约 0.5–0.7 秒。
- 全本地的回答最慢（1.05 秒），但前端只要 0.97 秒，整体最短。
- 同样是 DeepSeek 回答，全云 0.57 秒、混合 1.26 秒，差异来自上下文不同与测量窗口波动，不是模型变了。

### 3.2 分问题（首字输出中位数）

| 问题 | 全云 | 全本地 | 混合 |
| --- | ---: | ---: | ---: |
| Q1 human-in-the-loop | 3066 ms | 2226 ms | 3453 ms |
| Q2 流式输出 token | 2395 ms | 1824 ms | 2134 ms |
| Q3 Agent 调用工具 | 2434 ms | 1920 ms | 2330 ms |
| Q4 checkpointer 持久化 | 2576 ms | 925 ms | 2118 ms |
| Q5 搭建 RAG 检索链 | 2658 ms | 1795 ms | 2460 ms |
| Q6 条件边 | 2279 ms | 1916 ms | 1828 ms |
| Q7 文本分割器 | 2413 ms | 1992 ms | 2560 ms |
| Q8 子图与并行分支 | 2353 ms | 2001 ms | 1801 ms |

- 全本地每题 0.9–2.2 秒，问题之间差异最小。
- 全云每题 2.3–3.1 秒，比较均匀；换掉规划模型后不再出现 4 秒级的单题。
- 混合在 Q1 出现 3.5 秒，是远端回答首 token 偏慢的一次；其余与全本地接近。

## 4. 换更快的规划模型，对首字输出有用吗？

没有。规划模型的首 token 不是用户体验指标——用户看不到规划何时开口，只看得到回答的第一个字。按**首字输出**衡量，把全云的规划从 `DeepSeek-V4.1-Flash` 换成首 token 更快的 `Hunyuan-A13B-Instruct` 后：

| 指标 | DeepSeek 规划（上一轮） | Hunyuan 规划（本轮） |
| --- | ---: | ---: |
| 规划阶段耗时（内部） | 1628 ms | 1702 ms |
| 回答阶段首 token（内部） | 504 ms | 571 ms |
| **首字输出（体验）** | **2360 ms** | **2466 ms** |
| 有效样本 | 19 / 40 | 30 / 40 |
| 429 失败 | 3 | 0 |

- 首字输出没有变早，反而慢约 106 ms。Hunyuan 的模型首 token 更快（230 vs 411 ms），但它把两条查询写完的时间更长，规划阶段反而慢 74 ms；用户最终等到的首字也就没有改善。
- 回答、端到端的差别在 ±100 ms 内，属于窗口波动；两轮样本数、拥塞情况不同，不能当作严格 A/B。
- 结论：**首字慢在「规划写两条查询」，换模型解决不了**。要压这一段，应该减少输出（只生成一条短查询，或先判定再生成），或者把规划放到本地（混合方案，规划 0.90 秒）。

## 5. 解读

1. **全本地最快也最稳**：1.92 秒中位数、2.24 秒 P90、没有长尾；代价是 4B 的回答质量未评测。
2. **混合次之**：2.20 秒中位数，回答沿用 DeepSeek；但远端回答仍有 3 秒级尾部（P90 3.27 秒）。
3. **全云最慢**：2.47 秒中位数。即使规划换成更快的模型，前端仍要 1.86 秒（规划 1.70 + 召回 0.16），是三种方案里最重的固定成本。
4. **嵌入本地化收益最大**：全云召回 159 ms、混合/本地 34–37 ms；更重要的是远端嵌入一旦拥塞就是数秒级，本轮仍有 10/40 被剔除。
5. **首字时间主要花在规划**：规划 0.9–1.7 秒，回答开口 0.57–1.26 秒，召回 0.03–0.16 秒。用户等的是三者之和；其中规划占比最大，而规划的首 token 本身不进入体验。

## 6. 建议

| 场景 | 选择 | 理由 |
| --- | --- | --- |
| 追求稳定与速度 | **全本地** | 中位数 1.92 s、P90 2.24 s、无长尾；需接受 4B 的回答质量 |
| 需要云端回答质量 | **混合** | 中位数 2.20 s，回答沿用 DeepSeek；建议给回答加 3 秒超时并回退本地 |
| 必须全云 | 可继续用 Hunyuan 规划 | 与 DeepSeek 差别在噪声内，但本轮 429 更少；要提速应精简规划输出 |
| 想靠换规划模型提速 | 不推荐 | 规划首 token 不进入用户体验；换模型后首字输出并没有变早 |

- 下一步优化顺序：**精简规划输出 → 本地化嵌入 → 本地化规划 → 回答侧超时/降级**。
- 所有关思考的结论只针对延迟，回答质量需要单独评测。

## 7. 复现

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

- 过滤规则：`plan > 4000`、`retrieve > 1000`、`generate > 4000`（毫秒），并剔除 429 记录。
- 清洗后的数据单独保存为 `*-clean.json`，可直接复算本报告的所有数字。

## 8. 数据留档

| 文件 | 内容 | sha256 |
| --- | --- | --- |
| `data/bench/ttft-2026-10-27-local-clean.json` | 全本地，40 条有效样本 | `93753007ae6b8477b55e8559271300bc66dc2bf675582039a998f4b8cea6bd3e` |
| `data/bench/ttft-2026-10-27-hybrid-clean.json` | 混合，28 条有效样本 | `0d1f17d5f5f97d212222ffdbda9df6b89bc4eedea1bc3cc019e8e6137c2256f8` |
| `data/bench/ttft-2026-10-27-cloud-hunyuan-clean.json` | 全云（Hunyuan 规划），30 条有效样本 | `26b6ad03a78ec2f9cc63562000be7fe594cdcbc5b5bc3909954b6aaef2156cc0` |
| `data/bench/ttft-2026-10-27-cloud-hunyuan.json` | 全云（Hunyuan 规划）原始 40 条 | `6d723e1b420380b13fac735f0b564fe0c6f8eb42b4091fe75e6651b5539def5b` |
| `data/bench/ttft-2026-10-27-cloud-clean.json` | 上一轮全云（DeepSeek 规划）19 条，用于第 4 节对照 | `827f7a1ad1ca0f3fb5c57ba6f73bd49049a45b89ce01027ea4c5673f29ef93ac` |
| `data/bench/ttft-2026-10-27-local.json` | 全本地原始 40 条 | `bf790f87580ad8ad2e3743185a44c8bcd8c2e1351dc85e3a4ec47d1513f76a5f` |
| `data/bench/ttft-2026-10-27-hybrid.json` | 混合原始 40 条（含 4 次 429） | `5e7c27684b591bf63ffc3606a6a24c2a224133ba8d49e2bf91af35b66a3a4410` |
| `data/bench/ttft-2026-10-27-cloud.json` | 上一轮全云原始 40 条（含 3 次 429） | `090b4bbc8fee209ce245036e5edd500a115e60f315460160df66b9e5638e6843` |

- 日志：`ttft-2026-10-27-{local,hybrid,cloud,cloud-hunyuan}.log`，与 JSON 同目录。
- `data/` 已被 `.gitignore` 忽略，明细只在本机。

## 9. 局限

- 全云两轮（DeepSeek 规划 / Hunyuan 规划）在不同的拥塞窗口，样本数 19 vs 30，第 4 节的对比只能看趋势。
- 剔除阈值是事后按本次数据断点确定的；报告同时保留原始中位数，便于核对。
- 每题 5 次，问题级结论只适合看量级。
- 只测到首 token，没有评估回答质量、完整回答耗时与幻觉率。
- 本地回答重复同一问题时可能命中 llama.cpp prompt cache（40 次里 3 次首 token < 100 ms），影响很小。
- 本地向量（bge-m3 1024 维）与云端（2560 维）的检索质量未做量化对比。
- 同一台机器（Intel i7-11800H / 22 GB）；换机器、换量化都会变化。
