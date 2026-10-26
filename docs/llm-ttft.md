# 各对话模型首 token 时延（直接测模型）

- 日期：2026-10-26
- 分支：`bench`
- 脚本：`src/fastrag/llm_ttft.py`
- 主数据：`data/bench/llm-ttft-2026-10-26.json`（第 2 次运行，10:08Z）
- 对照数据：`data/bench/llm-ttft-2026-10-26-run1.json`（第 1 次运行，09:5xZ）
- 目的：只比较模型本身的首 token 时延（TTFT），不经过检索、规划、融合，也不评估回答质量。

## 0. 摘要

- **本地 `qwen3-4b-instruct` 最快：30 ms**（同一台机器、localhost，但见 5.1 的缓存说明）。
- 云端 200–300 ms 档：`Hunyuan-A13B-Instruct` **230 ms**、`Qwen3-30B-A3B-Instruct` **254 ms**、`GLM-4.5-Air` **269 ms**。
- 当前默认的 `DeepSeek-V4.1-Flash`：**开思考 1232 ms，关思考 411 ms**；关掉思考能省约 0.8 s。
- `Step-3.5-Flash`（3173 ms）和 `DeepSeek-V3.2`（1462 ms）明显更慢，不建议用于延迟敏感的角色。
- `Qwen3.5-35B-A3B` 两次运行差别巨大（87.1 s → 544 ms），说明厂家负载波动足以淹没模型差异，选型时应避开这类不稳定的服务。

## 1. 方法

- 固定提示词：`用一句话解释什么是 RAG。`，每个模型先预热 1 次（不计入），再测 3 次。
- 计时口径：`time.perf_counter()` 从发起 `llm.stream()` 到收到**第一个非空文本 chunk**；拿到就 `close()` 断流。
- 思考默认关闭（`enable_thinking = false`），保证首个文本 chunk 就是回答正文；另外单独保留一条 `DeepSeek-V4.1-Flash` 开思考做对照。
- 单次请求超时 30 s；失败（如厂家 503）记录为失败，不再重试。
- 云端走 SiliconFlow（直连，无代理），本地走 `[endpoints.local]`（`http://127.0.0.1:9938/v1`，llama.cpp）。
- 本地模型只纳入 `qwen3-4b-instruct`（唯一在跑的对话模型），其余本地模型不在本次统计内。

命令：

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  uv run python -m fastrag.llm_ttft --runs 3
```

## 2. 结果（第 2 次运行）

| 模型 | endpoint | 思考 | 中位数 (ms) | 平均 (ms) | 范围 (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3-4B-Instruct（本地） | local | 开 | 30 | 29 | 28–30 |
| Hunyuan-A13B-Instruct | siliconflow | 关 | 230 | 229 | 222–235 |
| Qwen3-30B-A3B-Instruct | siliconflow | 关 | 254 | 256 | 251–263 |
| GLM-4.5-Air | siliconflow | 关 | 269 | 262 | 220–295 |
| Qwen3-8B | siliconflow | 关 | 359 | 476 | 314–754 |
| Qwen3.5-9B | siliconflow | 关 | 362 | 904 | 347–2003 |
| DeepSeek-V4.1-Flash（关思考） | siliconflow | 关 | 411 | 627 | 384–1087 |
| Qwen3.5-35B-A3B | siliconflow | 关 | 544 | 502 | 362–599 |
| DeepSeek-V4.1-Flash（开思考） | siliconflow | 开 | 1232 | 1287 | 1094–1536 |
| DeepSeek-V3.2 | siliconflow | 关 | 1462 | 1358 | 814–1798 |
| Step-3.5-Flash | siliconflow | 关 | 3173 | 3843 | 3094–5261 |

排序按中位数。`Qwen3-8B` 与 `Qwen3.5-9B` 中位数接近，但后者有一次 2 s 的离群值（平均 904 ms）。

### 2.1 两次运行的差异

| 模型 | 第 1 次 (ms) | 第 2 次 (ms) |
| --- | ---: | ---: |
| Qwen3.5-35B-A3B | 87056 | 544 |
| DeepSeek-V4.1-Flash（关思考） | 726 | 411 |
| Qwen3-8B | 1043 | 359 |
| Qwen3.5-9B | 509 | 362 |
| DeepSeek-V3.2 | 1098 | 1462 |
| Step-3.5-Flash | 3610 | 3173 |
| Hunyuan-A13B-Instruct | 246 | 230 |
| Qwen3-30B-A3B-Instruct | 264 | 254 |
| GLM-4.5-Air | 252 | 269 |
| DeepSeek-V4.1-Flash（开思考） | 1457 | 1232 |
| Qwen3-4B-Instruct（本地） | 31 | 30 |

- 稳定的模型（Hunyuan、Qwen3-30B-A3B、GLM-4.5-Air、本地 Qwen）两次差距在 ±20 ms 内。
- `Qwen3.5-35B-A3B` 第 1 次遇到厂家 503/排队，首 token 到 87 s；第 2 次恢复正常。同一模型 160 倍的波动说明**公网服务不能只看一次结果**。
- DeepSeek 开思考时，短提示词下只多约 0.8 s；但换成 RAG 的真实长提示词，思考会让「首个正文 token」拖到 20 s 以上（见 [`bench-first-token.md`](bench-first-token.md)）。

## 3. 怎么用这些数

- 需要**最低首字延迟**且能接受本地服务：`qwen3-4b-instruct`（localhost，30 ms，缓存因素见 5.1）。
- 只能用云端、要稳定在 300 ms 内：`Hunyuan-A13B-Instruct`、`Qwen3-30B-A3B-Instruct`、`GLM-4.5-Air`。
- 继续用 `DeepSeek-V4.1-Flash`：把 `[llm.route]` / `[llm.answer]` 的 `thinking` 设为 `false`，短请求约 0.4 s；长上下文仍要实测。
- 不建议：`Step-3.5-Flash`（3 s+）、`DeepSeek-V3.2`（1.5 s）、`Qwen3.5-35B-A3B`（波动过大）。

## 4. 与 RAG 链路的关系

- 本文只测模型本身。RAG 链路里「生成」还包含拼上下文与请求往返，「检索规划」还包含结构化输出解析，所以链路里的耗时一定更高。
- 链路口径的数据见 [`bench-first-token.md`](bench-first-token.md)：关思考后，生成阶段首 token 中位数 491 ms，检索规划中位数 6937 ms。
- 换成 200–400 ms 档的模型只能改善链路里约 0.6 s 的规划耗时，无法解决检索规划占 90% 的结构性问题；要降低端到端还得减少规划的输出或调用次数。

## 5. 局限

1. **本地 30 ms 受 prompt cache 影响**：每次请求都是同一条提示词，llama.cpp 预热后命中前缀缓存，首 token 只剩一次解码的时间；换成新问题或冷启动会更高。云端模型也可能命中厂家的上下文缓存，只是本次数据里看不出明显影响。
2. 3 个样本、单条短提示词（约 20 token），不含 RAG 的长上下文预填充；长提示词下各模型的排序可能变化。
3. 只测首 token，不反映吞吐、总生成时间、回答质量或工具/结构化输出能力。
4. 公网服务受时段负载影响大（见 2.1），绝对数值只适合判断量级。
5. 本地只跑了一个模型、一种量化（Q4_K_M）和一份 llama.cpp 配置，换量化或参数会得到不同结果。

## 6. 数据留档

| 文件 | 内容 | sha256 |
| --- | --- | --- |
| `data/bench/llm-ttft-2026-10-26.json` | 主数据（第 2 次，11 个模型） | `efb3b25a560f22f9385382e0cbc777fddd0950c789875b6931dd32e1fd276e21` |
| `data/bench/llm-ttft-2026-10-26.log` | 第 2 次运行输出 | `5c99ebaf74ea16a330998c35500809894fe4c6f376e944c749efb22cd4907283` |
| `data/bench/llm-ttft-2026-10-26-run1.json` | 第 1 次（含 35B 的 87 s 离群） | `2305bfe1aa5995c7915073cc7ec21b92526808c986b16fdb9176c62c05595fb9` |

- `data/` 已被 `.gitignore` 忽略，这些文件只留在本机。
