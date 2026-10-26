# 首 token 基准（bench 新口径）

- 日期：2026-10-26
- 分支：`bench`（`upstream/dev` 的 `740322f` + 计时改动）
- 脚本：`src/fastrag/bench.py`
- 主数据：`data/bench/results-2026-10-26-first-token.json`（`generation_mode = first_token`、`answer_thinking = false`）
- 旧口径数据：[`bench-data-changes.md`](bench-data-changes.md)（完整回答 + 开启思考）

## 0. 摘要

- 生成阶段改为**首 token 延迟（TTFT）**：拿到第一个 token 立即断开，不再等整段回答。
- 这个模型是推理模型，会先流一段隐藏的 `reasoning_content`。开着思考时，「首 token」到底指推理 token 还是回答正文会差 20 倍以上：
  - 首个推理 token：约 **0.4–0.6 s**
  - 首个可见回答 token（Q1 真实提示词）：约 **23.7 s**，期间收到 **1422 个空 chunk**
- 因此基准默认**关闭回答模型的思考**（`enable_thinking = false`），让首 token 就是正文：9 次运行 **403–852 ms，中位数 491 ms**。
- 端到端中位数从旧口径的 **22.0 s 降到 7.5 s**。注意这是**口径变化**，不代表产品变快：生产环境仍开着思考、等完整回答，用户实际等待仍是 16–24 s（第 5 节）。
- 新口径下瓶颈彻底转移到**检索规划**：中位数 6937 ms，占端到端 **91.2%**；首 token 只占 6.7%，召回 1.8%，融合 0.0%。

## 1. 为什么改口径

### 1.1 旧口径的问题

- 旧基准的「生成」是 `graph.invoke()` 跑完整回答，耗时 10–29 s，占端到端 63–69%。
- 每次运行都要等模型把整段回答生成完，9 次基准要 3–4 分钟，而且测量值混入了「回答长度」这个与流水线无关的变量（Q3 旧数据回答只有 169–429 字符，端到端就掉到 12.6–15.5 s）。

### 1.2 推理模型的「首 token」陷阱

- `deepseek-ai/DeepSeek-V4.1-Flash` 会先输出 `reasoning_content`（思考过程），再输出 `content`（回答正文）。
- `langchain-openai` 1.7.2 **不解析 `reasoning_content`**（源码注释写明 not extracted），所以 `llm.stream()` 在思考阶段只会拿到 `content` 为空的 chunk：实测 Q1 真实提示词下，**23.7 s 内收到 1422 个空 chunk**，之后才出现第一个正文字符。
- 如果按「第一个非空正文」算 TTFT，数值约等于完整生成时间，改口径没有意义；如果按「模型吐出的第一个 token」算，又会把思考算进去且受厂家实现影响。
- 两种定义都别扭，所以直接把思考关掉，让「第一个 token」就是「第一个回答字符」。

### 1.3 关闭思考的接口

同一个请求加不同参数各试一次（短问题，记录首个 reasoning / content 的时间）：

| 参数 | reasoning | 首个 content | 结论 |
| --- | ---: | ---: | --- |
| 不加（默认） | 368 ms | 1313 ms | 会思考 |
| `enable_thinking = false` | 未出现 | 482 ms | **可用** |
| `thinking = {"type": "disabled"}` | 未出现 | 409 ms | 可用 |
| `reasoning_effort = "none"` | 未出现 | 429 ms | 可用 |
| `chat_template_kwargs.enable_thinking = false` | 464 ms | 1678 ms | 无效 |
| `thinking_budget = 0` | — | — | 接口 400（要求大于 0） |

基准采用 SiliconFlow 文档里的 `enable_thinking = false`，通过 `extra_body` 下发；开启思考时不下发任何参数，保持厂家默认行为。

## 2. 改动

| 文件 | 改动 |
| --- | --- |
| `src/fastrag/rag_graph.py` | `Pipeline` 新增 `first_token_only`、`answer_thinking`；`answer` 节点在首 token 模式下改为 `stream` 读取，`_first_token()` 拿到第一个非空 chunk 就 `close()` 断流；`get_pipeline()` 按口径缓存 |
| `src/fastrag/llm.py` | `build_llm(config, thinking=...)`；`thinking=False` 时给 `init_chat_model` 传 `extra_body={"enable_thinking": False}` |
| `src/fastrag/config.py` | `LLMConfig` 新增 `thinking: bool = True`，从 `[llm.*]` 读取 |
| `config/fastrag.toml` | `[llm.route]`、`[llm.answer]` 增加 `thinking` 字段，显式写出 |
| `src/fastrag/bench.py` | 默认首 token 口径；新增 `--thinking`（保留思考）、`--full-generation`（旧口径）；JSON 落盘 `generation_mode`、`answer_thinking`、逐条 `first_token_only` |

- 断流会取消后续请求，不只是「读到首 token 就忽略」，所以省掉了剩余生成的时间和 token。
- 生产路径（`uv run python -m fastrag.rag_graph` 交互问答、`get_pipeline()` 默认参数）仍为**完整回答 + 开启思考**，基准口径不会改变用户看到的行为。
- 首 token 模式下 `answer` 只保存第一个 chunk（本模型是 2 个字符），所以明细里 `answer_chars` 为 2；这是预期行为，不代表截断错误。

## 3. 方法

- 与之前一致：3 个问题 × 每题 3 次，另加 1 次预热；同一台机器、同组模型、`fusion.method = "rrf"`。
- 关闭代理直连 `api.siliconflow.cn`；语料与索引沿用（2625 个 chunk，复用已嵌入向量，初始化 60.9 ms）。
- 阶段耗时仍由 `timed` 装饰器记录；「首 token」即 `answer` 节点耗时（流式首 chunk 往返）。
- 命令：

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  uv run python -m fastrag.bench --runs 3
```

- 复现思考开启后的对照：`... uv run python -m fastrag.bench --thinking`（会等到首个正文字符，Q1 约 20 s 以上）。

## 4. 结果

主数据：2026-10-26T09:41:50Z，`generation_mode = first_token`，`answer_thinking = false`。

### 问题 1：LangGraph 中如何实现 human-in-the-loop？（3 次）

| 阶段 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) | 标准差 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检索规划 | 9010.2 | 6936.6 | 6186.7 | 13907.2 | 4257.5 |
| 召回 | 157.9 | 148.6 | 125.0 | 200.1 | 38.4 |
| 融合 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 |
| 首 token | 474.4 | 421.4 | 403.5 | 598.2 | 107.6 |
| **端到端** | **9649.1** | **7496.4** | **6914.9** | **14536.0** | **4242.2** |

### 问题 2：如何让 LangGraph 的图流式输出 token？（3 次）

| 阶段 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) | 标准差 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检索规划 | 8066.4 | 7783.9 | 5946.8 | 10468.5 | 2274.0 |
| 召回 | 143.4 | 148.1 | 132.4 | 149.7 | 9.6 |
| 融合 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 |
| 首 token | 585.4 | 467.0 | 437.2 | 851.9 | 231.3 |
| **端到端** | **8801.9** | **8388.8** | **6541.2** | **11475.6** | **2493.0** |

### 问题 3：LangChain 的 Agent 如何调用工具？（3 次）

| 阶段 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) | 标准差 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检索规划 | 6951.4 | 6752.0 | 5921.8 | 8180.3 | 1142.4 |
| 召回 | 139.2 | 139.4 | 136.7 | 141.6 | 2.5 |
| 融合 | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 |
| 首 token | 556.9 | 494.8 | 491.0 | 684.9 | 110.9 |
| **端到端** | **7652.1** | **7393.0** | **6557.0** | **9006.2** | **1245.0** |

### 全部 9 次运行

| 阶段 | 平均 (ms) | 中位数 (ms) | 最小 (ms) | 最大 (ms) | 标准差 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 检索规划 | 8009.3 | 6936.6 | 5921.8 | 13907.2 | 2635.8 |
| 召回 | 146.8 | 141.6 | 125.0 | 200.1 | 21.6 |
| 融合 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 |
| 首 token | 538.9 | 491.0 | 403.5 | 851.9 | 147.8 |
| **端到端** | **8701.0** | **7496.4** | **6541.2** | **14536.0** | **2682.1** |

- 9 次全部触发检索（`need_search = true`），没有再出现上一轮的误判。
- 按每次运行取占比中位数：检索规划 **91.2%**、首 token **6.7%**、召回 **1.8%**、融合 **0.0%**。
- 检索规划本身方差很大（5.9–13.9 s），端到端的标准差 2.7 s 几乎全部来自它。

## 5. 与旧口径对比

旧口径来自本文档之前的两次完整运行（`results-2026-10-26-post-dev-merge-run1/run2.json`：完整回答、开启思考），新口径为本文主数据。**两者不是同一件事**，只能用来判断各阶段量级。

| 阶段 | 旧口径中位数 (ms) | 新口径中位数 (ms) | 变化 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 检索规划（旧=路由+重写） | 7166.2 | 6936.6 | -3.2% | 两边都开着思考，可直接比 |
| 召回 | 180.2 | 141.6 | -21.4% | 网络波动范围内 |
| 融合 | 0.1 | 0.1 | 0.0% | 本地 RRF |
| 生成 | 12581.9 | 491.0 | **-96.1%** | 口径不同：完整回答 vs 首 token，且思考关闭 |
| **端到端** | **21997.8** | **7496.4** | **-65.9%** | 主要是口径变化，不是产品变快 |

- **不要把 -65.9% 当成优化成果**：生产环境仍开着思考并等完整回答，上一轮实测端到端是 22–24 s，和旧口径一致。
- 想比较「用户真正等到第一个字」的时间，用 `--thinking`：开启思考后 Q1 的首个正文字符约 23.7 s（单次探针，非基准），和完整生成时间同量级——也就是说这个模型的时间几乎都花在思考上，关掉思考才是真正让首 token 变快的原因。
- 首 token 只有 0.4–0.9 s，说明**网络往返 + 提示词预填充本身很便宜**；之前「生成 12–16 s」里绝大部分是思考与逐字输出。

## 6. 新的瓶颈

- 首 token 口径下，端到端 7.5 s 里有 6.9 s 是 `plan_search`。它现在包含一次结构化输出调用：判断是否检索 + 生成两路英文查询，输出 token 不少，且仍开着思考。
- 可选方向（未实施）：
  1. 给 `[llm.route]` 也关闭思考（`thinking = false`），用 `--thinking` 对比计划质量；
  2. 减少 `plan_search` 的输出（先判断是否需要检索，再只生成一条查询）；
  3. 换更小/更快的模型做规划，失败再回退；
  4. 对常见问题缓存规划结果。
- 召回（约 0.14 s）与融合（0.1 ms）在新口径下已经可以忽略。
- 各模型单独的首 token 时延对比见 [`llm-ttft.md`](llm-ttft.md)（本地 `qwen3-4b-instruct` 30 ms，云端最快约 230 ms）。
- 整条链路全本地（bge-m3 + qwen3-4b-instruct）的端到端首 token 见 [`bench-all-local.md`](bench-all-local.md)（中位数 2508 ms，比云端快约 3 倍）。

## 7. 数据留档与复现

| 文件 | 内容 | sha256 |
| --- | --- | --- |
| `data/bench/results-2026-10-26-first-token.json` | 本文主数据（首 token + 关思考） | `8648ab1e7c8666375dd331b14f187456c0997f4aa51c26b4449f02e730a15949` |
| `data/bench/bench-run-20261026-first-token.log` | 运行输出 | `515b34149a210d5e4f00b28910ace95b6b9fee44b5c0c9c5114832ce1cd423c6` |
| `data/bench/results-2026-10-26-post-dev-merge-run1.json` | 旧口径第 1 次 | `29c8b32643ef1ad686df9e66c2979059f53272e6e6145898c09bd77bcbde1b66` |
| `data/bench/results-2026-10-26-post-dev-merge-run2.json` | 旧口径第 2 次 | `c9c265e18c88ff2f7c17f1f3310b1846132610270ab1eb3917f31a84fb07c2b9` |
| `data/bench/results-2026-10-26-pre-dev-merge.json` | 合并前 | `f6fd9bab69424d24f926cb3ba5934520ef674ffaf57b2d7cfa9bd1096c74a13a` |

- `results.json` 始终是最近一次运行的副本，历史文件不再覆盖。
- 语料、向量、BM25 索引未重建，三次口径的运行共用同一份索引。

## 8. 局限

- 每题 3 次，检索规划方差很大（5.9–13.9 s），首 token 的绝对数值只适合看量级。
- 关闭思考会改变回答内容与质量，本文没有做答案质量评估；`thinking = false` 仅用于基准口径，生产配置保持 `true`。
- 首 token 延迟只覆盖「到第一个字」，不含用户读完回答的时间；需要完整回答耗时就用 `--full-generation`。
- `enable_thinking` 是 SiliconFlow / 通义系 OpenAI 兼容接口的参数；换厂家若参数被拒需要调整 `build_llm`。
- 探针数据（思考开启时首个正文 23.7 s、1422 个空 chunk）来自单次测量，用于说明量级，不是基准。
