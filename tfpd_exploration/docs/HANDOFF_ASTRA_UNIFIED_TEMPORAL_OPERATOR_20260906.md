# Astra 审查：跨数据集时间核分叉，需要更通用的结构

日期：2026-09-06  
给：Astra（规划 / 结构审核）  
来源：用户要求把「M1 / M2 / H1 时间核不一致」写成审查包，请 Astra 找更通用的结构。  
状态：**REVIEW_ONLY**。本文不授权训练、不打包、不注册 EvalAI、不改已封印权重。  
执行默认（在 Astra 写出命名结构之前）：B-transformer 新训可以先按 M2-like 因果核起步；**这是权宜，不是冻结。**

## 请 Astra 回答的问题

现在三条数据线上的 B-transformer **时间核不是同一个算子**。用户认为这不可接受：论文 / 主线需要一个跨 M1·M2·H1 可讲的通用结构，而不是三个各自长出来的加速故事。

请给出一个**命名的通用时间算子**（或有限的算子族），并写清：

1. 哪些层跨数据集必须共享（算子，不是超参）。  
2. 哪些数字可以按数据集变（窗长、E0 维、out_dim、官方 batch）。  
3. 默认先落哪一边：**M2-like**、**M1-like**，或 **融合**。用户要求三种都要认真考虑，不要因为执行者偏好而删掉后两种。  
4. 每种选项要重训哪条线、作废哪些现有点，以及第一格消融是什么。

执行者可以有工程默认，**不能代替这次结构决定。**

## 分叉是什么（不是包装幻觉）

骨架已经是同一族：因果 conv k=5、8 slot、d=256、8 head、4 层、width 256、FFN 512、256→128→任务维。分叉在 **时间核**：

| 线 | 时间算子 | 位置 | 窗 | 历史是否互相读 | 现况 |
|---|---|---|---:|---|---|
| **M2 SMALL** | 因果 Transformer | 正弦 PE | 50 | 是（层内 contextualize） | 581973 finished，HO **0.390**；exact-E 已官方跑通 |
| **M1 T e6** | `QueryTemporalStack`（current-query） | age-bias，16 bucket | 100 | 否（历史只当 memory） | 仅 source-dev；T−FULL **−0.004**；V3 包装 CONDITIONAL-GO，未注册 |
| **H1 V4/V6/V7** | FULL 因果或 query / recency | PE 或 age / log-age / recency τ | 700 | FULL 是；V6 query 否 | 未作为产品推出；V6 未保住 FULL（本地 T≈0.195） |
| **H1 因果 e5** | 与 M2 同族因果 | PE | 700 | 是 | 581940 / 581942 **naive 全窗超时** |
| **H1 C2** | SPINT，不是 B-transformer | 单位级 identity + 长窗 | 700 | C2 自己的路径 | **581920 finished，HO 0.376，延迟 0.035** |

M1 的 `QueryTemporalStack` 源码写明：这 **不是** 普通 Transformer KV cache。所以 M2 exact-E（frontend k=5 + last-layer last-query，**无** temporal KV）和 M1 V3（滚动 KV + current-query）是两套精确改写，不是同一合同的两个实现。

Frontend 也只是「同族」：M2 用 concat MLP + `MultiheadAttention`、E0=50；M1 用 FactoredTokenMLP、E0=100。统一时间核时，Astra 应同时说 frontend 是否一并收口。

## 数据集强迫 vs 训练线分叉

**数据集 / 评测只会强迫数字，不强迫时间算子：**

- 官方 batch：M1=4，M2=7，H1 按协议  
- 输出维、tag 数、session 长度（M1 公开约 16–18 万 bin，所以 naive 全窗先超时）  
- H1 手写语境下 **W=700** 比 M2 的 50 更像数据需求  

**不是数据集必然的：** PE 因果 vs query-memory、历史是否互相读、有没有 age-bias。这是 `two_mainlines` 里分开选的。

因此「各数据各用各的核」站不住。用户要的是通用结构；执行默认 M2-like 只是因为 **唯一已经官方成立的 B-transformer 是 M2 因果**，不是因为 M1/H1 不能用别的核。

## 三种选项都要审（不要提前删）

### A. M2-like（当前执行默认，合理但不冻结）

因果 self-attn + 正弦 PE + exact-E 合同（k=5 frontend、last-query、batched bank、inactive freeze）。

- 优点：581973 已证明算子 + 加速能在官方 CPU 上跑完并出 HO。包装合同已经清楚。  
- 代价：M1 要按这个时间核 **重训**；H1 B-transformer 要 W=700 exact-E，不能拿 581940/942 当核失败。  
- 不要齐的：窗长、E0 维、out_dim、batch。  
- 风险：把 M1 已接受的 query 归纳偏置丢掉，而那个偏置从未官方验证——也可能本来就不该当家族定义。

### B. M1-like（必须作为并列假说）

current-query：只有当前 query 跨层更新；历史 token 互不 contextualize；age-bias；V3 流式 KV。

- 优点：为流式而生；M1 上 T≈FULL（Δ −0.004）；容器 B4 ≈8.3 ms，相对旧 naive 全窗约 1/10。  
- 代价：M2 若改过来，**581973 不再是同核结果**，要重训再官方验证；H1 V6 已经显示 query **不会**自动等于 FULL。  
- 论文负担：必须讲清「历史不互读」为什么跨数据集成立，而不是 M1 的延迟补丁。

### C. 融合（必须作为并列假说，不能当默认）

例如 PE + age-bias、因果层 + query 读出、或「frontend 统一、时间核按任务开关」。

- 只有 Astra 能写出 **一个可命名的归纳偏置** 时才值得开。否则这是第三个新模型：两边现有点和 581973 都作废，还多一个没消融的核。  
- 若走融合：第一格必须是相对 A、相对 B 的配对，而不是「看起来都用上了」。

## 硬约束（Astra 方案不得打破）

1. **C2 不是 B-transformer。** 581920 仍是 H1 官方对照。通用结构是统一 B 族，不是把 C2 拧进 Transformer，也不是把 Transformer 融回 SPINT。  
2. **包装统一 ≠ 算子统一。** 只改 `predict` 不能宣称结构已齐。  
3. **官方分不能用来选点。** 581973 / 581971 / 581920 / 581919 只作历史对照。新核的 epoch-pick 仍只用本地可见面。  
4. **不打开 hidden / test query。**  
5. **不自动提交。** 本文不消耗额度。  
6. 已删的 M1 naive 全窗图（`optv2-t-ema-e6-w0`）不要救活。现活的是 V3 包装 `spint-m1:v3-t-ema-e6-t2-5e44fc37`，`register: false`，只当对照。  
7. 历史 result root 不覆盖。

## 执行者不得擅自做的

- 在 Astra 写出命名结构之前，把 M2-like 写成论文主句或「已决定的家族」。  
- 为了整齐把 H1 窗改成 50，或把 M2 窗改成 100/700。  
- 用 581973 的 HO 给未来 M1/H1 因果重训选 epoch。  
- 重开已关闭的 H1 FiLM / C3 / q3-AFC4 等路线来「绕过」这次结构决定。

## 请 Astra 交回的最小冻结

写一份短冻结（新文件即可），至少包含：

- `operator_name`  
- `family`: `m2_like` | `m1_like` | `fusion` | `other`（other 必须命名）  
- `shared_must`：跨 M1/M2/H1 必须相同的模块列表  
- `per_dataset_may`：允许变的超参  
- `first_ablation`：第一格只改哪一个结构变量  
- `retrain`：M1 / M2 / H1 各要不要新训  
- `keeps`：C2、581973、M1 T e6 source-dev 各如何引用  
- `rejects`：明确丢掉的核（例如「H1 V6 不当家族定义」）

交回之前，执行侧新 B-transformer 训练若必须开工，只许标 `provisional_m2_like`，并链到本文。Astra 冻结后以冻结为准。

## 证据入口（只读）

- M2 官方 e8：`tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/OFFICIAL_581973.md`  
- M2 exact-E（无 temporal KV）：`tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py`（`_FrontendWindowCache` / `_ExactEEngine`）  
- M1 query 核：`tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/core.py`  
- M1 V3 流：`tfpd_exploration/src/m1_runtime_v3/runtime.py`  
- M1 绑定：`tfpd_exploration/src/m1_optimized_v2/model.py`  
- M1 验收（source-dev，非官方）：`tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1/ACCEPTANCE.md`  
- H1 算子诊断：`tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/v8_readonly_operator_hypothesis_v1/README.md`  
- H1/M2 C2·T4 官方对：`tfpd_exploration/docs/SUBMISSION_H1_M2_EPOCH_PICK_PAIR_20260905.md`
