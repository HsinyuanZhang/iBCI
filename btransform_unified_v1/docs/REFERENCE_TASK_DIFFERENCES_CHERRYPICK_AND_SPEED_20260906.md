# 三任务差异对照、可选项菜单、推理加速指导（btransform_unified_v1 执行者用）

- 维护人：审核 agent（与 `WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md` 同步维护；工单改，本文跟着改）。
- 用途：执行者在此**选**。每个可选项有编号（`CAL-*` / `TRN-*` / `SEL-*` / `SPD-*`）；receipt 里只写编号 + 参数，不要重新发明名字。
- 本文只列事实与选项，不做训练授权、不定选点、不碰 EvalAI。
- 上位合同仍是 [NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md](../../tfpd_exploration/docs/NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md)（P0/P1/P2）。

**三句话总结**

1. 三个数据集的**标签**不同（M2 目标角度 → 2 维手指速度；M1 16 通道 EMG；H1 7 维 `tx,ty,tz,rx,g1,g2,g3`），所以 **T4 carrier 的估计算子必然不同**。这是数据集的事实，不是对齐缺陷。统一的是「每单元 4 维 carrier」这个**接口**，不是算子。
2. identity（B3S 系）三任务同一形状（pre_pool → 按 trial 平均 → 拼 carrier → post_pool），但 hidden / 是否拼 carrier / 输出维 / 来源 checkpoint 都不同；都是 **SPINT 系 checkpoint 训出来的冻结编码器**。
3. 速度：M1/M2 几何下 A 级（等权）优化就够；**H1 W=700 的全窗因果核在 A 级做完后地板仍 ≈25–30 ms/步（本机）**，官方已超时过一次，必须在训练前就选 C 级路线或缩 L。

---

## 1. 一页表：三任务的硬差异

| 项 | M2（finger velocity） | M1（EMG） | H1（7-DoF 运动） |
|---|---|---|---|
| bin | 20 ms | 20 ms | 20 ms |
| 评分窗 W（SPINT 定义） | 50（1 s） | 100（2 s） | 700（14 s） |
| 单元数 N | 96 | 64 | 176 |
| 输出维 | 2 | 16 | 7，列序 `tx,ty,tz,rx,g1,g2,g3` 冻结（P2-15） |
| 目标尺度桥（P0-3） | 训 `×5`，评 `pred/5`；`MSE(raw,5y) = 25·MSE(raw/5,y)` | divisor = 1 | 训 `20y`，评 `pred/20`；`MSE(raw,20y) = 400·MSE(raw/20,y)`（**比值**，不是差值） |
| 校准 trial 结构 | 100 bin 插值 trial；MOVE 段 = 未插值有效前缀的 bins[5:30]；trial 带**目标角度** | 支持 trial 10 个（M10），query trial 10..210；trial 带 **16 通道 EMG** | trial 长 **1024 bin**（`T=1024`）；trial 带 **7 维速度**；部署预算 **M3** |
| 校准预算（血统实际用值） | **M33**（`SUPPORT_HORIZON=33`；S1/581973 用这个） | **M10**（rSyn3 探过 10/6/4/2） | 训练 cycle **M7/M5/M4/M3**，部署 **M3**（C2） |
| 源/外部 session | 源 7（2020-10-19…10-28）；外部 6（ext6），常用 **ext4**（10-30 ×2、11-18、11-19；519/490/425/635 窗） | 源多 session，本地面 31,252 source-minival；官方 LOSO 后段 | 源 13 session、6 个日期；本地 **20,325** 完整流 / **2,908** 满窗选点 |
| 历史最优（官方 HO） | SPINT MOVE-T4 0.349；**B-transformer S1/EMA e8 0.390**（唯一正例） | **Original SPINT 0.649**；冻结 SPINT + B3S + rSyn3 0.640 | **C2 重训 SPINT e15 0.376**；Original 0.262 |
| 历史最优（本地同面） | S1 endpoint24 ext4 0.449（session-equal mean），last-8 0.439 ± 0.010 | Original 0.809 / Sfix 0.828 / QueryAge 0.812（31,252） | Original 0.961 / C2 0.888（20,325） |
| 新系列现状 | S1 = 本系列祖先 | QueryAge 本地 0.812，官方 0.574 | QueryAge formal12：pooled 0.278 / session-mean 0.268 / 选点 0.320 |
| 官方汇总法 | session 内 variance-weighted R²，再跨 session 平均（P1-11：≠ 本地 pooled） | 同 | 同 |
| 已知官方事故 | 581973 normalized latency **0.736**（SPINT 0.044） | 581980 拒非有限/非 C-contiguous float32 | 581940/581942 **naive 全窗超时** |

「新系列的 P_task 前缀 bin」是模型侧输入长度，**不是**数据集属性；三任务的流式评测都按 bin 顺序给全量历史，早段自然是冷启动。

---

## 2. 校准物：identity 与 carrier 逐任务写死（P0-2）

### 2.1 T4 carrier（4 维 / 单元）——**估计算子按任务不同，这是标签决定的**

| | M2 **MOVE-T4** | M1 **rSyn3** | H1 **H-C（normalized EB）** |
|---|---|---|---|
| 用到的标签 | 校准 trial 的**目标角度** θ | 支持 trial 的 **EMG（16 ch）** | 支持 trial 的 **7 维速度** |
| 输入统计 | 每 trial 在 MOVE 段 bins[5:30] 的 spike 和 / 长度 → 每单元发放率 | 每 bin 发放率 | 每 bin 发放率 |
| 估计 | 对 `[1, cosθ, sinθ]` 最小二乘（设计矩阵须 rank 3，≥3 个方向）→ `[a, c, √(a²+c²), baseline]` | EMG 先 `R(x)=max(x,0)` → RMS 缩放 → 源 session rank-3 **NNMF**（NNDSVDa，seed 42）→ 每单元对 3 个 synergy 做 ridge（λ=1，截距不罚）→ `[w1,w2,w3,intercept]` | 发放率投到冻结源 PC（q 维）→ 对 7 维速度 ridge（λ，截距不罚）→ 映回单元 → 投到冻结 4 维行为基 U → **Empirical-Bayes** 向源先验 μ 收缩（权 τ²/(τ²+投影方差)） |
| 归一化 | 源 7 session z-score（不继承旧 M33 normalizer） | 源 RMS | 源 RMS normalizer（H-S/H-C 共享） |
| 支持 trial 数 | 全部校准 trial（M33） | M10 | 恰好 3 或 4 |
| 代码 | `m2_dual_track_v1/champion.py::fit_move_t4` → `streaming_calibration_exp/src/data/falcon_t4_features.py::t4_from_trial_sums` | `m1_emg_syn3_fcm_v1/plan.py`（`RANK=3, RIDGE_LAMBDA=1.0, SUPPORT_TRIALS=10`）+ `DESIGN_M1_EMG_RSYN3_SUCCESSOR_20260902.md` | `SPINT-main/src/data/h1_m4_eb_pilot.py::fit_frozen_carrier / fit_deployment_carrier` |

执行者不需要、也不应该「统一」这三个算子。receipt 里按任务写：算子名、支持 trial 数、normalizer 来源、数组 SHA。

### 2.2 identity（B3S 系 E0）

统一形状：`pre_pool(trial → h) → 按 trial 平均 → [拼 carrier] → post_pool → d_e`。差异在参数：

| | M2 | M1 | H1 |
|---|---|---|---|
| 编码器 | **B3S**（`SideFeatureEarlyPoolEncoder`，hidden 64，side_dim 4） | **B3**（Sfix e11 `student.id_encoder`，`compute_identity(side_features=None)`） | C2 e15 `carrier_pre_pool(1024→32)+ReLU` / `carrier_post_pool(36→32→32→700)` |
| carrier 是否进 post_pool | **是**（MOVE-T4 拼进） | **否**（rSyn3 只在 token 里） | **是**（H-C 拼进） |
| d_e | 50（= W） | 100（= W） | 700（= W） |
| 来源 checkpoint | M2 T4 主线 SPINT champion（冻结） | Sfix epoch 011（SHA 见 `cross_dataset_functional_calibration_v1/plan.py`） | `c2_epoch_015.ckpt`，SHA `ce46267e…`，schema `h1_cal_aug_m3_aware_dual_selection_v2_checkpoint` |
| 抽取方式 | `champion.native_e0_and_u`（push_trial / finalize_identity，非 batched-mean GEMM） | `m1_optimized_v2/calibration.py::load_frozen_b3`（只抽 `student.id_encoder`） | `two_mainlines_long_v1/decoder/h1_calibration.py::load_frozen_c2_materializer` |
| SPINT 里怎么消费 | `src + identity`（加到 W 维 spike 窗上，共用 fc_in） | 同 | 同（`h1_carrierid_spint.py` forward） |
| 本系列怎么消费 | 静态 per-(unit,bin) 特征 concat → `W_e0 [256,d_e]` | 同 | 同；`W_e0 [256,700]` ≈ 17.9 万参数 |

**d_e = W 不是巧合**：B3S 的 post_pool 输出被设计成能加到 SPINT 的 W 维 spike 窗上。本系列不加、只 concat，所以 d_e 没有理由必须等于 W——这是 §3 `CAL-3` 的来源。

---

## 3. 可选项菜单（cherry-pick）

标记：**✓证** = 有官方或同面本地证据；**○用** = 血统用过但未证正向；**△新** = 未在该任务试过，进新格子须单独门。每选一项写编号 + 参数 + 证据行。

### 3.1 校准侧 `CAL-*`

| 编号 | 选项 | M2 | M1 | H1 | 说明 |
|---|---|---|---|---|---|
| **CAL-1** | **校准预算 prefix-cycle + 部署预算 aware**：训练时每 (session, epoch) 从 `{M_k}` 轮换取「session 前 M 个校准 trial」重算 E0/carrier；部署用最小预算 | △新（部署预算 = 全部校准 trial，cycle 意义待定） | △新（rSyn3 探过预算 10/6/4/2 对 carrier 的影响，未在 decoder 训练里 cycle） | **✓证**：C1 prefix-cycle M7/M5/M4 → M3 官方 HO 0.284 vs 固定 M7 的 T0 +0.043；C2 0.376 | 这是你说的「前缀循环 / M3-aware」的原意（H1 血统词汇）。实现代价：bank 不能 session 冻结，E0/carrier 要按 (session, M) 预算多份缓存；静态折叠在推理侧仍成立（部署时 M 固定）。 |
| CAL-2 | 固定预算冻结 bank（工单现状） | ✓（S1 M33） | ○（M10） | ○（M3 固定，formal12） | 最省；但 H1 上已证不如 CAL-1。 |
| **CAL-3** | **identity 用法**：(a) post_pool `d_e=W` concat（现状）；(b) 取 post_pool **之前**的 `joined`（h + 4）低维 concat；(c) SPINT 式：把 E0 时间对齐**加到** spike 窗；(d) E0 置零只留 carrier；(e) E0 列置换对照 | (a) ✓证 0.390 | (a) ○ | (a) ○ 0.32；(b)(c)(d)(e) = 工单 §6 对照臂 | (b) 让 token_in 三任务同宽（16 + h + 4），是真正的「一份结构」；但 M2 先按 (a) 复刻，再单独开 (b) 格子。 |
| CAL-4 | carrier 是否重复出现：E0 内已拼 carrier 时，token 是否再 concat 一次 | ✓（S1 两次都有） | —（M1 只有 token 一次） | ○（两次都有） | 保持各任务血统现状，不统一。 |
| CAL-5 | 单元列序 = `units` DataFrame 行序；unit_mask 全 1（M2）或按任务 | ✓ | ✓ | ✓ | 不是选项，是合同（P2-13），列在这里防漏。 |

### 3.2 训练制度 `TRN-*`

| 编号 | 选项 | M2 | M1 | H1 | 说明 |
|---|---|---|---|---|---|
| **TRN-1** | S1 配方：warmup 1 ep → cosine 到 0.1×，peak 3e-4，24 ep，AdamW wd 0.01 clip 1，EMA 0.9995，unit dropout 0.10，batch 32 | **✓证** | △新 | △新（formal12 只差 cosine/24 ep/peak；EMA、p0.1、wd、clip 相同） | **按 update 数迁移**，不是 epoch：M2 3165 upd/ep，H1 731 upd/ep。EMA 0.9995 视界 ≈ 2000 update = M2 0.6 ep / H1 2.7 ep；warmup「1 ep」在 H1 只 731 步。迁移前写明换算。 |
| TRN-2 | 恒定 LR 1e-4（formal12 / S0 legacy） | ✗（S0 劣于 S1） | — | ○（formal12） | 不推荐。 |
| TRN-3 | whole-unit dropout p=0.10（key_padding 级） | ✓ | ✓ | ✓ | 三任务通用；receipt 需可复现 generator（seed, epoch, batch）。 |
| TRN-4 | 冷启动左清零 prefix（p=0.5 把窗左侧历史置零） | — | — | ○（formal12 用过，结果低，不算证据） | 与 CAL-1 不是一回事：这是输入历史增广，不是校准预算。 |
| TRN-5 | 输入长度 L_in = W + P：P=0 / P=W / P=350 | S1 是 **P=0**；P=50 △新 | △新 | P=350 △新，且见 SPD-H1 | P>0 每步时间核代价按 L² 涨；H1 先看 §5 再定。 |
| TRN-6 | 微批 + 累积：H1 microbatch 8 → effective 32（formal12 实录） | 直上 32 | 直上 32 | **必须**（N=176 时单样本一张 `[L,N,256]` token 激活：L=700 ≈ 126 MB，L=1050 ≈ 189 MB；反传要留 6–8 张） | 3090 24 GB，batch 32 直上不可能。 |
| TRN-7 | H1 空间 preset unscaled-dot / local-balanced | 无 | 无 | ○（QueryAge 家族显式用） | 前端例外，receipt 单列（P1-8）；不是 CausalPE 的一部分。 |
| TRN-8 | 尺度桥：M2 `×5`、M1 `1`、H1 `20y//20` | ✓ | ✓ | ✓ | 合同，验收用 §1 的**比值**式 + `pred_std ≥ 1% target_std`。 |

### 3.3 选点与评测面 `SEL-*`

| 编号 | 选项 | M2 | M1 | H1 | 说明 |
|---|---|---|---|---|---|
| SEL-1 | EMA endpoint（24）+ last-4/8 mean±std | ✓（0.449 / 0.439±0.010） | △ | △ | 工单主统计量；稳态读数。 |
| **SEL-2** | **预注册本地面 epoch-pick**：在 ext4 / LOSO fold / 2,908 minival 上逐 epoch 扫，先写规则再看数 | ✓证（581973 = e8 by epoch-pick，官方 0.390 > e19 0.351 > e24 未提交） | ✓（M1 current_query e6 亦属 epoch-pick） | ○（formal12 = 「earliest maximum on 2,908」） | 581973 的挑选面是 ext6，被审计标为 dev-on-official-selected；**新系列只准在 ext4 / 本地 LODO 上挑**。 |
| SEL-3 | 官方面 | 提交需授权 | 同 | 同 | 零参与选点；只作历史对照。 |
| SEL-4 | 双报 pooled + session-mean | 合同 | 合同 | 合同 | P1-11。 |

---

## 4. 什么不该「统一」

- T4 算子（§2.1）。
- 尺度桥。
- 校准预算数值（M33 / M10 / M3）。
- H1 空间 preset 的有无。
- **不要把 M2 的 0.390 / 0.449 迁到 M1、H1**；不要把 SPINT 系数字写成本系列。

可以统一：8-slot + CausalPE4 结构常数、TRN-1 配方（按 update 换算）、CAL-1 机制（预算集合按任务）、SEL-2 规则、receipt 六行表、A 级加速清单。

---

## 5. 推理加速指导

### 5.1 现状数字（方向用，不是官方口径）

- 官方 `normalized_latency = Σ compute_time / Σ neural_time`（`falcon_challenge/evaluator.py:621`）。20 ms bin 下 0.736 ≈ 14.7 ms/bin 官方 CPU；SPINT 0.044 ≈ 0.9 ms/bin。
- 本机 P95 单线程（同一冻结 SPINT 容器，B7）：SPINT 13.2 ms，QueryAge FLAT/ROUTE 22.1 ms（1.68×）；双线程 10.5 vs 12.8 ms。
- 全路径分量（`REVIEW_UNIFIED_CORE_SPEED_QA_20260906.md`）：conv 2% / token MLP 33–41% / **slot MHA 51–61%** / temporal 4L 5% / readout ≈0。**空间前端占 92–94%**。
- exact-E 缓存后每步：M2 ≈ 9–18 ms；**H1 ≈ 29 ms（前端残差 4.2 + 时间核 L1–3 25.1）**。缓存前 H1 370–613 ms/步——581940/581942 就是这样超时的。
- 为什么 SPINT 快：它每步只做一次 set attention（N 个 token，token 特征 = W 维 spike 窗 + identity，`fc_in` 一个 GEMM），**没有 per-bin token、没有时间栈**。本系列是 N×L 个 token + 4 层时间栈；不缓存就是 W 倍工作量。

### 5.2 A 级：等权、必须逐点 parity（FP32 `max|Δ| ≤ 1e-6` + 全预测集 digest）

按收益排序，全部与 `model.py` 现有接口兼容：

| 编号 | 做法 | 收益点 | 状态 |
|---|---|---|---|
| **SPD-A1** | 第一层静态折叠：`static = W_e0·E0 + W_carrier·carrier + b` 只算一次；同时预计算 `slot_norm(slots)` 的 Q 及其 in-proj | 消掉第一层静态 GEMM + `expand/cat` 物化；官方 581973 wrapper 还在做整段 cat | `forward_static_folded` 已有；slot-Q 预计算未做 |
| **SPD-A2** | 前端逐 bin 缓存：保留最近 4 个 raw bin 做 k=5 conv，只对新 bin 的 N 个 token 走 token MLP + slot MHA，fused `[L,256]` 用环形缓冲 | 前端从 O(N·L) → O(N) /步 | M2 wrapper `_FrontendWindowCache` 已做；本包未做 |
| **SPD-A3** | 时间核 L4 只算 last-query（第 4 层只需最后一个位置的 Q；K/V 仍来自第 3 层全长） | 省 1/4 时间核 | M2 wrapper 已做 |
| SPD-A4 | 固定形状：预分配所有中间 buffer，`torch.inference_mode`，`set_num_threads(1)`（官方单线程口径），去掉 Python 层 dict/日志/断言；SDPA `is_causal=True` 走 fused 核 | 小对象与调度开销，在 ms 级别里占比不小 | 未做 |
| SPD-A5 | `torch.compile`（固定形状、`mode="reduce-overhead"`）或导出 TorchScript；LN+Linear 融合 | 10–30% | 未做；须在冻结路径上 parity |
| SPD-A6 | inactive 单元行冻结（mask 掉的单元不进 token MLP） | N 小时无用；H1 176 全用 | 需证明 masked 行不影响输出 |

A 级极限：前端 ≈ 1–4 ms/步；时间核 L1–3 仍要对全长 L 重算（原因见 5.3）。**M2/M1（L ≤ 200）A 级足以逼近 SPINT；H1（L = 700 或 1050）A 级地板 ≈ 20–30 ms/步，官方 CPU 上 normalized latency 仍会 > 0.7，不可接受。**

### 5.3 为什么 L1–3 不能像 L4 那样只算最后一个位置

窗内正弦 PE 是**窗相对**的：每前进一步，所有位置的 PE 都变，第 1 层所有 hidden 都变，逐层传递到第 3 层。第 4 层的 last-query 需要第 3 层全长 K/V，所以 L1–3 必须全长重算，每步成本 ≈ 3 × [L·(4d² + 2d·ffn) + L²·d]：

- M2 L=50：≈ 0.1 GFLOP → 1–2 ms
- H1 L=700：≈ 1.9 GFLOP → 25 ms（实测 25.1）
- H1 L=1050（P=350）：≈ 3.3 GFLOP → **≈ 45 ms**

跨窗 KV 缓存在这个 PE 下**不是等价变换**（老位置的 hidden 是用更长的历史算出来的），所以被归为 C 级：换了模型。

### 5.4 C 级：改模型，须重训 + 质量门，581973 数字不继承

只对 **H1** 必要；M1/M2 不要碰。

| 编号 | 做法 | 每步时间核成本 | 代价 |
|---|---|---|---|
| **SPD-C1** | **流式 KV 缓存 + 平移不变位置编码**（RoPE / ALiBi / 相对偏置），训练也用同样的流式语义（TBPTT 或滑窗 KV 记忆） | O(L·d)（只算新 token 的 QKV，读 L 个缓存 K/V）→ H1 ≈ 1 ms | 感受野变成无界，训练/推理必须同一语义；是新模型 |
| **SPD-C2** | **局部带宽 b**：每层只看最近 b 个位置（感受野 4b），配相对 PE + 长 b 的 KV 缓存 | O(b·d)；b=64 → 亚毫秒 | 比 C1 更贴近有限窗；仍是新模型 |
| SPD-C3 | 时间下采样：进时间栈前把 fused 按 2–4 bin 池化（stride），L → L/s | 1/s² | 分辨率损失；末 bin 需单独保留 |
| SPD-C4 | 缩 L_in（H1 用 L=256 而非 700）——**注意 W=700 只是 SPINT 的窗定义，P0-4 只约束目标坐标 `(session,end)`，不约束模型输入长度** | (256/700)² ≈ 1/7.5 | 是否丢信息要靠对照；Original cold≈0.956 说明 SPINT 不依赖 14 s 互读 |
| SPD-C5 | 时间栈层数 4 → 2 | 1/2 | 质量门 |

**给 H1 的建议顺序**：先做 SPD-A1/A2/A3/A4 在冻结 M2 路径上拿到真实 A 级地板（这一步和 H1 无关，但把工具链做出来）；H1 主线开训**之前**在 C1 / C2 / C4 中选一个并写进合同——不要先用全窗 CausalPE4 + P=350 训 24 ep 再发现超时。P=350 在 H1 上等于把 5.3 的成本再乘 2.25。

### 5.5 不要做的

- INT8 / FP16 / 剪枝当作「加速」：改数值，不是 A 级；官方 FP32 口径。
- 在未缓存全路径的旧数（370–613 ms）上做 go/no-go。
- 把本机 7950X 数字换算成官方 CPU 数字；只能用来排序。
- 未写 parity 证明就在 wrapper 里开跨窗 KV（M2 wrapper 里显式 `raise`，保留这条）。

---

## 6. receipt 里怎么登记选项

每次训练/评测 receipt 加一段：

```
picks:
  CAL: [CAL-1 {budgets: [7,5,4,3], deploy: 3}, CAL-3a, CAL-4]
  TRN: [TRN-1 {updates_per_epoch: 731, warmup_updates: 731, ema_horizon_updates: 2000}, TRN-3, TRN-6 {micro: 8, accum: 4}, TRN-8]
  SEL: [SEL-2 {surface: "2908-minival", rule: "earliest max"}, SEL-4]
  SPD: [SPD-A1, SPD-A2, SPD-A3, SPD-A4]
```

再附 NOTE §6 六行表。缺编号或缺六行表的数字不得写进任何对照。

---

## 7. 证据入口

- S1 配方与数字：`tfpd_exploration/src/m2_b_small_stability_v1/config.py`，`results/m2_b_small_stability_v1/20260905_123000/comparison.json`
- 581973 官方记录：`tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/OFFICIAL_581973.md`
- H1 prefix-cycle 证据：`tfpd_exploration/h1_series_20260830/docs/HANDOFF_H1_SUCCESSOR_AGENT_20260903.md` §（C1 vs T0，M3 部署预算）
- H1 formal12：`tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/receipt.json`
- 速度分量与 A/C 级分类：`tfpd_exploration/docs/REVIEW_UNIFIED_CORE_SPEED_QA_20260906.md`
- 本机 SPINT vs 家族测速：`tfpd_exploration/docs/PROGRESS_CRST_FAMILY_20260906.md`（P95 表）
- M2 exact-E wrapper：`tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py`（`_FrontendWindowCache` / `_ExactEEngine`）
- 静态折叠参考实现：`tfpd_exploration/src/two_mainlines_long_v1/latency_opt_v1/e_static.py`
