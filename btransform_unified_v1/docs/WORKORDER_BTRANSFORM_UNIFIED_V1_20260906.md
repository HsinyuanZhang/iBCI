> **ADDENDUM-EXPOSED-EVAL（2026-09-06，Level-1 公平对比完成）**：留出日 exam 面（2,952 窗，坐标逐位 join，暴露核验闭合——Original calib 27 键含 S5_set_1/2、C2 terminal m3_carrier_sessions 含该日）：
> | 系统 | 暴露 | pooled | eq-mean |
> |---|---|---|---|
> | Original (W=700) | exposed | **0.9635** | 0.9633 |
> | C2 e15 (W=700 流式) | exposed | **0.8873** | 0.8885 |
> | M-F250 (L=250 proj_add, clean) | clean | 0.5492 | 0.5522 |
>
> **判读：GAP_SURVIVES_EXPOSURE**——SPINT 系 exposed 数字与重叠面数字（0.961/0.888）几乎零缩水（重叠红利 ≈0），F250 0.552 与冠军 ~0.41 的差距不是 clean-vs-exposed 伪影。公平天平**不**向 F250 移动；Level-2（C2-LOSO 重训）为金标准、待授权。**对验收律的含义**：H1 上 proj_add 0.5522 尚未达到 exposed-SPINT 原始量级——"显著超过 SPINT 原始"在 H1 的可行面是官方 HO（Original 0.262 / C2 0.376，两系统隐测大幅缩水）或 Level-2 匹配训练；此为后续 cell 与提交设计的硬约束。
> **ADDENDUM-UNIFIED-ADD（2026-09-06 用户裁定，接口终审）**：**统一 identity 接口 = proj_add（加法），全任务适用**。joined 对比 cell 即刻作废（SUPERSEDED）。**验收律**：各任务选点后须**显著超过 SPINT 原始**（操作化：超出 SPINT 原始在治理面的读数 + 0.03，超出量须大于配方已知噪声）。当前对线状态：
> - **H1**：M-F250@1e-4 留出日期 0.5522（干净面）——对 SPINT 原始的公平读数 = exposed-Original 在同面的评测（Level-1，运行中）；官方 HO（Original 0.262 / C2 0.376）为干净跨系统参照，F250 提交另行授权。F150（L=150）运行中；F700 记败（L=700 平台 L 依赖发现）。
> - **M2（2026-09-06 用户更正，判定改写）**：endpoint24 = 0.3407 **不是最高点**——SEL-2 选点（ext4 earliest max）= **e9 0.4016 > SPINT 原始 0.3582**，且 SPINT 原始的 0.3582 本身是 visible 选点产物，endpoint-vs-选点的不对齐比较对 proj_add 不公。**改判：M2 proj_add@P16 在 SEL-2 口径下已过线**（+0.0434）。P32 加宽重试继续（目标：抬升 endpoint 与选点稳健性，非救急）。endpoint24 读数保留为稳态辅报。
> - **M1**：**proj_add 系列实验准备中**（用户指令）：训练方法 = 已验证的 M2 噪声对齐 TRN-1（S1-grade：seed42、S1 域 dropout、F.conv1d、bf16、EMA 0.9995、cosine），LR 起 1e-4（H1 教训：3e-4 需按任务验证）；P 维度系列 {16, 32}；对比目标 = Original minival 0.809（同面、标注暴露）与官方 0.649。


> **ADDENDUM-M2-PROJADD（2026-09-06 用户指令）**：M2 封存解锁一个接口验证 cell——**M2 + identity_mode=proj_add**（E0 50→16 投影加到 local；GPU1 上 M-F250 完成后接续执行）。目的：若 proj_add 在 M2 也成立，三任务共享同一 identity 接口（"结构统一"升级为"接口统一"）。设计：与 p1a_v2b 逐项同构（同 seed42/manifest/TRN-1/噪声对齐路径/CAL-2/M33/ext4 面），唯一变量 = identity_mode；**基线 = p1a_v2b concat seed42 逐位孪生 0.4501（同种子配对比较）**。门（预注册）：endpoint24 EMA ext4 equal_session_mean ≥ 0.42，配对差 vs 0.4501 照报并注明 mode 切换引入的 RNG 重排属新抽样（种子带 [0.337,0.449] 披露）；≥0.45 则 proj_add 成为 M2 统一接口候选冠军。

> **ADDENDUM-H1-FIRST（2026-09-06 用户裁定）**：优先级改为 **H1 新网络优先**——M1 全 session 训练**推迟**（用户判断 M1 问题可解决；M1 agent 已让出 GPU1，数据接入/会话枚举成果落 DEFERRED receipt 留复用）。执行序：M-F250（proj-add16 用户提案）首跑上 GPU1 → 按矩阵判读推进其余 cell；M1 训练等 H1 矩阵告一段落或用户指令重启。

> **ADDENDUM-M2-SEALED（2026-09-06，GLM5.3）**：路线 B **PASS**（`results/p1a_v2b_m2/20260906_080831/gate_receipt_PASS.json`）——**24/24 epoch train_mse 与 S1 逐位一致**（bitwise，e24=0.0001750921 双方相同）；EMA endpoint24 ext4 equal_session_mean **0.4501** vs S1 0.4495（逐 session 配对差 ≤+0.0014）；SEL-2 pick e19=0.4521。P1a 的 FAIL 确证为 dropout 域 + conv 精度路径两个噪声源所致，无实现缺陷。**M2 线关闭封存**：官方结果已有（581973 HO 0.390），不备新提交；产物 = 逐位复刻配方 + 骨架正确性证明，供 M1/H1 继承。

> **ADDENDUM-REPRIORITIZE（2026-09-06 用户裁定）**：
> **M2**：路线 B（p1a_v2b）PASS 后即封存——官方结果已有（581973 HO 0.390），不备新提交；M2 线关闭，产物 = 复刻配方 + 骨架正确性证明。
> **M1**：本线负责**一次全 session 训练**（全部 held-in session 含 20120924；TRN-1 按 update 口径；CAL-2 M10 血统默认 identity (a) B3-100 concat——未验证选项不混入单跑；checkpoint 规则 = endpoint24 EMA 预注册，不选面），产出封存 checkpoint + 交付契约（尺寸桥 divisor=1、列序、观测合同）移交用户的提交 agent；EvalAI 提交本线不做。LOSO 方法选优若需要，由提交侧另立。
> **H1**：本线主战场，按 [MATRIX_H1_L_IDENTITY_V1](MATRIX_H1_L_IDENTITY_V1_20260906.md) 训练矩阵执行（L × identity 喂法 分数设计，CAL-1 全局启用）。
> 执行顺序：GPU1 路线 B 完成后接 M1 全 session 训练；H1 矩阵等 GPU0 五臂终局 + 矩阵文档生效后执行。

> **ADDENDUM-SUBMISSION-PROTOCOL（2026-09-06，用户裁定，方法学律）**：**两段式协议**——
> **第一段（方法选择，本地）**：在留出 session 的 LOSO 面上做方法/配方/超参/结构选择；此阶段训练数据 = 除留出 session 外的本地 session。禁止用同-session minival 面做非劣/选择结论（实验二已证其失效）。
> **第二段（提交构建）**：配方冻结后，**用本地全部 held-in session 重训**，再提交 EvalAI 在真正未见过的 session 上测试。重训后留出面已被污染：**禁止对提交候选再打 LOSO 面**（该数字无意义），官方分是唯一读数；checkpoint 规则从第一段迁移（如 endpoint24 或预注册 update 数），不因全量数据重选。
> 推论：实验二的 −0.140 是「家族(3 session) vs Original(全 session)」的不对等比较——Original 见过 20120924 在本协议下是**合法的**（它就是第二段产物），故 −0.14 高估方法差距；但「minival 面遮蔽跨 session 掉分」的选择器结论不受影响。本地与 Original 的任何对比须标注双方训练 session 集。

> **ADDENDUM-THREE-CHEAP（2026-09-06，GLM5.3 采纳 [RECEIPT_THREE_CHEAP_DECISIVE_20260906.md](../results/RECEIPT_THREE_CHEAP_DECISIVE_20260906.md)）**：
> (1) **P2a 门改面**：主开发面从 31,252 source-minival 改为 **LOSO 外层 session（20120924，26,496 窗）**——receipt 实证该面判平（0.812 vs 0.809）而 LOSO 面 −0.140，minival 面作为选择器失效；31,252 降级为诊断面。参照 Original LOSO 0.798 附泄漏披露（其训练含 20120924，非真 LOSO）；门数值待 P2a 设计时按此裁定，禁止再用 31,252 单面下非劣结论。
> (2) **P2b-L 风险重估**：H1 Original 掩蔽扫描显示信息在长尾（7s=0.486、14s=0.961；掩蔽为 OOD 操作，与 cold-start 鲁棒不矛盾）——L-sweep 从"低风险白捡"改判"必须实验裁决"；M1/M2 短历史有支持（M1 1s=0.775/2s=0.809；M2 0.5s=0.41/1s=0.548）。P2b-L 三结果预案维持（≤0.01 平/0.02 每 100bin 视界/证伪）。
> (3) **五臂中期读数**（L=100, e12）：a_concat700 0.080 / b_joined36 0.178（PROXY）/c/d/e 待落盘——低维 identity 方向领先，若终局维持则 H1 选 CAL-3(b) 并更新 §2/§4。

> **ADDENDUM-P1a（2026-09-06，GLM5.3）：P1a 已执行，门检 FAIL 原判保留**（`results/p1a_m2/20260906_072314/gate_receipt_FAIL.json`，ours e24 EMA ext4 = 0.3767 < 0.42；10-30-R1 −0.151 / 11-19 −0.197 两 session 违配对门）。**诊断结论：无实现缺陷证据**——(1) S1 e24 EMA checkpoint 过本包模型+评测路径 = 0.4501 vs S1 记录 0.4495（逐 session 0.575/0.393/0.501/0.332，实现级无损的最强证明）；(2) 配方种子级方差：S1 seed43 同配方 endpoint24 = 0.3373（同样不过 0.42 门），种子带 [0.3373, 0.4495]，我们 0.3767 落带内且呈 seed43 式平台形态；(3) 复刻前提逐项断言通过（state_dict 键集一致、init 逐位相同、manifest/lr/EMA/bf16/×5 核验一致）。残余差异源（已声明噪声类）：dropout seed payload 域不同（RNG 流不同）+ 逐 tap conv 在 autocast 下走 fp32 而 S1 的 F.conv1d 走 bf16。
> **P1a-v2 门（预注册，替代原单种子门）**：二选一，须在开跑前选定写入 receipt——(A) 统计门：补 seed 43/44 两跑，三种子均值 ≥ S1 双种子均值 0.3934 − 0.01 且三种子内最小值 ≥ 0.3373 − 0.02；(B) 噪声对齐门：将 dropout payload 域与 conv 前向（F.conv1d+autocast）改为与 S1 逐位同源后单跑 seed42，门= endpoint24 ≥ 0.42（若对齐后仍 <0.42 则升为实现缺陷嫌疑，回到失败协议）。骨架正确性主张以对照 (1) 为准已成立；P2a/P2b 不因此阻塞。

- v2 修订：2026-09-06，GLM5.3。依据 [REVIEW_WORKORDER_BTRANSFORM_UNIFIED_V1_REQUIRED_CHANGES_20260906.md](REVIEW_WORKORDER_BTRANSFORM_UNIFIED_V1_REQUIRED_CHANGES_20260906.md)（下称 REVIEW）与 [REFERENCE_TASK_DIFFERENCES_CHERRYPICK_AND_SPEED_20260906.md](REFERENCE_TASK_DIFFERENCES_CHERRYPICK_AND_SPEED_20260906.md)（下称 REF）。阻断项 A–E、重要项 F–K、代码项 L–S 全部落入本版。
- 上位指令（用户）：B-transformer small 加速 CausalPE 统一网络，三数据集；「epoch-pick / M3-aware training / 前缀循环 / tuning profile」。**术语按 REF 落地：前缀循环/M3-aware = CAL-1 校准预算轮换（H1 血统原意，唯一官方证实增益成分）；输入长度前缀 = TRN-5，独立可选项。**
- **身份声明（修订版）**：本系列 decoder = 8-slot + 因果时间核 B-transformer，**不是 SPINT**；与 SPINT 系一切比较只写「同一评分面、不同系统」。**E0/carrier 由 SPINT 系训练的冻结 B3S 系编码器产生**（M2 T4 主线 champion / M1 Sfix e11 / H1 C2 e15），按 NOTE P0-2 披露，详见 §4 表。
- 治理链：[NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN](../../tfpd_exploration/docs/NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md)（P0/P1/P2 合同）→ REVIEW（必改）→ REF（可选项菜单 CAL/TRN/SEL/SPD 编号）→ 本工单。

## 0. 诊断结论（§1 精确化，REVIEW J 项）

1. **训练制度**：H1 formal12 已用 EMA 0.9995、p 0.1、wd 0.01、clip 1；与 S1 配方（TRN-1）相差的只是 **cosine / 24 ep / peak 3e-4**（formal12 = warmup 至 1e-4 后恒定、12 ep，receipt 实录）。
2. **选择律伪影**：M2 QueryAge FLAT 0.143 = source-only 选 e2（欠拟合点），endpoint24 未评；非算子读数。
3. **数据对齐未结案**（H1）：NOTE P0/P2 各项开放。
4. **面差异**（M1）：本地 31,252 全员 ~0.81 vs 官方 0.65；官方面更难。
5. S1 配方（TRN-1）从未在 M1/H1 上用过；统一系列在 M1/H1 的可行性未被真正检验。

## 1. 包与命名

`btransform_unified_v1/`（workspace 顶层）。receipt 原子写+0444+.sha256（RECEIPT_LAW）；每次训练/评测 receipt 附 **REF §6 picks 段**（CAL/TRN/SEL/SPD 编号+参数）+ NOTE §6 六行表，缺任一不得进对照。

## 2. 统一网络规格（结构常数统一；任务改「几何与校准物合同」）

| 项 | 规格 |
|---|---|
| 输入 | [B, L_in, N]；**L_in 训练=推理钉死**（代码 N 项：forward require 长度）；TRN-5 的 P 由各阶段门决定 |
| Local | 每单元 Conv1d(1→16,k=5,左pad4)+SiLU |
| Token | concat(local16,E0,carrier) → Linear(→256) GELU → 256→256, LN；第一层静态分解（SPD-A1）为默认路径 |
| 空间 | 8 learned slot MHA + slot FFN 1024；FLAT only；H1 若沿用 unscaled-dot/local-balanced preset 按 TRN-7 单列披露 |
| 时间 | 8·256→256 + 窗内正弦 PE + 4 层 pre-LN 因果 block（FFN 512）；CausalPE4。**A 级边界注记**：窗相对 PE 下跨窗 KV 非等价（C 级）；H1 若走 SPD-C1 则重训并重定义 exact-E |
| 读出 | 末 bin LN→256→128 GELU→out |

| 任务 | W | N | out | 尺度桥（TRN-8，**比值式**） | 校准预算（CAL） |
|---|---|---|---|---|---|
| M2 | 50 | 96 | 2 | ×5：`MSE(raw,5y)=25·MSE(raw/5,y)`，容差相对 1e-9；pred_std≥1% target_std | **M33**（S1 parity，SUPPORT_HORIZON=33） |
| M1 | 100 | 64 | 16 | divisor=1 | **M10** |
| H1 | 700 | 176 | 7（列序 tx..g3 冻结） | 训 20y 评 /20：`MSE(raw,20y)=400·MSE(raw/20,y)` | **训练 prefix-cycle {7,5,4,3} → 部署 M3**（CAL-1，✓证 +0.043） |

## 3. CAL-1（校准预算轮换，工单 v1「M3-aware/前缀循环」的正式定义）

- bank 按 (session, M) **多份预计算**（M2 {33}、M1 {10}、H1 {7,5,4,3}）；训练时每 (session, epoch) 从集合轮换取用并重算 E0/carrier；推理时 M 固定（H1=M3），静态折叠仍成立。
- TaskBank 增加 budget 必填字段（代码 S 项）；adapters `build_*_bank(..., budget)`（代码 R 项）。
- CAL-2（固定预算冻结 bank）仅作 M2 P1a 复刻用（S1 parity）；H1 主线禁用 CAL-2。

## 4. 校准物合同（REVIEW C 项表，逐任务写死）

| | M2 | M1 | H1 |
|---|---|---|---|
| 编码器 | B3S（hidden 64，side_dim 4，MOVE-T4 拼进 post_pool） | B3 Sfix e11 `student.id_encoder`，`compute_identity(side_features=None)`（rSyn3 **不进** identity） | C2 e15 `carrier_pre_pool(1024→32)`/`carrier_post_pool(36→32→32→700)`，H-C 拼进 |
| d_e | 50 | 100 | 700（CAL-3a 现状；见 §6） |
| carrier 算子 | MOVE-T4（`fit_move_t4`→`t4_from_trial_sums`，对 [1,cosθ,sinθ] 最小二乘） | rSyn3（NNMF rank-3 + ridge λ=1；`RANK=3, RIDGE_LAMBDA=1.0, SUPPORT_TRIALS=10`） | H-C normalized EB（冻结源 PC→7 维速度 ridge→4 维基 U→EB 收缩） |
| 抽取函数 | `m2_dual_track_v1/champion.py::native_e0_and_u`（push_trial/finalize，非 batched-mean） | `m1_optimized_v2/calibration.py::load_frozen_b3`（只抽 id_encoder，禁带 decoder 权重） | `two_mainlines_long_v1/decoder/h1_calibration.py::load_frozen_c2_materializer`（ckpt SHA `ce46267e…`） |

**不统一**：T4 估计算子、尺度桥、预算数值、H1 preset（REF §4）。真 identity 统一（如三任务统一用 B3S hidden 低维，CAL-3b）= P1 复刻后的独立 cell，不得混进 P1。

## 5. 训练/选择/评测

- 配方 = TRN-1 **按 update 数迁移**（代码 Q 项）：warmup_updates / total_updates / ema_horizon_updates(≈2000) 常数化；换算表：M2 3165 upd/ep（24ep=75,960）、H1 731 upd/ep（24ep=17,544；EMA 视界≈2.7ep，warmup 1ep=731 步）。H1 用 TRN-6 microbatch 8×accum 4。
- 选择律 = **SEL-2 预注册本地面 epoch-pick**（M2 ext4 / M1 LOSO fold minival / H1 2,908 minival；规则先于数字写进 receipt）+ SEL-1 endpoint24 与 last-4/8 稳态辅报 + SEL-4 双报 pooled/session-mean。官方面零参与（SEL-3）；**ext6 式 dev-on-official 挑选禁止**。
- 评测面：M2 ext4；M1 31,252 source-minival（与 Original 0.809/Sfix 0.828/QueryAge 0.812 同面）；H1 20,325 完整流 + 2,908 选点，坐标 join Original 档案 (session,end)。

## 6. H1 identity-usage 五臂对照（REVIEW B 项，替代 v1「或证明等价」分支）

**与 P2b-L 的交互规则（L 扫描先于五臂）**：(c) SPINT 式相加臂依赖时间对齐规则，L≠700 时需先定对齐（默认：E0 取末端 L bin 对齐到输入末端）；(a) 的 700 维静态 concat 与 L 无关（静态 per-unit 特征，非时间轴），L 扫描期间用 (a) 合法。顺序 = P2b-L 定 L → 五臂在选定 L 上跑（5 cell 而非 20）；若最优 L 由 (b)/(c) 臂产生且其 d_e 改变，几何表相应更新。

C2 消费 = `src = src + identity`（时间轴逐位相加、共用 fc_in）；本系列 = 静态 concat + 独立 W_e0。**同一物体不同算子，等价分支关闭**。五臂（同骨架/同时间核/同制度，只换 identity 用法）：
(a) post_pool 700-d concat（现状）；(b) post_pool **前** joined 36-d（pooled 32 + H-C 4）concat——token_in 三任务同宽；(c) SPINT 式时间对齐**相加**（定义前缀对齐规则，最简=只加最后 700 bin）；(d) E0 置零只留 carrier；(e) E0 列置换（结构破坏对照）。
**P2b 门 = 五臂结果落盘 + 选定臂写回 §2/§4**；d_e 的 PENDING 由此落地（(a)700 / (b)36 / (c) 不进 token）。

## 7. 阶段与门（REVIEW E/F/I 项修订）

| 阶段 | 内容 | 门 |
|---|---|---|
| P0 | 骨架+smoke（已过 19 tests）+ 本版代码项修复（L–S） | pytest 全绿 |
| **P1a** | M2 真数据，**P=0、init 用 `initialize_decoder`（S1 复刻）**，其余全同 S1（CAL-2/M33） | endpoint24 ext4 `R_session_equal_mean` ≥ **0.42** 且逐 session 配对差 ≥ −0.02（统计量= session-equal mean，明写） |
| **P1b** | TRN-5 P=50 新 cell | 同门 |
| P2a | M1 接入（CAL-1 {10}，TRN-1 按更新数） | 同 31,252 面 pooled+session-mean 双报，**相对 Original 0.809 的配对差**；官方门（≥0.649−0.01）如需判定须用户单独授权提交，不在本工单授权内 |
| **P2b-前置** | H1 延迟预算门：冻结路径测 SPD-A1–A4 真实地板 | 若仍超预算：在 SPD-C1/C2/C4 中选定一条**写进合同后再训**；**P_H1=350 在选定 C 级路线前不得启用** |
| **P2b-L** | **H1 输入长度扫描（SPD-C4 实例化，用户 2026-09-06 指令）**：L ∈ {150, 250, 350}（+700 对照仅在被要求时），CAL-3(a) 现状 identity，CAL-1 {7,5,4,3}→M3，TRN-1 按更新数，其余全同 | 预注册读数：2,908 minival SEL-2 epoch-pick + endpoint；**信息视界判读**：若 R²(L) 随 L 降不降（≤0.01），「有效信息不需 700」假设成立，选最短合格 L 写回 §2 几何表；若显著降（>0.02/100bin），记录视界并回退。每步预期成本（缓存态本机）：150≈5.4 / 250≈7.4 / 350≈10 ms |
| P2b | H1 五臂对照（CAL-1 {7,5,4,3}→M3）→ 选臂 → 正式训练 | 五臂落盘 + 选臂写回 + 2,908/20,325 双报 |
| P3 | A 级加速全量落地 + 冻结路径 parity ≤1e-6 + 延迟报告 | parity 门 + 延迟地板 |

## 8. 资源与纪律（REVIEW K 项）

**双卡由多 agent 共享，任何时刻不假设空闲**；GPU 任务前 preflight（外部 pid + UUID），CUDA_VISIBLE_DEVICES 钉单卡，FP32，TRN-6 显存预算（H1 单样本 L=700 token 激活 ≈126MB、L=1050 ≈189MB，反传 6–8 张）。不改历史 root、不动 EvalAI、不重选 checkpoint、receipt 先记哈希再引用。

## 9. 代码项修复清单（REVIEW §3，随本版执行）

L: `model.py` 改用 `m2_dual_track_v1.decoders.initialize_decoder`（P1a 复刻前提）；M: dropout 可复现 generator（seed,epoch,batch 派生）透传；N: forward 钉死 L_in；O: scale_bridge/plan H1 文案改**比值式**（`MSE(raw,20y)=400·MSE(raw/20,y)`，相对容差）；P: `TASK_GEOMETRY["h1"]["prefix"]` 改 `H1_PREFIX_PENDING` 哨兵；Q: update 口径常数+换算；R: adapters 补校准物来源+budget 参数；S: TaskBank budget 必填。
