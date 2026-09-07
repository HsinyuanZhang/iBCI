# Handoff: 合并路线图审核（两份外部分析 + 操作侧修正）

Date: 2026-08-23
Status: 决策图 handoff，供协作者讨论。合并对象：两份外部建议的**原文存档**在
`EXTERNAL_ANALYSIS_FOUR_PATHS_20260823.md`（逐字保留：Part 1 = 四路径分析——
Label-Budget-Matched Carrier Uncertainty / Canonical Identity Frame / Within-Session
Identity Tracking / Calibration-Cost-Normalized Protocol；Part 2 = 高维标签七项分析——
Effective Dimension / Dimension Scaling / Anisotropic Noise / Identity Separability /
D-optimal Snippet Selection / Shared Subspace Embedding / Per-DoF Attribution）。
本文是操作侧对其的合并审核与修正，凡与收据冲突处以本文 §3 为准。前置文档：
`HANDOFF_TFSR_PIRG_REVIEW_20260823.md`（TF-SR screen FAIL + PIRG 审核与收据索引）。

---

## 1. 背景一句话

TF-SR screen 决定性 FAIL（external −0.164 vs D，within +0.022 为唯一亮点）、PIRG 近零
（消费端后验第二次 null）之后，"加 widget"路线五连败（S2/C/W/PIRG×2/TF-SR）。两份外部
分析一致主张：转向零/低 GPU 的结构性贡献，用诊断先裁决再花算力。

## 2. 合并决策图

```
零成本门（先跑，裁决一切；~1-2h 推理级 GPU）
├─ 三线 oracle 诊断（oracle 载体 vs 散布校准 vs 连续校准，M30/M4）
│    含 D-最优覆盖对照——解开"散布胜连续"里覆盖度与时间分散的混杂
├─ 有效行为维数（行为协方差谱的 participation ratio，逐 session）
│    若 H1 有效维数 ≈ 2-3，则"高维展台"叙事坍塌，第 4/5 项降级
└─ 逐 DoF 增益归因（纯记账；预言增益集中于良态方向 = 可证伪测试）

分支（oracle 诊断裁决，只选一条 GPU 路径）：
├─ oracle ≈ 现状 → 瓶颈不在载体估计 → 家族 B（规范标架）为唯一升级；
│    不确定性线预测 null（PIRG 双 null 已在 2D 印证）
├─ oracle ↑ 且 散布≈oracle → 瓶颈=会话内漂移 → 家族 C（跟踪）升主贡献
└─ oracle ↑ 且 散布≈连续   → 瓶颈=估计方差 → 家族 A（不确定性）升主贡献

家族 A（不确定性）：预算标定曲线（预测→命中）+ 跨数据集维度标度
   + 各向异性修正（沿弱探索方向收缩；Mahalanobis 距离全面替换欧氏角距离）
家族 B（规范标架）：canonical/gauge-fixed 身份命题（activity 空间 ID 仅在不特定
   变换群下可辨识 vs 行为空间载体被任务规范）+ 共享子空间部分迁移
   （掩蔽必须实现为边缘化而非零填充——与 AM/IM 的 placeholder 教训同构）
家族 C（会话内跟踪）：RLS/Kalman 递推载体更新，解码器输出做伪标签；
   完全活在 SPINT 自称制度内（无标签无梯度）→ 唯一头对头战场；
   fail-closed 触发器用 Mahalanobis 度量（各向异性修正），否则被最噪方向支配
独立候选：D-最优校准选段（"选得好的 2 分钟胜连续 4 分钟"）——最可能升级为
   真第二贡献：它是载体估计器的逻辑推论而非附加模块，且顺手修复诊断混杂
装甲（全程并行）：校准成本归一协议 + 充分统计量命题（诚实口径）+ 域随机化重构
   + 语义化重命名 + 混合效应统计
```

## 3. 操作侧修正（对照本仓库收据；讨论时以此为准）

1. **"margin 在 2D 收窄到零"不成立**：d=2 的 sub-M 上 carrier vs 纯 activity 身份
   margin ≈ 0.5（Z4 external −0.146/−0.119/−0.111 三 seed + spintshape_z4 −0.294
   vs carrier 系 0.34–0.42）。正确表述：margin 在 d=2 已存在，预言其随有效维数
   扩大或饱和——这是带符号的机制预言，不是排行榜数字。
2. **D-最优选段的预算记账**：按 realized behavior 选段 = 使用候选池行为元数据
   （=解码目标），必须计入预算。诚实版本 = **按 cue 选段**（任务设计方先验已知，
   临床免费、审计干净）。评测协议里按 cue 版写。
3. **H1 解析预测的先决条件**：维度标度预言（同一闭式理论预报 7D 曲线）只在
   修好的估计器上可计算——冻结源域行为基 + 设计矩阵中心化/标准化 + ridge/GCV
   是前置，不是并列项（H1 裸 OLS 已失败：Date2 反转、截距 60× 尺度）。
4. **2D 消费端双 null 的重述**：M30 前缀下载体已准、2D 各向异性小——精度信号
   在该制度本无利可图；PIRG/posterior-consumer 的 null 因此是**制度依赖预言的
   印证**而非单纯失败。写作时按此救回。
5. **SPINT 不可能性表述降级**："structurally impossible" → "不额外锚定则不规范化"
   （审稿人可用锚定方案反驳强版本）。
6. **Hungarian 基线不豁免**：Path 2 的"更简"是方法主张；帕累托装甲里照样跑。
7. **M1/M2 协变量定义需先核实**（EMG vs 运动学、名义维数）——第二份分析自己
   也标了不确定；它决定第 1/2/6 项的数字。
8. 附带：TF-SR 的 two 收获（within 0.5919 首超 A2 全 seed 的 within/external 权衡
   实证；载体必要性跨主干表 zero 即崩 −0.635）应进论文负结果边界章节。

## 4. 与在跑/未决事项的关系

- TFSR seed 43（epoch ~40/48，jit 构建加速）约数小时后落地；GO 复制目的已失效，
  仅余 within/external 分裂的第二种子点价值。处置待操作者。
- Cell-D seeds 43/44 仍是任何 A2 优越性主张的硬前置（已四次顺延）；与分支选出的
  GPU 路径竞争同一批算力，需一并拍板。
- 五连败模式（S2/C/W/PIRG×2/TF-SR）= "别再上 widget"禁令的实证版本；两份外部
  分析的所有候选均为零/低 GPU 结构性贡献，与该禁令一致。

## 5. 建议的讨论议程（按序）

1. oracle 三线诊断 + PR 计算是否立即授权（推理级，~1-2h GPU）
2. H1/M1/M2 协变量与名义维数核实（读文档，零成本）
3. 分支预判：各自押哪个瓶颈（漂移 vs 方差 vs 都不是）——预测先行，诊断裁决
4. D-最优 cue 选段是否单列为第二贡献候选
5. GPU 算力分配：分支路径 vs Cell-D seeds 43/44 vs 两者
6. TF-SR seed 43 停/跑完

## 6. 收据索引（同 HANDOFF_TFSR_PIRG_REVIEW_20260823.md §5，增补）

- Z4 三 seed external：`results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json`
- D 家族与机制分解：`results/subpop_score_v1_r2/`、`results/aimask_score_v1/`、
  `results/subpop_step0c_v1/`
- H1/M1 前史（第一份外部分析 §5 的处方对象）：见其引用的 H1 侧收据
- FALCON 数据文档（协变量定义核实用）：
  FALCON NeurIPS 2024 Datasets & Benchmarks track（biorxiv 2024.09.15.613126）

## 7. 2026-08-23 低成本诊断终局（已执行）

本文 §2/§5 的零/低 GPU 门已全部执行或核对现有封存结果。新增评估使用同一个
sealed Cell-D SWA、同一批 query 输入、同一 M30 B3S calibration activity、同一
sealed ordinary-OLS normalizer；仅替换 T4 的标签支持集。全程无训练、无 target
backward/update/optimizer step，formal 未打开。C2/C3 是明确标注的 leakage/oracle
诊断，不是可部署系统。

### 7.1 四支持集定义

- C0 contiguous：前 M 个已标注试次（当前基线）。
- C1 cue-balanced early：只从前 50 个试次中用 cue 做 greedy D-optimal 选择。
- C2 cue-matched scattered：保持 C1 的方向多重集，但把试次分散到整个 session。
- C3 full-session oracle：使用整个 session 的真实标签拟合 T4。

### 7.2 固定模型的 governing R2

| surface | budget | C0 contiguous | C1 early balanced | C2 scattered | C3 full oracle |
|---|---:|---:|---:|---:|---:|
| within-6 | M4 | 0.3815 | 0.4403 | 0.3600 | 0.6082 |
| external-15 | M4 | 0.1147 | 0.1858 | 0.2375 | 0.4449 |
| within-6 | M30 | 0.5697 | 0.5607 | 0.5965 | 0.6082 |
| external-15 | M30 | 0.4179 | 0.3987 | 0.4204 | 0.4449 |

配对结果（平均 delta；10,000 次 session bootstrap 95% CI，seed 42）：

- M4 C3-C0：within `+0.2266` `[+0.1448,+0.3199]`，6/6 正；external
  `+0.3303` `[+0.2518,+0.4042]`，15/15 正。M4 的 carrier-estimation ceiling 很大。
- M4 C2-C1：within `-0.0803` `[-0.2533,+0.0594]`；external `+0.0517`
  `[-0.0204,+0.1254]`。单独的时间分散效应不稳定。
- M4 C3-C2：within `+0.2481` `[+0.1254,+0.3781]`，6/6 正；external
  `+0.2074` `[+0.1505,+0.2673]`，15/15 正。剩余缺口主要是大量真实标签/样本，
  不是只把同样 4 个 cue 分散到 session 全程。
- M30 C3-C0：within `+0.0385`，external `+0.0270`；C2-C0 分别只有
  `+0.0268` 和 `+0.0025`。现有 M30 载体已接近实用上限。

### 7.3 D-optimal 与载体忠实度

在 within-6 真实 replay 中，D-optimal 相对 chronological 的 carrier-fidelity
中位增益是 M10 `+0.0161`、M15 `+0.0093`，未达到预注册的两个预算均
`>=+0.03` 门，因此 FAIL。C1 在 M4 解码上有中等正值（within `+0.0588`，
external `+0.0711`），但符号不广泛，且 M30 反而下降。它可作为短前缀协议
技巧，但不足以作为第二个主贡献。

### 7.4 有效维数与逐 DoF

- M2 评估目标的 participation ratio 几乎是满 2D：within 均值 1.975，
  external 均值 1.923。改善不是只在一个坐标/一个特征向量上出现。
- H1 现有 CPU screen 中，7D 行为的 participation ratio 均值为 3.120
  (full recording)、2.902 (M4 support)；90% 方差中位数需 4 个成分。
  所以 H1 不是真正的 7 个独立轴，但也不是 2D 问题。
- H1 封存 query-label oracle 已给出反向结果：support carrier R2 0.52551，
  query-oracle 0.52265，delta `-0.00286`。因此 M2-M4 的大 oracle headroom
  不能外推成“所有数据集都需更好 carrier”。

### 7.5 更新后的分支裁决

1. **M30 性能路线**：关闭 D-opt/cue selection、posterior credibility、简单时间分散
   作为主攻方向。oracle 只剩约 `+0.03`，不足以解释对 A2 或新架构的大缺口。
2. **M4 短标定路线**：确认了一个很大但未被当前方法兑现的 ceiling。证据不支持
   `scattered ~= oracle`，所以按原决策图，不能自动把简单 RLS/Kalman tracking
   升级为主贡献。若继续，必须是一个单独、性能导向的方法：利用大量未标注
   query 但必须用高置信伪标签/不确定性门防止自强化漂移，而不是继续做
   posterior/PIRG widget 消融。
3. **H1**：carrier-refinement/tracking 关闭；query oracle 已表明此通道没有可见上限。
4. **论文可用贡献**：保留 calibration-cost-normalized protocol、M4->M30->oracle
   退化曲线、D-opt 负结果和 H1/M2 反例。这是很强的机制/边界故事，但不伪装成
   新的排行榜突破。

### 7.6 封存产物

- governing diagnostic receipt:
  `results/low_cost_calibration_diagnostics_v3/receipt.json`, SHA-256
  `c1adfd9f06d737a44a2b7120ec7fa82b844c30fed71bb618d8535195ab91694c`。
- terminal:
  `results/low_cost_calibration_diagnostics_v3/terminal.json`。
- 两个诚实 fail-closed predecessor 被 V3 绑定：V1 是 opaque-session module alias
  错误（无 experimental forward）；V2 是 float64 audit R2 与 governing float32
  R2 的口径错配（C0 prediction SHA parity 已通过）。

## 8. 2026-08-24 M4/M10 完整比较与路线终局

操作侧继续执行了全部低成本比较，并只给一个性能导向训练设计完整 48 epoch：
Calibration-Budget-Marginalized Cell D（CBM-D）。本轮共新增 204 个 score cells：
Phase-1 48、因果协议 factorial 32、原始 SPINT/Arm-A 12、lambda curve 72、
CBM-D terminal matrix 40。统一使用 last-bin、variance-weighted session R2、
equal-session mean 和逐 session 配对统计；target optimizer/backward/update 均为 0。

### 8.1 低成本可采用结果

- 固定 normalized ridge `lambda=0.1` 是部署默认。相对 OLS，M4 external
  `+0.0516`（11/15），M10 `+0.0119`（9/15）；per-session GCV 因短前缀
  过度正则化而失败。
- cue-only、严格 first-30 的 D-optimal selection 只在 M4 开启。对 total-calibration
  external，D-opt 相对 chronological：OLS `+0.0684`（11/15，CI 全正），ridge
  `+0.0532`（9/15，CI 全正）。到 M10 只有 `+0.0049/+0.0028`，且 within 下降，
  因此 M10 起关闭。
- 原始 SPINT B0 的 external 在 M4/M10/M30 全负（`-0.123/-0.135/-0.093`）；
  activity-only identity 可以提高 held-in，却不能形成跨被试身份。Cell D 相对
  Arm A 的 external 优势随预算增加：`+0.041/+0.082/+0.158`，再次确认整-unit
  dropout 主要买到跨被试鲁棒性。
- 当前诚实 total-calibration 部署表：M4 Cell D + D-opt + ridge = `0.1197`
  external；M10 Cell D + chronological + ridge = `0.2955`；M30 同配方 = `0.4286`。
  M4 label-limited `0.1902` 需要 B3S 仍看 M30，不能冒充四试次系统。

### 8.2 CBM-D 的决定性负结果

CBM-D 保持 Cell D 图和 3,510,842 参数不变，source batches 让 B3S activity 与
ordinary-OLS T4 共同覆盖每个整数预算 M4..M30。训练完整结束：48 epoch、
1,628,400 steps、loss `0.6015 -> 0.3175`、全部有限性/梯度检查通过。

在 exact recipe-matched external 对比中：

| Budget | CBM-D minus Cell D | Positive | 95% CI |
|---:|---:|---:|---:|
| M4 | `-0.1147` | 3/15 | `[-0.1782,-0.0645]` |
| M10 | `-0.0894` | 2/15 | `[-0.1383,-0.0429]` |
| M30 | `-0.1337` | 0/15 | `[-0.1734,-0.0978]` |

所有预注册门均 FAIL，终局 `STOP`。更有解释力的是 ordinary-OLS/chronological
对比：M4 total-calibration within `+0.1121`（6/6，CI 全正），external 却只有
`-0.0104`；M10 external `-0.0706`，M30 `-0.1195`。因此训练确实学会了 source
held-in 的短前缀适应，但没有学到跨被试 calibration-budget invariance，反而损坏
了原来 Cell D 的 transferable identity。

### 8.3 更新后的决策图

1. **关闭 CBM-D 家族**：不再做 prefix schedule、loss、sampling distribution
   消融。这个 treatment 的 within 正值不能为 external 负值辩护。
2. **保留低成本部署配方**：M4 用 D-opt-first30 + ridge 0.1；M10/M30 用
   chronological + ridge 0.1；权重仍用 sealed Cell D。
3. **论文定位**：低成本部分是 calibration-cost-normalized protocol + estimator
   robustness + 明确的 held-in/external 反例，不是新 decoder SOTA。它与 unit
   dropout 的正结果共同给出一个干净边界：随机化 population composition 可以迁移，
   随机化短 calibration budget 在当前监督目标下不能迁移。
4. **下一条 GPU 路线的资格**：必须是新的跨 session/subject invariance mechanism，
   或带严格置信门的无标签 target adaptation；不得再把 source held-in improvement
   当作 external improvement 的代理。H1 仍不走 carrier refinement，因为其 query
   oracle 已关闭该通道。

### 8.4 新收据

- Phase-1 comparator receipt: `results/calibration_budget_comparators_v1/receipt.json`,
  SHA `0ec107cc95cfb806336e6859e55fb8d7d30c365c2ff829757e100045cfb5c1cc`。
- Protocol factorial: `results/calibration_budget_protocol_factorial_v2/receipt.json`,
  SHA `ce283aa7d046d575ed50860090633a1cab9448ccfba51472ff7694812c7594b2`。
- Original SPINT/Arm-A: `results/original_spint_short_budget_v2/receipt.json`,
  SHA `526dc11460f44d67674287762273da74265fe05fa5aa0c10bab953b65fb06a68`。
- Lambda curve: `results/ridge_t4_lambda_curve_v1/receipt.json`, SHA
  `e82d917b348be09bd3888a924d8523e8c88dc7794d1a8d273aa3b333b725f257`。
- CBM-D training terminal: `results/calibration_budget_marginalized_cell_d_seed42_v3/terminal.json`,
  SHA `8e5c807dbf961cbda673d355b81ae530f6078b02b7aadd06deae92e95d95c3a1`。
- CBM-D matched score V2: `results/calibration_budget_marginalized_score_v2/receipt.json`,
  SHA `09ffbd837be8529ecd0a3cca60bbe010ebb989137346ddb92cf8023f98756f51`。
