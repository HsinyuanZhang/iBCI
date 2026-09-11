# NORM_ONLY 零校准基线：实现指导（M1 / M2 / H1）

日期：2026-09-10。作者：GLM5.3（规划）。**本文只给规格，不包含代码**；M1/M2/H1 的实现由另一 agent 按本文执行。参考实现（已完成、可直接对照）在 688：`btransform_unified_v2/dandi688_bench_v1`（臂 `norm_only`，variant `norm_only`）。

## 1. 为什么要这一级（预注册动机）

现有阶梯的最小一级是字面零 `NONE`（E0=0、T=0、原始输入）。但**它连领域标准的 per-session 输入归一化都没做**——几乎所有实用 BCI 解码器都会用当日校准数据做每通道率归一化（零标签、零反传、零成本）。所以 `NONE` 不是一个公平的操作点，且已在 M1 上被证明比平凡解更差：

| 任务 | NONE（字面零）官方 HO | 说明 |
|---|---|---|
| M1 | **−0.677**（582224） | 比"预测目标均值"(R²=0) 还差——不可部署 |
| M2 | 已训完未提交 | — |
| H1 | 已训完未提交 | — |

机制已定位（`final_ablation_official_v1/m1_srcbank_ho3_v1/receipt.json`）：M1 的 source-frozen bank 在 HO3 上 raw chR² = **−0.7527**，而 **DC-centered chR² = +0.3464**（Δ = +1.10）。崩塌的主体是**逐通道常量偏移**，来自 HO 相对 source 的**发放率抬升（≈2.1× / 3.3× / 1.3×）**，不是信息丢失。

**结论**：论文主表用 `NORM_ONLY` 作为零校准基线；`NONE` 降级为附录里的机制诊断（它已经是官方数字，不删）。`dc-centered R²` **只作统一附录诊断量，不作任何任务的指标**——换指标会把校准贡献从 1.30 压到 0.22，抹掉我们自己的主张。

## 2. 阶梯与嵌套（必须遵守）

四级的 E0 轴严格单调，每级只加一样东西——**这是唯一能与已有 ACT/FULL 臂嵌套的形式**（已有臂都是"原始输入 + 不同 E0"）：

| 级别 | E0 | T（carrier） | 输入 | 已有 |
|---|---|---|---|---|
| NONE | 0 | 0 | 原始 | ✅ 已训 |
| **NORM_ONLY（本文）** | **当日无标签逐单元平均发放率** | **0** | **原始** | ❌ 待训 |
| ACTIVITY_ONLY | encoder(当日活动, 零 side) | 0 | 原始 | ✅ 已训/已官方 |
| FULL | encoder(活动, carrier side) | carrier | 原始 | ✅ 已官方 |

**禁止**改成"输入侧 z-score"——那会让 ACT/FULL 也得重训才能嵌套（9 臂方案）。若 NORM_ONLY 在主表判据下不达标（见 §5），再单独申请升级到输入侧 z-score，并明确披露需要重训 ACT/FULL。

## 3. 构造规格（三任务同式，只换参数）

**身份向量**：

```
rate_sess[u]  = 当日 label-free 校准支持集上，单元 u 的池化发放率（总计数 / 总时长）
E0[u, :]      = z[u] 广播到 e0_dim 维（所有维度同值）
z[u]          = (rate_sess[u] − μ_src[u]) / σ_src[u]
```

- `μ_src[u] / σ_src[u]`：**同一条率统计量在 source（训练 session）上的逐单元均值/标准差**，冻结计算一次，不使用任何标签
- **支持集 = 该数据集 ACTIVITY_ONLY 臂 E0 所用的同一个支持集**（关键：与 ACT 只差"表示丰富度"，不差支持集）
- **为什么是对 source 标准化而不是 raw**：`z[u]` 的语义是"这个 session 的单元 u 相对 source 常态偏离了多少"——**保留漂移信号（这正是网络需要用来抵消输出偏移的量），同时把跨任务的量级统一到 O(1)**。raw 率在不同数据集差一个数量级（688 ~0.05 counts/bin vs M1 EMG 量级），会让 E0 通路的学习尺度失衡
- **禁止跨单元 z-score**（那会抹掉整体水平，正是要暴露的信号）
- padding 行（Nmax 填充）E0 = 0，且 `unit_mask` = false（沿用各数据集既有 padding 纪律）

**carrier**：全零（`T = 0`），形状与支持集一致。

**输入**：原始神经数据，不做任何归一化（与 NONE/ACT/FULL 一致）。

**训练**：**从零重训**，不在任何已有权重上换 bank——否则会混入已学到的标签信息。除上述两点外，训练配方与本任务 FULL/ACT **逐字相同**（同 seed、同 epoch 数、同 optimizer/EMA/dropout、同选择规则）。

## 4. 各任务参数

| 任务 | E0 支持集（与 ACT 相同） | e0_dim | carrier 支持 | 输出维 | 训练配方 |
|---|---|---|---|---|---|
| **M1** | 当日 label-free 神经活动 **M10** | 100 | M10（置零） | 16 | RIFT R100 / concat，seed42，24ep × 6665 = 159,960 updates |
| **M2** | 当日 label-free 神经活动 **M33** | 50 | M10（置零） | 2 | RIFT R50 / concat，seed42，24ep |
| **H1** | 当日 label-free 神经活动 **M3** | 700 | M3（置零） | 7 | RIFT R300，seed42，32ep（23,392 updates） |

命名建议（与本轮既有目录一致）：`formal_<task>_norm_only_s42_v1`，臂名 `NORM_ONLY`。

## 5. 判据（预注册）

**主判据（本地，各任务自己的 HO 面）**：
- **M1**：HO3 channel-weighted R²（chR²）**≥ 0.3**，且每 session 的 DC 罚项 **< 0.1**（DC 罚项 = raw chR² − DC-centered chR² 的绝对值）
- **M2**：EXT6 等权 R² ≥ **0.15**（M2 的 ACT 官方是 0.093，故零线不应高于它）
- **H1**：HO-M3 grouped-7 ≥ **0.25**（ACT 官方 0.345 是上界）

**合理性检查（必须同时成立，否则构造有 bug）**：
```
NONE ≤ NORM_ONLY ≤ ACTIVITY_ONLY ≤ FULL     （逐任务，同一面）
```
- 若 `NORM_ONLY < NONE`：构造错误（归一化反而更差），停下来查
- 若 `NORM_ONLY > ACTIVITY_ONLY`：说明均值率比 encoder 身份更强，需在论文里如实报告并重新审视 ACT 的设计
- 若 `NORM_ONLY > 0.1` 但 **DC 罚项仍 > 0.1**：说明偏移没修好，主表引用时需附 DC 罚项

**不达标的后备**：若 M1 的 HO3 < 0.3，升级为**输入侧每通道 z-score**（用当日校准统计量，label-free），并披露需要重训 ACT/FULL 才能保持嵌套。

## 6. 与 688 实现对齐的要点

688 已完成同一级的实现（臂 `norm_only`），三任务应与它**逐项对应**：

| 项 | 688 | M1/M2/H1 应对应 |
|---|---|---|
| 身份来源 | 当日无标签校准活动的逐单元率，对 source 分布标准化 | 同（换成各自支持集与 source 名单） |
| 身份形态 | 广播到 e0_dim，O(1) 尺度 | 同（e0_dim 各任务不同） |
| carrier | 全零 | 同 |
| 输入 | 原始（不变） | 同 |
| 训练 | 从零重训 | 同 |
| 嵌套位置 | NONE < norm_only < f_labelfree < t4 | 同 |

**688 的实现可作为对照读数**：`results/train_exp1_narrow_norm_only/`（score receipt 落盘后），可与 `floor=0.3247 / f_labelfree=0.8257 / t4=0.8729` 一起看四级分解。

## 7. 报告要求

每个任务交付：candidate ckpt + SHA、`train_receipt`、`score_receipt`（含 raw chR² 与 DC-centered chR² 两栏）、四级阶梯表（NONE/NORM_ONLY/ACT/FULL 同面对比）、以及 §5 的三个合理性检查结论。receipt 一律 seal（0444 + sha256）。

**若要让零线进官方面**：打包并按各任务既有提交路径准备（`register:false`），提交由用户执行；每任务 1 个名额。
