# HANDOFF：神经-only 载体（N4）与 RT 通用性格子

**日期：2026-08-06**
**给：负责执行的 AI / 研究者**
**状态：实验 A 的 M24 CPU 预检、M33 held-in LOSO Stage 1 和 M24 chronological-disjoint held-out 均已完成；held-out 上 N4−NS4 仅 +0.00159 R²（3/6 session 正），没有有意义的载体特异性证据。实验 B（RT）未启动。**

**前置阅读（数字引用前必读）**

- [`HANDOFF_GENERAL_CARRIER_PROGRAM.md`](HANDOFF_GENERAL_CARRIER_PROGRAM.md) — K4 的原始动机与 v2.1 修订
- [`../carrier_perf_program/docs/PROGRAM_STATUS.md`](../carrier_perf_program/docs/PROGRAM_STATUS.md) — P0N–P6 全程状态
- `../results/m2_k4_component_decomposition_v1/audit.json` — 本 handoff 的直接触发证据
- `../results/m2_m24_disjoint_heldout_v1/aggregate_heldout.json` — 冻结的 M24 held-out 数值
- `../carrier_perf_program/results/p2_rt_gocue_coverage_v1/audit.json` — RT 15 session 资格审计

---

## 一句话

M2 上的载体问题已经收敛到一个新假设：**载体的作用主要是给每个通道一个 session 内稳定的条件化标签，而"这个标签用调谐拟合算出来"这件事贡献很小**。本 handoff 提两个实验去检验它的两个推论——(A) 如果调谐不承重，那么一个**完全不碰行为**的描述子应当接近 T4；(B) 如果载体的价值是通用的，那么它应当在一个 **T4 根本无法定义**的任务上仍然成立。

---

## 二、触发本 handoff 的证据（均已逐条核对）

### 2.1 冻结 M24 held-out（6 session，chronological-disjoint）

来源 `../results/m2_m24_disjoint_heldout_v1/aggregate_heldout.json`（sha256 `1db3477417ffefa358a432404d1223cb4ea25a31667b922d9fd2f254c84ecca7`）

| 臂 | mean R² | 对 F0 |
|---|---:|---:|
| F0 | 0.17390 | — |
| KS4 | 0.20225 | +0.02835 |
| T4 | 0.22680 | +0.05290 |
| K4 | 0.24577 | +0.07187 |

`K4−T4 = +0.01899`（低于预先声明的 `+0.03` 主门，判负）；`K4−KS4 = +0.04352`。

**关键读数：KS4 把通道对应关系彻底打乱之后，仍保住 K4 全部增益的约 40%。** 也就是说载体价值里有一大块根本不是"这个通道是哪个通道"的信息。

### 2.2 内部 LOSO（7 折，2026-08-05 完成）

`p2_general_carrier_m2_k4` / `p2_general_carrier_m2_ks4`，`val_heldin/r2_mean`：

| fold | K4 | KS4 | Δ |
|---:|---:|---:|---:|
| 0 | 0.6303 | 0.5806 | +0.0497 |
| 1 | 0.7458 | 0.6418 | +0.1040 |
| 2 | 0.6323 | 0.4193 | +0.2131 |
| 3 | 0.5271 | 0.4886 | +0.0385 |
| 4 | 0.6228 | 0.5739 | +0.0489 |
| 5 | 0.5563 | 0.4627 | +0.0936 |
| 6 | 0.5436 | 0.3213 | +0.2223 |
| **mean** | **0.6083** | **0.4983** | **+0.1100** |

7/7 全正，精确符号检验 `p=0.0156`。**但内部 K4−KS4 (+0.110) 是 held-out 同一对比 (+0.0435) 的 2.5 倍——任何后续门槛都不得用内部数字标定。**

### 2.3 K4 四维分解（2026-08-06，CPU，7 held-in session，M24）

来源 `../results/m2_k4_component_decomposition_v1/audit.json`。预先声明阈值：residual R² < 0.5 判为被 T4 解释；split-half r ≥ 0.5 判为可靠。

| K4 维 | 平均 residR² | 平均 split-half r | 7 session 判定票数 |
|---|---:|---:|---|
| `baseline_rate` | 0.014 | 0.985 | 冗余 7/7 |
| `w_norm` | 0.225 | 0.861 | 冗余 7/7 |
| `w_x` | 0.539 | 0.592 | 冗余 4 / 新颖但不可靠 2 / 新颖且可靠 1 |
| `w_y` | 0.450 | 0.721 | 冗余 4 / 新颖且可靠 3 |

`baseline_rate` 逐 session 的 T4↔K4 Pearson 为 0.967–0.992。

`defensible_recommendation: "neither"`。receipt 内已记录：OR 聚合给出的 `fusion_worthwhile` 由 1/7 与 3/7 的通过率驱动；多数/全票聚合给出的 `hybrid_worthwhile` 由 4032 次匹配中多认对 7 次驱动（配对差 +0.001736，标准误 0.001909，符号检验 18 正 13 负 11 零，`p=0.473`，`hybrid_margin_within_noise: true`）。**两个正面头条都不成立。**

### 2.4 跨 session 通道识别率（42 个有序 session 对，96 通道，chance = 0.0104）

| 子集 | top-1 |
|---|---:|
| `hybrid_t4w_k4b` | 0.0625 |
| `k4_full` / `t4_full` | 0.0608 |
| `t4_w_only` | 0.0486 |
| `k4_b_only` | 0.0499 |
| `k4_w_only` | 0.0466 |
| `t4_b_only` | 0.0404 |

**所有载体都只有 chance 的 5–6 倍，没有一个能跨 session 认出通道。** 因此载体的机制不是跨 session 身份查表，只能是 session 内的描述子↔数据对齐。这同时解释了 §2.1 里 KS4 为何仍保住 40% 增益（打乱行破坏对齐，但分布正确的逐通道标量本身仍有用）。

### 2.5 P4 谐波分层（27 session，464 电极）

来源 `../carrier_perf_program/results/p4_harmonic_dispersion_strata_v1/audit.json`。
`monotonic_gain_vs_dispersion: false`，`transduction_consistent_harmonic_over_fund: false`，PV R² 与 rate MSE 的 fund−harm 分别为 `−0.00274` 与 `−0.07543`（方向相反）。**把调谐模型做得更丰富（加二次谐波）没有一致收益。**

### 2.6 合并推论

§2.3 说唯一高可靠维（`baseline_rate`，r≈0.985）恰恰是唯一**不含调谐信息**的那一维；§2.5 说丰富调谐模型无收益；§2.4 说没有跨 session 身份信息；§2.1 说 40% 的增益在打乱后仍存活。四条独立证据同向。

**因此提出待检验假设 H_N：一个完全不使用行为量计算的逐通道描述子，可以达到与 T4 相当的解码增益。**

---

## 三、实验 A：神经-only 载体 N4（M2）

### 3.1 目的

检验 H_N。注意这**不是** F0 已经回答过的问题：

| 臂 | 侧特征维度 | 计算是否使用行为 | 通道对应关系 |
|---|---:|---|---|
| F0 | 0 | — | — |
| Zero4 | 4（全零） | 否 | 无信息 |
| T4 / K4 | 4 | **是** | 正确 |
| TS4 / KS4 | 4 | 是 | **打乱** |
| **N4（本实验）** | **4** | **否** | **正确** |
| **NS4（本实验对照）** | **4** | **否** | **打乱** |

F0 回答"有没有逐通道标签值多少分"，回答不了"这个标签是否必须用行为算"。KS4 也不行——它的数值仍由行为算得，只是错位。**表中最后两行是当前证据网格里唯一的空缺。**

### 3.2 N4 的定义（执行前必须冻结，不得中途改）

沿用 K4 的 raw block 构造以保证可比：`K4_RAW_BIN_MS=20`，`K4_BLOCK_WIDTH_BINS=5`（100 ms），block 不跨 trial 边界，chronological 前缀 24 trial。

**纯度要求（最容易出错的一条）：必须去掉 K4 的速度激活过滤 `~all(abs(v) < 1e-3)`。那条规则读取行为量，用了它 N4 就不再是 behavior-free。** N4 使用标定 trial 内的全部 block。这会让 N4 的支撑集比 K4 略大，属预期，须在 receipt 中显式记录两者的 block 数。

对每个通道 i，在其 block rate 序列 `r_i[k]`（Hz）上计算：

| 维 | 定义 | 动机 |
|---|---|---|
| 1 `mean_rate` | `mean_k r_i[k]` | `baseline_rate` 的 behavior-free 对应物；§2.3 预测它承载主要价值 |
| 2 `fano` | `var_k(count_i[k]) / mean_k(count_i[k])`，原始计数非速率；分母 ≤ 1e-6 时按退化处理 | 离散度，与均值正交的二阶统计量 |
| 3 `lag1_autocorr` | `r_i[k]` 与 `r_i[k+1]` 的 Pearson，仅在 trial 内配对 | 时间尺度代理 |
| 4 `population_coupling` | `r_i[k]` 与去掉自身后的群体平均 `mean_{j≠i} r_j[k]` 的 Pearson | Okun et al. 2015 的 population coupling，已知的稳定单神经元属性 |

维度锁定为 4，以保住 `side_dim=4` 契约（B3S post_pool 首层仿射输入宽度 `hidden_dim(64)+4=68`；改宽会破坏 strict 加载，见 `../results/m2_m24_k4_t4_residual_stop_v1/stop_design_receipt.json`）。

**退化处理必须 fail-closed**：零脉冲通道、Fano 分母下溢、autocorr 样本不足，一律走 `sua_exploration/mc_maze/unit_side_features.py` 中 `enforce_direction_degeneracy_policy` 同款的显式策略（默认 `raise`），禁止静默填零。

归一化沿用 `fit_train_k4_stats` 的同款逐列 train-only z-score。NS4 沿用 `deterministic_k4_row_permutation` 的同款确定性整行置换，seed 与 session 名命名空间独立于 KS4/TS4。

### 3.3 阶段 0：CPU 预检（**GPU 前强制**，约十分钟）

在 7 个 held-in session 上算出 N4，复用 `carrier_perf_program/src/carrier_perf/k4_component_decomposition.py` 的同款度量：

1. 每维的 12/12 split-half 跨通道 Pearson
2. 每维对 T4 四维的 residual R²
3. `mean_rate` 与 T4 `baseline_rate` 的逐 session Pearson
4. 跨 session 通道识别 top-1（与 §2.4 同协议，同 chance 基线）

**放行条件（预先声明）**：至少 2 维的 split-half r ≥ 0.5（多数规则，≥4/7 session）。**聚合规则在此处显式声明为多数规则**——上一轮的教训是 OR 聚合会制造假阳性头条，不得重蹈。

不满足则停止，不进 GPU，把 CPU receipt 封存并报告"N4 描述子在 M24 预算下不可估"。

### 3.4 阶段 1：内部 LOSO 门

在 M2 内部 LOSO（与 §2.2 同协议、同 7 折）上跑 `N4` 与 `NS4` 两臂。已有的 `F0` / `T4` / `K4` / `KS4` 内部数值可直接复用，不得重跑后择优。

**放行条件（预先声明）**：`N4 − NS4 > 0` 且 ≥5/7 折为正。这只证明"描述子里有通道特异信息"，**不构成任何 held-out 结论**。

#### 3.4.1 2026-08-06 完成态审计

权威本地 receipt：`../results/n4_m2_internal_loso_v1/audit.json`（SHA-256
`3869867a305b690efc4badacbb3c8bb085b7f96d7badbbb02257f15dbeba95c9`）。正式集合为原始顺序
launcher 产生的 `N4/NS4 × folds 0..6 × seed42` 共 14 cells；一个 N4 1-epoch smoke
和 11:40 左右两条独立的 **M24 held-out** fold-0 run 已从本 M33 internal aggregate 排除（它们不是重复 run，见 §3.5）。所有正式 cells 都有 finite test metric；
N4 均到 epoch 11，NS4 fold5 在共同 EarlyStopping 规则下止于 epoch 10，因此同时报告
两种既有指标口径，禁止结果后选择较好者。

| 口径 | N4 mean | NS4 mean | mean Δ | 正 fold | 2SE 区间 | 判读 |
|---|---:|---:|---:|---:|---:|---|
| last available `val_heldin/r2_mean`（主） | 0.533608 | 0.524008 | **+0.009600** | **6/7** | `[-0.027319,+0.046518]` | 预声明 Stage-1 门机械通过 |
| selected-checkpoint `test_heldin/r2_mean`（敏感性） | 0.542713 | 0.532709 | **+0.010004** | **5/7** | `[-0.019092,+0.039100]` | 同样机械通过 |

主口径的 fold delta 为
`[+0.01695,+0.07493,+0.02963,+0.01922,+0.00081,+0.01304,-0.08738]`；
session bootstrap 95% 区间为 `[-0.02794,+0.03959]`，leave-one-fold-out mean 可低至
`-0.00129`。所以正确结论是：**内部 attachment screen 通过，但平均效应仅约
`+0.01 R²`、区间跨零且有一个明显反向 fold，是 weak/indeterminate efficacy
signal。**它不能证明 held-out 有效、不能证明 N4 接近 T4，也不能支持“标签不重要”。

Stage 1 当时只构成启动 held-out 的弱筛选信号；随后完成的 Stage 2 见下一节。退化政策、
Zero4 和 source-lineage 问题仍限制跨臂机制归因，但不改变 matched N4−NS4 已经接近零的事实。

### 3.5 阶段 2：冻结 M24 held-out（仅在阶段 1 通过后）

同 §2.1 的 chronological-disjoint 6 session 协议，同 M24 预算，同随机种子规程。新增 `N4`、`NS4` 两臂；`F0`/`T4`/`K4`/`KS4` 一律引用已冻结数值，**禁止重跑**。

**预先声明的三道门**：

- **G1 有效性**：`N4 − F0 ≥ +0.03`
- **G2 特异性**：`N4 − NS4 > 0`
- **G3 非劣（本实验的真问题）**：`N4 − T4 ≥ −0.03`

#### 3.5.1 2026-08-06 完成态审计

权威 receipt：`../results/n4_m2_m24_heldout_v1/audit.json`（SHA-256
`7a759f5e4c447d960ff0767f2c111df5c899c99c8bd63d9cc4d3d89203e38023`）。两臂均为 fold 0、seed 42，
使用同一 6 个 held-out session、chronological support `[0,24)`、`query_start=24`，所有
query window 均通过 full-history disjoint 审计。

| 臂 | held-out mean R² | 与另一臂之差 |
|---|---:|---:|
| N4 | 0.175550 | +0.001588 vs NS4 |
| NS4 | 0.173962 | — |

逐 session 的 `N4−NS4` 为
`[-0.05761,+0.00531,-0.00760,-0.00690,+0.05079,+0.02553]`：仅 **3/6** 为正，
paired SE `0.01491`，`mean ± 2SE=[-0.02823,+0.03141]`。所以 G2 虽按“均值大于零”这个
过弱的标量规则机械过线，但实际效应约为零且没有跨 session 一致性。**正确结论是：N4
在 local held-out 上没有显示正确 channel attachment 相对 row shuffle 的有意义收益。**

旧冻结 F0/T4/K4/KS4 receipt 是 fold 1，而本次 N4/NS4 是 fold 0；因此虽然 session、M24、
query window 与 seed 相同，source train/validation split 和 checkpoint 不同。仅作非匹配背景，
`N4−F0=+0.00165`、`N4−T4=-0.05124`；二者不得包装成 matched G1/G3 机制检验。
但 G2 已经近零，足以否定当前四维 N4 作为 T4 替代物的主要假设，不值得为它补跑更多 seed。

**关于 `+0.03` 这个数字的重要说明（勿误用）**：它是 M24 K4 cell **当时为该 cell 预先声明的效应门**，**不是**本程序已冻结的全局效应下限。`carrier_perf/protocol.py` 中的 `PRACTICAL_EFFECT_FLOOR` **故意保持为 `None`**，并有单测 `test_no_prewritten_effect_floor` 守着；P0N 明确要求"在 `fold2_seed43` 存在且有经审阅的 receipt 冻结之前，不得写入 `PRACTICAL_EFFECT_FLOOR` 或 GPU 阈值"，而 `fold2_seed43` 至今缺失（见 `../carrier_perf_program/docs/PROGRAM_STATUS.md` P0N 行）。

因此执行者**必须**：把 `+0.03` 作为本 cell 的、写在本 handoff 里的、逐 cell 预先声明门来使用，并在 receipt 中注明其来源是"沿用 M24 K4 cell 的同名门以保持跨 cell 可比"；**不得**把它写进 `protocol.py` 的 `PRACTICAL_EFFECT_FLOOR`（会破坏该守卫单测），**也不得**声称它是噪声底导出的。若 `fold2_seed43` 在此期间被补齐并冻结出正式效应下限，以正式值为准并在 receipt 中记录替换。

### 3.6 结论矩阵（执行前写死，看到数字后不得改写）

| G1 | G3 | 结论 | 论文含义 |
|---|---|---|---|
| 过 | 过 | **行为量对载体的计算不是必需的** | 适用范围从"有二维目标标签的任务"扩张到"任何多通道记录"，包括言语、手写、高自由度。这是数量级推广，也彻底绕开余弦调谐的二维假设 |
| 过 | 不过 | 载体可 behavior-free 构造，但调谐信息确有额外贡献 | 现有叙事成立且被加强，同时给出一个强对照 |
| 不过 | — | 行为量是必需的，调谐承重 | §2.6 的假设被推翻，`baseline_rate` 的高可靠性另有解释；现有叙事完全站得住 |

**三种结果都可发表。** 这是本实验优先级高于任何 K4 后续改法的原因。

---

## 四、实验 B：K4 在 RT 上（sub-C，DANDI 000688）

### 4.1 目的与它为何比 M2 更有价值

M2 是 T4 的主场，"K4 必须打赢 T4"是最苛刻的一条线，且已判负（§2.1）。RT 上的问题完全不同。

来源 `../carrier_perf_program/results/p2_rt_gocue_coverage_v1/audit.json`：

- `n_sessions: 15`，`n_eligible: 15`，`n_ineligible: 0`
- **`all_sessions_have_single_target_dir: true`**，**`t4_undefined_single_target_dir: true`**
- `min_finite_gocues: 2`，逐 trial finite go-cue 数均值 3.23，最大 4

**即：15 个 RT session 的 `target_dir` 全部是 rank-1，T4 在 RT 上根本无法定义。** 这不是"T4 表现差"，是"T4 不存在"。因此 RT 的主张是**类别性的而非增量性的**——K4 只需要在一个 T4 无法进入的格子里成立，不需要打赢任何东西。

附带的结构性论据：RT 的 trial 内含多个顺序随机目标（`go_cue_time_array`、`num_targets`、`num_attempted`）。T4 若要勉强工作必须先用 go-cue 把 trial 切成子运动段；**K4 不需要 go-cue**，它只按速度激活规则取 100 ms block。

### 4.2 协变量

`processing/behavior` 内有 `Position` / `Velocity` / `Acceleration`。**用二维手部速度 `Velocity`**，`k4_from_raw_calibration` 的 `covariates.shape[1] == 2` 契约天然满足，**K4 估计器一行都不用改**。

### 4.3 网格与协议

臂：`F0`、`K4`、`KS4`。**没有 T4 臂**，理由见 §4.1，须在 manifest 与 receipt 中显式写明"T4 因 rank-1 `target_dir` 不可定义"，不得留空或静默省略。

必须先冻结并在 receipt 中记录的协议参数（本 handoff 不替执行者决定，但要求执行前写死）：

1. **信号类型**：sorted SUA 还是 pseudo-MUA。sub-C 两者都有历史臂，必须二选一并说明，不得两者都跑后择优。
2. **解码目标**：二维 cursor velocity（与 sub-C CO 既有臂一致）。
3. **划分**：15 个 session 跨 2013-10 至 2015-03。建议先内部 LOSO，通过后再做 chronological-disjoint held-out；两者的 session 名单在执行前冻结。
4. **支撑预算**：M24，与 M2 对齐以便跨格子讨论。

**fail-closed manifest 政策（P2 审计已声明，必须遵守）**："RT k4 cells may only load eligible_sessions; ineligible_sessions must appear explicitly in any manifest exclusion list — silent drop forbidden."

### 4.4 预先声明的门

- **G1 有效性**：`K4 − F0 ≥ +0.03`
- **G2 特异性**：`K4 − KS4 > 0`，且多数 session/折为正

**明确不设 `K4 − T4` 门**（不可计算）。

`+0.03` 的性质与使用限制同 §3.5 末尾的说明——逐 cell 预先声明门，不是全局效应下限，不得写入 `PRACTICAL_EFFECT_FLOOR`。**额外注意**：RT 与 M2 是不同任务、不同动物、不同信号，绝对 R² 尺度不可比，因此 `+0.03` 在 RT 上是一个**借用**的门槛而非导出的门槛。执行者若认为 RT 的噪声尺度明显不同，**必须在跑任何 RT 臂之前**改写本节并说明理由，事后调整一律作废。

### 4.5 结论矩阵

| G1 | G2 | 结论 |
|---|---|---|
| 过 | 过 | **载体在 T4 无法定义的任务上成立，且信息是通道特异的。** K4 的存在理由被证实，与 M2 上没打赢 T4 并不矛盾 |
| 过 | 不过 | 载体有用但非通道特异；与 §2.1 中 KS4 保住 40% 增益的现象一致，须并入同一机制讨论 |
| 不过 | — | 载体价值不跨任务迁移；这是对整个载体程序外部效度的实质性限制，必须如实写入论文 |

---

## 五、纪律（两个实验共同适用，违反即作废）

1. **所有阈值与聚合规则在看到任何结果之前写死在代码常量里并进 receipt。** 上一轮的具体教训：per-dim 阈值预先声明了，跨 session 聚合规则没有，结果 OR 聚合制造出一个经不起推敲的 `fusion_worthwhile` 头条。**本 handoff 中凡涉及多 session 汇总，一律预先声明为多数规则（≥4/7 或 ≥5/7，见各节）。**
2. **任何比较在宣称为正之前必须过噪声下限。** 报配对差、其标准误、以及换算成绝对计数的量级。上一轮 `hybrid_worthwhile` 就是栽在 4032 次里多 7 次上。有序 session 对不独立，符号检验只能作描述性使用并须注明。
3. **内部 LOSO 数值不得用于标定 held-out 门槛。** 实测倍率约 2.5×（§2.2）。
4. **held-out 只允许在预先声明的内部门通过之后开一次。** 禁止 held-out 上的多次读取、择优、或事后调参。
5. **已冻结的臂一律引用，不得重跑。** F0/T4/K4/KS4 的 M24 held-out 数值见 §2.1。
6. **退化情形一律 fail-closed。** 禁止静默填零、静默丢 session。
7. **receipt 必须含每个输入文件的 SHA256**，格式对齐 `../results/` 下既有 receipt。
8. **不得触碰** `streaming_calibration_exp/outputs/`、`logs/`，以及任何正在运行的 native-M2 重训作业。

### 禁止的降级做法

- 把 N4 的维度从 4 改成别的宽度来"让它工作"（破坏 side_dim=4 契约与与既有臂的可比性）
- N4 使用任何行为量，包括 K4 的速度激活过滤规则
- 在 RT 上强行构造一个 T4 替身然后声称"K4 打赢了 T4"
- 事后更换聚合规则、阈值、或 session 名单
- 两种协议都跑完再报好的那个（信号类型、划分方式必须事前二选一）

---

## 六、执行顺序与成本

| 步骤 | 资源 | 前置 |
|---|---|---|
| A-阶段0 N4 CPU 预检 | CPU，约十分钟 | 无 |
| B-协议冻结（信号类型/划分/预算） | 纸面 | 无，可与 A-阶段0 并行 |
| A-阶段1 内部 LOSO（N4/NS4） | GPU，2 臂 × 7 折 | A-阶段0 通过 |
| B-内部 LOSO（F0/K4/KS4） | GPU，3 臂 × 折数 | B-协议冻结 |
| A-阶段2 held-out（N4/NS4） | GPU，2 臂 | A-阶段1 通过 |
| B-held-out（F0/K4/KS4） | GPU，3 臂 | B-内部门通过 |

A 与 B 相互独立，可并行。**A-阶段0 是唯一的 CPU-only 步骤，且能在花任何 GPU 之前砍掉一整条线**——上一轮的 K4 分解就是用十几分钟 CPU 排除了三套各三臂 × 7 折的 GPU 方案，同样的杠杆在这里再用一次。

---

## 七、本 handoff 不检验的事情

- 不检验融合载体（T4+K4 各形态）。`../results/m2_k4_component_decomposition_v1/audit.json` 的 `defensible_recommendation: "neither"` 已覆盖该问题，重开需要新的预注册假设。
- 不检验 M1。M1（DANDI 000941, Monkey L）的 NWB `acquisition` 只有 `eval_mask` 与 `preprocessed_emg`（16 块具名肌肉），**数据集内无任何运动学**，K4 的二维速度前提无对应物，需要单独设计 16 维 EMG 的降维方案与其预注册，本 handoff 明确不涉及。
- 不检验 P3 载体腐蚀增强。注意历史遗留问题：`corruption_config` 曾出现在实验 YAML 中但不在 `FalconDataModule.__init__` 签名里，因此那次运行实际未施加腐蚀。重开 P3 前必须先修数据模块并重跑。
- 不检验二次谐波。P4 已判负（§2.5）。
