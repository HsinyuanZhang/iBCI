# T4G 路线：Generalized Analytic Functional Carrier（通用解析功能载体）

**日期：2026-08-06**
**状态：Phase 0 的 circular primitive/合约测试与 Phase 1 native-M2 精确约化已通过；RT continuous-velocity AFC4 已在冻结矩阵中取得正向 development evidence 并按既定矩阵收口；H1 q3-AFC4 fold-0 clean nested-LOSO pilot 已判负并停止；M1 已移出主线。本文不是新 GPU arm、formal held-out 或 EvalAI 的启动授权。**
**范围：**把现有 T4 放入一个更一般、但仍然低计算量和 target-session 无反向传播的载体家族。本文不修改、不中止、也不重新解释正在运行的 N4/NS4 作业。

### 2026-08-07 H1 q3-AFC4 clean nested-LOSO fold-0 pilot：冻结门判负，路线不扩张

作为独立的人体 H1 feasibility cell，按本文的 q3 source-PCA、四维 affine carrier 和
target 无反向传播契约完成了一个严格 clean nested-LOSO outer fold。outer target
`ses-19250101T111740` 未在 fit 或 checkpoint selection 期间打开；inner-validation 为
`ses-19250101T112404`，source-only plan SHA
`b335c688c2854714fdc23deedfb2ddbe19d6116901c85da48410fdfeb6bc6168`。teacher、Full
(`afc4_h1q3`)、B4 和 Zero4 都在 inner validation 选择 checkpoint 后只做一次 outer
held-in minival scoring（`1,466` windows；无 optimizer/backward，model state SHA 不变）。

| fold-0 arm | outer variance-weighted R² |
|---|---:|
| source-only teacher | `-0.0403340407` |
| Full q3-AFC4 | `-0.0219134353` |
| B4 | `+0.0064380431` |
| Zero4 | `-0.0324103446` |

预注册 gate 为 Full absolute `>0`、Full−teacher/Full−B4/Full−Zero4 各至少 `+0.03`。
实际差值为 `+0.0184206054`、`-0.0283514784`、`+0.0104969093`，所以四项都失败，
aggregate 状态为 `STOP_FOLD0_GATE_FAILED`。权威回执为
`../results/h1_afc4_clean_nested_loso_v1/fold0/gate_aggregate.json`
（SHA-256 `709716b38d669e3210f1a10a444c307617b596c0d850c95b16055eb1a306e47b`）。按事前
协议不再运行 RS/XS、不扩展 13-fold，也不调 q、support、lag、ridge、epoch 或 width；
单 fold 的 B4−Zero4 `+0.038848` 只作 exploratory contrast，不能被解释成 B4 路线放行。
该 cell 只判定这一冻结 H1 pilot 未达 efficacy 门，不否定其他任务或未来另行注册的 AFC4
estimator。

### 2026-08-06 实施进展

已新增独立实现 `streaming_calibration_exp/src/data/afc4_features.py`，其 API 只接收 calibration response sums、exposure 与 source-declared 二维 task basis，不存在 query 输入；实现了截距不罚的闭式 OLS/ridge、circular specialisation、source-only normalizer、确定性 descriptor-row shuffle 和 label permutation。合约测试
`streaming_calibration_exp/tests/test_afc4_features.py` 共 `5/5` 通过，包括合成系数回收、rank/shape fail-closed、shuffle 确定性，以及 circular AFC4 与 T4 的精确约化。

Phase 1 receipt 为 `../results/t4g_m2_circular_equivalence_v1/audit.json`。审计采用 native-M2 fold 1 / seed 42 的真实 held-in 数据和各 budget 的真实冻结 B3S checkpoint，分别检查 `M=24` 与 `M=33`；每个 budget 均包含 6 个 source-train session 和 1 个 LOSO source-validation session，未构造 formal held-out dataset，也未读取 query label/covariate。结果如下：

| 层级 | M=24 最大绝对误差 | M=33 最大绝对误差 |
|---|---:|---:|
| pre-normalization descriptor | `0.0` | `0.0` |
| source-only mean / std | `0.0 / 0.0` | `0.0 / 0.0` |
| normalized descriptor | `0.0` | `0.0` |
| frozen row-shuffled descriptor | `0.0` | `0.0` |
| real-checkpoint B3S encoder output（aligned / shuffled） | `0.0 / 0.0` | `0.0 / 0.0` |

M24 每个 session 的 24-trial prefix 中有 12 个 finite directional trial、12 个 centre/rest unlabeled trial；M33 中为 16 个 finite directional trial、17 个 centre/rest unlabeled trial。所有 14 个 budget×session design 均为 rank 3，condition number 为 `1.584–2.643`。因此现在可以作出的结论是：**当 basis 为 `[cosθ,sinθ]`、`λ=0`，并锁定相同 exposure、valid mask、source normalizer 与 B3S checkpoint 时，`afc4_circ` 在实现层面就是 native T4 的精确特例。**这只通过了实现正确性 gate；它本身不替代上方独立的 RT efficacy receipts，也不把已经停止的 H1 pilot 变成正结果。

## 1. 结论先行

T4 的有价值部分不是 `sin/cos` 这两个数本身，而是一个部署契约：在新 session 的小段 calibration 中，以因果可得的任务条件为自变量，为每个通道解析地估计一个功能系数，再把该系数压缩到已有的四维 side-feature 接口。`sin/cos` 只是任务条件为圆周方向时的一个特例。

本文将该家族称为 **Analytic Functional Carrier, AFC4**；“T4G 路线”只作为讨论路线的总称，**不建议把实现臂命名为 `T4G`**。仓库已有已判负的 `T4GATE`，而 `E4`、`D4`、`K4` 已有固定含义，复用这些名字会造成结果和机制的混淆。建议的实现臂名为：

| 名称 | 含义 | 适用任务 |
|---|---|---|
| `afc4_circ` | 圆周方向 special case；应约化为 T4 | M2 / CO |
| `afc4_vel` | 连续二维速度的解析编码载体 | RT |
| `afc4_evt` | event/phase-aligned continuous basis 的解析载体 | RT；仅在 `afc4_vel` 后 |
| `afc4_emg` | source-frozen EMG 低秩基上的解析载体 | M1 |
| `afc4_rs` | 完整 descriptor-row shuffle（attachment null） | 所有 AFC4 臂 |
| `afc4_ls` | calibration label shuffle 后重新拟合（label–response association null） | 所有 AFC4 臂 |

`AFC4` 表示 **最终接入 decoder 的宽度为 4**，而不表示所有任务的原始任务基都必须是二维；AFC4 的通用性主张是“算法、接口、部署约束通用”，不是“方向、速度和 EMG 有相同的生理坐标语义”。

## 2. 经核验的起点，以及不能越过的证据边界

> **N4 状态边界（2026-08-06）：**M33 held-in LOSO Stage 1 已完成。N4 相对 NS4 在
> last-validation / selected-checkpoint 两个口径分别为 `+0.009600`（6/7 folds 正）和
> `+0.010004`（5/7 正），所以按预声明的纯符号门机械通过；但两种不确定区间都跨零，且
> leave-one-fold-out mean 可为负。随后完成的 M24 chronological-disjoint local held-out
> 上，`N4=0.175550`、`NS4=0.173962`，即 `N4−NS4=+0.001588` 且仅 3/6 session 为正，
> 没有有意义的 held-out attachment evidence。N4/NS4 使用 fold 0，而旧冻结 F0/T4 使用
> fold 1，所以跨谱系差值只能作背景、不能作 matched attribution；但 matched N4−NS4
> 已足以停止当前 N4 替代物。权威 receipts 为
> `../results/n4_m2_internal_loso_v1/audit.json` 与
> `../results/n4_m2_m24_heldout_v1/audit.json`。

### 2.1 已核验的事实

1. Native M2 的严格 M24 chronological-disjoint local held-out receipt报告 `F0=0.17390`、`T4=0.22680`、`K4=0.24577`、`KS4=0.20225`。因此 `T4-F0=+0.05290`，`K4-T4=+0.01899`，`K4-KS4=+0.04352`。K4 相对 T4 没有达到该 K4 cell 事前写下的 `+0.03` operational 门；该 `+0.03` **不是项目级噪声底**，也不可被本路线重新包装成全局 SESOI。
2. native T4 的实际估计式是每 trial rate 上的连续角度一阶谐波最小二乘；其输出为 `[a,c,sqrt(a^2+c^2),b]`，并对 rank-deficient design fail-closed。M2 的 `theta` 是相对于 `(0.5,0.5)` 的 target polar angle；M1 的 `theta` 是 `tgt_loc` 中的方位角。这和 SUA 的“8 canonical direction、逐方向均值”实现不是同一个 estimator，跨 setting 绝不可把二者都笼统称为同一个 T4 数值。
3. RT 的 trial-level `target_dir` 在 15/15 sub-C RT session 中为 rank-1；普通 T4 在该任务是**未定义**，不是一个可参与 `T4` vs `AFC4` 胜负比较的弱基线。RT 具有连续 `Position/Velocity/Acceleration`，且一个 trial 含多次 reach；整 trial 平均 velocity 已被既有审计证明会抵消运动方向，不能作为 AFC4 条件变量。
4. M1 的输出是 16-D EMG，而现有 T4 用目标方位角。M1 的旧 support/query overlap 结论已被撤回；仓库已有的可构造 local development endpoint 是 support `[0,10)`、checkpoint selection `[10,210)`、sealed report `[210,end)`。已完成的 D4、学习式 E4 与 temporal-prototype 分支各有自己的停机结论，不能把 AFC4-EMG 偷换成这些已停止分支的 rescue。
5. 现有 M2 N4 CPU precheck 通过 split-half gate，M33 held-in LOSO Stage 1 按其预声明符号门机械通过但效应弱；真正的 M24 local held-out 上 `N4−NS4=+0.001588`、3/6 session 正，当前 N4 的正确 attachment 没有可辨识收益。脚本和 production datamodule 实际传入 `degeneracy_policy="fill_median"`，而相关 handoff 写的是 fail-closed/`raise`；这进一步限制严格 deployment claim，但不是 held-out 近零效应的解释性补救。该结果排除的是当前四维 N4，不证明所有无标签载体都无效，也不单独证明 T4 标签机制。

### 2.2 如果未来观察到“N4 显著不如 T4”，什么时候才支持标签—响应关联重要？

只有在下列条件全部匹配时，`N4 < T4` 才支持这个较窄的结论：

> 在该 carrier family、该 calibration budget 和该 decoder consumer 下，目标 session 中正确配对的任务标签—神经响应关联，提供了目前这组无标签神经统计量无法恢复的功能信息。

必需的匹配条件为：

| 必须匹配/锁定的条件 | 原因 |
|---|---|
| 同一 task、相同 chronological support、相同 query 及完整 history-disjoint audit | 否则差异可能来自时间段或数据域，而非标签。 |
| 同一 `side_dim=4`、同一 B3S/decoder 架构、相同 source checkpoint/seed 规则 | 否则是容量或 checkpoint 选择差异。 |
| 对 N4 和 T4 都有 matched `Zero4`，并报告 `T4-Zero4`、`N4-Zero4` | F0 不控制四维 side path；宽度/零初始化本身必须分开。 |
| 对两者均做完整-row shuffle | 只有这样才可判断“数值分布”与“正确通道 attachment”。 |
| T4 做 **label shuffle → refit**，保留 neural rates、trial/exposure、label multiset 和 normalizer；N4/NS4 不能替代该 null | row shuffle 测 attachment；label shuffle 测 label–response association；两者不是同一个问题。 |
| 归一化只在各 LOSO source-train session 上拟合；feature/label 处理完全预注册 | target 统计量或选择会把 session 信息泄漏到 carrier。 |
| biological session 是推断单位；seed 是训练重复，不得把 session×seed 当独立样本 | 否则小样本下会虚增精度。 |

即使上表全过，结论仍然**不能**证明：

- “标签在一般意义上是必需的”；它只排除了当前 N4 的四个 marginal statistics，不排除另一种无标签或自监督的功能估计器。
- “T4 的 `sin/cos` 形式在任何任务都最优”；M2 的圆周方向只是恰好适合它的条件变量。
- “标签量必须很大”；二维线性模型的关键可能是任务基覆盖/条件数，而不是 trial 的绝对个数。
- “T4 比 K4 的 carrier form 更好/更差”。K4 使用 dense per-bin velocity，T4 只用 per-trial direction；这是 label-information 和时间分辨率均不等的 **operational comparison**。
- “有跨 session neuron identity lookup”。既有 descriptor 的跨 session channel matching 很低；AFC4 只声称 session 内正确的 row attachment 对 source-trained consumer 有用。

## 3. 统一定义：从任务条件到四维 AFC4

令 target-session calibration 中可用的神经计数/率为 `r_i(t)`，任务条件或标签为 `y(t)`；令 `S` 是严格早于 query 的 calibration support。固定任务基为

\[
z(t)=\phi_{\mathrm{task}}(y(t))\in\mathbb{R}^{q}.
\]

对每个通道 `i`，在 support 内以闭式、无反向传播的 encoding fit 估计：

\[
\hat r_i(t)=b_i+w_i^\top z(t)+\epsilon_i(t),
\]

\[
(\hat b_i,\hat w_i)=\arg\min_{b,w}\sum_{t\in S}\omega_t\,[r_i(t)-b-w^\top z(t)]^2+\lambda\lVert w\rVert_2^2.
\]

截距不罚；`\omega_t`、binning、exposure correction、lag、`\lambda`、link family 和任何 condition balancing 都是 estimator 的一部分，必须在 source-only 阶段锁定。若统计模型需用 Poisson/negative-binomial GLM，可用固定链接函数与固定收敛准则替换 squared loss；但它是一个**单独预先命名的 estimator**，不得在目标 session 或 held-out 上看到 OLS 表现后再切换。

最终供 B3S 使用的是固定宽度：

\[
e_i = C_{\mathrm{task}}(\hat w_i,\hat b_i)\in\mathbb{R}^{4}.
\]

目前可审计的压缩规则为：

| `q` | 固定四维 carrier | 备注 |
|---:|---|---|
| 2 | `[w_{i1}, w_{i2}, ||w_i||_2, b_i]` | T4/二维 velocity 的直接形式。 |
| 3 | `[w_{i1}, w_{i2}, w_{i3}, b_i]` | 适合 source-frozen EMG 3-D basis。 |
| `q>3` | `[w_i^T U_{source}]_1^3, b_i`，其中 `U_source∈R^{q×3}` | `U_source` 只能在 source session 上估计、版本化并冻结；target session 不能学习/旋转它。 |

T4 正是 `q=2` 的 circular specialization：

\[
\phi_{\mathrm{circ}}(\theta)=[\cos\theta,\sin\theta]^\top,
\qquad
e_i=[a_i,c_i,\sqrt{a_i^2+c_i^2},b_i].
\]

为做真正的 M2 reduction/equivalence，`afc4_circ` 必须使用与 native T4 相同的 trial sums、exposure/rate 定义、可用方向掩码、OLS (`λ=0`) 及归一化。只要引入 ridge、不同 time bin 或不同样本权重，就不再是“代码重述 T4”，而是一个待检验的新 estimator；不可把接近的数值称为 mathematical equivalence。

### Source-supervised training 与 target calibration 的分界

| 阶段 | 允许做什么 | 严格禁止 |
|---|---|---|
| Source stage | 用 source session 训练 B3S 如何消费四维接口；选择/冻结 `φ` 的结构、`U_source`、normalizer、ridge/GLM 超参数、lag 与 session split；可使用 source labels。 | 用 target calibration/query、official/future query 标签调任何超参数；按 target 结果换 basis。 |
| Target calibration | 仅用该 session 的合法 chronological support，累积足够统计量，做 per-channel closed-form ridge/固定 GLM，发出 `N×4` carrier。 | decoder/backbone/normalizer 的 target-session BP、early stopping、target basis learning、读取 query labels。 |
| Streaming query | 冻结 carrier、decoder 和所有参数后解码。 | query 内重新拟合 carrier，或以 query 误差选择 checkpoint/lag/λ。 |

该分界使 AFC4 可以有 source-supervised 表示学习，但其 target deployment 仍保持 **no target BP**；不得把“target 无 BP”误说成“全流程没有学习”。

## 4. 公共实验设计：两个比较口径与必须的对照

### 4.1 必须报告的两种比较

1. **Equal-label mechanism comparison：**完全相同的 calibration time bins/trials、同一 exposure、同一 label granularity、同一 `φ` 和同一 carrier fit，只改变 alignment 或 label–response association。这一组才能支持机制归因。
2. **Operational comparison：**比较不同可部署方案的总效用与真实成本，例如 T4（trial target labels）、AFC4-velocity（dense kinematics）和 per-session direct ridge。它们必须逐项报告：labeled trials、labeled bins/秒、label dimensions、是否需实时 kinematics、calibration compute/state。它们不能被改写为“只因 carrier 形式而赢”。

### 4.2 每个 AFC4 实验的最小矩阵

| 臂 | 作用 | 口径 |
|---|---|---|
| `F0` | 无 side path的历史/架构基线；只作辅助参照 | 非 width-matched。 |
| `Zero4` | 同侧路径、四维全零 | 所有 AFC4 efficacy 主比较都必需。 |
| `AFC4` | 正确 channel-attached解析载体 | 主臂。 |
| `afc4_rs` | 对拟合后的完整 `[w,b]` descriptor row 进行确定性非 identity 置换 | 测 channel attachment；保留 feature distribution。 |
| `afc4_ls` | calibration label 在预先指定的合法 exchange block 内置换后重新拟合 | 测 label–response association；保留 neural data、exposure、label multiset。 |
| `N4` / `NS4` | 行为无关的四维统计量及 row-shuffle | 若同一数据接口可定义，则是“当前无标签替代物”的负/正对照；不是 `afc4_ls` 的替代。 |
| `DirectRidge` | target support 上从原始 neural window 到行为输出的 session-specific ridge/Wiener readout | 高监督、非共享 decoder 的 operational reference；不是 AFC4 mechanism baseline。 |

`afc4_ls` 的置换单位取决于标签时间尺度：trial-level 标签必须以完整 trial 为单位、只在预先定义的 calibration-pool exchange blocks 之间置换；连续 kinematics 必须在不跨 calibration/query 边界、且保持 segment/exposure 与主要时间自相关结构的 blocks 内置换。不能打乱 feature rows 来代替 label shuffle，也不能在 event-aligned 数据上把不同 event phase 混成一个无物理含义的 null。

## 5. 三阶段执行计划

### Phase 0 — 共同的 source-only 实现与合规预检（先于任何新 GPU）

**目标：**使后续“通用”不只是命名，而是可验证的同一部署接口。

1. 在独立新模块实现 `afc4` primitives：time/segment pairing、intercept-unpenalized closed-form ridge、source-only `U_source`、row/label shuffle、normalizer 与完整 receipt。现有 `mc_maze/general_carrier.py` 已有一部分 movement-aligned Gate-A primitive，但明确“不实现 network side feature”；不可把它的 source proxy 直接当作 decoder evidence。
2. 为所有 carrier 写合约测试：shape 为 `[channels,4]`、同 seed non-identity row permutation、no query input、design/basis rank、degenerate channel policy、source-only normalizer、support/query history disjoint、synthetic coefficient recovery、`afc4_circ==T4` 的精确约化。
3. 统一校准账本：trial 数、有效 label bins、seconds、labels per bin、`q`、support coverage/condition number、有效 samples、lag、λ/link、每通道退化情况。没有该账本，一律不能比较“低标签”。
4. 对 N4 另立小修复/复审任务：要么把 CPU precheck 也改为 fail-closed 并重跑为新 receipt，要么清楚标为 exploratory median-imputed audit。不得静默改变已存在结果，亦不得用 live N4/NS4 run 作为修改理由。

**Phase-0 kill：**任何 carrier 若需要 query labels、target BP、target-trained basis、未记录的 raw support cache、或无法给出 deterministic `[N,4]` 输出，即停止；不得用 FiLM/gate、增加 side width或 target autoencoder 来绕过。

### Phase 1 — M2 circular reduction/equivalence（正确性，不是刷分）

**问题：**AFC4 是否真的包含 native T4，而非事后给 T4 换名？

**锁定设定：**M2 native T4 的连续 angle、同一 chronological support budget（分别在当前合法 M24/M33 setting 中独立标注）、同一 valid directional-trial mask、同一 trial-rate/exposure、`λ=0`、同一 train-only normalizer、`side_dim=4` 与同一 source checkpoint/seed/session manifests。该 phase 只使用已允许的 source/held-in development scope；它不自动打开任何 held-out 文件。

**必须检查：**

- `afc4_circ` 与 T4 的 pre-normalization carrier、normalizer 后 carrier、row-shuffle 以及 encoder forward 输出的逐元素一致性（仅允许预先写明的 float rounding tolerance）。
- aligned `T4/afc4_circ`、`TS4/afc4_rs`、`T4 label-shuffle-refit/afc4_ls`、`Zero4` 和 N4/NS4 的 source-only matrix；已冻结结果只可引用，不能为“匹配”而重跑择优。
- 方向 design rank、condition number、每 session directional exposure；M2 centre/rest target 仍为 unlabeled，禁止人为赋角度。
- 同 budget 的 direct ridge operational reference，但它不得进入 equivalence 判据。

**Go/no-go：**

- **GO（只进入 Phase 2，不是 held-out GO）：**数值约化、normalizer 及 encoder output 三层均通过预写 tolerance，且所有 lineage/hash/no-leakage tests 通过。
- **NO-GO：**任何一层不等、或只能靠 λ、binning、normalizer/epoch 改动“接近”T4 才相同。记录为 estimator difference，先定位差异；不得声称 T4 是 AFC4 special case。
- **机制读法：**只有在 Phase 1 的匹配 controls 下 `AFC4_circ - afc4_ls` 与 `AFC4_circ - afc4_rs` 的 session-level effect 同向、并超过一项**事先由独立 source-only uncertainty/MDE receipt 锁定**的可分辨性规则，才可说 M2 中标签—响应关联有 evidence。不得为本路线发明数值噪声底；也不得借用 K4 cell 的 `+0.03` 当普遍阈值。

### Phase 2 — sub-C RT：连续与 event-aligned AFC4

**问题：**当 T4 因 rank-1 trial target direction 未定义时，是否仍能以同一解析接口从合法连续任务条件获得通道特异的功能载体？

#### 2A. 第一优先：`afc4_vel`

\[
\phi_{RT}(y(t))=[v_x(t-\tau),v_y(t-\tau)]^\top,
\quad e_i=[w_{ix},w_{iy},\|w_i\|,b_i].
\]

`τ`、bin size、valid velocity/exposure rule 与 ridge `λ` 只能在 source sessions 的 nested split 中选定并冻结。配对不得跨 trial 或 event segment；使用 velocity 是 dense-label operational cost，必须以有效 calibration seconds/bins 报告，不能仅写成“24 trials”。RT 中没有 T4 臂是设计要求：receipt 必须写 `T4 undefined because trial target_dir is rank-1`。

#### 2B. 条件性第二优先：`afc4_evt`

若 `afc4_vel` 通过 source-only integrity/preflight 但显示明显的 phase-mixing，才测试固定 event/phase basis；不得并行铺开 basis 搜索。一个可审计定义为

\[
\phi_{RT}(t)=B_{phase}(s(t))\otimes[v_x(t-\tau),v_y(t-\tau)],
\]

其中 `s(t)` 由 predeclared go-cue/event parser 产生，`B_phase` 是 source-frozen 的固定低阶 temporal basis。若原始维度大于 3，使用 source-only `U_source` 压为三维，再与 `b` 形成 AFC4。go-cue 缺失、segment 不完整或跨 support/query 的 event 一律 fail-closed；“整 trial 平均速度”明确禁止。

**RT 最小对照：**`F0`、`Zero4`、`afc4_vel`（随后才可 `afc4_evt`）、相应 `afc4_rs`/`afc4_ls`、`N4/NS4`（若对该 signal view 先独立资格审计）、以及 label-cost 明示的 `DirectRidge`。先冻结 15 个 RT session 的 manifest、signal view（SUA 或 deterministic pseudo-MUA 二选一）、chronological source/validation split、support duration和 query boundary；不得两种 granularity、两种 split 都跑后报最好者。

**RT go/no-go/kill：**

- **GO to a new, separately authorized held-out evaluation：**source-only/LOSO 预注册的 `AFC4-Zero4`、`AFC4-afc4_rs` 与 `AFC4-afc4_ls` 都通过已封存的 session-level direction + uncertainty/MDE rule；所有 eligible sessions 有完整 no-leakage/event coverage receipt。
- **Stop as no content-specific RT evidence：**任一 matched control不支持正确 attachment 或 label–response association，或 effect 仅相对 F0/DirectRidge 好而不相对 `Zero4`/shuffles 好。
- **Stop as task-basis failure：**`afc4_vel` 不通过且其 source-only failure 不能由预先定义的 phase-mixing audit 解释；不允许随后扫许多 temporal bases、lag 或 FiLM。仅在预定义 phase-mixing 条件满足时可启动一次 `afc4_evt`。
- 所有数值 practical margin 都须在**看 RT decoder aggregate 前**由 sealed source uncertainty audit 写入；本计划不伪造未测的 RT 噪声底。

### Phase 3 — Native M1：source-frozen EMG low-rank AFC4

**问题：**对于 16-D EMG 输出，能否避免把高维输出硬塞入 4D side path，同时保留 label-conditioned、闭式和无 target BP 的 carrier？

在每个 outer LOSO fold，仅使用 source sessions 的 EMG calibration distribution，以**预先固定的 exposure-weighted PCA** 估计并冻结 `P_source∈R^{16×3}`。首个 pilot 不开放 PCA/PLS 后择优；supervised PLS 属于另一个 estimator family，只有 PCA pilot 终止且出现新的 source-only 机制理由时才可另立计划。以 source-frozen means/scales 处理 EMG：

\[
z(t)=P_{source}^\top\operatorname{standardize}_{source}(\mathrm{EMG}(t))\in\mathbb{R}^3,
\]
\[
e_i=[w_{i1},w_{i2},w_{i3},b_i].
\]

target M1 calibration 只投影、累积 closed-form sufficient statistics 与拟合 `w_i,b_i`；它不重估 PCA、不旋转 signs、不以 support/query 表现选择 basis。basis 顺序按 source singular value 固定；每一列的 sign 以 source loading 中绝对值最大的元素为正来确定，平局按肌肉通道索引确定。basis sign/order、training-session names、means/std、explained variance、basis hashes 和 each-target projection diagnostics 必须入 receipt。

**M1 先决条件：**

1. 只采用已构造的 support/query-disjoint local development semantics：support `[0,10)`，selection `[10,210)`，report `[210,end)`；不得复用历史 `query_start=0` overlap 数字，也不得把该 local endpoint改称 hidden official score。
2. AFC4-EMG 是新的、可解释的 closed-form hypothesis，不是已停止的 D4、learned E4 或 temporal-prototype 分支的自动延续。它先需要独立 source-only feasibility/uncertainty receipt；没有新的明确授权，不开 GPU、不重开 M1 EvalAI。
3. 先做 basis/reduction audit：outer source fold 的 `P_source` 是否稳定、target support 的 `z` 是否满秩/有足够有效 bins、label/exposure alignment 是否正确；任何 failure 不可用 target PCA 或更大 side width补救。

**M1 最小对照：**`F0`、`Zero4`、`afc4_emg`、`afc4_rs`、`afc4_ls`、相同 EMG-bin label budget 的 `DirectRidge`；AFC4 与 DirectRidge 都必须披露其使用的是 10 个 trial 内的密集逐-bin EMG，而不能只写成“10 labels”。T4/TS4 与 D4/DS4只作为已知的 task-label mismatch/categorical context，不能代替 AFC4 的 label-shuffle control。N4/NS4 如要加入，必须先为 M1 16-D/该时间 bin signal view 定义并通过独立的 fail-closed reliability audit。

**M1 go/no-go/kill：**

- **GO to one reviewed decoder pilot：**source-only audit在冻结 basis、budget 和 label accounting 后表明 AFC4 的可靠性、attachment null 与 label-shuffle null均可区分，并通过先封存的 session-level uncertainty/MDE rule；data module contract 已保证 disjoint endpoint。
- **NO-GO：**basis instability、rank/coverage failure、任何 target data参与 basis/normalizer/λ选择，或只有 future-label oracle proxy 改善而没有机制对照。proxy 结果只能叫 source-only later-label oracle，不能叫 behavior-decoding gain。
- **Decoder-pilot stop：**若 `afc4_emg` 未同时优于 `Zero4`、`afc4_rs`、`afc4_ls` 的预先封存规则，停止该 branch；不增加 rank、latent width、target autoencoder、FiLM/gate、mixture 或 support sweep来追逐结果。

## 6. 全阶段泄漏、统计与硬件约束

### 6.1 不可协商的数据与统计约束

- **Checkpoint：**每个 comparison 对使用同一 source training split、固定 epoch/checkpoint rule和新的 exclusive run directory；不得为每一臂挑不同 best epoch。已冻结 arm 的数值只可按原 receipt 引用。
- **Basis/normalization：**`φ` 结构、`P/U_source`、lag、λ、GLM link、bin/exposure rule、z-score parameters 必须 source-only 并版本化。target support 仅允许 fit `w,b`；query从不参与 fit/selection。
- **Chronology：**support 是 query 前的 prefix；任何 50-bin（或模型实际 receptive field）history 必须完全落在 query boundary 后。连续标签的 lag pairing 不能跨 segment/trial 或越过边界。
- **Budget：**同一机制比较必须 exact-match labeled trials、有效 bins/seconds、labels/bin、coverage/condition number；operational 比较必须披露它们不同，不能只报 `M`。
- **Uncertainty：**session 是 biological unit。报告每 session paired delta、每 seed session mean、seed correlation、paired session bootstrap/预定义 nonparametric summary和 MDE；不把 seed 当新增动物。内部 LOSO 只做筛选，绝不拿来标定 held-out effect。
- **Formal isolation：**任何 official held-out/EvalAI 文件只有在全新的单次授权及 source gate 后才可打开；不以 Phase 1/2/3 的 development aggregate 选择正式版本。缺失/不合格 session要显式列出，不能静默删除。

### 6.2 部署/硬件边界

AFC4 保留的硬件主张应严格限定为：target calibration 不做 BP；持久 descriptor 是每通道 4 个值（`4N`，精度另行声明）；已有 decoder 接口仍是 `side_dim=4`，因此 decoder query-stage MAC/状态不因更换 AFC4 内容而增加。校准时不保存原始 support trial cache；闭式 ridge 至多维护固定 `q×q` Gram、`q×N` cross-statistic、means/counts和最终 `4N` descriptor。

这**不是**“所有 AFC4 一律相同成本”的许可。`q`、event basis order、precision、GLM iteration count、per-bin preprocessing、support duration均必须在该 task/version 预先上界化并在 receipt 中测量。`afc4_evt` 若需要未界定的大型 basis、每 event 自适应搜索或原始波形缓存，就不满足本路线的固定状态/MAC边界。任何宣称 ASIC/INT8/latency 优势之前必须给出对应实现的 state/MAC/latency receipt；本文不外推已有 T4 的具体数字。

## 7. 明确不优先的方向

- **更多 harmonic：**既有 P4 分层没有给出二次谐波一致收益；在 M2 上先证明 circular reduction，再讨论非圆周 basis，比叠加 `cos(2θ),sin(2θ)`更有信息量。
- **FiLM/gate/新 fusion：**既有 static T4GATE、relation和多条 residual 分支已给出非正或 stopped 证据；本路线的假设是 carrier content/task-variable match，不是 decoder 不会融合四个值。
- **target autoencoder / target learned set encoder：**它破坏“解析、固定小状态、no target BP”的核心约束。已停止的 M1 E4不能被换名重开。
- **在 target 或 held-out 上 sweep λ、lag、rank、phase basis或 EMG projection：**这会把“通用性”变成 target-specific architecture search。仅允许 source-only nested selection和一次预先定义的 conditional `afc4_evt`。

## 8. 论文可用表述（在相应证据完成后才可使用）

**方法段（可在实现/协议层使用）：**

> We formulate T4 as the circular special case of an analytic functional carrier: a source-defined task basis maps calibration labels into low-dimensional coordinates, and each target-session channel obtains a fixed-width functional descriptor by closed-form encoding regression. The target session performs no backpropagation; it only accumulates bounded sufficient statistics, fits the per-channel coefficients, and supplies a four-value descriptor to the unchanged decoder interface.

**结果/结论段（只有三类 matched controls 与 disjoint endpoint 均通过后可使用）：**

> Across the evaluated tasks, the evidence supports a narrow claim: correctly channel-attached, label-conditioned functional coefficients can add information beyond a zero side path, row-shuffled descriptors, and label-shuffled refits. This does not imply that a sinusoidal direction code is universal or that labels are universally necessary; rather, circular tuning is one task-matched basis within a fixed-interface, no-target-backpropagation calibration framework.

中文对应：

> 我们将 T4 视为解析功能载体的圆周特例，而非普适的 `sin/cos` 公式：source 阶段冻结任务基，target calibration 只以有界充分统计量完成逐通道闭式拟合，并把四维结果送入不变的 decoder 接口。
> 若 matched Zero4、row-shuffle、label-shuffle-refit 与严格时间分离均通过，结论也只是在已测任务中“正确通道附着的标签条件功能系数”具有额外价值；它既不宣称余弦方向调谐普适，也不宣称所有无标签方法或所有标签需求均被排除。

## 9. 与现有 handoff 的冲突和主审必须决定的风险

1. `HANDOFF_GENERAL_CARRIER_PROGRAM.md` 的早期版本曾把 per-trial average behavior 当 general carrier，后文已自行撤回；本路线明确采用 time-aligned `φ(y(t))`，不继承该错误定义。
2. 历史 N4 handoff 把退化处理写为 fail-closed，但已存在的 `n4_cpu_precheck.py` 使用 median imputation。需要主审决定：重做严谨 CPU receipt，或把现有 precheck 永久降为 exploratory。当前不能两种说法同时成立；旧 handoff 已从 public tree 移除但保留在 Git history。
3. 旧 general-carrier handoff 有“先估噪声底、再定门槛”的纪律；另一个 neural-only handoff 为一个 M24 K4 cell 写过 `+0.03`。本路线选择前者：所有 AFC4 的 practical/MDE decision 由各任务独立 source-only uncertainty receipt封存，不把历史 K4 数字伪称为全局噪声底。
4. M1 的 local clean endpoint 已可构造，但先前 D4/E4/prototype branch 已停止且 M1 EvalAI 已使用。AFC4-EMG若要继续，必须有新授权与单独的 source-only gate；本文件绝不自动开启 M1 GPU或官方评测。
5. `afc4_evt` 的 event parser、phase basis order、signal view和 state/MAC上界尚未实现，也不应在 `afc4_vel` 失败后临时发明。主审应先批准“只做 velocity”还是批准一个预先固定的 conditional event extension。

## 10. 最短执行顺序

```text
Phase 0: AFC4 contract + N4 degeneracy-policy disposition (CPU/source-only)
    -> Phase 1: exact M2 circular reduction/equivalence (development only)
        -> Phase 2A: RT afc4_vel, matched controls and source gate
            -> [only if predefined phase-mixing trigger] Phase 2B: afc4_evt once
        -> Phase 3: M1 source-frozen EMG basis audit (new authorization required)

At every arrow: sealed source-only decision -> separate review -> at most one newly authorized
chronological-disjoint held-out evaluation. No arrow opens formal held-out automatically.
```
