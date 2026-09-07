# HANDOFF: 通用调谐载体（连续运动学回归）与 MUA / 论文证据程序

**日期：2026-07-31**
**给：负责执行的 AI / 研究者**
**目标：不是再找一个架构 trick，而是（a）把调谐描述子从"绑定 center-out 8 方向"换成任务通用的载体，（b）把机制搜索从已饱和的 SUA 转到未饱和的 MUA，（c）补齐论文缺失的证据宽度与外部对比。**

**状态：待执行。所有门槛数值一律在 E1 噪声底估出来之后才允许写死（见 §六 纪律 1）。**

**本版（v2）相对 v1 的结构性改动**——v1 的入口是 `MUA × T8`，现已降级。三条新核对的事实推翻了 v1 的中心假设：
1. **T4 有两套互不相干的实现**，SUA 与 FALCON 的对比是**估计器混淆**的（§2.6）。
2. **FALCON 侧根本没有 `t8`**，且给 FALCON 的连续角度强加 8-bin 是估计器降级（§2.6）。
3. **sub-C RT 的 `target_dir` 是 rank-1 的**，现有载体在同猴同阵列换任务即静默失效（§2.7）。

→ **新入口是 E2「通用载体」**（per-unit 连续运动学回归），`T8` 降为次要谐波臂。

**v2.1 修订（同日，重要）**：v2 曾把通用载体定义为**逐 trial 平均行为向量**的回归
（`rate_i(trial) ≈ b_i + w_i · mean_behavior(trial)`）。**这个定义是错的**，已被 §2.8 的
只读审计推翻，并已替换为**运动时间对齐**的形式 `r_i(t) = b_i + W_i y(t−τ) + ε_i(t)`。
同时删掉了 v2 里那条无效的自检条款（"CO 上 `k4 ≈ T4`，否则是实现错了"）——审计显示
它在 CO 上本身就不成立。

## 2026-08-01 execution audit（本 handoff 不再是执行命令）

本文件提出的假设已经按独立协议审计，最终状态以
[`M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md`](M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md)
为准：

- movement-aligned K4 的 support-only encoding Gate A 成立，但这不是 decoder
  efficacy；
- 单个 held-in Gate B 不能作为放行依据；严格 M24 held-out 的
  `K4-T4=+0.018987`，低于 `+0.03` 主门；
- full CPU factorial 显示 M24 的 flattened/weighted W 可复现
  (`.6937/.7812`)，但典型逐 channel 方向 cosine 只有 `.1837`；
- `||W||` 与 `b` 更可靠（Pearson `.8780/.9866`），平衡采样和 ridge 没有修复
  per-channel direction；
- K4 使用 dense per-bin velocity，T4 使用每 trial 一个方向标签，因此
  `K4-T4` 只能是 operational comparison，不能归因于 carrier form；
- 条件式 K4/KS4 GPU precision replication 已停止。未来若测试
  `[||W||,b]`，必须改称“modulation-depth + baseline-rate identity”，并作为新假设
  重新预注册。

---

## 一句话

SUA 上连续四个机制方向判负的一个候选解释是**饱和/冗余**：T4 已把功能测到 0.575，teacher 冻结，
zero-init 加法项拿不到梯度。FALCON M2 的合法 future-query endpoint 没有同样的高绝对分数：
修正 M33 四-session 子集为 F0 `0.2058`、T4 `0.2778`，严格 M24 audited cell
为 F0 `0.1739`、T4 `0.2268`。但这不足以单独证明“饱和机制不存在”。
但真正卡住下一步的不是"哪个描述子更强"，而是**现有描述子的载体绑死在"每 trial 一个方向标签"上**：
它在 RT 上 rank-1 静默失效，在 M1 上用**方位角**去描述一个输出为**16 维 EMG** 的通道。
因此下一轮的入口是**把载体换成对被解码量本身的 per-unit 回归**——
它在 CO 上退化回 T4，在 RT 上可用，在 M1 上自动修掉描述子/目标错配。

---

## 二、当前证据全景（引用前必读，数字均已逐条核对）

### 2.1 四个已完成的 setting：T4 增益与 baseline 强反相关

| setting | 信号 | 解码目标 | baseline | T4 | T4−baseline | artifact |
|---|---|---|---:|---:|---:|---|
| SUA sub-C CO | 分选单元 | 2-D cursor_vel | 0.23642 (`b0`) | **0.57498** | **+0.33856** | `results/sua_spint_t4_mainline_fp32_v1/aggregate.json` |
| pseudo-MUA sub-C | 同电极求和 | 2-D cursor_vel | 0.20838 (`F0`) | **0.52612** | **+0.31774** | `results/pseudomua_t4_bridge_v1/summary.json` |
| FALCON M2 | threshold crossings | **2-D finger_vel** | 0.20582 (`F0`) | **0.27780** | **+0.07198** | corrected M33-eligible four-session subset; 4/4 positive, exact `p=.125` |
| **FALCON M1** | threshold crossings | **16-D EMG** | — | — | **withdrawn** | same `query_start=0` overlap; a future M1 claim also needs disjoint re-inference |

**2026-08-01 protocol correction:** the historical local replay scored from
trial 0, so its support and evaluation windows overlap. For M2, two sessions
contain exactly 33 trials and have no post-M33 query at all. The former M2 row
is therefore void as held-out-calibration evidence; it cannot support a “full
calibration budget” claim. New test-only `query_start=33` inference has now
re-issued the row above on the four eligible sessions with a full 50-bin
history-disjoint receipt. It was not obtained by deleting two rows from the old
aggregate, and it remains a local development subset rather than a hidden test.

**⚠ 表内两种 baseline 不可混比**（已逐一核对 artifact）：
- 第 1 行的 `b0 = 0.23642` 是**原版 SPINT**（无 side-feature 通路），`T4−b0 = +0.33856`。
- 第 2 行 pseudo-MUA 的 `F0 = 0.20838` 是**side-feature 置零**（通路在、内容空），
  `T4 = 0.52612`、`TS4 = 0.16045` → `T4−F0 = +0.31774`、`T4−TS4 = +0.36567`。
- 同一份 bridge artifact 里的 **SUA view 用的也是 `F0`**：`F0 = 0.31399`、`T4 = 0.56675`
  → `T4−F0 = +0.25276`。**与 pseudo-MUA 严格同口径的对照是这一对**：
  `SUA +0.25276` vs `pseudo-MUA +0.31774`，即**同电极求和后 T4 增益反而更大**
  （因为 `F0` baseline 在池化后掉得更多：0.31399 → 0.20838）。
- 因此不得把 `+0.33856` 与 `+0.31774` 并排读成"池化损失了 0.02"——那是两个不同 baseline 的差。

**M1 一行的历史解读已失效**（见 §2.6）：旧 overlap replay 中
`T4−TS4 = −0.00374`，但该数不能再证明“内容效应都没有”；M1 尚未完成合法
future-query correction。
v1 把它写成"身份已解析 → 描述子无事可做"。但现在核实到 **M1 的解码目标是 16 维 EMG，
而 T4 的标签是 target 方位角**（`SPINT-main/configs/model/falcon_m1.yaml:19` `num_covariates: 16`；
`streaming_calibration_exp/src/data/falcon_t4_features.py:45-48`）。
**描述子和解码目标不是同一个物理量**，这是一个比"天花板"更简单、更可检验的解释。
"0.627 是否接近天花板"仍**无证据**，必须自己算上界（E9），不得用别的数据集的数字（如 POYO 0.935）类比。

修正 M33 四-session 子集逐 cell 的 `T4−F0` 为
`fold1_s42 +0.061517`、`fold1_s43 +0.055583`、`fold2_s42 +0.098840`；
三者全正。严格 M24 audited cell 为 `+0.052885`、5/6 session 正。前者只有四个
独立 session clusters，后者只有一个训练 cell；两者都不能扩写成隐藏测试或完整
multi-cell significance claim。

### 2.2 T8 已存在、SUA 上确认无用、MUA 上不存在实现

- 定义：`mc_maze/unit_side_features.py:85,104,776-802`。`t8` = 逐方向平均发放率的
  8 维非参数 profile（`dir_0..dir_7`，`TUNING_NUM_DIRECTIONS=8`，跨 session 固定列序）；
  置换对照 `ts8` 已实现（`:171`）。
- **SUA 上的实测（关键）**：`results/linear_decoder_control.json` 的 `e3_headline_mean_r2`
  = `F0 0.31399 / T4 0.56675 / T8 0.57073 / TS4 0.31452 / TS8 0.30185`。
  **`T8 − T4 = +0.00398`，即在 SUA 上多出的 4 个自由度买不到任何东西。**
- 另有旧的 `results/p3_e3_tuning_ablation_t8_*_seed4*.json`（0.594/0.572/0.598），
  但那是 `best_checkpoint_validation_r2`，属协议 V4 已作废的 argmax 口径，
  **不得与任何 V4 数字并列**；如需引用必须按 epoch 5–12 重算。
- **MUA 上不只是"没跑过"，是"没有实现"**：`streaming_calibration_exp/src/data/falcon_t4_features.py`
  只导出 `T4_DIM = 4` / `T4_FEATURE_NAMES`（`:20-21`），没有任何 8-bin 结构。见 §2.6。

### 2.3 秩 1 假设：**已降级为次要假设**（v1 曾把它当中心假设）

原假设：T4 拟合 `rate(θ) = b + a·cosθ + c·sinθ` 只有一个基频分量；MUA 通道混合方向相反的 unit
会使基频相消，而 `t8` 的方向 bin 不受此限。**可证伪预测**：`T8−T4` 在 SUA ≈ 0（实测 +0.004），在 MUA > 0。

**三条反向/削弱证据，合起来把它从中心假设降级**：

1. **同口径下 pseudo-MUA 的 `T4−F0 = +0.31774` > SUA 的 `+0.25276`**（§2.1）。
   "只要求和池化 T4 就废"是错的——sub-C 的同电极求和反而让 T4 **更**有用。
   所以秩 1 惩罚不是池化的必然结果，只取决于**通道内实际混进了多少方向相反的 unit**。
2. **SUA↔M2 的 T4 对比本身是估计器混淆的**（§2.6）。"M2 上 T4 只 +0.065"里有多少来自信号、
   多少来自估计器（逐 trial 率 vs 逐方向均值）**目前完全未分离**。
3. **M1 的 null 有了更简单的解释**（描述子/目标错配，§2.1），不需要秩 1。

**因此秩 1 假设的检验方式改为分级、且并入 E2**：`T8−T4` 应随「通道内 preferred-direction
离散度」单调上升；在 sub-C 同电极（离散度低）处接近 0，在原生 threshold crossing 处才转正。
**不测这个离散度，`T8−T4≈0` 就无法区分"假设错"与"这批通道本来就没混叠"。**

### 2.4 证据宽度的真实短板

- SUA 只有 sub-C center-out：**257/257 个含 `task` 字段的结果 JSON 全为 `CO`**。
- `sua_exploration/data/dandi_000688/sub-C` 下有 **15 个 `sub-C_ses-RT-*` NWB，零结果**（见 §2.7）。
- sub-J 已下载 3 个 CO session，**零结果**；sub-M 从未动。
- **外部已发表方法零对比**：全库 grep `NoMAD|ERDiff|LFADS|CycleGAN|ADAN|NDT2|KalmanFilter|procrustes` → 0 命中。
  现有外部参照只有 SPINT 自身、`ridge_raw_window 0.30777`、`population_vector 0.08702`。
- **FALCON leaderboard 从未提交**：所有 MUA 数字的 `evaluation_scope` 明写
  "not a hidden EvalAI query/test-set result"。

### 2.5 已判负、不得重开的方向

| 方向 | 结论 | artifact |
|---|---|---|
| 电极身份 embedding / gate / anchor | ineffective（`t4gate−T4 = −0.0108±0.0049`） | `results/t4_gate_screen/aggregate.json` |
| same-electrode relation（减组均值） | ineffective（`−0.00144±0.00088`） | `results/sua_electrode_relation_full_v1_scheduler/multiseed_strict_aggregate.json` |
| **电极空间邻域先验** | **Stage 0 判负**，`stage1_candidate=false` | `results/electrode_spatial_prior_stage0_v1/audit.json`；见 `HANDOFF_ELECTRODE_SPATIAL_PRIOR.md` §零 |
| decoupled K/V v1 / dynamic keys | 失败 | `CURRENT_RESULTS.md` 相应节 |
| attention B15/B16 | **已撤回**，不是可用证据 | `README.md:6-12,33,119-120` |
| **T4 向零收缩（W3）** | **ineffective**，`+0.00075`（§三） | `results/sua_t4_shrinkage_m15_v1/t4w3_m15_s42.json` |

**这六条不得以任何形式重启，除非有新的、独立的机制性理由。**

### 2.6 ⚠ 新发现：T4 有两套互不相干的实现（估计器混淆）

这是本版最重要的方法学发现。**"T4" 在两个 setting 下不是同一个估计量**：

| | sub-C SUA | FALCON M1/M2 |
|---|---|---|
| 实现文件 | `mc_maze/unit_side_features.py` | `streaming_calibration_exp/src/data/falcon_t4_features.py`（**不 import 前者**） |
| 方向来源 | `trials.target_dir` | `trials.tgt_loc`（`:41-42` 缺失即 raise） |
| 角度表示 | **snap 到 8 个 canonical bin**（`:588-599`） | **连续弧度**（M1 `deg2rad`；M2 `arctan2(Δy,Δx)` 绕 `(0.5,0.5)`，`:45-59`） |
| 拟合对象 | **逐方向均值**（`:602-626` 明写不按 trial 数重加权） | **逐 trial 率**（`:113` `rates = sums[usable]/lengths[usable,None]`） |
| 退化处理 | `<2` 方向 → **静默全零填充**（`:914-922`） | rank≠3 → **raise**（`:106-114`） |
| 特征名 | `("m_cos_phi","m_sin_phi","m","b")` | `("m_cos_phi","m_sin_phi","m","baseline_rate")` |

**三个必须记住的后果**：
1. **§2.1 的跨 setting 比较有一个未量化的估计器分量。** 逐方向均值是一种**方差缩减**技巧
   （先平均再拟合），它利用了 CO 的离散结构；逐 trial 拟合没有这个红利。
   低 `M` 下这个差别可能不小。**E3 必须把它量化出来，否则 §2.1 的"适用边界律"不可发表。**
2. **给 FALCON 加 `t8` 需要从零实现**，且必须先决定如何把连续角度离散化——
   那等于给 FALCON 引入 SUA 侧的 8-bin 假设，是**估计器降级**。这是 T8 降级为次要臂的主因。
3. **静默全零填充是一个真实隐患**：`:914` 的注释写 "Expected to essentially never trigger in
   practice"，但 §2.7 证明它在 RT 上 **15/15 session 必然触发**。见 §六 纪律 8。

### 2.7 ⚠ 新发现：sub-C RT 的方向标签 rank-1 —— 现有载体的任务泛化性为零

**只读实测（本轮，全部 15 个 RT session，无一例外）**：

| 事实 | 值 |
|---|---|
| RT session 数 | 15（`sua_exploration/data/dandi_000688/sub-C/sub-C_ses-RT-*.nwb`） |
| `trials.num_targets` | **每个 trial 都是 4**（一个 trial 内连续多次 reach） |
| `trials.target_dir` 有限值的**唯一取值数** | **1**，恒为 `0.785398`（= π/4）——15/15 session 一致 |
| trial 数 / 有限标签数 | 例：`RT-20131009` 187/168；`RT-20150318` 921/830 |

**代码后果**（`unit_side_features.py`）：`:868` 把每个 trial snap 到同一个 index →
`present_directions` 长度 1 → `:914` `len(present_directions) < 2` →
**整个 session 每个 unit 都拿到固定全零 T4/T8，且不抛异常、不报警**。
（`t4c`/`t4w3` 会在 `:905` 因 `rank=1` raise，但主力 `t4`/`t8` 不会。）

**但 RT 本身完全不缺方向信息** —— 这是关键区分。同一批文件里有连续运动学：

| 事实 | 值 |
|---|---|
| `processing/behavior` | `Position` / `Velocity` / `Acceleration` |
| `cursor_vel` | `(110511, 2)`（`RT-20131009`） |
| 运动样本方向的 8-bin 直方图 | `[5532, 4666, 5817, 6427, 5682, 4686, 6032, 5362]` → **近乎均匀** |
| `go_cue_time_array` 有 ≥2 个有限值的 trial | 154/187（可做 per-reach 切分） |

**结论**：**RT 不是"方向贫乏"，是"trial 表标签不可用"。** 方向信息就在 kinematics 里、
且覆盖均匀。所以"换一个更通用的载体"不是审美偏好，而是**解锁 15 个已在磁盘、
零结果的 session 的唯一途径**。这也直接回答"余弦嵌入是否只因 CO 轨迹而有效"：
**现有载体对同猴、同阵列、换个任务的泛化性是 0，且失效方式是静默的。**

### 2.8 ⚠ 新发现：逐 trial 平均行为向量**不能**代表该 trial 的运动方向

这一条直接否掉了 v2 对通用载体的定义（见文首 v2.1 修订）。**只读审计结果**：

| 数据 | 整 trial 平均速度模长 / 平均瞬时速度模长 | 与 target direction 的中位角误差 |
|---|---:|---:|
| sub-C **RT**（15 session） | **≈ 0.24** | **≈ 90.7°** |
| sub-C **CO**（53 session） | **≈ 0.664**（中位比值） | **≈ 41°** |

**机制**：RT 一条 trial 含 4 次 reach（`num_targets=4`，§2.7），含返回运动，
时间平均把约 3/4 的方向性抵消掉；CO 虽是单次 reach，但整段 trial 还包含等待、启动、
到达与非主运动区间，所以平均速度**不是** `speed × target_direction`。

**为什么这比 §2.7 更危险**：`target_dir` 的 rank-1 至少在 `t4c`/`t4w3` 上会 raise。
而逐 trial 平均速度的设计矩阵**仍然是 rank 3**——没有秩塌陷可抓，得到的是
**"数值上可拟合、物理上不代表运动"的连续特征**。这是与 §2.6/§2.7 同一类的**保秩静默退化**，
只是更难发现。

**两个直接后果**：
1. 若按 v2 的定义实施 E2/E4，**很可能得到一个错误的 negative，或一个无法解释的 positive**。
2. v2 写的自检"CO 上 `k4 ≈ T4`，否则是实现错了" **无效并已删除**——CO 上 41° 的中位角误差
   意味着这个近似本身不成立，它不能用来判断实现是否正确。

**（审计来源标注）** 上表两行来自本轮对本地 NWB 的直接只读审计，**尚未落成 artifact**。
执行方必须在 E2 的 `audit.json` 里**复算并写入**这两行（字段
`mean_velocity_magnitude_ratio`、`mean_velocity_angular_error_deg`，逐 session + 中位数），
复算不一致就以复算为准并在本文件更正。

---

## 三、W3 / M20 的处置（**W3 已用实测数据自己判负**）

### 3.1 结论先行：`T4W3@15` 已跑完，增益 ≈ 0

按 V4 口径（epoch 5–12 窗口均值）核对：

| 臂 | epoch 5–12 mean R² | artifact |
|---|---:|---|
| `T4@15` | 0.52496895 | `results/sua_t4_shrinkage_m15_v1/t4_m15_s42.json` |
| `T4W3@15` | 0.52572195 | `results/sua_t4_shrinkage_m15_v1/t4w3_m15_s42.json` |
| **delta** | **+0.00075** | — |

`+0.00075`，对照 `σ_seed ≈ 0.0385`——**这是彻底的零**。而 `M=15` 正是 W3 理应最有用的格子
（deficit 最大、拟合噪声最大）。**在最有利的条件下拿到零，比在中性条件下拿到零信息量更大。**

这与 Stage 0 的 train-only proxy 不矛盾：proxy 上 `zero_vs_ordinary` 的 MSE 比确实是 `0.9537`
（27/27 session 改善），但那只说明**收缩改善了发放率预测**；**下游解码 R² 完全没有跟着动**。
**这是本项目第二次观察到 rate-MSE proxy 不传导到任务指标**，值得单独写进论文的方法学部分，
并且**直接影响 E2 的判据设计**（见 §四 E2 的"proxy 传导性"要求）。

（注：同目录另有 `t4w3_m15_s42.invalid_receipt_pre_c4ac509.json`，是 `c4ac509` 之前的作废
receipt，**不得引用**；有效的是 `t4w3_m15_s42.json`。）

### 3.2 GPU 现状：没有 W3 在跑，无需 kill 任何东西

当前两个 compute PID `855235` / `855236` 跑的**不是** W3，而是
`sua_b3t_t4_efficiency_v1`（`--variant B3S` / `B3TS`，`--side_features t4`，
`--side_feature_pool_size 30`），即**标注效率曲线**，正好是 §四 E7 的工作，
产物目录 `results/sua_b3t_t4_efficiency_v1/`。

### 3.3 处置：W3 封口，不加 seed、不加档、不做 M20

- **不补 seed 43/44**：`+0.00075` 距 `+0.03` 门差 40 倍，再加 2 个 seed 不可能翻盘。
  按四态判决记 **ineffective**（须注明 `n_seeds=1` 这一诚实边界）。
- **`TS4W3@15` 不必再启动**：aligned 臂已是零，内容门无对象可测。
- **M20 取消**：收缩红利随 M 单调衰减（Stage 0 `zero_vs_ordinary`：M10 `0.9283` →
  M15 `0.9537` → M20 `0.9725`），而 M15 这个红利最大的档在下游已经是零 → M20 只会更零。
  对固定 margin 做 M 扫描在 `M→50` 时必然通过；M15 出零后再加档，形式上是预注册、
  实质是 forking paths。M20 的 runner/validator（commit `c50e8f9`，20 passed）
  **保留在库里不删**，只是不启动。
- **W3 的归宿**：E7 标注效率曲线上的一条平线 + 论文里"试过并诚实报负"的一段。

**执行方不得自行 kill 进程**：正在跑的 `sua_b3t_t4_efficiency_v1` 属 E7，应让它跑完。

---

## 四、要进行的实验

> **门槛数值一律留空，由 E1 的噪声底填入。** 本项目已两次因先写门槛后估噪声而设出不可达门槛。
> 第三次不允许发生。

### E1 —— FALCON M2 噪声底估计（**0 GPU，阻塞所有门槛**）

**为什么先做**：现有 MUA 网格只有 3 个 cell（`fold1_seed42`/`fold1_seed43`/`fold2_seed42`，
`fold2_seed43` 缺失），而 fold2 明显更吵（`0.024–0.077` vs fold1 `0.008–0.017`）。
在此之前写任何 MUA 门槛都是无根据的。

- 从 `results/native_mua_t4_v1/aggregate_m2.json` 的 `artifacts` 指向的 run 目录读 epoch 曲线，
  按 V4 口径（epoch 5–12 平均）算窗口内 `σ_run`、跨 seed `σ_seed`、跨 fold 分量。
- `σ_delta_paired = stdev(逐 seed 的 mean delta)/√n_seeds`，同时报 unpaired quadrature 与隐含 `ρ`。
  `n_seeds < 2` 必须 raise。
- **交付物**：`results/mua_noise_floor_v1/m2_sigma.json`，含各 σ 分量、`2σ_delta_paired`、
  以及"在现有 cell 数下可分辨的最小效应量"。
- **输出用途**：E4/E5 的门槛 = `max(2σ_delta_paired, 部署相关下限)`，填入本文件后冻结。
- 附带：若 `fold2_seed43` 缺失使 σ 估计不可靠，须明确写"需补该 cell"并列为 E4 前置。

### E2 —— ★ 通用载体 `k4`：免训练 proxy（**0 GPU，本轮最高优先，E4 的 go/no-go**）

**动机**：§2.7 证明现有载体绑死在"每 trial 一个方向标签"上；§2.1/§2.6 证明 M1 上它甚至
描述错了物理量。**通用载体**把"方向"这一层中间表示去掉，直接回归被解码量本身。

**定义**（新描述子，token `k4`，置换对照 `ks4`）——**运动时间对齐的 encoding signature**：
对每个 unit/channel `i`，在**短时间 bin** 上做最小二乘
```
r_i(t) = b_i + W_i · y(t − τ) + ε_i(t)
```
- `r_i(t)`：unit `i` 在 bin `t` 的 spike rate（bin 宽与模型输入 bin 一致，**不跨 trial 平均**）。
- `y(t)`：**同一时间尺度**的行为输出（sub-C：2-D `cursor_vel`；M2：2-D finger_vel；M1：16-D EMG）。
- `τ`：神经—行为延迟。**必须预注册一个固定值，或只用 train session 选**（nested、不看评估区间）；
  选择过程与最终值都要写进 audit.json。
- **RT 必须按 `go_cue_time_array` 切成 4 段 reach 分别取 bin，不得整条 trial 平均。**
- 特征 = `[W_i, ‖W_i‖_F, b_i]`；**2-D 输出时恰好 4 维** `[W_ix, W_iy, ‖W_i‖, b_i]`。
- **descriptor 拟合区间与效果评估区间必须 chronological disjoint**（同 session 内按时间切，
  拟合用前段、评估用后段），不得共用 bin。

**为什么是这个形式，不是逐 trial 平均**：§2.8 实测——整 trial 平均速度在 RT 上只保留
约 24% 的方向模长、中位角误差 90.7°，在 CO 上是 0.664 / 41°。**它的设计矩阵仍是 rank 3，
所以不会像 §2.7 那样显式塌陷，而是静默产出"可拟合但不代表运动"的特征。**

- **4 维 → 可直接复用现有 `side_dim: 4` 通路，不改模型**（`configs/model/streaming_b3s_t4*.yaml:7`）。
  这是选它作为入口的关键工程理由（16-D EMG 的 M1 需要另配 `side_dim`，见 E4）。
- **与 T4 的数学关系（只作为定性理解，不作为自检）**：T4 的 `[a,c]` 是 `rate` 对
  `(cosθ, sinθ)` 的回归系数（`unit_side_features.py:622-626` 的 design 是 `[1, cosθ, sinθ]`）。
  `k4` 把这个单位圆嵌入换成真实速度向量。**两者在 CO 上并不数值等价**（§2.8），
  所以**不得**用 "`k4 ≈ T4`" 判断实现是否正确。

**替代的正确性检查（三条，必须全做，全部 0 GPU）**：
1. **合成数据回收**：用已知 `W_i`、已知 `τ` 生成 Poisson/Gaussian 发放率，检验拟合能
   回收 `W_i` 与 `τ`（角误差、模长比）。**这才是实现正确性的检查，不是 CO 对比。**
2. **`τ` 的曲线单峰性**：在 train session 上扫 `τ`，held-out rate-MSE 应有明确极小；
   若曲线平坦，说明对齐无信息，须先查 bin/时间戳对齐再继续。
3. **置换基线 `ks4`**：打乱 `y(t)` 与 `r_i(t)` 的时间对应后重拟合，rate-MSE 必须显著变差。

**必跑的四个格子**：

| 格子 | 目的 | 预期 |
|---|---|---|
| sub-C **CO** | 标定（`τ` 选择 + 合成回收 + 与 T4 的**定性**比较） | `k4` 应 ≥ `F0`；与 T4 的差是**结果**，不是 bug 判据 |
| sub-C **RT** | 解锁性检查 | T4 **无定义**（rank-1，§2.7）；`k4` 可算 → 这本身就是结果 |
| FALCON **M2** | MUA 上是否更好 | 未知；`tgt_loc` 角度 vs 直接回归 finger_vel |
| FALCON **M1** | 检验"描述子/目标错配"解释 | `k4`（回归 16-D EMG）应 **> T4**（回归方位角） |

**RT 切分本身必须 fail-closed（新增，源于实测）**：`RT-20131009` 只有 **154/187** 条 trial
有 ≥2 个有限 `go_cue_time_array` 值，且 trial 0 的 `go_cue_time_array`/`target_id` 全 NaN。
所以 per-reach 切分**必须先审覆盖率**：逐 session 报可切分 trial 数 / 总 trial 数、
被丢弃的 bin 数，写进 audit.json；覆盖率低于预注册阈值就显式失败，**不许静默丢弃**（§六 纪律 8）。

**必须一起报的两项**（否则结论不可解释）：
- **秩 1 分级检验**（§2.3）：sub-C 有真 sorting，对每个电极算其 unit 的 preferred direction
  圆形离散度（如 `1 − |Σ m_u e^{iφ_u}| / Σ m_u`），把电极分低/中/高三层，
  检验 `T8−T4` 与 `k4−T4` 是否随离散度**单调上升**。FALCON 无 sorting，
  只能报通道级代理量 `higher_harmonic_ratio`（`t8` profile 的 DFT 去掉基频后的能量占比）。
- **proxy 传导性**（§3.1 的教训，**这是本项目第三次要用 proxy 做门，前两次一次成功一次失效**）：
  除 held-out rate-MSE 外，必须**同时**报一个更接近任务的量——
  用该描述子做 §七 的 `population_vector` / `ridge_pooled_rate` 式**线性解码 R²**（0 GPU）。
  **只有 rate-MSE 与线性解码 R² 同向改善才放行 E4**；只有前者改善则记录为
  "proxy 改善但不传导"，**不放行**。

**低 `M` 下的已知风险（必须预注册，不得事后解释）**：T4 在 SUA 侧拟合**逐方向均值**
（§2.6），这是利用 CO 离散结构的**方差缩减**；`k4` 逐 bin 回归的样本数虽然多得多，
但噪声也大得多（单 bin spike count 方差高），**在 `M=10~15` 时可能反而更差**。
缓解手段**必须事前选定并写死**：
(a) 对 `W_i` 加岭正则，λ 由 train-only nested 选；
(b) bin 宽与平滑核事前固定，不作为可调项；
(c) 系数标准化（`W_i` 的量纲随行为变量尺度变，跨 session 可比性必须显式处理——
M2 还有 `behavior_scaling_factor: 5.0`）。

- **交付物**：`results/general_carrier_proxy_v1/audit.json`，含逐格子的
  `k4_vs_t4` rate-MSE 比与线性解码 R² 差、bootstrap CI、逐 session 改善数、
  `tau_selected` 与 `tau_sweep_curve`、`synthetic_recovery`（`W` 角误差 / 模长比）、
  `rt_segmentation_coverage`、`mean_velocity_magnitude_ratio` 与
  `mean_velocity_angular_error_deg`（§2.8 的复算）、
  `direction_dispersion`（逐电极值 + 分层边界）、`higher_harmonic_ratio`、
  `{t8,k4}_minus_t4_by_dispersion_stratum`（单调性）、`transduction_consistent` 布尔、
  `chronological_disjoint` 布尔（拟合/评估区间无重叠的机器可检证明）、
  `stage1_candidate` 布尔。
- **纪律**：只开 train/calibration session；不碰 held-out / EvalAI 评测数据；产物 SHA 绑定。
  `k4` 消耗 behavior label，与 T4 同级，**必须注册进
  `unit_side_features.py:317` `side_features_use_behavior_labels()`**。
  注意 `k4` 消耗的是**连续 kinematics**（比 T4 的 trial 标签更强的信息），
  所以"校准预算"的口径要重新定义并写死：**用于拟合的时间长度**（秒 / bin 数），
  而不只是 trial 数，否则与 E7 的 `R²(M)` 曲线不可比。

**2026-08-01 attribution and naming correction:** matching the number of
support trials does not match label information. `K4−T4` mixes dense per-bin
kinematics access with carrier/estimator form and is therefore descriptive
operational comparison only—not a mechanism contrast. `K4−F0` remains an
end-to-end utility comparison that includes the cost/benefit of dense labels;
`K4−KS4` is the label- and width-matched mechanism gate for correct
channel-attached content. A same-dense-label direction-only estimator may be
added later, but is not required merely to report this limitation honestly.

The paper name must follow the component ablation. If standardized
`[||W||, b]` is equivalent to full `[Wx, Wy, ||W||, b]`, the claim is renamed
**modulation-depth + baseline-rate identity** and must not be called a
motion-aligned directional signature. If `b` alone is equivalent, the claim is
reduced again to baseline-rate identity. “Equivalent” requires a predeclared
R² equivalence margin derived from the uncertainty audit and an interval-based
equivalence decision; failure to find a significant difference is not enough.

### E3 —— 估计器混淆的量化（**0 GPU，§2.1 可发表性的前置**）

§2.6 显示 SUA 与 FALCON 的 T4 是两个估计量。在修好这一点之前，
§2.1 的"适用边界律"（E9）不能写进论文。

- 在 sub-C CO 上**同时**用两种估计器算 T4：(a) 现有 8-bin + 逐方向均值；
  (b) FALCON 式连续角度 + 逐 trial 率。比较其 held-out rate-MSE 与线性解码 R²，扫 `M ∈ {10,20,40,50}`。
- **交付物**：`results/t4_estimator_equivalence_v1/audit.json`，给出
  "估计器差异贡献了多少"的定量上界。
- **两条出路，事前二选一并写死**：若差异可忽略 → §2.1 可直接跨 setting 比较；
  若不可忽略 → **必须统一实现**（建议统一到连续角度 + 可选分组平均），
  且 §2.1 的表须重算后才可发表。

### E4 —— ★ `k4` 的 GPU 臂（**由 E2 放行、门槛由 E1 填**）

- 臂：`f0`、`t4`（现有）、`k4`（新）、`ks4`（新，置换对照）。
- 优先级：**M1 > M2**。理由反直觉但明确——M1 是"描述子/目标错配"假设的**判决性格子**
  （§2.1），且那里 T4 的 `−0.002` 给了一个干净的零基线；M2 上 T4 已经 +0.065，增量空间被占。
- **M1 的 side_dim 不是 4**：回归 16-D EMG 时特征是 `[W_i (16×?), ‖W_i‖, b_i]`，
  须先决定降维/汇总方案（如只取 `‖W_i‖` 加前 k 个主方向）**并在 E2 里就冻结**，
  否则 E4 会变成一次架构搜索。**这个决定必须写回本文件后才允许开 GPU。**
- sub-C **RT** 是第二个判决性格子：T4 在那里无定义，所以 `k4` 只要显著 > `F0`，
  就是"载体通用性"这一 claim 的正面证据（且是**新任务**，直接补 §2.4 的短板）。
  **前置：E6（静默失效修补）+ E2 的 RT 切分覆盖率审计通过。**
- 网格：补齐完整 fold × seed（含 E1 指出的缺失 cell），不得只跑单 cell。
- 沿用 V4：独占 run 目录、固定 epoch 预算关 early stopping、epoch 5–12 平均。
- 主判据：`k4 − t4`（机制增量）与 `k4 − ks4`（内容门），两者都要过。
- **descriptor 拟合区间必须与训练/评估在时间上不重叠**（同 E2），receipt 里要能验证。

### E5 —— T8 作为次要谐波臂（**降级，仅在 E2 显示离散度分级为正时才做**）

v1 曾把这个当主线。现在的条件是：**只有** E2 的 `T8−T4` 随 preferred-direction 离散度
单调上升、且高离散层显著为正，才值得上 GPU。
- **只在 sub-C（有真 sorting，离散度可测）上做**，臂 `t8`/`ts8` 已实现。
- **不给 FALCON 实现 `t8`**：需要给连续角度强加 8-bin，是估计器降级（§2.6）。
  若确要在 MUA 上测谐波阶数，正确做法是给 `k4`/连续角度版本加二次谐波项
  （`cos2θ, sin2θ`，即 `docs/T4_NEXT_ROUND_IDEATION.md:61-62` 的候选 #9），而不是移植 8-bin。

### E6 —— RT 静默失效的修补（**0 GPU，小改动，安全性质**）

§2.7 的静默全零填充是隐患：任何人在 RT 上跑 `t4`/`t8` 会拿到一堆零特征而看不到异常。
- 在 `unit_side_features.py:914` 的分支上，把"静默计数"改为**显式失败或显式警示**：
  至少要让 `insufficient_direction == num_channels`（即**整个 session 无一个可用方向**）
  这种情形无法被静默通过。
- 保留 receipt 里 `insufficient_direction` 的计数字段（向后兼容），只加硬失败/告警。
- 附带修正 `:914-921` 那段注释——"Expected to essentially never trigger in practice"
  已被 15/15 RT session 反证。
- **须有单元测试**覆盖 rank-1 方向输入。

### E7 —— 标注效率曲线（**已在跑，PID 855235/855236**）

`R²(M)`，`M ∈ {10,15,20,30,50}` × `{ordinary, W3}`。
- **已在执行**：`results/sua_b3t_t4_efficiency_v1/`，应让它跑完（§3.2）。
- 已有点：`T4@50 = 0.583811`、`T4@15 = 0.524969`（`delta = −0.058842`，6/6 session 下降）；
  `T4@15 − TS4@15 = +0.190956`（6/6 正，Wilcoxon `p=.03125`，CI `[+0.150568,+0.232131]`）——
  **15 trial 下 T4 内容效应仍很强，缺的只是均值水平**，这是曲线的核心叙事。
- **W3 分支已封口、是一条平线**：`T4W3@15 = 0.525722`，`W3−ordinary = +0.00075`（§3.1）。
  图上画这一个点并标注"收缩无效"，**不再补 M20 或额外 seed**。
- 若 E4 为正，**`k4` 应加进这张曲线**——低标签下的载体对比是很好的论文图。

### E8 —— FALCON leaderboard 提交（**M2 私有提交已完成；held-out 较弱**）

M1/M2 已有完成的本地结果；本轮首次把 M2 正式提交。结果返回后可获得与已发表方法的同台对比、
第三方私有测试集数字。**这是在不实现任何外部 baseline 的前提下拿到外部对比的唯一途径。**
- FALCON 的作者是 Karpowicz，也是 NoMAD 的作者；FALCON H1（DANDI 000954）
  就是 NoMAD 公开数据所属的人类 BrainGate 家族。仓库已有
  `streaming_calibration_exp/configs/data/falcon_h1.yaml`（`calibration_n_trials: 2`），
  `CURRENT_RESULTS.md:872` 记为「尚未完成」。**H1 若能跑通，等于在 NoMAD 作者的
  benchmark 上、用他们的私有测试集，对 NoMAD 系方法做对比。**
- 提交涉及外部服务，**须先经用户确认**，并遵守 test 封存纪律。
- **2026-08-01 更新**：用户已明确授权尝试 E8。首个候选已冻结为原始 SPINT M2
  `epoch_027`（不是 T4/K4）；host minival `R²=0.52384857`，容器 minival
  `R²=0.52384860`，远端目录/输出接口模拟通过。镜像
  `spint-m2:e8-epoch027-76f0fb2`，image ID `sha256:b179efcba8e3...`，容器内模型
  SHA 与冻结 pkl 完全一致。认证已确认 `HKU-ECE` team `41975` 与 Test Phase
  `few-shot-test-2319`（phase `4599`）。镜像 manifest 已上传且 digest 与本地 image ID
  完全一致；EvalAI 于 `2026-08-01T07:29:43.100855Z` 接受一次**私有**正式提交
  `578218`，最终状态 `finished`、无 stderr。官方 `test_split_m2` 返回：
  **Held-out `R²=0.18647872 ± 0.16214462`，Held-in `R²=0.56824267 ± 0.03116488`，
  Normalized Latency `0.11236103`**。这说明原始 SPINT 在 hidden held-in 上健康，但跨
  session held-out 明显较弱且异质，强化了“held-out few-shot calibration 才是核心终点”的
  问题动机；它**不是** T4/K4 的正向结果。提交是 private，不能声称 public leaderboard
  名次或相对某方法显著更好。旧 EvalAI CLI 与 Docker 29 的兼容修复、上传 UUID、官方
  文件哈希和正式回执见 `docs/E8_FALCON_EVALAI_SUBMISSION_RECEIPT.md`；机器回执见
  `results/e8_falcon_evalai_m2_spint_epoch27_v1/submission_578218_receipt.json`。

### E9 —— 适用边界律与 M1 天花板（**0 GPU + 可能 1 个小 run；被 E3 阻塞**）

- 把 §2.1 四个点做成定量分析。**现在有三个候选轴，不是两个**：
  (i) 身份是否已解析；(ii) 池化是否破坏描述子；(iii) **描述子与解码目标是否同一物理量**（§2.1）。
  E2 的 M1 格子直接判 (iii)。
- **必须自己确定 M1 的 0.627 是否接近上界**：用同数据的 within-session / held-in 训练性能
  作为上界估计。若需要新 run 就跑最小规模的那个；**不得用跨数据集类比代替**。
- **前置**：E3。估计器混淆未量化前，跨 setting 的定量律不可发表。

### E10 —— 变维不适用性论证（**低-中成本，替代实现 NoMAD**）

SUA sub-C 的 33 个 train+val session unit 数为 **38–91（中位 60，23 个不同取值）**，
且每天独立分选、无稳定身份。NoMAD/ERDiff 这类方法假设通道身份稳定。
- 用一个 fixed-dimension 对照（把每 session pad/truncate 到固定 N 的身份无关对齐基线）
  展示其性能随 N 变化而退化，从而**证明**（而非断言）现有 alignment 范式不适用于本 setting。
- 这比移植 NoMAD 便宜，且论文里是更强的 claim（定义新问题 > 在旧问题上刷分）。

---

## 五、执行顺序与依赖

```
E1 (0 GPU, 噪声底) ─────────────┐
                                ├──> E4 (GPU: k4, 优先 M1 与 RT)
E2 (0 GPU, ★通用载体 proxy) ────┤         │
   └─(离散度分级为正)──> E5 (T8 次要臂)   │
E3 (0 GPU, 估计器混淆) ──> E9 (边界律 + M1 上界)
E6 (0 GPU, RT 静默失效修补) ── 独立，E4 跑 RT 前必须先做
E7 (标注曲线) ───────────────── 沿用在跑的 run；E4 为正则加 k4
E8 (leaderboard) ────────────── `578218` 已完成；作为原始 SPINT M2 held-out 外部锚点
E10 (变维论证) ──────────────── 独立，低优先
```

**建议起手：E1 + E2 + E3 + E6 四项并行（全部 0 GPU）。**
E2 是本轮的科学入口；E1 给门槛；E3 解锁 E9；E6 是 E4 跑 RT 的安全前置。

---

## 六、纪律（不可协商）

1. **先估噪声底，后写门槛。** 本项目已两次犯反。§四所有门槛留空，由 E1 填入并冻结；
   填入后不得因结果修改。
2. **数据隔离**：只用 train + validation session。SUA 的 6 个 test session 的
   spike/behavior/trial 一律不加载，只允许读 unit-table 行数以固定 `N<100` regime。
   不创建 / 不修改 / 不删除 formal-test receipt。FALCON 的 held-out/EvalAI
   评测数据仅通过官方允许通道使用。**RT 的 15 个 session 不在封存集内**（封存的是 CO test 6 个），
   可自由使用，但一旦用于开发就必须在 manifest 里显式划分 train/val。
3. **V4 估计量**：独占 run 目录、固定 epoch 预算关 early stopping、epoch 5–12 算术平均、
   配对 `σ_delta_paired`、四态结论 `effective / effective_heterogeneous / ineffective / indeterminate`。
   只有 `ineffective` 才允许写成阴性结论。
4. **旧口径隔离**：任何 `best_checkpoint_validation_r2` 数字（如 §2.2 的旧 T8 结果）
   不得与 V4 数字并列比较，需要就重算。
5. **免训练关口优先**，但**必须检验 proxy 传导性**：W3 的教训是 rate-MSE proxy 改善
   （27/27 session）可以完全不传导到下游 R²（`+0.00075`）。因此 E2 的放行条件是
   **rate-MSE 与线性解码 R² 同向改善**，不是单看 MSE。
6. **不重开 §2.5 的六个已判负方向。**
7. **负结果照实写**：E2 若显示 `k4` 在 M1 上也无用，就写"描述子/目标错配"解释被否；
   E9 若显示 M1 远未到天花板，就写 T4 在 M1 失效另有原因。不得因不利于叙事而弱化。
8. **静默退化零容忍**：§2.7 暴露了"特征全零但不报错"的路径，§2.8 暴露了更隐蔽的一种
   ——**设计矩阵满秩但特征物理上无意义**。此后任何新描述子（含 `k4`）都必须：
   (a) 让"整个 session 无可用标签"无法静默通过；
   (b) 报告一个**物理合理性量**（如 `W_i` 与已知调谐方向的一致性、`τ` 曲线是否有极小），
   而不只是报"拟合成功/rank 足够"；(c) 有单元测试覆盖退化输入。
9. **时间对齐与 chronological disjoint**（新增）：任何消耗连续行为量的描述子，
   其拟合区间必须与效果评估区间**在时间上不重叠**，且延迟 `τ` 只许用 train session 选。
   receipt 必须包含可机器校验的区间边界。**不得用整条 trial 平均代替时间对齐**（§2.8）。

---

## 七、关键文件索引

- **SUA 侧调谐特征**：`mc_maze/unit_side_features.py:85,104,171,776-802`（`t8`/`ts8`）、
  `:588-599`（`_nearest_canonical_direction_index`，8-bin snap）、
  `:602-626`（`_fit_cosine_tuning`，逐方向均值）、`:905-922`（退化分支 / 静默全零填充）、
  `:317`（`side_features_use_behavior_labels`，新描述子必须注册）
- **FALCON 侧调谐特征（另一套实现）**：`streaming_calibration_exp/src/data/falcon_t4_features.py:20-21`
  （`T4_DIM`/名称）、`:24-60`（`calibration_target_angles`，连续角度）、
  `:76-79`（`_design`）、`:82-120`（`t4_from_trial_sums`，逐 trial 率 + rank 硬检查）
- **解码目标维度**：`SPINT-main/configs/model/falcon_m1.yaml:19`（`num_covariates: 16`，EMG）、
  `falcon_m2.yaml:19`（`num_covariates: 2`，finger_vel）
- 已完成 SUA 主线：`results/sua_spint_t4_mainline_fp32_v1/aggregate.json`
- 已完成 MUA：`results/native_mua_t4_v1/aggregate_m{1,2}.json`、
  `results/native_mua_heldout_t4_v1/aggregate_heldout.json`
- pseudo-MUA 桥：`results/pseudomua_t4_bridge_v1/summary.json`、`docs/PSEUDO_MUA_T4_BRIDGE_48H.md`
- 经典基线（E2 的线性解码 proxy 可直接借）：`scripts/linear_decoder_control_dandi688.py:20-32`、
  `results/linear_decoder_control.json`（含 `e3_headline_mean_r2` 的 T4/T8 对比）
- 免训练审计范式（可借鉴，勿修改）：`scripts/audit_electrode_spatial_prior.py`
- 判据：`docs/MEASUREMENT_PROTOCOL_V4.md`
- 已判负记录：`docs/HANDOFF_ELECTRODE_SPATIAL_PRIOR.md` §零
- FALCON 数据配置：`streaming_calibration_exp/configs/data/falcon_{m1,m2,h1}.yaml`
- MUA 程序与解释边界：`docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md:74-75,83-90,98-104`
- 未实现的候选（谐波、Poisson link）：`docs/T4_NEXT_ROUND_IDEATION.md:61-64`

---

## 八、本 handoff 不能声称的内容

- 以上全部为 validation / train-only development evidence，不构成 formal held-out 结论。
- **`k4` 尚未实现，一次都没跑过。** §四 E2 是它的第一次检验；在 E2 出结果前
  不得在任何文档里把"通用载体更好"写成结论。
- **不得声称 "CO 上 `k4 ≈ T4`"。** 这是 v2 写过的错误自检，已由 §2.8 的审计否掉
  （CO 上整 trial 平均速度与 target direction 的中位角误差 ≈41°）。`k4` 与 T4 在 CO 上
  的任何数值差都是**待解释的结果**，不是实现正确性的判据。
- **§2.8 的两行审计数字尚无 artifact**，属待复算证据；E2 复算落盘前不得引用进论文。
- §2.3 的秩 1 假设**仍未被检验**，且已从中心假设降级；不得写成已确立的机制。
- §2.1 的 M1 `−0.002` 目前支持"T4 在该 setting 无效"；**"因为描述子/目标错配"是 §2.1 提出的
  假设，由 E2 的 M1 格子检验**，未验证前不得当结论。"因为接近天花板"要等 E9 的上界估计。
- **§2.1 的跨 setting 比较带一个未量化的估计器分量**（§2.6），E3 完成前不得作为定量律发表。
- §2.7 的 RT 事实是**只读实测**（15/15 session，`target_dir` 唯一值 `0.785398`、
  `num_targets=4`、`cursor_vel` 方向近均匀），可直接引用；但"`k4` 能救 RT"
  仍是假设，待 E2/E4。
- FALCON 的一切本地数字都不是 leaderboard 结果（`evaluation_scope` 已明写）。
