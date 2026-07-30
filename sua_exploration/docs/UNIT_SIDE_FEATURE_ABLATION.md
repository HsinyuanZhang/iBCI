# Per-Unit 侧信息消融章程

**状态：章程已执行完毕（2026-07-26）。结果 `F1/F2 = indeterminate`，见
[`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §I。本文保留为该 screen 的
预注册章程。**
**数据隔离：validation-only。本消融在任何结果下都不得产出 formal held-out test。**

本文固定"给 NeuronID encoder 增加 per-unit 侧信息"的可证伪假设、数据事实、
接入点、成本论证、变体矩阵和预注册门槛。侧信息指**不随 trial 变化的
per-unit 属性**：波形形状、幅度/SNR、electrode 归属。

## 1. 要回答的主张

现有 identity `E_i` 只来自 calibration 期的 spike count 统计。假设是：
sorted SUA 携带了一些 firing statistics 之外的 per-unit 信息，把它们喂给
`ψ` 可以提升跨 session identity 的可用性。

要支持这条主张，至少需要：

1. 在固定的前向 calibration 协议下，侧信息变体在 validation sessions 上
   相对同 seed 的 B3 有稳定正向 paired delta；
2. 增益不能由 `ψ` 输入维度变宽本身解释（需要 shuffled-feature 对照）；
3. 增益在部署上是可付的（侧信息必须只在 calibration 期读取一次）。

## 2. 一个必须先纠正的直觉

早期讨论把波形对 identity 的价值排得过高。需要记录的反驳是：

- `E_i` 不是用来做**身份追踪**的。它被加到 `Z` 上、喂给 cross-attention，
  决定"这个 unit 该被赋予多少权重来解码速度"。它需要编码的是**功能属性
  （tuning）**，不是解剖属性。
- 波形与运动 tuning 的相关性很弱，已知的主要只有 narrow/broad-spiking
  这个粗二分。
- 存在循环问题：波形正是 sorter 用来定义这个 unit 的东西。session 内它
  当然高度指示 unit 身份，但那是平凡的，没有给 decoder 任何 firing
  statistics 之外的**功能**信息。

因此本消融把波形排在幅度/SNR **之后**，并预期整体增益可能为零。零结果
同样是有价值的记录，因为它会关闭一条看起来很自然的架构分支。

## 3. 数据事实（2026-07-25 用 h5py 实测）

实测文件：`sub-C_ses-CO-20151103`（validation）与 `sub-C_ses-CO-20131003`（train）。

> **⚠️ 数据隔离违规记录（2026-07-26 自查发现并纠正）**
>
> 本节初版的发放率表包含 `sub-C_ses-CO-20151119`，而它是 **6 个 held-out
> test session 之一**。撰写者（Claude）为做第 4 节的成本论证读取了该 session
> 的 `spike_times`、`waveforms` 与 trials，超出了「只允许读 test NWB 的
> unit-table 行数」这一许可范围。
>
> **影响评估**：该数据仅用于第 4 节的 spikes/s 量级估算，**未进入任何训练、
> 模型选择、评估或门槛设定**；已独立核实共享缓存
> `sua_exploration/cache/dandi688_subc_co_v1/` 中**不存在任何 test session
> 派生的条目**（对 6 个 test session × 4 个 feature group × 2 个 pool_size
> 枚举缓存键，命中 0 条）。因此科学结论不受影响，但**治理层面确属违规**。
>
> **纠正**：该行已替换为 train session `sub-C_ses-CO-20131003`（71 units，
> 与原行同量级），第 4 节的结论量级不变。此记录保留，不删除。

| 字段 | 实际情况 |
|---|---|
| `/units/waveforms` | `(23645280, 1)` int16，配 `waveforms_index` → **逐 spike 波形，48 samples/spike** |
| `waveform_mean` | **不存在**，per-unit template 必须自行离线平均 |
| `/units/electrodes` | 每 unit 对应的 electrode index，存在 |
| `/general/extracellular_ephys/electrodes` 列 | `bank, filtering, group, group_name, id, label, location, pin` |
| **电极坐标** | **不存在**。无 `x`/`y`/`z`/`rel_x`/`rel_y` |

实测发放率：

| Session | units | spikes | span | spikes/s | Hz/unit |
|---|---:|---:|---:|---:|---:|
| sub-C CO-20151103 (val) | 38 | 492,610 | 1,201 s | 410 | 10.8 |
| sub-C CO-20131003 (train) | 71 | 319,409 | 663 s | 482 | 6.8 |

### 3.1 这否掉了"先加电极坐标"这条路径

早期讨论把电极坐标排在第一位（零计算成本、MUA 模式同样存在、无模态
不一致）。**该数据集没有坐标。** Utah 阵列的 pin 编号不是空间单调的，
没有该实验室具体的 Blackrock `.cmp` array map 就还原不出拓扑位置。

因此本章程**不**包含坐标特征。若后续取得 array map，可作为独立追加。

### 3.2 但 electrode index 有一个未被讨论过的性质

对同一块植入阵列，**electrode index 是跨 session 稳定的，而 sorted unit
id 不是**。这正面指向本项目最难的跨 session 问题，因此它作为独立特征组
`F3` 进入消融——但检验的是"稳定物理索引"假设，不是"拓扑邻近"假设。

必须同时记录的反向风险：神经漂移意味着同一电极跨月/跨年并不记录同一
神经元，`F3` 很可能无效甚至有害。它排在最后。

## 4. 成本论证（修正版）

早期讨论认为逐 spike 波形编码会"压垮整个芯片"（估计 ~15 GMAC/s）。
按第 3 节实测发放率，该估计**对 sorted SUA 不成立**：

| 方案 | 计算量 | 相对实时路径（`N=64` 时 921 MMAC/s） |
|---|---:|---:|
| 逐 spike 编码 `48→32`（482 spikes/s） | ≈ 0.74 MMAC/s | ≈ 0.08% |
| per-unit template `48→32`，每 session 一次 | `N×1536` MAC/session，摊薄后 < 0.0001 MMAC/s | 可忽略 |

15 GMAC/s 那个数字对应 **10k events/s**，那是 96 通道 MUA threshold-crossing
的事件率，不是 ~40 个 sorted unit 的发放率（本数据约 410–482 spikes/s）。

**因此"必须喂 template 而非逐 spike"这条结论仍然成立，但理由不是算力：**

1. template 是 per-unit 常量，逐 spike 编码在信息上冗余——pooling 之后
   得到的本就是均值；
2. 逐 spike 引入**数据相关、事件驱动的负载**，破坏实时路径的确定性时序，
   对闭环延迟保证是负担；
3. identity 每 session 只需要一次，逐 spike 等于把 session 路径的工作
   搬进 event-rate 域，与 [`ASIC_DEPLOYMENT_CHARTER.md`](ASIC_DEPLOYMENT_CHARTER.md)
   第 3 节的不对称性论证直接冲突；
4. MUA 模式下不存在波形，逐 spike 分支会造成更严重的模态不一致。

侧信息本身的增量成本：`k` 个标量 concat 到 `ψ` 输入端，增加 `k×D` MAC
per unit per session。`k=6, D=64, N=64` 时为 24,576 MAC/session，相对 B3 的
`1.30e7 MAC/session` 约 **+0.19%**，实时路径**零增量**。

## 5. 接入点

`streaming_calibration_exp/src/models/components/streaming_encoders.py`
的 `EarlyPoolEncoder`（B3）即 SPINT 的 φ/pool/ψ：

- `pre_pool` (φ) = `Linear(trial_length=100 → D=64) + ReLU`，**逐 trial** 应用
- pool = 对 M 个 calibration trial 求均值 → `mean_feat[B,N,D]`
- `post_pool` (ψ) = affine stack `D → … → W=50`

**侧信息必须 concat 到 `mean_feat`、送进 `post_pool` 之前**，即
`finalize_identity()` 中 `return self.post_pool(...)` 那一行
（约 `streaming_encoders.py:332`）。

接在 φ 上是错的：φ 是 trial-wise 的，把不随 trial 变的量塞进去会被重复
`M` 次再被 pooling 平均掉，纯粹浪费。接在 ψ 输入端只需把 `post_pool`
第一层 `Linear` 从 `D→D` 改成 `(D+k)→D`——没有新 PE、没有新 dataflow，
只是一个更宽的 GEMM，且完全落在 session 路径。

`B15`/`B15D`/`B15P`/`B16` 继承同一 `finalize_identity` 模式，改法一致。
本轮只在 **B3 底座**上做，避免与 attention 变量混淆。

## 6. 特征组定义

所有特征在 **calibration pool 内的 spikes** 上计算，不使用 evaluation
trials、不使用行为标签。

| 组 | 内容 | 维度 |
|---|---|---:|
| `F0` | 无侧信息（= B3 基线） | 0 |
| `F1` | mean template 的 peak-to-peak 幅度、噪声 std、SNR | 3 |
| `F2` | `F1` + 峰谷宽度、峰谷比、复极斜率 | 6 |
| `FS1` | `F1` 的**对照**：在 unit 维度随机置换的同一批特征 | 3 |
| `FS2` | `F2` 的**对照**：在 unit 维度随机置换的同一批特征 | 6 |
| `F3` | `F2` + electrode index 的 learned embedding（第二阶段） | 6 + `e` |

置换对照是必须的。它保证 `ψ` 输入变宽、参数变多这件事本身被控制住；
没有它，`F2 > F0` 无法归因于特征内容。这与 `B15P` 对 `B15` 的作用是同一类控制。

> **2026-07-25 修订**：初版只定义了一个 6 维的 `FS`，导致 `F1 − FS` 在比较
> `side_dim=3` 与 `side_dim=6` 两个**不同架构**（`post_pool[0]` 的 fan_in
> 分别为 67 与 70，且 RNG 流从第一层起分叉）。这是初版章程的缺陷，不是
> 实现问题。现拆分为维度匹配的 `FS1` / `FS2`：每个特征组的内容门槛只与
> **同维度**的置换对照比较。

### 6.1 归一化契约

所有连续特征使用**仅由 train sessions 估计**的均值/标准差做 z-score。
统计量随 checkpoint 一起保存。**禁止**使用 validation 或 test session 的
统计量，也禁止 per-session 重新标准化——后者会把跨 session 的尺度漂移
偷偷消掉，而那正是被研究的对象。

### 6.2 模态一致性

MUA 没有波形。本轮**只做 SUA**，不对 MUA 下任何结论。若 `F1`/`F2` 通过
门槛，MUA 侧需要单独决策（modality dropout 训练，或两套 encoder 权重），
届时再写章程。`F1`/`F2` 只有 3–6 个标量，屏蔽它们比屏蔽一整个 48 维
template 分支温和得多，这也是本章程选择标量而非学习式波形编码器的理由
之一。

## 7. 数据隔离

- 数据：DANDI 000688、sub-C、CO，仅旧 unit-count regime（`N < 100`）；
- split：固定 chronological `27 train / 6 validation / 6 test`；
- 训练只读 train sessions；模型选择与全部 R² 只读 6 个 validation sessions；
- **test sessions 的 neural、behavior、trial data 一律不加载**；
- 固定前向评价协议：前 50 个 rewarded trials 为 pool，按 chronologic
  `first` 选 30 个 calibration trials，只评价 pool 之后的 trials；
- 评价时 encoder 与 decoder 全部冻结，不用 validation 行为标签更新任何参数。

### 7.1 formal test 已被占用

`sub-C/CO/27-6-6` 这个 formal-test scope 已有 receipt
`sua_exploration/results/p3_formal_test_816cdd8bf9f26abd1a3e6251e5fbf8537eb6c6cb4de1e8f3312980ddbf478379_receipt.json`。
**本消融在任何结果下都不得创建、覆盖或重跑该 scope 的 formal test。**
其全部证据严格限定为 validation development evidence。

若需要确认性证据，必须换未参与本次结构选择的数据：`sub-M`（external
subject）或 `sub-J`。

## 8. 预注册门槛

> **2026-07-25 修订**：初版沿用了 `attention_arch_screen_v3` 的 `+0.005`
> 门槛与 best-checkpoint 口径。诊断（[`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §H）
> 显示该口径的噪声底为 `σ = 0.0388`，`+0.005` 比噪声低约 8 倍，
> **不可能被有意义地通过**。本节改用
> [`MEASUREMENT_PROTOCOL_V4.md`](MEASUREMENT_PROTOCOL_V4.md)。

估计量、不确定度报告与三态判定全部遵循 V4：固定 `E=12` epoch、无 early
stopping、唯一 run 目录、变体分数取 epoch 5–12 的协议指标平均、seeds 42/43/44。

| Gate | 条件 |
|---|---|
| 变体可用 | 变体分数 > 0 |
| **内容有效** | `F_x − F0` 与 `F_x − FS_x`（**同维度**置换对照）的 mean delta 均 ≥ `+0.03`，且 6 个 session 中至少 5 个为正，且 3 个 seed 的逐 seed 均值全部为正 |
| **内容无效** | 上式不成立**且** `abs(mean delta) > 2·σ_delta` |
| **不确定** | 以上都不满足 |

`σ_delta` 必须从本轮数据实测，不得沿用先验估计。**"不确定"不得写成阴性
结论**——这正是 V3 犯的错。汇总前不因中间结果增删 seed、特征组或阈值。

## 9. 筛选后的决策树

| 结果模式 | 解释 | 动作 |
|---|---|---|
| `F1` 即通过，`F2` 不再增益 | 幅度/SNR 携带了可靠性信息 | 只保留 3 个标量，进入 MUA 模态决策 |
| `F2` 显著优于 `F1` | 波形形状确有功能信息 | 保留 6 标量，但需 replication |
| `F_x ≈ FS_x` 且判定为**无效** | 增益来自 `ψ` 变宽，不是特征内容 | 关闭侧信息路线，记录阴性结果 |
| 判定为**不确定** | 效应量落在噪声内，证据不足 | **不得记为阴性**；按 V4 §4.3 记为 indeterminate，决定是否加 seed |
| 全部判定为无效 | 侧信息在当前预算下无价值 | 关闭该分支，模态一致性问题随之消失 |

`F3`（electrode embedding）推迟到 `F1`/`F2` 有结论后再立项。

## 10. 本消融不能声称的内容

- validation 筛选结果不是 formal held-out test；
- SUA 上的结论不能外推到 MUA（本轮不含 MUA 臂）；
- 波形有效不等于跨 session 神经元身份被追踪到；
- 单个 subject（sub-C）的结果不构成 replication；
- 侧信息的部署成本论证只覆盖 MAC，未覆盖 sorter 到 encoder 的互连与
  地址映射代价——那是 RTL 侧必须提前预留的通路，见
  [`ASIC_DEPLOYMENT_CHARTER.md`](ASIC_DEPLOYMENT_CHARTER.md) 第 6 节。

## 11. 前置条件

按依赖顺序，三项都必须先完成：

1. **测量修复 M1–M3**（[`../ROADMAP.md`](../ROADMAP.md) M 段）。在 best-checkpoint
   口径下跑本消融只会产出噪声——这是 V3 的教训。
2. **`attention_arch_screen_v4` 先跑完**。它用同一套 V4 估计量、同一批
   validation session，可以顺带实测 `σ_delta`，为本消融提供分辨力标定；
   而且它的 B3 臂可直接复用为本消融的 `F0`（须校验超参数完全一致）。
3. **`learned_prior` 无标定对照修好**（[`HANDOFF_SIDE_FEATURES.md`](HANDOFF_SIDE_FEATURES.md)
   Task A）。当前它退化成 zero-identity，缺少可信的下界参照。

运行矩阵随 `FS1`/`FS2` 拆分与 3 seeds 更新为 **5 组 × 3 seeds = 15 runs**
（`F0/F1/F2/FS1/FS2` × seeds 42/43/44）。若复用 v4 的 B3 臂作 `F0`，则为 12 runs。
