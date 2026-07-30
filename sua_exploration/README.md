# SUA/MUA Shared Encoder Exploration

**状态：当前项目主线**  
**最近整理：2026-07-30**

> **2026-07-25 重要更新（含一次结论撤回）**
>
> 逐 epoch 曲线诊断发现当前测量的噪声底（SUA `σ=0.0388`）是被测效应的
> 6 倍，且存在不均等的 checkpoint 选择偏置和 run 目录冲突。因此
> **此前"attention 已被否定"的结论已撤回**——attention 是否有效当前是
> **未知**。诊断见 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §H，
> 修复计划见 [`ROADMAP.md`](ROADMAP.md) M1–M5。**在测量修好前不要跑新实验。**
>
> 本项目的部署定位（可重构芯片，性能非核心诉求）现记录于
> [`docs/ASIC_DEPLOYMENT_CHARTER.md`](docs/ASIC_DEPLOYMENT_CHARTER.md)。

## 研究目标

研究轻量 NeuronID encoder 在两类 intracortical spike signal 之间可以共享到什么程度：

- **MUA**：阈值 crossing 表示的 electrode-level population activity；
- **SUA**：spike sorting 后的 single-unit cluster activity。

本项目不以构建大型 foundation model 为目标，而是回答一个面向部署的问题：在约 18K–35K 参数的 encoder 预算下，哪些结构能够跨信号类型复用，哪些部分必须按信号类型重新训练？

## 关键问题及当前答案

| 问题 | 当前答案 | 证据状态 |
|---|---|---|
| 同一个轻量架构能否处理 SUA 与 MUA？ | **能。** B3 分别训练时都能接近各自 teacher。 | 已有单任务证据 |
| MUA 训练权重能否零样本迁移到 SUA？ | **不能。** 原始迁移 R²≈0.001，z-score 后更差。 | 当前设置下已否定 |
| calibration identity 是必需的吗？ | **是。** 置零后 validation R² 全部转为强负值（B3 `−0.277`、B16 `−0.515`、B15 `−2.298`）。 | 已确认的大效应结构性结论 |
| SUA 是否需要关系型结构（跨 neuron attention）？ | **未知。** 此前的阴性结论已撤回——测量不可靠（噪声底是效应的 6 倍）。 | 2026-07-25 撤回，待重测 |
| per-unit 波形/SNR 侧信息是否有帮助？ | **未定（`indeterminate`）。** 4 个主配对全部为负，但均未越过噪声底。 | 2026-07-26，不得写成阴性 |
| 标定期方向调谐特征是否有帮助？ | **有。** SUA T4 `+0.2528`、T8 `+0.2567`，正确内容对照 6/6 session、3/3 seed 全正。 | 项目首个稳定 `effective` |
| T4 增益能否跨到 pseudo-MUA？ | **能。** pseudo-MUA 上 T4−F0 `+0.3177`、T4−TS4 `+0.3657`，均为 `effective`。 | 9/9 run 完成；validation evidence |
| T4 在原生 FALCON MUA held-out session 上是否有效？ | **任务相关。** local held-out-calib replay 中，M2 `T4−F0=+0.06979`、`T4−TS4=+0.06201`，M1 两项均略为负。 | 同一 SPINT/B3 local 路径；非隐藏 EvalAI test |
| SUA 是否明显优于 pseudo-MUA？ | **有小幅 residual advantage，不是很大。** T4 下 SUA−pseudo-MUA `+0.0406 ± 0.0153 SE`；无 T4 时为 `+0.1056`。 | 同 session/view 的事后配对诊断 |
| 静态 electrode reliability gate 能否进一步改善 T4？ | **不能。** T4GATE−T4 `−0.0108 ± 0.0049 SE`。 | `ineffective`，停止该方向 |

因此，当前主线已经从“修测量”推进到：

> **保留 B3/B3T 作为部署骨架，把 T4 作为当前最强功能 identity 信号；先拆清
> T4 的有效成分，再优化 T4–activity 融合与 SUA/pseudo-MUA 共享。** E1/E2 已
> 证明延长训练和 SWA 不能解决跨 seed 主问题；E3 已给出大效应，pseudo-MUA
> bridge 已证明该效应不依赖 sorted-unit view。下一轮不得继续堆静态 electrode
> lookup，也不得在分量归因前把 `[a,c,m,b]` 的全部收益简称为“方向”。

## 已确认与暂定结论

### 已确认

- B3 的 per-neuron EarlyPool 结构不依赖 neuron 数量，能够适配 MUA 和 sorted SUA 的输入形状。
- 直接 MUA→SUA 权重迁移失败；输入尺度不是唯一根因，单纯 z-score 无法修复。
- 对新信号类型重新训练小 encoder 的成本远低于训练大型统一模型，因此“按信号类型重训”是当前可靠部署基线。
- T4/T8 的正确 per-unit 功能内容在 SUA 上带来约 `+0.25 R²`，并把
  `σ_seed` 约减半；完整置换对照回到基线。
- T4 在 pseudo-MUA 上同样 `effective`；electrode pooling 不会消除其价值。
- T4 下 SUA 绝对分数仍高于 pseudo-MUA，但只高 `+0.0406 R²`，不是数量级优势。
- 静态 per-electrode scalar gate 在 T4 上已被排除为 `ineffective`。

### ⚠️ 以下历史数字**全部低于后来实测的噪声底**，不可引用

MC_Maze 上 B15 `0.90977` > B16 `0.90848` > B3 `0.90781`（差值
`+0.00196`/`+0.00068`）、B16 在 M2 单 fold 的 `+0.011`——这些差值比
实测的 `2σ_delta`（`0.048–0.077`）小一到两个数量级，**没有信息量**。

它们产生于 2026-07-25 测量可靠性诊断之前。判断任何 R² 差值是否可引用的
标准见 [`docs/MEASUREMENT_PROTOCOL_V4.md`](docs/MEASUREMENT_PROTOCOL_V4.md) §4.1。

`B16` 的高阶统计路线因此处于**未测试**状态，不是"已试过收益很小"——
若要推进必须在足够 seed 下重做。

## 当前实验口径

### SUA（当前主线）

- 数据：**DANDI 000688 sub-C，CO 任务**，旧 unit-count regime（`N < 100`）39 个 session。
- Split：chronological `27 train / 6 validation / 6 test`（session-disjoint）。
- 评价协议：固定前向 calibration，`first / n=30 / pool=50`，encoder 与 decoder 全程冻结。
- 估计量：`E=12` epoch、无 early stopping、变体分数取 **epoch 5–12 的协议指标平均**
  （见 [`docs/MEASUREMENT_PROTOCOL_V4.md`](docs/MEASUREMENT_PROTOCOL_V4.md)）。
- **所有数字都是 validation development evidence**；6 个 test session 从未被加载。

### SUA（历史，MC_Maze 单 session）

DANDI 000128 的单 session internal validation，非跨 session。该口径下的历史
排名差值均低于噪声底，已不作为结论来源。

### pseudo-MUA bridge（当前受控跨 view 证据）

- 构造：同一 session、同一 electrode 上的 sorted SUA spike counts 逐 bin 求和；
- T4：聚合后按 electrode 重新拟合，不平均 unit-level T4；
- 结果：F0/T4/TS4 为 `0.2084/0.5261/0.1604`，T4 的两项主要配对均为
  6/6 session、3/3 seed 全正；
- 边界：这是同数据的受控 signal-view bridge，不是真实 threshold-crossing MUA；
- 详情：[`docs/PSEUDO_MUA_T4_BRIDGE_48H.md`](docs/PSEUDO_MUA_T4_BRIDGE_48H.md)
  与 [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) §K。

### MUA

原生 FALCON M1/M2 已完成 matched F0/T4/TS4 internal-LOSO 训练，并将同一批
冻结 checkpoint test-only replay 到 repository 现有的 local
`held-out-calib` 路径。M1 使用 chronological first 10，M2 first 33；实际
calibration 无 optimizer/backward。local replay 的结果为：

| Task | F0 | T4 | TS4 | T4−F0 | T4−TS4 |
|---|---:|---:|---:|---:|---:|
| M1 | .62725 | .62505 | .62879 | −.00221 | −.00374 |
| M2 | .22160 | .29139 | .22938 | +.06979 | +.06201 |

F0 是无 target-label calibration 基线；T4/TS4 都使用 calibration target
labels，TS4 只打乱 channel–T4 对应。因此 `T4−F0` 不是 label-information
matched，`T4−TS4` 才回答正确调谐内容是否有用。M2 两项均为 3/3 cells
正，M1 不支持增益。该 replay 使用本地 calibration NWB 中可见的 neural 与
behaviour arrays，**不是隐藏 EvalAI query/test-set 结果**，也不能与 SUA 的
绝对 R² 横向比较。

MUA 侧的 attention 对照同样受 2026-07-25 诊断影响（run 目录冲突 + 选择偏置），
结论已撤回。

## 当前优先级（2026-07-29）

完整计划见 [`ROADMAP.md`](ROADMAP.md) 的「当前实验计划」，章程见
[`docs/E3_E4_ENCODER_PROGRAM.md`](docs/E3_E4_ENCODER_PROGRAM.md)。

| # | 实验 | 状态 |
|---|---|---|
| **E1** | SWA/EMA 权重平均 | ✅ 完成；`+0.022`，不降 `σ_seed`，非主线贡献 |
| **E2** | 收敛性长 run | ✅ 完成；长训练无益并放大 seed 分歧，沿用 `E=12` |
| **E3** | 方向调谐特征 T4/T8 | ✅ 完成；项目首个 `effective` |
| **E4** | 时间基 `B3T` / trial attention `B3A` | ✅ 完成；B3T 六 seed `+0.0418`，B3A 放弃 |
| **P-MUA** | pseudo-MUA T4 bridge | ✅ 9/9 完成；T4 组级 `effective` |
| **T4GATE** | T4 + 静态 electrode reliability gate | ⛔ `ineffective`；不继续 |
| **N-MUA** | 原生 M1/M2 internal + local held-out-calib replay | ✅ 完成；M2 正向，M1 不支持跨任务泛化 |
| **SUA-REL0** | 同电极 relation | ⛔ `ineffective`；不进入 relative amplitude |
| **T4-NEXT** | T4 分量归因、T4×B3T、置信度融合、跨 view 一致性 | 下一轮候选，须先冻结章程 |
| — | fixed-slot router 的同 `K` top-K / random / activity 对照 | 排队 |
| — | **（需用户决策）** formal-test receipt 悬空，见 [`ROADMAP.md`](ROADMAP.md) G1 | 阻塞 |

下一轮的第一原则不是继续增加“身份字段”，而是先回答 T4 的 `a/c`、`m`、`b`
分别贡献多少。T4 含 `[a,c,m,b]`，其中 `m=sqrt(a²+c²)` 冗余、`b` 是 baseline
rate；在完成分量消融前，“方向调谐带来全部 `+0.25`”仍是过强表述。

本机两张 RTX 3090、CUDA 与 `spint` Conda 环境均已验证可运行。

完整验收标准见 [`ROADMAP.md`](ROADMAP.md)。

## 文档导航

| 文档 | 用途 | 权威性 |
|---|---|---|
| [`ROADMAP.md`](ROADMAP.md) | 当前执行顺序、验收标准、停止条件 | 当前计划 |
| [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) | 经过口径整理的结果与限制 | 当前结果摘要 |
| [`docs/MEASUREMENT_PROTOCOL_V4.md`](docs/MEASUREMENT_PROTOCOL_V4.md) | 估计量、不确定度、可达门槛、三态判定 | **当前测量协议** |
| [`docs/ASIC_DEPLOYMENT_CHARTER.md`](docs/ASIC_DEPLOYMENT_CHARTER.md) | 速率域划分、可重构主张、实验取舍标准 | 定位文档 |
| [`docs/ATTENTION_ARCHITECTURE_SCREEN.md`](docs/ATTENTION_ARCHITECTURE_SCREEN.md) | attention 参数匹配对照章程 | **结论已撤回，待用 V4 重测** |
| [`docs/FIXED_SLOT_ROUTER_PILOT.md`](docs/FIXED_SLOT_ROUTER_PILOT.md) | 固定 token 接口 pilot | 部署支线 |
| [`docs/UNIT_SIDE_FEATURE_ABLATION.md`](docs/UNIT_SIDE_FEATURE_ABLATION.md) | per-unit 侧信息消融章程 | 已执行，结果 `indeterminate` |
| [`docs/E3_E4_ENCODER_PROGRAM.md`](docs/E3_E4_ENCODER_PROGRAM.md) | 调谐特征与 encoder 架构变体章程 | 已执行的冻结章程 |
| [`docs/PSEUDO_MUA_T4_BRIDGE_48H.md`](docs/PSEUDO_MUA_T4_BRIDGE_48H.md) | SUA→pseudo-MUA T4 bridge 执行与结果 | **完成，当前跨 view 证据** |
| [`docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md`](docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md) | 原生 FALCON M1/M2 T4 协议、internal 与 local held-out-calib 结果 | **完成，M2-only 正向证据** |
| [`docs/T4_OPTIMIZATION_DIRECTIONS.md`](docs/T4_OPTIMIZATION_DIRECTIONS.md) | T4 分量归因、网络优化与两周 pilot | **下一主线候选，尚未启动新实验** |
| [`docs/ELECTRODE_ANCHOR_DESIGNS.md`](docs/ELECTRODE_ANCHOR_DESIGNS.md) | T4 上的 electrode gate/anchor 设计 | D 已判无效；C/A 未运行 |
| [`docs/SIDE_FEATURE_DRIFT_DIAGNOSTIC.md`](docs/SIDE_FEATURE_DRIFT_DIAGNOSTIC.md) | 侧特征跨 session 漂移诊断（假设被证伪） | 诊断记录 |
| [`docs/HANDOFF_SIDE_FEATURES.md`](docs/HANDOFF_SIDE_FEATURES.md) | 侧信息实施交接说明 | 已完成，历史记录 |
| [`docs/MULTISESSION_SUA_DATASETS.md`](docs/MULTISESSION_SUA_DATASETS.md) | 多 session SUA 数据集网络审计、信号类型排查与推荐 split | 2026-07-23 数据决策 |
| [`docs/B16_OPTIMIZATION_BRAINSTORM.md`](docs/B16_OPTIMIZATION_BRAINSTORM.md) | B16 候选架构与实验矩阵 | 背景材料（B16 现为未测试状态） |
| [`docs/RESULTS.md`](docs/RESULTS.md) | 2026-07-22 原始实验叙述 | 实验日志，含不同 teacher |
| [`docs/ARCHITECTURE_ANALYSIS.md`](docs/ARCHITECTURE_ANALYSIS.md) | 实验前架构假设和文献比较 | 背景材料 |
| [`PLAN.md`](PLAN.md) | 最初 SUA/MUA 假设与阶段设计 | 历史计划 |
| [`PAPERS.md`](PAPERS.md) | 论文索引 | 文献导航 |

发生冲突时，以本 README、`ROADMAP.md` 和 `docs/CURRENT_RESULTS.md` 为准。

## 代码与产物

```text
sua_exploration/
├── README.md / ROADMAP.md / PLAN.md / PAPERS.md
├── data/dandi_000688/sub-C/        # 53 CO sessions（当前主数据）
├── data/000128/ 000129/            # MC_Maze / MC_RTT（历史）
├── mc_maze/
│   ├── multisession_datamodule.py  # DANDI688 跨 session（当前主线）
│   ├── single_session_datamodule.py
│   ├── unit_side_features.py       # per-unit 侧信息 + 缓存
│   └── datamodule.py               # MC_Maze 单 session
├── scripts/
│   ├── train_variant_dandi688.py           # 训练入口（含 M2/M3 开关）
│   ├── eval_epoch_window_dandi688.py       # M3 估计量（epoch 5-12 平均）
│   ├── swa_utils_dandi688.py               # E1 权重平均
│   ├── select_gradient_free_protocol_dandi688.py  # 协议指标实现
│   ├── aggregate_*.py                      # 各 screen 的三态判定聚合器
│   └── run_*.sh                            # 各 screen 的启动脚本
├── results/                        # 各 screen 的 JSON 产物与 aggregate.json
├── checkpoints/
└── docs/                           # 见上方文档导航
```

Encoder 实现在 `streaming_calibration_exp/src/models/components/streaming_encoders.py`：

- B3：`EarlyPoolEncoder`（部署基线）
- B3S：`SideFeatureEarlyPoolEncoder`（ψ 输入端接 per-unit 侧信息）
- B15/B15D/B15P：`Relational` / `DiagonalRelational` / `PerNeuronResidual`（neuron 轴，结论已撤回）
- B16：`HighOrderStatsEncoder`（跨 trial 高阶统计，未测试状态）
