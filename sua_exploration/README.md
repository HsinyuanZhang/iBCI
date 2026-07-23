# SUA/MUA Shared Encoder Exploration

**状态：当前项目主线**  
**最近整理：2026-07-23**

## 研究目标

研究轻量 NeuronID encoder 在两类 intracortical spike signal 之间可以共享到什么程度：

- **MUA**：阈值 crossing 表示的 electrode-level population activity；
- **SUA**：spike sorting 后的 single-unit cluster activity。

本项目不以构建大型 foundation model 为目标，而是回答一个面向部署的问题：在约 18K–35K 参数的 encoder 预算下，哪些结构能够跨信号类型复用，哪些部分必须按信号类型重新训练？

## 三个问题及当前答案

| 问题 | 当前答案 | 证据状态 |
|---|---|---|
| 同一个轻量架构能否处理 SUA 与 MUA？ | **能。** B3 分别训练时都能接近各自 teacher。 | 已有单任务证据 |
| MUA 训练权重能否零样本迁移到 SUA？ | **不能。** 原始迁移 R²≈0.001，z-score 后更差。 | 当前设置下已否定 |
| SUA 是否需要关系型或高阶统计结构？ | **尚不能判断。** B16 在 MUA 单 fold/seed 也略有提升，可能是通用改进。 | 尚需多 fold/seed 与 B15 MUA 对照 |

因此，当前主线不是继续证明“权重天然共享”，而是：

> 先建立严格、可比较的 SUA 基线，再判断 B15/B16 的增益是否稳定、是否为 SUA 特有，最后决定采用按信号类型重训、共享 backbone，还是 context-conditioned 联合训练。

## 已确认与暂定结论

### 已确认

- B3 的 per-neuron EarlyPool 结构不依赖 neuron 数量，能够适配 MUA 和 sorted SUA 的输入形状。
- 直接 MUA→SUA 权重迁移失败；输入尺度不是唯一根因，单纯 z-score 无法修复。
- 对新信号类型重新训练小 encoder 的成本远低于训练大型统一模型，因此“按信号类型重训”是当前可靠部署基线。

### 暂定，不能作为最终结论

- B15 在一次 MC_Maze 内部 validation 比较中高于 teacher task R²。
- B16 可能以较小硬件代价捕获跨 trial 可靠性。
- B16 在 FALCON M2 fold 0 / seed 42 的公平对照中取得 held-out `0.248 ± 0.137`，略高于 B3 的 `0.236 ± 0.102`；4/6 sessions 提升，但单次证据不足以定论。
- B15/B16 的收益可能来自 SUA spike sorting 的 split/merge 或可靠性信息。

这些判断仍缺：SUA 同 teacher 的公平 B3、多 seed、B15 MUA 对照、B16 多 fold/seed，以及跨 session 或外部 SUA 验证。

## 当前实验口径

### SUA

- 数据：DANDI 000128 MC_Maze，sorted units。
- 当前 datamodule 使用一个带行为标签的 train NWB。
- 使用 NWB 内的 `train`/`val` trial split；这不是跨 session LOSO。
- 当前只使用 `heldout == false` 的 units。
- 代码将同一个 validation loader 同时返回为 `val_heldin` 和 `val_heldout`；两者目前不是独立集合。

所以现有 SUA 指标统一称为 **MC_Maze internal validation R²**，不能解释为真正的跨 session held-out 泛化。

### MUA

- 主要基线：FALCON M2，96 个 MUA channels。
- B3-D64 在正确 LOSO fold 0 + held-out session 协议下为 `0.236 ± 0.102`，接近论文 SPINT 的 `0.26 ± 0.13`。
- B16-D64 在完全相同协议下的首个 fold 0 / seed 42 结果为 `0.248 ± 0.137`，相对 B3 绝对提升 `+0.011`；这是候选信号，不是稳定增益结论。
- 该结果用于确认 B3 在 MUA 上的可行性；不能与 MC_Maze internal validation 的绝对 R² 直接横向比较。

## 当前优先级

1. 用当前 MC_Maze teacher 重训公平 B3-v2，并让比较脚本输出可追踪 JSON。
2. 对 B3/B15/B16 使用一致配置运行至少 3 个 seed。
3. 在 FALCON M2 上补 B16 多 fold/seed，并运行 B15 MUA 对照；M1 作为后续 task replication。
4. 建立真正的 SUA 泛化评估，而不继续沿用重复的 `heldin/heldout` loader 命名。
5. 只有在结构增益稳定后，再做 SUA+MUA 联合训练或 context conditioning。

训练与比较脚本已于 2026-07-23 补齐显式 seed、teacher SHA-256 校验、核心超参数一致性检查和 JSON 结果输出。当前公平 B3-v2 尚未运行；本机两张 RTX 3090 与 CUDA 驱动已通过 PyTorch smoke test，项目 `spint` Conda 环境正在按 `SPINT-main/environment.yaml` 创建。

完整验收标准见 [`ROADMAP.md`](ROADMAP.md)。

## 文档导航

| 文档 | 用途 | 权威性 |
|---|---|---|
| [`ROADMAP.md`](ROADMAP.md) | 当前执行顺序、验收标准、停止条件 | 当前计划 |
| [`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md) | 经过口径整理的结果与限制 | 当前结果摘要 |
| [`docs/B16_OPTIMIZATION_BRAINSTORM.md`](docs/B16_OPTIMIZATION_BRAINSTORM.md) | B16 稳定性诊断、候选架构与分阶段实验矩阵 | 当前优化策略 |
| [`docs/RESULTS.md`](docs/RESULTS.md) | 2026-07-22 原始实验叙述 | 实验日志，含不同 teacher |
| [`docs/ARCHITECTURE_ANALYSIS.md`](docs/ARCHITECTURE_ANALYSIS.md) | 实验前架构假设和文献比较 | 背景材料 |
| [`PLAN.md`](PLAN.md) | 最初 SUA/MUA 假设与阶段设计 | 历史计划 |
| [`PAPERS.md`](PAPERS.md) | 论文索引 | 文献导航 |

发生冲突时，以本 README、`ROADMAP.md` 和 `docs/CURRENT_RESULTS.md` 为准。

## 代码与产物

```text
sua_exploration/
├── README.md
├── ROADMAP.md
├── PLAN.md                         # 历史初始计划
├── PAPERS.md
├── data/
│   ├── 000128/                     # MC_Maze
│   └── 000129/                     # MC_RTT
├── mc_maze/datamodule.py
├── scripts/
│   ├── train_teacher_mc_maze.py
│   ├── train_b3_mc_maze.py
│   ├── train_variant_mc_maze.py
│   ├── eval_zero_shot_transfer.py
│   └── compare_neuronid_variants.py
├── checkpoints/
└── docs/
    ├── CURRENT_RESULTS.md
    ├── RESULTS.md
    └── ARCHITECTURE_ANALYSIS.md
```

Encoder 实现在 `streaming_calibration_exp/src/models/components/streaming_encoders.py`：

- B3：`EarlyPoolEncoder`
- B15：`RelationalEarlyPoolEncoder`
- B16：`HighOrderStatsEncoder`
