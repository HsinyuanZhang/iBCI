# SPINT Research Workspace

## 项目定位

最终目标是一颗**可重构的 intracortical 运动解码芯片**：NeuronID encoder
（使网络不依赖固定 unit id）+ cross-attention decoder。论文主张是**芯片架构
的可重构能力可跨应用复用，性能不是核心诉求**。

这个定位决定了所有实验的取舍标准——评估任何改动时先问它落在哪个**速率域**：
session-rate 的复杂度近乎免费，per-window（50 Hz）的复杂度和随 `N` 变化的
shape 才是真代价。详见
[`sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md`](sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md)。

## 当前主线

**SUA/MUA 共享 NeuronID 编码器探索**，入口
[`sua_exploration/README.md`](sua_exploration/README.md)。

建议阅读顺序：

1. [`sua_exploration/README.md`](sua_exploration/README.md) — 当前状态与文档导航
2. [`sua_exploration/ROADMAP.md`](sua_exploration/ROADMAP.md) — 实验计划与决策门
3. [`sua_exploration/docs/CURRENT_RESULTS.md`](sua_exploration/docs/CURRENT_RESULTS.md) — 结果与口径
4. [`sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md`](sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md) — **测量协议（引用任何数字前必读）**
5. [`sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md`](sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md) — 速率域与可重构主张

## 当前结论（2026-07-26）

- B3 EarlyPool 架构在 MUA 和 SUA 上分别训练都有效，**架构可复用**。
- MUA 训练的 B3 权重零样本迁移到 SUA 失败，**权重不可直接复用**。
- **calibration identity 是承重部件**：置零后 validation R² 全部转为强负值。
  这是目前唯一效应量远高于噪声底的结构性结论。
- **跨 neuron self-attention 是否有效：未知。** 此前的阴性结论已于 2026-07-25
  撤回（测量不可靠）。
- **per-unit 波形/SNR 侧信息：`indeterminate`。** 4 个主配对全部为负，但
  未越过噪声底，不得写成阴性结论。

## ⚠️ 测量可靠性（本项目吃过两次亏）

本项目**两次**把门槛设在噪声底之下，导致结论不可用：

| 轮次 | 门槛 | 实测噪声底 | 后果 |
|---|---|---|---|
| `attention_arch_screen_v3` | `+0.005` | `σ_epoch = 0.0388` | 结论撤回 |
| `side_feature_ablation_v2` | `+0.03` | `2σ_delta = 0.048–0.077` | 全部 `indeterminate` |

**引用任何 R² 差值前，先读
[`MEASUREMENT_PROTOCOL_V4.md`](sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md)
的 §4.1 修订框，确认该数字是否越过了当轮实测的 `σ_delta`。**
`indeterminate` ≠ 阴性。

## 目录结构

| 路径 | 定位 | 当前关系 |
|---|---|---|
| `sua_exploration/` | SUA/MUA 共享编码器 | **当前主线** |
| `streaming_calibration_exp/` | MUA 轻量 encoder 实验框架 | B3/B15/B16 的共享实现基础 |
| `SPINT-main/` | 原始 SPINT 实现与数据接口 | teacher 和模型语义来源 |
| `software-to-hardware/` | B3 INT8/QAT 与硬件交接 | 部署支线 |
| `planB_tempconv/` | 低成本时间卷积 decoder | decoder 压缩支线 |
| `docs_archive/` | 2026-07 上旬的历史分析与旧计划 | **仅作历史记录，不是当前优先级来源** |
| `sua_exploration/papers/` | 文献 PDF（7 篇） | 索引见 `sua_exploration/PAPERS.md` |

## 工作区注意事项

- 顶层仓库通过 GitHub `origin/main` 同步源码、配置、测试和研究文档。原始数据、
  checkpoint、缓存、日志及可再生实验结果不进入 Git；关键实验仍应在文档中记录
  配置、结果文件路径与内容哈希，确保本地产物可以核验。
- Conda 环境 `spint`（torch 2.5.1 + CUDA），2× RTX 3090。
- 数据：`sua_exploration/data/dandi_000688/`（sub-C 53 个 CO session 已下载）。
- **`sub-C/CO/27-6-6` 的 formal-test scope 已被 receipt 占用且状态悬空**，
  处置方案见 [`sua_exploration/ROADMAP.md`](sua_exploration/ROADMAP.md) G1。
  在此之前所有实验只能产出 validation development evidence。
