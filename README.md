# SPINT Research Workspace

## 当前主线

本项目当前聚焦于 **SUA/MUA 共享 NeuronID 编码器探索**：研究 sorted single-unit activity（SUA）与 multi-unit threshold crossings（MUA）能否共享轻量身份编码器的架构、训练方法或权重，并评估关系型结构与高阶统计对 SUA 的价值。

主线入口：[`sua_exploration/README.md`](sua_exploration/README.md)

建议阅读顺序：

1. [`sua_exploration/README.md`](sua_exploration/README.md)：当前状态、已确认结论和文档导航。
2. [`sua_exploration/ROADMAP.md`](sua_exploration/ROADMAP.md)：下一步实验顺序与决策门。
3. [`sua_exploration/docs/CURRENT_RESULTS.md`](sua_exploration/docs/CURRENT_RESULTS.md)：经过口径整理的当前结果。
4. [`sua_exploration/docs/ARCHITECTURE_ANALYSIS.md`](sua_exploration/docs/ARCHITECTURE_ANALYSIS.md)：SPINT、NDT2、UniBCI 的架构背景。
5. [`sua_exploration/PAPERS.md`](sua_exploration/PAPERS.md)：文献索引。

## 当前结论

- B3 EarlyPool 架构分别在 MUA 和 SUA 上训练都有效，说明**架构可复用**。
- MUA 训练好的 B3 权重零样本迁移到 SUA 基本失败，说明**权重不可直接复用**。
- B15（跨神经元关系）和 B16（跨 trial 高阶统计）在 MC_Maze 内部 validation 上表现出潜力；B16 在首个公平 M2 MUA fold/seed 上也略高于 B3，但仍缺多 fold/seed、B15 MUA 对照和真正跨 session SUA 验证。
- 当前最高优先级不是继续扩展模型，而是先关闭上述证据缺口。

## 其他目录的定位

| 路径 | 定位 | 当前关系 |
|---|---|---|
| `SPINT-main/` | 原始 SPINT 实现与数据接口 | teacher 和模型语义来源 |
| `streaming_calibration_exp/` | MUA 轻量 encoder 实验框架 | B3/B15/B16 的共享实现基础 |
| `software-to-hardware/` | B3 INT8/QAT 与硬件交接 | 部署支线；不决定当前 SUA 科学结论 |
| `planB_tempconv/` | 低成本时间卷积 decoder | decoder 压缩支线 |
| `sua_exploration/` | SUA/MUA 共享编码器 | **当前主线** |

根目录其余分析和旧计划保留为历史研究记录。除非主线文档显式引用，不应把它们当作当前优先级来源。

## 工作区注意事项

当前顶层 `.git/` 不完整，缺少 `HEAD` 和对象库，无法使用正常的 Git 状态或历史追踪。重要实验应依赖显式 checkpoint、配置、结果 JSON 和文档记录，而不能假设 Git hash 可用。
