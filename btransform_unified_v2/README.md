# B-transformer Unified V2 — RIFT

下一代时间结构路线：**RIFT — Recency-biased Incremental Finite-context Transformer**（近时优先的增量有限上下文 Transformer）。

当前状态（2026-09-07）：H1-R300 recency／flat 两臂均已完成 32 epochs 与 HO-M3 开发面 EMA 选模，分别选中 e22 / e19，七 session 等权平均 R² 为 0.374823 / 0.362731。同开发面的 BT-EORT H1 B2 e18 为 0.378414；RIFT 尚无官方 test 分数。L200-recency 质量对照已于 19:28 香港时间在 GPU0 正式启动（microbatch32 / true-local，PID 1915357），固定 R300 权重的 CPU 推理优化已完成首轮验收；flat 按用户要求稍后启动。

设计入口：[RIFT V1 结构与实施设计](docs/DESIGN_RIFT_STREAMING_TEMPORAL_V1_20260907.md)。
执行入口：[H1-R300 配对实现与自动开训计划](docs/EXECUTION_H1_RIFT_R300_V1_20260907.md)。
当前工作：[L200 质量对照与 CPU 推理优化](docs/EXECUTION_H1_RIFT_L200_AND_CPU_20260907.md)。

CPU 首轮优化已验收：固定 R300 e22 EMA、FP32，在同机两物理核/2线程的真实 B1/B8 端到端基准上，`CpuRiftRuntime(..., temporal_backend="cached")` 平均耗时为 2.707 / 15.656 ms，封存 BT-EORT 为 18.051 / 112.955 ms；本轮相对原 RIFT reference 额外降低平均延迟 13.3% / 29.6%。这是共享宿主上的工程实测，不是官方服务器结果。完整分布、保真、53 项测试与复现方式见当前工作记录 §8。

新训练已接入 microbatch 16/32（有效 batch32）和真正 local attention；dense 路径保留作数值 oracle。H1-R300 每层窗口为 `[75,75,75,74]`，H1-R200 为 `[50,50,50,49]`。这些执行改动不追溯改写已完成的 R300 运行；具体 CUDA 验收与新训练状态以当前工作记录为准。

## 统一命名（2026-09-07 用户约定）

- **BT-EORT**：旧的 B-transformer 结构／系列，BT 表示 B-transformer；此前讨论的 H1／M1／M2 E-ORT 系统以后统一使用此名称。具体前端、深度、上下文和 checkpoint 仍按各任务记录，不将三个系统说成同一个配置。
- **RIFT**：新的 Recency-biased Incremental Finite-context Transformer 结构／系列。
- **RIFT-recency / RIFT-flat**：均属于 RIFT；flat 是关闭 recency 偏置的配对消融，不是 BT-EORT。
- 结构／系列名与实际执行后端分开记录；例如 RIFT 将来使用 ORT 后端仍称 RIFT。历史路径、类名、镜像标签、提交 ID 与原始收据不追溯改名，历史 `E-ORT` 称呼保留可追溯映射。Original SPINT 也不因此改称 BT-EORT。

## 核心决策

- 稳定的逐时刻 frontend token + 相对时间 recency 偏置 + 逐层滑窗注意力 + 有界 KV 状态。
- H1 主要扩展候选为 **`H1-RIFT-D4-R300`**；旧 200 bin 是人工筛选的系统基线，不是通用上限。
- 其他数据集采用各自完整任务上下文；`R_task`、每层窗口、预处理历史和采样率分别配置，不统一为 200/300。
- 保持四层与主干宽度，不在首个对照中混入 HC、减深或新 runtime 的收益。
- `RIFT` 是新结构；旧 B-transformer 系列统一称 `BT-EORT`，其既有 fast exact-E + ONNX Runtime 执行方案与模型结构分别记录。

## 目录边界

本目录独立承载新设计及后续实现。`../btransform_unified_v1/` 和旧 submission 仅作为只读基线；HC/深度线由其原任务继续管理。

代码位于 `src/btransform_unified_v2/`，入口位于 `scripts/rift_v1/`，新结果位于 `results/rift_v1/`。用户已于 2026-09-07 授权：完成 GPU 训练前测试后，在 GPU 空闲时立即开始训练。只使用空闲设备，不停止他人任务，不覆盖 v1 或旧运行产物；正式训练状态以执行计划和新结果记录为准。

运行证据：[启动与配对验收记录](results/rift_v1/h1_r300_pair_20260907T075312Z/launch_receipt.json)、[实时队列状态](results/rift_v1/h1_r300_pair_20260907T075312Z/launch_state.json)。两臂从共同初始化独立训练；4-update smoke 的权重不用于正式训练。全部 32 epochs 结束后才进行 HO-M3 开发面 EMA 选模，不使用官方 test。
