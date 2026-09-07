# H1-RIFT-R300：实现、训练前验收与空闲 GPU 开训

日期：2026-09-07。依据：[RIFT 结构设计](DESIGN_RIFT_STREAMING_TEMPORAL_V1_20260907.md)。

命名更新（2026-09-07 用户约定）：旧 B-transformer 结构／系列统一称 **BT-EORT**，新结构／系列称 **RIFT**。`RIFT-recency` 与 `RIFT-flat` 都属于 RIFT，flat 不是 BT-EORT。历史 E-ORT 路径、提交 ID 与原始收据保持不变；实际执行后端另列，不由系列名推断。

用户已授权由主 agent 驱动并行 agents 搭建代码，完成 GPU 训练前测试，并在 GPU 空闲后立即开始训练。当前进度（2026-09-07 07:56 UTC）为 **IMPLEMENTATION PASS / CPU PASS / CUDA SMOKE PASS / BOTH FORMAL TRAINING HEALTHY**；不是训练完成或收益成立。

## 1. 本轮闭合范围

先落地 `btransform_unified_v2/` 中 H1-R300 的完整网络、批量 oracle、单步 KV、流状态和训练/评分接入。H1 首轮为同一个上下文配置下的两个配对模型：`recency` 与 `flat`（R0）；只改变 temporal recency bias。M1/M2 的完整上下文先在配置及模型接口中支持，不在本轮同时扩成多数据集训练矩阵。

两臂固定 D4、width256、8 heads、FFN512、conv k5/C16、P16、8 slots，H1 b=[75,75,75,74]、raw R=300。与 582044 的系统比较如实列出上下文由 200 增至 300，以及新 token/位置/逐层依赖语义。v1、HC、深度线及旧 submission 均不修改。

## 2. 并行所有权

| 工作项 | 执行 agent | 拥有范围 |
|---|---|---|
| temporal 与配置、缓存/梯度验证 | `rift_temporal` | `src/btransform_unified_v2/{config,temporal}.py`、`tests/test_temporal.py` |
| 稳定 frontend、完整 decoder、stream 生命周期 | `rift_frontend` | `src/btransform_unified_v2/{model,streaming,__init__}.py`、对应 model/stream tests |
| C2-CAL-1 B2 训练接入、真实数据预检 | `rift_training` | `scripts/rift_v1/h1_*.py`、`tests/test_h1_training_contract.py` |
| 接口、集成验收、GPU 启动与证据汇总 | 主 agent | 文档、集成检查和运行协调；不越权修改其他任务产物 |

各执行者不撤销他人编辑，共享接口问题直接同步。所有源文件编辑使用 apply_patch；生成的测试/训练记录存入新结果目录。代码完成后按实际 diff 和验证证据验收，不以口头完成替代。

## 3. 训练前必需验收

1. 配置与数值：各任务 R→逐层 b 推导正确；recency/flat 的共同可训练参数初始化字节一致；位置 bias 是两臂预期的区别。
2. 前向正确：CPU FP32 的 batch/stream 同模型误差满足 `1e-5 + 1e-5*abs(reference)`；稳定前端、冷启动、多次 ring 回绕、B1/B8 都有覆盖。
3. 依赖正确：未来扰动不影响过去；窗口外 decoder 输入不影响目标；足量 warm-up 的分块目标输出及梯度与完整图一致。
4. 状态正确：按 stream_id 绑定，重排、单独 reset、bank/mask/model 失效处理、重复 predict 幂等，不能漏掉不计分但真实存在的神经 bin。
5. 冷启动明确：`input_valid_mask` 只表示真实输入是否存在。reset 前 batch padding 不能成为有效 temporal KV；真实零放电 bin 仍然有效，prediction/eval mask 不能替代该掩码。
6. 训练合同：13 source sessions、目标坐标与输入切片核验，C2-CAL-1 B2 的 bank/预算/起点选择不变；真实数据 CPU 微批前后向、有限 loss/梯度、参数更新及尺度桥通过。
7. 运行准备：CLI、可恢复 checkpoint、独立输出路径和预估内存/成本清楚；不得覆盖旧权重或在 GPU 空闲判定前探测性占显存。

CPU 前检完成后再使用空闲 GPU 做小规模 CUDA/bf16 运行核验与内存检查，随后立即正式训练；smoke 产物单列，正式两臂从共同初始化开始。已通过的检查只在相关代码变化或新失败出现时重跑。

## 4. H1 首轮训练合同

以已核对的 B2 配方为基础：32 epochs、每 epoch 731 次 optimizer 更新、有效 batch32、seed42、AdamW peakLR=1e-4→1e-5、warm-up 一 epoch、EMA=0.9995、clip=1、unit dropout=0.10。神经输入 R300，训练目标为 20y，评分回到 native。

recency/flat 匹配共同初始化、目标 endpoint 序列、session/bank 序列、unit-dropout 随机流和更新数。若使用 microbatch，保持原有效 batch 的目标数与 loss 权重，梯度在循环外清零，LR/global_step/EMA 每 optimizer 更新计一次；尾批按实际目标数处理。

训练上下文按当前权重重算，不跨 optimizer step 复用 KV；序列内 mask 固定。近 reset 样本的有效输入掩码与在线相同。数据复用优先使用已核验缓存/只读函数，不重新定义 carrier estimator，不静默改变标签对齐或泄露未来。

训练中记录健康与成本，不用官方 test 选点。沿用 B2 在全程训练结束后做 HO-M3 开发面 EMA epoch sweep 的选择法，独立标记其为开发选模面而非 untouched test；正式 32-epoch 权重、优化器、EMA、RNG 与进度可恢复。

本轮是上述两臂，不自动增加 seed、HC/depth 组合或其他数据集的训练网格。未继承旧脚本的历史设备固定值或不明确的 5 小时硬中断；实际训练成本须在预检和运行记录中报告。

## 5. 空闲 GPU 启动规则

初始化设备快照：2026-09-07 07:21 UTC，两张 RTX 3090 均在运行其他 M1 P32 depth 训练，显存虽有余量，但算力占用约 98%，不视为空闲。

仅在预检闭合后选择没有其他计算进程且处于空闲状态的 GPU，并在启动前重新确认；不能依据“剩余显存足够”与他人并跑，也不停止/移动/改写既有任务。若发现已有明确设备预留，遵守其归属。

一张 GPU 先空闲时优先启动 `recency`；另一张空闲后可启动 `flat`。只有一张持续可用时两臂顺序执行，保持训练合同相同。对每次启动保存物理 GPU index/UUID、进程 PID、命令、代码/输入配置标识、输出目录和开始时间。

训练尚未启动时继续等待真实空闲状态；有新资源冲突时记录原因，不虚构“已开训”。启动后先检查真实 optimizer 更新、loss/梯度健康和 checkpoint/heartbeat，再报告运行中。

## 6. 状态与交付

分别报告 `IMPLEMENTATION / CPU_PREFLIGHT / CUDA_PREFLIGHT / TRAINING_RECENCY / TRAINING_FLAT / SELECTION`，不能用单一 PASS 代替不同阶段。结果目录将在启动前绑定，实际命令、PID、测试与训练记录在落盘后补充；本页不把计划中的命令或测试写成已执行证据。

## 7. 实际验收与启动记录

以下为 2026-09-07 07:56:48 UTC（香港时间 15:56:48）快照；后续以各 formal 目录的 heartbeat、metrics 和 checkpoint 为准。[完整启动收据](../results/rift_v1/h1_r300_pair_20260907T075312Z/launch_receipt.json)保存命令、GPU UUID、代码 SHA、配对校验和当时的运行数据；[ready 文件](../results/rift_v1/h1_r300_pair_20260907T075312Z/ready.json)是启动前的源码冻结门。

| 阶段 | 实际结果 |
|---|---|
| IMPLEMENTATION | `config.py`、独立向量化 batch temporal、单步 KV、稳定 `frontend_last`、多流 wrapper、H1 trainer/scorer 已接通 |
| CPU_PREFLIGHT | 正式 conda 环境 `38 passed in 2.29s`；真实 H1 B2 样本完成前后向、有限梯度、参数更新和 EMA 更新 |
| CUDA_PREFLIGHT | recency／flat 各 4 次有效 batch32 更新，bf16 autocast，均 exit 0；独立 SMOKE checkpoint，不作为正式初始化 |
| TRAINING_RECENCY | GPU 0，PID `1903714`，正式 step 300/23392，loss `0.0081186763`、grad norm `0.22791949` |
| TRAINING_FLAT | GPU 1，PID `1903715`，正式 step 300/23392，loss `0.0081620839`、grad norm `0.23484345` |
| SELECTION | PENDING；全程 32 epochs 后 HO-M3 开发面 EMA epoch sweep，未触碰官方 test |

两臂正式模型均于 07:55:44 UTC 初始化到 GPU 并开始更新。使用 `/home/xinyuan/miniconda3/envs/spint/bin/python`、`PYTHONNOUSERSITE=1`、PyTorch `2.5.1.post303` / CUDA `11.8`，避免 `~/.local` 中另一套 PyTorch 覆盖正式环境。CPU 测试关闭 GPU 可见性，正式队列在两次空闲快照后各自设置物理 `CUDA_VISIBLE_DEVICES`；子进程内的 `cuda:0` 指其唯一可见设备。

两臂共同可训练初始化 SHA 为 `ff3aa959fb7db33cfeea15201f8498bf27cd5719721c0e7362160e20e98e1c6b`，endpoint inventory、有效输入 mask inventory、bank roster SHA 均相等。4-update CUDA smoke 的实际 endpoint/bank/unit-mask 顺序 SHA 也相等，为 `cfe3e7e090fa3f67cd37987b47483b70a428abd66c56cce5fc192b8f64362ba8`；正式每 epoch 继续落盘实际顺序摘要。

[真实数据 CPU 预检](../results/rift_v1/h1_r300_cpu_preflight_20260907T154000Z/preflight.json)记录 13 source sessions、每 epoch 731 updates、输入 `[2,300,176]`、有效 bin 数、真实 endpoint 和 bank SHA，梯度 norm `3.6787848`，参数确有更新，raw/native MSE 尺度比约 400。该早期目录中的 `154000Z` 是历史命名误将本地时钟标为 Z；本节运行目录及收据使用真正 UTC，不据目录名重解释实际运行顺序。

首 300 updates 每臂用时约 58 秒；PyTorch 峰值 allocated memory 均为 `2442.59 MiB`（约 `2.385 GiB`），nvidia-smi 进程占用约 `3608 MiB`，两者口径不同。此时两卡利用率约 96%，只是早期训练吞吐与资源健康证据，不是 32-epoch 总时长承诺或部署推理测量。

正式输出：

- [recency formal](../results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/)、[recency 日志](../results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/train.log)。
- [flat formal](../results/rift_v1/h1_r300_pair_20260907T075312Z/flat_20260907_155347/formal/)、[flat 日志](../results/rift_v1/h1_r300_pair_20260907T075312Z/flat_20260907_155347/train.log)。
- [实时队列状态](../results/rift_v1/h1_r300_pair_20260907T075312Z/launch_state.json)：launcher PID `1903473`，持续写入 heartbeat 并追踪当前两臂，不需要再发启动命令。

当前实现采用有界 KV 列表及批量 attention，不是预分配 ring 的最终优化实现。K/V view 会保留共享 QKV backing storage，因此设计中的 H1-R300 FP32 `604160 B/stream` 是逻辑 K/V payload，不能直接当作本实现的真实 tensor storage；CPU 成本探针另行记录 unique backing storage，不能混淆。M1-R100／M2-R50 已有配置与模型接口，本轮未开始其训练；部署 adapter／ORT 导出也未列为已完成。

## 8. CPU 流式成本实测

正式证据为 [2026-09-07 08:00:05 UTC benchmark](../results/rift_v1/cpu_cost_20260907T080005Z/benchmark.json)，脚本为 [benchmark_cpu.py](../scripts/rift_v1/benchmark_cpu.py)。Ryzen 9 7950X、正式 conda PyTorch 2.5.1.post303、两线程、FP32、inference_mode；使用未训练 seed42 模型、合成 H1 raw／bank，不是精度评测。每配置先填满对应上下文，再测三轮各 30 次真实新 bin 推进，共 90 次。

| 配置 | 完整 stream_step median | mean | P95 |
|---|---:|---:|---:|
| R200 / B1 | 2.524 ms | 2.532 ms | 2.769 ms |
| R300 / B1 | 2.664 ms | 2.702 ms | 3.047 ms |
| R200 / B8 | 12.237 ms | 12.237 ms | 12.778 ms |
| R300 / B8 | 13.061 ms | 13.374 ms | 13.723 ms |

这里 B8 是一次推进八条流的总耗时，不是单条流的耗时。完整推进包括 stream_id 查询、bank/mask hash、raw 历史、单 token frontend、状态 pack/unpack、temporal 和 readout。当前短测中 R200→R300 的 median 增幅为 B1 约 5.6%、B8 约 6.7%，支持将 300 作为首轮历史候选；不是普遍性能上界。R300 分项 median：frontend_last B1/B8 为 0.883/6.852 ms，temporal.step 为 0.682/2.240 ms。

R300 每条流逻辑 K/V 为 604160 B，但 unique backing tensor storage 实测为 906240 B（885 KiB），因为 K/V view 保留完整 QKV base；B8 对应 7,249,920 B（约 6.914 MiB）实际 backing storage。这仍不含 Python 对象、raw 缓冲、临时 attention 或模型本体，不能当成总 RSS。各配置填满后继续推进 90 次，步后的逻辑及 backing storage 均不增长。

没有在本轮短探针中重测封存 582044 BT-EORT engine，因此不能宣称 RIFT 相对 BT-EORT 加速多少倍。早期 `cpu_cost_20260907T075007Z` 和 `075633Z` 保留为诊断记录，其中 component 测量存在 grad-enabled／环境口径问题；以本节 `080005Z` 的正式 inference-mode 结果替代这些分项结论。

## 9. 后续评分主机内存与早期 checkpoint

只读读取 14 个 HO calibration NWB 的 eval_mask，合计 33,613 个有效目标点；未为此资源核验物化 HO R300 窗口。当前 train endpoint inventory 为 23,212 点。

- 每臂 train R300 float32 X-store：`23212*300*176*4 = 4,902,374,400 B`，约 4.5657 GiB。
- 每臂 HO R300 float32 X-store：`33613*300*176*4 = 7,099,065,600 B`，约 6.6115 GiB。
- 当前 `run()` 调用 `score_stage()` 时仍保留 train；两臂同时评分的 train+HO X-store 及 valid masks 最低常驻约 22.386 GiB，低于主机 62 GiB RAM。这只是上述数组的底数，不等同于整个评分进程 RSS。

08:00 UTC 的后续核验已见两臂各自 `epoch_001.pt`、`epoch_002.pt`；每 epoch 731 次更新，正式前两个 epoch 的实际 endpoint/bank/unit-mask 顺序 SHA 逐 epoch 配对一致。首两 epoch 的训练 MSE 分别为 recency `0.0084697812 / 0.0070576724`、flat `0.0084709757 / 0.0070567899`；这些训练值不用于宣称开发面或官方精度胜负。

## 10. 下一轮训练的前置优化（用户指令，尚未实施）

2026-09-07 用户要求：下一次如果还训练 RIFT，将 microbatch 提到 16/32，并把训练 attention 改为真正计算局部窗口，而不是先算完整 300×300 再 mask。本节是下一轮开训前必须执行的要求，不是新增训练授权，也不改变当前已冻结并启动的两臂源码、参数或 checkpoint。

### 10.1 microbatch 16/32，保持有效 batch32

- 下一轮优先测试 `microbatch=32`，并测 `16` 作为备选；按真实 optimizer-step 吞吐和峰值显存确定，不继续默认沿用本轮的 8。
- 有效 batch 仍为 32：microbatch32 每个有效 batch 一次前后向；microbatch16 分两次累积。尾批按实际样本数加权，梯度每个有效 batch 清零一次，optimizer/LR/EMA/global_step 每次有效更新推进一次。
- recency／flat 使用相同 microbatch、有效 batch 和数据/endpoint/bank/unit-dropout 配对合同；不因为改 microbatch 同时改变学习率、训练更新数或目标尺度。
- 下一轮空闲设备预检记录 16/32 的 steady optimizer-step time、有效 samples/s、peak allocated/reserved 显存和数值健康。此处尚未执行新的 GPU 测试。

### 10.2 真正的局部窗口训练 attention

- H1-R300 保持输入长度 300、四层与 raw 感受野 300 不变；每层窗口严格沿用 `[75,75,75,74]`（均含当前 token），不是四层都改成 75，也不是把训练输入截成 75。
- 生产训练路径只对每个 query 的真实因果局部 K/V 执行 attention；有效 score/probability 计算为 `[B,H,T,b_l]` 或等价的局部分块布局，不构造完整 `[B,H,T,T]` 后再屏蔽。其他 RIFT 配置读取各层实际窗口，不硬编码所有任务为 75。
- 保持 cold-start 的 `input_valid_mask`、真实零放电 bin、因果边界、recency 相对年龄与半衰期、flat 零偏置以及残差/FFN 语义；训练上下文仍在当前权重下重算并保留梯度，不跨 optimizer step 复用旧 KV。
- 避免通过大规模显式复制滑窗 K/V 引入新的峰值内存瓶颈；验收看实际内存和吞吐，不仅看 attention 输出形状。
- 原 dense+mask forward 保留为独立正确性 oracle，不作为新的默认训练计算路径。新路径在开训前验证输出、输入/参数梯度及 warm-up 分块一致性，覆盖冷启动、边界、recency／flat；再做真实数据 CUDA/bf16 检查。

从长度 300 的 dense score 布局变为约 75 的 local 布局，attention score 的候选计算规模约降到四分之一；这不是全模型训练速度提升四倍的承诺。QKV、FFN、frontend、数据传输及内核效率均须计入端到端 optimizer-step 测量。两项优化落地后另存新来源 SHA 与预检收据，不重写本轮历史记录。
