# SPINT Dim256 / Top-K64 A8W8 混合 QAT 硬件说明书（SPEC）

| 项目 | 内容 |
|---|---|
| 文档版本 | 0.1（架构与接口冻结候选） |
| 目标网络 | FALCON m2 SPINT student，`Dim=256`、`Top-K=64`、A8W8 混合 QAT |
| 目标用途 | ASIC/FPGA 微架构、RTL、验证与后端流片输入 |
| 参考实现 | `src/models/components/spint.py`、`src/models/components/quantization.py` |
| 本 SPEC 的精度状态 | 网络拓扑、张量形状、量化边界已定义；逐层 scale 数值、FP 数据格式和最终 checkpoint 尚需量化签核后冻结 |

## 1. 范围与符合性

本 SPEC 规定推理网络的功能、模块边界、张量布局、权重形状、量化域及控制行为。实现若在同一已冻结 checkpoint 与 scale 表下，对指定输入产生与 Q/DQ 参考模型一致的输出，即认为符合本 SPEC。

本网络不是“全 INT8”网络。A8W8 只覆盖 8 个高算力 Linear；ID 分支为 W8A16，RMS pruning、LayerNorm、attention score/Softmax、残差和输出头保持浮点域（记为 `P_FP`）。本 SPEC 建议 `P_FP=FP16`，但当前 PyTorch QAT 代码没有把 FP16/BF16/FP32 硬件格式写死；在流片冻结前必须以最终 Q/DQ reference 选择并验证一种格式。不得将训练阶段的 fake-quant 计时当作芯片 INT8 时延。

## 2. 固定网络常量

| 符号 | 值 | 说明 |
|---|---:|---|
| `N` | 96 | 输入神经元/token 数 |
| `T` | 50 | 解码时间窗长度 |
| `M` | 33 | calibration trials 数 |
| `Tc` | 100 | 每个 calibration trial 的时间长度 |
| `C` | 2 | 行为协变量/查询 token 数 |
| `D` | 256 | 模型隐藏宽度 |
| `H` | 32 | attention heads 数 |
| `Dh` | 8 | 单头宽度，`D/H` |
| `K` | 64 | RMS Top-K 保留 token 数 |
| `L` | 1 | cross-attention layer 数 |
| `F` | 256 | FFN 隐层宽度 |

除 batch `B` 外，所有维度均为流片固定常量。实现可支持 `1 ≤ B ≤ B_MAX`，但不能改变上述网络语义；建议最小合规 profile 为 `B=1`。

## 3. 顶层接口与数据布局

### 3.1 外部输入/输出

| 端口/缓冲区 | 逻辑形状 | 元素域 | 语义 |
|---|---|---|---|
| `NEURAL_IN` | `[B][T][N] = [B][50][96]` | `P_FP` | 当前 50-bin 神经特征；软件张量轴为时间、神经元 |
| `CALIB_IN` | `[B][M][Tc][N] = [B][33][100][96]` | `P_FP` | 校准试次神经特征 |
| `WEIGHT_MEM` | 见第 7 节 | INT8 / `P_FP` | 冻结的参数、bias、scale、LayerNorm 参数及 learned `rep` |
| `CTRL` | 寄存器 | 整数 | `start`、`id_map_valid`、`output_mode`、状态/错误读取 |
| `BEHAVIOR_OUT` | `[B][T][C] = [B][50][2]` | `P_FP` | 网络完整输出 |
| `LAST_OUT`（可选） | `[B][C] = [B][2]` | `P_FP` | `BEHAVIOR_OUT[:,49,:]`；与训练时 `decode_last_timestep_only=true` 的评估接口一致 |

逻辑布局与 PyTorch 对齐。建议 DMA 在线性存储中采用最后一维连续：`NEURAL_IN[b][t][n]`、`CALIB_IN[b][m][tc][n]`。硬件内部将神经输入转置为 token-major 的 `SRC[b][n][t]`，将 calibration 转置为 `CAL_T[b][m][n][tc]`。任何物理分块/tiling 均不得改变此逻辑索引。

### 3.2 控制与缓存协议

该网络的 calibration ID 在 calibration 不变的连续解码窗口中可复用。顶层应支持两种合法模式：

1. **完整模式**：`id_map_valid=0`，读取 `CALIB_IN`，生成 `ID_MAP[b][96][50]` 后完成一次解码。
2. **缓存模式**：`id_map_valid=1`，忽略 `CALIB_IN`，从片上/片外 `ID_MAP` 读取已生成结果，再解码 `NEURAL_IN`。

缓存模式仅在 `CALIB_IN`、冻结权重、`P_FP` 格式均未变化时有效；软件必须在 calibration 更新、模型切换或复位后将 `id_map_valid` 清零。`ID_MAP` 容量为 `B×96×50` 个 `P_FP` 元素（若为 FP16，`B=1` 时为 9,600 B）。此缓存是数学等价的调度优化，不改变网络。

## 4. 顶层数据流

```text
CALIB_IN ──► Calibration-ID Engine ──► ID_MAP[B,96,50] ─┐
                                                          ├─► add ─► RMS/TopK64 ─► gather
NEURAL_IN[B,50,96] ─► transpose ─► SRC[B,96,50] ─────────┘                         │
                                                                                     ▼
 learned REP[1,2,50] ─┐                                                    FC_IN A8W8
                      ├─► FC_IN A8W8 ─► Q tokens[B,2,256]                 K/V[B,64,256]
                      │                                                               │
                      └────────────────► Explicit 32-head Cross Attention ◄─────────┘
                                                   │
                                                   ▼
                                           FFN A8W8 + residual
                                                   │
                                                   ▼
                                              FC_OUT P_FP
                                                   │
                                                   ▼
                                       BEHAVIOR_OUT[B,50,2]
```

执行顺序必须保持：`ID_MAP` 加到 `SRC` 后，先计算 RMS 和 Top-K/gather，随后才对 retained token 运行 A8 `fc_in`。Top-K 的计算算子本身在 `P_FP`；但 `ID_MAP` 来自 W8A16 分支，因此其量化误差会进入 Top-K 分数。这一行为必须与最终 Q/DQ reference 保持一致。

## 5. 模块功能与 I/O 规范

### 5.1 M0：Calibration-ID Engine（W8A16）

输入 `CALIB_IN[b,m,tc,n]` 先重排为 `CAL_T[b,m,n,tc]`。对每个 `(b,m,n)`，执行：

```text
U0 = Linear_100x256_W8A16(CAL_T)       # [B,33,96,256]
U1 = ReLU(Linear_256x256_W8A16(U0))    # [B,33,96,256]
U2 = Linear_256x256_W8A16(U1)          # [B,33,96,256]
V  = mean_m(U2)                        # [B,96,256]
V1 = ReLU(Linear_256x256_W8A16(V))     # [B,96,256]
V2 = ReLU(Linear_256x256_W8A16(V1))    # [B,96,256]
ID_MAP = Linear_256x50_W8A16(V2)       # [B,96,50]
```

这里的 `Linear_W8A16` 定义为 INT8 per-output-channel weight 与 `P_FP` activation 的矩阵乘法，bias 与输出均为 `P_FP`。`mean_m` 必须计算 33 项算术平均；允许使用较高精度累加器，最后按 `1/33` 归一化并舍入到 `P_FP`。

### 5.2 M1：输入融合、RMS Top-K 与 gather（浮点域）

```text
X[b,n,t] = SRC[b,n,t] + ID_MAP[b,n,t]       # [B,96,50]
score[b,n] = sqrt(mean_t(X[b,n,t]^2))       # [B,96]
idx[b,0:63] = TopK_descending(score[b,:])   # [B,64]
S[b,k,t] = X[b,idx[b,k],t]                  # [B,64,50]
```

要求：

- 每个 batch 独立排序；`idx` 的第一维为分数降序。
- 分数相等时，PyTorch `torch.topk` 的 tie 行为并不保证稳定排序。流片前必须在 software reference 中固定 tie-break（推荐：较小 neuron index 优先），并在硬件/软件共同实现；否则逐 token 输出可能不比特一致。
- `sqrt`、平方、累加和比较器的 `P_FP` 格式、舍入和 NaN/Inf 处理须在第 10 节冻结。
- 本配置中 `dropout_rate=0`、`tf_drop_rate=0`、`dynamic_dropout=false`；推理硬件不得实现随机 dropout。

### 5.3 M2：共享 Read-in MLP（A8W8）

同一套权重对 retained neural token 与 learned query `REP` 复用：

```text
ENC(z) = Linear_256x256_A8W8(ReLU(Linear_50x256_A8W8(z)))
KVTOK = ENC(S)                 # [B,64,256]
QTOK  = ENC(REP broadcast B)   # [B,2,256]
```

`REP` 是冻结的可学习参数，逻辑形状 `[1,2,50]`；对 batch 广播后不复制权重。M2 的输入 activation 与两个 Linear 的权重均执行 Q/DQ，输出返回 `P_FP`，再进入 LayerNorm。

### 5.4 M3：显式 32-head Cross Attention（混合精度）

设 `Q0=QTOK[B,2,256]`、`S0=KVTOK[B,64,256]`。先各自通过同一组 `LN1` 参数（硬件可按两个输入串行复用单元）：

```text
QN = LayerNorm1(Q0)                         # [B,2,256], P_FP
SN = LayerNorm1(S0)                         # [B,64,256], P_FP
Q  = reshape(Linear_Q_A8W8(QN), [B,32,2,8])
K  = reshape(Linear_K_A8W8(SN), [B,32,64,8])
V  = reshape(Linear_V_A8W8(SN), [B,32,64,8])
P  = Softmax((Q × K^T) / sqrt(8))           # [B,32,2,64], P_FP
O  = concat_heads(P × V)                    # [B,2,256], P_FP
X  = Q0 + Linear_O_A8W8(O)                  # [B,2,256], P_FP
```

`Linear_Q/K/V/O` 均为 `256×256` A8W8。注意力 score 矩阵、`1/sqrt(8)` 缩放、Softmax、head reshape 和残差加法均保持 `P_FP`。该网络不支持 `attn_mask` 或 `key_padding_mask`；硬件接口无需相应端口。

### 5.5 M4：FFN、输出头与输出重排

```text
XN = LayerNorm2(X)                           # [B,2,256], P_FP
F1 = ReLU(Linear_256x256_A8W8(XN))           # [B,2,256], P_FP after DQ
Y  = X + Linear_256x256_A8W8(F1)             # [B,2,256], P_FP
R  = Linear_256x50_FP(Y)                     # [B,2,50], P_FP
BEHAVIOR_OUT = transpose(R, axes=(0,2,1))    # [B,50,2]
```

M4 的 `Dropout` 在训练代码中存在但概率为 0；推理中应完全旁路。`FC_OUT` 不量化：其 weight、bias、输入和输出均处于 `P_FP`。

## 6. 量化与算术合同

### 6.1 W8A16 Linear

对输出通道 `o`，冻结权重 scale：

\[
s_{w,o}=\max_i\left|W_{o,i}\right|/127,
\quad q_{w,o,i}=\operatorname{clip}_{[-127,127]}(\operatorname{round}(W_{o,i}/s_{w,o}))
\]

硬件数值语义为：

\[
y_o=\sum_i x_i\,(q_{w,o,i}\cdot s_{w,o})+b_o
\]

`x`、`b`、`s_w` 和输出为 `P_FP`。实现可采用 INT8×FP MAC，也可将激活转换为适当的内部格式；在模块边界处的结果必须与上式及冻结 reference 一致。

### 6.2 A8W8 Linear

除权重量化外，对每个 Linear 输入张量使用一个对称、per-tensor activation scale：

\[
s_a=\max|x|/127,
\quad q_a=\operatorname{clip}_{[-127,127]}(\operatorname{round}(x/s_a))
\]

计算可等价为：

\[
y_o=(s_a s_{w,o})\sum_i q_{a,i}q_{w,o,i}+b_o
\]

当前 QAT observer 以观测期的 activation 最大绝对值的历史最大值来更新 scale，epoch 10 后冻结。流片权重包必须携带最终冻结的 scale；硬件不得在推理时重新 observer 或自适应更新 scale。

### 6.3 量化模块清单

| 模块 | Linear 个数 | weight | activation | 权重元素数 |
|---|---:|---|---|---:|
| Calibration ID（`fc_id_in`） | 3 | W8/输出通道 | `P_FP` | 156,672 |
| Calibration ID（`fc_id_out`） | 3 | W8/输出通道 | `P_FP` | 143,872 |
| Read-in `fc_in` | 2 | W8/输出通道 | A8/张量 | 78,336 |
| Attention Q/K/V/O | 4 | W8/输出通道 | A8/张量 | 262,144 |
| FFN | 2 | W8/输出通道 | A8/张量 | 131,072 |
| 输出 `fc_out` | 1 | `P_FP` | `P_FP` | 12,800 |
| 合计 | 15 | 14 个 W8 Linear | 8 个 A8 Linear | 784,896 |

W8 权重总数为 **772,096 B**（754 KiB，未含 scale）；A8W8 子集为 **471,552 B**。每输出通道一条 weight scale，共 **3,378** 条；8 个 A8 Linear 各一条 activation scale。若把全部 scale 固定为 FP16，weight scale 表为 6,756 B、activation scale 表为 16 B；最终位宽需通过第 10 节的签核确认。所有 Linear bias 当前保持 `P_FP`。

## 7. 参数存储规范

| 参数类别 | 数量 | 推荐存储域 | 备注 |
|---|---:|---|---|
| 14 个量化 Linear 的 weight | 772,096 | INT8 | row-major `[out][in]`；逐输出通道 scale |
| `fc_out` weight | 12,800 | `P_FP` | `[50][256]` |
| 全部 Linear bias | 3,428 | `P_FP` | 对应每个 Linear 输出通道 |
| 两个 LayerNorm 的 γ/β | 1,024 | `P_FP` | 每个 LayerNorm 为 256 γ + 256 β |
| learned `REP` | 100 | `P_FP` | `[1][2][50]` |
| 模型总参数 | 789,448 | 混合 | 与训练配置一致 |

需要为模型版本号、权重 CRC、scale CRC、`P_FP` 格式、rounding mode 和 Top-K tie-break 版本提供只读 metadata 寄存器。仅替换 INT8 weight 而不同时替换 scale/bias/浮点参数属于非法模型装载。

## 8. 计算量、带宽与推荐调度

下表为 `B=1` 的乘加（MAC）计数；不包含 ReLU、LayerNorm、Softmax、RMS、Top-K、量化/反量化及数据搬运。

| 阶段 | MAC/次完整推理 | 占总 MAC |
|---|---:|---:|
| Calibration ID `fc_id_in` | 496,336,896 | 94.66% |
| Calibration ID `fc_id_out` | 13,811,712 | 2.63% |
| Read-in `fc_in`（64 retained + 2 query） | 5,170,176 | 0.99% |
| Q/K/V/O 投影 | 8,650,752 | 1.65% |
| attention `QKᵀ` + `PV` | 65,536 | 0.01% |
| FFN | 262,144 | 0.05% |
| `fc_out` | 25,600 | <0.01% |
| **完整模式合计** | **524,322,816** | **100%** |
| **缓存模式：不含 ID 生成** | **14,174,208** | — |

因此建议调度如下：

1. 将 M0 作为低频 calibration-update task，生成并缓存 `ID_MAP`；
2. 将 M1–M4 作为每个 50-bin 解码窗口的实时 task；
3. 使用共享 INT8 GEMM 阵列处理 M0、M2、M3、M4 的 14 个量化 Linear，并以 scale/bias 后处理单元完成 DQ；
4. 使用独立的 `P_FP` 向量单元处理 LayerNorm、RMS、Top-K、Softmax、残差和 `fc_out`；
5. Q/K/V 可共享同一 GEMM 阵列但不得改变其各自独立 weight/scale/bias；M2 需支持同权重对 64 个 K/V token 与 2 个 query token 的两次调用。

Attention workspace 的固定最大逻辑尺寸为 `B×32×2×64 = B×4,096` 个 `P_FP` 元素；实现可按 head 或 query 分块流式计算 Softmax，不要求完整 score 矩阵落 SRAM。Top-K 只需维持 64 个 `(score,index)` 候选，推荐 streaming compare/select 实现。

## 9. 控制状态机、时序与异常处理

推荐顶层状态机：

```text
IDLE
 └─ start, id_map_valid=0 → LOAD_CALIB → ID_ENCODE → ID_CACHE_WRITE
 └─ start, id_map_valid=1 → ID_CACHE_READ
ID_CACHE_WRITE / ID_CACHE_READ → LOAD_NEURAL → ADD_RMS → TOPK_GATHER
TOPK_GATHER → READIN → ATTENTION → FFN_OUT → WRITE_OUTPUT → DONE → IDLE
```

最小状态/错误寄存器：`busy`、`done`、`error`、`id_map_valid`、`model_crc_match`、`scale_crc_match`、`topk_tie_seen`（可选计数）。出现非法 `K>96`、weight/scale CRC 不一致、未装载 `ID_MAP` 却请求缓存模式、输入 NaN/Inf 或 DMA 越界时，应置 `error` 并禁止 `done`；软件恢复前不得输出该次结果。

系统级 latency 由目标时钟、GEMM 阵列规模、SRAM/DRAM 带宽和 calibration 更新频率决定，不能只由网络结构给出。流片性能预算必须至少分别签核：

- `t_decode`：缓存模式下 M1–M4 每个解码窗口的 p50/p95 latency；
- `t_calib`：M0 生成一次 `ID_MAP` 的 latency；
- `t_amortized = t_decode + t_calib / R`：每 `R` 个窗口更新一次 calibration 时的均摊 latency；
- 峰值/平均 SRAM、外部带宽和功耗。

## 10. 流片前必须冻结并签核的项目

当前仓库已定义网络和 fake-quant 范围，但还没有一个已验证的 TensorRT/QDQ INT8 engine 或包含固定数值 scale 的交付包。以下项目是 **tape-out blocker**，不可由 RTL 团队自行猜测：

1. **最终模型版本**：从 A8W8 QAT run 的 callback metadata 选择 `best_model_path`，冻结 checkpoint SHA-256。
2. **量化参数包**：导出 14 组 INT8 weight、3,378 个 per-channel `s_w`、8 个 `s_a`、所有 bias、LayerNorm 参数、`REP` 与 `fc_out`；记录每项 CRC。
3. **精确数值格式**：确定 `P_FP` 为 FP16、BF16 或 FP32，确定 `s_w/s_a/bias/LayerNorm` 位宽、INT32 accumulator 宽度、DQ 舍入/饱和方式和 denormal 策略。
4. **Q/DQ reference**：导出显式 ONNX Q/DQ 或等价的独立 golden model；必须覆盖所有 14 个量化 Linear，且与硬件的 Top-K tie-break 一致。
5. **数值验证阈值**：对固定 held-in/held-out vector 集定义逐输出 `max_abs_error`、`mean_abs_error` 与 R² 退化阈值；分别比较 PyTorch QAT、Q/DQ reference 和 RTL/cycle model。
6. **输入前处理归属**：明确 neural/calibration 特征的生成、标准化、插值与 padding 是否在芯片外完成。依据当前数据管线，芯片接口接收的是已准备好的 `NEURAL_IN`/`CALIB_IN`；原始 spike 数据处理不在本 SPEC 范围。
7. **性能/功耗目标**：冻结 `B_MAX`、时钟、最大可接受 `t_decode/t_calib`、SRAM 容量、DRAM 带宽、PPA 目标及校准更新周期 `R`。
8. **复位与安全行为**：冻结 `ID_MAP` 是否掉电保留、模型热切换协议、CRC 错误/NaN 输入的故障上报与输出屏蔽策略。

在以上 8 项签核完成前，本文件可用于架构探索、RTL 规格和面积/带宽估算；完成后才可作为可比特验证的流片 SPEC 基线。

## 11. 验证交付物

流片验证至少应交付：

- 定版 checkpoint、权重/scale/bias 二进制包与 manifest；
- bit-accurate Q/DQ golden model；
- 覆盖 `B=1`（以及目标 `B_MAX`）的输入/输出向量集，含固定 Top-K tie 的案例；
- M0、M1、M2、M3、M4 的模块级中间张量 golden dump；
- RTL 仿真、门级仿真、FPGA/仿真器和 silicon bring-up 的逐层误差报告；
- 端到端 held-in/held-out R²、延迟、吞吐、峰值存储/带宽、功耗报告。

## 12. 代码与配置可追溯性

- 网络、Top-K、显式 Q/K/V/O：`src/models/components/spint.py`
- fake-quant、per-channel weight scale、per-tensor activation scale：`src/models/components/quantization.py`
- QAT observer 生命周期、蒸馏和 checkpoint 载入：`src/models/falcon_module.py`
- 模型配置：`configs/model/falcon_m2_dim256_prune64_a8w8_mixed_qat.yaml`
- 当前运行解析配置：`logs/train/runs/m2_quant_a8w8_qat_50ep/tensorboard/version_0/hparams.yaml`
