# 系统总体设计

状态：`PROVISIONAL`

## 1. 设计目标

目标是一颗可重构的 intracortical 运动解码加速器，使同一套 PE–SRAM
资源支持：

1. 每个 session 运行一次的 NeuronID calibration；
2. 每个解码窗口运行一次的 cross-attention decoder；
3. SUA/MUA 的 variable-`N` 输入；
4. 模型参数和 shape 的有限运行时可配置；
5. W8A8 整数推理与分层 bit-exact 验证。

第一版芯片优先回答：

> 一个共享的、可改变 token/output 并行度与 SRAM 驻留方式的计算阵列，能否在
> B3/B3T session adapter 和少量 behavior-query 的 cross-attention 之间复用，
> 并以可接受的面积代价满足实时约束？

## 2. 非目标

第一版不以以下事项为目标：

- 实现完整 spike sorting 前端；
- 支持任意 Transformer；
- 实现片上训练或反向传播；
- 支持无限层数和任意 tensor rank；
- 证明 SUA/MUA checkpoint 可以零样本互换；
- 以峰值 TOPS 作为主要贡献；
- 在缺少工艺与 SRAM macro 时给出可信植入功耗。

## 3. 双速率系统

### 3.1 Session-rate 路径

```text
host-preprocessed calibration trials [M,T,N]
    -> NeuronID encoder
    -> identity E [N,W]
    -> cache until session reset
```

特点：

- 每 session 一次；
- 无行为标签、无 optimizer、无权重更新；
- 计算量可以较大，但平均吞吐贡献很小；
- 关键硬件成本是 coefficient SRAM、`SUM_feat` 与可编程性；
- 允许使用同一 PE 阵列低速执行。

### 3.2 Frame-rate 路径

```text
live neural window X [W,N]
    -> transpose-free token view [N,W]
    -> add identity E
    -> read-in MLP
    -> LayerNorm
    -> one-layer cross-attention
    -> FFN
    -> last-timestep behavior output [C]
```

特点：

- 目标工作点按 20 ms/window，即 50 Hz 评估；
- `N` 随 session 变化；
- `H×H` projections 与 FFN 权重主导面积/访存；
- softmax/LN 主导非线性实现风险；
- PE 数、SRAM 带宽和时序由这条路径决定。

## 4. 系统框图

```text
                         ┌──────────────────────┐
host / DMA / sensor FIFO │ input and model DMA  │
───────────────┬────────►│ CRC / version check  │
               │         └──────────┬───────────┘
               │                    │
               │         ┌──────────▼───────────┐
               │         │ descriptor controller │
               │         │ loop/address/schedule │
               │         └──────┬─────────┬─────┘
               │                │         │
       ┌───────▼────────┐  ┌────▼─────┐  ┌▼────────────────┐
       │ activation and │  │ 64 PE    │  │ coefficient SRAM │
       │ session SRAM   │◄►│ array    │◄►│ weights/scales   │
       │ X/E/SUM/U      │  │ 8×8/1×64 │  │ LUT/model image  │
       └───────┬────────┘  └────┬─────┘  └──────────────────┘
               │                │
               │       ┌────────▼──────────────────────┐
               │       │ postprocess and reductions    │
               │       │ requant/ReLU/add/mean/max     │
               │       │ sumsq/reciprocal/rsqrt/exp    │
               │       └────────┬──────────────────────┘
               │                │
       ┌───────▼────────┐  ┌────▼─────────────┐
       │ cached E/state │  │ behavior output  │
       └────────────────┘  └──────────────────┘
```

## 5. 工作负载基线

### 5.1 Encoder 基线

原始 B3-D64：

```text
Linear(100→64) -> ReLU
mean over M
Linear(64→64) -> ReLU
Linear(64→64) -> ReLU
Linear(64→50)
```

M2 默认：

```text
M=33, T=100, N=96, D=64, W=50
params ≈ 18,034
MAC/session ≈ 21.4M
```

B3T 候选：

```text
fixed basis 100→12
Linear(12→64) -> ReLU
mean over M
same post MLP
```

在当前 `N=64,M=30` 成本工件中：

```text
B3:  18,034 params, 13.0M MAC/session, 16 KiB support state
B3T: 12,402 params, 4.5M MAC/session, 16 KiB support state
```

这些数字用于架构 DSE，不表示 B3T 已完成硬件量化 sign-off。

### 5.2 Decoder 基线

当前 M2 原始配置：

```text
N=96
W=50
H=512
C=2
heads=64
head_dim=8
layers=1
F=2048
```

在缓存 constant query 且只计算 `fc_out` 最后一行后：

```text
online MAC/frame ≈ 82.87M
online rate ≈ 4.14 GMAC/s at 50 Hz
```

其中真正 `QK + attention×V` 约为 `0.20M MAC/frame`；主项是 read-in 和
projection，而不是 attention score 本身。

## 6. 第一版建议范围

| 项目 | 建议 |
|---|---|
| PE | 64 INT8 MAC，支持 `8×8` 与 `1×64` 两种广播映射 |
| accumulator | INT32 |
| requant product | 至少 INT64 |
| encoder | 原始 B3 必须，B3T 通过 descriptor 支持 |
| decoder | 一层 cross-attention |
| runtime `N` | variable，初始 DSE 覆盖 16–160 |
| runtime `M` | 1–33 |
| `H` | tile 化支持 64/128/256/512 |
| `F` | 2H/4H，是否保留 2048 上限待 DSE |
| `C` | 1–16 |
| model memory | 可加载 SRAM image，不固化 checkpoint |
| nonlinearity | accurate integer baseline + optional approximation mode |
| host preprocessing | trial segmentation、插值、输入量化第一版放主机 |

## 7. 速率域调度

建议 controller 提供两种顶层命令：

```text
ENCODE_SESSION
DECODE_FRAME
```

### `ENCODE_SESSION`

```text
RESET_SESSION_STATE
LOAD/STREAM_CALIB_TRIAL
RUN_ENCODER_PRE
ACCUMULATE_SUM
repeat M
RECIPROCAL_MEAN
RUN_ENCODER_POST
WRITE_E
SET_E_VALID
```

### `DECODE_FRAME`

```text
CHECK_E_VALID
LOAD/UPDATE_X_WINDOW
RUN_READIN
RUN_LN1
RUN_SCORE_VALUE
RUN_SOFTMAX_AV
RUN_OUTPUT_PROJ_RESIDUAL
RUN_LN2_FFN_RESIDUAL
RUN_LAST_ROW_READOUT
WRITE_BEHAVIOR
```

Encoder 与 decoder 不并行运行，因此 activation/session SRAM 应尽量按
lifetime 复用。

## 8. 系统级不变量

以下性质必须跨 FP32、整数模型和 RTL 保持：

1. calibration 不使用 behavior label；
2. 部署期无 backward/optimizer；
3. 同一 neuron encoder 权重对所有 neuron 共享；
4. trial 顺序置换不改变 identity；
5. 同步置换 online/calibration 的 neuron 轴时，行为输出不变；
6. `N` 只影响循环边界/mask，不改变模型权重布局；
7. 同一 session 的 `E` 在显式 reset 前保持有效；
8. 任何 model image 都带版本、shape、量化和 hash；
9. 非法 `N/M/H/F/C` 必须报错，不能静默截断；
10. 性能统计按 session energy 与 frame energy 分开报告。

## 9. 总体验收门

第一版系统进入 tapeout 讨论前至少需要：

- B3 tiny/full bit-exact RTL；
- B3T 或最终 encoder 的独立整数 release；
- split API 与原始模型 FP32 等价；
- 一层 compiled cross-attention 的 FP32 等价；
- 完整 decoder 的 W8A8 task-level 验证；
- 64/128 PE cycle model；
- SRAM compiler 面积、时序、动态/漏电功耗；
- variable-`N` 与 permutation metamorphic regression；
- 20 ms deadline 的最坏配置时序；
- reconfigurable 与两套专用阵列的公平面积/能耗对照。

