# 术语与符号

## 1. 信号类型

| 术语 | 含义 |
|---|---|
| EEG | scalp electroencephalography；本仓库主任务不是 EEG |
| ECoG | 皮层表面电信号；BrainDistill 等参考工作使用 |
| intracortical spikes | 皮层内电极检测到的 spike/event |
| SUA | spike-sorted single-unit activity |
| MUA | multi-unit/threshold-crossing activity |
| neuron/unit | 当前模型中的 set token；MUA 时也可对应 channel |

## 2. 模型符号

| 符号 | 含义 | 当前常见值 |
|---|---|---:|
| `M` | calibration trial 数 | 1–33 |
| `T` | 单 calibration trial 长度 | 100 |
| `N` | neuron/unit/channel token 数 | variable |
| `D`/`Denc` | NeuronID hidden | 64 |
| `W` | online history/identity 维度 | 50 |
| `H` | decoder model dimension | 512 baseline |
| `F` | FFN hidden dimension | 2048 baseline |
| `C` | behavior query/covariate 数 | M2=2 |
| `heads` | attention head 数 | M2=64 |
| `head_dim` | `H/heads` | M2=8 |
| `E` | session identity `[N,W]` | cached |
| `X` | online neural window `[W,N]` | per frame |
| `Z` | `transpose(X)+E` | `[N,W]` |
| `U` | read-in+LN neuron token | `[N,H]` |
| `R` | static-query compiled score weight | `[C×heads,H]` |

## 3. Attention 类型

### Decoder cross-attention

```text
behavior queries attend over neuron tokens
```

当前硬件主线。复杂度的 score 部分是 `C×N`。

### B15 cross-neuron self-attention

```text
NeuronID finalize 时，neuron token 互相 attention
```

每 session 一次，科学证据当前不稳定；不是 decoder cross-attention。

### B3A trial-axis attention

```text
每个 neuron 在 M 个 calibration trials 上做 attention
```

需要保留 `M×N×D` feature；第一版不做。

### Linear attention

使用 kernel feature map 改写或替代 softmax attention。它改变模型语义；
在当前 `C,N` 很小而 projection 很大的 SPINT 中，不一定是主要硬件杠杆。

## 4. Encoder 变体

| 变体 | 定义 | 第一版状态 |
|---|---|---|
| B3 | `100→64` pre-pool + mean + post MLP | bring-up baseline |
| B3T | fixed temporal basis `100→12→64` + mean + post MLP | parameterized candidate |
| B3A | trial-axis attention 替代 mean | deferred |
| B15 | cross-neuron self-attention | 不属于第一版 encoder baseline |
| B16 | mean/variance statistics | 未作为第一版硬件合同 |

## 5. 硬件术语

| 术语 | 含义 |
|---|---|
| PE | processing element；本设计为 INT8 MAC + INT32 acc |
| Mode A | `8 token × 8 output` |
| Mode B | `1 token × 64 output` |
| coefficient SRAM | weights/bias/scales/LUT/descriptor |
| activation SRAM | intermediate tensors |
| session SRAM/state | E、SUM_feat、session metadata |
| output-stationary | output partial sum 留在 PE/accumulator |
| requant | accumulator 乘 multiplier、round、shift、clamp |
| NLU | nonlinear/reduction unit；softmax/LN/rsqrt/exp 等 |
| liveness overlay | 生命周期不重叠 tensor 复用同一 SRAM 地址 |
| model image | 可加载的 weight/config binary |
| golden | 软件生成、供 RTL 逐层比较的期望值 |
| bit-exact | 所有整数 stage 逐元素完全相等 |

## 6. 证据术语

| 术语 | 含义 |
|---|---|
| development evidence | train/validation 范围内的结构选择证据 |
| formal held-out | 配置锁定后一次性独立测试 |
| paired delta | 同 session/seed/control 的指标差 |
| indeterminate | 效应未越过当轮测量噪声，不等于阴性 |
| analytic MAC | 公式计算，不等于实测周期/功耗 |
| PPA | power, performance, area |
| sign-off | 指定版本/范围完成验收，不等于普遍结论 |

## 7. 状态标签

| 标签 | 含义 |
|---|---|
| `FROZEN` | 可进入合同/RTL |
| `PROVISIONAL` | 做参数化实现，不固化 |
| `OPEN` | 待证据 |
| `DEFERRED` | 第一版不做 |

