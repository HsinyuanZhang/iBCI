# 科研路线与实验计划

状态：`PROVISIONAL`

## 1. 研究问题

### Q1：双速率可重构是否真实有价值？

同一 PE–SRAM 阵列能否高效覆盖：

- session-rate DeepSets/NeuronID；
- token-rich decoder projection；
- query-poor FFN；
- attention reduction/NLU；

且统一架构的面积/leakage 小于两套专用硬件？

### Q2：固定 behavior query 是否能被硬件编译？

在一层 cross-attention 中，能否通过 query cache 与 QK folding：

- 去掉 K projection/K storage；
- 保持 standard softmax；
- 保持 variable N 与 permutation invariance；
- 在 W8A8 下保持任务性能？

### Q3：原始 backbone 是否严重 over-provisioned？

`H/F` 从 512/2048 降到 256/1024、128/512 或更小时：

- 跨 session R² 降多少；
- INT8 SRAM/MAC 降多少；
- distillation 能否弥补；
- 不同 `C`/信号域是否选择不同工作点？

### Q4：SRAM-first 优化是否比 attention 替换更重要？

对低频神经解码：

- coefficient leakage；
- bank activity；
- FFN weight residency；
- activation rematerialization；

是否比 softmax arithmetic 更决定平均功耗？

## 2. 候选池

| # | 候选 | 预期收益 | 风险 |
|---:|---|---|---|
| 1 | `8×8 ↔ 1×64` PE 重构 | 提高 token-rich/query-poor 利用率 | 互连/控制面积 |
| 2 | static query QK folding | M2 约 -23% MAC | 仅一层/小 C 最有利 |
| 3 | H/F scaling + distillation | 最大权重/MAC杠杆 | 精度风险 |
| 4 | B3T session adapter | encoder 参数/MAC下降 | 尚缺 W8A8 release |
| 5 | SRAM bank gating | 降低低 duty-cycle leakage | 需真实宏 |
| 6 | online integer softmax | 降 score SRAM/数据移动 | 数值复杂 |
| 7 | low-precision LayerNorm | 降 NLU/storage | 精度/QAT |
| 8 | FFN low-rank/width | 降最大权重块 | 重训练 |
| 9 | structured H/head pruning | 同时降 projection/state | 训练不稳定 |
| 10 | identity-based top-K neuron | 降 N 线性成本 | 当前 slot 路线掉点 |
| 11 | sparse spike read-in | 利用输入稀疏性 | 只影响第一层 |
| 12 | E→session bias folding | 精确减少加法/第一层计算 | 增加 N×H state |
| 13 | linear attention | 简化 softmax | 触及的 MAC 很小 |
| 14 | RMSNorm | 简化统计 | 改模型且创新弱 |

## 3. 收敛后的四条主线

### 主线 A：双数据流 PE–SRAM

论文硬件核心。

必须证明：

- Mode A/B mapping 的利用率；
- 重构互连面积；
- bank read energy；
- 与两套专用阵列相比的面积/leakage；
- B3/B3T、decoder projection、C2/C16 FFN 全覆盖。

### 主线 B：静态 query 编译

算法–编译–硬件协同核心。

必须证明：

- FP32 精确等价；
- variable N/permutation invariance；
- M2/H1/M1 不同 C/head_dim 的收益边界；
- compiled-graph W8A8；
- K SRAM/访问与 cycles 的真实下降。

### 主线 C：backbone–SRAM co-design

决定第一版 coefficient SRAM 是否可行。

必须证明：

- H/F Pareto；
- task-aware distillation；
- 多 seed/session；
- task R² 与 SRAM/PPA 同时报告；
- H512 baseline 是上界而不是默认 tapeout 规格。

### 主线 D：双速率 memory/power gating

系统贡献。

必须证明：

- encoder bank、decoder bank 的互斥生命周期；
- session retention state；
- 50 Hz duty cycle；
- bank leakage；
- unified vs separate model memory；
- energy/session 和 energy/frame 分开。

## 4. 三个关键实验

### E1：backbone–SRAM Pareto

矩阵：

```text
H = 64,128,256,512
F/H = 2,4
head_dim = 8,16
layers = 1
```

训练：

```text
FP32 student
teacher distillation
W8A8 QAT
```

数据：

- FALCON M2；
- DANDI SUA validation；
- FALCON M1 或 H1 shape/generalization。

输出：

```text
paired R²
per-session delta
weight bytes
activation bytes
MAC
cycles
energy
```

### E2：static-query编译

arms：

```text
A reference
B cached query + last-row readout
C B + QK folding
D C + compiled-graph W8A8 QAT
```

检查：

- FP32 max/mean error；
- stage error；
- score/attention distribution；
- permutation；
- variable N；
- M2/H1/M1 MAC/weight/read counts；
- task R²。

### E3：统一阵列公平 PPA

baselines：

```text
unified 64 PE
fixed 8×8
fixed 1×64
two dedicated engines
unified 128 PE
```

workloads：

```text
B3
B3T
fc_in
V/score
C2 FFN
C16 FFN
```

metrics：

```text
logic/SRAM area
frequency
utilization
cycles
reads/writes
dynamic energy
leakage
deadline
```

## 5. 研究候选的 kill criteria

### Static query folding

停止或降级，如果：

- FP32 无法稳定等价；
- compiled INT8 task delta 不可恢复；
- C/head_dim 范围中只有单一任务有收益；
- compiler/model image复杂度超过 K SRAM收益。

### H/F scaling

停止继续缩小，如果：

- paired worst-session出现灾难性下降；
- distillation/QAT无法恢复；
- seed噪声大于硬件化收益对应的可辨别性能差；
- H256 已满足面积/时序，继续到 H128 无系统收益。

### Online softmax

停止，如果：

- N≤160 时 score SRAM 很小；
- online rescale 增加能耗/面积；
- 定点误差需要大规模重训练；
- NLU不是 cycle/energy瓶颈。

### Fixed slots/top-K

不进入第一版，如果：

- 相对 variable-N baseline 仍有大幅 R² 损失；
- control 无法排除 selection bias；
- 固定 K 节省小于 H/F scaling。

## 6. 两周 feasibility pilot

### Day 1–2：合同与 exporter

产出：

- split API；
- graph reference/compiled schema；
- layer descriptor；
- memory image；
- tiny/full checkpoint表。

### Day 3–4：static-query FP32

产出：

- cached query；
- last-row readout；
- QK folding；
- FP32 equivalence；
- permutation/variable-N tests。

### Day 5–6：解析 DSE

产出：

- H/F/C/N/head sweep；
- weight/MAC/live-state；
- 64/128 PE ideal cycle；
- SRAM bandwidth。

### Day 7–9：cycle-accurate simulator

产出：

- Mode A/B；
- bank conflict；
- postprocess；
- encoder mean；
- LN/softmax；
- layer breakdown。

### Day 10–11：RTL microkernel

产出：

- tiny affine；
- 64-PE mode；
- INT32 acc；
- INT64 requant；
- B3 tiny/full compare。

### Day 12–13：小 backbone diagnostic

至少：

```text
H128/F512
H256/F1024
H512/F2048
```

先跑 development evidence，判断是否存在灾难性差距。

### Day 14：Go/No-Go

继续完整实现的条件：

- FP32 compiled graph通过；
- 64 PE 有 20 ms 潜力；
- SRAM 容量存在合理工作点；
- 至少一个 H≤256/512 候选有可接受性能；
- reconfiguration 不是明显面积负收益。

## 7. 两句话论文 pitch

现有跨 session neural decoder 将 session-rate identity calibration 与
frame-rate cross-attention 放进同一个通用 Transformer 数据流，造成实际
token/query 形状与 PE 利用率、SRAM residency 不匹配。我们提出一种双速率
PE–SRAM 架构，通过 token/output 并行度重构、静态 behavior-query 编译和
bank-level power gating，在保留 variable-neuron、permutation-invariant、
gradient-free calibration 语义的同时降低在线计算和存储成本。

## 8. 最强反对意见与回答

### 反对意见 1

“B3 每 session 一次，放 host 计算即可。”

回答：

- 不声称 B3 平均吞吐是主收益；
- 片上 B3 的价值是无 host autonomous calibration；
- 架构主收益来自共享阵列、在线 decoder 和 bank gating；
- 必须给出支持 encoder mode 的增量面积。

### 反对意见 2

“同一个 MAC array 跑两个 GEMM 不是可重构创新。”

回答：

- 重构对象是 token/output 并行度；
- 改变广播拓扑、reduction tree 切分和 SRAM 读宽；
- 用 fixed 8×8、fixed 1×64 和双专用阵列公平对照；
- 报告面积、leakage、利用率而非只画框图。

### 反对意见 3

“整数 softmax/LN 已经有很多工作。”

回答：

- 不把通用 NLU 当主要贡献；
- 采用已有 accurate/approximate baseline；
- 新意来自短 variable-N、少量固定 behavior query 和双速率系统；
- 重点报告 neural task 上的端到端影响。

### 反对意见 4

“SUA/MUA 本来就只是 variable N，不能支撑跨应用可重构。”

回答：

- 不把 SUA/MUA 切换本身当贡献；
- 跨应用证据要来自不同 `N/C/H/F`、session/frame 速率和 PE mode；
- 若未来包含 FIR/sorter，再额外证明短抽头与 GEMM 的数据流重构；
- 第一篇只主张当前有工件覆盖的范围。

## 9. 报告纪律

- `indeterminate` 不写成阴性；
- validation 不写成 formal test；
- MUA/SUA 绝对 R² 不横向排名；
- analytic MAC 不写成实测能耗；
- literature power estimate 不写成本芯片 PPA；
- query folding 必须说明一层/常量 query 边界；
- 量化后的 compiled graph 作为独立部署图；
- 每个优化单独 arm，避免错误归因。

