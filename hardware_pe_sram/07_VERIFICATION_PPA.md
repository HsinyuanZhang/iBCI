# 验证、验收与 PPA

状态：流程 `FROZEN`；阈值/PPA 数字 `OPEN`

## 1. 验证层级

```text
Level 0  FP32 reference graph
Level 1  FP32 compiled graph
Level 2  fake-quant graph
Level 3  integer software engine
Level 4  cycle-accurate PE/SRAM simulator
Level 5  RTL simulation
Level 6  gate-level simulation
Level 7  FPGA/prototype
Level 8  silicon
```

任何层级失败都应先在相邻层定位，不从最终 R² 反推底层原因。

## 2. 软件等价测试

### Encoder

- batch vs streaming；
- B3 `push_trial/finalize`；
- B3T；
- M sweep；
- trial permutation；
- neuron permutation；
- cached E reuse。

### Decoder

- original `forward` vs split API；
- cached query；
- last-row readout；
- static query folding；
- variable N；
- key padding mask；
- neuron permutation；
- C/H/F/head sweep。

## 3. Integer golden 层

### B3 checkpoints

```text
S0 input
S1 pre accumulator
S2 pre ReLU/requant
S3 SUM_feat
S4 mean
S5 post0
S6 post1
S7 E
```

### Decoder checkpoints

```text
D0 X/E/Z
D1 readin0
D2 readin1
D3 LN1 mean/var/output
D4 score/R projection
D5 V
D6 softmax max/sum/prob
D7 weighted V
D8 out projection/residual
D9 LN2
D10 FFN0
D11 FFN1/residual
D12 behavior output
```

每个 checkpoint 应有：

```text
logical shape
linear layout
dtype
scale
hash
valid range
expected saturation
```

## 4. Metamorphic tests

必须包含：

1. neuron 同步置换；
2. trial 顺序置换；
3. batch duplication；
4. padding/mask equivalence；
5. split/cached vs unsplit；
6. query cache invalidation on model reload；
7. E invalidation on session reset；
8. N tail tile；
9. M=1 与 M=max；
10. zero-input/max-input/negative-input；
11. abort and restart；
12. DMA backpressure。

## 5. RTL 单元验收

### PE

- signed multiplication；
- accumulator clear/hold；
- overflow assertion；
- Mode A/B mapping；
- partial output mask；
- partial token mask。

### SRAM

- bank mapping；
- simultaneous read/write；
- conflict/stall；
- address range；
- model CRC；
- retention/reset。

### Postprocess

- INT64 product；
- rounding；
- shift；
- saturation；
- ReLU；
- residual add。

### NLU

- reduction；
- max；
- sum/sumsq；
- reciprocal；
- rsqrt；
- exp LUT；
- materialized/online softmax。

## 6. Full-model acceptance

### Functional

```text
integer software == RTL at every frozen stage
```

### Task

报告：

```text
mean R²
per-session R²
paired delta
worst-session delta
seed/fold
development/formal scope
```

正式阈值应在实验前固定。当前 formal-test receipt 存在治理问题，因此新的
硬件化实验只能先形成 development evidence，不能伪装成最终 held-out 结论。

## 7. Cycle model

至少建模：

```text
PE compute
tile tails
weight SRAM bandwidth
activation SRAM bandwidth
bank conflicts
postprocess overlap
reduction
LN
softmax
DMA
controller bubbles
debug disabled/enabled
```

输出：

```text
cycles by layer
PE utilization by layer
stall breakdown
reads/writes by memory/block
peak live memory
end-to-end cycles
deadline margin
```

## 8. 性能指标

### Session-rate

```text
latency/session
energy/session
cycles vs M/N
encoder PE utilization
peak state
```

### Frame-rate

```text
latency/frame
energy/frame
average power at 50 Hz
worst supported shape
deadline margin
```

必须分开报告，不能把 session adaptation MAC 摊到 frame 后掩盖峰值 SRAM，
也不能把 frame-rate 主项用 session average 弱化。

## 9. PPA 流程

### Pre-PDK/analytic

允许报告：

- MAC；
- parameter/byte；
- SRAM read/write count；
- theoretical cycles；
- utilization；
- bandwidth；
- live state。

不得报告可信 mW/mm²。

### SRAM compiler 后

加入：

```text
macro area
read/write energy
leakage
frequency
bank/port overhead
ECC/padding
```

### RTL synthesis 后

加入：

```text
logic area
critical path
clock tree estimate
PE/postprocess/NLU/controller breakdown
reconfiguration interconnect overhead
```

### Post-layout

加入：

```text
wire/congestion
real clock
dynamic power from activity
IR/drop/thermal if applicable
```

## 10. 架构公平对照

论文必须比较：

```text
A. unified reconfigurable 64-PE
B. fixed 8×8-only
C. fixed 1×64-only
D. two dedicated engines
E. optional 128-PE unified
```

同一：

- process；
- frequency/voltage；
- SRAM capacity；
- bit width；
- model；
- workload；
- activity source；
- timing constraint。

报告：

```text
logic + SRAM area
reconfiguration overhead
energy/session
energy/frame
leakage
utilization
latency
```

只比较 TOPS/W 不足以支持可重构主张。

## 11. 推荐回归矩阵

| Profile | M | N | H | F | C | 用途 |
|---|---:|---:|---:|---:|---:|---|
| tiny encoder | 1 | 1/9 | 64 | — | — | 逐元素 |
| B3 M2 | 33 | 96 | — | — | — | encoder full |
| B3T M2 | 33 | 96 | — | — | — | candidate |
| decoder tiny | — | 9 | 64 | 128 | 2 | attention bring-up |
| M2 small | — | 96 | 128 | 512 | 2 | compressed |
| M2 medium | — | 96 | 256 | 1024 | 2 | compressed |
| M2 baseline | — | 96 | 512 | 2048 | 2 | upper baseline |
| SUA variable | — | 38/64/91/137 | selected | selected | task | N sweep |
| M1-shape | — | dataset | selected | selected | 16 | C stress |
| tail stress | 7/33 | 159 | 192/320 | odd tile | 7 | mask/tail |

## 12. Tapeout 前 sign-off 清单

- [ ] contract version frozen
- [ ] model release frozen
- [ ] all hashes recorded
- [ ] B3/B3T decision recorded
- [ ] backbone H/F/C/head frozen
- [ ] quant scales frozen
- [ ] LN/softmax LUT frozen
- [ ] integer software vs RTL exact
- [ ] gate-level regression
- [ ] SRAM macro integrated
- [ ] worst-shape timing
- [ ] worst-shape energy
- [ ] session reset/error handling
- [ ] permutation tests
- [ ] formal/development evidence labeled
- [ ] PPA comparison with dedicated baselines
- [ ] model reload and CRC
- [ ] silicon debug observability

