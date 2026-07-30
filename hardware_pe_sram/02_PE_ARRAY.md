# PE 阵列设计

状态：`PROVISIONAL`

## 1. 设计判断

当前 workload 同时包含：

- 大量 neuron/calibration token 的小/中型 affine；
- `N×H` 到 `H×H` 的 token-rich projection；
- 只有 `C=1...16` 行的 query FFN；
- B3/B3T 的 DeepSets mean reduction；
- attention score/value 与非线性后处理。

固定方形 systolic array 在 `C=2` FFN 上利用率差；纯 `1×P` vector array
则失去 token 维的 weight reuse。建议将 64 个 PE 组织为可以改变广播关系的
`8×8` 阵列，而不是两套独立计算单元。

## 2. PE 基本单元

每个 PE 至少包含：

```text
signed INT8 × signed INT8 multiplier
INT32 accumulator
accumulator clear/load/hold
activation/weight input mux
output register
clock enable
```

不建议每个 PE 内都复制：

- INT64 requant multiplier；
- exp/reciprocal/rsqrt；
- 大型 reduction tree；
- 模型 scale SRAM。

这些功能应在阵列后端共享或按小组配置。

## 3. 执行模式

### 3.1 Mode A：`8T × 8O`

含义：

- 8 个 token 并行；
- 8 个 output channel 并行；
- 对同一个 inner-dimension index `k`：
  - 读取 8 个 activation；
  - 读取 8 个 weight；
  - 形成 64 个乘积；
  - 每个 PE 累加一个 token/output pair。

```text
             weight o0 ... o7
token t0     PE PE ... PE
...
token t7     PE PE ... PE
```

适合：

- B3/B3T pre-pool；
- encoder post-pool；
- decoder `fc_in`；
- V projection；
- token-rich score projection；
- `N` 较大的常规 affine。

优点：

- 每个 weight 被 8 个 token 复用；
- 每个 activation 被 8 个 output 复用；
- 只需 8 activation bytes + 8 weight bytes/cycle；
- `N=96`、`D=64`、`H` 为 64 倍数时尾部利用率高。

### 3.2 Mode B：`1T × 64O`

含义：

- 一个 token；
- 64 个 output channel；
- 每周期广播一个 activation，读取 64 个 weight；
- 64 个 PE 分别累计不同 output。

适合：

- `C=1/2` 的 query FFN；
- 单 token/小 token-count GEMV；
- B3 `64→64` bring-up；
- attention output/query 小矩阵；
- latency 优先而 weight reuse 次要的路径。

代价：

- coefficient SRAM 需要逻辑 512-bit/cycle；
- 对 token-rich workload 不如 Mode A 节省 weight reads。

### 3.3 Mode C：分组 dot/reduction

可将 64 PE 划成：

```text
8 groups × 8 lanes
4 groups × 16 lanes
```

用于：

- head_dim 8/16 的短 dot product；
- LayerNorm 的 sum/sumsq partial reduction；
- attention weighted-V 的分组乘加；
- B3/B3T basis projection 的小维度处理。

Mode C 是否需要额外 crossbar，必须通过 RTL 面积 DSE 决定。若控制/互连开销
过大，可由 Mode A/B 加共享 reduction tree 实现，不把 Mode C 作为独立硬件模式。

## 4. 后处理单元

阵列输出进入共享 postprocess：

```text
INT32 accumulator
    -> bias/optional residual alignment
    -> INT32 × INT32 multiplier
    -> INT64 product
    -> round
    -> arithmetic shift
    -> clamp
    -> optional ReLU
    -> INT8/INT16 writeback
```

吞吐起点：

```text
8 or 16 requant lanes
```

理由：一次 inner-product 需要几十到上千个 MAC cycle，而 requant 只在一个
output tile 完成后运行，通常不需要与 64 PE 一一对应。cycle model 必须检查
`in_dim=8/12/16` 的短层是否使 requant 成为瓶颈。

## 5. Reduction 单元

建议提供：

```text
sum
max
sumsq
valid-count
```

输出位宽：

- B3 `SUM_feat`：INT32；
- LayerNorm sum/sumsq：至少 INT32，最终位宽待 range analysis；
- attention max：logit domain；
- softmax denominator：待量化实验；
- performance counters：至少 32/64 bit。

reduction tree 应支持切分，以便同时服务：

- 64-lane 全阵列 reduction；
- 8 个 head_dim=8 group；
- 4 个 head_dim=16 group；
- 8-token Mode A 的独立统计。

这里的“切分点可重构”是芯片架构贡献的一部分，不能只在软件 loop 中表述。

## 6. 典型映射

### 6.1 B3 `100→64`

Mode A：

```text
token tile = 8 neurons
output tile = 8 hidden channels
inner loop = 100 time bins
8 output tiles per token tile
```

每 8 个 neuron：

```text
8 × 100 = 800 MAC-array cycles
```

忽略后处理和 SRAM stall。

Mode B：

```text
1 neuron × 64 outputs
100 cycles/neuron
```

理想 MAC 周期相同，但 Mode A weight reads 更少，因此默认选择 Mode A。

### 6.2 Decoder `N×H -> N×H`

以 `N=96,H=512`：

```text
Mode A
12 token tiles
64 output tiles
512 inner cycles
```

理论：

```text
96 × 512 × 512 / 64 = 393,216 cycles
```

### 6.3 Query FFN `C×H -> C×F`

以 `C=2,H=512,F=2048`：

Mode B 每个 query：

```text
32 output tiles × 512 cycles
```

两个 query 理论：

```text
32,768 cycles
```

若固定 8×8 Mode A，则 token row 只有 2/8 有效，阵列利用率约 25%；这正是
Mode B 必要性的直接例子。

## 7. PE 数 DSE

M2 cached-query baseline：

```text
82.87M MAC/frame
```

静态 query 编译候选：

```text
63.90M MAC/frame
```

100 MHz 理想时间：

| PE | 原 baseline | compiled candidate |
|---:|---:|---:|
| 32 | 25.90 ms | 19.97 ms |
| 64 | 12.95 ms | 9.98 ms |
| 128 | 6.47 ms | 4.99 ms |

32 PE 在最理想情况下也几乎无法覆盖完整在线路径，因此不建议作为 H512
baseline。64 PE 是最小合理起点；128 PE 是否值得取决于：

- 实际利用率；
- SRAM 读宽；
- NLU 周期；
- SRAM macro 面积；
- 是否通过 backbone 压缩到 H≤256；
- 功耗/时钟目标。

## 8. 带宽合同

### Mode A

目标每周期：

```text
8 activation bytes
8 weight bytes
```

### Mode B

目标每周期：

```text
1 activation byte
64 weight bytes
```

### Writeback

每个 output tile 完成后：

```text
64 results in Mode A
64 results in Mode B
```

writeback 可由 8/16-lane postprocess 分多个周期完成，只要不阻塞下一组
accumulator。建议双 accumulator bank：

```text
bank A compute
bank B requant/writeback
swap
```

## 9. Clock/power gating

至少支持：

- 未使用 PE lane clock gate；
- Mode B 只启用必要 output lanes；
- `C<8` 时关闭无效 token row；
- encoder 完成后关闭 encoder coefficient bank；
- B3T 模式关闭原 B3 pre-pool weight bank；
- decoder layer 间关闭未访问 memory bank；
- session idle 时只保留 `E`/configuration retention。

对 50 Hz workload，应同时报告：

```text
active energy/frame
idle/leakage between frames
average power at actual duty cycle
```

不能只按连续满载 TOPS/W 评价。

## 10. RTL 实现顺序

1. `1×8` tiny dot-product；
2. 扩为 `1×64`；
3. INT32 accumulator overflow/assertion；
4. 8/16-lane requant；
5. B3 tiny golden；
6. `8×8` broadcast mapping；
7. banked SRAM address generator；
8. B3 full-shape；
9. reduction tree split；
10. decoder affine microkernels；
11. NLU integration；
12. full compiled decoder。

## 11. 验收

每种模式必须验证：

- negative/zero/max INT8；
- partial tile；
- `in_dim` 非 64 倍数；
- `token_count` 非 8 倍数；
- output lane mask；
- accumulator clear/load/hold；
- reset/abort；
- backpressure；
- coefficient SRAM stall；
- postprocess overlap；
- Mode A/B 相同 affine 的逐元素一致；
- cycle counter 与 simulator 一致。

