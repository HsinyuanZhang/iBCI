# SRAM 与数据驻留

状态：组织方式 `PROVISIONAL`；最终容量 `OPEN`

## 1. 设计原则

1. coefficient SRAM 面积与 leakage 很可能是第一版芯片主项；
2. encoder 与 decoder 不同时运行，activation/session memory 应复用；
3. variable `N` 通过循环边界与 valid mask 实现；
4. 不为 B3A 或失败的 fixed-slot 路线预留专用大容量；
5. bank 组织必须同时支持 `8×8` 和 `1×64` PE 模式；
6. 所有容量结论必须在具体 SRAM compiler 宏上重算。

## 2. 逻辑存储层次

### 2.1 Coefficient SRAM

存储：

```text
INT8 weights
INT32 bias
INT32 requant multiplier
shift
activation scales/IDs
LayerNorm gamma/beta
compiled query/score constants
softmax/rsqrt LUT
layer descriptors
```

建议按模型模块分 bank：

```text
ENC_PRE
ENC_POST
READIN
ATTN_VALUE_SCORE
ATTN_OUT
FFN_IN
FFN_OUT
READOUT
NORM_AND_LUT
```

这样可以：

- 按 encoder/decoder mode 关闭 bank；
- 独立替换模型块；
- 统计每块 leakage；
- 对 FFN、projection 等大块做容量 DSE。

### 2.2 Activation/session SRAM

存储：

```text
calibration trial tile
SUM_feat
mean feature
identity E
online circular window X
read-in/normalized token U
attention scratch
query state
FFN ping-pong
debug dump
```

不建议给每个 tensor 固定物理区；通过 memory planner 按生命周期 overlay。

### 2.3 Small register files

```text
64×INT32 accumulator bank A
64×INT32 accumulator bank B
head/query max and denominator
layer loop counters
address/stride registers
performance counters
```

## 3. 容量量级

### 3.1 Encoder

以 `N=96,D=64,W=50,T=100`：

| State | 格式 | 容量 |
|---|---|---:|
| one trial `[T,N]` | INT8 | 9.38 KiB |
| `SUM_feat[N,D]` | INT32 | 24.0 KiB |
| `E[N,W]` | INT8 | 4.69 KiB |
| B3 weights | INT8 主体 | 约 17.6 KiB + metadata |
| B3T learned weights | INT8 主体 | 约 12.1 KiB + metadata |
| B3T fixed basis `[12,100]` | INT8/INT16/ROM | 1.2–2.4 KiB |

若 `N=160`：

```text
SUM_feat = 160×64×4 = 40 KiB
E = 160×50 = 7.8 KiB
trial = 100×160 = 15.6 KiB
```

### 3.2 Decoder activation

以 `N=96,H=512,W=50,C=2`：

| State | 格式 | 容量 |
|---|---|---:|
| `X[N,W]` | INT8 | 4.69 KiB |
| `E[N,W]` | INT8 | 4.69 KiB |
| normalized tokens `U[N,H]` | INT8 | 48 KiB |
| full V `[N,H]`（若物化） | INT8 | 48 KiB |
| scores `[C,heads,N]` | INT16 | 24 KiB |
| output accumulator `[C,H]` | INT32 | 4 KiB |
| max/sum `[C,heads]` | 16/32 bit | <1 KiB |

不建议同时物化 `U`、full V 和 full scores。优先比较：

1. `U` 常驻 + head-group streaming；
2. full V + scores 的简单两遍方案；
3. online softmax，仅保留 max/sum/output；
4. 重算 score 与减少 scratch 的 compute–memory trade-off。

### 3.3 Decoder weights

M2 `H=512,F=2048,W=50`，在 query cache 与 last-row pruning 后，在线
INT8 weight 量级约 3.17M。静态-query score folding 后约 2.98M。

这两个数字：

- 不含 bias、norm、scale、bank padding、ECC；
- 不含 encoder；
- 不代表最终选定 backbone；
- 用于说明 coefficient SRAM 是面积主项。

## 4. Activation SRAM DSE

建议比较：

```text
128 KiB
192 KiB
256 KiB
```

### 128 KiB

优势：

- 面积较小；
- 对 `N≤160,H≤512` 的 `U` + X/E + attention state 可能可行；
- 强制采用 liveness overlay/streaming。

风险：

- debug dump 占空间；
- 双缓冲和 bank padding 后可能不足；
- H512 的 simple materialized-attention dataflow 难以容纳。

### 192 KiB

优势：

- 给 ping-pong、debug 和 head-group buffer 更多余量；
- 仍可能保持可接受面积；
- 更适合支持 N160/H512。

### 256 KiB

优势：

- dataflow 简单；
- 可减少重算；
- 对多个 shape 更鲁棒。

风险：

- 可能不是能耗最优；
- leakage 增长；
- 容易掩盖 streaming schedule 的研究价值。

最终选择必须结合 SRAM compiler 的：

```text
area
read/write energy
leakage
max frequency
aspect ratio
port/bank overhead
```

## 5. Coefficient SRAM 组织

### 5.1 逻辑读宽

Mode B 需要：

```text
64 INT8 weights/cycle = 512 bit/cycle
```

推荐起点：

```text
8 banks × 64 bit
```

其他可选：

```text
16 banks × 32 bit
4 banks × 128 bit
```

选择取决于 compiler 支持、布线和读能耗，不应只看总 bit。

### 5.2 Weight blocking

推荐逻辑布局：

```text
for output_tile:
  for input_index:
    store 64 output-lane weights
```

Mode A 读取其中的 8-output slice；Mode B 读取完整 64-output word。

若 Mode A 每次只激活部分物理 bank，应验证：

- 未访问 bank 可否 clock/power gate；
- 512-bit macro 的部分字读取是否仍消耗全读能量；
- 8×64-bit 多 bank 是否优于单宽宏。

## 6. Activation banking

Mode A 要同时读取 8 个 token activation。推荐 neuron/token index 映射到 bank：

```text
bank = token_index mod 8
row  = floor(token_index / 8) × feature_stride + feature_index
```

对最后不足 8 个 token：

- lane mask；
- 不读无效地址；
- accumulator 不更新；
- performance counter 记录 tail utilization。

`X[W,N]` host layout 与 token-major internal layout之间不做全量物理转置：

- 输入按 time-major 写入 bank；
- address generator 按 neuron token 和 lag 读取；
- 或在 circular-buffer write 时完成 bank interleave。

两种方案由读写能耗与控制复杂度 DSE。

## 7. 生命周期与 overlay

### 7.1 Session encoding

```text
trial buffer
  overlaps decoder token U region

SUM_feat
  occupies shared state region

final E
  moves to retention region

mean/post scratch
  uses PE ping-pong region
```

### 7.2 Online decode

```text
X circular buffer + retained E
  persistent within session

U[N,H]
  reuses former SUM_feat/trial region

attention scratch
  reuses read-in intermediate

FFN ping-pong
  reuses head-group scratch after attention
```

建议 model compiler 输出 `memory_map.json`，并进行 live-range collision
静态检查。RTL 不应依赖人工维护地址表。

## 8. Softmax memory 方案

### 8.1 Materialized scores

优点：

- 验证简单；
- max/exp/sum/AV 分阶段清晰；
- 容易与软件逐层对齐。

代价：

- `C×heads×N` 的 INT16/INT32 scratch；
- 需要多遍 SRAM 读取；
- C16 时增长明显。

### 8.2 Online normalization

状态：

```text
running max
running denominator
running weighted-V accumulator
```

优点：

- 避免 score SRAM；
- 更适合 variable N；
- 减少 data movement。

风险：

- max 更新时需要重缩放旧 accumulator；
- 定点误差路径复杂；
- N 较短时节省的 SRAM/能耗未必抵偿逻辑；
- 需要 compiled-graph/QAT 验证。

建议第一版验证路径保留 materialized accurate model；最终硬件是否采用 online
方案由 cycle/energy/accuracy 决定。

## 9. ECC、CRC 与可靠性

第一版至少支持：

- model image CRC；
- descriptor CRC/hash；
- DMA payload length；
- illegal address assertion；
- SRAM out-of-range；
- optional parity for coefficient image；
- session reset 清除 `E_VALID`；
- model reload 清除旧 session state。

是否给大 coefficient SRAM 加 ECC 取决于工艺、运行环境与面积预算；未决定前
容量表不得漏掉 ECC/padding overhead。

## 10. 需要输出的 memory 报告

每个模型 release 和 hardware DSE 应输出：

```text
weight bytes by block
bias/scale/norm/LUT bytes
peak live activation bytes
session retained bytes
reads/writes per session
reads/writes per frame
bank active cycles
bank idle cycles
estimated dynamic energy
estimated leakage
```

不能只报告参数量或总 SRAM bit 数。

