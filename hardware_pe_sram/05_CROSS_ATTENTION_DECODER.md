# Cross-Attention Decoder 硬件化

状态：算子语义 `FROZEN`；backbone/量化/compiled graph `PROVISIONAL` 或 `OPEN`

## 1. 原始语义

在线输入：

```text
X[W,N]
E[N,W]
```

token view：

```text
Z[N,W] = transpose(X) + E
```

随后：

```text
read-in:
  Linear(W→H)
  ReLU
  Linear(H→H)

cross-attention:
  query = fc_in(learned behavior representation)
  key/value = read-in neuron tokens
  pre-norm MHA
  residual
  pre-norm FFN H→F→H
  residual

readout:
  Linear(H→W)
  keep last time row only
```

第一版保持：

- 一层 cross-attention；
- standard softmax semantics 作为 reference；
- variable neuron `N`；
- learned behavior queries；
- neuron-axis permutation invariance；
- last-timestep behavior output。

## 2. M2 workload

```text
N=96
W=50
H=512
C=2
heads=64
head_dim=8
F=2048
layers=1
```

在 E 缓存、query 常量缓存和 last-row readout 后：

| 模块 | MAC/frame |
|---|---:|
| neuron read-in | 27,623,424 |
| MHA 在线 projections（K、V 与 output projection；Q 已缓存） | 50,855,936 |
| QK + attention×V | 196,608 |
| FFN | 4,194,304 |
| pruned readout | 1,024 |
| 合计 | 约 82.87M |

这张表决定优化优先级：

1. `H×H` projections；
2. `H×F` FFN weight SRAM；
3. model width；
4. softmax/LN 实现；
5. 最后才是 `N²` 类复杂度讨论。

本模型是 cross-attention，score 复杂度为 `C×N`，不是 self-attention 的
`N²`。

## 3. 合法常量折叠

### 3.1 Query cache

当：

```text
layers=1
inference mode
dropout disabled
behavior representation fixed
```

可在模型导出时缓存：

```text
fc_in(rep)
LayerNorm(query)
Q projection
```

在线不需要 query `fc_in` 和 Q projection。

### 3.2 Last-row readout

原：

```text
fc_out: H→W
```

部署只消费最后一个时间位置，因此只保留：

```text
one weight row: H→1
```

每个行为 query 输出一个标量。

### 3.3 V bias folding

softmax weights沿 neuron 求和为 1：

```text
Σ_n a_n (U_n W_V^T + b_V)
= Σ_n a_n U_n W_V^T + b_V
```

因此 V bias 可经 output projection 折入 output bias。是否实际采用取决于
整数 rounding 与 compiled-graph QAT。

## 4. 静态 query–key 编译

### 4.1 推导

对 behavior query `c`、head `j`：

```text
q[c,j] = constant
k[n,j] = U[n] W_K[j]^T + b_K[j]
```

score：

\[
s_{cjn}
=\frac{q_{cj}k_{nj}^{T}}{\sqrt d}
=U_n r_{cj}+\text{constant}_{cj},
\]

其中：

\[
r_{cj}=\frac{(W_K^j)^Tq_{cj}^{T}}{\sqrt d}.
\]

常数项对所有 neuron 相同，在 softmax 中抵消。部署图可以直接：

```text
score[n,c,j] = U[n,:] @ R[c,j,:]
```

在线不需要：

- K projection；
- K tensor；
- Q projection；
- Q weight。

### 4.2 M2 收益

```text
W_K: [512,512] = 262,144 INT8 weights
R:   [2×64,512] = 65,536 INT8 weights
```

解析 MAC：

```text
cached-query baseline ≈ 82.87M/frame
compiled candidate     ≈ 63.90M/frame
reduction              ≈ 22.9%
```

公平 coefficient baseline已经移除在线 Q weight 后：

```text
~3.17M weights -> ~2.98M weights
```

权重节省约 6%，计算节省更明显。

### 4.3 收益条件

标准 K projection：

```text
N H²
```

compiled score projection：

```text
N H (C×heads)
```

当：

```text
C×heads < H
```

即大致：

```text
C < head_dim
```

收益明显。

任务示例：

- M2：`C=2,head_dim=8`，有利；
- H1：`C=7,head_dim=16`，可能有利；
- M1：`C=16,head_dim=16`，K 计算收益接近消失。

论文必须跨任务报告，不得只选择 M2。

### 4.4 有效边界

该变换：

- 对一层且 query 常量的 FP32 real-arithmetic graph 精确；
- 保持 standard softmax；
- 保持 variable N；
- 保持 neuron permutation invariance；
- 不直接适用于后续 cross-attention layer；
- 不保证逐层独立量化后的 bit-exact。

正确流程：

```text
train/reference graph
  -> compile R and folded constants
  -> verify FP32 equivalence
  -> quantize/QAT compiled graph
  -> export compiled integer golden
```

错误流程：

```text
independently quantize W_Q and W_K
  -> multiply quantized matrices
  -> assume same rounding
```

## 5. 推荐在线数据流

### Stage 1：Z/read-in

```text
for neuron tile:
  read X and E
  align scale and add
  Linear(W→H) + ReLU
  Linear(H→H)
```

### Stage 2：LayerNorm

对每个 neuron token：

```text
sum
sumsq
mean
variance
rsqrt
gamma/beta
requant
```

输出 `U[N,H]`。

### Stage 3：score/value

compiled 模式：

```text
score = U × R^T
V = U × W_V^T
```

按 head group 执行，避免物化 full V/K。

### Stage 4：softmax + weighted V

可选：

1. materialized score baseline；
2. online normalization；
3. head-group scratch。

### Stage 5：out/residual/FFN

```text
concat head outputs
Linear(H→H)
add query residual
LayerNorm
Linear(H→F)
ReLU
Linear(F→H)
add residual
```

### Stage 6：readout

```text
Linear(H→1) for each of C queries
```

## 6. LayerNorm 设计

第一版 accurate baseline：

```text
INT8/INT16 input
INT32/expanded sum
expanded sumsq
fixed-point mean/variance
rsqrt LUT or piecewise approximation
INT16 gamma/beta
INT8 output
```

待比较：

- two-pass LN；
- one-pass sum/sumsq；
- low-precision statistics；
- RMSNorm 替代。

RMSNorm 会改变模型语义，必须重新训练；不能作为纯硬件等价优化。

## 7. Softmax 设计

### 准确 reference

```text
row max
subtract max
exp LUT
sum
reciprocal
normalize
```

### 可研究实现

- base-2/shift-friendly exponent；
- online normalization；
- 低 bit exponent result；
- denominator clipping；
- log2 probability representation。

这些方法已有较多硬件文献，建议作为实现组件而非单独论文主张。核心验证应在
本项目的短 `N`、variable `N`、小 `C` 和 task R² 上完成。

## 8. Backbone 扫描

第一轮：

```text
H ∈ {64,128,256,512}
F/H ∈ {2,4}
head_dim ∈ {8,16}
layers = 1
```

优先 distillation：

```text
teacher: current full SPINT
student: small backbone
loss:
  task prediction
  optional query representation
  optional attention/output matching
```

随后：

```text
compiled-graph QAT
```

必须报告：

```text
paired task R²
worst-session delta
INT8 weight bytes
peak activation bytes
MAC/frame
cycle/frame
energy/frame
```

## 9. 解析 Pareto 参考

假设：

```text
N=96,W=50,C=2,F=4H,head_dim=8
```

| H | baseline MAC/frame | compiled MAC/frame | baseline weights | compiled weights |
|---:|---:|---:|---:|---:|
| 64 | 1.59M | 1.28M | 52K | 49K |
| 128 | 5.68M | 4.47M | 203K | 191K |
| 256 | 21.38M | 16.61M | 799K | 750K |
| 512 | 82.87M | 63.90M | 3.17M | 2.98M |

这是解析成本，不是精度结果。它说明 `H/F` co-design 的收益远大于只改变
softmax。

## 10. 不建议的早期优化

### 只减少 heads

固定 H 时：

- Q/K/V/out weight 基本不变；
- projection MAC 基本不变；
- 只减少 head state/softmax control。

### 直接换 linear attention

当前真正 QK/AV 仅约 `0.20M/82.87M`。替换 softmax attention：

- 会改变模型；
- 不消除 read-in/V/out/FFN 大矩阵；
- 算术收益有限；
- 可以作为后续算法对照，不是首要硬件优化。

### 强制 fixed K slots

当前 fixed-slot router：

- 固定 shape 机械上可行；
- cached state 等价；
- 精度 gate 失败。

第一版应支持 variable N streaming，而不是依赖 router。

## 11. Decoder 验收

### FP32

- split API vs original；
- cached query；
- last-row pruning；
- static query folding；
- V bias folding；
- variable N；
- neuron permutation。

### Integer

- stage-by-stage；
- LN statistics；
- logits；
- softmax probability；
- weighted V；
- residual scale alignment；
- full behavior output；
- end-to-end R²。

### RTL/PPA

- cycles/frame；
- PE utilization per layer；
- NLU stall；
- SRAM reads/writes；
- peak state；
- deadline margin；
- energy/frame；
- C/N/H/F sweep。
