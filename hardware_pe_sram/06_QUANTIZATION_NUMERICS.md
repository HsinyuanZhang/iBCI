# 量化与数值格式

状态：B3 `FROZEN`；decoder `OPEN`

## 1. 原则

1. 任务指标优先于 identity cosine/RMSE；
2. FP32 reference、fake quant、integer engine 和 RTL 必须分开；
3. 量化 scale 只能使用允许的 train/calibration scope；
4. 所有 rounding、saturation、overflow 必须显式；
5. compiled graph 改变运算结合顺序后必须重新 QAT；
6. 不因单个 checkpoint 通过就宣布跨 fold/seed sign-off。

## 2. B3 冻结合同

| Edge/operation | 格式 |
|---|---|
| calibration input | signed INT8 |
| weight | signed INT8 |
| weight scale | per-output-channel symmetric |
| bias | INT32 accumulator domain |
| dot accumulator | INT32 |
| activation | per-edge shared per-tensor INT8 |
| pre-pool ReLU output | INT8 |
| `SUM_feat` | INT32 |
| reciprocal product | INT64 |
| mean output | INT8 |
| post0/post1 | W8A8, INT32 acc |
| E | signed INT8 + scalar scale |
| requant multiplier | INT32 per output |
| requant product | INT64 |

rounding：

```text
(product + 2^(shift-1)) >>> shift
```

其中 `>>>` 必须与软件 arithmetic right shift 语义一致。

## 3. Scale 管理

### B3

已存在的六个 edge scale：

```text
input
pre_out
mean
post0_out
post1_out
E
```

这些是 model release 内容，不是 RTL 常数。

### Decoder

至少需要：

```text
X
E
Z/shared-add-domain
readin0
readin1
LN1 output
score/logit
V
softmax probability
attention output
out projection
residual1
LN2 output
FFN hidden
FFN output
residual2
behavior output
```

不要在没有端到端实验时把全部 activation 强行共用一个 scale。

## 4. `X+E` 边界

三种候选：

### A. Shared INT8 scale

```text
scale_X == scale_E == scale_Z
Z_q = saturate(X_q + E_q)
```

优点：硬件最简单。  
风险：X/E 分布不同，可能损失精度。

### B. Shared INT16 add domain

```text
X_q -> INT16 shared domain
E_q -> INT16 shared domain
add
```

优点：保留范围。  
风险：增加带宽/第一层输入位宽。

### C. Separate requant to INT8 Z

```text
Z_q = requant_X(X_q) + requant_E(E_q)
```

优点：仍以 INT8 进入 PE。  
风险：增加两个 multiplier/shift 和 rounding 路径。

必须通过 decoder task R² 选择，不能只比较 Z RMSE。

## 5. Accumulator range

### B3

最坏粗界：

```text
K × 127 × 127
```

`K≤100/64` 时 INT32 充足，当前软件工作点无 overflow。

### Decoder

需分别分析：

```text
W=50
H up to 512
F up to 2048
head_dim 8/16
softmax×V
LayerNorm sumsq
```

尤其：

- affine INT32 通常足够；
- LayerNorm sumsq 可能需要更宽；
- softmax denominator 与 scaled numerator需独立格式；
- residual add 前必须 scale-align；
- INT16 activation 若与 INT8 weight 相乘可能使 INT32 累加边界收紧。

range analysis 必须基于：

1. 理论上界；
2. train/validation trace；
3. saturation/overflow counters；
4. adversarial max-input RTL tests。

## 6. LayerNorm 格式候选

建议 baseline：

```text
input INT8
sum INT32
sumsq INT40/INT48 candidate
mean/variance fixed point
rsqrt LUT + interpolation
gamma/beta INT16
output INT8
```

待扫：

```text
statistics precision
variance epsilon
rsqrt LUT entries
interpolation bits
gamma/beta bits
output clipping
```

必须保存：

```text
mean
variance
rsqrt
normalized pre-affine
final LN output
```

作为 golden checkpoint。

## 7. Softmax 格式候选

准确 baseline：

```text
logit INT16
row max INT16
delta INT16
exp LUT output UINT16
denominator INT32
reciprocal UINT16/INT32
probability UINT16
V INT8
weighted accumulator INT32
```

这只是起点，最终位宽必须由 QAT/trace 决定。

待扫：

- logit clip；
- exp 输入范围；
- LUT 分段；
- probability bits；
- denominator bits；
- reciprocal shift；
- materialized vs online；
- base-e vs base-2；
- max-update rescale precision。

## 8. Static `R` 量化

`R` 来自：

```text
R = function(W_K, W_Q, query constants, LN params)
```

三种候选：

1. FP32 编译后对 R 做 per-output-channel INT8 PTQ；
2. 将 R 注册为 compiled model 参数做 QAT；
3. 训练期直接使用 compiled score projection。

推荐顺序：

```text
FP32 equivalence
-> PTQ diagnostic
-> compiled-graph QAT
```

不能要求 R 与独立 WQ/WK 量化路径 bit-exact；两者是不同部署图。需要保持的是：

- FP32 语义等价；
- compiled integer graph 自身 bit-exact；
- task-level性能满足门槛。

## 9. QAT 验证路径

每个候选至少报告：

```text
anchor FP32
compiled/shadow FP32
fake quant
integer engine
RTL
```

检查：

```text
FP32 split equivalence
fake vs integer exact
integer vs RTL exact
task R² delta
per-session delta
saturation per edge
overflow count
multi-calibration draw
fold/seed scope
```

## 10. Mixed precision 的使用原则

只在定位到具体瓶颈时增加位宽：

- W8A32 消融判断 weight error；
- W32A8 判断 activation error；
- E INT16 判断 identity output；
- LN/softmax selective INT16/32；
- accumulator保持 INT32/更宽统计。

不建议一开始把完整 decoder 升为 INT16，因为：

- coefficient/activation SRAM 翻倍；
- PE 面积/能耗显著增加；
- 当前 B3 已显示“更高 identity 位宽”不必然改善 task R²；
- 可能掩盖真正的 scale/clip/QAT 问题。

## 11. 数值验收

### 必须 exact

- integer engine vs RTL 的所有合同 stage；
- weight/bias/mult/shift image；
- reciprocal mean；
- requant rounding；
- illegal overflow behavior；
- model hash。

### 允许 tolerance

- FP32 reference vs algebraic compiled graph，按定义 tolerance；
- approximate LN/softmax vs accurate integer reference；
- task prediction vs FP32。

允许 tolerance 的 stage 必须同时给：

```text
max_abs
mean_abs
RMSE
percentile error
task R²
worst-session delta
```

## 12. 禁止项

- 用 heldout/test session 估计量化 scale；
- 每个 eval session 单独选择 activation range；
- 只报 E cosine，不报 frozen-decoder task R²；
- 直接把训练期 CUDA fake path 当硬件合同；
- 将 best checkpoint 和 last checkpoint混用；
- 改 rounding 后不提升合同版本；
- 将 approximation error 与模型压缩 error 混在一个 arm。

