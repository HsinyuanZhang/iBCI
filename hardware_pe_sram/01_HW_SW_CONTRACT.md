# 软硬件合同

状态：B3 部分 `FROZEN`；decoder 与 shape envelope `PROVISIONAL`

## 1. 合同目的

本合同把“软件仍在调模型”与“硬件可以并行开发”分开：

- 软件可以继续改变 weights、scales 和受支持范围内的 `H/F/C/N/M`；
- 硬件冻结 tensor 布局、算子语义、rounding、model image 和 debug 接口；
- 超出当前 envelope 的网络改变必须提升合同版本；
- model checkpoint 与硬件合同版本相互独立。

## 2. 顶层 API

### 2.1 Session encoding

```text
encode_session(
    calib_q: INT8[M,T,N],
    M: uint,
    N: uint,
    encoder_descriptor,
    encoder_model_image
) -> E_q: INT8[N,W], E_scale
```

第一版固定：

```text
T=100
W=50
Denc=64
```

可配置：

```text
M=1...33
N=runtime
encoder_variant=B3 or B3T
weights/bias/scales/mult/shift
```

### 2.2 Online decoding

```text
decode_frame(
    X_q: INT8[W,N],
    cached_E_q: INT8[N,W],
    N,
    decoder_descriptor,
    decoder_model_image
) -> y_q[C], y_scale
```

第一版语义：

- `X` 与 `E` 对齐到同一 neuron 顺序；
- `Z=X+E` 在声明的共享域或显式 requant 后执行；
- decoder 为一层 cross-attention；
- 输出只保留原 `fc_out` 的最后时间行，即每个 covariate 一个标量。

## 3. Tensor 布局

### 3.1 Host-facing layout

| Tensor | 逻辑 shape | 推荐线性顺序 |
|---|---|---|
| calibration trial | `[T,N]` | time-major, neuron-minor |
| calibration set | `[M,T,N]` | trial-major |
| online window | `[W,N]` | lag-major, neuron-minor |
| identity | `[N,W]` | neuron-major |
| encoder sum | `[N,D]` | neuron-major |
| decoder token | `[N,H]` | neuron-major |
| behavior output | `[C]` | covariate-major |

硬件不需要物理执行 PyTorch `permute`；通过 address generator 以 neuron token
视角读取时间向量。

### 3.2 Weight layout

所有 affine 遵循：

```text
y[out] = Σ_in x[in] * W[out,in] + bias[out]
```

逻辑 weight shape：

```text
[out_features, in_features]
```

物理 weight image 以 PE 输出 tile 为最内层：

```text
[out_tile][in_index][out_lane]
```

从而在 `1×64` 模式每个 `in_index` 可读取一个 64-weight logical word。
`8×8` 模式允许从同一布局选择一个 8-output subword，并在 8 个 token 间复用。

## 4. Shape envelope

当前建议：

| 字段 | 最小 | 初始评估最大 | 说明 |
|---|---:|---:|---|
| `M` | 1 | 33 | calibration trials |
| `T` | 100 | 100 | 合同 v0.1 固定 |
| `N` | 1 | 160 | DSE 范围；RTL counter 可留到 255 |
| `W` | 50 | 50 | 合同 v0.1 固定 |
| `Denc` | 64 | 64 | encoder hidden |
| `H` | 64 | 512 | 64 的 tile；最终上限待 SRAM |
| `F` | 128 | 2048 | 必须是 PE tile 可处理的整数 |
| `C` | 1 | 16 | behavior queries |
| `heads` | 1 | 64 | 与 `H/head_dim` 一致 |
| `head_dim` | 8 | 16 | 当前任务候选 |
| `layers` | 1 | 1 | 第一版 |

这里的最大值不是最终硬件承诺。SRAM 宏或精度实验否定某个范围时，应修改
合同版本并记录原因。

## 5. Layer descriptor

硬件不实现通用 ISA，但每个 affine/reduction 使用固定格式 descriptor：

```text
op_type
src_base
dst_base
weight_base
bias_base
scale_base
in_dim
out_dim
token_count
src_stride
dst_stride
pe_mode
activation
residual_enable
residual_base
requant_enable
input_scale_id
output_scale_id
debug_checkpoint_id
```

推荐 `op_type`：

```text
AFFINE
AFFINE_RELU
ADD
MEAN_REDUCE
LAYERNORM
SCORE_PROJECT
SOFTMAX_AV
FFN
READOUT_LAST
```

`FFN` 可以在控制层展开成两个 `AFFINE`；保留该枚举只用于 profiling 或
schedule grouping，不要求专用 FFN datapath。

## 6. Attention descriptor

```text
N
H
C
heads
head_dim
score_mode = STANDARD_K or STATIC_QUERY_R
softmax_mode
logit_scale
valid_token_count
V_weight_base
score_weight_base
out_weight_base
query_state_base
```

第一版推荐 `STATIC_QUERY_R`，但调试模型应保留标准软件 reference graph。
是否在 RTL 中保留完整 `STANDARD_K` 路径由面积 DSE 决定；不是强制要求。

## 7. Model release

最终每个可部署模型应导出：

```text
model_release/
├── manifest.json
├── graph_reference.json
├── graph_compiled.json
├── layer_descriptors.bin
├── weights_int8.bin
├── bias_int32.bin
├── requant_mult_int32.bin
├── requant_shift.bin
├── scales.json
├── norm_params.bin
├── softmax_lut.bin
├── session_constants.bin
├── memory_map.json
├── tiny_inputs/
├── tiny_stages/
├── full_inputs/
├── full_stages/
├── permutation_tests/
└── eval_report.json
```

`manifest.json` 至少包含：

```text
contract_version
model_variant
checkpoint identifier and hash
source code identifier or archive hash
shape envelope
actual shapes
bit widths
rounding mode
LUT version/hash
descriptor hash
weight image hash
evaluation protocol
evaluation sessions
formal/development evidence flag
```

## 8. B3 已冻结数值合同

| 项目 | 合同 |
|---|---|
| input | signed INT8 |
| weight | signed INT8, per-output-channel symmetric |
| bias | INT32 accumulator domain |
| dot accumulator | INT32 |
| activation scale | shared per-tensor per edge |
| `SUM_feat` | INT32 |
| mean | integer reciprocal multiply + arithmetic shift |
| requant multiplier | per output channel |
| requant product | at least INT64 |
| rounding | add half then arithmetic right shift |
| ReLU | after requant contract所定义的位置 |
| output E | signed INT8 + scalar scale |

当前 B3 `reciprocal_shift=20`、layer requant `shift=31` 是已有工作点；应放
model image/CSR，而不是作为不可修改 RTL 常数。

## 9. Decoder 待冻结数值项

以下为 `OPEN`：

- `X+E` 的公共 scale 或双输入 requant 规则；
- LayerNorm 输入、统计量、gamma/beta、rsqrt 和输出位宽；
- attention logit 位宽；
- exp LUT 的函数、输入范围和输出格式；
- softmax denominator/reciprocal 位宽；
- softmax probability 与 V 的乘法格式；
- residual add 的 scale 对齐；
- FFN/output activation scales；
- compiled `R` 的训练/量化方式；
- 最终行为输出位宽。

在这些项目冻结前，RTL 可以做 parameterized datapath 和 accurate baseline，
不能做 model-specific sign-off。

## 10. CSR 建议

### 控制

```text
CTRL_START_ENCODE
CTRL_START_DECODE
CTRL_ABORT
CTRL_SESSION_RESET
CTRL_DEBUG_ENABLE
```

### Shape

```text
CFG_M
CFG_N
CFG_T
CFG_W
CFG_DENC
CFG_H
CFG_F
CFG_C
CFG_HEADS
CFG_HEAD_DIM
```

### 状态

```text
STATUS_BUSY
STATUS_E_VALID
STATUS_MODEL_VALID
STATUS_DONE
STATUS_SHAPE_ERROR
STATUS_CRC_ERROR
STATUS_OVERFLOW
STATUS_SATURATION
```

### Counters

```text
PERF_TOTAL_CYCLES
PERF_PE_ACTIVE_CYCLES
PERF_WEIGHT_READS
PERF_ACT_READS
PERF_ACT_WRITES
PERF_SRAM_STALLS
PERF_NLU_STALLS
PERF_SATURATION_COUNT
```

## 11. 版本升级条件

以下变化要求提升 major/minor contract version：

| 变化 | 版本动作 |
|---|---|
| 改 rounding/overflow/softmax/LN 语义 | major |
| 增加新 bit width | major |
| 改 tensor layout | major |
| 从一层扩到多层 cross-attention | major |
| 新增已有 datapath 可表达的 layer descriptor | minor |
| 扩大 `N/M/H/F/C` 但不改语义 | minor + PPA re-signoff |
| 仅替换 weights/scales | model release，不改合同 |
| 仅替换 calibration/session state | session image，不改合同 |

