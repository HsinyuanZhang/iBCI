# SPINT Few-shot Learning 计算、数据流与 ASIC 部署分析

> 目标：面向实验室科研流片，拆清 SPINT-M2 的 gradient-free few-shot adaptation 到底计算什么、数据怎样流动、哪些状态需要保存，并给出可实现的芯片划分。
>
> 结论依据：仓库当前源码与 M2 配置。MAC、参数量和张量容量按源码逐项推导；论文/本地实验结果单独标注。PPA 必须在确定工艺、SRAM macro、频率和量化格式后通过综合与功耗仿真获得。

## 1. 先澄清：这里的 few-shot 不是片上反向传播

SPINT 的测试期适配没有 optimizer、gradient 或权重更新。它的过程是：

1. 片外预训练：使用 held-in sessions 的有标签数据训练 IDEncoder 和 decoder 的所有权重。
2. 新 session 校准：输入少量无标签 calibration trials，固定权重的 IDEncoder 前向生成每个神经元的 identity embedding `E`。
3. 在线 query 解码：把 `E` 加到每帧神经时间窗上，固定权重 decoder 持续输出运动学预测。

因此更准确的硬件表述是：

- **on-chip gradient-free few-shot adaptation**；或
- **support-conditioned / amortized adaptation**。

如果芯片只接收主机已经算好的 `E`，芯片本身并没有执行 few-shot 计算，只执行 session-conditioned inference。若论文贡献要强调“片上 few-shot”，至少应把 IDEncoder 前向或一个更简单的 CORAL 统计适配器放到芯片上。

## 2. M2 基准规格与符号

| 符号 | 含义 | M2 值 | 源码/配置 |
|---|---|---:|---|
| `B` | batch size，部署时为 1 | 1 | 部署接口 |
| `M` | calibration trial 数 | 33，可扫 1/7/13/19/25/33 | data config |
| `T` | 每条 calibration trial 归一化长度 | 100 | `max_trial_length` |
| `N` | 神经元/通道数 | 96 | M2 Utah array |
| `W` | 在线因果窗口长度 | 50 | 1 s @ 20 ms/bin |
| `H` | hidden/model dimension | 512 | model config |
| `C` | 行为 query/输出维数 | 2 | x/y velocity |
| `F` | Transformer FFN dimension | 2048 | 构造函数默认值 |
| `h` | attention heads | 64 | 每个 head dimension = 8 |
| `f_bin` | 在线输入/解码速率 | 50 Hz | 20 ms/bin |

核心输入、状态和输出：

```text
calibration support set:  Xcal [M, T, N] = [33, 100, 96]
current causal window:     X    [W, N]    = [50, 96]
session identity cache:    E    [N, W]    = [96, 50]
online output:             y    [C]       = [2]
```

## 3. 训练期、校准期、在线期必须分开分析

### 3.1 离线 meta-training

训练 sample 返回：

```text
neural_window                   [W, N]
behavior_target                 [W, C]
calib_trialized_neural_features [M, T, N]
session_name
```

同一个 batch 只包含一个 session，但 `random_calibration=true` 时，每个 sample 可以抽到不同的连续 calibration trial 子集。训练时所有网络参数通过最后一帧 MSE 和 Adam 更新。

M2 每个 sample 的 calibration 张量为：

```text
33 * 100 * 96 = 316,800 values
FP32 = 1,267,200 B = 1.208 MiB/sample
batch 32 = 38.67 MiB，仅计 calibration 输入
```

`phi` 的最大逻辑中间张量为：

```text
B * M * N * H
= 32 * 33 * 96 * 512
= 51,904,512 values
= 198 MiB @ FP32 / 每个被 autograd 保留的中间层
```

所以训练显存和训练算力会被 IDEncoder 主导。这个成本不应进入部署 ASIC 预算；第一版芯片不应尝试片上训练。

### 3.2 新 session 的一次性校准

校准分两层：

1. **非学习预处理**：20 ms bin、eval mask、trial segmentation、cubic interpolation 到 `T=100`、量化。
2. **学习型 IDEncoder 前向**：`MLP1 -> mean over M -> MLP2 -> E`。

源码中的 cubic interpolation 是 SciPy 数据处理，不是网络层。第一版 ASIC 应把它留给主机，或通过固定长度采集协议直接得到 `T=100`；不要为一次性预处理增加 cubic resampler 数据通路。

### 3.3 50 Hz 在线解码

每 20 ms 输入 96 路新 spike-count bin：

```text
new x[N]
 -> update N x W circular history
 -> add cached E[N,W]
 -> per-neuron read-in MLP
 -> cross-attention + FFN
 -> last-timestep x/y prediction
```

参考 Python 每帧执行 `np.roll`、复制、转置，并重新运行 IDEncoder。ASIC 必须用 circular buffer 和 session-level `E_valid` 状态消除这些重复。

## 4. Few-shot IDEncoder 的精确计算图

源码 `num_id_layers=3` 的实际含义是输入侧 3 个 affine，加输出侧 3 个 affine，共 6 个 affine：

```text
对每个 trial m、神经元 n：

xcal[m,:,n] [T=100]
 -> Linear 100 -> 512 -> ReLU
 -> Linear 512 -> 512 -> ReLU
 -> Linear 512 -> 512
 = phi[m,n,:] [H=512]

对每个神经元 n：

phi_bar[n,:] = (1/M) * sum_m phi[m,n,:]
 -> Linear 512 -> 512 -> ReLU
 -> Linear 512 -> 512 -> ReLU
 -> Linear 512 -> 50
 = E[n,:] [W=50]
```

trial mean 位于两个 MLP 之间，因此：

- `fc_id_in` 的三层都必须运行 `M*N` 次；
- `fc_id_out` 只运行 `N` 次；
- 不能把整个 IDEncoder 先对 trial 求均值，否则 ReLU/MLP 非线性会改变模型。

## 5. IDEncoder 参数量与 MAC 推导

### 5.1 参数

`fc_id_in`：

```text
(T*H + H) + 2*(H*H + H)
= (100*512 + 512) + 2*(512*512 + 512)
= 577,024 parameters
```

`fc_id_out`：

```text
2*(H*H + H) + (H*W + W)
= 2*(512*512 + 512) + (512*50 + 50)
= 550,962 parameters
```

总计：

```text
IDEncoder = 1,127,986 parameters
FP32      = 4.30 MiB
全部按 8 bit 粗计 = 1.076 MiB
```

更实际的 W8A8 存储是：

```text
weight int8  = 1,125,376 B
bias int32   = 2,610 * 4 = 10,440 B
total        = 1,135,816 B = 1.083 MiB
```

### 5.2 MAC

定义一个 `phi` token 的 MAC：

```text
MAC_phi_token = T*H + 2*H^2
              = 100*512 + 2*512^2
              = 575,488 MAC
```

输入侧：

```text
MAC_id_in = M*N*(T*H + 2*H^2)
          = 33*96*575,488
          = 1,823,145,984 MAC
```

输出侧：

```text
MAC_id_out = N*(2*H^2 + H*W)
           = 96*(2*512^2 + 512*50)
           = 52,789,248 MAC
```

总计：

```text
MAC_ID(M) = M*55,246,848 + 52,789,248
MAC_ID(33) = 1,875,935,232 MAC
           = 3.752 GOP，若 1 MAC = 2 operations
```

不同 few-shot 预算：

| `M` | IDEncoder MAC | 等价 GOP |
|---:|---:|---:|
| 1 | 108.04 M | 0.216 G |
| 2 | 163.28 M | 0.327 G |
| 4 | 273.78 M | 0.548 G |
| 8 | 494.76 M | 0.990 G |
| 16 | 936.74 M | 1.873 G |
| 33 | 1,875.94 M | 3.752 G |

注意固定的 `fc_id_out` 使小 `M` 时成本不完全按 M 成比例。

## 6. IDEncoder 的片上数据流与 SRAM 下界

### 6.1 不需要物理执行 `permute`

源码把 `[M,T,N]` 变为 `[M,N,T]`。硬件可通过 SRAM address generator 直接按 neuron 读时间向量，不需要搬运一次完整转置。

若主机以 trial-major、time-major 方式流入 `[T,N]`，可对 layer-1 使用如下循环：

```text
for m in trials:
  for n_tile in neurons:
    for h_tile in hidden_outputs:
      acc[n_tile,h_tile] = sum_t Xcal[m,t,n_tile] * W1[t,h_tile]
```

### 6.2 推荐的 trial-major streaming schedule

```text
LOAD one preprocessed trial [T,N]
 -> phi layer 1/2/3, token tiled
 -> requantize phi[m,n,h]
 -> accumulate SUM_PHI[n,h]
repeat M times
 -> scale by 1/M
 -> psi layer 1/2/3 per neuron
 -> write E[n,w]
 -> assert E_valid
```

主要 SRAM：

| 状态 | shape | INT8/INT16/INT32 容量 | 说明 |
|---|---:|---:|---|
| ID weights + biases | 1.126M weights | 1.083 MiB | 主体面积 |
| one trial buffer | `100*96` | 9.38 KiB @ int8 | 主机已完成插值 |
| `SUM_PHI` | `96*512` | 96 KiB @ int16，192 KiB @ int32 | 跨 M 累加 |
| token ping-pong | `2*512` | 1 KiB @ int8 | layer 间 tile buffer |
| identity `E` | `96*50` | 4.69 KiB @ int8，9.38 KiB @ int16 | session state |

`phi` 每个 trial 若先 requantize 到 int8，`M<=33` 的和只需要约 13-14 个 magnitude bits，INT16 `SUM_PHI` 有机会成立；必须用量化仿真验证饱和率。保守版本用 INT32。

因此 IDEncoder-only W8A8 方案的片上 SRAM 下界约为：

```text
1.083 MiB weights/bias
+ 0.094 or 0.188 MiB SUM_PHI
+ about 0.015 MiB trial/token/E/control buffers
= about 1.19 MiB (INT16 sum) to 1.29 MiB (INT32 sum)
```

这是架构下界，不包含 ECC、bank padding、DMA FIFO、双缓冲和 SRAM compiler 对齐开销。

### 6.3 平均值不需要除法器

第一层 `psi` 是 affine：

```text
Wpsi * (SUM_PHI/M) + b
```

固定 `M=33` 时可把 `1/M` 合并进该层的 requantization multiplier。若支持多个 `M`，用小型 reciprocal LUT 保存允许值 `{1,2,4,7,8,13,16,19,25,33}` 的定点乘数和 shift；不需要通用 divider。

## 7. IDEncoder 性能配置

理想计算时延：

```text
latency = MAC_ID / (P_MAC * f_clk * utilization)
```

忽略访存/控制、假设 1 MAC/PE/cycle 且 utilization=100%，`M=33`：

| PE 数 | 100 MHz 理想时延 | 评价 |
|---:|---:|---|
| 8 | 2.345 s | 一天一次仍可接受 |
| 16 | 1.172 s | 推荐的低面积起点 |
| 32 | 0.586 s | 更好的实验交互性 |
| 64 | 0.293 s | 对一次性任务通常过度 |

`M=1`、16 PE、100 MHz 的理想时延约 67.5 ms。

IDEncoder 是一次/session 的任务。实验室流片应优先节省 SRAM/PE 面积，不必为亚毫秒时延铺大阵列。16-32 个可复用 INT8 MAC lane 已经足够；真实周期数应加入 tile 装载、requant、ReLU 和 SRAM conflict 后由 cycle model 给出。

## 8. 在线 SPINT decoder 的计算与数据流

缓存 `E` 后，每帧主要步骤：

```text
1. Z = X + E                                    [N,W]
2. fc_in: W -> H -> H                           [N,H]
3. LayerNorm keys/values                        [N,H]
4. Q/K/V projections
5. 64-head attention over N=96
6. output projection + residual                 [C,H]
7. LayerNorm + FFN H -> 2048 -> H               [C,H]
8. fc_out H -> W，软件只取最后一列           [C,W] -> [C]
```

逐项 affine/matmul MAC：

| 模块 | 公式 | MAC/frame |
|---|---:|---:|
| neural `fc_in` | `N*(W*H+H^2)` | 27,623,424 |
| query `fc_in` | `C*(W*H+H^2)` | 575,488 |
| MHA projections | `C*H^2 + 2*N*H^2 + C*H^2` | 51,380,224 |
| QK and attention*V | `2*C*N*H` | 196,608 |
| FFN | `2*C*H*F` | 4,194,304 |
| full `fc_out` | `C*H*W` | 51,200 |
| **reference cached-E total** |  | **84,021,248** |

这不含 LayerNorm、softmax、bias、ReLU、residual 和数据移动。

### 8.1 两个合法的常量折叠

M2 只有 1 个 cross-attention layer，训练结束后 behavior query 是常量，所以可缓存：

- `fc_in(rep)`；
- query LayerNorm 结果；
- Q projection。

另外部署只读取最后一个时间点，`fc_out:512->50` 可保留对应最后一行，变成 `512->1`。优化后：

```text
82,871,296 MAC/frame
4.144 GMAC/s @ 50 Hz
8.287 GOP/s，若 1 MAC = 2 operations
```

必须用 checkpoint 做逐元素等价测试；多层 cross-attention 时 query 不再对所有层恒定，不能直接推广。

### 8.2 实时 PE 下界

`84.02M MAC / 20ms`：

```text
100 MHz: ceil(84.02M / 2.0M cycles) = 43 MAC/cycle
200 MHz: ceil(84.02M / 4.0M cycles) = 22 MAC/cycle
```

这是 100% 利用率的数学下界。考虑 softmax、LayerNorm、SRAM bank conflict 和 tile 尾部，实际可从 64-128 MAC lanes @100 MHz 做架构探索：

| 配置 | 纯 MAC 理想时延 | 留给非 MAC 的 20 ms 余量 |
|---|---:|---:|
| 64 lanes @100 MHz | 13.13 ms | 6.87 ms |
| 128 lanes @100 MHz | 6.56 ms | 13.44 ms |

因此“必须数百 PE”并非由 MAC 数直接推出；是否需要更宽阵列取决于 SRAM 带宽、实际利用率和 softmax/LN 实现。

### 8.3 在线权重和外存带宽

在线 decoder 参数共 3,466,902。按 int8 weight、int32 bias、int16 LayerNorm 参数估算约 3.49 MB。

若每帧从片外至少装载一次每层权重：

```text
3.49 MB/frame * 50 frame/s = about 174 MB/s
```

带宽数值不高，但片外访问能耗通常远高于 MAC。完整 SPINT 最好把 decoder 权重放片上 SRAM/NVM；这会成为面积主项。

在线 activation 还包括：

- `X` circular buffer：4.69 KiB @ int8；
- `E`：4.69-9.38 KiB；
- token hidden `N*H`：48 KiB @ int8；
- K/V：各 48 KiB @ int8，是否同时物化取决于 schedule；
- attention score：`C*h*N=12,288` values，INT16 全物化为 24 KiB，也可用 streaming online-softmax 降低容量。

### 8.4 完整 SPINT 的非 MAC 难点

IDEncoder 本身只有 Linear、ReLU、mean，适合整数 ASIC。完整 decoder 额外需要：

- 64 组长度 N 的 softmax：max reduction、exp approximation、sum、reciprocal；
- LayerNorm：mean、variance、reciprocal square root；
- 变长 N 的 mask/control；
- 2.10M 参数 FFN，对 `C=2` query 的权重利用率低。

因此 full-SPINT 是一个小型 Transformer accelerator，不再只是 few-shot adapter。

## 9. 量化与数值格式建议

### 9.1 IDEncoder 优先做 W8A8

建议起点：

- weights：INT8，per-output-channel symmetric scale；
- activations：INT8，per-layer scale；
- dot-product accumulator：INT32；
- `SUM_PHI`：先验证 INT16，保守版本 INT32；
- `E`：INT16 起步，验证后再压 INT8；
- ReLU 和 requantization：整数 multiplier + shift + saturation。

最大 512 项的 INT8 乘积累加理论界：

```text
512 * 127 * 127 = 8,258,048
```

24-bit signed 只剩很小余量，叠加 bias 和非对称零点后不稳妥；统一 INT32 accumulator 更安全。

### 9.2 `X + E` 是关键量化交界

原图要求 elementwise addition，所以 `X` 和 `E` 必须：

- 使用相同 scale/zero-point；或
- 先各自 requantize 到共享的 INT16 域再相加。

原始 spike count 是非负整数，但 cubic interpolation 可能产生小数、负值和 overshoot。主机量化时需要固定 clipping/scale 规则，不能把原始 `uint8 spike count` 直接当作插值后 calibration 的数值格式。

### 9.3 完整 decoder 的混合精度

若做 full-SPINT，建议先验证：

- GEMM/FFN W8A8、INT32 accumulate；
- LayerNorm statistics INT24/INT32，gamma/beta INT16；
- attention logits INT16；
- exp LUT + reciprocal 使用至少 16-bit 中间精度；
- softmax output INT16，再与 V 相乘。

full-SPINT 的量化风险主要在 LayerNorm 和 softmax，不在 IDEncoder。

## 10. 三种 ASIC 边界比较

### 10.1 方案 A：主机预处理 + 片上 IDEncoder + 外部/已有 decoder

这是最贴合“研究 few-shot 部分 ASIC 化”的方案。

```text
Host:
  trial segmentation + cubic interpolation + quantization
      |
      v
ASIC ID adapter:
  [M,T,N] -> phi -> mean -> psi -> E[N,W]
      |
      +-> E readback to host/GPU decoder
      or
      +-> E SRAM consumed by another on-chip decoder
```

优点：

- 芯片确实从 support/calibration data 计算 session-specific 状态；
- 无 softmax、LayerNorm、backward；
- 16-32 MAC lanes 即可；
- 可独立发表 IDEncoder quantization/dataflow/gradient-free adaptation accelerator。

代价：

- 约 1.2-1.3 MiB 片上 SRAM 下界；
- `M=33` 仍有 1.876 GMAC，但只执行一次；
- 若 decoder 仍在主机，系统级实时功耗不代表全植入式 decoder。

### 10.2 方案 B：完整 SPINT on-chip

优点：保留论文原始算法、置换不变和变 N 能力。

代价：

- 全模型 4,594,888 参数；量化权重/偏置约 4.6 MB；
- 约 84 MMAC/frame @50 Hz；
- softmax、LayerNorm、rsqrt、变长控制；
- 需要 64-128 MAC lanes @100 MHz 级别和多 bank SRAM。

除非课题目标就是 Transformer neural decoder ASIC，否则不建议作为第一次 toy tapeout。

### 10.3 方案 C：CORAL few-shot + frozen depthwise-TCN

仓库 Plan-B 模型：

```text
per-channel causal depthwise FIR: N=96, K=9
 -> newest output only [96]
 -> Linear 96->32 -> ReLU -> Linear 32->2
```

参数：

```text
depthwise: 96*9 weights + 96 bias = 960
linear1:   96*32 + 32             = 3,104
linear2:   32*2 + 2               = 66
total                                4,130
```

ASIC 只计算 newest FIR output：

```text
96*9 + 96*32 + 32*2 = 4,000 MAC/frame
```

而当前 PyTorch 会计算全部 50 个卷积位置再丢弃 49 个，约 46,336 MAC/frame。硬件应使用每通道 `K-1` history shift/circular state。

CORAL 对每通道：

```text
x'_i = a_i*x_i + b_i
```

可以在 session reset 后折叠进第一层 depthwise FIR：

```text
w'_{i,k} = a_i*w_{i,k}
b'_i     = b_i*sum_k(w_{i,k}) + b_conv_i
```

所以在线期仍只有 FIR + MLP。当前本地 6-session `ref_raw` 结果：

| calibration | mean R2 |
|---:|---:|
| 4 s | 0.172 |
| 8 s | 0.180 |
| 16 s | 0.181 |
| 32 s | 0.186 |
| full fold | 0.184 |

对照：本地 ridge-CORAL 约 0.114；论文 SPINT M2 为 `0.26 +/- 0.13`。因此 Plan-B 是明确的精度/硬件折衷，不应声称等价替代 SPINT。

## 11. 推荐的第一版芯片

若研究题目聚焦“few-shot 部分如何在 ASIC 上运行”，推荐 **方案 A：M2 固定 shape 的 IDEncoder-only accelerator**，而不是把 `E` 全部在主机算完。

建议固定：

```text
N=96, T=100, W=50, H=512
M programmable: 1..33，或只支持实验 sweep 集合
W8A8, INT32 dot accumulator
16 or 32 MAC lanes @ 50-100 MHz
host-side trial interpolation
on-chip ID weights + SUM_PHI + E SRAM
```

控制状态机：

```text
RESET
 -> LOAD_WEIGHTS / CHECK_CRC
 -> LOAD_CALIB_TRIAL
 -> RUN_PHI
 -> ACCUMULATE
 -> repeat until trial_count == M
 -> RUN_PSI
 -> WRITE_E / E_VALID
 -> IDLE or STREAM_E_OUT
```

接口建议：

- AXI4-Lite/APB：shape、M、quant scale、start/status/error；
- AXI-Stream 或 DMA：calibration trial 输入和 E 输出；
- CRC：weights/calibration payload 完整性；
- performance counters：MAC active cycles、SRAM stall、saturation count、total cycles；
- debug readback：`SUM_PHI` 和选定 neuron 的 layer output，便于硅后对齐 golden model。

若首要目标是最低流片风险、展示闭环在线 decoder，则选择方案 C；但论文应明确它是 CORAL-conditioned TCN，不是原始 SPINT IDEncoder。

## 12. 软件先行的必要重构

在 RTL 之前，先把当前模型拆成两个显式 API：

```python
E = model.encode_calibration(calib)       # once per session
y = model.decode_with_identity(x, E)      # every 20 ms
```

必要测试：

1. FP32 原始 `forward(x,calib)` 与拆分 API 的 `max_abs_error` 应接近 0。
2. 同一个 session 连续多帧只允许 `encode_calibration` 调用一次。
3. query precompute 和 `fc_out` last-row pruning 与原输出最后一帧逐元素一致。
4. 神经元与 calibration 同步置换时输出保持不变。
5. `M` sweep 时缓存失效、reciprocal 和量化 scale 正确更新。

当前源码尚未提供这两个 API；直接按 `spint_decoder.py` 生成 trace 会把 IDEncoder 错误地放入每帧循环。

## 13. 验证与验收矩阵

### 13.1 算法精度

| 层级 | 指标 |
|---|---|
| FP32 split graph | 与原 forward 逐元素等价 |
| quant IDEncoder + FP decoder | 6 held-out session mean/std、逐 session R2、M sweep |
| full quant graph | 相对 FP32 的 `Delta R2` 和 worst-session drop |
| Plan-B | 4/8/16/32 s curve，不只报平均值 |

建议门槛：先以 `Delta R2 <= 0.01` 和所有 session 不出现灾难性负增益为目标，再依据实验结果决定 INT8/INT16 边界。不要只用单个 session 校准量化 scale。

### 13.2 RTL 功能

- affine layer bit-exact random tests；
- negative/zero/max input 和 accumulator overflow assertions；
- `M=1`、`M=33`、非法 M；
- SRAM bank conflict 和 backpressure；
- reset 中断校准、重复 session reset；
- weight/calibration CRC error；
- permutation metamorphic test；
- Python quant golden -> C/RTL bit-exact -> gate-level regression。

### 13.3 性能与 PPA

至少报告：

- cycle/session vs M；
- PE utilization；
- weight SRAM、activation SRAM、logic area 分项；
- dynamic/leakage power 分项；
- 每次 session adaptation 的 energy；
- 若含在线 decoder，再报告 energy/frame 和 50 Hz average power。

在工艺节点、SRAM macro、V/F、activity trace 未确定前，不能从 GOP 数可信地给出 mW。现阶段可以确定的是容量、MAC、带宽和理论周期下界。

## 14. 决策表

| 目标 | 推荐版本 | 理由 |
|---|---|---|
| 研究“SPINT few-shot 本体”的硬件映射 | IDEncoder-only ASIC | 真正片上从 calibration 生成 E，算子简单 |
| 第一次最小面积闭环流片 | CORAL + depthwise-TCN | 4.13K 参数、4K MAC/frame、无敌对算子 |
| 完整复现 SPINT 论文能力 | full-SPINT | 保留置换不变/变 N，但成为 Transformer ASIC |
| 最快系统 demo | 主机算 E + 片上线性/TCN | 风险最低，但不能称片上 few-shot 计算 |

## 15. 最关键的研究判断

1. SPINT few-shot 的核心硬件不是训练器，而是一个一次性运行的 6-layer MLP + DeepSets mean reducer。
2. IDEncoder 的 1.876 GMAC 很大但不实时；片上真正的限制是约 1.1 MiB 权重 SRAM，而不是吞吐。
3. 参考 decoder 每帧重算 IDEncoder 是软件实现问题，不能成为 ASIC 架构基线。
4. 如果只研究 few-shot block，IDEncoder-only 最干净；如果研究完整低功耗 neural decoder，Plan-B TCN 更实际。
5. 完整 SPINT 的困难来自在线 cross-attention、LayerNorm 和 3.47M decoder 参数，不来自 few-shot mean/MLP 本身。
6. 下一项最有信息量的实验不是 RTL，而是 IDEncoder-only PTQ/QAT：验证 `E` 和 held-out R2 对 W8A8、INT16 reducer、M sweep 的敏感度。

## 16. 源码依据

- SPINT 结构与 forward：`SPINT-main/src/models/components/spint.py`
- M2 模型配置：`SPINT-main/configs/model/falcon_m2.yaml`
- M2 数据配置：`SPINT-main/configs/data/falcon_m2.yaml`
- calibration 数据构造：`SPINT-main/src/data/falcon_datamodule.py`
- loss/最后一帧选择：`SPINT-main/src/models/falcon_module.py`
- FALCON packaged decoder：`SPINT-main/third_party/falcon_challenge/spint_decoder.py`
- ASIC-friendly TCN：`planB_tempconv/models/tcn_student.py`
- TCN/CORAL few-shot 实验：`planB_tempconv/scripts/b3_tcn_crossday.py`
- 本地曲线：`planB_tempconv/outputs/results/b3_tcn_fewshot_curve.csv`

