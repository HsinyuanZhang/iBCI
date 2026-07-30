# NeuronID Encoder 硬件化

状态：原始 B3 合同 `FROZEN`；B3T `PROVISIONAL`；B3A `DEFERRED`

## 1. 系统角色

NeuronID encoder 从无标签 calibration trials：

```text
Xcal[M,T,N]
```

生成：

```text
E[N,W]
```

部署时：

- 每 session 运行一次；
- 无行为标签；
- 无 backward/optimizer；
- E 缓存在片上或可加载 session state；
- 在线 decoder 每帧复用同一 E；
- neuron 轴 variable、置换等变。

## 2. 原始 B3

### 2.1 计算图

对每个 trial/neuron：

```text
x[T]
  -> Linear(T→D)
  -> ReLU
  -> feat[D]
```

跨 trial：

```text
SUM_feat[n,d] = Σ_m feat[m,n,d]
mean_feat = SUM_feat / M
```

每个 neuron：

```text
Linear(D→D) -> ReLU
Linear(D→D) -> ReLU
Linear(D→W)
```

默认：

```text
T=100
D=64
W=50
M=1...33
N=runtime
```

### 2.2 不可交换的操作

必须保持：

```text
mean(ReLU(Linear(x_m)))
```

不得改成：

```text
ReLU(Linear(mean(x_m)))
```

因为 ReLU 位于 trial mean 之前。

### 2.3 流式 state

```text
SUM_feat[N,64] INT32
trial_count
```

一次只需保留一条 trial 或一个 token tile，不需要保存全部 `M` 个
intermediate feature。

## 3. B3T

### 3.1 计算图

```text
x[100]
  -> fixed raised-cosine basis [12,100]
  -> basis_coeff[12]
  -> learned Linear(12→64)
  -> ReLU
  -> feat[64]
```

后续 mean/post MLP 与 B3 相同。

### 3.2 硬件映射

两阶段都用共享 PE：

```text
AFFINE_FIXED 100→12
AFFINE_RELU  12→64
```

fixed basis 选择：

1. 小 ROM；
2. coefficient SRAM 的只读 image；
3. 结构化生成器。

第一版建议 ROM/image，不引入函数生成器。

### 3.3 与 B3 的兼容

共同部分：

- tensor API；
- `SUM_feat`；
- mean reducer；
- post MLP；
- E format；
- neuron/trial permutation invariants；
- PE/requant/SRAM。

差异仅为 pre-pool descriptor，因此支持 B3T 不应新增专用 PE。

### 3.4 当前证据边界

当前开发结果表明 B3T：

- 参数更少；
- session MAC 更少；
- support state 与 B3 相同；
- 三 seed 点估计方向为正；
- B3T+SWA 有进一步正向信号。

但：

- formal held-out scope 尚未形成可引用结论；
- per-session 一致性不充分；
- 尚无与原 B3 同等成熟的 W8A8 release；
- 不得直接把 FP32/SWA checkpoint 用作 RTL sign-off。

因此硬件应支持 B3T shape，不固化 B3T 模型内容。

## 4. B3A 不进入第一版

B3A 在 trial 轴上学习 attention，需要保留：

```text
trial_feats[M,N,D]
```

而 B3/B3T 只保留：

```text
SUM_feat[N,D]
```

当前 `N=64,M=30,D=64` 成本工件：

```text
B3/B3T support state ≈ 16 KiB
B3A support state ≈ 491 KiB
```

同时 B3A 当前净效应约为零、跨 seed 混号。因此：

- 不为 B3A 预留专用 SRAM；
- 不实现 trial-softmax；
- 若未来新证据恢复 B3A，提升合同版本后再讨论。

注意：B3A 是 trial-axis attention；它与 decoder cross-attention、B15
cross-neuron self-attention 是三个不同算子和科学问题。

## 5. Encoder SRAM

以 `N=96`：

```text
trial buffer        100×96 INT8 = 9.38 KiB
SUM_feat            96×64 INT32 = 24 KiB
E                   96×50 INT8 = 4.69 KiB
B3 weights          ~18 KiB INT8
B3T learned weights ~12 KiB INT8
```

trial buffer 可进一步 tile 化；第一版是否保留完整 trial 由 DMA/地址生成器
复杂度决定。

## 6. Encoder PE schedule

### B3 pre-pool

```text
for trial m:
  for neuron tile n8:
    for output tile d8:
      for t in 0..99:
        Mode A MAC
      requant + ReLU
      add to SUM_feat
```

可以不物化完整 `feat[M,N,D]`。

### B3T pre-pool

```text
for trial m:
  for neuron tile n8:
    run fixed 100→12
    store 12-d coefficient scratch
    run learned 12→64 + ReLU
    add to SUM_feat
```

### Finalize

```text
mean_q = reciprocal_mean(SUM_feat, M)
post0
post1
post2
write E
set E_VALID
```

## 7. B3 整数合同

| Tensor/operation | 格式 |
|---|---|
| input | signed INT8 |
| weight | signed INT8, per-output-channel |
| bias | INT32 accumulator domain |
| dot accumulator | INT32 |
| affine output | INT8 shared activation scale |
| `SUM_feat` | INT32 |
| mean product | INT64 |
| reciprocal shift | model-configured，当前工作点 20 |
| requant product | INT64 |
| layer shift | model-configured，当前工作点 31 |
| output E | signed INT8 + scalar scale |

所有 edge saturation、overflow 和 rounding 必须可计数/断言。

## 8. B3T 量化待办

必须补：

1. fixed basis 的位宽和 scale；
2. `100→12` 输出 scale；
3. basis coeff 到 `12→64` 的 accumulator range；
4. 该两层是否可离线折叠作为精度 reference；
5. 显式两层硬件图的 W8A8 QAT；
6. B3/B3T 相同 protocol 的 paired R²；
7. full stage integer golden；
8. 多 `M` 与 variable `N`；
9. basis ROM/image hash；
10. final release manifest。

说明：在实数域中两个线性层可以折叠，但显式 B3T 的 MAC/learned-SRAM 优势来自
保留 `100→12→64` 结构；折成 `100→64` 会失去主要硬件收益，并改变定点
rounding。

## 9. Encoder 验收

### 功能

- tiny shape 逐元素 exact；
- `M=1`；
- `M=33`；
- `N=1/8/9/96/160`；
- partial neuron/output tiles；
- trial permutation；
- neuron permutation/equivariance；
- reset/abort/restart；
- E cache invalidation。

### 数值

- INT32 overflow=0；
- saturation rate；
- reciprocal error；
- stage exact；
- E exact；
- decoder task-level delta；
- 多 calibration draw。

### 性能

- cycles/session vs M/N；
- PE utilization；
- coefficient/activation reads；
- session energy；
- state peak；
- Mode A vs Mode B；
- B3 vs B3T。

