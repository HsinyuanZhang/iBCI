# B3 EarlyPool IDEncoder：网络做什么、如何对硬件

> 本文件位于可独立拷贝的 `software-to-hardware/` 目录。  
> 对应实现：同目录 `early_pool_encoder.py`（`EarlyPoolEncoder`，variant B3）。  
> 配套程序：同目录 `b3_hw_golden.py`（导出分层 golden，供 RTL 对照）。

> 实现状态（2026-07-14）：第一版硬件继续采用本文 D64 EarlyPool 图，不包含 cross-attention。当前 QAT-B LOSO0 候选与可冻结/可编程边界见 `B3_QAT_B_hardware_handoff.md`。

---

## 1. 在系统里的角色

部署时 **没有反向传播、没有权重更新**。新 session 的流程是：

1. 主机（或上游）给出无标签 calibration trials：`Xcal[M, T, N]`
2. **B3 IDEncoder 前向一次**，得到每个神经元的 identity：`E[N, W]`
3. 缓存 `E`；之后每 20 ms 在线帧只做 `Z = X + E` 再进 decoder（本文件不涵盖 decoder）

因此硬件是 **一次/session 的 session adapter**，不是实时每帧网络。

默认 M2 形状：

| 符号 | 含义 | 默认值 |
|------|------|--------:|
| `M` | calibration trial 数 | 1…33（可扫） |
| `T` | 每条 trial 时间长度 | 100 |
| `N` | 神经元数 | 96 |
| `D` | B3 hidden（`hidden_dim`） | 64 或 128 |
| `W` | identity / 因果窗长 | 50 |

输入约定：

- 一条 trial：`[T, N]`（时间 × 神经元）
- 网络内部按 **neuron 独立、权重共享** 处理：每个神经元取时间向量 `[T]`

主机侧的 cubic 插值 / trial 切分 **不属于** B3 网络；RTL 输入应是已定长、已量化的 trial（或 raw pad/trunc 到 `T`）。

---

## 2. 计算图（完整一次 session）

对每个 trial `m`、每个神经元 `n`：

```text
x[m, n, :] ∈ R^T
    │
    ▼
pre_pool:  Linear(T → D) → ReLU
    │
    ▼
feat[m, n, :] ∈ R^D
```

跨 trial 做 DeepSets 均值（顺序无关）：

```text
SUM[n, :] = Σ_m feat[m, n, :]
mean[n, :] = SUM[n, :] / M
```

再对每个神经元跑共享的 post MLP：

```text
mean[n, :] ∈ R^D
    │
    ▼
post_0: Linear(D → D) → ReLU
post_1: Linear(D → D) → ReLU
post_2: Linear(D → W)          # 末层无 ReLU
    │
    ▼
E[n, :] ∈ R^W
```

软件等价写法（streaming）：

```text
state = reset()                    # SUM=0, M=0
for each trial:
    state = push_trial(state, trial)   # SUM += pre_pool(trial)
E = finalize(state)                # post_pool(SUM / M)
```

**不能**先对 trial 求均值再进 `pre_pool`：ReLU 在均值之前，交换会改变结果。

---

## 3. 层参数与张量布局（RTL 对齐）

PyTorch `nn.Linear` 约定：`y = x @ Wᵀ + b`，权重 shape 为 `[out_features, in_features]`。

| 模块 | 权重 | bias | 输入 | 输出 | 激活 |
|------|------|------|------|------|------|
| `pre_pool.0` | `[D, T]` | `[D]` | `[T]` | `[D]` | ReLU |
| `post_pool.0` | `[D, D]` | `[D]` | `[D]` | `[D]` | ReLU |
| `post_pool.2` | `[D, D]` | `[D]` | `[D]` | `[D]` | ReLU |
| `post_pool.4` | `[W, D]` | `[W]` | `[D]` | `[W]` | 无 |

`post_pool` 为 3 个 Linear + 中间 ReLU，Sequential 下标为 `0,2,4`（奇数位是 ReLU）。

golden 导出的权重文件名：

| 文件 | shape |
|------|-------|
| `weights/pre_w.npy` | `[D, T]` |
| `weights/pre_b.npy` | `[D]` |
| `weights/post0_w.npy` / `post0_b.npy` | `[D, D]` / `[D]` |
| `weights/post1_w.npy` / `post1_b.npy` | `[D, D]` / `[D]` |
| `weights/post2_w.npy` / `post2_b.npy` | `[W, D]` / `[W]` |

跨神经元状态（片上应保留）：

| 状态 | shape | 说明 |
|------|-------|------|
| trial 缓冲（可选） | `[T, N]` | 一条 trial；也可边收边算不存满 |
| `SUM_feat` | `[N, D]` | 跨 M 累加 |
| `trial_count` / `M` | scalar | finalize 时除 |
| `E` | `[N, W]` | session 输出缓存 |

均值实现建议：固定 `M` 时把 `1/M` 并入 `post_0` 的 requant scale；多 `M` 用小型 reciprocal LUT。**不要**上通用除法器。

---

## 4. 成本（量级）

公式：

```text
MAC_session = M · N · (T · D) + N · (2 · D² + D · W)
params ≈ (T·D + D) + 2·(D·D + D) + (D·W + W)
```

| `D` | 参数量 | MAC @ M=33,N=96,T=100,W=50 |
|----:|-------:|---------------------------:|
| 64  | 18 034 | ≈ 21.4 M |
| 128 | 52 402 | ≈ 44.3 M |

---

## 5. 与原始宽 IDEncoder 的差异（背景）

| | 原始宽网络 | B3 EarlyPool |
|--|-----------|--------------|
| 池前 | 3 层 MLP，`T→H→H→H`，`H=512` | 1 层 `T→D` + ReLU |
| 池化 | trial 维 mean | 同左 |
| 池后 | 3 层 `H→H→H→W` | 3 层 `D→D→D→W` |

---

## 6. 必须保持的不变量

1. Calibration **不用** behavior 标签  
2. 测试期无 optimizer / 权重更新  
3. 同一套权重对所有 neuron 共享  
4. trial 顺序置换不改变 `E`（mean 交换律；浮点累加顺序误差除外）  
5. 同步置换 query 与 calib 的 neuron 轴时，`E` 同步置换  
6. 输出 shape 固定 `[N, W]`（batch 时为 `[B, N, W]`）

---

## 7. 软硬对照：分层检查点

用 `b3_hw_golden.py` 导出下列中间量：

| 检查点 ID | 名称 | shape | 用途 |
|-----------|------|-------|------|
| S0 | `x` / `S0_x_m0_n0` | `[T]` | 输入 |
| S1 | `pre_linear` / `S1_*` | `[D]` | MAC + bias |
| S2 | `feat` / `S2_*` | `[D]` | ReLU 后，累加前 |
| S3 | `sum_feat` / `sum_after_trial` | `[N, D]` | 每个 `push_trial` 后 |
| S4 | `mean_feat` / `S4_*` | `[N, D]` | `SUM/M` |
| S5 | `post0_*` / `S5_*` | `[D]` | |
| S6 | `post1_*` / `S6_*` | `[D]` | |
| S7 | `E` / `S7_*` | `[W]` | 最终 identity |

建议验收顺序：

1. 单 neuron、`M=1`、小 `T/D/W`（`--profile tiny`）逐层对齐  
2. 扩到 `N>1`（权重共享）  
3. `M>1` 累加与均值  
4. 全规格 `T=100,W=50,D=64/128,N=96`

浮点对照默认 `atol=1e-5`、`rtol=1e-5`。`b3_hw_golden.py` 提供 **FP32 分层参考**；定点 RTL 使用 `b3_int8_validate.py` 导出的 `stages_int8/*.npy` 做逐元素整数对照。QAT model-specific sign-off 使用冻结 release 包中的 integer stages，具体见 `B3_QAT_B_hardware_handoff.md`。

---

## 8. 快速使用

在本目录（拷贝后的绝对路径）下：

```bash
cd /path/to/software-to-hardware

python b3_hw_golden.py --profile tiny --out runs/tiny_sw
python b3_hw_golden.py --profile d64 --out runs/d64_sw --torch-check
python b3_hw_golden.py --compare runs/tiny_sw runs/tiny_rtl
```

输出目录主要内容：

- `meta.json`：形状、seed、检查点列表  
- `weights/*.npy`：各层 `W`/`b`  
- `stages/*.npy`：S0–S7 及全量中间结果  
- `E.npy`：最终 `[N, W]`  
- `summary.txt`：便于手对的短摘要

---

## 9. 并行开发版本边界

硬件可以冻结本文的算子顺序、`T=100,D=64,W=50`、W8A8、INT32 accumulator、reciprocal mean 和 integer mult+shift requant。`M`、`N`、weights、bias、requant mult/shift 与六个 activation scales 应保持可配置。

当前 epoch-14 checkpoint 是 LOSO0 硬件候选，不是应硬编码的最终常数。软件后续网络优化只允许通过新的 model release 替换参数包；若改变层数、D/T/W、量化位宽或 rounding 语义，则必须提升硬件合同版本并重新审核。
