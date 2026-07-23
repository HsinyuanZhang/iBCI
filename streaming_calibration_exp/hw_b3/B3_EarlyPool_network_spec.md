# B3 EarlyPool IDEncoder：网络做什么、如何对硬件

> 主线规格。对应实现：`src/models/components/streaming_encoders.py` → `EarlyPoolEncoder`（`variant="B3"`）。  
> 配套程序：同目录 `b3_hw_golden.py`（导出分层 golden，供 RTL 对照）。

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

输入约定（与 datamodule 一致）：

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

`post_pool` 由 `_build_affine_stack(D, D, num_post_layers=3, out_dim=W)` 生成，Sequential 下标为 `0,2,4`（奇数位是 ReLU）。

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

相对原始 B0（`T→512→512→512` + `512→512→512→50`）约 **1.13 M 参数 / 1.88 G MAC**，B3 是数量级压缩后的硬件主线。

---

## 5. 与原始 SPINT IDEncoder 的差异

| | 原始 (B0) | B3 EarlyPool |
|--|-----------|--------------|
| 池前 | 3 层 MLP，`T→H→H→H`，`H=512` | 1 层 `T→D` + ReLU |
| 池化 | trial 维 mean | 同左 |
| 池后 | 3 层 `H→H→H→W` | 3 层 `D→D→D→W` |
| 训练 | 与 decoder 端到端 | 当前实验多为 frozen decoder + 蒸馏 |

B3 **不**复制 teacher 的 `fc_id_*` 权重（与 B0/B1 不同），需单独训练。

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

用 `b3_hw_golden.py` 导出下列中间量（每个 neuron、每个 trial 可单独比）：

| 检查点 ID | 名称 | shape | 用途 |
|-----------|------|-------|------|
| S0 | `x` | `[T]` | 输入 |
| S1 | `pre_linear` | `[D]` | MAC + bias |
| S2 | `pre_relu` / `feat` | `[D]` | ReLU 后，累加前 |
| S3 | `sum_feat` | `[N, D]` | 每个 `push_trial` 后 |
| S4 | `mean_feat` | `[N, D]` | `SUM/M` |
| S5 | `post0_linear` / `post0_relu` | `[D]` | |
| S6 | `post1_linear` / `post1_relu` | `[D]` | |
| S7 | `post2_linear` / `E` | `[W]` | 最终 identity |

建议验收顺序：

1. 单 neuron、`M=1`、小 `T/D/W`（`--profile tiny`）逐层 bit/数值对齐  
2. 扩到 `N>1`（权重共享）  
3. `M>1` 累加与均值  
4. 全规格 `T=100,W=50,D=64/128,N=96`

浮点对照默认 `atol=1e-5`、`rtol=1e-5`；定点 RTL 应对同一输入比对 **量化后的整数 golden**（后续 PTQ 脚本扩展），本程序先给出 **FP32 分层参考**。

---

## 8. 快速使用

```bash
conda activate ks4
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp

# 导出 tiny 用例（便于手算 / 波形对照）
python hw_b3/b3_hw_golden.py --profile tiny --out hw_b3/runs/tiny_sw

# 导出接近部署的规格
python hw_b3/b3_hw_golden.py --profile d64 --out hw_b3/runs/d64_sw

# 与 EarlyPoolEncoder 交叉验证（需 torch）
python hw_b3/b3_hw_golden.py --profile d64 --torch-check

# 硬件侧把同名 .npy 放到 rtl_dir 后比对
python hw_b3/b3_hw_golden.py --compare hw_b3/runs/tiny_sw hw_b3/runs/tiny_rtl
```

输出目录主要内容：

- `meta.json`：形状、seed、检查点列表  
- `weights/*.npy`：各层 `W`/`b`  
- `stages/*.npy`：S0–S7 中间结果  
- `E.npy`：最终 `[N, W]`
