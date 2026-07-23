# B4b: 窗长匹配 Ridge 控制实验

> 目的：验证 B3 "TCN > ridge" 结论在相同时间窗口下是否成立。

## 实验设计

B3 中 TCN(W=50) 在 ref_raw 下拿到 R²≈0.20，而 ridge(N_HIST=7) 在 ref_raw 下只有 R²≈0.11。
但两者窗长不同：TCN 用 1000 ms，ridge 用 140 ms。存在混淆变量。

B4b 做法：把 ridge 的 N_HIST 从 7 拉到 50（96×50 = 4896 维平铺特征），使窗长与 TCN 一致（均为 1000 ms）。

```
ridge(N_HIST=50, ref_raw) = X
if X ≈ 0.20  →  gap 主要来自窗长，conv 结构没多少功劳
if X << 0.20 →  conv 结构贡献显著，机制故事成立
```

## 关键结果（6 session 均值，saturated budget）

| 方法 | 窗长 | 特征维数 | ref_raw R² | own_zscore R² |
|---|---|---|---|---|
| ridge N_HIST=7 (b4) | 140 ms | 672 | 0.11 | 0.114 |
| **ridge N_HIST=50 (b4b)** | **1000 ms** | **4896** | **0.116** | **0.126** |
| TCN W=50 (b3) | 1000 ms | — | **0.20** | ~0.15 |

## Per-session 明细（N_HIST=50，saturated）

| Session | own_zscore R² | ref_raw R² |
|---|---|---|
| ses-2020-10-30-Run1 | 0.225 | 0.235 |
| ses-2020-10-30-Run2 | 0.259 | 0.264 |
| ses-2020-11-18-Run1 | 0.147 | 0.126 |
| ses-2020-11-19-Run1 | 0.062 | 0.035 |
| ses-2020-11-24-Run1 | 0.090 | 0.085 |
| ses-2020-11-24-Run2 | -0.030 | -0.051 |
| **mean** | **0.126** | **0.116** |

## 结论

1. **窗长对 ridge 几乎没帮助**：N_HIST 从 7→50，ref_raw R² 仅从 0.11→0.116（+0.006）
2. **TCN 在相同窗长下大幅领先 ridge**：0.20 vs 0.116，gap = **0.084**
3. **0.084 的差距来自 conv 结构**（depthwise 卷积 + ReLU 非线性 + 权重共享），而非窗长差异
4. TCN 的 ref_raw (0.20) 显著优于 own_zscore (~0.15)，而 ridge 的 ref_raw (0.116) 反而略低于 own_zscore (0.126) — TCN 更能利用 CORAL 跨天对齐信息

## 各网络参数量对比

| 模型 | 参数量 | 结构说明 |
|---|---|---|
| **TCN k9 (B3)** | **4,130** | depthwise conv(96×1×9=864+b96) + pointwise MLP(96→32→2) |
| ridge N_HIST=7 (B4) | 1,538 | 平铺 768 维 FIR → 2 输出 (coef 2×768 + bias 2) |
| ridge N_HIST=50 (B4b) | 9,794 | 平铺 4896 维 FIR → 2 输出 (coef 2×4896 + bias 2) |
| SPINT (teacher) | ~4,600,000 | 完整 cross-attn + IDEncoder |

关键观察：
- TCN (4,130) 比同窗长的 ridge-W50 (9,794) **参数少 58%**，但 R² 高 72%（0.20 vs 0.116）
- TCN 的 depthwise conv 通过权重共享（每通道 9 个 tap 共享全部时间步）大幅压缩参数
- 4,130 参数远低于 BrainDistill IND 的 30K 量化 student，仍在植入级热预算内

## 对后续工作的影响

- B3 "TCN 时间卷积 > 线性 ridge" 的结论**在窗长匹配后依然成立**
- 轴 B（时间卷积）的机制故事有实验支撑，可以继续推进
- 下一步关注 B4（QAT + 功耗核算）：验证 TCN 的 W8A8 量化开销是否在植入热预算内

## 文件

- 脚本: `scripts/b4b_ridge_w50_ref_raw.py`
- 结果: `outputs/results/b4b_ridge_w50_ref_raw.csv`
- 图表: `outputs/figures/b4b_ridge_w50_ref_raw.png`

---

*注意：quick run（仅 2 session）给出 ridge ref_raw=0.250，是 sample bias — 那两个 session 恰好简单。全 6 session 均值 0.116 才是可靠数字。*
