# SUA-MUA Shared Encoder Experiment Results — 2026-07-22 实验日志

> **文档状态：原始结果叙述。** 本文包含不同 teacher/checkpoint 阶段，表格不能全部直接横向比较。经过协议审计的当前摘要见 [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md)，下一步见 [`../ROADMAP.md`](../ROADMAP.md)。

## Summary Judgment

**Direct weight transfer: NO VALUE. Architecture transfer: HIGH VALUE.**

The B3 EarlyPool encoder (18K params) works excellently on sorted SUA data when trained on it,
but learned weights are completely signal-type-specific. Zero-shot MUA→SUA transfer fails entirely.

---

## Experiment Results

| Experiment | R² | % of Teacher | Conclusion |
|---|---|---|---|
| Teacher (MC_Maze SPINT) | 0.868 | 100% | Strong baseline on SUA |
| P0: B3 trained on MC_Maze | 0.831 | 95.7% | Architecture works on SUA |
| P2: Zero-shot M2 B3 → MC_Maze | 0.001 | 0.1% | Weight transfer fails |
| P1: Zero-shot + z-score norm | -0.18 | <0% | Normalization hurts |
| FALCON M2 B3 (reference) | 0.630 | 97% of M2 teacher | Same architecture on MUA |

## P3: Representation Analysis

| Metric | Value | Interpretation |
|---|---|---|
| Cosine similarity (M2 vs MC identity) | 0.08 | Essentially orthogonal |
| Pearson correlation | 0.01 | Zero correlation |
| M2 B3 identity std on MC_Maze | 0.25 | Undifferentiated (noise) |
| MC_Maze B3 identity std | 0.92 | Strong per-neuron differentiation |

The M2-trained encoder produces near-constant output on SUA data — it cannot distinguish neurons.

---

## Key Findings

### 1. Architecture is signal-type-agnostic (P0)
B3's per-neuron independence (Linear(T→D)→ReLU → mean over M → MLP) makes no assumptions
about signal type. It achieves 95.7% of teacher on sorted SUA, matching its 97% on MUA (FALCON M2).

### 2. Learned weights are signal-type-specific (P2, P3)
The temporal features learned from MUA threshold crossings (population activity, -4.5×RMS)
are completely incompatible with sorted SUA (isolated single units, spike-sorted).
The encoder's internal representation has zero correlation across signal types.

### 3. Input normalization cannot bridge the gap (P1)
Z-score normalization per neuron makes transfer worse (-0.18 vs 0.001 raw).
The incompatibility is in the learned temporal filters, not input scale.

### 4. Why transfer fails: fundamental signal differences
| Property | MUA (FALCON) | SUA (MC_Maze) |
|---|---|---|
| Source | Threshold crossings (-4.5×RMS) | Spike-sorted single units |
| Units per electrode | 1:1 | 2.09:1 |
| Firing pattern | Population mixture | Isolated neuron |
| Spike count range | [0, 5] per 20ms bin | [0, 4] per 20ms bin |
| Identity meaning | Electrode-specific | Cluster-specific |

The encoder learns "what does neuron N's temporal pattern look like" — but "neuron N" means
fundamentally different things in MUA vs SUA.

---

## Practical Implications

### What transfers:
- **Architecture**: B3 EarlyPool (18K params) works on any binned spike data
- **Training recipe**: lr=1e-4, 40 epochs, task_plus_y_plus_E loss, freeze decoder
- **Convergence speed**: ~5 epochs to 95%+ of teacher (both MUA and SUA)

### What does NOT transfer:
- **Encoder weights**: Must retrain for each signal type
- **Identity representations**: Completely incompatible across signal types
- **Temporal filters**: Signal-type-specific features

### Deployment recommendation:
For a new signal type (SUA, MUA, or other):
1. Train SPINT teacher on calibration data (~20 epochs, ~5 min on GPU)
2. Train B3 student with teacher distillation (~5 epochs, ~2 min on GPU)
3. Total adaptation cost: ~7 minutes, 18K trainable params

This is cheap enough that transfer is unnecessary — just retrain.

---

## Comparison with Literature

| System | Params | SUA/MUA | Transfer approach |
|---|---|---|---|
| UniBCI (Hong 2026) | 3M+ | Both | Context embedding + large model |
| NDT2 (Ye 2023) | ~1M | Both | Multi-context pretraining |
| **SPINT B3 (ours)** | **18K** | **Either** | **Retrain per signal type** |

UniBCI/NDT2 achieve cross-signal-type transfer via large models with context conditioning.
We show that at 18K params, transfer is not achievable — but retraining is trivially cheap.
The lightweight encoder trades transferability for extreme efficiency.

---

## Conclusion: Is There Transfer Value?

**For direct deployment (zero-shot): NO.**
Cannot take a MUA-trained encoder and use it on SUA data.

**For architecture/methodology: YES.**
The same 18K-param B3 architecture + training recipe achieves 95%+ of teacher on both
MUA and SUA. The "transfer" is at the architecture level, not the weight level.

**For the original hypothesis ("本质上都是ID的不准"):**
Partially confirmed. Both SUA and MUA have identity uncertainty, and the same encoder
architecture can resolve it. But the resolution mechanism (learned temporal features)
is signal-specific. The problem structure transfers; the solution does not.

---

## NeuronID 重构实验：面向 SUA 的结构优化

### 动机

原始 B3 identity encoder 严格 per-neuron 独立：每个神经元的 calib 活动独立编码，跨神经元零信息交换。
这对 MUA（1:1 电极-unit）合理，但对 SUA（2.09 units/electrode）存在原理性错配：
spike sorting 误差是**关系型**的（split/merge/共电极耦合），per-neuron 独立结构无法表达。

### 新增变体

| Variant | 名称 | 核心改动 | 参数量 |
|---------|------|---------|--------|
| B15 | RelationalEarlyPoolEncoder | B3 + 跨神经元 self-attention（置换等变） | 34,802 |
| B16 | HighOrderStatsEncoder | B3 + 跨 trial 方差作为额外输入 | 22,130 |
| B3 | EarlyPoolEncoder (baseline) | 原始 per-neuron 独立 | 18,034 |

### 实验结果（MC_Maze SUA；teacher checkpoint validation R²=0.9061）

> 以下 B15/B16/teacher 数字来自单 session 内部 validation 的探索性比较。它们不是跨 session held-out 结果；旧 B3 还使用了不同 teacher，不能用于公平结构排序。

| Variant | R² | ID MSE | Cosine | Pearson | vs Teacher |
|---------|-----|--------|--------|---------|-----------|
| **B15 (关系型)** | **0.9204** | **0.0175** | **1.000** | **0.991** | **100.3%** |
| B16 (高阶统计) | 0.9112 | 0.0761 | 0.972 | 0.960 | 99.3% |
| Teacher (reference) | 0.9173 | 0.0000 | 1.000 | 1.000 | 100% |
| B3 (旧 teacher 训练)* | 0.281 | 0.451 | 0.792 | 0.776 | — |

*B3 使用旧 teacher（epoch 21, R²=0.868）训练，identity 空间与当前 teacher 不对齐，数值不可比。

### 初步观察

1. **B15 在本次内部 validation 中略高于 teacher 数值**（0.9204 > 0.9173）：这支持继续验证跨神经元
   self-attention，但在公平 B3、多 seed 和外部验证完成前，不能据此认定其泛化能力优于 teacher。

2. **B16 在本次内部 validation 接近 teacher**（0.9112, 99.3%）：仅多存一个跨 trial 平方和
   （+4K params），同时保持 per-neuron 独立，因而是值得优先复验的硬件友好候选。

3. **当前 checkpoint 下 identity 对齐差异较大**：B15 的 ID MSE=0.0175 约为 B16 的 1/4；
   是否稳定仍需多 seed 验证。

### 硬件部署分析

| | B3 | B16 | B15 |
|---|---|---|---|
| 运行态缓存 | N×D = 34KB | 2×N×D = 68KB | N×D = 34KB |
| finalize 峰值 | per-neuron 串行 | per-neuron 串行 | **O(N²) 全局，~244KB** |
| 可否逐神经元串行 | 是 | 是 | **否** |
| 硬件友好度 | ★★★ | ★★★ | ★☆☆ |

- **B16 最适合低缓存硬件**：保持 per-neuron 独立，无跨神经元数据依赖
- **B15 效果最优但需全局缓存**：self-attention 要求所有 N 个神经元 embedding 同时在片上
- 若 calibration 在 host 端完成、仅下发 identity [N,W]=27KB，则 B15 的 finalize 开销不影响实时路径

### 计算开销

B15 相比 B3：参数 +93%（+16.8K），计算 +45%（+4.65M MACs，仅 finalize 一次性）。
B16 相比 B3：参数 +23%（+4.1K），计算 +0%（仅多一次逐元素平方）。
两者绝对量仍极小（35K/22K vs UniBCI 3M+）。

---

## 未完成缺口 (TODO)

1. **B3 公平基线缺失**：当前 B3 checkpoint 使用旧 teacher（R²=0.868）训练，
   需用当前 teacher（R²=0.906）重训 B3 以获得同条件下的公平对比。
   命令：`conda run -n spint python sua_exploration/scripts/train_variant_mc_maze.py \
   --teacher_ckpt "sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt" \
   --variant B3 --out_name b3_mc_maze_v2`

2. **M1 (MUA) 对照实验**：B15/B16 尚未在 MUA 数据上测试。
   若 B16 在 MUA 上也有类似提升 → 跨 trial 方差是通用信号；
   若仅 SUA 有效 → 证实它捕获的是 sorting 可靠性（SUA 特有）。

3. **多 seed 统计显著性**：当前为单 seed 结果，需 3+ seeds 确认差异显著。

---

## Files

- Teacher checkpoint: `checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt`
- B3 student checkpoint (旧 teacher): `checkpoints/b3_mc_maze/best-epoch=019-val_heldin/r2_mean=0.8387.ckpt`
- B15 student checkpoint: `checkpoints/b15_mc_maze/best-epoch=016-val_heldin/r2_mean=0.9092.ckpt`
- B16 student checkpoint: `checkpoints/b16_mc_maze/best-epoch=017-val_heldin/r2_mean=0.9020.ckpt`
- Training scripts: `scripts/train_teacher_mc_maze.py`, `scripts/train_variant_mc_maze.py`
- Evaluation: `scripts/eval_zero_shot_transfer.py`, `scripts/compare_neuronid_variants.py`
- Datamodule: `mc_maze/datamodule.py`
- Encoder 实现: `streaming_calibration_exp/src/models/components/streaming_encoders.py` (B15, B16)
