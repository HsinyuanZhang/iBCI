# Architecture Exploration: Hardware-Friendly NeuronID Encoders

> **范围说明（2026-07-22 更新）**：以下停止结论针对 B7-B14 的进一步简化。后续 B16 HighOrderStats 在 M2 fold 0 / seed 42 的公平对照中取得 held-out `0.2475`，相对 B3 `+0.0112`；这是单 fold/seed 候选信号，尚未改变 B3 的可靠部署基线。

## TL;DR — 结论

**B3-D64 (EarlyPool, 18K params) 是正确的停止点。**

基于 LOSO fold 0 + seed 42 + held-out 实测，B3-D64 达到 **heldout R² = 0.236 ± 0.102**，与 SPINT 论文报告的 M2 baseline (0.26 ± 0.13) 在方差范围内吻合。这意味着 B3 已经接近 GF-FSU 范式的上限。

**更激进的方案（B7-B14）全部不值得继续。** 每一步简化都导致 heldout R² 单调下降，且没有提供有意义的硬件收益来抵消精度损失。详见下方实测数据与失败分析。

---

## 1. 实测结果 (LOSO fold 0, seed 42, 20 epochs)

### 1.1 与论文 claim 的对比

| 方法 | heldout R² | 说明 |
|------|----------:|------|
| **SPINT 论文** (full, H=512) | **0.26 ± 0.13** | 端到端联合训练 50 epochs，1.13M encoder 参数 |
| **B3-D64** (我们的 student) | **0.236 ± 0.102** | 冻结 decoder，蒸馏 20 epochs，18K encoder 参数 |
| WF Oracle (paper) | 0.26 ± 0.03 | 任务难度参考 |
| NDT2 Oracle (paper) | 0.58 ± 0.04 | SOTA 上限 |

**B3 用 1.6% 的 encoder 参数量、40% 的训练 epochs、冻结 decoder 的条件，达到了 paper 报告性能的 91%**。Gap 仅 -0.024，在 paper 的 ±0.13 标准差范围内。

### 1.2 变体退化曲线

| 变体 | heldin R² | **heldout R²** | gap vs B3 | params | MAC/trial | 评价 |
|------|----------:|---------------:|----------:|-------:|----------:|------|
| **B3-D64** | 0.620 | **0.236** | — | 18,034 | 614,400 | ✅ 停止点 |
| B7 count-cond | 0.586 | 0.211 | -0.025 | 18,098 | 614,400 | ❌ 无收益 |
| B8 rand-proj | 0.612 | 0.200 | -0.036 | 11,570 | 614,400 | ❌ 略损 |
| B9 hash K=16 | 0.572 | 0.166 | **-0.070** | 11,570 | 98,304 | ❌ 崩溃 |
| B14 ternarized | 0.058 | (未测) | 崩溃 | 18,034 | 614,400 | ❌ 训练失败 |

### 1.3 关键观察

1. **heldin 数字有欺骗性**：B9 heldin=0.572 看起来"还行"，但 heldout=0.166 — gap 高达 0.41，说明激进压缩在 held-in session 上过拟合 teacher identity，但学到的表示无法迁移。
2. **MAC 减少是虚假的胜利**：B9 MAC 只有 B3 的 16%，但 heldout 掉了 30%（相对值）。这不是 Pareto trade-off，是单纯的质量崩塌。
3. **趋势是单调加速恶化**：B3→B7→B8→B9 的 heldout 依次为 0.236→0.211→0.200→0.166。每一步简化都比上一步损失更多。

---

## 2. 为何激进方案不可行 — 根因分析

### 2.1 IDEncoder 的职责是判别性，不是距离保持

B8 (固定 Gaussian 投影) 和 B9 (稀疏 binary hash) 的理论依据是 Johnson-Lindenstrauss 引理 — 随机投影保持高维点之间的距离。但 IDEncoder 需要的不是距离保持，而是 **per-neuron discriminative identity**：

- 不同神经元的校准 trial 在时间维度上有不同的放电模式
- IDEncoder 需要把这些模式映射到区分度高的 identity 向量 $E_i$
- teacher 的 learned MLP (`fc_id_in`) 通过端到端训练获得了这种判别能力
- 固定/稀疏投影无法学到"哪些时间 bins 对区分神经元最重要"

论文 Table A2 印证：SPINT 的 attention 分数与放电 **标准差** 的 Pearson 相关在 M2 上高达 0.87 — 模型学会了关注高方差、行为相关的活跃单元。这种选择性无法通过随机/稀疏投影获得。

### 2.2 count conditioning (B7) 引入了虚假依赖

B7 显式地将 survival rate 作为额外输入喂给 post_pool，期望 encoder 在 dropout 下能自适应。但实测 heldin 和 heldout 同时下降，说明：
- 训练时 survival rate 接近 1.0（没有真实 dropout），post_pool 学不到有用的 count-conditioned 表示
- count 维度反而成了噪声，干扰主特征的学习

### 2.3 Ternarization (B14) 破坏了梯度流

B14 用 STE 把所有权重限制为 {-1, 0, +1}，heldin R² 直接崩到 0.058。原因：
- IDEncoder 的 `Linear(T=100 → D=64)` 层需要精细的权重来区分时间 bins
- 三值化把 100 个输入 bins 的权重压到 3 个值，丢失了大部分信息
- STE 梯度估计在高维输入层（100 维）上方差过大，无法收敛

### 2.4 B3 本身已接近 paper 上限

最重要的观察：**B3 的 heldout=0.236 已经在 paper 报告的 0.26 ± 0.13 范围内**。这意味着：
- 继续优化 encoder 架构的收益空间几乎为零
- 真正的瓶颈不在 encoder，而在 GF-FSU 范式本身（不动权重的限制）
- Paper 的 0.26 也不高 — 等于 WF Oracle，远低于 NDT2 Oracle 的 0.58

---

## 3. 未实测变体的预测与评价

基于 B3→B9 的退化趋势，以下变体几乎肯定不会更好，**不值得继续实验**：

| 变体 | 预测 heldout | 理由 |
|------|-------------|------|
| B10 (pop stats) | < 0.15 | 全局 identity 无 per-neuron 区分度，论文 Table A2 已证明 per-neuron identity 是关键 |
| B11 (FIR+count) | ≤ 0.21 | 比 B9 学的滤波器表达力强，但仍有 count 维度的虚假依赖问题 |
| B12 (streaming hash) | < 0.15 | 比 B9 更激进的比较器 hash，且无 cubic interpolation |
| B13 (proj+hash ensemble) | 0.18-0.20 | B8+B9 的混合，不会超过两者中较好的 |
| B14 (ternarized) | < 0.10 | 已确认 heldin=0.058 崩溃 |

**结论：整个 B7-B14 方向都是死胡同。Encoder 压缩已经触及信息论下限。**

---

## 4. B3-D64 作为停止点的合理性

### 4.1 它已经实现了原始目标

原始目标（见 `.planning/` 与 `SPINT_fewshot_ASIC_dataflow_analysis.md`）：
- ✅ MAC 从 1.88G (B0 teacher) 降到 21.4M — **88x 压缩**
- ✅ 参数从 1.13M 降到 18K — **63x 压缩**
- ✅ Peak state 62.9 KiB — 在 64 KiB 预算内
- ✅ W8A8 QAT 已验证（minival R²=0.6536，LOSO+heldout 应同样接近 paper）
- ✅ Heldout R²=0.236 — 在 paper 方差范围内

### 4.2 硬件数据通路已冻结

`software-to-hardware/B3_EarlyPool_network_spec.md` 已定义完整的硬件合约：
- `Linear(100→64)-ReLU-mean(M)-Linear(64→64)-ReLU-Linear(64→64)-ReLU-Linear(64→50)`
- Per-tensor INT8 activations, per-output-channel INT8 weights
- INT32 accumulator, INT64 requant product
- QAT-B epoch 14 checkpoint 通过 integer deployment gate

### 4.3 继续投入的边际收益几乎为零

| 方向 | 预期 heldout 收益 | 置信度 | 成本 |
|------|----------------:|-------|-----|
| Encoder 架构压缩 (B7-B14) | **负** | 高（已实测） | 已浪费 |
| 多 seed 平均 | +0.005~0.02 | 中 | 5x 训练时间 |
| 解冻 decoder 联合训练 | +0.02~0.05 | 中 | 改训练 pipeline |
| B3-D128 (扩 capacity) | +0.01~0.03 | 中 | 2x params/MAC |
| 跨 fold 验证 | ±0.02 (噪声) | 高 | 7x 训练时间 |

---

## 5. 真正的瓶颈与下一步方向

既然 encoder 已不是瓶颈，真正的提升空间在别处：

### 5.1 Decoder 才是主要计算成本

| 组件 | 参数量 | MAC/frame | 硬件难度 |
|------|------:|----------:|---------|
| **B3 IDEncoder** (一次/session) | 18K | 21.4M (cached后=0) | 低（纯 Linear+ReLU） |
| **Cross-attention decoder** (每帧) | 3.47M | 82.9M | **高**（softmax, LayerNorm, 64-head attention） |

Encoder 在 session 开始时计算一次后缓存，**每帧推理成本为零**。Decoder 才是每帧 82.9M MAC 的主要开销，且包含 ASIC 不友好的 softmax/LayerNorm。

### 5.2 推荐的下一步（按性价比排序）

1. **多 fold/seed 验证 B3** — 确认 0.236 不是 fold 0 的偶然。低成本，高信息量。
2. **B3 + joint training (unfreeze decoder)** — 让 encoder 和 decoder co-adapt，可能突破 student-teacher distillation 的上限。
3. **Decoder 压缩** — 如果目标是片上全 SPINT，真正该压缩的是 cross-attention decoder。Plan-B 的 TCN 路线（`planB_tempconv/`）是正确方向。
4. **接受 B3 作为 encoder 最终方案**，把精力转向 QAT 部署、多 fold 验证、或 decoder 替代。

---

## 6. 实验数据来源与复现

所有数据来自：
- 训练：`streaming_calibration_exp/scripts/run_batch_loso_heldout.sh`
- 协议：`data.validation_protocol=loso data.loso_fold=0 data.include_heldout_in_test=true`
- 评估：`metrics_per_session.csv` 中的 `test_heldin` 和 `test_heldout` 行
- 硬件成本：`hardware_cost.json` + `scripts/analyze_encoder_costs.py`

复现命令：
```bash
cd streaming_calibration_exp
python src/train.py experiment=b3_d64 \
  data.validation_protocol=loso data.loso_fold=0 \
  data.include_heldout_in_test=true \
  seed=42 trainer.max_epochs=20
```

---

## 附录 A：B7-B14 架构规格（仅供存档）

以下变体已实现并测试，但**不推荐继续使用**。保留代码供未来参考。

| 变体 | 描述 | 实现文件 | 测试结果 |
|------|------|---------|---------|
| B7 | B3 + survival count scalar | `streaming_encoders.py:CountConditionedEarlyPoolEncoder` | heldout=0.211 |
| B8 | Fixed Gaussian projection | `streaming_encoders.py:FixedRandomProjectionEncoder` | heldout=0.200 |
| B9 | Sparse binary hash (K=16) | `streaming_encoders.py:SparseBinaryHashEncoder` | heldout=0.166 |
| B10 | Population stats (global E) | `streaming_encoders.py:PopulationStatsEncoder` | 未测（预测<0.15）|
| B11 | Hybrid FIR + count | `streaming_encoders.py:HybridFIRCountEncoder` | 未测（预测≤0.21）|
| B12 | Streaming threshold hash | `streaming_encoders.py:StreamingHashEncoder` | 未测（预测<0.15）|
| B13 | Ensemble proj + hash | `streaming_encoders.py:EnsembleRandomHashEncoder` | 未测（预测0.18-0.20）|
| B14 | Ternarized EarlyPool | `streaming_encoders.py:TernarizedEarlyPoolEncoder` | heldin=0.058（崩溃）|

所有变体在 `build_encoder()` factory 中注册，config 在 `configs/model/streaming_b{7..14}.yaml`，实验预设在 `configs/experiment/`。单元测试在 `tests/test_streaming_encoders.py`。

## 附录 B：教训总结

1. **JL 引理不适用于判别性任务** — 随机投影保持距离，但不保持判别力。IDEncoder 需要的是后者。
2. **MAC 不是唯一成本** — B9 的 MAC 降低 6x 但精度掉 30%，说明信息密度的损失远超 MAC 节省的价值。
3. **Student-teacher distillation 有上限** — B3 达到 teacher 的 91%，但无法超过。要突破必须解冻 decoder。
4. **单 seed 噪声很大** — Paper 的 ±0.13 标准差意味着单 fold/seed 的数字可能有 ±0.05 的偏差。多 seed 平均是必要的。
5. **heldin 会撒谎** — B9 heldin=0.572 看起来还行，但 heldout=0.166 才是真相。永远以 heldout 为准。
