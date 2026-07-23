# Current Results: SUA/MUA Shared Encoder

**状态：当前结果摘要**  
**更新：2026-07-22**

## 结论摘要

1. **架构迁移成立**：B3 EarlyPool 分别在 MUA 和 SUA 上训练时都能逼近各自 teacher。
2. **直接权重迁移失败**：FALCON M2 的 B3 权重在 MC_Maze 上无法生成有判别力的 identity。
3. **B15/B16 是候选，不是定论**：B16 在首个公平 MUA fold/seed 上略高于 B3，但 session 方向混合；SUA 公平 B3、多 seed/fold、B15 MUA 对照和真正外部 SUA 验证仍未完成。

## 评估口径

现有 MC_Maze 结果使用：

- DANDI 000128 的一个带行为标签 train NWB；
- `heldout == false` 的 sorted units；
- NWB 内部 `train`/`val` trial split；
- 10 个 calibration trials，`T=100`，20 ms bins，`W=50`；
- teacher decoder 冻结，student 学习 identity encoder；
- 比较脚本默认评估 validation loader 的前 200 batches。

这属于 **single-session internal validation**，不是跨 session LOSO。当前 datamodule 还将同一 validation loader 同时标为 `val_heldin` 和 `val_heldout`，因此所有 SUA 文档只采用 “internal validation” 表述。

## A. 架构复用与零样本权重迁移

以下是早期 teacher/checkpoint 组的已报告结果：

| 实验 | MC_Maze internal-val R² | 解释 |
|---|---:|---|
| MC_Maze SPINT teacher | 0.868 | 早期 SUA teacher |
| B3 在 MC_Maze 上重新训练 | 0.831 | 达到 teacher 的约 95.7% |
| FALCON M2 B3 → MC_Maze zero-shot | 0.001 | 直接权重迁移基本失败 |
| zero-shot + per-neuron z-score | -0.18 | 简单输入归一化不能修复迁移 |

对应 representation 分析：

| 指标 | 值 | 解释 |
|---|---:|---|
| M2 identity 与 MC identity cosine | 0.08 | 近似正交 |
| Pearson correlation | 0.01 | 基本无相关性 |
| M2 B3 在 MC_Maze 上的 identity std | 0.25 | neuron 区分度弱 |
| MC_Maze B3 identity std | 0.92 | neuron 区分度明显 |

**支持的结论**：问题结构和网络架构可复用，但已学习 temporal filters 与 identity space 是信号类型相关的。

## B. SUA 结构候选 B15/B16

### 结构

| Variant | 改动 | 参数量 | 主要代价 |
|---|---|---:|---|
| B3 | per-neuron EarlyPool | 18,034 | 基线 |
| B15 | neuron-axis self-attention + residual + LayerNorm | 34,802 | finalize 需要全局 neuron feature，含 O(N²) attention |
| B16 | 跨 trial mean + variance | 22,130 | 需要额外二阶矩 accumulator |

### 当前 teacher 下的 checkpoint validation

| 模型 | checkpoint validation R² | 备注 |
|---|---:|---|
| Teacher epoch 83 | 0.9061 | 当前共享 teacher |
| B15 epoch 16 | 0.9092 | 使用当前 teacher |
| B16 epoch 17 | 0.9020 | 使用当前 teacher |
| B3 epoch 19 | 0.8387 | **使用旧 teacher，不可公平排序** |

### `compare_neuronid_variants.py` 探索性重评

| Variant | task R² | identity norm MSE | cosine | Pearson |
|---|---:|---:|---:|---:|
| B15 | 0.9204 | 0.0175 | 1.000 | 0.991 |
| B16 | 0.9112 | 0.0761 | 0.972 | 0.960 |
| Teacher | 0.9173 | 0.0000 | 1.000 | 1.000 |
| B3（旧 teacher） | 0.281 | 0.451 | 0.792 | 0.776 |

这里的 B3 与 B15/B16 不共享 teacher/identity space，不能用于判断结构优劣。B15 的 task R² 在该有限 validation 子集上略高于 teacher，只说明 student identity 与固定 decoder 的组合在这次评估中得到更高 R²；它不是 teacher 的严格上界突破，更不能证明外部泛化更强。

## C. MUA 参考结果与 B16 首次对照

| 模型 | Task | held-out R² | Paper SPINT | 协议 |
|---|---|---:|---:|---|
| B3 | FALCON M2 | 0.236 ± 0.102 | 0.26 ± 0.13 | LOSO fold 0 / seed 42 + 6 unseen sessions |
| B16 | FALCON M2 | 0.248 ± 0.137 | 0.26 ± 0.13 | 与 B3 完全相同；best epoch 1 |
| B3 | FALCON M1 | 0.630 | 0.66 ± 0.07 | LOSO fold 0 + 3 unseen sessions |
| B3 | FALCON H1 | — | 0.29 ± 0.15 | 尚未完成 |

M2 的 B16-B3 held-out mean delta 为 `+0.0112`，约 `+4.7%` 相对提升；逐 session 为 4 个提升、2 个下降。B16 增加 22.7% encoder 参数，但 session MAC 只增加 1.84%；代价主要是二阶矩 support state 翻倍。由于这里只有一个 fold/seed，当前只能说“B16 在 MUA 也出现正向候选信号”，不能宣称稳定优于 B3，也不能再把 B16 的潜在收益直接归因于 SUA sorting reliability。

这些 MUA 结果不能与 MC_Maze internal validation 的绝对 R² 直接比较。

## 当前证据边界

### 可以写入摘要

- 同一轻量架构可分别适配 SUA 和 MUA。
- 当前 B3 权重不支持 MUA→SUA 零样本迁移。
- 关系型和高阶统计结构值得继续做公平验证。
- 在一个严格匹配的 M2 fold/seed 上，B16 略高于 B3，但证据仍是探索性的。

### 暂时不能写入摘要

- B15 已经确定优于 B3 或 teacher。
- B15/B16 的增益来自 spike sorting split/merge。
- B15/B16 在跨 session SUA 上有效。
- B16 已经稳定优于 B3，或其增益是 MUA/SUA 通用机制。
- 18K 参数模型绝对无法通过联合训练获得跨信号权重共享。

## 关键产物

- Teacher：`sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt`
- 旧 B3：`sua_exploration/checkpoints/b3_mc_maze/best-epoch=019-val_heldin/r2_mean=0.8387.ckpt`
- B15：`sua_exploration/checkpoints/b15_mc_maze/best-epoch=016-val_heldin/r2_mean=0.9092.ckpt`
- B16：`sua_exploration/checkpoints/b16_mc_maze/best-epoch=017-val_heldin/r2_mean=0.9020.ckpt`
- 训练：`sua_exploration/scripts/train_variant_mc_maze.py`
- 比较：`sua_exploration/scripts/compare_neuronid_variants.py`
- 数据：`sua_exploration/mc_maze/datamodule.py`
- Encoder：`streaming_calibration_exp/src/models/components/streaming_encoders.py`
- MUA B16：`streaming_calibration_exp/outputs/streaming_calibration/b16_m2_loso_f0_s42_20260722_160720/`

下一步及验收标准见 [`../ROADMAP.md`](../ROADMAP.md)。
