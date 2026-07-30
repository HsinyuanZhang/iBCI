# P3 Cross-Session Analysis: B3 on DANDI 000688 sub-C CO

**日期：2026-07-24**  
**状态：Step 0/1/2 完成，但发现关键混淆 — 所有跨 session 结论待重做（见下方 critical finding）**

> ## CRITICAL FINDING: Unit Count Regime 混淆（2026-07-24）
>
> Chronological split 恰好把一个 recording-pipeline regime 跳变切成了 train/test 边界。
> sub-C 的 unit 数在 2016-09 前后断崖式跳变：
>
> | Split | sessions | unit 范围 | 均值 | regime |
> |---|---|---|---|---|
> | Train (37) | 2013–2015 全部 | 38–91 | 59 | 全 < 100 |
> | Val (8) | 2015-11 ~ 2016-09 | 53–353 | 232 | 混合 |
> | Test (8) | 2016-09 后全部 | 188–299 | 245 | 全 ≥ 188 |
>
> **影响**：Step 1/2 的 "跨 session" 实验实际上测的是 "train ~60 units → test ~250 units"，
> 即 **4× N regime 跳变**，不是同 regime 内的跨 session 波动。
>
> - Step 1 zero-shot R²=-0.124：decoder 从没见过 >91 units 的输入，test 却有 ~250。不是同 regime 跨 session 失败。
> - Step 2 enc_ft K=20=0.692：finetune 主要在学"如何处理 4 倍 unit 数"，不是"适应新 session"。
> - identity_mse≈0.97：train identity 空间建立在 ~60-unit 分布上，250-unit 的 identity 不对齐是预期内的。
>
> **架构本身支持可变 N**（encoder 参数 N-free，decoder cross-attention 输出与 N 无关，
> 已验证 N: 96→66 前向正确）。POYO 同样不假设跨 session unit 对应，但用 unit dropout
> 应对的是 "±30% 波动"，不是 "4× 跳变"。
>
> **结论**：用户确认目标是同 regime 内的波动（"今天 95 个 unit，明天 70 个"级别）。
> 旧 regime（<100 units）有 39 个 session，足够做干净的 27/6/6 chronological split。
> 所有 Step 1/2 结论需在修正后的 split 上重做。

## 实验配置

| 项 | 值 |
|---|---|
| 变体 | B3 |
| 数据 | DANDI 000688 `sub-C` CO，53 sessions |
| Split | 37 train / 8 val / 8 test（按日期 chronological） |
| Teacher | MC_Maze single-session（`r2_mean=0.9061`） |
| Encoder | ~18K params，frozen decoder |
| Seed | 42 |
| 结果 JSON | `results/p3_b3_dandi688_co_seed42.json` |

## 核心结果

| 指标 | 值 |
|---|---|
| Best val R²（epoch 0） | **-0.0349** |
| Test mean R²（8 held-out sessions） | **-0.0187** |
| 决策门（≥ 0.30） | **未通过** |
| Early stopping | epoch 10（patience 10） |

### Per-session test R²

| Session | R² |
|---|---:|
| sub-C_ses-CO-20160923 | -0.049 |
| sub-C_ses-CO-20160929 | -0.011 |
| sub-C_ses-CO-20161005 | -0.036 |
| sub-C_ses-CO-20161006 | -0.002 |
| sub-C_ses-CO-20161007 | -0.027 |
| sub-C_ses-CO-20161011 | -0.009 |
| sub-C_ses-CO-20161013 | **+0.017** |
| sub-C_ses-CO-20161021 | -0.032 |

8 个 test session 中 7 个 R² 为负，仅 1 个略正。模型在未见 session 上的行为预测**劣于均值基线**。

## 与 POYO 对比

| 维度 | POYO (NeurIPS 2023) | 本实验 B3 |
|---|---|---|
| 数据源 | 同：DANDI 000688 sub-C/M/J | 同 |
| 参数量 | ~13M transformer | ~18–35K lightweight encoder |
| Single-session CO R² | **≈0.935** | N/A（未在同一设置下测） |
| Cross-session 策略 | Per-unit learned embedding + session embedding；新 session 通过 unit ID / finetune 适配 | Frozen MC_Maze teacher decoder + 小 encoder；无 session embedding |
| Binning | 5–10 ms | 20 ms |
| Context window | 1 s context, 500 ms step | 50 bins × 20 ms = 1 s window |
| Unit 处理 | 每个 unit 独立 D-dim embedding lookup | NeuronID encoder（per-neuron pooling） |
| 训练目标 | End-to-end behavior decoding | Distillation（task + y_teacher + identity） |

### 解读

1. **数据本身可解码**：POYO 在相同数据上 single-session R²≈0.935，说明信号质量不是问题。
2. **跨 session 是主要难点**：本实验直接在 37 个历史 session 上训练、8 个未来 session 上测试，无 session-specific adaptation。负 R² 表明当前 pipeline 未能泛化到新 recording day。
3. **Teacher 域不匹配**：Teacher 在 MC_Maze（DANDI 000128，`hand_vel`，135 units 固定）上训练，decoder 权重冻结后直接用于 DANDI 000688（`cursor_vel`，可变 N=几十到上百）。这是**跨数据集**而非单纯跨 session。
4. **POYO 的关键设计我们未采用**：
   - Per-session / per-unit learned embeddings
   - End-to-end 训练（非 frozen 大 decoder distillation）
   - 更大模型容量捕获 session 间变异性
5. **Distillation 信号可能误导**：`prediction_distill_mse≈0.01`（低）但 task R² 为负，说明 student 学会了模仿 teacher 的输出分布，但该分布在目标域上本身无效。

## 可能的失败机制（按优先级）

1. **Frozen decoder 域迁移失败** — MC_Maze decoder 在 cursor_vel + 可变 N 上不适配
2. **无 session conditioning** — 跨日期非平稳性（电极漂移、unit 组成变化）未建模
3. **行为标准化** — train-session 统计量可能不足以覆盖未来 session 分布
4. **Calibration trial 选择** — 固定前 10 个 R trials，可能不代表新 session 的 unit identity
5. **Binning / window 差异** — 20 ms bin vs POYO 5–10 ms

## 决策

按 P3 决策门：**不继续 B15/B16 跨 session 大规模对比**（在当前 frozen-teacher + distillation 设置下，结构改进不太可能解决域迁移问题）。

> **更新（Step 0/1/2 后）**：Step 0 证明 pipeline 正确。下方旧 53-session Step 1/2 结果随后被发现受 unit-count regime 跳变混淆；其中 few-trial finetune 还使用 held-out behavior labels 和 backward gradients，只能作为 diagnostic oracle，不能推翻或替代冻结的 gradient-free streaming-calibration 协议。

---

## Step 0 — 单 session 上界（2026-07-24）

| 项 | 值 |
|---|---|
| session | `sub-C_ses-CO-20131003`（最早的 CO session） |
| split | 单 session 内 chronological 80/20 trial |
| 训练 | 端到端 task_only（freeze_decoder=False），60 epoch |
| 结果 | held-out trial R² = **0.6937** |
| POYO 参考 | single-session CO R² ≈ 0.935 |

**结论**：pipeline 正确（0.694 明显为正、高于 0.5 gate）。与 POYO 的 0.24 差距来自架构容量（18K vs 13M 参数）和 binning（20ms vs 5-10ms），不是 pipeline bug。frozen-decoder 跨 session 失败的主因是域迁移，进入 Step 1。

产物：`checkpoints/b3_ss_sub-C_ses-CO-20131003/`、`results/p3_step0_b3_ss_sub-C_ses-CO-20131003_seed42.json`。

---

## Step 1 — 同域端到端跨 session（2026-07-24）

| 项 | 值 |
|---|---|
| 变体 | B3 |
| 数据 | DANDI 000688 `sub-C` CO，53 sessions，37/8/8 chronological |
| 训练 | 端到端 task_only（freeze_decoder=False），40 epoch 上限 |
| Teacher 角色 | 仅定义 decoder 架构 + warm-start 权重，无蒸馏约束 |
| Seed | 42 |

| 指标 | 值 |
|---|---:|
| Best val R²（epoch 15） | 0.054 |
| Test mean R²（8 held-out sessions） | **-0.124** |
| 决策门（≥ 0.30） | **未通过** |

8 个 test session 全部 R² 为负或接近 0（最好 +0.062）。端到端比 frozen decoder（-0.019）还差，说明**跨 session 非平稳性是独立于域迁移的真问题**：`identity_mse ≈ 0.97`，calibration identity 在新 recording day 上几乎失效。

**关键认知（已修正）**：该 53-session zero-shot 结果跨越了 unit-count regime，不能用于判断同 regime 的部署式 streaming calibration；重做时仍以冻结、无梯度的 zero-shot 为主结果，不能以 few-trial finetune 改写决策门。

产物：`checkpoints/b3_dandi688_co_e2e_s42/`、`results/p3_b3_dandi688_co_e2e_s42_seed42.json`。

---

## 历史诊断 oracle — 带标签梯度 finetune 对照（2026-07-24，非 Step 2 目标）

> 该实验跨越 4× unit-count regime，且 finetune 使用 held-out behavior labels 和 backward gradients。结果不能作为部署、gradient-free streaming calibration 或下一步结构选择的证据；保留它仅用于记录诊断现象。

| 项 | 值 |
|---|---|
| 初始模型 | Step 1 best ckpt（`r2_mean=0.0541`） |
| 评估集 | 每个 test session 的 `trials[20:]`（所有配置同一评估集，严格无泄漏） |
| Calibration | `trials[0:10]`（所有配置一致） |
| Diagnostic oracle | `trials[0:K]`，K ∈ {5, 10, 20}，30 epoch，lr 1e-4，使用 held-out labels + gradients |
| 配置角色 | (a) zero-shot = 冻结部署基线；(b)/(c) = 非部署 oracle 对照 |

### 核心结果（8 test sessions 均值 R²）

| 配置 | mean R² | 过门(≥0.30) |
|---|---:|---|
| zero-shot | -0.122 | ✗ |
| enc_ft K=5 | +0.408 | ✓ |
| enc_ft K=10 | +0.614 | ✓ |
| **enc_ft K=20** | **+0.692** | ✓ |
| enc+dec_ft K=5 | +0.286 | ✗ |
| enc+dec_ft K=10 | +0.573 | ✓ |
| enc+dec_ft K=20 | +0.678 | ✓ |

### Per-session test R²

| Session | zero-shot | enc_ft K=5 | enc_ft K=10 | enc_ft K=20 | enc+dec K=20 |
|---|---:|---:|---:|---:|---:|
| 20160923 | -0.180 | +0.294 | +0.387 | +0.491 | +0.525 |
| 20160929 | -0.067 | +0.558 | +0.730 | +0.778 | +0.777 |
| 20161005 | -0.108 | +0.235 | +0.478 | +0.521 | +0.532 |
| 20161006 | -0.185 | +0.353 | +0.526 | +0.626 | +0.603 |
| 20161007 | -0.156 | +0.402 | +0.686 | +0.776 | +0.815 |
| 20161011 | -0.159 | +0.392 | +0.636 | +0.735 | +0.723 |
| 20161013 | +0.050 | +0.543 | +0.737 | +0.811 | +0.712 |
| 20161021 | -0.171 | +0.487 | +0.734 | +0.797 | +0.734 |

### 历史诊断观察（非方法结论）

1. 在旧且混淆的设置中，label-using `enc_ft K=20` 达到 +0.692；这不是 gradient-free 结果。
2. encoder-only oracle 在该单 seed、跨 regime 的比较中优于 encoder+decoder oracle；不可推广为部署机制。
3. K 敏感度只描述该 oracle，不能代替 calibration trial 数、覆盖或 identity 统计的无梯度消融。

### 与 POYO 对比（更新）

| 维度 | POYO (NeurIPS 2023) | 本实验 B3 |
|---|---|---|
| 数据源 | 同：DANDI 000688 sub-C/M/J | 同 |
| 参数量 | ~13M transformer | ~18–35K lightweight encoder |
| Single-session CO R² | **≈0.935** | 0.694（Step 0） |
| Cross-session 策略 | Per-unit embedding + session embedding；新 session finetune（POYO-mp） | 目标：共享 decoder + calibration identity 的冻结、无梯度推理 |
| 跨 session finetune 后 R² | POYO-mp 接近 single-session | 历史 oracle `enc_ft K=20 = 0.692`（非部署指标） |

### 解读

1. **数据本身可解码**（POYO + Step 0 双重确认）。
2. 旧 Step 1/zero-shot 跨越 unit-count regime，不能判断同 regime 的无梯度部署是否可行。
3. 历史 few-trial encoder finetune 是带标签 oracle，不是跨 session 适配机制或方法结论。
4. 下一个有效比较应保持 decoder/encoder 冻结，改变 calibration trial 数、方向覆盖或闭式 identity 统计，而不是比较 finetune 效率。

产物：`results/p3_step2_adaptation_b3_dandi688_co_e2e_s42_seed42.json`、`scripts/eval_adaptation_dandi688.py`。

---

## 建议的下一步（需讨论后执行）

| 优先级 | 方向 | 理由 |
|---|---|---|
| P3a | 在修正后的旧 regime 重跑冻结 Step 1/2 | 获得同 regime 的 gradient-free streaming-calibration 主结果 |
| P3b | calibration trial 数消融 | 固定 10 改为 20/30/50，不使用 held-out labels 更新权重 |
| P3c | 校准 trial 方向覆盖 | 与固定前 N 个 R trial 比较，保持纯前向协议 |
| P3d | 闭式 identity 归一化 | 以 calibration spikes 的矩统计对齐 identity 分布 |

## 相关文件

- Checkpoint: `checkpoints/b3_dandi688_co/best-epoch=000-val_heldin/r2_mean=-0.0349.ckpt`
- 训练日志: `results/p3_b3_dandi688_co_s42_train.log`
- 完整指标: `results/p3_b3_dandi688_co_seed42.json`
- DataModule: `mc_maze/multisession_datamodule.py`
