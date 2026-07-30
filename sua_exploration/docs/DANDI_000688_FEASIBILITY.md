# DANDI 000688 Multi-Session SUA 可行性分析

**日期：2026-07-23**
**状态：sub-J smoke test 数据已下载并验证**

## 1. 数据验证结果

### sub-J 已下载文件（3 sessions, 84 MB）

| Session | Units | Trials | Total Spikes | Duration |
|---|---:|---:|---:|---:|
| sub-J_ses-CO-20160405 | 38 | 322 | 300,060 | 1277 s |
| sub-J_ses-CO-20160406 | 18 | 296 | 152,389 | 1003 s |
| sub-J_ses-CO-20160407 | 19 | 331 | 130,191 | 1077 s |

### NWB Schema 确认

- `/units/spike_times`: event-level spike timestamps（非预分箱）
- `/units/waveforms`: shape `(n_spikes, 48, 1)` — 48 samples per waveform, 1 channel
- `/units/electrodes`: electrode index per unit
- `/intervals/trials`: columns = `start_time, stop_time, target_on_time, go_cue_time, target_id, target_corners, target_dir, result`
- `/processing/behavior/Position/cursor_pos`: shape `(N_samples, 2)`, irregular timestamps (~1 kHz)
- `/processing/behavior/Velocity/cursor_vel`: shape `(N_samples, 2)`, unit = cm/s
- `/processing/behavior/Acceleration/cursor_acc`: same structure

### 与 MC_Maze datamodule 的差异

| 特征 | MC_Maze (DANDI 000128) | DANDI 000688 |
|---|---|---|
| Units 表 | 有 `heldout` column | 无 heldout，全部可用 |
| Trials 表 | 有 `split` column (train/val) | 无 split，需自行按 session 划分 |
| 行为 | `hand_vel` (1 kHz fixed rate) | `cursor_vel` (irregular timestamps) |
| 文件结构 | 单文件含所有 trials | 每 session 一个文件 |
| Unit 数 | 固定 135 (90 train + 45 heldout) | 可变：18–38 (sub-J)，预计 sub-C 范围更大 |
| Trial 质量 | 已预处理 | 含失败 trial (result='F'/'A'/'I')，最短 0.07 s |
| Trial result | 无 | R (reward) / F (fail) / A (abort) / I (ignore) |

## 2. POYO 论文关键对照

POYO (NeurIPS 2023) 使用了**完全相同的数据来源**（Perich et al. 的 Monkey C/M/Ja，即 DANDI 000688 的 sub-C/M/J）：

- **架构**：spike-level tokenization → PerceiverIO cross-attention → self-attention → cross-attention readout。13M params，与我们的 18–35K param lightweight encoder 完全不同量级。
- **跨 session 处理**：每个 unit 有一个 learned D-dim embedding（lookup table）。新 session 通过 unit identification（学习新 embedding 行）或 finetuning 适配。
- **性能参考**：single-session POYO 在 CO 上 R²≈0.935，RT 上 R²≈0.840。Multi-session POYO-mp + finetune 在 same-animal new-day 达 0.971。
- **数据处理**：5 ms binning (NLB) / 10 ms binning (MP datasets)；1 s context window，500 ms sliding step；unit dropout augmentation（最小 30 units）。
- **关键设计选择**：不假设跨 session unit correspondence；session embedding 捕获隐含实验变量。

### 与我们项目的关系

POYO 验证了这组数据的质量和可解码性，但**不直接回答我们的问题**。我们的目标是：在 18–35K param 预算下，B3/B15/B16 哪个结构最适合跨 session NeuronID estimation。POYO 的 13M param transformer 不是我们的竞争者，而是数据质量的上界参考。

## 3. 可行性评估

### 3.1 Loader 适配（中等工作量）

需要新增：
1. **Multi-file session registry**：扫描 subject 目录下所有 `*_behavior+ecephys.nwb`，按日期排序
2. **Trial 过滤**：只保留 `result == 'R'` 且 duration > 阈值（建议 > 0.5 s）的 trials
3. **行为插值**：cursor_vel 的 timestamps 不规则，需插值到 bin centers（现有 `interp1d` 逻辑可复用）
4. **Session-level split**：按日期排序后 chronological split，不是 trial-level random split
5. **Per-session calibration**：每个 session 独立构建 `[M, T, N_session]` calibration tensor，N 可变

现有可复用：
- `bin_spikes()` 函数
- cubic interpolation to fixed trial length
- behavior standardization
- `MCMazeSessionDataset` 的 windowed sampling 逻辑

### 3.2 可变 Unit 数（核心挑战，架构已支持）

B3/B15/B16 的 `finalize_identity` 接口设计为 per-neuron 独立处理 + pooling，参数不随 N 缩放。sub-J 的 18–38 unit 范围已验证可变性。sub-C 预计范围更大（POYO 论文 Table 1 报告 Perich et al. 117 sessions / 11,557 units，平均 ~99 units/session）。

**风险**：如果 sub-C 某些 session 有 >200 units，calibration tensor `[M=10, T=100, N=200]` 的内存和计算量仍在合理范围（8 MB float32）。

### 3.3 统计效力

| 实验 | Sessions | 效力评估 |
|---|---:|---|
| sub-J smoke test | 3 | 仅验证 pipeline，不用于排名 |
| sub-C CO chronological | 53 (37/8/8) | **首个有效跨 session 比较**，8 test sessions 足够估计 R² 均值和方差 |
| sub-C RT replication | 15 (9/3/3) | 3 test sessions，效力有限但可验证 task 泛化 |
| sub-M external | 28 | 冻结模型后的跨 subject 测试，效力充足 |

### 3.4 磁盘和时间

- sub-J: 84 MB ✓ 已下载
- sub-C: 9.738 GB（53 CO + 15 RT）— 需要确认磁盘空间
- sub-M: 2.915 GB — 第二阶段
- 总计 SUA 子集: ~12.74 GB

### 3.5 已识别风险

1. **Trial 过滤标准**：result='F'/'A'/'I' 的 trials 可能占比较高（sub-J 20160405 有 322 trials 但部分仅 0.07 s）。需要统计各 result 类型的分布，确定过滤后剩余 trial 数是否足够。
2. **行为采样率不一致**：cursor_pos timestamps 不规则，不同 session 采样率可能不同。需要验证插值后行为质量。
3. **Unit 稳定性**：同一 session 内 units 稳定，但跨 session 无对应关系。这正是 NeuronID encoder 应处理的设置，不是 bug。
4. **sub-C 下载时间**：9.7 GB 取决于网络带宽，建议后台下载。

## 4. 结论

**可行性：高。** 数据 schema 已验证，与现有 datamodule 的差异可控，架构已支持可变 N，POYO 论文确认了数据质量和可解码性。主要工作量在 multi-session datamodule 的编写（估计 200–300 行新代码）。

## 5. 建议的下一步

1. **立即**：统计 sub-J 的 trial result 分布，确定过滤策略
2. **本周**：编写 `MultiSessionDataModule`，在 sub-J 上完成 smoke test（binning → trial alignment → variable-N calibration → forward pass）
3. **下周**：后台下载 sub-C（9.7 GB），运行 53-session CO chronological split 的首个 B3/B15/B16 比较
4. **冻结后**：下载 sub-M 做 external evaluation
