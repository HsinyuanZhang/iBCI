# Multi-Session SUA Dataset Audit

**核查日期：2026-07-23**  
**目标：为 B3/B15/B16 建立真正跨 session 的 sorted-SUA 验证，而不是继续使用单个 MC_Maze session 的 trial split。**

## 结论

首选是 [DANDI 000688, version 0.250122.1735](https://dandiarchive.org/dandiset/000688/0.250122.1735)，但必须只使用 `sub-C`、`sub-M` 和 `sub-J`。这三个 subject 合计 **99 个 manually sorted SUA sessions**；`sub-T` 的 12 个 session 是 threshold crossings，不能混入严格 SUA 结论。

第二选择是 [DANDI 000121, version 0.220124.2156](https://dandiarchive.org/dandiset/000121/0.220124.2156)：两只猴、12 个 session、明确的 sorted spikes 和行为，但 live asset list 约 39.48 GB，文件中还包含 LFP，首轮适配成本高于 000688。

若接受 MATLAB 输入，[Zenodo 3854034](https://zenodo.org/records/3854034) 的原始 MC_RTT 是同领域的强候选：两只猴共 47 个 session，约 24.01 GB。每个 channel 的第一个 unit 是未排序的 hash/threshold-crossing unit，严格 SUA 实验必须排除它，只保留后续 sorted units。

## 候选比较

| 数据集 | 可用 session/recording | 行为 | 体量 | 与当前项目的匹配度 | 结论 |
|---|---:|---|---:|---|---|
| DANDI 000688 | 99 | CO/RT reaching，cursor position/velocity，trial metadata | SUA 子集约 12.74 GB | 标准 NWB，和当前 motor decoding 最接近 | **主数据集** |
| DANDI 000121 | 12 | hand/eye/cursor position，4 种 cursor task | live assets 约 39.48 GB | 标准 NWB、macaque M1/PMd，但 LFP 使文件较大 | **外部复验** |
| Zenodo MC_RTT 3854034 | 47 | 连续 self-paced reaching，finger/cursor/target | 约 24.01 GB | 同 motor task 家族；需 MATLAB adapter 和 unit 过滤 | **强备选** |
| DANDI 000947 | 378 recordings | choice-reaction reaching | 约 1.05 TB | 真 SUA，但多为线性 probe/单元生理记录，疾病状态和脑区混合 | 不作为 population NeuronID 首选 |
| DANDI 001868 | 70 ephys sessions，10 mice | ICMS-cued wheel detection | 约 7.50 GB | sorted SUA、纵向、多 subject；任务和物种差异大 | 可做通用性压力测试 |
| DANDI 001775 | 85 files，6 mice | DNMP、foraging、position tracking | 约 66.16 GB | sorted SUA 和行为完整；海马任务而非 motor cortex | 可做远域验证 |

## 主候选：DANDI 000688

### 已核实的数据组成

发布版官方 API 返回 111 个 NWB assets、4 个 subjects、总计约 13.18 GB：

| Subject | CO | RT | 合计 | 大小 | 信号类型 |
|---|---:|---:|---:|---:|---|
| `sub-C` | 53 | 15 | 68 | 9.738 GB | manually sorted SUA |
| `sub-M` | 22 | 6 | 28 | 2.915 GB | manually sorted SUA |
| `sub-J` | 3 | 0 | 3 | 0.088 GB | manually sorted SUA |
| `sub-T` | 6 | 6 | 12 | 0.438 GB | threshold crossings，排除 |

SUA 子集因此是 `68 + 28 + 3 = 99` sessions，其中 CO 78、RT 21。

信号类型不是根据 `/units` 表名猜测的。DANDI 官方描述明确写明 3 个 subjects 为 manual spike sorting、第四个为 threshold crossings；[POYO 论文](https://arxiv.org/abs/2310.16046) 的数据附录进一步说明 C、M、Ja 的数据经过人工离线 single-neuron identification，并把 T 作为 12-session 新动物测试集。两处信息与 asset 计数交叉对应，因而可以确定应排除 T。

### 代表文件的远程结构检查

对 `sub-C_ses-CO-20150706_behavior+ecephys.nwb` 的 HDF5/NWB 结构进行了只读远程检查：

- `/units/spike_times` 和 per-unit index；
- `/units/waveforms`，保留逐 spike waveform；
- `/intervals/trials`，包含 start/stop、target、go cue 和 result；
- `/processing/behavior/Position/cursor_pos`；
- `/processing/behavior/Velocity/cursor_vel`；
- `/processing/behavior/Acceleration/cursor_acc`。

这是真正的 event-level SUA，不是把预分箱计数包装进 NWB。现有 MC_Maze datamodule 的 spike binning 和二维行为接口可以复用；主要新增工作是多文件 session registry、session-disjoint split、不同 task schema 的标准化和 per-session unit mask。

### 推荐实验顺序

1. **Schema smoke test**：先下载 `sub-J` 的 3 个 session（共约 88 MB），只用于验证 NWB loader、binning、trial 对齐和可变 unit 数；样本太少，不能用于模型排序。
2. **首个有效 cross-session 实验**：仅使用 `sub-C` 的 53 个 CO sessions，按日期排序后固定为 37 train / 8 validation / 8 test。所有模型共享完全相同的 session split，checkpoint 只能看 validation，最终只读一次 test。
3. **Task replication**：对 `sub-C` 的 15 个 RT sessions 做 9 / 3 / 3 chronological split。CO 和 RT 分开报告，不能因 session 数量不同直接求未加权总平均。
4. **跨动物测试**：结构和超参数在 `sub-C` 上冻结后，再以 `sub-M` 做 external subject evaluation；`sub-J` 只有 3 个 CO sessions，只作为小样本补充。

首轮优先 CO，因为 trial 边界清晰且有 53 个同 subject sessions。跨日期 split 比随机打散 session 更接近部署时面对未来 recording day 的问题。跨 session 时不得假设 unit index 对应同一个生物神经元；unit ID 必须保持 session-local，这正是 NeuronID encoder 应处理的设置。

## 第二候选：DANDI 000121

[官方版本](https://dandiarchive.org/dandiset/000121/0.220124.2156)明确描述两只 macaques、96-channel Utah arrays、M1/PMd、sorted spikes，以及 hand/eye/cursor position：

- `sub-Reggie`：7 sessions；
- `sub-JenkinsC`：5 sessions；
- live asset API：12 NWB、约 39.48 GB。

需要注意官方 `assetsSummary` 仍显示 15 files / 187.53 GB，而版本的 live asset list 是 12 files / 39.48 GB。规划磁盘时应以实际 asset list 为准，并在下载后锁定 manifest 和 checksum。

它比 000688 更适合作为独立来源复验，而不是第一套 loader：每个 NWB 约 2.3--5.3 GB，包含 LFP，且四种 cursor task 需要额外事件标准化。

## 其他可用来源

### 原始 MC_RTT

[Zenodo record 3854034](https://zenodo.org/records/3854034) 官方元数据列出 48 个文件，其中 47 个是 session `.mat`：Indy 37 sessions、Loco 10 sessions，总计约 24.01 GB。数据包含 spike timestamps、sorted waveform snippets、finger/cursor/target position。

它适合验证与当前 DANDI 000129 MC_RTT 的同源扩展，但 ingestion 规则必须显式记录：每个 channel 的 unit 1 是 unsorted/hash，不能计入 SUA；其余 cluster 才作为 sorted units。连续 RT 数据也不能直接套 MC_Maze 的固定 trial window。

### DANDI 001868 与 001775

[DANDI 001868](https://dandiarchive.org/dandiset/001868/0.260715.2016) 是 2026 年发布的纵向 mouse S1 数据。官方描述为 sorted single-unit electrophysiology；live assets 中有 70 个 ephys sessions，覆盖 10 个有 ephys 的 subjects，且带 behavioral trials/wheel signal。它体量适中，但目标是 ICMS detection，不是二维运动学回归。

[DANDI 001775](https://dandiarchive.org/dandiset/001775/0.260418.0538) 有 85 files、6 mice、MountainSort4 spike times、position 和 trial intervals。它可测试架构是否超出 motor cortex 仍成立，但不应与 macaque reaching R² 直接合并。

## 容易误判但不应作为严格 SUA 主结果的数据

### DANDI 000070

[DANDI 000070](https://dandiarchive.org/dandiset/000070/0.260528.0919) 有 10 个 sessions、两个 monkeys、约 53.34 GB，表面上有 NWB `Units` 和 `spike sorting technique` metadata。但 POYO 数据附录明确说明 Churchland et al. 数据使用 threshold-crossing processing，所有 units 都是 multi-units。因此它适合 MUA 对照，不是严格 SUA 外部集。

### NeuroTask 001055

[DANDI 001055](https://dandiarchive.org/dandiset/001055/draft) 将原始 RT 数据压成 5 个 NWB、47 sessions、约 1.254 GB，适合快速 multi-session pipeline 原型。但它没有标准 `/units` event table，而是在 processing module 中保存预分箱 `spikes_counts`；waveform 和 electrode metadata 已移除，unit column 也是 session-local。其来源还包含每 session 的一个 multi-unit column，若用于 SUA 必须按 datasheet 排除。

因此 NeuroTask 可以验证模型代码能否跨 session 运行，不能替代 000688 对 SUA sorting/reliability 机制的验证。

## 决策

- P3 的主 benchmark 固定为 **DANDI 000688 的 C/M/J sorted subset**。
- 首个模型比较固定为 **`sub-C` CO 53-session chronological split**。
- DANDI 000121 作为独立实验室/任务的第二阶段 replication。
- 000070 和 000688 `sub-T` 明确归入 MUA，不进入 SUA 汇总。
- NeuroTask 仅作为 compact engineering prototype，并在结果表中单独标记为 pre-binned derived data。

在下载大数据前，应先把 asset path、size、DANDI version、SHA/dandi-etag 保存为 manifest。数据集后续更新不得静默替换本实验使用的版本。
