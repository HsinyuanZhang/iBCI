# 审核：EXP1 / EXP2 / EXP3 V2 工单

日期：2026-09-07 11:45。对象：
- [WORKORDER_EXP1_E0_RELIANCE_V2_20260907.md](WORKORDER_EXP1_E0_RELIANCE_V2_20260907.md)
- [WORKORDER_EXP2_688_PAIRED_UTILITY_V2_20260907.md](WORKORDER_EXP2_688_PAIRED_UTILITY_V2_20260907.md)
- [WORKORDER_EXP3_STREAMING_STRUCTURE_V2_20260907.md](WORKORDER_EXP3_STREAMING_STRUCTURE_V2_20260907.md)

只读审核；不改工单原文、不启动任务。每条"核实"均对照工作区文件在 11:36–11:42 之间检查。标记：**[阻断]** / **[必改]** / **[建议]** / ✓ 核实通过。

## 0. 总判

三份 V2 已吸收前一轮审核（[REVIEW_E0_UTILITY_AND_688_COSTAWARE_V1](REVIEW_E0_UTILITY_AND_688_COSTAWARE_V1_20260907.md)）的全部阻断/必改项，且在几处比审核稿更严谨（EXP2 §5 对 AdamW decoupled weight decay 的说明；EXP3 §6 P1 "零缓存 ≠ oracle 冷启动"；EXP3 §1 拒绝把探针数字当地板）。**三份均可进入执行授权**，前提是下列 1 个共同阻断与少量必改落地。

**共同阻断（运营而非设计）**：EXP1 的 M1 主候选与 EXP3 的 D4-ZP 基线都指向 `results/m1_projadd_depth2/20260907_gpu0/depth4/`，而该目录**不存在，也没有任何进程/队列脚本准备启动 `--stage train --depth 4`**（11:40 核实：`ps` 中无 depth4 进程、无等待脚本；GPU0 已空闲 0%/455 MiB）。depth-2 已于 11:38 完成 24/24 epoch 并出 `score_receipt.json`（LOSO 20120924 R² 0.6886，source-minival 0.835）。若 depth2 线的负责人不补启 D4（约 45 min）+ 两个 score stage，EXP1-M1 将停在 `WAITING_FOR_FROZEN_CHECKPOINT`，EXP3 §4 的分支规则无法读取。两份工单对此的处理（等待、不擅自顶替）是正确的；需要的是**用户/depth2 线补授权启动 D4**，不是改工单。

## 1. EXP1（E0 reliance V2）

| 项 | 结论 |
|---|---|
| §2 H1 `e32 EMA` 可用性 | ✓ `epoch_032.pt` 含 `raw_state_dict` 与 `ema={decay:0.9995, n_updates:23392, shadow(78 keys)}`；EMA shadow 完整 |
| §2 M2 `ext4_scan_receipt.json` | ✓ 存在于 `results/m2_projadd/20260906_104300/`（另有 `ext6_epoch_pick/`，工单已禁止改用 ✓） |
| §2 M1 D4 e24 | **不存在**（见 §0）；工单的 `WAITING_FOR_FROZEN_CHECKPOINT` 处理正确 |
| §6 H1 grouped metric | ✓ `c2_protocol.py::grouped_session_metrics` 按 `key.split("_set_")[0]` 把 14 录音归为 S6–S12 共 7 组；bootstrap 单位应为 **7 组**，工单"14 录音不等于 14 session"正确 |
| §4 十条流、`identity_mode=zero` 禁用、P 无 bias 核验 | ✓ 与代码一致（`identity_variant.py`） |
| §4 "即使直接 forward，若传入预折叠 static 也须按臂重建" | ✓ 正确：`forward_static_folded` 接受外部 static；直接 `forward()` 不接受 static，天然安全 |

**[必改]**
1. §4 C-ZERO 的"归一化后 c=0"在三任务下含义不同，manifest 须逐任务写明：M1 rSyn3 = 源 RMS 缩放（无中心化）→ 0 即"系数全零"；H1 H-C = normalized EB → 0 的含义取决于 normalizer 是否中心化（若中心化则 0 = 源均值 carrier）；M2 MOVE-T4 = 源 z-score → 0 = 源均值。三者都表示"无单元特异 carrier 信息"，但只有 z-score/中心化情形等价于"均值 carrier"，解释表须区分。
2. §2 M2 行补"pick 权重是 EMA 还是 RAW"（读 `ext4_scan_receipt.json` 的 `view`），与 M1/H1 的 EMA 口径对齐后写入 manifest。

**[建议]**
3. §2 次级 exposed 诊断（M1 全 4-session e24、H1 full13 e24）成本极低，建议列为默认执行而非"仅在核清后"；它们与 clean 主读数并排，恰好能显示暴露是否夸大 reliance。
4. M1 单 session 的 trial 块 bootstrap：M1 有 trial 结构（M10 支持 + query trial 10..210），优先用 trial 块而非 5 s 块；工单已给两条路径，建议 manifest 预先固定选 trial 块。

## 2. EXP2（688 paired utility V2）

| 项 | 结论 |
|---|---|
| §6 A 档 58 ms/update 上限 | ✓ 算术：6 GPU-h × 0.7（30% 余量）/ 288k ≈ 52–58 ms，量级正确 |
| §3 `_go_cues` 引用 | 定义在 **`streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py:37`**（经 `sua_exploration/mc_maze/subm_v9_f0_pv_ridge` 导入）；语义：`go_cue_time_array` 为每 trial 一行、列数 = 最大 cue 数、缺位 NaN。工单应直接写这个路径 |
| §3 多 cue | **需补规则** [必改]，见下 |
| §5 units=100 / pad / E0 按 donor 原生 N 生成 | ✓ 与 `model.py` 约束和 SPINT encoder 用法一致 |
| §5 AdamW decoupled wd 使零输入分支权重仍变化 | ✓ 正确，优于审核稿"W_c dead"的说法 |
| §4 donor 审计 | ✓ B0 s42 receipt：manifest SHA 同、test 未装载、val 为 loader_0、`calibration_n_trials: 30`；teacher 影响链仍需按工单执行 |
| §7 B0 e11 同面参考 | ✓ B0 receipt `validation_protocol`：calibration `trials[0:30]`、evaluation `trials[30:]`、trial-disjoint —— 与本工单 M30/q30 **同构**，B0 参考在合同层面可比（仍须核对窗/mask/last-bin 规则） |

**[必改]**
1. §3 事件合同补多目标 trial 规则：688 trials 有 `num_targets` 列，`go_cue_time_array` 每 trial可含多个 cue（`_complete_reaches` 用 `cues[trial, :count]` 解析）。MOVE700 要求每支持 trial **恰一个方向**，故：`num_targets != 1` 或 cue 数 ≠ 1 的支持 trial → 在前 30 内排除并披露，不取首 cue；同时报告每 session 被排除数。CO 会话预期几乎全为 1，但必须证明而非假定。
2. §2 命名 688 评分器：建议明写复用 `sua_exploration/scripts/eval_epoch_window_dandi688.py` 的窗/mask/last-bin 规则（B0 receipt 指定的 M3 确定性协议入口），使 FULL/NOE0/NOC 与 B0 参考"同面"有代码级依据；若自写 scorer，须对 B0 e11 做一次数值一致性核对。

**[建议]**
3. §6 S1 8k 检查的"同点常数基线"：建议明确为"支持区（前 30 trial）行为均值"，与 B0 协议一致。
4. §7 加一行：2 seeds 均报后，若两 seed 符号相反，主结论为 `UNCERTAIN`，不取均值定性。

## 3. EXP3（streaming structure V2）

| 项 | 结论 |
|---|---|
| §3 depth2 配方数字 | ✓ `depth2/run_meta.json`：`total_updates 92016`、`updates_per_epoch 3834`、`warmup_updates 3834`、`peak_lr 1e-4 → lr_min 1e-5`、`ema_decay 0.9995`、`seed 42`、sessions 26/27/28、heldout 20120924 |
| §5 "`SharedCausalConv.forward` reshape 按输入 width 写死，不能只改 left_pad=0" | ✓ 代码确认：`reshape(batch, n_units, C, width)` 用输入 width，valid conv 输出 width−4 会失配 |
| §6 P1 "零 frontend 缓存 ≠ oracle 冷启动" | ✓ 正确（conv bias、static 项、set 网络非线性） |
| §4 D4-ZP "已有匹配基线" | **尚不存在**（§0）；D2-ZP 已完成：LOSO 0.6886 / minival 0.835，但 D2 门（ΔR² ≥ −0.01 vs D4）在 D4 出来前不可读 |
| §6 P2 15% B4 成本门 | ✓ 可达：探针 P5→P1 在 M1 B4/t1 约 −5.9/21.5 ≈ −27%；工单坚持实测而非比例 ✓ |
| §7 累计门（最终候选 vs D4-ZP ≥ −0.01） | ✓ 正确防止 0.01+0.01 叠加 |

**[必改]**
1. §5 训练采样：写明 HC 与同 depth ZP **共享 endpoint 流 (session, end) 与 dropout RNG**，HC 仅把 raw 切片向前多取 4 bin；end < 103 的窗按 reset 语义补零（每 session ≤4 个）。这样"匹配"是端点级的，`sampler_batch_sha256` 必然不同，须另存 `endpoint_stream_sha256` 作为匹配证明。
2. §4 D2 已完成，工单应把"D2-ZP 已落盘 0.6886（LOSO）"作为事实录入，并明确分支裁决**等待 D4-ZP**；不得用 QueryAge-family 0.658 或 Original 0.798 代替 D4 读门。

**[建议]**
3. §6 P2 计时也报 t2（提交容器实际用 t2），并与运行时线 `p1_timing_fast_t2.json`（B4 12.41 ms）同口径。
4. HC wrapper 的 raw 历史缓冲从 100 扩到 104；`observe()` 路径不变，写进 `history_contract.json`。

## 4. 三份工单之间

- EXP2 固定 D4 + 窗内补零 ✓ 与 EXP3 隔离；EXP3 §8 已写"若最终模型换 HC，EXP1 需在最终模型上复读" ✓。
- EXP1 的 C-ZERO 读数只作 EXP2 NOC 的先验，EXP2 §0 已明确不用其做晋级选择 ✓。
- 资源：EXP1 ≤1 GPU-h 推理、EXP2 ≤6 GPU-h、EXP3 ≤3 GPU-h + D4 基线（归 depth2 线）。当前 GPU0 空闲，GPU1 被 H1 C2-protocol 评分占用。

## 5. 给用户的裁决点

1. **是否立即由 depth2 线启动 D4-ZP 基线（GPU0 空闲，≈45 min + score）**——EXP1-M1 与 EXP3 分支都依赖它。
2. 三份工单在落实上述 [必改] 后可授权执行；建议顺序：EXP1 P0/P1（只读绑定 + parity，不占 GPU）→ EXP2 S0 CPU 审计 → EXP3 P0/P1 实现与 parity（CPU）；训练类（EXP2 S1、EXP3 P3）等各自 P0 闭合后再授权。
