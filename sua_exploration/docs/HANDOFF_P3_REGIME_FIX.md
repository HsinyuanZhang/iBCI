# HANDOFF: P3 Cross-Session — Regime Fix

**日期：2026-07-24**
**给：下一位接手的 AI**
**状态：Step 0/1/2 已跑完，但发现致命混淆，需要修正 split 后重做**

---

## 一句话

P3 的 chronological split 把 sub-C 的 unit 数量级跳变（train ~60 → test ~250）切成了 train/test 边界，所有跨 session 结论被混淆。下一步是在同 regime 内（39 sessions, 38–91 units）重做 split + Step 1 + Step 2。

---

## 发生了什么

1. P3 目标：测 NeuronID encoder 在**跨 session SUA**（DANDI 000688 sub-C CO）上能否泛化。
2. Step 0（单 session 上界）：端到端 R²=0.694，pipeline 正确。✓
3. Step 1（跨 session 端到端 zero-shot）：test R²=-0.124，看起来"跨 session 失败"。
4. Step 2（few-trial finetune）：enc_ft K=20 R²=0.692，看起来"20 trial 恢复"。
5. **然后发现**：sub-C 的 unit 数在 2016-09 前后断崖跳变。

| Split | sessions | unit 范围 | 均值 |
|---|---|---|---|
| Train (37) | 2013–2015 全部 | 38–91 | 59 |
| Val (8) | 2015-11 ~ 2016-09 | 53–353 | 232 |
| Test (8) | 2016-09 后全部 | 188–299 | 245 |

train 全是 ~60 units，test 全是 ~250 units。Step 1/2 实际测的是 "4× N regime 跳变"，不是 "同 regime 跨 session"。

## 架构对可变 N 的立场

- **架构支持可变 N**：encoder 参数 N-free（per-neuron shared weights），decoder cross-attention 输出与 N 无关。已用 NumPy 验证 N: 96→66 前向正确。
- **但设计预期是 ±30% 波动**（"今天 95 个 unit，明天 70 个"），不是 4× 跳变。
- POYO 用 unit dropout 应对的也是小波动，不是 4×。
- **用户明确确认**：目标是同 regime 内的波动，不需要处理 4× 跳变。

## 修正后的 split

旧 regime（unit < 100）有 39 个 session，日期 2013-10 到 2015-12。推荐 27/6/6 chronological split：

```
sessions: 39 个 sub-C CO sessions（unit 38–91）
train: 27（2013-10 ~ 2015-07）
val:   6 （2015-07 ~ 2015-11）
test:  6 （2015-11 ~ 2015-12）
```

旧 regime 的最后一个 session 是 `sub-C_ses-CO-20151201`（53 units）。新 regime 从 `sub-C_ses-CO-20160909`（246 units）开始。

## 具体要做的事

### 1. 改 split 逻辑

`mc_maze/multisession_datamodule.py` 的 `discover_nwb_files` + `chronological_session_split` 当前按日期切分 53 个 session。需要加 unit-count 过滤：只保留 unit < 100 的 39 个旧 regime session，然后按日期 27/6/6 split。

最简单的改法：在 `discover_nwb_files` 或调用处加一个 `max_units` 过滤（需打开 NWB 读 `len(units_df)`），或者硬编码 39 个旧 regime session 列表。

### 2. 重做 Step 1

```bash
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/spint/bin/python -u \
  sua_exploration/scripts/train_variant_dandi688.py \
  --variant B3 --task CO --split_counts 27,6,6 \
  --max_units_exclusive 100 \
  --loss_mode task_only \
  --out_name b3_dandi688_co_oldregime_s42 \
  --seed 42
```

`--max_units_exclusive 100` 严格保留 unit **小于** 100 的 session（不包含 100）。

Step 1 仅训练并以 validation sessions 选择、锁定最佳 checkpoint；它不运行 held-out test。validation 同样保持 trial-disjoint：每个 validation session 用 `trials[0:N]` 作 calibration，验证窗口只来自 `trials[N:]`。若 B3/B15/B16 都是候选，必须都在 test 前只用 validation sessions 完成结构选择和调参，再锁定最终 checkpoint；随后用 `select_gradient_free_protocol_dandi688.py` 仅在 validation 上 sweep 并写入 protocol lock。

历史结果中的 `zero_shot` 实际仍以 calibration trials 前向计算 identity；它不是本次新增的真正 no-calibration control。新的 `zero_identity_no_calibration` 才是不读 calibration spikes、也不调用 identity encoder 的严格对照。

### Validation-only Protocol Selection

```bash
~/miniconda3/envs/spint/bin/python -u \
  sua_exploration/scripts/select_gradient_free_protocol_dandi688.py \
  --ckpt /absolute/path/copied-manually-from-run_metadata-best_checkpoint.ckpt \
  --variant B3 --task CO --split_counts 27,6,6 \
  --max_units_exclusive 100 \
  --calibration_ns 10,20,30,50 \
  --selection_modes first,direction_coverage \
  --pool_size 50 --seed 42
```

将 `--ckpt` 替换为 Step 1 `run_metadata.json` 中 `best_checkpoint` 的唯一绝对路径，手工逐字复制；禁止 shell substitution、glob 或多个 checkpoint。该脚本只读取 train sessions 的行为统计和 validation sessions 的 trial/spike/behavior 来选择协议，绝不加载或评估 test 的 spikes、behavior 或 trial data；它仅使用预注册 split 名称和 unit-count metadata 做 regime 审计。它会写出完整 validation result 与供 Step 2 消费的 protocol lock。

### 3. 重做 Step 2：冻结的 gradient-free streaming calibration

```bash
CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/spint/bin/python -u \
  sua_exploration/scripts/eval_adaptation_dandi688.py \
  --protocol_lock sua_exploration/results/p3_gradient_free_protocol_selection_b3_s42_lock.json \
  --out_name b3_dandi688_co_oldregime_s42
```

先运行 validation-only 的 `select_gradient_free_protocol_dandi688.py` 并使用其完整 validation lock。正式 Step 2 只传 `--protocol_lock`，由 lock 权威指定 checkpoint、SHA-256、split/regime、calibration 数、选择模式和 pool size；不传裸 `--ckpt`、不使用 glob 或命令替换。

同样，`--max_units_exclusive 100` 表示严格 `unit < 100`。Step 2 输出两项：`gradient_free_calibrated` 用 pool 内的 few-shot calibration spikes 前向计算 identity，`zero_identity_no_calibration` 则完全不读 calibration spikes、也不调用 identity encoder，而是向公开 decoder 接口传入全零 identity；它是非学习 control，不是 learned population prior。两者共享同一 pool 后 evaluation trials/windows，并报告 paired delta。正式 test 必须以 `--protocol_lock` 消费 validation-only 锁定的 checkpoint、calibration 数、选择模式和 pool size；encoder 和 decoder 均冻结，绝不以 held-out behavior labels 做反向传播或权重更新。正式 test 失败只记录结论；后续应使用新数据或独立 replication，不能改变协议后重跑该 test。

如需复现历史上的诊断性 oracle 对照，可显式加 `--run_diagnostic_oracle_baselines --ks 5,10,20 --finetune_epochs 30`。它会使用 held-out behavior labels 和 backward gradients，因此不属于 Step 2 目标、部署协议或后续决策依据。

### 4. 尚未实现的 Gradient-free 适配探索（用户关心的方向）

用户明确指出："不需要梯度的 finetune 在我们的实验范围内吗"——架构卖点就是 streaming calibration（推理时算 identity，无梯度）。带梯度的历史实验只能作为 oracle 对照，不能作为方法有效性结论。

这些仅可在正式 Step 2 前的 validation-only protocol-selection 阶段探索（全部纯前向）；正式 Step 2 后不得改变这些设置或重跑 held-out test，失败只记录结论并转向新数据或独立 replication protocol：
- 增加 calibration trial 数（现固定 10→试 20/30/50）
- 校准 trial 选择（方向覆盖采样 vs 固定前 N 个）
- 闭式 identity 归一化（用矩统计对齐 test session 的 identity 分布到 train 域）

---

## 已有产物（不要删）

| 产物 | 路径 | 说明 |
|---|---|---|
| Step 0 checkpoint + JSON | `checkpoints/b3_ss_sub-C_ses-CO-20131003/` | 单 session 上界，有效 |
| Step 1 checkpoint + JSON | `checkpoints/b3_dandi688_co_e2e_s42/` | **被混淆**，保留作为对比 |
| Step 2 JSON | `results/p3_step2_adaptation_b3_dandi688_co_e2e_s42_seed42.json` | **被混淆**，保留作为对比 |
| Step 1 frozen decoder | `checkpoints/b3_dandi688_co/` | 旧 frozen teacher run，保留 |
| Step 2 脚本 | `scripts/eval_adaptation_dandi688.py` | 默认冻结的 gradient-free streaming calibration；oracle 仅显式 opt-in |
| Step 1 训练脚本 | `scripts/train_variant_dandi688.py` | 已加 `--freeze_decoder` / `--loss_mode` 开关，复用 |
| 单 session 脚本 | `scripts/train_single_session_dandi688.py` | 复用 |

## 环境信息

- Conda: `spint`（torch 2.5.1, CUDA, 2× RTX 3090）
- 数据: `sua_exploration/data/dandi_000688/sub-C/`（53 CO NWB，已下载）
- sub-J 和 sub-M 未用于 P3（留作 replication）

## 关键认知

1. **跨 session 的 unit 不同**：spike sorting 是 per-session 做的，unit ID 只在单 session 内有意义。这是 NeuronID encoder 要处理的设置，不是 bug。
2. **架构卖点是 gradient-free**：streaming calibration = 读 calibration trial → 前向算 identity → 冻结 decoder。梯度 finetune 不在方法论预期内（虽然有效）。
3. **P3 之前所有跨 session 结论被 4× N 跳变混淆**，不能作为架构有效性的证据。

## 参考文档

- [`docs/P3_CROSS_SESSION_ANALYSIS.md`](P3_CROSS_SESSION_ANALYSIS.md) — 完整实验记录 + critical finding
- [`ROADMAP.md`](../ROADMAP.md) — 总执行计划（P3 段已更新）
- [`docs/CURRENT_RESULTS.md`](CURRENT_RESULTS.md) — 结果摘要（P3 段待更新）
- [`docs/DANDI_000688_FEASIBILITY.md`](DANDI_000688_FEASIBILITY.md) — 数据可行性，含可变 N 讨论
