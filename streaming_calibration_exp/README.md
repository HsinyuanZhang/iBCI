# Streaming Calibration Experiment

独立实验目录，不修改 `SPINT-main/` 原始代码。数据与 teacher checkpoint 通过相对路径引用上级仓库。

## 环境

推荐使用已有 conda 环境（实测可用）：

```bash
conda activate ks4
cd /home/xinyuan/Work_host/SPINT/streaming_calibration_exp
pip install -e .  # 可选
```

官方 `spint` 环境当前缺少 torch；请勿直接依赖。

## 训练轮次上限

所有训练默认 **`max_epochs: 40`**（见 `configs/trainer/default.yaml`）。  
结构选择监控 **`val_heldin/r2_mean`**（held-out 仅用于最终对比，符合实验指南 §8）。

## 目录结构

```text
streaming_calibration_exp/
├── configs/          # Hydra 配置（B1–B6 变体 + experiment 预设）
├── src/
│   ├── models/components/
│   │   ├── spint.py                 # 自 SPINT-main 复制
│   │   ├── streaming_encoders.py    # B0–B6 encoder
│   │   └── streaming_spint.py       # frozen decoder 包装
│   ├── models/streaming_calibration_module.py  # 蒸馏训练 Lightning 模块
│   ├── data/falcon_datamodule.py    # 自 SPINT-main 复制
│   └── metrics/run_artifacts.py     # CSV / git / hardware 元数据
├── scripts/
│   ├── verify_gate0.py              # teacher baseline 元数据
│   ├── verify_gate1.py              # B0 vs B1 FP32 等价
│   └── smoke_check.sh
├── tests/
└── outputs/streaming_calibration/   # 实验产出（run_id 子目录）
```

## 审核修复（2026-07-10）

针对首轮审核的 10 项问题已修复，详见 `VALIDATION.md`（LOSO 方法学说明）。关键变更：

- 训练入口路径/run_id/硬件成本写入已修复
- `test_step`、EarlyStopping、TensorBoard 已可用
- B2–B6 参数量与 MAC 公式对齐文档；B5/B6 trial feature 已改正
- B4–B6 支持 `trial_length` 与 padding 掩码；raw bins 配置可选

## 快速验证（审核前建议运行）

```bash
conda activate ks4
cd streaming_calibration_exp
bash scripts/smoke_check.sh
```

## 下一轮实验（修订计划 2026-07-11）

修订执行方案见上级目录 `SPINT_streaming_calibration_revised_experiment_plan_2026-07-11_1146.md`。

### Phase R0 训练前检查

```bash
bash scripts/pre_run_checks_revised.sh
```

### Phase R1 Immediate Round（3 次新训练 + 复用 anchor）

```bash
bash scripts/run_immediate_round.sh
```

顺序：`B2-D512 protocol control` → `B3-D64 task_only` → `B3-D64 task_plus_y`；`B3-D64 task+y+E` 复用已有 anchor。

结果写入 `outputs/streaming_calibration/gate2_revised_matrix.csv`。

### Phase R1 决策评估

仅在 Immediate Round 三个新训练全部完成并注册后使用：

```bash
python scripts/evaluate_r1_decisions.py --fold 0 --seed 42 --emit-loss-overrides
```

退出码：`0`=已选出 winner；`1`=R1 数据不完整；`2`=stop architecture sweep；`3`=R² 平局需 seed 43。

**数据不完整或平局时不会返回 `winning_loss`。**

### Phase R2/R4 条件轮（需 R1 evaluator 返回 exit 0）

```bash
bash scripts/run_conditional_round.sh task_plus_y   # 必须与 formal winner 完全一致
```

脚本会先调用 `--require-ready --require-winner`；CLI loss 与矩阵 winner 不一致时拒绝启动。

### 手动注册已完成 run

```bash
python scripts/update_gate2_matrix.py outputs/streaming_calibration/<run_id> --refresh-d512-deltas
```

## 下一轮实验（LOSO anchor）

先导出并晋升 canonical B0 baseline（只需一次，或重复性验证时带时间戳）：

```bash
python scripts/export_b0_baseline.py --promote-to-canonical
python scripts/verify_gate0.py
```

LOSO fold 训练示例：

```bash
bash scripts/run_loso_fold.sh b2_d128_anchor 0 seed=42
bash scripts/run_loso_fold.sh b3_d64_anchor 0 seed=42
```

诊断 / loss ablation 预设：`b3_d128_diag`、`b2_d256_diag`、`b3_d64_task_only`、`b3_d64_task_plus_y`。

## 训练示例

```bash
# B2 screening（40 epochs，frozen decoder + 蒸馏）
python src/train.py experiment=b2_d128 seed=42

# B3
python src/train.py experiment=b3_d64 seed=42

# B16 MUA architecture-reuse pilot（协议已锁定：M2 LOSO fold 0 + held-out，seed 42）
python src/train.py experiment=b16_m2_loso

# B15 对应控制（第二顺位）
python src/train.py experiment=b15_m2_loso

# B4/B5/B6 screening（cubic，默认）
python src/train.py experiment=b4_stats
python src/train.py experiment=b5_ema_r4
python src/train.py experiment=b6_fir_r4_k5

# B4 raw bin streaming（分布与 cubic teacher 不同）
python src/train.py experiment=b4_stats data=falcon_m2_raw_bins
```

可选覆盖：

```bash
python src/train.py experiment=b2_d128 trainer.max_epochs=40 model.loss_mode=task_only
python src/train.py experiment=b2_d128 model.id_hidden_dim=64 model.lambda_E=0.0
```

## Loss ablation（§7）

通过 `model.loss_mode` 切换：

| 值 | 含义 |
|---|---|
| `task_only` | 仅 behavior MSE |
| `task_plus_y` | + prediction distillation |
| `task_plus_y_plus_E` | + identity distillation（默认） |

## Teacher checkpoint

默认路径（Gate 0 matched baseline）：

`../SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt`

held-out test R2 ≈ 0.236（同前次 eval）。

可在 `configs/paths/default.yaml` 或命令行覆盖：

```bash
python src/train.py model.teacher_ckpt_path=/abs/path/to/epoch_034.ckpt
```

## 已实现 vs 待完成

| 模块 | 状态 |
|---|---|
| B0/B1 streaming encoder + Gate 1 脚本 | ✅ |
| B2–B6 encoder（参数量/MAC 对齐） | ✅ |
| B15/B16 encoder + M2 LOSO 配置 | ✅（B16 fold 0 / seed 42 已完成；B15 待运行） |
| Frozen decoder + 蒸馏 + test metrics | ✅ |
| 40 epoch + early-stop + TensorBoard | ✅ |
| 训练后 hardware_cost.json / metrics 行 | ✅ |
| metrics_summary + metrics_per_session 导出 | ✅ |
| LOSO / rotation_5_2 验证协议 | ✅ |
| 训练期 held-out 隔离（仅 test 评估） | ✅ |
| B0 baseline 导出脚本 | ✅ |
| LOSO session CV 批量跑法 | ⏳ 用 `scripts/run_loso_fold.sh` |
| W8A8 量化 / progressive stop | ⏳ 未实现 |

## 产出位置

每次 `src/train.py` 会在 `outputs/streaming_calibration/<run_id>/` 写入：

- `resolved_config.yaml`
- `environment.txt`
- `git_state.txt`
- `source_manifest.json`
- `teacher_metadata.json`
- `hardware_cost.json`
- `metrics_summary.csv`（design + aggregate test）
- `metrics_per_session.csv`
- `checkpoint_manifest.json` + `checkpoints/best.ckpt`

`run_id` 格式：`{variant}_s{seed}_{timestamp}`，避免重复运行覆盖。

Hydra 训练日志与 checkpoint 在 `logs/train/runs/<timestamp>/`。
