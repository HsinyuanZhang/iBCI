# SPINT Streaming Calibration 修订实验计划

> 创建时间：2026-07-11 11:46 HKT  
> 状态：进入下一轮训练前的执行基线  
> 适用目录：`streaming_calibration_exp/`  
> 上游计划：`SPINT_streaming_calibration_AI_experiment_guide_2026-07-10_1622.md`  
> 修订原因：B2-D128 与 B3-D64 在 LOSO fold 0 均通过 MAC 条件、未通过 Gate2 精度条件；当前结果混合了结构压缩、student 训练协议和跨-session 泛化损失。

## 0. 执行结论

下一轮不直接展开完整 B2/B3/B5/B6 网格，只执行三个诊断 run：

1. `B2-D512 protocol control`：测量相同 student/frozen-decoder/LOSO 训练配方的容量与泛化上限。
2. `B3-D64 task_only`：移除全部 teacher distillation。
3. `B3-D64 task_plus_y`：只保留 prediction distillation，移除 identity distillation。

现有 `B3-D64 task_plus_y_plus_E` fold-0 anchor 作为第三个 loss 对照，不重复训练，前提是下一轮不改变 seed、trainer deterministic、epoch、数据预处理或 teacher checkpoint。

读取上述结果后才条件执行：

- B3-D128 容量恢复点；
- B2-D128 使用获胜 loss 的复核；
- B5-R4 与 B6-R4-K5 的 fold-0 最小硬件探针；
- 完整七折 LOSO 与多 seed 确认。

在结构和 loss 锁定前，禁止：

- 评测六个 FALCON benchmark held-out sessions；
- 进入量化、progressive stop 或 decoder fine-tune；
- 同时改变 encoder、loss、预处理和训练预算；
- 用 D512 诊断基准替换预注册的固定 B0 Gate2 基准。

## 1. 当前证据基线

固定 B0 fold-0 session R2：

```text
session = ses-2020-10-19-Run1
B0 R2 = 0.67167222
baseline CSV SHA256 = 8c316c48768cb33025e7435d113819d88d3866f1f13fb63bed10e7d7c6a17225
teacher checkpoint SHA256 = fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec
```

最新 anchors：

| Candidate | R2 | Delta vs B0 | MAC/session | MAC reduction | Peak state | Gate2 fold-0 |
|---|---:|---:|---:|---:|---:|---|
| B2-D128 | 0.61882681 | -0.05284541 | 148,119,552 | 12.665x | 87,552 B | Fail accuracy |
| B3-D64 | 0.63024879 | -0.04142343 | 21,368,832 | 87.788x | 62,976 B | Fail accuracy |

当前 anchors 已确认：

- 正常训练、测试和导出；
- 使用相同 baseline snapshot；
- checkpoint hash 与最佳 epoch 匹配；
- fold 0 训练六个 held-in sessions，验证一个 held-in session；
- benchmark held-out 在 fit/test 均未加载；
- B2/B3 source manifest 相同；
- 不是 baseline 漂移、checkpoint 选错或 partial run 导致的失败。

## 2. 修订后的问题分解

### Q1：student 训练协议本身能否在 fold 0 接近 B0？

使用 B2-D512：

```text
原始 3+3 LatePool topology
D = 512
frozen decoder
train sessions = fold 0 的其余六个 sessions
validation session = A
loss = 当前 task+y+E
```

注意：当前实现只为 B0/B1复制 teacher IDEncoder 权重。B2-D512 是随机初始化的 student encoder，不是 exact teacher replica。因此它回答：

```text
在相同 student 优化配方和 LOSO 数据暴露下，
未压缩 3+3 encoder 能达到多高？
```

它不参与 Gate2 硬件候选排名。

### Q2：identity distillation 是否损害 unseen-session task R2？

固定 B3-D64，只比较：

```text
task_only
task_plus_y
task_plus_y_plus_E  # 复用当前 anchor
```

除 loss 外不得改变任何变量。

### Q3：若协议上限足够，增加 B3 容量能否恢复精度？

使用 B3-D128，并继承 Q2 的获胜 loss。

B3-D128 仍满足 Gate2 的 10x MAC reduction；虽然 nominal peak state 可能高于 64 KiB，它仍可用于判断 EarlyPool 的容量趋势和 Gate2 literal accuracy/MAC 条件。

### Q4：无 trial-buffer 结构是否值得保留为 minimum-hardware 点？

在 loss 选定后，才比较：

```text
B5 EMA R=4
B6 FIR R=4, K=5
```

第一步使用 cubic T=100，保持与 teacher/B3 的输入分布一致，只回答结构精度问题。若候选有希望，再切 raw-bin 路径验证取消 interpolation 与真实 valid-bin streaming；两步不得合并。

## 3. 固定实验口径

### 3.1 两种 delta 必须同时保留

官方 Gate2 指标不改变：

```text
delta_fixed_B0 = R2_candidate_LOSO - R2_fixed_B0
```

新增诊断指标：

```text
delta_vs_D512_LOSO = R2_candidate_LOSO - R2_D512_LOSO_control
```

用途：

- `delta_fixed_B0`：回答部署性能距离当前 SPINT baseline 多远；用于 Gate2。
- `delta_vs_D512_LOSO`：回答相同 student/LOSO 配方下，压缩本身额外损失多少；只用于诊断。

禁止根据结果用 D512 重新定义 Gate2 baseline。

### 3.2 Session 角色

每个 LOSO fold：

```text
6 held-in sessions -> student encoder training
1 held-in session  -> validation/checkpoint selection
6 benchmark held-out sessions -> 不加载、不评测
```

LOSO session 对 student encoder 是 unseen；对 frozen teacher/decoder 不是严格 unseen。报告中必须写成：

```text
encoder-level LOSO under a fixed all-held-in pretrained decoder
```

不得写成 full-model LOSO。

### 3.3 下一轮固定项

```text
fold = 0
seed = 42
M = 33
T = 100 cubic
batch_size = 32
optimizer = Adam, lr=1e-4
freeze_decoder = true
max_epochs = 40
early-stop monitor = val_heldin/r2_mean
patience = 10
trainer.deterministic = false  # 为与现有 anchors 单变量比较，下一轮不改变
include_heldout_in_fit = false
include_heldout_in_test = false
```

`deterministic=false` 是复现风险，但此时切换会使现有 `task+y+E` anchor 失去严格对照。随机性在候选确认阶段通过 seeds 42/43/44 量化；不得在同一比较矩阵中途切换 deterministic 设置。

## 4. Phase R0：训练前配置与证据检查

### 4.1 新增显式 presets

建议新增而不是依赖长 CLI override：

```text
configs/experiment/b2_d512_protocol_control.yaml
configs/experiment/b5_ema_r4_loso_probe.yaml
configs/experiment/b6_fir_r4_k5_loso_probe.yaml
```

其中每个 preset 必须显式记录：

```yaml
data:
  validation_protocol: loso
  include_heldout_in_fit: false
  include_heldout_in_test: false
model:
  freeze_decoder: true
```

B5/B6 loss 不在 preset 中默认为最终值；必须等 Phase R1 决定后，将获胜 loss 固化为新 run config。

### 4.2 训练前检查

执行以下只读或短时检查：

1. `verify_gate0.py` 验证 baseline CSV、checkpoint 和 hash。
2. 运行当前 unit tests。
3. 对三个下一轮 presets 做 Hydra `--cfg job` composition。
4. 检查 split manifest 预期：6 train / 1 validation / held-out false-false。
5. 检查 teacher hash 与当前 anchors 相同。
6. 检查 run ID 包含 variant、width/loss、fold、seed 和时间戳。
7. 确认没有正在运行的同名训练进程。

任一项失败则不得启动长训练。

## 5. Phase R1：协议上限与 Loss Ablation

### 5.1 运行矩阵

按顺序执行，不并发占用同一 GPU：

| Order | Run role | Variant | Width | Loss | New run? |
|---:|---|---|---:|---|---|
| 1 | protocol control | B2 | D512 | task+y+E | Yes |
| 2 | loss ablation | B3 | D64 | task_only | Yes |
| 3 | loss ablation | B3 | D64 | task+y | Yes |
| Ref | loss ablation | B3 | D64 | task+y+E | Reuse current anchor |

每完成一个 run 立即导出并检查：

- best checkpoint 与 hash；
- fold/session split；
- R2 与 `delta_fixed_B0`；
- normalized identity MSE；
- prediction distillation MSE；
- best epoch 与最后 epoch；
- hardware profile；
- baseline/reference/source manifests。

### 5.2 D512 判据

| D512 fold-0 delta vs fixed B0 | 解释 | 决策 |
|---:|---|---|
| `>= -0.01` | student/LOSO 配方上限足够 | 重点处理压缩和 loss，进入 R2 |
| `[-0.03, -0.01)` | 协议与压缩可能共同限制 | 允许 R2，但同时保留协议风险 |
| `< -0.03` | 即使 D512 也明显落后 | 暂停宽度网格；先重新审视 teacher/student 初始化、训练目标与 matched baseline 口径 |

### 5.3 Loss 选择判据

主指标：fold-0 `delta_fixed_B0`，其次为 R2、prediction MSE、identity MSE。

```text
若两个 loss 的 R2 差 > 0.005：选择较高者。
若差 <= 0.005：视为单 seed 平局，追加 seed 43 fold-0 tie-break。
若 seed 43 仍平局：优先使用更少的 distillation 项。
```

identity MSE 不作为单独的选择指标；它只能解释 student 是否逼近 teacher E，因为当前证据已经显示更低 identity MSE 不保证更高 task R2。

### 5.4 R1 停止条件

若 D512 `< -0.03` 且三个 B3 loss 均 `< -0.03`：

```text
停止结构扩展和 B5/B6 长训练。
下一步改为训练协议诊断，而不是继续压缩。
```

## 6. Phase R2：Gate2 容量恢复

### 6.1 首选 run

```text
B3-D128
loss = Phase R1 winner
fold = 0
seed = 42
其他设置不变
```

### 6.2 条件 run

只有在 D512 `>= -0.03`、但 B3-D128 仍不理想时，才考虑：

```text
B2-D128 + winning loss
```

用途是判断 B2 之前的失败是否主要来自 identity loss，而不是重新启动全部 B2 宽度 sweep。

B2-D256 不作为优先 Gate2 恢复点，因为约 510M MAC 只提供约 3.7x reduction，无法满足 Gate2 的 `>=10x` MAC 条件；仅在需要绘制完整容量趋势时运行。

### 6.3 Fold-0 promotion zones

| Fold-0 delta | 状态 | 后续 |
|---:|---|---|
| `>= -0.01` | Green | 直接进入七折 seed-42 LOSO |
| `[-0.02, -0.01)` | Amber | 先运行 folds 1、2；三折平均优于 -0.02 才补齐七折 |
| `< -0.02` | Red | 不作为 high-fidelity 候选扩展 |

使用预先固定的 folds 1、2，禁止根据已知 baseline 高低挑选“容易的 folds”。

## 7. Phase R3：完整 Gate2 LOSO

对被 promotion 的候选完成 seed 42、folds 0..6。

### 7.1 聚合口径

```text
mean_delta = mean(session_delta over 7 folds)
worst_delta = min(session_delta over 7 folds)
mean_R2 = mean(session_R2 over 7 folds)
```

同时报告：

- 七个 session 独立 R2/delta；
- best epoch 分布；
- identity/prediction MSE 分布；
- mean、median、std；
- 失败 session 的 firing/calibration 统计；
- MAC、peak state、weight bytes。

### 7.2 Gate2 判据

Literal Gate2：

```text
mean_delta >= -0.01
ID MAC reduction >= 10x
```

High-fidelity Pareto：

```text
mean_delta >= -0.01
worst_delta >= -0.03
peak state < 64 KiB
MAC < 150M
```

未达到 literal Gate2 时不得用 D512-relative delta 宣称 Gate2 通过。

### 7.3 多 seed

只有 seed-42 七折候选达到 Gate2，或处于预先标记的边界区 `mean_delta >= -0.015`，才运行 seeds 43、44 的完整七折。

最终候选需报告 21 个 fold-seed 点，不能只报告最佳 seed。

## 8. Phase R4：B5/B6 最小硬件探针

R4 在 Phase R1 选出 loss 后启动；它可以与 R2 的分析阶段衔接，但不得抢先使用未经选择的默认 loss。

### 8.1 Cubic accuracy probes

```text
B5: R=4, D=64, power-of-two alpha
B6: R=4, K=5, D=64
fold=0, seed=42
preprocessing=cubic T=100
loss=R1 winner
```

该阶段只证明：

- 固定 teacher 输入分布下 EMA/FIR 的 task accuracy；
- 结构状态可以设计为无 trial buffer；
- 理论 MAC/state profile。

该阶段不证明：

- host cubic interpolation 已经取消；
- PyTorch training runtime 只保留 bounded streaming state；
- raw variable-length 输入精度；
- measured peak SRAM。

### 8.2 Probe 判据

| Fold-0 result | 决策 |
|---|---|
| `delta >= -0.03` | Aggressive green；进入更多 folds |
| `-0.05 <= delta < -0.03` 且与最佳 B3 相差不超过 0.01 | 保留 minimum-hardware research point；先跑 folds 1、2 |
| `delta < -0.05` 或比最佳 B3 差超过 0.02 | 停止该分支 |

### 8.3 Formal Gate3 的附加要求

若要正式声称 Gate3 完成，还必须：

1. 补 B4 statistics 完整结果；
2. B4/B5/B6 各至少一个完整结果；
3. 对 promotion 候选运行 raw-bin/no-interpolation 配置；
4. 将 cubic 与 raw-bin 结果分开，量化 preprocessing delta；
5. 验证 padded bins 不进入统计和 MAC；
6. 用真实 `start_trial/push_sample/end_trial` 路径测 bounded state；
7. 不把 `forward_batch` 的 GPU 内存峰值当作硬件 streaming SRAM。

## 9. Phase R5：候选锁定与最终 held-out

只有完成 held-in LOSO 结构选择和多 seed 确认后，才锁定：

```text
variant
width/R/K
loss mode and lambdas
preprocessing
epoch policy
quantization policy（此阶段仍为 FP32 时固定 FP32）
```

然后：

1. 使用全部七个 held-in sessions 重新训练锁定结构；
2. 不使用 benchmark held-out 做 checkpoint selection；
3. epoch policy 在 LOSO 结果基础上预先确定，推荐使用 LOSO best-epoch 的中位数或固定 40 epoch；
4. 生成唯一 locked checkpoint 与 manifest；
5. 六个 benchmark held-out sessions 只评测一次；
6. 评测后不再根据 held-out 结果回改结构/loss/threshold。

不能直接拿某一个 LOSO fold checkpoint 做最终 benchmark held-out 结论，因为该 checkpoint 只使用六个 held-in sessions 训练 student encoder。

## 10. 结果表与记录要求

建议新增聚合表：

```text
outputs/streaming_calibration/gate2_revised_matrix.csv
```

至少包含：

```text
run_id
comparison_role              # protocol_control / loss_ablation / gate2_candidate / hardware_probe
variant
width_or_RK
loss_mode
lambda_y
lambda_E
seed
fold_id
train_sessions
validation_session
teacher_seen_validation_session
preprocessing
R2
delta_fixed_B0
delta_vs_D512_LOSO
identity_mse
prediction_distill_mse
best_epoch
parameter_count
MAC_per_session
peak_live_state_bytes
trial_buffer_bytes
baseline_sha256
teacher_sha256
checkpoint_sha256
source_manifest_sha256
decision_status              # green / amber / red / diagnostic_only
```

每个 run 完成后立即写明：

- 是否正常结束；
- 是否命中预定义 promotion/stop 条件；
- 下一 run 与本 run 的唯一变量；
- 是否触碰 benchmark held-out；
- 任何人工排除或重跑原因。

## 11. 预算控制

本计划按信息增益而不是网格规模分配训练：

### Immediate Round（固定三次新训练）

```text
1 x B2-D512 protocol control
1 x B3-D64 task_only
1 x B3-D64 task_plus_y
```

### Conditional Round

```text
最多 1 x B3-D128 winning-loss fold0
可选 1 x B2-D128 winning-loss fold0
最多 2 x B5/B6 cubic fold0 probes
```

在 Immediate Round 结果出来前，不预授权完整 7-fold x 3-seed 网格。

## 12. 决策树

```text
R1: D512 control + B3 loss ablation
 |
 |-- D512 < -0.03 AND all losses < -0.03
 |     -> STOP architecture sweep
 |     -> diagnose student protocol / teacher target / baseline exposure
 |
 `-- D512 >= -0.03 OR a B3 loss materially improves
       -> choose loss
       -> R2: B3-D128 fold0
             |
             |-- delta >= -0.01 -> seven-fold Gate2
             |-- -0.02..-0.01  -> folds 1,2 triage
             `-- < -0.02        -> stop high-fidelity branch

After loss selection:
  -> B5-R4 and B6-R4-K5 cubic probes
       |
       |-- delta >= -0.03 or near B3 at much lower cost
       |     -> limited folds -> formal Gate3/raw-bin validation
       `-- otherwise stop branch

Only after held-in selection lock:
  -> all-held-in refit
  -> one-time benchmark held-out evaluation
  -> quantization/progressive stop
```

## 13. 本修订相对原指南的变化

原指南的 Gate 数值和最终 held-out 规则不变。本修订只增加实验可解释性和预算闸门：

1. 增加 D512 student-protocol control，分离训练协议上限与压缩损失。
2. 将 loss ablation 提前到宽度和 B5/B6 搜索之前。
3. 将 fold-0 定义为诊断/筛选，不再从单 fold 宣称 Gate2。
4. 增加 Green/Amber/Red promotion 条件，避免无条件展开七折多 seed。
5. B5/B6 先做 cubic accuracy probe，再做 raw-bin 真流式验证。
6. 保留 fixed-B0 official delta，同时增加 D512-relative diagnostic delta。
7. 明确最终 held-out 前必须用全部 held-in sessions 重新训练锁定候选。

## 14. 下一步执行许可边界

本计划本身不启动训练。进入下一轮前需先：

```text
[ ] 用户确认 Immediate Round 三个 run
[ ] 创建并审查 b2_d512_protocol_control preset
[ ] Hydra compose 三个配置
[ ] baseline verifier 通过
[ ] unit tests 通过
[ ] GPU/进程状态正常
[ ] 记录启动时间、完整命令和预期 output run_id
```

完成后按 R1 顺序逐个运行、逐个验收，不采用无人审核的全网格长跑。
