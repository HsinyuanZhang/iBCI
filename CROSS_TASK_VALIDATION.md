# Cross-Task Validation: SPINT B3 Encoder on M1/M2/H1

> **文档状态：MUA 通用性支线。** M1/M2 结果可作为 SUA/MUA 主线的 MUA 参考；H1 尚未完成。当前主线入口见 [`sua_exploration/README.md`](sua_exploration/README.md)。

## 目标

验证 B3-D64 encoder 压缩方案在不同 FALCON 任务上的泛化性。论文 Table 1 报告了三个任务的 held-out R²，我们在 M2 上已验证 B3 接近论文上限。现在扩展到 M1 和 H1，检查 B3 是否 task-agnostic。

## 论文 Reference (Table 1, held-out R²)

| 方法 | M1 | M2 | H1 |
|------|---:|---:|---:|
| WF Oracle | 0.53 | 0.26 | 0.21 |
| NDT2 Oracle | 0.78 | 0.58 | 0.63 |
| WF Zero-shot | 0.34 | 0.06 | 0.16 |
| CycleGAN+WF (FSU) | 0.43 | 0.22 | 0.12 |
| NoMAD+WF (FSU) | 0.49 | 0.20 | 0.13 |
| **SPINT (GF-FSU)** | **0.66** | **0.26** | **0.29** |

## 任务参数对比

| 参数 | M1 | M2 | H1 |
|------|---:|---:|---:|
| 神经元数 N | 64 | 96 | 176 |
| 窗口 W | 100 | 50 | 700 |
| 协变量 C | 16 | 2 | 7 |
| model_dim H | 1024 | 512 | 1024 |
| calib trial 长度 T | 1024 | 100 | 1024 |
| 训练窗口数 | 213K | 116K | 132K |
| Held-in sessions | 4 | 7 | 13 |
| Held-out sessions | 3 | 6 | 14 |
| behavior_scaling_factor | 1.0 | 5.0 (1/0.2) | 20.0 (1/0.05) |
| predict_scaled_behavior | false | true | true |

## 当前状态 (2026-07-21)

### M2 (已完成)

B3-D64 LOSO+heldout（fold 0, seed 42, 20 epochs）：
- **heldin R² = 0.620**
- **heldout R² = 0.236 ± 0.102**
- 论文 SPINT M2 = 0.26 ± 0.13
- **Gap: -0.024（在方差范围内）**

### M1 (已完成)

- **M1 teacher**: 20 epochs, best checkpoint at epoch 14
  - Run dir: `SPINT-main/logs/train/runs/2026-07-21-19-11-01/`
- **B3-M1 student** (LOSO fold 0, seed 42, 20 epochs):
  - heldin R² = 0.753 (ses-20120924)
  - **heldout R² = 0.630** (mean of 3 sessions)
    - ses-20121004: 0.730
    - ses-20121017: 0.589
    - ses-20121024: 0.570
  - 论文 SPINT M1 = 0.66 ± 0.07
  - **Gap: -0.03（在方差范围内）**
  - Student/Teacher ratio: ~0.95 (teacher best ~0.66)

### H1 (未开始)

- 数据已下载 (98 MB, 40 NWB files)
- 需先训 teacher（model_dim=1024, W=700, 预计显存占用大）
- H1 的 `use_calib_active_segments=true`（与 M1/M2 不同）
- 等 M1 student 完成后再启动

## 已创建的 Config 文件

| 文件 | 用途 |
|------|------|
| `configs/data/falcon_m1.yaml` | M1 数据加载（W=100, T=1024, calib=10 trials）|
| `configs/data/falcon_h1.yaml` | H1 数据加载（W=700, T=1024, calib=2 active segments）|
| `configs/model/_streaming_base_m1.yaml` | M1 streaming base（scaling_factor=1.0, predict_scaled=false）|
| `configs/model/streaming_b3_m1.yaml` | M1 B3 encoder |
| `configs/experiment/b3_m1.yaml` | M1 B3 实验预设 |
| `configs/paths/default.yaml` | 加入 `m1_teacher_ckpt_path` |
| `scripts/wait_and_run_m1_student.sh` | 自动等 teacher → 跑 student |

## 关键差异：M1 vs M2 config

B3 student 在 M1 上需要适配：
1. **T=1024（vs M2 的 100）**: `pre_pool: Linear(1024→64)` 参数量从 6464 增到 65600 — 注意这会让 B3-M1 的参数量远大于 B3-M2
2. **W=100（vs M2 的 50）**: `post_pool` 最后一层输出 100 维而非 50 维
3. **C=16（vs M2 的 2）**: decoder 输出 16 个协变量而非 2 个
4. **behavior_scaling_factor=1.0**: 不做 scaling（M2 是 5.0）
5. **predict_scaled_behavior=false**: 直接预测原始行为

## 跨任务汇总

| Task | B3 Student Heldout | Paper SPINT | Gap | Status |
|------|-------------------|-------------|-----|--------|
| M2 | 0.236 ± 0.102 | 0.26 ± 0.13 | -0.024 | ✓ 完成 |
| M1 | 0.630 | 0.66 ± 0.07 | -0.03 | ✓ 完成 |
| H1 | — | 0.29 ± 0.15 | — | 待开始 |

**结论**: B3-D64 encoder 在 M1 和 M2 上均接近论文 full SPINT 性能（gap < 0.03），验证了压缩方案的 task-agnostic 有效性。

## 后续步骤

1. ~~M1 teacher 完成 → auto-runner 跑 B3 student~~ ✓
2. ~~M1 student 完成~~ ✓ heldout=0.630
3. 启动 H1 teacher（model_dim=1024, W=700, 显存需求大）
4. H1 teacher 完成 → 跑 B3-H1 student
5. 三任务结果汇总 → 最终跨任务对比表
