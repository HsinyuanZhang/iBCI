# M2 训练方法大对照：峰值 LR × 调度

日期：2026-09-05  
状态：**AUTHORIZED_AFTER_INTERIM_S0_S1_COMPLETE**  
范围：同一 3,543,010 小 Transformer、同一 seed42 初始化、同一 24-epoch shuffled manifest、同一 EMA 0.9995。只改训练制度。不改架构，不开 Mamba / seed43 / 48 epoch。

父对照（只读，不重训）：`tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000/`  
新根：`tfpd_exploration/results/m2_b_small_stability_v1/20260905_131500/`

## 采纳与不采纳

来自更新后的 `ANALYSIS_B_TRAJECTORY_INSTABILITY.md`（`5e6fc8ac…`），只做我认为成立的训练杠杆：

| 采纳 | 不纳入本对照 |
|---|---|
| 全程 cosine 对两段 LR（已有 S1 vs S0） | 训到 train-MSE 平台再停（无界；不自动 48 epoch） |
| EMA 0.9995 作为权重视图（已有） | 把 RAW 峰值或 visible-ext4 峰值改写成主候选 |
| **峰值 LR 3e-4 对 1e-4**，整表按 10× 缩放（min = 0.1 × max） | 跨评价面把 minival 0.309 去减 REF 0.3582 |
| 固定 24 epoch、last-4/last-8、source-pick 与 visible-pick 分列 | SWA、蒸馏、降宽、改 warmup 长度 |

## 2×2

|  | 旧两段（warmup + e2–12 恒定 + e13–24 cosine 尾） | 同 warmup 后持续 cosine 至 e24 |
|---|---|---|
| 峰值 3e-4，min 3e-5 | **S0-SMALL-LEGACY**（已完成，不重训） | **S1-SMALL-COS**（已完成，不重训） |
| 峰值 1e-4，min 1e-5 | **N0-SMALL-LEGACY-1E4**（新训） | **N1-SMALL-COS-1E4**（新训） |

本轮主候选预先指定：**N1-SMALL-COS-1E4 / EMA**。不得评分后改选。S1/EMA 仍是 3e-4 对照的主候选，路由不改写。

N0/N1 与 S0/S1 共用初始化 SHA、manifest digest `a95255fa…`、decoder 参数量、dropout 域公式。唯一差别是 `lr_max/lr_min` 与调度族。
