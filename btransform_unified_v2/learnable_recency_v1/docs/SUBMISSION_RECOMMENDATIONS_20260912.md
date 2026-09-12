# 推荐提交配置（recency vs flat 官方对照，2026-09-12 修订二）

**范围**：待提交 5 位。H1 learned+full **不在表内**——已提交（`582241`，official HO R² = 0.4588），沿用即可；H1 flat 从未提交，列入。

## 待提交表（checkpoint 相对 `btransform_unified_v2/learnable_recency_v1/results/`）

| # | 提交位 | run 目录 | **epoch** | 本地 mean | 本地 worst (session) | 配置要点 |
|---|---|---|---|---|---|---|
| 1 | M2 learned+full | `m2_concat_full_stage2_s42` | **8** | 0.3832 | 0.2495（11-24） | Concat 两阶段：E0=冻结 trunk（s42 stage-1 选轮 e10）+MOVE-T4 拼接过 MLP；P16/D4/learned_slope@default；备选 s43 e7=0.3681 |
| 2 | M2 flat | `m2_concat_flat_stage2_s42` | **7** | 0.3622 | 0.2404（11-24） | 同冻结 trunk，八 None 半衰期=零 slope |
| 3 | M1 learned+full | `m1_concat_full_stage2_s42` | **2** | 0.7124 | 0.6287（20121024） | 同协议（M1 trunk 冻结自 stage-1 s42 选轮 e9）+肌肉载体；备选 s43 e2=0.7075 |
| 4 | M1 flat | `m1_concat_flat_stage2_s42` | **2** | 0.7195 | 0.6457（20121024） | 同冻结 trunk + 零 slope |
| 5 | H1 flat | `h1_flat_p16_s42` | **15** | 0.4658 | 0.2241（S7） | 原生 C2 材料化器 E0 + 零 slope；learned 对照=已提交的 582241 |

## 对照锚（不占提交位）

- **H1 learned+full = 582241（已提交）**：official HO R² **0.4588**（本地 0.4624/e16）。H1 的 recency-vs-flat 官方差 = 本表 #5 的官方分 − 0.4588。

## 共同配置（5 位一致）

proj_add **P16** / D4 / width 256 / 8 heads / local attention；carrier 走 token 通道（M2=MOVE-T4、M1=muscle_response16_svd4、H1=m3 H-C）；EMA checkpoint；M2/M1 位为 Concat 两阶段协议（stage-1 联合预训练 s42 → 选轮冻结 trunk → stage-2 重训 decoder），flat 位与 learned 位共享同一冻结 trunk、仅 recency 算子不同（learned_slope 24 标量 vs 八 None 零 slope）。选轮=各自 development 面 earliest-max EMA，未触碰 official test。

## 官方判定口径

- 每任务 recency−flat 官方差：M2 = #1−#2；M1 = #3−#4；H1 = #5−0.4588。本地预期差：M2 **+0.021**（learned 优）、M1 **−0.007**（flat 优）、H1 **−0.003**（平）。
- worst-case 本地观察：M2 learned +0.009、M1 −0.017、H1 −0.032（flat 的 worst 反而低）——官方侧建议同报 worst session。

## 打包注意

- M2/M1 的 4 个 concat 位使用新 checkpoint schema（`b3s_full_epoch_checkpoint_v1`、`trunk_side=concat`）：打包按 run_meta 用 `B3SFullRiftDecoder`/`ConcatSideTrunk` 重建，**不能**用旧 frozen-encoder 加载器；carrier 按 run_meta sha 重绑源文件（M2 的 EXT6 载体在冷归档，`activity_data.resolve_m2_ext6_root()` 已处理 `.RELOCATED`）。
- H1 flat 位沿用 `evalai_h1_rift_*` 打包路径（冻结 C2 材料化器 + EMA checkpoint），与 582241 同构仅零 slope。
