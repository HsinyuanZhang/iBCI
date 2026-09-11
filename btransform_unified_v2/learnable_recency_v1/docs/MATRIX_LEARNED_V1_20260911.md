# 统一实验矩阵（learnable_recency_v1 主线，2026-09-11 起）

## 设计（用户 2026-09-11 裁定）

每数据集四格；recency 轴 = **learned_slope@default** vs **无 recency（flat）**；身份轴 = **full / act / static**。**fixed-recency 臂取消**（历史参照数字沿用）。static 为用户新建的测试（非 norm；norm 已废）。flat 训练由用户后续安排。

载体口径（每数据集最后落盘代）：
- M1 = `muscle_response16_svd4/global_rms`（carrier_official4 包；官方锚 582205=0.6260）。旧 rSyn3-refit-v1 线（0.7047/0.7075/0.6949）作废留档。
- M2 = MOVE-T4（dual-track 冻结缓存 T.npy，官方线同源，缓存哈希互锁）。
- H1 = signed_state14 27-tag banks（2026-09-09 封印，官方 582196 同源）。

## 矩阵（每格两数：官方提交（若有）/ 本地选择）

### M2（本地面 = EXT6 earliest-max equal_session_mean，EMA；proj_add P16）

| 格 | 官方 | 本地 |
|---|---|---|
| learned + full | — | **0.3634**（s42 e10）；s43 0.3668（e11） |
| learned + act | — | **0.1194**（clean/label-free e10；E0 side 全零） |
| learned + static | — | —（用户搭建中） |
| no-recency + full (flat) | — | —（用户安排） |

旁注（不填格）：clean-ACT 阶梯中间点 ACT-token（E0 side 含 MOVE-T4）= 0.3595 → **E0 内标签 side 贡献 +0.240**；NORM = −0.0013（已废）；concat 接口 learned = 0.3553、concat fixed-recency 参照 = 0.3901；官方 concat 线 581973=0.3901 / 582217(ACT,旧配方)=0.0930。

### M1（本地面 = HO3 earliest-max equal_session_mean，EMA；proj_add，muscle_response16_svd4 载体）

| 格 | 官方 | 本地 |
|---|---|---|
| learned + full | — | —（muscle 线 learned P16 待跑） |
| learned + act | — | —（agent 在途） |
| learned + static | — | —（用户搭建中） |
| no-recency + full (flat) | — | —（用户安排） |

旁注：官方锚 582205 = 0.6260（concat + fixed-recency + muscle 载体，不填新格）；旧 rSyn3 线作废（learned P16 0.7075 / P32 0.6949 / 参照 0.7047）。

### H1（本地面 = HO-M3 grouped-seven earliest-max，EMA；proj_add P16）

| 格 | 官方 | 本地 |
|---|---|---|
| learned + full | — | **0.4624**（s42 e16）；s43 0.4516（e15） |
| learned + act | — | **0.2659**（e14，label-free） |
| learned + static | — | —（用户搭建中） |
| no-recency + full (flat) | — | **0.4686**（flat e15，signed_state 家族既有） |

旁注：官方锚 582196（signed_state fixed-recency，本地对应 0.4677）；官方 ACT 582218=0.3446（旧配方）；H1 P32 learned = 0.4797（宽度扫描旁支）。

## 维护规则

- 每臂落袋即更新对应格；官方提交由用户执行后回填（填提交号+分数）。
- 本地数字必须与 run 目录 + selection receipt 一一对应，判定文字（非劣/差）写在旁注。
- 载体/E0/阶梯任何变更必须同步更新"载体口径"节。

## 补充（2026-09-11）

- **static 已建成**（STATIC_SOURCE_ONLY.md）：source-only 共享 `static_identity[N,16]` 通道表（无 E0/无 carrier/无目标校准），脚本 `m1/m2/h1_static_train.py`，checkpoint 预注册为**末轮 EMA**（不做目标面选轮）。三格（learned+static）正式训练进行中（s42 先行）。
- **多 seed 政策（用户 2026-09-11 二次细化）**：**仅 learned+full 允许多 seed 训练并本地选最优提交官方**；其余格（learned+act、learned+static、no-recency+full）一律 seed42 单 seed。矩阵本地列按 seed 分记。注意口径：static 用预注册末轮 EMA、FULL/ACT 用目标面 best-epoch——跨格比较时在旁注标明选轮规则差异。
