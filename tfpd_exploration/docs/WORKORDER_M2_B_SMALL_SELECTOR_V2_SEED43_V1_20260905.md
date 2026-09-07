# 补充轮：预注册选择律 v2 + S1 seed43

日期：2026-09-05  
状态：**AUTHORIZED_AFTER_N0_N1_EXT4_SEALED__DO_NOT_LAUNCH_WHILE_131500_UNSCORED**  
父审核：`tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000/ANALYSIS_SMALL_STABILITY_REVIEW_NOTE.md`  
SHA：`c49bf3d75a33963a0c5ce28eb463ddd8fb82147d550839c8d84faab52f63526e`

## 裁决

- **补一轮：要。** 内容是选择律 v2 + 一条 seed43 确认，不是再加 LR/宽度/48 epoch cell。
- **现在不启动。** 必须先封存 `20260905_131500` 的 N0/N1 ext-4 与 2×2 表。N0/N1 仍按旧 source-minival argmax 评分，不回溯改 S0/S1 路由。
- 采纳注记：S1/EMA 末段 ext-4 0.45 是同面真读数；source-pick 停在 e2 是因为 held-in 10-20 把 equal-session 均值劫持，不是选择器实现坏了，也不是单靠 “early-EMA artifact” 能说完。
- 评分后把 last-k/endpoint 升成 S0/S1 主声明 = 违规。要主张 0.45，只能靠本工单预注册后的新跑。

## 预注册（写在 seed43 任何 step 之前）

主统计量：**S1-SMALL-COS / EMA / endpoint24**（cosine 终点，无数据依赖选点）。  
同时固定报告 last-4 / last-8 EMA、source-minival argmax（只作诊断）、minival **per-session median**。  
不得在 seed43 出分后改主统计量。Visible-ext4 argmax 仍只记 development evidence。

确认对象只有 **S1 配方**（3e-4、全程 cosine、EMA 0.9995、同一小模型、同一 24ep manifest 公式、seed **43**）。  
N1 seed43 仅当 N0/N1 ext-4 的 N1/EMA endpoint24 或 last-8 不低于 S1/EMA 同统计量时再另写一句话授权；默认不跑。

## 明确不做

再开 N2、降到 3e-5、48 epoch、SWA、Mamba、改 S0/S1 已封路由、用 minival 0.309 减 REF 0.3582。
