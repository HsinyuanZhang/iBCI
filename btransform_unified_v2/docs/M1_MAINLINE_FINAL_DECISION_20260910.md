# M1 论文主线最终决定（2026-09-10）

论文 M1 主线固定为 `muscle_response16_svd4/global_rms` 的 RIFT R100 FULL。固定 original FULL recipe 已形成 EvalAI official submission `582205`；其 `test_split_m1` Held Out R² 为 `0.6259910151098528`。权威结果见 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)，正式 recipe、公开选择面与 official 结果的边界见 [M1 official retrain record](M1_MUSCLE_OFFICIAL_RETRAIN_20260909.md)。

用户已授权以此固定 FULL 主线完成既定三任务 recency-versus-flat 最终消融。M1 配对基准为 `results/m1_muscle_r100_v1/formal_s42_gpu1`，flat run 为 `results/recency_flat_ablation_v1/formal_m1_muscle_full_flat_s42`，二者固定 seed 42、24 × 6665 updates、HO3 完整 24-epoch EMA 曲线和各自 earliest-maximum 选择。两者使用同一 [carrier_pack.npz](../results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz)，并在 run metadata 中绑定 carrier receipt、fit 和 original FULL reference。

`582205` 是固定 FULL 主线的 official provenance。该 official 分数不替代这次 recency-versus-flat 的开发面效应结果，也不把单 seed 消融写成 official 或 multi-seed 结论。三任务最终协议见 [protocol_muscle_full_v2.json](../results/recency_flat_ablation_v1/protocol_muscle_full_v2.json)。

不新增 M1 carrier、模型结构或 decoder seed 搜索。已完成的 original-muscle multi-seed 与 mean-rate4 fourth-column candidate 保留为历史、描述性证据；它们不替换这条论文主线。历史结果、收据和解释限制保留在 [multi-seed carrier plan](M1_MUSCLE_MULTISEED_CARRIER_PLAN_20260909.md)。
