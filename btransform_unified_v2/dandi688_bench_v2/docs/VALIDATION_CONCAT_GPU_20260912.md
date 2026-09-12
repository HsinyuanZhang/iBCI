# DANDI688 CONCAT validation and completion record — 2026-09-12

CONCAT 正式 campaign 已完成。12 个 neural task 都完成 `24 × 3,165 = 75,960` updates、`B32`；随后完成 static controls、development seal、一次 six-date final score 和 report。final 结果不是中途 development 分数：所有 19 cell 均为 `SCORED`，`final_sessions_opened=6`。已完成产物只读查阅，不应再次运行 final。

| 证据 | 核验结果 | 边界 / 链接 |
| --- | --- | --- |
| [CONCAT CPU smoke receipt](../results/smoke_concat_2015_m33_v1/receipt.json) | 2 Full pretrains、6 neural arms，各 2 updates | `SMOKE`，只作工程合同证据 |
| [aggregate validation](../results/concat_line_v1/VALIDATION.json) | 历史 CPU run：56 passed、1 HDMF warning、7.15 s | source-only smoke，不替代正式 completion |
| [SUA GPU profile](../results/gpu_profile_concat_20260912_sua/receipt.json) / [PMUA profile](../results/gpu_profile_concat_20260912_pmua/receipt.json) | 四臂各 8 B32 steps，loss/gradient finite，约 36–38 updates/s | feasibility only，不能选 checkpoint |
| [pre-final verification](../results/formal_campaign_concat_2015_m33_allseeds_20260912/PRE_FINAL_TRAINING_VERIFICATION.json) | `TWELVE_FORMAL_TASKS_VERIFIED`；12×24 checkpoint 共 288 个 SHA 已验 | final 前不访问 final session |
| [static completion verification](../results/formal_campaign_concat_2015_m33_allseeds_20260912/STATIC_CONTROLS_COMPLETION_VERIFICATION.json) | 两个 Raw-PMUA static control 已完成并进入 seal | 保留原定 cell，不减少矩阵 |
| [selection seal](../results/formal_campaign_concat_2015_m33_allseeds_20260912/selection_seal_concat_19/selection_seal.json) | `FROZEN_FORMAL_SELECTION`；15 main + 4 Full supplements 和两个 encoder 已绑定 | development selections 已冻结 |
| [final receipt](../results/formal_campaign_concat_2015_m33_allseeds_20260912/final_concat_19/final_score_receipt.json) | `FINAL_SCORED`；6 final dates、19/19 `SCORED` | 唯一完成的 final access |
| [final integrity verification](../results/formal_campaign_concat_2015_m33_allseeds_20260912/FINAL_RESULTS_INTEGRITY_VERIFICATION.json) | `PASS` | 19 NPZ、114 per-session arrays、finite 值、digest、seal 和 70 个执行源码 hash 已核验 |
| [final report](../results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/report.md) | main table、三 seed 汇总、配对结果已生成 | report artifact SHA 也由 integrity verification 记录 |

正式协议固定为 2015-only 的 18 source / 6 development / 6 final session、M33 support 与 MOVE-T4 cosine OLS。Full encoder 是 fresh 的无 FiLM concat：`pre_pool → trial mean → concat(representation-matched MOVE-T4) → post_pool`；它与 decoder 的 M2-like `P16/proj_add` frontend 是不同部件。ACT 排除 T，Raw-set 不使用 encoder。训练为 CUDA bf16 autocast，development 和 final scoring 为 CUDA fp32。旧 Full-FiLM 权重不具复用资格，见 [historical termination receipt](../results/formal_campaign_2015_m33_allseeds/TERMINATED_BY_USER.json)。

Full main cell 使用 seed42；Full seed43/44 为预注册 supplement。三个 decoder seed 都条件于同表示的固定 seed42 encoder，故报告的 sample SD 只反映 decoder-seed variability。三 seed final mean ± sample SD：SUA `0.777458595870 ± 0.006124236429`，PMUA `0.773771616853 ± 0.011641903378`；matched SUA−PMUA 为 `0.003686979017 ± 0.017149722311`。差值在三个 seed 间变号，seed42 为 PMUA 更高，不能报告 SUA 稳健胜出。详表见 [full seed summary](../results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/full_seed_summary.csv) 与 [paired differences](../results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/main_sua_pmua_paired_differences.csv)。

Full−ACT 混合了 source encoder 预训练监督量和冻结策略差异；WF-FSS 使用额外 dense velocity labels，而 Full 使用 trial-angle 信息。这些比较保留在主表中，但不构成同标签预算或纯因果对照。

已完成 campaign 的状态、task 完成次序和 final receipt binding 见 [completion worker](../results/formal_campaign_concat_2015_m33_allseeds_20260912/completion_worker.json)。新复现实验必须选用新的 fresh root，并遵守 [campaign plan](CAMPAIGN_PLAN_20260912_CONCAT.json) 的不可变设计字段；已完成 root 只读查 report/receipt。
