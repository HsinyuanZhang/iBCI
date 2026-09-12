# DANDI000688 v2：2015-only SUA／PMUA CONCAT 实验

正式 CONCAT base19 campaign 已完成：12 个 neural task 均完成固定预算，development selection 已封存，并在一次 final access 中评分 19 cell。[completion verification](../dandi688_bench_v2/results/formal_campaign_concat_2015_m33_allseeds_20260912/FINAL_CAMPAIGN_COMPLETION_VERIFICATION.json) 为 `PASS`；可审计结果见 [final receipt](../dandi688_bench_v2/results/formal_campaign_concat_2015_m33_allseeds_20260912/final_concat_19/final_score_receipt.json) 和 [report](../dandi688_bench_v2/results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/report.md)。当前进行的是独立的 Flat6 seed42 follow-up，不重开或改写 base19 final。

## 数据、模型与边界

数据限定 DANDI000688 `sub-C/CO/2015` 的 30 个 sessions：18 source、6 development、6 final。2013、2016 不参与本轮。source/dev cache 已准备为 48 个 paired NPZ；[prepared receipt](../dandi688_bench_v2/results/prepared_2015_m33_v2/prepared_receipt.json) 记录 `final_sessions_opened=0`。

每个 query 用前 33 个合法 rewarded trials 作 M33 support，Q50 从 rewarded index 50 开始。MOVE-T4 用 M33 的 `[GO+0.1 s, GO+0.6 s)` spike counts/25 建立 `[intercept, cos, sin]` OLS，并产出 `[a, c, hypot(a,c), baseline]`；仅 finite angle、rank 3 的 rows 可进入拟合。所有结果按每日期 physical-velocity variance-weighted R²，再对 6 日期等权平均。

SUA 是 sorted-unit counts；PMUA 是同一 spikes 按真实 hardware electrode 的 pooled sorted-unit counts。它们共享行为和 query hash，但各自重算 T、E0 与 source statistics。N_pad=100 只用 mask padding，不截断。当前 Full encoder 合同为 [CONCAT](../dandi688_bench_v2/docs/CONCAT_ENCODER_CONTRACT.md)：`pre_pool → M33 mean → concat(T4) → post_pool`；它不含 FiLM，也不能与 decoder `P16/proj_add` frontend 混同。

## 固定训练与选择

每项 neural task 固定 24 segments × 3,165 updates、batch32；AdamW、bf16 CUDA autocast、EMA=.9995、session-balanced sampler 和 M2-like R50/P16/proj_add recipe。SUA、PMUA 各有一个 seed42 source-only Full encoder pretrain，并固定使用最后一个 EMA。每个 Full decoder 从 fresh initialization 训练；同一表示的 Full seed42/43/44 共享该表示 encoder。ACT、Raw-set 都仅训练 seed42 decoder。

decoder 对完整 dev-6 的 24 个 EMA checkpoint 采用 earliest maximum；encoder 不使用 dev 选点。Full 的三个 decoder seeds 用 sample SD 汇报，且按相同 seed 汇报 SUA−PMUA。任何结果不得以 best seed 取代 seed42 主表。

## 19 格输出

15 格主表包括：6 个 seed42 neural、7 个 CPU baseline 和 2 个 static control。四个 Full supplements 为 `full_{sua,pmua}_s43` 与 `full_{sua,pmua}_s44`；它们和主表在同一次 final access 评分，但不会改写主表。可机读 roster、任务与路径见 [campaign plan](../dandi688_bench_v2/docs/CAMPAIGN_PLAN_20260912_CONCAT.json)。

baseline 与 static-control 的 development candidates 已有 CPU 证据，但每个 static control 必须绑定本轮完成后的 `raw_set_pmua` checkpoint。WF-FSS 使用 dense velocity，而 Full 使用 trial angle；两者的标签信息量不同。FA 只有在 rank/收敛门失败时才可明确标为 `UNAVAILABLE`，不能删日期凑平均。

## 运行与验证状态

当前 Flat follow-up 仅包含 seed42 的 6 cell：SUA/PMUA 各有 Full、Activity 和 Raw-set，使用 fixed flat（全零 recency slope）。其当前 root 为 [formal_flat_followup_2015_m33_seed42_20260912](../dandi688_bench_v2/results/formal_flat_followup_2015_m33_seed42_20260912/)。新 controller 已启动，当前 [watcher](../dandi688_bench_v2/results/formal_flat_followup_2015_m33_seed42_20260912/completion_watcher.json) 在等待两侧 Full42 worker；随后才会执行 Activity42 和 Raw42。[start verification](../dandi688_bench_v2/results/formal_flat_followup_2015_m33_seed42_20260912/START_VERIFICATION.json) 为 `PASS_SEED42_ADOPTION_AND_CONTROLLER_START`，并确认 `flat_final_sessions_opened=0`。因此尚无 Flat final 分数或科学结论。

[25-row comparison progress](../dandi688_bench_v2/results/formal_flat_followup_2015_m33_seed42_20260912/COMPARISON_PROGRESS.md) 将已评分的 base19 19 行与 6 个 `PENDING` Flat cell 并列；它是进度表，不是完整的 Flat 结果。已完成 base19 的 Full learned seed43/44 仅保留为 supplemental seed reference，不能构造 Flat 单 seed42 的三-seed mean、SD 或配对不确定性结论。fresh 目录不复用旧 FiLM 产物；旧任务的终止记录只作为历史排除证据，见 [TERMINATED_BY_USER.json](../dandi688_bench_v2/results/formal_campaign_2015_m33_allseeds/TERMINATED_BY_USER.json)。

GPU feasibility profiles 已以 B32 覆盖四个 arm：SUA [receipt](../dandi688_bench_v2/results/gpu_profile_concat_20260912_sua/receipt.json) 与 PMUA [receipt](../dandi688_bench_v2/results/gpu_profile_concat_20260912_pmua/receipt.json) 均为 8 steps/arm、finite loss and gradients、约 36–38 updates/s。CPU smoke 的 56-test 和 2-step 证据只验证合同且永不具备 formal 资格，见 [validation record](../dandi688_bench_v2/docs/VALIDATION_CONCAT_GPU_20260912.md)。
