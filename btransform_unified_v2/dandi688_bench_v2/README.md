# DANDI688 v2 — CONCAT / M2-like 19-cell campaign

CONCAT base 19-cell 正式 campaign 已完成：12/12 个 neural training task 均完成 24 段、75,960 updates、batch 32；development selection 已封存；一次 final access 已完成并评分全部 19 cell。288 个训练 checkpoint 由 [pre-final training verification](results/formal_campaign_concat_2015_m33_allseeds_20260912/PRE_FINAL_TRAINING_VERIFICATION.json) 核验；独立 [final integrity verification](results/formal_campaign_concat_2015_m33_allseeds_20260912/FINAL_RESULTS_INTEGRITY_VERIFICATION.json) 为 `PASS`，核验 19 个 final NPZ、114 个按日期 prediction digest、全部有限数组和 seal/source-hash binding。综合 [base19 completion verification](results/formal_campaign_concat_2015_m33_allseeds_20260912/FINAL_CAMPAIGN_COMPLETION_VERIFICATION.json) 也为 `PASS`。

用户随后授权加入独立 Flat 系列对比，当前范围固定为仅 seed42 的 6 cell：SUA/PMUA 各有 Full、Activity 和 Raw-set。它复用已完成的同表示 base seed42 encoder，并使用 fixed flat（全零 recency slope）；base19 的结果、预算、seed 和 final receipt 不会被改写。新范围的 [frozen Flat6 plan](results/formal_flat_followup_2015_m33_seed42_20260912/plan.json) 已封存，[adoption plan](results/formal_flat_followup_2015_m33_seed42_20260912/ADOPTION_PLAN.json) 将不中断的两个 Full seed42 run 绑定到此范围。新 controller 已于 `2026-09-12T08:39:51Z` 启动；[start verification](results/formal_flat_followup_2015_m33_seed42_20260912/START_VERIFICATION.json) 为 `PASS_SEED42_ADOPTION_AND_CONTROLLER_START`，确认 78 个执行源码和 adoption binding 未变、90 秒 watcher heartbeat、无自动重试，且尚未开启 Flat final access。当前 watcher 正在等待两队 Full42 worker 完成，之后才依序运行 Activity42 和 Raw42；最终 Flat6 评分及完整 25-row 比较仍待完成。final gate/report 验收的 [14 passed、0 xfail](results/formal_flat_followup_2015_m33_seed42_20260912/FINAL_PIPELINE_TEST_RESULTS.json) 与此前的 14 个 training/launcher tests 是独立计数。[25-row comparison progress](results/formal_flat_followup_2015_m33_seed42_20260912/COMPARISON_PROGRESS.md) 单列已评分的 base19 19 行和 `PENDING` 的 Flat6 6 行；它是进度表，不是完整的 Flat 结果。此前 10-task／29-row Flat plan 已被本次范围变更 supersede，原有 Full learned seed43/44 仅保留为 base19 supplemental seed reference，不会被纳入 Flat 的三-seed mean 或 SD。状态与历史边界见 [Flat follow-up record](docs/FLAT_FOLLOWUP_20260912.md)。

结果位于 [final receipt](results/formal_campaign_concat_2015_m33_allseeds_20260912/final_concat_19/final_score_receipt.json) 与 [report](results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/report.md)。final roster 为 6 个日期，`final_sessions_opened=6`；此后不要重复运行 `score-final`，已完成 campaign 应只读查看 report 和 receipt。

协议固定为 2015-only 的 18 source / 6 development / 6 final session、M33 support 与 representation-matched MOVE-T4 cosine OLS。Full encoder 是 fresh、无 FiLM 的 `pre_pool → trial mean → concat(MOVE-T4) → post_pool`；这里的 `concat` 仅指 Full encoder 的 pooled activity 与 side input 融合，不是 M2-like decoder 的 `P16/proj_add` frontend。ACT 不读取 T，Raw-set 不使用 encoder。训练采用 CUDA bf16 autocast；development 与 final 评分采用 CUDA fp32。详细合同见 [CONCAT encoder contract](docs/CONCAT_ENCODER_CONTRACT.md)。

## Final 主表：15 格

各项为 six-final-date 的 equal-session mean R²；Full 主格为 seed42，7 个 baseline 与两个 Raw-PMUA static controls 也均已评分。

| Cell | SUA | PMUA |
| --- | ---: | ---: |
| Full, seed42 | 0.770706 | 0.782560 |
| Activity, seed42 | 0.448701 | 0.174831 |
| Raw-set, seed42 | 0.091951 | 0.074689 |
| WF-FSS | 0.320400 | 0.315672 |
| WF-ZS-H0 | — | 0.109415 |
| Diag-Z WF | — | 0.124517 |
| CORAL WF | — | 0.124517 |
| Aligned FA WF | — | 0.236022 |
| Aligned FA-stable WF | — | 0.178184 |
| Raw-set Diag-Z | — | 0.088714 |
| Raw-set CORAL | — | 0.088618 |

完整的六日期分数、cell 状态与 prediction artifact 绑定见 [main results CSV](results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/main_results.csv)。全部 19 cell 都是 `SCORED`，没有 unavailable cell。

## Base19 Full 的三 seed 汇总与配对差

下表仅是已完成 base19 Full decoder 的 seed42/43/44 参考。它们均在同表示、固定的 source-only seed42 encoder 条件下从 fresh decoder initialization 训练；sample SD 使用 `ddof=1`，它只度量 decoder-seed 变异，不度量 encoder-seed 变异。当前 Flat6 每个 cell 仅有 seed42，不能计算 Flat 三-seed mean 或 SD。

| 表示 / 配对量 | seed42 | seed43 | seed44 | 三 seed mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| SUA Full | 0.770706 | 0.779015 | 0.782654 | 0.777459 ± 0.006124 |
| PMUA Full | 0.782560 | 0.778187 | 0.760568 | 0.773772 ± 0.011642 |
| SUA − PMUA | −0.011854 | 0.000828 | 0.022087 | 0.003687 ± 0.017150 |

平均配对差很小且三个 seed 的符号改变；例如 seed42 是 PMUA 高于 SUA。因此这些结果不支持把 SUA 表述为对 PMUA 的稳定胜出。逐日期配对差见 [paired-difference CSV](results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/main_sua_pmua_paired_differences.csv)，完整 Full seed 表见 [full seed summary](results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/full_seed_summary.csv)。

Full−ACT 同时改变 source encoder 的预训练监督量与冻结策略，不能解释为目标角度标签的纯因果效应。WF-FSS 使用额外 dense velocity labels，而 Full 使用 trial-angle 信息，因此二者的标签预算不同；两者的数值可并列报告，不能直接当作同预算对照。

## 完成与可审计证据

| 证据 | 已核验事实 |
| --- | --- |
| [pre-final training verification](results/formal_campaign_concat_2015_m33_allseeds_20260912/PRE_FINAL_TRAINING_VERIFICATION.json) | 12 formal task、每项 24 段 / 75,960 updates、288 checkpoint SHA 完整 |
| [selection seal](results/formal_campaign_concat_2015_m33_allseeds_20260912/selection_seal_concat_19/selection_seal.json) | 15 main + 4 Full supplement selections 与两个 seed42 encoder 已冻结 |
| [final receipt](results/formal_campaign_concat_2015_m33_allseeds_20260912/final_concat_19/final_score_receipt.json) | six final dates 一次访问，19/19 `SCORED` |
| [final integrity verification](results/formal_campaign_concat_2015_m33_allseeds_20260912/FINAL_RESULTS_INTEGRITY_VERIFICATION.json) | `PASS`；seal 与 70 个执行源码 hash 匹配，19 NPZ 和 114 prediction arrays 均通过核验 |
| [report directory](results/formal_campaign_concat_2015_m33_allseeds_20260912/report_concat_19/) | 可读 CSV、JSON 与 Markdown 总结 |

历史工程证据仍保留，但不替代正式结果：[CPU concat smoke](results/smoke_concat_2015_m33_v1/receipt.json) 是 source-only、2 updates 的 `SMOKE`；[aggregate validation](results/concat_line_v1/VALIDATION.json) 记录当时 56 passed、一个 HDMF warning。B32 GPU feasibility 的 finite loss/gradient 与吞吐证据见 [SUA profile](results/gpu_profile_concat_20260912_sua/receipt.json) 与 [PMUA profile](results/gpu_profile_concat_20260912_pmua/receipt.json)。

若需从新的 fresh campaign 复现实验，使用 [machine-readable campaign plan](docs/CAMPAIGN_PLAN_20260912_CONCAT.json) 中不变的 12-task、24×3165、B32、seed 和 19-cell 定义；不要将以下命令用于已完成的 campaign root。运行中 campaign 的正式入口是：

```bash
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
BENCH=btransform_unified_v2/dandi688_bench_v2
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PY" "$BENCH/run_campaign.py" start \
  --root "$BENCH/results/<fresh_campaign_root>" \
  --cache "$BENCH/results/prepared_2015_m33_v2"
```
