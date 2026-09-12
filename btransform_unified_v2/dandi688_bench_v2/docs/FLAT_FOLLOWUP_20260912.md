# DANDI688 Flat follow-up — seed42 范围记录（2026-09-12）

当前 Flat follow-up 的冻结范围是 6 cell，且全部为 seed42：SUA/PMUA 各包含 Full、Activity 与 Raw-set。它是在已完成的 base19 CONCAT final 之后独立加入的比较，不能改写 base19 的 19-cell final receipt。当前 [frozen Flat6 plan](../results/formal_flat_followup_2015_m33_seed42_20260912/plan.json) 已明确列出 6 个 final cell、24 × 3,165 = 75,960 updates 和 final roster；[adoption plan](../results/formal_flat_followup_2015_m33_seed42_20260912/ADOPTION_PLAN.json) 将两个尚在训练的 Full-flat seed42 child 绑定到新范围，并确认 `flat_final_sessions_opened=0`。

## 当前状态与结论边界

新 Flat6 controller 已于 `2026-09-12T08:39:51Z` 实际启动：[launcher](../results/formal_flat_followup_2015_m33_seed42_20260912/launcher.json) 为 `RUNNING`，SUA/PMUA worker PID 分别为 `709863`／`709864`，watcher PID 为 `709865`。[start verification](../results/formal_flat_followup_2015_m33_seed42_20260912/START_VERIFICATION.json) 为 `PASS_SEED42_ADOPTION_AND_CONTROLLER_START`，确认 plan `25c7e04…`、adoption binding、78 个执行源码及 74 个冻结训练源码均匹配，并确认启动后 90 秒 watcher heartbeat。当前 [watcher](../results/formal_flat_followup_2015_m33_seed42_20260912/completion_watcher.json) 为 `WAITING`／`WORKERS`，两队 Full42 worker 都在运行；验收快照为各 `50,640` steps／第 16 段。随后 controller 才会运行 Activity42 和 Raw42。

[final-pipeline tests](../results/formal_flat_followup_2015_m33_seed42_20260912/FINAL_PIPELINE_TEST_RESULTS.json) 为 `14 passed`、`0 xfail`，其 [validation](../results/formal_flat_followup_2015_m33_seed42_20260912/FINAL_PIPELINE_VALIDATION.json) 为 `PASS_FINAL_PIPELINE_VALIDATION`，绑定 canonical plan、adoption-plan SHA 和 78 个执行源码。这 14 个 final gate/report tests 与此前 14 个 training/launcher tests 分开计数。launcher/watch claim 防止重复启动，且 `automatic_retry=false`；完成规则仍是 6 个 task 全部核验后再一次性 seal、final6 和 merged25。因此目前没有 Flat development selection、Flat final access、Flat final score 或 Flat 科学结论；完整 25-row 比较仍待完成。

[comparison progress](../results/formal_flat_followup_2015_m33_seed42_20260912/COMPARISON_PROGRESS.md) 提供当前 25 行：19 个 `SCORED` base19 final 行直接来自 [base final receipt](../results/formal_campaign_concat_2015_m33_allseeds_20260912/final_concat_19/final_score_receipt.json)，6 个 Flat6 行均为 `PENDING`。表中的六日期和 mean final R² 仅来自 receipt；Flat `PENDING` 行不填写 development 分数或任何替代数值。对应的机器可读文件为 [comparison_progress.csv](../results/formal_flat_followup_2015_m33_seed42_20260912/comparison_progress.csv)。

Full-flat 的两个进行中 seed42 child 已由新 worker 接管且未中断：[queue handoff](../results/formal_flat_followup_2015_m33_seed42_20260912/QUEUE_HANDOFF.json) 记录旧 parent supervisor `703727`／`703728` 于 `2026-09-12T08:40:41Z` 在验证 adoption 后终止，新 worker 继续监管 preserved child `703871`／`703872`。旧 watcher 已停止；当前 controller 和 watcher 已运行。

## 已 supersede 的 10-task 历史范围

此前的 10-task Flat plan 位于 [old frozen plan](../results/formal_flat_followup_2015_m33_allseeds_20260912/plan.json)，其中含 Full-flat seed43/44，并以 base19 + Flat10 的 29-row report 为完成目标。它已经被当前 seed42-only Flat6 范围 supersede，不再是当前执行计划。其历史工程证据仍可只读查看，但不能当成当前 Flat6 计划或其执行状态。

已完成 base19 的 `full_sua_s43`、`full_sua_s44`、`full_pmua_s43` 与 `full_pmua_s44` 保留为 `Full learned` supplemental seed reference。它们属于 base19 完成结果，不能用于构造 Flat 单 seed42 的三-seed mean、SD 或配对不确定性结论。

## 冻结的 Flat6 roster

| Method | Representation | Cell | Seed | Temporal variant | Final status |
| --- | --- | --- | ---: | --- | --- |
| Full | SUA | `full_flat_sua` | 42 | flat / fixed all-zero recency slope | `PENDING` |
| Activity | SUA | `activity_flat_sua` | 42 | flat / fixed all-zero recency slope | `PENDING` |
| Raw-set | SUA | `raw_set_flat_sua` | 42 | flat / fixed all-zero recency slope | `PENDING` |
| Full | PMUA | `full_flat_pmua` | 42 | flat / fixed all-zero recency slope | `PENDING` |
| Activity | PMUA | `activity_flat_pmua` | 42 | flat / fixed all-zero recency slope | `PENDING` |
| Raw-set | PMUA | `raw_set_flat_pmua` | 42 | flat / fixed all-zero recency slope | `PENDING` |

Full-flat 复用对应表示的 completed base seed42 encoder 并冻结它；Activity-flat 和 Raw-set-flat 不使用 encoder。base19 的 CPU baseline、static controls 与封存的 19-cell final 不会重跑。

当前 controller 的运行状态、训练核验和最终评分边界以其 launcher、watcher、handoff 和 start-verification 证据为准；不应从历史 F10 watcher、旧 queue 或训练日志推导 Flat6 完成状态。
