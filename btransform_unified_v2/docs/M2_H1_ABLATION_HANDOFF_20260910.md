# M2/H1 消融执行归属交接（2026-09-10）

## 用户决定与范围

用户于 2026-09-10 约 01:31 UTC 决定将 M2/H1 ablation 的训练、打包和 official result 跟进交给用户指定的另一 agent。本 ROOT 线程之后只负责 M1 carrier 分析与 M1 本地消融。

这是一项执行归属记录，不表示另一 agent 已启动、接收、提交或完成任何任务；六个消融的实验合同、历史结果、固定输入和验收门槛均未改变。M1 此前“本线程不负责提交 EvalAI”的边界继续有效，不由本次交接产生上传、register 或 submit 授权。

## 历史共享队列与当前 M1 本地 pack

2026-09-10 01:32:03 UTC 的交接观察中，formal queue 与 `pack_recovery_v1` 都是共享控制器；该观察只保留为历史。之后 M1 NONE formal train 已于 `2026-09-10T01:58:33.766775+00:00` 启动，child pid `399478`；score 仍待 formal train 完成后由同一 [formal queue](../results/final_ablation_official_v1/root_formal_queue.json) 启动。正式 556-file freeze 仍为 `fd960824387c3b600f1132b233e85556eeb73f5b9023199b94c2d24b6eb2c3a6`。

旧 [pack_recovery_v1 queue](../results/final_ablation_official_v1/pack_recovery_v1/root_pack_queue.json) 已 `FAILED` 并关闭，因为其冻结输入中的 `tfpd_exploration/submissions/evalai_h1_rift_r300_ablation_v1/decode.py` 在启动后变更；这不是 M1 formal 输入或 M1 pack 依赖，本文不修复、也不归责该 H1 变更。M2/H1 后续状态由用户指定的另一 agent 负责。

ROOT 当前仅管理独立的 [M1 pack recovery queue](../results/final_ablation_official_v1/m1_pack_recovery_v1/root_pack_queue.json)：pid `402604`，状态 `WAITING_FOR_FORMAL_STAGE`，唯一 job 为 `m1_none` 且仍 `PENDING`。其 7-file M1-only input freeze SHA-256 为 `34deb9b53c26cc09c6b0c69b2011eb854b8fbb4413d075065a143ac62c54af7f`，仍绑定上述 556-file formal freeze。该 queue 只等待 M1 train/score 的 `COMPLETED`/`returncode=0`，随后才调用 canonical build、host、Docker offline smoke、container byte-hash 与 manifest gates；目前没有 payload、package-ready 状态、外部注册或提交。

H1 NONE 的 `19/32` 只是 2026-09-10 01:31:31 UTC 的历史交接观察，不是 final selection、performance 或 package readiness。

## 关键入口与历史边界

- 当前六消融合同与历史证据：[FINAL_ABLATION_OFFICIAL_PLAN_20260909.md](FINAL_ABLATION_OFFICIAL_PLAN_20260909.md)。
- H1 ACT 的有效本地包：`tfpd_exploration/submissions/evalai_h1_rift_activity_only_r300_v2`。
- H1 NONE 的原目标包：`tfpd_exploration/submissions/evalai_h1_rift_none_r300_v1`。
- 旧 pack controller [root_pack_queue.json](../results/final_ablation_official_v1/root_pack_queue.json) 是 `FAILED` 历史记录，不能作为当前 active queue。
- 冻结的 556-file formal manifest、旧 15-file pack snapshot、recovery 21-file snapshot 以及两条 controller 都不得因本交接修改。
- 2026-09-10 01:09 UTC 的 56-row/no-new-ID official GET 是历史只读观察，不构成未来持续监控承诺。

M2/H1 负责 agent 应自行核验后续训练、评分、package 与 official 状态。本文不复制日志或执行命令。
