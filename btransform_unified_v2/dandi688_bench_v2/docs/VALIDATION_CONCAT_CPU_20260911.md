# DANDI688 CONCAT CPU validation — 2026-09-11

本记录绑定 [concat CPU smoke receipt](../results/smoke_concat_2015_m33_v1/receipt.json)、[smoke audit](../results/concat_line_v1/SMOKE_AUDIT.json) 与 [聚合验证收据](../results/concat_line_v1/VALIDATION.json)。它是 source-only `SMOKE`，`ineligible_for_formal=true`，不产生 development 或 final 分数。

- 全套测试：`56 passed`，一个已知 HDMF namespace warning，7.15 s；smoke exit code 为 0，耗时 16.3563 s。
- 两个 Full encoder pretrain：SUA 与 PMUA 各一个，均为 2 updates；六个 neural train arms：SUA/PMUA × Full/ACT/Raw，各 2 updates。
- 六个 EMA checkpoint reload checks：in-memory EMA 与 reload prediction 完全相同，`max_abs_delta=0.0`；每项 probe 为 source session 的 8 个 query。这是功能验证，不是 development 评价。
- Full 的真实 T 与零 T E0 L2 分别为 SUA `0.008555717766`、PMUA `0.00633774139`；side columns 已学习。两个 Full-frozen 和两个 ACT 的 T 拒绝合同通过。
- 七个 CPU baseline artifacts 与两个 Raw-PMUA static adapters 完成。既有 144 个 CPU baseline 的 concat 兼容性见 [compatibility proof](../results/concat_line_v1/BASELINE_COMPATIBILITY.json)。8 个 stage 的 code hashes 全部与当前源码一致。
- receipt 禁止 non-2015、development、final 与 full training；`final_sessions_opened=0`。GPU policy 仍为 [enabled=false](../results/GPU_EXECUTION_POLICY.json)。

子 receipt 的 recipe 均记录 `encoder_line=concat`、`encoder_carrier_fusion=post_pool_input_concat`、`encoder_film=false`。正式 concat 训练、selection 和 final 评分尚未运行。旧 Full-FiLM pretrain 权重不能复用，见 [终止收据](../results/formal_campaign_2015_m33_allseeds/TERMINATED_BY_USER.json)。
