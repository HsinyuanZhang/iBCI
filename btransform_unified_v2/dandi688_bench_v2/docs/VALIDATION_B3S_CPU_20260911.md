# DANDI688 B3S CPU validation — 2026-09-11

本记录绑定无 FiLM B3S 实现的 CPU 验证，不是正式训练、开发选择或 final 评分的证据。完整 smoke receipt 为 [smoke_b3s_2015_m33_v1](../results/smoke_b3s_2015_m33_v1/receipt.json)。

## 已验证

- 全部测试：`54 passed`，耗时 6.84 s；唯一输出为已知 HDMF namespace warning。
- B3S smoke：只使用 `prepared_2015_m33_v2` 的两个 2015 source sessions；每个训练 stage 固定 2 optimizer updates，receipt 为 `SMOKE` 且 `ineligible_for_formal=true`。
- 两个 Full encoder pretrain：SUA 与 PMUA 各一个；六个 neural train arms：SUA/PMUA × Full/ACT/Raw；六个 checkpoint reload parity 均通过。reload 是两个 source session、每次 8 个 query 的功能验证，不产生 development 或 final 分数。
- 七个 CPU baseline artifacts：`wf_zs_h0`、`diag_z_wf`、`coral_wf`、`aligned_fa_wf`、`aligned_fa_stable_wf`、`wf_fss_sua`、`wf_fss_pmua`；Raw-PMUA 的 `diag_z` 与 `coral` 两个 static adapters 均完成。
- receipt 禁止 non-2015、development、final 与 full training；`final_sessions_opened` 为 0。
- 既有 144 个 CPU baseline 配置可保留：[baseline compatibility proof](../results/b3s_migration_v1/BASELINE_COMPATIBILITY.json) 记录旧 `common.py` 源 SHA 与 snapshot 精确匹配、除顶层 `SCHEMA`/`RECIPE` 外 AST 全等，并验证其依赖的当前 hash 全部通过。

## 当前边界

正式 B3S 训练尚未运行。GPU execution policy 仍为 [enabled=false](../results/GPU_EXECUTION_POLICY.json)，不会自动重启任何 campaign。此前的 SUA/PMUA Full-FiLM pretrain 各在 4 segments、12,660 updates 后终止；其权重不满足 B3S contract，不能复用。见 [终止收据](../results/formal_campaign_2015_m33_allseeds/TERMINATED_BY_USER.json) 和 [B3S encoder contract](B3S_ENCODER_CONTRACT.md)。
