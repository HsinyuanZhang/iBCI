# CONCAT encoder contract

CONCAT 是当前 DANDI688 统一 encoder 线。它只定义 Full encoder 内 carrier 的融合位置：

```text
support[M33, T100, N_pad=100]
→ pre_pool: Linear(T100, h64) + ReLU
→ trial mean over M33
→ Full only: concat(pooled activity h64, representation-matched MOVE-T4 side T4)
→ post_pool: three-layer MLP → E0[N_pad, 50]
```

SUA 与 PMUA 各自用 18 source sessions 预训练一个 representation-matched Full encoder。encoder 固定导出最后一个 EMA 并冻结；每个 Full decoder 使用 fresh initialization。ACT 只使用 `pre_pool → trial mean → post_pool`，不读取 T。Raw-set 不使用 encoder、M33 support 或 carrier。encoder 不含 FiLM。

`concat` 不表示 M2-like decoder 的 `P16/proj_add` frontend。Full decoder 仍有其既定直接 T4 carrier route；ACT 该 route 为零。DANDI 保持 T100/h64/E50、R50 和 learned temporal recipe。

`B3SIdentityEncoder` 类名与 schema 仅为 artifact migration continuity 保留。旧 FiLM weights 不能复用，见 [termination receipt](../results/formal_campaign_2015_m33_allseeds/TERMINATED_BY_USER.json)。当前正式工作仅以 [fresh campaign root](../results/formal_campaign_concat_2015_m33_allseeds_20260912/) 中的 outputs 为准，任务与选择规则见 [2026-09-12 plan](CAMPAIGN_PLAN_20260912_CONCAT.json)。
