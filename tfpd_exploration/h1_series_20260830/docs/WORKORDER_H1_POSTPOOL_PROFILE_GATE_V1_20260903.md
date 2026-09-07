# Work Order: H1 Post-Pool Profile Gate V1

日期：2026-09-03  
状态：`AUTHORIZED_GPU0_SOURCE_ONLY_12EP`

执行 `DESIGN_H1_POSTPOOL_PROFILE_GATE_V1_20260903.md`。绑定两个已完成负结果：

- direct identity screen terminal：`ec86e26c513c3ccb9ab4ca5f8d0f0ce244e6b162b507c635f581078559a1804a`；
- scalar output residual terminal：`8958985646dd801a590cd7bf5429d1be6d061e4eb95c60857e67b9ec007c0f5f`。

每折只能训练 700 维 profile；C1 模型参数必须冻结，optimizer 参数集合必须严格
等于 profile。source profile 封存前不得打开 outer date。训练固定 12 epoch，
不得基于中间 loss early-stop、选 epoch 或改学习率。

仅使用 GPU0 UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`；不得查询、占用、
发信号或修改 GPU1 进程。新结果根：
`tfpd_exploration/h1_series_20260830/results/h1_postpool_profile_gate_v1/`。

按设计 §4 裁决。通过才授权 all-source profile/package；本工单本身不授权
EvalAI POST。

