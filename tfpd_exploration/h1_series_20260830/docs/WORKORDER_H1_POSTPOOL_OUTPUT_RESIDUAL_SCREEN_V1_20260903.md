# Work Order: H1 Post-Pool Output Residual Screen V1

日期：2026-09-03  
状态：`AUTHORIZED_SOURCE_ONLY_GPU0_SCREEN`

执行 `DESIGN_H1_POSTPOOL_OUTPUT_RESIDUAL_V1_20260903.md` 的五折 source-only
LODO 屏。必须绑定上一格 terminal
`ec86e26c513c3ccb9ab4ca5f8d0f0ce244e6b162b507c635f581078559a1804a`
和 score `62ebd18db6ce504c01414bd392a84a617c2b2aa2049af1e68b90e34bfc238d83`。

每折只允许：加载对应 C1 checkpoint；构造 static 与完整 post-pool 两路预测；
按设计的等日期/等 recording R² 闭式目标拟合唯一 beta；封存 beta；再打开 outer
date 并评分。不训练模型，不构造 optimizer，不允许 target update。

运行只可见 GPU0 UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`。不得查询或触碰 GPU1。结果根必须
新建且不可覆盖：
`tfpd_exploration/h1_series_20260830/results/h1_postpool_output_residual_v1/`。

按设计 §4 原样裁决。通过才允许 final all-source beta/package；失败则关闭
H1 post-pool output-residual family。本工单不授权 EvalAI POST。

