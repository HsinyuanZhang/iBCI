# Current recency-flat ablation protocol — 2026-09-10

用户已授权三项最终消融。比较仅包含三个当前单 seed (`42`) recipe：论文 M1 muscle FULL R100、H1 signed_state14 R300、M2 aligned concat R50。每个 task 只将固定 temporal recency slopes 改为精确全零 flat slopes，并保持同族 trainable parameter 初始化配对。历史 H1、meanrate4 M1 partial run 和既有探索性结果只作历史材料，不进入此最终比较。

执行顺序固定为 H1 → M2 → M1_FULL。M1 original reference 是 `results/m1_muscle_r100_v1/formal_s42_gpu1`，flat run 是 `results/recency_flat_ablation_v1/formal_m1_muscle_full_flat_s42`；两者固定 24 × 6665 updates、HO3 的完整 24-epoch EMA 曲线、channel-variance-weighted R² equal-session mean 和各自 earliest maximum。两者使用 `results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz`，并绑定 `muscle_response16_svd4/global_rms` carrier、carrier receipt 和 fit。M1 fixed original FULL recipe 的 EvalAI `582205` 是论文主线 provenance；它不替代本消融的 development-surface effect。

H1 保留 32 个 epoch 的 grouped-seven HO-M3 曲线与各自 earliest maximum。M2 保留 24 个 epoch 的 EXT6 equal-session mean 曲线与各自 earliest maximum。每个 selected checkpoint hash、selected flat slope buffer、source contract 与 reference binding 都必须在汇总前通过检查。

最终汇总器读取 [protocol_muscle_full_v2.json](../results/recency_flat_ablation_v1/protocol_muscle_full_v2.json)，默认输出 `results/recency_flat_ablation_v1/current_three_task_muscle_full_summary_v2`。它要求已完成的成对 decoder CPU B1/B4/B8 benchmark、完整 pure-temporal matrix，以及每 task 五轮 alternating、fresh-parameter GPU train-step benchmark。任何缺失或错误的曲线、selection、source binding、checkpoint、flat all-zero slope、latency 或 cost receipt 都会在创建 output 前中止。

效果数值是单个 paired seed 的 development-surface 观察值，不是 official flat-ablation 结果，也不提供 multi-seed confidence interval。CPU 报告保留 raw per-round data、mean、median、p95 和 observed round range；比值仅描述配对测量。GPU training-step cost 把 fixture construction 与 controlled timed updates 分开。full-run receipt wall time 仅作为 resource-dependent observation。

decoder CPU input 必须为 `selected_ema_decoder_cpu_cached_v1`：三个 batch case、每 mode 五轮、aggregate raw samples、parity 和 unchanged runtime-source seal 全部必需。GPU input 必须为 `rift_recency_flat_real_train_step_gpu_v1`：每 mode 五轮、fresh-parameter pairing、10 warmup、30 timed updates、每轮 40 EMA updates、正确 flat/recency slope evidence，以及与 effect row 相同 recency metadata 的 fixture reference seal。汇总器按 serialized field 与 live SHA-256 验证，不重建任何测量。
