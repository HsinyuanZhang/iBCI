# H1 六折跨日期 LODO 三臂实现（2026-09-08）

本实现定义一个新的 H1 跨日期训练和评分协议，不修改既有 H1 official receipt、submission、旧结果或论文数值。数值执行由独立运行者完成；本文件仅说明已实现的代码契约和运行命令。

六个 held-in 日期 `1925-01-01`、`1925-01-08`、`1925-01-13`、`1925-01-15`、`1925-01-19`、`1925-01-20` 分别作为一个 fold 的 target date。source 是另外五个日期的全部记录，因此按 target date 的 session 数量自然得到 10 或 11 条 source recording，代码不假定固定 source 数量。整个协议只使用 13 条 held-in recording。

三臂共享同一 `R300/D4/recency/P16` decoder 初始化、训练预算、source sampler、dropout recipe 和 EMA。`Z_NONE` 不读取或构造 M3 target calibration；`B_ACTIVITY_ONLY` 只使用 M3 的 random C2-shaped activity identity；`D_JOINT` 使用同一 M3 activity 以及每 fold 由 source M3 重建的 H-C carrier。`Z_NONE` 物理上不实例化 encoder；B/D 在隔离 RNG domain 中各自实例化同字节 random C2。preflight 比较三臂共同 decoder key 的初值，并比较 B/D 的额外 encoder key；它报告真实 trainable 参数数和实际获得非零梯度的参数数。B/D 的 encoder 增量必须小于总参数的 5%。没有加载旧 e15 或任一旧 checkpoint。

`h1_prepare.py --surface source` 是唯一构建 H-C authority 的路径。它只打开 source recording 的前 3 whole trials，以 `q=12`、`lambda=10`、source PCA、SVD、EB 和 source RMS 生成并持久化 plan arrays。`--surface target` 只读取已持久化的 source authority 和当前日期 target recording，绝不重拟合或读取 source NWB。每条 recording 以 available eval-valid native trial 的前 3 条作 support；source train 使用 indices `3:-2`、stride 4，source validation 使用最后两条 indices `-2:`、stride 4，target score 使用 indices `3:` 的全部 valid bin。native trial ID 不假定从 1 开始或总数为 15。每个相邻 native trial segment 都 reset raw context，窗口不进入 M3/support 前缀。source validation 也独立 reset，且绝不进入梯度或选择以外的 target 流程。

训练固定为 32 epochs、batch 32、AdamW `1e-4`/`.01`、gradient clip `1`、EMA `.9995`、第一 epoch warmup 后 cosine floor `.1`、paired whole-unit dropout `.1`、behavior scale `20`。RIFT trunk 在 CUDA 下使用 bf16；random C2 encoder 明确以 fp32 运行。每 epoch 用 source-validation EMA equal-session mean 选最早最大值；score 同时写入该 selected EMA 以及固定 epoch 32 EMA 的 target result。target labels 只在最终 R² 评分中读取，不进入训练或 checkpoint 选择。

`preflight` 真正初始化三臂并检查：共同 decoder key 的初值 byte equality、B/D encoder key 的初值 byte equality、真实参数差小于 5%、有限的非零 parameter backward gradient、Z 对 calibration 不敏感、B 对 activity 敏感且对 carrier 不敏感、D 的 activity/carrier/双路径响应、source raw interval 与 support 的分离、stride 和 session 合同。它不打开 target arrays。每个训练 step 的第一个及每 50 step 记录 heartbeat JSON/stdout elapsed；每 epoch 写 atomic `resume_latest.pt`，其中含 model、optimizer、EMA、curve、Python/NumPy/Torch/CUDA RNG；每 epoch score checkpoint 仅保存 EMA、固定-buffer/构造合同和 recipe binding。`--resume PATH` 从完整 resume 精确状态继续。EMA 的完整 unique parameter key、shape、finite 和 no-alias 条件在 update/load/evaluate 时均检查；load 使用 `strict=True`，EMA evaluate 验证 fixed buffers 未变。

建议的实际运行顺序如下。`PREPARED` 和 `RUNS` 应为新的空目录，不能覆盖旧结果；每个 fold/arm 单独 `DEST`。这些命令是运行者执行的数值工作，不是本文档执行结果。

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/cross_session_v1/h1_prepare.py --dest "$PREPARED" --fold 1925-01-01 --surface source
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/cross_session_v1/h1_train.py --prepared "$PREPARED" --dest "$RUNS/1925-01-01/Z_NONE" --fold 1925-01-01 --arm Z_NONE --stage preflight --device cuda:0
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/cross_session_v1/h1_train.py --prepared "$PREPARED" --dest "$RUNS/1925-01-01/Z_NONE" --fold 1925-01-01 --arm Z_NONE --stage train --device cuda:0
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/cross_session_v1/h1_prepare.py --dest "$PREPARED" --fold 1925-01-01 --surface target
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/cross_session_v1/h1_train.py --prepared "$PREPARED" --dest "$RUNS/1925-01-01/Z_NONE" --fold 1925-01-01 --arm Z_NONE --stage score --device cuda:0
```

对其余五个 fold 与 `B_ACTIVITY_ONLY`、`D_JOINT` 重复相同流程。恢复使用同一 `--dest` 和精确的 `--resume "$DEST/resume_latest.pt"`。可审核产物包括 source/target manifest 和 source H-C authority、`preflight.json`、`run_meta.json`、heartbeat、每 epoch EMA checkpoint、atomic resume、`selection.json`、`train_receipt.json` 和 `target_score.json`。receipt 绑定 source/target manifest、实际 array hashes、initial parameter hash、每 epoch sampler/endpoint/keep hash、每个 checkpoint SHA-256，以及 `cross_session_h1_model.py`、`h1_prepare.py`、`h1_train.py` 和 H-C fitting helper `h1_m4_eb_pilot.py` 的 source SHA-256。
