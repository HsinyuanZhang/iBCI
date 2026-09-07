# SPEC_FABLE_TKD_M2_V1_WAVE2 — TKD M2 训练/评测/重放（Wave 2）

Date: 2026-09-04
For: 实现agent（flash）。前置：Wave 1 + Wave 1.5 已交付且 stage0 terminal=PASS（或 A1 披露带通过）。
父工单：`WORKORDER_FABLE_TKD_M2_V1_20260904.md`（含 ADDENDUM-1）。Wave 1 spec：`SPEC_FABLE_TKD_M2_V1_IMPL_20260904.md`。
范围：`train.py`、`evaluate.py`、runner、REF-SHUF（ts4）重放。**延迟/drop 曲线不在本波**（Wave 3）。

## 0. 硬纪律（同 Wave 1，另加）

- **GPU1 only**：runner 启动时先做 preflight 再 import torch——`nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader -i 1` 必须 = `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`；`nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader` 中 GPU1 上不得有任何 pid（有则拒绝并列出）；GPU1 util ≤ 5% 且 mem ≤ 500 MiB；随后设 `os.environ["CUDA_VISIBLE_DEVICES"]="1"`。参照 `tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/gpu.py`。**GPU0 绝不触碰**。
- **FP32 严格**：`torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False`；全模型/数据 float32。
- 确定性：每 run 开始 `random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)`；窗口 shuffle 用独立 `np.random.default_rng(s)` 派生流（记录用法）。
- Receipt 律同 Wave 1（原子写+0444+sha256 sidecar；attempt 先行；fail-closed）。
- 不改既有文件；不 git 操作；不动密封物。

## 1. plan.py 增补常量（只增不改）

- `ARMS = ("A2","A","SHUF","POOL","B","A2_GRU")`（内部名 A2 表示主臂 A′）；`ARM_TABLE`：arm → (epsilon, key_mode, time_model, init, frozen_readin) 映射：A2=(learnable,identity,ssm,pv,False)、A=(zero,identity,ssm,pv,False)、SHUF=(learnable,shuffled,ssm,pv,False)、POOL=(learnable,pool,ssm,pv,False)、B=(learnable,identity,ssm,pv,True)、A2_GRU=(learnable,identity,gru,pv,False)。注：GRU 臂走 PV read-in init（读入侧同律）但 GRU 本体默认 init（spec §3.2 已记）。
- REF 逐 session 常量 `REF_EXTERNAL_PER_SESSION`（act30 champion p0，来源 probe score.json，写死 + 注释路径）：{"ses-2020-10-30-Run1":0.45756, "ses-2020-10-30-Run2":0.41392, "ses-2020-11-18-Run1":0.29073, "ses-2020-11-19-Run1":0.15933, "ses-2020-11-24-Run1":0.25785, "ses-2020-11-24-Run2":0.19193}（实现时以 probe score.json 实际 full-precision 值为准抄录，此处 5 位小数仅为索引）。
- `GRAD_CLIP=1.0`、`EVAL_BATCH=1024`、`P0_R2_TOLERANCE=1e-6`。

## 2. train.py

`train_one(repo_root, arm, seed, device) -> dict`：

1. 数据：`data.py` 的 7 个 held-in session post-30 窗（全部窗，不采样）；输入 `x[n,50,96]`、raw target `y[n,2] = covariate[start+49]`（**不乘 5.0**——损失在 raw 目标上，模型内部已 ÷5.0；与 champion 的 ×5.0 训练只差损失整体尺度，receipt 里记录此选择）；per-session `(t, rho, t4_raw, digests)`。
2. 模型：`TKD(**ARM_TABLE[arm])`，`pv_beta=plan.PV_BETA`、uniform masses、authority mean/std。
3. 优化：`torch.optim.Adam(filter(requires_grad), lr=3e-4, weight_decay=0)`；batch 32；每 epoch 全窗 shuffle（rng 流见 §0）；`loss = MSE(model(x,t,rho), y)`；backward 前 `clip_grad_norm_(1.0)`；断言 loss finite。
4. 每 epoch 末：within 面评测（7 session 全 post-30 窗，batch EVAL_BATCH，no_grad，eval 模式）→ `equal_session_mean`（用 `variance_weighted_r2` 逐 session）→ 记录 per-epoch 表。
5. 30 epochs 结束：选 best epoch（within 等权均值最大；平手取更早），保存该 epoch 的 `state_dict` 到 `results/fable_tkd_m2_v1/runs/<arm>_s<seed>/best.pt`（atomic 写 + 0444 + sidecar），receipt `train.json`：arm/seed/常量 echo、每 epoch within 均值、selected_epoch、state sha256（对 `state_dict` 逐 tensor bytes 做 sha，仿 `state_hash` 律）、参数量、训练 wall time、`gpu_uuid`、TF32=False 断言记录。
6. 训练中任何 NaN/inf → fail-closed 写 failure.json。

## 3. evaluate.py

`evaluate_run(repo_root, arm, seed, device) -> dict`：

1. 载入 `best.pt`（sha 校验 vs train.json），模型 eval/no_grad。
2. **外部面**（6 session）：`dataset.window_indices` 原样；每 session 闭式 `(t, rho)`（M30 律，digest 断言与 authority 一致）；batch 前向；预测 ÷5.0 已在模型内；`variance_weighted_r2` 逐 session + `equal_session_mean` + `summarize_sessions`；paired_contrast vs `REF_EXTERNAL_PER_SESSION`。
3. **within 面**（7 session post-30 全窗）同法。
4. ε 终值（A2/SHUF/POOL/B 臂）、attention 静态权重 α 的 per-bin 质量摘要（ε=0 或训练后 ε≈0 时可缓存路径，bit-exact 断言）。
5. receipt `eval.json`（0444+sidecar）：全部上述 + model/train receipt 的 sha 引用链。

## 4. runner：`tfpd_exploration/scripts/run_fable_tkd_m2_v1.py`

CLI（默认 dry 打印计划，`--execute` 才动真）：
- `--execute --mode pilot`：GPU1 preflight → attempt.json（`fable_tkd_m2_v1/stage1_attempt.json`，若已存在则拒绝）→ train A2/s42 → evaluate → **sanity 门**：外部 `equal_session_mean ∈ [0.10, 0.45]` 且 within 第一 epoch → 最后 epoch 单调改善（允许 ≤2 个非单调 epoch，改善幅度 >0 为准）。过 → 写 `stage1_pilot.json`（PASS）；不过 → failure.json 并退出码 1。
- `--execute --mode grid --arms A2,A,SHUF,POOL,B,A2_GRU --seeds 42,43,44`：逐 (arm,seed) 串行（A2_GRU 只跑 seed 42）；每 run 独立 receipt；中断可续（已存在且 sha 链完整的 run 跳过并列出）；结束写 `stage1_grid.json` 汇总表（arm×seed → 外部均值/逐 session/within/ε）。
- `--execute --mode eval --arm X --seed Y`：单 run 重评。
- 每 run 前重新 preflight（GPU1 可能被占）。

## 5. REF-SHUF 重放：`tfpd_exploration/scripts/run_fable_tkd_ref_shuf_replay_v1.py`

1. 读 `m2_hold_film_probe_v1` 的加载路径（`sua_exploration.evalai_t4_m2.export_t4_payload.load_frozen_model_and_data`），把它重建的 champion ckpt 换成 **ts4 ckpt**（路径 `streaming_calibration_exp/outputs/streaming_calibration/e8_ts4_m2_submission_control_m33q33_v1_s42_20260801_162010/checkpoints/best.ckpt`，sha 必须等于 `e385e2f408d6b3fe65645be934a8f69b3ab4e68b986e56e64512611cac040399`；normalizer/teacher sha 沿用 probe 断言）。**前置 sanity**：同一加载路径先用正主 champion ckpt（sha 25d7bc72…）跑一遍外部面，断言 `equal_session_mean` == `SEALED_M30_EXTERNAL` within 1e-6（复现 probe p0），不过则 fail-closed（说明重放面漂移，G3 校准不可信）。
2. ts4 ckpt 外部面逐 session R² + `equal_session_mean` → `Δ_ref = SEALED_M30_EXTERNAL − ts4_mean`（G3 校准量）。
3. receipt `results/fable_tkd_m2_v1/ref_shuf_replay.json`（0444+sidecar）：两个 ckpt 的 sha、逐 session 表、p0 复现误差、Δ_ref。GPU1 允许（纯推理）或 CPU；脚本内做同样 GPU preflight（若用 GPU）。

## 6. 验收（agent 自验后报告）

1. pilot 模式全流程跑通（若 stage0 已 PASS 可直接跑）：A2/s42 的外部均值、逐 session、sanity 门结果、receipts 路径。
2. grid 模式 dry 输出（列 16 个 run 计划），**不要在 Wave 2 里替我跑 grid**（我审核 pilot 后另行下令）。
3. REF-SHUF 重放的 p0 复现误差 + Δ_ref。
4. deviations 列表 + 每 run wall time。
