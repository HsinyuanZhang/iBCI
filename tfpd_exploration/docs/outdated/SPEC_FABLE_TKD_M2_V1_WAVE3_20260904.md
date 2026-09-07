# SPEC_FABLE_TKD_M2_V1_WAVE3 — drop 曲线 + CPU 延迟（Wave 3）

Date: 2026-09-05
For: 实现agent（flash）。前置：Wave 2 交付（pilot PASS + REF-SHUF 重放完成）。
父工单：`WORKORDER_FABLE_TKD_M2_V1_20260904.md` §5（成本/延迟/drop 协议）+ ADDENDUM-1（G2 口径）。
范围：`dropcurve.py`、`latency.py`（包内）+ 一个 runner 脚本。结果 `results/fable_tkd_m2_v1/wave3/`。

## 0. 纪律

同 Wave 2（GPU1 preflight 若用 GPU； receipts 原子写+0444+sidecar；fail-closed；不改既有文件——`eval_neuron_drop_curve.py` 只读参照）。

## 1. drop 曲线（G4）

协议镜像 `streaming_calibration_exp/scripts/eval_neuron_drop_curve.py`：**被 drop 的 unit 在校准试次与在线窗同时置零**（电极退化语义；fraction ∈ {0.10, 0.25, 0.50}，draw seeds {42,43,44}）。

- **TKD 侧**（A′ 主 seed best.pt）：对每个 (fraction, seed)：mask = 从 96 channel 均匀无放回抽 drop 数；外部 6 session，每 session：校准块置零被 drop 的 channel → **闭式重拟合 T4/ρ**（fit_ridge_t4 在全零 rate 列上的退化拟合：a=c=b=0, m=0 → t 走正常归一化；ρ 对零方差列定义为 0）→ 在线窗置零 → 前向 → 逐 session R²。零梯度不变。
- **champion 侧**：优先直接用 `eval_neuron_drop_curve.py`（它接受 `StreamingCalibrationLitModule` ckpt——M2 champion best.ckpt 就是；数据面若脚本内置与 M2 不符，做**最小适配副本**放我们包内，不改原脚本，适配点记 deviations）。champion 的 identity E 从掩码后校准块重建（脚本 apply_mask_to_calib 同律）。同一 (fraction, seed) 集合。
- 读数：每个 fraction 的 (TKD 下降, champion 下降)（各自相对 fraction=0 的均值，3 draws 平均）。**G4 判定：drop 0.25 时 TKD 下降 ≤ champion 下降**。receipt `wave3/drop_curve.json`：全表 + G4 布尔 + 适配记录。

## 2. CPU 延迟（G2 延迟半门）

- 条件：纯 CPU（`CUDA_VISIBLE_DEVICES=""`），`torch.set_num_threads(1)`，机器无 GPU 任务（receipt 里记录当时 nvidia-smi 摘要），warmup 100 步丢弃，≥2000 计时步，报 mean/std ms。
- **champion**：frozen champion ckpt，batch=1 逐窗前向（每窗 = 一个输出 bin 的官方 stride-1 面语义），同一外部 session 的连续窗。ms/输出bin。
- **TKD**：A′ 主 seed best.pt **流式模式**——逐步：ψ 的因果 conv 单步（新 bin 卷积窗）+ Linear → u_t；静态 key（ε 训练后若非 0，用 ε·P u_t 修正项逐步算，ε 终值如实使用）→ α_t → z_t → merge → ssm_step×2 → readout。ms/输出bin。**流式实现必须先过 parity 断言**：与窗口批前向在同一 session 500 窗上预测逐元素 allclose ≤1e-5（conv 边界零填充语义两侧一致），否则 fail-closed。
- 附带：TKD 窗口批模式 CPU ms/窗（参照）。**G2 延迟门：TKD ms/bin ≤ champion ms/bin ÷ 3**。receipt `wave3/latency.json`：双侧计时 + 硬件线程上下文 + parity 断言结果。

## 3. 验收

`run_fable_tkd_m2_wave3_v1.py --execute`（内部：TKD drop → champion drop → latency）→ `wave3/terminal.json`。报告：G4 表、G2 延迟比、流式 parity、deviations。
