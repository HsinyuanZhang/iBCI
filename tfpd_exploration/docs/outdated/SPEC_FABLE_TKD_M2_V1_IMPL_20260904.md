# SPEC_FABLE_TKD_M2_V1_IMPL — TKD M2 实现规格（Wave 1：模型/数据/Stage 0）

Date: 2026-09-04
For: 实现agent（flash）。父工单：`tfpd_exploration/docs/WORKORDER_FABLE_TKD_M2_V1_20260904.md`（必读）。
设计来源：`FABLE0904_decoder_design1.md` §2–§3（必读）。
Wave 1 范围：`plan.py`、`data.py`、`model.py`、`stage0.py`、tests、stage0 runner。**不含**训练循环/评测 runner（Wave 2）。**CPU only**。

## 0. 硬纪律

- 只创建/修改：`tfpd_exploration/src/fable_tkd_m2_v1/**`、`tfpd_exploration/scripts/run_fable_tkd_m2_stage0_v1.py`、`tfpd_exploration/tests/test_fable_tkd_m2_v1.py`、`tfpd_exploration/results/fable_tkd_m2_v1/stage0/**`（新目录）。不改任何既有文件；不碰 GPU（不 import 之外的 cuda 初始化；stage0 全 CPU）；不 git commit/push。
- 密封物只读：champion run dir、probe results、所有 0444 文件。
- conda env `spint`：`source /home/xinyuan/miniconda3/etc/profile.d/conda.sh && conda activate spint`，跑 python 时 `PYTHONNOUSERSITE=1`。
- Receipt 律：任何 stage0 产物 = 原子写（mkstemp+fsync+replace）→ `chmod 0444` → 同名 `.sha256` sidecar（内容 `<hex>  <name>\n`）。`results/fable_tkd_m2_v1/attempt.json` 必须在任何数据访问前先写（O_EXCL 语义：已存在则拒绝）。所有断言 fail-closed（raise，不 warn）。

## 1. 常量（plan.py）

- `SCHEMA="fable_tkd_m2_v1"`；`REPO_ROOT` = 脚本向上第 3 级 parent。
- REF 锚：`SEALED_M30_EXTERNAL=0.29521985196829853`（引用，不重算）；context：`M33_EXTERNAL=0.2991329172968798`、`MEANS_FILM_EXTERNAL=0.32102`。
- champion ckpt sha `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`；ts4 ckpt sha `e385e2f408d6b3fe65645be934a8f69b3ab4e68b986e56e64512611cac040399`（路径 `streaming_calibration_exp/outputs/streaming_calibration/e8_ts4_m2_submission_control_m33q33_v1_s42_20260801_162010/checkpoints/best.ckpt`）。
- 数据/面：`CHANNELS=96, WINDOW=50, OUT_DIM=2, ACTIVITY_HORIZON=30, RIDGE_LAMBDA=0.1, BEHAVIOR_SCALE=5.0`。
- 模型：`D_V=64, D_K=64, N_QUERIES=8, D_H=128, SSM_LAYERS=2, CONV_KERNEL=10, CONV_CHANNELS=16`。
- 训练（Wave 2 用，先定常量）：`EPOCHS=30, LR=3e-4, WD=0.0, BATCH=32, SEEDS=(42,43,44), SHUFFLE_SEED=42, PV_BETA=4.0`。
- held-in 7 session 名单与外部 6 session 名单（从 champion run dir 的 `split_manifest.json` 读一次后写死为常量并附来源注释）。
- T4 归一化 authority：首次构建时计算 held-in 7 session M30 raw T4 的 pooled per-column mean/std，连同各 session raw T4 的 sha256 写入 `results/fable_tkd_m2_v1/stage0/t4_authority.json`（0444）；此后每次运行重算并断言与该文件 bit 一致（数据或拟合律漂移即 fail）。

## 2. 数据（data.py）

镜像 `tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py` 的加载路径（先读懂它）：

1. `load_sessions()`：复用 `sua_exploration.evalai_t4_m2.export_t4_payload.load_frozen_model_and_data()`（它 sha-pin champion ckpt/normalizer 并给出同一预处理的数据集对象；模型部分 stage0 忽略/`torch.no_grad`+eval）。从中取：per-session `neural_data[T,96]`（float32 spike counts）、`covariate_data[T,2]`、`trial_change`、`calib_trial_target_angles`、`calib_trial_spike_sums`（注意：T4 拟合用 spike_sums/lengths 的 trial 均速率——与 screen `_ridge_side` 完全同律，先读该函数再写）。
2. `m30_t4(session)`：support = `_support_indices` 律（chronological arange(30)，≥3 有限方向试次）→ `fit_ridge_t4(trial_mean_rates[selected], angles[selected], normalized_lambda=0.1)` → raw [96,4] + per-unit ρ：
   `ρ_i = 1 − Σ_j (r_ij − ŷ_ij)² / Σ_j (r_ij − mean_j r_ij)²`，ŷ = design@coef（与 fit_ridge_t4 同一 design/coef；要求 ŷ 用 float64）。ρ clip 到 [0,1]。返回 (raw, rho[96], evidence)，raw/rho 各附 array sha256。
3. `normalize_t4(raw)`：`(raw − mean)/std`，mean/std 来自 §1 authority（float32）。产出 `t`。
4. `heldin_windows(session)`：window starts = 该 session 全部 window_indices 中 ≥ trial-30 起点者（镜像 `select_common_post30_window_starts`：`trial_change` 的 trial 起点 + ACTIVITY_HORIZON=30）；返回 (starts, neural_windows[num,50,96] float32, targets[num,2] = covariate[start+49]×5.0)。
5. `external_windows(session)`：`dataset.window_indices` 原样全部窗，targets 同律（不 ×5.0 存原始，预测侧统一 ÷5.0 后再比——与 screen `_score_session`/probe 评分一致，先读确认乘除位置）。
6. digest：每 session 的 (t4_raw, t, rho, starts, targets) 都过 `array_sha256`（复用 `m2_t4_activity_budget_screen_v1/core.py` 的实现或同义）进 receipt。

## 3. 模型（model.py）

`class TKD(nn.Module)`，构造参数（arm 开关）：
`epsilon: "learnable"|"zero"`、`key_mode: "identity"|"shuffled"|"pool"`、`time_model: "ssm"|"gru"`、`init: "pv"|"default"`、`frozen_readin: bool`（B 臂）。维度常量来自 plan。

### 3.1 前向（window 批模式，训练用）

输入 `x[B,50,96]`、`t[96,4]`、`rho[96]`：
1. `psi`：x 转置到 [B,96,50]，per-unit 共享因果 Conv1d(1→16,k=10,padding=9) → Linear(16→64) → `u[B,50,96,64]`。
2. value：`mu = W_mu @ t + b_mu`（W_mu:[1,4]，输出 [96] 标量）、`sig = softplus(W_sig @ t + b_sig)+1e-3`；`v = rho[:,None] * (u - mu)/sig`（广播 over B,t,d）。**μ,σ 是 t 的线性映射（可学习），init 精确恢复 raw b 列与 m 列**（closed form：由 normalizer 的 mean/std 反解，W 列向量 = std_col/…，具体：raw_b = t[:,3]*std3+mean3 ⇒ mu_i = t_i[:,3]*std3 + mean3，即 W_mu=[0,0,0,std3], b_mu=mean3；sig_i = t_i[:,2]*std2 + mean2 ⇒ W_sig=[0,0,std2,0], b_sig=mean2 − softplus^{-1}… 注意 sig 走 softplus，init 时直接把 (W_sig,b_sig) 设为使 softplus(...)≡ raw m：令 pre = log(exp(m)−1) per-unit —— 但 pre 必须是 t 的仿射函数：pre_i = log(exp(raw_m_i)−1)，而 raw_m_i 是 t_i[:,2] 的仿射 ⇒ 需要仿射=log∘(exp−1)∘仿射，非线性不可行。**修正：sig 不用 softplus，用 `sig = W_sig@t + b_sig` 直接线性并 clamp min 1e-3**（init 精确 = raw m，仿射可行）。记此偏差：value 归一化为线性（非 softplus），写进 stage0 receipt 的 deviations 字段。）
3. key：`k = Phi_k(t)`（Linear(4→64)→ReLU→Linear(64→64)），[96,64]；`epsilon_mode=="learnable"` 时 `k_eff[B,50,96,64] = k + eps * (u @ P.T)`（eps 零初始化标量，P:[64,64]）；`"zero"` 时 k_eff = k 静态（实现里 k_eff = k broadcast）。`key_mode=="shuffled"`：对 t 的行施加固定置换 perm（见 §3.4）**再进 Phi_k 与 μ/σ**（与 champion ts4 "side features 整行置换" 同律——identity 消费全部走置换后的 t；rho 不置换）。`"pool"`：t 替换为 broadcast 单位均值 t̄（[96,4] 每行同），再走同一前向。
4. attention：`q[8,64]` 可学习；`alpha[B,50,8,96] = softmax_i(q·k_eff/8.0)`；`z_l[B,50,8,64] = Σ_i alpha v`；`z = merge(concat_l) → [B,50,128]`。
5. 时间模型：SSM 2 层 + GLU + 残差（见 §3.2）或 GRU(128)（`time_model=="gru"`，单层双向禁止——单向）。
6. `y = readout(h)[:, -1, :] / BEHAVIOR_SCALE`（只出末 timestep）。

### 3.2 SSM（对角线性，S4D/LRU 型）

每层：`h_t = diag(a) ⊙ h_{t-1} + B z_t`（a=exp(−softplus(a_log)) ∈ (0,1)，[128]；B:[128,128]），`o_t = GLU(C h_t)`（C:[128,128]，GLU = Linear(128→256) 取半门控），残差 `z_{t+1} = z_t + o_t`。
- 并行（训练）：kernel 法——每通道 kernel 长度 50：`K_c[n] = C_c ⊙ a_c^n ⊙ B_c`…（实现按标准 S4D-lite：`o = (C * kernels) @ z`，用 torch.cumprod 预计算 [128,50] kernel，等价于 conv1d(depthwise)）。允许最简实现：直接用 `torch.einsum` 显式 scan（seq len 50，代价可接受）——**但必须同时提供 step 递推函数** `ssm_step(h, z_t)` 并过 A2 parity。
- **init="pv" 时整段 SSM init 为恒等直通**：a_log 使 a=0（h_t=z_t 的纯记忆丢失……注意 a=0 ⇒ h_t = B z_t，取 B=I 则 h_t=z_t）、C=I、GLU 门 init 全开（gate sigmoid 权量≈大正数偏置使 gate→1，另一支线性支 init 复制输入）→ o_t = z_t，残差改为 **init 时 scale 0**（残差系数 γ 零初始化，可学习）：`z_out = z + γ·o`，γ=0 ⇒ 恒等。这样 init 时整网络 = 静态 PV 分箱函数，A1 才有意义。
- GRU 分支 init 不要求 PV 恒等（A′-GRU 不进 A1）。

### 3.3 PV 锚初始化（init="pv"）

- ψ conv：channel0 kernel = 全 1/10；channel1–15 = 0；Linear(16→64)：row0 权重 [1,0,…]，bias 0 ⇒ u[...,0] = rate10。其余 row 全 0（u 其余维 = 0，训练再学）。rate10 = 因果 10-bin 均值（与 conv 一致）。
- μ/σ init = raw b / raw m（§3.1 修正律）。
- `Phi_k` init：目标 `q_ℓ · k_i ≈ β·cos(∠(PD_i, ψ_ℓ))`，PD_i=[a_i,c_i]/m_i，ψ_ℓ=8 等间隔方向。实现：设 Phi_k 的复合映射为 t→[a,c] 子空间的精确仿射（构造法：令最后一层 Linear(64→64) 的权重把 ReLU 输出的一个基方向映到方向编码 d_i = [cos φ_i, sin φ_i, 0…]，具体做法自定，**必须给出 closed-form 构造并在 A3 单测中断言 `q_ℓ·k_i / (β) ≈ cos(∠)` 最大误差 < 1e-4**；ReLU 支路置为正线性不拦截该方向——允许把 Phi_k 简化为 t 的直接线性（跳过 64 隐层、隐层旁路）以保证精确仿射，参数仍在、训练可用；此简化记入 deviations）。
- q init：q_ℓ = 单位方向嵌入。softmax 温度 β=PV_BETA=4.0 并入 k 的尺度。
- merge init：`z_ℓ` 第 0 坐标 ≈ (1/M_ℓ)Σ_{i∈binℓ} ρ_i x_i（x_i=(rate10−b)/m，M_ℓ=该 session bin 内单位数）；merge 把各 bin 第 0 坐标乘 `[cos ψ_ℓ, sin ψ_ℓ]·M̄_ℓ`（M̄_ℓ 用 held-in 池化的期望 bin 质量，全局常数）累加到输出 2 维的对应通道，其余通道 0；readout Linear(128→2) init 把该 2 通道恒等映出、其余 0。GLU/SSM γ=0 直通。
- 显式 PV 参照（A1 用，独立实现于 stage0.py，不经 TKD；ADDENDUM-1 修订版）：`y_PV = Σ_i ρ_i·(rate10_i − b_i)·[a_i, c_i]`（raw a,c,b；rate10 同一窗同一因果 10-bin 均值；无除法）。

### 3.4 SHUF 置换

`perm = np.random.default_rng(SHUFFLE_SEED).permutation(96)`，断言非恒等；模型内对输入 t 施加 `t[perm]`（等价实现任选，A3 断言两种写法一致）。构造一次进 plan 常量（写死数组）。

## 4. Stage 0（stage0.py + runner）

runner `tfpd_exploration/scripts/run_fable_tkd_m2_stage0_v1.py`：`--execute` 才跑（默认 dry 打印计划）；流程：attempt.json → 依次 A1–A6 → `stage0/terminal.json`。每项一个 0444 receipt（含耗时、digests、断言结果）。任何失败：写 `failure.json` 并 raise。

- **A1 PV 锚**（参照经 ADDENDUM-1 修订）：前 2 个 held-in session（chronological），各取前 2048 个 post-30 窗；`TKD(epsilon="zero", key_mode="identity", time_model="ssm", init="pv")` eval 模式 CPU float32；预测 [n,2]（÷5.0 后）vs **经典深度加权 PV 参照** `y_PV = Σ_i ρ_i·(rate10_i − b_i)·[a_i, c_i]`（raw a,c,b；rate10 同一窗同一因果 10-bin 均值；无除法、无 m 归一化——原双重 1/m 参照作废，见工单 ADDENDUM-1）；Pearson（flatten 两列）per session；报告 min。判据：>0.99 pass；[0.90,0.99) pass_with_disclosure（receipt 里 `disclosed=True` + 数值）；<0.90 fail。额外报告 per-dim Pearson 与 PV 自身方差。
- **A2 SSM parity**：随机 z[3,2000,128]（seed 42），随机 SSM 层参数（合法范围），parallel scan vs 逐步 `ssm_step`，max|Δ| ≤ 1e-6。GRU 分支同测（nn.GRU 前向 vs 手写单步，1e-6）。
- **A3 构造单测**（也在 tests/ 里重复）：SHUF perm 非恒等、逐次构造一致；POOL 下 Phi_k(t̄) 行行相同且 A′ 参数量 == A/SHUF/POOL 参数量（`sum(p.numel())` 相等断言）；ε=0 时静态 key 缓存前向 == 逐窗重算前向 bit-exact（同一 batch 两次 forward `torch.equal`）；key cos 逼近断言（§3.3）。
- **A4 MAC 计数**：解析式实现，输出 JSON：TKD 流式 per-bin（ψ 每 unit 一次 conv 步 + Linear、静态 key 0、attention 2·8·96·64、merge、SSM step、readout）、TKD 整窗 per-window（同量 ×50 + key 一次）、champion per-window decode（读 champion run dir 的 `hardware_cost.json` 与 resolved_config 推 SpintModel+B3S decode MAC；E 已缓存（act30 部署律）——列公式与假设进 receipt）。报告 TKD 流式每窗等效（per-bin×50）与 champion 比值。
- **A5 t 偏移**：外部 6 session 的归一化 t 各列 (mean, std) vs held-in 池化 (0,1) 的偏离表。
- **A6 ρ 分布**：13 session ρ 的 (min/median/max/mean) 表 + 直方图数据（20 bins）。

## 5. tests

`tfpd_exploration/tests/test_fable_tkd_m2_v1.py`：A2/A3 的可离线部分（不加载 NWB 的：SSM parity、SHUF/POOL、ε 缓存、key cos、参数量、μ/σ init 反解 raw 列 bit 一致）；命名 `test_fable_tkd_*`；`cd /home/xinyuan/Work_host/SPINT && python -m pytest tfpd_exploration/tests/test_fable_tkd_m2_v1.py -q` 必须全绿（repo 根为 CWD，sys.path 加 REPO_ROOT 与 tfpd_exploration/src——仿照现有 tests 的路径注入方式，先看一个现有测试文件怎么写）。

## 6. 验收（agent 自验后报告）

1. tests 全绿（贴输出）。
2. `--execute` 跑完 stage0：A1 min corr 数值、A4 比值、A5/A6 摘要、terminal.json 路径（贴关键 JSON 字段）。
3. 若 A1 ∈ [0.90, 0.99)：**不要自行调参重跑超过 2 轮**；报告数值与 per-session 明细即停（工单 §9 规则）。
4. deviations 列表：实现与本 spec 的每一处偏差（如 §3.1 softplus→linear、§3.3 Phi_k 旁路）。
