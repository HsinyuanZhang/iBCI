# 计划：688 与 M2 的 carrier 侧迭代实验（2026-09-09）

> **ADDENDUM-VSTATE4-EXT6-VERDICT（2026-09-09 终局）**：vstate4 官方面 ext6 earliest-max pick = **e4 0.3502**，vs 基线 e9 0.3901 = **−0.0399**，推荐门 0.4201 未过 → **CANDIDATE_NOT_RECOMMENDED**，未打包。ext4 +0.037 未迁移到官方面——判读：小校准支持面上的增益疑似**面选择过拟合**；官方面上 e4 早峰后持续衰减的形态佐证。逐 session：6 session 中 5 输（11-24-R1 −0.089 最差、11-18 +0.020 唯一正）。**harness 忠实性已反证排除工具误差**（基线 e9 复算逐字节复现 0.39006 + 六 session prediction digest 全匹配）。证据链 sealed：ext6 bank `f0dbcc66…`、score `b032192a…`、audit `3cb9f970…`、replay `26fdbb37…`。
> **对 688 vstate 计划的含义**：M2 的 ext4→ext6 反转是"小面增益不可信"的教训——688 的 vstate 臂以 exp1_narrow exam 为主判面不变；M2 的 dense-标签先验由 H1 官方 +0.054 单独支撑，M2 证据降级为"ext4 上正、官方面未复现"。
状态：**执行中（2026-09-09 17:00 CST）**。688 仅 `exp1_narrow`；全程单 GPU（`cuda:0` / `CUDA_VISIBLE_DEVICES=0`）。M2 cache 被本机缺失的 champion ckpt 挡住，见文末「执行笔记」。上位分析：`CARRIER_UNIFIED_STATE_RESPONSE_M2_688_20260909.md`。

## 0. 结论先行

每个数据集两臂，外加必要的配对对照。decoder 配方、seed、数据面全部不动，只换 carrier（T）及由 T 派生的 E0。

| 数据集 | 臂 | 类型 | 一句话 | 期望 |
|---|---|---|---|---|
| 688 | `t4` | 对照 | 冻结 `[a_R,c_R,m_R,δ_b]`，M10 | 首个 688 RIFT 读数（此前没有） |
| 688 | `u1_m10` | 轻微优化 / 非劣 | 同标签同窗口，改用统一模板（Poisson 标准化 + 伪计数收缩 + 谐波读出） | ≥ t4 − 0.01 |
| 688 | `u1_m30` | 大变化 / 高期望 | 同上，但方向标签用全部 30 个 activity-support trial | ≥ t4 + 0.03 |
| M2 | 基线 | 已有 | `m2_r50_concat_s42_formal_v1`（ext4 逐 epoch 曲线；ext6 e9 0.390） | 复用，不重跑 |
| M2 | `u1` | 轻微优化 / 非劣 | M33、bins `[5,30)` 不变，统一模板重算 | ≥ 基线 − 0.01 |
| M2 | `vstate4` | 大变化 / 高期望 | H1 signed-state 同族：全部 33 个校准 trial（含回中心 trial）、100 ms 块、softplus(±v) 软状态 | ≥ 基线 + 0.03 |

时间估算：M2 R50 concat 实测 24 epoch ≈ 30 min（3165 upd/ep，~44 upd/s）；688 `exp1_narrow` 约 5.9k upd/ep × 12 ≈ 71k upd，同量级模型 ≈ 30–45 min/臂。两张 GPU 并行，全部 5 个训练在 **~3 小时** 内跑完（不含缓存构建 ~30–40 min）。

## 1. 统一模板（两数据集共用一份实现）

新模块 `btransform_unified_v2/src/btransform_unified_v2/carrier_profile_v3.py`：

1. `poisson_standardize(rates, duration_s)`：`z_tu = (r_tu − r̄_u)/σ_u`，`σ_u = sqrt(max(r̄_u·Δ, 1))/Δ`（与 H1 `h1_profiles._unit_standardize` 同式）。
2. `conditional_response(z, W, n0)`：`R_uk = Σ_t w_tk z_tu / (Σ_t w_tk + n0)`。
3. `harmonic_readout(R_dir, θ_k)`：`a = 2·mean_k cosθ_k R_uk`，`c = 2·mean_k sinθ_k R_uk`，`m = hypot(a,c)`；K=8 固定，缺失方向 `R_uk = 0`。
4. `signed_state_weights(v, rms)`：`[softplus(v/rms), softplus(−v/rms)]`。
5. `fit_column_normalizer(rows)` / `apply`：列 mean/std，std 地板 1e-6。

回归门（构建时自动断言）：`n0=0`、不标准化、one-hot 权重时，688 输出与 `descriptors._fit_t4` 在 8 方向齐全的 session 上逐值相等；M2 与 `t4_from_trial_sums` 的差只来自方向不平衡，逐 session 报最大绝对差。

## 2. 688 臂定义

协议：bench `exp1_narrow`（9 train：2015-06-29→07-10；exam：0713/0714/0715/0716），RIFT `proj_add`，12 epoch，e8–e11 平均，seed 42，batch 32，AdamW（rift_v1 同款）。判读面：exam 4-session equal-session mean R²。

### 2.1 `t4`（对照）
冻结 cache `dandi688_prepared_cache_contract_v2` 原样。必须跑：688 上还没有任何 RIFT 训练读数，非劣/增益都需要它做配对基线。

### 2.2 `u1_m10`（轻微优化，非劣）
- 标签：与 `t4` 完全相同的前 10 个 rewarded trial 的 `target_dir / target_on / go_cue`；窗口 R700、H300 不变；率原语 `_pool_trial_rate_matrix` 不变。
- 变化仅在估计器：R700 行 Poisson 标准化（Δ=0.7 s，同一 affine 用于 H300 行）；8 方向 one-hot 条件响应，`n0 = 1` 伪 trial/方向；hold 状态 `R_hold`（`n0 = 1`）；读出 `[a, c, m, b]`，`b = mean_k R_uk − R_hold`（即标准化单位下的 δ_b）。
- normalizer：27 个 train session 列 mean/std（与现行 `fit_source_normalizer` 同规则）。
- E0：用冻结 B3S student `post_pool(cat(pre_pool(calib_trials).mean, carrier_norm))` 重算（`dandi688_train.build_rows` 同式），与 H1 582196 "只换 T、重熔 E0" 同策略。
- 期望：与 `t4` 相当；差别来自收缩与标准化，不来自信息量。

### 2.3 `u1_m30`（大变化，高期望）
- 与 `u1_m10` 唯一差别：方向/事件标签取 **前 30 个** rewarded trial（`_phase_trials(..., support_positions=range(30), namespace="reliability_audit")`）。activity/E0 支持本来就是 M30，query 仍 Q50，**不新开任何 trial，只用其已有标签**。
- 依据：688 上最大的已知 carrier 侧杠杆——历史 T4@M10→@M30 差 +0.054（SUA，15 session×3 seed，非 matched）；profile split-half 0.78→0.92。`u1` 的收缩在 M30 下约 21%（M10 下约 50%），估计更接近无偏。
- 披露：这是标签预算变化（M10→M30），报告里必须与 `u1_m10` 并列，不能写成"估计器改进"。

### 2.4 否决的大改候选
- dense-velocity signed-state：688 生产路径禁止读 dense velocity；dense-speed CP-FiLM 已是 null。
- 二次谐波 6 维：改 `carrier_dim`，等于改模型，不属于 carrier 侧。

### 2.5 实现方式
- 新脚本 `dandi688_bench_v1/scripts/build_carrier_cache.py --variant {u1_m10,u1_m30} --dest dandi688_bench_v1/results/cache_<variant>`：读冻结 cache 的 33 行（neural/behavior/starts/mask 逐字节复制），`prepare_source_surface` 取 NWB 路径与 `calib_trials`，重算 carrier 与 E0，写同 schema 的新 `prepared_contract.json`（`rows_hashes` 重算，附 `variant`、估计器参数、源 cache SHA、实现文件 SHA）。
- bench runner 以 `--arm t4 --prepared-cache <variant cache>` 训练；receipt 内的 `prepared_contract_sha256` 与 `arm_carrier_sha256` 区分各臂。
- bench 的 `--stage train` 目前是 `GPU_FORBIDDEN` 占位。需要一次日期化协议修订：`plan.GPU_POLICY` 翻转、`stage_train` 复刻 rift_v1 循环（batches/unit dropout/AdamW/逐 epoch ckpt/e8–e11 平均/receipt seal）、`stage_score` 允许 GPU。**该代码改动已派给子代理实施（纯 CPU 测试，不启动训练）**，可用 `git checkout -- btransform_unified_v2/dandi688_bench_v1` 整体回退。

### 2.6 判读
- 主判读：exam equal-session mean R²（e8–e11 平均权重），三臂并列，附 4 session 各自 R²。
- `u1_m10`：≥ `t4` − 0.01 记非劣；`u1_m30`：≥ `t4` + 0.03 记增益，0–0.03 记"方向正但未过门"。
- 预注册门（T4−F0、T4−TS4）本轮不跑。`t4` 之后固定补两臂消融（对齐 M1/M2/H1）：`f0` = activity-only（E0 保留、carrier 置零）；`z0` = query-only（E0 与 carrier 都置零）。二者用冻结 cache，不进门，只报 exam 读数。
- `exp1_narrow` 的 exam session ⊂ `exp2_full` 的 train 集，本轮结果只在 EXP-1 内解释。

## 3. M2 臂定义

配方：`m2_concat_train.py`（RIFT R50 D4 concat，24 epoch，冻结 batch manifest，seed 42，EMA），ext4 逐 epoch 打分。基线 `results/rift_v1/m2_r50_concat_s42_formal_v1`（ext4 selected e7；ext6 pick e9 = 0.3901）。判读面：**ext4**（配对、同配方），ext6 pick 只对过门候选补做。

### 3.1 `u1`（轻微优化，非劣）
- 支持：M33 同 trial；率：bins `[5,30)` 的 trial 计数（`champion.move_t4_trial_sums`，不变）。
- 估计器：trial 计数 Poisson 标准化；目标角映射到最近的 8 个 canonical 方向后 one-hot，`n0 = 1`；中心目标（角为 NaN）与现行一样不进方向拟合；读出 `[a, c, m, b]`，`b = r̄_u/σ_u`（方差稳定化的基线增益）。
- normalizer：7 个 held-in session 列 mean/std（`fit_source_move_normalizer` 同规则，独立拟合）。
- E0：`champion.native_e0_and_u(encoder, calib_activity, empty_contrast_side(T_new))` 重算（与 `stage0_data` 同路径，encoder 为 `load_frozen_champion().student.id_encoder`）。
- 期望：与基线相当。

### 3.2 `vstate4`（大变化，高期望）
- 支持：M33 **全部** 校准 trial，包括当前被丢弃的回中心 trial（约一半）。
- 行：trial 内不重叠的 100 ms 块（5 bin）；率 = 块计数，速度 = 块内 `calib_covariates` 均值（与 `calib_neural` 同 eval-mask 时间轴，构建时逐点核对对齐）。
- 状态权重：`[softplus(v_x/rms_x), softplus(−v_x/rms_x), softplus(v_y/rms_y), softplus(−v_y/rms_y)]`，rms 只用 7 个 held-in session 的校准速度拟合。
- 估计：块计数 Poisson 标准化；`R = conditional_response(z, W, n0 = 10 块)` → `[96,4]`；读出 `a = R_{+x} − R_{−x}`，`c = R_{+y} − R_{−y}`，`m = hypot(a,c)`，`b = mean_k R_uk`（运动-静止调制，对应 688 的 δ_b 语义）。
- normalizer：7 held-in 列 mean/std；E0 重算同 §3.1。
- 依据：与 H1 582196 的估计器同族（H1：7 轴 ×2 = 14 状态；M2：2 轴 ×2 = 4 状态，无需 SVD，谐波读出闭式）；有效支持翻倍；权重跟随真实运动而非目标方向。M2 历史上的 dense-profile null（REAL < EMPTY）是无符号速度分位数 profile **叠加**在完整 T4 之上，不是有符号方向状态**替换** T4，不构成反证。
- 披露：消费校准 trial 的 dense 速度标签（FALCON 允许的 support 标签）；这与旧文档中 688 的"sparse-label"纪律不是同一个问题。

### 3.3 实现方式
- 新目录 `btransform_unified_v2/scripts/carrier_v3_m2/`：
  - `build_m2_variant_cache.py --variant {u1,vstate4} --dest results/carrier_v3_m2/run_<variant>`：`construct_source_datamodule()` + `_build_ext4_dataset()` 取 11 个 session 的 `calib_neural / calib_trial_change / calib_trial_target_angles / calib_covariates / calib_trialized_neural_features`；生成 `cache/{source_train,source_minival,ext4}/<session>/`，其中 `X_store.npy / target_store.npy / eligible_starts.npy / calib_activity.npy / mapping.json / extra.json` 以符号链接指向原 cache（字节不变），`T.npy / e0_u.pt / provenance.json` 新写；写 `cache/carrier_normalizer.json`（含 rms、n0、estimator、源 cache SHA）。
  - `train_m2_variant.py --run-root <dest> --dest results/carrier_v3_m2/train_<variant> --device cuda:1`：在导入前把 `old_plan.ACTIVE_RUN_RELATIVE` 与 `adapters._M2_CACHE_ROOT` 指到变体 run root，然后调用 `m2_concat_train.run_train` 与 `score_stage`（ext4 曲线）。trainer 的 `frozen_cache_hashes` 会记录新 T/E0 SHA。
- 不改 `m2_concat_train.py`、`data.py`、`adapters.py` 任一字节。

### 3.4 判读
- 主判读：ext4 equal-session R²，两种端点并列——固定 epoch（基线 selected e7 与 ext6 用的 e9）与各自独立 selected（earliest-max EMA）。
- `u1`：selected ≥ 基线 selected − 0.01 记非劣；`vstate4`：≥ +0.03 记增益。
- 过门候选再补 ext6 pick（需用新 T/E0 重建 6-session ext6 query bank，约 30 min，不在当晚关键路径）。**不提交 EvalAI**，除非用户另行批准。

## 4. 不变项

- 官方 27-tag H1 bank、13-tag M2 bank、688 `prepared_cache_contract_v2`、`m2_dual_track_v1/20260905_101500/cache` 全部字节不变；变体只写新目录。
- 不动 `scripts/rift_v1/`、`src/btransform_unified_v2/model.py|concat_model.py`；新模块只新增文件。
- 不开 688 formal-test；不用 last-date；不触碰 EvalAI。
- 每个训练按规则挂 10 分钟 watch。

## 5. 风险与解释边界

1. Poisson 标准化改变第 4 列语义（688：δ_b → 标准化 δ_b；M2：绝对基线 → 增益）。若 `u1` 掉分，先查第 4 列，回退方案是保留原始率的 `b`。
2. T 与 E0 同变（沿用 H1 582196 做法）。若某臂为正，需一个 "只换 T、E0 原生" 的补臂才能把增益归到 direct token；本轮先不排。
3. `u1_m30` 是标签预算变化，任何表格必须标明 M30。
4. `vstate4` 的 E0 重算把新 T 送进为 MOVE-T4 训练的冻结 FiLM encoder；列语义与尺度（z-score 4 维）相近，H1 有先例，但属于分布外输入，构建时记录 E0 与原生 E0 的相对 Frobenius 距离。
5. 单 seed 42；结果是配对开发面读数，不是显著性。

## 6. 执行顺序（批准后）

1. 完成 bench `--stage train` 修订（进行中，CPU 测试）；写 `carrier_profile_v3.py`；写两个 builder 与 M2 launcher。≈ 60–90 min。
2. 构建 688 两个变体 cache、M2 两个变体 run root；每个跑 CPU preflight / smoke（bench preflight；M2 `--max-updates-smoke 20`）。≈ 30–40 min。
3. GPU0：688 `t4` → `u1_m10` → `u1_m30`（顺序，每个 ~30–45 min，含 score）。GPU1：M2 `u1` → `vstate4`（每个 ~40 min，含 ext4 曲线）。
4. 10 分钟 watch；结束后每数据集一张表（臂 × 端点 × 各 session），写入本文件末尾的"结果"节。
5. 过门候选：M2 补 ext6 pick；688 视时间补 `f0`。

## 7. 执行笔记（2026-09-09）

用户修订：688 只跑 `exp1_narrow`；只用一张 GPU（`cuda:0`）。顺序改为 688 三臂跑完再跑 M2。

已启动：
- GPU 训练波：`run_wave_after_t4.sh`（`t4` 打分 → `f0` → `z0` → `u1_m10` → `u1_m30` → M2 若 cache 齐）。
- 688 变体 cache：`build_carrier_cache.py --variant all`（CPU）。
- 10 分钟 watch：`AGENT_LOOP_TICK_train_watch`。

M2 cache 首次构建失败：本机没有
`streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt`
（SHA `25d7bc72…`）。E0 重熔不能进行；不回退成「只换 T、沿用原生 E0」。688 不受影响。找到 ckpt 后重跑 M2 builder。
