# M2 / DANDI688-like 数据集的 carrier 统一构建梳理（2026-09-09）

状态：分析与建议文档。不改任何已冻结 bank、prepared cache、官方 payload。所有"建议"项在未过预注册验证前不进入生产。

输入证据：H1 `signed_state14`（official 582196）对 H-C（official 582073）的对照；M2 MOVE-T4 全部消融；688 sparse-event profile 设计/Stage-0/Stage-1 结果；M1 carrier refinement 与 carrier v2/v3。代码与数字来源见各节路径。

## 0. 结论

1. **M2 的 `[a,c,m,b]`、688 的 `[a_R,c_R,m_R,δ_b]`、H1 的 `signed_state14`、M1 的 `muscle_response16` 在数学上是同一个对象**：按行为状态加权的每单元条件响应均值，经 source 固定的 4 维基压缩，再做 source 归一化。M2/688 对 `[1,cosθ,sinθ]` 做最小二乘，等价于"8 个方向条件均值取一阶谐波"，是这个模板在圆形行为空间上的解析特例；688 的 `δ_b` 只是多加一个 hold 状态列。这一点使 M2 与 688 的讲述可以收成一条公式加一张参数表，无需改任何字节。
2. **H1 新证据能迁移到 M2/688 的不是"signed-state"这个名字，而是模板里的三个部件**：单元内 Poisson 地板标准化、occupancy 伪计数收缩、source 端逐列归一化。H-C 输给 signed-state 的根本差别是"群体读出权重"对"单元编码侧条件响应"；M2/688 本来就在编码侧，所以 H1 的胜利首先是对 M2/688 现行路线的确认，而不是要求换估计器。
3. **不应迁移的**：adaptive v3 的 Wiener/SNR 压缩（H1 outer fixed +0.009、selected −0.001，M1 反向）；dense velocity 软状态（688 生产禁止读 dense velocity，且 dense-speed CP-FiLM 已是 null）。方向 one-hot 本身就是圆形任务上的 ± 状态划分。
4. **"更高效"的实际含义**是把三套独立实现（`falcon_t4_features.py`、`unit_side_features._fit_cosine_tuning`、`h1_profiles.py`）收成一个 `conditional_profile(Z, W, n0) → basis → normalizer` 函数，把 rank-3 硬失败、D-opt 选 trial、Stage-0 列掩码、逐臂独立 normalizer 这些围绕"低预算下估计不稳"的补丁，替换成一个伪计数收缩参数 `n0`。需要预注册验证的只有两项：单元内标准化是否伤 M2/688；M2 carrier 支持从 M33 降到 M10 是否可行。

## 1. 现状对照

| 维度 | M2 MOVE-T4 | 688 sparse-event profile | H1 signed_state14（582196） | H1 H-C（582073） | M1 rSyn3（生产） |
|---|---|---|---|---|---|
| 行为标签 | trial target 角 `atan2(tgt−(0.5,0.5))`，中心目标 NaN | trial target 方向，映射到 8 个 canonical 方向 | 7 维速度（3 平移 + 1 旋转 + 3 grasp），100 ms 块均值 | 同左 | 整流 EMG（16 路），运动窗 |
| 神经率原语 | 原始 20 ms bin `[5,30)`（trial 起 100–600 ms）求和 /25，每 trial 一行 | spike time `searchsorted` 半开区间计数 / 时长；R700=`[go, go+0.7)`，H300=`[target_on−0.3, target_on)` | 100 ms 块 spike 和 /0.1 s | 同左 | 20 ms bin，运动窗内 |
| 单元标准化 | 无（原始率进回归） | 无 | **有**：`(r−mean)/ (sqrt(max(mean·0.1,1))/0.1)` | source 13 session first-4 trial 的 per-unit z-score | 无 |
| 状态/设计 | `[1,cosθ,sinθ]`，Ω=I，每 trial 一行 | 先按方向取均值再 `[1,cosθ,sinθ]`（等价 Ω=1/n_θ） | `W=[softplus(v/rms), softplus(−v/rms)]`，14 列软占用 | source 神经 PCA q=12 → ridge λ=10 读出到 7 维速度 → 系数投回单元 | `[1, syn3]`，syn3 = source NNMF-3 |
| 条件响应 / 收缩 | lstsq，要求 rank 3，否则失败 | lstsq，要求 rank 3；Stage-0 列可靠性掩码（当前全真） | `R_uk = Σ_t w_tk z_tu / (Σ_t w_tk + 10)`，occupancy 收缩到 0 | EB 收缩 `τ²/(τ²+v_i)` 向 μ | ridge λ=1 |
| 4 维压缩 | 解析：`[a, c, √(a²+c²), b]` | 解析：`[a_R, c_R, m_R, b_R−b_H]` | source 池化 SVD `[14,4]`，轴无物理标签 | 右奇异基 U `[7,4]` | `[w1,w2,w3,b]` |
| source 归一化 | 7 个 held-in session 行的列 mean/std | 27 个 train session 行的列 mean/std（无 winsor；父 M30 whole-trial T4 才有 1%/99% clip） | 池化投影后列 mean/RMS | 单标量 RMS `6.826e−6` | 列 mean/std |
| 支持预算 | **M33** | **M10** 标签（E0 用 M30；query Q50） | **M3**（前 3 个公开 trial） | M3 | M10 |
| 进模型 | direct token（concat 或 proj_add）+ 已进 E0（B3S side path） | direct token + E0 | direct token + C2 E0 重熔 | 同左 | 同左 |
| 代码 | `streaming_calibration_exp/src/data/falcon_t4_features.py` L76–117；`tfpd_exploration/src/m2_dual_track_v1/champion.py` L360–454 | `sua_exploration/mc_maze/unit_side_features.py` L728–752, L970–997；`mc_maze/dandi688_sparse_event_t4_v1/descriptors.py` L133–196, `core.py` L49–53 | `btransform_unified_v2/scripts/carrier_profile_v2/h1_profiles.py` L143–251 | `SPINT-main/src/data/h1_m4_eb_pilot.py` L395–521 | `tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py` |

M2 与 688 的差异只有 5 个协议参数：窗口锚点（trial 起 vs go-cue / target_on）、率原语（bin 和 vs spike-time 计数）、第 4 列（运动窗均值 vs 运动−hold）、支持预算（M33 vs M10）、normalizer 名单（7 held-in vs 27 train）。估计器、列语义、方向基完全相同。

## 2. 统一模板

对任一数据集，permitted support 给出行索引 `t`（M2/688 = 一个 trial×窗口；H1 = 一个 100 ms 块），单元 `u`，行为状态 `k=1..K`。

1. **率标准化**（单元内、support 内）：`z_tu = (r_tu − r̄_u) / σ_u`，`σ_u = sqrt(max(r̄_u·Δ, 1))/Δ`（Poisson 地板，Δ 为行时长）。
2. **状态权重** `W ∈ R^{T×K}`：
   - M2：`w_tk = 1[θ_t = θ_k]`，K=8，窗口 `[5,30)`。
   - 688：`w_tk = 1[θ_t = θ_k]·1[窗=R700]`，K=8，再加一列 `w_{t,hold} = 1[窗=H300]`，K=9。
   - H1：`w_t,·= [softplus(v_t/rms), softplus(−v_t/rms)]`，K=14。
   - M1：`w_tk = max(EMG_tk,0)/rms_k`，K=16。
3. **条件响应**：`R_uk = Σ_t w_tk z_tu / (Σ_t w_tk + n0)`。`n0` 是伪占用（H1 用 10 个 100 ms 块）。状态缺失时 `R_uk → 0`，不再报 rank 错误。
4. **4 维基** `P`（source 固定）：
   - 圆形状态空间（M2/688）：解析一阶谐波 `a = (2/K)Σ_k cosθ_k R_uk`，`c = (2/K)Σ_k sinθ_k R_uk`，`m = √(a²+c²)`，第 4 列 `b`（M2：`Σ_k R_uk / K`；688：`Σ_k R_uk/K − R_u,hold`）。方向平衡时与现行 lstsq 逐值相等；688 已按方向均值拟合，8 个方向齐全时恒等，缺方向时现行 lstsq 用剩余方向外推，模板则把缺失方向的 `R_uk` 收缩到 0——这正是两者在低预算下的差别所在。
   - 非圆形（H1/M1）：source 池化 SVD 取前 4 右奇异向量。
5. **source 归一化**：`c_u = (P(R_u) − μ_src) / s_src`，逐列。
6. **对照臂**：F0 = 零；TS4 = session 内行置换；support resample。

模板下的讲述：H1 的 14 列是 7 条轴上的 ± 条件响应；M2/688 的 `a, c` 是 x、y 两条轴上的 ± 对比（`cos` 权重 = +x 减 −x），`m` 是合成幅度，`b` 是基线。圆形任务的谐波基是这个 ± 划分的解析版本，所以 M2/688 不需要 dense velocity 就已经是 signed-state。

## 3. H1 新证据的可迁移性

| 证据 | 数值 | 对 M2/688 的含义 |
|---|---|---|
| signed_state14 vs H-C，local HO-M3（7 session 全正） | mean 0.3748 → 0.4677；worst 0.2059 → 0.2662 | 单元编码侧条件响应优于群体读出权重。M2/688 一直在编码侧，是确认，不是要求换估计器。 |
| official HO | 582073 0.403 → 582196 0.457（+0.054）；HI 0.668 → 0.650（−0.018） | 泛化到 held-out 更强，held-in 略降；说明该 profile 的增益在跨 session 稳定性，不在拟合精度。 |
| source-only first3–next3 池化 cosine | signed_state 0.62/0.55/0.38/0.65 vs H-C 0.32/0.45/0.35/0.60 | 可复现性提升与 decoder 提升同向。688 Stage-0 已有等价指标（split-half Fisher r 0.84–0.93），可直接沿用作 M2 的前置门。 |
| velocity7（线性 ridge 编码 → SVD4） | 池化 cosine 四次全弱于 H-C，未进 decoder | 直接对连续标签做线性回归不如"条件响应 + 收缩"。M2/688 现行 lstsq 恰好是条件均值的谐波形式，不是这类回归，所以不受此负面结论影响。 |
| adaptive v3 SNR-Wiener（H1 outer） | fixed e32 +0.009；source-selected −0.001；两 session 一正一负 | 4 维压缩方式的收益接近零。M2/688 的解析谐波基不需要 source 拟合，无需引入 Wiener。 |
| M1 muscle_response16（同模板） | inner +0.044，outer 09/28 −0.008 | 模板不保证任何数据集为正；M1 的 carrier 增量本身接近零或负（官方 T4−Original −0.004）。这是 M1 任务性质问题，不推翻模板。 |
| E0 与 T 同变 | 582196 重熔了 C2 E0 | H1 的 +0.054 未把 T 与 E0 的贡献隔离。M2 同样把 T 喂进 B3S E0 side path。任何 M2/688 的 carrier 变更必须同时说明 E0 是否重算。 |

## 4. 688 与 M2 各自证据对模板部件的支持

| 部件 | 688 证据 | M2 证据 |
|---|---|---|
| 方向列是主效应 | AC4 0.5628 ≈ T4 0.5750；MB4 0.3566；TS4 0.2845；B0 0.2364 | T4 vs TS4 +0.0956（旧 T4）；D vs unit-shuffle 0.3935 vs 0.2784 |
| 运动窗优于全 trial | POST700 vs whole 对独立参考 r(a) 0.78 vs 0.65；Stage-1 POST700−WHOLE +0.060 | MOVE `[5,30)` vs whole-trial +0.0263（6/6）；official 581899 vs 578221 +0.024 |
| 第 4 列（基线/δ_b） | δ_b 最可靠（0.934）但 SE-T4−PHASE-R +0.0001，decoder 增量为零 | b 对 688 成分格贡献很小；M2 未单独消融 |
| 低预算稳定性 | M10 需 Stage-0 掩码 + rank-3 检查；M10-to-M30 gap +0.054（历史，非 matched） | M8 重采样 26/40 有效（8 个缺方向、6 个 rank 不足）；whole-trial ridge M10/M30 官方 0.264/0.295 |
| dense 软状态 | dense-speed CP-FiLM：CP10−EMPTY −0.0035，null | FiLM profile REAL < EMPTY（0.3427 vs 0.3536），无语义 |
| 融合 | 待 bench（t4_concat 臂） | concat 0.4501 vs proj_add 0.4016（ext4 SPINT）；官方 concat 582047 0.390 |

两条线的结论一致：方向条件响应是全部有效内容；窗口选对比估计器精巧更重要；第 4 列可以按接口需要保留但不承担增益；低预算下的失败模式是设计矩阵缺秩，而不是信息不够。

## 5. 对 M2/688-like 数据集的建议

### 5.1 讲述层（可立即统一，不改字节）

- 一条公式 `c_u = N_src( P · [Σ_t w_tk z_tu / (Σ_t w_tk + n0)]_k )`，一张参数表（状态集、窗口、率原语、n0、支持预算、normalizer 名单、基）。M2 与 688 同行；H1/M1 只换状态集与基。
- 把 M2/688 描述为"方向条件响应的一阶谐波 profile"，688 的 `δ_b` 描述为"附加 hold 状态列的基线对比"，不再用"trig LSQ"与"sparse-event"两套词。
- 论文 §2 现有的 688/M2 小节已写明"等价于等权方向均值拟合"，只需把 H1 小节从 H-C 改为 signed-state 的条件响应形式，四个数据集即落入同一模板。M1 小节保留 rSyn3 但标注其在模板中的位置（EMG 通道权重 + ridge，无收缩）。

### 5.2 实现层（需预注册验证）

按收益/风险排序：

1. **单一实现**。新模块 `carrier_profile_v3/conditional_profile.py`：`standardize(rates, dt)`、`conditional_response(Z, W, n0)`、`harmonic_basis(theta_k)`、`svd_basis(pooled_raw, 4)`、`source_normalizer(rows)`。M2/688 只提供 `W` 与窗口。回归门：`n0=0`、不标准化、one-hot `W` 时，688 必须与 `descriptors.py` 输出逐值相等；M2 与 `falcon_t4_features.py` 的差只能来自方向不平衡，逐 session 报最大绝对差。
2. **伪占用收缩替代 rank-3 失败**。`n0` 建议以"每方向伪 trial 数"定义（M2/688 起点 1 个 trial；H1 是 10 个 100 ms 块 ≈ 1 s）。直接收益：M8/M4 重采样不再无效，D-opt 选 trial 不再需要；688 Stage-0 掩码退化为诊断。
3. **单元内 Poisson 标准化**。H1 有正证据；M2/688 无证据。风险：`b` 列在标准化后恒为 0，需改为保留未标准化的 `r̄_u/σ_u`（增益项）或接受 3 有效列。这是唯一可能伤 M2/688 的改动，必须作为独立臂测。
4. **支持预算统一到 M10**。688 已是 M10。M2 从 M33 降到 M10 会削弱 normalizer 与 E0 的输入；旧 whole-trial ridge 在 M10 输 0.031，但那没有窗口与收缩。只有在 1–3 通过后才值得测。官方 M2 T 保持 M33 不动。
5. **normalizer 名单规则统一**："decoder 训练用到的全部 source session"。M2 = 7 held-in，688 = 27 train，H1 = 13 held-in。删去 688 RIFT 契约中的"winsorizer"字样（sparse-event 路径实际无 winsor）。

### 5.3 验证方案（挂在 688 bench，不新开协议）

688 bench `exp1_narrow` 现有 `t4/f0/ts4/t4_concat`。增补两臂（不进现有门，只报读数）：

- `t4_u0`：模板重算，`n0=1`/方向，不标准化 → 检验收缩本身与冻结 `t4` 的差；
- `t4_u1`：`t4_u0` + 单元内 Poisson 标准化，4 列 `[a,c,m, r̄/σ]`。

判读面为 exam 4-session equal-session mean R²，与 `t4` 配对。通过条件：`t4_u1 − t4 ≥ −0.01`（非劣）即可采纳为统一实现；`> +0.03` 才算增益。M2 侧对应在 ext6 本地跑同两臂（RIFT concat 配方，seed 42，e9 pick），不提交 EvalAI。

### 5.4 不变项

- 官方 27-tag H1 bank、13-tag M2 bank、688 `prepared_cache_contract_v2` 全部字节不变。
- 688 生产不读 dense velocity；方向 one-hot 即 ± 状态划分。
- 不把 Wiener/SNR 压缩推到 M2/688。
- H1 的 held-in 下降（0.668→0.650）与 E0/T 未隔离两点在论文与后续实验里保持可见。
- last-date 0.564 不参与任何选择。

## 6. 引用的主要文件

- H1：`btransform_unified_v2/scripts/carrier_profile_v2/h1_profiles.py`；`btransform_unified_v2/scripts/h1_signed_state_r300_v1/build_banks.py`；`docs/CARRIER_PROFILE_V2_LAST_DATE_20260909.md`；`docs/CARRIER_ADAPTIVE_V3_METHOD_20260909.md`、`_RESULTS_20260909.md`；`docs/H1_FROZEN_CARRIER_METHOD_BINDING_20260908.md`；`results/h1_signed_state_r300_v1/recency_s42_formal_20260909/ho_m3_selection.json`；`tfpd_exploration/submissions/evalai_h1_rift_r300_signed_state_e16_v1/artifacts/OFFICIAL_582196.json`
- M2：`streaming_calibration_exp/src/data/falcon_t4_features.py`；`tfpd_exploration/src/m2_dual_track_v1/{plan,champion}.py`；`tfpd_exploration/docs/outdated/RESULT_ID_ENCODER_CONTENT_AND_CARRIER_CLOSURE_V1_20260905.md`；`btransform_unified_v2/docs/M2_SIX_ARM_MECHANISM_CONTROLS_20260908.md`；`btransform_unified_v2/results/chronological_last2_v1/`
- 688：`sua_exploration/docs/DESIGN_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md`（含 STAGE0_MASK_AUTHORITY）；`sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/`；`sua_exploration/docs/PSEUDO_MUA_T4_BRIDGE_48H.md`；`btransform_unified_v2/docs/EXECUTION_688_RIFT_V1_20260907.md`；`btransform_unified_v2/dandi688_bench_v1/`
- M1：`tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py`；`btransform_unified_v2/docs/M1_CARRIER_REFINEMENT_V1_20260908.md`
