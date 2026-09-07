# REVIEW — WORKORDER_BTRANSFORM_UNIFIED_V1 审核意见与必改清单

- 审核对象：`WORKORDER_BTRANSFORM_UNIFIED_V1_20260906.md` + `src/btransform_unified_v1/`（Phase 0 骨架）。
- 审核日期：2026-09-06。审核人：审核 agent（NOTE 作者）。
- 配套文档：[REFERENCE_TASK_DIFFERENCES_CHERRYPICK_AND_SPEED_20260906.md](REFERENCE_TASK_DIFFERENCES_CHERRYPICK_AND_SPEED_20260906.md)（下称 REF；本文引用的 `CAL-*/TRN-*/SEL-*/SPD-*` 编号来自 REF）。
- 结论：**Phase 0 骨架合格**（`pytest tests -q` 19 passed，CPU 114 s；结构与 S1 `SharedSetFrontend/CausalTransformerStack` 逐层对得上）。**工单作为上位文件有 5 处阻断项、6 处重要项、若干代码项**，改完再进 P1。
- 修改责任：工单与代码由执行者改；本文与 REF 由审核方维护。改完请在工单顶部加一行「已按 REVIEW 修订，版本 v2」。

---

## 0. 已核实、不必再查

| 工单陈述 | 证据 | 状态 |
|---|---|---|
| H1 formal12 = warmup→1e-4 恒定、12 ep | `results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/receipt.json`：`"lr": "linear 1..731 to 1e-4 then constant"`, `"epochs": 12`；终值 pooled 0.278 / session-mean 0.268 / 2,908 选点 0.320；状态 `COMPLETE_FIXED_FORMAL_NO_PROMOTION` | ✓ |
| M2 QueryAge FLAT 0.143 = source-only 选 e2 | `DIAGNOSTIC_M2_QUERYAGE_EXT4_ERROR_DECOMPOSITION_20260906.md`：FLAT plain-EMA epoch 002 | ✓ |
| S1 配方常数 | `m2_b_small_stability_v1/config.py`：LR 3e-4→3e-5 cosine、24 ep、wd 1e-2、clip 1、EMA 0.9995、p 0.10 | ✓ |
| M1 官方 −0.075 | 0.574 − 0.649 | ✓ |
| §3 结构常数 | `model.py` 与 S1 decoder 逐层一致（conv 1→16 k5、token 2 层 + LN、8 slot MHA + FFN 1024、slot_proj 2048→256、4 层 pre-LN FFN 512、readout 256→128→out） | ✓ |
| 静态折叠 parity、因果自证、bank 合同、seal | 19 项测试 | ✓ |

---

## 1. 阻断项（不改不得进 P1）

### A. 「M3-aware」「前缀循环」被重定义，偏离用户指令语义

**现状**（工单 L4 / L33 / L39 / L53）：
- 「前缀循环」→ 输入序列前缀 bin `P_task`，每 epoch 轮换切段。
- 「M3-aware」→ 「bank 冻结、query 与 support disjoint、每 epoch 重采样 session 组合」。

**事实**：这两个词在 H1 血统里是**校准预算**概念，不是输入长度概念：
- prefix-cycle = 训练时每 epoch 轮换「session 前 M 个校准 trial」(M7/M5/M4/M3) 重算 identity/carrier；M3-aware = 对齐部署时只有 3 个 trial。
- 出处：`tfpd_exploration/h1_series_20260830/docs/HANDOFF_H1_SUCCESSOR_AGENT_20260903.md` L86–93（「训练 identity 固定 M7 vs 确定性 prefix-cycle M7/M5/M4」「M3 是部署预算，靠 prefix-cycle 外推」）；C2 checkpoint schema 本身叫 `h1_cal_aug_m3_aware_dual_selection_v2_checkpoint`。
- 这是 H1 上唯一被官方证实的增益成分：C1 prefix-cycle 581748 HO 0.284 vs 固定 M7 的 T0 **+0.043**；C2 0.376。
- 工单的「bank session 冻结」**恰好禁止**了这个成分。
- 另：M2 若按字面 M3 建 bank 会破坏 S1 对照——S1/581973 的 E0 用 `SUPPORT_HORIZON = 33`（`m2_dual_track_v1/plan.py`）。

**改法**：
1. L4 上位指令原文保留；在 §3「校准 bank」行与 §5「M3-aware」句改为 REF `CAL-1` 的定义：
   > 校准 bank：E0 [N,d_e] + carrier [N,4] + unit_mask。**预算按任务**：M2 M33（S1 parity）；M1 M10；H1 训练 prefix-cycle {7,5,4,3} → 部署 M3。启用 CAL-1 时 bank 按 (session, M) 多份预计算，训练时每 (session, epoch) 轮换；推理时 M 固定，静态折叠仍成立。
2. 「前缀循环」行改名为「输入长度 L_in = W + P」（REF `TRN-5`），与 CAL-1 分开；P 的取值见 §F/§I。
3. 若用户确认原意确是输入前缀 bin，则只需把 L4 的词改掉并在 §1 注明「与 H1 血统的 prefix-cycle 不是一回事」——但仍需补 CAL-1 作为独立可选项。

### B. §6「或证明 concat [176,700] ≡ C2 early-pool 同一科学量」分支可以关掉：**不是同一用法**

**事实**：
- C2 消费方式：`src = src + identity`（`SPINT-main/src/models/components/h1_carrierid_spint.py` forward）。700 维 identity 与该单元的 700 bin spike 窗**逐位相加**，再共用 `fc_in`。identity 的 700 维是**时间轴**。
- 本系列消费方式：同一张量作为静态 per-(unit, bin) 特征 concat，另配独立 `W_e0 [256,700]`（≈17.9 万参数），训练中只见过 ≤ 176 × source-session 数 个不同向量。
- 同一物体、不同算子。M2 上 d_e=50 时这样做能 work（0.390），不代表 d_e=700 也能。

**改法**：§6 改写为具体对照臂（同一 8-slot 骨架、同一时间核、同一训练制度，只换 identity 用法；REF `CAL-3`）：
- (a) 现状：post_pool 700-d concat
- (b) post_pool **之前**的 `joined` 36-d（pooled 32 + H-C 4）concat
- (c) SPINT 式：把 E0 时间对齐**加到** spike 窗（需定义前缀 bin 的对齐规则；最简单是只加到最后 700 bin）
- (d) E0 置零，只留 carrier
- (e) E0 列置换（结构破坏对照）

d_e 的 PENDING 由此落地：(a) 700 / (b) 36 / (c) 不进 token。P2b 门改为「五臂结果落盘 + 选定臂写进 §3」。

### C. 「B3S + T4 identity」在工单里没有统一，§3「一份结构，任务只改几何」对校准物不成立

**事实**（逐任务，代码已核）：

| | M2 | M1 | H1 |
|---|---|---|---|
| 编码器 | B3S（hidden 64，side_dim 4，MOVE-T4 拼进 post_pool） | B3 Sfix e11 `student.id_encoder`，`compute_identity(side_features=None)`（`m1_optimized_v2/calibration.py` L51）——**rSyn3 不进 identity** | C2 e15 `carrier_pre_pool(1024→32)` / `carrier_post_pool(36→32→32→700)`，H-C 拼进 |
| d_e | 50 | 100 | 700 |
| 来源 | M2 T4 主线 SPINT champion | Sfix SPINT e11 | C2 SPINT e15 |

三者全部是 **SPINT 系 checkpoint 训出来的冻结编码器**。工单身份声明「不继承 SPINT」对 decoder 成立，对校准物不成立。

**改法**：
1. §4 加一张上表，每格补 checkpoint 路径 + SHA + 抽取函数（M2 `champion.native_e0_and_u`；M1 `load_frozen_b3`；H1 `load_frozen_c2_materializer`）。
2. 身份声明加一句：「decoder 不是 SPINT；E0/carrier 由 SPINT 训练的 B3S 系编码器冻结产生，按 P0-2 披露。」
3. §3 表头「任务只改几何」改为「任务只改几何与校准物合同」；若要真正统一 identity（例如三任务都用 B3S hidden + T4 的低维表示），作为 P1 复刻之后的**独立 cell**，不得混进 P1。

### D. P0-3 数学写错：是 400 **倍**，不是差 400

`MSE(raw, 20y) = mean((raw − 20y)²) = 400 · mean((raw/20 − y)²) = 400 · MSE(raw/20, y)`。差值 = 399 · MSE(raw/20, y)，不是常数。

**根源**：NOTE 原文「只差常数 400」措辞含糊，已改正为比值式。工单 L10 与代码照抄了差值。

**改法**：
- 工单 L10：`验收 MSE(raw,20y) = 400 × MSE(raw/20,y)（比值，容差 1e-9 相对）且 pred_std ≥ 1% target_std`。
- `scale_bridge.py` L7、L76 文案同改；H1 分支实现时照 M2 分支的比值写法（M2 分支 L57–62 是对的：`|mse_raw − 25·mse_bridge| ≤ 1e-9·max(1, mse_raw)`）。
- `plan.py` `alignment_table` 的 H1 `scale_text`（L172–177）同改。
- `H1_CONSTANT_ACCEPTANCE = 400.0` 语义改为比值常数；`H1_CONSTANT_TOL` 改为相对容差。

### E. P2a 门「LOSO 官方式 ≥ 0.649 − 0.01」跨面减，违反工单自己的 P1-11

0.649 是 Original 的**官方 HO**（session-mean）。本地 LOSO 后段是另一个面。工单 §0 P1 明文「官方汇总 ≠ 本地 pooled，对读必须换算或并列」。

**改法**（二选一，写清楚）：
- 同面本地参照：Original 本地 0.809 / Sfix 0.828 / QueryAge 0.812（31,252 source-minival）；门写成「同一 31,252 面 pooled 与 session-mean 双报，相对 Original 0.809 的差」。
- 官方门：明说「≥ 0.649 − 0.01 只能靠官方提交判定，提交需用户单独授权，不在本工单授权范围内」。

---

## 2. 重要项（应改，不改会让结果不可解释）

### F. P1 门「ext4 endpoint24 EMA ≥ 0.40 证明骨架无损」太松，且 P=50 不是 S1 复刻

- S1 EMA endpoint24 ext4 = **0.449**（`R_session_equal_mean`），last-8 0.439 ± 0.010（`results/m2_b_small_stability_v1/20260905_123000/comparison.json`）。0.40 允许掉 0.05，证明不了「无损」。
- S1 是 `window=50`、**无 prefix**、PE 长 50（`SmallConfig`）。工单 M2 P=50 → L_in=100，是新变量。

**改法**：P1 拆两格：
- P1a：P=0、init 差异声明（见 §L）、其余全同 S1；门 = endpoint24 `R_session_equal_mean` ≥ 0.42 **且** 逐 session 配对差 ≥ −0.02；明写统计量是 session-equal mean 还是 pooled。
- P1b：P=50，作为 TRN-5 新 cell，同门。

### G. 选择律与用户「epoch-pick」指令冲突

官方唯一 B-transformer 正例 581973 = S1/EMA **e8**（ext6 epoch-pick，0.390）；e19 端点 0.351；e24 未提交。工单只写「主统计量 = EMA endpoint24」。

**改法**：§5 增加 REF `SEL-2`：预注册本地面 epoch-pick（M2 ext4、M1 LOSO fold minival、H1 2,908 minival），规则先于数字写进 receipt；endpoint24 与 last-4/8 作为稳态辅报。官方面零参与不变。ext6 那种 dev-on-official 挑选**不准**再做。

### H. 配方按 epoch 迁移会变味

M2 3165 update/epoch（`UPDATES_PER_EPOCH`），H1 731（formal12 `updates_per_epoch`）。EMA 0.9995 视界 ≈ 2000 update：M2 0.6 ep，H1 2.7 ep；「1 ep warmup」在 H1 只有 731 步。

**改法**：§5 配方以 update 数写死（warmup_updates、total_updates、ema_horizon_updates），每任务列换算表（REF `TRN-1`）；`plan.py` 的 `EPOCHS/WARMUP_EPOCHS` 旁加对应 update 常数。

### I. H1 显存与延迟没有预算门

- 显存：N=176 时单样本一张 `[L,N,256]` token 激活 L=700 ≈ 126 MB、L=1050 ≈ 189 MB，反传保留 6–8 张；3090 24 GB 上 batch 32 直上不可能。formal12 实录 microbatch 8 → effective 32。
- 延迟：H1 CausalPE 全窗在 W=700 已官方超时（581940/581942）。exact-E 缓存后本机 ≈ 29 ms/步，其中时间核 L1–3 25 ms 且 ∝ L²；P=350 → ≈ 45 ms/步。官方 `normalized_latency` 口径下仍会 > 0.7。

**改法**：
- §5 加 REF `TRN-6`（microbatch/accum 明写）。
- §7 在 P2b 之前加「H1 延迟预算门」：在冻结路径上测 A 级地板（REF `SPD-A1…A4`）；若仍超，H1 主线在 REF `SPD-C1/C2/C4` 中选定一条**写进合同后再训**。P_H1=350 在未选 C 级路线前不得启用。
- §3「无跨窗 KV，保 exact-E」保留，但注明「这是 A 级边界；H1 若走 C1 则重训并重新定义 exact-E」。

### J. §1.1「S1 配方从未在 M1/H1 上用过」过重

formal12 已用 EMA 0.9995、p 0.1、wd 0.01、clip 1；差的只是 cosine / 24 ep / peak 3e-4。改成精确表述，否则会把「恒定 LR」以外的因素也归到训练制度。

### K. §8「双卡当前空闲」此刻不成立

审核时两卡各被 `h1_queryage_family_v1.continue_prefix_train` worker 占 ≈ 9.4 GB（pid 1792958 / 1792959，另一 agent）。preflight 会拦住；文字改为「双卡由多 agent 共享，任何时刻不假设空闲」。

---

## 3. 代码项（`src/btransform_unified_v1/`）

| # | 文件:位置 | 问题 | 改法 |
|---|---|---|---|
| L | `model.py` `_init_parameters`（L252–284） | 自写 uniform(±1/√fan_in) 初始化，不是 S1 的 `m2_dual_track_v1.decoders.initialize_decoder` | 二选一：改用 `initialize_decoder`（P1a 才算复刻）；或保留并在 P1 receipt 声明「init 与 S1 不同」 |
| M | `model.py` `_expand_keep`（L305–319） | 训练态 unit dropout 从全局 RNG 采样，不接 generator；S1 有 `unit_dropout_seed(seed, epoch, batch_id)` 可复现 | `forward/forward_scores/_expand_keep` 加 `dropout_generator: torch.Generator | None` 透传给 `whole_unit_dropout`；训练循环按 (seed, epoch, batch) 派生 |
| N | `model.py` `_check_input`（L299）+ `pe` buffer | 允许 `1 ≤ L ≤ l_in`，PE 从窗首绝对计数；若训练轮换前缀长度而推理固定 L_in，末 bin 的 PE 位置不一致 | 训练与推理钉死同一 L_in；或在 `forward` 加 `require(x.size(1) == self.l_in)`，把变长只留给 `causal_check` |
| O | `scale_bridge.py` L7、L76；`plan.py` L172–177 | 见 §D | 比值式 |
| P | `plan.py` `TASK_GEOMETRY["h1"]["prefix"] = 350` | 见 §I：未过延迟门前不应作为默认几何 | 改为 0，或加 `H1_PREFIX_PENDING` 哨兵同 e0_dim 处理 |
| Q | `plan.py` 训练常数 | 见 §H | 加 update 口径常数与换算函数 |
| R | `adapters.py` 三个 stub 的 docstring | 缺校准物来源（checkpoint/SHA/抽取函数/预算）与 CAL-1 接口 | 按 §C 表补；`build_*_bank` 签名加 `budget: int` |
| S | `bank.py` `TaskBank` | 单预算冻结；CAL-1 需要多预算 | 加 `budget: int` 字段进 `calibration_meta` 必填键，或 `TaskBankSet = {M: TaskBank}` |

以上均不影响 Phase 0 测试通过；改完重跑 `pytest tests -q`。

---

## 4. 改完的验收

- 工单 v2 中：A/B/C/D/E 五项各有对应段落；F/G/H/I 落到 §5/§7；J/K 文字修正。
- 代码：`pytest` 全绿；`scale_bridge` H1 文案与 M2 分支同形；`plan.TASK_GEOMETRY["h1"]["prefix"]` 不再是 350 的裸整数。
- 每个 H1/M1 数字仍附 NOTE §6 六行表 + REF §6 picks 段。
- 不改历史 result root；不动 EvalAI；GPU 前 preflight。

---

## 5. 证据文件

- `tfpd_exploration/docs/NOTE_H1_M1_BEST_VS_BTRANSFORMER_EXTERNAL_MISALIGN_20260906.md`（P0-3 已改为比值式）
- `tfpd_exploration/src/m2_b_small_stability_v1/{config,training,decoder}.py`
- `tfpd_exploration/src/m2_dual_track_v1/{plan,data,champion}.py`（`SUPPORT_HORIZON=33`、`native_e0_and_u`、`fit_move_t4`）
- `tfpd_exploration/src/m1_optimized_v2/calibration.py`（`side_features=None`）
- `tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_calibration.py`（C2 pre/post pool 形状）
- `SPINT-main/src/models/components/h1_carrierid_spint.py`（`src = src + identity`）
- `tfpd_exploration/h1_series_20260830/docs/HANDOFF_H1_SUCCESSOR_AGENT_20260903.md`（prefix-cycle / M3 语义与数字）
- `tfpd_exploration/results/m2_b_small_stability_v1/20260905_123000/comparison.json`（S1 endpoint24 / last-8）
- `tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/OFFICIAL_581973.md`（e8 vs e19）
- `tfpd_exploration/docs/REVIEW_UNIFIED_CORE_SPEED_QA_20260906.md`（速度分量、A/C 级边界）
