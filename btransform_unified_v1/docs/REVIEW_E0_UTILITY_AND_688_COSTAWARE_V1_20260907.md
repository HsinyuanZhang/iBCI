# 审核：实验一（E0 使用依赖与增量效用）与实验二（688 低成本 B-transformer）

日期：2026-09-07。审核对象：
- [DESIGN_E0_UTILITY_PAIRED_V1_20260907.md](DESIGN_E0_UTILITY_PAIRED_V1_20260907.md)（下称 E0-设计）
- [DESIGN_688_BTRANSFORM_COST_AWARE_V1_20260907.md](DESIGN_688_BTRANSFORM_COST_AWARE_V1_20260907.md)（下称 688-设计）

本审核只读；不启动训练、不改设计稿原文、不改代码。所有"已核实"项均对照工作区实际文件/receipt 检查过（路径见各条）。标记：**[阻断]** = 不改不得开跑；**[必改]** = 开跑前写进设计；**[建议]** = 可选。

---

## 0. 总判

两份设计的科学逻辑成立：E0 的"reliance（推理干预、不训练）"与"utility（同初始化配对重训）"分离正确；第二层配对重训落到 688 三臂里复用，是最省的落点。主要问题不在逻辑，而在**数值口径与对象指定**：

1. 688-设计的**更新数预算小了约一个量级**（8k updates ≈ 0.57 epoch），FULL−NOE0/NOC 的零结果将不可解释；GPU 预算（6 GPU-h）其实允许 5–10× 的更新数。**[阻断]**
2. E0-设计的三个 checkpoint 只给了"P16 e24"这类简称，其中 M1 e24 与 H1 full13 e24 都是全 session 训练 → 本地任何面都是 exposed；而工作区**已经在跑**能给出 clean 面的配对模型（M1 depth2 系列的 depth-4 基线；H1 C2-protocol L200）。设计应改为首选这些。**[必改]**
3. 688 decoder 的固定 `units` 几何与 padding 规则未写；`go_cue` 列名与实际 NWB 不一致。**[必改]**

---

## 1. 实验一（E0-设计）审核

### 1.1 已核实为正确的实现前提

| 设计声明 | 核实 |
|---|---|
| "P 无 bias，zero 等价" | `identity_variant.py`: `e0_proj = nn.Linear(base_e0_dim, proj_out_dim, bias=False)` ✓；bank.E0 置零 ⇒ P(E0)=0，static 项退化为 `carrier@W_cᵀ+b`，精确 |
| "不要直接用 `identity_mode=zero`" | 该模式与 concat 同参数化（token_in = 16+d_e+4），proj_add 的 state_dict 装不进去 ✓ 警告正确 |
| "最干净实现是 bank.E0 替换" | 非折叠路径 `_frontend` 直接把 raw bank E0 传 `frontend.forward`；折叠路径 `bank_static_term→static_term` 每次重算 P(E0) ✓。任何在 reset 时缓存 `proj_static`/`static` 的 wrapper（如运行时线的 `FastExactEEngine.proj_static`）必须逐 arm 重建——设计的告诫成立 |
| M1 E0 = B3 Sfix e11 activity-only、rSyn3 不进 E0 | 工单 §4 表 + `m1_projadd_series/P16_20260906T150917Z/run_meta.json` picks `CAL-3f {bank E0 [64,100] (B3 Sfix e11 …)}` ✓ |

**[建议]** 评分直接用训练模型 `forward()` 对 `[B, L, N]` 窗批量推理（GPU），不要经过 exact-E wrapper；这样根本不存在 static/frontend cache 过期的问题，也符合设计"算法消融评分，不是延迟证明"的定位。

### 1.2 checkpoint 与评分面 **[必改]**

设计 §2 "M1=P16 e24；H1=P16 L200 e24；M2 按 pick 记录" 需要具体化：

| 任务 | 设计所指 | 训练面 | 本地面暴露 | 审核建议 |
|---|---|---|---|---|
| M1 | `m1_projadd_series/P16_20260906T150917Z/epoch_024.pt` | 全 4 session（run_meta `sessions` 含 20120924） | **全 exposed**（score_receipt 的"LOSO 面" 0.9798 是训练覆盖面） | 降为**次级 exposed 读数**。**首选**改为 `results/m1_projadd_depth2/<root>/depth4/` 的匹配 depth-4 基线（26/27/28 训练，20120924 26,496 窗为真留出；GPU0 正在跑 depth-2，depth-4 随后）。同 root 的 depth-2 模型可作"深度稳健性"附读，不算第二 seed |
| H1 | "P16 L200 e24" 二义：`h1_stage2_full13/pair_…/L200` e24（13 session 全训 → 本地全 exposed）或正在跑的 `h1_c2protocol_l200_p16`（13 held-in 训练，32 ep，C2 HO-M3 在 14 个 public held-out-calib 录音上选点，见 [H1_C2_PROTOCOL_SELECTION_V1](H1_C2_PROTOCOL_SELECTION_V1.md)） | — | **首选** C2-protocol 运行：面 = HO-M3 14 录音（未训练）；checkpoint 二选一并预先写定：(a) 其 `select_epoch` pick（须披露 pick 与干预同面）或 (b) endpoint e32 EMA（无 pick 耦合，推荐）。full13 e24 作 exposed 次级 |
| M2 | `m2_projadd/20260906_104300` SEL-2 pick e9（ext4 0.4016） | 源 7 session | ext4 未训练 → clean；pick 在 ext4 上选 → 轻度选择暴露，照报 | 同意；写明 pick receipt 路径与 e9 SHA |

推论：M1 的 clean 第一层只有 1 个 session → 无跨 session 区间，只能给**时间块 bootstrap**（块而非窗）——设计 §4 已说"窗不是独立样本"，此处要明写 M1 clean 读数是单 session 点估计。

### 1.3 干预臂 **[建议增补，零训练成本]**

- **加 C-ZERO / C-SHUFFLE（carrier 干预）两臂**（+1 与 +3 条推理流）。理由：设计 §5 已预见"ZERO≥REAL 可能与直接 c 冗余"，而 688-设计的 B-NOC 是否值得晋级也需要这条信息；现在不做，之后仍要回来补。单列一张表，不与 E0 结论混写。
- E-MEAN 与 REAL 的差 = 单元特异信息；SHUFFLE 与 REAL 的差 = 正确对应。§5 判读表缺一行：**MEAN > REAL**（E0 的单元特异部分有害/噪声）→ 直接支持"无 E0 重训"，不要归为"无辨识力"。
- 诊断项再加一项：各 arm 下第一层预激活 `pre = local@W_localᵀ + static` 的 mean/std（按 session）。它把"分布冲击"（ZERO 的 OOD）与"信息缺失"分开，成本为零。

### 1.4 其他

- **[建议]** "每 session ≤2,048 endpoints" 对 M1/M2 没必要：26,496 窗 × [100,64] 的 GPU 推理是秒级。首轮直接用**完整合法面**（M1 20120924 26,496；M2 ext4 全部；H1 HO-M3 全部），上限只留给 H1 20,325 完整流。这样第一层结果与其他线的数字**同面可比**。
- **[必改]** H1 各 arm 必须共用同一个 **M3 部署 bank**（CAL-1 训练轮换，推理 M=3 固定），并写进 manifest；否则 REAL 与 ZERO 之间混入预算差。
- 第二层委托给 688：成立，但前提是 688 的训练预算足够（见 §2.1）。
- "6 条推理流" 若采纳 1.3 则为 10 条；仍然是零训练。

---

## 2. 实验二（688-设计）审核

### 2.1 更新数预算 **[阻断]**

设计 §6：S1 每臂 2,000 updates、horizon 8,000、batch 32。核对：

- **单步成本实测**：M1 depth-2 运行（W=100、N=64、B=32、bf16、3090）`budget_projection.json` = **0.028 s/update**。688（W=50、N≤100）同量级 → 8,000 updates ≈ **4 分钟**。
- **数据量**：cache `dandi688_subc_co_v1/sessions/sub-C_ses-CO-20131003_*.npz`：`neural (33161, 71)`、`valid_starts (18833,)`；27 源 session ≈ 4–5×10⁵ 合法窗 ⇒ **≈14k updates/epoch**。8,000 updates ≈ **0.57 epoch**。
- **家族配方参照**：M2 S1 官方 pick e8 ≈ 25k updates；M1 P16 24 ep ≈ 85k–160k updates。8k 是家族用量的 1/10–1/20。

后果：FULL−NOE0、FULL−NOC 在 8k 处若为零，唯一诚实的结论是"未收敛"（设计 §7 自己的"证据不足"条款必然触发；工单 §0.2 的 QueryAge FLAT e2 教训同类）。

**修改建议（仍在 6 GPU-h 内）**：

| 项 | 原 | 改 |
|---|---|---|
| horizon | 8,000 | **48,000**（≈3.4 epoch；≈25 min/臂） |
| S1 筛选点 | 2,000 RAW | **8,000 RAW**（与 warmup 200 / cosine 1e-4→1e-5 在新 horizon 上重述） |
| 预注册读点 | step 2000 | 16k / 32k / **48k endpoint EMA（主）**；不选点 |
| S1+S2 总量 | 6k + 12k + 16k | 3×8k + 2×40k(续) + 2×48k (+NOC 2×48k) ≈ 3×10⁵ updates ≈ **2.5 GPU-h** |

续跑规则（不重启 schedule/optimizer/EMA/RNG）保留不变。

### 2.2 固定 `units` 几何与 padding **[必改]**

`model.py::_check_input` 钉死 `x.size(2)==units`，bank E0 形状 `[units, d_e]`。27 源 session N=41–74（B0 receipt `session_unit_counts`），sealed 6 个未知（合同 <100）。设计 §5 "N 随 session 变化，用实际 N/valid mask" 需落成：

- `units = 100`（或 99），每 session pad 到该宽；pad 行 `unit_mask=False`、E0=0、carrier=0。
- pad 行不进注意力（`key_padding_mask`）但**仍消耗 token MLP/KV 计算**——写进吞吐预检的假设。
- `_expand_keep` 要求每窗至少 1 个有效单元 ✓（真实 N≥41 自然满足）。
- unit dropout 只作用于有效单元（`whole_unit_dropout(keep)` 以 keep 为基 ✓）。

### 2.3 事件列名 **[必改]**

设计 §3.1 写 `go_cue_time=g_m`。实际 NWB trials 列为 **`go_cue_time_array`**（`sua_exploration/mc_maze/rt_classical_comparators.py:389`，经 `_go_cues(trials["go_cue_time_array"][:], n)` 规约到每 trial 单值）。设计须：写实际列名；写多 cue / 缺 cue trial 的规则（沿用 `_go_cues` 的语义或独立定义并披露）；`target_dir` 列名 ✓ 存在（同文件 :352）。

### 2.4 已核实为一致的项

| 项 | 核实 |
|---|---|
| manifest SHA | `sha256sum sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` = `4607e979…13a0c9` ✓ 与设计一致 |
| donor `B0 s42 epoch_011.ckpt` | 存在；`lightning_logs/*/hparams.yaml`：`side_dim: 0`、`window_size: 50`、`trial_length: 100`、`identity_mode: calibrated`、`id_hidden_dim: 128`（内部宽度，非输出宽度）；receipt `p3_…_b0_…_s42_seed42.json`：`train_val_manifest_sha256` 同上、`calibration_n_trials: 30`、`session_splits.test` 6 个**未装载**（`formal_test_sessions_loaded_during_fit: false`）、val 6 个为 validation loader（visible dev）✓ 与设计 "6 val 有历史暴露" 一致 |
| teacher | `teacher_mc_maze/best-epoch=083…`，SHA `9b4a94ca…`；**其 session roster 不在 B0 receipt 中** → 设计的"审计 teacher roster"条款**必须执行**（读 teacher 的 hparams/receipt；B0 `loss_mode: task_only` 提示 teacher 可能未参与 loss，但要证明不是假设） |
| `t4_from_trial_sums(sums, lengths, angles, *, source)` | `streaming_calibration_exp/src/data/falcon_t4_features.py:82`：每 trial 一行、`rates = sums/lengths`、`lstsq` 无 ridge、design rank 必须 3 ✓。传 `lengths = 35` ⇒ counts/20-ms-bin ✓ |
| B0 无 T4 侧特征 | `side_dim: 0` ⇒ c-zero 确实移除全部方向信息（不像 B3S 把 T4 藏在 E0 里）✓ 这是本设计最干净的一点 |
| cache 无原始 spike times | `sessions/*.npz` 仅 `neural/behavior/calib_trials/valid_starts` ⇒ MOVE-T4 需一次 NWB 读（train/val allowlist）✓ 设计已说 |
| 现有 cache 的 `calib_trials (10,100,N)` 是 **M10** | 设计要求 M30 ⇒ E0 与 carrier 都要新建 M30 缓存，**不得复用 M10 数组**（写进 S0） |

### 2.5 其他 [建议]

- B-NOC：carrier 列输入恒 0 ⇒ `W_c` 不接收梯度、停留在初始化；这是合法的 matched ablation，但要在 receipt 写明"W_c dead"。
- 参考评分：加 (a) 同面常数基线、(b) B0 e11 自带 decoder 在同 support/query 合同下的 system-level 分。设计 §5 已含 (b)。
- 深度轴：家族可能转向 depth-2 / valid-conv（见 [ANALYSIS_FORWARD_TIME_STRUCTURAL_LEVERS_V1_20260907.md](ANALYSIS_FORWARD_TIME_STRUCTURAL_LEVERS_V1_20260907.md)）。688 无延迟约束，首轮保持 CausalPE4 + 现行窗内零填充 conv 以便与 M1/H1 既有 cell 可比，但 manifest 必须显式记录 `temporal_layers=4`、`conv_pad=in_window_zero`。
- §3.3 CPU sanity 再加一项：700 ms 窗与下一 trial start 的重叠比例（按 session 分布）。
- 三个 descriptor 候选的 split-half 稳定性用 **相关系数 + 符号一致率**两个读数，避免只报一个。

---

## 3. 两份设计的接口

- 第二层（配对重训）只在 688 上做；M1/H1 的 utility 结论仍待各自任务的小规模配对——设计已写明，审核同意。
- 若采纳 §1.3 的 carrier 干预臂，688 的 B-NOC 晋级规则（688-设计 §6 第 4 条）可以引用实验一的 C-ZERO 读数作为**先验**，但不能替代 matched FULL−NOC。

## 4. 修改清单（执行者用）

1. [阻断] 688 horizon 8k→48k，S1 筛选 2k→8k，读点 16k/32k/48k。
2. [必改] 688 `units=100` padding/mask 规则写进 §5/§6。
3. [必改] 688 §3.1 列名 `go_cue_time_array` + 多/缺 cue 规则。
4. [必改] 688 S0：M30 新缓存；不复用 M10 `calib_trials`。
5. [必改] E0 §2：M1 首选 depth2-root 的 depth-4 基线（clean 20120924 面）；H1 首选 C2-protocol L200 e32 EMA（HO-M3 面）；e24 全训模型降为 exposed 次级。
6. [必改] E0：H1 各 arm 共用 M3 部署 bank。
7. [建议] E0：加 C-ZERO/C-SHUFFLE；加 MEAN>REAL 判读行；加预激活分布诊断；M1/M2 用完整合法面替代 2,048 上限。
8. [必改] 688 §4：teacher roster 审计落到具体文件（teacher hparams/receipt），结论进 receipt。
