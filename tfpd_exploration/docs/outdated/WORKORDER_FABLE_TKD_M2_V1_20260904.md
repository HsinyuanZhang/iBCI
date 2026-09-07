# WORKORDER_FABLE_TKD_M2_V1 — 调谐键控解码器（TKD）M2 首发实验

Date: 2026-09-04
Type: 预注册工单（进入实现前冻结；门与臂的修订只允许按 §9 的显式规则并留偏差记录）
Parent design: `FABLE0904_decoder_design1.md`（TKD 设计提案）
Parent review: `FABLE0904_review1.md`
Experiment id: `fable_tkd_m2_v1`
Result root: `tfpd_exploration/results/fable_tkd_m2_v1/`
Status: FROZEN_FOR_IMPLEMENTATION

---

## 0. 主张与一句话

TKD = population vector 的学习型推广：**per-unit 调谐描述子 T4 经共享映射 Φ_k 直接生成 attention 的 key/value 读入权重**，一层 latent cross-attention + 低维 SSM 时间模型 + 读出。

主张（继承 design1 §0.3）：**identity 适配 + 成本**——新 session 零梯度部署（session 相关量只有闭式 T4 与 ρ）、流式成本比 SPINT champion 低 ≥ 一个数量级。绝对 R² 只设非劣门，不是主张。

## 1. 与 design1 的显式偏差（发散记录）

| # | design1 原文 | 本工单 | 理由 |
|---|---|---|---|
| D1 | 首轮 688（27/6），M2 第二 | **首轮 M2**，M1 第二、H1 第三、688 最后 | 用户 2026-09-04 指令：优先 M2/M1/H1 中运算快的，688 最低优先级。运算量排序 M2（分钟级）< M1（~1.7 h/run）< H1（50 ep × 206k steps）< 688 |
| D2 | PV 锚只在 688 断言，M2 复用 688 init | **PV 锚在 M2 本地断言**（M2 同为 cosine 调谐 + 方向试次，结构同 688）；M2 自带 init，不依赖 688 | M2 先行，688 init 不存在 |
| D3 | REF = 688 SPINT+T4 密封结果 | **REF = M2 act30 champion 密封格 `ridge_static_m30`，外部 6 session 等权均值 0.29521985196829853**（`m2_hold_film_probe_v1/plan.py` SEALED_M30_EXTERNAL；checkpoint sha 25d7bc72…） | 同数据集同面 |
| D4 | identity 门 +0.10（688 T4 增量量级） | 门值保留 +0.10，**预注册校准修订规则**（§7 门 G3）：先推理重放 champion ts4 对照（T4 整行置换，训练+部署同置换，`b3s_ts4_m2_loso_internal`，ckpt e385e2f4…），若 champion 自身 T4 内容增量 < 0.20 则门降为 ½×champion 增量 | M2 的 T4 增量量级未密封，需同面校准 |
| D5 | 688 臂 B（PV 锚冻结）等六臂 | 臂结构照搬（§4），粒度适配 M2：N=96、W=50、B=2、M=30 校准块 | — |

不偏离主线：架构、PV 锚、ε 设计、ρ 门、零梯度部署契约、成本主张全部照 design1 §2–§3。

## 2. 面（faces）定义

**数据**：Dandiset 000953（MonkeyN）。held-in 7 session（2020-10-19/20/27/28），外部 6 session（2020-10-30 ×2、11-18、11-19、11-24 ×2）。20 ms bins，N=96 channels，W=50 bins（1 s 窗），B=2（finger_vel），行为缩放 BEHAVIOR_SCALE=5.0（与 champion 同律）。

**T4（identity）**：`fit_ridge_t4`（`tfpd_exploration/src/calibration_budget_comparators_v1.py`，设计阵 [cosθ, sinθ, 1]，λ_norm=0.1，normal equations），试次选择律 = M30 法（chronological first-30，≥3 有限方向试次，`m2_t4_activity_budget_screen_v1/physical.py::_support_indices`）。产出 raw [96,4]=[a,c,m,b]。**归一化律**：held-in 7 session 池化的 per-column mean/std（source 律，冻结后用于外部 session），与 champion 的 train-only T4 归一化同族；per-session 归一化后 T4 记 t_i，digest 逐 session 钉死。

**ρ（可靠性门，不学习）**：同一次 cosine 拟合的 per-unit R²（闭式，1 − SS_res/SS_tot，对有限方向试次）。部署时用外部 session 自己的校准块计算。

**训练面**（7 held-in）：每 session 的 post-30 窗（window start ≥ trial-30 起点，`select_common_post30_window_starts` 同律），输入 neural[50,96]，目标 covariate[start+49]×5.0，T4/ρ 来自该 session M30 校准块。**模型从零训练，无教师、无 λ_E**（champion 是 B3S 编码器 + 冻结 SPINT 教师蒸馏；这是面差异，结果文档必须披露）。

**部署/评测面**（6 外部）：`dataset.window_indices` 原样（与 `m2_t4_activity_budget_screen_v1` 外部面逐窗一致），T4/ρ 闭式来自外部 session 首 30 校准试次，**零梯度**。指标：`variance_weighted_r2`（2-D 方差加权），等 session 均值 `equal_session_mean`。p0 容差：本工单不重放 champion（密封数直接引用）；若实现需要 sanity，可用 probe 的 p0 复现律（1e-6）。

**epoch 选择律**：与 champion 同族——每 epoch 在 held-in within post-30 面算等权均值 R²，取 best epoch 部署（champion 用 val_heldin 选了 12 中的 epoch 2；TKD 同等权利，无泄漏）。

## 3. 架构（design1 §2.2 原样，M2 实例化）

- ψ：因果 Conv1d(1→16, kernel 10, per-unit 共享) → Linear(16→64)（d_v=64）
- k_i = Φ_k(t_i) + ε·P u_i；Φ_k = Linear(4→64)→ReLU→Linear(64→64)（d_k=64）；ε 可学习标量零初始化
- v_i = ρ_i · u_i；L=8 可学习 query；α = softmax_i(q_ℓ·k_i/√64)
- z_t = Linear(8·64→128)（d_h=128）；SSM：对角线性 SSM（S4D/LRU 型）×2 层 + GLU，训练并行 scan、推理 O(d_h) 递推，Stage 0 断言 scan vs 递推 ≤1e-6
- y_t = Linear(128→2)；预测 /5.0；只训末 timestep 损失（MSE on ×5.0 目标）
- PV 锚初始化（design1 §3.4，M2 实例化）：ψ 近似 10-bin 均值速率、u 首坐标 = (rate10−b)/m；Φ_k 把 t_i 映到偏好方向子空间；q_ℓ = 8 等间隔方向单位向量；merge+读出按方向余弦投影到 2-D 速度。断言见 §6 Stage 0-A1。
- 置换不变、N 无关、无 session 专属参数；ε=0 时 key 静态可缓存。

## 4. 臂

| 臂 | 定义 | seeds | 回答 |
|---|---|---|---|
| REF | act30 champion 密封 0.2952（+ M33 0.29913、means-FiLM 0.32102 作 context，不进门） | — | 参照 |
| REF-SHUF | champion ts4 ckpt（e385e2f4…）在同一外部面推理重放 | — | champion 的 T4 内容增量（identity 门校准） |
| **A′** | TKD，ε 可学习 | 42/43/44 | 主臂 |
| A | TKD，ε≡0 | 42/43/44 | 内容依赖门控的价值 |
| SHUF | A′，t_i 行在 96 channel 间确定性置换（seed 42 固定，训练+部署同置换，与 champion ts4 同律） | 42/43/44 | identity 内容控制 |
| POOL | A′，key 全同（退化为 ρ 加权均值池化 + SSM） | 42/43/44 | 无 identity 下界 |
| B | PV 锚 init 后冻结 Φ_k/q/ψ，只训 SSM+读出 | 42/43/44 | 学习型映射比经典 PV 多买多少 |
| A′-GRU | A′，SSM 换 GRU(128) | 42 | 时间模型 ablation（组件选择，不进门） |

统一调度：Adam lr 3e-4、wd 0、batch 32、30 epochs、FP32、grad clip 1.0、全 post-30 窗/epoch、shuffle seed=run seed。A′/A/SHUF/POOL 参数量相同；SHUF 置换对三个 seed 相同（只依赖 SHUFFLE_SEED=42）。

## 5. 成本与延迟测量

- MAC：解析计数脚本（TKD 流式 per-bin、TKD 整窗 per-window、champion per-window——champion 侧 B3S 18,290 参数 + SpintModel 常数从 resolved_config 推）。
- 延迟：CPU 单线程 pinned，同一外部 session 的全部窗：champion 整窗解码 vs TKD 递推流式；报 ms/window 与 ms/bin。
- unit-drop 曲线：复用 `streaming_calibration_exp/scripts/eval_neuron_drop_curve.py` 协议（10/25/50%，3 draws，seed 42/43/44）：TKD（A′ 主 seed）与 champion 各测；champion drop 重放若脚本不适配 M2 ckpt，允许最小适配并记录。

## 6. 阶段

**Stage 0（CPU，无 GPU 占用）**
- A1 PV 锚断言：TKD(init, ε=0) 与显式 PV 参照（y = Σ_i ρ_i·(rate10_i−b_i)/m_i·PD̂_i，PD̂_i=[a_i,c_i]/m_i）在 2 个 held-in session 的 post-30 窗批上，预测矩阵 Pearson 相关：**目标 >0.99；[0.90, 0.99) 过但强制披露；<0.90 阻断 Stage 1**，只允许调 β（softmax 温度）与 merge init，不许改维度/臂。
- A2 SSM scan vs 递推 parity ≤1e-6（随机输入 2k bins）。
- A3 SHUF/POOL 构造单测：SHUF 置换为非恒等、逐 seed 一致；POOL 下 key 全同且与 A′ 同参数量；ε=0 时静态 key 缓存与逐步计算 bit-exact。
- A4 MAC 计数脚本 + 报告（JSON receipt）。
- A5 t_i 分布偏移报告：held-in 池化 mean/std 归一化后，6 外部 session t_i 各列 mean/scale 相对 held-in 的偏移。
- A6 ρ 分布报告（held-in + 外部）。

**Stage 0.5（GPU1，推理-only，~分钟）**：REF-SHUF 重放（ts4 ckpt，外部面）。

**Stage 1（GPU1）**
- 1a pilot：A′ seed42 单跑。sanity 门：外部等权均值 ∈ [0.10, 0.45]（高于平凡、低于 within 面 0.69），held-in within 单调改善。不过 → 停，调试，不许换面。
- 1b 全格：其余 15 runs（A′ s43/44、A/SHUF/POOL/B ×3、A′-GRU ×1）。
- 1c 评测：全部臂外部面 + within 面；drop 曲线；CPU 延迟；ε 终值；attention-偏好方向图（主 seed）。

**Stage 2+（另立工单）**：M1（rSyn3 4 列 identity，fold-local LOSO 面，REF=Z-Fix 0.6374/S-Fix 0.6203；PV 锚不适用，臂缩为 A′/SHUF/POOL）——触发条件：M2 §7 至少三门过。H1 第三（H-C carrier，±0.04 噪声面上只主张成本）。688 最后（design1 §5.1 原面）。

## 7. 预注册门（M2 版）

- **G1 非劣门**：mean(A′ 3 seeds) ≥ 0.2952 − 0.03 = 0.2652，且 ≥4/6 session 满足 A′_s ≥ REF_s − 0.05（per-session，3 seeds 均值）。
- **G2 成本门**：TKD 流式每窗等效 MAC ≤ champion 每窗 MAC / 10；实测 CPU 延迟 ≤ champion / 3。
- **G3 identity 门**：先校准：Δ_ref = REF − REF-SHUF（同面推理）。若 Δ_ref ≥ 0.20 → 门保持 A′−SHUF ≥ +0.10 且 A′−POOL ≥ +0.10；若 Δ_ref < 0.20 → 门修订为 A′−SHUF ≥ ½Δ_ref 且 A′−POOL ≥ ½Δ_ref（偏差记录：门值随校准变动，比较对象不变）。POOL 门值随 SHUF 门值。
- **G4 鲁棒门**：drop 25% unit 的 R² 下降（3 draws 均值）：A′ 下降 ≤ champion 下降。

决策法则（design1 §5.3 照搬）：
- 四门全过 → TKD 立为论文设计点 3（"identity 即读入几何 + 零梯度部署 + 数量级成本"），Stage 2 开 M1。
- G1 不过其余过 → 效率-精度权衡曲线，不主张替代；Stage 2 仍可开（M1 上 TKD 可能相对更强）。
- G3 不过 → Φ_k 未学到调谐→几何映射；回 §3 归一化律与 PV 锚检查（A1 披露、t_i 偏移 A5），**不扩数据集**。
- G4 单独不过 → 如实写"鲁棒性未占优"，不阻断其他结论。

## 8. 资源与工程纪律

- **GPU1 only**（UUID GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86；GPU0 归另一 agent，禁触）。每次启动前跑 ownership preflight（复用 `m1_emg_rsyn3_fold_local_v1/gpu.py` 的检查：foreign pid 拒绝、util/mem 阈值），`CUDA_VISIBLE_DEVICES=1`。
- Receipt 律照 `m2_kcurve_ext_v1/plan.py::RECEIPT_LAW`：attempt.json 先行（任何数据/模型访问前）、原子写 + 0444 + 每 receipt .sha256 sidecar、terminal.json O_EXCL、fail-closed。密封物（champion ckpt、probe results）只读。
- conda env `spint`，`PYTHONNOUSERSITE=1`；FP32 主线；无 EvalAI 提交（本工单纯本地）。
- 分工：GLM5.3 规划/文档/门判定/结果文档；flash agent 代码实现（spec 由本工单 + 实现规格给出；实现偏离 spec 需记录）。

## 9. 修订规则

只允许以下修订且必须留偏差记录（本文件不改，记入结果文档 §偏差）：
1. G3 门值按 §7 校准规则变动；
2. A1 锚断言落 [0.90, 0.99) → 披露继续；<0.90 → 仅允许 β/merge init 调整；
3. champion drop 重放的最小脚本适配；
4. pilot sanity 失败后的 bug 修复（面/臂/门不许动）。

## ADDENDUM-7（2026-09-05，用户评价律）

**本弧线及其后任何承接线的最高评判 = held-out 结果**：M2 = 外部 6 session 官方面（及 EvalAI 若提交）；M1 = fold-local LOSO 目标 session / later-day Surface B；H1 = 5-date LODO 目标日期；688 = dev-6 held-out session（及官方面）。within/minival/锚相关等一律只作诊断披露，不得作为方法价值的主张数字；任何新方法的过门判定必须以 held-out 面读数为准（选择律按各面已确立的 held-in 合法选择，选择增益与方法增益分列——承 review §1.2 与 H1 线的既定纪律）。

回溯核验：本弧线全部四条线的关门读数均已是 held-out 面（M2 外部 ≤+0.074、M1 LOSO 生成式 0.257、H1 LODO 父 ridge −13.98、688 dev-6 PV −0.007），关门结论不受本律影响。

## ADDENDUM-6（2026-09-05，r9 判读：选择律修正 + 降级前置 pilot r10）

**r9 事实**：A2-SSM 在 D15 全有界化后仍出现 −11.9 破坏性 episode——SSM 时间模型在本架构上**证伪**（负结果保留）；A2-GRU+蒸馏优化完全健康（within 0.65 单调升、ep29 仍在涨、MSE 5.3e-4 << 零解地板），但 external −0.053（11-18 一家 −0.387 拖垮，其余 ~+0.05）。

**选择律修正（Wave2 规格错误，规划侧）**：原规格用"训练所用同一批 post-30 窗"做每 epoch 选择 = 用训练集选 epoch，结构性选出最过拟合点（ep29）；champion 的选择面是 held-in **minival 划分**（val_heldin，故选了 epoch 2/12）。修正：每 session 的 post-30 窗按时间序前 80% 为训练窗、后 20% 为 **minival 窗**（不参与训练与蒸馏），部署 epoch = minival 等权均值 argmax。官方数字一律 minival 选择。

**诊断例外（明示披露）**：记录 external-vs-epoch 全曲线**仅用于线路继续/降级判定**，绝不用于选 epoch。

**pilot r10（降级判定前置）**：A2_GRU 两变体——λ_E=0.1（champion 惯例）与 λ_E=1.0（纯压缩惯例）——同调度同选择律修正。**继续/降级规则（预注册）**：若任一变体的 minival 选择 epoch 的 external ≥ 0.10 → M2 线继续，GRU 升主时间模型（SSM 记负结果），授权全格；若两变体均 < 0.10 → M2 按 ADDENDUM-5 降级条款处理（权衡曲线/负结果文档），程序转向 M1 另立工单。

**证据链（pilot r4–r8 + Stage A 闭式基线，全部 sealed）**：
- 校准后经典 PV 双面 R²≈0.03（锚在数学上精确 0.9975、物理上空）；
- ridge(rate10→y×5)：within 0.244 / external −0.026——线性逐 unit 读入跨 session 不迁移；
- TKD from-scratch 五次迭代（D8–D13 全部到位后 r8）：within 0.161 / external 0.036，且末段过拟合（within 0.047→峰→−0.23）；
- champion external 0.295 的来源 = 大容量预训练 SPINT decoder + 冻结教师蒸馏的 B3S 编码器 + 50 维活动 identity（T4 0.2952 vs 活动 identity 0.2991 本就平手）。

**判定**：G1 对 from-scratch 小 decoder 在 M2 外部面不可达，非训练技术问题，是容量+适配信息量的结构性下界。按预注册决策法则与用户主线边界（低成本 + 零梯度校准为硬约束、decoder 设计自由），转蒸馏主臂：

**D14（蒸馏主臂）**：训练损失 `L = MSE(readout_out, y×5) + λ_E·MSE(readout_out, teacher_out)`，λ_E=0.1（champion `task_plus_y_plus_E` 同律）。teacher = 冻结 act30 champion（sha 25d7bc72…）在**同一训练窗**上的末 timestep 输出（×5 空间）；teacher 的 identity E 沿用其自身部署律（M30 校准块 + B3S 编码器，probe 机器复用）。**部署契约不变**：TKD + 闭式 T4/ρ、零梯度、流式。教师仅为训练期压缩工具，不是部署依赖。臂结构、门、面、调度（lr 3e-4/30ep/batch 32）、seeds 全部不变；SHUF/POOL 语义保留（教师输出与学生置换 t 无关，identity 内容控制在学生读入侧仍然成立）。from-scratch pilots（a3/a4/a5）保留为负结果 receipts。A′-GRU/B 臂随主臂同步换训练损失。

pilot 判据带不变（external ∈ [0.10,0.45] + within 改善）。预期外部 ~0.2+；若蒸馏后仍 <0.10，M2 线按决策法则降级为权衡曲线并转向 M1（另立判定）。

1. **D13（训练面修正，追认 champion 惯例）**：
   (a) 损失改在 **×5.0 缩放目标**上（`loss = MSE(readout_out, y_raw×5.0)`，÷5.0 仅部署/评测侧保留）——与 champion `behavior_scaling_factor: 5.0` 同律；我原 Wave2 规格选 raw-target 损失使梯度尺度小 25×，属规格错误。
   (b) **init 输出尺度校准**：merge 权重乘 `target_RMS/anchor_RMS`（held-in 闭式），readout bias init = held-in 逐维目标均值（×5 空间）——仿射不变，A1 相关逐位不变；消除 epoch-0 尺度坍缩盆地。
   (c) **γ = tanh(γ_raw) 参数化**（SSM 残差增益有界于 (−1,1)，init γ_raw=0 保持精确直通），压制跨层残差×递归的自激环增益。
   全部闭式、保锚、不动 lr/epochs/batch/臂/门。A1 判据不变。
2. **补控制实验（应早在 Stage 0 就有的）**：同一 within/external 面上两个闭式基线——(i) 显式 PV 参照对目标的 R²；(ii) ridge(rate10[96] → y×5) held-in 训练、双面评测。判定律：若 ridge within < 0.10 或 PV/ridge 双双外部 < 0 → **怀疑面接线错误，阻断 r8 并上报**；否则按 D13 继续。
3. **事实记录**：raw target std ≈ 0.01/维；PV 锚 init 输出 std 0.115–0.152（12–15×）；r7 within 0.1368 / external 0.0035 / ε −0.245；epoch 0 MSE 8.2e-4 → 9.7e-5 ≈ 目标方差（零解盆地实测）。

## ADDENDUM-3（2026-09-05，pilot r4–r6 失败归因后的设计修正）

1. **D12（value 去逐单位 1/m 归一化）**：value 改为 `v = ρ·(rate10 − μ(t))/s_pooled`，其中 μ(t) 仍为恢复 raw b 的线性映射（逐单位中心化保留），`s_pooled` = held-in 池化的 (rate−b) RMS **全局常数**（闭式）。理由：(a) D6 已证 PV 机制不消费 value 侧 1/m——深度加权由 key 分数中的 m·cos∠（I₁(κm) 机制）承担；(b) 逐单位除以 m/中位数 floor 使半数 unit 的 value 尺度 O(20–100)（floor=0.0070 仍太低），训练 MSE 停在目标方差水平、模型退化为零解（pilot r4/r6 实测）。此修正属 §9 bug 修复道（init/参数化），臂/面/门/调度不动。A1 参照与判据不变；PV 配置（β/masses）允许为新 value 尺度重选一次（≤3 个 β 点取最优后冻结，记 stage0_r4）。
2. **G3 自动修订生效记录**：REF-SHUF 重放（`results/fable_tkd_m2_v1/ref_shuf_replay.json`，TERMINAL，p0 复现误差 0.0）测得 ts4 外部均值 0.20746，**Δ_ref = 0.29522 − 0.20746 = 0.08776 < 0.20** → identity 门按 §7 预注册规则自动修订为 **A′−SHUF ≥ 0.0439 且 A′−POOL ≥ 0.0439**。附注：champion 自身的 T4 内容增量在外部面仅 ~0.088（对比 688 的 +0.47~0.71），是 M2 线的重要 context。
3. **实现侧偏差（追认）**：D10（P~N(0,0.05) 打破 ε=0 鞍点）、D11（ψ 线性层尾部列 1e-2 播种，conv 通道梯度复活）；以及 Wave 2 报告的中途 bug 修复（eval digest 断言范围、eval resolver latest-wins、GPU preflight 自身 pid 豁免）。

## ADDENDUM-2（2026-09-05，用户主线边界澄清）

**发散的许可范围是 decoder 的设计本身；两条不可偏离的主线约束：**
1. **低成本**：参数量与流式 MAC 相对 SPINT champion 保持数量级优势（G2 门是下限不是目标）。
2. **无需反向传播校准**：新 session 的全部 session 相关量必须闭式（T4 ridge、ρ 及其闭式派生 key/value 归一化）；部署契约零梯度——排除任何 test-time 梯度适配、session 专属可学习参数。
任何替代/发散架构（静态加权、DeepSets 式、其他时间模型等）只要守住这两条即合法；`FABLE0904_review1.md` §5/§6 是论文定位的参照（T4=设计点1、紧凑消费者+零初始化槽=设计点2、新 decoder 证成 decoder 无关性）。

## ADDENDUM-1（2026-09-04，Wave 1 审核后；只追加不改动正文门值）

1. **A1 参照修订（偏差记录）**：原参照 `Σ ρ(r−b)/m·[cosφ,sinφ]` 含双重 1/m（value 归一化一次、方向单位化一次），对近零 m 的低 ρ 单元有 m² 放大——不属于任何经典 PV 形式。修订为**经典深度加权 population vector**：
   `y_PV = Σ_i ρ_i·(rate10_i − b_i)·[a_i, c_i]`（Georgopoulos 加权变体；无除法）。
   理由：(a) 物理标准形式；(b) TKD(init) 在均匀 bin 质量下经注意力一阶倾斜项恰好实现它（Σcosψ_ℓ=0 精确消共模），Wave 1 实测相关 0.981/0.962。判据带不变（>0.99 / [0.90,0.99) 披露 / <0.90 阻断）。
2. **G2 口径澄清（门值不变）**：比较基 = **每解码 bin**。官方外部面 stride=1，champion 每输出一个 bin 预测需重解码整个 50-bin 窗（84.0M MAC）；即便 champion 缓存 fc_in 的逐 bin token（最优缓存下界），MHA+FFN+fc_out ≈ 55.9M/bin 仍不可增量。TKD 流式 0.41M/bin。门 `TKD ≤ champion/10` 在两种 champion 口径下均以 >100× 通过（20.5M vs 4.2B 朴素口径 = 205×；0.41M vs 55.9M 最优口径 = 137×）。 receipts 两种口径都记。
3. **实现偏差（flash agent 记录，规划侧认可）**：D1 σ 线性+clamp（softplus 不可仿射反解）；D2 Φ_k 隐层旁路（dims 0–3 精确仿射携带 t，尾部 dims 初始化零贡献可训练）；D3 champion MAC 维度取自 teacher config_tree.log（resolved_config 无 decoder 维度）；D5 MAC 两侧一致排除 LayerNorm/softmax-exp。
4. **发现记录**：T4 raw m 最小 ~9e-4（σ clamp 每 session 触发 1–2 个 unit）；外部 t 分布轻微收缩（worst scale 0.545/0.594，无爆炸）；ρ 重左偏（中位 0.15，273/1248 unit < 0.05）。

## 10. 证据路径

- REF 密封：`tfpd_exploration/results/m2_hold_film_probe_v1/score.json`（p0 臂 1e-6 复现 SEALED_M30_EXTERNAL）；`tfpd_exploration/src/m2_hold_film_probe_v1/plan.py` L9–13
- champion 训练配方：`streaming_calibration_exp/outputs/streaming_calibration/m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/resolved_config.yaml`（batch 32 / window 50 / lr 1e-4 / λ_y 1.0 λ_E 0.1 / last-timestep / ×5.0 / 12 ep）
- ts4 对照：`…/e8_ts4_m2_submission_control_m33q33_v1_s42_20260801_162010/checkpoint_manifest.json`（ckpt sha e385e2f4…，源 run m2_spint_t4_mainline_fp32_v1_ts4_m2_s42_20260730_141614）；置换律 = `falcon_datamodule.py` L670–672 `deterministic_row_permutation`（归一化后整行非恒等置换）
- T4/M30 律：`tfpd_exploration/src/calibration_budget_comparators_v1.py::fit_ridge_t4`；`tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py`
- 指标/面：`tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py::variance_weighted_r2 / summarize_sessions`
- drop 协议：`streaming_calibration_exp/scripts/eval_neuron_drop_curve.py`
- GPU preflight：`tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/gpu.py`
- Receipt 律：`tfpd_exploration/src/m2_kcurve_ext_v1/plan.py::RECEIPT_LAW`
