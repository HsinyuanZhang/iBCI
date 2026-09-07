# Handoff: TF-SR screen 终局 + PIRG/posterior 线审核（供外部协作者讨论）

Date: 2026-08-23
Status: review handoff。本文是操作侧对两条已收口线的审核与未决问题清单，供第三方一起思考。
写作人：实验操作 agent（全部收据在本仓库可验）。

---

## 0. 口径与参考系（全部 sealed，governing = last-bin / equal-session / variance-weighted / final-four SWA）

| 系统 | external (15 sub-M) | within (6 sub-C) |
|---|---:|---:|
| Cell D（Arm A + whole-unit dropout） | **0.4179** | 0.5697 |
| Cell T（key_padding_mask 真移除） | 0.4081 | 0.5662 |
| A2 pooled（teacher-init，3 seed） | 0.3461 | 0.5776 |
| Arm A（无扰动） | 0.2604 | 0.5538 |
| G（纯增益） | 0.2699 | 0.5338 |
| R（逐元素） | 0.1877 | 0.5082 |
| W（temporal residual 头） | 0.1702 | 0.5387 |

机制结论（多轮 sealed）：D 效应 = 整 unit 联合原子消融（placeholder 无害、增益非机制、元素散布灾难、AM/IM 双单边皆崩、一致性目标负）。SPINT 的 dropout 与 D 同构（`spint.py:445-455` 作用在融合 token 上）——我们的贡献是分解与边界，不是新正则。

---

## 1. TF-SR：预注册 screen 结果 = 决定性 FAIL

### 1.1 设计与规则
TF-SR（Task-Frame Stateful Read-in）：B3S 校准编码 + 闭式归一 T4 → 因果逐时间步融合 unit token → 状态条件置换不变 read-in → GRU 递归 → velocity。带 D 的精确 dropout 律。预注册规则（HANDOFF_TASK_FRAME_STATEFUL_READIN §0）：
"**若不能明确胜过 sealed Cell D，停止该路线；不许用静默模块扫描去救。**"

### 1.2 结果（receipt: `results/tfsr_b3st4_ddrop_seed42_matched_score_v2/score.json`；v1 评分尝试失败于 data-root 绑定，v2 语义修正 `correct_canonical_data_root_binding_only` 后数字权威收录 v1_score）

| | TFSR seed42 | Cell D | Δ |
|---|---:|---:|---:|
| **external governing** | **0.2542** | 0.4179 | **−0.1638（1/15）** |
| within governing | **0.5919** | 0.5697 | **+0.0222（4/6）** |
| external full-window | 0.0969 | 0.3086 | −0.212 |
| 控制条件 zero-carrier（external） | **−0.6349** | — | 崩溃 |
| 控制条件 wrong_pair（external） | −0.0602 | — | aligned−wrong_pair = **+0.3144（14/15，CI [+0.212, +0.417]）** |

评分链完整性：双 surface 只读打开、零 target optimizer/backward/update 证据、Cell D 参考数与全部历史收据逐位一致（0.4179/0.5697）。

### 1.3 我的审核判读
1. **FAIL 是决定性的**：−0.164 的 external 差距远超任何 seed 噪声量级（A2 三 seed 离差 0.066）。
2. **这是"容量→拟合→迁移差"模式的第三次系统级复现**（W −0.090、DH governing −1.137、TFSR −0.164），但带一个新信息：**TFSR 是第一个 within 超过 A2 全部 seed 的系统**（0.5919 > 0.5849/0.5725/0.5755）。within/external 的权衡现在有了系统级实证：**时序容量买到的是分布内拟合，不是迁移**。这本身是论文可用的发现。
3. **载体依赖在 TFSR 中更强**（zero-carrier 即崩、aligned−wrong_pair +0.314）："载体的必要性跨主干成立"有数据支撑；但"载体增益跨主干成立"展示不出来——因为主干本身外部不竞争（0.254 连 Arm A 0.260 都不如）。外部顾问建议的"backbone-generality 消融"重定位，其可行形态是**必要性表**而非增益表。
4. 校准记录：我事前给 GO ~25% / ≈D ~45% / 更差 ~30%；结果落在"更差"尾部（within 方向兑现、external 幅度超我中位预期）。
5. **seed 43（GPU0，epoch 40/48，~6.5h 剩余）的原始目的（GO 后 matched 复制）已失效**。处置待决：停（省 6.5h）vs 跑完（作为 within/external 分裂的第二个种子点，价值有限但非零）。操作侧不擅自动运行中的进程。

### 1.4 按预注册的默认处置
停 TF-SR 路线（不做模块扫描救援）。可选的止损动作只有写作层面的（§1.3 第 3 点的必要性表 + 第 2 点的 within/external 权衡发现）。

---

## 2. PIRG / posterior-carrier 线：两连 null 的审核

### 2.1 经过
1. **Posterior carrier consumer**（把后验精度作为输入特征喂给解码器）：负结果（前置链 receipt：`posterior_carrier_budgetmix_d_seed42_*`，多次诚实 smoke 失败后跑通，结果为负）。
2. **PIRG**（WORKORDER_POSTERIOR_IDENTITY_RESIDUAL_GATE_20260823）：最小继任探针。frozen Cell-D，唯一活参数是单标量 α：`g_i = 1 + 0.5·tanh(α)·tanh(z_i)`，`z_i = clamp(log r_i − mean log r, −4, 4)`（r = 后验方向可信度），`decode_with_identity(neural, identity × g)`；3 逻辑 epoch（M4/M10/M30 轮换）；α=0 时与 Cell-D 位级相等。**结果：几乎无效果**（用户报告；PIRG 的 score 收据在实验 agent 侧，本仓库 `results/posterior_identity_residual_gate_score_v2` 尚未落盘）。

### 2.2 我的审核：~70% 原理问题 / ~30% 实现把测试做弱
**原理层（主因，三条独立同向）：**
1. **干预通道在不变性包络内部**。Cell-D 的 dropout 训练让解码器在 unit 的 identity **完全缺席**时仍出好输出（AM/IM 证明联合原子消融承重）。±50% 的 identity 乘性调制远温和于模型已被边缘化的扰动——**门在调一个被专门训练去忽略的旋钮**。
2. **PIRG 属"重加权/强制"家族**——Cell C（强制一致，−0.036）vs D（随机暴露，+0.158）已经定罪过这个家族：正确的正则化是**暴露多样组合**（训练期边缘化），不是**向偏好构型重加权**（消费端强制）。后验信息的"消费端"用法（作为输入特征、作为门控）已两连 null。
3. **信号假定的故障模式可能不存在**：M30 前缀下点载体已足够好；短前缀（M4）下瓶颈是载体估计方差本身，下游门控无法修复未编码进载体的信息——门只能抑噪，不能注信。

**实现层（次要）：** 单标量 α 经双重 tanh 压缩 + 仅 3 epoch + 3+3 session 小屏——功效不足是设计出来的；即便修好（自由逐 unit 门、全 15 session 屏），预测仍 null（原理 1 在前）。

### 2.3 未测的第三臂（唯一机制自洽的形式）
**训练期后验采样**：源域训练时 β ~ N(β̂, Σ̂_i)——与 dropout（存在性边缘化）在同一会话生成分布框架下并列的**参数不确定性边缘化**。它是"暴露"家族，绕开全部三条原理障碍；PIRG 的 null 恰为它清了路（工单原本禁止它作为 first cell）。预期管理：M30 制度下预计同样 null；它的战场只在短前缀（M4 / H1 型）。成本 = 一个训练 cell。

---

## 3. 汇总：五次设计尝试的结构性模式

| 尝试 | 类型 | external vs 参照 |
|---|---|---|
| S2 扇区几何 | 训练结构 | −0.041 vs D |
| C 一致性目标 | 消费端强制 | −0.036 vs D |
| W temporal residual | 容量追加 | −0.090 vs Arm A |
| PIRG/posterior 消费端 ×2 | 消费端重加权 | ≈0 |
| TF-SR 换主干 | 容量+时序 | **−0.164 vs D**（within +0.022） |

所有失败同族：**在 D 的扰动骨架之外加东西（容量、目标、几何、门控）或换更大骨架，外部迁移一律不涨反跌；唯一 +0.158 的来源（整 unit 消融）是 SPINT 已有机制的分解确认**。与此互补：消费端使用后验精度两连 null。

---

## 4. 开放问题（供讨论，按操作侧倾向排序）

1. **论文转向时机**：外部分析的修正版 P0（标签预算对齐基线组：Ridge/Wiener on prefix、Hungarian 匹配+冻结解码器、末层重训；充分统计量命题按"模型条件化+精确β"诚实口径写；D 家族重写为会话组成域随机化；全代号语义化重命名；混合效应统计）——是否现在执行？操作侧判断：证据收集阶段已过收益递减点，转向写作的证据密度已够。
2. **TF-SR seed 43 处置**：停 vs 跑完（~6.5h 剩余）。
3. **后验第三臂（训练期采样暴露）**：跑不跑？若跑，绑定标签预算扫描（M30/M10/M4）作为主战场，否则必 null。
4. **Cell-D seeds 43/44**：任何 A2 优越性主张的前置，已四次顺延；与 1 互斥占用 GPU。
5. **TF-SR 的写作收割**：within/external 权衡发现 + 载体必要性跨主干表，是否进论文（作为负结果边界章节）。

## 5. 收据索引
- TF-SR screen: `results/tfsr_b3st4_ddrop_seed42_matched_score_v2/score.json`（含 v1 失败谱系与 data-root 修正声明）
- TF-SR 训练: `results/tfsr_b3st4_ddrop_seed42_train_v2/`（CELL_TERMINAL，48ep）；seed43 `…_seed43_train_v1/`（进行中）
- 提速链: `results/tfsr_b3st4_ddrop_throughput_engineering_v1/v2/v3`（v3 收据 SHA `99c904f8…`；结论：jit_scripted 1.5× 唯一有效）
- PIRG 工单: `docs/WORKORDER_POSTERIOR_IDENTITY_RESIDUAL_GATE_20260823.md`；posterior 前置: `results/posterior_carrier_budgetmix_d_seed42_*`
- 机制分解轮: `results/aimask_score_v1/`（AM/IM 联合必要）、`results/subpop_score_v1_r2/`（T/C/G/W）、`results/subpop_step0c_v1/`（曲线+集成排除）
- SPINT dropout 代码事实: `streaming_calibration_exp/src/models/components/spint.py:445-455`
