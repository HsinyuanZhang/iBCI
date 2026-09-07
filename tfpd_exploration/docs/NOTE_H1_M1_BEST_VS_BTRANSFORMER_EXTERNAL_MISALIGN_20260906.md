# H1 / M1 历史最优指向哪里，以及新系列在 decoder 以外可能丢分的地方

给后续负责搭建/对齐的 agent。本文**不是**训练授权，**不是**选点规则。  
官方 HO 只作历史对照，禁止用来选 epoch 或决定下一个候选。禁止覆盖历史 result root。禁止未授权的 EvalAI 注册/提交。

分数不能跨行相减。同一行内的「本地」和「官方」也不是同一面。

---

## 1. 一句话

H1、M1 过去最好的官方成绩都指向 **SPINT 消费者**（出厂 Original，或重训/冻结的 SPINT），**不是** 现在这套 8-slot B-transformer。  
新系列如果只换时间核、却把 identity / 尺度 / 校准预算 / 输入合同换掉，会在 decoder 设计以外丢分。H1 上这已经发生过一次（近常数输出、差过零预测）。

---

## 2. 历史最优指向哪里

### H1

| 谁 | 它实际是什么 | 本地 20,325 | 官方 HO | 不要叫它 |
|---|---|---:|---:|---|
| **本地最强：as-shipped Original** | 出厂 `SpintModel`。`reset()` 只用 payload 里已有的 `calib_trial_features`，形状 **`[2, 1024, 176]`（两 trial）**。wrapper **不读** velocity/NWB。预测走 SPINT，不是 slot+因果栈。镜像 `spint-h1-released-code-lr-5e-5:epoch-049` / paper-LR 同胞。 | **0.961**（同一 cache 回放；2,908 选点子集约 0.964） | **0.262**（578474 paper-LR） | C2；B-transformer |
| **官方最强：C2 epoch 15** | **重训的 SPINT**，不是 B-transformer。early-pool 单位 identity + **H-C 4 维** carrier；训练期 prefix-cycle（M7/M5/M4/M3）；部署前 3 trial。581920。 | **0.888**（同一 `query_starts+699` 面） | **0.376** | Original；B-transformer；「把 C2 的 E0 喂进 Transformer = 在跑 C2」 |
| 家族 CausalPE / QueryAge | 8-slot + 时间核。E0 声称是 **C2 fused `[176,700]`** + H-C 4，但消费者是 Transformer。 | CausalPE 完整流约 **0.48 / 0.42**；QueryAge 选点约 **0.32**（完整面以收尾 receipt 为准） | CausalPE 超时；QueryAge 无可用官方分 | 「已经对齐 C2」 |

H1 出厂 Original 的 cold≈0.956、full≈0.963：好的 SPINT **不依赖** 14 秒互读。  
家族 B-transformer 在**同一份** `source_cache`（SHA `51ff9ebf…`）上远低于 Original/C2。题目和 target 对得上；差的是系统和校准物。

### M1

| 谁 | 它实际是什么 | 本地 31,252 | 官方 HO | 不要叫它 |
|---|---|---:|---:|---|
| **官方最强：as-shipped Original** | 出厂 `SpintModel`，epoch 19。**不是** B3S，**不是** rSyn3 主路径。578244。 | **0.809** | **0.649** | Sfix；B-transformer |
| 次优官方：581727 | **冻结的 Original SPINT decoder** + 训好的 **B3S**（early-pool MLP）+ **rSyn3** 4 维。神经 identity 用 M10；carrier 支持是 D-opt `tgt_loc` k=4。整体仍是 SPINT 消费者。 | — | **0.640** | 「整个网络都是 SPINT-like」；B-transformer |
| 本地很强但非 Original：Sfix | B3 + rSyn3 改过的系统。同源 31,252 上约 **0.828**。 | 0.828 | 未当作 Official Original | Original |
| 新系列 QueryAge / FULL | 8-slot；E0=**冻结 B3 `id_encoder` post_pool 100 维** + rSyn3 4。M1 `prediction_divisor=1`（与 H1 `/20` 不同）。 | QueryAge ~**0.812**；FULL 同构约少 0.004 | QueryAge **581982 = 0.574**（低于 Original 0.649） | 「本地 0.81 已官方非劣」 |

M1 本地 0.81 和官方 0.57/0.65 之间已经掉过一次。先排除 runtime/校准/观测 dtype，再怪时间核。

### M2（对照，不是本文主问）

唯一官方 B-transformer 正例：S1-SMALL-COS **581973 HO 0.390**（E0=50 + MOVE-T4 4，目标 `×5`）。不能把这个数迁到 H1/M1。

---

## 3. 新系列实际在搭什么（decoder 以外必须对齐的合同）

共同骨架：local conv k=5、8×256 slot、宽 256、4 层、FFN 512、`256→128→任务维`。  
时间核默认 CausalPE4（窗相对正弦 PE）；QueryAge16 只是对照。

| 合同 | H1 新系列 | M1 新系列 | 历史最优用的是 |
|---|---|---|---|
| 消费者 | 新训 Transformer | 新训 Transformer | **SPINT**（出厂或重训/冻结） |
| 活动身份 E0 | **C2 fused `[176,700]`**，concat 进 token | **B3 post_pool `[64,100]`**，concat 进 token | H1 Original：`[2,1024,176]` 校准特征；C2：early-pool 走 **SPINT identity 分支**；M1 Original：**不用** 这套 E0 |
| 功能 carrier | H-C 4，静态 bank | rSyn3 4，多来自 M10 / `rSyn3-refit` | C2 的 H-C 给 SPINT 吃；581727 的 rSyn3 给 **冻结 SPINT** 吃 |
| 校准预算 | 家族常用冻结 **M3**；训练可加 p=0.5 左历史清零 | 常用 **M10**；QueryAge 无 H1 那种 prefix | H1 Original **2 trial**；C2 训练 prefix-cycle、部署 3 trial |
| 窗 / 点 | W=700，目标 `velocity[s+699]` | W=100，末 bin EMG | Original/C2 本地对照已用同一 endpoint 定律；**校准物不同** |
| 目标尺度 | 训练 `×20`，评分 `/20` | `divisor=1`，16 维 EMG | H1 官方 loader **不再隐式 ×20**；早期 H1 temporal 在这里出过事故 |
| 观测 | cache：`load_nwb` 现分箱 `uint8` 计数 → float32 | 任务 EMG/神经合同与出厂 SPINT 需逐项核 | M1 官方曾拒非有限 float32（581980）；H1 官方超时过（naive 全窗） |
| 空间 preset | H1 显式 **unscaled-dot + local-balanced** | 标准 scaled-dot | Original/C2 **没有** 这套 slot 注意力 |
| 选择面 | 2,908 满窗选点 vs 20,325 完整流（含早段左补零） | 31,252 source-minival | Original 有点级 NPZ；C2 只有标量 receipt |

把 C2 的 E0 张量塞进 Transformer，**不等于在跑 C2**。C2 的成绩不能继承。

---

## 4. decoder 以外：按优先级核对

下面每一项都能在时间核不变时单独搞垮或抬高分数。负责搭建的人应逐项写「与历史最优是否同一科学量 / 同一面」。

### P0 — 不对齐就会把对照作废

1. **消费者不是同一类。** 历史最优 = SPINT。新系列 = Transformer。禁止把 0.96/0.89/0.65 写成「同一 decoder 换核」。只允许写「同一评分面、不同系统」。
2. **校准物不是同一物体。**  
   - H1：`[2,1024,176]` ≠ C2 early-pool 在 SPINT 里的用法 ≠ concat 用的 `[176,700]` E0。  
   - M1：出厂 SPINT 无 B3 E0；581727 的 B3S 给冻结 SPINT；新系列 B3-100 给 slot MLP。  
   同名 `E0` / `H-C` / `rSyn3` 不够。
3. **尺度桥。** H1 必须训练 `20y`、评分 `pred/20` 对 **native** velocity；H1 官方 loader 不隐式乘 20。M1 默认不除 20。早期 H1 近常数输出就是这类事故。验收：同一窗口上 `MSE(raw, 20y) = 400 × MSE(raw/20, y)`（是**比值** 400，不是差 400；差值恒为 `399 × MSE(raw/20, y)`，不是常数），且 pred std 不是目标的 <1%。
4. **同一 `(session, end)` 和同一 cache target。** H1 对照必须能 join Original 档案的 `(session_id, end)`。已证明 2,908 点 target 与 cache 一致。新预测必须落在这套坐标上，禁止另开一套 start 定律再比 0.96。
5. **观测 dtype / 有限性 / 内存布局。** 官方 H1/M1 走 Falcon `observe`。M1 581980 因非有限、非 C-contiguous float32 被拒。新 wrapper 必须 `ascontiguousarray(..., float32)` 并拒 NaN。禁止默认假设 cache 的 float32 计数 ≡ 官方 `observe` 张量。

### P1 — 常被误写成「核不行」

6. **校准预算。** Original 2 trial vs 家族 M3 vs C2 的 M7→M3 cycle。预算不同，identity 方差不同。
7. **训练面 ≠ 评分面。** 新 H1 训练 p=0.5 左清零 + unit dropout；选择是满窗、eval、无 dropout。完整流早段左补零。选点 0.32 和完整面不是同一个数。
8. **H1 空间 preset。** unscaled-dot、√45 / local-balance 是前端算子，不是 CausalPE。M1/M2 没有这一套。
9. **E0 维和用法。** H1 token 第一层 720 维里约 700 维是静态 E0。E0 尺度/列义错了，slot 注意力会吃噪声。静态折叠（`e_static.py`）只在线性可分解且 bank 正确时等价。
10. **M1 identity 来源。** 只许抽 `student.id_encoder`，禁止把 B3/Sfix 的 decoder 权重偷进新网却仍对 Original 0.809。Sfix 0.828 不是 Original。
11. **官方汇总 ≠ 本地 pooled。** 安装版 evaluator 是「每 session 的 variance-weighted R² 再平均」。本地 formal 是全点 pooled concat。和 581920/578474/578244 对读时必须换算或并列，不能直接减。

### P2 — 未闭合、不要当已证伪

12. **原始 NWB → cache 独立重分箱。** 本地路径保持 `load_nwb` 的数组，但没有离开 pynwb 的 `searchsorted` 重建。解释不了「同一 cache 上 SPINT 0.96、Transformer 0.32」，仍是科学对齐的剩余洞。
13. **单位列顺序。** loader 按 `units` DataFrame 行序。新系列与出厂 SPINT 必须同一 176/64 列序。
14. **eval_mask。** 历史窗里的 bin 不必都 eval-valid；只要求 endpoint 有效。完整流评分会扫所有 mask 点。
15. **H1 通道语义 `tx,ty,tz,rx,g1,g2,g3`。** 禁止打乱 7 列再和 Original NPZ 比。

---

## 5. 已经不像、不要再当主因去修

在**当前** formal/cache 路径上，审计已排除：

- 选点窗 off-by-one（不是 `+698/+700`）
- 选点窗误左补零
- 训练 `×20`、选择目标仍 `×20` 这种简单单位双乘
- 本地 cache 175/177 通道或 mask 未开
- Original 在另一套 label 上刷 0.96（2,908 点 target 与 cache 逐点相同）

这些关掉之后，**优先查第 4 节 P0/P1**，不要先改 CausalPE。

---

## 6. 给搭建 agent 的最短工作方式

每接到一个 H1/M1 数字，先填这张表，缺一行就禁止宣称「对齐了历史最优」或「核不行」：

| 问 | 必须写明 |
|---|---|
| 这是哪个系统？ | Original / C2 / 581727 / Sfix / 新 B-transformer（写时间核名） |
| 消费者？ | SPINT 还是 8-slot Transformer |
| 校准物？ | 形状、trial 数、估计算子、hash |
| 评分面？ | 点数、endpoint 定律、cache SHA、pooled 还是 session-mean |
| 尺度？ | `×20`/`/20`/`/5`/`1`，pred std / target std |
| 和历史最优只差了哪一项？ | 只许一项；多项就不是消融 |

**最值钱的对照（尚未做，需单独授权）：**  
同一 8-slot 骨架、**不动时间核**，只把 H1 的校准物换成 C2/Original 真正使用的那种 identity 用法（或证明 concat `[176,700]` 与 C2 early-pool 在 SPINT 里是同一科学量）。这比再训一套 CausalPE 更能回答「是不是 decoder 以外没对齐」。

权威入口：

- H1 坐标/cache：[AUDIT_H1_LOW_R2_COORDINATE_ALIGNMENT_20260906.md](AUDIT_H1_LOW_R2_COORDINATE_ALIGNMENT_20260906.md)
- H1 尺度：[AUDIT_H1_LOW_R2_NUMERICAL_CONTRACT_20260906.md](AUDIT_H1_LOW_R2_NUMERICAL_CONTRACT_20260906.md)
- H1 对照面：[AUDIT_H1_QUERYAGE_COMPARISON_SURFACES_20260906.md](AUDIT_H1_QUERYAGE_COMPARISON_SURFACES_20260906.md)
- 新系列骨架例外：[AUDIT_QUERYAGE_THREE_TASK_NETWORK_FAMILY_20260906.md](AUDIT_QUERYAGE_THREE_TASK_NETWORK_FAMILY_20260906.md)
- C2 不是 B：[FREEZE_ASTRA_FLAT_ROUTE_OPERATOR_FAMILY_20260906.md](FREEZE_ASTRA_FLAT_ROUTE_OPERATOR_FAMILY_20260906.md)
