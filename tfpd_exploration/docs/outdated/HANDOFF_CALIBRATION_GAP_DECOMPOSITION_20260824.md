# Handoff: Calibration-Gap Decomposition route (M4/M10 少标签主线)

Date: 2026-08-24
Status: 路线 handoff。定义零成本裁决批次 + 五条路径 + 操作侧修正。代码基础已在本仓库
新子包 `src/calibration_gap_v1/` 落地（见 §7）。外部建议原文（第三份分析）存档于
`EXTERNAL_ANALYSIS_GAP_DECOMPOSITION_20260824.md`（逐字保留）。前置：
`HANDOFF_CONSOLIDATED_ROADMAP_REVIEW_20260823.md` §7/§8（诊断终局 + CBM-D 判决）。

---

## 1. 路线主张（一段话）

M30 载体近饱和（oracle 余量 0.0163 ≈ M4 的 1/20），M4 有 0.3252 的实测天花板缺口，
M10 居中（0.1494，载体/活动 ≈ 50/50）。外部缺口在 external-15 上分解为
**载体估计项 78%（0.2547）/ 校准活动项 22%（0.0705）**；within 面 ~50/50。
CBM-D 的负结果（within +0.11 / external −0.01）由此被追认为分解的证据：它攻击的是
22% 项、在活动项占半的 within 面出了唯一真效应。**主指标改为
"各预算下 oracle 载体缺口关闭比例"，M30 近零是必须命中的机制预言而非遗憾。**

## 2. 零成本裁决批次（先跑，一并裁决；全部冻结权重推理或纯计算）

| # | 项 | 产出 | 模块 |
|---|---|---|---|
| Z1 | **honest-M4/M10/M30 oracle 缺失格**（全会话标签 + M-预算校准活动）+ 全天花板曲线 | 0.2547 是上界还是实界 | `oracle_cells.py` |
| Z2 | **覆盖度回归**：support Gram 条件数 / 方向计数 / PR → 逐 session R² 与 D-opt/ridge 增量；含 20150617 离群诊断 | D-opt 为何只 M4 有效的量化解释 + 失败 session 可诊断性 | `coverage.py`（协变量已在诊断收据逐 session 记录里） |
| Z3 | **逐 DoF 归因**：M4 残差缺口集中于 support 未覆盖的行为方向 = 机制可证伪测试 | 谱-增益对齐表 | `coverage.py` |
| Z4 | **U 稳定性诊断**：M30 活动降采样至 M4/M10，主子空间角 | Path 1 可行性上界（U 饥饿程度） | `subspace_stability.py` |
| Z5 | **H1 query-oracle 完整 surface 复验**（现有 −0.0029 为 CPU screen 口径） | Path 1 的 H1 战场开/关 | 现有 H1 机架 |
| Z6 | **FALCON 合同阅读**（评测流无标签活动是否 transductive；cue 元数据是否算预算） | Path 1/4 的门 | `contract_notes.md` |

## 3. 五条路径（裁决后只投一条 GPU/重计算线）

**P1 无标签活动子空间限制载体**：C ≈ U·A，U = 目标会话神经协方差 top-k 主子空间
（无标签），A = k×d 行为对齐（标签只花在这里）。各向异性、数据对齐的正则替换
各向同性向零收缩的 ridge。预言：M4 大增益、M10 小、M30≈0，最大增益在最病态 session。
先决条件：Z4（U 稳定性）+ Z6（U 需要无标签目标活动——严格 total-calibration 制度下
只有 M 个校准 trial 可用，U 本身挨饿；评测流输入 = transductive = 合同问题）+
Z5（H1 战场裁决）。相关工作定位必须覆盖 Sadtler 2018 intrinsic-manifold 约束、
Degenhart 2020、Farshchian ADAN、NoMAD——先做 related-work pass 再动工。

**P2 估计器层（折入 P1 同一机架）**：
- **强半**——跨 unit 池化 λ 选择（86 units 共享一个 λ = 1 参数问题 86 倍数据）+
  以无标签活动重构（而非行为残差）为选择准则；正对 GCV 失效（λ-性能相关
  −0.659/−0.702）与 20150617 灾难格（−0.3135 vs 其余 +0.078）。
- **弱半**——波形/ISR 特征预测先验均值：270-cell 矩阵已证波形类特征近乎不携调谐
  信息（N4 全负），预期先验退化为近零信息 → EB 塌回 ridge。便宜可顺手测，预期压低。

**P3 覆盖度作为预测协变量**（= Z2/Z3 的产出本身，零 GPU）。

**P4 活动项的无标签统计量**：活动项 ~0.07 在 M4/M10 恒定、M30 消失——"统计量来自
过少活动"的签名。归一化/白化统计可用无标签评测流替代 M-trial 校准估计，零标签成本。
门 = Z6 合同。放行则 M4/M10 各 ~0.07 免费；不放行则如实记账为诚实部署制度的
不可约成本（本身可发表）。**M10 专属含义：50/50 分解下这份合同阅读值 ~0.07，
等于全部载体工作——与 P1 同优先级，不是侧任务。**

**P5 门控递推载体精化（测试时）**：RLS/Kalman + 伪标签，Mahalanobis 门 + 仅在 P1
子空间内更新 + 时间平滑伪标签 + bootstrap 一致性置信 + fail-closed 单调性
（永不低于冻结载体基线）。排序最后：P1+P4 把 M4 抬到 ~0.30 后伪标签才可信。

## 4. 操作侧修正（相对外部原文）

1. **H1 冲突（外部原文漏掉）**：P1 的旗舰叙事（86×7=600 参数）在 H1，但 H1
   query-oracle 已示 −0.0029（完美载体无增益）→ 子空间估计受 oracle 上界约束，
   H1 不会获益。出路二选一：Z5 复验推翻该负结果，或接受 P1 现实战场仅 sub-M
   （参数缩减 180→~20，9×，有意义但非旗舰）。
2. **P1↔P4 耦合（外部原文漏掉）**：见 P1 先决条件——U 的估计依赖 Z6 合同答案。
3. **P2 拆强弱两半**（见上）。
4. **M10 的 50/50**（见 P4）。
5. 主指标与预言结构采纳外部原文（缺口关闭比例 + M30 近零预言）。
6. 禁令维持：不加架构/损失/训练配方（CBM-D 已用 1.6M 步买单）；
   held-in 改善不得再充当 external 改善的代理（§8.3 条款）。

## 5. 执行序

```
Z1..Z6（零成本批次，src/calibration_gap_v1/ 已建基础）
  → 裁决：载体项实界？覆盖可预测？U 可估？H1 战场开？合同放行？
  → P1+P2强半（若 Z1 说载体项实界且 Z4 说 U 稳定）
  → P4（若 Z6 放行；与 P1 并行——纯估计器侧）
  → P5（最后）
```

## 6. 收据绑定（ledger.py 全部 SHA 验证加载，禁硬编码）

| 收据 | SHA256（前 16） |
|---|---|
| `results/low_cost_calibration_diagnostics_v3/receipt.json` | `c1adfd9f06d737a4` |
| `results/calibration_budget_comparators_v1/receipt.json` | `0ec107cc95cfb806` |
| `results/calibration_budget_protocol_factorial_v2/receipt.json` | `ce283aa7d046d575` |
| `results/original_spint_short_budget_v2/receipt.json` | `526dc11460f44d67674287762273da74265fe05fa5aa0c10bab953b65fb06a68` |
| `results/ridge_t4_lambda_curve_v1/receipt.json` | `e82d917b348be09b` |
| `results/calibration_budget_marginalized_cell_d_seed42_v3/terminal.json` | `8e5c807dbf961cbd` |
| `results/calibration_budget_marginalized_score_v2/receipt.json` | `09ffbd837be8529e` |

阶梯格指针（已逐字段核验）：
- C0/C1/C2/C3（M30 activity）→ diagnostics v3 `cells[{budget,surface,support}].summary.equal_session_mean_r2`
- label-limited / total-calibration × OLS/ridge × chronological/D-opt → factorial v2
  `cells[{budget,estimator,regime,support,surface}]`（0.1902 = M4/ridge/doptimal/label_limited；
  0.1197 = M4/ridge/doptimal/total_selected）
- 经典基线（trial_rate_ridge / dense_w50_ridge / population_vector）→ comparators v1
  `regime=classical_prefix_fit`

## 7. 代码基础（新子包 `src/calibration_gap_v1/`）

| 模块 | 状态 | 内容 |
|---|---|---|
| `ledger.py` | **完整可用** | 四收据 SHA 验证加载、描述符寻格、缺口分解计算（全部从收据加载值计算）、缺失格登记（honest oracle 三格标 MISSING + 精确 spec） |
| `coverage.py` | **完整可用** | 诊断收据逐 session 覆盖协变量提取 + 相关/回归 + 20150617 离群诊断 |
| `oracle_cells.py` | 规格 scaffold | Z1 的精确 runbook（支持集/活动预算矩阵 + 与诊断机架的对接点）；实现待派工 |
| `subspace_stability.py` | 规格 scaffold | Z4 的降采样-主子空间角协议；实现待派工 |
| `contract_notes.md` | 待填 | Z6 的合同问题清单与裁决记录 |
| `tests/test_calibration_gap_v1.py` | 完整可用 | SHA 篡改拒载、分解自洽、覆盖回归、缺失格检测 |

实现约定：纯 numpy/json/hashlib（无 torch 依赖，CPU 即可）；收据数字一律加载非硬编码；
新增任何结果走 `tfpd_lane/receipt.py` 事务写。

## 8. Zero-cost gate batch RESULTS (2026-08-25, CPU-only; receipts results/calibration_gap_v1/)

**Z1 honest-oracle (6 cells; C3 anchor reproduced to 1.2e-7)**: external M4/M10/M30 =
**0.2413 / 0.3885 / 0.4449**. Pre-registered "~0.25 branch" fired: the M4 bottleneck is the
ACTIVITY pathway, not the carrier. Matched-activity decomposition REVERSES the earlier
bookkeeping (the 78/22 split was an M30-activity artifact):
- M4 external 0.325 = **0.2036 activity (63%) + 0.1216 carrier (37%)**
- M10 0.147 = 0.056 activity (38%) + 0.090 carrier (62%); M30 = carrier-only 0.016
- within M4 0.299 = 0.214 activity (72%) + 0.085 carrier
P1's true M4 ceiling is 0.1216, not 0.2547.

**Z4 U-stability**: median θ_max(U_M, U_ref) at k=6 = 56.1° (M4) / 38.4° (M10) vs
thresholds 35°/20° → **U starves in the strict regime; P1 NO-GO strict** (mean angles far
lower — leading directions align, deeper top-k rotate away; LW shrinkage changes nothing).

**Z6 contract verdicts (cited in src/calibration_gap_v1/contract_notes.md)**:
- Q1 eval-stream neural data: **PERMITTED** (FSU/TTA class; causal accumulation mandatory;
  must be declared transductive, never mixed with the strict column)
- Q2 cue metadata: **NOT free — label spending**; cue-conditional carriers and wider
  unlabeled D-opt pools fail closed; D-opt stays legal only as label allocation inside M
- Q3 few-shot budget = the released paired calibration split; eval-stream = separate TTA axis

**Go/No-Go**: P4 GO (main line; targets the 0.2036 activity term; recovery bounds
+0.0705 ext / +0.1546 within @M4 leaked-label ceiling). P1 NO-GO strict, demoted behind
P4; only a transductive-U re-scope (TTA column, k≤4) could revive it, after Z5/H1.
Cue carrier NO-GO. P4 launched immediately (CPU, P4a streaming-normalization +
P4b streaming-identity variants, causal discipline audited).

## 9. P4 RESULTS (2026-08-25, CPU; receipt results/calibration_gap_v1/p4_stream_stats.json)

**Decisively negative, both variants** (anchors perfect: Z1 63/63 prediction SHA bit-exact;
strict 0.1197/0.2955/0.4286 reproduced to 1.2e-7):

| external | strict (FSS) | P4a stats (TTA) | P4b volume (TTA) |
|---|---:|---:|---:|
| M4 | 0.1197 | 0.1020 (−0.018) | **0.0514 (−0.068, 3/15)** |
| M10 | 0.2955 | 0.2869 (−0.009) | 0.1701 (−0.125) |
| M30 | 0.4286 | 0.4008 (−0.028) | 0.2289 (−0.200) |

**Diagnosis (the valuable part): the activity term is a STRUCTURE term, not a
volume/statistics term.** Harm is monotone in budget (worst at M30 where the mechanism
predicted ~0 gain); P4a's per-unit rate check shows M-trial calibration rates already match
streaming rates (median gain ≈0.97) — the deficit is not scale. The identity pathway needs
REACH-STRUCTURED activity (time-warped, cue-locked 100-bin trials); the continuous eval
stream is ITI-heavy and unstructured (FALCON strips trial structure by design, Z6-Q2) —
pooling it through the frozen B3S trial-mean DILUTES the identity (warm start bit-exact,
monotone degradation as the stream pool grows).

**Route consequences**: the "~0.07 free" does not exist; the honest-deployment activity
cost is irreducible under the frozen encoder. The M4 mining program is nearly closed:
activity term (63% of M4 gap) irreducible; carrier term (37% = 0.1216) has P1 dead
(strict), leaving only P2a/b estimator slices and P5 tracking (small expected). Paper
point: FALCON's own no-trial-structure evaluation regime cannot transductively supply
what trial-structured calibration does — the few-shot calibration split is the ONLY
source of reach-structured activity, which is WHY short calibration breaks and TTA
cannot substitute.
