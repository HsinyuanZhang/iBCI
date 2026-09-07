# CPU 筛线终结记录：五个杆、四个 artifact、两条尺子的局限

**状态：CPU 筛线终结记录**
**日期：2026-08-09**
**范围：** H1 CarrierID 线的全部 CPU 侧筛选，含 lag、pooling/shrinkage、H-NF、以及 `[a,c]` 内容杆的重叠门。

> **证据等级：本轮全部为 CPU 侧 source-only 筛选，零 GPU arm、零新 R² 数字。**
> 产出的 receipt 由本会话的执行 agent 生成并带 SHA-256，但**未经过项目的正式双审与 `0444` 封存链**。
> 引用时必须标注为 development screen evidence，不得与封存收据同级使用。
> 所有 target 侧读数均为 **same-checkpoint forward-only 诊断、non-routing**，仅使用已被既有封存收据打开过的 endpoint。

---

## 1. 一句话

五个杆全部关闭或降级，**没有产生任何 R² 提升**；真正的产出是四个方法学 artifact 的暴露、一条被证伪的前提、以及一条被量化但未经 decoder 校准的线索。

---

## 2. 判定总表

| 杆 | 判定 | 关键数字 |
|---|---|---|
| 跨 session 先验 pooling | ⛔ 关闭 | `pool_loo − eb_global = −0.0096`（均值）/ `−0.0076`（中位） |
| 收缩形式 | ⛔ 关闭 | 不可灌水指标上三臂互差 `+0.003` 以内 |
| 逐通道 lag（Arm E） | ⛔ 关闭 | 池化秩检验对 null 呈均匀；`M=4` split-half 稳定性不超 null |
| 群体 lag（Arm P） | ⛔ 关闭（**artifact**） | 固定样本集后峰移至 `0–9` bins，LOO 降为 `7/11`、`p=0.55` |
| H-NF 无标签 carrier | 🔒 封存降级、未跑 target decoder | fold-0 prior scale 约 `+0.003`，不是实测结果或上限；对手是 `H-C0` 而非 SPINT |
| L-A 逐通道编码形式 | ⛔ 当前形式关闭 | W 列中位残差 `0.364` ≈ N4 非速率列 `0.368` |
| L-C 噪声归一化 `[W,b]/σ` | 🟡 **判定撤回，待非线性对照** | 逐列残差 `0.511/0.580/0.664`，明显高于 L-A |
| L-B `U` 的选择 | 🟢 **被指向且已量化** | 原始 W 中位 `0.449` vs U 投影后 `0.364`，差 `+0.085` |
| L-D 乘性消费 | ⚪ 未触碰 | 唯一未测的结构杆 |

---

## 3. 实测数据事实（含对既有文档的更正）

### 3.1 `M=4` 是严重超定，不是数据饥饿 —— **更正**

| 量 | 实测 |
|---|---|
| trial 时长中位数 | `151` blocks = **`15.1` 秒**（p10/p90 = `134/183`） |
| 每条 recording 的 trial 数 | `8–15` |
| `M=4` support 的 block 数 | `558–696`，约 **60 秒**校准 |
| carrier 设计矩阵 | 约 `627 × 17` |

[`HANDOFF_LEARNED_CARRIER_ESTIMATOR_20260808.md`](HANDOFF_LEARNED_CARRIER_ESTIMATOR_20260808.md) §2 的
"nearly underdetermined" **是错的**。

**下游后果：** 整条"改进估计器"的路线（收缩、pooling、lag、EST4 的正则部分）建立在这个错误前提上，
所以它们全部为零是**可预测的**，不是运气。这把 pooling/shrinkage 的阴性从"测出来是零"升级为
"有解释的零"。

同时它**降低了** §6 leakage oracle 的预期价值：该 oracle 测"carrier 被完美估计会怎样"，而
`627` 样本估 `17` 参数已接近完美估计。

### 3.2 velocity-finiteness 掩码不丢弃任何 bin

`eval_mask & isfinite(neural)` 在 13 条 recording 上共 `132,476` bins；追加 `isfinite(velocity)`
后仍为 `132,476`，丢弃 **`0` bins（`0.0000%`）**。

因此神经-only 支撑集与 `h1_m4_eb_pilot._trial_blocks` 的支撑集当前逐 bin 相同。该等价性必须被
**断言而非假设**——若将来出现非有限 velocity，两个臂会静默分叉。

### 3.3 channel 66 是永久坏电极

在全部 13 条 recording 上整段 spike 总和为 `0`（`1/176`）。既有臂不受影响（ridge 对全零行返回零系数），
但任何对退化通道 fail-closed 的新估计器会在**每一条** recording 上停机。需要 source 侧声明的冻结坏道表。

### 3.4 尺度失衡 `326,463 ×`

`b` 在 pooled descriptor 矩阵中的方差 `240.25`，`W` 各列平均方差 `0.0007`。
逐列标准差 `s_j = [0.050, 0.039, 0.027, 15.50]`，`b` 列约为 W 投影列的 **300 倍**。

---

## 4. 逐筛结果

### 4.1 Lag

**Arm E（逐通道）** — 阴性。round-1 报的 `frac > null q95 = 0.33–0.51`（零假设期望 `0.05`）
由**仅 8 个 null 偏移**估出，`q95` 属外推，**已被取代**。round-2 用 ≥40 偏移与池化秩检验重估，
结果对 null 呈均匀；`M=4` 预算下 `τ*_i` 的 split-half 稳定性不超其 null。整段记录与 `M=4` 两种设定一致。

跨记录 `τ*` 稳定性低（`r=0.12`）**不构成**否决理由，因为部署时逐通道权重是逐 session 重估的。
真正的否决来自秩检验与 split-half。

**Arm P（群体）** — artifact。演进过程本身是本轮最重要的方法学案例：

| 轮次 | 读数 | 处置 |
|---|---|---|
| round-2 | `τ* = +13..+18` bins（`260–360 ms`），10/11 同向 | 疑似真实 neural lead |
| round-3 去趋势 | 峰保留，LOO `10/11`、`p=0.012` | 看似坐实 |
| round-4 三控制 | **全部推翻** | 关闭 |

三个控制：

- **C2 固定样本集**（τ ∈ `[−10,+40]` 上都有效的 block 交集，保留约 93%）：峰从 `8–24` 移至 `0–9` bins，
  LOO 符号从 `10/11` 降为 `7/11`、`p=0.55`；
- **C1 null LOO 分布**：真实 delta `+0.0044` 落在 null delta `+0.0037 ~ +0.0040` 之内，
  真实正计数 `7` 与 null 正计数 `7` **在每个偏移上都相等**；
- **C3 样本外 R²**：real 与 null 在**每一个** lag 上相差 `< 0.001`，不分离。

结论：该峰由**样本内 R² 膨胀**与**随 τ 变化的样本集**共同造成。若无这三个控制，一个
"260–360 ms 真实神经 lead"的假阳性会进入 GPU 队列。

### 4.2 Pooling 与收缩形式

- 跨 session 先验：`pool_loo − eb_global = −0.0096`（均值）、`−0.0076`（中位），配对、公式匹配、仅先验来源不同。**先验来源这条子杆关闭。**
- 收缩形式：round-1 的 `raw 0.365 → eb_global 0.777`（`+0.412`）**是指标 artifact**。
  在不可灌水指标（对同记录高精度参考拟合的 cosine）上，三个收缩臂互差 `+0.003` 以内。**收缩形式无可提升。**

### 4.3 H-NF 无标签 carrier

判为 `DEPRIORITIZED_SEALED_NOT_ON_CRITICAL_PATH`，但没有 target-behavior decoder R²。理由：
它的对手是 `H-C0 = 0.48662` 而非 SPINT `H-S = 0.49683`——紧凑无行为 identity 的结果已经存在；
一个冻结四维逐通道统计量叠加在已学到的 32 维 activity path 之上，`H-LS − H-C0 = +0.0133`
只提供 fold-0 的 prior scale。由此写出的约 `+0.003` 是预期增量，不是实测 H-NF gain 或 ceiling。

模块保留并加状态横幅，六条已发现缺陷（D1–D6）随横幅留档，避免复活时重踩。

### 4.4 重叠门与 `[a,c]` 内容杆

**归一化审计（重要）：** 六个 carrier form 在 **4 维输出上均无任何归一化**。
（此前的诊断说"用错归一化家族"，方向对、机制错。）
H1 生产管线用的是**单个全局 RMS 标量**，它**修不了列间不均衡**——L-A 若上 GPU，必须先把逐列归一化
接进生产路径并重新验证 `H-C0` 严格为零，而该路径目前处于矩阵中途的冻结状态。

**秩塌缩与修复：** 逐列尺度归一化（除以 source 逐列 SD，**不中心化**，故 `H-C0` 仍严格为零）后：

| | 修复前第一成分占比 | 修复后 |
|---|---|---|
| L-A | `0.9948` | `0.3380` |
| L-C | `0.9942` | `0.3981` |

`comps_90`：H1 = **4**，N4 = **1**，L-A/L-C 修复后 = 4。reference cosine 不再恒等于 `1.000`。

**重叠门（参照 = `H-C0` activity path；`H-C` 参照偏差约 `0.006`，方向为偏袒现任）：**

| carrier | pooled 残差 | 逐列残差（中位） |
|---|---:|---|
| H1 生产 carrier（正对照） | `0.868` | `0.864 / 0.851 / 0.864 / 0.906` |
| L-C `[W,b]/σ` | `0.467` | W `0.511 / 0.580 / 0.664`，b `0.088` |
| L-A `[W,b]` | `0.293` | W `0.281 / 0.364 / 0.513`，b `0.001` |
| N4（负对照） | `0.249` | mean_rate `0.001`，Fano `0.368`，lag1 `0.225`，**pop `0.386`** |

pooled 统计量为**逐列 R² 的均值**（`h1_content_lever_screen.py:715`），而 R² 对列尺度不变，
故逐列归一化从未改变它——修复前的 pooled 数字一直有效。

**自检通过：** L-A 的 `b` 列与 N4 的 `mean_rate` 列残差均为 `0.001`（activity path 确实握有发放率）；
H1 四列均在 `0.85–0.91`。

**原始 W（U 投影之前，七列）：** `0.542 / 0.531 / 0.594 / 0.408 / 0.286 / 0.320 / 0.449`，中位 `0.449`。
U 投影后中位 `0.364`，**差 `+0.085`**。`U` 按运动学方差选，不按内容选。

**配对检验：** L-A 三个 W 列 vs N4 三个非速率列，逐记录配对：`9/11` 正、`p = 0.0654`、中位配对差 `+0.058`。
n=11 时 `9/11` 的双侧 p 为 `0.0654`，`10/11` 为 `0.0117`——**判定挂在 9 与 10 的差别上，属欠功效的正向趋势，不是干净的打平。**

**归一化比值（separability/drift）：** L-A `106.75`，L-C `103.94`，基本打平。

---

## 5. 被抓出的四个方法学 artifact

这四条是本轮真正的产出，均已在实际数据上验证：

| # | artifact | 若未发现会怎样 |
|---|---|---|
| 1 | **split-half cosine 被收缩灌水** | 会把 `+0.412` 报成收缩形式的重大改进 |
| 2 | **argmax-over-lags 的选择偏倚** | 比值 `> 1` 必然出现，`R²(τ*)/R²(0)` 无 null 时不可读 |
| 3 | **随 τ 变化的样本集** | 群体 lag 的假阳性会带着 `p=0.012` 进入 GPU 队列 |
| 4 | **缺逐列归一化导致的秩 1 塌缩** | 四维契约实际只用一维，且那一维是基线率——而 `B4 = 0.287 < Z4 = 0.326`，**上 GPU 的预期是负的** |

---

## 6. 这两把尺子本身的局限（引用前必读）

**重叠门只能证伪，不能证实。** N4 的非速率列残差为 `0.368`，而 N4 在 M2 held-out 上实际失败
（`N4 − NS4 = +0.001588`，3/6 session）。所以：

- 残差 ≈ 0 → 该候选不携带 activity path 之外的内容，**可排除**；
- 残差 > 0 → **不代表有用**。`0.368`（N4 非速率列）是实用底线，低于它属于已知失败区间。

**重叠门是线性检验。** `W/σ` 是一个**比值**——非线性重组。即使 activity path 同时握有 `W` 类与
`σ` 类信息，线性回归也造不出它们的商。因此 **L-C 的高残差可能部分是"非线性通过"的假阳性**，
在相信它之前需要非线性对照。这也是 L-C 判定被撤回但未重开的原因。

**2026-08-10 后续校准已经完成，但没有建立预测器。** 使用精确 H1 production carrier、同一
H-C0 activity reference 和同协议 `{H-C,H-RS,H-LS}` 三点后，`(residual, gain-over-H-C0)` 为
`(.804819,+.038895)/(.813399,-.027224)/(.825789,+.013280)`；顺序非单调，exact Spearman
`rho=-0.5,p=1.0`。加入明确标为跨 pipeline sensitivity 的 N4 后为 `rho=0,p=1.0`。因此不能从
L-A=`0.364` 或 L-C=`0.580` 外推 decoder gain，也不能据此放行 L-B。SUA/RT 的活动参考因冻结
Hydra/config provenance 尚无法无猜测重建，没有被伪造进统计量；这是一项范围限制，不是补点理由。

**结论仍是：重叠门只能证伪，不能证实。** 它现在不仅“未校准”，而且在可严格重建的 H1
同协议三点上校准为非单调。所有候选读数仍是代理量；“能不能显著提升 R²”只能由匹配的 GPU
candidate/control 回答。权威 receipt SHA 为 `624557b1...036cb7`、mode `0444`，root byte-for-byte
复现并复跑修正后的合同得到 `21 passed`。

---

## 7. 对既有文档的更正清单

| # | 位置 | 更正 |
|---|---|---|
| 1 | `HANDOFF_LEARNED_CARRIER_ESTIMATOR_20260808.md` §2 | "nearly underdetermined" 错误；实为 `627×17` 严重超定 |
| 2 | `H1_CARRIERID_EVIDENCE_HANDOFF_20260808.md` §4 | 跨日期"3–4x"由**五个日期中的前三个**得出，已被取代。全五日期等权均值 `+0.056287` 对同日期 `+0.026`，实际约 **2.2 倍** |
| 3 | lag screen round-1 | `frac > null q95 = 0.33–0.51` 由 8 个偏移估出，**已被取代**，以 round-2 池化秩检验为准 |
| 4 | 本会话 round-C-FIX2 的 L-C 关闭判定 | **撤回**。当时依据是归一化比值打平，但逐列内容显示 L-C 明显高于 L-A；L-C 状态改为"待非线性对照" |
| 5 | 本会话 C-FINAL 报告末条 | `la_old` 与 `la_frozen` 的逐列 R² 是**同一组值换序**，故旧 `U` 把 `b` **隔在独立一列、未混入其他列**，与 C-FIX 的 `b` 载荷 `1.0000` 一致。F1 是空操作这一结论不变 |
| 6 | 关于归一化的早期诊断 | 不是"用错归一化家族"，而是 4 维输出上**压根没有归一化** |

---

## 8. 仍然开放

| 项 | 状态 |
|---|---|
| **C1 漂移口径重报** | **已跑完，输出从未被读取**（receipt `a9ab7923...`）。同日期 `+0.026` vs 跨日期 `+0.056287`，零成本的口径修正 |
| **L-B（`U` 的选择）** | 被指向且量化（`+0.085`），但**在 P2 之前不可信**——若 `M=4` 的 4 个 trial 张不出 7 维运动学，则 `U` 什么都没丢，该读数作废 |
| **L-C** | 待非线性对照 |
| **L-D（乘性消费）** | 唯一未触碰的结构杆。机制理由独立于任务基匹配 |
| **群体结构载体** | 新候选，**结构性**通过 Q2（activity path 无任何跨通道操作），且无需标签故可用于 M1。协议见 [`AGENT_BRIEF_GENERALIZATION_P1_P3_20260809.md`](AGENT_BRIEF_GENERALIZATION_P1_P3_20260809.md) |
| **identity reliance 诊断** | 后续 M1 `full-zero=+1.848512` 已完成；只证明 trained identity 承重，不是 attainable headroom 或跨任务规律 |
| **重叠门校准** | 后续严格三点校准已完成但无功效：`n=3`、span `0.020970`、两个 corruption controls；不是 measured negative |
| **外部范围** | subject-M same-Dandiset cross-animal 已完成；旧 sub-C one-shot formal slot 消耗后无结果，未重跑 |

---

## 9. 收据

| 项 | 路径 / SHA-256 |
|---|---|
| C1 漂移口径重报 | `a9ab7923...` |
| C-FIX2 逐列归一化 | `src/data/h1_content_lever_receipts/h1_content_lever_cfix2_v1.json`，`e08cd05d...` |
| C-FINAL 逐列残差 | `src/data/h1_content_lever_receipts/h1_content_lever_cfinal_v1.json`，`e3233519...` |
| lag / pooling 各轮 | 见 `SPINT-main/src/data/` 下对应模块目录 |
| `H-C0` 模型 state hash | `91b00622...`，全部前向前后不变 |

全部前向为 source-only、`map_location="cpu"`、无梯度；未打开 formal、minival 或 EvalAI。

---

## 10. 参考

- [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) — 2026-08-09 条目
- [`HANDOFF_AC_CONTENT_LEVERS_CPU_GPU_SPLIT_20260809.md`](HANDOFF_AC_CONTENT_LEVERS_CPU_GPU_SPLIT_20260809.md) — 四个内容杆与 CPU/GPU 划分
- [`HANDOFF_LABEL_FREE_CARRIER_AND_HEADROOM_20260809.md`](HANDOFF_LABEL_FREE_CARRIER_AND_HEADROOM_20260809.md) — H-NF 分析与两条件过滤器
- [`AGENT_BRIEF_GENERALIZATION_P1_P3_20260809.md`](AGENT_BRIEF_GENERALIZATION_P1_P3_20260809.md) — 下一轮 P1/P2/P3 协议
- [`T4_OPTIMIZATION_DIRECTIONS.md`](T4_OPTIMIZATION_DIRECTIONS.md) — kill criteria 与已关闭方向清单
