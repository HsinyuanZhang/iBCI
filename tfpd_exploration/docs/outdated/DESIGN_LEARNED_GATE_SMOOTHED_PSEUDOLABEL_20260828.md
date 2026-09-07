# Design: Learned-Gated T4 Updates with Smoothed Pseudo-Labels（供外部审核）

Date: 2026-08-28 · Status: DESIGN v2（P1 平滑探针结果已填入并重路由，可送审）· Route: M2/SUA CDM 家族继任
作者：操作侧；审核对象：全部设计点，尤其 §3 两个核心机制、§4 P1 三个设计修正、§6 分支

---

## 0. 一句话主张

在 activity-only CDM（当前最优：M4 0.2265 / M10 0.3888 external，sealed Cell-D 冻结基座）
之上，用**平滑先验构造的伪标签**驱动**可学习的门控**，有选择地更新 T4 载体参数
`[a,c,m,b]`，目标是吃下载体侧残余池（M4 ≈ 0.218 / M10 ≈ 0.056，全泄漏 oracle 0.4449 界定），
同时不重蹈已封死的"手工门 + 原始伪标签"覆辙（实测净负：M4 −0.004 / M10 −0.038 / M30 −0.059）。

## 1. 证据地基（全部 sealed，审核者可查）

| 事实 | 数值 | 收据 |
|---|---|---|
| activity-only CDM | M4 0.2265 / M10 0.3888 ext | activity_only_quick_v2 |
| 全泄漏载体 oracle（满活动） | 0.4449 | diagnostics v3 C3 |
| 载体侧残余池 | M4 0.218 / M10 0.056 | 上两行差 |
| 手工门家族判决 | 信任门 0.1911 < 精度门 0.2226 < 删除 0.2265（M4）| V8/V2/activity-only |
| 伪标签原始版净贡献 | 全预算负 | 同上 |
| 循环性障碍 | 伪标签误差=当前载体误差的系统映照 | 机制分析+实测 |
| 训练侧改动基率 | 0/9 | S2/C/W/PIRG×2/TF-SR/CBM-D/CS-WG |
| 原生 M2 / H1 活动机制复现 | +0.068 (M2 M4) / +0.05 (H1 5/5) | audit §2.5 / h1 compare v2 |

## 2. 系统总览

```
冻结 sealed Cell-D（SWA 626f65d8…，永不重训）
├── 活动臂（不改动，保底）：B3S 因果 FIFO，M→30 无标签增长
└── 载体臂（本设计）：伪标签驱动的 [a,c,m,b] 更新
     伪标签构造 ← 平滑先验（§3.2）
     接受决策   ← 可学习门（§3.1）
     安全地板   ← fail-closed 逐 session 回退（§5.3）
```

## 3. 两个核心机制（审核重点）

### 3.1 可学习门控：用源域真标签监督"接受/拒绝"策略

**与已死的手工门的本质区别**：源会话上真标签免费——可以完整模拟部署过程并对每个
候选更新计算"接受了会怎样"（真值对照），门控因此有**直接对应部署目标的有监督信号**。
精度门是该策略的单特征手工特例；手工特例输了不等于策略空间没东西可学。

**训练数据构造（源侧模拟，解码器全程冻结）**：
```
for 源会话 s（严格 LOSO 折）:
    用 CDM 状态机模拟部署（M-trial 起步、活动 FIFO 增长）
    for 每个完成的 trial j（产生一个载体更新候选）:
        特征向量 x_j（§3.3，全部无标签可观测）
        真值结果 y_j：用真标签在同一支持集∪{trial j} 重拟合参照载体 β*_j；
                     y_j = 更新后载体向 β*_j 的距离变化（正=改善）
    样本集 D_s = {(x_j, y_j)}
```

**策略模型（刻意小）**：逻辑回归或深度 ≤2 的树，特征 5-8 个。不用深网络——
(a) 防 widget 基率；(b) 论文需要特征重要性可读（"门在看什么"必须可回答）。
输出：接受概率 p_j → 二值决策或接受权重 w_j = p_j。

### 3.2 平滑先验构造伪标签：用运动连续性去噪，而不是用原始解码输出

**动机**：行为平滑、解码器逐窗抖动——相邻窗预测在信号上相关、在噪声上独立。
原始逐窗伪标签把全部时间噪声喂进载体回归；平滑版是**方差缩减后的伪标签**。
（注意与已失败的"时间平滑救伪标签更新"论证的区别：当时结论是平滑去不掉循环**偏差**——
本设计不否认这一点，而是把平滑定位为**门控特征 + 伪标签方差缩减**两个用途，
偏差问题交给门去筛。）

**构造分支（P1 探针后重排主从）**：
- **B2 因果窗口族（主，探针实证胜出）**：当前窗与 trailing K-1 个过去窗的**各自末 bin
  输出**做加权平均（K4 均值 / α=0.25 EMA 为实测最优档）。数学上涵盖"平滑整轨迹再取
  末 bin"。探针实测：M10 +0.019 (13/15)、M30 +0.037~0.039 (15/15)，CI 全正。
- **B1 轨迹对齐（降级为非因果诊断）**：对被评分 bin 跨窗平均需读未来窗——部署不合法；
  且实证弱于 B2（上下文依赖误差结构主导）。仅作零延迟诊断上界保留。
- **B3 一致性加权（特征）**：不直接平滑，把"相邻窗预测一致性分数"作为门输入特征
  （f3，P1 已证其携带真实信号）。
- **预算自适应**：平滑增益随预算单调（M4 边缘/M10 中/M30 强）→ 伪标签构造按预算
  取 K：M4 用轻档（K2/α=0.5），M10/M30 用 K4/α=0.25。K16 已证滞后偏差有害，禁用。

### 3.3 门特征表（候选，最终由特征选择定）

| 特征 | 定义 | 来源 |
|---|---|---|
| f1 载体后验精度 | σ̂²(XᵀX)⁻¹ 的迹/条件数（M-拟合自带的闭式量） | 精度门遗产 |
| f2 departure 统计 | 更新候选相对当前载体的位移（B8 兼容量） | 信任门遗产 |
| f3 ★ 时间一致性 | 相邻窗预测的方差（B3 分数；P1 已确证其携带真实信号——因果平滑 external M10 +0.019/M30 +0.037，15/15） | 本设计新，P1 后一等特征 |
| f4 解码置信度 | 该 trial 窗口预测的 ensemble 离散度 | 新 |
| f5 栈大小/会话进度 | FIFO 当前长度、trial 序号 | 部署状态 |
| f6 预算 | M4/M10 | 部署状态 |
| f7 伪标签-载体残差 | 平滑伪标签与当前载体预测的偏差幅度 | 新 |

## 4. 执行序（探针纪律，两道零成本门在任何拟合之前）

```
P1 平滑探针（✅ 已完成，2026-08-28，results/continuity_probe_v1/，锚点 63/63 位级）：
    因果平滑（K4 均值 / α=0.25 EMA）external：
      M4 +0.0088 (12/15, CI [+0.003,+0.015]，边缘)；
      M10 +0.0191 (13/15, CI [+0.011,+0.027])；
      M30 +0.0365~+0.0394 (15/15, CI 全正)  → **+0.01 门在 M10/M30 通过**
    within：因果族 null/负（滞后偏差主导）；轨迹对齐族（非因果诊断）within 小正
    → 路由：【≥ +0.01 分支生效】f3 时间一致性升一等特征；平滑独立进部署配方
P2 oracle-gate 诊断（下一步）：上帝视角的门（只接受真值改善的更新）能吃多少池子？
    → 这是任何门（手工/学习）的上界；≈ activity-only 则载体臂撤案
P3 可学性检查：y_j 与特征 x_j 的可预测性（单特征相关 + 小模型 LOSO AUC）
P4 策略拟合：源折 LOSO，逻辑回归/浅树，特征选择
P5 matched score：与 activity-only 同引擎配对，预注册门（§5.4）
```

**P1 的三个设计修正（已并入 §3.2/§3.3/§6）**：
1. **B2（因果窗口族）升主构造，B1（轨迹对齐）降级为非因果诊断**——探针发现轨迹对齐
   必然读取未来窗（被评分 bin 不在过去窗内），非部署合法；且实证上因果族在 M10/M30
   **强于**非因果上界（解码器上下文依赖误差结构主导）。因果族数学上涵盖"平滑整轨迹再取
   末 bin"（均值可交换性，测试已证）。
2. **平滑增益随预算单调**（M4 < M10 < M30）：预测越贴行为，方差缩减越有的赚——M4
   边缘恰是载体最饿处，提示 f3 在 M4 的特征价值有限、在 M10/M30 更强；伪标签构造按
   预算自适应 K。
3. **C3 oracle（0.4449）只是载体轴天花板**：平滑轴与其正交可叠加（M30 静态 0.4286 +
   平滑 +0.039 ≈ 0.468 已越过 0.4449）——池子记账更新为双轴：载体轴（门的战场）+
   输出抖动轴（平滑的战场，已白捡）。

## 5. 安全与纪律

### 5.1 基座不可侵犯
活动臂 = activity-only，**永不改动**。载体臂只能叠加：门全拒时系统逐位退化为
activity-only（预注册该退化必须 bitwise）。

### 5.2 LOSO 纪律
策略永不评估自己的训练会话；特征选择也在内折完成。

### 5.3 fail-closed 回退
部署时逐 session 监控：门的累计接受使该 session 劣于 activity-only 超过阈值 ε
（建议 ε=0.005）→ 该 session 余下 trial 强制零接受。回退事件计数入收据。

### 5.4 预注册门
- 主门：载体臂版 − activity-only ≥ **+0.03** 且 ≥10/15 sessions 正（external governing）
- 次读数：M4 与 M10 分列（池子大小悬殊，预期 M4 主战场）；M30 门保持 no-op
  （载体臂在 M30 必须逐位不激活——继承 M30 部署规则）
- 池子关闭比例：增益 / 残余池（M4: x/0.218, M10: x/0.056）为论文主指标

## 6. 分支树（审核者请重点挑错）

```
P1 平滑探针结果
├─ external ≥ +0.01（某 K）──────→ 时间一致性确证：
│     f3 升一等特征；平滑独立进部署配方（白捡）；
│     B1 平滑伪标签为主构造 → P2
├─ 仅 within 正 ────────────────→ 特征保留但预注册改为 within 读数；
│     external 池子的伪标签路径先验下调 → P2 照跑（上界是硬事实）
└─ 全 null ──────────────────────→ f3/B1/B2 全部出局；
      伪标签构造退回原始逐窗；门特征只剩 f1/f2（手工已败的两个）
      → P2（oracle-gate）变成生死判：上界大才继续，否则撤案

P2 oracle-gate 上界（相对 activity-only）
├─ ≥ +0.05 @M4 ────────────────→ 池子可及性成立 → P3
├─ +0.02~+0.05 ────────────────→ 边缘：只有 P3 显示强可学性才进 P4
└─ < +0.02 ────────────────────→ 任何门都吃不到 → 全案关闭，
      "载体侧残余需真标签"成为定稿边界句

P3 可学性（LOSO AUC / 相关）
├─ AUC ≥ 0.65 或 top 特征 |ρ|≥0.3 → P4
└─ AUC ≈ 0.5 ────────────────→ 异质性不可预测 → 撤案（比 P2 更快的死法）

P5 结果
├─ 过主门 ───────────────────────→ 三臂系统定稿（activity + learned-gate T4 + 平滑）
├─ 0 ~ +0.03 ───────────────────→ 按预注册记负；"学习门也不够"进负空间地图
└─ 负 ──────────────────────────→ fail-closed 生效证据 + 撤案
```

## 7. 与失败前例的逐条区分（审核者必问）

| 前例 | 本设计为何不同 |
|---|---|
| 信任门/精度门（手工，净负） | 手工=单特征固定阈值；本设计=真标签监督的多特征策略 |
| CBM-D（训练侧预算边缘化，−0.09） | CBM-D 改编码器训练；本设计解码器逐位冻结，只拟合决策策略 |
| PIRG（消费端精度加权，null） | PIRG 把精度喂给网络；本设计把精度（等特征）喂给**外层决策器** |
| PMC-D（训练期后验采样，−0.11） | 采样改训练分布；本设计不改任何训练 |
| Cell C（一致性损失，−0.036） | C 是表征级约束；本设计是部署级接受/拒绝决策 |
| 0C 集成（输入掩码平均，低于 native） | 那是给输入加噪再平均；平滑是对最终预测去噪，对象不同 |

## 8. 收据与复现纪律

沿用 lane 全套：0444+sidecar、fresh-root、launch-final closure equality、
零 target optimizer/backward、oracle 诊断标 leakage、逐 session 配对 + bootstrap（描述性）、
fail-closed 事件计数。P1-P5 每步独立收据，`ADVANCE/ABORT` 判定字段预注册于收据内。

## 9. 留给审核者的问题

1. ε（fail-closed 阈值）0.005 是否合理？逐 session vs 滑动窗口监控？
2. y_j 的真值对照量（向 β*_j 的距离变化）用 L2 还是下游解码 R² 变化？（后者更贴目标但噪声大）
3. f3 一致性分数的定义窗口（trial 内相邻窗 vs 跨 trial）哪个更贴"伪标签可信度"？
4. 接受权重 w_j=p_j（软）vs 二值（硬）——软版引入连续自由度，是否值得？
5. 门是否也应控制**更新幅度**（shrinkage toward update）而非仅接受/拒绝？
6. 池子数字（0.218/0.056）由 C3 oracle 界定——该 oracle 用泄漏标签，是否高估了
   伪标签可达上界？（即：oracle 界与伪标签可达界之间的 gap 本身是否需要单独诊断？）

---

## 10. 独立审核补充（2026-08-28）

### 10.1 审核结论

**裁决：`GO` 修订后的 P2 oracle-policy 分解；`NO-GO` 当前版本的 P3--P5。**

这个方向的研究问题成立：手工信任门和 Precision V2 只覆盖单特征、固定阈值的很窄策略空间，
不能排除一个由 source 真标签监督、部署时只读取无标签特征的小型决策策略。冻结 Cell-D、保持
activity transition 与 carrier transition 独立、用逻辑回归优先于深网络、先做 oracle 再拟合
策略，都是正确选择。

但是当前 v2 把三个不同问题混在了一起：

1. 对最终窗口输出做时间平滑；
2. 为一个已完成 trial 构造更准确的 pseudo direction；
3. 决定是否把一个 carrier proposal 提交到因果状态。

三者的因果时钟、上界和必要对照不同。当前 ceiling 记账、P1 解释、fail-closed 和策略标签还不足以
识别 learned gate 的贡献。如果现在进入 P3--P5，即使得到正数，也无法判断增益来自 output
smoothing、activity memory，还是 learned gate。

### 10.2 阻塞 1：`0.218 / 0.056` 不是已建立的 carrier residual pool

设计用 activity-only 的 M4/M10 分数减同一个 M30-activity C3 oracle `0.4449`：

- M4: `0.4449 - 0.2265 ~= 0.218`；
- M10: `0.4449 - 0.3888 ~= 0.056`。

这个相减不匹配 activity exposure。已完成的 honest-M C3 oracle 是：

| budget | activity-only external | honest-M full-label C3 oracle | crude difference |
|---|---:|---:|---:|
| M4 | 0.226494940 | 0.241332265 | +0.014837325 |
| M10 | 0.388808712 | 0.388511779 | -0.000296933 |
| M30 | about 0.428592983 static reference | 0.444947871 | +0.016354888 |

即使这组粗相减也不是最终 matched ceiling：activity-only 使用会随已完成 query trial 改变的因果
activity state，而 honest-M oracle 使用固定 first-M activity。它们不是同一条状态轨迹。但这已经
证明 `0.218 / 0.056` 不能称为已知 residual pool，也不能用作 `gain / pool` 的论文分母。

**要求：** P2 必须在 exact activity-only 因果状态轨迹、相同 query windows、相同初始 carrier、
相同 output filter 上重建 proposal/action oracle。后续的 pool-closed 指标应改成：

```text
learned-policy gain / coherent matched oracle gain
```

### 10.3 阻塞 2：P1 测到的是静态解码输出平滑，不是 CDM pseudo-label 质量

`continuity_probe_v1` 使用的是封存 P4 静态 recipe：固定 calibration activity、静态 T4、每个
query window 独立 forward，然后对最终输出做平滑。其实现明确披露：activity-only CDM 收据没有
保存 raw trajectories，CDM output smoothing 需要重新 forward，本次 P1 没有运行。

因此 P1 目前只支持：

> frozen static decoder 的最终输出存在可由 causal low-pass filtering 利用的 external 时间结构。

P1 不直接支持：

- 平滑会在 activity-only CDM 上保留相同增益；
- 平滑会减小 completed-trial pseudo direction 的误差；
- f3 prediction consistency 可以预测 carrier-update utility。

这些必须由同轨迹 CDM forward 和 source true-label counterfactual 直接测量。P1 的 `63/63`
bit-exact anchor 证明数据/实现复现正确，不证明这些跨机制推论。

另一个统计边界是：P1 在 external-15 上跑了多个 K/alpha arm，再按 budget 选 best arm。该结果可作
exploratory evidence，但不能把 external 标签选出的 K 直接当作无泄漏 deployment recipe。最终 K
必须由 source-only/nested grouped validation 选择，或在看到 external 结果之前有固定理论值。

### 10.4 阻塞 3：output-time 与 post-trial update-time 的因果性被混淆

对于当前被评分 window，trajectory-aligned B1 读取未来 window，确实不可部署。但 carrier update
发生在 trial 完成以后：trial 内所有 window 已经是过去，update 只影响下一 trial。因此，只要不读
下一 trial，在 completed trial 内使用整段轨迹、双向/zero-phase 平滑或多个 overlapping-window
estimate，可以是合法的 post-trial pseudo-label 构造。

当前 CDM 本来就是：

1. 获取一个完整 completed-trial velocity trajectory；
2. 在 movement mask 上积分 displacement；
3. `atan2` 得到 pseudo direction；
4. snap 到 canonical direction；
5. 用 native trial rate 更新 carrier sufficient statistics。

所以 P1 的 current-output R2 不能决定 B1/B2 谁更适合 pseudo direction。修订 P2 应直接比较：

- raw completed-trial trajectory；
- trial-local causal filter；
- trial-complete trajectory-aligned/zero-phase filter；
- 每种构造的 pseudo-direction error 和 future counterfactual utility。

所有 smoothing 必须按 completed trial 独立执行并在 trial boundary reset。P1 的 causal kernel 沿
拼接 window 序列平滑，不消费 trial ID；直接复用会在 trial/gap 边界混入上一 trial 的输出。

文档还存在一个文字冲突：§3.2 把 B2 定为主构造，§6 的通过分支却写“B1 平滑伪标签为主构造”。
在新收据前不要预先指定 B1 或 B2；由 trial-level pseudo-direction/utility 结果决定。

### 10.5 阻塞 4：主对照会把 smoothing 增益算到 learned gate 名下

文档同时要求：

- 平滑独立进入 deployment recipe；
- gate 全拒时 bitwise 退化为 activity-only。

如果最终输出仍做 smoothing，gate 全拒只能退化成 **smoothed activity-only**，不能 bitwise 等于 raw
activity-only。M10 已知 output smoothing 约有 `+0.019` exploratory gain；若最终比较
`gate+smoothing - raw activity-only`，即使总增益达到 `+0.03`，gate 本身可能只贡献约 `+0.011`。

**要求：** learned gate 的 governing contrast 必须是：

```text
learned-gated carrier + fixed output filter
minus
activity-only + the exact same fixed output filter
```

output smoothing 必须作为独立 factorial axis 报告，不能进入 carrier pool-closed 分子。

### 10.6 阻塞 5：§5.3 的性能回退需要 target labels

部署时无法知道“累计接受使 session 比 activity-only 差多少 R2”。用 `epsilon=0.005` 比较真实性能会
读取目标行为标签，属于 target-label leakage。问题不是 epsilon 大小，而是该量部署时不可观测。

可部署的 fail-closed 只能使用无标签量：

- source-only 固定 gate threshold；
- feature OOD/Mahalanobis 或 conformal rejection；
- proposal 相对 frozen support initializer 的累计 credible displacement；
- complementary-group disagreement；
- 最大 carrier commit 次数；
- M30 永久 no-op。

可以并行跑 shadow activity-only predictions 并记录 prediction difference，但不能根据未知 R2 决定
是否回退。

### 10.7 阻塞 6：`beta*` 距离不是部署 utility，普通四维 L2 也不成立

当前策略标签使用 candidate carrier 向 true-label refit `beta*` 的距离变化。这个量不保证 Cell-D
future R2 改善，而且 `[a,c,m,b]` 不是四个独立自由度：生产 T4 明确使用
`m = hypot(a,c)`。普通四维 L2 会重复计算方向幅度，还忽略方向周期性、posterior precision、unit
validity、normalizer 和 decoder sensitivity。

主 source-supervision 应改为 future counterfactual utility：

```text
u_j = normalized_loss(reject on a fixed future horizon)
      - normalized_loss(accept on the exact same future inputs)
```

要求：

- action 在 completed trial j 后发生；
- utility 只在 j+1 或预先固定的 j+1...j+H 上计算；
- accept/reject 使用相同 activity state、neural inputs、targets 和 output filter；
- 当前 trial 不可同时用于 proposal 和 utility；
- 单 trial 主标签建议用 normalized SSE，不用不稳定的单-trial R2；
- session-level governing R2 仍是最终 score；
- `beta*` 的 constrained/Mahalanobis distance 只保留作 mechanism diagnostic。

### 10.8 阻塞 7：oracle 必须区分局部上界与 coherent policy

一次 accept/reject 会改变后续 active T4、后续预测、后续 pseudo direction 和后续特征分布。因此
“只接受真值改善的 update”至少有三种不同含义：

1. `NONCOHERENT_ONE_STEP_SWITCH_CEILING`：在同一父状态比较两种 action，对每个 future trial
   取较优预测；这是宽松诊断 ceiling，不是一条可执行 policy；
2. `COHERENT_GREEDY_ORACLE`：看固定 next-trial/future-horizon utility，选 action 后真实推进状态；
   可执行但不是全局最优；
3. session 全 action sequence 的 global oracle：需要枚举或搜索连续状态树。

P2 至少同时报告前两者。如 trial 数允许，可以增加固定-width beam search，但必须称为近似搜索结果，
不能把 beam value 称为严格 global upper bound。

### 10.9 阻塞 8：gate LOSO 不等于 base-decoder out-of-session

即使 gate 留一 session，如果 frozen Cell-D 曾训练过该 source session，该 session 上 pseudo prediction
仍是 base-decoder in-sample。由此学到的 feature-to-utility 关系可能在 external subject-M 上失效。

优先选择：

1. 用 base decoder 没见过的 within-6 session 构造 gate labels，再在 external-15 测试；或
2. 使用真正 source-session-LOSO decoder checkpoints 生成 gate 数据。

无论选择哪种：

- CV/bootstrap unit 必须是 session，不是 trial；
- 特征选择、threshold、model choice 全部 nested group CV；
- 收据必须报告每预算的 session 数、trial 数、正负 action 数和 utility 分布；
- 不能把同 session 内的 trial 当独立样本计算普通 CI。

### 10.10 阻塞 9：AUC 不是策略放行指标

少量 false-positive carrier updates 可能带来很大伤害；一个 AUC 很高的 gate 仍可能降低 R2。反过来，
一个只接受少数高价值 proposal 的保守 gate，AUC 不高也可能有正 utility。

P3 主读数应是：

- source grouped-LOSO realized policy utility；
- oracle recoverable fraction；
- policy regret；
- false-positive harm；
- high-utility proposal 的 PPV/recall；
- 若未来考虑 soft gate，再报告 calibration/Brier score。

AUC 与单特征相关只能做诊断，不能单独用 `AUC >= 0.65` 放行 P4。

### 10.11 修订版 P2：最小高信息 oracle 矩阵

所有行必须共享 exact query inputs、activity-only 因果 activity trajectory、initial carrier、trial
chronology、checkpoint/normalizer 和 output filter。

| row | pseudo direction | carrier action | output filter | role |
|---|---|---|---|---|
| A0 | none | reject all | raw | original activity-only |
| A1 | none | reject all | fixed smoothing | smoothed activity-only |
| C0 | raw completed-trial prediction | always accept / frozen B8 checks | same as A1 | ordinary CDM |
| C1 | smoothed completed-trial prediction | always accept / frozen B8 checks | same as A1 | isolates pseudo-label smoothing |
| O0 | raw pseudo | oracle accept/reject | same as A1 | raw-pseudo gate ceiling |
| O1 | smoothed pseudo | oracle accept/reject | same as A1 | gate + pseudo-smoothing ceiling |
| O2 | true completed-trial direction | oracle accept/reject | same as A1 | combined pseudo/gate loose ceiling |
| P | existing Precision V2 | frozen precision rule | same as A1 | hand-gate comparator |

关键 contrasts：

```text
O1 - A1  = matched learned-gate opportunity after holding output smoothing fixed
O1 - O0  = value of pseudo-label smoothing inside the gate problem
O2 - O1  = remaining pseudo-label bias/quality gap
C1 - C0  = effect of smoothing under always-accept dynamics
```

M4/M10 是部署主读数；M30 carrier 仍 no-op，但保留 oracle-only 诊断，验证 oracle 是否确实学到
“可靠长前缀几乎全拒”。

### 10.12 修订后的 kill criteria

1. `COHERENT_ORACLE(smoothed pseudo) - smoothed activity-only < +0.02 @M4`：停止 learned gate。
2. `O1 - O0 < +0.005`：平滑不改善 carrier action value；仅保留独立 output filter。
3. O2 很高但 O1 很低：瓶颈仍是循环 pseudo-label bias；先修 pseudo direction，不拟合 gate。
4. oracle 很高但 source grouped-CV realized utility 近零：headroom 存在但无法从无标签特征识别，撤案。
5. 只有 coherent oracle 和 source learned policy 都通过，才进入 external matched score。

最终 `+0.03 and >=10/15` 可以保留作 performance claim gate，但不能用它替代前面的 matched
oracle feasibility gate。

### 10.13 如果 P2 通过：P3/P4 的最小模型纪律

- 第一版只用一个逻辑回归和 hard accept/reject；
- threshold 只由 source/nested grouped CV 固定；
- 不使用 `w=p`；
- 不同时学习 update amplitude；
- f1 复用 Precision V2 的 exact covariance/evidence；
- f2 使用相对 frozen support initializer 的累计 Mahalanobis displacement；
- f3 是 completed-trial 内、trial-boundary-reset 的 prediction/direction stability；
- f4 必须明确为四个 complementary held-unit predictions 的离散度，不能含糊写 ensemble；
- f5/f6 是 progress 和 budget，但要防止成为 session shortcut；
- f7 使用 circular/Mahalanobis residual，不使用 raw `[a,c]` 坐标；
- 所有特征必须在 action 前可得；
- 只有离散 action-grid `{0, 0.5, 1}` 的 oracle 明确证明中间幅度有价值，才另开 shrinkage successor。

最终 learned-gate governing contrast 必须是：

```text
learned gate + smoothed pseudo + fixed output filter
minus
activity-only + the exact same fixed output filter
```

同时保留 ordinary CDM、Precision V2、raw-pseudo learned gate、smoothed-pseudo learned gate 和 oracle
ceiling。M30 learned carrier 必须与 smoothed activity-only 在 carrier state 上 bitwise no-op。

### 10.14 对 §9 六个问题的直接回答

1. **epsilon=0.005：**不可部署，不是阈值大小问题；真实 R2 差需要 target labels。改用无标签
   OOD/drift/commit-cap safety。
2. **L2 或 R2：**主标签用固定未来 horizon normalized delta-SSE；最终用 governing session R2；
   constrained/Mahalanobis `beta*` 距离仅作 secondary diagnostic。
3. **f3 窗口：**trial 内，trial boundary reset。跨 trial consistency 如需要，应另立 session-drift
   feature，不与局部 prediction noise 混合。
4. **软或硬：**第一版硬门。没有 calibrated action-value 证据前不要令 `w=p`。
5. **控制幅度：**第一版不控制。先证明二值 action 有 learnable value，再用独立 oracle action grid
   决定是否值得研究 amplitude。
6. **pool 是否高估：**是，而且是 activity-budget/surface 不匹配导致的已知记账问题；必须用 exact
   matched-trajectory oracle 替代。

### 10.15 最终建议句

> Approve a rewritten oracle-policy decomposition, not learned-gate training. Continue to source-supervised
> gate fitting only if a coherent, same-trajectory oracle shows at least +0.02 M4 gain over smoothed
> activity-only and that gain is measurably predictable under session-grouped out-of-fold evaluation.

先完成上述 P2 不需要深网络训练。它即使返回 STOP 也有明确科学价值：可以分别区分“gate 没有可吃
上限”“上限存在但 pseudo labels 不够”“上限存在但 benefit 无法从无标签特征预测”。

---

## 11. 操作侧对审核的响应（2026-08-29，文档状态 → v3-revised）

**裁决：九个阻塞项全部接受。**P3-P5 按审核 NO-GO 冻结；执行修订版 P2（§10.11 八行
oracle 矩阵）。三点需要向后续读者明确：

1. **池子记账认错**：§0/§5.4 的 0.218/0.056 作废（活动暴露不匹配）。按 §10.2 的
   honest-M 粗算，M10 匹配暴露下的载体机会 ≈ 0，M4 ≈ +0.015——**在跑 P2 之前，
   可学习门的期望值已贴着 kill 判据 1（< +0.02@M4）**。修订版 P2 的预期结局是
   关门而非开门；仍执行（推理级成本、无论结果都终结该问题），但预注册此预期。
2. **因果时钟认错**：输出时刻与 trial 完成后更新时刻是两个时钟、两种合法性——
   trial 内零相位/双向平滑对伪标签构造合法（阻塞 3）。B1/B2 主从预定撤销，
   由 trial 级 pseudo-direction/utility 结果决定；§3.2 与 §6 的文字冲突系我的笔误。
3. **fail-closed 认错**：§5.3 的 R² 劣化判据需要目标标签，不可部署。改用 §10.6
   的无标签安全集（OOD/Mahalanobis、累计位移上限、提交次数上限、M30 永久 no-op）。

执行计划（全部推理级）：
- P2' = §10.11 矩阵 + §10.8 的双层 oracle（NONCOHERENT 一步切换诊断 +
  COHERENT 贪心可执行）+ §10.12 五条 kill 判据逐条预注册进收据
- P1 补漏：K/α 的 source-only/nested 选择重跑（external 选 K 的探索性披露保留）
- P2' 通过（kill 判据 5 全过）才进 §10.13 的最小模型纪律版 P3/P4

---

## 12. P2' 结果与终局（2026-08-29，learned_gate_p2prime_v1 TERMINAL）

**最终判决：`SMOOTHING_NO_CARRIER_VALUE_KEEP_OUTPUT_FILTER_ONLY`；kill 判决 = [KC2]。**

| 量 | 数值 | 判定 |
|---|---|---|
| O1−A1（匹配连贯门机会） | **+0.0645** @M4 ext（CI [+0.026,+0.108]）；M10 +0.0602 | **KC1 通过**——粗算（+0.015）错了，审核要求的匹配轨迹 oracle 改变了结论 |
| O1−O0（平滑在门内的价值） | −0.0002/−0.0003 | **KC2 触发**——平滑伪标签机制死；O0≈O1 ⇒ raw 伪方向的门机会同为 +0.065 |
| O2−O1（伪方向质量缺口） | **+0.0922** @M4 ext | 方向质量（70% snap 错配、0.901 rad）是残余池的所在地 |
| A1−A0（输出滤波复合） | +0.0194/+0.0234/+0.0385 ext | 幸存：独立输出滤波轴 |
| P−A1（手工精度门） | ≈0/负 | 手工门确认无效 |
| M30 | 载体逐位 no-op；oracle 仍会接受 41-45% | 部署维持 no-op；"可靠前缀应近全拒"的预期被驳 |

**三层实测钉死的负结果**：伪方向误差 70%（子研究）、伪构型效用全负（唯一为正的是
TRUE）、平滑不改善方向（第四位小数）。**幸存资产**：A1 系统（0.2459/0.4122/0.4671
external）、门机会 +0.065（oracle 上界，未授权拟合）、方向质量战场 +0.092。

**KC5 未授予**：按 §10.12，平滑伪标签门设计不进行；本收据亦不授权 raw 伪方向门
（KC1-KC3 须全过，KC2 已触发）。任何后续（raw 门/方向质量改进）须新工单重新预注册。

操作侧预期修正记录：§11 预注册的"M4 匹配池 ~+0.015 贴着 kill 线"被连贯复合效应
推翻（+0.065）——非连贯单步上界（+0.0093）不约束复合策略，审核的双层 oracle 区分
正是为此，其工序价值在此得到实证。
