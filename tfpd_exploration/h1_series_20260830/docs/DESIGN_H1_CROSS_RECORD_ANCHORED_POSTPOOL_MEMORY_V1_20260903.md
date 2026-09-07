# H1 Cross-Record Anchored Post-Pool Memory V1

日期：2026-09-03  
状态：`FROZEN_BEFORE_SCORE`  
范围：FALCON H1 / DANDI 000954 / source-only date-LODO / calibration-recording → minival-recording

## 0. 一句话

H1 的同 recording 真 trial 记忆已经证明有内容（C1 M3→M7：`+0.078573 R²`，5/5 dates、11/11 recordings），但它没有回答从一个全新的 eval recording 冷启动时是否仍有用。本设计把冻结 C1/M3 作为不可变锚，只允许 query recording 的完整定长块产生一个残差；同时比较原生 pooling 与优先的 **MLP 后 pooling**，用 source dates 选择后再打 outer date。

## 1. 为什么必须新开这一格

已有 H1-CAC Stage 1 的 query 与 M3 support 来自同一个 calibration recording。M2 的 continual 提交 581763 已经说明：同 recording 的成熟状态增益可以在官方 calibration→hidden-eval 重置后消失。因此 H1 在提交前至少要回答：

1. 新 recording 的第一块会不会污染校准 identity；
2. activity 内容在跨 recording 后是否仍为正；
3. MLP 后 pooling 是否能在静态锚保护下保留这部分内容。

本格不把 held-in-minival 的绝对高 R² 当成官方预测。它只用来模拟官方的状态重置与冷启动。

## 2. 已考虑的候选

发散候选包括：固定块、能量中位数门、均值能量门、支持距离门、post-pool 距离门、单次提交、延迟提交、native growing、post-pool growing、指数记忆、FIFO、静态/动态输出融合、可学习标量门、低秩 identity 残差和整网联合重训。

收敛后只保留三个构件：

- 固定块：当前能量门已在 C1 上失败，继续换门会形成 selector sweep；
- 不可变静态锚：任何 query 更新失败时都能精确回到已提交 C1；
- native 与 post-MLP 两种同输入残差：把“内容是否有用”和“pooling 放置”分开。

暂不做低秩/整网训练。先证明跨 recording 上存在可利用的无标签方向，再决定是否值得 12 epoch 匹配训练。

## 3. 官方与本地合同

- H1 是 continual。官方 evaluator 对 H1 每个 bin 调 `predict`，不调用 `on_done`。
- `reset(dataset_tags)` 清空 query buffer 和 query memory，但保留对应 held-out-calib 预计算的 M3 activity 与 M3 H-C carrier。
- query labels、`TrialNum`、eval mask、`trial_change` 均不得进入状态更新。eval mask 只用于离线计分。
- 第一候选从 query recording 的 raw bin 0 开始，收满 768 个真实 bin 后才完成；不得把 699-bin decoder 零历史当作 activity candidate。
- decode-before-commit：bin 767 仍用旧 identity；第一块最早影响 bin 768。
- M3 calibration members 永远保护；query 最多提交四块，M3→M7 后冻结。

## 4. 两种 activity identity

记三条 calibration activity 为 `a_1..a_3`，已完成 query chunk 为 `q_1..q_k`，固定 carrier 为 `c`，C1 的两个 identity 子网为 `pre` 与 `post`。

静态锚：

`h0 = post(mean_i pre(a_i), c)`。

原生 growing：

`hn(k) = post(mean_{a,q} pre(member), c)`。

MLP 后 pooling：

`hp(k) = mean_{a,q} post(pre(member), c)`。

部署 identity 始终是锚定残差：

- `RN(g,k) = h0 + g * (hn(k) - h0)`；
- `RP(g,k) = h0 + g * (hp(k) - h0)`。

`g=+0.0` 必须直接返回 `h0`，不得经过乘加后再声称相等。候选网格冻结为：

`g ∈ {0.0, 0.05, 0.10, 0.20, 0.50, 1.00}`。

`RP` 是优先部署形式：每个 member 通过完整 post MLP 后可以缓存，持续均值为 O(1) 更新。`RN` 是必要的 matched-content 对照；若 RP 不非劣，不得仅因工程偏好选 RP。

## 5. 数据划分与选择

五个 outer dates：`19250108/13/15/19/20`。每折使用已封存的 C1 date-LODO epoch-49 checkpoint 及该折 source-frozen H-C plan。

每个 session：

1. 从 `held-in-calib` 最早三个合法 trial 生成 M3 activity 与 carrier；
2. 调用一次逻辑 reset；
3. 从对应 `held-in-minival` 的 bin 0 开始构造 decoder 历史和 query chunk；
4. source dates 的 minival 只用于该折候选选择；
5. 候选冻结以后才允许打开 outer date minival 并计分。

选择指标为 official-style variance-weighted R²，使用全部 eval-mask bin，包括 decoder warmup。另报 commit 后 R²，但不用于选型。

每折先在 source dates 等 recording、再等 date。选择 source mean R² 最高的 `(family,g)`；若 RP 距 RN 最优不超过 `0.002 R²`，优先 RP；同 family 内若多个 g 距最优不超过 `0.001`，选择更小 g。然后只用所选候选评 outer date。

## 6. 决策门

五折 outer OOF 的 selected arm 相对 A-STATIC：

- equal-date mean `≥ +0.005`；
- 至少 `4/5` dates 非负；
- worst date `≥ -0.010`。

若通过，允许在全部 13 个 source session 上按同一选择律确定最终 family/g，并进入 all-source package。若最终选择 RP，则论文可称为 post-MLP anchored memory；若选择 RN，只能称为 native anchored memory。

若不通过但 RP 的 mean 非负、worst `≥ -0.010`，允许一次 12-epoch source-only matched-training cell；不得直接提交。若 RP 为负或发生显著冷启动损失，则停止晚池化路线。

## 7. 三个验证实验

1. **代数/实现哨兵**：g=+0 bitwise 等于 static；同一 raw member 在 batched 与 cached post-pool 路径误差 `≤1e-6`；decode-before-commit 无越界。
2. **跨 recording OOF**：五折 source-select→outer-score，报告完整流与 commit 后的逐 session/date delta。
3. **官方形态哨兵**：all-source payload 在本地 minival 经官方 evaluator 运行，reset 后状态为空、第一块完成前预测与 C1 static 精确相同。

## 8. 最强反对意见

反对意见：minival 每 recording 只有约 1500–1800 bins，通常只容纳一个 768-bin query chunk，不能代表官方长流的 M7 状态。

回应：它不能估计长流最终增益，所以本格只把它用作冷启动安全与跨 recording 方向门；同 recording M3→M7 上界已另行封存。只有短流不伤且方向为正，才值得把长流行为交给一次 EvalAI 外部确认。

## 9. 两周上限（实际目标更短）

- Day 1：冻结实现与单测，完成五折冻结权重 OOF。
- Day 2：若门过，构造 all-source payload、官方本地 evaluator 哨兵。
- Day 3：独立审核后提交一次 EvalAI。
- 其余时间只在 RP 非负但未过门时用于单个 12-epoch matched cell；不做 selector/gate sweep。

