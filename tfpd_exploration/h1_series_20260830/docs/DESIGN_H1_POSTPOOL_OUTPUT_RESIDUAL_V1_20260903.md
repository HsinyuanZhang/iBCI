# Design: H1 Post-Pool Output Residual V1

日期：2026-09-03  
状态：`FROZEN_SOURCE_ONLY_LODO_SCREEN`

## 1. 假设

上一格证明直接在 identity 空间做正向 RP 融合为负，但它没有回答 post-MLP
候选是否携带**符号或尺度错误的互补输出方向**。本设计不重训网络，也不再
扫 identity gate；它把 static 与完整 post-pool identity 各解码一次，只在
source dates 上闭式拟合一个 signed scalar output residual。

这与事后放宽上一格不同：上一格的干预是
`h=h0+g(hp-h0), g>=0`；本格的干预是

`y = y0 + beta * (yp-y0)`，

其中 `y0=D(x,h0)`，`yp=D(x,hp)`，decoder 的非线性在融合之前已经分别执行。
它是 M2 AOF 思路在 H1 的最小、低容量迁移。

## 2. 冻结输入和状态律

- 五个 C1 date-LODO epoch-49 checkpoint；
- calibration-recording chronological M3 activity 和同一 M3 H-C carrier；
- query-recording 从 raw bin 0 起，完整 768-bin chunk；
- decode-before-commit；chunk 只能影响 endpoint `>=768`；
- calibration M3 永久保护，只提交这一块；
- `beta=+0` 必须 direct-return `y0`；
- target、eval mask、trial-change 不进入 identity 或状态构造；
- target 仅在 source recordings 上用于拟合 beta 和算 R²；outer date 在 beta
  收据封存后才可打开。

## 3. Beta 的唯一拟合律

对 source recording `s`，记完整官方评分面的

- `d_s = yp_s-y0_s`（commit 前严格为零）；
- `e_s = target_s-y0_s`；
- `TSS_s = sum((target_s-mean(target_s))^2)`。

为了直接匹配“先等 recording、再等 date”的 primary，每个 recording 的权重为
`w_s = 1 / (number_of_source_dates * recordings_in_that_date * TSS_s)`。

闭式解唯一为：

`beta = sum_s w_s <d_s,e_s> / sum_s w_s <d_s,d_s>`。

不 clamp，不做 grid，不按 outer 结果改符号；要求 denominator finite 且
`>1e-12`，并要求 `abs(beta)<=4`，否则该折失败关闭。

## 4. 五折与门

每折严格顺序：source calib/minival → 拟合 beta → 写 beta receipt → 才打开
outer-date calib/minival → static 与 AOF-P 打分。

Primary 是五折 `AOF-P - STATIC` 的等日期均值：

- mean `>= +0.005 R²`；
- 至少 4/5 date 非负；
- worst date `>= -0.010`。

过门才允许在全部 13 source recordings 上按相同公式拟合 final beta 并制作
all-source package。未过门即关闭 H1 post-pool 输出融合；不追加 matrix、MLP、
per-date target fit 或 seed sweep。

## 5. 三个验证实验

1. `beta=+0` 的 prediction SHA 与 static 完全相同。
2. source beta 在写入收据之前不得打开 outer-date 文件；文件访问顺序可审计。
3. 全部模型 state before/after 相同，target optimizer/backward/update 全为 0；
   GPU0-only，GPU1 untouched。

## 6. 最强反对意见

source minival 很短，每个 recording 只有一次有效 commit，beta 可能只校正在这批
短流上的瞬态误差，不能代表官方长流。回应：因此仍要求严格 date-LODO 和安全
worst-date 门；通过也只授权一次 EvalAI 外部确认，不把 source 数字写成官方
持续记忆收益。

