# Design: H1 Post-Pool Profile Gate V1

日期：2026-09-03  
状态：`FROZEN_SOURCE_ONLY_DATE_LODO_12EP`

## 1. 为什么仍值得做

已完成的 H1 cross-record RP 正标量和 output-residual 标量都没有过门。这关闭了
“一个全局幅度就能迁移”的说法，但小 gate 的损失很小，不能排除 post-MLP
identity 在不同历史位置上具有不同符号和可靠度。

本 successor 不改写两个负结果。它检验一个新的、低容量假设：MLP 后 pooling
产生的 residual 需要一个沿 SPINT 700-bin identity 轴的 source-learned profile。

## 2. 唯一模型

基础模型是每折 immutable C1 epoch-49。静态与完整 post-pool identity 分别为
`h0` 和 `hp`，二者均为 `[176,700]`。只新增一个跨 unit 共享的
`p in R^700`：

`h = h0 + tanh(p)[None,:] * (hp-h0)`。

- `p` 初始化为 IEEE `+0`，第一步前必须 bitwise 等于 `h0`；
- C1 的所有参数冻结且保持 eval/no-dropout；
- 只对 `p` 建 Adam；不训练 carrier、pre/post MLP、transformer 或 decoder；
- 参数量 700；不做 rank、宽度、正则或学习率 sweep；
- 这是 anchored post-pool profile，不是“从头训练的纯 late-pool 网络”。

## 3. Source-only matched training

每个 outer-date fold：

1. 只打开其余四个 source dates 的 held-in-calib 与对应 held-in-minival；
2. calib chronological M3 生成 `h0` 和固定 H-C carrier；
3. minival raw bin `[0,768)` 生成唯一 query member，再得到 `hp`；
4. 只在 endpoint `>=768` 的 source minival label 上训练；每隔 4 个有效 endpoint
   固定取一行，降低重复的高度相关窗口；
5. 12 epoch、seed 42、session-homogeneous batch 32、Adam lr `1e-2`、weight decay 0、
   last-bin MSE、输出 `/20`；
6. 封存 profile 后才打开 outer date，并在完整 official-style minival 面评分。

query chunk、eval mask、trial_change 和 target 的职责分离：chunk 只读 raw neural；
eval mask 只定义评分/训练 endpoint；target 只进入 source loss 或 outer 评分，永不
进入 memory state。

## 4. 五折门

Primary 为 profile arm 相对同 checkpoint static M3 的等日期增益：

- mean `>=+0.005 R²`；
- 至少 4/5 outer dates 非负；
- worst `>=-0.010`；
- 五折全部 base-model state before/after 相同；
- 每折训练步均 finite，且 profile gradient 非零。

通过后才训练一个 all-source profile，并允许制作 H1 package。失败则停止所有
H1 post-pool profile/capacity 扩展；不追加 per-unit gate、matrix 或整网解冻。

## 5. 风险与解释边界

最强风险是 700 维 profile 在短 source minival 上记忆 recording 特征。严格
date-LODO、只有一个固定配置、以及 4/5 + worst 门用于控制这一风险。即使通过，
EvalAI 前也只能称“source-grouped transferable profile”，不能称官方 held-out
成功。

