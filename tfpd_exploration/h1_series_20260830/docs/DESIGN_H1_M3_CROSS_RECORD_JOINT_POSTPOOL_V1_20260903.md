# Design: H1 Exactly-M3 Cross-Record Matched Training + Joint Post-Pool V1

日期：2026-09-03  
状态：`FROZEN_BEFORE_GPU0_SOURCE_ONLY_DATE_LODO`

## 1. 问题

H1 官方 held-out calibration 对每个新 recording 恰好只提供三条合法 calibration
trial。现有 C1 checkpoint 的训练 identity prefix cycle 是 M7/M5/M4，M3 从未进入
训练；训练 query 还来自 calibration recording 本身。官方部署却是：一个
calibration recording 的 chronological M3 生成 identity 与 H-C carrier，随后在另一个
隐藏 eval recording 上解码。

冻结 C1 上的 direct post-pool、signed scalar output residual、700 参数 profile 三层
后继都未通过五日期稳定性门。它们关闭的是“小补丁”，没有检验匹配官方 M3 与
cross-record 输入合同的联合训练。

## 2. 三臂

每个 outer-date fold 都从同一 immutable C1 epoch-49 checkpoint 开始：

1. `FROZEN-C1`：不训练，使用原生 early-pool M3 identity；
2. `N3-XR12`：原生 early-pool，使用 source calibration M3 identity/carrier，在配对的
   source minival recording 上继续训练 12 epoch；
3. `J3-XR12`：与 N3 完全相同的数据、batch、随机数、步数、优化器和初始 C1
   权重，但使用 anchored joint post-pool identity。

J3 的 identity 定义如下。令每条 calibration trial 经过共享
`carrier_pre_pool` 后为 `e_m`，固定的 M3 carrier 为 `c`：

`h_native = post_pool([mean_m(e_m), c])`

`h_post = mean_m(post_pool([e_m, c]))`

`h_joint = h_native + tanh(alpha) * (h_post - h_native)`

`alpha` 是唯一新增参数，是跨 unit、跨 identity-bin 共享的标量，初始化为 IEEE
`+0`。第一步之前 J3 prediction 必须与 N3 bitwise 相同。随后 N3 与 J3 都更新完整
C1 网络；J3 额外更新 `alpha`。原生分支永远保留，不能用纯 late-pool 替换。

## 3. Source-only date-LODO

- outer dates：19250108、19250113、19250115、19250119、19250120；
- 每折 model/plan/normalizer：既有 immutable C1 date-LODO authority；
- source support：每个 held-in-calib recording 的 chronological M3；
- source query/target：对应 held-in-minival recording；
- carrier：只用同一个 calibration M3，经该折 source-frozen H-C plan 拟合并冻结；
- query window：official W700 zero-left-padding；只用 official eval-mask endpoint；
- 训练采样：每个 recording 的 endpoint 按固定 stride 4，session-homogeneous batch；
- 12 fixed epochs、seed 42、batch 32、Adam lr `5e-5`、weight decay 0、无 scheduler；
- loss：last-bin MSE，prediction `/20`；
- checkpoint：固定 epoch 11，不按 source 或 outer 指标选 epoch；
- outer-date calib/minival 只能在两臂训练状态封存后打开；outer 上零 optimizer、零
  backward、零模型更新。

N3/J3 每个 batch 使用同一 row order，并 replay Python、Torch CPU、Torch CUDA RNG。
每一步两臂的 dropout 随机数必须完全相同；只在完成两臂之后推进一次 canonical RNG。

## 4. 预注册选择门

先在五折上计算：

- `gain_N = N3-XR12 - FROZEN-C1`；
- `gain_J = J3-XR12 - FROZEN-C1`；
- `increment_J = J3-XR12 - N3-XR12`。

最终候选必须满足：相对 FROZEN-C1 的五折 mean `>= +0.005 R²`、至少 4/5
日期非负、worst `>= -0.010`。

若 J3 同时满足最终候选门，并且 `increment_J` mean `>=0`、至少 3/5 日期非负、
worst `>=-0.010`，选择 J3；否则若 N3 满足最终候选门，选择 N3；否则停止本路线。
J3 只因“晚池化相对匹配原生训练非负”而获选，不能把共同的 cross-record/M3
训练收益归因于晚池化。

## 5. 通过后的唯一动作

只对五折选出的一个架构：

1. 从 immutable all-source C1 epoch-49 checkpoint 开始；
2. 用全部 13 个 held-in calibration→minival recording pair 按相同合同训练 12 epoch；
3. 制作包含 27 个 M3 calibration payload 的 H1 decoder package；
4. 本地 held-in minival official-interface sentinel 通过后，允许一次 EvalAI H1 提交。

未通过五折时不训练 all-source、不打包、不提交，也不追加 profile、matrix、rank、
per-unit gate 或学习率 sweep。

## 6. 解释边界

- N3 成功、J3 无增量：故事是 official M3/cross-record matched training，而不是晚池化；
- J3 对 N3 非负且最终通过：晚池化提供小的互补收益，但主体收益仍需按 N3 对照拆分；
- 三臂都无增益：H1 的 C1 官方收益已接近这条 identity 路线可转移的上限；
- 本设计不读取 held-out eval label，不把隐藏 query activity 叫 calibration trial，也不
  测试 continual memory。

