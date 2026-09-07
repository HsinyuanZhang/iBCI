# B1 Activity-Reliability Template Profiling (ARTP) V1

日期：2026-09-03

## 1. 为什么继续 B1

此前的 NO-GO 只否定了“跨日期 channel identity 丢失，所以需要重新识别 unit”这一假设。B1 的
85 个通道索引基本稳定，这是资产，不是障碍。它没有回答另一个问题：当前 query trial 的活动时间
形状，能否指出三条已发布 calibration 声谱中哪一种更像当前发声。

两次 template-anchored 神经网络 pilot 也给出明确边界：framewise 训练把 source loss 持续降低，
却在验证日期从第 1 epoch 起就落后 `TPL-M3-median`。因此下一格不再让一个高容量网络改写强模板，
而只让神经信息控制三个已发布模板的凸组合；任何失败都不能产生无界输出。

## 2. 方法

名称：**ARTP — Activity-Reliability Template Profiling**。

对目标日期前三条 calibration trial，保存原始声谱 `Y_j` 和神经 activity signature `s_j`。
`s_j` 是 1 ms、5 ms box-smoothed rate 在每个时间点跨 85 通道的均值与标准差拼接，得到
1,800 维向量，再做标量中心化和 L2 归一化。它保留原生通道集合，不做 channel matching。

声学锚点为 `T = median(Y_1,Y_2,Y_3)`。每个 calibration exemplar 的可靠性为

`d_j = official_MSE(Y_j, T) / median_k official_MSE(Y_k, T)`。

对当前 query neural 计算 `s_q`，权重为

`w = softmax(<s_j,s_q>/tau - rho*d_j)`，预测为 `sum_j w_j Y_j`。

V1 冻结 `tau=0.05, rho=2, mixing=1`。`rho*d_j` 是 tuning/reliability profile：它利用合法的
M3 标签抑制声学异常 calibration trial；`<s_j,s_q>` 是 activity signature 的 trial-specific
信息。两项缺一都不能冒充主方法。

## 3. 与 M2 最新结构的关系

ARTP 迁移的是 M2 已验证的设计原则，而不是照搬其具体张量：

- 强锚点：`TPL-M3-median` 对应 frozen native branch；
- 低容量校正：只改三个凸权重，不重训主解码器；
- matched support：训练/部署都严格是 M3 reference bank；
- 安全回退：删掉 query-neural logits 即为 neural-free reliability control；未来若加标量 gate，
  `gate=+0` 必须直接返回中位模板；
- 无目标更新：query neural 只用于本次预测，不读取 query spectrogram。

B1 是 non-continual whole-trial 接口，当前 trial 的完整 neural array 合法可见，因此不需要伪造 M2
的 chunk boundary。ARTP 的缓存量固定为三条 signature 和三张模板。

## 4. 对照与主读数

必须同时报告：

1. `TPL-M3-median`；
2. 完整 ARTP；
3. cyclic query-signature：把 query neural signature 循环错配，模板/标签/可靠性不变；
4. neural-free reliability：query-neural logits 置零，只保留 M3 声学 profile。

主门是三个 held-in 日期上同时满足：ARTP 优于中位模板、优于错配 signature、优于
neural-free reliability。读数按日期等权；45 条 trial pooled 只作补充。

## 5. 证据等级

`tau/rho` 是在查看 held-in source 结果后冻结的，所以当前三日期 screen 是探索性构造和机制证据，
不能称为无偏 LODO 估计。附近 `(0.02,4)` 与 `(0.005,16)` 只用于显示不是单点偶然，不能在
held-out 结果后替换主配置。真正确认只能来自一次冻结后的 held-out EvalAI 运行。

该方法属于 Tier 2：读取目标日期前三条 calibration spectrogram 标签。论文必须与零标签方法分表。

## 6. 后续

若 source screen 三门均过：实现六日期 payload、official-shape decoder、本地 metric parity，审核后才
打包 EvalAI。不要先把 ARTP 接回 TARM 大网络；若 hidden 结果确认后仍需统一网络，可把 ARTP 当
冻结输出锚点，并只训练严格零初始化的 residual gate。
