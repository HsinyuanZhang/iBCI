# Result — B1 Activity-Reliability Template Profiling V2

日期：2026-09-03

> **勘误（2026-09-03）**：本文 query stream 末尾的 2 条 held-in-minival trial 与 held-in-calib
> 第 0/1 条逐字节相同，即 ARTP-P 的 M3 模板本身。去重后日期等权增益为 +5.46e-6（+1.28%），
> 非 +7.46e-5（18%）；20210626 上 ARTP-P 劣于 cyclic 与 neural-free 对照。以
> `CLOSURE_B1_REMOVED_FROM_MAINLINE_20260903.md` §2 为准，下文主数字不再有效。

## 结论

B1 的旧 NO-GO 只适用于 channel re-identification；它不适用于 trial-level activity signature。
ARTP-P 使用 M3-frozen per-channel mean/std profile 与 raw population-time signature 的等权相似度，
再用 M3 声谱可靠性 profile 抑制异常模板。它在三个 held-in 日期上同时超过强
`TPL-M3-median`，而且同时超过 neural-shuffle 与 neural-free profile 两个机制对照。

## 主数字

固定 `M=3, tau=.05, rho=2, raw/profile mix=.5`：

| 日期 | query | TPL-M3 MSE | ARTP-P MSE | 相对改善 |
|---|---:|---:|---:|---:|
| 20210626 | 10 | 0.000350166 | 0.000272345 | +22.22% |
| 20210627 | 28 | 0.000384070 | 0.000366335 | +4.62% |
| 20210628 | 7 | 0.000510542 | 0.000382166 | +25.14% |

日期等权绝对改善 `+7.4643e-5`，约为日期等权基线的 `18.0%`。三日期均为正。

相对 cyclic query-signature 的日期等权改善为 `+1.2034e-4`，3/3 正；相对只保留声学可靠性、
不读 query neural 的对照为 `+8.6798e-5`，3/3 正。日期—trial 层次 bootstrap 的 95% 区间：

- vs TPL-M3：`[+3.09e-6,+1.852e-4]`；
- vs cyclic neural：`[+9.11e-6,+2.554e-4]`；
- vs neural-free reliability：`[+6.34e-6,+2.183e-4]`。

## M2 迁移读数

可迁移的是强锚点、低容量校正、matched support 与无标签 query 使用；不可直接迁移的是 continual
profile。严格 decode-before-commit 的 growing mean/std 相对 fixed M3 profile，日期等权差只有
`+1.956e-7 MSE`（正号表示 growing 更差），因此本路线停止 growing profile。

J-R1/TARM 两次神经网络 pilot 保留为负对照：whole-trial pilot 四臂几乎相同；framewise pilot
虽然 profile 相对 zero 在同 checkpoint 有小改善，但所有臂都落后模板并随 epoch 过拟合。ARTP-P
的意义正是不给高容量 residual 覆盖强模板，只允许 query activity 在三条合法模板间调权。

## 限制

`tau/rho/mix` 是查看 held-in source 后冻结的，因此这些数字是探索性 source 证据，不是无偏泛化
估计。唯一确认是冻结实现后的 held-out EvalAI。方法读取目标日期 M3 spectrogram，属于 Tier 2；
不得与 label-free 方法混报。

## 部署复现

六日期本地 payload 已构建，SHA-256 为
`7c73e803752f938d6ca961dce8f4aeb167e5b420f4e41d160d43b81c739418fa`，大小
38,245,168 bytes。三个 held-in 日期通过 public decoder 的 `reset/predict` 重放后，逐日期
prediction SHA 与 source screen 完全相同，MSE float 也完全相同；query-label access=0、model
updates=0。状态为 `LOCAL_PACKAGE_COMPLETE_NOT_SUBMITTED`，尚未构建 Docker 或推送 EvalAI。
