# Work Order — B1 ARTP V1 Source Screen

日期：2026-09-03

授权一个 CPU-only、held-in source-only screen。不得打开 hidden query、不得提交 EvalAI、不得使用
GPU。输入只包括三个 held-in 日期各自的 held-in-calib 与 held-in-minival NWB。

冻结主配置：population-time mean/std signature，`M=3`，`tau=0.05`，
`reliability_strength=2`，`mixing=1`。每个日期的 reference 是 held-in-calib trial 0..2；query
顺序固定为 held-in-calib trial 3..N-1，随后两条 minival。minival 标签只评分，不参与 payload、
超参或状态更新。

输出：`tfpd_exploration/results/b1_artp_v1/source_screen.json`。必须逐日期记录四个系统的 official
per-trial MSE、prediction digest、query roster、三项 paired delta，并做 seed42/10,000 次日期—trial
层次 bootstrap。主门是三项 delta 各自 3/3 日期严格为正。

允许添加 ARTP 独立包、测试与 inert CLI；不得改写原 B1-SFCJ/TARM 收据。source screen 通过只授权
进入部署打包审计，不自动授权 EvalAI push。
