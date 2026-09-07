# Work Order — B1 ARTP Profile Source Screen V2

日期：2026-09-03

本 successor 不改写 ARTP V1。它在 V1 的 raw population-time signature 之外增加一个 M3-frozen
profile view：仅用前三条 calibration neural 的全部 1 ms bin，计算每通道 mean/std；reference 与
query 都在同一 profile 下标准化，再生成 population-time mean/std signature。raw/profile cosine
等权，冻结 `mix=0.5`；声学可靠性仍为 `tau=0.05,rho=2`。

授权 CPU-only source screen，新根 `tfpd_exploration/results/b1_artp_v2/source_screen.json`。query 顺序、
四个主对照、三道 3/3 门与 V1 相同。另增加严格因果 growing-profile 描述臂：预测当前 trial 时只用
M3 加已完成 trial 的 activity 更新通道 mean/std，预测后再提交当前 activity；不得用当前或过去的
query spectrogram。若 growing 未优于 fixed，明确停止 continual-profile，不得以 M2 先例强推。

本 source screen 仍是查看 held-in 后的探索性结果；通过只授权部署实现与 hidden confirmation，
不自动授权 EvalAI push。
