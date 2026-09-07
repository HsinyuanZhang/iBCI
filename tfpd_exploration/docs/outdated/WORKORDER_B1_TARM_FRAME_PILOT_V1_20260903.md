# Work Order — B1-TARM Framewise M2-Compatible Pilot

日期：2026-09-03

授权 GPU0 单次 source-only `fold=0/seed=42/12 epoch`。新根：
`tfpd_exploration/results/b1_tarm_frame_v1/pilot_fold0_seed42`。GPU1、held-out query、
EvalAI 均不触碰。

本 successor 保留 TARM 四臂/profile/template/memory 合同，只把 decoder 从“整条 900 ms
压成一个 token 后生成整张声谱”改为 M2 风格的 last-frame decoder：每个 official valid
frame 使用截至该 frame 的 64 ms causal neural window，输出该 frame 的 158 维模板残差。
每条 source query trial 提供 700 个监督帧；activity identity 对该 trial 只计算一次，当前
trial 仍不进入自己的 history。

四臂是 `F-TA-N0/F-TA-NS9/F-TA-J0/F-TA-JS9`。carrier columns 使用同一非零随机初始化，
residual head 为零，因此 step 0 四臂仍精确等于 TPL-M3，但 profile arm 在第一次更新即有
不同的 head gradient；避免 whole-trial pilot 的双零冷启动。训练设置：width128、8 heads、
Adam `1e-4`、unit dropout `.10`、12 epoch。Pilot 只作方向读数，不单格淘汰。

