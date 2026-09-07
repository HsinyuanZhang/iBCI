# Work Order — B1 ARTP-P V2 Local Package

日期：2026-09-03

前置：`results/b1_artp_v2/source_screen.json` 三道主门全过。授权 CPU-only 构建六日期 payload：
三个 held-in 与三个 held-out 日期都只读各自已发布 held-in/held-out calibration 的前三条 neural 和
spectrogram；不得打开 hidden query。

输出为 `artp_profile_payload_v2.pkl` 与 `package_receipt.json`。必须在序列化再加载后，用 public
`B1ARTPProfileDecoder.reset/predict` 重放三个 held-in 日期的全部 calib 4..N + minival，逐日期要求
prediction SHA 与 source screen 完全相同、MSE Python float 完全相同、query-label access=0、
model updates=0。decoder 返回必须严格是 `[158,880]`。

本单不授权 Docker build、网络访问或 EvalAI push。通过只产生一个可供独立审核的本地候选。
