# Findings

## M1 ORT (sealed, 2026-09-07)

- intra2 container minival B4, 1039 predicts
- max_abs vs P0 = 2.50e-6, 0 violations
- R² 0.96298 identical
- steady mean 9.96 ms/call, Normalized Latency 0.163
- first call 163 ms, session init 0.069 s
- backend: onnxruntime 1.19.2 CPUExecutionProvider
- graphs: `ort_adv_b{B}.onnx` + `ort_rebuild_b{B}.onnx`, B=1..4

## GPU (2026-09-07 12:41 +0800)

- GPU0: M1 depth-2 fullsession train, pid 1873248, ~2.2 GiB
- GPU1: H1 C2-CAL-1 B2 HO-M3 sweep, pid 1870315, ~14 GiB, pick not sealed yet

## Sealed payloads

- H1: `tfpd_exploration/submissions/evalai_h1_projadd_exacte_v1/artifacts/h1_projadd_stage2_s42_ema_e24_L200.pkl`
- M2: `tfpd_exploration/submissions/evalai_m2_projadd_p32_exacte_v1/artifacts/m2_projadd_p32_s42_ema_e07_ext6.pkl`
