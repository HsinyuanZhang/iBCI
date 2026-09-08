# M1 RIFT e3 inference evidence — 2026-09-08

此脚本只汇总已完成的固定 e3 推理干预 ledger，不训练、评分或修改权重。它要求准确 25 个 conditions：`M10`、`ZERO`、三项 direct-carrier shuffle、M5 与 M8 各十个 support-trial resamples。所有条件保持 E0 为 M10 B3 identity；M5/M8/M10 仅表示 direct functional carrier 的标签 support（50%/80%/100%），不是总 calibration budget 减少，也不是重训 B/D arms。

执行命令（由协调者在 CPU 10,11 启动）：

```bash
cd /home/xinyuan/Work_host/SPINT/btransform_unified_v2
/home/xinyuan/miniconda3/envs/spint/bin/python scripts/diagnostics_v1/summarize_plot_m1_inference_evidence.py \
 --input results/diagnostics_v1/m1_rift_e3_inference_ablations_v1.json \
 --outdir results/diagnostics_v1/m1_rift_e3_inference_evidence_20260908
```

脚本先核验 checkpoint、score receipt、25 conditions、所有三个 session 的 3881 完整 windows、finite R²、等权 mean、prediction/carrier hashes；之后生成 `figure_data.json`、PNG 和 PDF。图中 resample spread 是 within-condition repeat range。若生成 session bootstrap，仅按三个 sessions 重采样，明确不是十个模型 seeds，且不作显著性结论。

## 已生成 figure data 的核对

`figure_data.json` 的 input SHA 与原始 25-condition ledger SHA 一致：`a6d281bc57298ac05ea10c8435100a69670c004b1dc813ba1c5f8907371d1e7f`。其 REAL equal-session mean 为 `0.7046861373`，ZERO 为 `0.6109810877`，三个 shuffle conditions 的 condition-mean 平均为 `0.6176515583`，对应 REAL−ZERO `0.0937050495`、REAL−shuffle-average `0.0870345789`。M5 十个 carrier resamples 的 mean 为 `0.6963491914`、condition-mean range `0.6865356512–0.7035501891`；M8 为 `0.7039792611`、range `0.6996409027–0.7081392480`；M10 为 `0.7046861373`。

这些范围是 condition repeats 的描述性范围，不是置信区间。文件中的 session bootstrap 明确只对三个 sessions 重采样，既不是 training-seed bootstrap，也不能称为 training-seed CI 或显著性结果。
