# M1 R100 NONE ablation runner

This is the fixed `NONE` information ablation for the official M1
`muscle_response16` RIFT R100 baseline. It keeps the source-four sampler,
W100/D4 `[25,25,25,24]` decoder, P16 concat frontend, batch 32, AdamW,
EMA, unit dropout, seed 42, and 24 × 6,665 updates. It uses a fresh decoder
initialization and never warm-starts from the M1 main or decoder checkpoint.

`NONE` retains the normal neural query. It supplies literal `E0[64,100]=0`
and direct `T[64,4]=0`, does not call B3S, freezes its unused encoder
parameters, and stores no calibration side path. The runner does not accept a
carrier pack. Its route receipt records both train and score proof points.

The completed CPU two-step smoke is recorded at
`btransform_unified_v2/results/final_ablation_official_v1/m1_none_smoke_cpu_v1`.
It used the current frozen original-muscle seed-42 run
`btransform_unified_v2/results/m1_muscle_r100_v1/formal_s42_gpu1` as its
baseline binding. The historical matched `B_ACTIVITY_ONLY` run is a source
identity reference only; it is not the default official baseline for NONE.

Run a GPU smoke only in a new directory when another smoke is needed. GPU 1
is selected explicitly and appears to the process as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/m1_muscle_r100_none_ablation_v1/train.py \
  --stage train --smoke-steps 2 --device cuda:0 \
  --baseline-run btransform_unified_v2/results/m1_muscle_r100_v1/formal_s42_gpu1 \
  --dest btransform_unified_v2/results/m1_muscle_r100_none_ablation_v1/smoke_s42
```

Run formal training only after the smoke receipt passes, using a previously
nonexistent destination:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/m1_muscle_r100_none_ablation_v1/train.py \
  --stage train --device cuda:0 \
  --baseline-run btransform_unified_v2/results/m1_muscle_r100_v1/formal_s42_gpu1 \
  --dest btransform_unified_v2/results/m1_muscle_r100_none_ablation_v1/formal_s42
```

Score only a matching completed 24-epoch formal run. This scans EMA epochs
1–24 across all visible HO3 windows and chooses the earliest maximum equal
session mean channel variance-weighted R².

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/m1_muscle_r100_none_ablation_v1/train.py \
  --stage score --device cuda:0 \
  --baseline-run btransform_unified_v2/results/m1_muscle_r100_v1/formal_s42_gpu1 \
  --dest btransform_unified_v2/results/m1_muscle_r100_none_ablation_v1/formal_s42
```

No package, image build, EvalAI registration, or submission is performed by
this directory.
