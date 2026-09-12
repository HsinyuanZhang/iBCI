# External gradient-free controls — frozen v1 archive

`FROZEN_PROTOCOL_INVALID_FOR_COMPARISON`: the v1 runs use one fixed source session and `ridge=1` without an official-preprocessing matched control. They are retained for traceability only. They are not fair RIFT comparisons, EvalAI candidates, or evidence for an alignment method. Do not push or register their images. The replacement plan is [fair_v2](fair_v2/).

This archived directory evaluated four CPU-only controls using a fixed causal
source-day Wiener ridge decoder with ten bins of history:

- `wf_raw`: raw neural activity to Wiener decoder;
- `coral_wf`: target neural-only CORAL adaptation then Wiener decoder;
- `fa_wf`: source FA representation then Wiener decoder;
- `aligned_fa_wf`: target neural-only FA, stable loading-row selection, and
  orthogonal Procrustes alignment then Wiener decoder.

Target labels are never supplied to CORAL or AlignedFA fitting, and there is no
target backpropagation. These are public-calibration local development controls;
they are **not** EvalAI results and should not be described as final or official
performance results.

## Historical reproduction only

Run from the repository root. Set `CUDA_VISIBLE_DEVICES=''` to keep the command
CPU-only and use two numerical-library threads:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  -m btransform_unified_v2.external_baselines_v1.run \
  --task m2 \
  --dest btransform_unified_v2/external_baselines_v1/results/m2_full_replay_v1 \
  --fa-dim 10 --fa-max-iter 1000 --fa-n-init 3 \
  --fa-stable-fraction 0.5 --ridge 1 --history 10 \
  --coral-ridge 0.001 --coral-shrinkage 0.1
```

The M1 and H1 full commands are independently copyable:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  -m btransform_unified_v2.external_baselines_v1.run \
  --task m1 \
  --dest btransform_unified_v2/external_baselines_v1/results/m1_full_replay_v1 \
  --fa-dim 10 --fa-max-iter 1000 --fa-n-init 3 \
  --fa-stable-fraction 0.5 --ridge 1 --history 10 \
  --coral-ridge 0.001 --coral-shrinkage 0.1

env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  -m btransform_unified_v2.external_baselines_v1.run \
  --task h1 \
  --dest btransform_unified_v2/external_baselines_v1/results/h1_full_replay_v1 \
  --fa-dim 10 --fa-max-iter 1000 --fa-n-init 3 \
  --fa-stable-fraction 0.5 --ridge 1 --history 10 \
  --coral-ridge 0.001 --coral-shrinkage 0.1
```

Every command above uses a fresh, empty `*_full_replay_v1` destination; the
runner's fresh-directory guard rejects existing output directories. The already
completed M2 score correction is a deterministic rescore record at
`results/m2_full_v1_corrected`; it retains the original full run and records the
current dual metric contract. Do not overwrite an existing result directory.

## Historical metrics and completed artifacts

M1 and M2 receipts report both metrics below for every method and session:

- `full_compatible_flattened_*`: the historical FULL selector's flattened
  helper;
- `standard_variance_weighted_*`: centered, per-output
  variance-weighted R².

The metrics use different aggregation denominators. They are reported side by
side and must not be directly subtracted or treated as interchangeable.

H1 includes those two metrics plus `grouped_seven`, the existing seven-session
protocol. `grouped_seven.per_recording_r2` preserves its two recordings per
session, while `grouped_seven.r2_mean`, `r2_std_population`, and
`worst_session_r2` are calculated across the seven session values. The H1
receipt states that alignment uses positional unit rows and that physical
channel correspondence is unverified.

| Task | Completed artifact | Local scope |
|---|---|---|
| M2 | `results/m2_full_v1/receipt.json`; corrected score: `results/m2_full_v1_corrected/receipt.json` | EXT6, 15,403 windows |
| M1 | `results/m1_full_v1/receipt.json` | HO3, 3,881 windows |
| H1 | `results/h1_full_v1/receipt.json` | public M3, 33,613 windows; grouped-seven |

The retained smoke receipts are `results/m2_smoke_v2/receipt.json` and
`results/m1_smoke_v1/receipt.json`. Smoke execution validates wiring only and
does not establish decoding quality. Independent reload, prediction, metric, and
H1 grouped-seven recomputation checks for all three full runs are recorded in
`results/final_verification/receipt.json`.

## M1 FactorAnalysis convergence record

The completed M1 AlignedFA receipt records that the shared source FA fit reached
the configured `fa_max_iter=1000` without declaring convergence. For target
session `20121017`, its target FA fit also reached 1,000 iterations without
declaring convergence. The target log-likelihood increment at that limit was
`0.002351`, above the configured tolerance `0.001`; parameters stayed finite and
all noise variances stayed positive. This is a finite-iteration-budget result,
not a numerical failure and not a successful-convergence claim. The other two
M1 target FA fits did converge (151 and 217 iterations); all per-session
diagnostics are retained in `results/m1_full_v1/receipt.json`.
