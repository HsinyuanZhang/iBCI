# M2 cross-session Z/B/D implementation

This is a new sealed implementation.  It does not modify any historical M2
model, runner, receipt, result directory, EvalAI artifact, or DANDI-688 path.

## Arms

All arms use the same M2 query-spike frontend, RIFT-R50/D4 decoder, batch-32
source-seven manifest, optimizer/schedule, EMA, seeds 42/43/44, and 24 epochs.
They are independently trained; an arm is never obtained by inference-time
intervention on another arm.

| Arm | M33 calibration encoder | T4 to FiLM | T4 direct decoder token |
| --- | --- | --- | --- |
| `Z_NONE` | absent; no calibration/E0/T4 read | off | off |
| `B_ACTIVITY_ONLY` | live M33 activity encoder | off | off |
| `D_JOINT` | live M33 activity encoder | on | on |

The identity encoder is `HoldContrastFiLMEarlyPoolEncoder`, a B3S substrate
with a hold-vs-reach FiLM extension.  It is initialized from scratch with
`--encoder-init random` (the only permitted value and the default), so no
historical champion, `p0`, EMPTY head, teacher checkpoint, or historical
EXT4-selected artifact enters this formal experiment.  B and D receive byte-
identical random encoder initial states at a given seed.  B still derives its identity from activity
and may retain activity-derived contrast internal to that encoder; it receives
zero carrier values through every T4 path.  D uses actual T4 both as the FiLM
side feature and the existing direct decoder-token carrier feature.

`Z_NONE` intentionally does not instantiate a calibration encoder.  The receipt
therefore reports both total and active training parameter counts.  Preflight
rejects a greater-than-5% active-parameter difference, rather than disguising a
parameter mismatch with an unused encoder.

## Data and selection isolation

The historical `source_minival` cache is excluded from this implementation:
its raw audit found calibration-prefix reuse.  Gradients use the first 80% of
post-M33 whole trials in each source-seven `source_train` session.  Validation
uses the fixed final 20% of those whole trials, rounded up.  The query timeline
remains continuous across trial boundaries inside each segment.  The splitter
uses the whole-trial boundary as the sole train/validation cut: train windows
must end before it, and validation starts at the cut itself: its first scored
endpoint is therefore cut+49, so the initial 49 endpoint bins are naturally not
scored while every retained 50-bin validation context begins inside validation.  The splitter
derives the cut from held-in-calib NWB covariate-clock trial boundaries, rebuilds
the deterministic 24-epoch batch-32 manifest from the actual training windows,
and records its digest and per-session trial/window contract.
The per-epoch EMA curve is scored only on that source-trial validation view.
EXT4 is not loaded during `train`; `score` loads EXT4 only
once after that selection and reports the selected EMA plus predeclared e24 as
a sensitivity endpoint.  Neither EXT4 target labels nor EXT4 score can affect
optimization or epoch selection.

Preflight verifies roster separation (source-seven vs EXT4), source-minival
mapping declarations, arm switches, encoder gradient behavior, exact shared
decoder initialization, and parameter counts.  It is the required executable
evidence before formal jobs.

The new query loader does not call the historical generic M2 bank builder.  It
opens only `X_store.npy`, `target_store.npy`, `eligible_starts.npy`, and mapping
metadata to construct a zero compatibility bank.  Thus Z never opens
`calib_activity.npy`, `T.npy`, or `e0_u.pt`; B opens only raw
`calib_activity.npy`; D opens raw `calib_activity.npy` and `T.npy`.  All arms
share the same query-window sampler and an all-observed 96-channel query mask,
rather than reading `e0_u.pt` to recover the historical frozen mask.

## Commands

Run from `btransform_unified_v2` with the project environment.  The inherited
M2 cache and frozen 24-epoch source manifest must be present at their existing
paths.

```bash
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/xinyuan/miniconda3/envs/spint/bin/python

$PY scripts/cross_session_v1/m2_train.py \
  --dest results/cross_session_v1/m2_z_s42_v1 --arm Z_NONE --seed 42 \
  --stage preflight --device cuda:0 --cpu-threads 2 --encoder-init random

$PY scripts/cross_session_v1/m2_train.py \
  --dest results/cross_session_v1/m2_z_s42_v1 --arm Z_NONE --seed 42 \
  --stage train --device cuda:0 --cpu-threads 2

$PY scripts/cross_session_v1/m2_train.py \
  --dest results/cross_session_v1/m2_z_s42_v1 --arm Z_NONE --seed 42 \
  --stage score --device cuda:0 --cpu-threads 2
```

Repeat for `Z_NONE`, `B_ACTIVITY_ONLY`, and `D_JOINT`, each at seeds 42, 43,
and 44.  Training always retains every EMA-only epoch state, so a source-
minival selected epoch before e24 and the predeclared e24 sensitivity can both
be scored later.  The only resumable full state is atomically replaced at `resume_latest.pt`; resume
uses `--stage train --resume <dest>/resume_latest.pt` with the same destination.

## Formal done criteria

For each of the nine jobs, require: completed preflight; matching source hashes
and whole-trial split-manifest digest; the receipt's dynamically derived update
count; complete 24-row source-trial-validation EMA curve; selected EMA state;
`target_query_labels_used_for_selection: false` and
`target_query_labels_used_for_gradients: false`; and one EXT4 score receipt
with both source-minival-selected and e24 reports.  Aggregate the three seeds,
never EXT4 sessions as independent seeds.
