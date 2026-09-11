# M2 RIFT R50 fixed ACTIVITY_ONLY / NONE ablations

This directory owns an isolated cache, training adapter, and EXT6 selector for
the two fixed M2 information ablations.  It matches the official M2 baseline
**RIFT concat e9, submission 582189, held-out R² 0.34654225938843214**.  The
baseline's local EXT6 value is a development value and is never an official
metric.

`ACTIVITY_ONLY` uses literal `T[96,4]=0`. It rematerializes static `E0[96,50]`
with the same frozen EMPTY-head encoder, `empty_contrast_side(zero_T)`, and
native chronological `reset_stream` / `push_trial` / `finalize_identity`.
`NONE` uses literal zero E0 and T, and never calls the encoder. The decoder is
always the original static `RiftConcatDecoder`; no live or trainable encoder
is added.

ROOT runs the following separately for each arm. Every `--dest` must be fresh.
No command here packages, registers, pushes, or submits to EvalAI.

```bash
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
ROOT=btransform_unified_v2
PKG=$ROOT/scripts/m2_rift_r50_ablation_v1
BANK=$ROOT/results/rift_v1/m2_r50_activity_only_banks_v1
SMOKE=$ROOT/results/rift_v1/m2_r50_activity_only_s42_smoke_v1
RUN=$ROOT/results/rift_v1/m2_r50_activity_only_s42_formal_v1
SEL=$ROOT/results/rift_v1/m2_r50_activity_only_s42_ext6_v1
PYTHONNOUSERSITE=1 $PY $PKG/build_banks.py --arm ACTIVITY_ONLY --dest $BANK
PYTHONNOUSERSITE=1 $PY $PKG/m2_ablation.py --arm ACTIVITY_ONLY --bank-cache-root $BANK --dest $SMOKE --device cuda:0 --smoke-steps 2
PYTHONNOUSERSITE=1 $PY $PKG/m2_ablation.py --arm ACTIVITY_ONLY --bank-cache-root $BANK --dest $RUN --device cuda:0
PYTHONNOUSERSITE=1 $PY $PKG/score_ext6.py --arm ACTIVITY_ONLY --bank-cache-root $BANK --run-dir $RUN --dest $SEL --device cuda:0
```

Replace `ACTIVITY_ONLY` with `NONE` and use distinct BANK/RUN/SEL paths for
the NONE arm. Formal training is fresh 24 epochs, 3,165 updates per epoch
(75,960 total), seed 42, R50/D4 local attention windows 13/12/12/12, batch 32,
AdamW 3e-4 with one-epoch warmup, EMA .9995, and unit dropout .1. The selector
requires all 24 EMA checkpoints and chooses the earliest maximum unweighted
six-session EXT6 mean. It is a local selection receipt, never an official
held-out result.
