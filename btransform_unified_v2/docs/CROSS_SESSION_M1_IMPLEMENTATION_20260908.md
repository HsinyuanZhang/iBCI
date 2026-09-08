# M1 cross-session Z/B/D implementation contract

This new program replaces neither old M1 runs nor their HO3 evidence.  It runs four public held-in LOSO folds, with target M10 `[0,10)`, reset query `[10,210)`, source train `[10,310)`, and source validation `[310,end)`.  Target labels never select an epoch and target gradients are zero.

Arms are `Z_NONE` (E0/T4 zero; no calibration tensor attached), `B_ACTIVITY_ONLY` (randomly initialized live B3S E0; carrier zero in both paths), and `D_JOINT` (the same B3S construction plus carrier in B3S-side and direct-decoder paths).  B/D share decoder and B3S random initialization per fold/seed.  The calibration encoder is never loaded from Sfix or an all-source checkpoint.

Each fold must fit rSyn3 dictionary and normalizer only from the three source sessions' support and train region.  The target and all source-validation labels are prohibited from that fit.  Source-validation EMA selection is earliest maximum equal-session mean; target reports only that selected EMA and the predeclared epoch-24 EMA.

The present command interface is:

```bash
python scripts/cross_session_v1/m1_train.py --dest results/cross_session_m1_v1/<fold>/<arm>/s42 --target ses-20120924 --arm Z_NONE --seed 42 --stage prepare
python scripts/cross_session_v1/m1_train.py --dest results/cross_session_m1_v1/<fold>/<arm>/s42 --target ses-20120924 --arm Z_NONE --seed 42 --stage preflight
```

`prepare` materializes only the three source sessions and atomically caches the
fold-local rSyn3 arrays. Its NMF basis sees source EMG trials `[0,310)` and
each source carrier sees only source M10. `train` repeats that source-only
reader and therefore never opens the target NWB, target M10, or target labels.
Only `score` materializes the target support and reset query stream.

Before every rSyn3 NNMF fit and NNLS projection, EMG is explicitly rectified
with `maximum(emg, 0)`. The source cache records the negative count, fraction,
and minimum of each source input. The cache also seals the fitted dictionary,
scaling, and source normalizer arrays.

All ranges are physically sliced at trial boundaries before constructing a
`FalconDataset`; this gives each support, train, validation, and query range a
separate left-padded raw stream. It prevents the last raw bins of a preceding
range from becoming context for the next range.

Run each stage in order (use the same destination for the arm and fold):

```bash
export PYTHONPATH=/home/xinyuan/Work_host/SPINT/btransform_unified_v2/src:/home/xinyuan/Work_host/SPINT/btransform_unified_v1/src:/home/xinyuan/Work_host/SPINT
python3 scripts/cross_session_v1/m1_train.py --dest results/cross_session_m1_v1/ses-20120924/Z_NONE/s42 --target ses-20120924 --arm Z_NONE --seed 42 --device cuda:0 --stage prepare
python3 scripts/cross_session_v1/m1_train.py --dest results/cross_session_m1_v1/ses-20120924/Z_NONE/s42 --target ses-20120924 --arm Z_NONE --seed 42 --device cuda:0 --stage preflight
python3 scripts/cross_session_v1/m1_train.py --dest results/cross_session_m1_v1/ses-20120924/Z_NONE/s42 --target ses-20120924 --arm Z_NONE --seed 42 --device cuda:0 --stage train
python3 scripts/cross_session_v1/m1_train.py --dest results/cross_session_m1_v1/ses-20120924/Z_NONE/s42 --target ses-20120924 --arm Z_NONE --seed 42 --device cuda:0 --stage score
```

The numerical preflight performs fresh initialization-byte, parameter-count,
range-reset, finite-forward/gradient, and arm-isolation checks. Training uses
an explicit paired `[1,64]` whole-unit dropout mask, local RIFT backend, BF16
decoder autocast with the calibration encoder kept in FP32, atomic per-epoch
EMA/resume checkpoints, and complete RNG restoration under `--resume`.
