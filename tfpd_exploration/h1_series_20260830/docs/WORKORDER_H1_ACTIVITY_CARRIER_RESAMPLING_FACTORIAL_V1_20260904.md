# Work Order: H1 Activity/Carrier Resampling Factorial V1

Date: 2026-09-04  
Status: `AUTHORIZED_TRIGGERED_SOURCE_ONLY_GPU0_TWO_ARM`

Execute `DESIGN_H1_ACTIVITY_CARRIER_RESAMPLING_FACTORIAL_V1_20260904.md`
exactly once.

- main-result predecessor terminal SHA256:
  `c047bbf5bd6e324fcd250ef2702ab507fff54b6a13c6bc1314963983bc298a94`;
- train only `LP-AR-CF` and `LP-AF-CR`; reuse sealed `LP-F3` and `LP-R3`
  outer scores only after each new fold has completed training and scoring;
- preserve the exact main experiment optimizer, endpoint, support-index,
  branch-only, checkpoint-before-outer-open, and zero-target-update contracts;
- use physical GPU0 only; never query, signal, or touch GPU1;
- no formal held-out, EvalAI, all-source refit, hyperparameter sweep, or model
  selection.

Fresh result root:

`tfpd_exploration/h1_series_20260830/results/h1_activity_carrier_resampling_factorial_v1/`

Publish attempt, two branch checkpoints and training/fold receipts per date,
score, and terminal.  Do not retry or overwrite the root after failure.  This
root produces a mechanism classification, not a deployable winner.

