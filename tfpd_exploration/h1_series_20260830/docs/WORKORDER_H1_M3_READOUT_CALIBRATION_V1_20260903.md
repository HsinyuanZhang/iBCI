# Work Order: H1-M3RC Source-Only Five-Fold Screen V1

Date: 2026-09-03  
Authority: `DESIGN_H1_M3_READOUT_CALIBRATION_V1_20260903.md`

## Authorized action

Run one five-fold source-only/date-LODO screen on physical GPU0.  The GPU is
used only for frozen C1 inference.  All readout fits are CPU float64 closed
forms.  GPU1 must not be queried or touched.

## Fixed execution

- seed: 42 (bookkeeping only; the closed form is deterministic);
- official support: chronological M3;
- model: exact sealed C1 epoch-49 checkpoint for each outer fold;
- families: `DIA7`, `MAT7`;
- ridge grid: `0, 1e-6, 1e-4, 1e-2, 1, 100`;
- target fit/backward/model updates: zero;
- no outer-date bytes before `selection_<date>.json` is immutable;
- no formal held-out, EvalAI, or network access.

## Result topology

Fresh root:

`tfpd_exploration/h1_series_20260830/results/h1_m3_readout_calibration_v1/`

Success contains attempt, five source-selection receipts, five fold receipts,
score, and terminal, each with a SHA sidecar.  Failure is immutable and the
root is never overwritten or retried.

## Decision

Use only the gates frozen in the design.  A passing screen authorizes an
all-source selector/package work order, not an automatic EvalAI push.

