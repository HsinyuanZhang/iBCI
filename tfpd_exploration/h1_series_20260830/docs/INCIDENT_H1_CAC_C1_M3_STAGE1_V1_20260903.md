# Incident: H1-CAC C1/M3 Stage-1 V1 admission failure

Date: 2026-09-03  
Status: `CLOSED_BY_ADDITIVE_V2_SUCCESSOR`

## Immutable V1 failure

The first Stage-1 execution reserved and sealed:

`tfpd_exploration/h1_series_20260830/results/h1_causal_activity_completion_v1/stage1_c1_m3/`

- `attempt.json` SHA-256: `8324d2a5c8747f47d56f9854dcecf33b644574e185c1c48395ef7fd3f0669446`
- `failure.json` SHA-256: `6a14335da0e25f6cbe5dc51602ad454f3af3e658fe351d87754a269d5b9d2162`
- exception: `PilotDataError: frozen carrier requires exactly four TrialNum values`
- published prefix: `attempt.json` only
- target optimizer/backward/model updates: all zero
- no score or scientific result was produced

The V1 root is immutable and must not be retried or overwritten.

## Root cause

The Stage-1 implementation accidentally called the historical
`fit_frozen_carrier`, whose contract is exactly M4. The frozen design and the
existing deployment implementation instead require
`fit_deployment_carrier`, which accepts exactly three or four unique support
trials, forbids duplication/padding, and otherwise uses the same frozen
ridge/EB equations.

This is an integration error before scoring, not a checkpoint, data, model,
or scientific failure.

## Authorized successor

The additive successor changes only that helper call:

- old: `fit_frozen_carrier(record, plan, M3)`
- corrected: `fit_deployment_carrier(record, plan, M3)`

All five C1 checkpoints, source-selected carrier plans, M3 identity support,
query surface, four arms, thresholds, and frozen-weight/no-update contract
remain unchanged. The successor writes only to:

`tfpd_exploration/h1_series_20260830/results/h1_causal_activity_completion_v1/stage1_c1_m3_v2/`
