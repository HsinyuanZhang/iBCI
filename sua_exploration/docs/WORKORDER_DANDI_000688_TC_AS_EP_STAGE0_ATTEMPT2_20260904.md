# Work Order — DANDI 000688 TC-AS-EP Stage-0 Attempt 2

Status: `AUTHORIZED_SELECTOR_REVISION_1_OF_1`

Date: 2026-09-04 (Asia/Hong_Kong)

## Authority and predecessor

The user authorized the experiment and the governing design permits exactly
one selector revision using source-only constructibility/bias/stability
evidence before any decoder R2.

Attempt 1 is retained unchanged at:

`sua_exploration/results/dandi688_tc_as_ep_v1/stage0_attempt1/`

Its body SHA-256 map is:

- `attempt.json`: `b69383410265ceb6fc0cc23d4609ab9ae51d23908f12d731f8329f3d6058d7b8`
- `selector_authority.json`: `e529fd7adf4d9cbab93f40eb0d7a384e1c30f0f8bf988c008c668887196f1fd3`
- `selector_decision.json`: `1d0f2bac594452e7b637eeda851b57296faa9079a38845c4cba0c6f84d3c0f90`

Attempt 1 passed source C50/Q50 authority, direction coverage, leave-one-unit
stability, stateless random entropy, and global RNG isolation. It failed only
because `sub-C_ses-CO-20151110` had absolute selected-membership/duration
point-biserial correlation `0.22597925813276998`, above the frozen `0.20`
ceiling. No decoder, R2, GPU, formal target, or external target was opened.

## Sole revision

The governing amended design is:

`sua_exploration/docs/DESIGN_DANDI_000688_ACTIVITY_SELECTED_EARLY_POOLING_V1_20260904.md`

SHA-256:

`3e04774363f8e35ff7ebb158d72a6492f04ce317fc7ed46903accba1db3d8752`

Before robust centering and MAD scaling, regress each unit's
`log1p(raw_spike_rate_hz)` linearly on centered `log(trial_duration_seconds)`
using all first-50 activity trials. No other selector formula, gate, threshold,
support budget, label budget, query boundary, or roster changes.

## Execution

Run the same complete 27-train/6-development source authority and six-session
selector gate under a fresh root:

`sua_exploration/results/dandi688_tc_as_ep_v1/stage0_attempt2/`

Attempt 2 must bind attempt 1 and the amended design. If any selector gate
fails, TC-AS-EP V1 stops: there is no third selector attempt and GPU training
must not start. If it passes, proceed to the remaining no-R2 operator parity
and teacher/loss checks before issuing the already authorized GPU0 training
capability.
