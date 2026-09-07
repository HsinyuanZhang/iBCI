# Workorder: low-cost calibration-support diagnostics

Date: 2026-08-23

Status: authorized non-formal diagnostic, V3 successor. V1 is retained as an
immutable failed predecessor: it completed input materialization and then
failed before any experimental forward because the route checked the opaque
session type against the wrong module alias. V2 corrected that alias and
entered experimental inference; prediction SHA parity passed, but it compared
a NumPy float64 audit R2 to the governing TorchMetrics float32 R2 and failed on
the estimator-precision mismatch. V3 must validate both exact predecessor
attempt/failure graphs before reserving its fresh root. The governing value is
the unchanged reviewed `runtime.score_result`; float64 decomposition is
reported only as an audit. Performance claims remain governed by
the sealed mainline receipts, not by this result.

## Objective

Determine whether the remaining useful carrier headroom comes from cue
coverage, within-session temporal drift, or neither. Reuse the sealed Cell-D
SWA and the reviewed V4 fixed-input scorer. Do not train a model.

## Fixed graph and evaluation

- Model: sealed Cell-D final-four SWA, seed 42.
- Activity identity: the same M30 B3S calibration tensor in every cell.
- Carrier normalizer: the sealed ordinary-OLS source normalizer; no target
  refit.
- Query windows: the exact V4 within-6 and external-15 inputs.
- Score: valid last bin (bin 49), variance-weighted R2 per session, then equal
  session weighting.
- Evaluation batch size: 128.
- No optimizer, backward, update, target adaptation, cache, or formal surface.

## Support cells

Run M4 and M30 for every support rule:

1. `C0_contiguous`: first M rewarded trials. This must reproduce the immutable
   V4 sealed Cell-D row exactly.
2. `C1_cue_balanced_early`: greedy D-optimal cue selection from the first 50
   rewarded trials. Selection sees target cue angles only.
3. `C2_cue_matched_scattered`: the exact C1 direction multiset, but selected
   deterministically across the full session timeline. This is a leakage/oracle
   diagnostic, not a deployable calibration rule.
4. `C3_full_session_oracle`: all rewarded trials. This is an explicit labeled
   upper-bound diagnostic and is identical across the M4/M30 labels.

The carrier-only CPU audit also reports M4/M10/M15/M20/M30/M50 for C0-C3 and
the pre-existing D-optimal M10/M15 gate.

## Required outputs

For every session/cell report:

- governing variance-weighted R2;
- coordinate R2 for x and y;
- R2 after rotating into the target-covariance eigenbasis;
- target covariance eigenvalues and participation ratio;
- prediction, target, carrier, selected-row, and input digests;
- selected direction counts, design condition, and temporal span.

Aggregate only by equal-session mean/median and paired per-session deltas.
The C0 parity failure, a state change, active dropout, any target update, or any
formal access fails the run closed.

## Decision

- If C3 is near C0 and C2 is near C0, carrier estimation/tracking is closed.
- If C3 improves and C2 approaches C3, temporal tracking is the only surviving
  carrier branch.
- If C3 improves but C2 stays near C0, finite-support estimation is the likely
  bottleneck.

No new training is authorized by this workorder.
