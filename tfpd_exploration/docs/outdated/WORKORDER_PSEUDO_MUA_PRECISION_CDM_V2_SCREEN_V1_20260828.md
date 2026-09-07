# Pseudo-MUA Precision-CDM V2 Screen V1

## Purpose

Test whether the accepted Precision-CDM V2 carrier gate transfers from sorted SUA to pseudo-MUA. This is an engineering replication, not a new formal benchmark.

## Fixed experiment

- Data: the same 15 DANDI 000688 sessions used by the frozen paired SUA/pseudo-MUA screen.
- Signal view: `pseudo_mua` only.
- Model: frozen `shared_t4`, seed 42.
- Query: all valid 50-bin windows strictly inside rewarded trials after the first 30 eligible rewarded trials.
- Support:
  - M4: D-optimal four trials selected from the first 30.
  - M10: chronological first ten trials.
  - M30: chronological first thirty trials.
- M4 and M10 systems: static T4, activity-only FIFO, ordinary independent-activity CDM, and Precision-CDM V2.
- M30 systems: static T4 and Precision-CDM V2 literal no-op. The M30 predictions must be byte-identical.
- Metric: last-bin two-coordinate variance-weighted R2 per session, then equal-session means and paired deltas.

## Causal and safety rules

- A query trial is predicted before it can update state.
- State may use neural activity and frozen-model predictions only.
- Query behavior is used only after prediction for R2.
- No target gradient, backward call, optimizer step, normalizer refit, or model update is permitted.
- The activity-only arm appends each valid completed B3S activity trial but never changes its support carrier.
- Ordinary and Precision arms share the same core proposal. Precision V2 may only veto an otherwise accepted carrier commit against the frozen support posterior.

## Decision

The cross-view mechanism signal is the paired Precision-minus-ordinary delta at M4 and M10. M30 is only a no-harm/no-op invariant. Seeds 43 and 44 are authorized only after seed 42 gives a positive mechanism signal without violating M30.
