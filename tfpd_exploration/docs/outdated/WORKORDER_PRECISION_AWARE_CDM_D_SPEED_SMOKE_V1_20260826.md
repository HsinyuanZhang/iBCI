# Precision-Aware CDM-D speed smoke V1

## Purpose

This is a narrow, non-governing GPU0 engineering smoke for the accepted
`PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V2` result.  It
measures the already-reviewed O1 identity-cache and O2 deterministic sampled
repeat machinery without changing a carrier proposal, precision decision,
normalizer, decoder, query ordering, metric, or score gate.

## Frozen historical evidence

The route must descriptor-read the exact 12-leaf V2 graph (two authority body
pairs and four result body pairs).  It binds the supplied body digests, the
canonical attempt identity digest, final/launch closure, and the one selected
canonical Precision row.  A current execution has a distinct, root-selected
GPU0 compatible profile; historical V2 GPU1 evidence is provenance only.

## Bounded matrix

Exactly one Precision-V2 dynamic cell runs:

1. M4 external `sub-M_ses-CO-20150615`.

M30 is not rerun and no sealed comparator is rerun.  Before a physical forward
the route reconstructs and re-hashes the exact historical V2 cell row and
requires the selected row's input record, prediction digest, R2, window count,
model-state digest, sealed-load proof, and carrier-transition commit count.

## Baseline and numerical-equivalence contract

The single session runs two independent causal evaluations back-to-back on the
same materialized input records and sealed model state.

- The eager baseline uses physical B128 and must reproduce the accepted V2
  prediction SHA256, R2, and full transition accept/reject/state sequence
  exactly.
- The O1/O2 arm computes a B=1 cached identity and uses the deterministic
  sampled repeat coordinates.  It first attempts physical B1024; a genuine
  CUDA OOM may retry the same untouched session from the beginning at B512,
  then B128.  The selected physical batch and any OOM fallback are receipt
  fields.
- GPU evidence established that direct cached-identity decoding is not
  bitwise-identical to eager B128 on RTX3090.  The optimized prediction SHA is
  therefore disclosed separately.  It must have maximum absolute prediction
  error at most `1e-6` against the live eager array, R2 absolute difference at
  most `1e-7`, and an exactly equal causal transition accept/reject/state
  chronology.  These numeric allowances do not weaken the eager historical
  anchor.

The receipt persists both row anchors, compact O1/O2 evidence, logical and
actual forward counts, selected repeat coordinates, wall time,
windows-per-second, and current/peak memory.  A speedup strictly greater than
1.0 completes this smoke; `1.5x` is recorded as a recommendation for a later
engineering decision, not a governing scientific threshold.

The design is motivated, but not authorized, by a separate GPU0 synthetic
precursor: B1024 completed in approximately `0.07970s` versus eager-repeat
B128 in approximately `0.32850s` (`4.12x`), with maximum prediction error
`5.960464477539062e-07`, R2 difference `5.98374754190445e-08`, TF32 disabled,
and optimized peak allocation/reservation `954813952`/`1199570944` bytes.
Those values are descriptive only; the live smoke records its own selected
batch, parity evidence, timing, and memory and never treats this precursor as
a result or acceptance threshold.

## Lifecycle and safety

Public CLI is dry only.  A root-only opaque capability is required to reserve
fresh authority and score roots.  Exact launch variables are checked before
any reservation/publication and again before prepare/finalization:
`CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and the two canonical
source-root strings.  GPU1 is an explicit rejection for this smoke.  The
shared immutable score lifecycle owns attempt-before-prepare, atomic
score+terminal publication, and honest failure publication.  The historical
V2 graph is re-held/revalidated before authority reserve, capability issue,
score reserve, physical prepare, and final publication.

No target optimizer, backward, update, source/target refit, comparator
retraining, TF32, AMP, compile, data cache, or result reuse is permitted.

## Required no-data tests

Synthetic tests cover held exact topology/SHA/sidecar and row reconstruction,
identity and closure drift, GPU0-only environment gating before any write,
stale predecessor, device drift, reserved-root validation, attempt-before-
prepare, parity mismatch failure, speed-evidence tampering, and atomic success
or failure topology.  The public CLI must remain Torch-free and inert.

## Deferred live gate

Only an independently reviewed root may select a compatible idle GPU0,
construct the capability, and run the one-cell smoke.  A terminal is an
engineering parity/throughput observation, never a governing scientific gate.
