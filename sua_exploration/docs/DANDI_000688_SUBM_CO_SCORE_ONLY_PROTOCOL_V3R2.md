# External sub-M score-only T4-vs-TS4 protocol V3R2

Status: `STATIC_PRELAUNCH_ONLY_NOT_AUTHORIZED`

V3R2 is an append-only successor to the V1/V2/V3 score-only packages. It is
prepared for execution only after the separate Native-M2 experiment has
finished. This document, the V3R2 prelaunch bundle, and its authorization
template do not authorize external scoring, create a nonce claim, open a
sub-M NWB file or checkpoint, run a forward pass, or compute R2.

## Scientific endpoint

The frozen cohort is the same ordered set of 15 eligible sub-M center-out
sessions in DANDI 000688. The matrix is fixed before any external score is
opened:

- views: `sua`, `pseudo_mua`;
- arms: `shared_t4`, `shared_ts4`;
- seeds: 42, 43, 44;
- total cells: `15 * 2 * 2 * 3 = 180`.

The deployment budget is exact and asymmetric by design:

- activity identity consumes the first 30 chronological rewarded trials;
- the T4/TS4 descriptor fit consumes the first 50 chronological rewarded
  trials;
- evaluation uses only valid windows strictly after rewarded trial 50.

This is the budget verified by the consumed-sub-C V5R2 parity execution. The
obsolete V1 statement "calibration is the first 50 trials" is not an adequate
description because it incorrectly lets the activity carrier consume 50.

The primary endpoint is SUA `shared_t4 - shared_ts4`. Deterministically pooled
pseudo-MUA is the key secondary endpoint. Each view passes only if all frozen
gates pass: mean paired delta at least +0.03 R2, all three seed means positive,
at least 12 of 15 session means positive, the frozen hierarchical-bootstrap
lower bound positive, and absolute shared-T4 performance positive.

The claim is a correct-vs-shuffled T4 attachment/content contrast. It is not a
T4-vs-SPINT claim: no same-window shared B0/SPINT arm exists in this matrix. It
is a same-Dandiset cross-animal confirmation, not an independent laboratory or
native threshold-crossing-MUA result.

## V5R2 parity inheritance and banner-only drift

V3R2 retains the immutable V5R2 parity artifacts and their historical runtime
source map. Two historical parity-protocol Markdown files later received the
same four-line, 276-byte `SUPERSEDED` banner. Their bodies did not change:
removing exactly that known prefix in memory reproduces both V5R2 historical
SHA-256 values. V3R2 records this as a banner-only transition and pins the
current complete files; it does not edit the historical artifacts or weaken
ordinary source verification.

## Runtime and authorization boundary

Execution is CPU-only and one-threaded. CUDA, autocast, TF32, optimizer steps,
backward, target updates, normalizer fitting, cohort reselection, checkpoint
selection, and retries are forbidden. All six epoch-11 checkpoints, the
teacher, source normalizers, current source closure, exact Python/host runtime,
external-NWB root, fresh output root, and claim boundary are policy-bound.

A future execution requires a canonical V3 detached authorization envelope,
an Ed25519 signature made outside the workspace, a fresh 256-bit nonce, and a
validity interval of at most 15 minutes. The nonce is atomically claimed only
after signature, expiry, policy, source, runtime, output-root, and external-root
checks succeed. Authorization expiry limits when execution may begin; a
successfully claimed one-shot CPU run may finish later.

The scoring phase uses the parity-proven V5 chronology bridge to construct the
30-trial activity carrier while retaining the 50-trial T4 pool and post-50
query boundary. It writes immutable prediction/target arrays and per-cell
metrics for all 180 cells, seals the complete matrix, and then recomputes every
cell R2 from the sealed float32 arrays before producing the aggregate. Partial
results do not authorize a retry or a claim.

## Preparation/launch separation

The V3R2 prelaunch writer and dry-run are permitted before Native-M2 finishes.
They must report zero external-NWB/checkpoint/normalizer opens, zero model
forwards, zero R2 computations, zero nonce claims, and no authorization. A
future root reviewer may generate and sign one short-lived execution envelope
only after Native-M2 completion and a final byte-level V3R2 review.
