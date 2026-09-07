# Work Order: PACD V2 Successor Source Smoke

Date: 2026-08-31

Status: authorized for implementation review and, only after an independent
root audit, one bounded successor source smoke on an otherwise idle GPU0.
This work order does not authorize full training, target scoring, deletion,
or reuse of the V1 result root.

## 1. Purpose

PACD V1 reached a real Cell-D paired optimizer step and then failed closed in
its route-owned finite-state assertion.  The assertion called `numel()` on an
inactive `UninitializedParameter` belonging to a lazy branch that is not used
by the ordinary coupled Cell-D forward.  This is an implementation-checking
failure, not a PACD performance result.

V2 changes only that checking boundary.  It must skip exactly Torch
`UninitializedParameter` objects, continue to test every materialized model
parameter for finiteness, and record materialized and skipped-lazy counts per
step.  The sealed Cell-D model, initialization, optimizer, schedule, paired
loss, RNG replay, dropout mask, data route, P0/P1/P2 arms, batch size, and
eight-step exposure remain identical to V1.

## 2. Immutable V1 predecessor

Before reserving or publishing a V2 attempt, resolving source data, loading a
checkpoint tensor, importing Torch, or initializing CUDA, V2 must open the V1
directory with a held directory descriptor and `O_NOFOLLOW` and validate the
exact topology below:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_v1/smoke_seed42/
  attempt.json
  attempt.json.sha256
  failure.json
  failure.json.sha256
```

Every leaf must be a regular, non-symlink file with mode `0444`.  Each sidecar
must contain the canonical body basename and the rehashed body digest.  There
must be no terminal or extra leaf.

Exact body SHA-256 literals:

```text
attempt.json  e1d6cd813b2bc49d89121b3341271a0bf94176abe3811d2f3b6f6ffa4a28b986
failure.json  82ca6850cdd50f0b5ea5073eb89809a0c6ea426ac3f8f7d9e03fc44d1bd7e273
```

The V1 attempt must bind execution closure
`0848f4ca2ce2753757cd557f46a9cba30c4ac5352fb5dbf77fe357c7fd74bdb6`
and work-order SHA
`843240ea69b4770f0a8cb4bd25618d4edf94f63488f263f2e74b19c091937dc8`.
The V1 failure must be `CELL_FAILED`, bind the exact attempt SHA above, report
`terminal_published=false`, `target_access=false`, source roster 27, and the
`ValueError` traceback at the historical V1 `core.py` finite-parameter loop.

V2 must repeat the held-descriptor predecessor validation immediately before
its terminal or failure receipt.  It must never edit, chmod, rename, delete,
or add a leaf to the V1 directory.

## 3. V2 result identity and lifecycle

The only V2 canonical result root is:

```text
/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42
```

The root must be absent at root preflight.  V2 must reject every alternate
path and every symlinked parent or root.  It publishes an immutable attempt
before Torch/source/checkpoint/CUDA access and then publishes exactly one
immutable terminal on success or one immutable failure on error.  No retry or
overwrite is allowed.

The V2 attempt and terminal/failure must bind:

- the exact V1 predecessor body digests and topology;
- the current successor execution closure;
- the repaired PACD core SHA and focused-test review SHA;
- the unchanged sealed Cell-D terminal, SWA, and canonical initial-state
  digests;
- the exact physical/logical GPU identity and source-only/no-target facts.

## 4. Smoke and resource contract

The scientific and resource contract is inherited exactly from
`WORKORDER_PACD_V1_20260831.md`: P0 M30/M30, P1 M30/M4, P2 M30/M10; eight
optimizer steps per arm; batch size 32; one optimizer update per paired step;
identical realized whole-unit dropout probability and mask; `num_workers=0`;
one numerical CPU thread; physical GPU0 only, UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`.

GPU1 and its active experiment remain entirely outside scope.  A successor
launch fails closed if GPU0 has a compute process or if the V2 root is not
fresh.  Failure after attempt is terminal for this work order and must not be
automatically retried.

## 5. Role separation and launch gate

- Terra owns only V2 additive code/tests and backward-compatible seams needed
  to reuse V1's reviewed executor.  Terra may not launch or touch canonical
  V1/V2 roots, data, checkpoints, or GPUs.
- Root owns this work order, independent source/test/closure audit, live root
  and resource preflight, and the sole launch decision.
- Luna Max performs read-only monitoring only after launch and may not edit,
  signal, retry, restart, score, or deserialize checkpoint tensors.

No V2 launch is permitted until root independently reproduces the real
Cell-D lazy-parameter regression, validates the V1 failure graph through held
descriptors, confirms the V2 root is absent, and rechecks GPU isolation.

