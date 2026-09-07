# Precision-Aware CDM-D speed smoke V2 diagnostic successor

## Purpose

This is a one-session, non-governing engineering successor to the immutable
`PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_SPEED_SMOKE_V1` failure. It does
not retry, alter, populate, or otherwise reuse the V1 result root. It repeats
the same bounded M4 external eager-versus-O1/O2 comparison so that the live
numeric comparison is durable in either a terminal or a failure receipt.

## Immutable predecessors

Before any V2 authority or score-root reservation, the route descriptor-reads
the exact V1 six-leaf failed result graph:

- `attempt.json`: `68419c40f381737c3700b9893cfd7d4fa489c4a887efdaf50ea999a8a98d992b`
- `input_authority.json`: `18b58d85cea9099360f7f0f4123ff99e45a258b3ea9a6928f2cf1240e77e111c`
- `failure.json`: `dd1f6b9bd4ced42ff24a4a4ef1e43eb031987541282f1e8be638cb1b772f37a4`

Each body and its canonical basename sidecar is a regular immutable `0444`
leaf; no extra, missing, score, or terminal leaf is accepted. The failure is
exactly `budget_m4`, class `SpeedSmokeError`, with error digest
`db39bc6275e036c0fb90da7bdb0d487ed988cab9fae32fbb183bf1b2048d9e81`,
and has zero target optimizer/backward/update calls. Its accepted Precision
V2 predecessor remains separately revalidated through the frozen V1 typed
identity.

## Fixed computation

The sole physical row remains M4 external
`sub-M_ses-CO-20150615`. The eager B128 baseline must exactly reproduce the
historical accepted Precision-V2 row, including prediction SHA256, R2, input
record, model state, and full causal transition chronology. The optimized arm
remains O1 identity caching plus O2 sampled repeats with ordered genuine OOM
fallback `B1024 -> B512 -> B128`.

V2 records the baseline row, optimized row summary, full numerical comparison,
transition-sequence digest, forward counts, timing, memory, selected batch,
and fallback evidence before making any terminal decision. If a later
validation/finalization failure occurs after the comparison exists, the
failure receipt carries that same typed safe comparison payload. No source or
target data values, checkpoint bytes, or raw predictions are serialized.

## Informational numeric policy

The immutable V1 policy is disclosed exactly:

- maximum absolute prediction error `<= 1e-6`;
- absolute R2 delta `<= 1e-7`;
- speedup strictly greater than `1.0`.

V2 keeps the exact historical eager anchor and exact causal transition
chronology, but uses predeclared engineering numerical limits
`max_abs <= 2e-6` and `abs_delta_r2 <= 2e-7`. It records whether each V1
numeric predicate passed. Speedup is descriptive only: a terminal requires
finite positive timing but does **not** require speedup greater than one.
`>= 1.5x` remains a recommendation for a later engineering decision, not a
scientific or terminal threshold.

## Lifecycle and boundaries

V2 uses fresh authority and score roots and an opaque root-only capability.
The public CLI is dry-only. It lexical-checks the exact GPU0/CVD0/PCI source
environment before authority publication, capability issue, score reservation,
attempt, prepare, and finalization. The shared profiled lifecycle remains the
sole owner of attempt-before-prepare, atomic score/terminal publication, and
honest failure publication.

There is no retraining, target gradient/backward/update, normalizer refit,
M10/M30 evaluation, sealed comparator rerun, AMP, TF32, compilation, or
formal scientific verdict.

## Required no-data tests

Synthetic tests cover held V1 predecessor topology/SHA/semantic tampering,
V1-versus-V2 numeric predicate reporting, V2 tolerance rejection, diagnostic
comparison persistence in terminal and failure payloads, speedup being
non-blocking, source-root/GPU0 gate ordering, attempt-before-prepare, atomic
lifecycle success/failure, closure drift, and the inert Torch-free dry CLI.

## Deferred live gate

Only independent root review may mint authority or select an idle compatible
GPU0. A V2 terminal is an informational numerical-equivalence and throughput
observation, never a governing performance result.
