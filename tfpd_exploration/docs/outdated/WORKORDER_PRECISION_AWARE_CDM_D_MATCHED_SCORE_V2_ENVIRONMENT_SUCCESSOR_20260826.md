# Precision-Aware CDM-D Matched Score V2 — Environment Successor

## Purpose and scope

This is a narrow, additive successor to the immutable failed
`PRECISION_AWARE_CAUSAL_DUAL_MEMORY_CELL_D_MATCHED_SCORE_V1` attempt.  It
does not alter the four-cell M10/M4 Precision-V2 versus V8-sealed matrix, the
immutable V8 witness, the V5 independent-activity state machine, the sealed
Cell-D SWA, the normalizer, the parser, the target metric, or any model
parameter.  Its sole semantic addition is a durable, exact launch-environment
gate for the two already-approved evaluation roots.

## Immutable predecessors

V1 is failed historical evidence and must never be retried or overwritten.
Before a V2 capability, authority reservation, authority publication, score
reservation, attempt, or final revalidation, V2 descriptor-validates both:

| V1 location | Required JSON bodies | Exact SHA-256 |
| --- | --- | --- |
| `tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v1_authority` | `official_preflight.json` | `cc51447ff154d57d40504c860114793259ec1083bdb2889c8996badbaba20874` |
| same | `root_authorization.json` | `3d00ef126c3c358f77f7ee098623ccf929d5c7277a73990186b7ea8ec6157483` |
| `tfpd_exploration/results/precision_aware_causal_dual_memory_cell_d_matched_score_v1` | `attempt.json` | `59469ac2c250fb22f1df0c94c49d17473f8d1553c1d3ea5a81bf074d6be6c849` |
| same | `failure.json` | `435ccf2c300ec4d6180738c89a51737050b768225fcee28bc165d6ae0179e253` |

Each predecessor directory has exactly the listed JSON bodies plus canonical
basename `.sha256` sidecars, all regular, non-symlink, `0444`, and `nlink=1`.
V1 failure is exactly `stage=materialize_inputs`,
`error_class=PhysicalCDMDScoreError`,
`error_sha256=058a6e19dd1fee1b1febed44a30014b0b803f74cac6c587f3bf4238f4a1a936d`,
zero within/external opens and forwards, no target updates, and truthful
prepare-time sealed-checkpoint/CUDA initialization.  V2 binds this as an
environment failure, never as a performance observation.

## Required lexical launch environment

The only accepted source-root strings are exactly:

```text
SUBC_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
SUBM_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
```

They are compared as literal canonical absolute strings before any route root
is reserved or published.  Relative paths, aliases, swapped variables,
missing values, and symlink spellings are rejected without `resolve`, `stat`,
or an evaluation-data open.  Existing selected-device `CUDA_VISIBLE_DEVICES`
and `CUDA_DEVICE_ORDER` checks remain mandatory.  The exact environment
contract is carried in V2 identity, target-free preflight, durable authority,
attempt, score/terminal/failure identity, and final revalidation.

## Execution and scientific boundaries

V2 uses fresh roots
`precision_aware_causal_dual_memory_cell_d_matched_score_v2_authority` and
`precision_aware_causal_dual_memory_cell_d_matched_score_v2`.  It uses the
shared V1-compatible profiled lifecycle and a route-local environment-guarded
delegating backend; it does not copy a parser, forward loop, metric,
optimizer, model, or lifecycle.  The public CLI is dry-only.  Real execution
requires a durable target-free authority pair and an opaque in-process root
capability.  Attempt remains durable before `prepare` and therefore before
checkpoint/CUDA/input access.  No target gradients, optimizer, backward,
update, labels-for-state, normalizer refit, or model update is authorized.

## Acceptance tests

Synthetic/no-CUDA tests must prove exact V1 authority/failed-graph topology
and semantic binding; missing, swapped, relative, and symlink-spelling source
environment rejection; environment drift at final revalidation; no write or
reservation before the environment gate; V1 default route non-mutation; V2
delegation seam; V1-compatible four-cell M10/M4 order and V8 M30 witness; and
a complete mock authority/capability/attempt-to-terminal lifecycle.  This
work order authorizes no data, checkpoint tensor, CUDA/GPU, result-root
reservation, minting, launch, or retry during implementation.
