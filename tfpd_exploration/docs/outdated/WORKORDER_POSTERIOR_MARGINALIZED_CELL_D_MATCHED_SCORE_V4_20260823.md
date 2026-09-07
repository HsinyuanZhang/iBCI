# Workorder: PMC-D matched scorer V4 — B128 paired replay

Status: additive implementation only.  This workorder does not authorize a
data open, CUDA initialization, result-root reservation, or score launch.

## Purpose

V3 stopped at the historical sealed Cell-D M30 exact-R2 replay gate.  The
fixed historical scorer used evaluation batches of 128, whereas V3 used 32.
All fixed session assets, valid-window counts, neural/calibration/target/mask
digests, and normalized M30 point-carrier digests matched.  V4 repairs the
evaluation geometry rather than widening a numeric tolerance.

## Frozen scientific surface

V4 is a scoring successor only.  It binds the completed PMC-D final-four SWA
and the sealed Cell-D SWA, uses deterministic ordinary OLS point-T4 inference,
and opens only the fixed strict-within6 and external15 evaluation assets after
a durable attempt.  It evaluates exactly these eight cells:

| Surface | Budgets | Systems |
| --- | --- | --- |
| within6, external15 | M30, M4 | sealed Cell-D, completed PMC-D |

No M10, zero/wrong control, target update, target normalizer refit, posterior
inference carrier, checkpoint selection, or training is allowed.  The query
is bin 49 of every 50-bin window and the metric is per-session
variance-weighted R2 followed by equal-session summaries.

## V3 predecessor and input authority

The following immutable V3 failure graph is a predecessor, never an output to
overwrite:

```text
tfpd_exploration/results/posterior_marginalized_cell_d_score_v3
```

V4 validates its exact preflight, authorization, attempt, input-authority,
and failure body/sidecar pairs before any V4 authority/root reservation.  It
also rematerializes all 21 approved inputs exactly once and requires its
per-session records to equal V3's input-authority records byte-for-field.

## B128 historical bridge and paired screen

The physical forward loop is fixed to `EVAL_BATCH_SIZE = 128`, matching
`scripts/run_a2_matched_rescore.py`.  The sealed historical bridge uses the
historical `spintshape.build_spintshape_model(seed=42)` builder and the sealed
SWA.  PMC uses the producer-native reviewed Cell-D graph and completed PMC
SWA.  Both run on the same selected device, input materialization, order, and
metric implementation.

V4 never widens historical R2 equality.  It records each surface's bridge
result and, on a mismatch, the first mismatch `{session, field, expected,
observed, prediction_sha256}`.  A bridge mismatch is contextual: it changes
the historical absolute reference label, but does not suppress the direct
same-evaluator PMC-minus-sealed paired screen.  The score receipt must state
whether the historical absolute table was exactly reproduced.

## Lifecycle and deployment boundary

Fresh prospective roots are:

```text
tfpd_exploration/results/posterior_marginalized_cell_d_score_authority_v4
tfpd_exploration/results/posterior_marginalized_cell_d_score_v4
```

V4 publishes `preflight`, `authorization`, and `attempt` before target
asset derivation; then `input_authority`, `historical_bridge`, and an atomic
`score` plus `terminal`, or an honest failure terminal.  Public CLI is
dry-only.  Live execution requires two public flags plus a distinct opaque
in-process root-reviewed capability after current provenance, V3 predecessor,
V3 input authority, V4 closure, chosen compatible device, and prospective-root
freshness are independently revalidated.

The exact V3 `SUBC_DATA_ROOT`/`SUBM_DATA_ROOT` launch environment is checked
before V4 authority reservation, authority-pair publication, and capability
issuance, as well as at execution.  Capability issuance descriptor-reloads
the durable official authority pair itself; a caller-supplied typed authority
is only a comparison value and cannot replace that reload.  Issuance also
rebuilds the current V2→V3→V4 closure chain and verifies the held V3 input
authority SHA before minting.  The physical forward path rejects any runtime
whose batch literal differs from 128 before it accesses the runtime, data, or
model.

## Required no-data tests

- dry CLI does not import Torch and public execution fails closed;
- V3 graph/input sidecar, schema, topology, and digest tampering fail before
  reservation;
- V4 matrix contains only M30/M4 and exactly eight cells;
- V3 input records must match exactly;
- B128 is literal and a B32 subclass/forward seam is rejected by the contract;
- stale/forged durable authority, V2→V3→V4 closure drift, and a wrong launch
  environment fail before any authority output/root reservation;
- historical bridge records rather than masks an R2 mismatch;
- paired delta/mean/median/positive summaries are deterministic;
- mock success publishes terminal atomically; failure has no terminal;
- closure drift and fresh-root collision fail closed.
