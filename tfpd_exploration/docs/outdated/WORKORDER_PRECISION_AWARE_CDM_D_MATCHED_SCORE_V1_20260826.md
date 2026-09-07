# Work Order: Precision-Aware CDM-D Matched Score V1

Date: 2026-08-26  
Status: additive, no-data/no-CUDA candidate  
Role: non-adaptive performance screen for the accepted Precision-Aware CDM-D V2 transition rule.

## Scope

This route is a thin composition over the accepted CDM-D V8 matched scorer and
the accepted V5 independent-activity source gate.  It does not change a
model, checkpoint, ordinary OLS normalizer, fixed evaluation authority,
support selection, query chronology, metric, or target boundary.  It must not
copy the V8 evaluator or the immutable lifecycle and must not monkeypatch
globals.

## Immutable predecessors

The route descriptor-loads and validates the exact V8 four-pair terminal graph:

| body | SHA-256 |
| --- | --- |
| `attempt.json` | `557bf8071094d6512be862b1b56f0d8a949df57f5fe79356d76771d1fea2a376` |
| `input_authority.json` | `ada5427650d5e612232e74b12159b332721fd475bb218f9894ec3b30ecffdd07` |
| `score.json` | `98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e` |
| `terminal.json` | `80b2dff139a1298e1447c9313ae70548191c7406f2643a1cfb45a1867dec8096` |

The historical V8 closure is
`62c685fb0d288107ed43d7a0ef469bbb655c32e68b0d7b677aac8b371ab94939`.
It is historical receipt evidence only: this successor binds its own current
implementation closure separately.  The V8 graph must also prove the accepted
V5 independent-activity source gate and its complete 88-pair source evidence.

## Matrix and metric

Run exactly four new cells in this order:

1. M10 within-6 Precision-V2;
2. M10 external-15 Precision-V2;
3. M4 within-6 Precision-V2;
4. M4 external-15 Precision-V2.

For each new cell, descriptor-reload the corresponding immutable V8 sealed
Cell-D M10/M4 row; do not rerun the sealed comparator.  Re-materialize input
once per session through the V8 parser and exact-compare every input record to
the V8 input authority before scoring.  M30 is not run: it is reported as the
immutable V8 sealed deployment-equivalent reference.

Use the same post-first-30 causal query pool, final-bin variance-weighted
two-output R2, equal-session mean/median, session order, eval/no-grad/dropout
off behavior, and ordinary OLS normalizer as V8.  No target label is used by
state, and target backward/optimizer/update counts are exactly zero.

## Precision transition

For M4 and M10 only, construct the V2 conditional Gaussian-ridge posterior
from exact support-only duration rates and frozen support direction indices.
After the inherited independent-activity proposal gates accept a carrier
proposal, compare its `[a,c]` coefficients to the *frozen support-only initial*
carrier.  A proposal outside the Bonferroni FWER credible region is carrier
rejected with the typed `precision_credible_region` reason while independent
activity still commits.  Existing B8/design/departure gates remain first and
unchanged.  Precision evidence records posterior covariance law, family-wise
threshold, frozen-reference digest, decision summaries, and acceptance rates.

M30 is a literal deployment no-op: it has no posterior construction, proposal,
forward, or state update in this successor.

## Receipt and lifecycle

Use the shared V1 profiled immutable lifecycle: durable attempt before parser,
data, checkpoint, or CUDA access; atomic score plus terminal or honest
failure.  The successor has fresh authority and score roots.  A successor
preflight/capability binds both the V8 terminal graph and V5 source gate.
Terminal revalidation repeats both predecessor checks and verifies the
successor closure.  V8 sealed comparator rows and M30 references are immutable
receipt payloads; new Precision-V2 rows are the only physical cells.

This work order authorizes only code, dry CLI, synthetic tests, py_compile,
and closure calculation.  It does not authorize source/target/NWB access,
checkpoint tensor loading, CUDA/GPU initialization, authority/result-root
creation, or launch.
