# TFSR Phase-E matched score V2

## Purpose

This is a bounded successor to the immutable Phase-E V1 attempt.  V1 failed
at `resolve_inputs` before opening an evaluation file, constructing a model,
or making a forward/backward/optimizer call because the operator supplied
parent directories rather than the canonical sub-C and sub-M data roots.
V2 changes only that deployment binding.  It does not change the Cell-D or
TF-SR models, training lineage, evaluator, score matrix, metric, controls, or
scientific decision rule.

## Immutable V1 predecessor

V2 must descriptor-read, sidecar-check, schema-check, and recheck after its
forwards the following exact V1 failure topology before it may reserve either
a V2 authority or score root:

| Leaf | SHA-256 |
| --- | --- |
| `attempt.json` | `dc5255069e1ede930b511df5280a5e596c017c8d61ca34ca59880557c0d3be3a` |
| `failure.json` | `c4054cf120a631332f01cf019a4d51b08a0f0ff09d882c89e2ac28d5761c6404` |

Both bodies and canonical basename sidecars must be regular `0444` files.  No
other V1 score leaves may be present.  The failure must be V1's honest
`resolve_inputs` failure: within/external resolved, all data unopened, zero
forwards/backwards/optimizer calls, no input authority, score, or terminal.
The V1 authority pair is also revalidated using the frozen V1 scorer before
V2 publication or execution.

## Sole deployment correction

The V2 physical route accepts only these exact roots, supplied through the
existing `SUBC_DATA_ROOT` and `SUBM_DATA_ROOT` variables after a durable V2
attempt:

```
/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
```

Any other spelling, parent directory, symlink, omission, or caller override
fails before a V2 output root is reserved or an evaluation path is formed.

## V2 authority and execution contract

V2 publishes separate immutable target-free preflight and root-authorization
pairs under `tfsr_b3st4_ddrop_seed42_score_authority_v2`, and separate score
artifacts under `tfsr_b3st4_ddrop_seed42_matched_score_v2`.  Each V2 identity
binds the exact V1 authority payload, V1 failed-attempt/failure pair, V1
training terminal/SWA lineage, exact correct-root contract, and an explicit
V2 closure.  No V2 preflight, authorization, attempt, input authority, score,
or terminal can be valid if those predecessor facts drift.

The physical implementation is a route-local subclass with no overridden
evaluation/parser/model/metric method: it inherits the frozen V1
`PhysicalMatchedScoreBackend`.  V2 only wraps the lifecycle receipts so the
successor provenance is durable.  It uses V1's exact fixed authorities,
training terminal/SWA, rosters, last-bin variance-weighted equal-session
metric, Cell-D replay, TF-SR aligned/zero/wrong-pair matrix, no-target-update
boundary, and final held-FD revalidation.

Public CLI behavior is static/dry/no-Torch/no-data/no-write by default.
Execution needs exactly `--execute --i-have-phase-e-root-authorization`; the
route still requires a reloaded V2 root authorization and its internal
execution capability.  No CLI exposes data roots, budgets, models, or a
backend replacement.

## Required no-data gates

Tests must cover wrong-root rejection before root reservation or data factory,
descriptor-safe V1 authority/failure topology and tampering, exact physical
inheritance seams, successful injected V2 lifecycle with atomic
score+terminal publication, post-forward V1/V2 lineage revalidation, and
closure drift.  No candidate is a launch authorization until a separate root
audit accepts the frozen bytes.
