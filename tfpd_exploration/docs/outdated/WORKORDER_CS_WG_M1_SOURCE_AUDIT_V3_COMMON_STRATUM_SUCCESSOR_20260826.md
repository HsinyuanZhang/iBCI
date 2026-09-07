# Work Order: CS-WG M1 Source-Audit V3 Common-Stratum Successor

Date: 2026-08-26
Status: additive, source-only audit construction; no-data/no-CUDA freeze required

## Immutable predecessors

V1 is immutable at
`tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1`.
Its exact regular mode-0444 body pairs are attempt
`5d0cd206644d29b4ac81139d9a7cbe4aaedc404935a1029597ef1a0f3f9a1e75`,
launch `895fb18d342a51a91e97e5a89ef13e53d662bd556fc2b0881ca1e7be5a70a99d`,
and failure `f5ec355bd26d4bb13f48117e122a2f2b4c5eb3dac2e18cc9b357dfa10554436b`.
The V1 failure is `SourceReaderError` with error digest
`7730762e166a649a38f201a078398d00337352875f4fa2169150fac19c517e65`.

V2 is immutable at
`tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2`.
Its exact regular mode-0444 body pairs are attempt
`163e73fc5c793b009e7ec86f4af7cc1841bec0bbf5865629c7013a3df468350b`,
launch `9af2bb8c400dad927b0d3dbf711d9d24242c0e4bfb55b8163a96e0a5e4ea4fcc`,
and failure `945a7b5f843e14a6cb4008309c9df4ab510b5f60dbda4a33ba54e6dc563889cb`.
The V2 failure is `SourcePhysicalError` with error digest
`52dde6c1a3dbbe88f47a5f32e281da20c1b3aeeb32166a14cb1c7fe36108e543`.
It has no source authority, audit, or terminal.  It records source resolution
only; model, checkpoint, CUDA, optimizer, and target surfaces were untouched.

V3 validates both exact six-leaf graphs under held no-follow directory FDs,
with canonical basename sidecars, body hashes, modes, topology, semantic
links, and named identity rechecks before V3 capability issuance and again
before V3 root reservation / terminal.  V2 binds the exact V1 graph; V3 must
cross-bind that nested V1 evidence to its independently held V1 graph.

The inherited V2 closure is
`ca77c59103830fcb38e6d51af49b355485b6b916a3b4402e26d4b20e21588fa0`; its
identity is
`e9565bce5d0f2e4e683b599f5e0e0e65a089eae4d0308afd7aa2d58170a6b581`.
The sealed metadata-only M1 manifest remains
`4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6`.

## Scientific successor

This successor changes only the explicit Stage-0 allowed deterministic
fallback for unequal source-stratum support.  It does not change the pooled
source-only quantile fit, `TaskStratum` definition, 11/11/10 quota rotation,
lambda/tau, M1 graph, labels, calibration ownership, target boundary, or
network policy.

For the exact three source sessions in the inherited audit fold, fit the same
pooled source-only stratum authority, derive every original per-session
`TaskStratum` count, and retain all original labels and rows in evidence.
Let `raw_common` be strata present at least once in every source session.
Let `eligible_common_min2` be exactly those `raw_common` strata with count at
least two in every source session.  Constructibility requires at least ten
eligible strata.  The route-owned training pools contain only those eligible
strata, identically in all three sessions.  This makes an 11/11/10 B32 episode
need at most two rows per represented stratum/session and preserves
deterministic rotation without duplicate rows.

The authority and audit receipts bind, in canonical stratum order:

1. original per-session stratum count tables and sets;
2. union, each session's missing-versus-union table, and raw intersection;
3. eligible-min2 set/count, exact fallback mode, and minimum-count law;
4. retained window count/fraction and dropped noncommon/sparse counts per
   session;
5. exact pruned training-pool stratum sets, step-zero 11/11/10 episode, row
   ownership, and concat compatibility.

If fewer than ten eligible strata exist, the source audit publishes only a
typed failure after durable attempt / launch.  Its stage, reason, original
topology, candidate intersection, and eligibility evidence must be durable;
it must not construct a model, initialize CUDA, or publish a terminal.

## Namespace and lifecycle

The reviewed V3 bootstrap imports the route only as
`tfpd_exploration.src.cross_session_worst_group_v1...`.  It leaves top-level
`src` unoccupied until the historical `streaming_calibration_exp` parser owns
it, rejects a preexisting non-streaming top-level `src`, and never mutates
`sys.modules` entries.

V3 uses the fresh prospective root
`tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3`.
The public CLI is dry only.  A root-only opaque source-audit capability is the
only execution seam.  Current GPU smoke remains non-issuable; V3 creates no
GPU identity or capability.

## Ownership and boundary

Only these additive files may be created or changed:

```text
tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_AUDIT_V3_COMMON_STRATUM_SUCCESSOR_20260826.md
tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v3.py
tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v3.py
tfpd_exploration/tests/test_cross_session_worst_group_m1_source_audit_v3.py
```

No V1/V2 production file, immutable root, shared model/datamodule, source or
target NWB, checkpoint, CUDA/GPU, root reservation, receipt mint, or launch is
authorized during this implementation.

## Required no-data gates

1. exact V1/V2 held predecessor topology, body/sidecar/mode/semantic/link
   tamper rejection;
2. clean bad-namespace rejection and reviewed deferred-parser seam success
   without NWB/CUDA;
3. exact-equal pools remain row-identical;
4. unequal pools with at least ten min2-common strata prune deterministically;
5. count-one strata are excluded, and fewer than ten candidates produce typed
   pre-model/CUDA non-terminal failure evidence;
6. source-session order, target-label injection, stratum-table permutation,
   and row/pool substitution fail closed;
7. deterministic 11/11/10 episodes have no duplicates, use only eligible
   strata, and represent each chosen stratum evenly;
8. synthetic V3 attempt-before-source terminal / typed-failure transactions;
9. static CLI, compile, explicit no-glob closure, and V1/V2/V3 regression.
