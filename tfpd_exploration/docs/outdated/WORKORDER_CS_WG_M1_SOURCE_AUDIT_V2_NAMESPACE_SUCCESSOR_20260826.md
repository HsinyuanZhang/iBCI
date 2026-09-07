# Work Order: CS-WG M1 Source-Audit V2 Namespace Successor

Date: 2026-08-26  
Status: additive no-data/no-CUDA successor implementation only  
Scientific role: repair the audited V1 namespace-launch failure without
changing any M1 data, model, objective, fold, calibration, or source-audit
semantics.

## 1. Immutable V1 predecessor

The V1 source-audit root is immutable and is never retried, populated,
modified, deleted, or used as a V2 output root:

```text
root:
tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1

attempt.json SHA256:
5d0cd206644d29b4ac81139d9a7cbe4aaedc404935a1029597ef1a0f3f9a1e75

launch.json SHA256:
895fb18d342a51a91e97e5a89ef13e53d662bd556fc2b0881ca1e7be5a70a99d

failure.json SHA256:
f5ec355bd26d4bb13f48117e122a2f2b4c5eb3dac2e18cc9b357dfa10554436b

failure class: SourceReaderError
failure error SHA256:
7730762e166a649a38f201a078398d00337352875f4fa2169150fac19c517e65
```

Exactly the six regular, non-symlink, mode-0444 leaves exist: each named JSON
body and its canonical basename sidecar.  There is no source authority,
audit, terminal, partial body, temporary leaf, or extra leaf.  V1 failed after
attempt and launch but before source authority, parser/NWB body access, model,
checkpoint, CUDA, or optimizer work.  The failure was caused by an incorrect
launcher import of `src.cross_session_worst_group_v1`, which preoccupied the
historical top-level `src` namespace.

The V2 issuer and executor must hold one no-follow directory FD while reading
all six predecessor leaves, exact-check body SHA, canonical sidecar, mode,
topology, and the failure's source-only/no-model/no-CUDA/no-update semantics.
That validation occurs before V2 capability issuance and again before V2 root
reservation.

## 2. Frozen inherited contract

V2 inherits the accepted repaired V1 implementation closure exactly:

```text
V1 repaired closure SHA256:
b5d4fdf5505fd53f160d56160daa642e36e8077171f654d9c5cdff6ee8fb7a28

sealed metadata-only M1 authority SHA256:
4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6
```

The inherited CPU source audit remains source-only, has outer target 20120924,
uses source sessions 20120926/20120927/20120928, does not construct a model or
initialize CUDA, and performs zero optimizer steps.  The current GPU smoke is
still non-issuable.  V2 must not issue or imply a GPU smoke capability.

No network, source reader recipe, calibration ownership law, physical source
protocol, common-stratum rule, M1 fold, objective, model, or target boundary
changes in this successor.

## 3. Namespace repair

The reviewed V2 bootstrap is the sole physical entry seam.  It must:

1. add the repository root needed for
   `tfpd_exploration.src.cross_session_worst_group_v1...`, never the
   `tfpd_exploration` directory that exposes its `src` child as a top-level
   package;
2. import the route only by its fully qualified
   `tfpd_exploration.src.cross_session_worst_group_v1` name;
3. reject an already-loaded top-level `src` unless it is demonstrably the
   closure-bound `streaming_calibration_exp/src` runtime; and
4. never delete, replace, or otherwise mutate `sys.modules`.

The historical streaming runtime remains responsible for importing its own
top-level `src` later, inside its existing deferred parser loader.  A clean
subprocess must demonstrate the old bad import fails closed and the V2
bootstrap reaches that deferred parser seam without opening an NWB or
initializing CUDA.

## 4. V2 identity and lifecycle

V2 uses a fresh root only:

```text
tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2
```

Its typed identity binds the inherited V1 audit identity, the current exact
V1 repaired closure, the sealed metadata-only manifest, the V2 explicit
closure, and the literal V1 failed predecessor expectation.  A root-only
opaque V2 capability binds the held predecessor graph digest.  The lifecycle
is:

```text
V1 held predecessor validation
-> V2 capability/freshness validation
-> V2 fresh-root reservation
-> attempt -> launch -> source_authority -> audit -> terminal
```

Any exception after attempt produces a V2 immutable failure receipt with
honest source/model/CUDA/optimizer progress.  A V2 terminal may only be
published after exact held revalidation of all V2 published pairs and a second
held reload of the V1 predecessor.  The public CLI is static and can never
issue a capability, reserve a root, open source, import Torch, or initialize
CUDA.

## 5. Additive ownership

Only these new files may be added or changed for this successor:

```text
tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_AUDIT_V2_NAMESPACE_SUCCESSOR_20260826.md
tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py
tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v2.py
tfpd_exploration/tests/test_cross_session_worst_group_m1_source_audit_v2.py
```

Do not modify V1 source lifecycle/reader/physical code, V1 result bytes,
shared M1 modules, data, checkpoints, configurations, score routes, or any
other experiment.

## 6. Mandatory no-data tests

1. V2 workorder, V1 closure, and sealed metadata literal binding;
2. all V1 predecessor body/sidecar/mode/topology/semantic tamper cases;
3. held no-follow predecessor root/parent/leaf identity substitution;
4. V1 predecessor validation before V2 capability and root reservation;
5. fresh V2 root/collision and no V1 reuse;
6. full synthetic attempt-before-source V2 audit terminal and honest failure;
7. public static CLI imports no Torch and rejects execution flags;
8. a clean bad-namespace subprocess reproduces V1's failure; and
9. a clean reviewed-bootstrap subprocess loads the deferred parser seam,
   preserves `src` for streaming, opens no NWB, and leaves CUDA uninitialized.

## 7. Stop conditions

Stop without source, CUDA, root reservation, receipt minting, or launch if
the V1 failure graph is not exact, the two package namespaces require module
replacement, V1 closure or sealed metadata drift, the V2 root exists, or the
route cannot preserve the inherited source-only/no-GPU boundary.
