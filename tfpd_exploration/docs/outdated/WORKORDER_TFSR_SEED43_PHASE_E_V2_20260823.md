# Seed43 Phase-E score V2 replication addendum

## Scope

This is an additive seed43 scoring successor.  It does not modify seed42
Phase-E V2, seed43 training, the frozen seed43 V1 scorer, Cell-D, TF-SR, or
any shared evaluator.  Its purpose is to prepare a post-terminal seed43
replication while seed43 training remains live.  It is not authorized to mint
an authority, open an evaluation asset, construct a model, initialize CUDA,
or score until a later root review.

## Required upstream: completed seed42 Phase-E V2

The V1 seed43 addendum is superseded as code evidence only because it requires
the failed seed42 Phase-E V1 terminal.  It is not a result prerequisite.
V2 must instead descriptor-read under one held no-follow directory FD the
exact successful seed42 Phase-E V2 graph:

| Leaf | SHA-256 |
| --- | --- |
| `attempt.json` | `d1e97ba3c3d0dae077b4482b244d282cfe57722fdbccaeaf9816b8c56937ea03` |
| `input_authority.json` | `8d00e184c3ccd0cfde152a567c779e86f2724198388ff8751878e342a47b963d` |
| `score.json` | `8551ddf2fe437c9bd37e4767991bec5f8d9328778ad4c883afcbac3fd185a9dd` |
| `terminal.json` | `3007383ef0e4d2b8cc78aea262729fca876261a8bf25c300212e6cc573a2b579` |

Each body and canonical-basename sidecar must be a regular `0444` leaf.  No
failure, temporary, or extra leaf is permitted.  V2 validates the outer V2
attempt/input/score/terminal graph using seed42 V2 validators, then extracts
and independently validates its nested V1 input-authority and V1 score using
the frozen V1 validators.  The upstream verdict is exactly `STOP`; that fact
is reported but does not prevent a non-governing replication.

The current seed42 V2 closure must descriptor-rebuild identically to the
terminal's launch/final closure before V2 preflight and after seed43 forwards.

## Seed43 prerequisite

Seed43 must have a descriptor-valid completed native training terminal, final
four checkpoint/SWA graph, and strict accelerated-build disclosure.  Until
then this route is hard NO-GO.  No incomplete epoch/checkpoint may be used as
a model substitute or authority-mint input.

## Unchanged scientific semantics

The route composes the frozen seed43 V1 physical backend and lifecycle helpers
instead of copying their parser/model/metric logic.  It retains the native
seed43 SWA loader, eager-versus-accelerated-v2 parity on every TF-SR forward,
six TF-SR cells (within/external × aligned/zero/wrong-pair), upstream Cell-D
replay, strict rosters, last-bin variance-weighted R², equal-session
aggregation, and zero target optimizer/backward/update semantics.  It never
reruns Cell-D.

The only lineage change is that the upstream Cell-D/input/score evidence is
the nested evidence in the completed seed42 V2 terminal graph, not a failed
V1 score root.  The old seed43 V1 source/CLI/test remain closure-bound as
superseded code evidence only; no V1 score result is read or required.

## V2 authority and deployment

Fresh roots are required:

```text
tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_score_authority_v2
tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_matched_score_v2
```

V2 binds the complete seed42 V2 graph, nested V1 evidence digests, both current
import closures, seed43 terminal/SWA evidence, exact root names, and the only
legal target roots:

```text
/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
```

Any other string fails before a V2 score-root reservation or target factory.
Public CLI is static/dry by default.  Execution requires exactly
`--execute --i-have-seed43-phase-e-v2-authorization` plus a reloaded,
in-process V2 authorization capability.

## Required no-data gates

Focused tests cover the four-pair V2 graph, nested V1 payload tampering,
current seed42 V2 closure drift, canonical-root ordering, incomplete seed43
training rejection before output reservation, the narrow inherited physical
seam, and a complete injected success lifecycle.  A separate root audit must
approve frozen bytes and a completed seed43 terminal before authority mint or
execution.
