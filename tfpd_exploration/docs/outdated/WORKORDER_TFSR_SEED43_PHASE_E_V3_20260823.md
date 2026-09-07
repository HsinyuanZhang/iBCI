# TF-SR seed43 Phase-E matched score V3 — exact physical `ScoreSpec` repair

## Purpose and scope

This is a minimal additive deployment successor for completed TF-SR seed43.
It does not alter the seed43 training graph, the frozen seed42 score V2,
Cell-D, TF-SR, the data authorities, a metric, a roster, a control, or the
GPU1 physical profile.  It exists because the immutable seed43 score V2
attempt failed honestly immediately after attempt publication, before any
input receipt, forward, update, or terminal.

The V2 failure is an interface-identity failure, not a scientific result.
V2 passed the statically loaded seed42 V1 `PUBLIC_SPEC` to the inherited
physical backend.  Its field payload is identical to the backend's public
specification, but it is a distinct `ScoreSpec` class/object loaded under a
different static package.  The physical backend correctly rejects it with
`spec != PUBLIC_SPEC`.

V3 changes exactly this call boundary: it obtains `PUBLIC_SPEC` from the
same frozen physical seed42 scorer module that constructs the inherited
seed43 backend, and passes that exact object to `resolve_inputs`.  The
payload must still equal the static V2/V1 public specification.  No schema
relaxation, type coercion, monkeypatch, or replacement backend is permitted.

## Immutable predecessors

Before V3 authority reservation, descriptor-safe validation requires all of:

1. The completed seed42 Phase-E V2 attempt/input/score/terminal graph already
   bound by the V2 route.
2. The completed native seed43 training terminal/SWA/final-four graph.
3. The seed43 V2 authority pair:

   | leaf | SHA-256 |
   | --- | --- |
   | `official_preflight.json` | `b870bf0ea9f01bcd64441f22612e3bb9cfbb0a18eb909ac44b95bc366c6b0f07` |
   | `root_authorization.json` | `4c999af634eb0caa718cc377b5bb165c10fd8cd876ab90d7892541905491adc4` |

4. The exact failed V2 result topology, with no additional leaf:

   | leaf | SHA-256 |
   | --- | --- |
   | `attempt.json` | `ac9f4a54ce8d3a719791b401caecd4f35f05bb8124374a90d181f865989cc29f` |
   | `failure.json` | `e228395570a22904d99c678fc523fa01e14f78a4d5528735ad5e1a6652262458` |

Every predecessor body and canonical-basename sidecar is a regular `0444`
leaf read under held no-follow directory descriptors.  The failed receipt must
remain exactly `resolve_identical_seed42_v2_inputs_after_attempt`, with no
input/terminal, no opened evaluation asset, no forward, no backward, and no
optimizer update.

## Unchanged scientific and deployment contract

V3 inherits the frozen seed43 V1 physical backend by narrow composition.  It
retains the GPU1 attestation, native seed43 SWA loading, eager-versus-
accelerated-v2 parity for each TF-SR forward, fixed within6/external15
authorities, the six TF-SR cells (within/external × aligned/zero/wrong-pair),
same-input Cell-D replay evidence, last-bin variance-weighted equal-session
R², and zero target optimizer/backward/update semantics.  It does not rerun
Cell-D or use a new target authority.

The only legal data roots remain the V2 literals:

```text
/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
```

## Fresh V3 roots and authorization

```text
tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_score_authority_v3
tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_matched_score_v3
```

Authority publication requires an in-process root-only capability.  Physical
execution requires exactly both public flags plus a reloaded, in-process V3
authorization capability:

```text
--execute --i-have-seed43-phase-e-v3-authorization
```

The public zero-argument CLI is static, dry, no-data, no-Torch, no-CUDA, and
no-write.  No authority mint, input resolution, model construction, CUDA
initialization, or scoring is authorized by this work order.

## Required review gates

Focused CPU/no-data tests must prove: (1) the static and physical score specs
have equal payloads but are distinct objects/classes; (2) V3 uses the exact
physical object and an injected lifecycle reaches a post-`resolve_inputs`
stage; (3) V2 authority/result pair topology, mode, sidecar, body SHA,
semantic identity, and failure facts are bound before reservation; (4) V3
closure drift fails closed; and (5) V3 score/terminal publication is atomic
and failure-only paths are honest.  Root must independently audit frozen bytes
and mint a fresh V3 authority before any launch.
