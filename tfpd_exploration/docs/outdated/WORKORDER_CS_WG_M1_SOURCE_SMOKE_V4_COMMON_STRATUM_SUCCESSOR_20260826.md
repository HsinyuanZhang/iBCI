# CS-WG M1 source-smoke V4 common-stratum successor

## 1. Scope and purpose

This work order defines a new, additive, source-only 100-optimizer-step smoke
route for `CROSS_SESSION_WORST_GROUP_SPINT_M1_V1`.  It is a successor to the
accepted CPU/source-only common-stratum audit V3.  It is not an evaluation,
target-adaptation, checkpoint-selection, or hyperparameter-search route.

The route exists to exercise the already-reviewed M1 CS-WG graph on a selected
compatible GPU after a root reviewer has separately authorized execution.  A
public CLI is dry only.  This work order authorizes implementation and
synthetic/CPU tests only; it does not authorize reading an NWB, loading a
checkpoint tensor, initializing CUDA, reserving a result root, minting a
capability, or launching the smoke.

## 2. Immutable predecessor

Before capability issuance, root reservation, source resolution, and again
before terminal publication, the route must hold-open and validate exactly the
following V3 source-audit graph.  The predecessor directory contains exactly
these five regular `0444` bodies and their canonical basename sidecars, hence
ten leaves and no `failure.json`.

| body | SHA-256 |
| --- | --- |
| `attempt.json` | `f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287` |
| `launch.json` | `04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c` |
| `source_authority.json` | `0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b` |
| `audit.json` | `ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9` |
| `terminal.json` | `2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230` |

The root is
`tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3`.
Its accepted closure, identity, V1/V2 failure-chain binding, and fallback
evidence are respectively:

- `d52168e567188b8ede816f4764cf829ecd920b2540c323fee14569ec7503fa1e`
- `90892d8022be400b1b9e9046f003e86f81e9c48ae9ee892c4a93b99547c489ff`
- `47d588d2762276c49d9ff8ae1c63d9185fc5b8ebf1ede88480df5c3659328112`
- `2e763763bf55276bfef175815ec7b6cb454393a87278513f86f8d99677618089`

The loaded source authority must prove the accepted common-stratum facts:
43 eligible min2 strata; per-session retained/original rows
`54467/54476`, `49189/49228`, and `54766/54783` in fixed source-session
order; and an exact step-zero B32 `10/11/11` episode with no duplicates.
These are provenance constraints, not a license to recompute or overwrite V3.

## 3. Fixed science and physical contract

The successor reuses the frozen M1 graph and the reviewed V1 physical smoke
runner.  It must use all of the following literal conditions:

- source-only outer target `20120924`, with its three ordered held-in source
  sessions; no target, held-out, minival, formal, EvalAI, or non-training
  surface may be resolved or opened;
- the deterministic V3 common-stratum min2-pruned training pool only;
- total B32 three-session rotating quotas `10/11/11`, then the normal
  rotation; exactly one concatenated mixed-session forward per optimizer
  step, preserving every row's own calibration ownership;
- 100 optimizer steps, seed 42, M1 `W=100`, `U=64`, raw output 16, exact M10
  calibration `[10,1024,64]`, 15,007,496 materialized trainable parameters,
  final-bin raw-output MSE, and Adam `lr=1e-5`, `weight_decay=0`;
- centered smooth worst-group objective `lambda=1`, `tau=0.01`, with all
  three current source-session losses participating;
- a root-selected one-visible-device compatible profile.  The route must
  attest the selected profile, disable TF32 before model construction, and
  keep AMP and compile disabled.  It must not hard-code an ordinal.

The receipt must record the selected runtime attestation, finite objective,
model, Adam state and all initialized trainable gradients; final/best strict
checkpoint reloads; seeded/restored RNG proof; resources, throughput and peak
memory; source-only/zero-target-update facts; step-zero calibration ownership;
session quotas; and worst-group objective weights/derivative evidence.

## 4. Route-local digest-cache requirement

V3's source material has one immutable, contiguous `float32 [10,1024,64]`
calibration backing per source session.  V4 must preserve the exact
`core.SourceEpisodeRow.input_digests()` values, but must not recompute the
already-validated session calibration digest once per row.  It may cache and
return the literal session calibration digest while calculating each distinct
`x` digest normally.  The cache must:

1. be route-local and never mutate V1/V2/V3 materials or their receipts;
2. prove exact equality to the ordinary row digest on synthetic fixtures;
3. bind per-session backing digest, row count, cache topology, and semantic
   equivalence in V4 source authority evidence; and
4. leave the actual B32 concatenation unchanged: it still explicitly stacks
   32 per-row calibration tensors and may not use a batch-wide broadcast.

The V4 common-stratum fallback computed from cached rows must exact-match the
immutable V3 fallback payload and SHA before any model/CUDA work.

## 5. Lifecycle and safety

The fresh prospective root is
`tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4`.
An opaque, in-process V4 capability binds current V4 closure, selected device,
and the held V3 completed graph.  The lifecycle order is:

1. validate current code/metadata identity, held V3 graph, environment, and
   fresh V4 root;
2. reserve the V4 root and publish immutable attempt before source work;
3. publish launch, resolve only fixed source descriptors, build V4 source
   authority, run the inherited V1 smoke runner, publish smoke/checkpoints,
   and atomically terminalize; or publish one honest failure receipt;
4. immediately before terminal, reread every published V4 pair under the held
   artifact root and reread the exact V3 graph.  No live final check may treat
   the route's own already-reserved root as a collision.

Failure receipts must carry actual source/model/CUDA/optimizer progress.  No
retry, root overwrite, implicit fallback, source-descriptor glob, or global
module/RNG monkeypatch is permitted.

## 6. Required no-data tests and freeze boundary

Focused tests must cover canonical body/sidecar/topology/V3 semantic drift;
capability ordering before reserve/source; current closure drift; fresh-root
rejection; V3 revalidation before terminal; cached-vs-ordinary digest parity
and no calibration rehash in `input_digests`; cache receipt drift; exact
common-stratum and B32 ownership preservation; inherited 100-step runner seam;
complete synthetic success/failure lifecycle; source-only boundary; and a dry
public CLI that imports neither Torch nor a parser and cannot execute.

At freeze, report the exact workorder/module/physical/CLI/test hashes, the
explicit no-glob closure, isolated no-CUDA test command/output, compile/dry
result, and remaining live-only gates.  Do not claim launch authorization.
