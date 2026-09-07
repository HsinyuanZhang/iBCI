# CS-WG M1 source-smoke V5: granular derivative successor

## 1. Purpose and immutable predecessors

`CROSS_SESSION_WORST_GROUP_SPINT_M1_V1` V5 is one additive, source-only,
100-optimizer-step smoke successor.  It does not change the M1 graph,
source-only labels, common-stratum fallback, mixed-session B32 schedule,
optimizer, learning rate, weight decay, lambda, tau, random-seed law, or
target boundary.  It exists solely because the completed V4 physical run
failed after all 100 steps at the opaque V1 aggregate predicate
`session_objective_derivatives_nonnegative`.

V5 must descriptor-hold and validate, before any V5 capability or fresh-root
reservation and again before terminal publication:

* accepted V3 audit graph under
  `tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3`:
  attempt `f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287`,
  launch `04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c`,
  source authority `0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b`,
  audit `ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9`,
  terminal `2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230`;
* immutable V4 failure graph under
  `tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4`:
  attempt `0d4179d77cd8aa66bdad70d1164682dba38b3d960fab3d524cf5f213e4601b27`,
  launch `422dec3d014bf3fa0fed716b8634cc6e8da6076f36f0f303f657a388f7b417ad`,
  source authority `40336ed6ea1f6dd2b6511eee4d22f5cf76c4af29510f8c258819bd396c78b4fe`,
  failure `1a8237db7ddf34535bebd83cc773da7efde6b5aa97517a09638b391d6eae91d3`,
  whose exact error class/repr digest is
  `SourceSmokeV4Error('CS-WG V4 inherited physical smoke evidence drift')` /
  `bfeff083be483479e40fb82078a3b8b1e5755b0f04ec0f341cc840af892cf813`.

All predecessor leaves are regular non-symlink `0444` body/sidecar pairs;
sidecars are exact `SHA256<two spaces>basename<newline>`.  V5 treats their
recorded V3 closure `d52168e567188b8ede816f4764cf829ecd920b2540c323fee14569ec7503fa1e`
and historical V4 closure `46cf82d19bb80f79cbb4df3a8ea34a4778af4cf8bc5f1f8674e4fe2ba729a676`
as immutable historical evidence.  Its own successor closure is rebuilt from
current V5-listed bytes; it must never assert that an old closure remains a
current-byte closure after the expressly authorized observer seam.

The V3 identity schema is intentionally split and V5 must preserve that exact
producer topology: V3 attempt, launch, source authority, and terminal carry
the full canonical nested identity payload, whereas V3 `audit.json` carries
only its exact `identity_sha256`.  V5 must validate the audit digest against
the full identity and must neither require nor invent an audit-side nested
identity.  Conversely, a missing or substituted nested identity in launch,
source authority, or terminal is a fail-closed predecessor drift.

## 2. Narrow implementation ownership

The only shared change is an optional, default-`None`, read-only derivative
observer on `TorchCSWGSmokeRunner` in
`tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py`.
When absent it must neither call nor read the observer payload and must retain
the historical V1 result/optimizer/model/RNG behavior.  The observer is called
immediately after the existing `torch.autograd.grad` and before zero-grad,
backward, and optimizer-step; it receives only detached scalar loss/derivative
values and no model, optimizer, graph-bearing tensor, or RNG handle.

All other V5 work is additive:

* `src/cross_session_worst_group_v1/source_smoke_v5.py`;
* `src/cross_session_worst_group_v1/source_smoke_physical_v5.py`;
* `scripts/run_cross_session_worst_group_m1_source_smoke_v5.py`;
* `tests/test_cross_session_worst_group_m1_source_smoke_v5.py`.

V5 reuses the V4 cached common-stratum provider and the exact V1 runner.  It
does not copy the optimizer loop, mutate module globals, add a second forward
or backward pass, retain step tensors, or modify V1/V2/V3/V4 receipt roots.

## 3. Fixed derivative evidence gate

For every one of the 100 completed optimizer steps, V5 records a compact
canonical scalar row and a domain-separated digest.  It retains no GPU tensor.
The session MSE values and raw FP32 autograd weights must have three values in
canonical source-session order.  The frozen gate is:

* every session MSE is finite and in `[0.0, 10.0]`;
* raw FP32 weights are finite;
* the independent reference is FP64 `softmax(session_mse / 0.01)`;
* `abs(sum(raw)-1) <= 2e-6`;
* `max(abs(raw-reference)) <= 2e-5`;
* `min(raw) >= -2e-5`.

The thresholds are fixed before a V5 run.  The synthetic calibration includes
the known FP32-cancellation case `[0,1,1]`, which may have a small negative
raw component but passes the fixed reference gate, and material cases
`[0,20,20]`, `[0,1000,1000]`, and `[0,1e6,1e6]`, which fail.  A wrong sum,
wrong reference, nonfinite value, out-of-domain loss, or malformed ordering
also fails.  This replaces only V1's opaque Boolean derivative predicate; all
other V1 smoke checks remain mandatory.

## 4. Lifecycle and receipt law

The fresh V5 root is
`tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5`.
An opaque in-process root capability is required.  Capability issue and
execution revalidate the exact V3/V4 held graphs, successor closure, selected
device/thread environment, and fresh root.  Durable attempt publication
precedes source resolution, parser, model construction, CUDA, forward,
backward, or optimizer work.

The success topology is attempt, launch, source authority, smoke, two strict
checkpoint bodies, checkpoint manifest, and terminal, with canonical 0444
pairs.  A terminal binds each prior SHA and current successor closure.  A
failure is mutually exclusive with a terminal and records honest progress plus
one typed `failure_stage` / `failed_predicate` pair from:

`main_schema_scalars`, `gradient_coverage`, `rng`, `resources`, or
`derivative_numeric_gate` (or a documented physical observer exception before
the next optimizer boundary).  Its observed summary contains only safe compact
scalar/digest/count information—never source rows, checkpoint bytes, target
data, or retained tensors.

## 5. Boundary and tests

Public CLI is stdlib-only and dry/fail-closed.  No public route imports Torch,
opens source/target data, reserves a root, issues a capability, or launches.
No data/NWB/checkpoint tensor/CUDA/GPU/result-root access or execution is
authorized for this build/test phase.

Focused no-data tests must cover held predecessor topology/tampering, V1
default observer non-interference, observer ordering and exception progress,
FP32 tolerance/reference failures, granular receipt preservation, full mocked
100-step success lifecycle, checkpoint/terminal topology, closure drift,
freshness and public dry isolation.  The test environment uses no user site,
empty CUDA visibility, and all numerical thread controls set to one.
