# CS-WG M1 Source Smoke V6: Audit-Spec Successor

## Scope

V6 is an additive, source-only 100-step smoke successor.  It exists solely
because immutable V5 failed before its source authority, model construction,
CUDA initialization, or optimizer update: the V5 provider passed a smoke
run-spec into the V3 common-stratum builder, which accepts only an audit
run-spec.  V5’s result root and all previous roots are immutable and must not
be retried, changed, or reused.

This work order permits code plus synthetic/no-CUDA tests only.  It permits no
NWB/source/target access, checkpoint loading, CUDA/GPU use, result-root probe,
reservation, capability issuance, receipt creation, or launch.

## Immutable predecessors

Before any V6 capability or root reservation, hold one no-follow directory
descriptor for each predecessor and validate no extras, exactly mode-0444
body/sidecar pairs, canonical basename sidecars, exact body SHA-256 values,
and receipt semantics.

Accepted V3 source-audit graph remains the V5-held 10-leaf graph.  The V5
failure graph is exactly six leaves under
`tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5`:

| body | SHA-256 |
| --- | --- |
| `attempt.json` | `ae6bf550dfff4179d51dbb62616a7fa97d2bfb33d1e7de5ab8c4917096afb9c4` |
| `launch.json` | `03a33580948f9c95dd8f022ea1947ed9f696f551974fe546c5edb03c94d3756f` |
| `failure.json` | `0cbd5c0551fa654bf7d9b2b45dfe334d7023bcfb142087fefd297ed85199ce11` |

The exact failure is
`SourceAuditV3Error('CS-WG V3 source selection/outer-target ordering drift')`,
whose `repr` SHA-256 is
`840a7a668324571626ba202fd63a3f9590252701be5b91707f2a0b11d2d3d9db`.
It has no source authority or terminal, `model_constructed=false`,
`cuda_initialized=false`, `optimizer_steps_completed=0`, and all target flags
false/zero.

## The one repair

The V6 provider must obtain the exact accepted V3 audit spec from the typed
identity:

```
audit_spec = identity.accepted_v3_identity.inherited_v1_identity.spec
descriptors = provider.resolve_exact_sources(audit_spec)
audit = build_common_stratum_prepared_audit(..., spec=audit_spec, ...)
prepared = rebind_v3_audit_prepared_to_smoke(..., audit_prepared=audit)
```

The `audit_spec` is exactly the historical V3 audit source-selection contract.
The re-bound object is then the unchanged V1 smoke spec consumed by the V5
derivative observer and original `TorchCSWGSmokeRunner`.  Passing the smoke
spec directly to the V3 audit builder must fail closed.  V6 must not copy the
optimizer loop, change task strata/quotas/model/data/optimizer, or alter the
V5 derivative numerical gate.

## Lifecycle and test boundary

V6 uses a fresh root
`tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6`.
Attempt publication precedes launch and source preparation.  A successful
future route will publish the exact V6 authority, V5 derivative evidence,
inherited V1 smoke/checkpoint facts, and terminal; failures must preserve the
actual progress and cannot publish a terminal.

Required no-data tests cover the direct smoke-spec error, audit-spec/rebind
success, exact held predecessor/tamper/topology/collision protection,
attempt-before-prepare ordering, V5 derivative-gate retention, a complete
temporary 100-step mock lifecycle, public dry CLI inertness, and no CUDA.

## Closure-file integrity

The V6 successor closure is also authorization material.  Each direct V6
closure leaf must be read through one held `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`
file descriptor: reject a symlink or non-regular leaf, compare the opened
regular FD with its no-follow named entry, hash bytes from that same FD, then
revalidate the named leaf and held parent/root directory identities before
accepting the digest.  The no-data suite must directly reject a symlink leaf
and a named-leaf replacement after the FD opens.  This is a closure-read
integrity repair only; it does not change V6 provider, lifecycle, model, or
science semantics.
