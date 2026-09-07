> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md`](DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md).
> Preserved as append-only audit evidence; content unchanged.

# External sub-M three-arm score-only V4 preparation

Status: `BLOCKED_MISSING_ZERO4_TERMINALS`

V4 is an append-only control-plane preparation for the formal external sub-M
three-arm endpoint.  It does not modify V2, V3, V5R2, the zero4 parity V1, any
r6d source, or any sealed artifact.  It does not authorize external data
access or scoring.

## Frozen matrix

The matrix is fixed before any score exists:

```text
15 external sub-M CO sessions
× 2 views: SUA, pseudo-MUA
× 3 seeds: 42, 43, 44
× 3 arms: shared-T4, shared-zero4, shared-TS4
= 270 cells
```

All arms use one common N=15 cohort and the same chronological query windows
strictly after rewarded trial 50.  Activity identity remains first-N30.  The
checkpoint table contains exactly nine `epoch_011.ckpt` slots.  The six
existing T4/TS4 hashes are copied from the already-sealed scope metadata,
without opening checkpoint files.  The three shared-zero4 hash slots are null,
which is the current blocker.

## Independently registered comparisons

The two comparisons remain different claims and cannot substitute for one
another:

1. `T4 - zero4`: descriptor-present versus direct standardized neutral
   coordinate;
2. `T4 - TS4`: correct descriptor attachment/content versus seed-specific
   complete-row permutation.

For each comparison, SUA and pseudo-MUA are evaluated separately.  Each view
must pass the existing mechanism gates: grand paired mean at least +0.03 R2,
all seed means positive, at least `ceil(0.75*N)` positive session cross-seed
means, hierarchical session-by-seed bootstrap lower 95% bound positive, and
positive shared-T4 absolute grand/seed means.  No cross-view rescue or
cross-comparison pooling is allowed.  A formal three-arm claim requires both
comparisons to pass in both views.

The bootstrap remains 100,000 PCG64 replicates with seed `68820260805` and the
existing session-then-seed hierarchical resampling rule.

## Adapter evidence

- T4/TS4 chronology is bound to the successful consumed-sub-C V5R2 receipt and
  the V5 start/stop-only owner bridge;
- shared-zero4 is bound to parity V1, including positive-bitwise `float32
  [N,4]` zero, channel-count-only construction, no descriptor label/rate/T4
  normalizer reads, first-N30 activity, post-50 owner valid starts, both signal
  views, and shuffle/drop invariance;
- the future zero4 runtime branch is separate from `load_unit_side_features`:
  only after authorization it may call
  `attach_standardized_zero4_to_evaluation_record` directly.

## Authorization-first boundary

A later append-only completion must bind all nine checkpoint hashes, this exact
contract and cohort, both parity bundles, the complete source snapshot, one CPU
runtime, a real dedicated Ed25519 public key, a fresh 256-bit nonce, a maximum
15-minute validity window, and exact external-NWB/output roots.

Signature verification and atomic nonce claim must finish before importing
Torch or any model/data adapter, and before opening any checkpoint,
normalizer, or NWB.  CPU-only execution forbids CUDA, normalizer fitting,
optimizer/backward, and target updates.

V4 implements and tests these control-plane primitives but intentionally has
no public key, no signature, no complete execution policy, and no complete
checkpoint table.  Consequently its runner rejects `score` while still at the
blocked prelaunch boundary, before reading even an authorization path.

## Atomic result ledger

The future execution ledger is already fixed:

- one exclusive `0444` JSON receipt per exact cell key;
- duplicate cells fail closed;
- resume accepts only semantically valid complete cells bound to the exact
  contract hash;
- partial, unknown, mutable, malformed, wrong-arm/seed/view/cohort files fail
  closed;
- the full aggregate can be published once and only after exactly 270 complete
  cells exist;
- the aggregate must contain separate `T4-zero4` and `T4-TS4` sections.

## Current receipt

The writer may seal only a blocked metadata receipt whose status is
`BLOCKED_MISSING_ZERO4_TERMINALS`.  It must report zero checkpoint/model/NWB/
normalizer/Torch/forward/R2/GPU/signature operations and explicitly state that
no executable prelaunch or external capability was sealed.  Completion of the
three zero4 runs requires another review and another append-only version; it
does not mutate V4 into an executable scorer.

