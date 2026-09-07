> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md`](DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md).
> Preserved as append-only audit evidence; content unchanged.

# DANDI 000688 sub-M CO three-arm score-only protocol V6

Status: **BLOCKED_MISSING_ZERO4_TERMINALS**. V6 is an append-only control-plane
repair. V4 and V5 artifacts remain immutable predecessors. This package is not
an authorization, verified grant, executable prelaunch, or external-scoring
capability.

V6 accepts no directly constructed grant. The capability class can only be
minted after a real Ed25519 detached-signature verification, an aware UTC
validity interval no longer than 15 minutes, exact output/external-root checks,
all frozen contract/source/parity/runtime/closure bindings, and an atomic
`O_EXCL` nonce claim sealed `0444`. Its canonical
`verified_grant_sha256` is carried through every prediction artifact, cell,
resume operation, and aggregate.

The expected contract is rebuilt as a complete object from the frozen V5
predecessor cohort/query map, fixed V6 protocol constants, and the reviewed
nine checkpoint slots. Validation uses exact object equality, covering query
totals, support/activity budgets, CPU policy, claim separation, gates, and
bootstrap policy; a caller cannot authorize a mutation by rehashing it.

A future zero4 slot is not accepted from metadata alone. Its real closure and
authority receipt must be regular, no-symlink, canonical JSON, `0444`, and must
match their path/hash/bytes pins, exact schema, arm/seed/epoch, checkpoint
path/hash/bytes/mode, and canonical complete-slot binding. The Ed25519
authorization also binds the live verified closure-bundle digest.

Prediction/target artifacts are safe NPZ files with exactly two C-contiguous
finite float32 arrays named `prediction` and `target`, both shaped
`[frozen_query_rows, 2]`. Object arrays, extra ZIP members, wrong shapes,
wrong rows, NaN/Inf, mutable files, hash/size drift, and symlinks fail closed.
R² is never supplied by a caller: the ledger computes variance-weighted R²
from the verified arrays and rejects undefined, nonfinite, or greater-than-one
results. Resume reopens and rehashes every NPZ and recomputes every R².

An existing aggregate is legal only after all 270 cells revalidate. Resume
then rebuilds all two-comparison/two-view statistics and the fixed 100,000-draw
PCG64 hierarchical bootstrap and requires canonical exact equality with the
stored aggregate. All security-sensitive file reads use `O_NOFOLLOW`, `fstat`
before/after, inode/device checks, fixed bounds, and path-layer symlink checks.

The present blocked package lacks all three future zero4 closures, a public
key, signed authorization, execution roots, and a verified closure-bundle
digest. Its runner therefore refuses score mode before any checkpoint, NWB,
Torch, GPU, model, or external R² access.
