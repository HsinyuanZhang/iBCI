> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md`](DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md).
> Preserved as append-only audit evidence; content unchanged.

# DANDI 000688 sub-M CO three-arm score-only protocol V5

Status: **BLOCKED_MISSING_ZERO4_TERMINALS**. This is an append-only repair of
the V4 control plane. It does not amend or replace any V4 artifact and it is
not an external-scoring authorization.

## What V5 fixes

V5 makes an independently verified `AuthorizationGrant` the authority for the
contract hash. A contract cannot authorize a modified cohort by recomputing
its own declared hash. The same grant (authorization hash, nonce, absolute
output root, contract/cohort/query-map/checkpoint-slot digests) is required by
cell publication, resume, and aggregate publication.

Every cell has an exact schema and binds one exact session/asset/view/seed/arm,
a finite R2 scalar, the frozen per-asset query count, and one fixed-path
prediction/target artifact. Cell JSON and binary artifacts are exclusive,
regular, no-symlink, hash/byte-size checked, and mode `0444`. Resume re-reads
canonical JSON and re-hashes every referenced artifact; duplicate, partial,
unknown, mutable, malformed, NaN, wrong-query, and symlink states fail closed.

The aggregate API accepts no statistics. It rebuilds both comparisons
(`T4-zero4`, `T4-TS4`) separately for SUA and pseudo-MUA from exactly 270
verified cells. It applies the frozen grand-mean, three-seed, 12/15-session,
T4-absolute, and lower-bootstrap-bound gates. The interval is exactly 100,000
hierarchical session/seed resamples from NumPy `PCG64(68820260805)`, in the
declared comparison/view draw order, using linear 2.5%/97.5% quantiles.

Each checkpoint slot binds arm, seed, epoch, path, SHA-256, bytes, required
execution mode, and closure evidence. Existing T4/TS4 terminal evidence is
pinned without opening a checkpoint. A shared-zero4 digest alone is invalid:
each of seeds 42/43/44 must arrive through a future `0444`
`sealed_terminal_checkpoint_closure_v5` whose semantic binding covers the
complete slot. The completed nine-slot digest must then be independently
included in the external grant.

## Current prohibition

Until those three closures and a separately verified grant exist, the runner
supports dry-run inspection only. It must not open sub-M NWB data, a
checkpoint, or a normalizer; import Torch; use a GPU; run a model; calculate
R2; create a signature; or create an external capability. Synthetic fixtures
may exercise only the ledger and deterministic aggregate implementation.
