# External sub-M three-arm score-only V7 protocol

## Status

**`BLOCKED_MISSING_ZERO4_TERMINALS`**.  V7 is an append-only replacement for
the V6 control plane; it is not a result, not a formal-test opening, and not a
permission to read external sub-M NWB files.  The checked-in V7 package does
not open an external NWB or a real checkpoint, import Torch, call a model,
compute an external R², use a GPU, create a complete policy, create a
signature, or mint a run grant.

The target endpoint remains unchanged:

| axis | frozen value |
| --- | --- |
| sessions | 15 external sub-M held-out sessions |
| neural views | SUA and deterministic pseudo-MUA |
| arms | shared-T4, shared-zero4, shared-TS4 |
| seeds | 42, 43, 44 |
| total score cells | `15 × 2 × 3 × 3 = 270` |
| score window | valid 50-bin windows strictly after rewarded trial 50 |
| calibration policy | T4 uses the fixed trial-50 calibration policy; no backpropagation at deployment |

The three missing shared-zero4 epoch-011 terminal checkpoints (seeds 42, 43,
44) remain the immediate scientific blocker.  Existing T4/TS4 terminal
evidence is **not** treated as a V7 closure: before any formal score, all nine
slots must receive a fresh independent V7 closure attestation.

## Why V7 replaces V6

V6 bound much more evidence than earlier versions, but an adversarial review
correctly identified a trust-root ambiguity: a runtime caller could provide a
self-consistent root payload and its own key pins.  That means a valid-looking
policy, signature, and grant could all be self-issued.  V7 removes that
transition completely.

The production `install_trusted_roots()` API takes **no payload**.  It first
reads the source-pinned, immutable anchor
[`dandi_000688_subm_v7_pinned_trust_anchor.json`](../configs/dandi_000688_subm_v7_pinned_trust_anchor.json).
That anchor is deliberately a blocked marker in this revision.  Thus no
runtime command can turn V7 into a formal runner by supplying a different
policy root, checkpoint root, run-auth root, or Ed25519 key.  A future active
root configuration needs an explicit reviewed successor revision whose anchor
path, byte count, SHA-256, mode, and public-key fingerprints are all pinned in
source.  This is intentionally stricter than an environment variable or a
CLI option.

The isolated synthetic test factory is private, carries a `synthetic_test_only`
origin, and is rejected by every formal transition.  It exists solely to test
the cryptographic/file-integrity mechanics without contacting an external
dataset or checkpoint.

## Formal chain after a reviewed activation revision

The future transition is ordered and fail-closed:

```text
source-pinned V7 successor anchor
        │
        ├── fixed policy root + pinned policy Ed25519 public key
        ├── fixed checkpoint root + pinned independent closure public key
        └── fixed run-auth root + pinned authorization public key
                    │
                    ▼
signed complete policy (no public key or root path fields)
                    │
                    ▼
verify all 9 independently signed closures
                    │
                    ▼
live openat/fstat/SHA/bytes/mode/path validation of all 9 epoch-011 files
                    │
                    ▼
construct deterministic 270-cell contract
                    │
                    ▼
short-lived signed run authorization + one canonical nonce claim
                    │
                    ▼
CPU score/write/recompute/aggregate ledger
```

The policy has an exact schema that intentionally excludes every public-key
field, filesystem root, output root, external-data root, or nonce claim path.
The root object owns those locations.  The policy must first be an immutable
canonical JSON file with a detached Ed25519 signature.  Only after signature
verification, all nine closure verifications, and live checkpoint verification
does `build_expected_contract(VerifiedPolicy)` construct the contract.  It
does not accept a bare policy mapping.

Each closure is at a derived, slot-specific path under the fixed checkpoint
root and has a separately derived signature path.  Its payload binds
`arm/seed/epoch`, checkpoint relative path, SHA-256, bytes, mode, and a slot
binding digest.  The checkpoint file is then read through the fixed directory
file descriptor and checked again.  A closure signature alone is never enough;
a moved, mutable, wrong-mode, replaced, or wrong-byte checkpoint fails before
the contract exists.

## Run authorization and output ledger

The future run authorization is also under a fixed run-auth root and must be
Ed25519-signed by a different root-owned key.  It binds the signed-policy
digest, deterministic contract digest, root identifier, `run_id`, cell count,
short validity interval (at most 15 minutes), and a 64-hex nonce.  It cannot
supply arbitrary output or external-data paths.

The output root is derived as:

```text
fixed_output_parent / runs / signed_run_id
```

The nonce claim path is equally derived rather than caller-selected:

```text
fixed_claim_root / claims / nonce[:2] / nonce.claim.json
```

All ledger reads/writes use directory file descriptors, `openat`,
`O_NOFOLLOW`, `fstat` identity checks, `O_EXCL`, `fsync`, and immutable `0444`
files.  No path traversal, symlink substitution, duplicate artifact, or
alternate nonce spelling is accepted.  The aggregate remains a reconstruction
from exactly 270 verified cell artifacts; it cannot trust a caller-provided
aggregate or caller-provided scalar R².

## R² semantics and archive safety

The V7 internal scorer reproduces the frozen
`torchmetrics==1.5.1` CPU-float32 behavior of:

```python
R2Score(multioutput="variance_weighted")
```

including the source implementation's `torch.isclose(..., atol=1e-4)`
near-constant target and residual branches.  A constant/near-constant target
does not silently use the simpler global-SSE formula.  The ledger rejects a
non-finite result rather than turning it into a claim.  Synthetic golden tests
compare ordinary, exact-constant, non-perfect constant, near-constant, and
perfect near-constant cases against the frozen installed TorchMetrics class to
an absolute tolerance of `1e-4`.

Prediction/target artifacts remain exact C-contiguous float32 arrays of
`[query_windows, 2]`.  Before any array is materialized, V7 bounds the archive
and each member, requires exactly `prediction.npy` and `target.npy` in
canonical order, rejects encryption and unsupported compression, parses the
NPY magic/version/header with a bounded header length, requires little-endian
float32 C layout and the exact shape, and checks exact payload length.  This
prevents pickle loading, duplicate members, oversized headers, decompression
surprises, and shape substitution.

## Verified implementation evidence

The source is
[`subm_co_three_arm_score_only_v7.py`](../mc_maze/subm_co_three_arm_score_only_v7.py).
The blocked writer and runner are
[`write_dandi688_subm_co_three_arm_score_only_prelaunch_v7.py`](../scripts/write_dandi688_subm_co_three_arm_score_only_prelaunch_v7.py)
and
[`run_dandi688_subm_co_three_arm_score_only_v7.py`](../scripts/run_dandi688_subm_co_three_arm_score_only_v7.py).

The focused V7 test suite covers:

- exact retained three-zero4 blocker and hard score refusal;
- policy rejection when a runtime public key is injected;
- rejection of tampered trusted-key pins;
- rejection of caller-created self-signed root sets at formal transitions;
- positive synthetic independent-closure verification plus checkpoint mutation,
  bad-signature, and closure-symlink attacks;
- canonical nonce destination and exclusive immutable writes;
- TorchMetrics golden parity at `1e-4` including constant/near-constant cases;
- NPY-header-first NPZ acceptance and malformed/duplicate-member rejection;
- no module-level Torch import or authorization execution.

These fixtures are temporary, synthetic files only.  They neither read nor
write a real experiment checkpoint, external sub-M session, score cell, formal
authorization, complete formal policy, or grant.

## What this does and does not establish

V7 strengthens the credibility of a future held-out BP-free calibration
endpoint.  It does **not** establish an accuracy gain, does not evaluate T4,
does not compare T4 with SPINT/zero4/TS4, and does not make a performance
claim.  The only current conclusion is operational: formal external scoring
remains correctly blocked until the zero4 terminals, independent closure
receipts, and a separately reviewed active trust anchor all exist.
