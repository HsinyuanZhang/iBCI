> **SUPERSEDED — historical evidence only, not current authority.**
> Current successor: [`DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md`](DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V5.md).
> Preserved as append-only audit evidence; content unchanged.

# Consumed sub-C C1/sub-M data-adapter parity — v4 execution package

Status: `STATIC_V4_CPU_PARITY_EXECUTION_PACKAGE_NOT_AUTHORIZED`

V4 is append-only. It does not modify score-only v1/v2 or parity v1/v2/v3.
It adds the future *execution capability* that a root reviewer may sign later.
This delivery does not create an authorization, signature, claim, output root,
private key, checkpoint/NWB/NPZ read, model import, model forward, or external
sub-M score.

## Fixed authority

The v4 prelaunch writer fixes and verifies all five v3 source files and the
three immutable v3 bundle artifacts. It fails closed if any pinned byte, status,
or mode drifts.

| v3 source | SHA-256 |
|---|---|
| protocol | `f8c3da98d38c3d061f2d89d5514cf1d5453424b0a458d5440a4a64dd3c7bf802` |
| helper | `c7951beb13618b849c43a2d296a3b1550f0a644efb68e872567fd14368759a9b` |
| runner | `e70da4456c3c78ebbf0c02c34533ea4b5b2ed0313a22aee5c9451948d6c9986f` |
| prelaunch writer | `fceeeaec8e72346f1dd41e849d74a00831f91df5c02d545b9d3fa1608a6a6f8d` |
| tests | `2f19994b1104307b6cf3b7fe491f28328ff22d3e75f594c0332f7b2930fe87c5` |

The required v3 immutable bundle is draft
`96200f471c7a38277508071ab9c483e623defbc9038946468408cd495a4e6157`,
receipt `c52796b13f9e76f3dd98d62fa882158c6ae3ad4237f8896753741c259b383d8b`,
and seal `e0088a63b97717b9d984409485da20fdaca8e378a01750733f4bb60b02191cb1`.
Each must remain mode `0444`.

Fixture pins remain exactly those frozen by v3: shared-T4 seed-44 terminal
checkpoint, C1 teacher checkpoint, already-consumed
`sub-C_ses-CO-20151103` NWB, SUA behavior/T4 normalizer pins, C1
`first_n30` activity, T4 pool/query boundary 50, and post-50 query windows.
V4 does not open the pinned data while constructing or checking prelaunch.

## Runtime identity and CPU boundary

The prelaunch records one exact SPINT interpreter file SHA/bytes/path, CPython
version, and hostname. The execution authorization must bind the same values.
It further requires:

- CPU device only; CUDA is forbidden.
- `CUDA_VISIBLE_DEVICES=""` before importing v3/Torch owners.
- one Torch intra-op and inter-op thread; the runner sets
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` immediately after
  a successful authorization and before runtime imports.
- no optimizer, backward, normalizer fitting, external sub-M score, or retry.

## Detached authorization

The production trust anchor is the dedicated checked-in root **public** key
`sua_exploration/configs/dandi688_subc_parity_v4_root_ed25519_public.pem`,
pinned to SHA-256
`a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9`
in the v4 prelaunch. No private key is read, generated, written, or accepted
from the workspace.

A future authorization consists of canonical JSON bytes in an authorization
file and a separate base64-encoded 64-byte Ed25519 `.sig` file over those
exact bytes. Its exact authorization body must bind:

- v4 draft/receipt/seal hashes, all v3 source/bundle hashes, all fixture pins,
  the root public-key pin, and the exact SPINT host/Python/CPU policy;
- permitted action `concrete_parity_after_future_authorization` only;
- one absolute, new output directory below the dedicated v4 output parent;
- one 256-bit lowercase-hex nonce, authorization identifier, and short
  timezone-aware issue/expiry window (at most 15 minutes);
- `external_subm_scoring_permitted=false`.

The verifier checks signature, canonical JSON, expiry, all bindings, source
hashes, runtime identity, CPU preconditions, and output-root freshness **before
importing** the v3 helper or any Torch, PyNWB, NPZ, checkpoint, or data owner.

Only then does it atomically create the one immutable nonce claim using
`O_CREAT|O_EXCL`. A claim collision is a replay and fails before data access.
An already-existing output root fails before a nonce claim is consumed.

## Future successful execution

After authorization and the atomic claim, the runner imports only
`concrete_parity_after_future_authorization` from v3 and invokes it once. V3
therefore independently calls both real loader chains and performs its exact
input gate before the shared model forward.

A successful v4 run writes, with exclusive create, flush, fsync, SHA-256
indexing, and mode `0444`:

1. `input_trace.json`;
2. `environment.json` (host, Python, CPU policy, complete source hashes);
3. `parity_execution_receipt.json` (input result, prediction/target hashes,
   both R2 values, observer counters, and signed-authorization/claim hashes);
4. `seal.json`.

This compatibility result never opens an external sub-M asset and never grants
external sub-M scoring. A subsequent, separately reviewed action would still
be required for any score-only release.
