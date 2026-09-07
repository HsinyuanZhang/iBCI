# Consumed sub-C parity v5 — schema bridge and one-time CPU execution prelaunch

Status: `STATIC_V5_SCHEMA_BRIDGE_CPU_PARITY_EXECUTION_PACKAGE_NOT_AUTHORIZED`

V5 is append-only. It preserves all v3 and v4 sources, bundles, the consumed
v4 nonce, empty v4 output root, and the v4 incident receipt. It neither retries
the consumed v4 capability nor changes score-only code, C1 preprocessing, the
experiment matrix, external sub-M scope, or scientific endpoint claims.

## Incident binding

V5 fixes only the documented failure:

- incident: `dandi_000688_subc_scorer_adapter_parity_execution_incident_v4`;
- receipt SHA-256:
  `4c676146c5855f91591aaa13fc1d0b312ebbb51138584bd39c02c2888f47d0bc`;
- status:
  `FAILED_CLOSED_AFTER_AUTHORIZATION_BEFORE_MODEL_FORWARD_SCHEMA_TYPE_MISMATCH`;
- no checkpoint load, model forward, R2 computation, external sub-M asset, or
  external sub-M scoring occurred.

The v4 nonce remains consumed. V5 requires a different, future, detached
signature and nonce.

## Minimal schema bridge

The actual score-only datamodule owner
`list_datamodule_rewarded_trials` exposes semantically integral bin
coordinates as float-typed `start` and `stop`. C1's existing calibration
builder uses those values as slice indices and therefore requires integer
types.

V5 keeps the complete raw owner chronology as evidence. Before and after its
bridge, it serializes the original owner sequence and requires identical
SHA-256s. For every trial it creates a distinct mapping copy and:

1. rejects missing coordinates, booleans, nonnumeric values, NaN, infinity,
   fractional values, and values outside signed int64;
2. treats Python and NumPy integers directly, without conversion through float,
   preserving exact values at and around 2**53 and up to INT64_MAX;
3. treats float values only if finite and exactly integral;
4. converts only the copied `start` and `stop` fields to Python integers;
5. proves that every other copied key/value and the key set are unchanged.

The bridge does not recompute bins. It does not change trial ordering or
selection semantics, T4 pool/features, neural/behavior values, or the
owner-produced post-50 `valid_starts`. The trace names this precisely as
`*_changed_by_schema_bridge=false`; it does not falsely claim that the
existing selection and T4 owner calls are absent.

The corrected concrete parity call then reuses v3's C1 reference fixture,
shared observer, exact input gate, model loader, one forward helper, and metric
observer. Only the adapter chronology supplied to the already-existing C1
calibration builder is bridge-typed.

## V5 execution boundary

V5 fixes the same dedicated public root anchor used by v4:

`sua_exploration/configs/dandi688_subc_parity_v4_root_ed25519_public.pem`

SHA-256: `a541b5aabc7922251e797da13727b97ceb4ed48de7e95d5b1c4031ba96c602a9`.

No private key is in the workspace or accepted by the runner. The authorization
is canonical JSON plus a separate strict-base64 Ed25519 64-byte signature,
short-lived for at most 15 minutes, bound to v5 policy SHA, all v3/v4/incident
pins, exact SPINT Python/host, CPU-only one-thread policy, one new output
root, and a fresh 256-bit nonce.

The runner reads policy only from the stored 0444 v5 bundle, recomputes the
policy SHA, validates the signature and every binding, verifies freshness, then
claims the nonce with O_CREAT|O_EXCL. Only after that can it import the v5
corrected helper and call it once. V5 remains CPU-only and prohibits normalizer
fitting, optimizer/backward, external sub-M assets, and external sub-M scoring.

A future success writes raw bridge evidence plus shared input trace,
prediction/target digests, R2, observer counters, exact environment/source
hashes, and an immutable seal. Asset-open counts are explicitly
not-instrumented and never fabricated; the pre-import audit is frozen before
runtime imports.

