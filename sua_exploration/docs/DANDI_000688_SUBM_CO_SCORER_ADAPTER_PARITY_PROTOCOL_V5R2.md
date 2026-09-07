# Consumed sub-C parity V5R2 — explicit V5-source closure

Status: `STATIC_V5R2_EXPLICIT_V5_SOURCE_CLOSURE_PACKAGE_NOT_AUTHORIZED`

V5R2 is a minimal, append-only static revision of the sealed V5 package. It
does not change, replace, reopen, or authorize V5. V3, V4, V5, their sealed
bundles, the consumed V4 nonce, and the V4 schema-type incident remain intact.
No data, NWB, checkpoint, normalizer, model, scoring asset, or external
sub-M resource is opened by V5R2 prelaunch or dry-run.

## Correction

V5's combined `source_snapshot` contained its six source hashes, but its
authority and authorization bindings did not expose the V5 six-file closure as
a named map. V5R2 supplies both forms:

1. `v5_sources`: the exact six sealed V5 source paths mapped directly to their
   SHA-256 values;
2. `source_snapshot`: the combined absolute-path snapshot containing all V3,
   V4, V5, and V5R2 source pins.

The maps are checked for exact consistency: every `v5_sources` relative path
is resolved under the repository and must occur in `source_snapshot` with the
same digest. Missing, extra, or changed entries fail closed. The V5 map is also
included explicitly in the detached V5R2 authorization bindings, alongside the
V5R2 source map, predecessor bundles, incident, runtime identity, CPU policy,
and public-key pin.

## Fixed V5 closure

The explicitly bound V5 paths are the V5 protocol, bridge helper, execution
core, runner, prelaunch writer, and test. Their exact digests are fixed in the
V5R2 writer and independently rechecked before V5R2 prelaunch or dry-run.
They are additionally represented in V5's existing sealed prelaunch snapshot.

## Future execution boundary

V5R2 preserves V5's authorization-first shape. A future real execution would
require a new, external, short-lived detached Ed25519 signature and fresh
nonce. The V5R2 runner has no caller policy/key/checkpoint/NWB/model override.
It reconstructs policy from a stored 0444 V5R2 bundle, verifies the explicit
V5 closure, claims the fresh nonce atomically, and only afterwards may import
the V5R2 helper, which delegates to the already-pinned V5 corrected bridge.

This static revision is not such authorization. It performs no model/data
operation and reports the package as not authorized.
