# Workorder: PMC-D matched scorer V3 successor

Status: additive, fail-closed launch-contract successor; no live execution is
authorized by this document.

## Scope and scientific hold

V3 is a thin deployment successor for the failed
`posterior_marginalized_cell_d_score_v2` attempt.  It reuses the reviewed V2
physical runtime, the inherited Cell-D graph, deterministic ordinary-OLS
point-T4 inference, fixed last-bin input authority, strict-within6 and
external15 surfaces, variance-weighted per-session R2, equal-session
aggregation, target optimizer/backward/update zero, and terminal/failure
lifecycle.  It introduces no posterior inference field, normalizer, mean,
credibility gate, model, loss, scheduler, or data-selection change.

The only deployment repair is an exact launch contract, checked before root
reservation or root-reviewed capability issuance:

```text
SUBC_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C
SUBM_DATA_ROOT=/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M
```

The public V3 CLI is dry-only.  A production root reviewer must issue the
in-process V3 capability after fresh provenance, V2 predecessor, V3 closure,
and compatible device-profile validation.  No capability is minted by this
workorder, and no NWB, target/formal data, checkpoint tensor, CUDA, GPU, or
result is opened by the scaffold tests.

## Immutable V2 predecessor gate

The V2 root is read-only and must remain exactly the input-stage failure graph
below.  V3 must not reserve, write, or alias this root:

```text
tfpd_exploration/results/posterior_marginalized_cell_d_score_v2
```

The body and sidecar SHA-256 values are:

```text
preflight.json   76567e94c46dd7ea35ffc8c0ae551760ab6c49da968f68af8a69f231dd2d9c71
preflight.json.sha256
                 41a69ceba567237d6ae9fb89ca00966601153075fc6c03b9d73d06d347a6a522
authorization.json
                 2fe2396e88dbcdc49ba2da83e1d98f4010f9d413de0f6314f4be277aca664ba5
authorization.json.sha256
                 c4f3b954d825a621562314b1e8e32be51267b48135fdae7fc2312d91e298f605
attempt.json     69d67e268253f070845646e2ba1f1051891196d12dcf58ccd34babfded25fb07
attempt.json.sha256
                 0ec17e155f6312690f2d666c37c94df1960c6709a0eaa9b172234a3714425f83
failure.json     cd72f49762a4917fb940a7368acf1f27b33db95bedd338b71fa556c36af540eb
failure.json.sha256
                 e2d5730190592c300028518ee20c417b091fb5ee369b22bed950940f9210d6b8
```

All four bodies carry identity SHA
`cbc12c8285f579b40253ea1403e857eebb1df28b419016a27eda384fe7f076db`.
The failure body must remain `stage=input`, `status=SCORE_FAILED`,
`error_class=PhysicalScoreError`, and error SHA
`672dc861532297fd1601eae610b9ef833072dfbacfadb027d2247ee7b7a64d0d`.
It must have `terminal_published=false`, null input-authority and sealed-M30
parity, `target_opened=true`, and zero target optimizer/backward/update calls.
The failure's preflight/authorization/attempt body bindings must continue to
equal the canonical JSON digests of those exact bodies.  Any topology, mode,
sidecar, body, identity, stage, or binding drift is a hard V3 no-go.

## V3 identity, closure, and lifecycle

The fresh canonical output namespace is:

```text
tfpd_exploration/results/posterior_marginalized_cell_d_score_v3
```

The V3 identity carries an additive closure whose base is the reviewed V2
physical closure, whose local leaves are this workorder, the V3 package,
physical wrapper, dry-only CLI, and focused test, and whose payload contains
the exact failed-V2 graph and launch environment above.  Every preflight,
authorization, attempt, input, score, terminal, and failure receipt is
therefore identity-bound to the V3 closure and the immutable V2 predecessor.

The authorized route is ordered as follows:

1. Require the exact SUBC/SUBM environment.
2. Hold and validate all eight V2 predecessor body/sidecar leaves.
3. Fresh-load verified PMC/sealed provenance, recompute the V2 physical and
   additive V3 closures, and construct a V3 identity.
4. Require the V3 root-review token and exact compatible device profile;
   reject V1/V2 capability objects, stale closure leaves, stale provenance,
   stale predecessor graph, and profile mismatch.
5. Reserve only the fresh V3 root and publish the inherited attempt group
   before fixed input authority derivation.
6. Compose the reviewed V2 physical backend/runtime and inherited lifecycle;
   materialize each surface input once, score all six matched cells, validate
   sealed-M30 parity, then publish score and terminal or an exact failure.
7. Final-reverify provenance, predecessor graph, and V3 closure before
   terminal publication.

Any missing or wrong environment value fails before reservation, capability
issuance, or input-authority derivation.  V2 remains immutable throughout.

## Validation boundary

The focused no-data suite must cover: exact predecessor graph and no-forwards
semantics; missing/wrong environment with zero reserve calls; V3 issuer and
identity closure binding plus leaf drift rejection; V2/non-V3 capability
rejection; inherited synthetic lifecycle receipt identity/closure bindings
through terminal; and the actual sealed producer payload's CPU-only,
weights-only, strict-load proof (top-level `state_dict` and `swa_manifest`,
manifest schema exact, finite CPU state, no CUDA initialization).  It must not
open NWB/target/formal data or create the canonical V3 root.

The V3 CLI accepts `--dry-run` only.  `--execute` and capability-like public
arguments fail closed.  Live launch requires an independent root-reviewed
issuer and fresh terminal/SWA/provenance gates; this workorder is not that
issuer and does not authorize a launch.
