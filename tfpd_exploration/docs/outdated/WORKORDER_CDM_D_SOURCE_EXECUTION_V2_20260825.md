# Work Order: CDM-D Source Execution V2

Date: 2026-08-25  
Status: authorized for additive implementation and no-data/no-CUDA tests only  
Scientific role: byte-domain repair of the CDM-D source smoke and source gate

## 1. Reason for V2

Source Execution V1 failed closed during `prepare`, before any source body was
opened by the execution route, before checkpoint load, before CUDA, and before
any model forward.  The failure was not a T4 value drift.  V1 compared two
different SHA-256 domains:

- the sealed theta authority stores `SHA256(contiguous float32 raw-T4 bytes)`;
- V1 converted raw T4 to float64 and called
  `posterior_carrier_v1.core.tensor_digest`, which also prefixes dtype and
  shape.

The authorized read-only replay of `sub-C_ses-CO-20131003` proved:

```text
shape: [71, 4]
all raw values finite: true
sealed raw-T4 SHA256:
d85783bc3636799bf75adb0dffa2023fa4bdaed5738a249e1f10a52a099265e1
live contiguous-float32-byte SHA256:
d85783bc3636799bf75adb0dffa2023fa4bdaed5738a249e1f10a52a099265e1
live theta versus sealed theta: bitwise equal
live valid mask versus sealed valid mask: exactly equal
V1 float64 tensor_digest versus sealed SHA: unequal
```

V2 changes only that authority-checking law.  It does not change T4 values,
the theta/valid rule, support selection, normalizers, Cell D, B3S, K=4 groups,
carrier estimation, causal memory, B8, thresholds, scoring, or GPU policy.

## 2. Immutable V1 predecessor

V1 remains immutable and failed.  Before V2 reserves either canonical root,
hold one `O_NOFOLLOW` directory descriptor and validate the exact six-leaf,
mode-0444 topology with canonical basename sidecars:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v1

attempt.json SHA256:
13825a85ec3937653a27245b55db7dd1d1df024822e3b33dafe71633aa31b29f

launch.json SHA256:
b76b1d721e460297b45c7986615fd01aef8c429f20946891a7120c2120e4f963

failure.json SHA256:
a283cca52947fe54d681ae4444bd7885515a62bdf20adb2350362629b14f16ea
```

The failure must say `status=FAILED`, `stage=prepare`,
`source_resolved=true`, `source_opened=false`, `checkpoint_opened=false`,
`cuda_initialized=false`, zero forward/backward/update/optimizer calls, no
target/within/external/formal access, no terminal, and exact V1 closure:

```text
77f9495780605d05be177500ff0ef61b3bd35a427ddd12af4fbab97b3486428b
```

Missing, extra, replaced, symlinked, writable, sidecar-mismatched, or
semantically altered predecessor leaves are fatal before V2 capability issue
or root reservation.

## 3. Additive ownership

Create only:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v2.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v2.py
```

Do not edit V1, Stage-0, Source-Audit, shared models, parsers, data,
checkpoints, existing results, or other routes.  Reuse V1 by narrow
composition/subclassing.  Do not copy its full lifecycle or physical route.
The worker is not alone in the repository; preserve unrelated changes.

## 4. The only semantic repair

Define one route-local helper with the theta authority's already sealed law:

```python
SHA256(np.ascontiguousarray(raw_t4, dtype=np.float32).tobytes())
```

Use it only when comparing live ordinary M30 raw T4 to the theta authority's
`raw_t4_sha256`.  The downstream numeric tensor remains the same ordinary raw
T4 produced by the closure-bound source adapter.  Do not round or replace the
numeric tensor after validation.

Still require all of the V1 topology checks and additionally prove:

1. live raw T4 has exact `[n_units, 4]` topology and finite values;
2. recomputed theta `atan2(c, a)` is exactly equal to the sealed float64 theta;
3. recomputed validity under the sealed `raw_m > MODULATION_EPS` rule is
   exactly equal to the sealed bool mask;
4. known invalid-unit counts and canonical unit order remain exact.

No alternative hash domain, numeric tolerance, authority regeneration, or
fallback is permitted.

## 5. V2 lifecycle and roots

Use new immutable roots:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v2
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v2
```

V2 identity, attempt, launch, source authority, failure, smoke/gate evidence,
and terminal must bind:

- this work order and the explicit V2 closure;
- the exact immutable V1 predecessor graph above;
- the accepted V1 implementation closure and all inherited dependencies;
- the exact `theta_float32_contiguous_raw_bytes_v1` hash-law label;
- launch closure equal to final closure.

Preserve V1's attempt-before-resolution ordering, source-only boundary,
one-session smoke, strict-27 gate, M30-to-M10-to-M4 fail-fast order, unchanged
B8 thresholds, breadth threshold 14/27, device profiles, resource evidence,
transactional publication, and failure semantics.

The public CLI remains dry and fail-closed.  No public flag combination may
mint the in-process root capability.

## 6. Required no-data tests

At minimum, test:

1. synthetic raw values whose contiguous float32 bytes match authority pass;
2. the numerically equal float64 `tensor_digest` does not equal that authority
   and is never used as a fallback;
3. one-float32-ULP raw mutation fails;
4. theta mutation, valid-mask mutation, invalid-count drift, nonfinite raw,
   shape drift, and unit-order drift fail;
5. exact V1 predecessor success plus body, sidecar, mode, symlink, topology,
   semantic, and closure tampering failures;
6. V1 roots are never modified and V2 roots are fresh/spec-scoped;
7. inherited smoke/gate schemas and fail-fast decisions are unchanged;
8. authorization rechecks predecessor, current V2 closure, and root freshness
   before backend construction or root reservation;
9. static CLI imports no Torch and performs no data/CUDA/write/launch action.

Run the V1 focused suite together with V2 tests under no-user-site and
`CUDA_VISIBLE_DEVICES=''`.  Report exact file SHA-256 values, explicit closure,
test commands, test counts, dry output, and root freshness.

## 7. Authorization boundary

This work order authorizes only additive implementation and no-data/no-CUDA
verification.  It does not authorize source NWB access, checkpoint tensor
load, CUDA initialization, a GPU smoke, the strict-27 source gate, scoring, or
any result/receipt creation.  Root will independently audit the frozen V2
candidate and issue a separate launch decision.
