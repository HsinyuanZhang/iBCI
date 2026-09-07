# Work Order: CDM-D Source Execution V4 Loader Successor

Date: 2026-08-25  
Status: **BUILD AND TEST ONLY; NO DATA, CUDA, RESULT ROOT, OR LAUNCH UNTIL ROOT REVIEW**

## 1. Purpose

V3 corrected the CDM-D state machine: every valid completed-trial B3S row can
advance the unlabeled activity memory independently of the pseudo-label-gated
carrier. Its only authorized smoke failed before source materialization because
the inherited physical runtime loader re-entered V1's frozen implementation
closure. V1 correctly rejected the new V3 `core.py` bytes.

V4 is a narrow loader successor. It does not change the V3 scientific system,
the Cell-D network, data selection, carrier estimator, trust gates, thresholds,
normalizers, device policy, or causal update order. It only supplies the
physical provider and executor with the current explicit successor closure
instead of forcing them through the historical V1 closure validator.

## 2. Immutable V3 failed predecessor

Before V4 capability issue or root reservation, hold one `O_NOFOLLOW`
descriptor for this exact directory:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v3
```

Require exactly six regular mode-0444 leaves, canonical basename sidecars,
and no extras:

```text
attempt.json
attempt.json.sha256
launch.json
launch.json.sha256
failure.json
failure.json.sha256
```

Exact body SHA-256 values:

```text
attempt.json  f77d2f552d52d03e756422f5c1526babbefe9afb1463e3a5ea0b79db73edef3f
launch.json   6b6e170d6d845734f18fcd8eea9d1a95dac2a851e9b7ab0dae4404fbd9c9c47a
failure.json  667808ded590acbb79f143043638e97e64b20f2aeacc7f719d1a1c1858907ccc
```

The three bodies must bind one identity SHA:

```text
1d42783bfb67d6360ea3fcc782a980063ba75f9ceaff414327845a22ae1cb331
```

and the exact V3 implementation closure:

```text
080c8bdd274893da0da86cd6b881ab722286252fd88fdf01578c73e9ba0f4584
```

The failure must say `status=FAILED`, `stage=prepare`,
`source_resolved=true`, `source_opened=false`, `checkpoint_opened=false`,
`cuda_initialized=false`, zero model forward/backward/optimizer/update calls,
all within/external/formal/target access false, no source authority, and no
terminal. Its error class is `SourceExecutionError` and its exact error digest
is:

```text
8e44da12b00a343cefe1c2af68f35b42847ba99bedebf612090d74b2745338f7
```

Missing, extra, writable, symlinked, replaced, sidecar-mismatched, or
semantically altered predecessor leaves are fatal before any V4 write or data
resolution. V3 is not retried and its result root is never modified.

## 3. Frozen scientific contract

V4 must preserve V3 exactly:

- sealed Cell-D model, SWA, parameter count, graph, and FP32 inference;
- ordinary sealed source normalizers;
- M4 D-optimal, M10 chronological, and M30 chronological support;
- fixed-ridge-by-trial carrier initialization with lambda 0.1;
- budget-specific K=4 complementary groups and all existing B8 thresholds;
- B3S FIFO capacities M4=26, M10=20, M30=0;
- valid B3S activity transition independent of carrier acceptance;
- current-trial predictions use the pre-transition state;
- M30 source-audit positions 30--59 remain offline/no-deployment-commit;
- source-only, target-zero-gradient, and fail-fast boundaries;
- V3 M10 smoke: session `sub-C_ses-CO-20131003`, support positions 0--9,
  completed-query positions 10 and 11;
- full gate order M30 -> M10 -> M4 and exact fixed-pool trace binding.

No network layer, trainable parameter, loss, checkpoint, label, support row,
normalizer, threshold, batch size, or GPU profile may change.

## 4. Minimal loader seam

The V1 physical loader currently calls `source_execute.execution_closure_payload`
inside `_load_closure_bound_module()` for both provider materialization and
executor SWA loading. Add a minimal backward-compatible closure-injection seam
to `source_execute_physical.py`:

1. the default V1/V2 path must still use the historical V1 closure builder;
2. the provider and executor must each call a small overridable runtime-module
   method rather than the global loader directly;
3. the loader must accept only an explicit callable/current closure payload,
   re-hash the named module bytes under the same held-descriptor/no-follow law,
   and execute only those exact bytes in the existing private module namespace;
4. V4 route-local provider/executor subclasses override only that runtime-module
   seam and inject V4's current explicit closure;
5. no module-global mutation, monkeypatch, mutable-path import fallback, copied
   parser, or copied SWA loader is allowed;
6. closure validation occurs before module execution, source materialization,
   checkpoint load, or CUDA initialization.

The two directly executed helper files remain exactly:

```text
tfpd_exploration/src/tfpd_lane/pop_robust.py
tfpd_exploration/src/tfpd_lane/arm_common.py
```

Both must be explicit V4 closure members and be descriptor-rehashed against the
current closure row before execution.

## 5. Ownership and roots

Terra may edit only:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py
```

for the minimal backward-compatible hook, and may create only:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v4.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v4.py
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v4.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v4.py
```

Use fresh prospective roots:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v4
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v4
```

Do not edit V1/V2/V3 lifecycle modules, core, state-machine code, tests,
receipts, or result roots. Do not touch Z1/Z4/Z6/P4 or unrelated workspace
changes. V4 identity and every receipt must bind this work order, the complete
current no-glob closure, the V3 scientific contract, and the exact failed V3
predecessor graph.

## 6. Mandatory no-data/no-CUDA tests

At minimum prove:

1. the default V1 runtime-module path still invokes V1's historical closure
   validator and fails closed on the current V3 core bytes;
2. V4 provider and executor independently use the V4 closure-injection seam;
3. both `pop_robust.py` and `arm_common.py` are exact V4 closure members and are
   descriptor-rehashed before private execution;
4. missing/changed module row, byte drift, symlink, wrong parent identity, or
   wrong closure builder fails before module execution;
5. no global loader or package state is monkeypatched;
6. exact V3 predecessor graph validation plus body, sidecar, mode, topology,
   identity, closure, flag, and error-digest adversarial failures;
7. V4 capability issue rechecks the predecessor, current closure, selected
   device environment, and fresh spec-scoped root before any backend/data call;
8. the inherited V3 M10 smoke spec and independent-activity receipt semantics
   are unchanged;
9. the inherited V3 full M4/M10/M30 trace cardinality/order/digest-chain rules
   remain reachable and unchanged;
10. static CLI imports no Torch and performs no data, checkpoint, CUDA, write,
    root reservation, or launch action.

Run focused V3+V4 tests under exact no-user-site/no-CUDA isolation. Report
file SHA-256 values, explicit closure rows and closure SHA, exact commands,
test counts, dry output, whitespace checks, and fresh-root status.

## 7. Live sequence after root audit

This work order authorizes implementation and no-data/no-CUDA testing only.
After root independently accepts the frozen candidate:

1. run exactly one V4 M10 source smoke on an idle compatible GPU;
2. validate the independent activity trace and immutable terminal;
3. only then run the strict-27 V4 source gate;
4. only after a complete accepted V4 gate may a successor matched scorer be
   built and separately authorized.

Stop on any drift or exception. Never retry automatically.
