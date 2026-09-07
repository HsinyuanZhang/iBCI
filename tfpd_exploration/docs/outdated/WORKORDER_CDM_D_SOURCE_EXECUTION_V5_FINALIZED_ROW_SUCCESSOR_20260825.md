# Work Order: CDM-D Source Execution V5 Finalized-Row Successor

Date: 2026-08-25  
Status: **BUILD AND TEST ONLY; NO DATA, CHECKPOINT TENSOR, CUDA, RESULT ROOT, OR LAUNCH UNTIL ROOT REVIEW**

## 1. Purpose

The V3 scientific state machine is valid and the V4 M10 physical smoke proved
its decisive behavior: two valid completed-query B3S rows advanced the M10
activity FIFO even though both carrier transitions were rejected. V4 also
fixed the closure-loader seam and reached the strict-27 physical route.

The V4 full source gate then failed deterministically on its first M30 row.
This is not a carrier, B3S, theta, model, threshold, or state-transition
failure. It is a backend-layer type error:

```text
executor event: FinalizedFourGroupPseudo
    -> base source-only audit join
B8 input row: GroupedPseudoAuditRow
```

The V1 base `_finalized_row()` intentionally returns the second object. The V3
override calls that base method and then incorrectly asserts that the returned
object is still the first object. V5 repairs only where the independent-state
trace is captured. It does not change the scientific system.

## 2. Accepted V4 smoke predecessor

Before any V5 capability issue or root reservation, hold one `O_NOFOLLOW`
descriptor for:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v4
```

Require exactly ten regular non-symlink mode-0444 leaves: five bodies and their
canonical basename sidecars, with no extras or failure.

```text
attempt.json           c9dfd75b173d92c95d85e77327a2f02e1190b91afb594aada245a7ed691c8ca9
launch.json            c8c7c1e72b5904286ee5a3c139e5b44d952af153ee29aa4b23f2f28019074736
source_authority.json  2d473889d0d206e43834f34ef814dd0511d34352e6252d9c67549e425c23d42e
smoke.json             bef1637ea02b54f6f51ea5c0516b3322eeb2d9e29664702e874bd69591a3c02f
terminal.json          305499831e169670e3ff0988a52ae8d3425960daf250e028c7182e5e3a8c73ca
```

The graph must bind identity
`38291a4af71d0ef03b9060a59f49ae1df194375d64d8bbb3d6c14a01907e6f50`,
V4 closure
`7854b667abfdec7cb004eff64e52c9f662e5eed293d7f0faef9d4fc45a094200`,
status `SMOKE_COMPLETED`, M10 support positions 0--9 and audit positions 10--11,
two committed activity transitions, zero committed carrier transitions, two
carrier rejections, exact activity change, exact carrier preservation, one
continuous state chain, source-only access, and zero target/backward/optimizer/
parameter updates.

## 3. Failed V4 gate predecessor

Also hold one `O_NOFOLLOW` descriptor for:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v4
```

Require exactly eight regular non-symlink mode-0444 leaves: four bodies and
canonical basename sidecars, with no terminal, budget evidence, or extras.

```text
attempt.json           8adff90f9686c6aaa9e1ba7306fee3e417009e0e74b6e09ca82182031cdb8da9
launch.json            45de7f3d0df03f33aef240fb8214559c434108cb0d0f2f9f96047d323d8c03a2
source_authority.json  7264c4c5a57f7edc8fecd1f7144f4603cdb4f92a745fc8cfccf46afbcb5e8324
failure.json           7105726660dbc481d791df7a2e63fa8870c8dd66dc28be02f841dbb3fa05720e
```

The four bodies must bind identity
`4c2f6fa1a1359b0f393340f37de90786de3fd72ab538cefdb45dfcf69e95cbe4`
and the same accepted V4 closure. The failure must be
`status=FAILED`, `stage=budget_m30`, class `SourceExecutionPhysicalV3Error`,
error digest
`8e918cf7fad3d75ce559664025520b92f12bb3591698d8937ab2602c315dbb04`,
with source/checkpoint/CUDA opened, exactly eight model-forward calls, zero
backward/optimizer/parameter/target updates, no normalizer refit, no OOM retry,
all within/external/formal/target surfaces unopened, and no terminal.

Missing, writable, symlinked, replaced, sidecar-mismatched, semantically
altered, or extra predecessor leaves are fatal before a V5 write, source
resolution, checkpoint load, or CUDA initialization. V4 roots remain immutable
and are never retried or edited.

## 4. Frozen scientific contract

V5 preserves V3/V4 exactly:

- sealed Cell-D graph, SWA, parameters, FP32 inference, and normalizers;
- M4 D-optimal, M10 chronological, and M30 chronological support;
- fixed-ridge-by-trial carrier initialization with lambda 0.1;
- budget-specific K=4 complementary groups and all frozen B8 thresholds;
- B3S FIFO capacities M4=26, M10=20, and M30=0;
- valid B3S activity commit independent of carrier acceptance;
- current-trial prediction before post-trial state commit;
- M30 positions 30--59 offline with exact no-deployment-state mutation;
- full gate order M30 -> M10 -> M4, fixed pools, trace cardinality/order, and
  digest-chain validation;
- source-only execution and zero target gradient/update.

No model layer, trainable parameter, loss, checkpoint, support row, label,
normalizer, threshold, batch size, parser, truth join, or device profile may
change.

## 5. Narrow V5 physical seam

Create a route-local V5 composition. Do not edit V1/V2/V3/V4 files or shared
model/data code.

1. Subclass V4's independent-activity executor. Its single
   `execute_completed_trial()` call must return the unchanged exact
   `FinalizedFourGroupPseudo` and additionally retain that exact raw event in a
   private one-shot map keyed by `(session, budget, trial_id)`. Reject duplicate
   keys, missing consumers, stale leftovers, wrong identity, or a second read.
2. Subclass V4's physical backend and override only `_finalized_row()`.
3. Invoke `PhysicalSourceExecutionBackend._finalized_row()` exactly once as an
   unbound V1 base call. This remains authoritative for raw-event validation,
   the source-only truth join, rejection-reason validation, and construction of
   `GroupedPseudoAuditRow`. Do not copy those steps.
4. After that one base call returns, consume the executor's exact retained raw
   event. Validate exact session/budget/trial identity, exact raw type, the V3
   independent-activity contract, and the transition payload.
5. Append exactly one transition row to the inherited V3 trace store, then
   require the base return value to be the exact closure-bound
   `GroupedPseudoAuditRow` and return it unchanged to B8.
6. No second executor call, model forward, truth join, label access, carrier
   update, activity update, or result conversion is allowed.
7. The V5 factory must require exactly its V4 provider plus V5 executor/backend
   types. No monkeypatch, global mutation, mutable import fallback, copied
   parser, copied SWA loader, or copied B8 implementation.

The raw event is pre-audit-join mechanism evidence. The grouped row is the
source-audit/B8 input. The receipt must never conflate these types.

## 6. Lifecycle, closure, and ownership

Terra may create only:

```text
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v5.py
tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v5.py
tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v5.py
tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v5.py
```

Use one fresh prospective root:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v5
```

V5 lifecycle may compose V4 artifact and validation primitives but must use a
new schema/identity/closure and must bind this work order, the complete current
no-glob implementation closure, accepted V4 smoke graph, failed V4 gate graph,
V4 closure, V3 scientific contract, strict roster, fixed assets, selected
device, and source-data capability. Recheck both held predecessors and current
closure before capability issue, before root reservation, before backend/data,
and before terminal/failure publication.

Attempt must publish before source resolution. Success topology is attempt,
launch, source authority, all exact session/aggregate budget evidence, and
terminal pairs. Failure topology is the honest completed prefix plus exactly
one failure pair and no terminal. Every durable leaf is a regular non-symlink
0444 file with a canonical basename sidecar. No retry or resume exists.

## 7. Mandatory no-data/no-CUDA tests

At minimum prove:

1. an actual V3/V4 full-gate `_finalized_row()` call reproduces the historical
   wrong post-conversion type assertion;
2. the V5 full-gate MRO path captures exact `FinalizedFourGroupPseudo`, calls
   the raw executor once, calls the audit truth join once and only after raw K4
   completion, records one trace row, and returns exact `GroupedPseudoAuditRow`;
3. duplicate, absent, stale, mismatched, forged, or twice-consumed raw events
   fail closed;
4. a returned wrong grouped-row type fails even when the retained raw event is
   valid, and a forged raw event fails even when the grouped row is valid;
5. M30 records an offline transition with identical state/activity/carrier
   digests and no deployment commit;
6. M10 and M4 trace rows preserve exact independent activity-chain semantics,
   including carrier-rejection activity commits;
7. full M30/M10/M4 row order/cardinality/parent/digest validators remain exact;
8. both predecessor graphs reject topology/body/sidecar/mode/identity/closure/
   semantic tampering before V5 reservation;
9. a complete temporary V5 full-gate lifecycle publishes exact per-budget
   rows/aggregates and a terminal, with reload-before-downstream validation;
10. failure at every stage publishes honest flags and no terminal;
11. current closure, selected GPU environment, source capability, and
   spec-scoped root freshness are rechecked before any action;
12. public CLI is static, Torch-free, no-data, no-checkpoint, no-CUDA,
   no-write, and cannot mint the opaque capability.

Run isolated V3+V4+V5 focused tests and the complete V5 lifecycle tests under
no-user-site/no-CUDA settings. Report exact test command/count, pycompile,
whitespace, dry output, owned file SHAs, explicit closure rows/SHA, and
prospective root status. Tests must directly cover the full-gate boundary; a
smoke-only test is insufficient.

## 8. Live sequence after root review

This work order authorizes implementation and no-data/no-CUDA testing only.
After independent root acceptance, run exactly one V5 strict-27 source gate on
an idle compatible GPU. A redundant M10 smoke is not required because V5 does
not change the already accepted executor/scientific smoke path; its mandatory
full-boundary test is the relevant pre-launch gate.

Stop fail-closed on any drift or exception and never retry automatically. Only
after a complete independently accepted V5 terminal may a successor matched
scorer be built or launched.
