# Work Order: CDM-D Independent-Activity Successor V3

Date: 2026-08-25  
Status: **BUILD AND TEST ONLY; NO DATA, CUDA, RESULT ROOT, OR LAUNCH UNTIL ROOT REVIEW**

## 1. Why V3 is required

The intended CDM-D system has two different target-time memories:

1. an unlabeled B3S activity FIFO; and
2. a pseudo-label-gated carrier sufficient-statistics memory.

The activity FIFO is allowed to consume every completed query trial because it
uses neural activity only. The carrier may change only when all complementary
prediction, trust, snapping, conditioning, and departure gates accept the
trial.

The frozen V2 implementation does not have this separation. Its
`CausalDualMemory.observe_completed_trial()` creates a candidate activity
state only after the carrier proposal passes. Therefore any carrier rejection
also rejects the unlabeled activity row. This contradicts the performance
design and matched-score work order.

V2 remains immutable historical evidence. V3 is a narrow successor, not an
in-place reinterpretation of the V2 receipts.

## 2. Frozen scientific system

Keep all of the following unchanged:

- sealed Cell-D model, SWA, parameters, and inference graph;
- M4 D-optimal, M10 chronological, and M30 chronological support rules;
- fixed-ridge-by-trial carrier initialization with lambda 0.1;
- K=4 budget-specific complementary groups;
- all pseudo-direction, trust, snapping, conditioning, departure, and freeze
  rules;
- B3S capacities M4=26, M10=20, and M30=0;
- source-only and target-zero-gradient boundaries;
- all source-gate thresholds and fixed audit pools.

No network layer, trainable parameter, normalizer, loss, checkpoint, or label
budget changes in V3.

## 3. Required independent transition semantics

For completed query trial `j`, after all predictions for `j` are finished:

1. validate the typed B3S activity capability against the current activity
   state;
2. if the B3S capability is invalid, commit no transition;
3. if the B3S capability is valid, construct the next activity state
   independently of every native-count, velocity, pseudo-label, and carrier
   gate;
4. attempt the carrier proposal using the unchanged full typed join and all
   unchanged gates;
5. if the carrier proposal is accepted, commit the new activity state and the
   new carrier state;
6. if the carrier proposal is rejected or its carrier-side capability is
   invalid, commit the new activity state and retain the exact old carrier;
7. increment the completed-query transition count whenever the valid B3S
   activity transition is committed;
8. expose the new state only to trial `j+1`.

The outcome must report these facts separately:

```text
activity_transition_committed
activity_fifo_changed
carrier_transition_committed
carrier_rejection_reason_or_null
state_before_sha256
state_after_sha256
activity_before_sha256
activity_after_sha256
carrier_before_sha256
carrier_after_sha256
```

The old single `accepted` / `committed` bit must not be reused to represent
both memories.

### M30

M30 has literal activity capacity zero. A valid completed trial may advance
the completed-query counter, but the B3S activity stack, query-row count, and
activity digest must remain exact. The existing source-audit M30 positions
30--59 remain an offline constructibility pool and do not enter deployment
memory during the source gate. The later matched score may still apply the
unchanged accepted carrier transition after each post-support target trial.

## 4. Implementation boundary

Prefer a backward-compatible addition to the existing core plus additive V3
route files. Do not change the behavior of the frozen V2 methods or mutate V2
result roots.

Terra owns only:

- the minimal backward-compatible additions in
  `tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py`;
- new additive V3 source-execution modules under
  `tfpd_exploration/src/causal_dual_memory_cell_d_v1/`;
- one new V3 dry CLI;
- one new focused V3 test;
- the later additive matched-score successor only after the V3 source gate
  passes and root issues a separate work order.

Terra is not alone in the workspace. It must preserve all unrelated changes,
must not touch Z1/Z4/Z6, and must not edit or delete V1/V2 receipts.

## 5. Mandatory no-data tests

The focused suite must prove at least:

1. M4/M10 carrier rejection advances the B3S FIFO and completed-query count
   while the carrier digest remains exact;
2. carrier acceptance advances both memories;
3. malformed or wrong-session/wrong-channel B3S causes no transition;
4. malformed native counts or velocity evidence cannot update the carrier but
   a separately valid B3S row still advances activity;
5. stale pending updates fail closed;
6. a transition from trial `j` cannot affect any prediction for trial `j`;
7. M30 activity stack/query count/digest remain exact at capacity zero;
8. FIFO capacity and eviction remain exactly 26/20/0;
9. host Python/NumPy/Torch RNG state is unchanged;
10. the frozen V2 API still has its old behavior and V2 immutable evidence is
    labeled superseded rather than rewritten;
11. receipt validators reject a forged outcome that collapses the two booleans
    back into one;
12. zero-argument CLI imports no Torch and performs no data, CUDA, write, or
    launch action.

## 6. Live sequence after root audit

After the no-data candidate freezes and root independently accepts it:

1. run one V3 source-only smoke on the selected idle compatible GPU;
2. validate its immutable receipt and the independent-transition evidence;
3. run the complete strict-27 V3 source gate with the original thresholds and
   audit pools;
4. bind the fresh V3 closure and terminal into a new matched-score successor;
5. only then run M30 -> M10 -> M4 matched performance scoring.

The source gate is expected to be short. It does not authorize target scoring
unless its exact V3 terminal is complete and root separately approves the
matched-score successor.

## 7. Stop conditions

Stop before data or GPU if any of the following is true:

- independent activity advancement requires a target label;
- a carrier rejection can still suppress a valid activity transition;
- activity can advance before all predictions for the current trial finish;
- M30 activity state changes at capacity zero;
- V3 cannot distinguish activity and carrier transition facts in receipts;
- the implementation requires changing Cell D, its SWA, or any target
  optimizer boundary;
- the current implementation closure cannot be reconstructed exactly.

