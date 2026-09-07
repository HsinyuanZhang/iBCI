# H1 CarrierID next-route audit: naming, source-only selection, and capacity controls

**Status:** planning/audit only.  This document starts no job, opens no NWB,
does not inspect target metrics, and does not modify the sealed H1-C,
D-S4e/D-Q4e, or RT artifacts.

**Authority boundary:** it resolves the ambiguous next-route wording in
[`H1_CARRIERID_FRESH_MATCHED_DISTRIBUTION_PROTOCOL.md`](H1_CARRIERID_FRESH_MATCHED_DISTRIBUTION_PROTOCOL.md).
The D-S4e/D-Q4e comparison is explicitly a development leakage diagnostic.
Its frozen result can classify a *hypothesis*, but it must not select a
checkpoint, width, epoch, or final model.  Every route below has a separate
source-only gate before it is eligible for a fresh target evaluation.

## 1. Retire the overloaded `C7` name

`C7` is not defined anywhere in the H1 CarrierID implementation or protocol,
yet the distribution protocol says “test C7, then H64.”  It is already used
elsewhere for unrelated work (notably the T4 labelled-budget experiment and a
fixed-K hyperparameter proposal).  It is therefore not an auditable H1 name.

The canonical H1 names are:

| Canonical name | Meaning | Replaces |
| --- | --- | --- |
| `H1-EST4-SLODO` | 4-D learned closed-form carrier-estimator source-date-LODO screen | vague “estimator/Q1/shrinkage route” |
| `H1-CI64-SLODO` | **C**arrier **I**nterface width 64 while neural carrier hidden width stays 32 | undefined H1 “C7” |
| `H1-H64-SLODO` | full CarrierID hidden-width-64 escalation | ambiguous “H64” follow-up |
| `H1-DIST-XDATE-REPL` | fresh, predeclared cross-date D-S4/D-Q4 replication preparation | unresolved branch |

No new H1 receipt, configuration, run directory, or paper text may use bare
`C7` after this audit.

## 2. The three valid D-S4e/D-Q4e exits

The exposure terminal preflight already defines the only valid decisions:
validity failure, then `estimator-limited evidence`, `consumer-limited
evidence`, or `unresolved`.  Validity failure is a failure of the diagnostic,
not evidence for an architecture.  Conditional on validity passing, the three
scientific exits map to one unique *next preparation*:

| Frozen D-S4e/D-Q4e interpretation | Unique next preparation | What it can answer | What it cannot do |
| --- | --- | --- | --- |
| `estimator-limited evidence` | `H1-EST4-SLODO` | Whether a source-loss-trained, label-structurally-required 4-D analytic estimator improves source-date generalization over the frozen estimator | It cannot use D-Q4e, fold-0 target R², or oracle R² to select learned width/epoch |
| `consumer-limited evidence` | `H1-CI64-SLODO` | Whether the bottleneck is the 36-to-32 carrier attachment interface rather than the carrier estimate | It cannot call H64, select width, or claim capacity from D-Q4e alone |
| `unresolved` | `H1-DIST-XDATE-REPL` | Whether the diagnostic’s sign/magnitude is reproducible on one predeclared, still-unopened date under the same paired/exposure contract | It cannot choose estimator or consumer architecture after inspecting mixed fold-0 recordings/seeds |

If D-S4e fails its predeclared absolute validity gate, the disposition is
`H1-DIST-REPAIR-ONLY`: repair or abandon the diagnostic.  Do **not** enter any
of the three architecture routes.

This is a frozen decision tree for development bookkeeping, not a license to
use query-local labels for deployment.  A D-Q4e-dependent choice still carries
development-selection risk.  Therefore the route name above merely determines
which source-only screen may be *considered*; its eventual candidate is fixed
only by that screen and cannot be rescued or re-ranked with fold-0 target
scores.

## 3. What “consumer-limited” does and does not establish

After an exposure-valid, near-zero D-Q4e−D-S4e result, the supported statement
is narrow: the matched h=32 consumer did not profit from the deliberately
different carrier distribution.  It does not prove that the existing carrier
is optimal, that H64 will help, or that an interface width is the cause.  The
oracle displacement audit is consistent with attenuation, but not diagnostic
of capacity versus source-distribution mismatch.

Accordingly the first consumer experiment must isolate the point at which the
carrier meets the neural representation:

```text
H1-CI32: pooled[32] + carrier[4] -> 36 -> 32 -> 32 -> 700   (current H1-C)
H1-CI64: pooled[32] + carrier[4] -> 36 -> 64 -> 32 -> 700   (interface-only probe)
H1-H64 : pooled[64] + carrier[4] -> 68 -> 64 -> 64 -> 700   (whole carrier encoder widening)
```

`H1-CI64` leaves the 1024-to-32 neural pre-pool and the 32-wide token delivered
to the unchanged SPINT decoder intact.  It expands only the joint carrier/
pooled-activity interface.  `H1-H64` doubles both the neural pre-pool and the
post-pool path; it is therefore a broad identity-encoder capacity experiment,
not a clean interface test.  **H1-H64 is not the next experiment.**

## 4. Required retraining, parameter accounting, and controls

Neither CI64 nor H64 is checkpoint-compatible with H1-C: tensor shapes in the
first/second carrier-post layers change.  Both must be independently trained
from scratch on their source data.  Loading H1-C and padding, copying, or
fine-tuning its weights is prohibited; it would confound capacity with a
warm-start advantage.

All counts below are static session-identity encoder parameters.  The decoder
and downstream SPINT path remain unchanged.

| topology | pre-pool | post-pool | identity params | whole-model params | reduction versus SPINT ID MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `H1-CI32` / current H1-C | 32,800 | 25,340 | 58,140 | 10,947,836 | 102.606x |
| `H1-CI64` | 32,800 | 27,548 | 60,348 | 10,950,044 | 98.852x |
| `H1-H64` | 65,600 | 54,076 | 119,676 | 11,009,372 | 49.847x |
| SPINT identity MLP reference | — | — | 5,965,500 | 16,855,196 | 1.000x |

The increment for CI64 is only 2,208 parameters and doubles the carrier-entry
weight block from 128 to 256.  H64 adds 61,536 parameters over H1-C and also
changes the neural rate pathway; treating it as an “interface-only” control
would be false.

For each candidate width, all source runs must be separately trained with:

1. `full` — correct M=4 support carrier;
2. `C0` — literal zero carrier at the model boundary, with the same topology;
3. `LS` — label-rotated carrier, separately trained with the same topology;
4. `RS` — row-shuffled carrier, separately trained with the same topology.

The full and C0 models need identical random initial states; the carrier-entry
columns are zero at initialization for every arm.  LS and RS receive the same
initial state too.  A CI64-vs-CI32 comparison alone only measures capacity.
CI64-full must also beat CI64-C0 and CI64-LS (and should beat CI64-RS) to
support a carrier-content interpretation.

## 5. Strict source-only selection: feasible in principle, absent in current code

It is possible to decide **whether H64 is eligible** without target data, but
the present H1 implementation cannot do it yet:

- `src/data/h1_m4_eb_pilot.py` fixes `FOLD0_DATE = "19250101"` and exposes
  only a fixed fold-0 source/target split;
- `H1CarrierIdSpint` rejects any `carrier_hidden_dim != 32`;
- `H1CarrierIdLitModule` and the distribution wrappers hard-code h=32 and
  fold date `19250101`;
- no generic source-date-LODO trainer/selector, CI64 model, H64 model, or
  matching checkpoint schema exists.

Thus a claimed “source-only H64 selection” cannot be honestly run by changing
a YAML scalar.  The missing prerequisite is a new, isolated source-date-LODO
preflight that opens only the eleven existing fold-0 **source** recordings,
leaves date `19250101` unopened, and partitions the five source dates
`19250108/13/15/19/20`.  Every outer source date must score only strict
post-support windows from that held source date.  It must record source file,
schedule, normalizer, initialization, and checkpoint hashes and forbid
minival/formal/EvalAI routes.

The fixed source-only kill gate before H64 is eligible is:

```text
Run H1-CI32 and H1-CI64 on the same five source-date LODO folds and fixed epochs.

CI64 mechanism pass requires all of:
  mean_date R2(CI64-full - CI32-full) > 0 and at least 4/5 dates positive;
  mean_date R2(CI64-full - CI64-C0)  > 0 and at least 4/5 dates positive;
  mean_date R2(CI64-full - CI64-LS)  > 0 and at least 4/5 dates positive.

If any clause fails: stop the consumer-width route; do not run H64.
If all pass: H1-H64-SLODO becomes an eligible, predeclared single escalation,
with H64-full/C0/LS/RS and H1-CI64-full as its controls.
```

This gate is deliberately source-only.  It does not claim that a source proxy
is the final decoding endpoint; it merely prevents the already opened fold-0
development target from selecting a larger consumer.

## 6. Minimal implementation readiness

[`scripts/h1_carrierid_next_route_plan.py`](../scripts/h1_carrierid_next_route_plan.py)
is an intentionally data-free static plan.  It fixes names, topology arithmetic,
controls, selection prerequisites, and the H64 escalation condition.  It does
not import torch, a DataModule, or an evaluator, and has no launch mode.  Its
test only checks the algebra and fail-closed source-only contract.

No GPU experiment is authorized by this audit.  In particular, do not add an
H64 override to an existing H1-C/D-S4e/D-Q4e configuration: that would evade
both the retraining requirement and the source-only selection gate.
