# T4 + CDM Cross-Dataset Validation Matrix

Date: 2026-08-28

Status: execution design. Existing immutable results remain authoritative; this
document specifies only the missing experiments needed for a defensible
cross-dataset claim.

## 1. Claim to test

The useful claim is not that one DANDI implementation can be copied unchanged
to every benchmark. The useful claim is:

> A support-only functional carrier and causal post-support activity memory can
> improve short-calibration decoding without target-session gradients, when the
> task exposes honest trial labels and completed-trial boundaries.

The method has two separable state variables:

- **carrier state:** T4 or a task-specific T4 analogue, initialized only from
  the labeled support prefix;
- **activity state:** a bounded collection of completed, unlabeled neural
  trials used by the calibration-activity identity path.

The DANDI CO and native FALCON M2 tasks admit the same directional T4 law. M1
admits a directional T4 but has different label geometry and weak existing T4
evidence. H1 uses the related H-C carrier and must be reported as a mechanism
transfer, not as the identical T4 system.

## 2. Common protocol

### 2.1 Two endpoints, never pooled

For each budget `M in {4, 10, 30}`, report two distinct endpoints when the data
permit them:

1. **Common-query endpoint:** support is the predeclared M-trial recipe, but every
   system is scored only on windows wholly after trial 30. This is the primary
   M4/M10/M30 degradation curve because all budgets have identical query rows.
2. **Deployment endpoint:** score after the support budget M. This measures the
   practical amount of usable data at that budget, but its rows differ across
   budgets and therefore cannot be used as the primary budget comparison.

Every scored window must have its complete history after the declared query
boundary. All arms in one cell must bind the same target, validity-mask, unit
order, support identities, query identities, checkpoint, and metric digests.

The selected support recipe is part of the estimand. Current deployment uses
causal D-optimal selection inside the first-30 cue pool at M4, and
chronological support at M10/M30. Because D-opt reads candidate cue directions,
its M4 row must be labeled `cue-budgeted M4`; it is not interchangeable with a
strict chronological first-four row.

### 2.2 Minimal system matrix

The following systems are sufficient. A larger ablation grid is not required
before the main performance result.

| System | Initial activity | Initial carrier | Post-support activity | Post-support carrier | Purpose |
|---|---|---|---|---|---|
| Static no-T4 baseline | first M | none | frozen | none | original SPINT/B3 or task-native baseline |
| Static T4 | first M | support-only OLS and fixed ridge 0.1 | frozen | frozen | establishes carrier value at the same budget |
| Activity-only CDM | first M | best frozen support-only carrier | causal update | frozen | isolates the activity-memory mechanism |
| Carrier-only CDM | first M | best frozen support-only carrier | frozen | causal gated update | isolates pseudo-label carrier refinement |
| Full CDM | first M | best frozen support-only carrier | causal update | causal gated update | complete method |
| Precision-CDM | first M | best frozen support-only carrier | causal update | precision/credible-region gated update | selected robustness successor |

At M30, static T4 is the governing safety anchor. The current CDM definition has
zero activity capacity at M30, and adaptive carrier updates are already known to
be unsafe on DANDI. Therefore M30 is primarily a non-regression check. A rolling
30-trial activity window may be reported as a separate drift diagnostic, but it
must not be silently called the same M30 cell.

### 2.3 Comparators and controls

Every dataset table should include the strongest available task-native
comparators:

- original SPINT/B3 or the accepted task baseline;
- static T4 with ordinary OLS;
- static T4 with fixed normalized ridge 0.1;
- dense-window ridge, trial-rate ridge, and population vector where their input
  contracts are defined;
- zero-carrier and channel-row-shuffled carrier as a single mechanism check,
  not a full three-budget sweep.

For CDM receipts, additionally report the number of completed trials, activity
commits, carrier proposals, carrier commits, rejection reasons, carrier
precision, worst session, latency, and peak memory. These diagnostics are needed
to distinguish a useful memory from a method that only updates more often.

## 3. Dataset-specific experiments

### 3.1 DANDI 000688 sorted SUA

Already complete:

- static T4/zero/shuffled external-15 evidence;
- sealed Cell-D, CDM-D V8, and Precision-CDM seed-42 M4/M10 results;
- M30 safety result showing that ordinary adaptive carrier updates are harmful.

Completed after this plan was written:

1. activity-only CDM at M10 and M4 on within-6 and external-15. External means
   are 0.388809 at M10 and 0.226495 at M4, improving over sealed by +0.093352
   (15/15) and +0.106811 (14/15), respectively.
2. the previously finalized three-seed label-budget curve has been folded into
   this plan. With B3S activity fixed at M30, SUA T4 is 0.304264 at M10 and
   0.358154 at M30; M15 is the smallest budget whose point estimate and all
   three seed means are within 0.03 of M50.

Still missing:

1. seeds 43 and 44 for the selected activity-only route;
2. carrier-only only as a bounded attribution diagnostic, not a new main route;
3. optional rolling-30 activity diagnostic, kept separate from governing M30;
4. one aligned table with paired per-session deltas, bootstrap intervals, sign
   counts, worst-session loss, and update/acceptance counts.

### 3.2 DANDI 000688 deterministic pseudo-MUA view

Use the exact electrode-sum transformation and the same session/trial/query
identities as the accepted paired-view result. Do not create a new pseudo-MUA
pooling rule after seeing behavior scores.

Already complete:

- the external-15 three-seed label-budget curve with B3S activity fixed at
  M30: M10 0.259185, M15 0.288234, M20 0.298447, M30 0.305291, M40 0.301075,
  and M50 0.306073. M15 is the smallest all-seed budget within 0.03 of M50;
- the M50 controls: F0 seed means -0.202339/-0.162278/-0.249558, PV50
  0.104219, Ridge50 0.410193, T4 0.306073. PV/Ridge use denser behavior labels
  and are performance comparators rather than equal-information controls.

Missing after deduplication:

1. at M10, the matched activityM arm needed to isolate the value of expanding
   B3S activity from 10 to 30 while holding the carrier and query fixed;
2. at M4, cue-budgeted D-opt/ridge static versus activity30 on the same fixed
   query; chronological M4 OLS is not assumed identifiable;
3. the selected full or Precision-CDM winner only if activity-only leaves a
   material residual and the view-specific carrier gate passes;
4. paired SUA-minus-pseudo-MUA degradation for every newly selected row;
5. exact carrier transformation-law and electrode-pooling evidence in every
   receipt.

Pseudo-MUA replication should follow, not precede, the sorted-SUA activity-only
decision. It is a granularity test, not an independent animal/task replicate.

### 3.3 Native FALCON M2 threshold-crossing MUA

M2 is the most important independent task replication because directional T4 is
well defined and official T4 system-level evidence is already positive. The CDM
evaluator must nevertheless be native-M2-specific.

The first frozen-weight local screen is complete:

- seven within post-30 sessions and six external official-query sessions;
- fixed-ridge static versus activity30 at M4/M10, plus M30 safety;
- external M10 +0.049070 (6/6) and M4 +0.068272 (5/6);
- no gradients or parameter updates; M4 activity is within 0.004227 of M30.

Remaining local development matrix:

- seven outer-LOSO held-in calibration sessions;
- budgets M4/M10/M30 on a common post-30 query, plus separate post-M deployment
  rows where legal;
- original SPINT, static T4 OLS, static T4 ridge 0.1, activity-only CDM, selected
  full/Precision-CDM, and fixed classical comparators;
- seed 42 for the complete screen, followed by seeds 43/44 only for the selected
  static and CDM systems.

The adapter must prove native 96-channel order, centre-trial exclusion,
continuous direction about `(0.5, 0.5)`, support-only fit rank/conditioning,
variable-length B3S support, completed-trial chronology, next-trial-only state
updates, and exact last-bin R2.

The organizer-hidden M2 interface does not currently expose the same trial
updates. Static T4 may use the existing official path. Online CDM is not an
official-submission result until either the interface exposes trial boundaries
or a separately validated causal trial/change detector is included in the
sealed inference package.

### 3.4 Native FALCON M1

M1 has a defined directional T4, but existing evidence is weak: the official T4
system did not improve the original system, and the corrected local post-support
effect is only descriptive. Therefore M1 must not begin with full CDM.

Required sequence:

1. audit first-4, first-10, and first-30 direction design rank, conditioning,
   trial counts, and a common post-30 query on the four held-in calibration
   recordings;
2. if M4 is rank deficient, mark M4 T4 undefined and use the smallest
   predeclared rank-3 support instead of a pseudo-inverse chosen post hoc;
3. score frozen-checkpoint static T4 OLS/ridge and activity-only memory first;
4. run carrier-only/full CDM only if static T4 attachment and the source-only
   pseudo-label constructibility gate are positive;
5. use outer-session folds and matched static checkpoints; do not reuse the
   historically contaminated minival endpoint.

This is a mechanism-boundary experiment. It should not be pooled with the M2 or
DANDI mean.

### 3.5 FALCON H1

H1 uses an H-C functional carrier rather than the identical 2-D velocity T4.
The accepted deployment support is M4. Existing activity-headroom experiments
already show that causal activity choice matters, while one variable-exposure
training attempt did not improve it.

Required sequence:

1. audit whether each accepted H1 recording has legal M10/M30 support and a
   common post-30 query; otherwise report only M4 and the largest common legal
   budget;
2. compare static H-C, activity-only causal memory, and the accepted H-C
   checkpoint on the same five-date surface first;
3. treat M10/M30 carrier refits as new H-C estimator experiments, not as a
   direct replay of the M4 model;
4. add carrier adaptation only after a source-only constructibility test defines
   how seven-output predictions become the trial-level carrier label;
5. retain the existing full-session activity arm only as a noncausal oracle.

The official H1 stream also lacks a validated trial-update interface. A CDM
official submission therefore has the same trial/change-detector prerequisite
as M2.

### 3.6 Datasets without a legal T4/CDM contract

Do not force this matrix onto datasets that lack either a trial-level functional
label or completed-trial boundaries. DANDI tasks with ICMS detection, position
decoding without a compatible target-direction law, or continuous official
streams without trial events require a new carrier and/or a change detector.
They can test the abstract memory principle later, but they are not direct T4
replications.

## 4. Execution order

1. Treat activity-only as the selected DANDI route and replicate seeds 43/44.
2. On deterministic pseudo-MUA, reuse the finalized activity30 label-budget
   curve and run only the missing activityM controls at M10 plus the matched
   cue-budgeted M4 static/activity30 pair.
3. Add F0/OLS/classical comparators to the completed native-M2 ridge/activity
   matrix; do not rerun the same 65 rows.
4. Replicate the selected M2 static/activity pair on seeds 43/44.
6. Run M1 rank/query feasibility and frozen-weight activity-only screen.
7. Run the H1 common-budget feasibility and activity-only screen.
8. Only after local gains exist, build a causal trial/change detector for an
   official FALCON CDM submission.

## 5. Minimum evidence for a broad claim

A defensible broad T4+CDM claim needs all of the following:

- positive paired M4 and/or M10 evidence on sorted SUA;
- preservation under the deterministic pseudo-MUA transformation;
- independent positive native-M2 local post-support evidence;
- at least two training seeds for each selected trained substrate, preferably
  three;
- M30 non-regression under the declared no-adaptive-carrier rule;
- explicit negative or inapplicable M1/H1 findings rather than omitted rows;
- no pooled average across SUA, pseudo-MUA, M2, M1, and H1;
- a separate statement that local trial-aware CDM is not yet equivalent to an
  official continuous-stream submission.

M1/H1 success would strengthen the generality claim, but it is not required to
establish cross-granularity plus independent-task replication. A clean negative
M1/H1 result is scientifically useful because it identifies the boundary of the
directional-carrier/trial-memory mechanism.
