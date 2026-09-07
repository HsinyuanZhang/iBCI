# Design: Support-Anchored Causal T4 Memory

Date: 2026-08-30

Status: design and execution guidance for independent review. This document
does not authorize data access, result-root creation, scoring, training, or a
GPU launch.

Primary surface: DANDI 000688 sub-C/sub-M center-out, with explicit completed
trial boundaries and M4/M10/M30 labeled-support budgets.

Relationship to the paper: this is the conditional carrier-state extension of
**Causal Calibration Memory**, the proposed second contribution beside the T4
functional carrier. Activity-only memory is already supported by positive
results. Continuous T4 memory remains a research candidate until it beats the
activity-only system.

---

## 0. Executive decision

Continuous T4 adaptation is **not closed as a research direction**.

The evidence supports a narrower statement:

> The continuous carrier updates implemented so far have no incremental
> performance value over activity-only CDM. They do not prove that every causal
> T4 update is useless.

The existing negative result combines three possible failures:

1. the pseudo-direction measurement may be biased and self-referential;
2. the recursive per-trial update may accumulate that bias;
3. the gate may measure support precision without measuring proposal quality or
   new direction coverage.

These factors have not been separated. The first required experiment is
therefore a **causal oracle headroom test**. It asks whether a correct
completed-trial direction, used only after the trial ends, can improve T4 on the
next trial. If a support-anchored oracle refit does not beat activity-only CDM,
the continuous-T4 line stops. If it does, the line proceeds to a deployable
cross-fitted pseudo-direction estimator.

The recommended method is not another unrestricted online RLS. It is a
**support-anchored, block-refit, trust-region carrier memory**:

```text
frozen labeled support carrier
    + bounded sufficient statistics from finalized completed trials
    + cross-fitted trial-direction measurement
    + direction-coverage gate
    + support-posterior trust region
    -> optional next-trial T4 correction
```

M4 and M10 are the primary budgets. M30 is a safety and headroom diagnostic;
carrier updates remain disabled by default at M30.

## 1. Evidence that fixes the design

### 1.1 Immutable evidence anchors

| Evidence body | SHA-256 |
|---|---|
| `results/causal_dual_memory_cell_d_activity_only_quick_v2/result.json` | `48d658c866b0bcb45a9f03c6e0c84204ecb87286e940d850c846a46f67b33bb8` |
| `results/causal_dual_memory_cell_d_matched_score_v8/score.json` | `98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e` |
| `results/precision_aware_causal_dual_memory_cell_d_matched_score_v2/score.json` | `4e06ffc544210fbb7cba68c21fd86831c7d0a0407d43ef379a33b9a05bb2bcb1` |
| `results/budget_matched_posterior_cal_aug_v1/c2_c3_posterior_gpu_score_v8/terminal.json` | `c04c0f86a682fe3aa256268b543021e0733dfb52169273e81a9a5791e86beb75` |
| `docs/DESIGN_CAUSAL_CALIBRATION_MEMORY_SECOND_INNOVATION_20260830.md` | `36d6e869faf316cc7ccfac545730379fa3562b780423d14a842de5c918604763` |

All comparisons in this document use activity-only CDM as the incremental
carrier-update reference. Beating sealed static Cell-D is not enough to claim
that carrier memory adds value.

### 1.2 Activity-only memory is the positive baseline

The activity-only system freezes the support-derived carrier and updates only
the completed-trial B3S activity state.

| External budget | Sealed static | Activity-only CDM | Delta | Breadth |
|---|---:|---:|---:|---:|
| M4 | 0.119684 | 0.226495 | +0.106811 | 14/15 |
| M10 | 0.295456 | 0.388809 | +0.093352 | 15/15 |

Within improvements were `+0.156670` at M4 and `+0.073769` at M10, both
positive in 6/6 sessions. This is the minimum baseline that any carrier-memory
successor must beat.

### 1.3 The current carrier state removes part of that gain

Full CDM V8 allowed gated pseudo-label carrier updates in addition to activity
memory.

| External budget | Full CDM V8 | Delta vs sealed | Delta vs activity-only |
|---|---:|---:|---:|
| M4 | 0.191126 | +0.071442 | -0.035369 |
| M10 | 0.340083 | +0.044627 | -0.048726 |
| M30 | -- | -0.059092 | unsafe update row |

The full system remained above sealed static at M4/M10, but the carrier state
reduced the activity-memory benefit. At M30, where the support carrier is
already strongest, carrier updates caused a material loss.

The accepted-update pattern was also scientifically backwards: the system
accepted many updates at M30 and few at M4, although a useful uncertainty-aware
rule should permit smaller movement when the labeled support is more precise.

### 1.4 Precision gating repairs safety but does not add performance

| External budget | Precision CDM V2 | Delta vs sealed | Delta vs activity-only |
|---|---:|---:|---:|
| M4 | 0.222630 | +0.102946 | -0.003865 |
| M10 | 0.351002 | +0.055546 | -0.037807 |

At M4, precision gating removed almost all of the carrier-update damage. At
M10, substantial damage remained. This proves that bounded rejection can make
the state safer. It does not prove that the accepted proposals contain useful
new information.

### 1.5 E02/E03 do not test continuous carrier memory

The E02/E03 development results provide weak motivation for posterior
shrinkage at M50:

| Cell | Mean R2 | Delta | Breadth |
|---|---:|---:|---:|
| E01 ordinary T4 | 0.6137 | -- | -- |
| E02 posterior mean | 0.6287 | +0.0150 | 5/8 |
| E03 posterior plus reliability | 0.6367 | +0.0080 vs E02 | 5/8 |

They are static, wide-budget, same-subject development results. They neither
test causal online updates nor establish external M4/M10 transfer. The local
budget-matched C2/C3 result also rejects unconditional posterior replacement
and a free q consumer feature. Therefore posterior information may constrain a
future update, but it must not replace the support carrier or enter the decoder
as an unrestricted feature.

## 2. What is closed, open, and conditional

### 2.1 Closed implementations

Do not repeat these exact systems:

- recursive per-trial pseudo-label RLS that treats the previous pseudo-updated
  carrier as the next prior;
- unconditional posterior-mean replacement of the support carrier;
- posterior reliability q concatenated as a free decoder feature;
- M30 carrier updates enabled by default;
- additional seeds of the current full-CDM or C2/C3 implementations;
- threshold tuning on the current departure statistic without changing the
  pseudo-direction measurement or update estimator.

### 2.2 Open scientific questions

The following questions have not been answered:

1. Does a correct completed-trial direction create causal carrier headroom over
   activity-only CDM?
2. Is the failure caused by recursive accumulation rather than carrier
   adaptation itself?
3. Can a direction derived from complementary unit groups avoid direct
   self-labeling?
4. Can full-trial trajectory aggregation improve the direction measurement?
5. Can a gate jointly enforce measurement confidence, direction coverage, and
   support-posterior distance?
6. Is useful carrier adaptation limited to M4, or does it also exist at M10?

### 2.3 Conditional paper position

The second contribution remains:

```text
Causal Calibration Memory
|-- activity state: validated positive core
`-- carrier state: conditional support-anchored extension
```

Until the carrier extension beats activity-only CDM, figures and prose must use
a dashed or explicitly experimental carrier branch. The validated claim is the
activity state, not full dual-memory superiority.

## 3. Required first experiment: causal oracle carrier headroom

### 3.1 Scientific question

The first experiment must separate carrier-update headroom from pseudo-label
quality:

> If the true direction of a completed query trial were available immediately
> after that trial ended, could a legal next-trial T4 update improve the frozen
> decoder beyond activity-only CDM?

The true direction is diagnostic target-label leakage. It is never a deployable
input and must be recorded as:

```text
target_label_leakage = true
checkpoint_selection_from_target = false
hyperparameter_selection_from_target = false
```

### 3.2 Causal discipline

For trial `i`:

1. score all query windows of trial `i` using the state available before it;
2. finalize the trial and its immutable prediction/input evidence;
3. reveal the oracle direction only after finalization;
4. construct a carrier proposal;
5. commit or reject the proposal;
6. allow the resulting state to affect trial `i+1` only.

No update may change the predictions or score of the trial that supplied the
measurement. No target gradient, target backward, optimizer step, model update,
normalizer refit, or decoder-state update is allowed.

### 3.3 Primary cells

| Cell | Activity state | Direction measurement | Carrier estimator |
|---|---|---|---|
| O0 | activity-only CDM | none | frozen support T4 |
| O1 | same | true completed-trial direction | existing incremental update |
| O2 | same | true completed-trial direction | support-anchored block refit |

O1 is diagnostic. It determines whether the old estimator fails even with a
correct direction. O2 is the actual headroom test.

### 3.4 Budgets and surfaces

- M4 and M10 are co-primary.
- M30 is a mandatory diagnostic and safety column.
- Report within-6 and external-15 separately.
- Never average M30 with M4/M10 to rescue or reject a low-budget effect.
- Use exactly the same support trials, query trials, chronological order,
  decoder, normalizer, activity-memory state, query-window authority, and
  variance-weighted R2 implementation as the accepted activity-only scorer.

### 3.5 Oracle headroom gate

Use a three-way decision. **GO** requires O2 to satisfy all of the following on
at least one low budget:

```text
O2 - O0 >= +0.03 equal-session external R2
positive external sessions >= 10/15
the other low budget >= O0 - 0.01
within every budget >= O0 - 0.02
all target updates/backward/optimizer steps = 0
```

**HOLD** applies when the primary delta is between `+0.01` and `+0.03`, at
least 9/15 external sessions are positive, and no safety condition fails. HOLD
authorizes only one source-selected block-size/shrinkage sensitivity check; it
does not authorize the full pseudo-direction program. The sensitivity rule must
be frozen before returning to the held target surface.

**STOP** applies when O2 is below `+0.01`, fewer than 9/15 external sessions are
positive, the result depends on a causality leak, or a safety condition fails.

Interpretation:

- `O2 > O0`, `O1 <= O0`: the recursive update is wrong, but carrier adaptation
  has headroom.
- `O2 > O1 > O0`: both the measurement and estimator can help.
- `O1/O2 <= O0`: close continuous T4 for the current decoder and activity
  system.
- positive M4 only: continue as an M4-only carrier extension.
- positive M10 only: treat M4 pseudo-direction quality as the likely bottleneck.
- nonpositive M30: keep M30 carrier movement exactly disabled.

The oracle gate decides whether headroom exists. It cannot establish a
deployable method claim.

## 4. Support-anchored block refit

### 4.1 Frozen support sufficient statistics

For each unit, preserve the exact production T4 regression convention and the
frozen labeled-support sufficient statistics:

```text
A0 = Xs^T Xs + lambda I
b0 = Xs^T rs
T4_support = solve(A0, b0)
```

`Xs` contains the same carrier design columns and direction convention as the
accepted support-only T4 estimator. `rs` uses the same native trial-rate domain,
valid-unit mask, direction grouping, unit order, and ridge law.

The support carrier and its sufficient-statistic digests never change.

### 4.2 Bounded pseudo or oracle evidence

Finalized completed trials contribute separate statistics:

```text
At = A0 + rho_M * sum_i w_i x(theta_i) x(theta_i)^T
bt = b0 + rho_M * sum_i w_i x(theta_i) r_i
T4_candidate = solve(At, bt)
```

Requirements:

- evidence is accumulated in complete trial blocks;
- every row binds finalized trial ID, chronology, rate digest, direction digest,
  confidence, group identity, and valid-unit mask;
- pseudo evidence has a fixed maximum effective mass relative to labeled
  support;
- a rejected block never changes the accepted carrier statistics;
- a committed block is still recomputed from `A0/b0` plus the accepted
  evidence bank, not from a previous pseudo-updated carrier;
- arithmetic is fixed and receipt-bound; no silent FP32/FP64 substitution is
  allowed.

### 4.3 Trust-region projection

The first successor uses a support **design-precision geometry**, not an
unstated sandwich covariance and not a target-fitted posterior covariance. Let:

```text
delta = T4_candidate - T4_support
P_support = Xs^T Xs + lambda I
D2 = delta^T P_support delta
```

The active carrier is:

```text
T4_active = T4_support + alpha_M * project(delta, D2 <= c_M)
```

where:

- `P_support` is derived only from the frozen labeled-support design;
- `c_M`, `alpha_M`, block size, and maximum pseudo mass are selected on source
  grouped-OOF data only;
- target labels do not select them;
- the candidate set for `alpha_M` begins with `{0, 0.125, 0.25, 0.5}`;
- M30 uses `alpha_M = 0` in the deployable route unless a separate preregistered
  successor is authorized after a positive oracle M30 result.

`c_M` is a source-calibrated geometric trust threshold. It must not be described
as a nominal 95% credible region. A later Bayesian or sandwich-covariance
successor would need to name its exact variance estimator, aggregation law, and
coverage interpretation in a separate work order.

This rule makes support design precision a bound on movement. It does not treat
precision as evidence that a proposal is correct.

### 4.4 Why block refit differs from the failed update

The failed system recursively promoted pseudo-updated carrier state. A biased
pseudo direction could therefore influence the next prediction and then label
the next update.

The proposed system:

- keeps a permanent labeled-support anchor;
- caps the total pseudo-data influence;
- commits only after a direction-coverage block is complete;
- can remove a rejected block without reconstructing a hidden recursive state;
- separates proposal quality from support confidence;
- preserves exact activity-state commits when carrier proposals are rejected.

## 5. Deployable direction measurement

This section is implemented only after a positive O2 oracle gate.

### 5.1 Cross-fitted complementary-group direction

Direct self-labeling is forbidden. When updating units in complementary group
`g`, the pseudo direction must be derived only from predictions made without
group `g`:

```text
theta_hat_minus_g = circular_mean(theta_hat_h for h != g)
dispersion_minus_g = circular_dispersion(theta_hat_h for h != g)
```

The activity rates of group `g` appear only on the carrier-regression response
side. They do not contribute to their own pseudo-direction label.

The group partition and unit order are frozen. Joint group/unit permutation
must preserve the reconstructed full carrier exactly. A same-group estimator is
retained only as a self-labeling diagnostic.

### 5.2 Completed-trial trajectory direction

Do not use a single bin's velocity direction. Wait for the trial to finish and
integrate the complete predicted trajectory:

```text
d_hat_g = sum_t v_hat_g[t] * dt
theta_hat_g = atan2(d_hat_g.y, d_hat_g.x)
```

For center-out data, any snap to a canonical direction must use one frozen,
source-selected rule and report the unsnapped angle, snap margin, and selected
direction. The update still becomes available only after the trial is complete.

This is the valid role for an output-continuity prior:

> continuity aggregates noisy per-bin predictions into one trial-level carrier
> measurement; it does not directly smooth the governing velocity output.

### 5.3 Confidence evidence

The direction confidence may use only predeclared functions of:

- complementary-group circular dispersion;
- agreement between early and late completed-trial displacement;
- final displacement magnitude;
- distance and margin to the nearest canonical direction;
- deterministic trajectory-coherence statistics.

It must not use query behavior labels, target R2, future trials, or carrier
outcomes. A learned confidence model is not authorized until a fixed low-capacity
version beats the required controls.

## 6. Three-factor commit gate

A carrier proposal may commit only if all three independent gates pass.

### 6.1 Measurement-confidence gate

Question: is the pseudo direction internally reliable?

Evidence:

- cross-group dispersion below a source-frozen threshold;
- trajectory-coherence pass;
- nondegenerate displacement;
- canonical-direction margin if snapping is used.

### 6.2 Direction-coverage gate

Question: does this block add identifiable carrier information?

Evidence:

- increase in `logdet(X^T X + lambda I)`;
- nondecreasing minimum eigenvalue;
- fixed maximum repetition per canonical direction;
- required number of distinct directions in the block;
- no invalid direction or rate row silently dropped.

A repeated high-confidence direction is not enough to update a full cosine
tuning carrier.

### 6.3 Support-trust-region gate

Question: is the proposed displacement plausible under labeled support?

Evidence:

- support-only covariance or precision digest;
- proposal design-precision distance;
- projected and unprojected delta norms;
- maximum per-unit and aggregate displacement;
- familywise/session-level aggregation fixed before target scoring.

For the first successor, this evidence is the source-calibrated design-precision
distance from section 4.3. The gate must not call it posterior coverage and must
not rely on a per-unit `0.95` maximum that changes its effective false-rejection
rate with unit count.

### 6.4 Required precision ordering

The accepted carrier movement should shrink as labeled support becomes more
precise. The required diagnostic is:

```text
median carrier movement at M30
    <= median carrier movement at M10
    <= median carrier movement at M4
```

M30 remains an exact no-op in the first deployable route, so its movement is
zero by construction.

## 7. Experimental program

### 7.1 Stage O: oracle headroom

Run O0/O1/O2 exactly once after independent review. This stage requires no new
decoder training.

Primary output:

- paired equal-session R2 deltas versus activity-only;
- 15 external and 6 within per-session deltas;
- trial-by-trial carrier movement and state chronology;
- old incremental versus support-anchored refit attribution;
- M4/M10/M30 budget separation.

If O2 is STOP, end all later stages. If O2 is HOLD, run only the one permitted
source-selected sensitivity check and then issue a new GO/STOP decision.

### 7.2 Stage P: deployable pseudo-direction

Only a positive O2 authorizes this matrix:

| Cell | Direction measurement | Carrier update |
|---|---|---|
| P0 | none | activity-only, frozen T4 |
| P1 | cross-group circular ensemble | anchored block refit |
| P2 | trajectory-smoothed cross-group ensemble | anchored block refit |
| P3 | P2 direction with constant confidence | anchored block refit |
| P4 | P2 direction with deterministic group/confidence shuffle | anchored block refit |
| P5 | same-group pseudo direction | self-labeling diagnostic only |

P2 versus P1 tests the incremental value of trajectory continuity. P2 versus
P3/P4 tests whether the confidence mapping contains semantic information.

### 7.3 Stage G: bounded carrier state

Only a positive P1 or P2 target score authorizes broader online-state work:

- source-frozen precision/coverage thresholds;
- delayed block commits;
- optional change-point-triggered carrier proposals;
- optional posterior-residual shrinkage inside the same support trust region.

Do not train a learned gate before a deterministic P1/P2 system has a positive
incremental result.

### 7.4 Deferred ideas

The following ideas are lower priority and require a positive deterministic
carrier result first:

- learned gate;
- particle or ensemble carrier state;
- low-rank learned carrier-update directions;
- carrier change-point detector;
- joint decoder/carrier training;
- M30 online carrier adaptation.

## 8. Promotion, attribution, and stopping rules

### 8.1 Deployable carrier promotion gate

A deployable continuous-T4 cell must satisfy all of:

```text
candidate - activity_only >= +0.01 equal-session external R2
positive external sessions >= 10/15
other low budget >= activity_only - 0.01
within every budget >= activity_only - 0.02
M30 carrier state is exact no-op
target optimizer/backward/model/normalizer updates = 0
```

It must also beat:

- constant confidence;
- deterministic confidence/group shuffle;
- the same support anchor with no pseudo evidence.

Beating sealed static but losing to activity-only CDM is a failed carrier result.

### 8.2 Continuity attribution

Claim that trajectory continuity helps only if:

```text
P2 - P1 >= +0.01 external R2
positive sessions >= 10/15
P2 beats constant and shuffle controls
```

If P1 is positive and P2 is null, the carrier method may survive, but the
continuity-specific claim is closed.

### 8.3 Confidence attribution

Claim that uncertainty or agreement is useful only if the correct binding beats
both constant confidence and a deterministic row/group shuffle on the same
prediction records. A lower update count is not itself a positive result.

### 8.4 Stop conditions

Stop the continuous-T4 line if any of the following holds:

- O2 receives a STOP decision after the initial run or the one permitted HOLD
  sensitivity check;
- oracle refit improves only the current trial through a causality leak;
- P1/P2 fail to beat activity-only CDM;
- gains disappear under constant/shuffle controls;
- positive aggregate gain is carried by fewer than 10/15 external sessions;
- M4/M10 gains require target-selected thresholds;
- carrier updates require target gradients or decoder changes;
- repeated carrier proposals drift outside the support trust region.

## 9. Required receipt evidence

Every scored session must record:

- support trial IDs, directions, native-rate digest, and frozen T4 digest;
- query trial IDs and exact chronological order;
- state-before and state-after digest for every completed trial;
- activity-state transition independently from carrier-state transition;
- direction estimator type, group exclusion, raw angle, snapped angle, and
  confidence evidence;
- frozen support-statistics digest;
- proposed block-statistics digest;
- direction-coverage evidence;
- unprojected and projected carrier delta;
- proposal, rejection, and commit reason;
- cumulative accepted pseudo mass;
- carrier movement by budget and session;
- exact decoder state before/after;
- exact target optimizer, backward, parameter-update, and normalizer-update
  counts, all zero;
- `target_label_leakage=true` for oracle cells and false for deployable cells.

The attempt must publish before checkpoint, target data, source data, model,
CUDA, or evaluation input access. Score and terminal publication remain atomic;
any failure terminalizes with honest access and progress flags.

## 10. Implementation boundaries

The first implementation should reuse the reviewed activity-only scorer,
completed-trial state machine, parser, sealed decoder, metric, and lifecycle.
It should add only:

1. a route-owned oracle/pseudo direction provider;
2. a support-sufficient-statistic carrier bank;
3. a block refit and trust-region projection;
4. typed three-factor gate evidence;
5. a new score codec for O0/O1/O2 and later P0--P5.

Do not modify the accepted activity-only result or reinterpret full CDM V8 as a
positive carrier baseline. Do not copy a scorer or bypass held-descriptor
validation. Historical immutable receipts remain evidence; a successor binds
current implementation bytes separately.

The first stage is inference/scoring only. It does not require new network
training or a GPU training job. GPU use, if required for scoring speed, is a
separate launch decision after code and root review.

## 11. Strongest objection and response

### Objection

Complementary groups can share the same decoder bias. Low group disagreement
therefore does not imply a correct direction. A cross-fitted update may still
reinforce common-mode error.

### Response

This objection is valid. The design does not treat agreement as ground truth.
It requires:

1. a positive true-direction causal oracle result first;
2. a permanent labeled-support anchor;
3. group exclusion to remove direct self-labeling;
4. trajectory and coverage evidence in addition to agreement;
5. constant and shuffle controls;
6. an incremental result over activity-only CDM.

If oracle O2 is positive but P1/P2 fail, the correct conclusion is:

> continuous carrier adaptation has headroom, but the current deployable
> direction measurement is inadequate.

That is a useful boundary result. It does not authorize a learned gate by
itself.

## 12. Two-sentence method pitch

> Support-Anchored Causal T4 Memory adapts functional tuning from finalized
> unlabeled trials while preserving the original labeled-support carrier as an
> immutable statistical anchor. Cross-fitted trial-direction measurements may
> contribute only bounded block-level evidence that improves direction coverage
> and remains inside a support-derived trust region.

## 13. Recommended execution order

1. Implement O0/O1/O2 as a score-only successor over the accepted activity-only
   runtime.
2. Run no-data synthetic tests for causality, anchor reconstruction, coverage,
   trust-region projection, state independence, and failure lifecycle.
3. Independently audit current bytes, immutable predecessor evidence, input
   authority, and fresh result roots.
4. Run O0/O1/O2 once on M4/M10/M30 within-6 and external-15.
5. If O2 is HOLD, run only the one permitted source-selected sensitivity check;
   then issue GO or STOP.
6. Stop on O2 STOP. On O2 GO, implement P0--P5 with fixed source-selected
   hyperparameters.
7. Promote a carrier state only if it beats activity-only CDM and all semantic
   controls.
8. Keep activity-only Causal Calibration Memory as the validated second
   contribution unless and until step 7 succeeds.

This sequence protects the strong activity-memory result while giving
continuous T4 adaptation one decisive, technically credible opportunity.
