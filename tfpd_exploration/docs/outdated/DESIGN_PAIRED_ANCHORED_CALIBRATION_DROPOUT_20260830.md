# PACD: Paired Anchored Calibration Dropout

Date: 2026-08-30

Status: bounded production-path source smoke passed; the matched P0 full/full
control is actively training under the V3 evidence-policy successor. P1/P2
full training and all matched R2 evaluation remain gated and unmeasured.

Primary scope: DANDI 000688 sub-C/sub-M center-out, sealed Cell-D producer

Primary budgets: M4 and M10

Safety budget: M30

## Current execution boundary and naming

PACD `P0`, `P1`, and `P2` are source-training arms. They are unrelated to the
Stage-P carrier-memory `P0/P1/P2` labels in
`support_anchored_t4_stage_p_v1`.

The distinction is:

| Family | What changes | Current evidence |
|---|---|---|
| Stage-P carrier P1 | causal deployment-time T4 carrier state | completed positive M4 result on DANDI 000688 |
| PACD P0 | matched M30/M30 two-branch training control | active full training; no R2 result |
| PACD P1 | M30 anchor + M4 calibration-activity dropout during source training | forbidden until PACD P0 terminal |
| PACD P2 | M30 anchor + M10 calibration-activity dropout during source training | forbidden until PACD P0 terminal |

The Stage-P carrier result neither validates nor invalidates PACD. It changes
the later factorial question: if a PACD static checkpoint passes, it should be
combined first with activity-only CCM as originally planned. A combination
with the Stage-P carrier is a separate, conditional factor and must not be
assumed additive, because the completed C1-by-carrier factorial already showed
that independently positive calibration mechanisms can interact negatively.

The active PACD P0 run is an infrastructure and causal-control cell, not a
performance candidate. P1/P2 remain mandatory after a valid P0 terminal; P0
loss or throughput cannot select either one out. A post-P0 admission/device
successor must descriptor-validate the complete P0 graph before it can issue
either treatment capability. Until P0 terminalizes, implementation files in
its execution closure must not be edited.

The read-only shared-lifecycle seam audit is frozen in
`AUDIT_PACD_V4_SHARED_LIFECYCLE_SEAM_20260831.md`. It requires a staged V4
runtime path so CUDA binding is recorded before execution-stack loading, while
the legacy V1/V2/V3 device and receipt behavior remains unchanged.

## 0. Recommendation

Build one method family around a single new idea:

> For every source-training query, train the same decoder on a full M30
> calibration-activity view and on a paired short M4 or M10 view. Supervise
> both views with the same task target. The full view is a permanent anchor;
> the short view is structured calibration-trial dropout.

The method name is **PACD: Paired Anchored Calibration Dropout**.

The first PACD version must be simple:

- no new network layer;
- no posterior carrier;
- no uncertainty feature;
- no learned gate;
- no prediction-consistency loss;
- no target update;
- no change to the sealed Cell-D inference graph;
- no D-optimal support selection in the first causal test;
- no CDM state transition during training.

PACD changes only the source-training calibration view and the loss exposure.
At deployment it uses one ordinary forward, so it adds no inference parameter,
latency, state, or label cost.

## 1. Two-sentence pitch

Short calibration prefixes are not merely smaller inputs: a decoder trained
only on full calibration can treat M4 or M10 identities as an unfamiliar
session domain. PACD trains every short view beside its own full M30 anchor,
so the model learns low-budget robustness without being allowed to forget the
full-budget solution.

## 2. Why this experiment is justified now

### 2.1 CAL-AUG C1 found a real but incomplete signal

The completed chronological prefix-cycle experiment changed only the B3S
calibration-activity prefix. Relative to its matched T0 control, C1 produced:

| Readout | C1 minus T0 |
|---|---:|
| source prefix-robustness recovery at M4 | +0.246854 |
| source prefix-robustness recovery at M10 | +0.063518 |
| external M4 R2 | +0.026478, 10/15 positive |
| external M10 R2 | +0.034960, 10/15 positive |
| external M30 R2 | -0.022008, 6/15 positive |

This is not a null. It shows that exposing the B3S identity encoder to short
activity prefixes can improve low-budget transfer. But the same one-view
training schedule damaged M30 enough to fail its continuation rule.

### 2.2 Unanchored budget marginalization already failed

CBM-D exposed source training to calibration budgets from M4 through M30, one
budget at a time. It learned strong held-in M4 robustness but did not transfer:

| Surface | M4 | M10 | M30 |
|---|---:|---:|---:|
| within, ordinary total calibration | +0.1121 | +0.0216 | -0.0171 |
| external, ordinary total calibration | -0.0104 | -0.0706 | -0.1195 |

CBM-D therefore rules out the claim that more unstructured budget exposure is
enough. PACD is not another budget schedule. Its load-bearing change is that
the short view is always optimized beside the same query's full view.

### 2.3 The uncertainty-feature neighborhood is not the first move

PIRG learned a posterior-credibility gate amplitude near zero and changed R2
only at approximately 1e-4 scale. C3-Real later supplied angular reliability
as an explicit fifth side feature, but the correctly aligned feature did not
beat its same-checkpoint row-shuffle control. Those results do not test B3S
activity-prefix uncertainty exactly, but they make a generic learned gate a
poor first experiment.

### 2.4 Strong consistency is specifically contraindicated

Cell C trained two whole-unit-subset views with task loss on both views plus a
prediction-consistency loss. It lost 0.0357 external R2 to Cell D, with only
4/15 sessions positive and a confidence interval excluding zero.

PACD v1 therefore does **not** require the full and short predictions to be
equal. Each branch is simply required to solve the supervised task. The full
branch anchors the optimization distribution; it is not a teacher target.

## 3. What is new

The following operator has not been executed in this project:

```text
same query x, same target y, same session s, same ordinary M30 T4

C_full  = first 30 calibration-activity trials
C_short = selected M4 or M10 subset of C_full

y_full  = f_theta(x, C_full,  T4_M30)
y_short = f_theta(x, C_short, T4_M30)

L_PACD = 0.5 * MSE(y_full,  y)
       + 0.5 * MSE(y_short, y)
```

The distinction from earlier routes is exact:

| Earlier route | What it did | What PACD adds |
|---|---|---|
| CAL-AUG C1 | one M30, M10, or M4 view per forward | simultaneous M30 anchor for every short view |
| CBM-D | one matched budget per batch | paired full and short views of the same query |
| PIRG/C3 | consumed a reliability estimate | no reliability input in v1 |
| Cell C | forced two predictions to agree | no agreement penalty |
| D-opt protocol | selected informative M4 trials at deployment | not used until a conditional PACD-M4 successor |
| CDM | updated causal deployment memory | no online state in the core PACD test |

## 4. Method definition

### 4.1 Name and terminology

**PACD** means **Paired Anchored Calibration Dropout**.

- **Paired:** full and short calibration views use the same query and target.
- **Anchored:** the M30 view is present in every optimizer step.
- **Calibration Dropout:** whole calibration trials are omitted from the B3S
  identity input. Query bins and behavior targets are not dropped.

PACD calibration dropout is distinct from Cell D's existing dynamic
whole-unit dropout. PACD drops calibration trials; Cell D drops neural units.
The Cell-D whole-unit dropout law remains active and unchanged.

### 4.2 Frozen producer

The starting producer is sealed Cell D:

- 3,510,842 live parameters;
- B3S calibration activity;
- four-column ordinary OLS T4 side features;
- two attention heads;
- dynamic whole-unit dropout during source training;
- batch size 32 unique query samples;
- Adam, learning-rate schedule, source roster, batch order, valid-bin loss,
  epoch count, checkpoint rules, and final-four SWA inherited unchanged;
- no target optimizer, backward, update, refit, or checkpoint selection.

PACD v1 adds no trainable parameter and changes no tensor shape.

### 4.3 Training views

For every source batch, construct two calibration-activity views:

```text
anchor view: C30 = ordered first 30 authorized calibration trials
short view:  CM = P_M(C30), M in {4, 10}
```

The query neural tensor, behavior target, session identity, T4 tensor, valid
mask, and batch indices are identical across the two branches.

The short-view law is predeclared per arm:

- `PACD-M4`: chronological positions `[0,4)`;
- `PACD-M10`: chronological positions `[0,10)`;
- `PACD-Joint`: M alternates deterministically between 4 and 10 while every
  step still contains the M30 anchor.

`PACD-Joint` is conditional. It is not part of the first causal comparison.

### 4.4 Carrier law

During source training, both branches use the same ordinary M30 T4 carrier and
the same frozen source normalizer. This isolates calibration-activity dropout.

The short branch must not silently receive:

- posterior-mean T4;
- short-prefix T4;
- a refitted T4 normalizer;
- a reliability column;
- target-derived T4;
- CDM-updated T4;
- a different unit order.

Deployment scoring later reports two surfaces:

1. **Activity-isolation surface:** B3S=M and ordinary T4=M30.
2. **Honest total-calibration surface:** B3S=M and the frozen deployment T4
   recipe for M, using D-opt plus fixed ridge 0.1 at M4 and chronological plus
   fixed ridge 0.1 at M10/M30.

The first surface tests the PACD mechanism. The second tests practical value.
Neither may be relabeled as the other.

### 4.5 Loss

The v1 loss is fixed:

```text
L = 0.5 * L_task(anchor)
  + 0.5 * L_task(short)
```

Both task losses use the same dense valid-bin supervised MSE as Cell D. The
weights are fixed at 0.5/0.5 before any result.

Forbidden in v1:

- `MSE(y_full, y_short)`;
- KL or R-Drop consistency;
- stop-gradient teacher loss;
- contrastive loss;
- identity-embedding alignment;
- learned branch weighting;
- budget-conditioned weighting;
- target-label tuning.

These exclusions are scientific, not merely engineering choices. They keep
the first result attributable to paired supervised exposure.

### 4.6 Coupling Cell-D whole-unit dropout across the pair

The two PACD branches must use the same realized Cell-D whole-unit dropout
probability and the same per-example unit keep mask. This preserves Cell D's
marginal dropout law while making calibration-view length the only difference
inside a pair.

Required evidence per audited step:

- identical dropout probability;
- identical unit-mask digest;
- identical kept-unit count per paired example;
- different calibration-view digest for PACD-M4/M10;
- identical calibration-view digest for the full/full control;
- no dropout at evaluation.

If exact mask coupling cannot be implemented without changing the sealed
dropout semantics, stop and redesign the seam. Do not substitute independent
masks and call the result a calibration-only intervention.

### 4.7 Execution form

The safe memory form is sequential paired backpropagation:

```text
optimizer.zero_grad()
backward(0.5 * loss_anchor)
free anchor graph
backward(0.5 * loss_short)
optimizer.step()
```

There is one optimizer step, one unique query batch, and two calibration
views. The branch order is fixed as anchor then short and recorded.

No claim of compute matching is allowed without the full/full paired control.

## 5. Experimental family

### 5.1 Mandatory cells

| Cell | Anchor branch | Second branch | Purpose |
|---|---|---|---|
| `P0-FullFull` | M30 chronological | M30 chronological | matched two-branch/compute/RNG control |
| `P1-PACD-M4` | M30 chronological | M4 chronological | primary short-budget candidate |
| `P2-PACD-M10` | M30 chronological | M10 chronological | companion short-budget candidate |

All three cells use:

- seed 42;
- the same source sessions and source batch order;
- the same unique query examples per optimizer step;
- the same number of optimizer steps;
- the same paired dropout-mask stream;
- the same initialization;
- the same checkpoint/SWA rule;
- the same scorer and records.

Historical T0 and C1 remain required comparators, but `P0-FullFull` is the
causal control for PACD because it matches the new two-branch computation.

### 5.2 Conditional cells

| Cell | Condition to build | Question |
|---|---|---|
| `P3-PACD-Joint` | P1 and/or P2 gives positive external signal with M30 safety | can one checkpoint cover M4 and M10? |
| `P4-PACD-M4-Dopt` | P1 proves the paired mechanism or shows source-positive/external-weak behavior | does matching the known M4 deployment selection law improve transfer? |
| `P5-PACD-ActivityUncertainty` | a PACD arm is positive but effects track support instability | does activity-only support count/dispersion improve the paired model? |
| `P6-PACD-plus-Activity-CDM` | best static PACD arm passes | are training robustness and causal activity memory additive? |

These cells are not pre-authorized by this document. Each needs its own frozen
work order after its condition is observed.

### 5.3 Explicitly excluded cells

Do not build as part of this family:

- PACD plus prediction consistency;
- PACD plus posterior-mean T4;
- PACD plus angular-reliability side feature;
- PACD plus learned gate in the first round;
- M10 D-opt support dropout;
- complementary-group trajectory contrastive training;
- new recurrent/stateful backbone;
- target-selected branch weights;
- target backpropagation.

## 6. Staged execution plan

### Stage 0: code and synthetic contract

No data and no CUDA are needed.

Required tests:

1. M30/M30 control produces equal calibration tensors.
2. M30/M4 and M30/M10 use exact nested authorized rows.
3. Query, target, T4, unit order, and valid mask are identical across branches.
4. The paired dropout probability and unit mask are identical.
5. The 0.5/0.5 loss gives gradients to every existing Cell-D critical group.
6. No new trainable parameter exists.
7. Evaluation executes one ordinary no-dropout branch.
8. Branch order cannot change the declared receipt identity.
9. A future or non-support calibration row is rejected.
10. An M4 D-opt row cannot enter the chronological v1 arm.
11. Posterior/reliability/CDM inputs are structurally unreachable.
12. The dry CLI cannot open data, load a checkpoint tensor, initialize CUDA,
    reserve a result root, or launch.

### Stage 1: bounded source smoke

Run `P0`, `P1`, and `P2` for the same fixed short source schedule. The smoke is
only a feasibility and invariance gate; it does not select a scientific arm.

Required smoke evidence:

- one attempt before source or checkpoint access;
- exact producer and source authority;
- paired-view shapes and digests;
- paired dropout-mask equality;
- both branch losses finite;
- all critical gradient groups nonzero and finite;
- one optimizer update per scheduled batch;
- model and optimizer finite;
- no target data or target update;
- measured steps/s, synchronized wall time, and peak memory;
- projected full-run duration for the actual GPU profile.

All three smokes must be allowed to pass if they are technically sound. Do not
choose P1 or P2 from a noisy smoke loss.

### Stage 2: seed-42 full training

Train the mandatory cells under identical exposure:

```text
P0-FullFull
P1-PACD-M4
P2-PACD-M10
```

The route retains the sealed 48-epoch, 33,925-step-per-epoch source schedule
unless an independent implementation audit proves that the paired runner must
use a different matched count. Any count change applies to all three cells and
must be frozen before the first full attempt.

The expected paired-run cost is approximately twice the forward/backward cost
of a single-view Cell-D run. The Stage-1 measured throughput, not a hand
estimate, must bind the wall-clock projection. Free GPU memory is not authority
to enlarge the scientific batch size; batching may be optimized only after an
exact prediction/gradient equivalence test and must apply identically to P0,
P1, and P2.

### Stage 3: source-only mechanism evaluation

Use immutable completed checkpoints and identical source evaluation records.
No source result may be selected after looking at external sessions.

For checkpoint `X`, define:

```text
prefix_degradation_X(M)
  = R2_X(B3S=M, T4=M30) - R2_X(B3S=M30, T4=M30)

paired_anchor_recovery_X(M)
  = prefix_degradation_X(M) - prefix_degradation_P0(M)
```

Report for all 27 source sessions:

- M30, M10, and M4 R2;
- paired anchor recovery;
- positive-session counts;
- anchor/short task-loss difference;
- anchor/short prediction difference;
- anchor/short identity difference;
- full-branch gradient norm, short-branch gradient norm, and their cosine;
- exact checkpoint and prediction digests.

The gradient and identity diagnostics are descriptive. They cannot select a
checkpoint or relax a performance gate.

### Stage 4: one matched development score

Score P0, P1, P2, historical T0, historical C1, and sealed Cell D on identical
within-6 and external-15 records.

For each system, report:

- activity-isolation M4/M10/M30;
- honest total-calibration M4/M10/M30;
- variance-weighted two-coordinate last-bin R2 per session;
- equal-session mean and median;
- paired delta for every session;
- positive-session count;
- fixed-seed bootstrap interval;
- worst-session R2 and worst-session paired delta;
- no-target-update proof;
- model-state-before/after equality.

The external records are opened once after source artifacts and arm identities
are fixed. They cannot select a checkpoint, loss weight, support law, branch
weight, or retry.

### Stage 5: conditional replication

Seed 43 and seed 44 are authorized only after one seed-42 PACD arm satisfies
both the performance and M30 safety conditions below. Replication uses the
identical arm; do not choose different budgets or support laws by seed.

## 7. Predeclared decision rules

### 7.1 Infrastructure validity

`P0-FullFull` must be a scientifically harmless matched control:

- no new parameter or inference path;
- full/full branch predictions bitwise equal under the coupled mask;
- source training finite;
- external M30 no worse than historical matched T0 by more than 0.01 R2;
- no target update;
- identical scoring records and metric law.

If P0 fails, P1/P2 are not interpretable. Repair the paired runner under a new
attempt; do not score around the failed control.

### 7.2 PACD mechanism gate

For arm `PM` at its trained short budget M:

```text
paired_anchor_recovery_PM(M) >= +0.01 R2
positive source sessions >= 18/27
source M30 delta versus P0 >= -0.01 R2
```

This is source-only mechanism evidence, not a deployment claim.

### 7.3 Primary development performance gate

A PACD arm passes its primary performance gate only if all are true:

```text
external total-calibration delta versus P0 at trained M >= +0.03 R2
positive external sessions >= 10/15
external M30 delta versus P0 >= -0.01 R2
within M30 delta versus P0 >= -0.01 R2
worst external session delta versus P0 >= -0.05 R2
```

M4 is the primary budget for P1. M10 is the primary budget for P2. A gain at
the other budget is secondary and cannot rescue failure at the trained budget.

### 7.4 Incremental claim over CAL-AUG C1

PACD earns the stronger claim that pairing is better than unpaired prefix
exposure only if:

```text
external delta versus C1 at trained M >= +0.01 R2
positive external sessions versus C1 >= 9/15
external M30 delta versus C1 >= +0.01 R2
```

The M30 term is important: PACD is designed to solve C1's full-budget harm.
If PACD merely matches C1 at M4/M10 and still loses M30, the anchor did not do
its intended job.

### 7.5 Interpretation table

| Outcome | Interpretation | Next action |
|---|---|---|
| low-budget gain, M30 safe, beats C1 | paired anchoring works | replicate exact arm, then consider Joint |
| low-budget gain, M30 safe, does not beat C1 | useful budget exposure, pairing claim unsupported | stop new pairing ablations; report equivalence |
| source gain, external null/negative | session fingerprint learning | stop PACD family |
| low-budget gain, M30 harm | anchor failed to protect the full solution | stop; do not rescue with gates |
| P1 positive only | M4-specific calibration-dropout effect | replicate P1; consider D-opt successor |
| P2 positive only | M10-specific effect | replicate P2; do not build M4-Dopt |
| P1 and P2 positive | broad short-budget effect | build one predeclared Joint successor |
| both null | paired calibration dropout is not the missing mechanism | close PACD; do not append B/C/D |

## 8. Conditional successors

### 8.1 PACD-Joint

If P1 and P2 are both safe and at least one passes, alternate the short branch
between M4 and M10 while retaining the M30 anchor on every step. The schedule
must be deterministic and session-balanced.

PACD-Joint answers a deployment question. It is not needed to establish the
basic paired-anchor mechanism.

### 8.2 PACD-M4-Dopt

This is the former candidate C, narrowed to its defensible role.

It changes only the P1 short-view selection law:

```text
anchor: chronological M30
short:  causal first-30 D-opt-selected M4
```

The source direction labels used by D-opt are allowed source supervision and
must be counted. The target rule must use only the already authorized causal
cue metadata. No later-session or query label may enter selection.

Do not run M10-Dopt: existing protocol evidence says D-opt is useful at M4 and
should be turned off by M10.

### 8.3 PACD-ActivityUncertainty

This is the only retained part of the former candidate B. It is allowed only
if a positive PACD arm still has a support-quality-dependent residual.

Permitted features are activity-only and source-normalized:

- `log(M)`;
- leave-one-calibration-trial-out B3S identity dispersion;
- an activity-only effective sample size or standard error;
- an explicit missing/invalid indicator if needed.

Required controls:

- constant feature;
- within-session row shuffle;
- same parameter count and initialization;
- source-only selection;
- exact M30 safety.

Posterior angular reliability, posterior-mean T4, and a generic free MLP gate
are excluded because they overlap completed negative or unidentified routes.

### 8.4 PACD plus activity-only CDM

If a static PACD checkpoint passes, compose it with the already defined
activity-only CDM rollout without retraining PACD. This tests whether source
calibration robustness and target-time causal activity accumulation are
additive.

The comparator must include:

- P0 static;
- best PACD static;
- P0 plus activity-only CDM;
- best PACD plus activity-only CDM.

This is a 2x2 factorization, not a new blended method claim. Carrier updates
remain off unless separately justified.

## 9. Required receipts and integrity

Every executable successor must bind:

- exact work-order and implementation closure;
- sealed Cell-D source terminal, SWA, and normalizer;
- historical T0/C1 mechanism and deployment terminals;
- exact source roster and batch-order authority;
- paired calibration-row identities;
- paired dropout probability and mask digests;
- branch loss weights and branch order;
- optimizer-step and unique-query counts;
- source-only checkpoint-selection rule;
- final-four checkpoint and SWA graph;
- one attempt before data/checkpoint/CUDA access;
- atomic terminal or honest failure;
- target optimizer/backward/update counts equal zero.

Execution-critical source bytes and review-only documents should be separated.
A review-comment or documentation edit that cannot affect numerical execution
must be recorded as review drift and must not force a live scientific job to
restart. A change to the paired operator, loss, model, data records, metric,
checkpoint selection, RNG law, or scorer is execution drift and must fail
closed.

## 10. Resource and scheduling policy

PACD is a two-forward training family. Do not hide its cost.

- Stage 0 is CPU/synthetic.
- Stage 1 measures actual GPU throughput and memory.
- P0/P1/P2 may use separate idle GPUs only after their own capabilities and
  roots are independently valid.
- No process may preempt or alter an active unrelated job.
- Increasing batch size is engineering-only until prediction, gradient,
  optimizer-step, and selected-checkpoint equivalence are demonstrated.
- P0, P1, and P2 must use the same logical batch and update count even if their
  physical microbatching differs.
- Seeds 43/44 are conditional replication, not parallel speculative jobs.

Suggested two-week pilot if implementation review is positive:

| Time | Work |
|---|---|
| Days 1-2 | route-owned paired operator, coupled-mask seam, synthetic tests |
| Day 3 | no-data audit and three bounded source smokes |
| Days 4-7 | P0 and one priority PACD arm on independent idle GPUs |
| Days 7-10 | remaining mandatory PACD arm; immutable source terminals |
| Days 10-11 | source mechanism score, no external selection |
| Day 12 | one common within/external matched score |
| Days 13-14 | interpretation and exact replication decision |

The order of P1 and P2 can follow resource availability, but neither smoke may
select the other out. If only one treatment GPU is available after P0, P2 has
the slightly stronger prior because C1's external M10 gain was +0.0350 with a
positive bootstrap lower bound, while M4's +0.0265 interval crossed zero. M4
still has larger headroom and remains mandatory unless a hard safety failure
stops the entire family.

## 11. Strongest objections

### Objection 1: this is just CAL-AUG C1 with twice the compute

No. C1 sees one budget on a step. PACD sees the full and short views of the
same query on every step. P0-FullFull controls for the two-branch compute and
gradient accumulation. PACD must also beat C1 incrementally to claim that the
pairing itself matters.

### Objection 2: CBM-D already proved budget marginalization fails

CBM-D proves that unanchored, one-budget-at-a-time exposure does not transfer.
That failure is the motivation for the permanent M30 anchor. If PACD also
fails, the correct conclusion is to close this family, not to invent another
budget schedule.

### Objection 3: Cell C proved paired views are harmful

Cell C added an explicit equality constraint across different unit subsets.
PACD adds no equality constraint. It gives each calibration view its own task
loss, allowing the model to use view-specific evidence while retaining a full
view in every update.

### Objection 4: the full branch may dominate the gradient

The 0.5/0.5 weights are fixed, and branch gradient norms/cosines are reported.
Do not learn or tune branch weights in v1. If the short gradient is consistently
negligible, PACD has failed as specified.

### Objection 5: this still may learn source fingerprints

Yes. The source mechanism gate cannot establish transfer. The external-15
paired result is required, and a source-positive/external-null outcome closes
the family.

### Objection 6: D-opt or uncertainty may be the real improvement

That is why neither is in v1. Chronological P1/P2 establish the paired-anchor
effect first. D-opt and activity uncertainty are conditional, separately
identified successors.

## 12. Paper claim if successful

The strongest defensible claim is:

> Full-view anchoring turns calibration-trial dropout from unanchored domain
> randomization into safe low-budget robustness. A single decoder improves M4
> or M10 transfer while preserving its M30 solution, without target gradients,
> new inference parameters, or online state.

The paper must not claim:

- universal invariance across datasets;
- posterior uncertainty modeling;
- learned gating;
- target adaptation;
- improved T4 estimation;
- trajectory contrastive learning;
- zero-cost training.

Cross-dataset validation comes only after a positive DANDI result. The first
replication target should expose the same trialized calibration-activity
contract. M1/H1 require their own loader and calibration-headroom audit and are
not silently included in the PACD claim.

## 13. Paper claim if null

A null is still decisive:

> Short-prefix CAL-AUG gains do not arise because the one-view schedule lacks
> a full-budget anchor. Paired full/short supervision does not improve transfer
> beyond matched two-view training, so further budget-dropout, uncertainty-gate,
> D-opt-training, and cross-group-consistency combinations are not justified.

This is the explicit stopping node. Do not turn a null PACD result into an
open-ended B/C/D ablation program.

## 14. Naming decision

Names considered during convergence:

| Name | Decision |
|---|---|
| Full-View Anchored Calibration Dropout | accurate but long |
| Anchored Calibration Trial Dropout | misses the paired same-query fact |
| Paired Calibration-View Training | too generic; hides the dropout mechanism |
| Dual-View Calibration Marginalization | too close to the failed CBM-D terminology |
| Anchor-Paired Support Dropout | “support” can be confused with labeled T4 support |
| **Paired Anchored Calibration Dropout (PACD)** | selected: short, specific, and distinguishes the method from CBM-D |

Recommended cell names:

```text
P0-FullFull
P1-PACD-M4
P2-PACD-M10
P3-PACD-Joint          # conditional
P4-PACD-M4-Dopt        # conditional
P5-PACD-ActivityUncertainty  # conditional
P6-PACD-plus-Activity-CDM    # conditional
```

## 15. Independent-review questions

1. Does paired full/short supervision isolate a mechanism that CAL-AUG C1 and
   CBM-D did not test?
2. Is full/full paired P0 sufficient to control the doubled computation and
   paired gradient accumulation?
3. Must Cell-D whole-unit dropout be coupled exactly across the two branches,
   or is there a safer route-owned construction?
4. Is holding ordinary M30 T4 fixed during source training the cleanest way to
   isolate B3S calibration-activity dropout?
5. Are the M4/M10 performance gates large enough to exceed existing C1 noise
   and small enough to detect a useful effect?
6. Does the M30 safety rule directly test the proposed advantage over C1?
7. Is the conditional ordering correct: PACD first, D-opt second, activity
   uncertainty third, CDM composition last?
8. Is any completed experiment closer to the exact paired-anchor operator than
   the routes listed in Section 3?

## 16. Final decision

**GO for design and independent review. NO-GO for launch until a separate
implementation work order, no-data tests, source smoke, closure audit, device
authority, fresh roots, and explicit launch authorization exist.**

Among the previously discussed candidates, only PACD deserves a new primary
training family. Direction-balanced M4 selection is a conditional PACD
variant. Activity uncertainty is a conditional PACD feature. Cross-group
trajectory consistency is excluded by existing negative evidence.

## 17. Implementation update: PACD V2 source smoke

The bounded production-path smoke is complete. V2 ran P0 M30/M30, P1 M30/M4,
and P2 M30/M10 for eight source-only optimizer steps per arm on an isolated
GPU0. All paired dropout-mask, RNG, gradient, finite-state, exact P0
prediction/identity, sealed-predecessor, closure, no-target, and immutable
receipt gates passed. The concurrent Stage-P experiment continued on GPU1.

This update changes the engineering verdict from implementation NO-GO to
source-smoke PASS. It does not provide R2 or establish a performance gain.
Exact receipts, throughput, lineage, and the V1 fail-closed repair are recorded
in `RESULT_PACD_V2_SOURCE_SMOKE_20260831.md`.

The next decision is whether to authorize matched full source training. That
decision requires a separate work order; this design document and the bounded
smoke do not authorize it.
