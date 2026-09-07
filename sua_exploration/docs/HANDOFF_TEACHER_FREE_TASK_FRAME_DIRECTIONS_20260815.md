# HANDOFF: from teacher--student inheritance to task-frame population decoding

**Date:** 2026-08-15  
**Status:** research-direction handoff; authorizes no code change, data access, GPU launch, target scoring, or formal evaluation  
**Scope:** next-generation non-H1 method design, with SUA/sub-M and RT/M2 as the primary evidence-bearing domains  

> ## Outcome note (2026-08-17): the clean-sheet arms were scored, and the verdict is mixed
>
> This document set up the clean-sheet TFPD route. The arms it proposed have now run, and the parts
> of it that are load-bearing for scheduling need qualifying. Numbers are external native R² on
> 15 sub-M sessions unless stated (`tfpd_exploration/results/gate4_arm_external_v1/final_external_matrix_v2_receipt.json`).
>
> **The clean-sheet arms lost to the SPINT-shaped comparator.** `bl_t4` external **−0.9597**,
> `large_t4` **−0.0848**, against teacher-free `spintshape_t4` **+0.0902** — and under a 48-epoch
> warmup+cosine schedule with final-four SWA the same SPINT-shaped backbone reaches **+0.1610**
> (within 0.5163). Teacher-**initialized** A2 remains **0.341**, i.e. `+0.180` ahead externally,
> versus only `+0.059` within. So §0's decision to not let SPINT define the algebra has held up as
> *framing*, but as *engineering* the SPINT-shaped topology plus a longer schedule is currently the
> best teacher-free system, and source/teacher initialization is still the frontier. The gap is
> wider externally than within, so closing the within gap is not evidence of closing the
> deployment gap.
>
> **The transparent PV baseline is void, which removes this route's intended interpretability
> anchor.** `tfpd/population_vector.py` decomposes `[a, c] / max(m, eps)` on a **z-scored** carrier;
> 65.7% of units get `m_z` clamped to `1e-6`, median direction magnitude 3.24×10⁵. It cleared
> Stage-0 only because `tfpd/synth.py` emits raw-unit carriers, so G5 does not transfer to real
> data. Details in `tfpd_exploration/docs/TFPD_RESULTS_LEDGER_20260816.md` §4. Until it is fixed, no
> claim contrasting learned aggregation against a simple population vector can be made.
>
> **One prediction from this route was confirmed, and it is the best mechanism result available.**
> Carrier-derived identity transfers markedly better than calibration-activity-derived identity:
> `large_t4` vs `spintshape_z4` is `+0.058` within but `+0.209` external (activity-only identity is
> *negative* externally at −0.2936). Topology-confounded and measured across two estimators, so cite
> the direction and not the magnitude — but it supports this document's core claim that per-unit task
> correspondence from calibration-derived coordinates is the right invariance to build on, and it
> warns that adding capacity to the activity-identity encoder carries a session-fingerprint risk.
>
> **Scheduling authority has moved.** Ranked next steps, and the list of closed directions with
> receipts (hidden-space port, decoupled-K/V key replacement, SetKV, staged carrier admission,
> state-conditioned queries), are in
> `tfpd_exploration/docs/HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md` §5–§6.

## 0. Executive decision

The next method should not be framed as a student that learns a residual improvement over SPINT.
SPINT remains an important inspiration, initialization source, and strong comparator, but it should
not define the algebra, naming, or contribution boundary of the new system.

The central problem should be restated as:

> Given a small calibration support set, a variable and permuted population of recorded units, and
> a low-dimensional task-frame description for each unit, how should a complete decoder map that
> population into a shared behavior space so that it transfers across sessions and subjects?

The leading clean-sheet hypothesis is a **Task-Frame Population Decoder (TFPD)**.  It treats T4 as
a conditional read-in or observation coordinate, not as an additive NeuronID waveform.  The first
test should be an explicit low-rank bilinear activity--carrier read-in, not an unconstrained
hypernetwork.  If that premise survives, a constrained carrier hypernetwork can generate each
unit's contribution to a shared task latent; a permutation-invariant population aggregator and
shared temporal dynamics then predict behavior.  The complete model is trained end to end on
behavior, without teacher-output or teacher-identity matching.  Existing evidence motivates this
hypothesis but does not yet establish it as the strongest consumer.

## 1. Factual correction: the completed A2 system is not literally `SPINT + delta`

The present software uses teacher/student names, but three implementation facts constrain the
scientific interpretation.

1. `StreamingCalibrationLitModule.setup()` loads the SPINT checkpoint, creates an isomorphic
   decoder, and strict-copies the teacher state.  It also passes teacher identity layers into the
   streaming encoder builder.  This is inherited architecture and initialization.
2. The sealed A2 source launcher uses `loss_mode=task_only`.  During source training, the objective
   is behavior MSE; prediction and identity distillation are not active training losses.
3. A2 has `freeze_decoder=false`; the optimizer covers every trainable decoder and encoder
   parameter.  At deployment the teacher is absent.  The forward path adds the inferred identity
   to the neural input before the jointly trained decoder; it does not evaluate a teacher output
   and add a residual prediction.

Therefore A2 should be described as a **SPINT-initialized, task-only, jointly trained,
carrier-conditioned decoder**, not as knowledge distillation and not as an output-level
`S + delta` system.  Its terminal evidence remains valid and should not be discarded.

The criticism instead applies to the *research frame*: new architectures, losses, and interfaces
have repeatedly been forced through a teacher-derived decoder and an additive identity port.  That
frame makes independent structural contributions look like SPINT patches and can cap the search at
teacher-compatible functions.

## 2. When distillation is scientifically meaningful

Knowledge distillation can be a contribution when at least one real deployment asymmetry exists:

- the student is materially smaller, faster, or cheaper;
- the teacher uses privileged modalities, future context, dense labels, or expensive offline state
  that the deployed model cannot access;
- the teacher is an ensemble or smoother and the deployed model is a causal single model;
- distillation measurably improves calibration efficiency, robustness, or uncertainty under a
  constrained deployment budget.

For a same-input, same-task, similar-capacity teacher and student, teacher-output or teacher-identity
matching is primarily an optimization regularizer.  It may be useful, but it is not the core method.
The existing `task_plus_y` and `task_plus_y_plus_E` families should consequently be treated as
supporting ablations or historical scaffolding unless they can establish one of the asymmetries
above.

Teacher weights may still be used as an engineering warm start.  If so, the paper should call this
pretraining or initialization, report it transparently, and keep the architecture and scientific
claim independent of the initializer.

## 3. The abstraction change: carrier is not identity

The inherited coupled interface forms an identity tensor with the neural window width and applies

\[
x_i(t) + E_i(t)
\]

before the decoder read-in.  This requires a four-dimensional analytic carrier to be converted into
a pseudo-neural waveform.  Many recent ideas then differ only in where that additive signal is
inserted: input, hidden state, token set, or attention logit.

The alternative abstraction is:

> T4 estimates how a unit's activity should be interpreted in a shared task coordinate system.
> It is a condition on the population observation/read-in operator, not a persistent identity tag.

For query activity `x_i(t)`, support trials `S_i`, and carrier `c_i`, a generic task-frame decoder is

\[
a_{it}=f_\theta(x_i[t-W:t], S_i),
\]

\[
(B_i,g_i)=H_\phi(c_i, \operatorname{stats}(S_i)),
\]

\[
z_t = \frac{\sum_i \sigma(g_i) B_i a_{it}}
            {\epsilon + \sum_i \sigma(g_i)},
\qquad
\hat y_{1:T}=D_\psi(z_{1:T}).
\]

Here `a_it` is an `r`-dimensional activity feature, `B_i` is a `d x r` read-in matrix, `f` is a
small shared activity encoder, `H` is a carrier-conditioned map, `g_i` represents reliability
rather than identity, and `D` is a shared temporal decoder.  Unit aggregation is permutation
invariant and accepts different unit counts as a type-level property; robustness across unit-count
and tuning-density regimes must still be measured.  A final implementation must also carry an
explicit count/confidence statistic or justify the normalized sum, because normalization can erase
absolute population confidence.

The minimum first instantiation is deliberately simpler than the generic formula:

\[
z_t = \frac{1}{\sqrt{N}} \sum_i
      (U a_{it}) \odot (V g(c_i)),
\]

with a small fixed rank and a shallow causal temporal decoder.  This explicit bilinear model tests
the task-coordinate premise without giving a large hypernetwork enough capacity to memorize a
session-level carrier cloud.

This design has no algebraic SPINT base, no teacher prediction at inference, and no requirement that
carrier information imitate neural activity.

## 4. Evidence that motivates the change

### 4.1 H1: identity capacity is not the general bottleneck

The architecture-preserving H1 width study compressed the activity-only identity path from
5,965,500 to 60,124 identity parameters (99.22x) and from approximately 2.710B to 28.813M identity
MACs.  W32 was selected on the initial fold-0 routing date, then passed the frozen non-inferiority
criterion on four predeclared confirmation dates; the five-date summary is useful development
context but is not five independent confirmatory selections.  This is not a no-identity result, but
it argues strongly against making a larger NeuronID network the general innovation.

### 4.2 A2: task-frame content has large relative value under subject shift

The sealed A2 means are:

| domain | T4 | Z4 | T4 minus Z4 |
|---|---:|---:|---:|
| within sub-C | 0.574976 | 0.326008 | +0.248968 |
| external sub-M | 0.341367 | -0.143399 | +0.484766 |

The cross-domain interaction is `+0.235799`, with crossed seed-by-session bootstrap interval
`[+0.100852,+0.371768]`.  This establishes increased *relative* carrier value under the observed
subject shift.  It does not prove general biological correspondence or an absolute T4 improvement
of `+0.235799`.

### 4.3 Additive consumer changes have delivered little absolute improvement

- A1 produced a `+0.0128` interface-by-content interaction on six within-sub-C development
  sessions and stopped below its `+0.03` routing gate.  A1 did not measure an external-sub-M
  absolute improvement.
- Mix-T4 produced a carrier interaction of about `+0.0718` but only `+0.0226` absolute external T4
  improvement and therefore missed the frozen `+0.03` gate.
- Frozen SetKV post-hoc routing failed sharply; this rules out that frozen intervention, not every
  jointly trained set consumer.
- The value-mask intervention moved external T4 by only about `+0.002`.

Together these results establish that carrier *content* is valuable and that the tested add-site,
post-hoc-routing, mixing, and masking interventions did not add the required external absolute
lift.  Inefficient conversion by the inherited consumer is one live explanation, not an identified
fact: A2 may already capture much of the attainable information, and remaining error could instead
come from neural nonstationarity, normalizer transfer, T4 estimation noise, or task ambiguity.
The appropriate next step is therefore a falsifiable non-additive consumer screen, not a claim that
the consumer has already been diagnosed.

### 4.4 CEBRA: calibration-support adaptation may contain separate headroom

The completed Track-B Subject-M Stage-P development cell found target-support-only kNN R2 of
`0.291775316` for SUA and `0.340311825` for pseudo-MUA, above the corresponding source-only values
`-0.042026777` and `0.182215780` in that single date/seed cell.  The immutable paired completion is
`cebra_exploration/results/track_b_v2_subject_m_stagep_paired_real_producer_v2/aggregate/paired_completion_v2.json`,
SHA-256 `1f3db2d9fb8946e43be86774d433e913c10e1ab1c0527d9d452f20bb1ba86b8c`;
its status is `PAIRED_STAGEP_PILOT_COMPLETE__NO_POPULATION_INFERENCE`.  The cell is target session
`sub-M_ses-CO-20140307`, seed 42, with 24,708 query rows.  It uses dense behavior within the target
support and is constructive one-cell evidence only.  It suggests, but does not establish, that the
deployment system may benefit from designing around how calibration support creates a
target-serviceable representation rather than around teacher fidelity.

## 5. Divergent clean-sheet candidates

The candidates below are intentionally broader than a single architecture.  They are retained here
so later work does not collapse prematurely back into add-site variants.

### 5.1 Structural candidates

1. **Carrier-generated read-in hypernetwork.**  T4 generates a per-unit low-rank projection, value
   vector, or observation matrix.  Activity supplies time-varying amplitude.
2. **Learned population vector.**  Start from a transparent population-vector computation, but
   learn nonlinear activity features, confidence, temporal filtering, and task-space basis.
3. **Task-frame neural field.**  Treat units as samples from a continuous field over carrier/task
   coordinates, then integrate the field into a shared latent.
4. **Carrier-aware DeepSets or Set Transformer.**  Aggregate tokens containing query activity,
   support statistics, carrier, and reliability without any persistent unit slot.
5. **Carrier-geometry graph network.**  Construct unit-to-unit edges from carrier geometry and use
   message passing to combine complementary tuning.
6. **Task-frame state-space model.**  Let the carrier generate the session observation matrix while
   a shared dynamical model represents behavior evolution.
7. **Task-equivariant decoder.**  When carrier coordinates have an explicit two-dimensional task
   geometry, impose the corresponding rotation or coordinate equivariance.
8. **Low-rank bilinear activity--carrier decoder.**  Use an explicit multiplicative interaction
   `f(x)^T A g(c)` rather than an additive identity embedding.

### 5.2 Training candidates

9. **CEBRA-inspired cross-session contrastive task latent.**  Align different sessions at matched
   behavior states instead of matching a teacher representation.
10. **Episodic target-session training.**  Repeatedly treat a source session as unseen: use only its
    calibration prefix to create the read-in and optimize its post-prefix query performance.
11. **Self-supervised neural pretraining.**  Use masked-neural reconstruction, future-neural
    prediction, or cross-view consistency as a teacher-free initializer.
12. **Calibration-subset consistency.**  Require equivalent support subsets to yield stable task
    latents and predictions; the reference is the deployment function, not a teacher identity.
13. **Distributionally robust correspondence training.**  Use clean, mix, and frozen swap
    environments as a worst-case training set rather than as separate architectural stories.
14. **Uncertainty-aware carrier learning.**  Carry a confidence or posterior over carrier estimates
    and train the consumer to attenuate unreliable units.

### 5.3 Deployment candidates

15. **Differentiable closed-form target adapter.**  Train a source feature map through the exact
    ridge/Kalman solve that will be applied on the target prefix; deployment itself needs no
    backpropagation.
16. **Amortized Bayesian readout.**  Infer a posterior over target-session read-in/readout parameters
    from the calibration prefix.
17. **Two honest deployment tiers.**  Tier 1 is T4 forward-only with zero target parameter update;
    Tier 2 permits an explicitly counted target-support closed-form update.  They share a backbone
    but are never pooled into one headline comparison.

## 6. Convergence and proposed priority

### Priority 1: explicit low-rank bilinear activity--carrier read-in

This is the cleanest evidence-to-cost test of the new abstraction.  It directly asks whether an
aligned task coordinate should multiply or parameterize a unit's live activity contribution rather
than be added as an identity waveform.  The rank, parameter count, and aggregation can be tightly
matched, so a negative result is informative and cannot easily be blamed on an oversized
hypernetwork.

### Priority 2: learned population vector

This is both a strong simple baseline and a plausible winning model.  It should expose the direct
two-dimensional task-coordinate computation and learn only the activity nonlinearity, confidence,
and temporal dynamics needed beyond the analytic carrier.  Any more complex TFPD must beat this
baseline under the same supervision and initialization contract.

### Priority 3: constrained carrier hyper-read-in or task-frame observation model

If the explicit bilinear premise passes but leaves performance headroom, promote a low-rank
carrier-generated observation matrix and shared state-space/temporal decoder.  A generic Set
Transformer, neural field, or unconstrained carrier-cloud hypernetwork remains held until this
constrained extension is justified.

### Priority 4: episodic target-session training

Repeatedly treating a source session as an unseen target aligns source training with the actual
support-to-query deployment contract and is more relevant than teacher fidelity.  It can accompany
Priority 1, but must preserve exact prefix/query separation and must not introduce dense target
information into a Tier-1 sparse-carrier claim.

### Priority 5: honest Tier-2 closed-form meta-adaptation

Differentiable ridge/Kalman adaptation may have the highest short-term performance upside, but it
changes the target-update and label budget.  It should be developed as a separate Tier-2 deployment
result and never used to rescue a failed Tier-1 forward-only system.

### Supporting component: uncertainty/reliability gating

Reliability is likely important for the negative-session heterogeneity seen in several screens, but
it is a component of the read-in rather than a standalone method.  It should be added only after a
minimal carrier-generated read-in is attainable and non-vacuous.

## 7. Candidate kill criteria

Reject or demote a direction if any of the following holds:

- its only novelty is where an additive carrier is inserted into the SPINT consumer;
- it needs teacher-output or teacher-identity fidelity but has no compression, privileged-input, or
  deployment asymmetry;
- it increases NeuronID capacity without a dataset-specific headroom demonstration;
- it improves only a T4--Z4 interaction but misses the frozen absolute external T4 gate;
- it relies on post-hoc frozen routing after the consumer has already learned to ignore the new
  interface;
- it uses target labels or target updates without separate, explicit label-density and state-update
  accounting;
- it permits the carrier to be ignored and provides no aligned/zero/swap mechanism test;
- it is selected on one external target and then reported as a general cross-subject method.

## 8. Minimal evidence package

The clean-sheet route should avoid another large factorial.  The smallest defensible package is:

1. **One teacher-free architecture gate.**  Synthetic tests prove unit-permutation invariance,
   variable-`N` support, finite gradients, exact query/support separation, and an attainable
   carrier effect.  The first model is the explicit fixed-rank bilinear read-in, not a general
   hypernetwork.
2. **Seed-42 matched new-model T4/Z4 source pair.**  Use the A2 strict-27 roster, M30 support, source-only
   normalizer, 12-epoch schedule, query policy, within/external scorer, and formal-data seal, but
   standard-initialize and task-only train the complete new model.
3. **One standard-initialized SPINT-shaped T4 comparator.**  This is needed to separate architecture
   from teacher initialization.  The sealed teacher-initialized A2 T4 remains the historical
   deployment benchmark, not the sole causal architecture comparator.  Do not reopen a large
   distillation-loss lattice.
4. **Strong simple baselines.**  Include a learned population vector, an explicit carrier-free
   capacity-matched read-in, and supervision-budget-matched direct/ridge decoding.  A label-rich
   ridge may be reported as a non-information-matched upper comparator, never as a fair sparse-label
   peer.
5. **Primary performance decision.**  Compare external absolute T4 first against the
   standard-initialized SPINT-shaped T4 and also report the sealed A2 T4 deployment reference;
   require a predeclared practical lift with a within-sub-C non-inferiority floor.  The T4--Z4
   contrast is secondary mechanism evidence, not a rescue.
6. **Same-checkpoint mechanism diagnostics.**  Score aligned T4, exact zero carrier, the frozen
   matched-swap carrier, and an activity-destroyed diagnostic without retraining.  This distinguishes
   task-coordinate use from a carrier-only/session-signature shortcut.  Swap worse than zero is
   strong evidence but not a necessary condition if the model conservatively attenuates corrupt
   carriers.
7. **Conditional replication.**  Only after the seed-42 practical gate passes should seeds 43 and 44
   be added.  No formal subject-C test session is opened by this development package.
8. **Prospective boundary.**  sub-M is now a repeatedly used development external subject.  It can
   select and falsify development directions but cannot support a new general cross-subject claim.
   Such a claim requires a never-used subject/dataset/task frozen before evaluation, with the
   subject/session rather than query windows as the inferential unit.

## 9. Paper framing

Before the mechanism experiments, the primary statement must remain a hypothesis:

> Unconstrained NeuronID representations may encode session-specific fingerprints.  In tasks with
> a well-defined low-dimensional behavior coordinate and calibration-estimable unit tuning, a more
> transferable decoder may instead anchor each observed unit in that shared task frame and let the
> task coordinate parameterize population read-in.

Two-sentence prospective pitch, usable as a method proposal but not yet as a result claim:

> We test whether cross-session neural decoding is limited by unconstrained, session-specific unit
> representations.  In two-dimensional reaching tasks, we treat each unit as a task-coordinate
> observation and use a calibration-derived carrier to generate its read-in into a shared motor
> latent, yielding an end-to-end decoder that does not require teacher fidelity or persistent unit
> slots.

SPINT then has three honest roles:

- inspiration for calibration-conditioned decoding;
- a strong baseline/reference system;
- an optional disclosed initializer in a small ablation.

It is not the algebraic base of the claimed method.  Existing A2 evidence remains important because
it establishes the value of task-frame content under shift, while the clean-sheet system asks
whether a better consumer can convert that content into higher absolute performance.

## 10. Strongest objections and required responses

### Objection 1: T4 uses target calibration labels

An improvement could reflect target information rather than architecture.  The response is a strict
matched deployment contract: compare new T4 only against existing T4 with identical M30 labels,
normalization, support/query split, and target-update policy.  Dense target-support adaptation is a
separate Tier-2 result.

### Objection 2: teacher-free training may underfit limited source data

If a warm start helps, report it as pretraining rather than distillation.  Self-supervised source
pretraining is a teacher-free alternative.  A dependence on initialization is an engineering fact;
it must not redefine the architecture as a student.

### Objection 3: a hypernetwork may memorize session identity through the carrier cloud

Exclude session identifiers, use held-session and held-subject evaluation, bind population
statistics before target scoring, and require aligned/zero/swap diagnostics.  If the model succeeds
only through a session-level carrier signature, the swap and external-subject results should expose
that failure.

### Objection 4: the new model may merely be a learned population vector

That is not fatal if it outperforms under the same deployment contract.  The scientific novelty
would be the task-frame-generated, variable-population read-in and its demonstrated transfer, not
complexity for its own sake.  A strong simple mechanism is preferable to a large inherited
teacher--student stack.

### Objection 5: sub-M is no longer an untouched external test

sub-M has been used repeatedly for route selection, mechanism screens, and practical gates.  Seeds
do not restore subject-level independence.  All new work must call it a development external
subject, and any general cross-subject claim must wait for a prospectively frozen, never-used
subject/dataset/task.  Formal held-out sub-C sessions can confirm held-session behavior but cannot
by themselves establish subject transfer.

## 11. Disposition of existing directions

- Preserve A2 as the terminal matched carrier-content/subject-shift result.
- Preserve original SPINT/B0 as a comparator, not a parent model.
- Demote teacher-domain and distillation-loss lattices to initialization/regularization ablations.
- Treat A10, MATCH, value-mask, and similar interventions as deployment or diagnostic baselines.
- Treat POP-HYP as a cheap bridge to the hyper-read-in idea, but not as the final method: it still
  expresses only a low-rank delta on the inherited SPINT `fc_in`.
- Treat swap/mix as robustness environments and mechanism tests.  They can inform the new model's
  training, but should not define its architecture.
- Implement any clean-sheet successor in an independent model/module.  Reuse audited data,
  normalizer, query, scorer, and receipt infrastructure, but do not inherit
  `StreamingCalibrationLitModule` merely for compatibility.

## 12. Recommended decision

Stop using “improved SPINT student” as the innovation mainline.  Promote **task-frame-conditioned
population decoding** as a falsifiable research hypothesis, beginning with an explicit low-rank
bilinear activity--carrier read-in and learned-population-vector baseline.  Promote a carrier
hypernetwork or task-frame state-space observation model only if that minimal premise passes.  Keep
SPINT as a transparent baseline and optional initializer; keep only the smallest T4/Z4,
aligned/zero/swap/activity-destroyed, and initialization controls needed to identify why the
complete new system works.

## 13. Independent sol-medium review addendum

Two `gpt-5.6-sol` reviewers at medium reasoning independently read the same draft SHA-256
`06f6793ccb2e923672c3919342dede760c59aa55a9a873a7e3eb392e7a7bf3d6`.  They were forbidden to
edit files, run experiments, or exchange opinions.  Both independently returned **GO** for
demoting teacher--student/distillation as the innovation mainline and **GO as a hypothesis**, not a
proven answer, for task-frame population decoding.

Their independent top-five rankings were:

| rank | reviewer A | reviewer B |
|---:|---|---|
| 1 | #8 low-rank bilinear activity--carrier decoder | #8 low-rank bilinear activity--carrier decoder |
| 2 | #2 learned population vector | #2 learned population vector |
| 3 | #1 constrained carrier hyper-read-in | #6 task-frame state-space model |
| 4 | #10 episodic target-session training | #10 episodic target-session training |
| 5 | #17 two honest deployment tiers | #1 constrained carrier hyper-read-in |

Reviewer A confidence was `91/100`; reviewer B confidence was `89/100`.  The robust consensus is:

1. begin with #8, not a general hypernetwork;
2. require #2 as a strong simple comparator and possible winner;
3. treat #1/#6 as the conditional extension after the bilinear premise;
4. align training with support-to-query deployment through #10;
5. keep target-updating methods in a separately reported Tier 2.

The reviewers independently identified the same factual error and major scientific risks.  This
revision corrects A1 `+0.0128` from an alleged external lift to a within-sub-C routing interaction;
records the exact Track-B completion path/SHA and one-cell boundary; distinguishes the H1 selection
date from four confirmation dates; changes consumer inefficiency and session-fingerprint language
from conclusions to hypotheses; adds initialization-matched and strong-simple baselines; adds an
activity-destroyed shortcut control; and designates sub-M as a repeatedly used development external
subject rather than a prospective cross-subject test.

This review addendum changes research priority only.  It still authorizes no implementation, data
access, GPU execution, target scoring, or formal test.
