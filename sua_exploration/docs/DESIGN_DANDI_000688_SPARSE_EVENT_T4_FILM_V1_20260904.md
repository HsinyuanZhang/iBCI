# Design: DANDI 000688 Sparse-Label T4 Estimation + Conditional FiLM V1

Date: 2026-09-04
Status: **REV2_REVIEWED__CPU_STAGE0_IMPLEMENTATION_MAY_BEGIN__NO_GPU_AUTHORITY**
Short name: **SL-T4 / SE-T4-FiLM V1**
Rev2 review edits (2026-09-04, independent reviewer): era counts corrected to
5 short-delay / 9 no-delay / 13+6 long-delay; Stage-1 parent authority changed
from validation-argmax `best_checkpoint` to final-epoch `epoch_011.ckpt`;
Stage-1 estimator screen made non-gating; Stage-2 warm-start asymmetry
disclosed; Section 22 start sequence added.
Primary view: **SUA**
Conditional replication view: **pseudo-MUA**
Revision basis: **pre-decoder read-only geometry audit of the exact 27-train / 6-validation source manifest; no test file opened**

This document authorizes neither GPU use nor result-root creation. It freezes a
scientific design for independent review. A separate work order, fresh closure,
explicit GPU authorization, and immutable lifecycle are required before execution.

---

## 0. Executive decision

The dense-speed calibration-profile FiLM line on the complete DANDI 000688 T4
carrier is closed. The observed profile is reliable, but its contents did not
beat a matched empty-profile capacity control on the full-T4 substrate, and
opening `post_pool` did not rescue the semantic effect.

The successor now separates two questions that the first draft incorrectly
bundled together:

1. **Estimator question:** with the same first-M10 direction/cue labels, can a
   fixed post-cue firing-rate window produce a better low-budget T4 carrier
   than the historical whole-trial estimator?
2. **Conditional FiLM question:** only if a sparse-event profile is
   independently reliable, can a zero-initialized FiLM extract additional
   row-attached information beyond a matched empty-profile head and a
   single-phase control?

The estimator question is primary because it is simpler and the read-only
audit found a stronger signal for it. The FiLM question is conditional on a
decisive CPU reliability gate. Neither route is required to match or exceed a
dense-kinematic or M30/M50 upper bound.

The revised FiLM profile, if admitted, combines movement-window direction
tuning with a center-hold-to-movement baseline contrast:

\[
q_i^{SE}=[a_i^R,c_i^R,m_i^R,b_i^R-b_i^H].
\]

Both candidate routes use the same first-M target-direction labels and sparse
`target_on_time`/`go_cue_time` events. Rates are computed directly from spike
times with half-open intervals. No per-bin velocity is read to construct any
carrier or profile.

---

## 1. Scientific question and claim boundary

### 1.1 Primary question

At a first-M10 target calibration budget, does matched training with
`POST700-T4` improve decoder R2 over matched training with the ordinary
whole-trial `WHOLE-T4`, using the exact same sessions, query windows, targets,
activity support, initialization, optimization budget, and randomness?

### 1.2 Secondary questions

1. Is the four-column sparse-event profile reliable enough, before decoder scoring,
   to justify a FiLM experiment at all?
2. If admitted, does `SE-T4-FiLM` beat a parameter-matched `EMPTY` head, a
   `PHASE-R` second-observation control, and a frozen within-session
   `ROW-SHUFFLE` attachment control?
3. If the SUA route passes, does the same mechanism transfer to a pseudo-MUA view when
   pseudo-MUA phase-T4 rows are refitted from pooled spikes rather than averaged
   from SUA rows?
4. How much of the historical M10 deficit disappears from matched M10 training
   or improved rate estimation, without interpreting that deficit as a FiLM-
   recoverable information quantity?

### 1.3 What a positive result may claim

A positive estimator result may support:

> Under a fixed sparse M10 label budget, choosing a movement-relevant post-cue
> rate window improves T4 estimation and downstream decoding relative to the
> historical whole-trial rate estimator.

A positive conditional FiLM result may additionally support:

> Sparse target direction and target/go event annotations contain reliable
> movement-window direction tuning and hold-to-movement baseline information
> that a zero-initialized FiLM can use to recalibrate an early-pooled T4
> identity at low label budget.

It may not support:

- that sparse labels equal dense labels;
- that no dense source behavior target was used to train the decoder;
- that the method is label-free;
- that the method is an EvalAI/FALCON result;
- that the method improves every carrier or dataset;
- that pseudo-MUA equals or exceeds SUA;
- that M4 is as reliable as M10;
- that any selected T4@50 checkpoint was reproduced.

### 1.4 What a negative result means

A matched negative estimator result closes the frozen `POST700-T4` estimator.
A Stage-0 reliability failure closes the cue-phase FiLM route without spending
GPU compute. A matched FiLM null closes this exact profile/operator. Neither
outcome proves that all sparse-event carriers or two-state pooling are
impossible.

---

## 2. Why this successor is scientifically distinct

### 2.1 Dense-speed CP-FiLM predecessor

The predecessor profile partitioned calibration bins by the 25th and 75th
percentiles of the norm of dense two-dimensional velocity. It therefore
consumed per-bin kinematics to define low/high state.

Its full three-seed result on the M30/T4@30 substrate was a semantic null:

- `CP10@M10 - EMPTY = -0.003502` mean R2;
- `CP30@M30 - EMPTY = -0.002793` mean R2;
- `CP10@M10 - SHUFFLE10@M10 = +0.000066` mean R2.

The profile itself was not absent or dominated by split-half noise:

- M10 split-half Pearson mean `0.7765`;
- M30 split-half Pearson mean `0.9163`.

The failure is therefore evidence against adding that dense-speed profile to an
already complete T4 carrier under the tested operator, not evidence against all
profile-conditioned FiLM.

### 2.2 Historical low-budget deficit: context, not recoverable headroom

The frozen V9 label-budget audit reported the following retrospective means
over 15 sessions x 3 seeds:

| View | T4@M10 | T4@M30 | M10-to-M30 gap |
|---|---:|---:|---:|
| SUA | 0.304264 | 0.358154 | +0.053890 |
| pseudo-MUA | 0.259185 | 0.305291 | +0.046106 |

These cells were generated by changing the deployment label budget of a single
T4@50-trained checkpoint. They conflate train/deploy mismatch, noisy M10 OLS
estimation, and direction-design conditioning. They are retrospective context,
not a FiLM-recoverable information gap, not a fresh selection surface, and not
a matched M10 training comparison. The selected T4@50 checkpoint bytes are also
not locally recoverable.

Any fresh sparse-budget denominator must instead be the Stage-2 matched
`WHOLE-T4@M10` arm. No recovery fraction may use the historical table as its
governing denominator.

### 2.3 H1 sparse-event precedent

The H1 `H-SE5` carrier used only movement-event endpoint pairs and per-event
neural rates. Its complete evidence is mixed:

- `H-SE5 - Zero5 = +0.028469` in the matched M4 fold-0 cell;
- both target recordings positive;
- correct-minus-label-shuffle and correct-minus-intercept positive on all 13
  public recordings at M3 and M4;
- `H-SE5 - H-S = +0.003204`;
- `H-SE5 - dense H-C = -0.025474`;
- the strict second-date replication gave `H-SE5 - Zero5 = -0.022590`, with
  only `2/3` recordings positive.

H-SE5 is therefore a carrier-only boundary result, not a replicated sparse
success and not evidence for FiLM. It motivates testing sparse estimators while
requiring cross-session/date discipline; it cannot serve as positive prior
evidence for the proposed DANDI effect.

### 2.4 Two distinct mechanisms

The primary `POST700-T4` route changes no label count and adds no model
parameters. It tests whether ordinary whole-trial rates are a statistically
poor low-budget estimator compared with rates in the fixed 700 ms immediately
after go cue.

The conditional `SE-T4-FiLM` route asks a different question. Ordinary T4 fits
one direction-tuning curve after averaging a whole trial; the revised candidate
adds:

- direction tuning `[a^R,c^R,m^R]` measured in a fixed 700 ms movement window;
- a scalar baseline contrast `Delta b=b^R-b^H` between that movement window
  and a 300 ms center-hold window before target appearance.

A single whole-trial T4 does not explicitly expose this window-specific
direction estimate or hold-to-movement baseline change. The claim is narrower
than "state-by-direction interaction": `a^R/c^R/m^R` may simply be a better
direction estimator, while `Delta b` is the only explicit state difference.
`PHASE-R` separates these explanations at decoder level.

The first-draft profile used `[Delta a,Delta c,Delta m,Delta b]` with both phases
anchored around go cue. Its disclosed M30 split-half evidence was:

| Profile column | all-33 mean r | validation-6 mean r |
|---|---:|---:|
| `Delta a` | 0.34 | 0.35 |
| `Delta c` | 0.34 | 0.32 |
| `Delta m` | 0.35 | 0.42 |
| `Delta b` | 0.65 | 0.65 |

Those values retire the first-draft subtraction of directional columns. The
revised `[a^R,c^R,m^R,Delta b]` profile requires a fresh, predeclared per-column
Stage-0 reliability gate before any decoder score.

---

## 3. Exact supervision ledger

### 3.1 Two meanings of sparse

This design keeps two sparsity axes separate.

1. **Temporal annotation sparsity:** carrier/profile construction uses one
   target direction and event timestamps per calibration trial, not a dense
   velocity sample at every 20 ms bin.
2. **Trial-budget sparsity:** the primary carrier/profile horizon is the first
   ten rewarded trials, `M=10`.

### 3.2 Primary target-session inputs

For each of the chronological first ten rewarded calibration trials, the
carrier/profile materializer may read:

- `target_dir`;
- `target_on_time`;
- `go_cue_time`;
- `start_time` and `stop_time` for legality checks;
- neural spikes/counts belonging to the trial and the fixed cue windows.

It may not read:

- per-bin cursor velocity;
- per-bin position;
- movement onset inferred from velocity;
- trial 10+ target directions for M10 construction;
- query targets or predictions;
- external/formal-session labels during selection.

### 3.3 Counts that must be reported separately

Every session receipt must report:

- rewarded trial horizon: `10`;
- finite target-direction count;
- distinct canonical direction count;
- cosine-design rank and condition number;
- finite cue count;
- finite target-on count;
- hold/reach cue-window count;
- scalar target-direction values consumed;
- target-on and go-cue timestamp values consumed, reported separately;
- dense velocity scalars consumed: exactly `0`;
- neural values consumed, separately and never described as labels.

`M10` denotes a chronological horizon, not a promise that all ten direction
fields are finite. Existing source evidence includes two sessions with nine
finite first-M10 directions; both retain rank three. The exact finite count must
be disclosed rather than silently imputed.

Stage 0 additionally reads M30 odd/even and source trials 50-109 labels in a
separate source-only reliability-audit namespace. Those labels choose only the
predeclared column mask; they are never candidate inputs, never available for a
target session, and never passed to decoder training or scoring. Receipts must
account for audit labels and candidate labels separately.

### 3.4 Source training is still supervised

The SPINT decoder is trained on source-session behavior targets. This design
does not claim an entirely sparse-supervision training pipeline. The sparse
contract applies to target-session calibration inputs and to the newly added
carrier/profile construction.

---

## 4. Frozen V9-compatible deployment surface

### 4.1 Independent knobs

The implementation must expose and bind three independent quantities:

- `activity_support_n = 30`;
- `carrier_profile_label_horizon = 10`;
- `query_start_trial = 50`.

They must not be represented by one overloaded `calibration_n_trials`
argument.

### 4.2 Activity support

The identity activity input remains the chronological first 30 rewarded trials,
using the existing early-pooling activity representation. It is label-free with
respect to target direction in this experiment.

This design does not add:

- random activity selection;
- Top-K activity selection;
- late pooling;
- continual query memory;
- growing support;
- cardinality cycling.

### 4.3 Carrier/profile support

Only the chronological first ten rewarded trials may supply target direction,
target-on, or go-cue labels to `WHOLE-T4`, `POST700-T4`, or `q_SE`.

### 4.4 Query surface

All scored query windows begin strictly after rewarded trial 50, matching the
V9 evaluation boundary. Trials 10 through 49 contribute no direction or event
labels to the M10 carrier/profile.

The choice deliberately holds activity, label horizon, and query boundary
separate. A gain cannot be attributed to moving the query start or seeing more
activity.

---

## 5. Cue-window contract

### 5.1 Why two sparse event anchors are used

A read-only audit of the 27 train and six validation source NWBs found the same
trial columns in all 33 sessions:

- `start_time`;
- `target_on_time`;
- `go_cue_time`;
- `stop_time`;
- target/result fields.

There is no `move_onset_time` or `movement_onset_time` field.

`target_on_time -> go_cue_time` is not a stable cross-session duration: nine
source sessions have an approximately 1 ms interval, while other eras have
approximately 0.3 s or 1.0 s intervals. V1 therefore does not use that variable
interval as a state. Instead, `target_on_time` anchors a fixed pre-target hold
window and `go_cue_time` anchors a fixed post-go movement window.

### 5.2 Deterministic spike-time intervals

There is no independent "official 20 ms bin timestamp" authority in the
current loader. Its dense neural bin grid begins at the first observed spike,
so anchoring cue windows to that grid would introduce a session-dependent
offset and a second rate estimator.

For finite target-on `o` and go-cue `g`, V1 freezes two half-open
continuous-time intervals:

```text
H300 = [o - 0.300, o)
R700 = [g, g + 0.700)
```

`H300` is center hold before target appearance. `R700` is the movement-window
rate used both by the primary estimator comparison (`POST700-T4`) and the
conditional FiLM profile.

Counts are obtained from each unit's sorted raw spike times with
`numpy.searchsorted(..., side="left")`, exactly matching the half-open law in
`unit_side_features._pool_trial_rate_matrix`. Rate is count divided by the
fixed interval duration. No interpolated `[trial,100,unit]` tensor or neural-bin
grid participates in these estimators.

The read-only audit inspected 300 ms and 700 ms post-cue windows before any new
decoder score. Dense cursor velocity was opened only in that separate audit to
check state geometry; production materialization remains forbidden from
opening it. The audit found that the first 300 ms after go cue was already
movement at median speeds of approximately 11-20 cm/s in the five short-delay
sessions but remained at approximately 0.3-0.8 cm/s in the other 28 sessions.
Movement in those sessions fell mainly in
`[g+0.300,g+0.700)`. The 700 ms direction fit also had substantially higher
M10-to-independent-reference correlation. This is disclosed selection
evidence: `R700` is now frozen, and V1 may not try a third duration or choose a
duration from decoder R2.

### 5.3 Legality

A trial contributes to phase T4 only if:

- result is the exact rewarded code used by the parent T4 route;
- target direction is finite and maps to a canonical direction;
- target-on event is finite;
- cue is finite;
- every interval start and stop is finite;
- `[o-0.300,o)` lies inside `[trial_start,o)`;
- `[g,g+0.700)` lies inside `[g,trial_stop)`;
- unit spike-time arrays are finite and nondecreasing;
- neither interval overlaps another trial or query material.

The audit found minimum `start -> go` approximately 0.62 s and minimum
`go -> stop` approximately 0.73 s for first-M10 trials. Stage 0 must still
revalidate both exact revised interval-containment predicates trial by trial.
An invalid trial is excluded and counted; no padding, interpolation, event
repair, or velocity-derived movement onset is allowed.

### 5.4 Frozen revision policy

The two revised intervals are frozen for V1. They were changed from the first
draft using a pre-decoder read-only geometry audit; this does not consume an
in-run Stage-0 repair allowance. Any further geometry-driven revision requires
a new design version and document hash. Decoder R2 may not select the duration.

### 5.5 Era reporting

The discarded first-draft pre-go window, denoted `H_pre-go` here to avoid
confusing it with the revised `H300`, had different meanings across eras:

- nine training sessions have `target_on -> go_cue` near 1 ms, so `H_pre-go` is
  center hold without a presented target;
- five training sessions (20131003, 20131022, 20131023, 20131031, 20131101)
  have an interval near 0.3 s, so `H_pre-go` spans most of the visual delay;
- thirteen training and all six validation sessions have an interval near
  1.0 s, so `H_pre-go` is late preparatory delay.

| Era | Training sessions | Validation sessions | discarded pre-go window |
|---|---:|---:|---|
| no-delay, about 1 ms | 9 | 0 | center hold without target |
| short delay, about 0.3 s | 5 | 0 | target response / most of delay |
| long delay, about 1.0 s | 13 | 6 | late preparatory delay |

Era membership is computed from the median `go_cue_time - target_on_time` over
the first 30 rewarded trials (nan-ignoring), not over the first 10, because
session 20131031 has a nonfinite first-10 value.

The revised `H300=[target_on-0.300,target_on)` precedes target appearance and had
audited mean speeds of approximately 0.3-0.65 cm/s in every era. Therefore all
27 training sessions are predeclared to enter the FiLM-head training set; none
is excluded based on delay era. Stage 0 must nevertheless report profile
statistics/reliability by era, and every decoder stage must report
`D_semantic` by era, because acquisition-era effects may remain.

---

## 6. T4 and phase-profile estimator

### 6.1 Ordinary T4 carrier

For unit/channel `i`, ordinary T4 fits the existing equal-per-direction cosine
law on whole-trial firing rates:

\[
r_i(\theta)=b_i+a_i\cos\theta+c_i\sin\theta,
\qquad m_i=\sqrt{a_i^2+c_i^2}.
\]

The emitted column order remains:

\[
T4_i=[a_i,c_i,m_i,b_i].
\]

No estimator change is permitted in the baseline carrier.

### 6.2 Window-specific trial rates

For every legal support trial, compute one rate per channel in each frozen
interval. `H300` divides spike count by `0.300` seconds; `R700` divides by
`0.700` seconds. `POST700` and `R700` are the same rate authority under two
experimental roles.

Use raw spike times through the reviewed ordinary-T4 half-open rate primitive,
not the interpolated `[trial,100,unit]` activity tensor, binned neural counts,
or a dense behavior trace.

### 6.3 Phase-specific T4 fits

Using the same target direction attached to the trial, fit the exact ordinary
T4 cosine estimator separately to `H300` and `R700` rates:

\[
T4_i^H=[a_i^H,c_i^H,m_i^H,b_i^H],
\]

\[
T4_i^R=[a_i^R,c_i^R,m_i^R,b_i^R].
\]

The same per-direction averaging law, least-squares convention, canonical
direction mapping, and missing-direction handling used by ordinary T4 must be
reused rather than reimplemented approximately.

### 6.4 Sparse event profile

The raw four-dimensional profile is:

\[
q_i^{raw}=[a_i^R,\;c_i^R,\;m_i^R,\;b_i^R-b_i^H].
\]

The column order is fixed as `[a_R,c_R,m_R,Delta b]`, with
`Delta b=b_R-b_H`. Directional `Delta a/Delta c/Delta m` are not computed:
center hold is intentionally pre-target and should not carry target-direction
tuning, so subtracting its directional fit would add noise.

The Stage-0 reliability mask may retain or zero individual columns only under
the predeclared rule in Section 11. A failed column is exact positive zero in
every semantic/control arm after a new document hash is frozen. `m_R` still has
norm-induced positive noise bias, and `Delta b` overlaps the previously tested
hold/move contrast; both facts must be reported.

### 6.5 Primary `POST700-T4` estimator

For every unit, fit the unchanged ordinary four-column T4 cosine law to the
first-M10 `POST700` rates. The result remains `[a,c,m,b]`; only the firing-rate
interval changes. `POST700-T4` uses exactly the same ten directions and cue
timestamps as `WHOLE-T4` and adds no labels or model parameters.

The already-inspected descriptor-only reference was whole-trial T4 fitted on
non-overlapping source trials 50-109. The observed correlations were:

| M10 estimator | r(a), all / val | r(c), all / val |
|---|---:|---:|
| historical whole-trial | 0.65 / 0.57 | 0.69 / 0.71 |
| first 300 ms post-go | 0.61 / 0.71 | 0.59 / 0.72 |
| `POST700` | 0.78 / 0.79 | 0.79 / 0.85 |

These numbers motivate and disclose the frozen estimator choice; they are not
fresh decoder evidence and cannot be used as a performance claim.

### 6.6 Carrier and profile normalization

Normalization is route-specific and frozen before scoring:

1. **Stage-2 matched estimator route:** fit one source-train-only four-column
   normalizer for `WHOLE-T4@M10` and a separate source-train-only normalizer for
   `POST700-T4@M10`. Each arm uses its own estimator-matched normalizer.
2. **Frozen-parent OOD diagnostic:** ordinary `WHOLE-T4@M10` uses the exact
   parent M30 normalizer, preserving the historical label-budget deployment
   law and its zero anchor. `POST700-T4@M10` uses a source-train-only M10
   normalizer and must be compared against a `WHOLE-T4@M10` arm using an
   independently refitted M10 normalizer, so normalizer change is not credited
   to the estimator. The conditional FiLM screen fits its `q_SE` normalizer on
   all 27 source-training sessions only.
3. **Matched FiLM route:** the frozen matched `WHOLE-T4@M10` carrier and its
   M10 normalizer come from the held matched baseline. Fit a separate
   source-train-only profile normalizer on all 27 training sessions before
   applying the frozen column mask. `PHASE-R` uses the same fitted scales for
   its retained first three columns and exact zero in column four.
4. No target-session or validation-session row contributes to any fitted
   mean/scale.

Requirements:

- scale has a fixed floor declared in the work order;
- all four raw and normalized columns are finite;
- raw profile column order remains `a_R, c_R, m_R, Delta b`;
- the all-source deployment normalizer, if later authorized, is fitted only
  after source-development decisions are frozen;
- no target-session robust-z refit may silently change the information law.

---

## 7. FiLM operator

### 7.1 Early-pooling substrate

The successor keeps the existing early-pooling B3S/T4 order:

\[
u_{ij}=\phi(x_{ij}),
\qquad
h_i=\frac{1}{K}\sum_j u_{ij},
\]

followed by one `post_pool` application.

No post-MLP/late-pooling or per-trial identity averaging is introduced.

### 7.2 Context and modulation

For the conditional FiLM route, let `c_i` be the stage-specific frozen
normalized `WHOLE-T4@M10` carrier: parent M30 normalization in Stage 1 and
source-fitted M10 normalization in Stage 2. Let `q_i` be the arm-specific
normalized profile:
zero, retained columns of `[a_R,c_R,m_R,+0]`, retained columns of
`[a_R,c_R,m_R,Delta b]`, or the row-shuffled version of that same full
sparse-event profile. Failed reliability columns are exact positive zero under
the frozen Stage-0 mask for every applicable arm.

\[
[\gamma_i,\beta_i]=F([c_i,q_i]),
\]

\[
h_i'=(1+\gamma_i)\odot h_i+\beta_i,
\]

\[
z_i=\psi([h_i',c_i]).
\]

The initial FiLM head is the existing rank-8 form:

```text
Linear(8, 8) -> ReLU -> Linear(8, 128)
```

The final linear layer is initialized to exact positive zero so that
`gamma=beta=+0`. The production operator must additionally take an explicit
direct-native branch when both modulation tensors are exact IEEE positive zero;
it may not rely only on evaluating `(1+0)*h+0` and then claim bitwise parity.

### 7.3 Zero-anchor law

Before any optimizer step, for every arm and at least one complete real source
session:

- identity tensor must equal the T4-native identity bitwise;
- prediction SHA must equal the T4-native prediction SHA;
- R2 must equal within the metric serialization law;
- all non-FiLM state must match the parent state;
- target/model update counts must be zero during the anchor.

Anchor failure stops the route before training.

The direct-native branch is an implementation law, not an inference-time
selection gate. Once a trained FiLM emits any nonzero modulation, the normal
modulated path is used.

---

## 8. Experimental arms

### 8.1 Parallel estimator arms

| Arm | M10 rate estimator | FiLM | Purpose |
|---|---|---|---|
| `WHOLE-T4` | historical whole rewarded trial | none | same-budget estimator anchor |
| `POST700-T4` | fixed `[go_cue, go_cue+0.700)` | none | primary sparse-label estimator |

They are first compared as a frozen-parent deployment-OOD diagnostic
(descriptive only, see Stage 1) and, if Stage-0 step 9 passes, under paired
matched training with their own source-train-only estimator-matched
normalizers. Within either comparison, the
unique scientific difference is the neural rate interval used to estimate the
same four T4 columns from the same first-M10 direction/cue annotations.

### 8.2 Conditional five-arm FiLM matrix

This matrix is implemented or trained only if the Stage-0 FiLM reliability gate
passes. Its admission is independent of the `POST700-T4` decoder sign.

| Arm | Carrier | FiLM profile | Purpose |
|---|---|---|---|
| `WHOLE-NATIVE` | correct matched M10 `WHOLE-T4` | no trained FiLM | FiLM-family product anchor |
| `EMPTY` | same | exact positive-zero four-vector | matched capacity/global-affine control |
| `PHASE-R` | same | retained normalized `[a_R,c_R,m_R,+0]` | movement-window direction control |
| `SE-T4` | same | retained row-attached `[a_R,c_R,m_R,Delta b]` | full sparse-event method |
| `ROW-SHUFFLE` | same | deterministic nonidentity permutation of the same retained `q_SE` rows | attachment control |

`PHASE-R` is required. Without it, a positive `SE-T4` result could be due
entirely to supplying a movement-window direction estimate. The full method's
increment over `PHASE-R` isolates the retained `Delta b` hold-to-movement
baseline contribution.

### 8.3 Row-shuffle law

For each session and seed:

- derive a dedicated PCG64 seed from schema, session, view, and training seed;
- construct a nonidentity permutation;
- apply it only to complete retained `q_SE` rows;
- do not permute T4, activity, query, targets, or unit roster;
- freeze the same permutation for all epochs and scoring;
- report permutation and SHA.

### 8.4 FiLM estimands

`EMPTY` has the same FiLM parameter count, optimizer, step count, T4 context,
and initialization family as `SE-T4`. Therefore:

\[
\Delta_{semantic}=R2(SE\text{-}T4)-R2(EMPTY)
\]

is the primary profile-content estimand.

Two additional contrasts are mandatory:

\[
\Delta_{baseline}=R2(SE\text{-}T4)-R2(PHASE\text{-}R),
\]

\[
\Delta_{attachment}=R2(SE\text{-}T4)-R2(ROW\text{-}SHUFFLE).
\]

`SE-T4 - WHOLE-NATIVE` is the product increment, but it mixes semantic
content with any global/capacity effect learned by an empty FiLM head. A
positive `PHASE-R` with null `Delta_baseline` supports "movement-window T4
helps," not an additional hold-to-movement baseline-profile claim.

### 8.5 Optional controls not in V1

The following are explicitly excluded from the initial matrix:

- dense-speed CP;
- direction-label shuffle;
- cue jitter;
- direct-concat sparse-event profile;
- two-state activity pooling;
- M10-to-M50 teacher correction;
- profile-width/rank sweep.

They require additive successors and cannot be introduced after seeing a V1
score.

---

## 9. Views and pseudo-MUA construction

### 9.1 SUA is primary

The first governed experiment is SUA because the hypothesis is explicitly
per-unit/channel state-tuning attachment.

### 9.2 pseudo-MUA is conditional

Pseudo-MUA execution is authorized by a future work order only after the SUA
primary passes its semantic and attachment gates.

### 9.3 Correct pseudo-MUA estimator order

For pseudo-MUA:

1. compute half-open per-SUA-row trial rates, then sum those rates into the
   frozen pseudo-MUA electrode/channel view with the reviewed
   `pool_trial_rates_by_electrode` authority;
2. recompute whole-trial, `H300`, and `R700/POST700` rates in that pooled view;
3. refit `WHOLE-T4`, `POST700-T4`, `T4^H`, and `T4^R` per pooled row;
4. compute `q_SE=[a_R,c_R,m_R,b_R-b_H]` from the refitted pooled rows;
5. fit pseudo-MUA-specific normalizers from source-training rows only.

Forbidden shortcuts:

- averaging SUA T4 rows;
- averaging SUA `q_SE` rows;
- pooling already normalized SUA descriptors;
- reusing SUA permutations or normalizers.

### 9.4 Cross-view estimands

Within-view effects:

\[
\Delta_{SUA}=R2(SE\text{-}T4_{SUA})-R2(EMPTY_{SUA}),
\]

\[
\Delta_{pMUA}=R2(SE\text{-}T4_{pMUA})-R2(EMPTY_{pMUA}).
\]

Secondary interaction:

\[
I=\Delta_{SUA}-\Delta_{pMUA}.
\]

`I` is not a prerequisite for a within-view success. Absolute SUA-minus-pMUA
performance remains a separate system comparison.

---

## 10. Source-development split and freshness

### 10.1 Source split

Use the exact existing
`sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`
27-train / 6-validation sub-C session manifest. Session groups and order are
frozen before execution. The geometry audit opened only those 33 source files
and no test file.

### 10.2 Freshness statement

The six validation sessions and external-15 have been used in prior development
and are not fresh formal evidence. This experiment is a source-development
mechanism study.

### 10.3 Forbidden selection surfaces

The following may not select windows, budgets, ranks, losses, epochs, or arms:

- external-15 retrospective rows;
- disputed formal sub-C files;
- EvalAI/FALCON results;
- H1 or M2 target-session scores;
- pseudo-MUA scores before the SUA decision is frozen.

### 10.4 Seeds

Use paired seeds `42, 43, 44`. A single-seed result cannot govern the scientific
conclusion.

---

## 11. Execution stages

### Stage 0 — CPU constructibility and information audit

No decoder training and no GPU.

Required checks:

1. Validate the exact 33-session source allowlist.
2. Prove dense velocity arrays are never opened by the descriptor materializer.
3. Materialize M10 `WHOLE-T4`, `POST700-T4`, `T4^H`, `T4^R`, and
   `q_SE=[a_R,c_R,m_R,Delta b]` for SUA through the reviewed spike-time rate
   primitive.
4. Require every relevant cosine design to retain rank three after declared
   invalid-trial exclusions.
5. Report finite label/cue counts, distinct directions, conditions, raw and
   normalized profile digests.
6. Verify exact half-open `H300` and `R700/POST700` interval geometry.
7. Verify no trial >=10 label enters any candidate carrier/profile presented to
   a model.
8. Verify no query target/prediction enters candidate materialization. The
   source-only reliability-audit namespace may separately read trials 50-109
   direction/target-on/cue labels to construct reference descriptors, but those arrays
   must be receipt-separated, unavailable to model code, and deleted from the
   runtime handoff before any training/scoring import.
9. Reproduce the disclosed `WHOLE-T4` versus `POST700-T4` descriptor-reference
   comparison. On the 27 training sessions only, require `POST700` to exceed
   `WHOLE` for both `a` and `c` aggregate correlation; otherwise the estimator
   route closes before GPU use.
10. Partition sessions by the first-30-trial nan-ignoring median
    `target_on -> go_cue` into the frozen no-delay (9 train), short-delay
    (5 train), and long-delay (13 train + 6 validation) eras. Publish `q_SE` statistics and reliability by era before
    any model score. All 27 training sessions are predeclared to train the FiLM
    head; era does not select membership.
11. For each raw `q_SE` column, compute an M30 odd/even-trial split-half
    reliability on the 27 training sessions. A column passes this first gate
    only if at least `24/27` session split pairs have finite full-rank fits, its
    Fisher-z aggregate correlation is `>= 0.50`, its median raw session
    correlation is `>= 0.40`, and at least `20/27` session correlations are
    positive.
12. Separately build an independent reference from rewarded source trials
    50-109. Compare first-M10 and reference `q_SE` rows within each training
    session. A column passes this deployment-budget gate only if its Fisher-z
    aggregate correlation is `>= 0.40`, median raw session correlation is
    `>= 0.30`, and at least `18/27` correlations are positive.
13. The column-retention rule is fixed before Stage 0: retain a column only if
    it passes both steps 11 and 12; otherwise replace it with exact positive
    zero in `SE-T4`, `PHASE-R` where applicable, and `ROW-SHUFFLE`. Publish the
    four-bit mask and a new design hash before any decoder/model import. If all
    four columns fail, close FiLM at Stage 0. Thresholds, windows, and failed
    columns may not be repaired after observing the result.
14. Report `corr(a_R,a_whole)` and `corr(c_R,c_whole)` within each session at
    M10 to distinguish a better direction-rate estimator from an additional
    state descriptor.
15. Compare correct versus direction-shuffled descriptors as a diagnostic, not
    a decoder-selection surface.
16. Test row-shuffle determinism and no fixed identity permutation.
17. Test pseudo-MUA refit order without running the conditional model.

The estimator and FiLM decisions are independent. Undefined geometry, leakage,
nonfinite values, rank failure, or failed step 9 closes the estimator route.
Failure of all columns in step 13 closes FiLM before implementation/training
but does not close `POST700-T4`. Stage 0 may not tune any window, threshold, or
column set against decoder R2. The already-inspected 300/700 ms descriptor
comparison is disclosed; no additional duration is permitted in V1.

### Stage 1 — Frozen-parent deployment-OOD screens in SUA

Stage 1 binds the locally present seed-specific M30/T4@30 parents at their
final epoch, `sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s{42,43,44}/epoch_ckpts/epoch_011.ckpt`
(SHA-bound in the work order), and keeps every base parameter frozen. It is a
screen, not a matched-training result.

`run_metadata.best_checkpoint` may not be used: it is an argmax over the noisy
per-epoch validation metric on the same six validation sessions that govern
this design, and the parent's own metadata forbids it for deterministic
protocols. `epoch_011.ckpt` is also the exact parent byte authority already
bound by the CP-FiLM V1 predecessor, so `WHOLE-NATIVE`/`EMPTY` rows remain
comparable across the two experiments.

**Estimator screen (descriptive, non-gating):** replay three no-update rows on
identical Q50 windows:

- `WHOLE-PARENTNORM`: whole-trial M10 T4 under the exact parent M30 normalizer;
- `WHOLE-M10NORM`: the same whole-trial M10 T4 under a source-fitted M10
  normalizer;
- `POST700-M10NORM`: R700 M10 T4 under its source-fitted M10 normalizer.

The descriptive contrast is `POST700-M10NORM - WHOLE-M10NORM`; the first row
isolates normalizer change. This screen does **not** gate Stage 2. The frozen
parent learned its `post_pool` mapping on whole-trial T4@30 z-scores, so a
`POST700` carrier is a different variable under that mapping, not merely a
budget shift; a negative or null frozen-parent contrast is therefore
uninformative about the matched estimator question and may not close the
route. The matched estimator branch of Stage 2 opens directly on Stage-0
step 9. The screen is still run because it is nearly free and provides the
validation-session analogue of the historical deployment-OOD table.

**FiLM screen:** only if at least one profile column survived Stage 0, keep
`c=WHOLE-T4@M10` under the exact parent M30 normalizer, replay
`WHOLE-NATIVE`, and train the four matched FiLM heads `EMPTY`, `PHASE-R`,
`SE-T4`, and `ROW-SHUFFLE` for 12 epochs. The parent, activity M30, Q50 query,
loss, batches, and non-profile RNG are identical. A positive mean
`D_semantic`, positive mean `D_attachment`, and at least `4/6` positive
semantic sessions open matched FiLM training; this is also a weak screen, not
the final efficacy gate.

For both screens, report the governing six-session validation result and
non-governing in-sample source-session effects stratified into the three eras.
Because all six validation sessions are long-delay, the latter are descriptive
only and cannot establish cross-era generalization.

The selected M30/T4@50 V9 checkpoints may not be named as parents because their
manifest-bound bytes are absent locally.

### Stage 2 — Conditional matched sparse-label training

The estimator branch is conditional on Stage-0 step 9 only. The FiLM branch is
conditional on the Stage-0 reliability mask and its Stage-1 opening predicate.

**Matched estimator branch:** construct paired clones from each held local
parent (`epoch_011.ckpt`); train `WHOLE-T4@M10` and `POST700-T4@M10` with their
separate source-training M10 normalizers, identical M30 activity/Q50 query, the
parent's trainable/frozen decoder policy, task loss, optimizer/LR, batches,
targets, dropout masks, and 12-epoch budget.

Disclosed asymmetry: both arms warm-start from a parent trained on whole-trial
T4@30, so the `WHOLE-T4@M10` arm begins with a carrier of the same variable
type while the `POST700-T4@M10` arm must re-learn its carrier mapping within
the same budget. This biases `D_estimator` against the candidate and is
accepted as conservative. A from-scratch paired replication is a permitted
additive successor, not a V1 substitute.

**Matched FiLM branch:** construct five paired clones using
`WHOLE-T4@M10` and its source-fitted M10 normalizer. Train the base according to
the parent policy in every arm; `WHOLE-NATIVE` has no FiLM head, while the four
FiLM arms have identical rank-8 head parameters and optimizer steps. Use the
frozen retention mask, with `PHASE-R=[a_R,c_R,m_R,+0]` and
`SE-T4=[a_R,c_R,m_R,Delta b]` before masking. No joint variant may change
`post_pool`/decoder trainability relative to the shared matched parent policy.

Every trained arm saves all epochs and governs by the fixed epoch 9-12
parameter average. Report validation contrasts and descriptive in-sample
effects by era. Only Stage 2 may support a matched sparse-label efficacy claim.

### Stage 3 — Conditional pseudo-MUA replication

Only after both SUA decisions are frozen:

- first repeat the two-arm matched estimator comparison in pseudo-MUA;
- instantiate the conditional five-arm FiLM design only if the SUA FiLM route
  passed;
- refit every carrier/profile after spike pooling;
- use the same M10/M30/Q50 and seed rules;
- report within-view effects before the cross-view interaction.

### Stage 4 — Optional retrospective stability score

External-15 may be scored once only after all method choices, gates, and state
SHAs are frozen. It must be labeled retrospective and cannot promote a failed
source-development result.

---

## 12. Training and numerical hygiene

### 12.1 No new optimization sweep

V1 freezes:

- 12 epochs;
- seeds 42/43/44;
- the seed-specific final-epoch `epoch_ckpts/epoch_011.ckpt` from each locally
  present M30/T4@30 parent as the held initialization authority (never the
  validation-argmax `best_checkpoint`);
- rank 8 for the conditional FiLM heads;
- existing task loss;
- existing parent optimizer/LR policy for matched training;
- fixed epoch 9-12 averaging.

No LR, rank, cue-window, loss, or weight-decay sweep is included.

### 12.2 Paired arm execution

Within a seed and resident batch:

- matched arms see the same source session and query/target tensors;
- dropout masks and stochastic operations are paired;
- in the Stage-1 estimator contrast, `POST700-M10NORM` versus
  `WHOLE-M10NORM`, the assigned rate estimator is the only scientific
  difference; `WHOLE-PARENTNORM` is an explicitly separate normalizer control;
- in the Stage-1 FiLM contrast, the arm-specific profile is the only scientific
  difference;
- in Stage 2, the matched estimator pair differs only in rate estimator and
  its necessarily estimator-specific source normalizer, while the matched
  FiLM matrix differs only in arm-specific profile/head presence;
- RNG state is restored/compared under the established paired-arm law.

In Stage 1 the three estimator rows are forward-only, `WHOLE-NATIVE` is
forward-only, and the four parameter-matched FiLM heads have identical
optimizer steps. In Stage 2 the two estimator arms have identical base-model
optimizer surfaces. In the matched FiLM branch, every arm has the identical
base optimizer surface and the four FiLM arms additionally have identical head
optimizer surfaces. Receipts compare counts within the correct matched set and
never invent FiLM steps for the native arm.

### 12.3 Heavy-tail protection

The previous FiLM runs showed volatile per-epoch mean loss. V1 does not change
the loss because that would confound the sparse-profile test. It uses a frozen
last-four-epoch parameter average for every trained arm instead of a last-epoch
point. Average every trainable floating parameter in zero-based epochs 8-11 in
float64, cast once to its declared storage dtype, and require all frozen and
non-floating state to remain identical to the held initialization. Optimizer
state is never averaged.

Huber loss or per-session normalized loss requires a successor factorial and
cannot be added only to the candidate.

### 12.4 Model/update receipts

Receipts must prove:

- exact trainable parameter names/counts per stage;
- no target-session optimizer or backward step;
- parent state before/after;
- per-epoch head/full-state SHA;
- averaged-state construction inputs and SHA;
- optimizer step counts;
- finite loss/gradient/parameter checks at sparse sentinel steps;
- target, carrier, profile, activity, and query digests.

Although `task_only` excludes the teacher term from the training loss, the
current validation path still computes teacher diagnostics. The exact teacher
checkpoint SHA is therefore a required dependency shared by all matched arms
and must be bound before model construction.

---

## 13. Primary estimands and gates

### 13.1 Per-cell quantities

For each seed and validation session, report the following on the Stage-2
matched surfaces. Stage-1 analogues carry an explicit `_OOD` suffix and cannot
replace them:

```text
D_estimator = R2(POST700-T4) - R2(WHOLE-T4)
D_semantic = R2(SE-T4) - R2(EMPTY)
D_baseline = R2(SE-T4) - R2(PHASE-R)
D_attachment = R2(SE-T4) - R2(ROW-SHUFFLE)
D_product = R2(SE-T4) - R2(WHOLE-NATIVE)
D_capacity_or_global_correction = R2(EMPTY) - R2(WHOLE-NATIVE)
```

Never substitute `D_product` for `D_semantic` in the mechanism claim.
`D_capacity_or_global_correction` is not pure capacity: an EMPTY head still sees
carrier `c` and can learn a global correction to its distribution.

### 13.2 Aggregation

1. Average the three paired seeds within each validation session.
2. Report all six session means.
3. Compute equal-session grand mean and median.
4. Run a deterministic 10,000-draw paired session bootstrap over the six
   seed-averaged sessions.
5. Report seed-specific grand means separately.

Windows may not be treated as independent bootstrap units.

With six sessions, a size-six bootstrap has only 462 distinct count vectors.
The bootstrap interval is retained as a transparent paired summary, not
presented as high-resolution asymptotic evidence.

### 13.3 SUA estimator gate

The Stage-2 matched `POST700-T4` route passes only if all are true:

1. mean `D_estimator >= +0.015 R2`;
2. at least `5/6` seed-averaged sessions have `D_estimator > 0`;
3. the paired session-bootstrap 95% lower bound for `D_estimator` is `> 0`;
4. worst seed-averaged session `D_estimator >= -0.030`;
5. every seed's equal-session `D_estimator` is nonnegative;
6. all implementation, label-budget, and no-leakage gates pass.

No gate requires reaching the historical M30/M50 score. The historical
M10-to-M30 deficit is not the denominator for this decision.

### 13.4 Conditional SUA FiLM gate

The FiLM route is first required to pass the Stage-0 reliability gate and its
Stage-1 opening predicate. Its Stage-2 matched decoder result passes only if all
are true:

1. mean `D_semantic >= +0.015 R2`;
2. at least `5/6` seed-averaged sessions have `D_semantic > 0`;
3. paired session-bootstrap 95% lower bound for `D_semantic` is `> 0`;
4. worst seed-averaged session `D_semantic >= -0.030`;
5. mean `D_attachment > 0`;
6. if `Delta b` survived Stage 0, mean `D_baseline > 0` for a
   hold-to-movement-baseline claim; if it did not survive, `SE-T4` and
   `PHASE-R` must be prediction-identical and no such claim is allowed;
7. every seed's equal-session `D_semantic` is nonnegative;
8. all zero-anchor, era, implementation, and no-leakage gates pass.

If semantic and attachment gates pass but retained `Delta b` has
`D_baseline <= 0`, the result may claim that movement-window T4 helps through
FiLM, but not that the hold-to-movement baseline contrast is useful.

### 13.5 pseudo-MUA gate

Pseudo-MUA is a replication, not a prerequisite for either SUA claim. Its
within-view estimator or semantic effect must nevertheless have:

- positive mean;
- positive paired-bootstrap lower bound;
- at least `4/6` positive session means;
- positive mean attachment contrast for the FiLM branch.

The same `+0.015` absolute threshold is reported but not required because the
pseudo-MUA budget gap and noise may differ.

---

## 14. Outcome taxonomy

### Outcome A — sparse estimator supported in SUA

`POST700-T4` passes the estimator gate over matched `WHOLE-T4`. This supports
the narrow conclusion that sparse-label T4 performance depends materially on
the firing-rate estimation window. It is not a FiLM result.

### Outcome B — estimator null

`POST700-T4` does not pass. Close the estimator route. Descriptor correlation
alone cannot promote it. The FiLM route remains governed solely by its separate
Stage-0 reliability decision.

### Outcome C — FiLM closes at Stage 0

All four `[a_R,c_R,m_R,Delta b]` columns fail the fixed reliability gates. Do
not train FiLM. Report that the proposed sparse-event descriptor is not
estimable reliably enough at M10 under this data surface.

### Outcome D — sparse FiLM mechanism supported in SUA

The conditional route passes semantic and attachment gates. It supports only
the retained, explicitly reported movement-window/baseline columns; it does not
license a claim about columns removed at Stage 0.

### Outcome E — movement-window profile helps, baseline contrast does not

`PHASE-R` is positive relative to `EMPTY`, while retained `Delta b` has
`D_baseline <= 0`. The useful effect is movement-window direction tuning, not
the hold-to-movement baseline contrast.

### Outcome F — capacity or global-carrier correction only

`EMPTY - WHOLE-NATIVE > 0`, but `SE-T4 - EMPTY <= 0`. The head may correct a
global carrier-distribution mismatch or add useful capacity; profile contents
are not supported.

### Outcome G — session-level but not row-attached

`SE-T4 - EMPTY > 0`, but `SE-T4 - ROW-SHUFFLE <= 0`. Profile distribution or
session context may help, but there is no evidence for per-channel attachment.

### Outcome H — pseudo-MUA also positive

Both views pass within-view semantic/attachment gates. The mechanism does not
require spike sorting under the tested pseudo-MUA construction.

### Outcome I — SUA positive, pseudo-MUA null

Interpretation: sorted-unit row identity preserves sparse-event tuning that
is blurred by electrode pooling. This is compatible with the existing absolute
SUA advantage.

### Outcome J — both views null

Close sparse-event T4 FiLM. Preserve the result as a boundary condition:
the admitted descriptor does not add usable information to the selected sparse
T4 carrier under this early-FiLM operator.

No hidden expansion to dense-speed CP, Top-K, late pooling, or arbitrary
profile search is permitted.

---

## 15. Controls against alternative explanations

### 15.1 Extra parameters

Controlled by `EMPTY`, which has identical FiLM capacity and training steps.

### 15.2 Session-level calibration rather than row alignment

Controlled by `ROW-SHUFFLE`, which preserves profile column distributions but
breaks unit/channel attachment.

### 15.3 More labels or later labels

All arms use the same first-M10 direction/cue horizon. No trial 10+ label is
available to the candidate.

### 15.4 More activity

All arms use the same chronological M30 activity support.

### 15.5 Easier query surface

All arms use identical Q50 starts, targets, and masks.

### 15.6 Dense velocity leakage

The descriptor materializer must expose no dense-behavior argument and must
record zero dense velocity acquisition/read calls. Tests must fail if a dense
velocity object is passed or accessed.

### 15.7 Different pseudo-MUA semantics

Pseudo-MUA descriptors are refitted after pooling; no SUA descriptor averaging
is allowed.

---

## 16. Artifact governance

### 16.1 Missing selected V9 checkpoint bytes

The selected T4@50 manifest binds three final checkpoint paths and SHA256
values, but the corresponding local `epoch_ckpts` directories contain zero
files. The historical receipts remain evidence of prior reported values, but
the selected artifact is not locally rerunnable.

Consequences:

- Stage 1 must use locally present, independently SHA-bound T4@30 parents;
- no document may call Stage 1 a selected V9 reproduction;
- Stage 1 is frozen-parent deployment-OOD evidence only;
- Stage 2 creates the new matched sparse-estimator and/or FiLM lineages;
- recovery of the exact selected bytes must match the historical SHAs and is a
  separate governance repair;
- old manifests/results may not be overwritten.

### 16.2 Immutable roots

Each executed stage requires a fresh root. Failures are terminal for that root;
no retry, deletion, replacement, or overwrite is permitted.

### 16.3 Descriptor-first receipts

Before model construction or GPU initialization, publish/hold:

- document/work-order hashes;
- source manifest and input authority;
- sparse-label ledger;
- phase-window law;
- normalizer authority;
- T4/profile digests;
- forbidden dense-velocity access evidence.

---

## 17. Two-week feasibility sequence

This is scheduling guidance, not execution authorization.

### Days 1-2

- implement a route-owned whole/POST700/H300/R700 spike-time materializer;
- reuse the reviewed ordinary T4 cosine fit;
- add half-open spike-time interval, label horizon, era, and no-dense-velocity
  tests.

### Day 3

- run CPU Stage 0 on the exact 33 source sessions;
- publish constructibility, rank/condition, estimator-reference comparison,
  era-stratified profile reliability, and label accounting;
- freeze the four-bit reliability mask under a new document hash, or stop the
  FiLM route if all columns fail.

### Days 4-5

- implement the frozen-parent estimator and conditional five-arm FiLM screens;
- add parent-normalizer/M10-normalizer controls, zero-anchor, paired RNG,
  row-shuffle, checkpoint-average, teacher-dependency, and lifecycle tests.

### Days 6-7

- independently audit Stage 0 and the executable work order;
- only then consider GPU authorization.

### Days 8-9

- run the three-seed SUA Stage-1 frozen-parent screens if authorized;
- freeze the FiLM opening decision; record the descriptive estimator screen.

### Days 10-12

- implement and run the Stage-2 matched estimator branch if Stage-0 step 9
  passed, and the matched FiLM branch only if its Stage-1 opening predicate
  passed, under a new work order.

### Days 13-14

- if the corresponding SUA route passes, execute the conditional pseudo-MUA
  estimator and/or FiLM replication;
- synthesize results with H1 sparse-event and existing SUA-vs-pseudo-MUA
  evidence without reopening external selection.

---

## 18. Explicitly deferred successors

### 18.1 Direct sparse-event profile concat

Feed retained `[a_R,c_R,m_R,Delta b]` as additional carrier columns or a
route-local carrier residual. This is an operator-capacity oracle, not part of
V1.

### 18.2 Two-state activity pooling

Pool neural identity features separately in H and R states before `post_pool`.
This changes the identity operator and is opened only if the profile is
constructible/informative but FiLM cannot use it.

### 18.3 M10-to-M50 source teacher

Train a source-only correction from causal T4@M10 and reliability fields toward
a T4@M50 teacher. Deployment remains M10, but source training consumes M50 and
must disclose the asymmetric label budget.

### 18.4 H1 sparse-event FiLM replication

Add an event-context FiLM to the existing H-SE5 substrate. This is a cross-
dataset replication only after DANDI V1 establishes a useful operator.

### 18.5 Event-only broad-phase profile

Use `[start,go_cue)` versus `[go_cue,stop)` duration-normalized rates. This has a
larger duration/nuisance surface than the fixed event windows and is not an
automatic fallback.

### 18.6 Peripheral-hold baseline

Replace center hold with `H=[stop_time-0.300,stop_time)`. It consumes no new
event annotation, but represents direction-specific peripheral posture rather
than neutral stationary hold. It is scientifically distinct and cannot replace
`H300` after a V1 result.

---

## 19. Strongest objection and answer

### Objection

`POST700-T4` was chosen after inspecting descriptor reliability, and
the sparse-event profile uses the same target-direction labels as its carrier while adding a
new parameterization plus a FiLM head. A positive decoder score might reflect
an estimator choice, a second phase observation, global correction, or added
capacity rather than new sparse-event information.

### Answer required from the experiment

1. `WHOLE-T4` versus `POST700-T4` is a separate no-extra-parameter estimator
   comparison; its 300/700 ms descriptor inspection is disclosed and no more
   durations may be tried.
2. `EMPTY` removes the extra-parameter/global-affine explanation.
3. `PHASE-R` separates movement-window direction tuning from the `Delta b`
   hold-to-movement baseline term.
4. `ROW-SHUFFLE` removes a distribution-only/session-context explanation.
5. Whole-trial T4 and the sparse-event profile use the same M10 directions, so the
   candidate receives no additional trial labels.
6. The FiLM route is forbidden unless at least one profile column passes both
   frozen reliability gates; failed columns are exact-zero masked before any
   decoder score.
7. Zero initialization makes every candidate start exactly at native.
8. A hold-to-movement baseline claim is allowed only if semantic, attachment,
   and `SE-T4 - PHASE-R` contrasts pass with `Delta b` retained.

If those controls fail, the narrower alternative explanation wins and the
corresponding claim is prohibited.

---

## 20. Independent review questions

Before an executable work order is frozen, an independent reviewer should
answer:

1. Does the descriptor path truly avoid all dense velocity/position reads?
2. Is one target direction plus event timestamps per M10 trial an honest sparse
   label accounting?
3. Are M30 activity, M10 labels, and Q50 query independently parameterized?
4. Are the exact continuous-time half-open windows deterministic across all
   source sessions and implemented through raw spike times?
5. Does `H300` use pre-target center hold in every era, with all 27 training
   sessions included and era-stratified reporting retained?
6. Does the phase-T4 code reuse ordinary T4 direction mapping and fitting?
7. Is raw `q_SE` fixed to `[a_R,c_R,m_R,Delta b]`, with only the predeclared
   per-column reliability mask allowed to zero columns?
8. Are estimator/profile normalizers separate, source-train-only, and bound?
9. Is T4-native bitwise reproduced at FiLM zero initialization?
10. Does `EMPTY` have exactly the same parameter count and optimizer steps?
11. Does `ROW-SHUFFLE` preserve all non-profile inputs and distributions?
12. Are seeds and epoch averaging paired and fixed before scoring?
13. Does Stage 0 independently close the estimator and FiLM routes under the
    declared reliability predicates?
14. Is Stage 1 honestly labeled frozen-parent deployment-OOD, with matched
    training claims reserved for Stage 2?
15. Is pseudo-MUA refitted after pooling rather than averaged from SUA?
16. Are external-15 and disputed formal sessions excluded from selection?
17. Are bootstrap units sessions rather than windows?
18. Is `+0.015` interpreted as a matched same-budget material-effect threshold,
    never as a fraction of the historical M10-to-M30 deployment deficit?
19. Are the missing selected V9 bytes treated as a governance blocker?
20. Are dense-speed CP, Top-K, late pooling, continual memory, and loss sweeps
    excluded from V1?

---

## 21. Final frozen summary

The V1 hypothesis is deliberately narrow:

```text
DANDI 000688 sub-C
  -> primary SUA
  -> first-M10 direction + target-on + go-cue labels only
  -> chronological M30 label-free activity
  -> Q50 query surface
  -> Stage 0: spike-time geometry + estimator check + decisive FiLM reliability
  -> Stage 1 frozen-parent OOD screens
  -> conditional Stage 2 matched WHOLE-T4@M10 vs POST700-T4@M10
  -> three paired seeds
  -> fixed epoch 9-12 parameter average
  -> estimator gate
  -> independently conditional all-era zero-init early-pool FiLM
  -> q_SE = retained [a_R,c_R,m_R,Delta b]
  -> WHOLE-NATIVE / EMPTY / PHASE-R / SE-T4 / ROW-SHUFFLE
  -> semantic + baseline + attachment gates
  -> conditional pseudo-MUA replication
```

The primary success criterion is a material paired estimator improvement under
the same M10 annotations. FiLM is a separately gated secondary mechanism.
Matching or exceeding a dense-kinematic or M30/M50 upper bound is not required.

---

## 22. Start sequence (Rev2)

This section tells the executing agent exactly what may begin now and in what
order. It authorizes CPU-only work; GPU use still requires a separate work
order and explicit authorization.

### 22.1 May begin immediately (CPU only, no result root)

1. Route-owned package, e.g. `sua_exploration/mc_maze/dandi688_sparse_event_t4_v1/`,
   with `plan.py` (frozen constants: `H300`, `R700`, `M=10`, `activity_support_n=30`,
   `query_start_trial=50`, era thresholds, reliability thresholds from Stage-0
   steps 11-12, parent `epoch_011.ckpt` paths and SHA256, teacher SHA256),
   `descriptors.py` (materializer), `reliability.py` (source-only audit
   namespace), `core.py` (mask/decision logic).
2. Materializer contract:
   - trial listing via `multisession_datamodule.list_datamodule_rewarded_trials`
     extended route-locally with `target_on_time` and `go_cue_time` read from
     the same `trials_df` row; do not modify the shared datamodule;
   - rates via `unit_side_features._pool_trial_rate_matrix` called with
     synthetic `start_time/stop_time` dicts for `H300` and `R700`; no bin grid;
   - fits via `unit_side_features._fit_cosine_tuning` and
     `_nearest_canonical_direction_index`; no reimplementation;
   - pseudo-MUA via `pool_trial_rates_by_electrode` before any fit;
   - the materializer signature must accept no velocity/position object; a
     test must assert that `nwb.processing["behavior"]` is never accessed by it
     (e.g. wrap/patch and count accesses).
3. Tests (pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, run under the
   `spint` conda environment):
   - half-open interval containment predicates of Section 5.3;
   - label-horizon: trial index >= 10 never enters candidate arrays;
   - era assignment reproduces 9 / 5 / 13+6 on the manifest;
   - `q_SE` column order and `Delta b = b_R - b_H`;
   - row-shuffle nonidentity and determinism;
   - zero-dense-velocity access;
   - reliability-audit namespace and candidate namespace share no arrays.
4. Stage-0 CPU run on the 33 manifest sessions only, writing to a fresh
   `sua_exploration/results/dandi688_sparse_event_t4_v1/stage0/` root with
   receipts for every item in Section 11 Stage 0 and Section 3.3.
5. Freeze the four-bit reliability mask and publish the new document hash.

### 22.2 May be implemented but not executed until GPU authorization

- Stage-1 forward-only rows and the four FiLM heads (reuse the CP-FiLM V1
  operator in `dandi688_cp_film_v1/core.py`: `build_film`, `film_identity`,
  zero-anchor law), with `PHASE-R` added and the profile source replaced.
- Stage-2 paired matched-training clones.

### 22.3 Forbidden during the start sequence

- opening any test-split NWB;
- modifying `unit_side_features.py`, `multisession_datamodule.py`, or any
  mainline config;
- reading decoder R2 of any arm;
- changing `H300`, `R700`, thresholds, or the column-retention rule after
  observing Stage-0 statistics.

### 22.4 Deliverables at the end of the start sequence

- Stage-0 receipt JSON with per-session, per-era, per-column statistics;
- the frozen mask and design hash;
- a one-page summary stating whether the estimator route (step 9) and the
  FiLM route (step 13) are open, closed, or partially open, with no decoder
  numbers.
