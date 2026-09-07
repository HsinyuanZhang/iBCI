# Design Review: Continual Calibration Next Routes for FALCON M2/M1

**Date:** 2026-09-02

**Status:** review draft, revision 3; **not** an execution work order

**Primary scope:** FALCON M2 and FALCON M1

**Out of scope:** DANDI 000688 experiments, H1 expansion, modification or
interruption of the currently running M1 rSyn3 `Z-Fix` / `S-Fix` jobs

**Purpose:** convert the proposed A--F directions into auditable hypotheses,
matched controls, causal deployment contracts, and predeclared stop rules.

---

## 1. Executive decision

The proposal contains three strong directions, one inexpensive successor, one
direction that needs a redesign, and one baseline-repair direction:

| Route | Role | Numerical potential | Research-story value | Decision |
|---|---|---:|---:|---|
| **A. Trial-free chunked activity memory** | label-free deployable method | small / unknown | **highest** | **GO after C/A0** |
| **READOUT. A10 closed-form readout adaptation** | labeled-calibration strong baseline | potentially highest | medium | **GO, M2 first** |
| **C. Continual-calibration curve** | mechanism characterization and paper figure | indirect | high | **immediate GO** |
| **D. MATCH moment alignment** | label-free domain-alignment diagnostic | uncertain | low--medium | **NO-GO as written; redesign first** |
| **E. Training-recipe repair** | matched baseline and fairness control | small--medium | low | **conditional GO; not a novelty claim** |
| **F. Causal EMA identity** | label-free fixed-effective-memory successor | negative in completed local screen | mechanism null | **STOP current EMA family; retain uncapped uniform** |

The recommended paper-facing composition is **C + A**:

> Trial-structured activity memory is locally useful, but official continuous
> streams do not expose trial boundaries. We replace trial events with causal,
> non-overlapping activity chunks and obtain boundary-free, label-free continual
> calibration, while explicitly measuring the performance--exposure--memory
> frontier.

READOUT is important but must be reported separately because it consumes
target calibration labels. Route D is not ready for execution. Route E protects
fair attribution but should not be presented as a new method.

---

## 2. Evidence already available

### 2.1 M1 activity headroom is real on the measured local fold

The frozen-weight M1 fold-20120924 activity-headroom receipt reports:

| Activity system | R2 | Delta vs static M10 | Causality/status |
|---|---:|---:|---|
| `STATIC_SUPPORT` | 0.5707439184 | -- | causal, cardinality matched |
| `ROLLING_FIXED_M` | 0.5842786431 | +0.0135347247 | causal, cardinality matched |
| `CAUSAL_GROWING_CAP30` | 0.5933129191 | +0.0225690007 | causal, cardinality OOD |
| `FULL_SESSION_ORACLE` | 0.5999125242 | +0.0291686058 | label-free but noncausal |

Authority:
`tfpd_exploration/results/m1_h1_activity_headroom_v1/m1_fold20120924.json`.

This establishes four limited but useful facts:

1. Updating activity memory can improve M1 without target optimization.
2. A fixed-cardinality rolling M10 memory already captures about +0.0135 R2.
3. Growing to cap30 adds about +0.0090 beyond rolling M10 on this fold.
4. The measured gap from causal growing-cap30 to the noncausal full-session
   activity oracle is only about +0.0066 on this fold.

It does **not** establish official-test improvement, multi-fold breadth, or
trial-free equivalence. The proposed M1 official expectation of roughly
`+0.006 to +0.014` is plausible but remains a prior, not a result. In
particular, the `+0.0066` cap30-to-full-oracle gap is **not** a hard upper bound
on Route A: rolling refresh at fixed capacity is a different operator from
growing cap30. The subsequently completed EMA family is reported separately
in Section 2.7 and does not advance.

### 2.2 Exact official M2 anchors and the matched comparator

The official M2 rows already include a cached first-30 activity identity:

| Submission | System | Held-out R2 |
|---|---|---:|
| `578218` | original packaged SPINT | 0.186479 |
| `578221` | submitted T4 | 0.303244 |
| `581362` | M30 T4 + first-30 cached B3S activity | 0.295161 |
| `581359` | M10 T4 + first-30 cached B3S activity | 0.263833 |
| `581361` | cue-budgeted M4 T4 + first-30 cached B3S activity | 0.289744 |

Submission `581359` is the mandatory primary official comparator for an M10
Route A candidate because it matches the target-label carrier budget,
checkpoint family, and initial first-30 activity information. It is
`cached-identity` with no adaptive state. Therefore trial-free *static*
deployment is not new: the remaining novelty is causal continual identity
refresh after that same initial state.

The official surface has not isolated activity expansion by itself:
`581359` versus `578221` changes both carrier budget (M10 versus M30) and
activity support (first 30 versus the packaged static contract). Consequently,
the lower `581359` score must not be interpreted as proof that activity30 is
harmful, and the local activity gain must not be counted again against an
original-SPINT or T4-only comparator.

Authority:
`tfpd_exploration/docs/HANDOFF_M2_SUA_COMPLETE_EXPERIMENT_AUDIT_20260828.md`
and
`sua_exploration/evalai_t4_m2_activity_budget/artifacts/evalai_submission_581359_terminal_receipt_v1.json`.

### 2.3 Cardinality OOD is an observed contract fact

The current growing-memory result explicitly records
`CAUSAL_CARDINALITY_OOD`: the identity encoder is evaluated with activity
cardinalities beyond the calibration cardinality represented by the frozen
training contract. Existing downward prefix cycles do not constitute upward
cardinality training.

This motivates a cardinality analysis but does not prove that upward
cardinality training is the only solution. Saturation can also arise from stale
history, pooling dilution, chunk/trial distribution mismatch, or genuinely
exhausted activity information.

### 2.4 Neural-only boundary reconstruction has already failed its source gate

The Causal Calibration Memory P1 audit measured three preregistered causal,
neural-only completed-trial boundary statistics. Their best per-source-session
AUC values were only `0.4866--0.5306`, all below the frozen `0.65` gate. This
does not prove that every conceivable boundary detector is impossible, but it
does reject the tested low-cost neural-only reconstruction family and gives a
strong reason to test deterministic chunks instead of silently assuming trial
events.

Authority:
`tfpd_exploration/docs/RESULT_CAUSAL_CALIBRATION_MEMORY_P1_20260831.md`.

### 2.5 The official decoder API is state-capable but trial-free

The packaged FALCON decoder exposes persistent decoder state across repeated
`predict()` calls and resets it in `reset()`. Neural observations arrive bin by
bin. Therefore causal chunk memory is structurally possible without trial
events.

However, official evaluation may run multiple data streams in one batch. A
valid implementation must maintain separate chunk and memory state for every
stream, honor reset/done behavior, and never use a global counter shared across
sessions.

### 2.6 The 20-epoch and 50-epoch M1 results are recipe-confounded

The existing phase-3 table explicitly records:

- the earlier 20-epoch route used constant learning rate `1e-5` and no
  scheduler;
- the 50-epoch route used warmup plus cosine with peak learning rate `1e-4`.

Thus the observed 20- versus 50-epoch difference is a cross-recipe comparison,
not a clean horizon ablation. It cannot yet be summarized as “50 epochs
overfits.”

Authority:
`tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/phase3_table/table_selected.json`.

### 2.7 Newly completed M2 memory-law and early-start results

Three local, inference-only M2 routes reached immutable terminal after the
initial version of this review. None claims the official contract.

#### Memory-law scan

Against `UNIFORM_CAP30`, the completed true-trial memory-law scan reports:

| Policy | within M4 delta | positive within sessions | external M4 delta | external M10 delta | Decision |
|---|---:|---:|---:|---:|---|
| `EMA_A080` | -0.06741 | 0/7 | -0.01126 | -0.02641 | fail |
| `EMA_A090` | -0.02219 | 0/7 | +0.00155 | +0.00004 | fail |
| `EMA_A095` | -0.00521 | 1/7 | -0.01416 | -0.00033 | fail |
| `UNIFORM_UNCAPPED` | **+0.01393** | **7/7** | 0.00000 | 0.00000 | selected locally |

Thus the tested EMA family is closed. Uniform uncapped history remains useful
on the longer within sessions, but the local external sessions do not expose
enough additional chronology for a difference from cap30. This supports a
continued-memory characterization, not an official gain claim.

The historical scan labels `1/(1-alpha)` as `effective_pool`; revision 3 treats
that field only as a time-constant proxy. The alpha-specific R2 measurements
remain valid, but that field must not be cited as Kish effective sample size.

#### Early-start and reblocking scans

Chronological-first-four decoding starts at realized trial 9 but is much worse
than the D-opt M4 support: D-opt minus chronological-four is `+0.11599`
external and `+0.02819` within for the CDM family. The follow-up first-10
reblocking route is also negative: its primary CDM-K4 external delta versus the
old first-30 law is `-0.02918`, positive in only 2/6 sessions. K-all remains
negative at `-0.02021`.

Consequences:

1. preserve first-30/D-opt support quality;
2. do not trade carrier/support selection quality for an earlier start;
3. do not rerun the completed EMA alpha family;
4. the next unmeasured mechanism is deterministic fixed-chunk activity versus
   the matched true-trial operator, initialized from the same first-30 state.

Authorities:
`tfpd_exploration/results/m2_memory_law_scan_v1/terminal.json`,
`tfpd_exploration/results/m2_chrono4_strict_v1/terminal.json`, and
`tfpd_exploration/results/m2_reblock10_v1/terminal.json`.

---

## 3. Reporting tiers

Every result must belong to exactly one tier.

### Tier 1: label-free deployment

Permitted target information:

- chronological neural activity already observed;
- dataset/session reset and done signals exposed by the evaluator;
- source-frozen model, normalizer, T4/carrier, hyperparameters, and chunk law.

Forbidden:

- target behavior labels;
- future target activity;
- full-session moments computed before earlier predictions;
- target-query R2 used to choose chunk length, phase, capacity, or checkpoint.

Routes A, C's causal arms, F, and a corrected D belong here.

### Tier 2: labeled calibration

Permitted target information:

- the officially authorized labeled calibration/support portion only;
- source-frozen ridge hyperparameters and rung definitions.

Forbidden:

- query labels for fitting, hyperparameter selection, or rung selection;
- bin-random cross-validation that leaks temporally adjacent samples.

READOUT belongs here.

### Oracle/diagnostic

Any use of future activity or query labels must be marked explicitly, for
example:

- `FULL_SESSION_ACTIVITY_ORACLE_NONCAUSAL`;
- `TARGET_LABEL_AFFINE_ORACLE`;
- `target_label_leakage=true`.

Oracle rows may bound headroom but may never select an official candidate.

---

## 4. Route C: continual-calibration curve

### 4.1 Scientific question

How does decoding quality change as a causal decoder consumes more unlabeled
activity, and how much of that change is due to exposure versus retained memory
capacity?

### 4.2 Why C comes first

C is inference-only and answers the decisions needed by A:

- whether cap30 is sufficient;
- whether cap60 is worth implementing;
- whether rolling recent memory beats growing memory;
- whether the current encoder benefits from or is harmed by cardinality OOD;
- how long a continual-calibration burn-in takes;
- whether the curve supports an official submission at all.

### 4.3 C-Pre: freeze the feasible grid from metadata before scoring

Before fixing any `(n,K)` grid, publish a source-metadata-only session-length
inventory: total chronological activity units, support units, remaining query
units, and the longest common suffix available for each proposed exposure.

This is mandatory because two of the six M2 post-33 sessions contain exactly
33 trials and therefore have no post-support query at all. A shared exposure
grid containing `n=120` cannot be constructed on every M2 session while also
preserving a common query suffix. By contrast, the measured M1 fold contains
414 trials and can support the full proposed grid.

The inventory, eligibility law, and reduced dataset-specific grid must be
frozen before target query labels are scored. An exposure may be omitted for
insufficient chronology; it may not be replaced after observing its R2.

Authority:
`sua_exploration/docs/M2_CPU_CORRECTION_AND_CARRIER_AUDIT_V1.md` and the M1
activity-headroom receipt cited above.

### 4.4 Separate exposure from capacity

Do not use one ambiguous axis called “cap.” Measure:

\[
R^2(n,K),
\]

where:

- `n` is the number of completed past activity units consumed;
- `K` is the maximum number retained in memory.

Suggested preregistered grid:

- exposure `n in {M, 30, 60, 120, all-past}` where available;
- capacity `K in {M, 30, 60, infinity}`.

The exact grid must be the output of C-Pre. It may be reduced when a dataset
has fewer observations, but never after scoring target query labels.

### 4.5 Required systems

At minimum:

1. `STATIC_SUPPORT_M`;
2. `ROLLING_RECENT_M`;
3. `ROLLING_RECENT_30`;
4. `EMA_EFFECTIVE_M`, with source-frozen decay;
5. `GROWING_CAP30`;
6. `GROWING_CAP60`, if enough past activity exists;
7. `CAUSAL_ALL_PAST`;
8. `FULL_SESSION_ACTIVITY_ORACLE_NONCAUSAL`.

`CAUSAL_ALL_PAST` and `FULL_SESSION_ACTIVITY_ORACLE_NONCAUSAL` are different
systems and must never share a label.

### 4.6 Common-query requirement

All systems in a primary comparison must be scored on the same query suffix.
For example, a comparison involving exposure 60 cannot score cap10 on early
trials and cap60 on later trials. That would confound memory with temporal
drift.

Required reporting:

- a common-suffix paired table;
- an optional prequential curve covering the whole chronological stream;
- per-session deltas, not only a pooled average;
- both equal-session and pooled-bin summaries when session lengths differ.

### 4.7 Interpretation law

- A rising curve supports continual activity adaptation.
- A cap30 plateau indicates little immediate benefit from retaining more raw
  history under the current frozen operator; it does not identify the cause.
- A drop beyond the training cardinality is evidence consistent with
  cardinality mismatch, but it is not by itself proof that upward-cardinality
  training will fix it.
- Upward-cardinality training may be authorized only after the curve and a
  source-side diagnostic show that additional activity contains usable signal
  that the current encoder fails to exploit.
- EMA must be tested before upward-cardinality training because it can expose
  unlimited causal history while keeping statistical effective memory near the
  training cardinality.

### 4.8 Deliverable

One paper-ready figure per dataset:

- x-axis: causal activity exposure;
- line/color: retained capacity;
- y-axis: governing R2 on a matched query surface;
- noncausal oracle shown as a dashed upper bound;
- explicit vertical or annotation marks for cardinality-matched versus OOD
  regions.

---

## 5. Route A: trial-free chunked activity memory

### 5.1 Scientific question

Can non-overlapping causal activity chunks replace true trial boundaries while
preserving the downstream identity and R2 benefit of activity memory?

This is not merely an engineering convenience. The preregistered low-cost
neural-only boundary statistics failed their `0.65` source AUC gate
(`0.4866--0.5306`), so deterministic chunks are the cheapest remaining causal
construction that does not invent unavailable behavioral events.

### 5.2 Proposed deployment system

Primary official candidate:

`M10 T4 + causal non-overlapping activity chunks + rolling/growing cap30`.

The candidate must initialize from the same first-30 cached B3S activity state
as official submission `581359`. Its first prediction before any continual
update must reproduce the matched static comparator. Route A's novelty is the
subsequent **continual refresh**, not the existence of a trial-free cached
identity.

M2 is the primary dataset. M1 is a secondary transfer evaluation.

No target labels, optimizer steps, backward passes, or model-state changes are
permitted.

### 5.3 State machine

For every parallel stream `s`, maintain separately:

- current incomplete chunk buffer;
- number of valid bins accumulated in that chunk;
- completed-chunk activity memory;
- memory capacity and eviction state;
- current cached identity;
- reset/done status.

For time bin `t`:

1. predict using only support plus chunks completed strictly before `t`;
2. append the newly observed neural bin to stream `s`'s current chunk;
3. if the chunk becomes complete, finalize it after the prediction;
4. update the memory and identity for future bins only;
5. never reuse overlapping window bins as separate memory observations.

The implementation must define handling for padding, invalid bins, partial
final chunks, stream completion, and asynchronous batch members.

### 5.4 A0 source-selected chunk law

Chunk length, stride, padding rule, and phase must be frozen using source-only
information. The six local M2 target sessions may evaluate noninferiority but
may not choose the winning chunk geometry.

At least two preregistered phase offsets should be audited to reveal phase
sensitivity. If only one phase is legal under the evaluator contract, that fact
must be documented before query scoring.

### 5.5 A0 noninferiority experiment

Compare, on identical model/checkpoint/query bins:

- `TRUE_TRIAL_MEMORY`;
- `TRIAL_FREE_FIXED_CHUNK_MEMORY`.

Keep T4, initial support, activity capacity, normalization, batch ordering, and
metric identical.

Primary noninferiority release gate, with
`delta = R2(chunk) - R2(true-trial)`:

- mean delta at least `-0.005`;
- worst-session delta at least `-0.01`;
- all causal and state-isolation invariants pass;
- no target labels or future activity are read;
- chunk geometry was source-selected.

If chunks are better than true-trial memory by more than `+0.005` on average,
do **not** call the operators equivalent and do not reject the result. Register
it as a separate positive result and test the predeclared explanations,
especially segmentation density, recency, and boundary semantics.

Secondary evidence:

- per-session identity cosine and norm ratio;
- per-session prediction digest;
- phase sensitivity;
- time and memory complexity;
- chunk completion/update counts.

Identity equality is diagnostic. Downstream R2 noninferiority is governing;
material superiority is retained as a separate result.

### 5.6 Capacity selection

The first official candidate should use cap30 unless Route C supplies a
source-selected, matched-surface reason to prefer cap60.

Do not use causal all-history as the first official candidate because:

- runtime can grow with stream length;
- repeated full-history identity reconstruction can become expensive;
- cardinality is increasingly OOD;
- official-test submissions must not be used as a hyperparameter sweep.

### 5.7 Official evaluation and stop law

One source-selected primary submission per dataset:

1. M2 first;
2. M1 second if A0 passes and resources permit.

Primary comparator: official submission `581359`, exactly matched on M10
target-label carrier budget, checkpoint, first-30 cached activity identity,
normalization, and evaluation surface. The candidate's first pre-update output
must match this comparator. Secondary context rows may include `578221`,
`581362`, and original SPINT, but none may be used to recount an activity30
gain already present in `581359`.

Pragmatic continuation threshold:

- official delta at least `+0.01` to justify further engineering.

Scientific interpretation:

- an official delta below `+0.01` stops deployment optimization, but does not
  erase a valid local paired mechanism result;
- the scientific claim depends on the local multi-session paired evidence;
- the official point validates end-to-end compatibility and the direction of
  transfer, not statistical separation by itself.

The `581359` held-out session standard deviation is about `0.099`; with roughly
six sessions its marginal standard error is about `0.040`. The paired official
standard error is unavailable. Therefore `+0.01` is deliberately an
engineering continuation threshold below the marginal noise scale, not a
significance threshold and not sufficient evidence for a scientific transfer
claim.

### 5.8 Strongest objection

**Objection:** fixed chunks are not behaviorally homogeneous trials, so their
activity distribution may not match the identity encoder's calibration
contract.

**Response:** that is exactly what A0 tests. No architecture or training claim
is permitted unless source-selected fixed chunks preserve downstream utility
across the preregistered local sessions.

### 5.9 Route F: causal exponentially weighted identity (completed local null)

Route F tested whether the observed plateau was caused by stale-history
dilution rather than lack of additional activity information. It replaced the
unweighted growing mean with

\[
m_t = \alpha m_{t-1} + (1-\alpha)\phi_t,
\]

initialized from the same support feature state. The tested alphas were
`0.80`, `0.90`, and `0.95`. The operator is label-free, causal, constant-memory,
and has no learned target parameter, but none of the three alphas passed the
primary local M4 gate.

For independent equally weighted observations under the normalized EMA law,
the long-run Kish effective sample size is

\[
N_{\mathrm{eff}} = \frac{1+\alpha}{1-\alpha}.
\]

To match a training cardinality `M`, the principled anchor is
`alpha = (M-1)/(M+1)`. The expression `1/(1-alpha)` is a time-constant proxy,
not the statistical effective sample size, and must not be used as the
cardinality proof.

Required Route A/F matrix on every feasible common suffix:

1. `A0-STATIC581359`;
2. `A1-FIFO30`;
3. `F1-EMA_EFFECTIVE_M`;
4. `A2-GROW60`, where feasible;
5. `A3-ALLPAST`, where feasible;
6. `O-FULLSESSION`, explicitly noncausal.

This matrix is now partly historical: `F1` is a completed null, while
`UNIFORM_UNCAPPED` is the surviving true-trial memory law. Do not run another
target-selected alpha sweep. A future EMA successor would require a genuinely
new source-side law and independent motivation, not a denser alpha grid.

---

## 6. READOUT: A10 closed-form readout adaptation

### 6.1 Scientific question

How much authorized dense calibration supervision remains unused when T4
compresses many labeled bins into a small carrier, and can that information be
recovered by a closed-form output-layer update without target backpropagation?

### 6.2 Scope and reporting tier

- M2 first;
- target calibration/support labels are allowed;
- target query labels are forbidden for fitting or selection;
- report as **Tier 2 labeled calibration**, separate from label-free A/CDM;
- M1 is lower priority because prior M10 DirectRidge evidence is weak and
  small-sample affine estimation has been unstable.

Dataset output dimensions and semantics must be explicit: M2 uses `C=2`
velocity covariates, M1 uses `C=16` EMG covariates, and H1 uses `C=7`. Do not
describe M1 calibration labels as velocity.

### 6.3 Frozen components

Freeze:

- neural encoder;
- identity encoder;
- carrier/T4 path;
- normalizer;
- all hidden representations;
- source-selected regularization and rung definitions.

Only the output-coordinate map is fit on authorized target support.

### 6.4 Preregistered rungs

#### R0-NONE: no adaptation

Exact original output head. This must reproduce the sealed baseline.

#### R1-DIAG: per-covariate affine

For `C` output covariates:

\[
\hat y'_c = a_c \hat y_c + b_c,
\]

for `2C` fitted parameters total.

#### R2-AFFINE: full affine

\[
\hat{\mathbf y}' = A\hat{\mathbf y} + \mathbf b,
\]

where `A` is `C x C` and `b` is `C`-dimensional, for `C^2+C` parameters.

#### R3-HEADDELTA: output-head delta

For frozen hidden representation `h in R^H`:

\[
\hat{\mathbf y}' = (W + \Delta W)h + (b + \Delta b),
\]

with `Delta W in R^(C x H)` and optional `Delta b in R^C`, fit by ridge around
the source head. This is `C*H` parameters without per-covariate intercepts or
`C*(H+1)` with them.

The source-selected ridge penalty must penalize the delta, not refit an
unanchored head unless an explicit unanchored control is preregistered.

### 6.5 Effective sample disclosure

Report all of:

- number of labeled trials;
- number of valid labeled bins;
- temporal block length used for cross-validation;
- effective number of blocks;
- hidden dimension and fitted parameter count;
- design rank and condition number;
- ridge penalty selected on source data.

Bins from the same trajectory are correlated and must not be described as
independent samples.

### 6.6 Hyperparameter-selection law

Choose ridge penalties and the primary rung using source LOSO or source
grouped-OOF only. Do not use target query R2 to choose among R1/R2/R3.

All preregistered rungs may be reported descriptively on target query, but the
primary row must be frozen in advance.

### 6.7 Controls

1. `DELTA_ZERO`: exact baseline parity.
2. `TRIAL_SHUFFLE`: permute whole-trial behavior assignments or use
   within-trial circular shifts; do not shuffle individual bins independently.
3. `SOURCE_GROUPED_OOF_REFIT`: test whether the rule generalizes beyond its
   support fit.
4. `UNANCHORED_HEAD`, optional descriptive control, must not replace the
   anchored primary ridge.

The source-side refit is not required to be bitwise no-op. It is required not
to produce a systematic source-query loss and not to rely on sample-in support
fit quality.

### 6.8 Stop law

Close the entire readout-adaptation line if the source-selected densest rung:

- improves local held-out query R2 by less than `+0.03` on average; or
- is positive in fewer than 4/6 local sessions; or
- does not exceed the trial-preserving shuffle distribution; or
- has gains driven only by support/sample-in fit.

The exact breadth denominator must match the available preregistered M2 local
session set.

### 6.9 Claim boundary

A positive READOUT result supports efficient use of labeled calibration, not
label-free continual adaptation. Comparisons with fully supervised few-shot
methods must disclose model, source training, calibration size, and evaluation
surface differences.

---

## 7. Route D: MATCH moment alignment

### 7.1 Decision on the current proposal

The current gate -- “target input-moment shift exceeds source-session shift by
2 standard deviations” -- is insufficient. Detectable shift is not evidence
that forcing moment equality improves decoding.

The route remains NO-GO until it has a source-side causal utility gate.

### 7.2 Structural risks

1. Neural channel indices may not be functionally aligned across sessions.
   Per-channel source/target moment matching can therefore align unrelated
   units.
2. Source-frozen normalization may already transform these moments; a second
   alignment can double-correct them.
3. Applying a transform only to query activity but not calibration activity
   places the identity and query paths in different coordinate systems.
4. Full-target moment computation before early predictions is noncausal.
5. The original proposal is a shared **per-bin** diagonal affine, not a
   per-unit affine. A per-bin transform can be folded into the first linear
   layer if it is defined on the entire waveform entering `fc_in`. However,
   SPINT computes `fc_in(x + e)`, where `e` is the activity identity. For a
   desired raw-input-only transform `Ax+c` while keeping `e` unchanged,
   `W_1(Ax+c+e)+b_1` is not reproduced by the naive substitution
   `W'_1=W_1A`, because that instead yields
   `W_1A(x+e)+W_1c+b_1`; the difference is `W_1(A-I)e`. Exact folding
   therefore requires either applying the transform to the full `x+e`
   waveform, transforming the identity consistently, or retaining an explicit
   inexpensive preprocessing operator. Per-unit affine transforms remain
   generally non-foldable into one shared `fc_in`.

### 7.3 Required redesigned pre-gate

Run source-session LOSO pseudo-target evaluation:

1. hold out one source session as a pseudo target;
2. estimate moments using only its first M unlabeled chunks/trials;
3. freeze the transform;
4. transform both calibration and future query activity consistently;
5. score only its future query portion;
6. repeat over source sessions.

Candidate locations:

- `MATCH-X`: shared per-bin transform of raw `x`, with an explicit operator and
  unchanged identity;
- `MATCH-Z`: shared per-bin transform of the full `x+identity` waveform, which
  can be folded exactly but changes both paths together;
- global raw-input robust mean/scale;
- distribution across units rather than unit-index correspondence;
- permutation-invariant identity representation;
- pre-output hidden representation, if causally estimable.

### 7.4 Release gate

Target evaluation is authorized only if a preregistered transform achieves:

- source-LOSO mean delta at least `+0.01`;
- positive delta on a majority of source folds;
- no future activity and no labels;
- stable clipping/conditioning;
- identical transformation law for calibration and query paths.

Moment shift greater than `2 sigma` remains descriptive evidence only.

---

## 8. Route E: matched training-recipe repair

### 8.1 Scientific role

E protects the fairness and strength of all later comparisons. It is not a
standalone algorithmic contribution.

### 8.2 Claim correction

Do not state that “50 epochs overfit compared with 20 epochs” until horizon is
isolated under an otherwise identical recipe. The existing comparison changes
learning-rate magnitude and scheduler as well as duration.

### 8.3 Minimal matched schedule experiment

Hold fixed:

- seed;
- data exposure and batch sampler;
- architecture;
- loss;
- optimizer;
- learning-rate law;
- dropout law;
- checkpoint-selection surface.

Compare checkpoints/horizons at `10`, `15`, and `20` epochs. A single longer
run may supply these checkpoints only if its learning-rate trajectory up to
each checkpoint is exactly the intended matched trajectory.

Checkpoint and recipe selection must use source grouped-OOF or the frozen
source validation rule, never the formal target query.

### 8.4 Learning-rate experiment is separate

If needed, compare separately:

- constant `1e-5`;
- constant `1e-4`;
- a frozen warmup/cosine schedule.

Do not combine horizon and LR changes into one causal interpretation.

### 8.5 Dropout experiment is separate

The existing M1 route already contains dynamic whole-unit dropout with a broad
dropout-probability distribution. “Stronger dropout” is therefore undefined
until a specific operator is named.

Possible later cells include:

- a different predeclared probability distribution;
- calibration-prefix dropout;
- paired anchored calibration dropout;
- path-specific activity versus calibration dropout.

Each is a distinct intervention and requires a matched T0. Do not bundle it
with the initial horizon sweep.

### 8.6 Interaction with the running rSyn3 experiment

The current GPU1 M1 `Z-Fix` / `S-Fix` jobs use a different 12-epoch recipe,
fold-local configuration, and carrier topology. They cannot provide a causal
horizon comparison against the old 20- or 50-epoch routes. Do not use their
result to claim that a shorter horizon is better or that 50 epochs overfit.

No new E job should contend with, modify, or restart those jobs. Their terminal
receipts may inform resource scheduling and method performance only; a horizon
claim still requires the matched experiment in Section 8.3.

---

## 9. Revised execution queue

This document does not authorize execution. A later work order should follow
this dependency order.

### Stage 0: protect current work

- Do not signal, restart, modify, or contend with GPU1 M1 rSyn3 training.
- Do not use its incomplete training loss as a method comparison.
- Wait for matched `Z-Fix` / `S-Fix` terminal and R2 receipts before changing
  the M1 baseline plan.

### Stage 1: C-Pre metadata inventory

- Publish the per-session chronology/support/query inventory without scoring
  target query labels.
- Freeze the feasible exposure grid and common-suffix eligibility per dataset.
- Do not assume M2 can support `n=60` or `n=120` merely because M1 can.

### Stage 2: C0 true-trial continual-calibration characterization

- Datasets: M2 local sessions first; M1 measured folds second.
- Frozen weights, no target optimization.
- Separate exposure and capacity.
- Use common query suffixes.
- Bind the completed cap30/uncapped/EMA receipts rather than rerunning them;
  add only missing feasible exposure points identified by C-Pre.
- Produce the paper-ready curve and capacity decision.

### Stage 3: A0 trial-free noninferiority

- Implement per-stream chunk state.
- Freeze chunk law from source only.
- Compare against true-trial memory on identical query bins.
- Enforce mean delta `>= -0.005` and worst-session delta `>= -0.01`.
- If chunks materially beat trials, register a distinct positive result rather
  than rejecting “equivalence.”

### Stage 4: READOUT R0--R3 closed-form local screen

- M2 only initially.
- Run R0-NONE/R1-DIAG/R2-AFFINE/R3-HEADDELTA with source-frozen
  regularization.
- Include trial-preserving shuffle and grouped-OOF controls.
- Stop the route if the dense rung fails the `+0.03` / breadth / shuffle gate.

### Stage 5: one official A submission per dataset

- Submit only one source-selected state law, cap, and chunk law.
- Compare the M10 candidate primarily against official `581359`.
- Require first-prediction parity with `581359` before continual updates.
- M2 first.
- M1 second only if local noninferiority and resource gates pass.
- Never tune capacity from the official score.

### Stage 6: E matched recipe, if still needed

- Isolate epoch horizon before LR or dropout.
- Do not treat the completed rSyn3 route as a horizon ablation.
- Keep the result as baseline repair.

### Stage 7: D redesigned CPU screen

- Source LOSO and causal prefix only.
- No target run without source-side utility.

---

## 10. Required decision table

| Observation | Decision |
|---|---|
| C shows no causal activity benefit on matched suffixes | Stop A official route; retain boundary result |
| True-trial helps but fixed chunks fail A0 noninferiority | Do not submit A; investigate source-selected segmentation only if independently justified |
| Fixed chunks materially beat true trials | Keep the positive result, label it non-equivalent, and test recency/density explanations |
| A0 passes and cap30 is near the curve optimum | Submit one cap30 candidate |
| A0 passes and cap60 is source-selected, materially better, and computationally bounded | Submit cap60 instead of cap30, not both as a test sweep |
| Completed EMA family remains null while uncapped uniform wins locally | Close EMA tuning; retain uncapped uniform as the true-trial reference |
| R3 gains at least +0.03, passes breadth, and beats shuffle | Advance Tier-2 READOUT adaptation |
| R3 fails monotonicity/utility/shuffle gates | Close all A10 READOUT rungs |
| D detects large shift but source-LOSO correction is null/negative | Stop D; do not treat shift magnitude as headroom |
| Short matched schedule improves source-OOF and held-out without target selection | Replace the baseline recipe and rerun only necessary matched arms |
| All new routes are null | Stop experimentation and write the measured applicability boundary |

---

## 11. Paper-facing outcomes

### Outcome A: continual boundary-free memory succeeds

Primary contribution:

> Starting from the same cached calibration identity as the matched official
> baseline, boundary-free continual neural identity refresh preserves or
> improves the benefit of activity memory under a streaming interface.

Required evidence:

- true-trial/chunk noninferiority, or an explicitly non-equivalent positive
  result;
- exposure-capacity curve;
- official M2 delta against `581359`;
- M1 transfer if available;
- label-free causal audit.

### Outcome B: A fails, READOUT succeeds

Primary practical result:

> Dense labels inside a small number of calibration trials contain substantial
> unused output-coordinate information that can be recovered by a closed-form,
> source-regularized readout update.

This is a Tier-2 claim and must not be described as label-free CDM.

### Outcome C: activity helps only with true trials

Boundary result:

> Activity memory is effective under explicit behavioral episodes but does not
> transfer to boundary-free streaming through naive fixed segmentation.

This is scientifically useful because it identifies the deployment assumption
on which the mechanism depends.

### Outcome D: all methods are null

Write rather than continue an open-ended search:

- measured activity curve and ceiling;
- cardinality-matched/OOD boundary;
- readout-adaptation upper bound;
- session-shift diagnostics;
- failure of label-free correction under the official contract.

---

## 12. Audit questions

Reviewers should answer these before a work order is frozen:

1. Does the official/local evaluator preserve chronological per-stream state,
   and are reset/done semantics completely specified?
2. Is the chunk law source-selected, causal, non-overlapping, and independent
   for every batch stream?
3. Are activity exposure and retained capacity separated in Route C?
4. Was the feasible `(n,K)` grid frozen from the C-Pre metadata inventory
   before any target query label was scored?
5. Are all primary curve comparisons scored on common query bins?
6. Is `CAUSAL_ALL_PAST` distinguished from the noncausal full-session oracle?
7. Does every M10 official A delta use `581359` as its primary matched
   comparator and prove first-prediction parity before continual updates?
8. Is EMA decay source-selected, and is its claimed effective cardinality based
   on `(1+alpha)/(1-alpha)` rather than a time-constant proxy?
9. Does READOUT fit only authorized support labels and select all
   hyperparameters on source data?
10. Are shuffle controls trial-preserving rather than bin-independent?
11. Is the R3-HEADDELTA dimension parameterized by `C` and is its anchoring
   contract explicit?
12. Does Route D prove source-LOSO utility rather than merely detect shift?
13. Are calibration and query activity transformed consistently in Route D?
14. Does Route E isolate horizon, LR, and dropout into separate interventions?
15. Are Tier 1, Tier 2, and oracle rows separated in every table and claim?
16. Is one official candidate frozen before submission, with no leaderboard
    hyperparameter sweep?
17. Does every route have a terminal null-result node rather than an automatic
    successor experiment?

---

## 13. Final recommendation

Approve preparation of two small, non-GPU work orders after the currently
running M1 jobs reach terminal:

1. **C-Pre + C0/A0:** metadata-constrained completion of the
   continual-calibration curve plus trial-free chunk noninferiority; bind the
   completed EMA null and uncapped-uniform result rather than rerunning them;
2. **READOUT R0--R3:** M2 closed-form output readout screen.

Do not yet authorize:

- an official A submission before A0;
- causal all-history as the primary deployment candidate;
- D/MATCH target evaluation under the current 2-sigma-only gate;
- an epoch-plus-LR-plus-dropout bundle;
- any claim that 50 epochs alone caused the prior M1 degradation.

The highest-value remaining method claim is continual chunk-based A over
matched official baseline `581359`; the highest-value characterization is C,
and the strongest likely numerical baseline is READOUT. The current EMA/F
family is a completed local null.
