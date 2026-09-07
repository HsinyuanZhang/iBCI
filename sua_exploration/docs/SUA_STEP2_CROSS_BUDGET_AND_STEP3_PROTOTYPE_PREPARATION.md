# SUA Step 2 / Step 3 preparation: causal budget correction and fixed-K temporal prototypes

**Prepared:** 2026-08-02 (Asia/Hong_Kong)  
**Status:** design and implementation-boundary audit only.  This document does **not** authorize a
GPU launch, a formal SUA test read, a change to the active factorized-T4-residual experiment, or
an M1 representation restart.

## 1. Decision context and containment

The preceding SUA evidence is useful for narrowing this work:

- selected, aligned T4 is the viable SUA substrate;
- static SNR/waveform concat, electrode lookup, same-electrode relation, confidence-FiLM and the
  present spatial-prior route did not establish a controlled gain;
- the active low-rank T4 attention-logit residual has its own frozen seed-42 protocol.  Its
  result must not be used to tune either proposal below;
- the M1 D4/DS4 local gate and the source-only E4 Gate A have stopped.  Neither is a reason to
  repurpose M1 as a second development endpoint here.

The two ideas below answer different, falsifiable questions.

1. **Step 2 — estimator-aware cross-budget correction:** can a source-trained, zero-backprop
   correction reduce *the actual small-sample error of a causal T4 estimator* at a small number of
   labelled calibration trials?
2. **Step 3 — fixed-K temporal-prototype memory:** before any decoder experiment, do compact,
   neural-only, temporally routed sufficient statistics carry source-LOSO predictive information
   beyond rate and a shuffled-slot control?

They are not two arms of one hyperparameter search.  In particular, Step 2 is a label-efficiency
question for an already useful carrier; Step 3 is a new representation question.  They shall not
be launched in parallel or advanced automatically.

### Serialized decision rule

```text
active factorized-T4 residual screen finishes unchanged
    -> Step-2 CPU estimator / cache / leakage audit
       -> explicit review: stop, or authorize the bounded Step-2B validation pilot
          -> written Step-2 decision (including the no-dropout / TS4 controls)
             -> only then may Step-3 CPU Gate A be considered
                -> separate review before any Step-3 decoder pilot
```

The Step-3 Gate A is deliberately CPU-only, but it remains blocked until Step 2 receives a written
stop/no-go or a written pilot decision.  If Step 2's CPU audit finds low-budget estimates
undefined or its review declines the pilot, that written stop unblocks Step 3; it does not make
Step 3 positive by default.  A positive Step-2 CPU audit gives its one validation pilot priority
and leaves Step 3 dormant.  No formal-test sessions are opened anywhere in this sequence.

This ordering prevents the common failure mode of turning two vague mechanisms into a simultaneous
GPU lottery.  It also keeps the active residual's files, outputs and provenance outside both
branches.

## 2. Step 2 — causal cross-budget T4 correction

### 2.1 Question, time boundary, and two different budgets

Let `r_it` be unit `i`'s rate on the chronological rewarded calibration trial `t`, and let
`theta_t` be that trial's target direction.  For each

```text
M_T4 in {10, 15, 20, 50},
```

fit the ordinary T4 descriptor directly from the **first `M_T4` actual rewarded trials**:

```text
T_i^(M) = cosine_fit({(r_it, theta_t): 0 <= t < M}).
```

The fit must use the same rewarded-trial filter, chronological `first` rule, direction mapping and
per-trial-rate semantics as the existing T4 computation.  This is crucial: independently adding
noise to four cached T4 coefficients is *not* an estimator-aware corruption experiment.  It loses
the dependence on direction coverage, exposure, rank, rate heteroscedasticity and the shared
design matrix.

All decoder evaluations begin at trial 50.  Thus every candidate sees the same post-calibration
evaluation window.  The standard online neural-activity calibration remains fixed at
`M_activity=30`; only the number of trials whose labels/rates enter the T4 fit varies.  This
isolates **labelled T4 calibration budget**, rather than silently changing the activity support
budget as well.

There are two label concepts which must never be conflated:

| Quantity | Allowed use | Not allowed use |
|---|---|---|
| `M_T4=M` labels in the first `M` trials of the evaluated session | Fits `T^(M)` and all reliability fields derived from that same fit.  This is the deployment label budget. | Query labels; labels from trials `>=M`; a hidden `M=50` feature at inference. |
| First 50 labels on a **source training session** | Offline target/teacher for learning a session-independent correction; all source views may be constructed before training. | Any held-out validation/session query label, selection on a formal test, or an input required at low-budget deployment. |
| First 50 labels on the held-out validation session | Builds the ordinary `T^(50)` *boundary arm* and offline descriptor-consistency diagnostic. | Input to a model advertised as `M=10/15/20` deployment. |

`T^(50)` is therefore a teacher/consistency boundary, not a valid low-label inference feature.  The
fact that evaluation starts at 50 makes the boundary causal for its own `M=50` arm, but does not
make it available to `M<50` deployment.

### 2.2 Minimal estimator-aware mechanism

This subsection is **Step 2A: correction-only**.  It is the smallest causal control: it asks
whether the error of the low-budget estimator is measurable and predictable at all.  It is not
the full modality-dropout/cross-budget robust-decoder proposal described in section 2.5.

For a low-budget fit, retain only statistics computable from the exact same first `M` trials:

```text
q_i^(M) = [fit residual/deviance, labelled spike count, labelled time exposure,
           total-prefix spike count, total-prefix time exposure,
           modulation-to-residual, design rank, log condition,
           directional count histogram],
q_global^(M) = [M, direction balance, design rank, log condition].
```

The exact final subset is not yet selected; the CPU audit reports each field's degeneracy and
cross-budget predictability before any learning.  A source-trained shared per-unit map has the
following constrained form:

```text
Delta_i^(M) = g_phi(T_i^(M), q_i^(M), q_global^(M))
Tilde_i^(M) = T_i^(M) + Delta_i^(M).
```

`g_phi` is shared across units and sessions, has no electrode or unit-ID table, and is zero
initialized so `Tilde^(M) == T^(M)` at initialization.  Offline training may regress toward the
source session's `T^(50)` and/or add a decoder-consistency loss, but at validation/deployment it
consumes only `T^(M), q^(M)` from the first `M` trials.  It is finalized once after calibration:
there is no target-side backpropagation, gradient state, or query-label update.

This is intentionally a correction of the **estimator**, not a generic confidence-FiLM retry.
The already non-positive confidence-FiLM result rules out treating cached confidence scalars as a
new untested identity mechanism.  Here confidence fields are admissible only as inputs that
describe the causal corruption of a low-M estimate, and only with the content controls below.

### 2.3 Step-2A required arms and what each can establish

For every auditable budget, retain this arm vocabulary:

| Arm | Calibration feature at inference | Role |
|---|---|---|
| `T4@M` | `T^(M)` | Ordinary causal T4 baseline. |
| `ACM@M` | `Tilde^(M)` from the aligned correction map | Estimator-aware correction candidate. |
| `ACM-TS4@M` | Same correction computation, but the final per-unit corrected descriptor rows are deterministically non-identity shuffled while all dimensions/marginals remain matched | Content/attachment control for the new carrier. |
| `TS4@M` | Row-shuffled ordinary `T^(M)` | Contextual ordinary-T4 content control; it is not a substitute for `ACM-TS4`. |
| `T4@50` plus descriptor consistency to `T^(50)` | First 50 labels only | Causal full-budget boundary and offline teacher target, never a low-M input. |

The primary comparison of a future pilot is `ACM@M - T4@M`, with the matched corrected-row
attachment comparison `ACM@M - ACM-TS4@M` required before claiming content-specific correction.
`T4@50` can show the remaining headroom and permit descriptor-space consistency reporting; it
cannot convert an `M=10` or `M=15` result into a `M=50` deployment result.

There is deliberately no `ACM@50` performance arm in the initial pilot: at `M=50` it is too easy
to create a capacity confound rather than a small-budget correction test.  `M=50` is a boundary
and source teacher first.  A later exception would need a new protocol and a parameter-matched
baseline.

### 2.4 Step-2A CPU-only preflight and kill gates

The following audit must be written and reviewed before any training integration or GPU request.
It uses the strict SUA train/validation development split only and leaves formal test files unopened.

1. **Chronology / cache receipt.**  For every opened session and each `M`, record the exact first
   `M` rewarded trial indices, starts/stops, directions, source fingerprint, fit version, and the
   common evaluation start `50`.  Verify prefix nesting (`T@10` uses exactly the prefix of `T@15`,
   etc.) and reject any cache that omits `M`, signal view, bin/window, reward filter or fit version
   from its key.
2. **Identifiability.**  Report direction histogram, rank, condition number and finite-fit count.
   A rank-deficient estimate is `undefined`, never a zero-filled positive example.  The audit must
   tabulate what fraction of units/sessions are defined at each M.
3. **Actual-estimator stability.**  Using a fixed-seed, disjoint two-way multinomial partition
   of each unit/trial spike count (with each partition rate rescaled by two),
   quantify `T@M` repeatability and `T@M -> T@50` error separately for `(a,c)`, modulation `m`,
   and baseline `b`.  The resampling must preserve the first-M trial/label design; it must not
   synthesize independent coefficient noise.  Raw counts remain in RAM and are never cached;
   a rank-deficient partition is reported undefined rather than imputed.  Report session paired uncertainty/MDE rather than
   inventing a +0.03 decoder threshold at this stage.
4. **Reliability-field sufficiency.**  On source-fold training only, evaluate whether `q@M`
   predicts descriptor error to `T@50` out of fold.  If it does not outperform a constant/`M`
   baseline in a distinguishable way, the proposed correction has no estimator-aware signal and
   stops before GPU work.
5. **Leakage and permutation contracts.**  Test that all `q@M` fields are invariant to joint unit
   permutation; that a complete deterministic row shuffle is non-identity and preserves column
   marginals; and that `ACM` at zero initialization is exactly `T4@M`.  Confirm that no target
   label after M reaches `Tilde@M`.

CPU **stop** conditions are: no usable rank/coverage at the proposed low budgets; no out-of-fold
relationship between causal reliability and `T@50` error; failed prefix/cache/leakage contracts;
or a noise/MDE receipt showing the planned development matrix cannot distinguish its stated
effect.  Any stop is a negative/inconclusive estimator finding, not a rationale to widen the MLP,
add waveforms/electrodes, or sweep arbitrary noise levels.

The CPU audit does not yet choose one of `10/15/20`, and it does not itself authorize a GPU pilot.
That avoids selecting a budget from the same six validation sessions by decoder score.  A separate
review must freeze one budget-selection rule, source-fold training objective, model dimension,
seed count, epoch rule, MDE-informed practical margin and test-isolation receipt.

### 2.5 Step 2B — offline episodic cross-budget robustness (the authorized dropout extension)

Step 2B is deliberately broader than `ACM`: it tests whether **pre-training the selected T4
decoder path on real, causal, low-budget descriptor views and occasional whole-modality absence**
produces a robust low-label decoder.  It is not permitted to simulate that training by adding
independent noise to coefficient columns.

#### Source episodes and held-out deployment boundary

For every *source training* session, an episode first samples
`M in {10,15,20,50}` and materializes `T^(M)` by refitting the cosine estimator from that same
session's exact first-M chronological rewarded trials.  Descriptor attachment is then fixed by
the arm, not randomly mixed within the aligned candidate:

- **`XBR-no-dropout` and `XBR`:** the descriptor is always the real, aligned `T^(M)` while it is
  present.  `XBR` additionally samples a whole-modality presence state; when absent, it replaces
  the complete four-vector by the normalized zero/sentinel representation and provides a one-bit
  *presence* indicator to the new adapter.  This is not coordinate-wise dropout and does not hide
  only direction or only baseline rate.
- **`XBR-TS4`:** when the modality is present, apply a deterministic non-identity permutation to
  complete four-vector rows, retaining side width and exact column marginals.  It uses the same
  whole-modality dropout schedule as `XBR`; when absent, both arms see the identical sentinel.

Thus TS4 is never an ordinary augmentation of the aligned `XBR` arm.  It remains a clean,
parameter/schedule-matched content-and-attachment control.

The sampled budget, corruption state and dropout state are source-training augmentation variables.
They neither create a new behavioural label nor use a query label.  In particular, a `T^(50)`
teacher on a source session is permissible only as a target below; it is not passed through the
student input on an `M<50` episode.

At held-out deployment/evaluation the model is a **single forward pass**.  It receives the
selected, pre-registered `T^(M*)` fit from the first `M*` trials (plus its known `M*` code) and
online neural activity.  It does not run adaptation, backpropagation, a consistency update, a
teacher branch, or a hidden `T^(50)` computation.  `M*` must be chosen from source-only Step-2A
evidence before the six validation sessions are scored.

#### Frozen substrate and the only trainable path

The reference is the selected ordinary B3S/T4 substrate: pooled calibration activity is passed
through `pre_pool`; the four T4 coordinates concatenate before `post_pool`; the coupled SPINT
decoder consumes the resulting identity.  The Step-2B student starts from an exact selected-T4
checkpoint and freezes all pre-existing layers:

```text
frozen:  B3S pre_pool, original post_pool (including its existing T4 columns),
         coupled SPINT decoder, teacher, normalizations and behaviour readout
trainable: one new shared low-rank per-unit identity residual adapter and its M/presence codes
```

For a pooled activity feature `h_i`, masked descriptor `D_i`, budget embedding `e_M`, and presence
bit `p`, the proposed future integration is constrained to

```text
E_base_i = post_pool([h_i, D_i])
delta_E_i = W_up sigma(W_down [h_i, D_i, e_M, p])
E_i = E_base_i + delta_E_i,
```

with a small fixed rank and zero-initialized `W_up`, so the aligned, no-dropout path is exactly the
selected substrate at step zero.  The residual is shared across units; it introduces no unit or
electrode table.  Freezing `post_pool` and the decoder matters: it prevents a full decoder retrain
from being mislabeled as a calibration-robustness gain, while still giving dropout episodes a
trainable path conditioned on frozen activity `h_i` and the explicit presence bit.

This is not a revived generic confidence-FiLM: neither static SNR/waveform nor the historical
confidence descriptor is an input, no feature-wise scale/shift is applied to activity, and the
intervention is a zero-init, budget-conditioned identity residual.  Its only sources are actual
T4 views, their whole-row attachment and the known calibration budget/presence state.

#### Offline teacher terms: allowed, but source-only

In addition to the ordinary source-session behavior/distillation objective, the training protocol
may use both of these predeclared auxiliaries, with fixed weights selected *before* the validation
pilot:

```text
L_T4 = Huber(H(E_i(T^(M))) - stopgrad(T_i^(50)))
L_E  = MSE(E_i(T^(M)), stopgrad(E_base_i(T^(50)))) .
```

Let `D_E` be the selected B3S identity width mechanically read from the exact selected-anchor
checkpoint/model receipt.  `H` is a small linear `D_E`-to-4 prediction head and is trained only
with the adapter.  The future preflight must receipt `D_E`, the anchor SHA, expected adapter shapes
and parameter count, and fail closed if the instantiated identity width differs; no document or
runner may hard-code `64` (the current selected-cost receipt indicates a different identity width).
The targets come from the source session's true first 50 trials.  They are never evaluated as a
held-out adaptation loss and are never needed for held-out forward inference.  If parameter budget
is too tight, the future protocol must preselect *one* of `L_T4` or `L_E`; it may not choose
between them from the six-session decoder score.  The all-source `T@50` descriptor-space audit
remains useful even if neither auxiliary is used.

#### Parameter-matched controls and prospective kill gate

The prospective Step-2B matrix at the one source-selected budget `M*` is:

| Arm | Training schedule | Student input at normal evaluation | Purpose |
|---|---|---|---|
| `T4@M*` | Frozen ordinary selected T4 reference | aligned `T^(M*)` | Causal low-budget baseline. |
| `XBR-no-dropout@M*` | Same adapter, same M episode distribution, same optimizer steps and the same optional teacher terms; dropout probability fixed to zero | aligned `T^(M*)` | Parameter- and cross-budget-matched control.  It isolates modality dropout rather than low-M exposure. |
| `XBR@M*` | Same as no-dropout, plus pre-registered whole-modality dropout probability | aligned `T^(M*)` | Cross-budget robust candidate. |
| `XBR-TS4@M*` | Identical to `XBR`, but source episode descriptors are complete-row TS4-corrupted before the student branch; teacher targets remain aligned | TS4-corrupted `T^(M*)` | Tests whether the training gain needs aligned descriptor content. |

For an honest attachment control, the required scored `XBR-TS4` evaluation uses its own
TS4-corrupted `T^(M*)` input; `XBR` and all ordinary baselines stay aligned.  The optional teacher
terms stay aligned targets in both arms, so the only altered source information is the student
descriptor-to-unit attachment.

No Step-2B run is authorized merely because it has an attractive teacher loss.  Step 2A may
recommend *one* Step-2B GPU pilot only when, source-only and out of fold:

1. at least one low-M candidate has valid/covered fits and a nontrivial, MDE-resolvable
   `T@M -> T@50` error gap (there is something to make robust);
2. causal first-M fields or the correction-only map predict/reduce that gap beyond an M-only or
   constant baseline with the prespecified resampling uncertainty; and
3. the cache/chronology, zero-init, whole-modality, TS4 and teacher-isolation contracts pass.

Step-2A's uncertainty/MDE is in **descriptor-error units**, not decoder R², and must never be
converted numerically into an R² threshold.  Before launch, the separate Step-2B protocol must set
its prediction-endpoint practical margin `d*` from either (a) an independent, already completed
validation run/seed/session noise receipt that matches the locked endpoint, or (b) a new,
pre-registered precision receipt with no candidate selection.  It may not derive `d*` from the
Step-2A descriptor MDE, from a running Step-2B arm, or from formal-test observations.

`XBR@M*` is a **stop** unless it exceeds *each* of `T4@M*`, `XBR-no-dropout@M*`, and the scored
`XBR-TS4@M*` by at least this independently fixed R² `d*`, with the prespecified paired
interval/session-direction rule also satisfied.  Failing any one comparison stops Step 2B: no
dropout-rate sweep, adapter-rank sweep, unfreezing, confidence-FiLM, extra budget fishing, or
formal SUA test.  Passing only the ordinary T4 comparison supports neither the modality-dropout
claim nor the aligned-content claim.

## 3. Step 3 — fixed-K temporal-prototype memory, CPU Gate A only

### 3.1 Narrow hypothesis and deployment state

This is not a revival of the unsuccessful learned fixed-slot router.  That router compressed the
decoder token interface through a learned NeuronID routing matrix.  The present question is
smaller: whether **fixed, session-local temporal prototype summaries** provide any neural-only
functional information before they are offered to a decoder.

For M1 Gate A, lock the initial representation to `K=4` slots and temporal rank `r=4`.  At a
support bin, a unit's recent neural history is filtered by fixed causal rank-r filters to give
`z_i(t) in R^r`.  Training source sessions supply four globally ordered anchor vectors
`p_1,...,p_K` only within each nested source fold.  A unit routes each support-bin feature to an
anchor with a deterministic shared rule (hard nearest-anchor or a frozen shared soft rule; choose
one before the audit), and maintains only streaming sufficient statistics:

```text
n_ik       <- n_ik + route_ik(t)
s_ik[1:r]  <- s_ik + route_ik(t) * z_i(t)
P_ik       = s_ik / max(n_ik, eps).
```

The calibration memory is `P_i[4,4]` plus four counts per unit, i.e. `O(N*K*(r+1))`, and causal
filter state is `O(N*r)`.  It never stores an `M x T` array of raw support trials/bins.  All
weights/anchors are frozen after source-only fitting; deployment has a streaming accumulation and
one forward finalization, not target-side backpropagation.

This definition also fixes a subtle comparability problem: per-unit online k-means would allow
the meaning/order of slot 1--4 to permute across units.  The anchors here are shared and ordered
within a source fold, so a slot index has a common source-trained temporal meaning.  The entire
operator is shared across units and is unit-permutation equivariant; no absolute channel/unit ID
is permitted.

### 3.2 CPU Gate A data/endpoint and oracle boundary

Gate A is source-only nested LOSO over the four M1 source sessions.  It has no official M1
held-out access and no decoder/GPU training.  Support is the fixed first ten calibration trials
using the already accepted M1 support/query semantics.  Reliability resampling must split bins
within every support trial (or use deterministic within-trial spike thinning), not chronological
first-five/last-five, because the latter can delete condition coverage.

The primary target is a later **neural-rate** predictive proxy, evaluated after the support
boundary.  A source-fold readout may be fit on source sessions and scored on the left-out source
session; its fitting, nested anchor construction and uncertainty all must be logged.  It is not a
claim of backprop-free offline training.

No dense behavior, velocity or future condition is permitted in a prototype **value**, anchor,
route, or deployed carrier.  If a diagnostic ever places dense behavior in those values, it is a
separate `label-rich oracle` analysis: useful only to locate an upper bound and categorically
barred from advancing this Gate A or from deployment language.  Later neural targets may be used
only to score the offline predictive proxy and must be labeled as later-neural oracle targets.

### 3.3 Required Gate-A comparators

All comparators share the same source-fold support/query boundary, unit ordering discipline,
readout capacity and total side-vector width (zero-pad where needed):

| Carrier | Purpose |
|---|---|
| `D4` | Existing closed-form support-condition carrier reference.  Its earlier decoder failure is not erased by a proxy. |
| rate-only | Per-unit support rate/exposure summary, dimension-matched without prototype assignment. |
| `K=4` temporal prototypes | Candidate neural-only compact state `P_i`. |
| slot shuffle | Within every outer fold, apply a deterministic complete non-identity **session-keyed** permutation of the four prototype slots to every source-readout and left-out carrier, shared across units within each session; preserve each slot's complete values/counts but break cross-session anchor-to-slot semantics. |

The shuffle must permute whole `(count, r-vector)` slot blocks, not unit rows and not individual
coordinates.  Its session-keyed permutation is predeclared and receipted within each outer fold;
it is never optimized from the later-neural score.  It thereby distinguishes use of coherent
cross-session prototype labels from an arbitrary wider rate feature.  A candidate that is no
better than slot shuffle cannot be described as a temporal-prototype mechanism.

### 3.4 Gate-A decisions

Before interpreting an error ratio, report source-LOSO paired resampling intervals and the minimum
distinguishable proxy effect for four source sessions.  The review must then require all of:

1. state-size, no-raw-`M x T` and unit-permutation contracts pass;
2. temporal state is repeatable under within-trial matched split/thinning, with degeneracy reported
   rather than imputed;
3. out-of-fold prototype improvement is distinguishable from rate-only **and** slot-shuffle under
   the measured MDE; and
4. no behavior/value oracle crossed the carrier boundary.

Failure of any item stops the branch at Gate A.  It does not authorize more `K`, a learned router,
cross-neuron attention, decoder fitting, a GPU run, or M1 held-out evaluation.  A pass merely
permits a new, separately reviewed one-cell decoder protocol; it is not a GPU authorization in
this document.  No fixed numerical percentage is asserted until the four-source MDE audit exists.

## 4. Implementation touchpoint audit

### 4.1 Safe new, isolated files (allowed only after review of their own implementation)

| Scope | New file(s) | Responsibility |
|---|---|---|
| Step 2 protocol/cache | `sua_exploration/mc_maze/t4_cross_budget_protocol.py`, `sua_exploration/mc_maze/t4_cross_budget_features.py` | **Implemented readiness boundary only:** versioned first-M trial receipts (ordinal/times/raw+snapped target direction), exact causal fits, rank/condition/coverage and explicitly named spike-count/time reliability fields, plus a new cache namespace.  It is not registered as a training feature. |
| Step 2 CPU audit | `sua_exploration/mc_maze/t4_cross_budget_audit.py`, `sua_exploration/scripts/audit_sua_t4_cross_budget.py`, `sua_exploration/tests/test_t4_cross_budget_protocol.py` | **Guarded source-only CPU runner:** RAM-only count partitions, undefined handling, nested source-LOSO q-to-error proxy, target-free development scoring and frozen-manifest validation.  It executes only with `--execute-source-audit` **and** `STEP2A_REVIEWED_SOURCE_AUDIT=YES`, requires a fresh output directory, and never resolves formal-test paths. |
| Step 2B future adapter | `streaming_calibration_exp/src/models/components/streaming_spint_t4_cross_budget_adapter.py`, plus isolated tests | Only after a separate protocol approval: zero-init residual, whole-modality presence handling, source-only teacher terms and parameter-count receipt. |
| Step 3 state | `sua_exploration/mc_maze/fixed_k_temporal_prototypes.py` | Pure streaming filters, ordered-anchor routing, sufficient statistics and no-raw-memory invariant. |
| Step 3 CPU audit | `sua_exploration/scripts/audit_m1_fixed_k_prototypes.py`, `sua_exploration/tests/test_fixed_k_temporal_prototypes.py` | Nested LOSO anchors/readout, source-only receipt, MDE and all four comparators. |

The Step-2A protocol/cache/test/dry-run files now exist solely as isolated readiness code; they
do not open data in dry-run mode, do not register an encoder feature token, and do not authorize a
source/validation audit.  The Step-3 and Step-2B-adapter names remain proposals.  This separation
keeps new feature/cache semantics testable before any model integration.

### 4.2 Existing code that may be *read as reference*, not modified now

- `sua_exploration/mc_maze/multisession_datamodule.py` owns
  `list_datamodule_rewarded_trials()` and `calibration_pool_end_time()`.  Step 2 must call the
  former rather than recreate trial filtering/order.
- `sua_exploration/mc_maze/unit_side_features.py` contains ordinary T4/T4C fitting, directional
  geometry and current cache-key conventions.  Existing `t4c` has an M=50 confidence meaning;
  it is not the Step-2 low-M cache and must not be silently repurposed.
- `streaming_calibration_exp/src/models/components/streaming_encoders.py` and the older fixed-slot
  router are reference only.  They do not satisfy the fixed-K temporal-prototype Gate-A definition
  by themselves and must not be called an implementation shortcut.
- The future Step-2B adapter has no approved training entrypoint yet.  It must not be added to the
  active `train_variant_dandi688.py` or coupled to an existing residual switch while this
  preparation/active-screen boundary is in force.

### 4.3 Frozen / excluded during this preparation

The following active factorized-residual touchpoints must remain untouched until that experiment
is terminal and a separate integration review approves any change:

- `streaming_calibration_exp/src/models/t4_logit_residual_module.py`;
- `streaming_calibration_exp/src/models/components/streaming_spint_t4_logit_residual_adapter.py`;
- `streaming_calibration_exp/tests/test_t4_logit_residual_module.py` and
  `streaming_calibration_exp/tests/test_streaming_spint_t4_logit_residual_adapter.py`;
- `sua_exploration/scripts/train_variant_dandi688.py`,
  `audit_sua_t4_factorized_logit_preflight.py`,
  `run_sua_t4_factorized_logit_one_arm.sh`, and
  `schedule_sua_t4_factorized_additive.sh`;
- all active residual checkpoints, logs, result folders and formal-test scopes.

Consequently no existing side-feature token, datamodule registration, shared training entrypoint,
decoder component, or runner is changed in the preparation phase.  A future GPU pilot needs a
separate immutable protocol and a deliberately reviewed integration path, after the serialized
decision rule in section 1.

## 5. What this preparation does and does not claim

It establishes a falsifiable, causal route for testing whether low-label T4 error can be corrected
from the actual estimator, and a compact neural-only Gate A for a fixed-K temporal memory.  It
does not claim that either will improve R², that `M=10/15/20` is sufficient, that a teacher target
is deployable, or that a previous static/confidence result has been reversed.  The next artifact
should be the Step-2 CPU receipt, not a GPU dashboard.

## 6. Step-2A execution decision (2026-08-02)

The guarded source-only audit completed on the frozen 27-train / 6-development-validation split.
No formal-test path was resolved, no GPU was used, and the endpoint is descriptor-error MSE rather
than decoder R2. The immutable source receipt is
`results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json` (SHA-256
`42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d`).

The implemented predictor is more narrowly `q_unit + M`: seven unit reliability fields plus the
known budget. In nested source LOSO it beat M-only by mean `0.7671` descriptor-MSE units in
21/27 sessions. Applied target-free to the six development-validation sessions, however, the
increment fell to `0.0226`, only 3/6 sessions favored it, the normal-approximation 95% interval
crossed zero (`[-0.3268,0.2817]`), and the paired MDE was `0.4349`. Dropping the single most
favorable validation session reversed the mean. The simpler M-only effect remained visible in
all 6/6 sessions, so the collapse is specific to the incremental reliability fields rather than
proof that the audit could detect nothing.

The proxy also predicts only scalar error magnitude, not the signed residual
`T4@50-T4@M`; it therefore cannot by itself specify the proposed correction. It omits the proposed
global design fields and does not contain the required per-field degeneracy/contribution table.
These limitations strengthen the stop decision rather than authorizing a post-hoc feature patch.

**Decision: Step-2B is NO-GO.** Do not launch cross-budget/dropout GPU training, a wider q model,
or formal evaluation. The useful descriptive result is that all 33 sessions are rank-defined,
descriptor error falls with M, low-M `(a,c)` is substantially less repeatable than `b`, and `b`
is already near `0.99` split-half Pearson. Under the serialized rule this written stop unblocks
only the separately gated Step-3 CPU audit; it does not authorize a Step-3 decoder or GPU run.

The machine decision is
`results/sua_t4_cross_budget_source_audit_v1_20260802/step2_decision.json`; independent review is
in `SUA_STEP2A_INDEPENDENT_RESULT_AUDIT.md`.

## 7. Step-3 execution decision (2026-08-02)

The independently reviewed v3 source-only CPU audit completed on the exact four hash-bound M1
`held-in-calib` source sessions. It opened no minival, held-out, formal-test, or EvalAI path,
constructed no decoder, and forced CUDA off. The endpoint is the predeclared later-neural oracle
proxy, not behavior-decoding R2. The authoritative result is
`results/m1_fixed_k_temporal_prototype_gate_a_v3/source_gate_a.json`, SHA-256
`b2b1e9ff3288ef57fc26d2b446fd803155b1ec5e0ce14dc2f9cc1cbddb92f6dd`.

Prototype minus rate-only was strongly positive and consistent: mean `+0.100526`, 4/4 positive,
paired 95% CI `[+0.090798,+0.110253]`, versus MDE `0.012719`. The repeatability and state/oracle
contracts also passed. The stronger mechanism conjunction did not: session-keyed slot shuffle
caused highly heterogeneous failures, giving prototype-minus-shuffle CI
`[-56.427143,+110.100156]` and MDE `108.863822`. Counting the 256 technical resamples as extra
sessions or replacing the frozen inference post hoc is forbidden.

**Decision: Step 3 stops at Gate A.** The source proxy supports a stable signal beyond support
rate/exposure, but coherent cross-session slot semantics are not established precisely enough to
authorize decoder/GPU/held-out work. No K/r/router expansion or quantization follows. The full
root interpretation is in `M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_RESULT_AUDIT.md`.
