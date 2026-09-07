# Design — DANDI 000688 T4-Covered Activity-Selected Early Pooling V1

- Date: 2026-09-04
- Dataset: DANDI `000688`, sub-C centre-out SUA
- Working method: **TC-AS-EP** (`T4-Covered Activity-Selected Early Pooling`)
- Training method: **COV-RAND10-EP**
- Deployment method: **COV-TOP10-EP**
- Parent deployment protocol: **V9 only** (`activity observed through trial 50`, `T4@50`, `query after trial 50`)
- Status: `REVISED_AFTER_SECOND_AUDIT__STAGE0_DESIGN_GO__CODE_AND_WORKORDER_NOT_AUTHORIZED`
- Supersedes in place: draft SHA `e8725cc66497cdf37b7da5a5bcb799891c9fe80353b18a296c8b411e5c121c83`

Stage-0 amendment (attempt 2): the first immutable source-only selector
attempt passed C/K/Q authority, direction coverage, jackknife stability, and
random-support RNG gates, but one of six development sessions had absolute
selected-membership/duration point-biserial correlation `0.2259792581`, above
the frozen `0.20` ceiling. The sole permitted selector revision is therefore
the duration-residualization step in Section 7.2. No decoder, R2, GPU, formal
target, or external target was opened before this amendment. If attempt 2
fails, V1 stops.

> This is a revised research design, not a work order. It does not authorize
> data access, cache construction, GPU execution, checkpoint creation, target
> scoring, result-root creation, or submission. The current shared
> `random_calibration` implementation cannot express this design; Stage 0 must
> first add and test a route-owned three-axis data contract.

## 0. Revision verdict

An independent review returned `REVISE`. The core decomposition was retained,
but the first draft had three material problems:

1. its raw/interpolated activity-centrality selector was empirically biased
   toward shorter trials and could remove an entire movement direction;
2. `K/C = 30/50` made Top-K and random supports overlap too much to expose a
   small selector effect;
3. the existing data module couples candidate-pool size, returned support size,
   and query-start index, so its advertised random-support branch is a no-op.

This revision therefore:

- changes the activity support from `K=30` to `K=10` while retaining
  candidate pool `C=50` and query start `Q=50`;
- replaces interpolated-tensor centrality with raw-spike, duration-normalized
  activity features;
- uses the same direction labels already consumed by `T4@50` to guarantee
  tuning-regime coverage, rather than claiming a behavior-free selector;
- adds fixed early and fixed late coverage-complete controls;
- requires a route-owned `candidate_pool_n / activity_support_n /
  query_start_trial` separation before training;
- uses seeds `42/43/44` for a positive result;
- treats the existing formal sub-C scope as consumed and unavailable.

A second independent review accepted the revised scientific structure and
returned `Stage 0 = GO; executable work order = REVISE`. Its remaining valid
points concern exact Stage-0 definitions, historical-surface wording, and
receipt labels; they are incorporated below. Stage-0 design GO is not authority
to read data or implement/run the route.

One reviewer statement is explicitly corrected: the old paired-view C1 route
trained activity at `M10`, but the repository's selected ordinary B3S/T4
deployment checkpoint lineage trains at `M30`. These are distinct lineages and
must not be called the same champion.

## 1. Two-sentence research pitch

T4 describes what each unit encodes, while activity pooling estimates how that
code is instantiated in a particular session. We train the native early-pooled
identity on diverse, direction-covered 10-of-50 activity supports and deploy a
deterministic T4-label-budget-neutral activity coreset, while fixing T4@50,
query timing, architecture, loss, and parameter count.

The intended contribution is not “another Top-K heuristic.” It is a matched
test of whether calibration identity should be trained as a **set-valued
object** instead of an arbitrary chronological prefix.

## 2. Frozen factual corrections

### 2.1 Current selected checkpoint versus historical C1

The repository contains at least two relevant training contracts:

1. Historical paired-view C1:
   - `source_activity_calibration_n_trials = 10`;
   - evaluation activity support `=30`;
   - T4 label pool `=50`;
   - evaluation starts after trial `50`.

   Evidence:
   `sua_exploration/scripts/train_paired_view_c1_dandi688.py`.

2. Current selected ordinary B3S/T4 checkpoint lineage:
   - manifest `activity_calibration_n = 30`;
   - all selected seed-42/43/44 run metadata record
     `training.calibration_n_trials = 30`;
   - T4 label pool `=50`;
   - selected architecture is `coupled_ordinary_B3S_T4`.

   Evidence:
   `sua_exploration/manifests/sua_t4_final_architecture_selection_v1.json`
   and the three referenced `run_metadata.json` files.

The selected lineage's `validation_protocol.evaluation_windows` is
`trials[calibration_n_trials:] only`; with `calibration_n_trials=30`, its
source training/model-selection query surface begins at trial 30. The manifest
V9 deployment/scoring boundary of trial 50 is a different surface. Therefore
historical M30 performance is context only: the new `M30-V9-REPRO`, trained
with `Q=50`, is not required or expected to reproduce that performance.

The selected training blocks do not record a non-null `random_calibration`
law. Given the old coupled pool/support construction, any purported K30-of-K30
random selection would be inert; chronological/fixed support is therefore an
implementation inference, not an explicitly receipted historical field.

All three selected runs bind the same teacher SHA. The seed-42 metadata omits
the `encoder_warmstart_*` fields, while seed 43/44 record them as explicit
`null`; none of the three supplies a non-null encoder warm-start. The schema
presence difference is disclosed, but it is not evidence of different
warm-start content.

Therefore:

- the selected M30 checkpoint and its historical numbers are context only;
- `M30-V9-REPRO` is the sole matched Q50 product anchor for this route;
- the historical M10 C1 line is useful evidence that the topology can train at
  a smaller support cardinality;
- neither may be silently substituted for the other;
- the new K10 arms are new matched training arms, not “free champion
  reproductions.”

### 2.2 Parent protocol is V9, not A2

TC-AS-EP is legal only under the V9 timing contract:

```text
activity candidate observation: trials 0..49
T4 label fit:                 legal labels among trials 0..49
query:                        strictly trials 50+
```

It is not directly comparable with the A2 protocol, whose activity, side
feature, and evaluation boundary are all 30. Under A2, reading activity from
trials 30..49 would move the query boundary and change the protocol.

Every result table must include `parent_protocol=V9` and
`not_comparable_to_A2_start30=true`.

### 2.3 Formal sub-C evidence is unavailable

The receipt

`sua_exploration/results/p3_formal_test_816cdd8bf9f26abd1a3e6251e5fbf8537eb6c6cb4de1e8f3312980ddbf478379_receipt.json`

has status `started`, and historical documents already treat the scope nonce as
consumed. Conservatively, this design must not use, reopen, overwrite, or
reclaim that formal scope.

The selected-architecture manifest also records
`formal_test_files_opened=false`. That is evidence in favor of the underlying
files remaining unopened, but it does not undo the already-claimed scope nonce.
Both facts must be retained if a future governance process revisits whether a
new formal authorization is possible.

Consequences:

- the six sub-C development sessions are development evidence only;
- external sub-M15 is retrospective evidence because it has been repeatedly
  inspected;
- V1 cannot make a fresh confirmatory claim;
- a confirmatory result requires a genuinely untouched new surface under a
  separate authorization.

### 2.4 One missing direction label

For source session `20150313`, one first-50 trial has direction index `-1`.
Activity from that trial remains legally observable. It is excluded from the
direction-covered selector because it lacks a valid direction label, and the
T4 fit uses 49 valid labels. Every receipt must expose:

```text
candidate_activity_count = 50
valid_direction_label_count = 49
invalid_direction_indices = [27]  # subject to independent Stage-0 recheck
```

No imputed direction is allowed.

## 3. Scientific decomposition

### 3.1 Primary questions

V1 asks three separable questions:

1. **support-augmentation effect:** does training on many legal K10 support
   realizations outperform training on one fixed K10 realization?
2. **fixed-support content sensitivity:** does an early versus late fixed K10
   support materially change the result?
3. **activity-coreset effect:** under one checkpoint, does deterministic
   activity-based selection outperform a typical direction-covered random K10
   support?

### 3.2 Why “randomness” is not the mechanism claim

Random support training necessarily exposes the model to more distinct trials
than a single fixed support. Therefore the governing language is **support-set
augmentation** and **support-composition robustness**, not “randomness itself
causes the gain.”

The fixed-early and fixed-late arms reveal whether a result can be explained by
which fixed trials are exposed. A strong augmentation claim requires the
random-support arm to exceed both fixed controls, not merely the weaker one.

### 3.3 Explicitly excluded questions

V1 does not test:

- late pooling or J-R1;
- continual query-memory updates;
- learned selection;
- CP-FiLM;
- a K30/K10/K4 cardinality cycle;
- new T4 content;
- extra target labels;
- target-session parameter updates.

Those are successor questions and cannot rescue a negative V1.

## 4. Frozen early-pooling topology

For activity trial `j`, let `A_j` be the canonical trial activity and `c` the
fixed normalized T4@50 tensor:

```text
u_j      = pre_pool(A_j)
u_bar(S) = arrival_order_mean({u_j : j in S})
e(S)     = post_pool(concat(u_bar(S), c))
y_hat    = decoder(x_query, e(S))
```

This matches B3S:

1. apply `pre_pool` separately to each trial;
2. sum and divide by trial count;
3. concatenate side features;
4. apply nonlinear `post_pool` exactly once.

V1 must not average per-trial `post_pool` outputs.

### 4.1 What “fixed carrier” means

“Fixed carrier” means that all matched arms use the same:

- T4@50 raw descriptor values;
- T4 source-frozen normalizer;
- label pool and valid-label mask;
- side-feature tensor presented to the network.

It does **not** mean all weights that consume T4 are frozen. B3S `post_pool`
jointly consumes activity and T4. If `post_pool` is trainable, its T4-consuming
weights change.

### 4.2 Trainable parameter contract

The selected ordinary B3S/T4 lineage records `freeze_decoder=false`. To avoid
introducing an unacknowledged training-contract change, every new training arm
uses the same full source-training trainable/frozen parameter roster as the
selected lineage, including its decoder policy.

All matched arms have identical architecture and parameter count. Only the
activity support law differs. Target-session optimizer, backward, and update
counts remain zero.

If a later experiment freezes the decoder or trains only the identity encoder,
that is a separate factorial—not V1.

## 5. Three independent data axes

The implementation must separate:

```text
candidate_pool_n   = C = 50
activity_support_n = K = 10
query_start_trial  = Q = 50
```

These values have different meanings:

- `C`: how many calibration activity trials may be considered;
- `K`: how many trials enter one identity support;
- `Q`: which trials may contribute query windows.

### 5.1 Current hard blocker

The current shared data module builds exactly `calibration_n_trials` activity
trials and also excludes exactly that many trials from query windows. Its
random branch samples only when the stored pool is larger than
`calibration_n_trials`; under the current construction they are equal, so the
branch never activates.

The existing help text that describes historical random resampling is not
runtime evidence. No V1 work order is legal until a route-owned adapter or
reviewed shared seam proves the three axes are independent.

### 5.2 Required Stage-0 implementation proof

On a synthetic and one authorized source-only fixture, prove:

```text
stored candidate activity shape[0] == 50
returned identity support shape[0]  == 10
first query trial index             == 50
```

For random support, additionally prove:

- more than one unique subset is emitted;
- subset entropy is nonzero;
- every subset contains exactly ten unique indices;
- every subset is sorted before tensor indexing and reduction;
- sampling never changes query/target/T4 tensors;
- global NumPy/Python/Torch RNG streams are unchanged.

### 5.3 Stateless per-sample sampling law

Do not use global `np.random.choice`. Derive a local counter-based seed from:

```text
SHA256(
  route_domain,
  training_seed,
  epoch,
  session_id,
  sample_or_window_id
)
```

Use a local generator, sample without replacement, and sort the selected trial
indices before materializing the support. The receipt records the generated
indices and the domain digest.

The sampling semantics are **per sample**, not ambiguously “per optimizer
step.” If all samples in a resident batch intentionally share a subset, that
alternative must be separately frozen and named.

## 6. Why K=10 and C=50

For a random K-subset and a fixed Top-K subset from a C-candidate pool, expected
overlap is `K^2/C`.

```text
K=30, C=50: expected overlap = 18/30; only 40% differ
K=10, C=50: expected overlap =  2/10; 80% differ
```

At K30, TOP30 and BOTTOM30 must overlap by at least ten members. That weakens
both the selector contrast and its strongest negative control. K10 permits
disjoint positive and anti-selected supports and increases the observable
identity perturbation.

K10 is not claimed to be the selected checkpoint's historical training
cardinality; the selected checkpoint is M30. K10 is a deliberate new few-shot
identity-capacity choice, motivated by:

- stronger counterfactual separation;
- the historical existence of an M10 C1 training path;
- the desire to test support robustness rather than near-identical K30 means.

Because K10 may reduce absolute performance, the M30-V9 reproduction arm in
Section 8 remains the product anchor.

## 7. T4-covered activity selector

### 7.1 Retraction of the first selector

The first draft's “Robust Activity Centrality Top-K” used mean/std statistics
from cubic-interpolated 100-bin trial tensors. Read-only review found:

- selected trials were shorter than rejected trials in all six development
  sessions;
- two sessions lost an entire movement direction;
- approximately one third of interpolated elements were negative, with a
  reported minimum near `-0.650`;
- applying `log1p` inside the interpolated-bin aggregation amplified cubic
  interpolation undershoot;
- trial duration and firing-rate structure remained physiologically
  correlated, so an activity-only ranking could still encode vigor/duration.

That selector is retired before execution. Its formula must not be implemented
as V1.

The correction is not claimed to eliminate duration dependence. Interpolated
per-bin means can approximate a duration-normalized rate up to bin-scale and
warping details. The real defects are the nonlinear `log1p` placement,
negative interpolation artifacts, missing direction coverage, and an
unbounded physiological duration/rate association. Raw spike-time rates and
the Stage-0 bias gate below constrain these problems; they do not prove
behavioral independence.

### 7.2 Raw activity feature

For each first-50 trial `j` and unit `i`, compute directly from raw spike times:

```text
rate[j,i] = count(spike_time in [start_j, stop_j))
            / (stop_j - start_j)
f[j,i]    = log1p(rate[j,i])
```

This follows the half-open and duration-normalized firing-rate convention used
by the T4 feature code. It does not use cubic-interpolated activity, padding,
or negative-valued pseudo-counts.

The sole Stage-0 revision removes the calibration-pool linear duration
component before centrality is computed. On all 50 activity-authority trials,
in CPU float64, define:

```text
q[j]       = log(stop_j - start_j)
q0[j]      = q[j] - mean_j q[j]
f0[j,i]    = f[j,i] - mean_j f[j,i]
beta[i]    = sum_j q0[j] * f0[j,i] / sum_j q0[j]^2
f_res[j,i] = f[j,i] - q0[j] * beta[i]
```

Centering both operands is the frozen numerically stable intercept form and
ensures a constant feature has an exact zero numerator. The denominator must
be finite and strictly positive. This is a fixed
calibration-only nuisance projection, not a fitted selector hyperparameter.
It uses trial-boundary duration already required to construct rate, but no
query activity, dense behavior target, prediction, loss, gradient, or R2. The
method is consequently described as **duration-residualized activity
centrality**, not activity/behavior independence.

Within each session, robust-normalize each residualized unit coordinate across
the valid first-50 activity trials:

```text
center[i] = median_j f_res[j,i]
scale[i]  = 1.4826 * MAD_j(f_res[j,i]) + epsilon
z[j,i]    = (f_res[j,i] - center[i]) / scale[i]
```

Freeze `epsilon = 1e-12` in CPU float64. The robust center and MAD use all
first-50 trials that are valid members of the activity authority, including a
trial whose direction label is `-1`; that trial remains excluded from every
direction-covered support. Zero-MAD coordinates contribute exactly zero after
an explicitly frozen rule.

### 7.3 Honest supervision label

The selector is no longer called behavior-free or label-free. It uses canonical
direction labels already consumed by T4@50 to prevent loss of tuning regimes.

Correct description:

> target-update-free and label-budget-neutral: it consumes no labels beyond
> the same first-50 direction labels already required by T4@50.

It does not use dense query behavior, query activity, model predictions,
losses, gradients, or validation R2.

### 7.4 Direction coverage law

Among first-50 trials with valid canonical directions:

1. require all eight canonical direction bins to be represented;
2. allocate one mandatory support slot to each direction;
3. allocate the remaining two slots with at most two total trials per
   direction;
4. never impute or snap an invalid `-1` direction beyond the existing frozen
   T4 snapping law.

The same coverage constraints apply to fixed, random, Top-K, and anti-selected
K10 supports. This prevents the selector contrast from being explained by one
arm seeing different direction cardinalities.

### 7.5 COV-TOP10 selection

Define robust pairwise activity distance:

```text
D(j,l) = median_i abs(z[j,i] - z[l,i])
```

For each direction, select the medoid that minimizes summed distance to the
other candidates in that direction. These eight medoids fill the mandatory
coverage slots.

For the remaining two slots, greedily choose the candidate with the largest
reduction in total nearest-selected distance over all valid candidates,
subject to the two-per-direction cap.

Both argmin and argmax use one canonical, traversal-independent rule. For a
medoid argmin, first compute the complete finite value vector, set
`m=min(values)`, form
`T={j: math.isclose(value[j], m, rel_tol=1e-12, abs_tol=1e-12)}`, and choose
the member of `T` with the smallest chronological trial index. For a greedy
marginal-gain argmax, use the same construction with `m=max(values)`. Never
define ties by sequential pairwise comparison between candidates, because
`math.isclose` is not transitive.

This is called **coverage Top-K** even though the score is a deterministic
greedy marginal-coverage gain rather than ten independent scalar ranks. Every
step and marginal gain is recorded.

### 7.6 COV-RAND10 selection during training

For each sample:

1. draw one valid candidate uniformly within each of the eight directions;
2. draw two additional unused candidates uniformly without replacement;
3. enforce at most two selected trials per direction;
4. sort final indices chronologically before activity reduction.

The subset uses the dedicated stateless RNG domain in Section 5.3.

### 7.7 Fixed controls

`COV-FIX-E10`:

- select the earliest valid candidate in every direction;
- fill two remaining slots with the earliest legal unused candidates under the
  two-per-direction cap.

`COV-FIX-L10`:

- select the latest valid candidate in every direction;
- fill two remaining slots with the latest legal unused candidates under the
  same cap.

These are fixed support-content controls. They do not match the total unique
trial exposure of random augmentation, but they reveal whether the result is
dominated by a favorable early or late fixed support.

### 7.8 Selector Stage-0 bias and stability gates

On the six source-development sessions, before any GPU training:

- use an exact leave-one-unit-out jackknife: recompute the complete selector
  once after deleting each unit coordinate, with no random coordinate-drop
  fraction;
- the median leave-one-unit-out Top10 Jaccard within every session must be at
  least the exact value `2/3`, corresponding to no more than two membership
  replacements in the median replicate;
- absolute point-biserial correlation between selected membership and trial
  duration must be at most `0.20`;
- standardized selected-versus-rejected duration mean difference is reported
  descriptively but is not a second gate. Here `SMD_full` is explicitly
  defined using the standard deviation of duration over the complete
  candidate pool, not a pooled within-group Cohen's-d denominator. For
  selected fraction `p=K/C`, this definition is algebraically linked to the
  point-biserial correlation by
  `SMD_full = pb / sqrt(p(1-p))`; at `K/C=10/50`, `|pb|<=0.20` corresponds to
  `|SMD_full|<=0.50`. If Cohen's d is also reported, the relation is only
  approximate and it cannot be used as a second gate;
- all eight directions must be present in every K10 support;
- repeated execution must produce identical indices and marginal-gain bytes;
- no selector input may contain query, prediction, loss, gradient, or dense
  behavior-target arrays.

All medoid sums, distances, and marginal gains are computed in CPU float64 and
use the global-extremum candidate-set rule from Section 7.5. These definitions
and thresholds are frozen before the exact C50 source-only Stage 0.

Stage 0 permits at most one selector revision. Every attempt, including the
first failed attempt, must publish an immutable constructibility receipt and
bind the design-document SHA. A revision must be justified only by
target-free constructibility/bias/stability evidence, must receive a new
document SHA, and must occur before any decoder R2 or other performance metric
is computed. If the second selector attempt fails, the route stops. No
unreceipted or unlimited pre-performance selector tuning is allowed.

An audit-only C30 proxy using clipped/interpolated cached tensors found that
the direction-coverage law restored all eight directions and reduced the
reported duration association relative to unconstrained centrality. It is
supporting design evidence only: it used neither C50 nor the governing raw
spike-time rate feature and cannot satisfy or calibrate the Stage-0 gate.

## 8. Matched training arms

Within each seed, all four arms start from byte-identical copies of one
seed-specific canonical initialization and use the same V9 source roster,
query trials 50+, T4@50, normalizer, task batches, loss, optimizer,
learning-rate law, dropout, and 12-epoch horizon. Byte equality is required
within a seed, not across seeds: the training seed is intended to change
stochastic initialization and sampling.

The historical selected metadata bind the same teacher SHA for seeds 42/43/44;
seed 42 omits `encoder_warmstart_*`, while seeds 43/44 store explicit nulls.
The new work order must choose one unambiguous initializer construction and
record the pre-step model-state SHA for every arm. It must not infer a non-null
warm-start from metadata-field presence.

The selected run metadata contain `lambda_E=0.1` together with
`loss_mode=task_only`. Source inspection confirms `lambda_E` is inactive in
that mode: identity MSE enters the loss only for
`loss_mode=task_plus_y_plus_E`. Stage 0 must retain a regression that proves
changing dormant `lambda_E` cannot change a task-only update, while the
governing receipt labels the objective simply as task-only MSE.

Validation is different from the training loss path: the current module runs
the teacher forward and records teacher/identity diagnostics whenever
`not self.training`, even under `task_only`. Therefore the exact teacher
checkpoint and SHA are required validation-time runtime dependencies for all
four arms. They must be byte-identical within every paired seed and included
in source closure, launch, and terminal receipts even though teacher outputs
do not enter the task-only optimization loss.

### 8.1 Arm A0 — `M30-V9-REPRO`

- fixed activity support: chronological trials `0..29`;
- support cardinality: 30;
- candidate observation and T4 pool: first 50;
- training queries: trial 50+;
- purpose: reproduce the selected M30-style product under the new explicitly
  decoupled V9 loader.

This arm is required because historical M30 training used a coupled data
construction and cannot automatically certify the new three-axis loader. It
is the only new-route product anchor. Historical selected-M30 performance,
obtained from a query-from-30 training/model-selection surface, is contextual
and has no required performance-parity tolerance against this Q50 arm.

### 8.2 Arm A1 — `COV-FIX-E10-EP`

- fixed earliest direction-covered K10 support;
- training queries trial 50+;
- same parameter roster and all other inputs as A0/A2/A3.

### 8.3 Arm A2 — `COV-FIX-L10-EP`

- fixed latest direction-covered K10 support;
- otherwise identical to A1.

### 8.4 Arm A3 — `COV-RAND10-EP`

- stateless per-sample direction-covered random K10 support;
- otherwise identical to A1/A2.

### 8.5 Training schedule

- fixed 12 epochs;
- no early stopping;
- no post-hoc checkpoint selection;
- final epoch is the governing checkpoint;
- seeds `42`, `43`, and `44` are required for a positive result;
- no learning-rate, support-size, selector, epoch, or pooling sweep.

A compute-saving sequential rule may be frozen:

- run all four arms for seed 42;
- stop the route if the primary system contrast is negative;
- if nonnegative, complete seeds 43 and 44;
- seed 42 alone can reject the route but can never establish success.

### 8.6 Efficient paired execution

The four arms may share one source DataModule and resident query batch to raise
GPU utilization, provided that:

- each arm has an independent model and optimizer state;
- model initial bytes are equal;
- canonical RNG is snapshotted/replayed for paired operations;
- support RNG is isolated and used only by A3;
- no arm's gradients, buffers, or optimizer state can reach another arm.

This is an engineering optimization, not a change to the estimand.

## 9. Scoring matrix

Score every checkpoint under these support laws:

- `CHRONO30` for the M30 product anchor;
- `COV-FIX-E10`;
- `COV-FIX-L10`;
- a frozen distribution of `COV-RAND10` supports;
- `COV-TOP10`;
- `COV-ANTI10`, which chooses the least representative legal member within
  each coverage stratum and is the governing negative selector control.

### 9.1 Required checkpoint-by-selector table

| Checkpoint | CHRONO30 | FIX-E10 | FIX-L10 | RAND10 dist. | TOP10 | ANTI10 |
|---|---:|---:|---:|---:|---:|---:|
| M30-V9-REPRO | primary anchor | K10 OOD diagnostic | K10 OOD diagnostic | K10 OOD diagnostic | K10 OOD diagnostic | K10 OOD diagnostic |
| COV-FIX-E10 | diagnostic | native | required | required | required | required |
| COV-FIX-L10 | diagnostic | required | native | required | required | required |
| COV-RAND10 | product context | required | required | required | **primary** | required |

All selectors see the same first-50 candidate activity and the same T4@50.
Only the selected support differs.

Every K10 cell in the M30-trained row is cardinality-OOD and cannot provide a
clean selector estimand. It is retained only to diagnose how the historical
M30-style checkpoint reacts to reduced support. Governing selector effects are
the K10 cells of the three K10-trained rows.

## 10. Estimands

Let `R(A,L)` be paired session R2 for checkpoint `A` and support law `L`.

### 10.1 Support-augmentation effects

Under every K10 scoring law `L`, report:

```text
Delta_aug_early(L) = R(COV-RAND10, L) - R(COV-FIX-E10, L)
Delta_aug_late(L)  = R(COV-RAND10, L) - R(COV-FIX-L10, L)
```

The primary augmentation effect uses `L=COV-TOP10`, but all K10 laws are
reported. Claim support-composition robustness only if the random checkpoint
does not win solely against one weak fixed support.

### 10.2 Fixed-support content sensitivity

```text
Delta_fixed_content(L) = R(COV-FIX-L10, L)
                         - R(COV-FIX-E10, L)
```

Large values imply strong dependence on which ten fixed trials are used and
weaken a pure augmentation interpretation.

### 10.3 Selector effects for every checkpoint

For each K10-trained checkpoint `A`, report as a governing estimand:

```text
Delta_select(A) = R(A, COV-TOP10)
                  - E_S[R(A, COV-RAND10_S)]
```

This is not restricted to the random-trained checkpoint. The checkpoint ×
selector interaction is directly visible across the three K10-trained rows.
The same arithmetic is reported for M30-V9-REPRO only as a cardinality-OOD
diagnostic and is never pooled with the governing selector effects.

### 10.4 Full-system product contrast

```text
Delta_product = R(COV-RAND10, COV-TOP10)
                - R(M30-V9-REPRO, CHRONO30)
```

This is the practical product comparison. It combines cardinality, support
augmentation, and selector changes, so it is not a pure mechanism contrast.

### 10.5 Anti-selector control

```text
Delta_top_vs_anti(A) = R(A, COV-TOP10)
                       - R(A, COV-ANTI10)
```

Because K10 allows disjoint supports under C50, this is a materially stronger
negative control than the original TOP30/BOTTOM30 comparison.

## 11. Evidence surfaces

### 11.1 Source train and development

- optimization: frozen 27 source-training sessions;
- design/gating: frozen six source-development sessions;
- seeds: 42/43/44;
- no formal-test or external-subject session used for architecture, selector,
  epoch, or threshold choice.

The exact roster and closure must be rebound in a future work order.

### 11.2 External sub-M15

External sub-M15 has been repeatedly scored in prior research. It is available
only as retrospective cross-subject transfer evidence after all V1 choices are
frozen. It cannot validate freshness or support a confirmatory p-value.

### 11.3 Confirmation boundary

The consumed formal sub-C scope is excluded. V1 remains development evidence
unless a separate, genuinely untouched cohort is identified and independently
authorized before its names, neural arrays, or targets are opened.

## 12. Gates

### 12.1 Stage-0 constructibility gate

Require all of the following before GPU execution:

- V9 parent protocol exact;
- `C=50`, `K=10`, and `Q=50` independently represented;
- all required sessions have at least 50 legal activity trials;
- all eight canonical directions represented among valid first-50 labels;
- raw spike-time rate feature finite and nonnegative;
- selector bias/stability gates in Section 7.8 pass;
- sampling branch executes with nonzero subset entropy;
- sorted support indices preserve canonical reduction order;
- with one frozen historical M30 checkpoint, the C30 loader activity trials
  `0..29` and the first 30 entries of the C50 loader are bitwise identical;
- under that same checkpoint and identical first-30 support, the resulting
  identity tensor is bitwise identical;
- prediction forward parity is tested only on the common query subset whose
  windows belong to trials `50+`, and predictions there are bitwise identical;
  no prediction comparison is attempted on old-only trial-30..49 query
  windows;
- task-only loss is shown to be independent of dormant `lambda_E`;
- the exact common teacher checkpoint SHA is bound and its validation-only
  forward dependency is exercised without changing the task-only loss;
- no formal or external target opened;
- `20150313` invalid direction is retained in activity authority but excluded
  from coverage selection and T4 label fit.

Stage 0 is a constructibility/safety gate, not a performance gate.

### 12.2 Multi-seed development gate

A positive product result requires:

- three seeds `42/43/44` complete;
- grand mean `Delta_product >= +0.020`;
- crossed seed-by-session bootstrap 95% lower bound `> 0`;
- at least `4/6` session means nonnegative;
- worst session mean `>= -0.030`.

This replaces the first draft's `+0.010`/single-seed proposal, which was below
the observed small-effect noise scale.

### 12.3 Mechanism gates

For a full TC-AS-EP mechanism claim:

- both primary augmentation contrasts are positive;
- `Delta_select(COV-RAND10) > 0`;
- TOP10 beats mean RAND10 on at least `4/6` session means;
- TOP10 beats ANTI10;
- selector effect is not restricted to a single checkpoint without explicit
  interaction disclosure.

There is no performance-parity gate between M30-V9-REPRO and the immutable
historical selected-M30 result. Their source training/model-selection query
surfaces begin at trials 50 and 30 respectively. Historical performance is
context only; M30-V9-REPRO is the sole matched Q50 product anchor.

### 12.4 Decision table

| Augmentation | Selector | Product | Decision |
|---|---|---|---|
| positive | positive | positive | retain full TC-AS-EP |
| positive | null | nonnegative | retain COV-RAND training; simple support at deployment |
| null | positive | nonnegative | retain selector-only result; no augmentation claim |
| either | either | negative | no product promotion; mechanism result may remain diagnostic |
| null/negative | null/negative | any | stop route; do not open cycle or FiLM as rescue |

## 13. Minimal receipt requirements

Reuse existing receipt infrastructure for generic lifecycle evidence. The
route-specific attribution fields are:

### 13.1 Training

- exact `C/K/Q` values;
- subset indices per sample;
- support-law name;
- subset RNG domain and counter input digest;
- support entropy/coverage counts;
- T4@50 raw/normalized digest;
- shared batch/query/target/dropout evidence;
- trainable/frozen parameter roster;
- optimizer steps and final checkpoint/state SHA;
- target optimizer/backward/update counts equal zero.

### 13.2 Selector and score

- raw spike-rate feature digest;
- direction-label validity mask and counts;
- robust normalizer digest;
- deterministic medoid/marginal-gain trace;
- selected indices and digest;
- duration/direction bias diagnostics;
- identity/query/target/prediction digests;
- paired per-session R2;
- recomputed Section 10 estimands;
- evidence-surface freshness label.

No duplicate generic receipt field list is specified here; inherited lifecycle
templates remain authoritative where compatible.

## 14. Optional successor A — random-support cardinality cycle

Do not include a K cycle in V1. Membership variation and cardinality variation
are different factors.

If V1 is nonnegative and low-K robustness is a verified limitation, a separate
successor may use:

```text
K_t in (30, 10, 4), deterministic schedule
S_t = direction-covered random K_t subset from first 50
T4 = fixed T4@50
query start = 50
```

It must be called a **random-support cardinality cycle**, not a prefix cycle,
because the support is not a chronological prefix.

No cycle is allowed to rescue a negative augmentation result.

## 15. Optional successor B — calibration-profile FiLM

CP-FiLM is a separate task-state hypothesis. It may be opened only if the V1
activity route is at least product-noninferior.

A future DANDI profile may use:

- delay: `[target_on_time, go_cue_time)`;
- reach: `[go_cue_time, trial_end)`;
- per-unit reach-minus-delay and log-rate contrast;
- source-frozen robust normalization;
- zero-initialized residual FiLM at the early identity interface.

Required controls are full profile, zero profile, session constant, and row
shuffle. FiLM must not be described as evidence for support selection, nor may
it rescue failed V1 gates.

## 16. Anticipated objections

### 16.1 “This is no longer label-free”

Correct. Direction coverage uses labels already consumed by T4@50. The honest
claim is **no extra label budget** and **zero target-session updates**, not
label-free selection. This is preferable to knowingly deploying an
activity-only selector that drops entire directions.

### 16.2 “Why choose only ten activity trials after observing fifty?”

The system has 50-trial latency because T4 already requires that pool. K10 is a
compressed activity identity chosen to increase support counterfactual
separation. It is not a 10-trial early-exit method.

### 16.3 “The product contrast confounds K10 with selection”

Correct, which is why it is labeled a product contrast. The K10 matched arms
and selector matrix provide the mechanism estimands; M30-V9-REPRO provides the
practical anchor.

### 16.4 “Random training sees more unique trials than a fixed arm”

Correct. That broader exposure is part of support-set augmentation. Early and
late fixed controls measure sensitivity to fixed support content. The paper
must not claim that stochasticity alone caused a gain.

### 16.5 “Raw activity still correlates with behavior”

Yes. Neural activity necessarily reflects behavior. The method does not claim
statistical independence from behavior; it claims no extra dense target labels
and no query/metric feedback. Duration normalization, direction coverage, and
frozen source-only bias gates prevent the known pathological selector behavior.

## 17. Execution sequence

### Stage 0 — code seam and CPU authority

1. implement the three independent data axes;
2. implement stateless sorted per-sample support sampling;
3. implement raw spike-rate feature extraction;
4. implement direction-covered fixed/random/Top/Anti supports;
5. run synthetic and source-only constructibility tests;
6. run bias/stability diagnostics without decoder R2;
7. prove M30 loader/operator parity;
8. freeze source rosters, closure, gates, and work order.

No GPU or performance authorization is implied.

### Stage 1 — seed-42 four-arm screen

Train M30-V9-REPRO, COV-FIX-E10, COV-FIX-L10, and COV-RAND10 for exactly 12
epochs with coordinated paired batches. If the frozen seed-42 stop rule fails,
terminate the route.

### Stage 2 — seeds 43/44

Only after seed-42 continuation, train the exact same four arms for seeds 43
and 44. No configuration change is permitted.

### Stage 3 — source-development scoring

Score the complete checkpoint-by-selector matrix, publish all paired rows
before aggregate metrics, and apply the frozen product and mechanism gates.

### Stage 4 — retrospective transfer

If source gates pass, score external sub-M15 once under the frozen route and
label it retrospective. Do not alter the route afterward.

### Stage 5 — successors

Open CP-FiLM or cardinality cycling only under their separate triggers and new
work orders.

## 18. Outcome language

### Outcome A — full positive

Random support augmentation beats both fixed K10 controls, COV-TOP10 beats the
random-support distribution and COV-ANTI10, and the product beats M30-V9-REPRO.
This supports the full TC-AS-EP story.

### Outcome B — augmentation only

The random-trained checkpoint improves across selectors, but deterministic
Top10 adds nothing. Retain support-set augmentation and deploy a simpler fixed
or random support law.

### Outcome C — selector only

Top10 improves multiple checkpoints, but random training adds nothing. Retain
the label-budget-neutral activity coreset without a new training claim.

### Outcome D — fixed-content dependence

FIX-E10 and FIX-L10 differ materially and RAND10 does not exceed both. The
result is support-content sensitivity, not robust set learning.

### Outcome E — product negative

K10 loses too much absolute activity information relative to M30. Do not
promote the method even if one mechanism contrast is locally positive.

### Outcome F — complete null

Stop the route. Do not add FiLM, a learned selector, or a cardinality cycle to
manufacture a positive result.

## 19. Revised paper language

If all gates pass:

> Calibration identity is a set rather than an arbitrary prefix. Under the V9
> first-50 calibration contract, we train the native early-pooled identity on
> direction-covered random K10 supports and deploy a deterministic activity
> coreset using only the direction-label budget already required by T4@50. A
> matched checkpoint-by-selector matrix separates support augmentation,
> fixed-support content, selector, and full-system effects.

Cross-dataset formulation, only if separately supported:

> T4-like carriers describe what units encode; activity-selected early pooling
> estimates a stable session realization of that code; calibration-profile
> FiLM supplies only task-state information not already captured by carrier and
> activity identity.

Prohibited claims:

- “label-free selector”;
- “10-trial deployment latency”;
- “randomness alone caused the gain”;
- “fresh formal confirmation on sub-C”;
- “directly comparable to A2 start-30 results.”

## 20. Independent review checklist for the revised design

Before a work order, an independent reviewer must decide:

1. Does the selected M30 lineage evidence in Section 2.1 resolve the prior
   M10/M30 anchor dispute?
2. Is M30-V9-REPRO sufficient to control the new query-start-50 loader, or is
   another immutable anchor required?
3. Is K10 justified despite the risk of losing absolute performance versus
   M30?
4. Are COV-FIX-E10 and COV-FIX-L10 sufficient fixed-support controls, or is a
   deterministic five-block exposure cycle necessary?
5. Does raw spike-time duration-normalized rate exactly match the available
   NWB and T4 rate convention?
6. Is the T4 direction label budget legally reusable for activity support
   coverage without changing the supervision tier?
7. Is the medoid-plus-coverage Top10 algorithm fully deterministic and free of
   query/model/metric feedback?
8. Are the duration bias and jackknife stability thresholds justified before
   any decoder performance is observed?
9. Does excluding the one invalid direction trial from coverage selection but
   retaining it in activity authority preserve a coherent budget?
10. Can the three-axis data seam be implemented route-locally without changing
    old result semantics?
11. Does the stateless per-sample RNG law preserve paired dropout and batch
    equality?
12. Does matching the selected lineage require joint encoder/decoder training,
    and is the parameter roster exact?
13. Is the four-arm × three-seed cost proportionate to the expected effect, or
    should seed 42 be the only exploratory screen with no paper claim?
14. Are the `+0.020`, bootstrap-lower-bound, `4/6`, and worst `-0.030` gates
    appropriate for the actual historical variance?
15. Is any genuinely untouched confirmation surface available, or must all
    outputs be labeled development/retrospective?
16. Are CP-FiLM and cardinality cycling still cleanly separated?

The reviewer should return `GO`, `REVISE`, or `NO-GO`, cite exact code/data
evidence, and list the minimum changes required before an executable work order.

## 21. Review boundary

This revision incorporates the independent audit through failure analysis,
decomposition, and the simplicity test:

- failure analysis retired the interpolated centrality selector and exposed the
  no-op random-calibration path;
- decomposition separated augmentation, fixed-support content, selection, and
  product effects;
- the simplicity test keeps native early pooling, fixed T4@50, no query memory,
  no learned selector, and no FiLM in V1.

No result, target surface, checkpoint, cache, NWB, GPU, or formal-test scope was
opened by this document revision. All numeric audit facts remain subject to
independent reproduction before they become execution authority.
