# Design — Anchored Post-Fusion Capacity Screen V1

Date: 2026-09-02  
Status: design freeze candidate; source-only implementation may begin after independent review  
Working family name: **APFC** (Anchored Post-Fusion Capacity)  
Resource scope: physical GPU0 only; physical GPU1 and its M1 process are out of scope

## 1. Starting point

The completed same-surface APFG result resolves the implementation question
but misses the scientific promotion bar:

| Evidence | Result |
|---|---:|
| Corrected PF-R1 minus PF-MEAN, external | +0.051466, 4/6 |
| Corrected PF-R1 minus native POOLED, external | -0.060687, 1/6 |
| Scalar APFG source validation minus zero | +0.002203, 2/2 |
| Scalar APFG minus native POOLED, external | +0.004612, 4/6 |
| Scalar APFG external bootstrap CI | [-0.003402, +0.013194] |
| Scalar APFG gate increment over uncapped memory, within | +0.000069, 5/7 |
| Uncapped activity memory alone, within | +0.013933, 7/7 |

The exact native/zero control passed on all 13 sessions. Therefore the next
question is not whether the wrapper or checkpoint is broken. It is whether one
global scalar is too restrictive for a residual tensor with shape `[B,N,50]`,
or whether the transferable Post-Fusion residual is intrinsically too small.

The six local external labels have now been observed. They must not select a
new gate, epoch, basis dimension, feature, or hyperparameter. The successor is
selected entirely on source-grouped data and requires an untouched evaluation
surface for any confirmatory claim.

## 2. Problem and insight

**Problem.** Pure Post-Fusion training exposes complementary information but
loses the strong cross-session POOLED solution; exact scalar anchoring restores
that solution but leaves less than half of the required external gain.

**Insight.** Preserve the exact native POOLED zero state while adding only a
very low-dimensional, session-transferable gate over the Post-Fusion residual.
This tests a real capacity bottleneck without reopening the backbone, target
adaptation, carrier updating, or high-capacity architecture search.

All permitted arms have the form

```text
h = h_native + g(state, t) * (h_post - h_native)
```

and must return `h_native` bitwise when every gate parameter is IEEE `+0.0`.

## 3. Divergence: candidates considered before selection

The following candidates were considered before any new source run:

| Candidate | Decision | Reason |
|---|---|---|
| More scalar epochs | Reject | scalar validation already plateaued at epoch 11/12 |
| Scalar seed sweep | Reject | changes uncertainty, not function-class capacity |
| Refit scalar on the six external sessions | Reject | target leakage and invalid deployment |
| Jointly retrain the full PF backbone | Reject | already below POOLED by about 0.06 external after operator correction |
| Per-unit gate `[N]` | Reject | unit identities do not align across sessions |
| Full gate `[N,50]` | Reject | 4,800 target-facing degrees of freedom and severe overfit risk |
| Free per-session alpha | Reject | unavailable for a new target session without labels |
| Positive-only convex blend | Reject | source selected a negative coefficient; it would remove the observed direction |
| Learned recency or EMA memory | Reject | the memory-law scan already favored uniform uncapped accumulation |
| Output smoothing after decoding | Reject | prior work showed low-frequency/common-mode error rather than removable jitter |
| Target pseudo-label gate adaptation | Reject | reopens the failed carrier/pseudo-label problem and weakens attribution |
| Four-parameter smooth temporal gate | **Keep: A-TB4** | low capacity, unit-permutation compatible, tests structured residual geometry |
| Two-parameter disagreement-conditioned scalar | **Keep: A-DC2** | causal and label-free at deployment; tests whether one scalar should depend on pool reliability |
| Temporal gate plus disagreement conditioning | Defer | only legal if both simpler arms independently pass source OOF |

This screen intentionally keeps only two new function classes. It is not an
open hyperparameter search.

## 4. Matched arms

All three arms start from the exact Selected-T4 POOLED checkpoint:

```text
checkpoint_sha256 = 25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
student_state_sha256 = 2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20
```

All inherited parameters remain frozen and in evaluation mode. Only the
listed gate parameters are trainable.

### 4.1 A-S1: matched scalar control

```text
g(state,t) = tanh(a)
trainable parameters = 1
initial a = IEEE +0.0
```

This reimplements the completed scalar APFG under the coordinated runner. It
is the causal control for A-TB4 and A-DC2, not an optional convenience.

### 4.2 A-TB4: four-coefficient temporal-basis gate

Let `B` be the first four fixed orthonormal DCT-II basis vectors over the 50
identity time coordinates, ordered from constant to increasing frequency:

```text
q[t] = sum_k B[t,k] * c[k],  k=0..3
g(state,t) = tanh(q[t])
trainable parameters = 4
initial c = [+0.0,+0.0,+0.0,+0.0]
```

The same 50-value gate is broadcast across units. No learned basis, per-unit
coefficient, spline-knot choice, or basis-size sweep is permitted. The DCT
basis must be constructed analytically in float64, frozen, then converted once
to the model's float32 device tensor with its SHA recorded.

A-TB4 tests whether the scalar gate hides a smooth temporal structure in the
identity residual. Its value is structured capacity, not output smoothing:
the operator acts on the calibration identity before decoding.

### 4.3 A-DC2: disagreement-conditioned scalar

For each causal pool, compute per-trial Post-Fusion values `v_i` using the
already frozen branch. Define the label-free state statistic:

```text
d = sqrt(mean_i mean_{n,t} (v_i[n,t] - mean_i(v)[n,t])^2)
r = sqrt(mean_{n,t} mean_i(v)[n,t]^2)
s = log((d + 1e-8) / (r + 1e-8))
z = (s - source_fit_mean) / source_fit_std
g(state,t) = tanh(a0 + a1*z)
trainable parameters = 2
initial [a0,a1] = [+0.0,+0.0]
```

`z` is scalar and broadcast over unit/time coordinates. During the 5/2 screen,
the mean/std are fitted on the five lexical fit sessions only and frozen for
the two validation sessions. During all-seven refit, they are recomputed once
from all seven source sessions and frozen for deployment. Target statistics
cannot alter them.

The receipt must bind the exact state statistic before any label is read. The
feature set is exactly `{s}`; cardinality, T4 magnitude, session ID, target
prediction, and hidden labels cannot be added after seeing results.

## 5. Common source contract

The screen inherits the completed APFG V1 source contract exactly:

- seven source sessions;
- lexical first five fit / last two grouped validation;
- POOLED/G00m linear activity, never PIT cubic activity;
- first-30 candidate pool with D-opt-k4 support;
- selected-support4 fixed-ridge T4, frozen for each session;
- causal source pool controller cycling only M4/M10/M30;
- task-only scaled last-bin source MSE;
- 12 epochs, Adam `lr=1e-4`, weight decay 0, existing betas/epsilon;
- earliest best equal-session UNCAPPED validation R2;
- no teacher forward, no inherited gradient, no dropout, no target access;
- selected-epoch reset and refit on all seven source sessions.

The same ordered source batches, frozen native/post branch tensors, neural
windows, targets, and controller state must be shared by the three arms.
Python, NumPy, Torch, and CUDA RNG evidence must be matched even though the
inherited branch is in evaluation mode.

## 6. Coordinated GPU runner

The screen is one GPU0 job, not three independent low-utilization jobs:

1. prepare one PIT source DataModule and strict-load one POOLED model;
2. compute each frozen native/post branch once per causal pool state;
3. build the three gated identities from those shared branches;
4. concatenate the three arm batches along the batch dimension;
5. run one frozen decoder forward;
6. split the three losses;
7. backpropagate once and step three independent Adam optimizers;
8. prove only the declared gate parameters received gradients.

Concatenation changes floating-point batching relative to historical APFG V1,
so byte equality to the old scalar curve is not required. Before launch, a
small synthetic/real-source equivalence test must prove that coordinated and
separate execution agree within `2e-6` prediction max-absolute error and
`2e-7` R2, and produce the same selected arm under the frozen source gate.

The job may inspect and use only physical GPU0. It must set
`CUDA_VISIBLE_DEVICES=0`, require one visible logical device, bind its UUID,
and never query, signal, attach to, reserve, or inspect GPU1 or the M1 process.

## 7. Source-only selection and kill criteria

Each arm publishes all 12 validation rows and the two validation-session R2
values. A new arm is eligible only if all conditions hold at its own earliest
best epoch:

```text
candidate mean validation R2 - A-S1 mean validation R2 >= +0.003
candidate positive validation sessions vs A-S1 = 2/2
candidate worst validation-session delta vs A-S1 >= -0.002
candidate mean validation R2 - exact-zero mean validation R2 >= +0.005
```

These are source-only feasibility gates, not paper claims. If neither new arm
passes, stop PF capacity expansion and retain scalar APFG only as a bounded
mechanistic ablation.

If both pass, select the larger equal-session mean. If their means differ by
less than `0.001`, choose A-DC2 because it has fewer parameters and a direct
causal reliability interpretation. Do not build the combined arm in this
screen.

The matched scalar arm must also reproduce the qualitative V1 facts: selected
coefficient is negative, both validation sessions are nonnegative versus zero,
and its validation gain lies within `0.001` absolute R2 of the completed V1
gain `+0.0022026004`. Failure of this control invalidates the screen rather
than favoring a new arm.

## 8. Evaluation boundary

The completed six-session local external surface cannot select or confirm the
new arm because its labels motivated this successor after being observed.
After a source-only winner is frozen:

1. package the winner and its exact A-S1 control for an untouched EvalAI M2
   submission, if the evaluator supplies a legal completed-trial boundary;
2. otherwise stop before submission or create a separately named trial-free
   successor with its own equivalence evidence;
3. if another untouched dataset/fold with the same operator contract is
   available, score both winner and A-S1 exactly once there;
4. treat any re-score on the already observed six local sessions as
   descriptive only.

A meaningful improvement requires at least `+0.010` over the same-surface
native/zero POOLED control on the untouched evaluation and a breadth measure
when per-session scores are available. A source-only win is not sufficient.

## 9. Three validation experiments

### V0 — zero-anchor and geometry

- every arm at all-`+0.0` parameters is bitwise native POOLED;
- A-TB4 basis SHA/orthogonality and unit broadcast are exact;
- A-DC2 statistic is invariant to trial permutation, finite at zero
  disagreement, and changes only with completed causal evidence;
- no target/data/GPU access is required for synthetic coverage.

### V1 — matched source screen

- one GPU0 process, one DataModule, one frozen checkpoint;
- three arms, 12 epochs, all source rows and paired deltas;
- coordinated-versus-separate numerical equivalence;
- source OOF selection and kill decision only.

### V2 — untouched transfer

- one frozen winner plus A-S1/native control;
- no target fitting or target-selected hyperparameters;
- exact same-input/control receipt;
- `+0.010` meaningful-effect gate.

## 10. Strongest objection and response

**Objection.** A-TB4 and A-DC2 are being proposed after observing that scalar
APFG is heterogeneous on six external sessions, so a subsequent local win
would be target-driven overfitting rather than discovery.

**Response.** The local six-session surface is explicitly retired from model
selection and confirmation. The new arms, dimensions, features, source gates,
and tie-break are frozen before implementation; selection uses source-grouped
OOF only; the method claim requires a separate untouched evaluation. If no
untouched surface is available, this work can yield only a source-screen
diagnostic and must not be promoted.

## 11. Two-week feasibility envelope

This is an upper bound, not a reason to delay the first screen:

- days 1-2: adapter/controller implementation, synthetic zero-anchor tests;
- days 3-4: coordinated-runner and real-source numerical equivalence audit;
- day 5: one source-only GPU0 screen;
- days 6-7: independent receipt audit and winner freeze;
- week 2: untouched EvalAI/fold packaging, execution, and final attribution.

No additional PF architecture, seed sweep, target adaptation, or combined arm
is authorized inside this envelope.

## 12. Two-sentence pitch

Native pooling transfers well, while Post-Fusion retains complementary signal
but destroys that strong solution when trained as a replacement. APFC exactly
nests the native decoder and asks whether four smooth temporal coefficients or
one causal disagreement feature can extract a transferable residual that the
completed one-scalar gate was too restrictive to capture.
