# Design: Learnable Causal Output Filter — Frozen-Output and Training-Integrated Routes

Date: 2026-08-29  
Status: DESIGN v1 — for independent review; no training or launch authorization  
Primary route: M2/SUA frozen Cell-D and activity-only CDM  
Optional breadth routes: native M2, H1, and M1 only after the primary gates pass

---

## 0. One-sentence claim

First keep the decoder, activity memory, T4 carrier, normalizers, and all
target-update boundaries frozen, and test whether a tiny stable causal filter
can smooth decoder jitter while following genuine rapid changes with minimal
lag. Only if that matched effect survives may a second route retrain the decoder
with the same deployed filter and an explicit raw-output preservation loss.

The intended contribution is not another temporal decoder. It is a source-trained,
low-parameter inference layer whose gain over a fixed EMA must be demonstrated
under the exact same frozen predictions.

## 1. Why this experiment exists

The completed continuity probe found a nontrivial external signal from causal
output smoothing on the frozen static T4/Cell-D system:

| Budget | Raw static R2 | Best causal smoothed R2 | Delta | Positive sessions | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| M4 | 0.119684 | 0.128441 | +0.008757 | 12/15 | `[+0.002865,+0.014737]` |
| M10 | 0.295456 | 0.314517 | +0.019060 | 13/15 | `[+0.011353,+0.027026]` |
| M30 | 0.428593 | 0.468022 | +0.039429 | 15/15 | `[+0.029712,+0.050279]` |

The corresponding causal within-session effects were approximately null or
negative. This suggests a variance-versus-lag trade-off: cross-subject outputs
may contain more removable high-frequency jitter, while already accurate within
predictions pay more for temporal lag.

This result is exploratory for three reasons:

1. it used the frozen static P4/T4 recipe, not the stronger activity-only CDM;
2. K/alpha were read from a multi-arm external grid, so they are not yet a
   source-only selected deployment recipe;
3. the original smoothing kernel did not use trial IDs to reset at every
   trial/discontinuity boundary.

The first task is therefore not to train a filter. It is to reproduce and
factor the effect on the exact activity-only output stream.

### Immutable evidence anchors

- continuity probe body:
  `tfpd_exploration/results/continuity_probe_v1/continuity_probe_v1.json`,
  SHA256 `8afaa9109f2dbeb1fb68e1044c4e7ae275771113c4cb3daf5c5565ea77c588b1`;
- activity-only body:
  `tfpd_exploration/results/causal_dual_memory_cell_d_activity_only_quick_v2/result.json`,
  SHA256 `48d658c866b0bcb45a9f03c6e0c84204ecb87286e940d850c846a46f67b33bb8`.

These anchors license a new matched experiment. They do not pre-authorize
parameter selection on external target labels.

## 2. Exact intervention

For every chronologically ordered query output, the frozen system produces a raw
governing prediction:

```text
y_t = frozen_decoder(neural_window_t,
                     frozen_or_causal_activity_state_t,
                     frozen_T4_t)
```

The filter produces the emitted/scored prediction:

```text
y_filtered_t = F_theta(y_t, filter_state_{t-1})
```

For the primary frozen-output route, only `y_filtered_t` and the small filter
state differ from the raw system. A conditional training-integrated successor
is specified separately in §7. That successor trains decoder parameters and
therefore must never inherit the frozen-decoder parity claim.

### 2.1 Frozen-output route: what must remain exact

- frozen Cell-D graph and checkpoint;
- all decoder parameters and buffers;
- source-only behavior and T4 normalizers;
- support selection and label budget;
- B3S activity-memory input and transition sequence;
- initial and active T4 carrier;
- carrier proposal count and carrier commit count;
- neural query windows and target arrays;
- last-bin governing metric and equal-session weighting;
- target optimizer, backward, gradient, and parameter update counts: all zero.

For the governing activity-only arm, carrier proposals and carrier commits must
both remain zero. This document does not authorize pseudo-label generation or a
learned carrier gate.

### 2.2 Frozen-output route: what may change

- the final emitted prediction;
- a route-owned causal filter state;
- filter reset events;
- source-trained filter parameters, but only after the fixed-filter and oracle
  gates pass.

### 2.3 Non-goals

In the frozen-output route, this design is not a modification of B3S or Cell-D.
Across both routes, this design is not:

- a T4 update rule;
- pseudo-label smoothing for CDM carrier updates;
- target-time parameter adaptation;
- a new GRU, TCN, Transformer, or temporal decoder;
- a replacement for the learned-gated T4 oracle experiment.

Output filtering and carrier-policy learning must remain separate factorial
axes in every receipt and table.

### 2.4 Two distinct scientific questions

This document keeps two hypotheses separate:

1. **Frozen-output filtering:** does a small causal filter remove error already
   present in an accepted decoder's output stream?
2. **Training-integrated filtering:** does a decoder generalize better when it
   is trained with the same causal output dynamics that will be deployed?

The first is cheap, exactly matched, and scientifically cleaner. The second may
produce a larger gain, but it changes the trained system and introduces a new
decoder/filter identifiability problem. A positive frozen-output result is the
entry condition for the training-integrated route, not evidence that the
training-integrated route will work.

## 3. Causal clock and reset contract

At emitted time `t`, the filter may read only:

- the current raw prediction `y_t`;
- previous raw predictions `y_{<t}`;
- previous filter state;
- current deployment metadata already available without labels.

It must never read:

- `y_{t+1}` or a future neural window;
- current/future target behavior;
- a future trial or recording;
- target-derived hyperparameters.

### 3.1 Ordering proof

Every row must bind:

- session ID;
- recording/trial ID when available;
- exact output-bin index or timestamp;
- window start and governing output bin;
- predecessor row ID or an explicit reset marker.

Duplicate, reordered, or backward timestamps fail closed.

### 3.2 Required resets

Filter state resets before the first prediction of:

- a new session;
- a new recording;
- a non-contiguous window/time gap;
- a new trial in the primary internal-protocol arm;
- a dataloader/shard boundary that cannot prove chronological continuity.

Two reset policies may be studied, but they must be separately named:

1. `TRIAL_RESET`: governing internal M2/SUA protocol when trial IDs are
   available;
2. `STREAM_GAP_RESET`: formal/official-compatible candidate that resets only
   on recording/gap evidence when trial IDs are unavailable.

They cannot share one result label. `STREAM_GAP_RESET` is not allowed until the
formal adapter proves exact chronological ordering and reset semantics.

## 4. Filter ladder

The experiment follows a strict simplicity ladder. A higher level is tested only
if the previous level leaves measurable oracle headroom.

### F0 — Raw output

```text
y_filtered_t = y_t
```

Parameters: 0.  
Role: identity baseline and bitwise fallback.

### F1 — Fixed K2 mean

```text
y_filtered_t = y_t                              at a reset/start row
y_filtered_t = (y_t + y_{t-1}) / 2             otherwise
```

Parameters: 0.  
Role: smallest causal low-pass baseline.

### F2 — Fixed/source-selected EMA

```text
y_filtered_t = alpha * y_t + (1 - alpha) * y_filtered_{t-1}
```

At reset, `y_filtered_t = y_t`.

The alpha grid, if a grid is used, must be frozen before external scoring and
selected only by source/grouped validation. External-15 cannot select alpha.

Parameters: 0 after selection, or 1 if alpha is fit continuously on source.

### F3 — Source-learned causal FIR-K4

```text
y_filtered_t = sum_{k=0..3} w_k * y_{t-k}
w_k >= 0
sum_k w_k = 1
```

Weights are parameterized by a softmax. Available history is renormalized at a
reset/stream head. The same scalar weights are shared across the two M2/SUA
velocity dimensions.

Effective degrees of freedom: 3.

The simplex constraint guarantees:

- bounded output;
- DC preservation;
- no arbitrary rotation or scale transformation;
- a transparent impulse response.

### F4 — Adaptive scalar-gain filter (main learned candidate)

```text
g_t = sigmoid(theta^T z_t + b)
y_filtered_t = (1 - g_t) * y_filtered_{t-1} + g_t * y_t
```

At reset, the route emits `y_t` and records `g_t = 1` by contract.

Interpretation:

- small `g_t`: trust history; suppress jitter;
- large `g_t`: trust the current prediction; reduce lag during real change.

For M2/SUA, `g_t` is one scalar shared by both velocity dimensions. Arbitrary
2x2 gain matrices are forbidden in v1 because they can rotate/scale the
behavior coordinates and learn source-specific axes.

Candidate label-free features `z_t`:

| Feature | Definition | Invariance/safety |
|---|---|---|
| innovation norm | `||y_t - y_filtered_{t-1}||` | rotation invariant |
| raw first-difference norm | `||y_t - y_{t-1}||` | rotation invariant |
| short-history dispersion | trace or norm of trailing raw-output covariance | rotation invariant |
| predicted speed | `||y_t||` | rotation invariant |
| direction-change angle | circular angle between current and previous nonzero velocity | coordinate-rotation invariant |
| budget | M4/M10/M30 literal | deployment metadata |
| activity progress | FIFO length / completed-trial index | label free |
| boundary flag | reset/head/gap indicator | prevents hidden edge behavior |

No raw x/y coordinate, target error, target R2, or future statistic may enter
`z_t`.

Expected parameter count: approximately 5--15.

### F5 — Optional alpha-beta/Kalman-like successor

Only if the adaptive-oracle gate in §8 proves material headroom beyond F4's
function class:

```text
v_pred_t = v_state_{t-1} + a_state_{t-1}
innovation_t = y_t - v_pred_t
v_state_t = v_pred_t + alpha * innovation_t
a_state_t = a_state_{t-1} + beta * innovation_t
y_filtered_t = v_state_t
```

Parameters: 2--6 constrained scalars.

This is not part of the first implementation. GRU/TCN/Transformer filters are
explicitly out of scope unless every constrained filter and state-space oracle
leaves a large reproducible gap.

## 5. Rotational and stability constraints

For two-dimensional M2/SUA velocity, the v1 filter must be SO(2)-equivariant:

```text
F(R y_1, ..., R y_t) = R F(y_1, ..., y_t)
```

for any 2D rotation `R`, up to the frozen numeric tolerance.

This is enforced by:

- shared scalar gains/weights across x/y;
- rotation-invariant adaptive features;
- no coordinate-specific bias;
- no unconstrained output matrix.

Additional requirements:

- finite input implies finite output/state;
- constant input remains constant after initialization (DC preservation);
- state and output remain bounded by the convex-hull rule for F1--F4;
- explicit bypass yields bitwise F0 output;
- repeated identical runs produce identical outputs and state digests;
- no RNG is allowed in evaluation.

## 6. Source-only fitting discipline

### 6.1 Data

Trainable filter parameters may use only approved source/development sessions
with true behavior labels. External-15 and formal/organizer-hidden labels may
not select features, alpha, architecture, threshold, reset policy, or stopping
epoch.

The frozen decoder produces the raw prediction stream once. All filter levels
must consume the exact same raw prediction bodies and chronology.

### 6.2 Loss

Primary fitting loss is session-balanced normalized SSE:

```text
L(theta) = mean_s [
    sum_t ||target_s,t - filtered_prediction_s,t(theta)||^2
    / sum_t ||target_s,t - mean_target_s||^2
]
```

The denominator and any variance weighting are calculated inside each source
session. A long session cannot dominate by window count.

The final evaluation remains the frozen governing convention:

- last-bin prediction;
- variance-weighted multi-output R2 per session;
- equal-session mean;
- paired per-session deltas and sign counts.

### 6.3 Cross-validation

- outer unit: session, never window;
- feature selection, filter level, and parameter tuning occur inside grouped
  inner folds;
- report fold-wise chosen parameters;
- report session counts, window counts, and reset counts;
- ordinary window bootstrap is forbidden;
- confidence intervals resample sessions, not windows.

If the base decoder was trained on a candidate source session, its filter-fit
role must be disclosed. Prefer decoder-held-out within sessions or genuine
source-session-LOSO prediction streams when available.

### 6.4 No target-time adaptation

After source fitting, all filter parameters are immutable. Deployment updates
only the causal filter state, not `theta`, alpha, FIR weights, checkpoint, or
normalizer.

## 7. Conditional training-integrated route

### 7.1 Why integrating the filter during training may help

A post-hoc filter can only attenuate the residual spectrum produced by a fixed
decoder. If the same filter is present during source training, the decoder can
learn representations whose errors are easier for a causal low-pass mechanism
to remove. This can act as a weak temporal regularizer and may improve
cross-session robustness.

The hypothesis is plausible but not free. Joint training can also make the raw
decoder deliberately pre-emphasize, overshoot, or shift its predictions so that
the filter cancels the distortion. The filtered score may then look acceptable
while the raw decoder has become worse or unstable. For that reason the design
uses a dual loss and reports both raw and filtered outputs.

This route is not authorized merely because a fixed filter improves a frozen
checkpoint. It starts only after P1 proves that filtering the stronger
activity-only stream has a real matched effect.

### 7.2 Architecture

For each ordered deployment row:

```text
r_t       = D_phi(neural_window_t, identity_t, carrier_t)
y_hat_t   = F_theta(r_t, filter_state_{t-1})
state_t   = update(filter_state_{t-1}, r_t)
```

`r_t` is the raw decoder prediction and `y_hat_t` is the emitted prediction.
The primary training-integrated filter is the same isotropic scalar EMA or
source-selected fixed FIR used by the frozen-output route. It is not a hidden
GRU or an additional temporal decoder.

The permitted arm ladder is:

| Arm | Decoder | Filter during training | Trainable filter | Purpose |
|---|---|---|---|---|
| J0 | accepted baseline | none | no | raw reference |
| J1 | exact J0 checkpoint | yes, post hoc | source-fit only | frozen-output reference |
| J2 | retrained from the accepted initialization | yes | no; freeze the J1 filter | isolate training integration |
| J3 | retrained from the same initialization | yes | yes, constrained | conditional joint-learning test |

J2 is the first GPU cell. J3 is allowed only if J2 is positive and the adaptive
oracle or source-grouped F3/F4 result shows remaining filter-parameter headroom.

For J3, permitted trainable parameters are only:

- one scalar EMA logit;
- K4 nonnegative FIR logits normalized by softmax; or
- the F4 scalar-gain logistic parameters already defined in §4.

No output-coordinate-specific matrix, recurrent hidden network, TCN, or target
conditioner is permitted. The initial filter must be a source-selected fixed
filter or an exact bypass, and its choice must be frozen before the run.

### 7.3 Loss and anti-compensation constraint

The primary joint source loss is:

```text
L_raw      = normalized_session_MSE(r_t, target_t)
L_filtered = normalized_session_MSE(y_hat_t, target_t)
L_joint    = 0.5 * L_raw + 0.5 * L_filtered
```

Both terms use the same valid rows, source-session normalization, and equal
session weighting. The equal `0.5/0.5` weights are primary and are not selected
on external labels.

The raw auxiliary loss is mandatory. It prevents the decoder from using the
filter as an unrestricted inverse pair and gives a directly auditable
anti-compensation signal. Every checkpoint must therefore retain two source
validation curves and every score must report both `r_t` and `y_hat_t`.

J2/J3 fail the anti-compensation gate if the source-grouped out-of-fold raw R2
falls by more than `0.01` from J0, even when filtered R2 improves. A successor
may test another source-frozen loss weight only after the primary equal-weight
cell is terminal and only with a separate work order.

### 7.4 Chronological training data is a hard requirement

The existing Falcon training path returns a full `[B,W,C]` prediction but, with
`decode_last_timestep_only=true`, computes loss only on `[:, -1:, :]`. The
existing `SessionBatchSampler(..., shuffle=True)` may shuffle windows and batch
order. Therefore an EMA state cannot be carried across ordinary training
batches: adjacent batch rows are not guaranteed to be adjacent in time.

The training-integrated route must use a route-owned chronological sequence
sampler with all of the following properties:

- each sequence contains last-bin predictions from ordered query windows in one
  session and one uninterrupted segment;
- rows inside a sequence are never shuffled;
- complete sequences or blocks may be shuffled only after their internal order
  and predecessor context are sealed;
- a reset marker is supplied at the exact trial/session/recording/gap boundary;
- no filter state crosses sessions or an ambiguous discontinuity;
- every scored row appears exactly once per logical epoch;
- state burn-in rows are explicitly identified and excluded from the loss;
- source-session exposure and optimizer-step counts remain matched to J0.

For the internal M2/SUA trial protocol, the preferred training unit is a full
ordered trial trajectory with an exact trial reset. For a stream-gap contract,
use chronological blocks with exact predecessor context; never introduce an
artificial block reset and call it deployment-equivalent.

It is not valid to replace this with a causal filter over the `W` internal bins
of one model window. The existing governing stream consists of successive
last-bin predictions from successive windows. An earlier internal output from
the current window need not be bitwise or numerically equal to the prior
window's last-bin prediction. An in-window filter would be a different system.

### 7.5 Gradient and state contract

- gradients may flow through `F_theta` and the decoder within one sealed
  chronological training sequence;
- the incoming filter state at a block boundary is reconstructed from exact
  predecessor/burn-in rows and detached before the scored block;
- no gradient crosses a session, trial reset, recording gap, or source fold;
- target-session backward/optimizer/update counts remain zero;
- evaluation uses the identical filter equations, initialization, and reset
  contract as training;
- training and evaluation state-transition digests must be reproducible;
- filter parameters are source-trained and immutable at deployment.

For a fixed EMA/FIR, training-time state storage is negligible. The cost comes
from chronological decoder evaluation and its activation graph, not from the
filter itself. Sequence length, burn-in length, gradient accumulation, and peak
memory must be receipt-bound rather than changed opportunistically.

### 7.6 Required contrasts and gates

The primary training-integration estimand is:

```text
J2_filtered - J1_filtered
```

It asks whether exposing the decoder to a fixed deployed filter during source
training improves on applying that exact filter after training. The contrasts
`J1-J0` and `J2_raw-J0` remain mandatory diagnostics.

Advance J2 beyond source validation only if:

- source grouped-OOF `J2_filtered-J1_filtered >= +0.005`;
- source grouped-OOF `J2_raw-J0 >= -0.01`;
- onset lag, peak lag, and overshoot do not cross their source-frozen safety
  limits;
- no reset, chronology, exposure, or state-digest gate fails.

Claim an external training-integration benefit only if:

- `J2_filtered-J1_filtered >= +0.01` on the predeclared primary budget;
- at least 10/15 external sessions are positive;
- within regression is no worse than `-0.005`;
- J2 uses the same source selection rule and seed policy as J0.

J3 must additionally beat J2 by at least `+0.005` source-grouped OOF before it
may be externally scored. If J2 is positive but J3 is not, the result supports
training with known smoothing dynamics, not learning a more complex filter.

### 7.7 Interpretation boundary

A positive J2 result is a system-training result, not proof that output jitter
alone was the original bottleneck. A positive J1 result with null J2 says the
post-hoc filter is sufficient. A positive J2 with null J1 would be unexpected
and must be treated as a new regularization effect requiring replication, not
as confirmation of the continuity probe.

Because J2/J3 retrain the decoder, they require matched seeds and a matched J0
retraining control. Comparing a new J2 run only with an old sealed checkpoint is
not sufficient to attribute the gain to the integrated filter.

## 8. Adaptive-filter oracle

Before F4 is fitted, measure whether time-varying smoothing strength has value
beyond the best fixed F1--F3 filter.

### 8.1 Noncoherent switch ceiling

For each output row and a fixed candidate gain grid, select the candidate with
the smallest true instantaneous error while holding the same parent history.

This reads target labels and is a leakage diagnostic. It is not a deployable
policy and not a coherent state trajectory.

### 8.2 Coherent greedy gain oracle

At row `t`, compare the predeclared candidate gains using a fixed current or
short future horizon, select one using true source/evaluation labels, then
advance the actual filter state with the chosen gain before row `t+1`.

This is a coherent clairvoyant policy but not necessarily a global optimum.
Every result must retain the word `oracle` and `target_label_leakage=true`.

### 8.3 Oracle disposition

Let `F_best_fixed` be the best source-selected F1--F3 filter.

```text
coherent_adaptive_oracle - F_best_fixed < +0.005
    => STOP adaptive learning; fixed filter is sufficient.

+0.005 to +0.015
    => fit F4 only if grouped source predictability is stable.

>= +0.015
    => adaptive gain has material headroom; proceed to F4.
```

The thresholds apply to the matched equal-session governing R2 scale and are
reported separately for M4, M10, and M30.

## 9. Experiment sequence

### P0 — Runtime/contract audit

Before any new score:

- prove exact chronological ordering;
- prove reset markers;
- prove raw output parity with accepted static/activity-only receipts;
- prove model/activity/carrier state is unchanged by the filter wrapper;
- prove F0 bypass is bitwise equal;
- prove future perturbations do not change earlier causal outputs;
- prove trial/gap boundary tampering fails.

### P1 — Fixed-filter factorial on frozen outputs

Run the exact four-cell interaction matrix:

| Cell | Activity identity | Output filter |
|---|---|---|
| A0 | frozen static support | raw F0 |
| A1 | frozen static support | source-fixed F1/F2 |
| B0 | causal activity-only | raw F0 |
| B1 | causal activity-only | exact same F1/F2 |

Primary readings:

```text
A1 - A0                  static output-filter gain
B1 - B0                  activity-only output-filter gain
(B1 - B0) - (A1 - A0)    activity/filter interaction
```

This stage does not fit F3/F4 and does not create pseudo labels.

### P2 — Source-learned fixed filter

Fit F2 alpha and F3 FIR simplex weights using source/grouped validation only.
Compare against F1 K2 and the predeclared fixed EMA. Select at most one fixed
filter for the next stages.

### P3 — Adaptive oracle

Run §8 on the exact B0 raw activity-only streams and the selected fixed filter.
If the oracle disposition stops, do not implement F4.

### P4 — Source-fit F4

Fit one logistic scalar-gain filter. No tree, MLP, soft carrier gate, or state
space expansion is allowed in this stage.

Primary source readout is grouped out-of-fold realized R2 gain over the selected
fixed filter, not AUC.

### P5 — External matched score

Freeze all filter bytes and score once on the predeclared external roster.

Required external rows:

| Row | System |
|---|---|
| R0 | raw static |
| R1 | fixed-filter static |
| R2 | raw activity-only |
| R3 | fixed-filter activity-only |
| R4 | learned-FIR activity-only, if different from R3 |
| R5 | adaptive-gain activity-only, only if P3/P4 passed |
| O | adaptive oracle diagnostic, separately leakage-labelled |

No learned-filter number may be reported only against raw output. Its primary
contrast is always against the best source-selected fixed filter.

### P6 — Conditional training-integrated pilot

Only if P1 and the source gates in §7 pass, run J0/J1/J2 on one primary source
fold and one seed. Treat this as a mechanism pilot, not a performance claim.

J3 is not part of the first pilot. It is enabled only by positive J2 plus
independent filter-parameter headroom. A full seed/fold matrix is authorized
only after the pilot passes chronology, raw anti-compensation, lag, and matched
filtered-score gates.

### P7 — Formal/official evaluation

Only after a separate adapter audit proves:

- exact temporal ordering;
- persistent state lifetime;
- recording/gap resets;
- no batch/shard reorder;
- exact output-shape mapping;
- measured latency and state memory;
- no target-label or target-parameter access.

The local internal-protocol result and official `STREAM_GAP_RESET` result must
remain separate if their reset information differs.

## 10. Pre-registered performance gates

### 10.1 Fixed filter

Advance a fixed filter only if, on the activity-only primary surface:

- external equal-session mean delta is at least `+0.01` on M10 or M30;
- at least 10/15 external sessions are positive on that budget;
- within equal-session delta is not below `-0.005`;
- no unexplained state/reset/input mismatch occurs.

M4 is reported independently and may advance with a smaller effect only if its
session CI is positive and no safety gate fails.

### 10.2 Learned fixed FIR

F3 must beat the best nonlearned F1/F2 filter by at least `+0.005` source-grouped
OOF before external scoring. Otherwise retain the simpler filter.

### 10.3 Adaptive gain

F4 advances to a performance claim only if it beats the best source-selected
fixed filter by:

- at least `+0.01` external equal-session R2 on a predeclared primary budget;
- at least 10/15 positive external sessions;
- within regression no worse than `-0.005`;
- worst external session regression no worse than `-0.03` unless a stronger
  pre-existing safety convention is required;
- no target adaptation or reset violation.

Any `0` to threshold result is recorded as a null relative to the fixed filter,
even if both filters beat raw output.

### 10.4 Training-integrated filter

The training-integrated route uses the additional J2/J3 gates in §7.6. Its
headline comparison is against J1 with the identical filter, not only against
raw J0. A J2/J3 result that lacks a matched same-seed J0/J1 chain is descriptive
and cannot support a causal design claim.

## 11. Receipt requirements

Every cell/session records:

- raw prediction SHA256;
- filtered prediction SHA256;
- target/input authority SHA256;
- filter type and exact parameter payload SHA256;
- state-in/state-out SHA256;
- ordered reset event list and digest;
- chronological row/trial digest;
- per-session raw and filtered R2;
- paired delta;
- maximum and mean gain `g_t` for adaptive filters;
- gain/weight histograms with fixed bins;
- number of filter-state transitions;
- model/activity/carrier state-before/state-after digests;
- target optimizer/backward/update counts;
- wall time, throughput, and added latency;
- source-selection and outer-fold lineage;
- oracle/leakage flags where applicable.

Training-integrated cells additionally record:

- J0 initialization/checkpoint and same-seed pairing;
- raw and filtered loss curves and checkpoint-selection values;
- raw and filtered prediction digests;
- filter parameter/state digests at every checkpoint;
- sequence/block membership, order, reset, predecessor, and burn-in digests;
- scored versus burn-in row counts;
- source-session exposures and optimizer-step counts;
- raw/filtered gradient norms at the decoder output;
- raw R2, filtered R2, onset lag, peak lag, and overshoot;
- filter bypass evaluation from the same trained decoder;
- proof that target backward/update counts are zero.

Artifacts follow the lane conventions: immutable body plus canonical sidecar,
fresh root, explicit closure, attempt-before-data/model, and atomic terminal or
failure publication.

## 12. Cross-dataset extension

### 12.1 Native M2 and SUA/pseudo-MUA

These are the primary breadth targets because the output is 2D velocity and the
same isotropic filter class is meaningful. Each view keeps its own raw prediction
authority. A filter trained on one view cannot silently claim transfer to another.

### 12.2 H1

Start with one shared scalar gain across all seven outputs. If behavior-group
timescales are demonstrably different, a successor may use a small predeclared
grouping such as translation/rotation/grasp, with one gain per group.

Per-output unconstrained filters are not authorized in v1. H1 activity-memory
results remain a separate mechanism; this filter must compare on the identical
frozen H-C/H-S outputs and chronology.

### 12.3 M1

M1 may contain genuine high-frequency EMG structure. Before filtering, run a
source-only frequency/lag screen. A low-pass transferred from M2 is not assumed
safe. M1 advances only if a source-grouped fixed filter is nonnegative and an
oracle demonstrates removable rather than task-relevant high-frequency energy.

## 13. Stop conditions

Stop the entire learned-filter branch if any of the following occurs:

1. B1 does not improve over B0 after exact activity-only replay;
2. fixed-filter gain disappears after source-only parameter selection;
3. gain requires target-selected alpha/K/reset policy;
4. adaptive oracle leaves less than `+0.005` beyond the best fixed filter;
5. source grouped-OOF adaptive gain is null/negative;
6. external improvement comes with material within or worst-session harm;
7. official ordering/reset semantics cannot support a causal state;
8. a higher-capacity filter is required to match a simpler fixed EMA;
9. filter state or output is nondeterministic;
10. any target parameter, normalizer, carrier, or decoder state changes;
11. chronological source sequences cannot be reconstructed without changing
    the governing sample/exposure contract;
12. J2 improves filtered output only by materially degrading its raw output;
13. J2 fails to beat applying the same filter post hoc to matched J0;
14. J3 fails to beat fixed-filter J2 after source-grouped validation.

Do not respond to a STOP by adding a GRU, longer K, more layers, target-time
fine-tuning, or a larger feature search.

## 14. Interpretation map

| Outcome | Interpretation | Next action |
|---|---|---|
| Fixed filter positive; adaptive oracle null | simple output jitter is removable; no state-dependent headroom | publish/retain fixed filter |
| Adaptive oracle positive; F4 OOF positive | smoothing strength is predictably state dependent | run one frozen external score |
| Adaptive oracle positive; F4 OOF null | action value exists but is not identifiable from deployed features | stop learned filter; record boundary |
| External positive; within negative | cross-subject denoising/lag trade-off | retain external-only claim with safety disclosure |
| All filters null/negative | output jitter is not the bottleneck | close route |
| F4 only matches F1/F2 | learning unnecessary | retain simpler filter |
| J1 positive; J2 null | post-hoc filtering is sufficient | do not retrain decoder for filtering |
| J2 positive over J1; raw safe | training with deployed dynamics adds value | replicate matched seeds/folds |
| J2 filtered positive; raw collapses | decoder/filter compensation | reject integrated route |
| J2 positive; J3 null | known filter is useful, learned complexity is not | retain fixed integrated filter |

## 15. Questions for independent review

1. Should the primary internal reset policy be `TRIAL_RESET`, or should the
   formal-compatible `STREAM_GAP_RESET` be primary from the beginning?
2. Is source-session normalized SSE the right fitting objective, or should the
   filter optimize a source-only differentiable variance-weighted R2 surrogate?
3. Should adaptive gain use budget/activity progress, or would that risk a
   dataset-specific shortcut that weakens cross-dataset transfer?
4. Is a single SO(2)-equivariant scalar gain sufficient for M2/SUA, or is there
   evidence that radial and tangential errors require a constrained two-gain
   decomposition?
5. What is the minimum oracle-over-fixed gain that justifies fitting F4: `+0.005`,
   `+0.01`, or a fraction of the raw-to-fixed improvement?
6. For H1, should output groups be defined from task semantics before any score,
   or should v1 remain a single shared scalar regardless of DoF heterogeneity?
7. Is the equal raw/filtered joint-loss weighting sufficient to prevent decoder
   pre-emphasis, or should the primary route also anchor raw predictions to the
   matched J0 checkpoint?
8. For trial-aware M2/SUA training, should a complete trial be the only legal
   sequence unit, or is a predecessor-complete fixed block equally defensible?
9. Should J2 be run only with the best fixed source-selected filter, or should
   K2 and EMA each receive one matched pilot to separate kernel shape from
   training integration?

## 16. Recommended first execution

The first executable successor should contain only P0 and P1:

```text
raw static
fixed-filter static
raw activity-only
fixed-filter activity-only
```

Use one source-frozen conservative filter, preferably K2 mean or a source-selected
single-alpha EMA, with exact trial/gap resets. Do not implement F3/F4 until the
activity-only `B1-B0` effect is known.

Do not start J2/J3 in parallel with this first execution. If B1-B0 is positive,
the next engineering task is a chronology-preserving source sequence sampler
and a one-fold J0/J1/J2 pilot. J2 should use the already selected fixed filter;
J3 remains conditional on additional oracle or source-OOF headroom.

Final design principle:

> Learn only the amount of temporal filtering that the fixed-filter residual and
> coherent adaptive oracle can justify. A one-parameter or ten-parameter causal
> filter is a plausible contribution; another temporal neural network is not.

---

## 17. Operator disposition on the completed frozen-output route (2026-08-29) — LINE FROZEN

P0-P5 executed and sealed (incl. the successor P5 receipt
`results/learnable_output_filter_v1_p5/r3_external.json`, anchors exact vs sealed
stage-A B1 on both surfaces). Final disposition:

| Item | Decision |
|---|---|
| Scientific integrity | **PASS** |
| Engineering usability | **α=0.7 retained** as `SAFE_CAUSAL_OUTPUT_FILTER_ALPHA_0P7` — a frozen, pre-registered secondary configuration |
| Main-method value | **STOP** — not a positive result worth mining |
| Complex filters (F3/F4/F5, further ladders) | **STOP** |
| Training-integrated route (J0′/J0/J1/J2/J3) | **NO-GO** — the J-spec is retained as a document only; no training authorized |
| Paper position | Appendix / minor ablation |
| Default scoring in successor carrier experiments | **raw output** (no filter in the primary chain) |
| α=0.7 | Secondary sensitivity/configuration only; never re-selected against target results |
| α=0.25 | External-within trade-off diagnostic only |

Recorded reading of the whole line: (1) output-side removable noise exists but is
minor; (2) strong smoothing only moves along an external-within Pareto front;
(3) the low-label problem must be solved at the carrier/identity/trial-state
level; (4) action continuity, if it becomes a contribution, must stand as
representation learning with carrier utility — never as final-output low-pass.

Research resources shift to AC3-0 (amended, R-GE primary baseline).
