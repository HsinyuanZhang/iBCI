# T4 network optimization directions

**Status:** analysis and candidate convergence; no new experiment launched  
**Updated:** 2026-07-30<br>
**Evidence base:** DANDI 000688 sub-C/CO validation development evidence only

## 0. Starting point

This document starts from five measured facts, not from an unconstrained architecture search:

1. SUA T4 is `effective`: `T4-F0=+0.2528`, 6/6 validation sessions and 3/3 seeds
   positive.
2. pseudo-MUA T4 is also `effective`: `T4-F0=+0.3177`, again 6/6 sessions and
   3/3 seeds positive.
3. Under T4, the residual absolute SUA advantage is only
   `SUA-pseudo-MUA=+0.0406 ± 0.0153 SE`; without T4 it is `+0.1056`.
4. B3T improves the no-side-feature backbone by `+0.0418 ± 0.0090 SE` over six
   seeds while reducing learned parameters by about 31% and session-path MACs by about 65%.
5. A static electrode-index reliability scalar does not help:
   `T4GATE-T4=-0.0108 ± 0.0049 SE`, group verdict `ineffective`.

The optimization target is therefore:

> Preserve the per-session functional identity that makes T4 work, determine which part of
> `[a,c,m,b]` actually carries the gain, and improve its interaction with the temporal
> activity embedding without returning to array-specific lookup tables.

This is a **functional conditioning** problem, not an invitation to add more generic
attention or a larger encoder.

## 1. Deployment-cost and supervision audit

The current protocol has two distinct calibration budgets that must not be collapsed into one:

- B3/SPINT activity identity uses `calibration_n=30`, selected as the first 30 trials from
  the protocol pool;
- T4 itself is fitted from the firing rates and `target_dir` labels of the first
  `side_feature_pool_size=50` rewarded trials.

Formal evaluation begins after trial 50 for every matched arm. Thus `pool_size=50` is a common
evaluation boundary for F0 as well as T4, but only T4 consumes all 50 trials' behavioral
direction labels and rates. The measured result is therefore **T4@50**, not T4@30. No existing
result establishes that 30, 15, or 10 labeled trials are sufficient.

At the encoder cost reference shape `N=64, T=100, M_activity=30`, T4 is B3S with four
side-feature inputs appended only to the first post-pool affine layer:

| Quantity | F0 / B3 | T4 / B3S | Increment |
|---|---:|---:|---:|
| Learned parameters | 18,034 | 18,290 | +256 (+1.42%) |
| FP32 weight bytes | 72,136 | 73,160 | +1,024 B |
| Session-path MACs | 13,017,088 | 13,033,472 | +16,384 (+0.126%) |
| 50 Hz online-path MACs | unchanged | unchanged | 0 |
| Stored T4 values at `N=64`, FP32 | 0 | 1,024 B | +1,024 B |

The model `cost_profile()` does not include raw T4 extraction. The present Python
implementation counts spikes in 50 trial windows, averages rates by up to eight canonical
directions, and calls a three-coefficient least-squares fit per unit/channel. That computation
is session-rate and cacheable, but the generic NWB/Python path is I/O-heavy: in the
pseudo-MUA bridge, cached construction for 27 train plus 6 validation sessions took about two
minutes with four CPU workers. This is an engineering preprocessing cost, not a 50 Hz decoder
cost.

For deployment, the fit can be made much cheaper than the current generic implementation:
stream per-direction spike-rate sums and counts, precompute the pseudoinverse for the observed
canonical direction set, and apply one batched `3 x D` projection to all channels. There is no
need to retain all 50 trial-rate rows or run a generic SVD per channel. The scientific cost that
remains is the need for labeled target directions and the latency of a structured calibration
block, not sustained MAC throughput.

A read-only prefix diagnostic on the six current validation sessions gives a useful, but
non-performance, lower-budget bound. The first 5 trials cover only 4–5 of the 8 directions;
the first 10 cover 7–8; the first 15 still leave one direction absent in two sessions; and the
first 20 cover all 8 directions in all six sessions. However, at 20 trials the least-observed
direction has only 1–2 repeats per session, versus 4–6 repeats at 50. This makes
`M_T4=20` a reasonable first efficiency target and explains why a balanced target schedule or
uncertainty-aware shrinkage may matter, but it does **not** establish that T4@20 preserves R².

## 2. What is and is not established

### 2.1 Established

- Correct per-unit/channel T4 content matters; full-row permutation returns SUA to baseline
  and hurts pseudo-MUA.
- T4 is not SUA-specific; it survives substantial electrode pooling.
- A session-functional descriptor is more useful than a static electrode-index scalar.
- Smooth temporal-basis regularization (B3T) is independently useful and hardware-favorable.

### 2.2 Not established

Current T4 is:

```text
[a, c, m, b] = [m*cos(phi), m*sin(phi), sqrt(a^2+c^2), baseline_rate]
```

This representation conflates at least three mechanisms:

- preferred direction / phase (`phi`);
- modulation strength (`m`);
- baseline firing-rate scale (`b`).

`m` is mathematically redundant given `a,c`, and TS4 permutes the entire row. Therefore the
existing result proves that **aligned T4 content** is valuable, but does not prove that the
entire `+0.25–0.32 R²` is specifically caused by preferred direction. In particular,
pseudo-MUA sums units within an electrode, so a correctly aligned baseline-rate or scale
descriptor could plausibly become more valuable after pooling.

This attribution gap is the first thing the next screen must close. Optimizing the network
before closing it risks building a more elaborate mechanism around the wrong feature.

## 3. Ideation frameworks used

The candidate list below was generated with three complementary lenses:

1. **Failure and boundary probing:** explain the residual SUA/pseudo-MUA gap, the failed
   T4GATE, and session-specific reversals.
2. **Composition and decomposition:** split T4 into causal components and combine the two
   independently positive mechanisms, T4 and B3T.
3. **Simplicity test:** ask whether structured affine conditioning or a direct functional
   readout can retain T4 performance with less machinery than unconstrained concatenation
   plus a deep post-pool MLP.

## 4. Divergence: raw candidate list

The list is intentionally broader than the recommended program. “Kill/defer” entries are
retained so they are not rediscovered and rerun later.

| ID | Raw idea | Mechanism tested | Initial disposition |
|---|---|---|---|
| C1 | T4 component ablation: phase, magnitude, baseline | What actually drives T4? | **Required first** |
| C2 | Shuffle calibration `target_dir` labels before fitting T4 while preserving spike rates | Directional content vs rate metadata | **Required first** |
| C3 | Combine B3T temporal basis with T4 side features | Composition of two independent positives | **High priority** |
| C4 | Add T4 fit-confidence descriptors | Use per-session measurement reliability, not electrode ID | **High priority** |
| C5 | FiLM-style affine modulation of activity features by T4/confidence | Explicit multiplicative T4–activity interaction | **High priority** |
| C6 | Low-rank bilinear fusion instead of full FiLM | Hardware-cheaper multiplicative interaction | Merge into C5 pilot |
| C7 | Vary labeled T4 budget `M_T4=5/10/15/20/30/50` with activity budget tracked separately | Deployment value and uncertainty robustness | High priority after C1 |
| C8 | Paired SUA/pseudo-MUA view augmentation | Make one weight set robust to split/merge granularity | Medium/high priority |
| C9 | Cross-view prediction-consistency loss | Preserve behavior prediction under electrode pooling | Merge into C8 |
| C10 | Direct nonlinear functional readout `w_i=f(T4_i, confidence_i)` | Strong simplicity/control for the current decoder path | Medium priority |
| C11 | T4-guided soft functional slots | Fixed hardware interface organized by tuning, not index | High risk; later |
| C12 | Phase/lag-dependent tuning descriptors | Static T4 may miss temporal tuning shifts | Later, after C1/C3 |
| C13 | Circular/SO(2)-equivariant phase encoding | Coordinate-rotation generalization | Later external-scope idea |
| C14 | Static electrode anchor/embed tables | Stable array-specific prior | **Kill/defer:** gate already failed |
| C15 | Larger attention, longer training, or generic SWA tuning | More capacity/optimization | **Kill:** no mechanism and prior evidence is weak/negative |
| C16 | T4-conditioned decoupled cross-attention with identity-only keys and activity-only values | Separate “which unit?” from “what is it doing now?” in the decoder | Conditional after Rank 3; decoder-changing experiment |

## 5. Convergence filters and kill criteria

Every candidate is filtered by:

- **Attribution:** does it isolate a mechanism rather than change several things at once?
- **Paired measurability:** can same-seed, same-session contrasts cancel the dominant seed
  effect?
- **Deployment:** does it stay on the session-rate calibration path, or materially reduce
  online token/state cost?
- **Cross-view relevance:** can the mechanism plausibly work for both SUA and electrode-level
  signals?
- **Simplicity:** is the smallest version enough to test the hypothesis?
- **Array transfer:** does it avoid a table whose indices only make sense for one implant?

Unless a dedicated non-inferiority objective is stated, the V4 gate remains:

- `effective`: mean delta at least `+0.03`, at least 5/6 sessions positive, all seed means
  positive;
- `ineffective`: `mean + 2*paired_SE < +0.03`;
- otherwise `effective_heterogeneous` or `indeterminate` per the shared classifier.

Additional rules:

- no directional claim from fewer than three seeds;
- no selective arm/seed addition after looking at results;
- every content mechanism needs its own dimension/parameter-matched control;
- no held-out test session is opened while developing these variants;
- an efficiency variant may pass by **non-inferiority** if its lower 2SE bound is above
  `−0.03` and it reduces both learned weights and MACs by at least 25%;
- any candidate that requires an array-specific lookup must outperform an equally sized
  session-functional alternative, not merely F0.

## 6. Ranked directions

### Rank 1 — Mechanism screen: factorize T4 before changing the network

This is a diagnostic screen, but it has the highest priority because every later architecture
depends on its answer.

Minimum useful arms:

- `Tphase = [cos(phi), sin(phi)]`;
- `Tmag = [m, b]`;
- `Tac = [a, c]`;
- full `T4 = [a, c, m, b]`;
- `T4-label-shuffled`: permute `target_dir` across calibration trials before the cosine fit,
  preserving each unit's firing-rate samples and therefore its baseline-rate information;
- existing row-shuffled `TS4`.

Primary questions:

1. Does direction-only retain most of T4?
2. Does `[m,b]` explain the larger pseudo-MUA gain?
3. Does target-label shuffling remove the gain while preserving baseline rate?
4. Is redundant `m` useful to the finite MLP despite being derivable from `a,c`?

This screen can simplify T4, sharpen the scientific claim, and determine what the confidence
model in Rank 3 must quantify.

### Rank 2 — T4 × B3T: compose the two established positive mechanisms

Current T4 uses B3S's raw `Linear(100→64)` temporal projection. B3T replaces that projection
with a fixed 12-bump raised-cosine basis followed by `Linear(12→64)`.

The clean factorial is:

| Temporal encoder | No tuning | Real T4 | T4 row-shuffled |
|---|---|---|---|
| raw B3/B3S | B3 | T4 | TS4 |
| B3T family | B3T | **B3T+T4** | B3T+TS4 |

Primary contrasts:

- `B3T+T4 − T4`: does temporal regularization add on top of T4?
- `B3T+T4 − B3T`: does T4 remain effective on the compressed temporal backbone?
- `B3T+T4 − B3T+TS4`: is correct content still required?

The reviewed `B3TS` implementation now makes this composition explicit. Since
2026-07-31, B3T/B3TS also expose a true chronological bin API: the hardware-facing
execution retains `[N,K=12]` current-trial basis coefficients and never a complete
`[N,T]` trial. This is the `B3TStream` execution of the same learned network, so it
does not introduce a second set of weights or a redundant accuracy arm. At the frozen
reference shape `N=64,T=100,M=30`, its measured `cost_profile()` is:

| Encoder | learned parameters | MAC/session | support state | transient trial state |
|---|---:|---:|---:|---:|
| fresh T4/B3S | 18,290 | 13,033,472 | 16,384 B | 25,600 B (`N*T`) |
| B3TStream+T4/B3TS | 12,658 | 4,524,032 | 16,384 B | 3,072 B (`N*K`) |

This is a 30.79% parameter reduction and 65.29% session-path MAC reduction with no support
state increase and an 88% reduction of current-trial transient state. The fixed
raised-cosine basis remains a non-persistent buffer. Batch versus vectorized full-trial is
exact; bin-stream execution agrees within `atol=2e-5` under variable lengths and adversarial
padding, and the targeted B3T/B3TS CPU suite reports `19 passed`. These are
implementation/profile results, not an R² result.

This variant is worth keeping even if it merely matches T4 within a predeclared `−0.03`
non-inferiority margin, because it would dominate on deployment cost.

The fail-closed experiment uses fresh, matched `T4`, `B3T+T4`, and `B3T+TS4` runs with the
same strict split, seeds, first-30 activity/T4 support, 12-epoch window and trials `[30:]`
evaluation. Deployment effectiveness requires the paired seed-level lower-2SE bound for
`B3T+T4−T4` to be at least `−0.03`, both measured reductions to be at least 25%, no state
increase, and the full strict `B3T+T4−B3T+TS4` content gate. Accuracy superiority remains a
separate sufficient path.

### Rank 3 — Confidence-conditioned FiLM, not electrode reliability lookup

The matched SUA mainline fits T4 from 30 rewarded calibration trials; the label-efficiency
screen additionally builds an ordinary `T4@50` reference. Neither receives explicit
information about fit uncertainty. The failed T4GATE used a fixed scalar indexed by electrode;
it cannot know whether **this session's** T4 estimate is reliable.

Candidate confidence descriptors, computed only from the calibration block:

- residual variance or deviance of the cosine fit;
- modulation-to-residual ratio;
- total calibration spike count / exposure;
- directional condition balance;
- bootstrap or analytic standard error of `a,c`.

The smallest structured fusion is:

```text
h_i = temporal_activity_embedding(unit_i)
[gamma_i, beta_i] = small_mlp(T4_i, confidence_i)
h'_i = (1 + gamma_i) * h_i + beta_i
E_i = post_pool([h'_i, T4_i])
```

Zero-initialize `gamma/beta` so the model starts exactly as the selected T4 baseline. Use a
small or low-rank modulation dimension; this keeps the mechanism on the session-rate path and
tests an explicit T4–activity interaction rather than another identity table.

The most valuable endpoint is not necessarily higher R² at `M_T4=50`. A stronger deployment
claim would be:

> At `M_T4=10`, `15`, or `20`, confidence-conditioned T4 comes within `0.03 R²` of ordinary
> T4 at `M_T4=50`, without increasing online state.

This is a hypothesis and an experiment target, not an existing result. Unless a joint-latency
study is explicitly declared, keep the B3 activity budget fixed at `M_activity=30`, vary only
the number of trials whose target labels/rates enter T4, and keep the common evaluation start
at trial 50 so all arms see identical evaluation windows.

#### Frozen Stage-0 implementation contract, confidence v2 (2026-07-31)

The reviewed implementation uses a cached six-vector `[T4(4), C(2)]` from exactly the same
chronological first-`M_T4` rewarded labelled/rate support. It does **not** reuse activity from
trials outside that T4 support to manufacture confidence. The two confidence coordinates are:

\[
C_i =
\left[
\log(\hat{\sigma}_i^2+\epsilon),\
\frac12\log\kappa(G_{ac})
\right],
\qquad
G_{ac} =
\left[(X^\top X)^{-1}\right]_{a,c}.
\]

Here `X=[1,cos(theta),sin(theta)]` is built only from valid target-labelled trials.
`hat(sigma)^2` is the trial-rate residual variance around the **actual selected T4
equal-per-direction-mean fit**, not around a second trial-weighted refit. Thus the descriptor
measures uncertainty of the T4 value the network really consumes under direction imbalance.
Rank-3 and finite-condition checks fail closed before fitting.

Before any FiLM candidate was launched, a 27-session **train-only input audit** found that the
earlier covariance-area coordinate was nearly a duplicate of residual variance
(`r=0.975` after within-session centering) and its between-session standard deviation was
only `0.0087`. It was therefore replaced, without looking at validation performance, by the
scale-free a/c uncertainty shape above. The resulting session geometry ranges from `0.024`
to `0.180` on train sessions (`SD=0.0409`). Confidence v2 consequently has one
unit-specific noise coordinate and one session-level directional-balance coordinate, and
uses a distinct cache semantic version so v1 artifacts cannot be silently reused.

A second train-only audit tests predictive validity rather than mere non-duplication. For
each of the same 27 training sessions, it fits T4/confidence on rewarded trials `[0:M]` and
scores the frozen cosine fit on later rates (`[M:50]` for `M=10/15/20/30`; `[50:80]` for
`M=50`). A leave-one-session-out ridge model using T4 alone is compared with the same model
plus confidence. Current confidence improves future-error R² by `+0.155`, `+0.175`,
`+0.204`, and `+0.237` at `M=10/15/20/30`, and by `+0.358` at `M=50`. At `M=50`,
residual variance alone has `rho=0.932` with future error and gives `R²=0.8908`; geometry
has only `rho=0.070`, and the two-coordinate model gives `R²=0.8902`. Exposure/entropy/
analytic-SE expansion does not improve it (`R²=0.8885`).

This does not establish decoding gain, but it sharpens the failure branch: if the matched
FiLM screen fails, do not declare confidence uninformative and do not widen the descriptor
set. Run one parameter-matched **residual-only** modulation round with a residual-only row
shuffle. The initial full-FiLM and shuffled-C fit-validation trajectories are nearly
identical, so this fallback freezes the selected-T4 temporal/post-pool substrate and
decoder and trains only the four confidence-head tensors (1,208 parameters) for all three
matched residual arms. If that also fails against T4 continuation and NoFiLM, kill FiLM
and advance to the independently motivated decoupled-K/V and B3TStream+T4 candidates.

A simpler low-label branch now has stronger train-only support than another M50 fusion.
Shrink only T4's modulation coefficients with
`q=(a²+c²)/(a²+c²+3·sigma²·trace([(X'X)^-1]ac))`, set
`(a',c',m')=q·(a,c,m)`, and leave `b` unchanged. Nested LOSO across the 27 training
sessions independently selected Wiener strength 3 in all 27 folds at M10, M15 and M20.
At M15 the later-trial geometric rate-MSE ratio is `0.9537`, with improvement in 27/27
sessions. This motivates one fixed M15 decoding pilot—ordinary T4, aligned T4W3 and
unit-shuffled TS4W3—against the existing M50 reference. It is a candidate, not yet a
decoding result; validation and formal data played no role in selecting its formula,
budget or strength.

All five arms copy the same ordinary T4 final-epoch student encoder **and decoder**, discard
optimizer state, and start with newly added residual heads at exactly zero:

- `T4 continuation`;
- `FiLM(C)`;
- `FiLM(shuffle C)`, shuffling only confidence columns while T4 remains aligned;
- `NoFiLM-match(C)`, the same conditioning/head parameter count but additive-only, with no
  `gamma*h` interaction;
- `FiLM(TS4)`, shuffling only T4 columns while confidence remains aligned.

#### Seed-42 matched final result (2026-07-31)

All five arms completed strict validation, passed their individual protocol/receipt checks,
and were independently recomputed against the final aggregate:

| Arm | mean R² |
|---|---:|
| T4 continuation | 0.590273 |
| FiLM(C) | **0.593672** |
| FiLM(shuffle C) | 0.590890 |
| NoFiLM-match(C) | 0.592974 |
| FiLM(TS4) | 0.277239 |

The required contrasts are already negative under the predeclared mechanism gates:

- `FiLM−T4 continuation = +0.003399`, 4/6 sessions positive, 95% CI
  `[-0.000141,+0.006661]`, exact paired Wilcoxon `p=.21875`;
- `FiLM−shuffle C = +0.002782`, 5/6 positive, 95% CI
  `[-0.004685,+0.008329]`, `p=.4375`;
- `FiLM−NoFiLM-match = +0.000698`, 3/6 positive, 95% CI
  `[-0.002204,+0.003696]`, `p=.6875`.
- `FiLM−TS4 = +0.316433`, 6/6 positive, 95% CI
  `[+0.221798,+0.410750]`, `p=.03125`; the T4-content gate passes.

All three miss the `+0.03`, 6/6-session, CI and Wilcoxon gates. Consequently this
Confidence-FiLM version cannot pass seed-42 Stage 0 and will not expand to more seeds,
while the large aligned-T4 versus TS4 contrast confirms that T4 content itself remains
important. This rejects the current confidence fusion mechanism, not T4 or the train-only
predictive value of residual variance. Formal held-out data remain sealed.

At initialization the aligned-T4 arms (T4 continuation, FiLM, confidence-shuffle and
NoFiLM-match) are bitwise equal to the anchor; `FiLM(TS4)` deliberately differs only through
its shuffled T4-to-unit assignment. Confidence never enters the original 68-wide post-pool
input directly: only `[h',T4]` does. The final cached online identity remains `E[N,50]`;
confidence, gamma and beta are calibration-finalization temporaries, so persistent online
state and 50 Hz online MACs are unchanged.

The seed-42 Stage-0 screen is only a triage result. Calling the mechanism `effective` requires
three predeclared seeds and all three contrasts `FiLM−T4`, `FiLM−shuffle-C`, and
`FiLM−NoFiLM-match` to pass the common `+0.03`, 3/3 seed, 6/6 session, bootstrap-CI and exact
Wilcoxon gates. `FiLM−TS4` remains a T4-content diagnostic.

### Rank 4 — T4-conditioned decoupled cross-attention

This candidate changes how the decoder consumes identity. It is not another neuron-axis
self-attention layer and it does not prune tokens. The current coupled path is approximately:

```text
z_i = readin(x_i + E_i)
K_i = W_K z_i
V_i = W_V z_i
```

so functional identity and instantaneous activity jointly determine both attention selection
and content. The minimum decoupled candidate is:

```text
Q_j = learned_behavior_query_j
K_i = f_key(E_i, T4_i)
V_i = f_value(x_i)
alpha_ji = softmax_i(Q_j K_i^T / sqrt(d))
y_j = sum_i alpha_ji V_i
```

The intended interpretation is: the key answers “what functional unit is this?”, the value
answers “what is this unit doing now?”, and the fixed query asks which population evidence is
relevant to each behavioral output. All `N` tokens remain. Shared per-unit projections keep
parameter count independent of `N`, and attention remains `O(CND)` rather than adding an
`O(N^2)` neuron-neuron interaction.

The hardware attraction is stronger than the asymptotic statement alone: `K(E,T4)` is
session-static and can be cached after calibration, while only `V(x)` must be recomputed on
the 50 Hz path. The implementation must report the actual cached-key state and online MACs;
it may not count session-static work as free without including its storage.

This is a decoder-changing experiment. It cannot be presented as a frozen-decoder encoder
ablation, because the existing SPINT attention projections were trained for `readin(x+E)`.
The clean comparison must train or distill the coupled and decoupled decoder variants under
the same train sessions, optimizer budget, checkpoint rule and parameter envelope.

Minimum first-stage arms:

- coupled T4 baseline: current `readin(x+E)` path;
- decoupled `K(E,T4), V(x)`;
- decoupled `K(E,TS4), V(x)` content control;
- parameter-matched decoupled `K(E), V(x)` without direct T4;
- activity-key control `K(x), V(x)` to measure the cost of removing instantaneous activity
  from attention selection.

Frozen Stage-0 implementation contract (2026-07-31):

- the legacy coupled implementation is untouched; every arm trains a fresh decoder from the
  same teacher with `M_activity=30`, `M_T4=50`, common `eval_start=50`, 12 epochs and the
  fixed epoch 5–12 score rule;
- the identity encoder always receives aligned real T4. The TS4 control permutes only the
  decoder-key T4 rows, so it isolates key content rather than destroying the encoder input;
- the decoupled path is one layer with two heads and `D_k=D_v=32`; it caches only the
  projected `K[N,32]`, recomputes `V(x)` online, and introduces no `N²` neuron attention;
- at the actual `D=512`, coupled-64-head teacher configuration and reference `N=64`, the
  configured decoder-path receipt is 57,970,688 MAC/frame and 12,800 B persistent
  `E[N,50]` state for coupled T4, versus 4,997,120 MAC/frame and 8,192 B persistent
  `K[N,32]` state for cached decoupled K/V: `−91.38%` MAC and `−36%` state. These are
  implementation-derived decoder-path counts, not a joint end-to-end latency measurement.

A checkpoint-only spectral audit, run without opening train, validation or formal data,
pre-registers the main capacity diagnostic if this first stage fails. Rank 32 retains only
`29.82%` of teacher `Wq` energy and `39.37%` of `Wk` energy, while retaining `76.94%` of
the effective `Wo@Wv` energy. At rank 48 those figures are `38.89%`, `47.49%` and
`83.78%`. Consequently an under-trained or capacity-limited first stage should test key
rank before attributing failure solely to value rank. `Dk=48,Dv=64` still gives a configured
`91.11%` decoder-MAC reduction at `N=64`; its static key width is 48, no larger than the
existing 50-wide identity state. This audit is not an R² result and does not alter the
frozen `32/32` first-stage protocol.

The source audit also exposes a second, distinct confound that width alone does not fix.
The coupled teacher applies its pretrained `fc_in: 50→512→512` MLP to
`x_i+E_i`; the Stage-0 decoupled path instead feeds raw 50-bin activity directly to a
new random `50→32` value projection. Thus a large v1 accuracy loss cannot be attributed
to key/value factorization alone: it may reflect removal of the teacher activity read-in
and simultaneous random initialization of Q/K/V/out. If v1 fails beyond the `−0.03 R²`
deployment margin, the first representation-preserving follow-up is pre-registered as:

```text
h_E_i = teacher_readin(E_i)
h_x_i = teacher_readin(x_i)
K_i = low_rank_key(h_E_i) + W_T4 T4_i # calibration-time, then cached
V_i = low_rank_value(h_x_i)           # online
```

The candidate must use a single-head rank-48 approximation initialized from the teacher
Q/K bilinear map and a rank-64 approximation of the effective `Wo@Wv` map (or use an
explicitly declared prediction-distillation warm-up); it may not compare another fully
random small decoder and call the result a factorization test. This initialization is
**not** functionally equivalent to the teacher's 64 separate heads and their separate
softmaxes. The direct `4→48` T4 branch starts at zero, and the T4/TS4/e-only arms must
share bitwise-identical E, read-in, low-rank factors and any train-only bootstrap
checkpoint. A per-arm distillation adaptation would invalidate the key-content contrast.

With `Dk=48,Dv=64`, `N=64`, `C=2`, `D=512`, and the same counted
Linear/attention/FFN MAC convention, this read-in-preserving static-key path is estimated
at 25,462,784 decoder MAC/frame versus 57,970,688 for coupled (`−56.08%`). Its cached
`K[N,48]` is 12,288 B, still `4%` smaller than `E[N,50]`. Calibration-only work is
20,000,768 MAC: 18,415,616 for `teacher_readin(E)`, 1,572,864 for the hidden key
projection, and 12,288 for the optional direct-T4 projection. The dynamic `K(x),V(x)`
control cannot cache a key and costs 27,035,648 MAC/frame because its per-frame key
projection adds 1,572,864 MAC. These figures are a source-derived design budget, not an
R² result or measured end-to-end latency.

An isolated implementation scaffold now exists in
`streaming_calibration_exp/src/models/components/decoupled_kv_v2.py`. It is deliberately
not imported by `StreamingSpintModel` or the v1 runner while the sequential five-arm v1
screen is in flight. Its CPU contracts cover the weight-only, Q/K/V-bias-omitting SVD
orientation and score scale, the non-equivalence to 64 head-wise softmaxes, a dedicated
dynamic `K(h_x),V(h_x)` path with no direct-feature call or cache, exact static/dynamic
cost receipts, projected-K-only state, gradients and unit permutation. The related
decoupled suite initially reported 31 passing tests.

An equally isolated streaming adapter now lives in
`streaming_calibration_exp/src/models/components/streaming_spint_v2_adapter.py`. For a
static arm it executes `fc_in(E)` only when deriving K, while cached online decode executes
only `fc_in(x)` and `fc_in(rep)`; a single-session key uses a non-owning batch-expanded
view and is never reprojected. The dynamic x-only deployment API accepts neural activity
without identity/calibration, while framework-side E is explicitly marked metric-only.
The e-only and x-only modes reject supplied direct features, unused legacy decoder modules
are excluded from optimizer state, and adapter dispatch is explicit rather than relying
on the base class's generic non-coupled branch. The combined new/legacy/formal decoupled
suite reports 42 passing tests. Neither bypass module is imported by the active v1 screen.
This is implementation readiness, not accuracy evidence; production-selector integration
is conditional on the v1 result diagnosis above.

The v1 result determines which optimization is justified:

- if epochs 9–12 are still improving coherently, extend the matched training budget
  before changing capacity;
- if the curve has plateaued and direct T4 content is positive, test `48/64` capacity;
- if the curve has plateaued with a large coupled gap, use the read-in-preserving,
  teacher-initialized/distilled candidate above rather than width-only random Q/K/V;
- if `e_t4−e_ts4` is non-positive, do not claim that direct T4-conditioned keys work.

A complementary selected-anchor audit uses only the 27 strict training caches. Across
1,613 units, T4 occupies `4/54=7.41%` of the joint key coordinates but contributes median
`8.00%` and mean `10.00%` of the energy after affine-free joint LayerNorm. Thus simple
four-coordinate dilution is not supported. However, replacing T4 with zeros for the
`e_only` control also changes the normalized E coordinates: median relative L2 drift
`5.81%`, 90th percentile `17.37%`. The `e_only` comparison is consequently not a perfectly
isolated direct-T4 deletion, while `e_t4−e_ts4` remains the primary content test. A later
clean-mechanism round may use `W_E LN(E)+W_T LN(T4)`, but this is not justified as a default
capacity fix. The audit opened 27 train caches, zero NWBs, zero validation sessions and zero
formal sessions.

Confidence is deliberately excluded from this first stage. It enters the key only if Rank 3
first establishes that calibration-block confidence is useful. Optional attention bias is
also a later ablation, not part of the minimum candidate.

Advance by either of two predeclared outcomes:

1. superiority: at least `+0.03 R²` over the coupled T4 baseline with the shared V4
   consistency gates; or
2. deployment non-inferiority: paired lower 2SE bound at least `−0.03`, a positive
   `T4−TS4` key-content contrast, and at least 25% lower measured online decoder MACs and
   no larger persistent session state with cached keys.

The main kill condition is mechanistic: if removing activity from the key costs more than
`0.03 R²`, do not add confidence or larger key networks to rescue the factorization. That
failure would show that state-dependent attention selection, not just functional identity,
is important in this decoder.

### Rank 5 — Direct functional readout as the simplicity control

The current path maps pooled activity plus T4 through a three-layer MLP into a 50-bin identity,
adds it to neural tokens, and lets the frozen decoder's cross-attention interpret it.

A stricter simplicity test is a small shared nonlinear function:

```text
decoder_key_or_weight_i = f(T4_i, confidence_i)
```

with activity entering linearly or through the existing temporal basis. This is the nonlinear
extension of the currently weak `tuning_proj` control. It asks whether the full identity MLP is
necessary, or whether the core contribution is simply a shared nonlinear mapping from
functional descriptors to per-unit readout parameters.

Keep this small and factorized. If it approaches T4 within `0.03 R²` while using much less
weight SRAM, the simpler model is the contribution. If it fails, it strengthens the case that
the richer activity-conditioned identity is necessary.

### Rank 6 — Paired-view co-training for one SUA/pseudo-MUA weight set

The bridge currently trains separate models. Because pseudo-MUA is deterministically derived
from SUA with a known unit→electrode map, it provides a controlled paired augmentation:

- randomly present the same training session as SUA or electrode-pooled pseudo-MUA;
- share all encoder weights;
- retain view-specific T4 construction at the correct unit/channel level;
- optionally penalize disagreement between decoded behavior for paired views.

The primary outcome is not “beat separately trained SUA T4.” It is:

1. one set of encoder weights is non-inferior to separate T4 models in both views;
2. the residual SUA/pseudo-MUA gap does not grow;
3. real MUA external replication is then attempted under a separately frozen protocol.

This direction directly addresses the project's original shared-encoder question. It is more
scientifically valuable than squeezing another few thousandths of R² from SUA alone.

## 7. Reserve direction — T4-guided functional slots

The existing fixed-slot router failed its accuracy gate, and the cross-session linear
direction-bin control also lost substantial information. A T4-guided slot interface should
therefore not be the first follow-up.

It becomes justified only if Rank 1 shows that phase is the dominant T4 component. A later
model could soft-assign units/channels to learned circular functional slots using T4 while
retaining multiple residual slots per direction. This would test whether T4 can solve both
cross-session identity and fixed-shape hardware tokenization, but it must beat:

- equal-K activity routing;
- random-K/routing controls;
- the existing variable-token T4 baseline;
- a hard direction-bin control.

## 8. Winner: two-sentence pitch

> Current T4 succeeds because per-session functional descriptors align otherwise
> non-corresponding units, while B3T independently improves temporal regularization and
> deployment cost; the current concatenation neither separates which T4 component matters nor
> represents its confidence. We propose a factorized, confidence-conditioned T4+B3T encoder
> that uses low-rank affine modulation to combine functional identity with smooth temporal
> activity, aiming to preserve T4's cross-view gain with fewer weights, fewer calibration
> trials, and no array-specific electrode table.

Core tension: **session specificity vs cross-session sharing**. The functional descriptor must
be remeasured each session, but the function that consumes it should be shared.

## 9. Four validation experiments

### Experiment A — T4 attribution screen

- Views: SUA and pseudo-MUA.
- Arms: factorized components and label/row controls from Rank 1.
- Seeds: at least 42/43/44, same frozen 12-epoch estimator.
- Stop rule: if `[m,b]` matches full T4 within `0.03`, do not claim a directional mechanism
  and optimize rate/scale conditioning instead.

### Experiment B — 2×2 temporal-functional factorial

- Arms: B3, B3T, T4, B3T+T4, B3T+TS4.
- Co-primary outcomes:
  - performance superiority or non-inferiority of B3T+T4 versus T4;
  - exact parameter/MAC/state profile;
  - preservation of the correct-content contrast.
- Stop rule: if B3T+T4 is worse than T4 by more than `0.03` at the lower 2SE bound, do not add
  FiLM on that backbone.

### Experiment C — confidence and calibration-budget pilot

- Primary sweep: fix B3 activity identity at `M_activity=30`, vary labeled T4 fit budget
  `M_T4=10/15/20/30/50`, and keep evaluation fixed to trials `[50:]`.
- Secondary joint-latency sweep, only if the primary sweep succeeds: set
  `M_activity=M_T4` and compare total calibration latency at a separately frozen common
  evaluation boundary.
- Arms: selected-T4 continuation, confidence-FiLM, confidence-shuffled FiLM,
  parameter-matched additive NoFiLM, and T4-shuffled FiLM.
- Primary claim: reduce required labeled calibration trials relative to the measured
  `T4@50`, not merely improve a nominal 30-trial condition that has not been run.
- Stop rule: confidence must either add at least `+0.03` at a fixed budget or reach the
  `T4@50` baseline within `0.03` using at most 20 labeled trials.
- Coverage rule: report the number of distinct target directions and per-direction counts at
  every budget. Before low-budget runs, replace the current `<2 directions` guard with a
  design-matrix rank/condition-number guard: fitting `[b,a,c]` requires rank 3, so two
  directions are still underdetermined even though `np.linalg.lstsq` returns a minimum-norm
  value. A rank-deficient arm is a declared measurement degeneracy, not evidence against the
  network.

### Experiment D — decoder key/value factorization

- Run only after the selected T4 representation is frozen; add confidence only if Experiment C
  passes its confidence gate.
- Compare the coupled baseline, decoupled T4, decoupled TS4, identity-only key and activity-key
  control under the same decoder training/distillation and fixed checkpoint rule.
- Report paired R², the correct-content contrast, exact learned parameters, cached-key bytes,
  session-rate MACs and 50 Hz online MACs.
- Primary question: can session-static functional keys preserve behavior decoding while moving
  a material fraction of key computation out of the online path?
- Kill the branch if activity-free keys are worse than coupled T4 by more than `0.03 R²`; do
  not respond by adding a larger decoder in the same screen.

## 10. Two-week feasibility pilot

### Days 1–3: close attribution

- freeze T4 component definitions and controls;
- add cache namespaces and leakage tests;
- run synthetic invariance/label-shuffle tests and one train/one validation real-data smoke.

### Days 4–7: run the attribution screen

- execute three seeds on two GPUs;
- aggregate same-session/same-seed deltas;
- choose the minimal sufficient T4 representation before architecture work.

### Days 8–10: implement T4+B3T

- reuse the existing raised-cosine basis and side-feature loader;
- add exact zero-init equivalence, permutation, streaming-state and `cost_profile()` tests;
- do not add FiLM yet.

### Days 11–14: factorial pilot and decision

- run B3T+T4/B3T+TS4 with fresh matched controls;
- decide among:
  - keep T4+B3T as an efficiency-dominant model;
  - proceed to confidence-FiLM;
  - stop architecture work and prioritize paired-view co-training / real MUA replication.

This pilot deliberately does not combine component changes, B3T, confidence and multi-view loss
in one run. Each addition earns the right to enter the next stage.

The decoupled decoder is explicitly outside this first two-week pilot. It begins only after the
T4 representation and confidence decision are frozen, so an encoder-side interaction and a
decoder-side key/value factorization are not confounded.

## 11. Strongest objection and response

**Objection:** T4 uses behavioral direction labels, and its apparent “functional identity”
gain may mostly be aligned baseline firing rate or an information-rich supervised calibration
shortcut. The pseudo-MUA result is derived from the same recordings and therefore does not
establish external MUA generalization.

**Response:** accept both points as current evidence boundaries. The component and
target-label-shuffle screen separates direction from rate scale; the calibration-budget study
quantifies the supervision cost; pseudo-MUA remains a controlled invariance test only; and
real threshold-crossing MUA requires a separate external replication before any cross-signal
generalization claim.

**Decoder-specific objection:** a key computed only from session-static identity cannot change
attention weights with instantaneous population activity, so the factorization may remove a
useful behavior-dependent routing mechanism.

**Response:** treat this as the central falsifiable boundary, not an implementation defect.
The activity-key control measures it directly; a loss greater than `0.03 R²` kills the branch,
while non-inferiority plus lower online MACs supports the cached-functional-key deployment
claim.

## 12. External design precedents

These papers motivate mechanisms, not claims about this dataset:

- [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html):
  shared permutation-invariant/equivariant processing of variable-size sets.
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html): attention-based set
  processing and inducing-point reductions; relevant only if a functional-slot model earns a
  later screen.
- [FiLM](https://arxiv.org/abs/1709.07871): feature-wise affine conditioning, the template
  for the proposed zero-initialized T4/confidence modulation.
- [POYO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html):
  multi-session neural decoding with variable populations and cross-attention; a scale-heavy
  precedent for the shared-set problem, not the target architecture here.
- [CANDY](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ebc8728759f26169f09d9d74497a430b-Abstract-Conference.html)
  and [TCLA](https://arxiv.org/abs/2601.19963): adjacent task-conditioned/contrastive
  cross-session alignment approaches that motivate the paired-view objective while differing
  substantially from the lightweight streaming constraint here.
