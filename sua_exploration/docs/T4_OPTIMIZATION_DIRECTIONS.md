# T4 network optimization directions

**Status:** analysis and candidate convergence; no new experiment launched  
**Updated:** 2026-07-29  
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

Estimated from the existing cost models, adding four T4 inputs to B3T's first post-pool layer
would give roughly 12,658 learned parameters versus T4's 18,290, and about 4.52M versus
13.03M session-path MACs at the reference shape. That is approximately 31% fewer parameters
and 65% fewer MACs; the implementation must make `cost_profile()` compute the exact values
rather than treating this estimate as a result.

This variant is worth keeping even if it merely matches T4 within a predeclared `−0.03`
non-inferiority margin, because it would dominate on deployment cost.

### Rank 3 — Confidence-conditioned FiLM, not electrode reliability lookup

T4 is currently fitted from 50 rewarded calibration-pool trials, but the network receives no
explicit information about fit uncertainty. The failed T4GATE used a fixed scalar indexed by
electrode; it cannot know whether **this session's** T4 estimate is reliable.

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
E_i = post_pool(h'_i)
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

### Rank 4 — Direct functional readout as the simplicity control

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

### Rank 5 — Paired-view co-training for one SUA/pseudo-MUA weight set

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

## 9. Three validation experiments

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
  `M_T4=5/10/15/20/30/50`, and keep evaluation fixed to trials `[50:]`.
- Secondary joint-latency sweep, only if the primary sweep succeeds: set
  `M_activity=M_T4` and compare total calibration latency at a separately frozen common
  evaluation boundary.
- Arms: concat T4, concat T4+confidence, confidence-FiLM, confidence-shuffled control.
- Primary claim: reduce required labeled calibration trials relative to the measured
  `T4@50`, not merely improve a nominal 30-trial condition that has not been run.
- Stop rule: confidence must either add at least `+0.03` at a fixed budget or reach the
  `T4@50` baseline within `0.03` using at most 25 labeled trials.
- Coverage rule: report the number of distinct target directions and per-direction counts at
  every budget. Before low-budget runs, replace the current `<2 directions` guard with a
  design-matrix rank/condition-number guard: fitting `[b,a,c]` requires rank 3, so two
  directions are still underdetermined even though `np.linalg.lstsq` returns a minimum-norm
  value. A rank-deficient arm is a declared measurement degeneracy, not evidence against the
  network.

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
