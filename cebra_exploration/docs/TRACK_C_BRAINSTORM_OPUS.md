# Track C brainstorm — how our method can take inspiration from CEBRA

**Track:** C2 (independent parallel brainstorm; C1 not read)
**Date:** 2026-08-13
**Author role:** research ideation. **No production code written.** Three throwaway CPU diagnostics were run
and are reported in §0 with their provenance.
**Read before writing:** `cebra_exploration/docs/BACKGROUND_BRIEF.md` (full),
`bci_paper_overleaf/paper_6pp.tex` (full),
`sua_exploration/docs/HANDOFF_H1_SPARSE_MAINLINE_STRENGTHENING_20260812.md` (full, incl. §10–§12),
`sua_exploration/mc_maze/h1_sparse_event_endpoint.py`,
`sua_exploration/mc_maze/h1_sparse_event_endpoint_v2.py` (API only),
`SPINT-main/src/data/h1_sparse_event_endpoint.py`,
`SPINT-main/configs/data/falcon_h1_sparse_event_endpoint.yaml`,
vendored CEBRA `cebra/models/criterions.py` and `cebra/integrations/sklearn/metrics.py`.

---

## 0. Three measurements I made first, because they change the ranking

Everything below is a **throwaway diagnostic with no receipt**. None of it is citable. It exists only to
stop me proposing ideas against an imagined version of our pipeline. All of it is CPU, seconds of runtime,
H1 **held-in only**, routed through `h1_sparse_event_endpoint.index_heldin_calib`.

**Provenance gate I ran first.** For all 13 held-in recordings I recomputed the LODO endpoint basis and the
M4 carrier with `h1_sparse_event_endpoint_v2.fit_source_all_event_basis` /
`fit_session_range(start_index=0, budget=4)` and compared against
`results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json`.
**`basis_sha256` and `carrier_sha256` match the sealed receipt in 13/13 sessions.** So the numbers below
were computed on exactly the sealed estimator objects, not on a lookalike.

### 0.1 The H-SE5 carrier is numerically dominated by its least informative coordinate

The deployed sparse carrier is 5-wide: four latent displacement slopes plus an intercept
(`SE5_DIM = 5`, `LATENT_DIM = 4`, `RIDGE_LAMBDA = 3.0`). Before it is concatenated with the activity
summary it is divided by **one global scalar** `s_src = sqrt(mean(source_cache^2))`
(`SPINT-main/src/data/h1_sparse_event_endpoint.py`, `SparseScalarNormalizer`).

Per-coordinate statistics over 11 held-in sessions (my proxy `s_src = 1.0804`):

| carrier coordinate | RMS after `s_src` | std | mean |
|---|---:|---:|---:|
| slope 1 | 0.0340 | 0.0367 | 0.0002 |
| slope 2 | 0.0179 | 0.0194 | 0.0003 |
| slope 3 | 0.0179 | 0.0193 | 0.0006 |
| slope 4 | 0.0285 | 0.0306 | 0.0033 |
| **intercept** | **2.2355** | **1.1840** | **2.1051** |

Because `s_src` is a *single scalar*, **the ratios between coordinates are exact and independent of how
`s_src` was computed**, so the proxy is harmless here. The four coordinates that carry signed tuning —
the component the paper's own lattice identifies as the active ingredient (AC4 retains the effect, MB4
collapses it) — enter the identity MLP `ψ_c` at 1/32 to 1/61 of the intercept's *standard deviation*, and
at 1/66 to 1/125 of its *RMS*, since the intercept is mostly a constant offset (mean 2.11) that is
identical for every channel and therefore carries no per-channel identity at all.

The paper's own H-C carrier, the dense arm that **works** (`+0.0563` over matched SPINT), is 4-wide and is
built only from `B_slope` in eq. (15) — the intercept `b` is computed and then discarded. So the arm that
works has no intercept coordinate; the sparse arm that fails has one, and it is 60× larger than everything
else in the vector.

### 0.2 Split-half carrier stability predicts H1 forward transfer at ρ = 0.81; event count does not

Splitting each session's M4 support by trial parity, fitting two carriers, and taking the median
per-channel cosine between the two weight matrices:

| statistic vs sealed per-session `median_delta_intercept` | Spearman ρ | p | n |
|---|---:|---:|---:|
| **split-half carrier weight cosine** | **+0.808** | **0.0008** | 13 |
| carrier tuning-coefficient RMS | +0.637 | 0.019 | 13 |
| support event count | +0.175 | 0.57 | 13 |

The absolute values are as informative as the correlation. Per-session split-half cosines run from
**−0.036 to +0.229, median +0.035**. Caveat that matters: each half is fitted on ~10 events against 5
parameters (2.0 obs/param) versus 20 events for the deployed carrier, so these cosines **understate** the
deployed carrier's reliability and should be read as a session-ranking statistic, not as the deployed
reliability. Even so, a Spearman–Brown-style correction leaves the implied reliability low.

Two consequences. First, **at M4 the per-channel signed tuning direction is largely sampling noise**, which
is the mechanism behind the `+0.0285 → −0.0226` date flip and behind C1's monotonic no-plateau curve.
Second, we have a **calibration-time statistic that ranks sessions by how much the carrier will help**,
computed from the same 20 labels, with no decoder and no extra supervision.

### 0.3 Source-side carrier noise is already budget-matched

`build_sparse_source_assets` fits a fresh M4 carrier at **every legal contiguous 4-trial start** in every
source session. The source consumer therefore already sees carriers with deployment-matched sampling noise.
This kills one idea (§2, R2) and enables another for free (§1, S4): the positive pairs a contrastive
objective needs are already sitting in `SparseCarrierCache`.

---

## 1. Ranked ideas

Ranking follows the stated priority: strengthens the sparse narrative on H1 > respects the no-target-backprop
constraint > cheap to test > addresses a named weakness.

---

### S1 — Rebalance the carrier vector before injection (and drop or re-express the intercept)

**Statement.** The four signed-tuning coordinates of the H-SE5 carrier reach `ψ_c` at 1/32–1/61 of the
intercept's scale, so the coordinate the paper claims is the active ingredient is numerically the quietest
input to the identity path; standardize per coordinate at source, or factor the carrier the way CEBRA
factors an embedding, into a unit direction plus a log magnitude.

**Mechanism.** Change the descriptor→token interface only. Three variants, in increasing distance from the
current code:
- **(a) per-coordinate source standardization.** Replace the single scalar `SparseScalarNormalizer.s_src`
  with a source-frozen 5-vector of per-coordinate means and scales, computed on the same source carrier
  cache. Two lines. Everything downstream is unchanged.
- **(b) drop the intercept coordinate.** Make the sparse carrier 4-wide like H-C. The intercept is
  `log1p(baseline rate)`, which is close to what the activity summary `f_pre` already supplies, so it is both
  redundant and dominant.
- **(c) spherical factorization.** Inject `[ŵ_i/‖ŵ_i‖, log‖ŵ_i‖]`. This is CEBRA's own convention —
  `FixedCosineInfoNCE` compares L2-normalized latents with a temperature precisely because raw feature
  magnitudes are not comparable — and it matches the paper's lattice finding that phase alone (PH4) is
  intermediate while amplitude weighting is a real component, i.e. direction and magnitude are separately
  meaningful and should not be entangled in one unnormalized vector.

Stage: **source training** (the normalizer is a source-frozen constant). Online decode and the closed-form
solve `β_i = X†r_i` are untouched.

**Named weakness addressed.** The H1 sparse failure (brief §2.4). Also the general
"attachment sensitivity" story: a 60×-dominant channel-independent DC offset is exactly the kind of content
that cannot support unit–behaviour pairing.

**Violates no-target-session-backprop?** **No.** The normalizer is fitted on source carriers only, and it is
a fixed affine map applied at calibration. **Cost is at source-training time only, i.e. free for us.**

**Cheapest decisive test.** A CPU diagnostic on the existing sealed H-SE5 checkpoint, ~1 hour: load `ψ_c`'s
first layer and compute each carrier coordinate's *effective* contribution to the token, `|W[:,j]| · RMS(c_j)`,
and the variance of `E_i` attributable to each coordinate by finite differences at the empirical carrier
distribution.
- **Confirms:** the intercept coordinate accounts for the majority of token variance while the four tuning
  coordinates account for a small minority — the learned weights did *not* compensate for the input scale.
- **Refutes:** the tuning columns have large learned weights that cancel the scale gap, so the network
  already fixed it and there is nothing to win.
If confirmed, the follow-up is 2 GPU runs (variant (a) or (b) Full vs sealed Full on fold 0) — the cheapest
GPU confirmation on this list.

**Expected failure mode.** The zero-initialized carrier columns plus Adam's per-parameter scaling may have
already absorbed the imbalance, since Adam is approximately scale-invariant in the gradient magnitude. That
is the single strongest objection to this idea and it is exactly what the checkpoint test measures. Note the
imbalance still hurts through weight decay and through the shared initialization scale even under Adam, but
the effect would be much smaller than the raw 60× suggests.

**Kill criterion.** The checkpoint diagnostic shows the four tuning coordinates already account for ≥40% of
token variance, **or** variant (a)/(b) fails to beat sealed Full on fold 0 in the 2-run confirmation.

**Confidence.** Moderate-to-high that the imbalance is real and unintended (I measured it, on
SHA-matched sealed objects). **Low-to-moderate** that fixing it produces a large decoder gain, because of the
Adam objection above. My uncertainty is driven entirely by not having read `ψ_c`'s forward pass or its
optimizer configuration.

---

### S2 — Calibration self-assessment and abstention, from CEBRA's goodness-of-fit

**Statement.** CEBRA reads its own InfoNCE value as a label-free verdict on whether an embedding is any
good; we can compute an analogous verdict on the carrier at calibration time and, when it fails, emit the
zero-carrier token instead — turning "the sparse carrier sometimes hurts" into "the sparse carrier never
hurts, it sometimes abstains".

**Mechanism.** Insert one step between the closed-form solve and the single forward pass. After fitting
`β_i` on the M support events, compute a **stability statistic** and compare it to a source-calibrated
threshold:
- **primary:** split-half carrier cosine (§0.2), which already exists in code as
  `h1_sparse_event_endpoint.coefficient_split_stability`;
- **alternative:** leave-one-event-out or held-out-event `ΔR²` of the carrier against the intercept model,
  i.e. the estimator-level `median_delta_intercept` computed on the *support* block by cross-validation
  rather than on later events.

If the statistic is below threshold, set `c_i = 0` and use the carrier-free token of eq. (9). The threshold
is chosen on source dates only. The paper's attachment result makes the fallback provably safe in the right
direction: a bad carrier is *worse* than a zero carrier (TS4/LS4 fall below the zero-content floor), so
abstention is not a neutral hedge, it is a recovery of known lost value.

Stage: **session calibration**. Adds one extra `q×q` solve on a subset — negligible next to the existing
one.

**Named weakness addressed.** The H1 sparse failure, head on. It does not make date 2 positive; it makes date
2 *not negative*, which is a different and defensible claim: *the carrier is a monotone improvement over the
activity-only baseline under a calibration-time acceptance rule.* It also gives the paper the boundary
statement it currently asserts without measuring — the boundary variable is carrier reproducibility, not
event count (ρ = 0.81 vs 0.17).

**Violates no-target-session-backprop?** **No.** Closed-form statistics on the calibration prefix plus a
threshold comparison. No gradients anywhere.

**Cheapest decisive test.** Partly done (§0.2, CPU, seconds). The remaining decisive step is forward-only
and needs no training: take the existing sealed fold-0 and date-2 (19250108) H-SE5 checkpoints, apply the
gate at a source-chosen threshold, and check that the gated system's per-recording deltas against Zero5 are
≥ 0 on both dates.
- **Confirms:** the threshold that rejects date 2 does not also reject the fold-0 recordings, and gated
  deltas are non-negative on all four recordings.
- **Refutes:** no threshold separates them, i.e. the sessions that gain and the sessions that lose have
  overlapping stability statistics.

**Expected failure mode.** ρ = 0.81 is measured against the *estimator-level* `median_delta_intercept`, not
against decoder R². The link from estimator-level effect to decoder-level effect is assumed throughout this
project and is supported by exactly one calibration point (CPU `+0.0144` → decoder `+0.0165`, handoff §12).
If that link is weak, a gate tuned on the estimator will not sort decoder outcomes. The second failure mode
is honesty optics: a reviewer can read an abstention rule as a post-hoc excuse. The defence is that the
threshold is fixed on source dates and the rule is stated before the target is seen — that must be
pre-registered, not fitted.

**Kill criterion.** No source-chosen threshold gives non-negative gated deltas on all four available H1
recordings, or the statistic's correlation with *decoder* delta is under about 0.4 once measured.

**Confidence.** High that the statistic ranks sessions (measured). Moderate that a usable gate exists,
because n = 4 recordings with decoder numbers is very thin. This is my highest-confidence *evidence* and
only my second-highest-confidence *outcome*.

---

### S3 — Latent-anchored carrier: unsupervised per-channel loadings, sparsely-supervised shared rotation

**Statement.** CEBRA-Hybrid pairs an unsupervised temporal objective with a small label-driven component;
transplanted in closed form, that means estimating each channel's loadings from the *unlabelled* calibration
block, where we have thousands of bins, and spending the ~20 labels only on a single low-dimensional
rotation shared across all 176 channels.

**Mechanism.** Replace the per-channel supervised fit with a two-stage closed-form estimator.
1. **Unsupervised, label-free.** On the whole calibration prefix (all bins, not just movement events),
   compute the top-`p` principal components of the binned rate matrix, giving per-channel loadings
   `L ∈ R^{N×p}` and per-bin scores. This uses data the current pipeline discards — the activity summary
   `f_pre` reads only per-unit marginals and never the population covariance.
2. **Supervised, tiny.** Average the latent scores over each of the ~20 labelled movement events to get
   `Z_e ∈ R^{n×p}`, and fit one shared map `A ∈ R^{p×q}` from latent coordinates to the source-frozen label
   basis `φ(a_e)` by ridge. Set the carrier to `C = L A` (plus the existing intercept treatment from S1).

The accounting is the point. Today, 176 × 5 = 880 supervised parameters are estimated from 20 events. Under
this scheme the 176 × p per-channel numbers come from thousands of unlabelled bins and only `p × q` ≈ 16
numbers are estimated from the 20 events. **The residual estimation noise becomes a single rotation shared
by every channel rather than 176 independent per-channel errors**, and a shared rotation preserves the
*relative* unit–behaviour structure that the paper's attachment controls identify as the mechanism, whereas
independent per-channel noise destroys it.

Stage: **session calibration**, still one closed-form solve (two small ones).

**Named weakness addressed.** The H1 sparse failure and its measured cause: split-half cosine ≈ 0.03, i.e.
per-channel direction estimates that do not reproduce. It is also a principled account of the boundary —
sparse labels are sufficient when they only have to fix a shared low-dimensional alignment, and insufficient
when they have to fix a per-channel vector.

**Violates no-target-session-backprop?** **No.** PCA and ridge, both closed form.

**Cheapest decisive test.** CPU, all 13 held-in H1 recordings, reusing `forward_transfer`'s metric
unchanged: does `C = LA` beat the direct per-channel fit on later-event `median_delta_intercept`, and does
its split-half cosine rise?
- **Confirms:** `LA` beats the direct fit on ≥10/13 sessions at M4 **and** reaches at M2 the
  `delta_intercept` the direct fit reaches at M4 — that would be the sparsity headroom C1 proved does not
  exist for the current estimator, recovered by changing the estimator.
- **Refutes:** no improvement at any budget, meaning the per-channel loadings do not span the tuning
  directions.
Cost note, honestly stated: this is **not** a pure reuse. `EventSession` exposes events only, not a binned
rate matrix, so the screen needs a small binned-rate reader built on the existing `_unit_spikes`. Call it a
day of CPU work rather than an hour.

**Expected failure mode.** The dominant principal components of a 20 ms-binned H1 calibration block are
likely to be non-task variance — drift, common-mode gain, artifact — rather than movement tuning, so `L`
may not span the directions the labels need. The second failure mode is rigidity: forcing every channel's
tuning into `range(L)` is a hard rank constraint that biases channels whose tuning is off-subspace. `p` is
the whole game and I have no basis for guessing it.

**Kill criterion.** `LA` fails to beat the direct fit on `delta_intercept` at any `p ∈ {4, 8, 16, 32}` and
any budget, or its split-half cosine is not materially above the direct fit's ~0.03.

**Confidence.** Moderate. The variance argument is sound and the diagnosis it addresses is measured, but
whether unsupervised population structure contains the tuning subspace on human threshold-crossing data is
genuinely unknown to me.

---

### S4 — Carrier-view InfoNCE at source training

**Statement.** The truest transplant of CEBRA's actual objective: add a source-only InfoNCE term on the
identity tokens in which two carriers of the same channel fitted on different calibration windows are a
positive pair, so `ψ_c` is trained to be invariant to calibration sampling noise while remaining
discriminative between channels.

**Mechanism.** At source training only, alongside the existing MSE of eq. (8), add
`L_NCE` over L2-normalized tokens with a temperature, exactly `FixedCosineInfoNCE`:
- **positives:** `(E_i^{(w)}, E_i^{(w')})` — the same channel `i`, carriers fitted on two different
  contiguous 4-trial windows `w ≠ w'` of the same source session;
- **negatives:** tokens of other channels, drawn within and across source sessions.

The pairs already exist: `build_sparse_source_assets` caches one carrier per legal contiguous 4-trial start
per session, so the multi-view structure is a by-product of code that already runs. The two properties this
buys are precisely the two the paper already names as its mechanism: **invariance** to which calibration
window you happened to get (the date-2 failure mode), and **discriminability** between units (attachment
sensitivity, the reason TS4/LS4 fall below the zero floor). A second positive-pair definition is available
and worth trying in the same harness: channels in *different sessions* whose carriers are close, which is
the direct analogue of CEBRA's multi-session shared latent.

Stage: **source training**. Online decode and calibration are byte-for-byte unchanged; deployment cost is
identical.

**Named weakness addressed.** The H1 sparse failure, via robustness to carrier estimation noise rather than
via reducing that noise. Also gives us a CEBRA-style goodness-of-fit number for free, which feeds S2.

**Violates no-target-session-backprop?** **No.** Purely a source-training auxiliary loss. **Cost is at
source-training time only — free for us at deployment.**

**Cheapest decisive test.** A CPU pre-screen that can kill it in minutes, before any GPU time: over
**non-overlapping** 4-trial windows within each source session, compute the mean cosine between carriers of
the *same* channel and of *different* channels.
- **Confirms:** same-channel cosine is clearly above different-channel cosine — there is a stable per-channel
  identity for InfoNCE to latch onto.
- **Refutes:** the two are indistinguishable, in which case the positives are not positives and the objective
  has nothing to learn.

**Expected failure mode.** My own §0.2 measurement predicts this pre-screen may fail at M4: if split-half
cosine is ~0.03, cross-window same-channel cosine will also be low, and InfoNCE cannot manufacture identity
that the estimator did not capture. The honest reading is that S4 is **downstream of S1/S3** — it is worth
running only once the carrier itself is more reproducible. A second failure mode is that the objective
optimizes token-space geometry that the cross-attention decoder does not read, since the token is *added* to
activity rather than consumed directly.

**Kill criterion.** The CPU pre-screen shows same-channel and different-channel cross-window cosines within
noise of each other at every budget available in the source cache.

**Confidence.** Moderate on mechanism, **low on it working at the current M4 carrier quality**. I rank it
fourth rather than higher for exactly that reason, despite it being the most faithful CEBRA transplant on
this list.

---

### S5 — Empirical-Bayes shrinkage toward a source-learned activity→carrier prior

**Statement.** H-C already shrinks its carrier toward a source prior mean (eq. 15) while H-SE5 shrinks toward
zero; replace that zero target with a **per-channel** prior predicted from the channel's unlabelled activity
summary, learned at source time.

**Mechanism.** At source training, fit a map `h: activity summary → expected carrier` on the source carrier
cache (closed-form ridge, or a small MLP — either way, source-only). At calibration, form the raw carrier as
now, then shrink per channel toward the prediction:
`c_i = ĥ(a_i) + (τ²/(τ² + ν_i)) · (c_i^raw − ĥ(a_i))`, with `ν_i` the per-channel estimator variance already
available in closed form from the ridge solve. This is exactly eq. (15)'s structure with `μ_C` upgraded from a
global constant to a per-channel, label-free prediction.

Why the prior is non-trivial rather than circular: the paper's own linear probe reports that the activity-only
token is **phase-poor rather than phase-free** (null-relative phase advantage `+0.090` for Z4 versus `+0.517`
for AC4). So activity predicts some of the tuning, and that predictable part should not have to be re-estimated
from 20 events. It also degrades gracefully — as the labels get scarcer, `ν_i` grows and the carrier falls
back to the activity prior, which is the Zero5 behaviour, so the estimator itself acquires the "never worse
than zero-carrier" property that S2 achieves with a gate.

Stage: **source training** for `h`, **calibration** for the shrinkage.

**Named weakness addressed.** H1 sparse failure; and it converts the paper's floor claim from an empirical
observation into a structural property of the estimator.

**Violates no-target-session-backprop?** **No.** `h` is frozen after source training; the target-side step is
a closed-form convex combination.

**Cheapest decisive test.** CPU, LODO over the 6 H1 dates: fit `h` on the source dates' carriers, predict
carriers for the held-out date's channels, and measure (i) `delta_intercept` of `ĥ(a_i)` alone on later
events, and (ii) `delta_intercept` of the shrunk carrier vs the raw carrier.
- **Confirms:** `ĥ` alone has positive `delta_intercept`, and shrinkage beats raw on ≥10/13 sessions.
- **Refutes:** `ĥ` alone is at or below zero — activity does not predict tuning on H1 — in which case this
  reduces to a plain λ sweep.

**Expected failure mode.** `+0.090` null-relative phase advantage is a small number and it was measured on
center-out SUA, not on H1 threshold crossings; the human data may have no activity→tuning regularity at all.
Also, an activity-predicted carrier is by construction correlated with the activity summary that is
concatenated next to it, so the consumer may gain nothing it did not already have.

**Kill criterion.** `ĥ(a_i)` alone gives `delta_intercept ≤ 0` under LODO on H1.

**Confidence.** Moderate. The mechanism is standard and safe; the size of the win depends on a regularity I
have not measured on this dataset.

---

### S6 — Behaviour-free carrier from decoder pseudo-labels

**Statement.** CEBRA's positives only need to say "these two are similar", not what the value is — so the
auxiliary variable does not have to be a *ground-truth* label, and the frozen zero-carrier decoder's own
predictions on the unlabelled calibration block may be a good enough auxiliary variable to fit the carrier
against, giving a **zero-label** variant of the method.

**Mechanism.** At calibration: (1) run the frozen decoder with the carrier-free token of eq. (9) over the
calibration block; (2) integrate or read off predicted per-event displacements `Δq̂_e`; (3) fit the carrier
`β_i = X†r_i` against `φ(Δq̂_e)` instead of the true endpoints; (4) recompute `E_i` and stream. This is
CEBRA-Behavior with a self-generated behaviour variable, and it is distinct from the cited self-recalibration
line (`jarosiewicz2015`, `prit2025`) in that no decoder weight moves — only the identity token's *content*
changes.

Stage: **session calibration**. Honest protocol note: this makes calibration **two** forward passes rather
than the one stated in eq. (2). That is a real, if small, amendment to the paper's protocol sentence and
should be declared rather than glossed.

**Named weakness addressed.** Not the H1 sparse failure directly — something bigger. If it works, the paper's
weakest axis (sparse labels) is replaced by its strongest possible version (no labels), which would put us on
the same supervision footing as the unlabelled alignment family (`degenhart2020`, `nomad2025`) while keeping
the no-backprop property that they do not have.

**Violates no-target-session-backprop?** **No.** Forward passes and a closed-form solve.

**Cheapest decisive test.** CPU, **no decoder and no checkpoint required**: corrupt the true endpoint labels
with noise calibrated to the zero-carrier decoder's own accuracy (H1 Zero5 pooled `R² = 0.4716`), refit the
carrier, and measure `delta_intercept` and split-half cosine across all 13 held-in sessions as a function of
label noise.
- **Confirms:** at the noise level implied by `R² ≈ 0.47` the carrier retains most of its `delta_intercept`,
  so a real pseudo-label carrier is worth building.
- **Refutes:** `delta_intercept` collapses well before that noise level — dead before any implementation.

**Expected failure mode.** Two. First, circularity: pseudo-label error is *correlated with the neural data*
that the carrier is fitted from, so the induced bias is systematic, not the zero-mean noise my cheap screen
simulates — my screen is therefore an **optimistic upper bound** and I should say so explicitly rather than
treat a pass as a green light. Second, the obvious reviewer question: if the zero-carrier decoder already
predicts the movements well enough to supervise the carrier, what is the carrier adding? The answer would
have to be an amplification argument, and it may not hold.

**Kill criterion.** The noise-tolerance screen shows collapse at noise levels below the decoder's own error,
or a built version fails to beat Zero5 on fold 0.

**Confidence.** Low-to-moderate on it working, **high on it being worth the screen**, because the screen is
hours of CPU with no checkpoint and the upside is the single largest change to the paper's positioning
available anywhere on this list.

---

### S7 — Choose the 7→4 label basis to predict neural responses, not label variance

**Statement.** `fit_source_all_event_basis` picks the four latent displacement directions by PCA on
standardized 7-DoF endpoints — a criterion that never looks at a spike — and retains only ~0.72 of
displacement variance; a source-fitted reduced-rank regression or PLS between displacement and event log-rates
would choose the four directions that actually drive firing.

**Mechanism.** Replace the SVD in `fit_source_all_event_basis` with a partial-least-squares / reduced-rank
regression fit pooled over the source dates, mapping standardized 7-D displacement to the 176-channel event
log-rate matrix, and keep the top four components. Everything else is identical: still a fixed 7×4 linear map,
still LODO-frozen from source, still `q = 4`, still a 5-wide carrier, identical deployment cost, no decoder
change. This is CEBRA's framing applied to our label basis — the auxiliary variable's job is to organize the
*neural* space, so it should be reduced with reference to the neural space.

Stage: **source training** (the basis is a source-frozen constant).

**Named weakness addressed.** The 0.719 retained-variance figure, which is currently a gate criterion and an
unexamined loss. Also §12.2's finding that at fixed `q` you cannot add information without paying in retained
variance — this changes *which* information the four slots hold rather than adding more.

**Violates no-target-session-backprop?** **No.** Source-only.

**Cheapest decisive test.** CPU, LODO over dates, `forward_transfer` unchanged.
- **Confirms:** `median_delta_intercept` rises with ≥10/13 sessions positive under strict date-LODO.
- **Refutes:** no change, which would say the bottleneck is per-channel estimation noise, not basis choice —
  a result that would also strengthen S3.

**Expected failure mode.** PLS fitted on pooled source events can overfit the source dates' particular
event mix, and the strict LODO makes that visible. More fundamentally, my §0.2 measurement says the binding
constraint is per-channel reproducibility, and no choice of basis changes the 20-events-against-5-parameters
arithmetic. I expect this to be a small positive at best.

**Kill criterion.** Under strict date-LODO, `delta_intercept` does not improve, or improves in-sample only.

**Confidence.** Moderate that the change is directionally right; **low that it is large enough to matter.**
It is on the list because it is nearly free and because a negative result is diagnostic for S3.

---

### S8 — Closed-form per-session activity standardization, as a stand-in for CEBRA's session-specific input layer

**Statement.** CEBRA gives each session its own learned input layer; we add a token to *raw* binned activity
(`src_i = x_i + E_i`), so any session-to-session change in activity scale silently rescales the effective
token strength, and a closed-form, label-free per-channel affine map to source-matched moments is the
constraint-respecting analogue of what CEBRA learns.

**Mechanism.** At calibration, compute per-channel mean and scale of the binned rates on the **unlabelled**
calibration block, and map them to the source-pooled per-channel moment distribution before the window enters
eq. (3), `src_i = x_i + E_i`. Source statistics frozen; target statistics closed form. This is the one idea that
touches the *activity* path rather than the identity path, and it is the closest structural counterpart to
CEBRA's multi-session input layers, which the brief correctly identifies as the learned solution to the
problem our token solves in closed form.

Stage: **session calibration**, label-free.

**Named weakness addressed.** Cross-session robustness generally; the held-in/held-out gap (0.4731 vs 0.2749)
if that gap is partly a scale-domain gap rather than a tuning gap.

**Violates no-target-session-backprop?** **No.** Moment matching is closed form.

**Cheapest decisive test.** CPU, all 13 held-in sessions: compute each session's per-channel rate moment shift
relative to the source pool and correlate with the sealed per-session `median_delta_intercept`.
- **Confirms:** sessions with larger moment shift have systematically lower `delta_intercept`.
- **Refutes:** no relationship.

**Expected failure mode.** I already have evidence against the strong version: carrier tuning-coefficient RMS
varies only **1.30×** across the 13 sessions (§0.2), which argues the scale drift on H1 is modest, and the
response is already `log1p`-compressed. Also `delta_intercept` is an estimator-level quantity that never
touches the token, so the proposed screen is a weak proxy for the thing I actually care about.

**Kill criterion.** Moment shift shows no relationship to per-session effect, and H1 per-channel moments are
within a small factor across dates.

**Confidence.** Low that this is a live problem on H1. I include it because it is the honest answer to
"what is CEBRA's session-specific input layer, in our vocabulary", and because it is nearly free.

---

### S9 — **CONSTRAINT-VIOLATING.** Train a target-session input layer, CEBRA-style

**Statement.** Fit a small per-session input map on the target session by gradient descent, exactly as CEBRA
does, and report what the backward pass buys.

**Mechanism.** Add a session-local linear layer before `g(·)` and train only it on the target session's
calibration prefix, everything else frozen.

**Violates no-target-session-backprop?** **YES, explicitly.** This is not an improvement to our method; it is
a different method, and it destroys the paper's defining commitment and its ~12% latency claim.

**Why it might still be worth the loss.** The brief itself says quantifying what that backward pass buys is a
genuine result either way, and the number bounds how much our closed-form path is leaving on the table.

**Where it belongs.** **Track B, not Track C.** I am listing it for completeness and to record that I
considered it, not recommending Track C spend anything on it. If the coordinating agent wants this number, it
should come from Track B's comparator, where it is a baseline rather than a proposal.

**Confidence.** High that it would work; high that it should not be adopted.

---

## 2. Ideas I considered and rejected

**R1 — De-whiten the carrier: use the correlational form `Φᵀr` (spike-triggered-average style) instead of
inverting `ΦᵀΦ`.** This was my first idea and it looked strong: CEBRA never inverts a design matrix, its
objective is a first-moment alignment, and `(ΦᵀΦ)⁻¹` is exactly where variance blows up at 4 obs/param.
**It is already ~90% implemented.** `RIDGE_LAMBDA = 3.0` with the penalty `n·λ·diag(0,I)` and per-coordinate
standardized latent scores gives `S = ZᵀZ/n ≈ I + E`, so the estimator is
`(S + 3I)⁻¹Zᵀy/n ≈ ¼(I − E/4)Zᵀy/n`. The scalar ¼ is absorbed by the consumer MLP and the residual `E/4`
term is second order. The pure first-moment limit is worth at most one extra cell inside a λ sweep, not an
idea. **Do not re-propose this as a headline.**

**R2 — Match the source-side carrier's sampling noise to the deployment budget.** Rejected: already done.
`build_sparse_source_assets` fits a fresh M4 carrier at every legal contiguous 4-trial start of every source
session, so the consumer is already trained on deployment-noise-matched carriers. I had this as a strong
candidate until I read the datamodule.

**R3 — Choose *which* events to annotate (max-spread / optimal design event selection).** Rejected: measured
and dead. Handoff §10.2 gives `maxspread − first` at `+0.0013` (9/13, p=0.27) and `+0.0028` (8/13, p=0.58).
Chronological selection is already near-optimal, so there is no deployment gain here. Worth flagging because
"optimal experimental design for the calibration block" is an idea that *sounds* excellent and has already
been closed.

**R4 — Enrich the carrier with within-event temporal structure (K-point, curvature, sub-event splitting).**
Rejected for the first two: run, negative, best contrast `+0.0019` against a `+0.010` gate, with a mechanism
(§12.4 — the response is one scalar per event, `log1p(rate)`, so there is nowhere to attach predictor-side
temporal detail; and at fixed `q` it dilutes retained variance from 0.719 to as low as 0.602). Sub-event
splitting is *not* dead but it is already proposed and analysed in §12.5 by a previous agent, so re-proposing
it is duplication rather than contribution. I note only that my §0.2 measurement is directly relevant to it:
splitting raises observations but the risk §12.5 identifies is directional collinearity, and split-half cosine
≈ 0.03 says directional reproducibility is precisely what is broken, so I would expect sub-event splitting to
improve gain estimation and not direction — which is the failure mode §12.5 already predicted.

**R5 — Use CEBRA's `consistency_score` as a label-free cross-session drift diagnostic.** Rejected **as
described in the brief**. I read the implementation: `cebra/integrations/sklearn/metrics.py`,
`_consistency_datasets` raises if `labels is None` — "computing consistency between datasets requires labels" —
and it discretizes them into `num_discretization_bins = 100` bins to align embeddings before taking the R² of
the linear map. It also rejects anything but **1-D** labels. So cross-session consistency in CEBRA is
*label-requiring* and *scalar-label-requiring*; only the between-*runs* variant (same session, sample
correspondence available) is label-free. For H1's 7-DoF there is no 1-D label to bin. The salvageable idea —
comparing token-cloud second moments `EᵀE/N ∈ R^{W×W}`, which is well-defined without unit correspondence and
without labels — is folded into S2 rather than listed separately, because on its own it is a measurement, not
an improvement.

**R6 — Replace our MSE objective with InfoNCE end to end, i.e. make our decoder a CEBRA.** Rejected on
constraint grounds. InfoNCE produces an embedding, and CEBRA's own pipeline then fits a kNN or linear decoder
**on the target session** as a separate downstream step. That readout is either a target-session fit (which
guts the no-backprop claim's spirit even if it is closed form) or a source-fitted readout applied to an
unaligned target embedding (which will not transfer). It also abandons the calibrated 7-DoF velocity output
that the organizer endpoint scores. This is Track B's comparator by construction.

**R7 — Per-session renormalization of the carrier to match source token statistics.** Considered as a
mechanism for the date flip; rejected on my own measurement. Carrier tuning-coefficient RMS spans only
0.0228–0.0296 across the 13 held-in sessions, a 1.30× range, and the intercept RMS spans 2.297–2.540, a 1.11×
range. There is not enough cross-session scale drift for renormalization to explain a sign flip from `+0.0285`
to `−0.0226`. The *within*-vector imbalance (S1) is 30–60× and is the live problem; the *between*-session
drift is not.

**R8 — Raise `q` on H1 to recover the 28% of displacement variance the basis discards.** Rejected: it moves
the estimator the wrong way along the axis that §0.2 says is binding. At `q = 4` the estimator already sits at
4.0 observations per parameter with split-half cosine ≈ 0.03; adding latent dimensions lowers obs/param
further. §12.2 measured exactly this trade-off in the K-point family. If the retained-variance loss is to be
addressed, S7 (choose better directions at the same `q`) is the only version that does not pay in
observations.

---

## 3. What I could not verify

**About my own measurements**
1. The §0.1 and §0.2 numbers are **throwaway diagnostics without receipts**. They are not citable. The
   `basis_sha256` / `carrier_sha256` 13/13 match against `source_audit_v2r2.json` establishes that I fitted
   the sealed estimator, and nothing more. My `s_src = 1.0804` is a **proxy** computed at `start_index=0` on
   11 sessions, not the deployed value from the full fold-0 source cache over all legal starts; the
   *coordinate ratios* are unaffected because `s_src` is a single scalar, but the absolute normalized
   magnitudes are approximate.
2. My split-half statistic uses trial-parity halves of the M4 support, so each half has ~10 events against 5
   parameters. This is noisier than the deployed 20-event fit and **understates** its reliability by an amount
   I did not quantify. I asserted a Spearman–Brown-style correction leaves it low; I did not actually compute
   it, and cosine-of-weight-vectors is not the quantity Spearman–Brown was derived for.
3. ρ = 0.808 is against the estimator-level `median_delta_intercept`, and the split-half statistic and the
   target quantity are computed from overlapping data within each session, so part of the correlation is
   "session SNR predicts session SNR". That is fine for a *ranking gate* and not fine as a causal claim, and I
   have not separated the two.

**About our codebase**
4. I did **not** read `ψ_c`'s forward pass, its optimizer configuration, or whether any input normalization
   layer sits between the carrier and the first affine. The entire S1 Adam objection turns on this and I
   could not settle it.
5. I inferred that the **H-C** carrier has no intercept coordinate from eq. (15) using only `B_slope`, from the
   paper's `d_c = q+1 = 4`, and from `SE5_DIM = 5`. I did not read the H-C carrier construction code, so the
   "the arm that works has no intercept, the arm that fails does" contrast is an inference, not a verified
   fact — and it is load-bearing for S1(b).
6. I assumed the second H1 date in the `+0.0285 → −0.0226` comparison is **19250108**, from the config names
   `h1_hse5_lodo_{full,zero5}_19250108`. I did not confirm this against the run receipts, and S2's decisive
   test depends on it.
7. I did not verify that `legal_contiguous_starts` produces **non-overlapping** windows. S4's pre-screen needs
   non-overlapping windows to be meaningful; if the cache is a stride-1 sliding window, same-channel cosines
   across nearby starts will be spuriously high and the pre-screen must be restricted.
8. I did not count how many binned samples the unlabelled H1 calibration block actually contains. S3's whole
   argument is "thousands of unlabelled bins versus 20 labelled events"; the 4-trial prefix at 20 ms bins with
   `max_trial_length: 1024` suggests a few thousand bins, but I did not check, and if it is only a few hundred
   the variance argument weakens.
9. I did not measure the carrier coordinate scales on **center-out / RT (T4)**. My claim that H1's imbalance is
   unusually severe — and therefore that the scale pathology tracks where the sparse claim fails — is a
   prediction, not a measurement. The check is cheap and should be run before S1 is written up as an
   explanation rather than a fix.

**About CEBRA**
10. I read `criterions.py` and `metrics.py` but ran **no** CEBRA training. Everything I say about what its
    objective would do on our data is extrapolation from the source, not observation. The brief's statement
    that multi-session fitting with mismatched neuron counts works in our environment is taken on trust; I did
    not re-verify it.
11. I did not examine `cebra/distributions/` in detail, so my characterization of exactly how the
    auxiliary-variable positive distribution is sampled (particularly for continuous multi-dimensional
    labels, which is our case) is from the paper and the class names, not from reading the sampler.

**About the inferential chain everyone in this project relies on**
12. Every CPU screen proposed here, and every screen already run, measures an **estimator-level** quantity
    (`delta_intercept`, carrier fidelity, split-half cosine), while the paper's claims are **decoder-level**
    R². The only calibration point I found for that link is a single pair (CPU `+0.0144` → decoder `+0.0165`,
    handoff §12). I have used it throughout and it is `n = 1`. If a coordinating agent wants one piece of
    infrastructure out of this document that is not an idea, it is a proper estimator-to-decoder transfer
    curve, because every cheap screen in this project is currently interpreted through it.
