# Track C brainstorm — what our method can borrow from CEBRA

**Author:** Track C1 (Grok), 2026-08-13
**Status:** independent ideation. No production code. No C2 file was read.
**Sources read:** `cebra_exploration/docs/BACKGROUND_BRIEF.md`, `bci_paper_overleaf/paper_6pp.tex`, vendored CEBRA 0.6.1 (`criterions.py`, `distributions/continuous.py`, `solver/multi_session.py`, `integrations/sklearn/{cebra,metrics}.py`), H1 evidence handoffs (`HANDOFF_H1_MASTER_ANALYSIS_AND_PAPER_LINE_20260812.md`, `HANDOFF_H1_SPARSE_MAINLINE_STRENGTHENING_20260812.md`, `HANDOFF_H1_CONTEXT_PROGRAM_STATE_20260812.md`, `CURRENT_RESULTS.md`), `HANDOFF_COMPARATORS_20260812.md`. Sample-complexity audit JSON inspected for bin/event counts only.

---

## 0. Diagnosis that ranks the ideas

H1’s sparse claim fails in **two different places**. Mixing them produces the wrong CEBRA transplant.

**Failure A — estimator starvation (CPU, 13/13 held-in).** H-SE5 is per-channel ridge of `log(1+rate)` onto a source-PCA event feature `z_e ∈ R^4`, fitted on the event snapshots only (~20 events/session, 5 parameters, 4.0 obs/param). The event-budget sweep is monotonic with no plateau: 8 events retain 2.7% of the full effect. Condition number, retained variance, and retained energy all fail to predict forward transfer; directional diversity does. So “add regularisation / fix posedness” is a dead end (the brief already forbids reinstating posedness as cause). The unused resource is the rest of the four-trial prefix: ~2,300–2,500 eval-bins/session at M3 in `h1_sample_complexity_audit/audit.json`, against ~15–23 events. H-SE5 throws those bins away for `β_i`.

**Failure B — independently-trained Full−Zero5 does not replicate across source dates (GPU decoder).** Date 1: H-SE5 `0.500037` vs Zero5 `0.471569` (`+0.0285`). Date 2 (`19250108`): Full `0.495670` vs Zero5 `0.518260` (`−0.0226`). **Full is stable (~0.50); Zero5 jumped ~0.047.** The sparse token did not collapse. The replication rule `Full − independently-trained Zero5` is an interaction with a volatile activity-only baseline under source-date LOSO, plus the paper’s own attachment result: wrong-ish carrier content is worse than zero, so injecting a sparse `c_i` during source training can *hurt* the consumer relative to never seeing a carrier. The 13/13 estimator audit (correct − shuffle positive) is compatible with this: `β_i` can be informative and still poison `ψ_c` on a new source split.

CEBRA’s useful mechanic is not “contrastive learning” in the generic sense. It is:

1. Labels define a **positive distribution**, not a calibrated regression target (`TimedeltaDistribution`, `DeltaNormalDistribution`, discrete class pairing). That is cheaper in label information than `r_i = b + w^T φ`.
2. **CEBRA-Time** densifies the graph with temporal adjacency, no labels.
3. Multi-session alignment is enforced by **InfoNCE across separately instantiated encoders** (vendored sklearn: `nn.ModuleList` of full models, one per session, shared latent only through the loss — see §Unverified).
4. **Consistency** = `LinearRegression().score` between embeddings (optionally label-aligned and 1-D-discretised). **GoF** = `(log(batch_size × n_sessions) − InfoNCE) × log2(e)` bits.

Anything that uses (1)+(2) on the calibration prefix, closed form, attacks Failure A. Anything that uses (3)+(4) at source-training or as a gate attacks Failure B. The paper’s highest-value outcome is either a genuine cross-date sparse H1 result **or** a testable boundary. Ranked list below is ordered by that, then constraint, then cheapness.

---

## Ranked ideas

### 1. Hybrid time+event spectral basis (closed-form CEBRA-Hybrid `φ`)

**Statement.** Replace H-SE5’s event-only design matrix with the leading eigenvectors of a calibration graph whose edges are temporal neighbours **plus** sparse-event similarity, then keep `β_i = X^† r_i`.

**Mechanism.** Calibration only; `ψ_c`, cross-attention, and online decode unchanged. On the four-trial prefix, index **all** bins `t = 1…T` (~2.4k/session, CPU-trivial), not the ~20 event snapshots:

- `A_time[t,s] = 1` if `|t−s| ≤ τ` and `t,s` are in the same trial (CEBRA-Time; `TimeContrastive` uses a fixed offset, we use a short band).
- `A_event[t,s] = K(z_t, z_s)` if both bins fall in labelled events, else 0. `K` is the CEBRA `DeltaNormalDistribution` kernel: isotropic Gaussian on the existing event descriptor `z_e = P_src Standardize_src(x_e)` (displacement, or Context’s `[Δq, midpoint, tag]`). Sparse labels never enter as regression targets; they only mark which non-adjacent bins are positives.
- `A = A_time + λ A_event`. `Φ` = `q` leading eigenvectors of the normalised Laplacian (`q ∈ {3,4}` so the descriptor width still matches T4/H-SE5).
- `X = [1 | Φ]`, `β_i = X^† r_i`, `c_i = [ŵ_i, b̂_i]` (recompute `m_i = ||ŵ_i||` after, as now). Inject via eq. (carrier_injection).

**Pooling linearity is preserved if and only if `X` is built from time and labels, never from neural similarity.** Then `β_j = Σ_{i∈S_j} β_i` still holds. Neural kNN graphs would break it and leak the activity-only token into the carrier — that is a kill, not a variant.

Source training: independently retrain with the new `c_i` (GPU, later). The CPU screen does not need that.

**Named weakness.** H1 sparse claim / Failure A (event-only OLS ignores ~100× more bins); secondary, the “usable granularity is a function of event geometry” sentence in the conclusion, which currently has no constructive densification other than “use dense 7-DoF”.

**Violates no-target-session-backprop?** No. Eigendecomposition + OLS. Source retrain is free for the constraint (not free for the GPU queue).

**Cheapest decisive test.** CPU screen in the existing H-SE5 estimator-audit shape (13 held-in recordings, `index_heldin_calib` only): hybrid vs (i) sealed H-SE5, (ii) **time-only** `λ=0`, (iii) **label-shuffled** `A_event`, (iv) event-only Nyström of `A_event` with `A_time=0`. Metrics already used: correct−shuffle, delta-intercept, carrier fidelity vs full H-SE5. Confirm: hybrid beats H-SE5 **and** beats time-only on correct−shuffle, and shuffled `A_event` collapses toward time-only. Refute: hybrid ≈ time-only (labels add nothing) or hybrid ≤ H-SE5 on both correct−shuffle and fidelity.

**Expected failure / kill.** Failure B dominates: a better `φ` moves the 13/13 estimator audit and leaves date-2 `Full−Zero5` untouched, because Full is already stable at ~0.50. Kill the *recovery* claim if a later independently-trained date-2 (or a third source date) still has `Full < Zero5`. Keep the idea as densification of Failure A only if the CPU screen shows labels-plus-time beat both controls; otherwise abandon.

**Confidence: 0.45.** Graph construction is the right CEBRA mechanic for “labels as pairing + time as free observations.” Uncertainty is almost entirely Failure B: I do not know whether date-2 is an estimator problem. Also unverified: whether event windows are already so long that `A_time` inside events duplicates H-SE5, and whether trial-boundary handling of `A_time` matters.

---

### 2. Closed-form InfoNCE / encoding GoF as a per-unit carrier gate

**Statement.** Treat noisy `c_i` as the paper’s own “wrong content” (TS4/LS4 < zero). At calibration, compute a label-free-to-fit GoF for each unit; write `c_i = 0` (or shrink toward 0) when GoF is below a **source-only** threshold.

**Mechanism.** Calibration. After `β_i = X^† r_i` (H-SE5, T4, or Idea 1), compute one of:

- Encoding `R^2_i` of the OLS fit (already a by-product).
- Closed-form CEBRA alignment: for event pairs `(e,e')` sampled from `K(z_e,z_{e'})` vs uniform negatives, `align_i = mean_pos ⟨r̃_i(e), r̃_i(e')⟩` and `uniform_i = logmeanexp_neg` on z-scored event rates. This is the `align` / `uniform` split in vendored `infonce()` without a learned encoder.
- Session-level GoF: mean_i `R^2_i` or mean alignment. If the session fails a source-calibrated cutoff, emit the zero carrier for **all** units (abort to the activity-only token). Per-unit gating is the finer version.

`ψ_c` already zero-inits carrier columns; a literal zero `c_i` is in-distribution. Online path unchanged. Threshold fitted on source sessions only (e.g. the 10 source recordings of a date split), never on the target date’s decode score.

**Named weakness.** Failure B and attachment sensitivity (paper Fig. lattice: TS4/LS4 below Z4). Also the missing *principled* sparse boundary: currently the paper reports a sign flip and disclaims, without a statistic that would have predicted the flip.

**Violates no-target-session-backprop?** No. Scalar arithmetic on the calibration prefix.

**Cheapest decisive test.** CPU, no new decoder. On the sealed H-SE5 carriers for all 13 held-in sessions, compute per-unit `R^2_i` and pair-alignment. (1) Does session-mean GoF rank-order the two decoder dates’ `Full−Zero5` sign? Date 1 recordings `ses-19250101T111740/112404` vs date 2 `ses-19250108*`. (2) Source-only threshold: freeze a cutoff on date-1 source recordings; ask whether it would have aborted date-2 target blocks to zero. Confirm: GoF is lower on date-2 targets **and** a source-only cutoff would have written zero there, converting `−0.0226` into a predicted “no sparse claim / fall back” rather than a surprise. Refute: GoF distributions overlap across dates, or the cutoff that saves date 2 also aborts date 1 (the only positive cell).

**Expected failure / kill.** If Failure B is Zero5 luck and Full’s carriers are equally well-fit on both dates, GoF will not separate them — that is the kill for the *gate*, and it is itself a result (it redirects to Idea 4). Kill if any threshold that fires on date 2 also fires on the 13/13 estimator-positive sessions in a way that would have deleted the date-1 sparse cell.

**Confidence: 0.55.** Cheapest test that can *explain* the boundary; may not *recover* a sparse H1 number. Uncertainty: date-2 Full is 0.496 vs date-1 0.500, so unit-level encoding quality may be indistinguishable; the poison may be in `ψ_c`’s source-training distribution of `c_i`, which this gate does not see.

---

### 3. Nyström / kernel-mean carrier: consume sparse labels as a positive kernel, not as `φ(t)`

**Statement.** Stop regressing rate on a calibrated 4-D/7-D target. Represent each unit by its firing-weighted measure over the event-similarity graph, then take a 4-D Nyström coordinate — the closed-form object InfoNCE would have aligned.

**Mechanism.** Calibration. Let `K` be `E×E` (`E ~ 20`) with `K_{ee'} = exp(−||z_e − z_{e'}||^2 / 2σ^2)` on the existing event descriptors (CEBRA-Behavior positives). Nyström: top-`q` eigenpairs `(U, Λ)` of `K`, features `ψ(e) = Λ^{-1/2} U_{e,:}`. Unit carrier

```
c_i = Σ_e  r̃_{i,e}  ψ(e)     (rate-weighted kernel mean; r̃ = z-scored log-rate)
```

optionally concatenated with `b̂_i = mean_e r_{i,e}`. Width still `q+1`. Injection via `ψ_c` unchanged.

This is **not** OLS onto `z_e`. OLS asks for a linear encoding map `w` from behaviour coordinates to rate (needs the coordinates to be a well-spread basis; H1’s events are near-straight, arc/chord 1.023, K-point screen already dead). The kernel mean only asks “where on the similarity graph does this unit fire,” which is defined as soon as `K` is. Discrete native tags can enter as a block-diagonal component of `K` (same-tag positives) without being a one-hot in `φ` — relevant because Context’s tag supplies 67% of its M4 gain in the independently-trained decomposition, while same-checkpoint tag shuffle understated that by ~4×.

**Named weakness.** H1 sparse / Failure A: linear `φ` on poorly diverse events; also Context’s dependence on tags that H-SE5 throws away, without going back to the killed tag-free position screen.

**Violates no-target-session-backprop?** No. `E×E` eigen-decomposition, `E ≤ 23`.

**Cheapest decisive test.** Same CPU estimator audit as Idea 1, plus a **tag-augmented kernel** vs **displacement-only kernel** vs **shuffled-K** vs sealed H-SE5. Confirm: kernel mean beats H-SE5 correct−shuffle, shuffled-K falls to intercept-only, tag-augmented kernel captures a fraction of Context’s +0.014 vs H-SE5 without midpoint (midpoint already measured as ~0). Refute: kernel mean ≤ OLS on the same `z_e` (then the linear encoding model was not the bottleneck) or tag-augmented kernel fails to beat displacement-only (tags only helped Context via the jointly trained consumer, not via the estimator).

**Expected failure / kill.** With `E ~ 20`, Nyström of `K` is a rotation of a rank-`q` view of the same 20 points OLS already sees; it may be a cosmetic rewrite of H-SE5. Kill if, on the 13-session audit, kernel-mean fidelity to H-SE5 `c_i` is high (mean cosine ≳ 0.9) **and** correct−shuffle does not improve. Also kill if it needs `σ` fitted on target decode — that is peeking.

**Confidence: 0.35.** Right consumption of labels (pairing vs regression); high risk of being H-SE5 in a different basis. Idea 1 is the version that actually adds observations.

---

### 4. Source-only InfoNCE on identity-conditioned embeddings (CEBRA multi-session loss, our `E_i`)

**Statement.** Keep closed-form `c_i` and frozen target-session weights, but train the shared latent so that time points (or units) with similar auxiliary variables are aligned *across source sessions* — the actual multi-session CEBRA mechanism — instead of relying on MSE alone to teach `ψ_c` to read a noisy carrier.

**Mechanism.** Source training only. Online decode and calibration unchanged.

Let `h_i(t) = g(x_i(t) + E_i)` as in eq. (src). Add to the MSE decode loss (eq. loss) an InfoNCE term whose positive distribution is CEBRA’s, not a velocity target:

- **Time-point variant (closer to CEBRA-Behavior):** sample reference bins across source sessions; positives are bins with similar behaviour (`DeltaNormal` on velocity for RT/H-C, or similar `θ` / event tag for T4/H-SE5); negatives are the rest of the batch. Embeddings are mean-pooled `h` over units, or the `C` query outputs `ŷ_c` before readout. This is the vendored `MultiSessionSolver._inference` pattern: per-session objects, contrast in a shared `R^d`.
- **Unit variant (closer to our identity story):** positives are units (possibly from different source sessions) whose `c_i` are close in cosine; negatives are units with dissimilar or shuffled `c`. This directly trains `ψ_c` so that carrier pairing is the identity, which is what TS4/LS4 showed the decoder uses.

Temperature: start with CEBRA’s fixed cosine InfoNCE. Carrier columns of `ψ_c` remain zero-initialised.

This is the idea that attacks Failure B’s Zero5 jump: if source training with a sparse `c_i` currently *hurts* relative to Zero5 on some dates, an InfoNCE that treats `c_i` as a pairing variable (robust to affine rescaling of the coefficients) rather than as a calibrated vector that `ψ_c` must regress through may stop the poison. It also attacks the named held-in/held-out gap (0.4731 vs 0.2749, ~0.20 vs FALCON oracle ~0.04): CEBRA’s selling point is cross-session latent consistency.

**Named weakness.** Failure B (date-2 Full < Zero5 with Full stable); held-in/held-out gap; attachment sensitivity as a training-time, not only test-time, phenomenon.

**Violates no-target-session-backprop?** No. Gradients only on source sessions. Highlight: **deployment cost is exactly zero.**

**Cheapest decisive test.** Not a full GPU source train. Two CPU screens first: (a) on saved source-session `h_i` or `E_i` from existing H-SE5 date-1 vs date-2 checkpoints, compute CEBRA `consistency_score` (linear R²) between dates, Full vs Zero5, aligned on a 1-D index (preferred-direction bin or event-tag id — the sklearn helper **rejects `ndim>1` labels**). If Full embeddings are already more consistent than Zero5, Idea 4’s premise is weakened. (b) Tiny synthetic InfoNCE: take existing `c_i` tables, check whether a linear probe from `c_i` to `E_i` has lower R² on date 2 than date 1 (is `ψ_c` failing to read the carrier?). GPU only if (a) or (b) shows the consumer, not the estimator, is the date-2 fault. Confirm: after one source retrain, date-2 `Full−Zero5` flips positive **or** Zero5’s date-variance shrinks while Full stays ≥ date-1. Refute: InfoNCE matches MSE on both dates; or decode R² drops because the latent was hijacked away from velocity.

**Expected failure / kill.** InfoNCE on `h` can ignore the decode task (CEBRA itself needs a separate kNN/linear decode). Kill if a source-held-in velocity R² drops > 0.03 relative to the MSE-only sibling, or if the only way to keep decode is to down-weight InfoNCE until it is a no-op. Kill the unit-variant if it requires a cross-session unit matching (we do not have correspondence; pairing must be in `c_i`-space only).

**Confidence: 0.40.** Right attack on Failure B and the held-out gap; expensive to *confirm* (GPU source train), which is why it is not Idea 1. Uncertainty: I have not inspected whether existing checkpoints store `E_i` / `h_i` in a form a CPU consistency screen can consume.

---

### 5. Consistency of `E_i` (or query-space `h`) as the paper’s missing drift / sparse-boundary statistic

**Statement.** Adopt CEBRA’s consistency metric as a first-class, label-light diagnostic of “did the latent stay put,” and use it both as a paper figure and as a calibration abort (Idea 2’s session-level version, but on the token the decoder actually sees).

**Mechanism.** After the one forward pass of `ψ_c` (already allowed at calibration), we have `{E_i}` and can run frozen `g` on the calibration windows to get `{h_i}`. Consistency:

- **Across source sessions:** R² of a linear map between mean-pooled calibration embeddings, aligned on a 1-D discretised behaviour index (trial `θ` on center-out; event tag or 1-D PCA of `z_e` on H1). This is `_consistency_datasets` in `metrics.py`.
- **Source template vs target calibration:** same, using only the target prefix (no query labels). Low consistency → write zero carrier (gate) or report “sparse carrier off-manifold.”
- **Full vs row-shuffled `E_i`:** a label-free analogue of TS4 at representation level, cheaper than a decoder.

Does not change online decode unless used as the Idea-2 abort. Three-phase boundary intact.

**Named weakness.** Held-in/held-out gap; H1 sparse boundary currently described as a sign flip with no statistic; the six-minute cross-recording transfer result (`0.519024` vs `0.516518`) which already shows that “re-fit every session so it cannot drift” is false at short timescale — consistency is the quantity that would have shown that without a decoder swap.

**Violates no-target-session-backprop?** No. Linear R². The sklearn implementation fits `LinearRegression` on embeddings; that is not a decoder backward pass.

**Cheapest decisive test.** CPU forward of frozen `ψ_c` + `g` on held-in calibration prefixes (13 H1 recordings; also subject-M if tokens are on disk). Confirm: (i) consistency(Full) > consistency(Zero5) on date 1 and the reverse or a drop on date 2, **or** (ii) consistency of Zero5 across the five H1 source-date folds is more variable than Full, matching the 0.47 vs 0.52 Zero5 jump. Refute: consistency is flat across dates and arms, so it cannot explain or gate anything we care about.

**Expected failure / kill.** Cross-dataset consistency in CEBRA requires **1-D labels** (`ndim>1` raises). H1 is 7-DoF; a 1-D discretisation (speed, or PC1 of `z_e`) may be semantically empty. Kill if 1-D alignment R² is ~0 for every arm, or if Full vs TS4 does not separate (then the metric is deaf to the paper’s own attachment effect).

**Confidence: 0.50** as a **paper diagnostic**; **0.25** as a **method improvement**. Uncertainty: 1-D label restriction; whether calibration-prefix embeddings (4 trials) are long enough for a stable linear map.

---

### 6. `DeltaNormal` sample weights on the existing OLS (soft positives, same `φ`)

**Statement.** Keep H-SE5/T4’s `φ`, but weight calibration rows by a CEBRA-style kernel in auxiliary space so the fit emphasises locally supported positives instead of treating 20 heterogeneous events as equal linear observations.

**Mechanism.** Calibration. Current: unweighted `β_i = X^† r_i`. Proposed: WLS with `W = diag(w_e)`, `w_e = Σ_{e'} K(z_e, z_{e'})` (density) or, closer to InfoNCE, a two-term fit that upweights pairs inside `DeltaNormal` bandwidth `σ` and downweights isolated events. Source-frozen `σ` from source-session event pairwise distances (median heuristic). Same `q`, same `ψ_c`. On center-out, this is a continuous relaxation of “only use well-covered directions,” which is what the H1 conditioning sweep said actually predicts fidelity.

**Named weakness.** H1 conditioning sweep: diversity-at-fixed-count is real, condition number is the wrong statistic. Uniform OLS on ~20 events lets singleton outlier events (the K-point screen noted speed CV 0.569) leverage `w`. Also RT’s signed-content story: isolated endpoint directions should not dominate.

**Violates no-target-session-backprop?** No. Weighted OLS, still `β` linear in `r` for fixed `W(z)` (pooling linearity preserved).

**Cheapest decisive test.** CPU WLS re-fit of sealed H-SE5 design on 13 held-in sessions, σ from source median pairwise distance only. Compare correct−shuffle and fidelity to unweighted H-SE5; plus a **target-peeking σ** control that must *not* be used for selection. Confirm: WLS improves correct−shuffle at k=10 and k=14 (the budgets where diversity mattered) without needing more events. Refute: WLS ≈ downsampling to the dense core of `K`, i.e. it reproduces the budget sweep’s starvation (effective `k` drops, fidelity drops).

**Expected failure / kill.** Kernel weights *reduce* effective sample size, and the budget sweep says sample count is the binding constraint (monotonic, no plateau). Kill if effective `k = (tr W)^2 / ||W||_F^2` falls below ~14 and fidelity drops. That would mean Idea 1 (add observations) is the only OLS-family move, not reweighting.

**Confidence: 0.30.** Cheap and on-mechanic, but likely fights Failure A’s sample-count result. Worth a morning only because a negative is reusable (“soft positives without extra bins do not help”).

---

### 7. Source-frozen CEBRA-Behavior embedding of the auxiliary variable, then OLS onto that `z(t)`

**Statement.** Learn, on source sessions only, a low-D embedding of behaviour (or of event descriptors) with InfoNCE; on the target prefix, map sparse labels through that **frozen** map and OLS `r_i` onto the 3-D `z`, instead of onto 7-DoF or source-PCA `P_src`.

**Mechanism.** Source: fit CEBRA-Behavior (or a tiny InfoNCE MLP) on source **behaviour** `v(t)` or source event descriptors `x_e` — input dimension is `C` or the event-feature width, **not** neuron count, so there is no new-session input layer. Freeze. Target calibration: `z_e = f_src(x_e) ∈ R^3`, `β_i = X^† r_i` with `X = [1 | Z]`. Descriptor width 4, injection unchanged.

This is the precise difference from H-SE5’s `z_e = P_src Standardize(x_e)`: PCA maximises variance of the event features; CEBRA-Behavior maximises local similarity structure of the auxiliary variable (the `DeltaNormal` / InfoNCE geometry). Labels are still sparse; they are consumed as points on a source-learned manifold.

A CPU-only precursor that must be run first: **Isomap / Laplacian eigenmaps of source event descriptors, Nyström-extend to target events.** If the spectral map already beats `P_src` on the estimator audit, the GPU CEBRA-on-behaviour step is optional.

**Named weakness.** H1 sparse: `P_src` is a linear, variance-driven compression of 7-DoF event features that may not preserve the pairing geometry InfoNCE cares about; Failure A’s “4-D because we PCA’d to 4” is inherited from the consumer interface, not from a similarity model.

**Violates no-target-session-backprop?** No. `f_src` is frozen; target is one forward + OLS. Source GPU for CEBRA-on-behaviour is optional if the spectral precursor works.

**Cheapest decisive test.** CPU Nyström/Isomap of source `x_e` → 4-D, apply to target events, OLS, 13-session H-SE5 audit vs sealed `P_src`. Confirm: correct−shuffle improves and label-shuffle of target `x_e` before embedding kills the gain. Refute: ≈ `P_src` (then the linear PCA already captured what a similarity embedding would).

**Expected failure / kill.** Source and target event geometries differ (date-2 is a different source-date split; event tags and 7-DoF coverage may not overlap). Nyström from source `K` onto target events can silently extrapolate. Kill if target reconstruction error of `K` (Nyström residual) correlates with the sessions where H-SE5 already fails, **and** clipping those sessions is the only “gain.” Also kill if this becomes “train CEBRA on target behaviour” — that is Idea 9.

**Confidence: 0.35.** Conceptually clean CEBRA-Behavior transplant; `P_src` may already be good enough, and CEBRA-on-behaviour with `E~20` points per session is data-poor (source pooling of events across sessions is the only hope).

---

### 8. Closed-form session input layer: OLS of unit rates onto a source-frozen behaviour latent reconstructed from sparse labels

**Statement.** CEBRA’s new-session move is “train an encoder into an already-aligned latent.” Our closed-form analogue: reconstruct that latent from the sparse auxiliary variable with a source-frozen map, then take each unit’s loading onto it as `c_i`.

**Mechanism.** Source: train CEBRA-Behavior or the existing decoder’s query-space so we have a map `v ↦ z ∈ R^d` (or reuse Idea 7’s `f_src`). Target: from sparse events/trials, interpolate or hold `z(t)` on the prefix (trial-constant `z` on center-out; event-window-constant on H1). Then `c_i` **is** the OLS loading of `r_i` on `z(t)` — which is exactly `β_i` with `φ := z`. The architectural slogan: `E_i` remains the session-specific input; `z` is the shared latent CEBRA would have aligned by SGD.

If `z` is the decoder’s own query embedding `q_c` (eq. query), this is “encoding against the consumer’s queries,” not against the task basis. That is a real change: `φ` becomes source-consumer-dependent, so `c_i` lives in the space `g` already understands. Calibration: OLS only. Online: unchanged.

**Named weakness.** Failure B (`ψ_c` may not speak the same coordinates as OLS-on-`P_src`); the held-out gap (shared latent not actually shared). Distinct from FA alignment: we still emit a **per-unit** token, not a population-aligned activity matrix.

**Violates no-target-session-backprop?** No, if `z(t)` is produced by a frozen map from labels. Yes, if `z(t)` is produced by a target-session neural encoder (that is CEBRA’s real new-session input layer → Idea 9).

**Cheapest decisive test.** CPU: take existing source-trained query vectors `Q_c^{raw}` / `q_c`, form a `C`-D (or 4-D projected) `φ` from sparse labels via a linear map fitted on **source** `(v, q)` pairs, OLS `r_i` on that `φ`, run the H-SE5 estimator audit. Confirm: correct−shuffle beats `P_src` and a query-shuffled control. Refute: loadings are a rotation of T4/H-SE5 (consumer queries are aligned with the task axes already).

**Expected failure / kill.** On center-out, `q_c` may just recover `[cos θ, sin θ]`. Kill if cosine(new `c_i`, T4) ≳ 0.95 on subject-M. On H1, reconstructing 7-DoF-aligned `z(t)` from 20 events may smuggle dense interpolation — kill if the construction uses per-bin target velocity on the target session.

**Confidence: 0.30.** Fine as a slogan; likely collapses to Idea 7 or to T4. Included because it is the honest closed-form reading of “session-specific input layers.”

---

### 9. Constraint-violating upper bound: train a CEBRA session encoder on the target prefix

**Statement.** Do what CEBRA actually does on a new session — instantiate a new encoder (`offset1-model`, `num_input = N_target`) and train it with InfoNCE into the source-aligned latent — then either decode from that embedding (kNN/linear) or distill the encoder’s first layer into our `E_i`.

**Mechanism.** Breaks the three-phase boundary: the target session receives SGD. Two sub-variants:

- **9a (Track B, not us).** Full CEBRA-Behavior or Hybrid on the target prefix; kNN decode. This is a comparator, not an improvement of the carrier method.
- **9b (distill back into our objects).** After target-session InfoNCE, read out a per-unit vector (first-layer columns, or Jacobian of the embedding w.r.t. each unit, CEBRA has `attribution/`) and set `c_i` to that vector; freeze, one `ψ_c` pass, our decoder. Still **has** target backprop; it only preserves the online path.

**Named weakness.** Quantifies what the backward pass buys — the brief calls this a genuine result either way. Also the held-out gap vs NDT2-Multi few-shot (0.52 vs our 0.2749), which *does* backprop on the new session.

**Violates no-target-session-backprop?** **Yes.** That is the point. Worth it only as a numbered upper bound in the paper’s seven-axis table: “if we allow a target-session input-layer fit, here is ΔR².” Not as a replacement for the method. Track B owns the fair CEBRA comparator; this note exists so Track C does not “improve” the method by deleting its constraint.

**Cheapest decisive test.** Do not run this in Track C. If ever run: CPU smoke on a 20-vs-31 synthetic (already verified in-env) plus one held-in H1 prefix **without scoring the organizer endpoint**. Confirm: target-session InfoNCE GoF > source-zero-shot transform GoF. That still does not license a paper number.

**Expected failure / kill.** As a method idea, it is already dead (constraint). As an upper bound, kill if Track B’s comparator already reports the same Δ; do not duplicate.

**Confidence: 0.8 that it is the wrong method; 0.4 that the upper-bound number would be paper-useful.** Uncertainty: I did not check whether Track B is already specifying this.

---

## Ideas I considered and rejected

1. **Fit CEBRA-Time on the target prefix and use the embedding as `E_i`.** Requires a new encoder for a new `N` (backprop), is unlabeled (weakens the sparse-supervision thesis rather than strengthening it), and duplicates the activity-only token `f_id(mean f_pre(X_i))`. Time structure belongs in `φ` with labels as extra edges (Idea 1), not as a second identity path.

2. **Reinstating posedness / PCA-of-ridge / “underdetermined readout” as the thing CEBRA would fix.** Already tested and refuted (PCA k=8/16 *hurt* the H1 ridge; condition number does not predict H-SE5 transfer; date-2 H-SE5 is overdetermined at 4.0 obs/param and still loses `Full−Zero5`). A contrastive loss does not resurrect a causal claim the project killed.

3. **Cross-session unit matching via consistency (Hungarian on embedding R²).** The architecture is permutation-invariant *because* we refuse correspondence. Matching would silently reintroduce a channel table. Consistency as a *space* diagnostic (Idea 5) is the version that does not.

4. **Replace the velocity MSE decoder with kNN-on-CEBRA at online decode.** Changes the method family, needs a stored labelled embedding or target labels at query time, and fights the latency claim (~12% organizer-measured). CEBRA itself treats decode as a separate step; we already have a decode step.

5. **Sub-event splitting / K-point sampling as a “CEBRA positives along the trajectory.”** Already proposed in `HANDOFF_H1_CONTEXT_PROGRAM_STATE` §5, not from CEBRA, and the K-point CPU screen had **no arm passing** `+0.010` (best `+0.0019`) because events are spatially near-straight (arc/chord 1.023). InfoNCE along a straight chord does not create directional diversity. Idea 1’s time edges are different: they add **off-event** bins, not extra samples on the same displacement.

6. **Concatenate a source-trained CEBRA neural embedding of the calibration window into `c_i`.** Neuron count changes; a source encoder will not accept the target `N` without a new input layer (Idea 9). If we instead embed the activity summary we already have, we are widening `f_pre`, which is the Zero/B2 path the carrier is supposed to beat.

---

## What I could not verify

- **Nature paper vs vendored v0.6.1 multi-session architecture.** The brief (and the Nature narrative) say session-specific *input layers* into a shared network. The sklearn wrapper builds `nn.ModuleList` of **full independent models**, one per session, aligned only by InfoNCE (`_prepare_model`). I did not read the Nature PDF or non-sklearn pytorch configs to see whether a shared-trunk variant exists in this tag (`d1842ccc`).
- **I did not read C2’s file** (independence). I also did not read the CEBRA Nature methods in full, only vendored code + our brief §5.
- **Date-2 Zero5 jump as a fact about checkpoints.** I inferred it from the published pooled R²s (Full 0.500 vs 0.496; Zero5 0.472 vs 0.518) in the master handoff. I did not re-open the date-2 receipt to check seed, epoch, or whether Zero5 and Full used identical source lists beyond the handoff’s sentence that they did.
- **Whether `E_i` / `h_i` for H-SE5 date-1/date-2 checkpoints are on disk** in a form Idea 4/5’s CPU consistency screen can consume without a GPU replay.
- **CEBRA `consistency_score` on 7-DoF.** Code path `_consistency_datasets` requires 1-D labels. I did not check `align_embeddings` for a supported multi-D workaround, nor whether using PC1 of `z_e` is semantically valid on H1 events.
- **H1 event-window lengths in bins.** Audit JSON gives eval-bins and support-event *counts*, not duration of each event. Idea 1’s overlap between `A_time` and event windows depends on that.
- **Pooling-linearity numerically** for Laplacian `φ`: algebraically true if `X` is neural-independent; I did not run a throwaway pooled-vs-sum check.
- **Track B’s planned CEBRA comparator.** Idea 9 may already be their main arm; I did not read Track A/B docs (would leak other tracks).
- **InfoNCE temperature / batch size** that would be compatible with our windowed decoder batches; I did not inspect training-loop batch construction in `sua_exploration/`.
- **Whether FALCON H1 channels are threshold crossings** (handoff A5 still says “verify”). Irrelevant to ranking but relevant if someone proposes spike-sorting-dependent CEBRA features.
- **Area2_Bump / CEBRA monkey_reaching** label fields — not needed for Track C, not inspected.

---

## Suggested order of actual work (not extra ideas)

1. **Run Idea 2 + Idea 5 CPU diagnostics first** (hours, held-in only). They decide whether the date-2 failure is estimator GoF, token-space inconsistency, or Zero5/source-training volatility. That choice splits the remaining list.
2. If the estimator is the bottleneck: **Idea 1** (hybrid Laplacian `φ`), with Idea 3/6 as ablations in the same screen (kernel mean, WLS). Time-only and shuffle-event controls are mandatory.
3. If the consumer is the bottleneck: **Idea 4** source InfoNCE — but only after the CPU probe that existing `E_i` are inconsistently read. Do not spend GPU until that probe says so.
4. Do not implement Idea 9 inside Track C.
