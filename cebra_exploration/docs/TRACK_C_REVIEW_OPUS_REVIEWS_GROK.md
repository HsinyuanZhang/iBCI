# Track C cross-review — C2 (Opus) reviews C1 (Grok)

**Reviewer:** Track C2
**Reviewed:** `cebra_exploration/docs/TRACK_C_BRAINSTORM_GROK.md`
**Also read this round:** `COORDINATOR_VERIFIED_FINDINGS.md` (F1–F10),
`HANDOFF_H1_MASTER_ANALYSIS_AND_PAPER_LINE_20260812.md` §2.3,
`HANDOFF_MAINLINE_CLOSURE_20260811.md` L207–217,
`SPINT-main/src/models/components/h1_sparse_event_spint.py`,
`SPINT-main/src/models/components/h1_carrierid_spint.py`,
`SPINT-main/configs/model/falcon_h1_sparse_event_endpoint.yaml`,
plus one H-C checkpoint tensor read.
**Rule followed:** their document was not edited; my own document was not edited. All revisions are here.

---

## 1. Verdict in one paragraph

This is a good document and in one respect it is better than mine. Its §0 failure decomposition —
separating estimator starvation from decoder-level non-replication, and noticing that on date 2 it is
**Zero5 that moved, not Full** — is the single most useful reframing produced by either track, and I did
not have it. I verified those numbers exactly. But the inference C1 draws from that observation does not
survive contact with the same-checkpoint label-shuffle control, which they did not look up and which
reverses their conclusion. Their Idea 1 is well-motivated and attacks the mechanism the project itself
identified, but it constructs a **per-session** basis and never says how it is aligned across sessions —
precisely the failure mode of coordinator finding F8 — and its stated decisive test is entirely
within-session and would score a broken idea as a success. Their Idea 2 rests on a false claim about
`ψ_c` that I can now disprove from the code and the trained weights, and the gate it proposes would have
made date 2 **worse**, not better — which is equally a problem for my own S2. Their Idea 3 is
algebraically OLS on a rotated basis and can be killed in twenty minutes of algebra rather than by a
13-session screen. Their recommended first experiment is partly **not executable**: it needs H-SE5
`E_i`/`h_i` from checkpoints that are not on disk.

---

## 2. Factual errors and unsupported claims

### 2.1 The load-bearing error: "a literal zero `c_i` is in-distribution" (L67)

> "`ψ_c` already zero-inits carrier columns; a literal zero `c_i` is in-distribution."

This is a non-sequitur and it is wrong. Zero-initialization describes the **weights at step 0**; it says
nothing about whether a zero **input** lies in the distribution the trained network expects. I can now
show it is not. From my round-1 measurement, the H-SE5 carrier's intercept coordinate has mean **2.105**
and RMS **2.235** after the global normalizer, while the four tuning coordinates have RMS 0.018–0.034.
A zero vector therefore sits about 2.1 units from the centre of the carrier distribution **along the
coordinate that dominates the input**. It is strongly out of distribution.

`h1_sparse_event_spint.py:66` does implement exactly this substitution (`torch.zeros_like(carrier)`),
but that flag is used for the **separately trained** Zero5/H-C0 arm — a model that saw zeros throughout
training, for which zeros are trivially in-distribution. C1's phrase "abort to the activity-only token"
(L65) silently conflates *substituting zeros into a Full-trained model* with *switching to the
independently trained Zero5 model*. Those are different objects with different scores, and the project
has measured both.

### 2.2 The consequence, which damages their Idea 2 **and my own S2 equally**

`HANDOFF_H1_MASTER_ANALYSIS...` L48 records `H-SE5 − same-checkpoint zero = +0.011442` on date 1. So
zeroing the carrier on a Full checkpoint **costs 0.0114**; it does not recover the independently trained
Zero5 level. On date 2, Full = `0.495670` and Zero5 = `0.518260`. Applying the same-checkpoint zero
intervention to date-2 Full would land near **0.484**, i.e. *below* Full and far below Zero5.

**A gate that aborts to a zero carrier would have made date 2 worse, not better.** The only way to reach
Zero5's 0.518 is to ship and switch to a second, separately trained model — a deployment cost neither
document acknowledged. This is a genuine refutation of the naive gate in both our documents and I am
recording it against my own S2 as much as against their Idea 2.

*Caveat I must state:* the `+0.011442` figure is measured on date 1 only. Extrapolating the same offset
to date 2 is an assumption, not a measurement. But the burden of proof now sits with the gate.

### 2.3 "The sparse token did not collapse" (L15) is contradicted by the content-isolating control

C1's central inference is that because Full is stable (`0.500 → 0.496`) while Zero5 jumped
(`0.4716 → 0.5183`), the sparse carrier did not fail — the baseline moved. The *observation* is correct
and I verified it (`HANDOFF_H1_MASTER_ANALYSIS...` L74–76). The *inference* is not.

`HANDOFF_MAINLINE_CLOSURE_20260811.md:213` reports that on date 2 the **same-checkpoint endpoint-label
shuffle is above Full by `+0.001448`**. That contrast compares Full against a corrupted version of
*itself*, so it is immune to Zero5's volatility — it is the contrast that isolates carrier *content*.
On date 1 that contrast is `+0.007092` in Full's favour (L45); on date 2 it flips to `−0.001448`.

So the content-controlled contrast fails on date 2 as well. C1's "Full is stable, therefore the token
did not collapse" is not supported: the token's *content* stopped carrying signal, and the pooled Full
score stayed flat because the activity path picked up the slack. This does not erase their finding —
Zero5's volatility is real and important — but it removes the basis for their Failure-A/Failure-B split
being *clean*, and it weakens their argument for prioritizing consumer-side ideas (Idea 4) over
estimator-side ones.

### 2.4 An overstatement in Idea 1: the observation gain is not ~120×

L13 and L34 motivate Idea 1 by "~2,300–2,500 eval-bins/session" against "~20 events", i.e. a two-orders
gain in regression rows. The bin count is plausible and consistent with the `~3,100 eligible rows` at M4
in the strengthening handoff §B2, but **rows are not independent observations**. 20 ms bins are strongly
autocorrelated, and `A_time` (L36) deliberately makes neighbouring bins similar, so the Laplacian
eigenvectors are smooth temporal functions and the regression is effectively fitting a handful of smooth
templates. The effective sample size is governed by the autocorrelation time, not the bin count. The
gain is real but it is nowhere near 120×, and their own posedness arithmetic (obs/param) should not be
restated with the raw bin count.

### 2.5 Idea 3 re-enters a route the project has explicitly closed, without saying so

L93 and L99 propose a **tag-augmented kernel** and justify it by Context's tag contribution. The
strengthening handoff §7 says plainly: *"Do not try to revive Context Full or tag content. Ten CPU
design programs (TCE5, nested context, meta-learned basis, semantic V4, C2F5, LRT5, NLE5, PNO5, QC2F5)
all plateaued at +0.012 to +0.016, below the +0.02 gate. That route is exhausted."* C1's mechanism (a
block-diagonal same-tag kernel) is genuinely different from a one-hot in `φ`, so it is not automatically
excluded — but the document must engage with the prohibition and does not. As written, a coordinator
reading only C1's document would authorize an eleventh attempt at a closed route.

### 2.6 Claims I checked that are correct — credit where due

- **L21, multi-session CEBRA is an `nn.ModuleList` of full independent models.** Correct, independently
  derived, and it caught an error in the brief. F2b credits them and so do I.
- **L22, `GoF = (log(batch_size × n_sessions) − InfoNCE) × log2(e)` bits.** Matches
  `infonce_to_goodness_of_fit`'s docstring (`S = log N − InfoNCE`, N covering batch size and session
  count) with the nats→bits conversion. Correct.
- **L15 date-2 numbers** `0.495670 / 0.518260 / −0.022590`. Verified exactly.
- **L13**, condition number fails to predict forward transfer while diversity does. Matches §10.2.
- **L41**, pooling linearity survives iff `X` is neural-independent. Algebraically right, and the
  "neural kNN graphs are a kill, not a variant" call is exactly correct.
- **L166**, WLS keeps `β` linear in `r` for fixed `W(z)`, so pooling linearity survives. Correct.
- **L148**, a `LinearRegression` fit on embeddings is not a decoder backward pass. Correct.
- **L152**, `consistency_score` rejects `ndim > 1`. Correct, and they flagged it as a live risk before
  the coordinator confirmed it (F7).

### 2.7 One thing they mention that I could not verify at all

L146 cites a "six-minute cross-recording transfer result (`0.519024` vs `0.516518`)". I did not locate
it. Flagging so the coordinator can confirm it exists before it is used to support Idea 5.

---

## 3. Their top three ideas

### Idea 1 — hybrid time+event spectral basis

**The motivation is the best in either document.** Handoff §12.4 concluded that dense wins *"because it
has temporal resolution on both sides, predictor and response"*, and that enriching the predictor is
futile while the response stays one scalar per event. Idea 1 is the only proposal in either document
that gives temporal resolution on **both** sides using nothing but sparse labels and time. It also keeps
`β_i = X†r_i` verbatim, so the paper's pooling-linearity theorem transfers unchanged. That is a real
advantage over my own S3, which estimates the latent basis from the neural data and therefore does
**not** preserve pooling covariance — the PCA basis itself changes when channels are summed. I say that
plainly: **on the pooling-linearity axis, their Idea 1 dominates my S3.**

**The serious defect: no cross-session identification, and the test cannot see it.** `Φ` is the leading
eigenvectors of a graph built on the **target session's own** calibration block. Eigenvectors are
defined only up to sign, up to ordering when eigenvalues are near-degenerate, and up to rotation within
a near-degenerate subspace — and none of those are anchored to anything the source-trained `ψ_c`
learned. H-SE5 avoids this by freezing `P_src` so coordinate *k* means the same physical thing in every
session; `_canonicalize_component_signs` (`h1_sparse_event_endpoint.py:425-431`) exists precisely
because even a *source-pooled* SVD needed its signs pinned. A per-session eigenbasis makes that strictly
harder, and sign canonicalization does not fix ordering or subspace rotation.

This is exactly coordinator finding **F8**: *"any idea that borrows 'CEBRA aligns sessions' must specify
what plays the role of the cross-session positive pair. Alignment is not a property of the architecture;
it is a property of the sampling."* Idea 1 inherits CEBRA's basis-construction move and drops its
alignment mechanism.

There is a further tension the design does not resolve. `A = A_time + λ A_event`. The only cross-session
anchor is `A_event`, built on the source-frozen `z`. At small `λ` the eigenvectors are the session's own
slow temporal modes — maximally densifying, minimally identified. At large `λ` they approach the
source-frozen event geometry — identified, but the time densification that motivated the idea is gone.
**Densification and identifiability trade off directly along `λ`**, and the document treats `λ` as a
free hyperparameter.

**Their stated decisive test (L49) would not catch this.** Every metric they list — `correct−shuffle`,
`delta_intercept`, carrier fidelity — is computed **within a session**: `forward_transfer` fits on the
support events and predicts *later events of the same session*. A session-local basis with 100× the
rows will do *very well* on all of them, plausibly beating sealed H-SE5, while being useless to `ψ_c`
because the coordinates do not mean the same thing across sessions. Their confirm criterion would
return a strong positive for an idea that is broken at the decoder level.

**Fix, and it is cheap:** add a cross-session identifiability metric to the same screen. Fit `Φ`
independently on two sessions, map both to the shared source event coordinates, and measure the
principal angles between the two `q`-dimensional subspaces (or the stability of the carrier's meaning
under the session-to-session basis change). If the subspaces do not align, the idea is dead regardless
of how good `delta_intercept` looks. With that added, the screen becomes genuinely decisive and I would
rank Idea 1 in my top three.

**Verdict:** strong idea, wrong test. Sound after the alignment fix; the `λ` tension should be made an
explicit axis of the screen rather than a hyperparameter.

### Idea 2 — goodness-of-fit-gated carrier

Same family as my S2. Assessment of the parts:

- **The statistic.** They propose per-unit encoding `R²_i` and a closed-form align/uniform split. Both
  are reasonable and cheap. Neither was measured. My round-1 screen measured a different statistic —
  split-half carrier weight cosine — against the sealed per-session `median_delta_intercept` and got
  Spearman **ρ = 0.808, p = 0.0008, n = 13**, while support event count gave ρ = 0.175. So the *idea
  class* is supported by evidence; the specific instrument they chose is untested and, for `R²_i`, I
  expect it to underperform, because on the fold-0 session the sealed receipt records
  `median_r2_correct = −0.026` — the carrier's out-of-sample event-level R² is already at or below zero,
  so it has very little dynamic range to threshold on.
- **The fallback target is wrong** — see §2.1/§2.2. Gating to zero would have hurt date 2.
- **The test is not decisive.** L73 asks whether session-mean GoF rank-orders "the two decoder dates'
  `Full−Zero5` sign". **n = 2.** Their own refute condition ("GoF distributions overlap across dates")
  is not assessable at n = 2. This is the weakest cost/design claim in their document, and it matters
  because Idea 2 is one of the two things they want run first.

**But their test aims at the right target and mine does not.** They score against the *decoder-level*
`Full − Zero5`; I scored against the *estimator-level* `delta_intercept`, which is exactly the
inferential gap I flagged as item 12 of my own unverified list. The right screen is the union, and §6
below turns that into a concrete first experiment.

### Idea 3 — Nyström kernel-mean carrier

**It is algebraically OLS on the Nyström features, up to a per-session diagonal.** With
`Ψ = U_q Λ_q^{−1/2}` (their `ψ(e)`), their carrier is `c_i = Ψᵀ r̃_i`, while OLS on the same design is
`(ΨᵀΨ)^{−1}Ψᵀ r̃_i = Λ_q Ψᵀ r̃_i`. So kernel-mean and OLS differ by exactly `Λ_q`. C1 flags "high risk of
being H-SE5 in a different basis" (L103) and sets a kill criterion for it — good instinct — but this is
not a risk to be screened on 13 sessions, it is an **identity provable in five lines**, and it should be
settled by algebra before any data is touched.

Two further points they missed:

1. `Λ_q` is the eigenvalue spectrum of the **target session's own** `E×E` kernel, so it is
   session-dependent and therefore *not* absorbable by `ψ_c`'s fixed first layer. The idea has Idea 1's
   identification problem in a more acute form, since the basis is estimated from only `E ≈ 20` points.
2. The span is likely the same as H-SE5's. For a Gaussian kernel on `z_e ∈ R⁴` at wide bandwidth, the
   leading eigenfunctions are approximately the constant and the linear coordinates — i.e. `[1, z]`,
   which is exactly H-SE5's design. At narrow bandwidth they become localized bumps, which at `E ≈ 20`
   is overfitting. **There is no bandwidth at which this is both different from H-SE5 and stable.**

**What is salvageable, and it is not nothing:** the *tag* block of `K` is genuinely outside the span of
`[1, z]`, so a same-tag kernel block is a real change rather than a reparametrization. That is the only
live part of Idea 3 — and it is the part that collides with the §7 prohibition (§2.5 above).

**Verdict:** correctly ranked third by them; I would rank it lower still and convert it from a screen
into a twenty-minute algebra check plus a decision about whether the tag route may be reopened at all.

---

## 4. Per-unit versus per-session gating — the coordinator's question

**Both granularities are right, for different jobs, and both documents get the fallback target wrong.**

**Per-unit is the right granularity for the estimator**, and it is better supported than C1 realizes:
the paper's own dense arm already does it. Eq. (15) shrinks per channel,
`C_i = μ_C + τ²/(τ²+ν_i)(C_raw,i − μ_C)`, with a per-channel variance `ν_i` available in closed form.
C1's per-unit gate is the hard-threshold version of something H-C already does softly, and they never
cite it. My split-half statistic is *also* natively per-channel — I took a median only to get a session
summary — so the per-unit version costs nothing extra.

**Per-session is the right granularity for the claim**, not for the intervention. The paper's stated
need is a *principled, testable boundary* for where the sparse mechanism applies. That is a per-session
question ("does the sparse claim hold on this recording?"), and it is answered by a session-level
statistic. My ρ = 0.808 result is evidence for exactly that use.

**Both are wrong about the target.** Neither hard-zeroing a unit (theirs) nor zeroing the session
(mine) is safe, because zero is out of distribution for a Full-trained `ψ_c` (§2.1) and empirically
costs 0.0114 (§2.2). The correct fallback is **shrinkage toward the source carrier mean `μ_C`**, or
better toward a per-channel activity-predicted prior — which is my S5 and is eq. (15)'s existing
structure. Shrinkage keeps the input inside the training distribution by construction and degrades
continuously instead of discontinuously.

**Synthesis I would actually build:** per-unit empirical-Bayes shrinkage toward a source-learned prior,
with the shrinkage weight driven by a per-channel reliability statistic (split-half cosine or `ν_i`),
plus a per-session summary of the same statistic used **only** as a reporting rule for where the sparse
claim is asserted. That takes the best of their Idea 2, my S2, and my S5, and it avoids the zero-target
defect that sinks the naive version of all three.

---

## 5. Constraint violations

**None undeclared.** I checked every idea against the no-target-session-backprop rule.

- Idea 9 is correctly and prominently labelled as violating, and correctly assigned to Track B
  (L229). Identical to my S9, including the "do not implement in Track C" conclusion.
- Idea 8 (L208) is the one place where a violation could have hidden, and C1 catches it themselves:
  *"No, if `z(t)` is produced by a frozen map from labels. Yes, if `z(t)` is produced by a target-session
  neural encoder."* That is the correct discrimination and it is stated explicitly.
- Ideas 1, 3, 6, 7 are closed-form on the target and clean.
- Idea 4 is source-training only, correctly flagged as zero deployment cost.
- Idea 5's `LinearRegression` on embeddings is closed form and is not a decoder gradient step.

One thing to watch that is not a violation but is adjacent: Idea 3's kill criterion (L101) says "kill if
it needs `σ` fitted on target decode — that is peeking." Correct and well-policed. Idea 8's kill
criterion (L212) similarly forbids per-bin target velocity. Both good.

---

## 6. Cost claims, and resolving the two different first experiments

### 6.1 Their recommended first experiment is partly not executable

C1 recommends (L273) running Idea 2 + Idea 5 CPU diagnostics first, "hours, held-in only". Idea 5 needs
`{E_i}` and `{h_i}` from the H-SE5 date-1 and date-2 checkpoints, and C1 flagged as unverified whether
those are on disk.

**I checked. They are not.** `SPINT-main/pilot_artifacts/` holds 48 `.ckpt` files, none under
`h1_sparse_event_endpoint/` or `h1_hse5_lodo_19250108/` — those directories contain only
`runtime_logs/`, `source_snapshot/` and `fidelity_dated/`. Idea 5's screen therefore requires a GPU
replay of two 50-epoch runs before its "morning of CPU" can begin. The cost estimate is wrong by a
training run, and Idea 4's CPU screen (a) has the same dependency (L130 flags it, correctly).

Idea 2's estimator-side half **is** executable and cheap. Idea 1's, 3's, 6's and 7's screens are all
executable and cheaply costed — those estimates I believe.

### 6.2 Their *framing* is better than mine; my *instrument* is better than theirs

C1's stated reason for going first — decide whether the H1 failure is in the estimator or in the
independently-trained consumer, because that choice splits the remaining list — is **the right first
question, and better than my framing.** My round-1 recommendation (run the carrier-coordinate-imbalance
checkpoint diagnostic first) implicitly assumes the answer is "the interface", which is exactly the
thing that should be tested rather than assumed. I concede that point.

But their instrument cannot answer it: half of it needs checkpoints that do not exist, and the other
half is an n = 2 comparison.

### 6.3 The experiment I now recommend instead, which is neither of ours

Combining C1's per-recording decomposition with my split-half statistic produces a decoder-level test
that needs no checkpoints and no training. The date-2 per-recording deltas are
`−0.042554 / +0.001088 / +0.027104` (`HANDOFF_H1_MASTER_ANALYSIS...` L76) — **2 of 3 positive**, with
the pooled negative driven by one recording. Date 1 is `+0.02590 / +0.03593`. Against my round-1
split-half cosines:

| date | recording (assumed order) | split-half cosine | decoder `Full − Zero5` |
|---|---|---:|---:|
| 19250101 | `...T111740` | +0.0656 | +0.02590 |
| 19250101 | `...T112404` | +0.0598 | +0.03593 |
| 19250108 | `...T110520` | **−0.0358** | **−0.042554** |
| 19250108 | `...T111022` | −0.0175 | +0.001088 |
| 19250108 | `...T111455` | +0.2000 | +0.027104 |

On the failing date the ordering is **monotone, 3/3**: the statistic ranks the single catastrophic
recording as the worst carrier and the only clearly positive recording as the best. Across all five,
Spearman ρ ≈ **0.70**.

**Two caveats, both serious.** First, the handoff lists the three deltas without naming recordings; I
assumed the canonical `H1_HELDIN_SESSIONS` order. I searched for the underlying receipt and it is not on
disk, so **this ordering is unverified and the whole table collapses if it is wrong.** Confirming it is
step zero and costs one grep of the date-2 terminal receipt if it can be recovered. Second, n = 5 is far
too small for a p-value; this is a direction-and-outlier check, not a test.

**Recommended first experiment:** verify the recording ordering, then compute per-recording split-half
cosine and per-unit encoding `R²` for all 13 held-in sessions and regress both against the five
available decoder-level deltas. Hours of CPU, no checkpoints, no GPU. If the statistic tracks the
decoder-level delta, the failure is estimator-side and reproducibility-driven, which promotes Idea 1
(with the alignment fix) and my S3/S5. If it does not, the failure is consumer-side, which promotes
Idea 4 and my S1. **This answers C1's question with an instrument that exists.**

---

## 7. Idea overlap, gaps, and who is better where

### Real overlap — do not double-count

| Theirs | Mine | Note |
|---|---|---|
| Idea 2 + Idea 5 (GoF gate, consistency) | S2 (abstention gate) | Same family. Mine has a measured statistic; theirs has the better target and the finer granularity. See §4. |
| Idea 4 unit-variant | S4 (carrier-view InfoNCE) | Same family, different positives: theirs pairs units with similar `c_i`, mine pairs two calibration windows of the same unit. |
| Idea 7 | S7 (PLS basis) | Nearly identical: replace `P_src` with a similarity- or response-driven basis. Theirs adds an Isomap/Laplacian CPU precursor, mine uses PLS. Merge them. |
| Idea 1 | S3 (latent-anchored carrier) | Same goal — spend the unlabelled prefix — different mechanism. Theirs preserves pooling linearity, mine does not. |
| Idea 9 | S9 | Identical, both correctly deferred to Track B. |

### Theirs that I did not have

1. **The §0 failure decomposition, and the Zero5-moved observation.** The most valuable thing in either
   document. It is why I went looking for the per-recording deltas, which produced §6.3.
2. **Idea 1's neural-independent graph basis**, which keeps pooling linearity where my S3 breaks it.
3. **Idea 6, `DeltaNormal` sample weighting.** I did not have this. It is cheap, preserves pooling
   linearity, targets the measured diversity result, and its kill criterion (effective
   `k = (tr W)²/‖W‖_F²` falling below ~14) is the sharpest kill criterion in either document. Their own
   confidence of 0.30 is about right, but the negative would be reusable.
4. **Idea 8's "encode against the consumer's queries `q_c`".** Genuinely novel and I like it more than
   they do. Every other basis proposal (theirs and mine) optimizes the basis for the *label* or the
   *neural response*; this one optimizes it for the object that actually has to read the carrier. Given
   that §2.3 shows date-2's failure is a content-readability failure, that is arguably the best-aimed
   basis idea on either list. Their kill criterion (cosine to T4 ≳ 0.95 on subject-M) is exactly right.

### Mine that they did not have

1. **The carrier coordinate scale imbalance and the H-C/H-SE5 structural asymmetry** (§8 below).
   Entirely absent from their document.
2. **A measured gate statistic** (split-half cosine, ρ = 0.808, n = 13). They propose gating without
   measuring anything.
3. **Empirical-Bayes shrinkage toward a learned activity→carrier prior** (my S5). They have no
   shrinkage-target idea at all, which is why their gate zeroes to the wrong target.
4. **The behaviour-free pseudo-label carrier** (my S6): fit the carrier against the frozen decoder's own
   predicted endpoints, consuming zero labels. They rejected the neighbouring idea (CEBRA-Time as `E_i`)
   but never considered using the decoder's own output as the auxiliary variable. Highest upside on
   either list, with a CPU-only pre-screen that needs no checkpoint.
5. **Per-session activity standardization** (my S8). They propose nothing on the activity path.

### The plain comparison the coordinator asked for

**Their Idea 1 is better than my S3**, on pooling linearity and on directly answering §12.4's
predictor-and-response diagnosis. It needs the alignment fix and a cross-session metric in its screen.
**My S2 is better instrumented than their Idea 2**, but their granularity and target choice are better
than mine, and §2.2 damages both.

---

## 8. What I settled this round about my own document

The coordinator asked me to close two load-bearing items. Both are closed.

### 8.1 `ψ_c`'s forward pass and optimizer — settled, and it cuts both ways

`h1_sparse_event_spint.py:42-49, 58-67`:

```python
self.carrier_post_pool = nn.Sequential(
    nn.Linear(HIDDEN_DIM + CARRIER_DIM, HIDDEN_DIM), nn.ReLU(), ...)
with torch.no_grad():
    self.carrier_post_pool[0].weight[:, HIDDEN_DIM:].zero_()
...
return self.carrier_post_pool(torch.cat((pooled, effective.to(pooled)), dim=-1))
```

- **There is no input normalization** anywhere between the carrier and the first affine. The raw 5-vector
  I measured goes straight into `Linear(37→32)`. My round-1 premise is confirmed.
- The imbalance is in fact **wider** than I reported: the concatenation puts 32 activity features next to
  5 carrier features with no relative scaling either, so there is a second unmanaged scale boundary I
  did not name.
- `configs/model/falcon_h1_sparse_event_endpoint.yaml`: `torch.optim.Adam`, `lr: 5.0e-5`,
  **`weight_decay: 0.0`**.

The weight-decay half of my Adam objection is dead — there is no decay to penalize large compensating
weights. The scale-invariance half is real in principle: Adam's update is ≈ `lr · sign(gradient)` on a
zero-initialized weight, so growth rate does not depend on input scale. **But the required weight
magnitude still scales as 1/input, and growth is capped at about `lr` per step.** So the question is
empirical: did the trained network compensate?

**Measured, on the closest available checkpoint** (`h1_carrierid_distribution/.../q4/.../epoch_049.ckpt`,
`net.carrier_post_pool.0.weight`, shape `(32, 36)`):

| block | init | learned RMS | max abs | mean column norm |
|---|---|---:|---:|---:|
| activity columns 0:32 | `U(±0.167)` | 0.0959 | 0.167 | 0.541 |
| carrier columns 32:36 | **exactly 0** | **0.00328** | **0.0121** | **0.0176** |

The `s4` sibling gives 0.00252 / 0.00697 / 0.0140. Two readings:

1. **No scale compensation.** If the network were compensating for a smaller carrier input it would grow
   those weights *larger* than the activity weights; they end **29× smaller**. The activity columns are
   still at their initialization RMS (0.0962 expected for `U(±0.167)`), i.e. they barely moved.
2. **The growth-budget argument holds empirically.** From exactly zero, at `lr = 5e-5` over 50 epochs,
   the carrier columns reached a max magnitude of 0.012 — roughly 240 steps' worth of consistent
   movement. They are nowhere near the magnitude that would be needed to compensate a 30–60× input
   deficit.

**Net effect on my S1:** the premise is confirmed and the main theoretical objection is empirically
weakened, so S1 survives. But a new caution appears that I did not have: **H-C achieves its `+0.0563`
while operating in exactly this low-magnitude regime.** The carrier is currently a small perturbation on
the activity path and it works. Amplifying it is therefore a genuine risk as well as an opportunity, and
S1 must be run as a **rescaling sweep with the sealed scale included as a control**, not as a one-shot
"fix". *Caveat:* this checkpoint is H-C (4-wide, normalized EB carrier), not H-SE5, so I measured the
learned weights but not the H-C input scales, and cannot form the `w·x` product ratio directly.

### 8.2 "H-C has no intercept coordinate" — settled, my inference was right

Confirmed from four independent places:

- `h1_carrierid_spint.py:44`: `concat(normalized_carrier[B,N,4]) -> 36->32->32->700`.
- `h1_carrierid_spint.py:64`: "H1 CarrierID requires a four-dimensional normalized EB carrier".
- The trained checkpoint's `carrier_post_pool.0.weight` is `(32, 36)` = 32 activity + **4** carrier.
- `h1_sparse_event_spint.py:16`: H-SE5's identity path is `IDENTITY_PARAMETERS = 58_172` against the
  paper's `58,140` for H-C. The difference is exactly **32** = one extra input column × 32 hidden units.

Together with eq. (15) building `C_raw` from `B_slope` only and discarding `b`: **the dense arm that
works carries four slope-derived coordinates and no intercept; the sparse arm that fails carries the
same four plus an intercept whose mean is 2.105 and which is 30–60× larger than the other four.** This
was my S1(b) and it is now a verified structural asymmetry rather than an inference.

Two further items from my round-1 unverified list are also closed: the second date **is** `19250108`
(confirmed, `HANDOFF_H1_MASTER_ANALYSIS...` §2.3), and the estimator→decoder inferential link now has
**five** decoder-level data points instead of one (§6.3).

---

## 9. Their rejections

**Rejection 1 — fit CEBRA-Time on the target prefix and use the embedding as `E_i`** (L241). Right
conclusion, three reasons, one of which is wrong.

- *"Requires a new encoder for a new `N` (backprop)"* — correct and by itself sufficient. It is a
  constraint violation, full stop.
- *"is unlabeled (weakens the sparse-supervision thesis rather than strengthening it)"* — **this reason
  is wrong.** A zero-label method that worked would strengthen our position, not weaken it: it moves us
  strictly further along the supervision axis, onto the same footing as the unlabelled alignment family
  (`degenhart2020`, `nomad2025`) while keeping the no-backprop property they lack. The thesis is about
  what deployment costs, not about consuming labels for their own sake. Left uncorrected, this reasoning
  would also kill my S6, which is the highest-upside idea on either list.
- *"duplicates the activity-only token"* — **this is the good reason and it is under-argued.**
  CEBRA-Time's positives are temporal adjacency on the calibration window, and `f_id(mean f_pre(X_i))`
  already summarizes exactly that window. The citable evidence is B2-D1024: scaling the activity-only
  path to 1024 units reaches 0.1451 on RT against the carrier's 0.4419. If a CEBRA-Time embedding is in
  the same information class, B2-D1024 is the measured ceiling. That argument should have carried the
  rejection on its own.

**Rejection 2 — posedness.** Correct and correctly grounded in the brief's explicit prohibition.

**Rejection 3 — cross-session unit matching (Hungarian on embedding R²).** Correct and well-reasoned:
the architecture is permutation-invariant *because* it refuses correspondence, and matching would
silently reinstate a channel table. The best rejection in their document.

**Rejection 4 — kNN-on-CEBRA at online decode.** Correct; changes the method family and fights the
latency claim.

**Rejection 5 — sub-event splitting.** Defensible conclusion, misattributed evidence. They cite the
K-point screen's failure (best `+0.0019` against a `+0.010` gate) as if it settled sub-event splitting.
It does not: §12.5 explicitly distinguishes the two, K-point adding *feature dimensions* while splitting
adds *observations*, and calls splitting "worth a cheap CPU screen". Their actual argument — "InfoNCE
along a straight chord does not create directional diversity", arc/chord 1.023 — is correct and is the
same risk §12.5 names. So: right worry, wrong citation. I rejected it too, for the different and I think
cleaner reason that it is already proposed in §12.5 and re-proposing it is duplication.

**Rejection 6 — concatenate a source-trained CEBRA neural embedding into `c_i`.** Correct, and the
"widening `f_pre` is the B2 path the carrier is supposed to beat" argument is the right one. This is the
argument that should have been used in rejection 1.

---

## 10. Updated ranking

My ranking **changes**. Four things moved it: their §0 decomposition, my §2.2 finding that gating to
zero would have hurt date 2, my §2.3 finding that the content-isolating control also fails on date 2,
and the checkpoint measurement in §8.1.

| # | Idea | Change | Reason |
|---|---|---|---|
| 1 | **Estimator-vs-consumer decision screen** (§6.3) | **new, promoted to first** | C1's framing, my instrument. Answers the question that splits both idea lists, at hours of CPU with no checkpoints. |
| 2 | **S1 — carrier rebalancing, primarily drop/centre the intercept** | held, sharpened | Premise and structural asymmetry now verified (§8.1, §8.2). Weight-decay objection dead; growth-budget argument empirically supported. Must run as a sweep, not a fix. |
| 3 | **Their Idea 1 + cross-session alignment fix** | **new entry, above my S3** | Best motivation in either document; preserves pooling linearity where S3 does not. Enters only with the identifiability metric attached. |
| 4 | **S5 + their Idea 2, merged: per-unit EB shrinkage toward a learned prior** | S5 promoted, S2 folded in | §4. This is what makes any gate work at all, since the zero target is refuted. |
| 5 | **S2 demoted and reframed as a reporting rule, not a deployment intervention** | **demoted** | §2.2. It cannot recover Zero5's level without shipping a second model. It survives as the paper's missing boundary statistic, which is still valuable. |
| 6 | **S6 — behaviour-free pseudo-label carrier** | held | Unaffected by anything this round. Highest upside, CPU-only pre-screen, no checkpoint needed. |
| 7 | **S3 — latent-anchored carrier** | **demoted below their Idea 1** | Same goal, and it breaks pooling covariance because the PCA basis is estimated from the neural data. |
| 8 | **S7 merged with their Idea 7; their Idea 8 (`q_c` basis) attached** | merged, Idea 8 added | One "choose a better basis" slot with three criteria: neural predictability (mine), similarity geometry (theirs), consumer compatibility (their Idea 8, which I rate above both). |
| 9 | **S4 / their Idea 4 unit-variant** | held, still low | My split-half ≈ 0.03 undermines both: positives defined by carrier similarity are positives defined by noise. |
| 10 | **Their Idea 6, WLS soft positives** | **new entry, low** | Cheap, pooling-safe, sharp kill criterion. Likely fights the sample-count result. |
| 11 | **S8 — activity standardization** | held, lowest | Still no evidence of a live scale problem across sessions. |
| — | **S9 / their Idea 9** | unchanged | Constraint-violating, Track B's, not ours. |

**Does my recommended first experiment change? Yes.** I no longer recommend running the
carrier-coordinate-imbalance diagnostic first. That diagnostic is largely *done* — §8.1 answered it —
and more importantly C1 is right that the estimator-versus-consumer question should be settled before
committing to either half of the list. I now recommend the §6.3 screen first, and S1's rescaling sweep
second, conditional on it.

---

## 11. What I still could not verify

1. **The recording ordering in `−0.042554 / +0.001088 / +0.027104`.** My strongest new finding (§6.3)
   depends on it. The receipt is not on disk; I assumed canonical `H1_HELDIN_SESSIONS` order.
2. **H-SE5 input scales at the checkpoint.** §8.1's weight measurement is from an **H-C** checkpoint. I
   measured learned weights but not H-C's carrier input RMS, so I cannot form the `w·x` product ratio,
   and I am transferring a conclusion across arms.
3. **Whether the date-1 `+0.011442` same-checkpoint-zero cost transfers to date 2.** §2.2's conclusion
   is directionally strong but the date-2 number is not measured.
4. **C1's "six-minute cross-recording transfer" (`0.519024` vs `0.516518`, their L146).** Not located.
5. **C1's "~2,300–2,500 eval-bins/session at M3".** Consistent with the M4 `~3,100 eligible rows`, but I
   did not open `h1_sample_complexity_audit/audit.json` myself.
6. **Training step count for the H-SE5/H-C runs.** §8.1's growth-budget argument would be much stronger
   with the actual optimizer step count; I inferred "~240 steps' worth" from the weight magnitude and
   `lr`, which assumes near-perfectly consistent gradient sign and is therefore a lower bound on steps.
7. **Whether the temporal autocorrelation objection to Idea 1 (§2.4) is quantitatively fatal.** I argued
   the effective sample size is far below the bin count but did not estimate the autocorrelation time.
8. **All of my round-1 diagnostics remain unreceipted throwaways**, as stated there. The only provenance
   claim is the 13/13 basis/carrier SHA match against `source_audit_v2r2.json`.
