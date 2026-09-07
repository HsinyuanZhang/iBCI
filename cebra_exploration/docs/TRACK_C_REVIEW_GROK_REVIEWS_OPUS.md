# Track C review — Grok (C1) reviews Opus (C2)

**Reviewed:** `cebra_exploration/docs/TRACK_C_BRAINSTORM_OPUS.md`
**Against:** my brainstorm `TRACK_C_BRAINSTORM_GROK.md`, coordinator-verified F2b/F3/F7/F8 in `COORDINATOR_VERIFIED_FINDINGS.md`, and the code they said they did not read.
**Rule:** I did not edit either brainstorm. Ranking revisions live only here.

---

## 1. Factual errors

### 1.1 The provenance gate does not cover the load-bearing numbers

They SHA-matched `basis_sha256` and `carrier_sha256` against `results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json` after calling `h1_sparse_event_endpoint_v2.fit_source_all_event_basis` / `fit_session_range(start_index=0, budget=4)` on 13 held-in sessions (Opus §0, lines 24–29). That is a real check. It establishes **only** that the V2 objects at `start_index=0` are the sealed estimator.

It does **not** establish:

- The §0.1 RMS table (a derived statistic, 11 sessions, proxy `s_src=1.0804`).
- The §0.2 split-half cosines or Spearman ρ = 0.808. Split-half is not in `source_audit_v2r2.json`.
- That §0.2 was computed on **V2**. They cite `h1_sparse_event_endpoint.coefficient_split_stability` (Opus S2, line 167). That function lives in the **V1** module (`sua_exploration/mc_maze/h1_sparse_event_endpoint.py:592-618`), which fits a **4-wide, q=3, λ=0.1** carrier (`CARRIER_DIM=4`, `LATENT_DIM=3`, `RIDGE_LAMBDA=0.1` at lines 27–31 of the same file). Deployed H-SE5 is V2: **5-wide, q=4, λ=3.0** (`h1_sparse_event_endpoint_v2.py:14-16`). V1 `fit_carrier_from_arrays` will refuse a V2 `[E,4]` latent (`got {z.shape}` at v1 line 493).

They *describe* a V2 split-half (“~10 events against **5** parameters”, Opus §0.2 line 74) but *cite* the V1 function. Those cannot both be true of the same call. I cannot tell which they ran. Treat §0.2 as **unreplicated** until it is recomputed with `v2.fit_carrier_arrays` and correlated against the **V2** `median_delta_intercept` already in the audit. Mixing V1 stability with V2 transfer would manufacture ρ.

The 11-session RMS set is at least the right *population*: `H1_M4_FOLD0_SOURCE = H1_HELDIN_SESSIONS[2:]` is 11 sessions (`h1_m4_eb_pilot.py:58`), i.e. fold-0 source, which is what `ψ_c` sees. That part is coherent. The proxy `s_src` does not affect **ratios**, as they say, because `SparseScalarNormalizer` is a single scalar (`h1_sparse_event_endpoint.py:110-113`). Absolute RMS values remain approximate.

### 1.2 “The intercept is identical for every channel and carries no per-channel identity”

False on their own table. Opus §0.1: intercept mean 2.1051, **std 1.1840**. A coordinate with std 1.18 is not a shared DC offset. It is per-channel `log1p` baseline rate (the V2 intercept is unpenalized OLS of `log_rates` on a column of ones; `fit_carrier_arrays` at v2 lines 120–125, intercept first in the solve, then moved last). That *is* identity content — the kind the paper’s B4 arm showed is **worse than zero** on center-out. The right claim is: intercept variation is ~30–60× the slope variation **and** it is the wrong identity. Dropping “identical / no identity” matters, because S1(b) then has a lattice-shaped justification rather than a “this coordinate is blank” justification.

### 1.3 CEBRA multi-session “session-specific input layers” (S8, S9)

Stale relative to F2b/F3/F8, and they did not read `cebra/distributions/` (their own unverified item 11).

- **S8** (lines 463–475): “the closest structural counterpart to CEBRA’s multi-session input layers.” Multi-session CEBRA shares **zero** weights (F2b). The input-layer pattern exists only on single-session `adapt=True` (F3): first layer re-init, trunk frozen, 20% of params. Per-channel moment matching is not that object. CEBRA’s first layer is `Linear(N_session → hidden)` mixing units; S8 is a per-channel affine on `x_i`.
- **S9** (lines 502–508): “exactly as CEBRA does.” F8: `adapt=True` with a single-session InfoNCE **does not land in the source latent** (synthetic target R² −1.23 vs joint-fit +0.99). Alignment is a property of **cross-session positive sampling**, not of freezing a trunk. S9 never specifies the loss or the positive pair. As written it is the broken Track B arm.

They correctly parked S9 in Track B. The mechanism sentence is still wrong, and F8 makes it harmful if anyone implements it.

### 1.4 Consistency rejection overreaches (R5)

They correctly read that `_consistency_datasets` requires labels and rejects `ndim>1` (`metrics.py:350-352`). F7 agrees. They then reject the *idea* “as described in the brief” (R5, lines 556–565). F7: this is an API limit, not a conceptual one. A 7-DoF reimplementation with an explicit correspondence rule stays live. Folding “token-cloud second moments `EᵀE/N`” into S2 is a different, weaker statistic (no correspondence, so it cannot answer “did the latent stay put”).

### 1.5 H-C “has no intercept” — inference is right, citation is thin

They inferred it from paper eq. (15) `C_raw ← B_slope` and `d_c = q+1 = 4` (unverified item 5). Confirmed without their missing code read: compact H-C is `36→32` (32 activity + 4 carrier) in `paper_6pp.tex`; H-SE5 is `Linear(HIDDEN_DIM+CARRIER_DIM, …)` with `CARRIER_DIM=5` in `h1_sparse_event_spint.py:43-46`; the older residual consumer is `[N,4]` (`h1_m4_eb_normalized_v2_residual_spint.py:39-43`). Load-bearing for S1(b) and actually true. Still confounded: H-C also uses dense 7-DoF and shrinkage, so “works because no intercept” is not identified.

### 1.6 Date-2 = `19250108`

They flagged this as unverified (item 6). It is correct: master handoff and `h1_hse5_lodo_date2_terminal_evaluate.py`. Not an error.

---

## 2. Headline claim (S1) — measurement vs inference vs Adam

### Measurement

Likely **directionally correct** for V2 H-SE5, even though I did not recompute the table.

- Stored layout is `[w1,w2,w3,w4,b]` (`fit_carrier_arrays` returns `column_stack((coefficient[1:].T, coefficient[0]))`).
- One global `s_src = sqrt(mean(source_cache²))` (`SparseScalarNormalizer`, lines 101–113 of `h1_sparse_event_endpoint.py`). Ratios survive any scalar.
- Intercept is **unpenalized**; slopes get `λ=3` on source-standardized `z` (`penalty = diag([0]+[1]*4)*(n*3)`, v2 line 122). `transform` divides by `score_scale`, so `ZᵀZ/n ≈ I` on source-like events and slopes are shrunk by ~¼. The ridge **widens** the injection imbalance; it does not cause it alone (`log1p` rate vs standardized displacement scores would already differ).

I did not rerun their 11-session RMS. The 32–61× figure is theirs, uncitable, and should be regenerated from the sealed cache with the deployed `s_src`, not a proxy. I would be surprised if the order of magnitude moved.

### `ψ_c` — they did not read it; I did

`H1SparseEventSpint.carrierid_identity_projection` (`h1_sparse_event_spint.py:58-67`):

```
pooled = Linear(1024→32)+ReLU, mean over trials          # 32-D activity
cat = [pooled, carrier]                                   # 37-D, no LayerNorm
E = Linear(37→32)+ReLU → Linear(32→32)+ReLU → Linear(32→700)
carrier columns of the first affine are zero-initialized
```

No input normalization sits between `c_i` and the first affine. Concatenation is raw. That is the path their Adam objection “turns on,” and it does **not** rescue the input imbalance by architecture.

Optimizer for this module: **Adam, `weight_decay: 0.0`, `lr: 5e-5`** (`configs/model/falcon_h1_sparse_event_endpoint.yaml:8-12`). Their leftover “it still hurts through weight decay” (S1, line 143) is **false for H-SE5**. The live remainder is Adam’s per-column second moment on the linear map, plus ReLU mixing with the 32-D activity features.

### Does Adam defeat it?

**Partly the linear map, not the concat/ReLU.** For a single column, `∂L/∂W_j ∝ c_j`, so Adam’s v-term scales as `c_j²` and `W_j · c_j` is approximately scale-invariant. That is exactly the checkpoint test they proposed (`|W[:,j]| · RMS(c_j)`, S1 lines 129–136) and **did not run**. Until that number exists, “the intercept dominates the token” is an input-space claim, not a token-space claim.

What Adam does **not** cancel:

1. **32 activity dims vs 5 carrier dims** into 32 hidden units, with carrier columns starting at 0 and the activity path live from step 1. Tuning coordinates at RMS ~0.02 are small against both the intercept **and** `pooled`.
2. **ReLU** after the mixed pre-activation. A large intercept column can gate hidden units that the tiny slope columns never recover.
3. **Zero-init of carrier columns** is a stated contract (`h1_sparse_event_module.py:29-30`). The network must *learn* to read `c_i`; a 60× quiet coordinate is a 60× quieter learning signal at the start, even if Adam later rescales.

So: the measurement is probably real; the inference that this *is* the H1 sparse failure is **not yet licensed**; their own Adam objection is the right objection and is **stronger than they think on weight decay (there is none) and weaker than they think on concat/ReLU**. S1 remains the correct first *probe*, not yet a diagnosed cause.

S1(c) spherical factorization is a real CEBRA mechanic (`FixedCosineInfoNCE` on L2-normalized latents). S1(a)/(b) are plumbing. Ranking (a) first is right.

---

## 3. Constraint violations

S1–S8: no target-session backward pass. S6 honestly adds a second forward pass (lines 389–391). S9 is labelled. Good.

Soft spots, not violations:

- **S3** builds `L` from neural PCA of the target prefix. Closed-form, but `X` now depends on `r`, so pooling linearity (`paper_6pp.tex` eq. pooling) dies. They do not say so. My Idea 1 treated neural graphs as a kill, not a variant.
- **S6** is still no-backprop. The protocol sentence in the paper changes. They declared it.
- **S4** secondary pair (“channels in different sessions whose carriers are close”, line 293) is the F8-shaped pair; the **primary** pair is within-session same-channel different windows. That is SimCLR on calibration noise, not CEBRA multi-session alignment. They should swap the order after F8.

---

## 4. Cost claims

| Claim | Verdict |
|---|---|
| S1 checkpoint `|W|·RMS` / finite differences, ~1 hour CPU | **Yes.** One frozen `carrier_post_pool[0].weight` and the sealed normalized cache. This is the actual cheapest decisive test on their list. |
| S1 follow-up 2 GPU runs | Yes if the checkpoint confirms. Do not GPU until it does. |
| S2 “forward-only, no training” on fold-0 and date-2 | Decode of ~9k + ~13k windows through a 58k identity path is CPU-plausible. **Protocol is wrong:** gating `c_i→0` on a Full-trained net is *same-checkpoint zero*, not independently trained Zero5. Handoff: same-checkpoint controls understate content dependence ~4×. Their confirm clause (“deltas against Zero5 ≥ 0”, lines 191–194) mixes those. |
| S3 “a day of CPU, not an hour” | Honest. `EventSession` is event-only. |
| S4 pre-screen “minutes” | Cheap **if** restricted to non-overlapping windows. `legal_contiguous_starts` is stride-1 (`h1_m4_eb_pilot.py:772-783`). Their unverified item 7 is now resolved: the cache **is** overlapping; nearby-start cosines will be spuriously high. |
| S5/S7 LODO `delta_intercept` | CPU, yes. |
| S6 noise screen “no checkpoint” | Yes, and correctly labelled optimistic (correlated pseudo-label error, lines 408–410). |
| S8 correlating moment shift with `delta_intercept` | They themselves call it a weak proxy (lines 489–492) because `delta_intercept` never touches the token. The cheap test cannot decide the idea. |

They are more honest about estimator-vs-decoder transfer (`n=1` calibration point, unverified item 12) than most of this project. That item is the best infrastructure request in either document.

---

## 5. Genuine disagreements (not manufactured)

**D1. What the date flip is.** I read Full 0.500→0.496 and Zero5 0.472→0.518 as a volatile activity-only baseline under source-date LOSO. They read split-half cosine ≈ 0.03 as the mechanism of `+0.0285 → −0.0226` (Opus §0.2 lines 77–78). Their ρ = 0.81 is against **estimator** `median_delta_intercept` on 13 sessions, not against the two decoder dates. They cannot have shown that split-half predicts the *sign flip*. I cannot have shown that Zero5’s jump is “luck” rather than Full-training poison. **Leaving this open.** A decoder-level table of Full, same-checkpoint-zero, and independent Zero5 on both dates would settle it; neither of us has it.

**D2. Neural structure inside the carrier.** Their S3 *wants* population PCA in `L`. I treated neural-dependent `φ` as a kill because it leaks the activity-only token and breaks pooling linearity. This is a real design disagreement. A CPU screen with (i) label-shuffle of `A` only, (ii) time-or-label graph `φ` with neural-independent `X` (my Idea 1), (iii) `C=LA`, scored on `delta_intercept` **and** on a pooling-additivity residual, is the right court. I am not conceding S3 as an improvement to *this* method until (i) falls as far as H-SE5 label-shuffle.

**D3. Consistency.** They reject it (R5). F7 plus my Idea 5 keep a reimplemented 7-DoF version. Token-cloud `EᵀE/N` is not a substitute. I am not dropping this.

I am **not** disagreeing that intercept/slope scale imbalance is real and unintended. I missed it. That is their win.

---

## 6. Overlap, misses, and which of theirs is better

### Real overlap (do not double-count)

| Theme | Mine | Theirs | Notes |
|---|---|---|---|
| Calibration abstention / GoF gate | Idea 2 | **S2** | Same idea. Theirs is better specified: split-half already exists, ρ measured (caveat §1.1), paper-claim shape “never hurts, sometimes abstains” is sharper. Fix the Zero5 vs same-checkpoint protocol. |
| Source-only InfoNCE on tokens | Idea 4 | **S4** | Same family. After F8, *their* secondary pair (cross-session similar `c_i`) is the CEBRA pair; *my* time-point/behavior pair across sessions is the other legal one. Their primary same-channel/window pair is multi-view invariance, downstream of carrier quality, as they said. |
| Unused calibration bins | Idea 1 (time+event Laplacian `φ`) | **S3** (`L` from PCA, shared `A`) | Related, not the same. S3 shares the noisy rotation; Idea 1 densifies observations with a neural-independent `X`. |
| Source-frozen auxiliary reduction 7→4 | Idea 7 | **S7** (PLS/RRR) | Same slot. **S7 is better:** CPU, LODO, same `q`, no CEBRA training. My Isomap/CEBRA-on-behaviour precursor was overbuilt. |
| Constraint-violating target encoder | Idea 9 | S9 | Both parked in Track B. F8 now says neither mechanism as written works. |
| Consistency diagnostic | Idea 5 | R5 (rejected) | Not overlap; conflict. See D3. |

### Theirs that I missed (real)

1. **S1 — rebalance / drop intercept / spherical factor.** Best miss. Cheap, named-weakness-shaped, and the lattice (AC4 vs B4) already predicts the intercept is the wrong dominant coordinate. I had nothing on injection scale.
2. **S5 — activity→carrier EB prior.** Upgrades H-C’s global `μ_C` to a per-channel, label-free `ĥ(a_i)`. Structurally “never worse than Zero5” as `ν_i` grows. I did not have this. Kill criterion (`ĥ` alone `delta_intercept ≤ 0` under date-LODO) is clean. The center-out +0.090 phase-poor number may not transfer to H1; they said so.
3. **S6 — zero-carrier pseudo-labels as the auxiliary variable.** Highest-upside long shot I missed. Their optimistic-bound caveat on the noise screen is exactly right. The reviewer question “if Zero5 already predicts the movement, what is the carrier adding?” is especially sharp given date-2 Zero5 at 0.518.

### Mine they missed

- Hybrid **time+event** spectral `φ` with a hard ban on neural edges (Idea 1). Their S3 uses the bins but spends them on neural PCA.
- F8-shaped requirement that **cross-session positives** are what align, stated as a design constraint rather than a comparator fact.
- Kernel-mean / Nyström carrier (Idea 3) and DeltaNormal WLS (Idea 6). Their R1 kills the *linear* `Φᵀr` version; it does not kill a kernel on event similarity. Low priority after their §0.2.

### Which of theirs is better than one of mine

**S2 > my Idea 2.** Same gate, they already measured a ranking statistic (if V2-replicated). **S7 > my Idea 7.** **S1 has no counterpart in mine** and would have been in my top three if I had seen the normalizer.

S3 is *better aimed at their §0.2 diagnosis* than my Idea 1, and *worse aimed at the paper’s pooling claim*. I am not swapping Idea 1 out until D2 is empirical.

---

## 7. Their rejections — especially R1 de-whitening

**R1.** The algebra is roughly right and the conclusion stands: do not headline `Φᵀr`. V2 does

```
(S + λI) ŵ = Zᵀy / n ,  λ = 3 ,  S ≈ I  on source-scaled z
⇒ ŵ ≈ (1/4) Zᵀy / n
```

(`fit_carrier_arrays` lines 120–125; `transform` line 38). The ¼ is absorbed by `ψ_c`. A λ-sweep cell is the residue, not an idea.

Two nits, neither of which revives it: (i) `S≈I` is a source-distribution statement; a target session with 20 events can deviate. (ii) The intercept is **outside** that shrinkage (`diag(0,I)`). R1’s “already ~90% implemented” applies to the **slopes**. Combined with S1, the unpenalized intercept is a cause of the injection imbalance. They did not draw that arrow. Still: do not re-propose de-whitening as a headline.

**R2, R3, R4, R6, R8.** Correct, and they saved the coordinating agent time. R3 (max-spread) and R4 (K-point) are closed by existing CPU screens. R6 is Track B plus F8. R8 (raise `q`) fights obs/param.

**R5.** API reading correct; idea-kill too broad (F7). See §1.4.

**R7.** Between-session scale drift 1.3× vs within-vector 30–60× is a good argument that S1 is the live scale problem and session-wise renormalization is not. Independent of whether Adam later cancels S1.

---

## 8. Ranking update

The four facts and their document **do change my ranking.** I am not converging for politeness: S1 is a real miss, F8 rewrites what “borrow alignment” means, and S2 is a better gate than mine.

**New order I would run (not an edit of my file):**

1. **S1 checkpoint probe, then (a)/(b) only if it fails to self-kill.** Highest information per hour. I can now state that there is no LayerNorm, `weight_decay=0`, and ReLU-after-concat, so the probe is well-posed.
2. **S2 abstention, on a V2-replicated split-half, with same-checkpoint-zero as the gated baseline.** Best paper-shaped sparse-boundary claim if the statistic separates the four recordings.
3. **Unused-bin estimators, two arms in one CPU screen:** S3 (`C=LA`) **and** my Idea 1 (neural-independent time+event Laplacian). Label-shuffle and pooling-additivity as mandatory controls. Lets D2 die empirically.
4. **S7 PLS basis.** Nearly free; a negative diagnoses S3.
5. **S5 EB activity prior.** After S7, same LODO harness.
6. **S4 / my Idea 4 InfoNCE, only after F8-legal positives exist and after S1/S3 have made same-channel cosine distinguishable.** Their own ranking of S4 as downstream of carrier quality still holds; F8 additionally requires a cross-session pair.
7. **S6 pseudo-label screen.** High upside, optimistic CPU bound; run when the estimator is no longer the bottleneck.
8. **Reimplemented 7-DoF consistency (my Idea 5, F7).** Diagnostic, not a method. Budget the correspondence rule.
9. **S9 / my Idea 9.** Track B, and only a joint-fit or frozen-source-plus-cross-session-sampling variant (F8). Not `adapt=True` as published.

**Demoted / dropped from my original top:** Idea 8 (closed-form input layer) — F2b, there is no trunk. Idea 6 WLS — likely fights the budget sweep. Their S8 — wrong CEBRA analogue.

**First experiment: changed.** I previously wanted Idea 2/GoF first because of the Zero5 jump. I now want **S1’s `|W[:,j]|·RMS(c_j)` / finite-difference token-variance probe** first. Reasons: (i) it is strictly cheaper than any gate that needs decoder forwards; (ii) `ψ_c` inspection shows the imbalance is architecturally unblocked; (iii) if intercept dominates `E_i`, that is a B4-shaped poison that can explain Full stuck at ~0.50 *and* Full < Zero5 without invoking Zero5 luck; (iv) if the probe refutes (tuning columns already ≥40% of token variance), S1 dies in an hour and S2 becomes first again. I am not promoting S1 to “the H1 fix” until that probe exists.

---

## 9. What I am not taking from them

- §0.2 as a citable mechanism of the date flip (provenance + estimator-level ρ + V1/V2 citation clash).
- S3 as obviously pooling-safe.
- S8 as the CEBRA input-layer analogue.
- Any “CEBRA aligns sessions” sentence that does not name the cross-session positive pair (F8).
