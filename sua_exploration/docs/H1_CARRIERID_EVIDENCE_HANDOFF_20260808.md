# H1 CarrierID: Evidence Gaps and Next Experiments — Analysis Handoff

**Status:** analysis plus sealed development evidence. The fold0 same-checkpoint
dose-response item is complete; this handoff authorizes no additional run. Every
remaining item below still needs the normal pre-registration and review chain before launch.
**Date:** 2026-08-08
**Scope:** H1 CarrierID date-LODO line (SPINT-main) and its relation to the RT/AFC4
component-attribution results.

## 1. The question

H1 is no longer "does CarrierID have any predictive gain." The open question is whether
the gain comes from the *correct kinematic content* of the carrier. The evidence has
three layers, and each layer has a different weakness.

## 2. Current evidence state

| Layer | Question | Evidence | Verdict |
|---|---|---|---|
| 1 | H-C better than matched SPINT H-S? | fold0 two seeds: `+0.02868/+0.02257` (inside the pre-registered borderline band). Cross-date LODO: `19250108 +0.11931` (3/3 recordings), `19250113 +0.08273`, `19250115 +0.08189`. Currently 3/5 dates, 7/7 recordings positive. | Credible positive trend, not sealed. Dates 19/20 pending. |
| 2 | Is the carrier pathway doing the work? | fold0 `H-C−H-C0 = +0.03890/+0.03965` (two seeds, range 0.00075). Same-checkpoint zero/row/label interventions drop `0.084–0.153`. | Yes on fold0. No cross-date H-C0 exists. |
| 3 | Is correct neural-kinematic pairing necessary? | fold0 `H-C−H-LS = +0.02562` pooled, per-recording `+0.05461/−0.05814`. The strong-null audit is sealed. In the new four-repeat same-checkpoint dose response, both label- and row-corruption pooled mean curves, and both recording-level mean curves, decrease monotonically from nominal `p=0` to `p=1`; pooled endpoint losses are `−0.11658` and `−0.15875`, respectively. | The trained checkpoint is dose-dependently sensitive to pairing and attachment. Training-time necessity remains a separate claim for the fresh H-LS controls. |

Related RT-side evidence (different dataset/carrier, already sealed or in flight):
`Full−MB4 = +0.2575`, 15/15 folds, exact sign `p=6.1e-5` (signed `[Wx,Wy]` is
load-bearing); `R-C−R-RS = +0.287`, 15/15 (channel attachment necessary); XLSv2 strong
cross-reach label permutation is 11/11 folds positive so far. The old within-reach
cyclic LS was confirmed a weak null (`+0.017`, `p=0.607`) — the reason null-strength
audits are now mandatory.

## 3. Diagnosis per layer (reviewer attack surfaces)

**Layer 1.** The trend is credible; the weaknesses are structural. (a) fold0 pooled R²
is carried by one recording holding 75.13% of the windows (per-recording
`+0.043/−0.014`). (b) All cross-date results are seed42 only. (c) Nothing is formal;
G1 (the dangling formal receipt) still blocks any formal claim.

**Layer 2.** This is the most stable contrast in the project (seed range 0.00075),
which means the carrier-pathway effect is the stable part and the MLP baseline is the
noisy comparator. But it exists only on fold0, with the same per-recording sign flip
(`+0.072/−0.056`). Same-checkpoint interventions prove the trained network reads the
carrier; they do not prove training-time necessity, because a frozen network cannot
compensate. Both evidence types need a second date.

**Layer 3.** The dangerous one. There is an exact decomposition:

```
H-C−H-C0 = (H-C−H-LS) + (H-LS−H-C0)
fold0:     +0.039  ≈  +0.0256     +  ~+0.013
```

So part of the pathway gain may be *pairing-free* per-unit structure (a misaligned
carrier still encodes unit-specific modulation). A reviewer can say: "the carrier is
just a per-unit gain/baseline feature; correct velocity pairing adds little." No
The fold0 dose-response now answers the narrower deployment-intervention question:
progressively corrupting correct pairing or attachment progressively hurts the already
trained checkpoint. It does **not** by itself answer training-time necessity, because
the checkpoint is frozen and cannot relearn around the intervention. The five-date H-LS
plus cross-date H-C0 still close that stronger question, because they give the full
(pairing, misfit-residual, pathway) decomposition per date.

## 4. Two observations that turn into strengths

1. **The cross-date effect (+0.08 to +0.12) is 3–4x the same-date fold0 effect
   (+0.026).** If dates 19/20 hold, the main claim should be reframed: CarrierID's
   advantage *grows under distribution shift*, which is exactly the deployment regime
   (new day, zero target backprop). fold0 becomes the lower bound, not the headline.
2. **H-C0 is slightly worse than H-S** (0.4866 vs 0.4968; 0.5013 vs 0.5184). The
   compact architecture alone costs ~0.01–0.017 R². The carrier does not just beat the
   MLP — it first pays back an architecture tax, then adds +0.039. Reporting this
   decomposition preempts "the small model is just better regularized."

## 5. Recommended evidence experiments (priority order)

**Tier 1 — already committed, execute cleanly.**
1. Finish dates 19/20. Report with the date as the inference unit (n=5): exact sign
   test (5/5 → p=1/32≈0.031) plus paired bootstrap CI. Report pooled *and*
   equal-recording aggregation, with the full per-recording table.
2. Run the pre-registered five-date H-LS. This is the only cross-date channel for
   Layer 3; it outranks any new design.

**Tier 2 — high value, low cost.**
3. **Cross-date H-C0** on at least 2–3 pre-declared LODO dates. Closes Layer 2 and,
   with Tier 1.2, completes the per-date pairing decomposition.
4. **Completed — same-checkpoint dose-response shuffle.** The sealed fold0 H-C
   epoch-49 checkpoint and exact `665fe535...` query pool were used for every arm. The
   registered nominal doses were `p={0,.25,.5,.75,1}` for label and row corruption,
   with four deterministic repeats. The current-runtime CPU `p=0` forward first
   reproduced the sealed CUDA metric within the pre-frozen `1e-6` absolute tolerance:
   pooled delta `+1.31e-8`, recording deltas `+1.85e-9/+4.55e-8`; checkpoint state,
   carrier hashes, query hash, and sample counts matched exactly. Only then were
   nonzero doses run.

   | corruption | nominal p=0 | p=.25 | p=.50 | p=.75 | p=1 | endpoint delta |
   |---|---:|---:|---:|---:|---:|---:|
   | label, pooled mean R2 | 0.525511 | 0.520479 | 0.501433 | 0.475824 | 0.408930 | -0.116580 |
   | row, pooled mean R2 | 0.525511 | 0.482594 | 0.457637 | 0.424436 | 0.366763 | -0.158748 |

   Both pooled mean curves are monotonically nonincreasing. The same is true separately
   for both recordings:

   | corruption / recording | nominal p=0 | p=.25 | p=.50 | p=.75 | p=1 | endpoint delta |
   |---|---:|---:|---:|---:|---:|---:|
   | label / `111740` | 0.598179 | 0.592472 | 0.567058 | 0.541226 | 0.459150 | -0.139029 |
   | label / `112404` | 0.314797 | 0.311711 | 0.311041 | 0.286038 | 0.262925 | -0.051873 |
   | row / `111740` | 0.598179 | 0.547914 | 0.525404 | 0.487553 | 0.418000 | -0.180180 |
   | row / `112404` | 0.314797 | 0.293054 | 0.260986 | 0.241179 | 0.217755 | -0.097042 |

   Repeat-level honesty matters: pooled label and row curves were fully monotone in
   `3/4` repeats each. The one label exception was the negligible seed-2026080801
   `p=0→.25` increase (`+9.38e-5` R2); the one row exception was seed-2026080802
   `p=.25→.50` (`+0.00270`). Per-recording fully monotone counts were label `4/4` and
   `1/4`, row `3/4` and `4/4`, for `111740/112404`, respectively. Thus the robust claim
   is monotonicity of the pre-registered four-repeat mean curves, not determinism of
   every random corruption realization. `p` is a nominal dose: row fractions are exact
   because `N=176`; label block counts are 627/660, so round-half-up gives at most
   `8.0e-4` absolute fraction error, recorded per shard as `actual_selected_fraction`.

   This result upgrades same-checkpoint sensitivity from a single binary contrast to a
   dose response, with row attachment showing the larger effect. It must **not** be
   rewritten as training-time necessity. Preserve immutable result
   `SPINT-main/pilot_artifacts/h1_carrierid_dose_response/H1_CARRIERID_HC_FOLD0_SAME_CHECKPOINT_DOSE_RESPONSE_RESULT_v2.json`
   (SHA-256 `e0382d0f89f43653b33ee51b2c3c3099ad8d468ccb992aefee4227d3f58cc1d3`).
   The superseded v1 preflight and its two early `p=.25` shards remain preserved but
   are explicitly excluded from v2; all included doses were rerun after the p=0 gate.
5. **Carrier-fit-quality moderator analysis**: per-session support-only OLS fit R² vs
   the H-C−H-S delta. If positive, the per-recording heterogeneity flips from a
   weakness into mechanism-consistent evidence (better support fit → better carrier →
   bigger gain). Support-only, no target access, pre-registerable now.

**Tier 3 — robustness and mechanism.**
6. Gauge-invariant post-training audit (rank/spectrum/subspace of learned U,
   deterministic carrier-output RMS) — required before any parameter-level mechanism
   sentence, per the EST4 identifiability audit.
7. One extra seed on 1–2 LODO dates to bound the seed×date interaction (fold0 seed SD
   is 0.0043, but cross-date has no seed check).
8. If budget remains: H-RS (row permutation, isolates channel attachment) or H-MB4
   (`[0,0,‖W‖,b]`, isolates signed `[Wx,Wy]`) arms on 1–2 dates — one arm each
   replicates the sealed RT 15/15 results on the H1 side.

**Governance.** G1 must be decided (options A/B/C in ROADMAP): the paper needs at
least one untouched formal/hidden evaluation. Everything above is development
evidence.

## 6. Architecture improvements

Current CarrierID structure
(`SPINT-main/src/models/components/h1_carrierid_spint.py:38`): per channel, calibration
activity `Linear(1024→32)+ReLU → mean over trials`, concat the 4-dim carrier, three-layer
MLP → 700-dim identity token, added additively to the neural window. The carrier is a
forward input; the target side has zero gradient.

Structural bottlenecks:

1. The carrier enters only additively at identity construction. Instantaneous activity
   `x_i` never interacts with the carrier. No multiplicative "modulation depth → readout
   weight" mechanism is expressible.
2. The carrier has no time structure. Static `[Wx,Wy,‖W‖,b]` folds each neuron's
   lead/lag into attenuated magnitude at lag 0.
3. The OLS fit runs on M=4/5 support trials — nearly underdetermined. Estimation noise
   flows straight into the identity. This is a plausible cause of the sign-flipped
   recordings.
4. h=32 is contract-locked and the learned projection has an exact scale/rotation
   gauge (EST4 audit), so capacity honesty and parameter-level attribution are both
   currently blocked.

Candidates, ranked by value/cost:

| # | Change | Hypothesis | Addresses | Cost |
|---|---|---|---|---|
| A | **Carrier-only identity arm** (activity path zeroed; completes the ±activity × ±carrier 2×2) | Decomposes what the identity actually uses | Layer 2 attribution | Trivial, 1 arm/date |
| B | **Zero-init multiplicative gain head**: `g: carrier(4)→700`, `src' = src·(1+g)+E` (~3.5K params) | Carrier modulates reading of instantaneous activity; a badly fit carrier can be down-weighted | Negative recordings | Low; needs gain-LS control |
| C | **Lag-aware carrier** (4→8–12 dims: W at lags −1/0/+1) | Lag is more stable across days than magnitude; recovers real modulation the static W attenuates | Effect-size ceiling | Medium; needs dim-matched LS control |
| D | **Support-period shrinkage** (empirical-Bayes/Wiener; population-level partial pooling, no unit correspondence needed) | At M=4/5, OLS variance dominates; shrinkage improves the carrier input directly | Small, heterogeneous fold0 effect | CPU-only audit, runnable now |
| E | **Gauge-fixed projection** (orthogonal parameterization / fixed whitening of U) | Makes parameter-level mechanism claims legal | Blocked attribution | Medium; only after B-C/L-C shows signal |
| F | **CI64 consumer-saturation** (already queued) | If CI64−CI32≈0, "58K is enough" becomes a selling point; if large, the headline is an underestimate | Efficiency-claim honesty | Already pre-registered; waits for the five-date aggregate |

Design notes:

- **A** is the cheapest Layer-2 strengthener. Current arms only have H-C (both paths)
  and H-C0 (carrier zeroed). Adding carrier-only closes the factorial: if carrier-only
  ≈ H-C, the activity pre-pool can be deleted (simpler model, stronger claim); if not,
  the interaction value is quantified. A mean+std trial-pooling variant is a cheap
  add-on.
- **B** must be explicitly separated from the failed SUA confidence-FiLM (that was
  confidence-conditioned, at M50, effect +0.003). Here the condition is the carrier
  content itself, zero-init keeps bitwise identity with H-C at init, and a gain-LS
  control (misaligned carrier into g) is mandatory. The expected benefit is not higher
  mean R² but **downside protection on badly fit recordings** — so run the moderator
  analysis (item 5) first; if negative recordings are not the badly fit ones, deprioritize B.
- **C** is the main v2 carrier-content candidate and aligns with the RT Full−MB4
  component attribution. Every added dimension triggers the project's own
  dimension/parameter-matched control rule; the LS arm must be padded to the same
  width.
- **D** is estimator-side, not network structure, but has the best ROI: a support-only
  CPU audit (no target, no GPU), with the SUA T4W3 train-only precedent (M15, 27/27
  sessions improved). Cross-session unit correspondence does not exist, so the prior is
  population-level: shrink direction toward 0, magnitude toward the population median.
  Fully deployment-legal.
- **E** matters only if the whole-pipeline B-C/L-C comparison shows a signal.
- **Decoupled cached-K decoder** stays out of the mainline: SUA v1 failed at −0.44.
  Follow the existing oracle chain's pre-registered branches only; cite it in a
  deployment-roadmap section.

Explicitly do **not** do: hidden-width sweeps as a claim (contract-forbidden; CI64 is a
control only), a confidence-FiLM redo (SUA negative), electrode lookup tables (charter
ban + gate already ineffective), larger attention / longer training (C15 killed),
K4/SSC/REL derivatives.

## 7. Execution order

1. **Now, no GPU needed:** D support-only shrinkage audit; A/B arm definitions with
   synthetic tests; pre-registration drafts for items 3–5.
2. **After the five-date aggregate seals:** A rides along with cross-date H-C0 (shared
   source training); B runs as a single-date pilot behind its gate; the now-complete
   dose-response (item 4) and future moderator (item 5) decide B's expansion.
3. **v2 stage:** C plus an H-MB4 arm as the carrier-content upgrade screen; F's result
   determines how the efficiency chapter is written.

If only one package is funded: **A + D**. Near-zero cost, and together they attack both
Layer-2 attribution and effect heterogeneity. **B** is the only architecture lever
aimed at the sign-flipped recordings. **C** is the only lever that can plausibly raise
the effect size itself.

## 8. Paper-framing notes

- State the supervised-calibration assumption explicitly (E3 lesson): the carrier comes
  from a labeled calibration block; still gradient-free and deployment-realistic, but a
  stronger assumption. Do not let it look hidden.
- Keep the null-strength methodology paragraph (weak RT cyclic LS → mandatory null
  audit → strong H1 rotation). Reviewers cite this kind of paragraph.
- Always pair R² with the efficiency numbers: 102.6x fewer identity parameters, 98.9x
  fewer identity MACs, zero target backprop. A small positive effect with this profile
  is already a result; do not oversell training-protocol robustness (D-S4e lost 0.0624
  R², 2.44x the fold0 headline — the fixed-e49/no-selection protocol is why the numbers
  are believable).

## 9. Key references

- `sua_exploration/docs/CURRENT_RESULTS.md` — 2026-08-08 entries: 21:47 (3/5 dates),
  19:36 (RT Full−MB4 15/15), 18:24 (five-date H-LS pre-registration), 17:05 (LS
  null-strength audit), 16:51 (EST4 gauge audit), 11:17 (first clean cross-date pair),
  10:45 (exposure-training diagnostic), 04:26 (fold0 seed43 three-arm table),
  2026-08-07 21:10 (fold0 seed42 table, parameter accounting).
- `SPINT-main/src/models/components/h1_carrierid_spint.py` — current CarrierID
  architecture.
- `sua_exploration/docs/T4_OPTIMIZATION_DIRECTIONS.md` — ranked candidate framework,
  kill criteria, and the killed-direction list this handoff respects.
- `sua_exploration/ROADMAP.md` — G1 formal-receipt decision options.
