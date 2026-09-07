# Results: behavior_manifold_v2 — E1..E7 (2026-08-25)

Status: complete. Receipts: `results/behavior_manifold_v2/` (10 files, 0444+sidecar, all
`sha256sum -c` OK). Frozen artifacts bit-unchanged; zero target backward/optimizer; formal
not opened; oracle cells labelled leakage-diagnostic. Tests 16/16 new + 8/8 existing green.
Verification anchors: M=10 reproduces the frozen receipts bit-exactly (prediction SHA);
DirectRidge baseline exact (0.3623304528); one real bug caught by math tests (LOO/GCV hat
diagonal must use the intercept-aware smoother; fixed and verified against brute force).

## Headline table

| cell | value | reading |
|---|---:|---|
| E1 oracle-q8 ceiling | **0.477283** | ≤ 0.50 → **representational gap**; aux-head justified, zero-training upgrades dead |
| E1 cross-cell | q8 M10-deployable 0.4342 > DirectRidge full-oracle 0.4218 | the bottleneck is manifold capacity, not the estimator |
| E2 budget sweep | +0.0746 / +0.0718 / +0.0614 at M4/M10/M30 (4/4 each) | variance-limited regime; M-specificity worry resolved positively |
| E3 participation ratio | 3.94 (full) / 3.79 (M10) / 3.93 (query) | synergy band; q=8 comfortably above |
| E4 loading attribution | r = 0.6665, p = 0.0048 (n=16) | gains concentrate in well-reconstructed outputs; PECmaj 2nd-lowest recon — confirmed |
| E5 estimator upgrades | GCV −0.011 (harmful); pooled λ +0.000; EB +0.0001 | nothing closes the M10→oracle gap — consistent with E1 |
| E6 PCA-8 projection | −0.027642 vs manifold −0.021957 | ≈ equal → projection loss is a **generic 8-dim effect**; the 91%-on-manifold claim downgraded to on-any-8-dim-basis |
| E7 supervised-mismatched control | **+0.064777 = 90% of the +0.0718 gain** (4/4) | **supervised nonlinear reparameterization is the dominant ingredient; deployment-matching adds ~+0.007 residual** |

## Corrected attribution of the M1 closed-form gain

The handoff claim "the deployment-matched objective is the active ingredient" is **weakened
by E7**: a supervised objective without the deployment rule in the loop recovers 90% of the
gain. Corrected decomposition of +0.0718:
- ~+0.065 (90%) supervised nonlinear reparameterization (generic);
- ~+0.007 (10%) deployment-matching (the manifold trained through the M10 ridge rule);
- 0% dimensionality alone (PCA null, reconstruction-MLP negative — unchanged);
- E4 keeps a genuine output-level mechanism signature (gains follow per-output
  reconstruction quality), but E6 removes projection-level specificity.

Narrative consequence: the honest name is **supervised low-dimensional reparameterization
with a small deployment-matching bonus**, not "behavior manifold discovery". The E4 loading
structure and E3 synergy-band PR remain true geometry facts worth reporting.

## What the numbers now say about the route

1. The closed-form q8 family is **capacity-limited at 0.477** (E1/E5): no estimator-side fix
   reaches the full decoder (0.670). The remaining 0.19 gap is representational.
2. The gain is **robust** (E2 all budgets 4/4; q4-q12 flat; 94% route vs 6% ensemble) —
   a deployable improvement for the M1 closed-form constraint regime.
3. The aux-head Stage-2 experiment (dual-head SPINT regularizer) keeps its E1 justification
   (representational gap), BUT E7 lowers its manifold-specific prior further: the correct
   prediction from E7 is that a raw-16D aux head (control arm B) may match the manifold aux
   head (arm C). The B-arm is therefore mandatory, not optional, and the pre-registered
   mechanism claim must be "aux-supervised reparameterization", with manifold-specificity as
   the falsifiable secondary.

## Supersedes / corrects

- `HANDOFF_M1_CALIBRATION_AWARE_BEHAVIOR_BOTTLENECK_20260824.md` §3's causal sentence and
  §5's aux-head rationale (corrected above); numbers in §2/§4 stand unchanged.
- The "91% on-manifold" decomposition circulated on 2026-08-24 (computed from the three-fold
  receipt): downgraded by E6 to a generic 8-dimensional-basis effect.

## Receipts index (results/behavior_manifold_v2/)
e1_oracle_latent.json · e2_budget_sweep.json · e3_participation.json ·
e4_output_attribution.json · e5_estimator_upgrades.json · e6_pca_projection.json ·
e7_supervised_control.json (+ selection/meta receipts; all with .sha256 sidecars).

## 2026-08-25 addendum: probe + E8 — the aux-head question closed at both ends

**Probe (representation-manifold alignment)**: ridge probe from the frozen fold decoders'
penultimate representation (the unique fc_out input, documented in-receipt) to {q8 latent
(3 seeds), raw 16-D, PCA-8}. Equal-cell R²: latent **0.7483** > raw **0.6517** (PCA-8
0.6415) — ratio **1.148**, all 9 cells 1.11–1.22, λ-robust. The manifold codes are MORE
linearly present in the representation than raw behavior (denoising code), manifold-specific
(PCA-8 shows no surplus). Pre-registered kill threshold (≥0.9×) hit decisively →
**aux-head training cell cancelled**: its gradient signal is redundant with the main loss.

**E8 (deployment-legal alternative readout)**: rep → M10-support probe(z) → frozen dec →
behavior, vs the decoder's own raw head on the published replay query:
ensemble **0.5637** vs raw head **0.6702** (Δ −0.1065) → **READOUT_ADDS_NOTHING**.
Decisive diagnostics: probe-raw-decode 0.5487 ≈ probe-latent-decode 0.5637 (manifold
contributes only ~+0.015; the deficit is generic M10-support linear re-readout, ~1,300
bins vs 16,384 features); trivial mix 0.6687 ≈ raw head (no complementary signal). The
probe's latent-space surplus does not transfer to a behavior operating point.

**Route closure**: the M1 behavior-manifold line is experimentally complete — E1 ceiling /
E2 budgets / E3 PR / E4 attribution / E5 estimators-dead / E6 generic-8-dim / E7 supervised-
reparam-dominant / probe structural / E8 readout-dead. Remaining work is claims-side only
(held-out scoring of the frozen q8 ensemble when authorized; paper packaging with the
corrected attribution and the cross-dataset synthesis: label-supervised low-dimensional
reparameterizations transfer; unsupervised ones do not).

Receipts: `probe_alignment.json` (digest e104b15c…), `e8_readout.json` (digest 7444108f…);
tests 34/34.
