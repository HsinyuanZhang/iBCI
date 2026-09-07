# Method Writeup: Anchored Post-Fusion Gate (APFG) — paper-ready material

Date: 2026-09-03
Sources: `tfpd_exploration/src/m2_anchored_postfusion_gate_v1/adapter.py` (verbatim math),
`RESULT_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md` (numbers),
`RESULT_M2_POSTFUSION_OPERATOR_CORRECTED_SCORE_V1_20260902.md` (jointly-trained PF comparison),
`DECISION_HOLD_APFG_STATIC_PROBE_PUSH_20260903.md` + `SUBMISSION_581713_APFG_STATIC_PROBE_20260903.md` (deployment).

## 1. Setup (notation for the paper)

A frozen, pretrained SNN decoder $f_\theta$ (coupled SPINT) maps a window of
binned spikes $x_t \in \mathbb{R}^{W \times N}$ ($W{=}50$ bins of 20 ms,
$N{=}96$ units) to finger velocity, conditioned on a per-session calibration
identity (memory) matrix $E \in \mathbb{R}^{N \times W}$ added to the decoder
source:

$$\hat{y}_t = \tfrac{1}{\gamma} f_\theta(x_t,\; E), \qquad \gamma = 5.$$

The identity is produced from the session's causal activity pool
$\mathcal{P} = \{X_1,\dots,X_M\}$ ($X_i \in \mathbb{R}^{100 \times N}$ trial
activity rows) and a per-neuron task-carrier side feature $s \in \mathbb{R}^{N\times 4}$
(fixed-ridge fit on $k{=}4$ D-opt-selected calibration trials; frozen).

Two frozen operators produce identities from $\mathcal{P}$:

- **Native (early-pool) operator** — the pretrained B3S encoder $g$: trials are
  pooled *inside* the learned encoder,
  $$E_{\text{nat}} = g(X_{1:M},\, s).$$
- **Post-fusion (late-pool) operator** — each trial is encoded independently
  and the resulting per-trial identities are averaged,
  $$E_{\text{pf}} = \frac{1}{M}\sum_{i=1}^{M} p\!\big(e(X_i),\, s\big),$$
  with $e$ the per-trial encoder and $p$ the frozen fusion head (arrival-order
  float32 accumulation, matching the trained post-fusion probe exactly).

## 2. The gate (one equation)

$$\boxed{\;E_{\alpha} \;=\; E_{\text{nat}} \;+\; \tanh(\alpha)\,\big(E_{\text{pf}} - E_{\text{nat}}\big)
\;=\; \big(1-\tanh\alpha\big)E_{\text{nat}} + \tanh(\alpha)\,E_{\text{pf}}\;}$$

- a single scalar $\alpha \in \mathbb{R}$ interpolates (for $\tanh\alpha \in [0,1]$)
  or mildly extrapolates (for $\tanh\alpha<0$) between the two operators along
  the identity residual $E_{\text{pf}}-E_{\text{nat}}$;
- $\alpha = +0.0$ is an **exact operational no-op**: the implementation
  short-circuits to $E_{\text{nat}}$, and the same-surface control verified
  bitwise-identical predictions to the native decoder on 13/13 sessions
  (prediction SHA, R², targets, window starts, window counts);
- everything else ($g, e, p, f_\theta$, carrier $s$, normalizer) stays frozen
  and in eval mode; the trainable parameter count is **1**.

## 3. How $\alpha$ is fit (source-only, 12 epochs)

1. Seven held-in source sessions, lexical 5/2 fit/validation split.
2. Causal pool states only (decode-before-commit; pools are
   support$_4$ / support$_4$+recent$_6$ / support$_4$+recent$_{26}$ under a
   frozen 4→10→30 size controller; no future/current-trial leakage).
3. Loss: task-only **last-bin behavior MSE** (scaled targets); no teacher
   forwards, no auxiliary losses; only $\alpha$ receives gradient (Adam).
4. Selection: earliest epoch maximizing equal-mean validation R² on the 2
   held-out source sessions subject to a safety gate versus the $\alpha{=}0$
   state; refit $\alpha$ from $+0.0$ on all 7 sessions for the selected
   duration (11 epochs).
5. Outcome: $\alpha^\ast = -0.20759029686450958$ ($\tanh\alpha^\ast \approx -0.205$),
   then frozen.

**Sign interpretation.** $\tanh\alpha^\ast<0$ means the source data prefers a
small step *away* from pure post-fusion averaging — the same all-negative
signature as every jointly-trained post-fusion gate (PF-R1: $\tanh\alpha=-0.65$;
PF-R50: all 50 coefficients negative). Post-fusion information is real but
enters only as a bounded residual around the native operator, never as a
replacement for it.

## 4. Numbers for the paper (external = 6 held-out local sessions)

Equal-session mean R², UNCAPPED law, corrected deployment operator:

| System | Params added | External R² | Δ vs native |
|---|---:|---:|---:|
| Native POOLED (anchor) | 0 | 0.2991 | — |
| PF-MEAN (jointly retrained) | 0 | 0.1869 | −0.1122 (1/6) |
| PF-R1 (jointly retrained) | 1 | 0.2384 | −0.0607 (1/6) |
| PF-R50 (jointly retrained) | 50 | 0.2335 | −0.0655 (1/6) |
| **APFG (anchored, source-fit)** | **1** | **0.3037** | **+0.0046 (4/6)**, CI [−0.0034, +0.0132] |

Two sentences this table supports:

- *Anchoring removes the catastrophic transfer loss*: jointly retrained
  post-fusion variants land 0.06–0.11 R² below the anchor they were initialized
  from, while the anchored gate starts exactly at the anchor and can only move
  along a one-dimensional residual.
- *The residual signal is positive but bounded*: +0.0046 mean with a CI
  crossing zero did **not** pass the pre-registered promotion gate
  (+0.010 mean, ≥4/6 sessions), so APFG is reported as a mechanistic
  ablation, not a promoted method.

Within-surface factorization that must accompany any APFG table (7 sessions):

| Contrast | Δ R² | Positive |
|---|---:|---:|
| Total: APFG\|UNCAPPED − NATIVE\|FIXED30 | +0.0140 (CI [+0.0104, +0.0178]) | 7/7 |
| Memory only: ZERO\|UNCAPPED − ZERO\|FIXED30 | +0.0139 (7/7) | 7/7 |
| Gate only: LEARNED\|UNCAPPED − ZERO\|UNCAPPED | +0.00007 | 5/7 |

The within-surface gain is **causal activity memory**, not the gate. Reporting
the +0.0140 as a post-fusion effect is forbidden by the result doc.

## 5. Pool-size dependence (deployment note, submission 581713)

The correction term scales with the native-vs-average gap $\|E_{\text{pf}}-E_{\text{nat}}\|$,
which shrinks as pools homogenize: measured gate effect is +0.0046 on small
growing pools (4→~20), +0.0002 at capacity-30 pools, and −0.0008 on a frozen
30-trial calibration block. The official-protocol submission of the static
variant (EvalAI 581713) is expected to tie the static baseline; a tie
confirms the mechanism analysis on official data.

## 6. Paper-safe statement (verbatim, from RESULT V2 §7)

> Anchoring a source-learned Post-Fusion residual at the exact native pooled
> decoder eliminated the large degradation of jointly trained Post-Fusion
> models and yielded a small positive external point estimate (+0.0046 R2,
> 4/6 sessions), but did not pass the pre-registered +0.010 promotion gate and
> had a bootstrap interval crossing zero. In contrast, uncapped causal
> activity memory improved the within-session surface by +0.0139 R2 in all
> seven sessions. We therefore treat Post-Fusion gating as a bounded
> mechanistic signal rather than a promoted contribution.

Do not claim: APFG as a headline method; the within +0.0140 as a gate effect;
external significance; V1's cross-device SHA failure as a negative result.

## 7. Suggested placement

1. **Method section**: Section "Anchored identity adaptation" — Eq. (gate) +
   the nesting property + the 3-line training protocol. One parameter added;
   no runtime cost (identities precomputed per session).
2. **Results**: the external table (Section 4) as the post-fusion ablation,
   immediately followed by the within-surface factorization table, attributed
   to continual activity memory (the paper's deployable contribution).
3. **Analysis**: the all-negative learned gates across variants + the
   pool-size dependence — one short paragraph each.
4. **Reproducibility footnote**: α*, checkpoint SHA, V2 result graph SHAs,
   and the official probe submission ID for the deployment claim.
