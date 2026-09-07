# Frozen Protocol: Cross-Session Consistency of the Pooled Representation $\hat{y}$ (CSCS)

**Date predeclared:** 2026-08-13, **before any arm was executed.**
**Status:** CPU-only screen. **Trains nothing.** Loads sealed A2-v2 source checkpoints only.
**Scope:** DANDI 000688 center-out, SUA view, sub-C source domain and sub-M external cohort.
**Out of scope:** H1, RT, M2, the paper, and any sealed sub-C formal-test session.

---

## 1. The question this screen answers

CEBRA aligns multiple sessions with an **explicit cross-session contrastive term**: a reference
sample from session A is paired with a positive from session B and the loss pulls their embeddings
together. A separate control established that this sampling is the *only* thing producing alignment
in CEBRA — removing it collapsed synthetic-control target transfer from +0.99 to −1.23.

Our source loss (paper eq. 9) has no such term. It is plain MSE on behaviour, minimised jointly over
27 held-in sub-C sessions with all decoder weights shared. Cross-session consistency of the pooled
representation

$$\hat{y}_c = \tilde{q}_c + \mathrm{FFN}(\mathrm{LN}(\tilde{q}_c)) \in \mathbb{R}^{d}, \quad c = 1 \dots C$$

(paper eq. 7, **post-FFN, pre-readout**) is therefore only an *indirect by-product* of weight sharing.

The candidate change — **not** tested here — is to add $\mathcal{L} = \mathcal{L}_{\mathrm{MSE}} +
\lambda \mathcal{L}_{\mathrm{consist}}$ at source-training time, pulling together the $\hat{y}$ of
behaviour-matched samples from different source sessions. The mechanistic argument for it is that
$\hat{y} \mapsto \hat{v} = W_{\mathrm{ro}}\hat{y} + b_{\mathrm{ro}}$ is many-to-one, so a null space
exists in which two sessions' $\hat{y}$ differ substantially while producing identical outputs. MSE
cannot constrain that null space; a consistency term can.

**That argument is worthless if the representations are already consistent.** This screen asks:

> Is there measurable cross-session inconsistency in $\hat{y}$ that a consistency loss could
> plausibly fix, and does that inconsistency predict external-domain performance?

**A KILL verdict is a fully successful outcome of this screen** and is arguably the more useful one,
because it saves scarce GPU time. The analysis will not be massaged toward PROCEED.

---

## 2. Sealed inputs — nothing is trained, nothing is modified

| Artifact | Binding |
|---|---|
| Checkpoints | `sua_exploration/checkpoints/a2_matched_subject_shift_v2_source_{t4,z4}_dandi688_co_s{42,43,44}` |
| Epoch window | `epoch_ckpts/epoch_004..011.ckpt` = protocol epochs **5–12**, from `a2_matched_subject_shift_v2_core.EPOCH_WINDOW`. Never `best_checkpoint`. |
| Source roster | 27 sub-C CO train sessions from `configs/subc_co_27_6_strict_train_val_manifest.json` |
| Held-in-domain roster | 6 sub-C CO validation sessions from the same manifest |
| External roster | 15 frozen sub-M CO sessions from `mc_maze.gpu_contract_common.SUBM_EXTERNAL_SESSIONS` |
| Formal test | The 6 sealed sub-C test sessions are **names only**. No test NWB is resolved or opened. |
| Normalizers | Behaviour and T4 normalizers refit from the strict 27-session source roster with `cache_dir=None`, then verified bitwise against the training-path cache **and** against `run_metadata.side_features.normalization_sha256`. Reuses `a2_matched_subject_shift_v2_score._fit_source_normalizers` unchanged. |
| External R² | Read from the existing immutable receipts `results/a2_matched_subject_shift_v2/external_subject_M_source_{arm}_s{seed}.json`. **Not recomputed.** |

Sealed modules under `sua_exploration/mc_maze/` and `SPINT-main/src/` are imported and reused, never
modified. New code lives in `mc_maze/cross_session_consistency_screen.py`,
`scripts/run_cross_session_consistency_screen.py`, `tests/test_cross_session_consistency_screen.py`.

**Execution constraints.** `CUDA_VISIBLE_DEVICES=""` set before Torch import and re-asserted after;
`PYTHONNOUSERSITE=1`; `torch.no_grad()` throughout; no optimizer, no backward, no `.grad`, no
checkpoint write. The runner fails closed if `torch.cuda.is_available()` is True.

---

## 3. Where $\hat{y}$ is taken from

All six bundles are `decoder_mode == "coupled"`, so the live forward path is
`StreamingSpintModel.decode_with_identity`, whose last two lines are

```
transformer_output, _ = self.decoder.transformer(rep.repeat(...), src)   # this is y_hat, eq. (7)
output = self.decoder.fc_out(transformer_output)                          # this is W_ro, eq. (6)
```

$\hat{y}$ is captured with a **forward pre-hook on `model.student.decoder.fc_out`**, which records
that layer's input. This is exactly the post-FFN, pre-readout tensor of eq. (7) and requires no edit
to any sealed module. Shape `[B, C, d]`; flattened to $D = C \cdot d$ per window. The runner asserts
`decoder_mode == "coupled"` and records the observed `C`, `d`, `D`.

**Identity reuse.** $E_i$ depends only on the calibration prefix and side features, not on the query
window, so it is computed once per (session, arm, checkpoint) with
`student.compute_identity(...)` and reused across windows. The runner **proves** this is exact by
running one batch through the full `student(...)` path and asserting `torch.equal` on the captured
$\hat{y}$; the parity result is written to the receipt. (Verified bitwise-identical during design.)

---

## 4. Behavioural matching — the condition grid

Center-out gives a discrete shared behavioural coordinate. Both sub-C and sub-M NWB trial tables
carry `target_dir` at exactly the eight values $\{0, \pm 45, \pm 90, \pm 135, 180\}$ degrees, so the
direction bin is

$$k_{\mathrm{dir}} = \mathrm{round}\!\left(\theta / (\pi/4)\right) \bmod 8 \in \{0,\dots,7\}.$$

The runner asserts every used trial's `target_dir` is within $10^{-6}$ rad of a multiple of $\pi/4$.

**Direction alone is not enough.** Eight bins cannot identify a linear map on a $D$-dimensional
representation, and $\hat{y}$ varies strongly *within* a trial. Sessions differ in trial-duration
distribution, so averaging all windows of a trial would let phase composition masquerade as
inconsistency. The matched condition is therefore **direction × within-trial phase**:

- **Phase axis:** window start offset $o$ from trial start, $o \in \{0, 2, 4, \dots, 48\}$ (25
  offsets). Window $[start+o,\ start+o+50)$. Covers trial time 0–1.96 s at 20 ms bins.
- **Trial eligibility:** post-30 query trial (`trials[30:]`, matching the frozen A2-v2 query policy
  exactly, so the calibration prefix that built $E_i$ is never scored), finite `target_dir`, and
  duration $\ge 98$ bins so that every one of the 25 offsets exists.
- **Trials per direction:** exactly **`TRIALS_PER_DIRECTION = 10`**, the chronologically first 10
  eligible query trials of that direction. This equalises estimator noise across sessions whose
  trial counts range from 143 to 1008, and makes the split-half ceiling apply uniformly.
- **Condition grid:** $K = 8 \times 25 = 200$ conditions. $Y_{\mathrm{sess}} \in \mathbb{R}^{K
  \times D}$ holds the mean $\hat{y}$ over the 10 trials of each condition.

**Session eligibility.** A session enters the primary analysis iff it has $\ge 10$ eligible query
trials in **all eight** directions. Excluded sessions are named in the results with their external
R², and are re-admitted only in the pre-declared secondary variant of §7.

A feasibility scan of trial counts per direction (no representation, no R², no model loaded) was run
on 2026-08-13 before this document was frozen, to avoid declaring an infeasible grid. It found 46 of
48 sessions satisfy the rule; the two exceptions are `sub-M_ses-CO-20140626` and
`sub-M_ses-CO-20140627`, which do not contain all eight target directions at all. This is disclosed
here rather than presented later as a discovery.

**Split halves.** Within each session and direction, the 10 selected trials are split by
chronological rank: even ranks $\to$ half 1, odd ranks $\to$ half 2 (5 trials each). Condition means
are recomputed per half. Halves are used for the ceiling and for the noise-matched cross-session
comparison of §6.

---

## 5. The consistency metric

This is the CEBRA cross-session consistency metric reimplemented. CEBRA's own API is not called: it
accepts 1-D labels only and our matched condition is 2-D (direction × phase).

**Step 1 — source-only dimensionality reduction.** $D = C\cdot d$ is far larger than $K$, so an
unregularised linear map between two $K \times D$ matrices is exactly determined and returns $R^2 =
1$ vacuously. Per (arm, seed, epoch) a PCA basis is fitted on the row-stack of the **27 source-train
sessions'** condition-mean matrices only ($27 \times 200 = 5400$ rows $\times\ D$), centred on the
grand mean of those rows, and the top $D'$ right singular vectors are retained. The identical fixed
basis is applied to every session (train, validation, external) and to both raw and null arms. This
mirrors the house convention that every normalizer is source-only. Cumulative variance explained is
reported.

- **Primary $D' = 32$.** Sensitivity reported at $D' \in \{8, 16, 64\}$.
- Because the basis is orthonormal, $R^2$ measured in PCA coordinates is the $R^2$ of the original
  representation restricted to that subspace. Inconsistency living entirely in the discarded
  directions is invisible to this metric; this is a stated limitation, not a claim.

**Step 2 — cross-validated linear map.** For an ordered pair $(A, B)$ of aligned condition-mean
matrices $Y_A, Y_B \in \mathbb{R}^{K \times D'}$:

- Folds: 5 deterministic folds, $\mathrm{fold}(k_{\mathrm{dir}}, k_{\mathrm{off}}) = (25\,
  k_{\mathrm{dir}} + k_{\mathrm{off}}) \bmod 5$. No RNG. Stratified over both axes.
- For each fold: ordinary least squares with intercept, fitted on the training conditions, mapping
  $Y_A \to Y_B$; predictions taken on the held-out conditions.
- $$R^2(A \to B) = 1 - \frac{\sum_{k}\lVert Y_B[k] - \hat{Y}_B[k]\rVert^2}{\sum_{k}\lVert Y_B[k] - \bar{Y}_B\rVert^2}$$
  accumulated over all held-out conditions, with $\bar{Y}_B$ the mean over **all** $K$ conditions of
  $B$. Pooled over output dimensions, i.e. variance-weighted.
- Reported pair value is the symmetric mean $\tfrac{1}{2}[R^2(A\to B) + R^2(B\to A)]$. Both
  directions are also retained.

**Step 3 — matched null (mandatory).** A raw $R^2$ is uninterpretable alone.

- **Null-A (primary):** permute the condition index of $B$ against $A$ uniformly at random over all
  $K$ conditions, then run the *identical* fold structure and estimator. $R = 20$ permutations,
  `numpy.random.PCG64` seeded by `SHA-256("cscs-null|" + arm + "|" + seed + "|" + epoch + "|" + A +
  "|" + B + "|" + r)`, so every null is reproducible and independent of run order. This is the
  "pair bins at random across sessions" null: it preserves each session's own geometry, the
  dimensionality, and every degree of freedom, and destroys only the behavioural correspondence.
- **Null-B (secondary, stricter):** permute the 8 direction labels of $B$ only, keeping the phase
  axis aligned. Isolates direction correspondence while leaving temporal correspondence intact.

**Step 4 — reported quantities.** Following the house convention of the tuning-phase probe, every
consistency number is reported as a triple:

| Symbol | Meaning |
|---|---|
| `raw` | cross-validated $R^2$ on the true correspondence |
| `null` | mean over the 20 matched permutations |
| `adv` | `raw − null`, the **null-relative advantage**, the primary statistic |

---

## 6. The ceiling — what "already consistent" would mean

"T4 is already at the achievable ceiling" must be a measurement, not an assertion. The ceiling is
**within-session split-half consistency**: $C_{\mathrm{ceil}}(A) = $ the same metric applied to
$Y_A^{(1)} \to Y_A^{(2)}$. Two independent trial samples from the *same* session share the same
units, the same $E_i$, and the same recording, so no cross-session method can exceed this; what
separates it from 1.0 is pure trial-sampling noise.

A split-half estimate is built from 5 trials per condition while the full cross-session estimate uses
10, so they are **not** noise-matched. The headroom test therefore uses the noise-matched pair:

- $C_{\mathrm{ceil}}$: `adv` of $Y_A^{(1)} \to Y_A^{(2)}$, averaged over sessions.
- $C_{\mathrm{cross}}^{\mathrm{half}}$: `adv` of $Y_A^{(1)} \to Y_B^{(1)}$, averaged over ordered
  source pairs. Same 5-trial noise per side.
- **Headroom fraction** $H = 1 - C_{\mathrm{cross}}^{\mathrm{half}} / C_{\mathrm{ceil}}$.

$H$ is the fraction of the achievable consistency that cross-session sharing has *not* already
captured — the size of the null space a consistency loss would have to work in. Full-sample
cross-session values are reported alongside as the more precise descriptive number.

---

## 7. Analyses, in the order they will be run

1. **A1 — source-side residual inconsistency.** All $27 \times 26 = 702$ ordered source-train pairs,
   both arms, three seeds, eight epochs. Raw / null / adv. This is the set on which the proposed loss
   would actually operate.
2. **A2 — ceiling and headroom.** §6, on the same 27 sessions.
3. **A3 — T4 versus Z4.** Paired by (seed, epoch, session pair). Mean difference, **per-pair sign
   count**, and a paired bootstrap 95% interval over source sessions (10 000 resamples of sessions,
   not of pairs, to respect the dependence between pairs sharing a session).
4. **A4 — held-in-domain generalisation of the finding.** The same metric on the 6 sub-C validation
   sessions, which are same-subject but were never optimised.
5. **A5 — the decisive analysis.** For each external sub-M session $m$, its consistency to the source
   pool is the mean over the 27 source sessions of the symmetric `adv` value $C(m, s_j)$. Relate that
   to the session's external mean R² from the sealed A2-v2 receipt (itself the mean over epochs
   5–12), using Spearman $\rho$ within each (arm, seed) cell over the eligible external sessions.
   Aggregate as the mean $\rho$ over the three seeds of an arm, with a 95% bootstrap interval built
   by resampling **sessions** (10 000 resamples; each resample recomputes $\rho$ per seed and then
   averages), which respects the fact that the three seeds score the same sessions.
6. **A6 — confound controls for A5, reported whether or not they are favourable.**
   - Partial Spearman controlling for the session's unit count $N$.
   - Partial Spearman controlling for the session's mean firing rate over the calibration prefix.
   - The arm-level dose-response: 6 points (2 arms × 3 seeds) of mean source consistency against mean
     external R². $n=6$ is stated as underpowered; it is reported, not leaned on.
7. **Secondary variants, pre-declared, reported in the same table and font as the primary.**
   - $D' \in \{8, 16, 64\}$.
   - Null-B.
   - Pair-specific shared-direction grid (intersection of directions available in both sessions),
     which re-admits `sub-M_ses-CO-20140626` and `sub-M_ses-CO-20140627` at a smaller $K$.
   - All-window aggregation with no phase axis (direction bins only, all post-30 windows), reported
     as the "no phase control" contrast to show how much the phase axis matters.

---

## 8. KILL / PROCEED — fixed before any number is looked at

Let $H$ be the headroom fraction of §6 for the **T4** arm, averaged over the three seeds and the
eight epochs. Let $\bar{\rho}$ be the A5 mean Spearman for the **T4** arm with its 95% bootstrap
interval $[\rho_{lo}, \rho_{hi}]$.

**Recommend KILL if either holds:**

- **K1 — no null space left.** $H < 0.10$. T4's noise-matched cross-session consistency is within
  10% of the within-session split-half ceiling, so there is essentially nothing left for a
  consistency term to constrain.
- **K2 — consistency does not predict external performance.** $\bar{\rho} < 0.30$ **or**
  $\rho_{lo} \le 0$. Improving consistency is then not evidently a route to a better external number.

**Recommend PROCEED only if both hold:** $H \ge 0.10$ **and** $\bar{\rho} \ge 0.30$ with
$\rho_{lo} > 0$.

### Thresholds chosen arbitrarily, disclosed as such

Both numeric thresholds are conventions, not derived quantities, and are recorded here so the choice
cannot be made after seeing the data:

| Threshold | Value | Basis |
|---|---|---|
| Headroom fraction $H$ | 0.10 | **Arbitrary.** "Within 10% of ceiling" is a convention. Chosen to match the 0.10 gate magnitudes already used by the A4-v2 tuning-phase probe (`GATE_DELTA_PHASE_MEAN_MIN`, `GATE_AC4_NULL_ADVANTAGE_MIN`). |
| Spearman $\bar{\rho}$ | 0.30 | **Arbitrary.** Conventional "moderate" effect size. With 13 eligible external sessions, $\rho = 0.30$ is near the limit of what this sample can resolve; the CI condition, not the point estimate, does the work. |
| Permutations $R$ | 20 | Chosen for runtime. It estimates the null *mean*, which is all `adv` needs; it is not used for a permutation p-value. |
| $\mathrm{TRIALS\_PER\_DIRECTION}$ | 10 | Forced: the minimum over eligible sessions is exactly 10 (`sub-C_ses-CO-20150629`). Any larger value would drop sessions. |
| Primary $D'$ | 32 | **Arbitrary** within $\{8,16,32,64\}$; all four are reported. |

$H$ is a ratio of two `adv` values and is undefined or unstable if $C_{\mathrm{ceil}} \le 0.05$. If
that occurs, $H$ is reported as undefined, K1 cannot fire, and the verdict rests on K2 alone; this is
recorded as a limitation rather than resolved by substituting another statistic.

**Interpretation limit fixed in advance.** A5 is observational across sessions. Even a strong
$\bar{\rho}$ would show that consistency *co-varies* with external R², not that raising consistency
would raise R². A PROCEED verdict therefore licenses a training-time experiment, never a claim.

---

## 9. Reporting requirements

- Per-session and per-pair paired contrasts with **sign counts** and a bootstrap uncertainty
  interval, never a bare mean.
- Unfavourable results in the same table and the same font as favourable ones.
- One immutable JSON receipt written `O_EXCL` / `fsync` / `0444` with a SHA-256 sidecar, carrying:
  input NWB SHA-256 by session, all 48 checkpoint SHA-256s, `run_metadata.json` SHA-256 per bundle,
  the normalizer value hashes, the SHA-256 of this protocol document, the resolved configuration,
  the sealed external-R² receipt hashes, and an environment fingerprint (Python, NumPy, Torch,
  platform, thread count, `CUDA_VISIBLE_DEVICES`, `PYTHONNOUSERSITE`, `cuda.is_available()`).
- An explicit list of what could not be verified.

---

## 10. Known limitations of this screen, stated before it runs

1. The metric sees only the top $D'$ principal directions of the source-train representation. Null
   space that is orthogonal to all of them is invisible.
2. Trial start is used as the temporal anchor for the phase axis. Whether it is the same task event
   in sub-C and sub-M is assumed, not verified from the NWB schema.
3. Consistency is measured on a frozen model. It is not established that a consistency loss would
   actually move this metric, nor that moving it is achievable without harming the MSE term.
4. A5 is a between-session association with $n = 13$ and known confounds; §7.6 controls two of them
   and cannot control the rest.
5. Z4's external R² is near or below zero, so its A5 correlation is measured across a range of
   mostly-failed sessions and is expected to be weakly informative regardless of outcome.
