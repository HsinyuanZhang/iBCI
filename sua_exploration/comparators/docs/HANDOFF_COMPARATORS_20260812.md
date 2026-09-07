# Handoff: Comparators — Complete Plan

**Date:** 2026-08-12
**Purpose:** the single document for every comparison the paper needs. What to cite, what to
reproduce, how to reproduce it, what we already have, and what to skip.

**Supersedes and replaces:** `HANDOFF_COMPARISON_PLAN_20260812.md`,
`HANDOFF_COMPARISON_PLAN_20260812_zh.md`, `HANDOFF_BASELINE_GAP_ANALYSIS_20260812.md`,
`HANDOFF_RELATED_METHOD_COMPARISON_POINTS_20260812.md`. Those four are merged here and deleted.
**Chinese translation:** `HANDOFF_COMPARATORS_20260812_zh.md`

---

## 1. The rule

**You may cite a number from another paper only if it was computed on the same data, the same split,
and the same metric. Otherwise you must reproduce it yourself.**

This splits our four datasets in two:

| Group | Datasets | Why |
|---|---|---|
| **Citable** | FALCON M2, FALCON H1 | The organizer holds a hidden test set and scores every submission identically. Other papers' numbers on this endpoint are directly comparable. |
| **Not citable** | subject-M (DANDI 000688), RT | No common benchmark, split, or metric. Every comparator must be run by us. |

"Close enough" is not citable. Different sessions, window lengths, R-squared conventions, or
train/test boundaries all make a published number unusable.

---

## 2. Coverage today

| Dataset | Internal controls | SPINT-family | Ridge | Population vector | Kalman |
|---|---|---|---|---|---|
| subject-M SUA | done | none | done, 0.4179 | done, 0.1154 | **missing** |
| subject-M pseudo-MUA | done | none | done, 0.4102 | done, 0.1042 | **missing** |
| FALCON M2 | partial | done, matched | done, 0.1139 | **missing** | **missing** |
| RT | done | done, B2-D1024 | done, 0.2002 | proven inapplicable | **missing** |
| FALCON H1 | same-checkpoint only | done, matched | done, 0.2582 / tuned 0.2889 | likely inapplicable | **missing** |

Carrier references: subject-M 0.3568 / 0.3061, M2 0.2268, RT 0.4482, H1 0.5000.

---

## 3. Free comparators: FALCON Table 1

The FALCON benchmark paper reports baselines on **the same hidden endpoint we submit to**. Zero
compute required. This is the highest-value item in this document.

### H1, held-out R2 / held-in R2

| Method | Class | Held-out | Held-in |
|---|---|---:|---:|
| NDT2 Multi | oracle | 0.63 ±0.08 | 0.68 |
| **NDT2 Multi** | **few-shot supervised** | **0.52 ±0.04** | 0.62 |
| RNN | oracle | 0.44 ±0.13 | 0.51 |
| **ours, all-source CarrierID** | **few-shot supervised, no backprop** | **0.2749 ±0.1272** | 0.4731 |
| ours, matched SPINT reproduction | same | 0.2615 ±0.1487 | 0.4704 |
| Wiener Filter | oracle | 0.21 ±0.04 | 0.24 |
| Wiener Filter | zero-shot | 0.16 ±0.03 | 0.20 |
| **NoMAD + WF** | **few-shot unsupervised** | **0.13 ±0.10** | 0.21 |
| **CycleGAN + WF** | **few-shot unsupervised** | **0.12 ±0.06** | 0.15 |
| Wiener Filter | held-out calibration | 0.11 ±0.03 | - |
| NDT2 | zero-shot | 0.10 ±0.10 | 0.32 |
| RNN | zero-shot | 0.09 ±0.18 | 0.31 |

### M2, held-out R2 / held-in R2

| Method | Class | Held-out | Held-in |
|---|---|---:|---:|
| NDT2 Multi | oracle | 0.58 ±0.04 | 0.62 |
| RNN | oracle | 0.56 ±0.04 | 0.59 |
| NDT2 Multi | few-shot supervised | 0.43 ±0.08 | 0.63 |
| Wiener Filter | oracle | 0.26 ±0.03 | 0.27 |
| CycleGAN + WF | few-shot unsupervised | 0.22 ±0.06 | 0.32 |
| NoMAD + WF | few-shot unsupervised | 0.20 ±0.10 | 0.35 |
| Wiener Filter | held-out calibration | 0.14 ±0.05 | - |
| RNN | zero-shot | 0.13 ±0.09 | 0.17 |
| Wiener Filter | zero-shot | 0.06 ±0.04 | 0.15 |
| NDT2 | zero-shot | -0.03 ±0.15 | 0.28 |

### What this resolves

**The unsupervised-alignment gap closes with no reimplementation.** Threats to Validity currently
concedes we never compared against latent alignment. On the same endpoint, NoMAD + WF scores 0.13
and CycleGAN + WF scores 0.12 on H1 against our 0.2749; on M2 they score 0.20 and 0.22. Citing was
always permitted — only reimplementation was the risk.

A **Wiener Filter** reference also exists on the same endpoint, organizer-scored, at 0.11 (H1) and
0.14 (M2) when trained on held-out calibration. That is the closest published analogue to our ridge.

**Before printing:** verify our submission split matches FALCON's held-out split exactly, and
retrieve our absolute M2 organizer-held score from the receipts — the ledger currently stores only
the `+0.116765` delta over the original system.

---

## 4. Where we win and lose — state both

**We beat on H1 held-out:** every zero-shot method (WF 0.16, NDT2 0.10, RNN 0.09), both few-shot
unsupervised alignment methods (NoMAD 0.13, CycleGAN 0.12), the oracle Wiener Filter (0.21), and WF
on held-out calibration (0.11).

**We lose on H1 held-out:** NDT2 Multi few-shot supervised (0.52) and the RNN oracle (0.44).

**Do not hide the NDT2 Multi result.** It is public and on the same endpoint. The defensible framing
is that our contribution is not state-of-the-art FALCON accuracy. NDT2 Multi is a large
multi-session transformer doing few-shot **supervised** adaptation. Our claim is different:
competitive with or better than every zero-shot and unsupervised-adaptation method, **without any
target-session backward pass**, with an identity path 102.6x smaller, about 12% lower
organizer-measured latency, and event-level rather than dense supervision.

Framed on those axes the FALCON table strengthens the paper. Framed as an accuracy claim it sinks it.

---

## 5. The axes that actually separate method families

Accuracy alone does not distinguish these approaches. These seven axes do, and the paper should use
them as its comparison frame.

| Axis | Range across methods |
|---|---|
| Target-session supervision | none (zero-shot), unlabelled only (alignment), sparse labels (ours), dense labels (WF/ridge, NDT2 FSS) |
| Backward pass on the new session | required by NDT2 FSS, CycleGAN, NoMAD, MPA; **not** by us or closed-form readouts |
| Channel correspondence required | ridge/WF/Kalman need a fixed unit map; permutation-invariant identity paths do not |
| What is adapted | input alignment, latent alignment, decoder weights, or an identity token (ours) |
| Calibration data needed | H1 uses 4 trials, about 60 s; alignment methods typically need a full unlabelled block |
| Source multi-session requirement | NDT2 Multi and ours need it; WF/Kalman/PV do not |
| Deployment cost | parameters, MACs, latency. Only we and MPA report this axis |

Our distinctive position is the **conjunction**: sparse labelled supervision, no target-session
backward pass, no channel correspondence, and a small identity path. No other method in the table
occupies all four at once.

---

## 6. Numbers we already reproduced

All ours, all with immutable receipts.

### 6.1 Ridge, subject-M — receipt `b6a080c4...ba58`

| Budget | SUA dense | SUA direction-only | pMUA dense | pMUA direction-only |
|---|---:|---:|---:|---:|
| M15 | 0.0726 | −0.4599 | 0.1291 | −0.4244 |
| M30 | 0.3222 | −0.2086 | 0.3414 | −0.1635 |
| M50 | **0.4179** | −0.1220 | **0.4102** | −0.0879 |

Carrier: SUA 0.3381 / 0.3582 / 0.3568, pMUA 0.2882 / 0.3053 / 0.3061.

Two things follow. The **direction-only** arm receives the same trial-direction labels the carrier
uses and is negative at every budget; the carrier beats it at every budget with 14/15 or 15/15
session signs. And the carrier's loss to the dense ridge exists **only at the longest budget** — at
M15 SUA the ridge scores 0.0726 against the carrier's 0.3381.

Two constraints on using this. The direction-only comparison is **system-level**: the carrier also
carries a source-pretrained decoder, so it is not information- or architecture-matched. A
matched-label T4-Ridge `+0.25 R2` claim was already made, reviewed, and **struck**; do not reinstate
it. And the budget crossover is **mean-driven**: the carrier has a positive grand-mean delta in 3/6
cells but a majority of positive session deltas in **0/6**. Report it with session counts shown.

### 6.2 Ridge family, H1 — receipt `37e32b01...9e84`

Integrity gate reproduced the sealed `0.25823473332303337` at delta exactly **0.0**.

| Arm | pooled R2 | vs sealed | vs carrier 0.500037 |
|---|---:|---:|---:|
| sealed, fixed lambda | 0.258235 | - | −0.2418 |
| PCA k=8 | 0.071827 | **−0.186** | −0.428 |
| PCA k=16 | 0.140643 | −0.118 | −0.359 |
| PCA k=32 | 0.212532 | −0.046 | −0.288 |
| PCA k=64 | 0.217619 | −0.041 | −0.282 |
| lambda-CV | 0.284103 | +0.026 | −0.216 |
| W sweep | 0.284103 | +0.026 | −0.216 |
| **PCA + lambda-CV** | **0.288861** | **+0.031** | **−0.211** |

**Favourable:** even the fully tuned, PCA-reduced ridge stays 0.211 below the carrier, about a 2x
deficit on the identical query. A reviewer demanding a tuned baseline now has a pre-computed answer.

**Unfavourable, and it revises an earlier recommendation:** reducing the ridge's dimension *hurts
badly*. The two best-posed arms (k=8 at 15.2 obs/param, k=16 at 7.6) are the two worst performers,
and within-calibration CV selected the least-reduced options. Most of the gain came from lambda
selection alone. **Therefore the paper must not claim the ridge loses because it is
underdetermined.** Observations-per-parameter joins design condition number, retained variance, and
retained energy as a **fourth** proxy statistic that fails to predict performance. What survives is
the descriptive statement that the two estimators pose problems with very different
sample-to-parameter structure — the carrier's ratio independent of channel count, the readout's
degrading as `1/N` — as structure, never as cause.

*Minor defect:* the receipt's pooled obs/param figures appear to be sums of per-session values
rather than pooled ratios. Fix before printing; conclusions unaffected.

### 6.3 Ridge and PV, RT — receipt `c51cb0ff...b042`

All 15 folds bound to the sealed Stage-2 query identities.

| Arm | mean R2 | carrier higher |
|---|---:|---|
| T4d, sealed | **0.448176** | - |
| dense Full, sealed | 0.445189 | - |
| Ridge W50 lambda=1 | 0.200202 | **15/15** |
| Ridge W50 lambda-CV | 0.198913 | **15/15** |
| Zero4, sealed | 0.179272 | - |
| B2-D1024, sealed | 0.145148 | - |

Mean `T4d − Ridge` is +0.2480 fixed and +0.2493 tuned; the ridge never wins a fold, and lands only
+0.021 above the zero-content Zero4 arm. This closes the paper's most exposed hole: the largest
margin no longer sits on the only dataset without a classical baseline.

**Population vector: `PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE`.** The native direction field has
exactly one unique value across all 15 sessions; 0/15 non-degenerate. Report as evidence supporting
the endpoint-derived carrier, not as a missing number. A secondary diagnostic PV on endpoint-derived
directions scores 0.00139 and must never be presented as a native-annotation baseline.

### 6.4 Within-ridge density ablation, A2b-v2 — receipt `0d4cd01b...7361`

Same target, features and solver; only labelled windows per trial vary. `K=all − K=1` is
+0.289/+0.328 at M30/M50 SUA, 14/15 sessions positive, bootstrap excluding zero. A valid
**within-ridge** causal statement about supervision density. It does **not** make the carrier and
the ridge information-matched.

### 6.5 Source-pooled ridge — Part A only, no valid Part B

Protocol `SOURCE_POOLED_RIDGE_PROTOCOL_20260812.md`. The constructibility audit completed:

| View | correspondent channels | Verdict |
|---|---:|---|
| subject-M SUA | **0** | `SPR_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` |
| subject-M pseudo-MUA | **0** | `SPR_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` |

Channel counts vary across sessions (37, 41, 54, 55, 57, 58, 59, 62, 66, 68, 90, 92) with zero
correspondent channels in either view. **A source-pooled ridge is therefore not constructible on
subject-M at all.** If this holds up, it is a mechanism argument in the paper's favour: a direct
readout cannot benefit from source data without a channel correspondence that does not exist,
whereas a permutation-invariant carrier requires none.

**Two blockers before using it.** The `0` count must be verified to distinguish *identifiers
re-indexed per session* (a data-format fact) from *electrodes genuinely not corresponding* (a
neuroscience claim). Only the second supports the argument. And **no Part B receipt exists**: the
run was interrupted while the direction-only scratch arm was failing its integrity gate, which under
the frozen protocol voids all downstream numbers.

---

## 7. Still missing, ranked

| # | Item | Compute | Why |
|---|---|---|---|
| 1 | **Kalman filter**, all four datasets | CPU | The most conspicuous absence. The velocity Kalman filter is the standard clinical iBCI decoder and H1 comes from that lineage. Comparing against PV and ridge but not Kalman reads as incomplete. |
| 2 | Source-pooled ridge Part B | CPU | Code written; would remove the source-pretraining confound — but see 6.5, it may be structurally undefined |
| 3 | PV on M2; applicability statement for H1 | CPU | M2 has a discrete direction field so PV is definable and cheap. H1 is 7-DoF with no discrete direction target, so produce an evidenced applicability statement like RT's |
| 4 | Separately trained corruption arms on H1 | **GPU** | H1's content and attachment controls are same-checkpoint only; those were measured to understate dependence about 4x. Preflight passes; the run died producing nothing |
| 5 | Cost-of-no-backprop baseline | **GPU** | Readout-only probe first, then full fine-tuning. The paper's premise is that a backward pass is unaffordable and never says what that costs |

---

## 7.5 Skeletons built 2026-08-12 — review outcome

Two comparator skeletons were built and unit-tested. **Neither has been run as an experiment.**
Review findings below must be addressed before any run.

### 7.5.1 Scope rule correction — the prohibition is PER DATASET, not global

An earlier instruction said "never open a path containing `held-out`". **That is wrong as a blanket
rule** and it caused errors in both directions.

| Dataset | Is `held-out-calib` in scope? | Why |
|---|---|---|
| FALCON M2 | **Yes** | The `held-out-calib` split is FALCON's released few-shot calibration data, and the sealed M2 arms use exactly it as the M24 budget |
| FALCON H1 | **No** | `h1_sparse_event_endpoint.reject_path_scope` fails closed on `held-out` and requires `sub-HumanPitt-held-in-calib`; the entire H1 evidence chain is scoped to the 13 public held-in recordings |
| subject-M, RT | n/a | No such split |

Never confuse `held-out-calib` (released calibration data) with the private evaluation set. The
former is legal where the sealed protocol uses it; the latter is never legal.

### 7.5.2 Kalman skeleton — built, 9/9 unit tests pass, NOT run

Files: `sua_exploration/mc_maze/kalman_comparator.py`, `scripts/run_kalman_comparator.py`,
`tests/test_kalman_comparator.py`. Dataset-agnostic core plus per-dataset adapters, with a
`--dry-run` mode a reviewer should run first.

Adapters report ready for subject-M (both views), RT and H1, with RT binding the sealed Stage-2
triple hash and H1 binding the sealed 8,965-window manifest `665fe535...e4da`.

**Three items to resolve before running:**

1. **Position is synthesised, not measured.** The adapter sets position `= cumsum(velocity) * bin_size`
   with zero initial offset, because the subject-M and RT loaders expose no cursor-position channel.
   A Kalman filter's position state is normally real position, and integrating velocity accumulates
   drift. Either justify this explicitly in the paper or restrict the state to velocity only.
   **This is the most consequential unreviewed design decision.**
2. **subject-M query identity is a Kalman-specific composite**, not the sealed ridge receipt fields.
   It is anchored to the same V9 artifacts, but equality with the sealed query set must be proven
   before any contrast is reported, or the arms are not on identical windows.
3. **M2 is marked `blocked_data_access`** solely because of the over-broad rule in 7.5.1. The binding
   is code-complete. Unblock it.

### 7.5.3 Population vector skeleton — built, 6/6 unit tests pass; H1 audit QUARANTINED

Files: `sua_exploration/mc_maze/population_vector_comparator.py`,
`scripts/run_population_vector_comparator.py`, `tests/test_population_vector_comparator.py`.

**M2: `PV_DEFINABLE`, 6/6 sessions.** Direction derived from `trials.tgt_loc` as a centre-relative
`arctan2`, excluding centre/rest targets at `(0.5,0.5)`, matching native T4 feature conventions.
Seven to nine unique directions per session, six to eight within the M24 budget. The scoring arm is
implemented, reuses the sealed subject-M PV primitives, and binds `query_target_bins_sha256` from
the M2 split manifest. Receipt retained.

**H1: audit QUARANTINED for scope violation.** The module globs `*held-out-calib*.nwb` at lines 281
and 297, bypassing the sealed loader, and read 14 held-out-calib files. Quarantined to
`sua_exploration/results/population_vector_comparator/QUARANTINE_scope_violation/` with a README.

Severity: **not a label leak** — only `nwb.trials` presence was inspected and reported absent, so no
neural or behavioural values were consumed. But the receipt is inconsistent with the H1 evidence
chain and must not be cited or bound downstream. The verdict
`PV_INAPPLICABLE_NO_NATIVE_DISCRETE_DIRECTION_FIELD` is likely correct on its merits but was
established on the wrong files and must be **re-derived on the 13 held-in-calib sessions**, routing
through `h1_sparse_event_endpoint.index_heldin_calib` rather than a raw glob.

---

## 8. How to reproduce each

### 8.1 Ridge — conventions identical everywhere
Reuse existing implementations, do not write new ones:
- H1 `sua_exploration/scripts/run_h1_ridge_baseline_v2r2.py`
- RT `sua_exploration/mc_maze/rt_classical_comparators.py`
- subject-M the A2a-v2 weighting-control code, receipt `b6a080c4...ba58`

Fixed: causal `W`-bin by `N`-channel history, normalised lambda, **unpenalised intercept**,
support-block normalisation statistics only, per-session fitting. Every hyperparameter selected
**inside the calibration block**, never using query windows.

### 8.2 Kalman filter — recipe
Standard velocity Kalman filter, all parameters closed-form from the calibration block, no
backpropagation, which keeps it inside the paper's comparator class.

State `x_t` = position and velocity (2-D for subject-M / M2 / RT, 7-D for H1) plus a constant.
Observation `z_t` = binned firing rates.

```
state:        x_t = A x_{t-1} + w_t,   w_t ~ N(0, W)
observation:  z_t = H x_t + q_t,       q_t ~ N(0, Q)
```

Least-squares fits from calibration:
```
A = X1 X0^T (X0 X0^T)^-1
W = (X1 - A X0)(X1 - A X0)^T / T
H = Z X^T (X X^T)^-1
Q = (Z - H X)(Z - H X)^T / T
```
`X0`/`X1` are lagged and current state matrices, `Z` the rate matrix. Then run the standard
predict/update recursion on the query block, reporting the same metric as every other arm. Add and
record a `Q` regularisation floor. Fit per session; never transfer across sessions.

### 8.3 Source-pooled ridge — follow the frozen protocol
Part A first: measure whether unit identifiers correspond across sessions. Where they do not, report
`SPR_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` and run no arm. **Never manufacture a correspondence.**
Then pool `beta_prior = mean over source sessions` and solve
`beta = (X'X + (lambda+gamma) I)^-1 (X'y + gamma beta_prior)`, selecting `lambda` and `gamma` by
leave-one-session-out **among source sessions only**. `gamma = 0` must exactly recover the scratch
ridge; assert it in a test.

### 8.4 Integrity gate for every reproduction
Before interpreting any new arm, reproduce that dataset's sealed reference and report the maximum
deviation. Standards already met: H1 ridge to `0.0`, subject-M ridge to `2.24e-6`. If a gate fails,
**stop** — the new numbers are void. This already caught one run (6.5).

---

## 9. Recent work, 2025-2026

**PRI-T**, hidden Markov target inference, Nature BioMed Eng 2025. Unsupervised recalibration. Its
central finding helps us: PRI-T, FA stabilization and ADAN perform **equivalently** on day pairs but
**all fail when days are widely separated**, and only *target-labelling* strategies sustain
long-term control under chaining. Ours is a sparse target-labelling strategy. **Cite as motivation,
not as a comparator** — different task and data.

**MPA**, test-time adaptation for spiking networks, arXiv 2606.14866, 2026. LoRA-restricted TTA
updating under 9% of parameters, explicitly motivated by NoMAD being too expensive for implantable
hardware. Our nearest competitor **on the efficiency axis**, sharing our framing. Difference to
state: MPA is unsupervised but still performs gradient updates at test time; we perform none. Not
numerically comparable (different dataset) — contrast on the Section 5 axes.

**ALIGN**, adversarial session-invariant speech decoding, PMLR 2026. Same problem, different
modality. Cite for breadth.

**NoMAD**, Nature Communications 2025. Already in our bibliography and now also directly citable via
FALCON Table 1 on both H1 and M2.

---

## 10. What to skip

**Reimplementing** unsupervised latent alignment (ADAN, NoMAD, CycleGAN, Degenhart-style) or
multi-session pretrained models (POYO). Each is a full reimplementation with its own tuning surface,
and a badly tuned reimplementation is worse than no comparison because a reviewer who knows the
method will see it.

This is now moot for H1 and M2 anyway — Section 3 supplies NoMAD and CycleGAN numbers on the same
endpoint. **Citing is always allowed; only reimplementation is the risk.**

---

## 11. Reporting rules

1. Select hyperparameters inside the calibration block only. Never use query windows.
2. Report per-session paired contrasts with sign counts and an uncertainty interval, the same way
   carrier arms are reported. A single mean is not enough.
3. Report unfavourable results in the same table and font as favourable ones. The carrier loses to
   the dense ridge on subject-M at M50 and to NDT2 Multi on H1; both must stay visible.
4. State the supervision each comparator consumes next to its score, keeping accuracy and label cost
   separable.
5. Never tune a comparator where tuning helps us and leave it fixed where it does not. If the ridge
   is tuned on H1, it must be tuned on subject-M too, even though that widens our deficit there.
6. Every published number cites its receipt SHA and verifier status.
7. Against cited FALCON baselines, make comparisons **descriptive only**. Our held-out standard
   deviation (±0.1272) overlaps several baselines and the organizer provides no per-session scores,
   so paired tests are impossible.

---

## 12. Order

| # | Action | Compute |
|---|---|---|
| 1 | Verify our submission split matches FALCON's held-out split | none |
| 2 | Add the FALCON Table 1 comparison for H1 and M2 | none |
| 3 | Retrieve our absolute M2 organizer-held score from receipts | none |
| 4 | Rewrite Threats to Validity — we now *do* compare against alignment | writing |
| 5 | Reposition the contribution on the Section 5 axes, not accuracy | writing |
| 6 | Cite PRI-T for target-labelling, MPA for the efficiency axis | writing |
| 7 | Kalman filter, all four datasets | CPU |
| 8 | Verify the SPR correspondence finding, then decide on Part B | CPU |
| 9 | PV on M2, applicability statement for H1 | CPU |
| 10 | Readout-only probe | GPU |
| 11 | Separately trained H1 corruption arms | GPU |

Items 1-6 cost no compute and are the highest-value remaining work on the paper.

---

## 13. Open caution

FALCON argues that oracle models show only about 0.04 R2 held-in/held-out variability, so larger
gaps indicate real instability. Our held-in 0.4731 against held-out 0.2749 is a gap of about 0.20.
Expect a reviewer to ask about it and prepare the answer before submission.
