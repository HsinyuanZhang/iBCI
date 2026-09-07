# C2 — M2 sampling-objective contract (legacy vs equal-session)

**Date:** 2026-08-13
**Status:** CPU scaffolding / pre-registered contract. **Authorizes no GPU run, no training, and no launch.**
**Screen ID:** `m2_sampling_objective_c2_v1`
**Scope:** FALCON M2 internal-development source training only. No SUA/A2 claim, no validation-sampler claim, no batch-level differentiable R² loss.

This document does not mint an official preflight. The runner refuses `--launch`. The six sealed sub-C formal-test sessions are never opened.

---

## 0. Factual finding: the equal-session lever already exists

Section 5.2 item 4 of `HANDOFF_FOUR_LANE_BRAINSTORM_SYNTHESIS_20260813.md` is **correct for the M2 streaming path** and **incorrect if read as a claim about every `SessionBatchSampler`**.

There are three `SessionBatchSampler` definitions. Only one implements equal-session interpolation:

| Path | Equal-session interpolation | Fixed equal window budget |
|---|---|---|
| `streaming_calibration_exp/src/data/falcon_datamodule.py` | **Yes.** `SessionBatchSampler.__init__` takes `balance_sessions=False` at line 758 (class at 751). Strength interpolates empirical session batch counts toward equal counts at lines 854--904 while preserving epoch length; shorter sessions cycle deterministically. Exposed on `FalconDataModule` as `balance_session_batches=False` at line 1001 and passed into the **training** sampler at lines 1497--1503. Ordinary `configs/data/falcon_m2.yaml` currently fixes it to `false`. | **Sampler-only.** `window_budget_per_session=None` at line 760; the ordinary M2 DataModule does **not** route this argument from config. |
| `sua_exploration/mc_maze/multisession_datamodule.py:805--848` | **No.** Constructor is `(dataset, batch_size, shuffle=False, seed=42)` only. | **No.** |
| `SPINT-main/src/data/falcon_datamodule.py:224--258` | **No.** Constructor is `(dataset, batch_size, shuffle=False)` only. | **No.** |

C2 therefore **uses the existing M2 streaming lever** rather than inventing a sampler. The switch remains **default off**. An exact-null CPU test proves that `balance_sessions` in `{False, 0, 0.0}` and the omitted default are bitwise identical to an independent reconstruction of the legacy remainder-dropping sampler. C2 does **not** implement SUA/A2 balancing and does **not** wire `window_budget_per_session` into the ordinary M2 DataModule; those would be separate contracts.

The names `equal_session` and `equal_window_budget` do not appear as identifiers. The existing names are `balance_sessions` / `balance_session_batches` (interpolation) and `window_budget_per_session` (fixed budget, unrouted on ordinary M2).

---

## 1. Claim under test (must be able to fail)

Training MSE is window-count weighted — `SessionBatchSampler` drops remainders, so long sessions receive more gradient steps — while scoring is an unweighted session-mean R². Turning on the existing equal-session interpolation (`balance_session_batches=true`) during M2 source training, with ordinary inherited MSE, improves unweighted session-mean validation R² of T4 by at least `+0.03` relative to the inherited sampler, and the matched Z4 sibling does not reproduce that lift.

Batch-level differentiable R² is held: it is not the same object as the final unweighted session-mean R².

If the gate fails, equal-session sampling is not a printable accuracy route on this substrate.

---

## 2. Frozen matrix

| Factor | Levels |
|---|---|
| Sampling | `legacy` (`balance_session_batches=false`) · `equal_session` (`balance_session_batches=true`, strength 1.0) |
| Carrier | `t4` · matched `z4` (ordinary T4 fit and source normalizer, then `zeros_like`) |
| Seeds | `{42, 43, 44}` |
| Folds | M2 internal LOSO `{0, 1, 2}` |

**Total cells:** `2 × 2 × 3 × 3 = 36`. Each cell is one LOSO left-out development session. The primary mean is the unweighted mean of the 3×3 session-by-seed T4 sampling deltas.

The only intended scientific difference between `legacy` and `equal_session` is the existing training-sampler switch. Validation sampling remains the unbalanced, unshuffled `SessionBatchSampler`. `reshuffle_train_sampler_each_epoch` stays `false`. `window_budget_per_session` is not used.

### Frozen protocol (identical across all 36 cells except the two factors)

| Parameter | Frozen value |
|---|---|
| Variant | B3S, `side_dim=4`, decoder frozen |
| Loss | inherited ordinary MSE: `task_plus_y_plus_E`, `lambda_y=1.0`, `lambda_E=0.1`. **Not** R²-native. |
| Calibration | chronological first `33` trials, `random_calibration=false` |
| Epochs | `12`, no early stopping, no best-validation selection |
| Reported cell score | unweighted mean of validation R² at logical epochs 5--12 |
| Teacher / decoder / add site | unchanged relative to ordinary M2 B3S/T4 |
| Formal EvalAI / sealed sub-C test | never opened |

Z4 is a C2-local wrapper (`C2M2MatchedZ4DataModule`), not B1's class.

---

## 3. Frozen gate

For fold `f` and seed `s`:

```text
delta_T4(f,s) = R(equal_session, T4, f, s) - R(legacy, T4, f, s)
delta_Z4(f,s) = R(equal_session, Z4, f, s) - R(legacy, Z4, f, s)
interaction(f,s) = delta_T4(f,s) - delta_Z4(f,s)
```

`R` is the epoch-5--12 mean validation R² of that LOSO cell.

Call the experiment confirmatory-positive only if all of the following hold on the complete 36-cell matrix:

1. mean `delta_T4 >= +0.03`
2. all three session means of `delta_T4` are positive
3. all three seed means of `delta_T4` are positive
4. mean `interaction >= +0.03`

A crossed session/seed bootstrap interval is descriptive only. No Wilcoxon or exact sign test: both are unattainable at `n = 3`.

Kill interpretations:

| Classification | Meaning |
|---|---|
| `t4_session_mean_r2_lift_below_floor_stop` | equal-session sampling did not deliver the T4 session-mean R² lift |
| `generic_z4_lift_or_no_carrier_specificity_stop` | T4 moved, but Z4 reproduced enough of the move that the interaction missed `+0.03` |
| `sampling_objective_pass` | both the T4 floor and the anti-generic interaction passed |

Incomplete matrices, sealed-session names, R²-native loss flags, or a non-null `window_budget_per_session` fail closed and do not produce a scientific verdict.

The CPU tests include one synthetic matrix that must pass and two that must fail with different classifications.

---

## 4. Implementation bindings

| Artifact | Path |
|---|---|
| Existing lever | `streaming_calibration_exp/src/data/falcon_datamodule.py` `SessionBatchSampler.balance_sessions` |
| Matched Z4 | `streaming_calibration_exp/src/data/c2_m2_matched_z4_datamodule.py` |
| Four configs | `streaming_calibration_exp/configs/experiment/c2_v1_m2_{t4,z4}_{legacy,equal_session}.yaml` |
| Core / preflight / runner / aggregate | `src/metrics/c2_m2_sampling.py`, `scripts/preflight_c2_m2_sampling.py`, `scripts/run_c2_m2_sampling.py`, `scripts/aggregate_c2_m2_sampling.py` |

Launch state: **NO-GO.** `run_c2_m2_sampling.py --launch` always refuses.

**This document authorizes nothing.**
