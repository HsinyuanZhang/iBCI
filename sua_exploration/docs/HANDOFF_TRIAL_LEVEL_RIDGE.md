# HANDOFF: trial-target rate ridge — audited matched-support comparator

**Date:** 2026-08-11
**Status:** Corrected A2a-v2 and A2b-v2 are complete and independently audited. A2a supports a qualified target-formulation comparison; A2b supports a within-ridge same-target label-density effect at M30/M50. Earlier A2 receipts remain invalid diagnostics. Section 10 is authoritative.
**Compute class:** CPU only. Zero GPU. Zero training. Zero new data scope.
**Written in:** simple direct English.
**Original receipt:** `sua_exploration/results/trial_level_ridge_v1/trial_level_ridge_receipt.json`
**Original receipt SHA-256:** `9420a58cfba0e5b9d12ff0111259cf407c2de96eca78fabe421bcbce90259e34`
**Valid corrected receipts:** A2a-v2 `b6a080c4...ba58`; A2b-v2 `0d4cd01b...7361`

> **Audit notice.** The immutable receipt is a numerical record, not a valid
> equal-information proof. The implementation fits thousands of window rows
> carrying repeated 2-D trial/segment-mean velocity targets; it does not fit an
> `[M,N] -> [M,2]` design. Those target vectors are computed from dense
> kinematics, whereas T4 consumes a trial direction scalar. Existing Ridge50
> also uses a different `50*N` temporal feature representation. SUA/pseudo-MUA
> therefore supports only a matched-support-trial system comparison. RT and M2
> are exploratory and non-comparable to their cited references because their
> scoring masks include support bins. Section 10 freezes the corrective CPU
> experiments. Corrected A2a/A2b-v2 remove the solver/weighting confound and isolate a same-target density effect
> within their frozen ridge protocol, but they still do not make T4 and ridge equal-information or equal-architecture.

---

## 1. Why this exists

The paper compares the T4/carrier against a ridge readout that uses dense per-bin velocity (~5000 labeled bins per 50-trial calibration). This CPU experiment asks a narrower question: can a target-session linear readout trained from trial-constant targets compete with the source-pretrained frozen T4 system when both are restricted to the same chronological support-trial count?

A reviewer will ask whether a direct target-session linear readout can use sparse trial annotations as effectively as the frozen carrier decoder.

The implemented ridge uses all legal calibration windows from the first M support trials. Each window from trial j receives the same 2-D trial-mean velocity vector. Thus there are M distinct target vectors on SUA/M2, but thousands of weighted window rows. The 2-D vectors are computed from dense behavior traces and are not the scalar target-direction annotations consumed by T4. On RT, first-24 trial rows contain 60--88 accepted reach segments, so even the number of distinct segment targets is not M=24.

The ridge solver is reused unchanged (mean-normalized lambda=1).

---

## 2. What changes

| Property | Existing Ridge50 | Trial-level ridge (new) |
|---|---|---|
| Calibration features | `[~5000, 50*N]` flattened spike history | `[~3000, N]` 50-bin window-mean rates |
| Calibration targets | `[~5000, 2]` per-bin velocity | `[~3000, 2]` trial-mean velocity (M distinct values) |
| Label density | ~5000 dense 2-D bins | M repeated 2-D vectors on SUA/M2; 60--88 segment vectors on RT |
| Solver | `numerical.fit_ridge` (mean-normalized lambda=1) | Same, unchanged |
| Query prediction | same valid_starts, same scoring code | Same |

This is not a label-only ablation. Relative to existing Ridge50 it also changes the feature width from `50*N` to `N`, removes within-window temporal structure, changes row weighting, and changes the calibration topology. The solver and normalized penalty are reused, but the representation is not.

---

## 3. The three cohorts

### 3.1 SUA center-out (DANDI 000688, 15 external subject-M sessions)

- **Calibration:** all 50-bin causal windows from the first M rewarded trials. Each window gets its trial's mean velocity as the target. Features and queries use the same window-mean-rate representation `[rows, N]`; this does not assert that their statistical distributions are identical.
- **Budgets:** M in {15, 30, 50}.
- **Query:** the same V9 byte-identical target trace, same valid_starts, scored by torchmetrics R² 1.5.1.
- **Views:** SUA and pseudo-MUA.
- **T4 sealed reference:** 0.3381/0.3582/0.3568 (SUA), 0.2882/0.3053/0.3061 (pMUA) at M=15/30/50.
- **Exposure qualification:** at M=15 the T4 label prefix is 15 trials, but its activity-identity path remains fixed at 30 unlabeled trials. The ridge M15 fit uses neural windows from only 15 trials. Disclosure is necessary but does not turn this into a label-only causal comparison; it remains a system comparison with different unlabeled neural exposure and pretrained prior.

### 3.2 RT random-target (DANDI 000688, 15 sub-C sessions)

- **Calibration:** 5-bin causal window rates from reach segments within the first 24 trial-rows. Each window gets its segment's mean velocity as the target.
- **Budget:** M24 (the native RT budget).
- **Query:** 5-bin causal window rate at each eval bin, predicted by the ridge readout. Target is dense velocity at the same bins.
- **Historical reference:** 0.4419 is the dense continuous-velocity Full carrier, not sparse T4 (single seed, 15-fold nested LOSO).
- **Comparison note:** the reference is cross-session LOSO while this ridge is a within-session refit; it also uses a different temporal window. The ridge query includes its first-24 support bins. This comparison is exploratory, unpaired, and cannot be used for an equal-information claim.

### 3.3 M2 native (DANDI 000953, 6 held-out sessions)

- **Calibration:** 50-bin causal window rates from the first 24 trials. Each window gets its trial's mean velocity as the target.
- **Budget:** M24 (the native M2 budget).
- **Query:** 50-bin causal window-mean rate at each eval bin, predicted by the ridge readout.
- **T4 sealed reference:** 0.2268. Ridge-W50 (dense 50-bin): 0.1139. K4: 0.2458.
- **Comparison note:** the ridge uses 24 repeated 2-D trial-mean velocity vectors computed from dense kinematics, not the same labels as T4. Its query includes all of its 329--573 calibration windows, whereas the cited q24 references require a post-support query. The aggregate values are therefore descriptive only.

---

## 4. Results summary

### SUA: T4 dominates at every budget

| Budget | View | Trial-ridge mean | T4 mean | T4 − Ridge | T4 higher |
|---|---|---|---|---|---|
| M=15 | SUA | 0.0677 | 0.3381 | +0.2704 | 15/15 |
| M=30 | SUA | 0.0795 | 0.3582 | +0.2787 | 14/15 |
| M=50 | SUA | 0.0892 | 0.3568 | +0.2676 | 14/15 |
| M=15 | pMUA | 0.0637 | 0.2882 | +0.2245 | 14/15 |
| M=30 | pMUA | 0.0745 | 0.3053 | +0.2308 | 14/15 |
| M=50 | pMUA | 0.0828 | 0.3061 | +0.2233 | 14/15 |

T4 beats the trial-level ridge by +0.22 to +0.28 at every budget, in both views.

### RT: ridge is far below carrier

| Cohort | Trial-ridge mean | Carrier reference |
|---|---|---|
| RT M24 | 0.1413 | 0.4419 (cross-session LOSO) |

The trial-level ridge scores 0.1413 mean R², far below the carrier's 0.4419.

### M2: trial-level ridge is negative

| Cohort | Trial-ridge mean | T4 | Ridge-W50 (dense) |
|---|---|---|---|
| M2 M24 | −0.0354 | 0.2268 | 0.1139 |

This implementation is negative on M2, but the value is not on the same post-support query contract as the cited T4 and Ridge-W50 references. It cannot identify label density as the cause.

---

## 5. Conditioning

At M=15 on SUA, the ridge sees ~3000 windows with ~92 features. The design is full-rank (rank=92), while the labels carry only M=15 distinct velocity values and the effective degrees of freedom (trace(H)) is ~30. Its poor query performance is consistent with limited target variation being difficult for this specified readout, but these rank and trace statistics do not isolate the cause from representation, target semantics, weighting, or solver effects.

The system-level result is consistent with a source-pretrained frozen decoder using sparse trial-level direction information more effectively than this direct readout. It does not by itself prove which mechanism creates that advantage.

---

## 6. Non-interference contract

| Rule | Value |
|---|---|
| GPU | None. `CUDA_VISIBLE_DEVICES=""` for all processes. |
| Threads | `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `nice -n 10`. |
| Processes | No start/stop/signal of any process not started by this work. |
| Output | All writes to `sua_exploration/results/trial_level_ridge_v1/` only. |
| Artifacts | All existing artifacts read-only. No git/pip/conda. No formal/minival/EvalAI. |

---

## 7. Read rules. Frozen before reading the receipt

| Outcome | What the paper may say |
|---|---|
| Trial-target ridge is below T4 at every budget, in both views | On matched SUA/pseudo-MUA assets and post-trial-50 queries, the frozen T4 system uses sparse trial supervision more effectively than this fixed-lambda window-mean linear readout. This is a system comparison, not an equal-information or label-only causal comparison. |
| Trial-level ridge is above T4 at some budgets | Report it plainly. The linear readout is competitive even at low label density on those budgets. |
| Trial-level ridge is catastrophic (negative R²) | This specified readout is not competitive under the tested representation, target, weighting, and solver formulation. Do not assign the failure to label budget alone. |

**Observed outcome:** Row 1 applies only to the paired SUA/pseudo-MUA cohort (14--15/15 session-level contrasts positive). RT 0.141 versus dense Full 0.442 and M2 -0.035 versus historical references are exploratory observations under mismatched query/feature contracts.

---

## 8. Evidence status

- Every number is development evidence.
- The SUA comparison uses the same sessions and byte-identical post-trial-50 query targets. It remains a qualified system comparison because T4 is source-pretrained, its M15 activity path sees 30 unlabeled trials, and its session score is a three-seed mean while ridge is one deterministic solve.
- The RT and M2 comparisons are exploratory, unpaired, and support/query-overlapping in the current runner. They must not be used as matched evidence.
- This does not supersede any sealed result.
- The ridge solver is reused unchanged from the existing implementation. No new solver, no new penalty, no new standardization.
- The receipt pins its own bytes but does not pin runner/core/input/reference/query-vector SHA-256 values. It is not a complete independently replayable provenance chain.

---

## 9. References

- `sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py` — the reused ridge solver
- `sua_exploration/mc_maze/trial_level_ridge_core.py` — conditioning and rate-building functions
- `sua_exploration/scripts/run_trial_ridge_three_cohort.py` — the runner
- `sua_exploration/results/trial_level_ridge_v1/trial_level_ridge_receipt.json` — the receipt

---

## 10. Corrective experiments and frozen interpretation

All corrective runs are CPU-only. They do not retrain T4 and must not read a
formal or organizer-hidden endpoint.

### 10.1 Priority A — same-X target-supervision ablation on SUA/pseudo-MUA: numerical run complete, causal gate not closed

**Receipt:** `sua_exploration/results/trial_level_ridge_v1/priority_a_label_density_receipt.json`
**SHA-256:** `6fd6feef0f41482f89fc2a9a52b18f389d68f275057c36e574f7aae001d04521`

The run keeps the `50*N` flattened features, calibration windows, post-trial-50 query, support-only feature standardization, and fixed lambda=1. It changes the target and the fitting weights:

- `dense-Y`: per-bin endpoint velocity with the existing unweighted Ridge50 solve;
- `sparse-dir-Y`: unit-magnitude `[cos(theta), sin(theta)]` from the trial-table direction scalar, repeated per trial, with an intended equal-trial-weighted solve.

This is therefore a controlled same-X comparison of two supervision schemes, but it is not yet a pure label-density ablation. Besides supervision granularity, the sparse arm removes within-trial speed and temporal-profile information, changes target magnitude/semantics, and changes sample weighting.

**Results:**

| Budget | View | Dense-Y | Sparse-dir-Y | Δ(dense−sparse) | T4 |
|---|---|---|---|---|---|
| M=15 | SUA | 0.0726 | −0.4599 | +0.5325 | 0.3381 |
| M=30 | SUA | 0.3222 | −0.2086 | +0.5308 | 0.3582 |
| M=50 | SUA | 0.4179 | −0.1220 | +0.5399 | 0.3568 |
| M=15 | pMUA | 0.1291 | −0.4244 | +0.5535 | 0.2882 |
| M=30 | pMUA | 0.3414 | −0.1635 | +0.5050 | 0.3053 |
| M=50 | pMUA | 0.4102 | −0.0879 | +0.4981 | 0.3061 |

**Audited findings:**

1. Dense-Y reproduces the known Ridge50/budget-curve aggregate values (`0.0726/0.3222/0.4179` SUA and `0.1291/0.3414/0.4102` pseudo-MUA), which validates the dense branch numerically at aggregate level. At M50 the 30 session/view cells match the sealed Ridge50 results within maximum absolute error `2.24e-6`, but only 5/30 are bit-exact; therefore use “high-precision numerical reproduction,” not “exact reproduction.” The receipt does not bind the historical reference artifacts strongly enough to call the run independently replayed.
2. With the same `50*N` neural representation, the tested direction-only direct ridge is negative at every budget. The large dense−sparse gap (`+0.50` to `+0.55`) is consistent with the dense target carrying dynamic information absent from the direction-only target, but it does not isolate label *density alone* from speed, temporal profile, target scale, weighting, or solver effects.
3. T4 beats the direction-only sparse ridge at every budget (14/15 or 15/15 session signs). T4 does **not** beat dense-Y at every budget: it has a positive grand-mean delta in only 3/6 cells and a majority of positive session deltas in 0/6; dense-Y is higher in grand mean for SUA M50 and pseudo-MUA M30/M50. The supported statement is that the source-pretrained T4 system produces useful dynamic predictions from sparse trial-direction input whereas this direction-only direct ridge does not; this remains a system-level result rather than isolated evidence for the decoder mechanism.
4. The intended equal-trial weighting needs a solver correction before causal use. The current sparse solver subtracts an unweighted target mean while applying unequal row weights and does not solve an explicit unpenalized intercept. A weighted ridge with explicit intercept (or correct weighted centering) is required.
5. The receipt's `conditioning_median` field is not a median: the runner stores `cond[0]`. It must be renamed or recomputed before this conditioning statistic is cited.

### 10.2 Priority B — true M-row diagnostic

Separately fit the design that the original receipt incorrectly described:
trial-mean neural rates `[M,N]` to a trial-level target `[M,2]`. Keep it named
`M-row trial-mean ridge`; do not merge it with the window-row comparator. Run a
small source-only lambda sensitivity or freeze a lambda grid before reading the
15-session target aggregate.

### 10.2A Priority A2a — corrected weighting control complete

**Invalidated receipt:** `sua_exploration/results/trial_level_ridge_v1/priority_a2_weighting_receipt.json`
**SHA-256:** `6364b99f110a489172b5262c289605975e12b8d18e59615dc396befa20aa686f`

The first A2a run used an unnormalized weighted Gram and is invalid for weighting, density, or accuracy claims. Its
immutable receipt is retained only as a diagnostic. A2a-v2 instead uses

```text
(Z_w^T Z_s / W + lambda I) beta = Z_w^T Y_c / W
```

with `W=sum(row_weights)`, an explicit unpenalized intercept, and fail-closed direction validation. The valid receipt
is `priority_a2_weighting_control_v2_receipt.json`, SHA
`b6a080c48d36adc74050f0a6672623585320e32bada45001a962622379bcba58`. Its 90-cell gate reproduces the sealed
dense-uniform references within `5e-5` (maximum error `2.24e-6`). The frozen 2×2 control is:

| target | row weighting |
|---|---|
| dense per-bin velocity | uniform-window and equal-trial |
| direction-only `[cos θ, sin θ]` | uniform-window and equal-trial |

Across budgets, views, and weighting rules, dense−direction mean R² ranges from approximately `+0.46` to `+0.56`;
at M50 the range is `+0.46` to `+0.53`. Equal-trial versus uniform weighting changes dense mean R² by only `+0.004`
to `+0.076`. Thus row weighting alone does not explain the gap. The arms still differ in target content/semantics as
well as granularity, so A2a is a supervision-formulation comparison, not a pure label-density ablation.

### 10.2B Priority A2b — same-target label-density dose response

The corrected A2b-v2 experiment is complete. It fixes dense two-dimensional velocity, `50*N` neural features,
support trials, post-trial-50 query, standardization, normalized `lambda=1` solver, explicit intercept, and equal-trial
weighting. Only the target-blind number of labelled windows per trial varies:

```text
K = 1, 2, 4, 8, 16, all labelled windows per support trial
```

Rows are selected by nested deterministic permutations without reading targets, using seeds `42/43/44`, budgets
`M={15,30,50}`, and both SUA and pseudo-MUA views. The valid 1,620-cell receipt is
`priority_a2_same_target_density_v2_receipt.json`, SHA
`0d4cd01b5fea86e2dca4640731f99faa2abde57d3d163bf7dee6ac34ed167361`; all support/query intersections are zero.
Its `K=all` arm reproduces A2a-v2 `dense_equal_trial` in 90/90 cells with maximum R² difference `0.0` and exact
coefficient SHA in 90/90 cells.

The predeclared primary contrast `R²(all)−R²(K=1)`, after averaging each session over the three mask seeds, is:

| view | M15 | M30 | M50 |
|---|---:|---:|---:|
| SUA | `+0.160` (CI crosses 0) | `+0.289`, 14/15 positive | `+0.328`, 14/15 positive |
| pseudo-MUA | `+0.199` (CI crosses 0) | `+0.297`, 14/15 positive | `+0.312`, 14/15 positive |

The M30/M50 exact sign-test p-value is `0.00098` and the paired bootstrap intervals exclude zero. M15 remains
inconclusive by the bootstrap criterion. The three-seed aggregate rises from K1 to K4 in every view/budget cell.
At K8/K16, however, negative grand means conflict with positive medians and 14/15 positive session means: one
recording (`sub-M_ses-CO-20150512`) and mask sensitivity dominate the aggregate. No condition-number, solve-residual,
or precision/primal-dual diagnostic was recorded, so this must not be called numerical ill-conditioning or a
population-wide reversal.

This supports a within-ridge causal statement that target density matters under the frozen A2b protocol. It does not
make T4 and ridge equal-information or equal-architecture: T4 uses one trial-direction scalar plus a source-pretrained
decoder, while A2b directly fits dense velocity. T4-versus-ridge remains a system-level trade-off.

### 10.3 Priority C — strict RT/M2 replication, only if retained in the paper

Use the exact carrier reference query boundary and temporal representation.
Every calibration/query row intersection must be zero and must be recorded in
the receipt. RT must compare against the corresponding sparse T4d arm rather
than the dense Full carrier. M2 must use the authoritative q24 post-support
mask. Emit fold/session-matched deltas rather than subtracting one global
aggregate constant.

### 10.4 Receipt requirements

The corrected receipt must bind the runner, core module, NWB inputs, reference
aggregates, calibration-row vector, query-row vector, predictions, and targets
by SHA-256. It must record exact shapes, distinct-label counts, per-trial row
weights, support/query overlap count, source-only lambda policy, and package
version used for R2.

### 10.5 Current paper-safe conclusion

> A2a-v2 shows that dense velocity and trial direction are not interchangeable targets for the same direct ridge,
> even after normalized-solver and weighting corrections; this is a target-content/granularity result. A2b-v2 then
> isolates supervision density within the dense-target ridge protocol: all labelled windows outperform one labelled
> window per trial robustly at M30/M50 in both SUA and pseudo-MUA, while M15 remains uncertain. This does not identify
> the cause of the system-level T4-versus-Ridge50 gap, because their targets, pretrained priors, architectures, and
> deployment computations remain different.
