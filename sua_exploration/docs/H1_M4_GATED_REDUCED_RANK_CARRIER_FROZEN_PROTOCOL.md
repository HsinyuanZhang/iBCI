# Frozen protocol: H1 M=4 gated reduced-rank population carrier CPU audit

**Status:** source-only CPU identifiability audit. This is a new, independently
named M=4 question. It does not alter any sealed M=2, raw-M4, EB-M4, V1, or
V2 receipt, and it does not itself authorize GPU use.

## Scope and fixed matrix

The audit may open exactly the 13 public H1 held-in-calib recordings. It may
not open any separate minival, held-out, formal-test, EvalAI, or external-query
recording; it constructs no GPU object and performs neither target optimization
nor target backpropagation. Within each held-in-calib recording, use finite,
eval-valid, trial-bounded 100-ms blocks made from five native 20-ms bins.
Calibration support is trials 1--4, independent attachment halves are trials
1--2 and trials 3--4, and the strictly later development-scoring interval is
every legal block in trials 5+. This trials-5+ interval is never
activity-gated.

Run exactly this 2x3 matrix:

| gate | q_kin=1 | q_kin=2 | q_kin=3 |
| --- | --- | --- | --- |
| G0: all finite legal support blocks | sensitivity | sensitivity | sensitivity |
| G1: RT-style active support blocks | sensitivity | sensitivity | **primary** |

The sole decision cell is G1, q_kin=3. No result from another cell may rescue
or replace a failure of that primary cell.

`G1` is fixed at the raw-bin level. A raw 20-ms velocity sample is active iff
`~all(abs(v) < 1e-3)` across all seven H1 kinematic coordinates. A five-bin
block is retained iff each of its five raw samples is active. This is the
literal RT active-sample rule with H1's existing no-lag block pairing; there is
no threshold adaptation, block-mean threshold, retained-fraction selection, or
query filtering.

## Source-only plan and deployed reconstruction

For each outer calendar date and gate condition, exclude every outer-date
recording from all source objects. On eligible outer-source M=4 support blocks,
fit the neural mean/scale and neural PCA. Freeze `q_neural=16` and
`lambda=100`; there is no grid or target-selected hyperparameter. Fit the
source population ridge decoders, back-project their slopes to raw-channel
rows `R_s [N,7]`, concatenate them, and take the first `q_kin` right singular
vectors as `U [7,q_kin]`. Fix deterministic column signs and record source
singular values, energy, hashes, sessions, and inputs.

For the same source row pool and each rank, form the existing isotropic EB
prior in carrier coordinates `E_s=R_s@U`: its coordinate mean and its sole
isotropic ML variance are source-only. Target ridge covariance, output residual
variance, EB weights, and shrinkage use target support only, with no sweep or
electrode prior.

For a target fit, the deployable carrier is `E [N,q_kin]` and its only score
path is the actual reduced-rank reconstruction:

```text
R_carrier = E @ U.T
yhat_query = (rates_query - source_mean) @ R_carrier + pooled_support_intercept
```

Thus a rank change is not evaluated merely by replacing the attachment
statistic. If a later, separately proposed model needs the fixed four-wide
interface, it may zero-pad this carrier after the CPU gate; padding neither
changes nor replaces the rank-k reconstruction above.

## Frozen null and controls

Activity eligibility is always computed once from the correct, unpermuted
support behavior and then frozen. For G1, all 31 label nulls retain precisely
the same neural rows, retained block indices, support counts, and exposure as
the correct fit. Within each support trial, labels of those retained blocks are
rotated by a deterministic nonidentity circular offset. The gate is **not**
recomputed after rotation. The null refits ridge/covariance/EB carrier and
scores the unchanged correctly paired trial5+ query through `E@U.T`.

The row-shuffle control applies a deterministic complete nonidentity channel
row permutation to `E`, then scores `E_shuffled@U.T` with the same target
intercept and query. It is neither a separately fit decoder nor a full-rank
score.

## Gate and terminal decision

For each cell, aggregate query R2 within each date by summed SSE/TSS; use dates
as the six primary units. Attachment is the equal-recording mean of each
recording's median channel cosine between independently EB-shrunk split-half
carriers. Record all six date values and the 31 replicate date-level nulls.

The primary G1,q_kin=3 cell is eligible only if all six dates are defined and,
on at least four of six dates: (1) reconstructed correct query R2 is positive;
(2) it exceeds the fresh rotation-null q95; (3) split-half carrier cosine is
at least 0.5; and (4) reconstructed correct R2 exceeds row shuffle. A failed
or undefined primary cell seals the new route: do not select another matrix
cell, threshold, rank, normalizer, prior, source data, fusion, seed, epoch, or
GPU rescue.

If the primary cell passes, the receipt may say only
`eligible_for_separate_gpu_proposal`; it has
`gpu_authorized_by_this_receipt=false`. Any GPU proposal must be separately
reviewed, use an equal-M=4 neural-only H1 comparator, preserve the V2 numerical
contract, and may not freeze or reuse an M2 decoder across tasks.
