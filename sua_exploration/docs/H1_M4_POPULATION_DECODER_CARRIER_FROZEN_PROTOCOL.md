# Frozen protocol: H1 M=4 population decoder-direction carrier CPU audit

**Status:** New, separate CPU-only feasibility question. It does not reopen or
modify any sealed M=2 carrier result, and it is not comparable to the original
M=2 neural-only SPINT budget.

## Hypothesis

Four paired target calibration trials may be sufficient for a source-regularized
low-rank population decoder `y <- X` to yield a reproducibly channel-attached,
four-dimensional decoder-coefficient-row carrier. The target fit is closed
form; it uses no target optimizer or backward pass. A future performance test,
if this CPU gate passes, must compare only against an equally M=4 neural-only
SPINT baseline.

## Scope and timebase

- Open exactly the 13 public H1 `held-in-calib` recordings; reject minival,
  query, formal held-out, EvalAI, GPU, and receipt overwrite paths.
- Use only finite 20-ms bins aggregated into non-overlapping, trial-bounded
  100-ms blocks by the signed H1 authority. No activity threshold or trial
  boundary crossing is permitted.
- Each target recording must have at least seven legal chronological trials,
  so trials 1--4 are support and at least three trials (5+) remain query.
  Record the trial IDs, 100-ms block counts, and exposure for support/query.

## Six-date LODO source plan

For each outer calendar date, exclude every recording from that date from the
neural mean/scale normalizer, population PCA, q/ridge selection, and 4-D
coefficient-row carrier basis. Fit those objects on the other five dates only,
using their trials 1--4 as source support. Select from the frozen grids
`q in {2,4,8,16}` and `ridge lambda in {0.1,1,10,100}` by equal-recording mean
source M=4-support-to-later-query R2. Ties choose smaller q then smaller
lambda. The source-only plan, source input hashes, protocol hash, and source
selection score are recorded before target scoring.

## Target fit, attachment, score, and null

For a target recording, fit the pooled population ridge decoder on trials 1--4
only. Independently fit it on split pair A (trials 1--2) and split pair B
(trials 3--4). Convert all coefficients to raw-channel 7-D rows under the
frozen source PCA/normalizer; their median channel-row cosine is the required
attachment statistic. The pooled decoder is scored only on trials 5+ by
variance-weighted seven-output R2, relative to the pooled support-side target
mean.

The target carrier is the pooled raw-channel coefficient matrix times the
source-only 7-by-4 row basis. For each of 31 deterministic replicates, rotate
the labels independently and nontrivially within every support trial, refit,
and score against the unchanged, correctly paired trials-5+ rates and labels.
No query label/rate influences a target fit, source selection, or null fit.

## Frozen CPU gate

The conjunction passes only if all six date estimates are defined and each of
the following holds on at least four of six dates:

1. correct later-query R2 is greater than zero;
2. correct later-query R2 is greater than the date-level q95 of the 31 rotation
   null replicates; and
3. date mean of recording-level split-pair coefficient-row median cosine is at
   least 0.5.

Failure is terminal for this frozen M=4 program: do not alter q, ridge,
normalization, PCA, timebase, null, attachment threshold, width, fusion, or
source training to rescue it; do not use a GPU or attach the carrier to SPINT.
