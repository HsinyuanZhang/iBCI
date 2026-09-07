# Budget-Matched Posterior CAL-AUG V1

Date: 2026-08-29

## 1. Purpose

This work order turns the useful part of the teammate's GitHub experiment E02 into a
calibration-budget experiment for DANDI 000688 sub-C/sub-M centre-out data.
It is not an E02 replication.  E02 used 50 calibration trials.  This design
tests the actual deployment budgets M30, M10, and M4.

The primary question is:

> Does training Cell D with a budget-matched posterior-mean carrier improve
> M4 and M10 beyond activity-prefix CAL-AUG alone?

The first implementation stage is CPU-only contract and numerical validation.
It must not read NWB files or checkpoints, initialize CUDA, touch result roots,
or interfere with the active T0/C1 jobs.

## 2. Evidence imported from E01--E10

### 2.1 Exact evaluation scope

The GitHub E01--E10 numbers are real held-out results, not training-set or
within-window scores.  Their exact label in this work order is:

```text
sub-C session-heldout-8 / M50 / seed42 development screen
```

The split was 37 sub-C training sessions, 8 sub-C validation sessions, and 8
different sub-C target-test sessions.  No target-test optimizer or backward
was used.  This is useful same-subject cross-session evidence.  It is not the
project's governing `sub-C -> sub-M external-15` surface, and it contains no
M4/M10 budget result.

The eight test sessions were repeatedly consulted to decide whether later
E-cells should run.  They are therefore `held-out-selected development`, not
an independent blind formal set.

### 2.2 Held-out numbers

The seed-42 M50 session-heldout-8 experiment reported:

- E01 ordinary T4: mean/worst/session-SD R2 = 0.6137/0.4280/0.1136.
- E02 posterior-mean T4: 0.6287/0.4462/0.1067; mean delta +0.0150,
  positive in 5/8 sessions.
- E03 posterior mean plus angular reliability: mean 0.6367 and worst 0.4725;
  incremental mean delta over E02 +0.0080.
- E09 direct analytic residual and E10 local-frame residual raised the mean by
  +0.0141 and +0.0243 versus E01, but worsened the worst session to 0.3512 and
  0.2816 and increased session dispersion.

These are development results from one seed, M50, and a validation-argmax
checkpoint rule.  They are positive held-out motivation, but not accepted
cross-subject or M4/M10 evidence.

### 2.3 GitHub reproducibility boundary

E01's aggregate, audit, and per-session CSV are committed in the summary
branch.  The consolidated document contains the E02/E03/E04 and E09/E10 raw
per-session numbers.  However, the E02/E03/E04 aggregate links named in that
document are not present at those paths in the visible summary-branch result
tree, and the E09/E10 formal `p3_*.json`, logs, and checkpoints were gitignored.
The consolidated table is therefore the available remote evidence for those
cells; it is not a substitute for new immutable local receipts.

## 3. C2 system: one matched budget per training forward

The deterministic global training-forward cycle is exactly:

```text
M = 30, 10, 4, 30, 10, 4, ...
```

For each training forward:

1. choose M using integer cycle arithmetic;
2. use chronological support positions `[0, M)` for the B3S activity prefix;
3. use the exact same trial IDs and order for the carrier fit;
4. compute the posterior-mean raw carrier `[a, c, hypot(a,c), b]`;
5. normalize it with one frozen source-only posterior normalizer;
6. run the unchanged Cell-D query forward, loss, dropout, optimizer, schedule,
   batch order, and target definition.

M30 carrier bytes must never be substituted at M10 or M4.  A reordered,
different, future, or partly overlapping carrier prefix is a hard failure.

## 4. Exact posterior estimator

For one session and one unit, use trial-level rates, not equal-direction means.
Let `X = [1, cos(theta), sin(theta)]`.  First compute OLS and its residual
variance:

```text
beta_ols = solve(X'X, X'y)
sigma2 = ||y - X beta_ols||^2 / (n - rank(X))
```

The posterior mean is:

```text
beta_post = solve(
    X'X + diag(0, sigma2/tau2, sigma2/tau2),
    X'y,
)
```

The intercept is unpenalized.  `tau2` is a single positive isotropic source
prior variance for `(a,c)`.  It is fitted once from the strict source roster at
M30 and then reused unchanged at M30/M10/M4 and on every target session.  It
must not be fitted per budget or on target data.

The E02 value `tau2=1.4556518254` is provenance evidence only.  It came from a
different 37-session M50 source population and must not be copied into the new
authority without a strict-source audit.

The source prior law is:

```text
raw_second_moment = mean(a_ols^2 + c_ols^2) / 2
expected_ols_noise = mean(sigma2 * trace(inv(X'X)[a,c])) / 2
tau2 = max(floor, raw_second_moment - expected_ols_noise)
```

All solves, prior fitting, covariance calculations, and normalizer moments are
float64.  Only the final normalized side feature entering Cell D is cast once
to float32.

## 5. Source-only normalizer

Fit one posterior-specific normalizer from deterministic posterior means on
the strict source roster.  Concatenate rows in this exact order:

```text
budget M4, then M10, then M30;
within each budget: strict source roster order, then source unit order.
```

Each budget must contribute the same number of unit rows.  Use population
moments (`ddof=0`).  Ordinary M30 OLS normalizer moments are forbidden.  The
normalizer is frozen before training and cannot be refitted on a target.

## 6. Mandatory feasibility gate

The exact E02 residual-variance law needs `rank(X)=3` and `n-rank(X)>0`.
Therefore every strict-source prefix at M4/M10/M30 must pass both conditions
before any GPU smoke can be authorized.  M4 is the critical case: four trials
leave one residual degree of freedom only when their directions span rank 3.

The audit reports, per budget and source session:

- trial count, distinct direction count, rank, and residual degrees of freedom;
- exact ordered support-ID digest;
- PASS or `STOP_INSUFFICIENT_DIRECTION_RANK_OR_DOF`.

No fallback, pseudoinverse, borrowed M30 variance, or changed trial selection is
allowed inside C2.  A later D-optimal successor would be a different design.

## 7. Comparison cells and decision order

The matched cells are:

- T0: current recipe, matched-seed retraining control.
- C1: activity-prefix CAL-AUG, ordinary M30 carrier (already running).
- C2: budget-matched activity prefix plus posterior-mean carrier.
- C3-Const: C2 plus one constant reliability column, trained with the same
  five-dimensional side-input width as C3-Real.
- C3-Real: C2 plus E03 angular reliability.
- C3-RowShuffle-Eval: the C3-Real checkpoint with only q rows deterministically
  permuted at evaluation; no extra training run.

C2 is compared primarily with C1, not only with a historical Cell-D
checkpoint.  Headline surfaces are external M4 and M10; M30 is a safety row.
Use identical sessions, query records, seed, optimizer-step count, checkpoint
rule, and scoring implementation.

Minimum promotion evidence for a full C2 run:

- external M4 or M10 equal-session mean delta versus C1 at least +0.02;
- at least 10/15 external sessions positive on that promoted budget;
- neither external M30 nor within M30 below C1 by more than 0.01;
- no material worsening of the external worst-session result.

These are promotion rules, not formal statistical claims.

### 7.1 Fast E02/E03 execution policy

E02 and E03 are the active method family.  Their code, source audit, receipt
schema, and smoke tests are prepared together now.  C3 implementation does not
wait for a completed C2 external score.

After T0/C1 finish and the source rank/DOF audit passes:

1. run one C2 smoke;
2. if the smoke is finite, deterministic, and source-grouped-OOF non-harmful,
   authorize C2, C3-Const, and C3-Real matched training on available GPUs;
3. score all three on identical M4/M10/M30 source and external records;
4. run C3-RowShuffle-Eval on the already trained C3-Real checkpoint.

C3-Real must be compared both with C2 and C3-Const.  The row-shuffle evaluates
whether the trained model uses the correct unit-to-reliability binding.  A
separate fully trained shuffled-q arm is deferred unless these two controls
leave the mechanism ambiguous.  This removes one expensive full run without
dropping the two main identification controls.

The E03 claim needs all of the following:

- C3-Real improves C2 on external M4 or M10;
- C3-Real improves C3-Const, so the result is not just the wider input layer;
- C3-RowShuffle-Eval loses the relevant gain;
- posterior q correlates with source held-out carrier angular error in the
  expected direction;
- external M30 and worst-session safety are not materially worse.

## 8. Later successors

### C3: angular reliability feature

Implement now alongside C2.  Add E03's scalar

```text
q = -log((u_perp' Sigma_ac u_perp) / (||mu_ac||^2 + eps) + eps)
```

with fail-closed `q=-20` for near-zero modulation.  The exact real/constant/
row-shuffle design is frozen in section 7.1.  Do not reuse E04's monotone
attention-logit bias; it failed.

### C4: C2 plus activity-only CDM

If C2 improves the initial sparse carrier, evaluate the same checkpoint with
activity-only CDM at deployment.  Keep the carrier fixed.  This is the main
candidate for a larger M4/M10 gain because C2 addresses carrier estimation and
activity-only CDM addresses the changing activity identity.

### C5: precision-gated carrier update

Only after C4.  Use posterior precision to decide whether carrier updates are
allowed.  Compare with the best activity-only rule.  M30 carrier updating is
off by default.

### E09/E10 disposition

E09 and E10 are explicitly parked until C2/C3 have completed M4/M10 and
external-15 scoring.  Do not implement or launch either route in parallel with
the E02/E03 queue.  Their small session-heldout-8 mean gains came with worse
tail behavior.  After C2/C3, E09 may be reconsidered only with a residual-only
control and a worst-session non-degradation rule.  E10 is considered only if
E09 first passes; its local frame may then be tested as an auxiliary
source-training loss while the governing Cartesian output remains unchanged.

## 9. Stage boundaries

Stage 0 (authorized now): pure NumPy estimator, prior, normalizer, budget
binding, rank/DOF gates, E03 q computation, inert dry CLI, and synthetic tests.

Stage 1: metadata/source-only feasibility audit.  It may read only explicitly
authorized strict-source support metadata and rates.  It does not train.

Stage 2: C2 followed immediately by C3 wiring/no-harm GPU smoke, only after
T0/C1 finish and after a separate root review and launch authorization.

Stage 3: matched C2, C3-Const, and C3-Real full training plus common
M4/M10/M30 scoring and the C3 row-shuffle evaluation.

Stage 4: C4 activity-only CDM composition if C2 or C3 provides a safe external
gain.  E09/E10 remain parked at this boundary.
