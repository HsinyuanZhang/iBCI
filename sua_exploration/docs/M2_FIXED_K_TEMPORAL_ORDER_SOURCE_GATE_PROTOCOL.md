# M2 fixed-K temporal prototype — label-free temporal-order source Gate

**Status:** protocol only; no M2 NWB has been opened for this work.  This document
does not authorize an execution, GPU process, decoder, local held-out replay,
minival access, formal evaluation, or EvalAI submission.

**Terminal status update (2026-08-02): not executable.** M1 Gate A2 stopped at Stage 0 because
P20 lost to B20 in all four sessions. The entry condition below therefore failed, and this P20
M2 protocol is retained as an unexecuted design rather than being repurposed for B20.

**Entry condition:** this experiment may be prepared and run **only after the M1
fixed-K Gate A2 protocol passes all of its frozen content, relative-slot, and
common-bin temporal-order gates**.  The earlier M1 v3 Gate A result is not that
pass and cannot open this M2 branch.  If M1 Gate A2 fails, is invalid, or is
inconclusive, this M2 protocol is permanently unexecuted rather than a rescue
path for M1.

## 1. Narrow question and scope

The question is deliberately narrower than decoding:

> From the first 24 chronological, unlabeled M2 calibration trials, does the
> fixed-width P20 temporal carrier predict a **future per-unit temporal profile**
> better than an equally wide higher-order marginal carrier, and is any advantage
> removed when only common within-trial time order is destroyed?

The seven M2 held-in-calibration sessions are the only prospective source
sessions.  They are the seven source sessions previously used by the M2 M24
CPU audit; a future, separately reviewed manifest must enumerate their exact
paths and SHA-256 values.  It must not discover files recursively.

This is a source-LOSO, later-neural oracle experiment.  It is neither a
behaviour-decoding result nor an unseen-session held-out result.  In particular,
it cannot validate T4, K4, D4, B3/SPINT, a frozen decoder, calibration utility,
or a hardware claim by itself.

No target-direction, object, velocity, EMG, behavioural, minival, local
held-out, formal, or EvalAI label is read.  There is no label-dependent arm and
no scorer-only direction endpoint in this audit.

## 2. Why the later-neural target is an autocorrelation profile

Several apparently convenient targets were considered before freezing this
protocol.

| candidate later target | decision | reason |
|---|---|---|
| future P20 itself | reject | It repeats the carrier's routed-anchor functional on a different time range.  A good P20-to-P20 map could arise from the shared construction even if no independently meaningful temporal property transfers. |
| future ordered-anchor/routing statistics | reject | It changes names but keeps the same anchor, hard-routing, and slot-coordinate machinery on both sides of the regression.  This is too close to a structural self-prediction. |
| future target direction/category tuning | reject for this gate | It requires labels, tests a different supervised claim, and would repeat the label-information asymmetry that complicates K4/T4 comparisons. |
| future within-trial autocorrelation profile | **select** | It is computed directly from raw future bin-count pairs, has no anchor/router/slot/trial-label input, is fixed-width and per-unit, survives variable trial lengths, and gives the common-bin chronology null a direct but non-tautological target. |

The selected target intentionally shares only the scientific construct
“temporal structure” with P20.  It does **not** share P20's feature map.  This
does not prove causal transfer: persistent recording/channel properties could
still help.  The B20 control and the common-bin time-order null are included to
separate those explanations as far as a source-only neural oracle permits.

## 3. Immutable raw-data and temporal boundary

For source session `s`, let trialized raw count data be
`X_s[t,b,i]`, where `i` is a channel/unit and a valid trial has `L_st` bins of
20 ms counts.  The support is exactly chronological trials `[0,24)`; the
future scorer range is exactly `[24,end)`.  There is no gap selected by labels,
activity, performance, or target availability.

The executor must use raw, noninterpolated trial bins with

```text
smooth_calibration = false
interpolate_trials = false
use_calib_intertrials = false
pad_value = -1
```

It must verify, before any feature or target construction, all of the following:

1. exactly seven manifest-listed held-in-calibration NWBs, and no other NWB;
2. a matching SHA-256 for every source file;
3. at least 25 trials for every session, so `[24,end)` is nonempty;
4. finite, nonnegative, integer valid prefixes and exactly `-1` padded tails;
5. each retained raw-prefix channel sum equals the production raw trial-sum
   receipt; and
6. the first 24 trial boundaries and every future trial boundary remain intact
   after intertrial removal.

The runner must reject all paths containing `held-out`, `minival`, `test`,
`evalai`, `formal`, or `report`; it must reject recursive source discovery.
Future raw bins are scorer-only: they cannot enter anchors, carrier fitting,
normalization, routing, ridge fitting, parameter selection, or a state object.

The deployed P20 state cannot retain a trial-by-bin matrix.  A temporary
offline source-anchor workspace is permitted only for the six outer-train
sessions, is discarded before the carrier receipt is returned, and must be
reported separately from deployment state.  During oracle scoring, future data
may be streamed to sufficient statistics for the target but must not be fed
back to the support state.

## 4. Frozen carrier arms

### 4.1 P20 — canonical fixed-K temporal carrier

P20 is line-for-line the existing fixed-K implementation:

```text
K = 4 hard-routed globally shared slots
temporal rank r = 4
causal EWMA alphas = (0.5, 0.25, 0.125, 0.0625)
per-trial filter reset
12 deterministic sorted-feature k-means anchor iterations
lexicographically canonical anchor order
```

For every bin, each unit's four causal filter values are routed to one of four
anchors.  The final unit carrier consists of four complete blocks:

```text
[slot count, mean filter_1, mean filter_2, mean filter_3, mean filter_4]
```

and is thus width `4 × (1 + 4) = 20`.  Fold `s` fits anchors from the complete
first-24 supports of exactly the other six sessions.  The left-out session's
support, all future bins, and all labels are forbidden to the anchor fit.

### 4.2 B20 — high-order marginal baseline

B20 is an equally wide, label-free comparator designed to rule out nonlinear
rate and count-distribution information without any chronological state.  For
one unit's 24 support trials, define the trial log-rate

```text
r_t = log((sum_b X[t,b,i] + 0.5) / (0.020 × L_t)).
```

Its 20 coordinates are:

1. the ten empirical quantiles of the 24 values `r_t`; and
2. the ten empirical quantiles of all pooled valid-bin values `log1p(X[t,b,i])`.

Both groups use the frozen probability vector
`(0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)` and NumPy's
`method="linear"` convention (or an exactly numerically equivalent, receipt
recorded implementation).  The arm is invariant to support-trial order and to
the bin order within each trial.  It carries nonlinear mean rate, trial-rate
heterogeneity, zero mass, burstiness, and the marginal count distribution, but
no slot, anchor, temporal filter, timestamp, channel identifier, label, or
future information.

No legacy rate-only, D4, T4, K4, or learned arm is a substitute for B20 or can
rescue this gate.

## 5. Primary label-free later-neural target

For each unit `i` in a source session, use all raw future trials `t >= 24` and
the eight fixed lags in bins

```text
L = (1, 2, 4, 8, 16, 32, 64, 128).
```

For a future trial, let

```text
mu_it = mean_b X[t,b,i]
v_i = sum_{t,b} (X[t,b,i] - mu_it)^2 / sum_t L_t.
```

For lag `ell`, compute the exposure-weighted, trial-mean-centred lag product

```text
C_i(ell) =
  sum_{t : L_t > ell} sum_{b=0}^{L_t-ell-1}
       (X[t,b,i]-mu_it)(X[t,b+ell,i]-mu_it)
  / sum_t (L_t-ell)_+.

Y_i(ell) = C_i(ell) / max(v_i, 1e-12).
```

`Y_i` is the fixed `[8]` future autocorrelation-profile target.  Trial-mean
centering removes between-trial rate shifts; normalizing by future within-trial
variance makes the target a temporal-shape profile instead of merely a firing
rate target.  It is calculated directly from raw count pairs and never invokes
P20 anchors, slots, filters, a carrier normalizer, a target label, a behaviour
array, or a learned encoder.

For a unit with `v_i <= 1e-12`, or no valid pair at any listed lag, its entire
target row is `undefined`, not zero-filled.  A session is valid only if all
eight lags are defined for at least 90% of its units.  The result must record
the per-lag pair exposures and defined-unit fractions.  This rule keeps the
target legal for variable-length sessions without silently favouring a subset
of channels.

## 6. Source-LOSO readout and equal-session statistic

There are exactly seven outer source-LOSO folds.  In fold `s`:

1. build P20 and B20 independently from the six train-session supports and
   the left-out session's first-24 support;
2. fit a separate width-20 ridge multi-output readout from carrier rows to the
   eight-dimensional target rows using **only defined train-session unit rows**;
3. estimate feature and target mean/scale only from those same train rows;
4. apply those frozen transforms to left-out carrier and target rows; and
5. score every defined left-out unit × lag element.

The ridge penalty is fixed at `1.0`; the intercept is unpenalized. Features are
z-scored columnwise using only the six train sessions in the fold; any feature
SD at or below `1e-12` is replaced by `1.0`. Targets are mean-centred using
only those train rows and are not variance-scaled. The readout is the direct
float64 solve

```text
W = (Z^T Z + 1.0 I)^(-1) Z^T (Y - mean_train(Y)),
```

implemented with a linear solve rather than an explicit matrix inverse; the
train target mean is restored at prediction. No penalty, lag, P20 constant, carrier width,
target, normalizer, or session inclusion can be selected using this run.

For arm `A` and session `s`, use the usual unbounded but interpretable
per-session multi-output score

```text
R2_A,s = 1 - RSS_A,s / TSS_s,
```

where both sums are over that session's identical defined target elements and
the TSS baseline is the fold-train target mean.  `TSS_s <= 0`, a nonfinite
prediction, or different P20/B20 target masks is an execution failure.  The
primary content effect is

```text
Delta_B,s = R2_P20,s - R2_B20,s.
```

Sessions, not units, lags, resamples, or permutations, are the only
biological/inferential units.  Report all seven deltas, mean, median, sample
SD, the paired Student-t two-sided 95% CI (`df=6`), and

```text
MDE80 = (t_0.975,6 + t_0.80,6) × SD(Delta_B) / sqrt(7).
```

MDE is a sensitivity receipt, not a claim that seven sessions identify a
universal effect size.  Bootstrap, per-unit, or per-lag intervals may be
descriptive only and cannot replace this equal-session interval.

For the chronology test use the bounded fold utility

```text
U_s = TSS_s / (TSS_s + RSS_P20,s) = 1 / (2 - R2_P20,s),
H = mean_s U_s.
```

It prevents a single finite but catastrophically negative R² null draw from
dominating an otherwise valid randomization statistic.  Raw per-session R²,
RSS, TSS, and `U_s` remain mandatory outputs; `H` is not reported as R².

## 7. Common-bin time-order null

The only mechanism null is a common-bin chronological permutation of P20
support.  For each of the seven source sessions and each of the first 24
support trials, draw a bijection of that trial's valid bin indices and apply
the **same** bijection to all units/channels in the trial.  Padding is never
permuted.

Thus every schedule preserves exactly:

- each unit/trial total count and exposure;
- each unit/trial multiset of raw bin counts;
- all P20-independent B20 coordinates; and
- each instantaneous population count vector and therefore zero-lag
  cross-unit synchrony.

It destroys only the common ordering of population bins within every support
trial.  The permuted support is passed end-to-end through causal EWMA, source
anchor fitting, routing, train-only normalization, and the fold ridge readout.
The future autocorrelation target and every future bin remain untouched.

The primary test is exactly 4,095 independently seeded global schedules plus
the chronological observation.  A schedule contains all seven sessions and
all 24 trial permutations.  Sampling is with replacement: duplicate and
identity schedules remain, with their multiplicities logged.  A non-bijection,
padding touch, source/hash mismatch, nonfinite value, or failed fold aborts
the entire audit rather than being redrawn, omitted, or replaced.

The Monte-Carlo p-value is frozen as

```text
p_time = (1 + #{b in 1..4095 : H_b >= H_observed}) / 4096.
```

Ties count against P20.  The receipt must include all schedule seeds/hashes,
identity and duplicate counts, all fold scores, and the checksum of sorted
`H_b`.  It must also report a one-sided 97.5% binomial upper confidence bound
for the underlying null exceedance probability.  That bound measures
Monte-Carlo error only; it is not an eighth population-level confidence
interval.

There is intentionally no session-keyed slot-permutation mechanism test in
this M2 protocol.  M1 Gate A2 must establish the shared-coordinate premise
first.  Adding a second large randomization family here would make M2 an
uncontrolled retry after a negative/inconclusive M1 result, and it is not
needed to answer whether temporal ordering contributes beyond B20.

## 8. State, permutation, compute, and no-label contracts

All arms are label-free.  The executor must fail if an imported data object
exposes/reads target location, object ID, velocity, EMG, behavioural covariates,
or query labels.  The audit's data adapter may expose only raw trial counts,
valid lengths, trial boundaries, session ID, path, and source hash.

### Unit-permutation equivariance

Before an execution, a pure-array contract test must draw at least 100 seeded
unit-row permutations on synthetic variable-length integer trials.  It must
verify, to numerical tolerance, that applying a row permutation before P20/B20
construction merely applies that same permutation to P20/B20 rows and to the
future autocorrelation target.  It must separately verify that source anchor
values are invariant to source-unit and source-session presentation order.
The test may not call an NWB loader.

### Deployment state and MAC receipt

For M2's nominal `N=96` channels, P20's FP32 deployed state is exactly:

| state component | scalar count |
|---|---:|
| four causal filter values / unit | 384 |
| four slot counts / unit | 384 |
| four slots × four filter sums / unit | 1,536 |
| rate mean/second-moment/exposure control / unit | 288 |
| **total** | **2,592** |

That is `10,368` bytes at FP32.  The four shared rank-four anchors add only 16
scalars (64 bytes) and are not per-unit state.  Retained raw support matrices
are exactly zero.  Per unit per bin, the frozen accounting is 25 scalar
multiplications, 37 additions, and 3 comparisons.  The receipt must report
these numbers and separately report all offline-only anchor workspaces; it may
not present the latter as deployed state.

## 9. CPU execution budget and prelaunch receipt

This gate is CPU-only.  `CUDA_VISIBLE_DEVICES` must be empty and a prelaunch
receipt must prove no CUDA/torch accelerator initialization.  It may use at
most 16 CPU workers.  Before the full audit, an actual no-GPU benchmark at the
manifest dimensions must project the complete 4,095-schedule run, including
feature extraction and all seven folds, to finish within 24 hours.  If it does
not, the protocol is declared `fixed_null_budget_infeasible_stop`; lowering
the null count, modifying seeds, changing the target, or extending only a
near-significant run requires a new protocol.

The prelaunch receipt must bind the M1 Gate A2 pass artifact/hash, the M2
source manifest/hash, code hashes, constants, seeds, raw-boundary assertions,
output directory, target formula, carrier semantics version, solver contract,
and expected `N`/target widths.  A fresh output directory is mandatory;
existing results and all M1 results are immutable provenance and cannot be
inputs except for the Gate A2 pass receipt.

## 10. Frozen decision and kill criteria

The only success label is `pass_for_separate_decoder_review`; it does **not**
authorize a decoder run.  Every condition below is required:

1. **Beyond high-order marginals:** all seven `Delta_B,s` values are positive,
   mean `Delta_B >= +0.030 R²`, paired 95% lower bound is strictly positive,
   and `MDE80 <= 0.030 R²`.
2. **Temporal order matters:** `p_time < 0.025`, its one-sided 97.5% MC upper
   bound is also `< 0.025`, chronological `H` exceeds the time-null median,
   and at most one source session has `U_observed,s` no greater than its own
   time-null median.
3. **Validity:** all source/hash/raw-count/fold/mask/finiteness/common-width/
   no-label/no-CUDA/state/unit-permutation receipts pass; every session reaches
   the 90% defined-unit target threshold.

Any failure has the indicated terminal disposition:

| failure | terminal label and consequence |
|---|---|
| P20 does not beat B20 | `marginal_baseline_not_beaten_stop`; no decoder or P20 tuning |
| chronology does not beat common-bin null | `temporal_order_not_distinguishable_stop`; no temporal-mechanism claim or decoder |
| CI/MDE does not meet precision gate | `precision_insufficient_stop`; do not add sessions post hoc or relax the SESOI |
| data, label, mask, state, permutation, or finite-value contract fails | `invalid_execution_stop`; no numerical interpretation |
| 4,095 nulls cannot meet fixed CPU budget | `fixed_null_budget_infeasible_stop`; no count reduction rescue |

No stop outcome permits changing `K`, rank, EWMA constants, target lags,
autocorrelation normalization, ridge, B20, the statistic, schedule count,
seeds, support budget, session subset, or output filtering.  It also cannot
open a GPU, a decoder training run, a held-out/minival/formal file, EvalAI, or
quantization branch.  A pass only permits a **new**, independently reviewed
M2 decoder-development protocol with a separate source/development boundary.

## 11. Required outputs if separately authorized

The executor must write a new directory containing:

- immutable prelaunch and source manifests with hashes;
- per-session P20/B20 raw `RSS`, `TSS`, `R²`, target-mask counts, lag-pair
  exposures, and `Delta_B`;
- the equal-session CI/MDE receipt;
- all common-bin schedule seeds/hashes/multiplicities and raw `H_b` values;
- chronological and null per-fold utilities, plus their sorted checksum;
- raw-boundary, no-label, no-CUDA, unit-permutation, state/MAC, anchor-source,
  and future-target-isolation assertions; and
- one machine-readable terminal decision JSON.

No report may substitute a pooled unit-level score for the seven-session
inference, call the oracle proxy behavioural R², or call this a held-out test.
