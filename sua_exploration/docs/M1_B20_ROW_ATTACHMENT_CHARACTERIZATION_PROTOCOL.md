# M1 B20 row-attachment and component-characterization protocol

**Status:** protocol only. No source file, NWB, GPU, held-out partition, decoder, or EvalAI
endpoint has been opened or run for this protocol. This is a new, narrow **source-only
characterization** of an already-seen M1 source set. It does not revise, rescue, or replace
Gate A2's `marginal_baseline_not_beaten_stop` decision.

## 1. Question, scope, and non-claims

The immutable Gate A2 artifact
`results/m1_fixed_k_temporal_prototype_gate_a2_v1/source_gate_a2.json`
(`SHA-256 379f3c3b85fe120735802af887d8668fc4d84c3e89cb77acdaabecd485565981`) found:

| Carrier | Mean R2 | Per left-out source session |
| --- | ---: | --- |
| B20 marginal distribution | 0.904757650 | 0.892449, 0.942571, 0.963411, 0.820599 |
| P20 temporal prototype | 0.862794286 | 0.880273, 0.891359, 0.920905, 0.758640 |
| rate-only | 0.762268600 | 0.774948, 0.794927, 0.826736, 0.652464 |

Thus B20 minus rate-only was `+0.142489050` (positive 4/4; paired 95% CI
`[+0.108828,+0.176150]`; MDE80 `0.044507`) and P20 minus B20 was negative in all four
sessions. Gate A2 was correctly a P20-versus-B20 test and stopped before its P20 mechanism
nulls; it did not test B20's provenance.

This protocol asks one narrower question:

> Is B20's source-proxy signal attached to its channel row, and does it derive from sorted
> ten-trial log-rates, pooled-bin quantiles, or both?

It can answer that question only as an exploratory characterization of these four
already-inspected M1 sessions. It cannot establish a deployment effect, behavioural decoding
effect, frozen-decoder compatibility, formal held-out effect, or GPU authorization. Even a
complete pass has `gpu_authorized = false` and `decoder_authorized = false`. Any validation
must first be specified separately on data not used here.

## 2. Immutable data and oracle boundary

If separately root-authorized, a fresh manifest shall bind exactly the four hash-bound M1
held-in-calib source files used by A2, in this order:

```text
ses-20120924, ses-20120926, ses-20120927, ses-20120928
```

The current four-source manifest is only the checksum reference:
`manifests/m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json`
(`SHA-256 da3a6397a0739ddca09e999c8a2a64fc968d3b8ca860e96d6e600570bd624902`).
The execution manifest must reject recursive discovery and every minival, held-out,
formal-test, report, EvalAI, decoder/checkpoint, CUDA, and GPU path. It uses raw,
non-interpolated support trials `[0,10)` only: `smooth_calibration=False`,
`interpolate_trials=False`, pad value `-1`, no calibration intertrials. Valid prefixes must
be finite non-negative integer counts, padded tails exactly `-1`, and valid-prefix sums equal
production `calib_trial_spike_sums`. Every source must have exactly 64 channel rows; any
mismatch is invalid, not an excuse to sort, truncate, or pad.

The only future information is the frozen scorer target in `[210,end)`:

```text
Y[channel, object] = log1p(mean over future trials of object
                           (raw_count / (valid_bins * 0.020 s))).
```

Future object labels select entries of `Y` inside the scorer and nowhere else. They cannot
enter B20 construction, normalizers, permutations, random seeds, component choice, plots, or
early stopping. The runner API must expose this: carrier/permutation constructors receive no
target argument; only `score_fixed_carriers(..., Y)` receives one. The endpoint is a
later-neural source-LOSO proxy, never behavioural R2.

The previous A2 JSON is not an input to a carrier, normalizer, null, or scorer. Its values are
an immutable reproduction checkpoint only: the fresh unpermuted B20 computation must reproduce
all four A2 B20 R2 values to absolute tolerance `1e-10`, or emit
`a2_b20_reproduction_mismatch_invalid_stop` and stop.

## 3. Exact B20 decomposition and common estimator

For row `i`, trial `t in {0,...,9}`, valid-bin count `x[t,b,i]`, and valid length `L_t`, define

```text
r[t,i] = log((sum_b x[t,b,i] + 0.5) / (0.020 * L_t))
R_i    = sort_ascending(r[0,i], ..., r[9,i])                         # 10 coordinates
Q_i    = quantile_linear({log1p(x[t,b,i]) over valid t,b},
                         p=(.05,.15,.25,.35,.45,.55,.65,.75,.85,.95)) # 10 coordinates
B_i    = [R_i, Q_i]                                                   # B20, width 20
```

`quantile_linear` is explicitly NumPy `method="linear"`; no library default is permitted.
R and Q use no labels, trial/bin/channel identity, anchor, temporal filter, target, or future
information. B20 is invariant to support-trial and within-trial-bin order.

Every arm below is a 64-by-20 matrix. There are four outer source-LOSO folds. In each fold,
coordinate normalization, intercept, and ridge readout with A2-locked `lambda=1.0` are fit
only on the other three sessions. The target, folds, source baseline, normalization rule,
coefficient width, and regularization are otherwise identical in every arm.

The conditional component controls in section 4 retain all 20 B20 coordinates and the same
20 ridge coefficients while breaking only a complete ten-coordinate block's association with
a channel row. Those *conditional nulls* are therefore width- and coefficient-parameter
matched.

This does **not** make standalone R10 or Q10 parameter-matched to B20. When a downstream API
requires a width-20 matrix, a direct R10/Q10 descriptive carrier is represented respectively
as `[R,0_10]` or `[0_10,Q]`; its effective rank is ten, the zero coordinates carry no fitted
information, and it must never be described as a width/parameter-matched comparator. R10/Q10
exist here only as simpler descriptive candidates for a later independently preregistered
validation. The present component decision is based on the full-width conditional nulls, not
on a false equality of effective degrees of freedom.

## 4. Robust row-attachment randomizations

Replicate `b=0` is the one canonical identity arrangement. For every randomized replicate
`b in {1,...,4095}` and source session `s`, draw a uniform permutation `pi[b,s]` of the 64 row
indices. Permutations are independent over the four sessions. Thus every family has exactly
one identity observation plus 4,095 deterministic random schedules, or 4,096 scored rows in
total. The randomized schedules are sampled with replacement: an identity permutation or a
duplicate random schedule remains in the table. The same random schedule table is used in
every family. The receipt records every seed, permutation checksum, identity/duplicate count,
and full 4,096-row score-matrix checksum. A non-permutation, row-count change, nonfinite value,
or seed/hash mismatch aborts; nothing is redrawn or silently discarded.

For each session B20 matrix `[R,Q]`, define three complete-block transformations:

```text
A-all(b,s) = [ R[pi[b,s]], Q[pi[b,s]] ]   # detach complete 20-D B20 row
A-R(b,s)   = [ R[pi[b,s]], Q            ] # detach only sorted trial-log-rate block
A-Q(b,s)   = [ R,            Q[pi[b,s]] ] # detach only pooled-bin-quantile block
```

The target rows, raw trials, future labels, normalizers, and row order in every other object
are never permuted. Each ten-coordinate source block moves whole: scalar-coordinate shuffles
are prohibited. Thus each null keeps the exact per-session feature distribution, width, and
coefficient count but tests whether the block must remain attached to the channel whose later
target is scored. A single shuffle, raw-R2 mean, hand-picked seed, or deleting a catastrophic
finite score can never support an inference.

## 5. Fixed statistic, multiplicity, and execution order

Raw R2 is mandatory output, but the randomization rank uses A2's bounded utility:

```text
U_s = TSS_s / (TSS_s + RSS_s) = 1 / (2 - R2_s)
H   = mean over the four source-LOSO sessions of U_s.
```

`TSS_s <= 0`, nonfinite values, or negative RSS are invalid. Bounded H prevents a finite
catastrophic row permutation from dominating an ordinary mean; it does not make schedules or
target objects independent biological sessions.

For family `j in {all,R,Q}`, let `H_B` be the canonical replicate-0 score and count only
randomized replicates `b=1,...,4095`:

```text
p_j = (1 + count_{b=1..4095}[H_j(b) >= H_B]) / 4096,
```

with ties against B20. Report the canonical replicate-0 values and every randomized
replicate-1..4095 per-fold U/R2/RSS/TSS value (4,096 score rows per family), plus the 97.5%
one-sided binomial upper bound on the Monte-Carlo exceedance probability. This interval is
simulation uncertainty only. Three directional families share Bonferroni
`alpha = 0.05 / 3 = 0.0166666667`. A family is distinguishable only if both its p-value and
its 97.5% upper bound are strictly below alpha, and canonical U exceeds that family's median
in every one of the four sessions.

Run in this immutable compute-saving order:

1. validate all data/oracle/width contracts, reproduce A2 B20, and score canonical B20;
2. score all 4,095 `A-all` schedules;
3. only if A-all passes, score every one of the 4,095 `A-R` and 4,095 `A-Q` schedules;
4. compute target-free reliability receipts in section 7.

If stages 1 or 2 fail, later stages are `not_run_due_to_predeclared_early_stop`. If stage 3
starts, both component families must finish. No schedule count, seed namespace, statistic,
loss clipping, source-session subset, or permutation rule may be changed after an outcome.
The no-NWB prelaunch must benchmark all 12,285 randomized schedule scores plus the one
canonical score under a fixed CPU-worker cap and a 24-hour CPU-only ceiling. If that
worst-case path cannot meet the ceiling, this protocol is not run; it is not reduced after
seeing an intermediate rank.

## 6. Component attribution and forced simplification

For each session calculate robust permutation-median raw-R2 losses:

```text
D_R,s = R2_B,s - median_b R2_A-R(b,s)  # loss when R attachment is broken
D_Q,s = R2_B,s - median_b R2_A-Q(b,s)  # loss when Q attachment is broken
```

These are sensitivity summaries, not biological p-values. The practical `0.030 R2` amount is
inherited from A2's predeclared content threshold rather than selected post hoc.

| Result pattern | Permitted conclusion and required action |
| --- | --- |
| A-all fails | `b20_row_attachment_not_distinguishable_stop`: no row-attached B20 claim. |
| A-all passes; **both** A-R and A-Q are not distinguishable; and every `D_R,s < 0.030` **and** every `D_Q,s < 0.030` | The two source blocks are redundant or their provenance is unresolved in this source proxy. Retire the composite B20 claim. By a fixed hardware prior—not by score magnitude—retain **R10 only** as the lower-state candidate: it needs ten trial totals/exposures and no pooled 256-bin count histogram. It still requires separate independent validation. |
| A-all passes; **only** A-Q is not distinguishable and every `D_Q,s < 0.030` | Q has no practically important unique row-attached value conditional on R in this source proxy. Retire the composite B20 claim and retain only simpler 10-coordinate R as a candidate for separate independent validation. |
| A-all passes; **only** A-R is not distinguishable and every `D_R,s < 0.030` | R has no practically important unique row-attached value conditional on Q. Retire composite B20 and retain only simpler 10-coordinate Q as an independent-validation candidate. |
| A-all, A-R, and A-Q all distinguishable; all `D_R,s >= 0.030` and all `D_Q,s >= 0.030` | Both blocks have practically material, conditionally row-attached source-proxy value. Only a narrow composite-B20 source-characterization statement is permitted. |
| A-all passes but no preceding component rule applies | `b20_component_attribution_indeterminate_stop`: do not promote full B20. |

Failure to reject a component family alone is never called equivalence: the all-session
`D < 0.030` rule is what forces simplification. The both-negligible row has precedence over
the one-block rows. Its R10 choice is a pre-data hardware/state prior, not a comparison of
post-result scores: R10 requires trial totals/exposures, whereas Q10 requires maintaining a
pooled within-bin count distribution. Conversely one session, a mean, one seed, or reliability
cannot make a component necessary. This protocol intentionally chooses the simpler descriptor
whenever B20's extra ten coordinates have no predeclared practical incremental value.

## 7. Target-free measurement reliability

Reliability is a frozen diagnostic only: it uses neither Y, future labels, ridge fitting, nor
source-LOSO scores; it is not a multiplicity family, a gate, a stopping condition, or an
authorization criterion. It can only qualify measurement stability; it never rescues a
row-attachment failure or authorizes GPU.

### 7.1 Spike thinning

For exactly 1,024 deterministic seed pairs (no adaptive extension or truncation) and each
valid count x, draw
`x_A ~ Binomial(x,0.5)` and set `x_B=x-x_A`. Build B20 from intensity-scaled counts `2*x_A`
and `2*x_B`, retaining original exposure in the trial-rate definition. The halves retain the
same bin support and have expected full intensity. No bin is discarded and no future value
chooses a seed.

### 7.2 Odd/even bins

Within every valid support trial, build one carrier from bin indices `0,2,4,...` and the other
from `1,3,5,...`. Each has its actual exposure `0.020 * L_half` in R. Q uses native 20-ms
counts from its retained bins, with no factor-of-two scaling of a per-bin marginal. An empty
parity half invalidates the run rather than being filled or interpolated.

### 7.3 Receipt

For B20, R, and Q separately, center/scale every coordinate over 64 rows using the pooled
two-split mean and standard deviation. For every row with nonzero finite norm in both halves,
compute the cosine between its two standardized vectors. Report, per source session:

- all 1,024 thinning median-row cosines and their 2.5%, 50%, 97.5% quantiles, plus the minimum
  defined-row fraction;
- odd/even median-row cosine and defined-row fraction; and
- per-coordinate two-split Pearson over 64 rows, marking undefined coordinates explicitly.

The 1,024 thinnings are technical resamples, not independent sessions. The pooled-split
standardization, per-coordinate Pearson values, and defined-row fractions above are frozen
reporting rules; they cannot be replaced by a favorable normalization, summary, or threshold.

## 8. Audit requirements and prohibitions

The no-NWB receipt must bind protocol/manifest/code hashes, seed namespaces, fresh output,
source-only confirmation, CPU cap, and CUDA-disabled environment. On synthetic fixtures it
must prove B20 trial/bin-order invariance; complete-block marginal preservation; no target
argument in carrier/null functions; legal train-only normalization; rank and Monte-Carlo-bound
arithmetic; and thinning/parity support conventions.

The final immutable receipt records contracts, A2 reproduction, full raw score matrices,
permutations, component losses, reliability, runtime, and one terminal decision. It must state:

```text
source_only = true
M1_source_already_seen = true
formal_held_out = false
decoder_authorized = false
gpu_authorized = false
EvalAI_authorized = false
```

No result permits changing support size, quantiles, pseudocount, ridge, target window,
normalizer, schedule count, source subset, or permutation; adding labels, MLP/FiLM, decoder,
quantization, or a GPU job; or using formal held-out data as rescue. Its value is to reveal
whether a simpler ten-coordinate B20 component explains the already-observed source proxy,
without laundering that result into a deployment claim.
