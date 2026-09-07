# M1 fixed-K temporal prototypes — independent source-only Gate A2

**Status:** protocol only. No NWB has been opened and this document authorizes neither an
execution nor a GPU job. It is a new experiment; it does not revise, rescue, or replace the
failed v3 Gate A.

**Purpose.** v3 found a large source-LOSO later-neural-oracle advantage for the fixed-K
prototype over its simple rate/exposure control, but a single session-keyed slot shuffle gave a
heavy-tailed, statistically unresolvable mechanism contrast. Gate A2 asks a stricter question:
does the prototype beat a width-matched nonlinear marginal-distribution baseline, and is its
advantage specifically dependent on (i) a common slot coordinate and (ii) chronological
within-trial order?

Only a complete pass permits preparation of a separately reviewed one-cell decoder GPU protocol.
It never permits a decoder run, a held-out report, EvalAI, a parameter sweep, or a behavioural
claim by itself. A failure or an inconclusive result closes this branch: no changes to K, rank,
EWMA bank, ridge penalty, feature width, test statistic, null count, or target can be used to
rescue it.

## 1. Why this independent test is necessary

The v3 result remains useful but narrow:

- prototype minus rate-only was +0.100526 R2, positive in all four source-LOSO sessions;
- within-trial thinning repeatability was high;
- the single whole-slot-shuffle contrast was also positive in sign, but two shuffled folds had
  catastrophic negative R2 and its paired-t interval crossed zero.

It does **not** establish a reusable coordinate mechanism. Plausible alternatives are:

1. v3 rate-only was too weak: prototype slots may encode nonlinear rate, zero mass, dispersion,
   exposure, or a support-trial count distribution without temporal order.
2. The future neural target comes from the same recording session as support. Long-lived channel
   gain, recording quality, and stationarity can predict it without improving behaviour decoding.
3. Raw R2 is unbounded below. One particularly disruptive permutation can dominate an ordinary
   mean/SD despite not being the relevant alignment question.
4. The causal EWMA representation may use chronology, or merely the marginal distribution of
   bin counts. v3 did not separate them.

Gate A2 tests exactly these four alternatives and nothing about SUA, pseudo-MUA, T4, or official
FALCON performance.

## 2. Immutable source and oracle boundary

If separately authorized, a fresh manifest may name only the same four hash-bound M1
held-in-calib source files used by v3:

    ses-20120924, ses-20120926, ses-20120927, ses-20120928.

It must lock paths, SHA-256 values, support [0,10), future scorer window [210,end), channel/target
counts, code hashes, constants, all seeds, and a fresh output directory. It must reject recursive
discovery and every minival, held-out, formal-test, report, and EvalAI path. v3 outputs cannot
enter the new run as data or as a selection source.

Trialization stays raw and noninterpolated: smooth_calibration=False,
interpolate_trials=False, pad_value=-1, and no calibration intertrials. Valid prefixes
must be finite nonnegative integer counts, padded tails must be exactly -1, and each prefix
sum must match production calib_trial_spike_sums. Temporal state resets per trial.

The only future-label use is the frozen scorer-only target:

    Y[channel, object] =
      log1p(mean_future_trials_of_object(raw_count / (valid_bins * 0.020 s))).

Future object labels cannot enter a carrier, anchor, normalizer, permutation, ridge fit, state,
or model selection. The sole endpoint is a **source-LOSO later-neural oracle proxy**, not
behavioural decoding R2.

## 3. Common estimator and inference unit

There are exactly four outer source-LOSO folds. In fold s, anchors, feature normalization, and
the width-20 ridge readout are fit solely from the other three sessions' complete first-ten raw
supports. The frozen canonical representation is v3's K=4, rank r=4, causal EWMA bank
(0.5, 0.25, 0.125, 0.0625), deterministic source/unit-order-invariant anchors, hard routing,
and one manifest-fixed ridge penalty. The left-out session never fits a fold object.

All candidate and control features are width 20 and use the identical target construction,
normalization rule, ridge degrees of freedom, folds, and output scorer. Nonfinite features,
normalizers, anchor fits, targets, or predictions are a hard execution failure. A finite poor
prediction is not discarded.

Outer sessions are the only biological/statistical units (n=4). Bins, target objects,
permutations, and thinning repetitions are not additional independent sessions.

## 4. Frozen arms

### 4.1 P20 — canonical temporal prototype

P20 is exactly the v3 neural-only prototype: four routed complete slot blocks, each carrying
its count and rank-vector, for width 20. It remains causal/no-backprop and has v3's streaming
state and unit-permutation contracts.

### 4.2 B20 — nonlinear marginal-rate/distribution baseline

B20 is the primary comparator, not v3's weaker rate-only arm. For a channel and its ten
support trials, define:

    r_t = log((sum_b x[t,b] + 0.5) / (0.020 * L_t)),

where L_t is valid-bin exposure. Its exactly 20 coordinates are:

1. sort(r_0,...,r_9), ten ascending trial log-rates; and
2. ten empirical quantiles of pooled valid-bin log1p(x[t,b]) at probabilities
   (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95), with one frozen quantile
   interpolation rule.

It therefore represents nonlinear firing rate, exposure-normalized trial heterogeneity, zero
mass, burstiness, and the marginal within-bin count distribution. It is exactly invariant to
support-trial order and to bin order within every trial. It uses no labels, slot, anchor, temporal
filter, bin/trial index, channel ID, or future information. Legacy rate-only/D4 outputs may be
reported only descriptively; neither can pass Gate A2.

### 4.3 Slot-coordinate null P20-slot(pi)

Construct canonical P20 first, then apply a permutation to a **whole** slot block (its count
and rank-vector move together) for every channel of a given session. There is no scalar/row
shuffle and no redefinition of an anchor.

A global common permutation applied to all four sessions is only a shared ridge-column renaming:
train-only column normalization covaries with the renaming and isotropic ridge is prediction
invariant. It is therefore an exact gauge symmetry, not new evidence. Gate A2 fixes
ses-20120924 as the gauge/reference identity and enumerates only the other sessions'
relative assignments:

    G_rel = S4(ses26) x S4(ses27) x S4(ses28), |G_rel| = 24^3 = 13,824.

For each configuration, all three train sessions and the left-out session receive their assigned
complete-block permutation; the fold ridge is legally refit. This quotient is equivalent to
enumerating (S4)^4 after removing the 24-fold global-column permutation redundancy. The
all-identity relative configuration is the observed canonical carrier.

### 4.4 Temporal-order null P20-time(rho)

For every source session and each of its ten support trials, draw a bijection of valid time-bin
indices and apply the **same** bijection to all channels of that trial. This preserves exactly:

- each channel/trial exposure and total spike count;
- each channel/trial multiset of bin counts; and
- every population bin-vector, hence instantaneous cross-channel synchrony.

It destroys only their chronological order. Padding is excluded. The permuted supports are
processed end to end: EWMA, anchors, routing, train-only normalization, and ridge readout are
refit in their legal outer folds. The future target is untouched.

The primary temporal test uses exactly 4,095 independently seeded global schedules plus the
unpermuted observation. A schedule includes all four sessions and all ten trials, with each trial
permutation sampled uniformly from its full symmetric group (identity is allowed). Sampling is
with replacement: an identity or duplicate schedule is retained and reported with its
multiplicity, because replacing it after inspection would invalidate the Monte-Carlo law. A
non-bijection, padding touch, feature-fit failure, or seed/hash mismatch aborts the entire
execution rather than being redrawn or dropped.

An independently per-channel/per-trial time shuffle may be output as a destructive diagnostic,
but it removes synchrony and is non-primary; it cannot replace this common-bin null.

## 5. Precommitted robust randomization statistic

Raw R2 equals 1 - RSS/TSS and is unbounded below. Instead each fold uses the bounded, monotone
predictive utility:

    U_s = TSS_s / (TSS_s + RSS_s) = 1 / (2 - R2_s), U_s in (0, 1].
    H = mean_s U_s.

TSS_s and RSS_s are computed over precisely the frozen target elements in that fold.
TSS_s <= 0 is a hard contract failure. This utility has a direct loss interpretation: it is
the fraction of total-plus-residual error attributable to target variance, and it maps a
catastrophic R2 smoothly toward zero instead of allowing -104 to carry 100 times the weight of
-1. It has no arbitrary hard clipping threshold. Raw per-session R2, RSS, TSS, U, and
unclipped R2 mean/median remain mandatory outputs; H alone must not be described as an R2 gain.

### 5.1 Exact relative-slot test

For every g in G_rel, calculate H(g). With e the identity configuration:

    p_slot = #{g : H(g) >= H(e)} / 13,824.

Ties count against the candidate. This is exact over relative session coordinates: no raw-R2
paired t-test, MDE from a heavy-tailed slot null, outlier deletion, or alternative shuffle may
replace it. The implementation also retains the complete `[13,824,4]` matrix of per-fold bounded
utilities. For every left-out session `s`, compare its canonical `U_s(e)` with the median of
`U_s(g)` over the complete quotient; all four canonical values must be strictly above their own
null median. This is the session-consistency gate and prevents the global `H` rank from being
driven by one fold.

A 24-way canonical-other-identity conditional rank, including the equivalent re-gauged orbit for
the reference session, may be reported descriptively. Ranks over every possible conditioning
stratum are not a gate and are not required; that stronger-looking rule has no predeclared
population interpretation and would make the result depend on thousands of correlated slice
tests.

### 5.2 Monte-Carlo temporal-order test

Let a_0 be chronological data and a_1 through a_4095 the logged time schedules. The frozen
conservative randomization p-value is:

    p_time = (1 + #{b in 1..4095 : H(a_b) >= H(a_0)}) / 4096.

The receipt records all seeds, schedule hashes, duplicate/identity multiplicities, all raw
per-fold scores, and a checksum of sorted H. It additionally reports the one-sided 97.5% binomial
upper confidence bound for the Monte-Carlo exceedance probability. This is MC-sampling uncertainty,
not uncertainty over biological sessions.

## 6. Content effect, multiplicity, uncertainty, and runtime limits

The primary content estimand is:

    Delta_B,s = R2_P20,s - R2_B20,s.

Report all four values, mean, median, SD, two-sided paired Student-t 95% interval, and:

    MDE80 = (t(0.975,df=3) + t(0.80,df=3)) * SD_session / sqrt(4).

This is a sensitivity receipt, not artificial sample-size expansion. With four sessions a useful
exact sign test cannot establish a small-alpha effect, so Delta_B is a predeclared practical
gate, not a third formal p-value.

p_slot and p_time are the two formal directional mechanism tests. They are one family and
use Bonferroni: both must be strictly below 0.025; for time, its 97.5% MC upper bound must also
be below 0.025. All other arms, plots, losses, subsets, and diagnostics are descriptive and
cannot rescue either test.

The no-NWB prelaunch receipt must benchmark the implementation on the conservative frozen raw
contract dimensions (four sessions, ten trials, 64 channels, and the maximum 1,024 valid bins per
trial) and report a transparent operation/runtime projection. The real source dimensions are
recorded only after reviewed execution begins; the prelaunch must not open NWB merely to improve
its estimate. The receipt must prove a deterministic execution plan with at most 8 CPU workers for
slot enumeration and at most 16 total CPU workers for time schedules. Slot work is divided into disjoint lexicographic chunks
whose union is exactly 13,824 configurations; no configuration is omitted or duplicated. The
maximum wall time is 24 hours for the complete all-gates-pass execution path, including both
nulls, with no CUDA visible. If the fixed 13,824 / 4,095 design cannot complete under those limits,
Gate A2 is not run; reducing the count, changing seeds, or conditionally extending a
near-significant run requires a new protocol.

The conjunction is evaluated in a predeclared compute-saving order:

1. build and validate the canonical P20/B20 carriers and evaluate the section-7 content,
   precision, raw/oracle and repeatability prerequisites;
2. only if stage 1 passes, enumerate all 13,824 relative-slot configurations;
3. only if stage 2 passes, execute all 4,095 random time schedules.

A failed earlier stage writes a complete immutable stop receipt and marks later stages
`not_run_due_to_predeclared_early_stop`. It is not missing data and cannot be followed by running
only the skipped stage. Conversely, a stage that starts must finish its entire frozen null count;
partial ranks or conditional stopping inside a null are forbidden.

## 7. CPU gate and kill criteria

All conditions below are necessary for gate_a2_pass_cpu_only, which only allows preparation of
a separately reviewed decoder GPU pilot:

1. **Beyond marginal distribution.** All four Delta_B,s values are positive; its paired 95%
   lower bound is at least +0.030 R2; and MDE80 <= 0.030 R2.
2. **Shared slot coordinate.** Exhaustive p_slot < 0.025; canonical identity's H exceeds the
   relative-slot null median; and for all four left-out sessions canonical `U_s` is strictly above
   the corresponding median across all 13,824 relative-slot configurations. A failed slot test
   means no shared-coordinate claim.
3. **Chronological information.** p_time < 0.025 and its one-sided 97.5% MC upper bound is
   <0.025; chronological H exceeds the time-null median; and at most one session has
   U_P20,s <= median_b U_time,s,b. A failed time test means no temporal-order claim.
4. **Validity.** All source/hash/raw-count/fold/oracle/finiteness/common-width/no-CUDA/fresh-output
   receipts pass. At least 90% of channels are prototype-defined in every left-out support. The
   v3 split-half thinning prerequisite is independently reproduced (both 2.5% cosine quantiles
   >= 0.5 and >=90% value-defined units in each session).

Any failure terminates the branch with the frozen label below:

| Condition that fails | Disposition |
|---|---|
| B20 content criterion | marginal_baseline_not_beaten_stop |
| exact slot criterion | slot_coordinate_not_distinguishable_stop |
| time-order criterion | temporal_order_not_distinguishable_stop |
| MDE/interval sensitivity | precision_insufficient_stop |
| any data/oracle/finiteness receipt | invalid_execution_stop |
| CPU budget cannot meet frozen count | fixed_null_budget_infeasible_stop |

No stop state permits K/r/router/FiLM changes, decoder/GPU work, held-out testing, or quantization.
A pass supports only the narrow neural-oracle statement; it does not show that frozen SPINT
decoding will improve.

## 8. Required no-NWB prelaunch audit

Before source access, an independent receipt must prove:

- exact manifest-indexed source loading; no datamodule setup, recursive discovery, minival/report/
  EvalAI/held-out imports, or decoder/checkpoint path;
- B20's exact 20 coordinates, quantile convention, and invariance to trial and within-trial-bin
  reorderings;
- P20's v3 state, width, causal reset, and unit-permutation contracts;
- complete-block slot movement, global-permutation prediction invariance, the fixed ses24 gauge,
  and exhaustive unique 13,824 relative configurations;
- common-bin time-permutation conservation of per-channel/trial totals and bin multisets,
  conservation of population bin vectors, padding exclusion, legal fold refits, schedule hashes,
  and duplicate/identity accounting;
- correct U/TSS/RSS arithmetic, exact p_slot, conservative p_time, and MC upper-bound calculation
  on synthetic fixtures;
- the fact that changing a finite R2 from -2 to -200 only smoothly decreases U toward zero, while
  an invalid estimator result aborts rather than being hidden by a robust statistic; and
- the stated CPU worker/wall-time ceiling and CUDA-disabled execution plan.

The root authorization must bind the receipt SHA, manifest SHA, code SHA, exact confirmation
phrase, and fresh output directory. A subsequent execution receives an independent result audit
before any document is updated with a conclusion.

## 9. Explicit prohibitions

Do not treat a v3 sign pattern, a different one-off shuffle seed, a utility result alone, or a
time-null catastrophe as GPU authorization. Do not tune a hyperparameter or add attention, FiLM,
labels, an MLP, a decoder, or extra support trials after seeing Gate A2. A negative Gate A2
result would still be scientifically useful: it would identify v3's source proxy signal as
marginal distribution/stationarity or an unstable slot effect, and would close the only remaining
plausible fixed-K route before it consumes formal held-out evaluations.
