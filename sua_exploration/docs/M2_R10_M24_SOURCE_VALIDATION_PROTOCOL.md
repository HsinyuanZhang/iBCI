# M2 R10 M24 — independent seven-source CPU validation protocol

**Status:** terminal source-gate stop. The first authorized source attempt (`source_gate_v1`)
stopped before any feature, target, or score because an unsigned `uint8` neural array could not
represent the production `-1` pad value. The independently reviewed mechanical v2 converted
validated counts to signed `int64`, then stopped at the frozen target prerequisite with
`target defined-row fraction below 0.90`. Both immutable terminal receipts are retained. Neither
attempt produced R10/L10 scores, reliability, or attachment-null values, so there is no R10 effect
estimate. This frozen M2 gate does not permit lag/mask/session rescue and authorizes no SUA, GPU,
decoder, local held-out, formal, EvalAI, or quantization work.

## 1. Motivation, without rewriting the M1 decision

M1 Gate A2's B20 decision is **indeterminate**, not a B20 pass. In particular, it must not be
relabelled as permission for cross-domain B20/P20 work. The reason to consider R10 is narrower and
new: before A2 was stopped, the predefined M10 trial-rate distribution component alone exceeded
the older rate-only carrier in all four source sessions by:

    +0.11275, +0.13576, +0.13186, +0.15447 R2.

Its difference from full B20 was only:

    +0.00476, +0.01188, +0.00482, +0.01366 R2,

and all conditional `A-R` block-attachment losses were at least 0.03. R10 split-thinning was also reliable. These
are motivation for a **new low-state marginal trial-rate hypothesis**, not evidence that R10 is a
protocol-approved simplification of B20, a full-B20 result, a temporal carrier, or a decoder
feature.

This protocol asks only:

> From the first 24 chronological unlabeled M2 trials, does a channel's ten-quantile distribution
> of trial log-rates predict its later label-free within-trial autocorrelation profile more
> reliably than a fair low-order rate control, and is the benefit attached to the correct channel
> row?

It does not test behavioural decoding, velocity/direction, T4/K4, P20, B3/SPINT, or a hardware
accuracy claim.

## 2. Immutable source, support, target, and label boundary

The only permitted data scope is the seven exact hash-bound M2 held-in source sessions already
listed by the existing M2 source-only temporal-order protocol. A new manifest must name their
paths and SHA-256 values; it must neither recursively discover data nor resolve a path containing
held-out, minival, test, formal, report, or EvalAI.

These seven held-in source sessions have been globally seen in earlier M2 analyses. This experiment
is independent of the M1 candidate-selection domain and remains source-only under its own frozen
boundaries, but it is **not** a pristine, hidden, or formal M2 confirmation set. It may not be
reported as such, and none of the already-viewed M2 held-out files may be opened.

The support and scorer boundary are exactly:

    support trials = [0,24)
    scorer-only future trials = [24,end)
    raw bin width = 20 ms

No target direction, velocity, EMG, object/location, reward, electrode/channel identity,
waveform/SNR, or query label may enter R10, L10, the attachment null, feature normalization, or
any deployed state. Future neural bins may enter only after those objects are sealed, solely as
the source-fitted/readout-scored oracle target defined in section 2.1; they can never alter a
carrier, null schedule, normalization, or calibration state.

The source-only runner must use the already defined raw M2 boundary:

    smooth_calibration = false
    interpolate_trials = false
    use_calib_intertrials = false
    pad_value = -1

Trial inclusion must be reconstructed only from `trials.start_time`, `trials.stop_time`, and the
shared M2 acquisition timestamps used as the 20-ms neural bin time base. The runner may read unit
spike times and those timestamps, but it must not read finger-velocity values or the NWB
`acquisition/eval_mask`. It constructs a geometry mask that is true on each half-open
`[start_time, stop_time)` trial interval and passes that mask, plus zero placeholder covariates, to
the production `FalconDataset` trializer. The receipt must bind the start-index, stop-index, and
geometry-mask checksums and prove that changing any unavailable behavior-side mask cannot alter
the trialized neural prefixes.

The runner must verify finite nonnegative integer valid prefixes, exactly -1 padding tails,
trial-boundary preservation, and equality of valid-prefix sums to production trial-count receipts.
A failure invalidates the run; no session/unit/mask can be selectively omitted.

### 2.1 Existing future target is a prerequisite, not a redesign opportunity

This protocol may execute only after an exact seven-path source manifest and raw source-only
loader are frozen. The future target need not already have Python code: a new pure implementation
is permitted, provided it is line-for-line the formula below, is synthetic-array tested before
source access, has a hash-bound code receipt, and makes no semantic redesign. The required target
is:

    lags = (1, 2, 4, 8, 16, 32, 64, 128) raw 20-ms bins
    target trials = [24,end)
    within every future trial subtract that trial/unit mean
    normalize exposure-weighted lag products by future within-trial variance

More exactly, for future raw counts X[t,b,i], with t >= 24 and valid length L_t:

    mu_i,t = mean_b(X[t,b,i])
    v_i = sum_t,b (X[t,b,i] - mu_i,t)^2 / sum_t L_t

    C_i(ell) =
      sum_{t:L_t>ell} sum_{b=0}^{L_t-ell-1}
        (X[t,b,i]-mu_i,t) * (X[t,b+ell,i]-mu_i,t)
      / sum_t max(L_t-ell, 0)

    Y_i(ell) = C_i(ell) / max(v_i, 1e-12).

It is the eight-dimensional per-channel target specified in
M2_FIXED_K_TEMPORAL_ORDER_SOURCE_GATE_PROTOCOL.md. It is materialized only after R10/L10 features
and attachment-null schedules are sealed, then serves solely as the source readout-training and
scoring target. It has no anchor/router/slot input and never feeds back into a carrier,
feature-normalizer, null schedule, or calibration state.

The pure target implementation must mark a channel row undefined when its future within-trial
variance is at or below 1e-12, or when even one required lag has zero valid-pair exposure. It must report
per-lag defined-row fractions and pair exposures. Every one of the eight lags must be defined for
at least 90% of units in every session. The resulting target row/mask is frozen before either
R10 or L10 fitting; both arms must score exactly the same defined channel-by-lag elements. No
per-arm, per-lag, or outcome-dependent exclusion is permitted.

Before source access, the receipt must prove that the exact seven paths, raw source-only loader,
manifest and formula-compatible target implementation are frozen and synthetic-tested. The
terminal target_or_manifest_not_ready_stop applies only if those exact seven source paths and
their raw loader cannot be frozen. The absence of an already-existing Python target module is not
itself a stop and may not justify changing the target formula.

## 3. Frozen M24 R10 and fair low-order rate control

For channel i and support trial t, with valid length L_t and integer bin counts x_i,t,b:

    r_i,t = log((sum_b x_i,t,b + 0.5) / (0.020 * L_t)).

### 3.1 Candidate R10

R10 is the following exact width-10 vector:

    R10_i = quantile_p(r_i,0, ..., r_i,23)

where:

    p = (0.05, 0.15, 0.25, 0.35, 0.45,
         0.55, 0.65, 0.75, 0.85, 0.95)

and the convention is NumPy method="linear" or a bitwise documented equivalent. R10 is invariant
to trial order and within-trial bin order. It retains the empirical distribution of 24
exposure-normalized trial rates, but no bin timing, count histogram, unit ID, cross-unit statistic,
or label.

R10 is normalized only with train-session channel rows in each outer fold. It has no learned
feature encoder, backpropagation, calibration optimization, or session-specific parameter fitting.

### 3.2 L10: width-matched low-order rate control

The control is not the former under-specified rate-only arm. It is a deterministic width-matched
nonlinear map of the first two moments of the same 24 trial log-rates, plus pooled support rate:

    mu_i    = mean_t(r_i,t)
    sigma_i = sqrt(mean_t((r_i,t - mu_i)^2))
    g_i     = log((sum_t,b x_i,t,b + 0.5) / (0.020 * sum_t L_t))

    L10_i =
      (mu_i, sigma_i,
       mu_i^2, mu_i*sigma_i, sigma_i^2,
       mu_i^3, mu_i^2*sigma_i, mu_i*sigma_i^2, sigma_i^3,
       g_i).

L10 has the same width, train-only z-score policy, ridge penalty, source rows, target mask, and
readout degrees of freedom as R10. It contains no empirical quantile/order statistic, histogram,
third moment, temporal ordering, or label. The fixed degree-at-most-three map prevents the
comparison from confusing “R10 has ten ridge columns” with “R10 has a richer distributional
statistic.” It is a control, not an optimized learned competitor.

The primary content contrast is:

    Delta_L,s = R2_R10,s - R2_L10,s.

No B20, P20, T4, K4, D4, B3, or full decoder arm is a substitute for L10 or may rescue this
contrast.

## 4. Strict seven-session source LOSO

There are exactly seven outer folds. In fold s:

1. build R10/L10 from each session's legal first-24 support only;
2. estimate feature mean/scale and target mean from complete defined channel rows of the other six
   source sessions only;
3. fit a separate width-10 multi-output ridge readout for each arm using only those train rows;
4. predict the left-out session's defined target rows; and
5. score all defined left-out channel-by-lag elements using the same fold-train target-mean TSS
   baseline for both arms.

The ridge penalty is fixed to lambda=1.0; its intercept is unpenalized. A feature standard deviation
at or below 1e-12 is set to 1.0 for both arms. The target is centered from train rows but is not
target-variance-scaled.

Per-session scores are:

    R2_A,s = 1 - RSS_A,s / TSS_s,

with TSS_s computed over identical defined left-out target elements for R10 and L10. A nonpositive
TSS, nonfinite fitted feature/prediction, target-mask mismatch, any session-lag defined fraction
below 90%, or zero-source-row count is a hard invalid-execution stop.

Sessions, rather than channels, lags, resamples, trial bins, or null schedules, are the only
operational comparison units. Report all seven Delta_L values, mean, median, sample SD, the
two-sided paired Student-t interval computed with df=6, and:

    MDE80 = (t(0.975,df=6) + t(0.80,df=6)) * SD(Delta_L) / sqrt(7).

These seven LOSO folds are not seven independent experiments: their training sets overlap, and
same-day Run1/Run2 sessions may be correlated. Therefore the interval and MDE are frozen
**operational stability/precision gates**, not a population-level confidence interval or a claim
of independent-session generalization. Passing them cannot upgrade this globally seen source set
to a formal confirmation set.

## 5. Robust complete-row attachment null

R10 is meaningful only if its complete vector is attached to the corresponding channel. The
attachment null uses 4,095 precommitted global schedules plus the aligned observation. In schedule
b, each source session receives an independently seeded uniform draw from the full permutation
group of its channel rows. Fixed points, an identity draw, duplicate within-session draws, and
duplicate seven-session schedules are all legal and must be retained. The same permutation moves
all ten R10 values together; no columnwise shuffle, scalar shuffle, feature recomputation, target
permutation, or row deletion is allowed.

For every schedule, the six legally permuted train sessions fit a new ridge readout and the
correspondingly permuted left-out session is scored against its unpermuted target. A full uniform
permutation preserves sessionwise feature values, all ten-column marginal distributions,
dimensions, source/session count, target, and ridge capacity, while randomizing channel-to-vector
attachment. Some rows remain fixed by chance; excluding those schedules would put the null draws
and aligned identity observation in different conditional permutation spaces and would invalidate
the formal Monte-Carlo p-value. A zero-fixed-point derangement may be reported only as an explicitly
descriptive sensitivity analysis and may never enter the alpha gate.

Raw R2 is unbounded below. The precommitted null statistic is the bounded, monotone fold utility:

    U_s = TSS_s / (TSS_s + RSS_R10,s) = 1 / (2 - R2_R10,s)
    H = mean_s(U_s).

TSS_s must be positive. U maps catastrophic finite R2 values smoothly to zero and avoids allowing
one unstable permutation to dominate a mean/SD. Raw R2, RSS, TSS, U, H, and raw R2 mean/median
must all be reported. H is not an R2 claim.

With H_0 aligned and H_b the 4,095 null values:

    p_attach = (1 + #{b in 1..4095: H_b >= H_0}) / 4096.

Ties count against R10. The run must also report the one-sided 97.5% binomial upper confidence
bound for the Monte-Carlo null exceedance probability. Schedules, seeds, complete per-session
permutation vectors, fixed-point counts, identity-draw counts, duplicate schedule multiplicities,
all H values, and a sorted-H checksum are mandatory receipt fields. Sampling is with replacement:
duplicates remain; any non-bijective or out-of-range permutation is a hard invalid-execution stop
and is not replaced after inspection.

## 6. Technical repeatability, state, and compute

### 6.1 Thinning prerequisite

For every left-out support, form 256 seeded complementary within-trial Binomial(count, 0.5)
partitions. Multiply both halves by two before computing their trial totals and R10 vectors. Do
not refit a readout or count these 256 values as sessions.

Raw R10 vectors are positive log-rate vectors and their cosine can be dominated by a shared
offset, making a raw-vector cosine largely vacuous. For each split seed and session, first retain
only rows finite in both halves. For every coordinate d, compute one pooled two-half channel-row
mean and scale:

    mu_d = (sum_i Fa_i,d + sum_i Fb_i,d) / (2*n_defined)
    sd_d = sqrt((sum_i (Fa_i,d-mu_d)^2 + sum_i (Fb_i,d-mu_d)^2)
                / (2*n_defined)).

Standardize both half matrices by these same pooled coordinate values, with sd_d at or below
1e-12 treated as undefined rather than silently replaced. Then compute each retained row's cosine
between its two standardized ten-coordinate vectors and report their median. Also report the
defined-row fraction and the ten per-coordinate Pearson correlations across channel rows between
the two standardized halves.

The lower 2.5% seed quantile of this **median standardized-row cosine** must be at least 0.5 in
every source session, and the defined-row fraction must be at least 0.90 in every partition.
Undefined coordinates, a nonfinite Pearson calculation, or a failed fraction is a measurement
failure. The per-coordinate Pearson values are mandatory diagnostics, not extra biological
samples. This is a measurement prerequisite, not evidence of source generalization.

### 6.2 Exact low-state streaming receipt

R10 needs no raw bins and no histogram. During calibration it stores only:

    M=24 uint32 per-channel trial totals: 96 B/channel
    one uint32 current-trial total:       4 B/channel
    one aligned validity/overflow word:   4 B/channel
    M=24 uint16 exposures, shared:        48 B, plus current/flags

At N=96 this is:

    per-channel state = 104 B
    channel state     = 9,984 B
    shared rounded    = 56 B
    accumulator       = 10,040 B (about 9.8 KiB).

Finalization uses a padded 32-input bitonic network: 240 compare-exchanges and 24 rate/log
evaluations per channel. It emits ten FP32 values (3,840 B at N96) or symmetric INT8 values plus
ten FP32 scales (1,000 B). Peak accumulator-plus-output is 13,880 B (13.6 KiB) for FP32 output
or 11,040 B (10.8 KiB) for INT8 output. Post-finalization, only that static output remains.

Per valid bin/channel the accounting is one checked integer addition into the current trial total.
There are zero floating MACs, no temporal recurrence, no cross-channel operation, no histogram
SRAM, and no raw-bin retention. A uint32 trial-total overflow is a hard descriptor failure, never
a wrap or saturation. Output INT8 is a feature-export receipt only; it is not decoder
quantization and does not authorize QAT/PTQ.

## 7. Decision, multiple testing, and early stops

The only pass label is:

    r10_m2_source_pass_for_separate_review

It is CPU-only and only permits a new review of a target-free SUA replication protocol. It does
not authorize that SUA run, a decoder, GPU, held-out data, formal evaluation, EvalAI, or
quantization.

All requirements below must pass:

1. **Content beyond the declared low-order rate family:** all seven Delta_L,s values are positive;
   mean Delta_L is at least +0.030 R2; the operational paired interval's lower bound is strictly
   positive; and the operational MDE80 is at most +0.030 R2. L10 is width-matched but its effective
   degrees of freedom derive only from `mu`, `sigma`, and `g`; this gate does not establish equality
   of effective rank or expressivity between the arms.
2. **Correct attachment:** p_attach is strictly below 0.025; its one-sided 97.5% MC upper bound
   is also below 0.025; aligned H exceeds the null median; and at least six of seven source
   sessions have aligned U_s above that session's null-U median.
3. **Measurement and contracts:** every session passes thinning; all source/hash/raw/padding/fold/
   mask/finiteness/no-label/no-CUDA/state/row-permutation/fresh-output receipts pass.

The attachment test is the only formal randomization test in this protocol; alpha is frozen at
0.025. The content threshold is a practical-effect and sensitivity gate, not a second small-n
p-value. No other arm, null, target, loss, session subset, pooled-channel score, or bootstrap may
rescue a failure.

Execution is serial:

1. complete the no-NWB prelaunch receipt and source-file guard;
2. run only aligned R10 and L10, all seven source-LOSO folds, plus thinning;
3. if any content, precision, thinning, or validity condition fails, write the terminal stop and
   do **not** run the 4,095 attachment schedules;
4. only if content/prerequisites pass, run all 4,095 frozen schedules and issue the final decision.

The terminal outcomes are:

| Failure | Required disposition |
|---|---|
| exact seven source paths/raw loader cannot be frozen | target_or_manifest_not_ready_stop |
| low-order control criterion fails | r10_low_order_control_not_beaten_stop |
| operational interval/MDE sensitivity fails | r10_precision_insufficient_stop |
| thinning fails | r10_measurement_unreliable_stop |
| attachment null fails | r10_row_attachment_not_distinguishable_stop |
| source/raw/label/fold/finite/state receipt fails | r10_invalid_execution_stop |
| all tests pass | r10_m2_source_pass_for_separate_review |

No terminal stop permits a B20 claim, changes to R10 probabilities/L10 basis/M24/target/lags/ridge,
additional M2 sessions, held-out reuse, SUA, decoder/GPU work, or quantization.

## 8. No-NWB prelaunch proof obligations

An independent prelaunch receipt must prove, entirely on synthetic arrays and static inspection:

- the exact seven-source manifest/read path exists and cannot resolve prohibited paths;
- the formula-compatible future-autocorrelation target implementation is either an exact existing
  interface or a new pure implementation of section 2.1, with synthetic-array equivalence tests;
  no semantic rewrite is permitted;
- exact R10 values agree with a reference linear-quantile implementation for variable trial
  lengths/exposures; L10 uses only declared first/second rate moments plus pooled rate;
- trial/bin order invariance, joint channel-row permutation equivariance, raw-prefix/padding
  handling, and no-raw-retention all hold;
- each attachment schedule contains seven valid complete-row draws from the full permutation
  group, preserves every ten-value row unchanged, permits and records fixed points/identity draws,
  and never permutes target rows;
- U/H/p_attach/MC-upper-bound arithmetic, ties, duplicates, and terminal early-stop behavior are
  correct on fixtures;
- all 4,095 schedules can run with at most 16 CPU workers, CUDA disabled, and a source-dimension
  benchmark projected to complete inside 24 hours; and
- the state/MAC/overflow figures in section 6 are emitted by the same feature implementation.

If the frozen schedule budget cannot meet the 24-hour CPU-only receipt, write
r10_null_budget_infeasible_stop. Reducing schedule count or changing seeds is a new protocol, not
an implementation optimization.

## 9. Explicitly forbidden conclusions

A positive R10 M2 source result would say only that a channel-attached distribution of 24
unlabelled trial rates transfers to its later neural autocorrelation profile across these seven
source sessions, beyond the declared L10 control. R10 and the later autocorrelation target can
both retain stationary-rate, exposure, and finite-sample trial-demeaning effects; the experiment
does not isolate a temporal mechanism. It would not establish B20, functional identity, a decoder
gain, few-shot behaviour calibration, population-level independent-session generalization,
cross-subject generalization, or an INT8 model.

Do not call R10 a hardware winner merely because its accumulator is smaller. Do not attach it to
SPINT, concatenate it with T4/P20, add a learned MLP/FiLM/attention, use source labels, or open
SUA/formal/held-out data until the independent M2 gate passes and a new protocol is reviewed.
