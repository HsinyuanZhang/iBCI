# B20 marginal-distribution carrier — cross-domain and hardware audit

**Status:** historical analysis/protocol audit with post-execution disposition, 2026-08-02. The
prospective gates below are retained as design provenance only. This document authorizes no further
source or target data access, SUA work, GPU, decoder, local held-out, formal, EvalAI, or INT8
experiment.

## Bottom-line verdict

B20 is an important scientific control and a plausible no-label static carrier, but its exact
pooled-bin quantile half is **not** a lightweight streaming feature. It can be computed exactly
without retaining raw bins, but only by keeping a per-unit count histogram. At a conservative
8-bit count bound, that histogram dominates state:

| deployment setting | exact accumulator only | peak with FP32 B20 output | peak with INT8 output |
|---|---:|---:|---:|
| M2, N=96, M=24, max trial length 1024 | about 57.8 KiB | about 65.3 KiB | about 59.8 KiB |
| SUA, N=137, M=10, max trial length 100 | about 74.9 KiB | about 85.6 KiB | about 77.7 KiB |

These are accounting bounds, not measured latency. They show that exact B20 is excellent as a
strong source-only falsification baseline, but it should not be advertised as a low-state
replacement for B3/T4/P20 unless a separately tested compression retains its effect.

If the reported M1 source proxy finding is that B20 exceeds both the simple rate arm and P20, the
immediate scientific interpretation is not “B20 is a decoder feature.” It is “the former P20
signal may be explained by a high-order non-temporal marginal carrier.” Under the frozen M1 A2
protocol, P20 then fails the B20 content gate and cannot open the P20 M2/SUA/GPU branch. The
separate B20/R10 marginal-persistence proposal below is historical rather than an active route.

### Post-execution disposition: full B20 did not pass component attribution

The later guarded M1 characterization ended at
`b20_component_attribution_indeterminate_stop`. All three complete-row/block attachment nulls
were statistically distinguishable, but Q10's conditional median-permutation loss was only
`0.00938--0.01745 R2`, below `0.03` in every source session, while R10's was
`0.03895--0.08250`. Because Q10 was still distinguishable, the frozen rule does not permit a
full-B20 claim or an assertion that Q10 is absent. Consequently the B20 M2/SUA sequence described
below is not authorized as written.

R10 was a separate, explicitly post-M1 lower-state hypothesis because it beat rate-only in 4/4 M1
sessions and full B20 exceeded it by only `0.00476--0.01366 R2`. R10 does not require the pooled
256-bin histogram. With the same aligned accounting convention used below, its calibration state
would have been:

| R10-only setting | accumulator | peak + FP32 R10 | peak + INT8 R10 |
|---|---:|---:|---:|
| M2, N=96, M=24 | 10,040 B (9.8 KiB) | 13,880 B (13.6 KiB) | 11,040 B (10.8 KiB) |
| SUA, N=137, M=10 | 6,600 B (6.4 KiB) | 12,080 B (11.8 KiB) | 8,010 B (7.8 KiB) |

These are accounting bounds, not latency measurements. The independent M2 R10 source-only gate
has now been attempted and is closed without a performance result: v1 was invalid at the
unsigned-padding check (`OverflowError: Python integer -1 out of bounds for uint8`), and the one
authorized mechanical v2 retry was invalid before scoring because the target defined-row fraction
was below `0.90`. Neither run produced R2 values, R10--L10 deltas, reliability, or schedule-null
results. This is not evidence that R10 performs negatively. It is an execution/data-validity stop,
and it cannot inherit the B20 result, decoder authorization, or a full-B20 hardware claim.

Combined with `b20_component_attribution_indeterminate_stop`, these outcomes close every
B20/R10-to-SUA, decoder, GPU, formal, and INT8 escalation path. The remaining gate descriptions
and hardware accounting are historical specification material, not a recommendation or
authorization to resume the program.

## 1. Frozen B20 semantics by domain

B20 has two conceptually separate components:

    trial-rate part:      empirical distribution of support-trial log-rates
    pooled-bin part:      empirical distribution of valid support-bin log1p(counts)

No component uses a label, condition, unit ID, electrode, anchor, temporal filter, bin timestamp,
or later neural data.

| Domain | Support | Trial-rate coordinates | Pooled-bin coordinates | Important qualification |
|---|---|---|---|---|
| M1 A2 | chronological M=10 | all ten sorted trial log-rates | ten fixed pooled-bin quantiles | This is already frozen by the M1 A2 protocol and must not be changed retrospectively. |
| M2 | chronological M=24 | ten linear empirical quantiles of the 24 trial log-rates | ten linear pooled-bin quantiles | This is the existing M2 source protocol definition. |
| SUA | chronological eligible M=10, no reward/object/velocity filtering | ten sorted trial log-rates | ten fixed pooled-bin quantiles | This is the M1-compatible M10 definition, not numerically identical to M2's interpolated M24 feature. |

For M2, and for any future cross-domain comparison that requires identical numerical semantics,
the probability vector is fixed as:

    p = (0.05, 0.15, 0.25, 0.35, 0.45,
         0.55, 0.65, 0.75, 0.85, 0.95)

with the NumPy linear empirical-quantile convention or a bitwise documented equivalent. B20-M2
therefore has ten quantiles of 24 rates and ten pooled-bin quantiles, exactly as requested.

There is an unavoidable small-M semantic decision:

- M1 A2 and proposed SUA M10 retain all ten trial-rate order statistics. This is lossless for
  ten support rates and preserves the already frozen M1 definition.
- A new common B20-Q feature could use the probability vector above for both M2 and SUA, including
  interpolated M10 quantiles. It is more literally cross-domain-identical, but it is a **new**
  candidate. It cannot replace the M1 A2 B20 feature after seeing M1 and cannot inherit a positive
  result from it.

The recommended immediate route is M1-compatible B20-S10 for the SUA audit, with the mismatch to
M2 disclosed. Do not make a “same numeric carrier across all domains” claim unless B20-Q is
predeclared and independently tested.

## 2. Exact streaming implementation: no raw bins

Let x_i,t,b be a nonnegative integer count for unit i, trial t, valid bin b. Let L_t be the valid
bin count and M the fixed number of calibration trials. Exact B20 needs no trial-by-bin cache:

1. For each current trial and unit, accumulate the integer trial total C_i,t.
2. For every valid bin, increment that unit's count histogram H_i[x_i,t,b].
3. At a trial boundary, store C_i,t in a length-M per-unit trial-total buffer and store L_t in a
   shared length-M exposure buffer.
4. After trial M, form rate r_i,t from C_i,t and L_t, sort or select the required ten rate
   statistics, scan H_i to select the required ten pooled-bin order statistics, then apply log
   LUT/affine feature normalization.
5. Emit the static 20-value carrier and free the trial-total buffer and histogram. No raw count
   bin, raw trial matrix, or temporal history survives finalization.

The monotonicity of log1p means histogram selection may occur in raw count space and the ten
selected counts are mapped through log1p only after selection. This avoids a per-bin logarithm.
The design is forward-only and carries no online state after calibration beyond the emitted B20
vector itself.

### 2.1 Exact accumulator layout

Use the conservative hardware contract below until a source-only, prelaunch range receipt justifies
another bound:

    raw count admissible range: 0 <= x <= 255
    histogram:                  H[0..255], uint16 counters
    per-trial totals:           M uint32 values per unit
    current trial sum:          one uint32 per unit
    overflow/valid flag:        one byte logically, four-byte aligned in this accounting
    exposures:                  M uint16 values shared across units, plus current exposure

The raw-count upper bound is a deployment interface contract, not a fitted threshold. A new
dataset cannot silently choose a smaller range because its source values looked benign.

For one unit, the aligned accumulator size is:

    256 * 2 + 4*M + 4 + 4 = 520 + 4*M bytes.

Thus it is 616 bytes at M24 and 560 bytes at M10. The per-unit histogram is the dominant term.
This layout is integer state; calling it “FP32” or “INT8” would be misleading. Quantizing a
histogram counter to INT8 would overflow at ordinary calibration exposures, while storing it as
FP32 wastes 2x state and loses exact integer semantics.

### 2.2 M2 and SUA state receipt

The following uses M2's protected maximum padded trial length of 1024 and SUA's current
M10/max-trial-length-100 configuration. It assumes the manifest also proves M times maximum
length is at most 65,535, so each histogram counter is safely uint16.

| component | M2: N96, M24 | SUA: N137, M10 |
|---|---:|---:|
| unit accumulators | 96 x 616 = 59,136 B | 137 x 560 = 76,720 B |
| shared exposure/current/flags, rounded | 56 B | 24 B |
| accumulator subtotal | 59,192 B | 76,744 B |
| finalized FP32 B20, 20 x N x 4 B | 7,680 B | 10,960 B |
| finalized symmetric INT8 B20, 20 x N plus 20 FP32 scales | 2,000 B | 2,820 B |
| peak accumulator plus FP32 output | 66,872 B (65.3 KiB) | 87,704 B (85.6 KiB) |
| peak accumulator plus INT8 output | 61,192 B (59.8 KiB) | 79,564 B (77.7 KiB) |
| post-finalize FP32 output only | 7,680 B | 10,960 B |
| post-finalize INT8 output only | 2,000 B | 2,820 B |

The INT8 output row assumes symmetric per-feature affine quantization and twenty FP32 scales
(80 bytes). It excludes a decoder because B20 has not earned decoder integration. If a decoder
needs per-session zero points, additional metadata must be counted. The histogram/trial-total
accumulator itself is unchanged by output INT8; this is why “INT8 B20” cannot be used to imply an
8x calibration-state reduction.

### 2.3 Operation count

During each valid calibration bin and per unit:

| operation | count |
|---|---:|
| input-range comparison x <= 255 | 1 comparison |
| add x to current trial total | 1 integer addition |
| increment H[x] | 1 integer addition and one indexed memory update |
| multiply-accumulate | 0 |

At each trial boundary there is one trial-total write per unit and one shared exposure write.
At finalization:

- M2 M24 may use a fixed 32-input padded bitonic sorting network for the 24 rates: 240
  compare-exchanges per unit, plus 24 rate/log evaluations.
- SUA M10 may use a padded 16-input network: 80 compare-exchanges per unit, plus 10 rate/log
  evaluations.
- Each unit scans 256 histogram counters: 256 integer additions, at most ten threshold advances,
  and ten log-LUT reads.
- Carrier affine normalization/INT8 conversion costs 20 fixed scalar affine operations per unit.

There are no per-bin floating MACs, no temporal recurrence, no raw-bin DMA replay, and no
cross-unit operation. The real bottleneck is histogram SRAM bandwidth, not arithmetic.

## 3. Histogram upper bound and overflow policy

Exact pooled-bin quantiles cannot be produced from only mean/variance/Fano statistics. They require
either an exact histogram, a retained sample, or an approximation. The exact B20 branch uses the
histogram and must follow all rules below:

1. Every input count above 255 sets a sticky overflow flag for that unit/session and makes the
   calibration descriptor undefined. It must not be clamped into bin 255, folded into a last
   overflow bin, or converted to a nearby quantile.
2. A uint16 histogram counter overflow is a hard error. The manifest must prove
   M times L_max <= 65,535 before execution; otherwise the implementation uses uint32 counters
   with their larger receipt or does not run.
3. Current trial totals use uint32. Their addition is checked for overflow. A detected overflow
   invalidates the descriptor rather than wrapping.
4. The final result reports input-range maximum, count-overflow units, histogram-counter overflow,
   undefined feature rows, and calibration-abstention rate. A source-only experiment with any
   overflow is invalid for the exact 8-bit B20 claim; it cannot declare success after hiding
   those units.
5. A histogram bound selected from source data must be presented only as a future hardware
   candidate bound, then confirmed on a fresh scope. It must not be selected on an evaluation or
   target-free development session.

This deliberately makes B20 conservative. A lossy log-histogram, t-digest, adaptive histogram
range, reservoir sample, or clipping policy is a different carrier, not an optimized
implementation of exact B20.

## 4. Historical cross-domain replication order (superseded)

The ordering below recorded the prospective test of whether B20's apparent signal was a general
marginal-persistence effect before attachment to a frozen decoder. It is superseded by the
post-execution disposition above and authorizes none of the listed gates.

### Gate 0 — source-free implementation and hardware contracts

Before any NWB:

- pure-array tests prove B20 invariance to unit-row permutation, support-trial reordering, and
  valid-bin reordering;
- integer histogram/trial-total output equals a reference exact batch calculation on synthetic
  variable-length integer trials;
- overflow, histogram-counter boundary, padding, trial reset, fixed quantile interpolation,
  output-state release, and no-raw-retention tests pass;
- output FP32/INT8 parity is tested as a feature-export contract only, not decoder accuracy;
- source-only loading/no-label/no-CUDA receipt and the above state/MAC table are hash-bound.

Any failure is implementation invalidity, not a tuning opportunity.

### Gate 1 — M2 M24 source-only later-autocorrelation replication

Only after Gate 0 can M2 use the seven exact held-in source sessions, chronological support
[0,24), no behavioural/velocity/direction labels, and the existing label-free future
within-trial autocorrelation profile:

    lags = (1, 2, 4, 8, 16, 32, 64, 128) raw 20-ms bins,
    future target = all trials [24,end).

The target is trial-mean-centred and variance-normalized, so it tests future temporal shape rather
than merely future mean rate. All target construction is scorer-only. Use nested source LOSO,
equal-session aggregation over seven sessions, one fixed width-20 ridge readout, train-only
normalization, and no target-dependent hyperparameter selection.

The primary scientific question is not whether B20 “beats P20” on a new target. It is whether the
pooled-bin distribution adds stable predictive value beyond the trial-rate component. The frozen
component controls are:

- R10: the ten trial-rate statistics, zero-padded to width 20;
- H10: the ten pooled-bin quantiles, zero-padded to width 20;
- B20: their concatenation.

R10/H10 are component ablations, not new learned architectures. P20 may be reported only as a
descriptive historical comparator; it is not a B20 gate and cannot rescue or invalidate B20.

M2 passes to SUA only if all seven B20 minus R10 session deltas are positive, the equal-session
mean is at least +0.030 R2, the paired 95% lower bound is above zero, MDE80 is at most +0.030 R2,
and all no-label/raw/oracle/overflow/state contracts pass. The threshold is deliberately a
prospective SESOI for this new B20 hypothesis, not imported from M1 P20 or M2 T4/K4 results.
Failure is B20-marginal-replication stop: do not change lags, target, M, feature probabilities,
histogram range, or add a decoder.

### Gate 2 — SUA M10 source plus target-free later-profile replication

Only an M2 pass permits sorted SUA sub-C CO:

- source: nested LOSO across the fixed 27 source sessions;
- application: the six fixed target-free development sessions, with no session-specific refit;
- support: first ten chronological eligible neural trials, 20-ms bins, no reward/object/velocity/
  target-direction/electrode/waveform/SNR input;
- future target: the same eight-lag within-trial autocorrelation profile, built only from a
  predeclared neural-only post-trial-50 interval after B20 has finalized;
- formal SUA test paths remain unresolved.

The source readout, anchors (none are used by B20), normalization, and any equal-session
subsampling policy are fitted from source sessions only. The target-free application is genuinely
target-free: later neural bins are unavailable until the fixed B20 vector and source readout
prediction have been produced.

The principal target-free test is B20 minus R10 across the six application sessions. It passes only
if at least five of six deltas are positive, mean delta is at least +0.030 R2, the paired 95%
lower bound is above zero, MDE80 is at most +0.030 R2, and every label/oracle/overflow/state
receipt passes. The 27-session source result is required but cannot replace the six-session
target-free criterion. A failure is terminal for B20 decoder escalation; it is not a reason to
use the sealed formal sessions, add a FiLM, tune quantiles, or remove a dev session.

Even a Gate-2 pass establishes only a session-independent, label-free mapping from support
marginal statistics to a later neural temporal profile. It does not establish behavioural decoding,
functional unit identity, frozen-decoder compatibility, or a hardware accuracy advantage.

### Gate 3 — only then consider decoder compatibility

After both domains pass, a separate root review may consider one zero-init, width-matched
decoder-compatibility CPU audit. It must compare a frozen baseline, B20, R10, and a complete B20
row permutation. It must prove zero-init equality, preserve variable-N/unit permutation behavior,
state/output accounting, and use a sealed selection/report policy. GPU, formal, EvalAI, and
quantization remain separately unauthorized until that audit passes.

## 5. Historical components versus prohibited follow-ups

The following were prospective compression/attribution probes, not ways to rescue a negative B20
result. They are not authorized under the closed B20/R10 program.

| Component | State cost and purpose | Required comparison |
|---|---|---|
| R10 trial-rate distribution | M uint32 totals per unit; 96 B/unit at M24, 40 B/unit at M10, plus no histogram | Determines whether pooled-bin distribution adds information at all. |
| ZFV4: zero fraction, mean, Welford variance/Fano, exposure-normalized rate | four to six integer accumulators per unit; no trial buffer or histogram | Low-state hardware proxy; must be compared with R10/B20 without claiming exact quantiles. |
| CDF8 fixed thresholds | eight uint16 counters plus trial totals; thresholds frozen before source access, e.g. count CDF at 0,1,2,3,4,7,15,31 | Candidate quantile compression; must first match B20's target-free result within a separately frozen noninferiority margin. |

The following are prohibited in this program:

- retaining raw bins, trial matrices, a reservoir sample, or a per-bin sort buffer;
- adaptive histogram edges, source-fitted quantile thresholds, t-digest/KLL sketches, learned
  quantile estimators, or post-result clipping; these are different models and need a new
  hypothesis;
- P20/B20 concatenation, T4/D4/K4 fusion, labels, electrodes, waveform/SNR, static IDs, FiLM,
  attention, an MLP, or a decoder before the cross-domain gates;
- counting source units, lag values, bootstrap seeds, or histogram bins as extra sessions;
- changing M24/M10, future target, lag grid, B20 probabilities, range policy, or session subset
  after a negative result; and
- calling an INT8 feature export “quantized B20 deployment” while its exact uint16 histogram
  still dominates calibration state.

## 6. Honest claims and final disposition

The useful immediate role for B20 is methodological: it is the high-information control that can
separate temporal carriers from marginal count-distribution/stationarity effects. Its hardware
analysis does not make it cheap enough to skip that scientific test.

Historical prospective sequence (now closed):

    exact B20 prelaunch/state audit
        -> M2 seven-source M24 future-autocorrelation test (B20 vs R10)
        -> only if positive, SUA 27-source plus six target-free M10 later-profile test
        -> only if positive, separately reviewed zero-init decoder compatibility audit

The M2 R10 run did not reach a performance comparison, so it must not be summarized as a negative
R10/B20 result. The correct final disposition is narrower: B20's M1 source proxy and its
hardware-accounting role remain diagnostic, but the B20/R10 cross-domain program is closed at
component attribution plus execution/data-validity gates. It supplies no basis for SUA, decoder,
GPU, formal, or INT8 escalation.
