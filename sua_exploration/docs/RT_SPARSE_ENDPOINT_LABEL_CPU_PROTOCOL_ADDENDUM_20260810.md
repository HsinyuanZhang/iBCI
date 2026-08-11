# RT sparse cursor-endpoint direction — CPU protocol addendum

**Status:** proposed and frozen before any sparse-endpoint performance statistic. Root review is required before any all-15-session Stage 0B or Stage 1 run.

**Parent protocol:** `HANDOFF_SIMPLE_LABEL_MAINLINE_20260810.md`.

**Stage-0 input receipt:** `results/rt_simple_label_v1/stage0/RT_SIMPLE_LABEL_STAGE0_METADATA_RECEIPT_v1.json`, SHA-256 `ad8b2c583fddb8eef852fa4c39408beda9927a2c0dfbc714e4725df4cec9d37b`.

**Scope and claim boundary:** This is a third, accuracy-only Stage-0B/Stage-1 branch. It derives one scalar direction per accepted RT reach from two sparse endpoint positions in `processing/behavior/Position/cursor_pos`.

There are two non-interchangeable claims.

1. **Native annotation cost remains void.** Stage 0 established that no RT session exposes nondegenerate target metadata. This route must not say that the dataset natively supplies a target/direction, that position endpoints are a new annotation field, or that continuous behavioural measurement is unnecessary.
2. **A conditional algorithmic supervision-density claim is possible.** If the frozen gates pass, the estimator may be said to consume one derived angle per reach and only the two event-time 2-D positions that define it, rather than a per-bin velocity vector as its fitting target. This is an algorithmic reduction in supervision density after the position trace has been recorded; it is neither an annotation-cost claim nor a claim that the continuous position trace was not measured.

This new document is append-only: it does not alter the parent protocol, an RT loader, any existing receipt, or paper sources.

---

## 1. Read-only timing finding and limit

The existing RT event implementation, `streaming_calibration_exp/src/data/rt_k4_loader.py`, treats one trial as a sequence of `num_targets` reaches. A cue-complete trial has finite, strictly increasing declared cue times inside `[trial start, trial stop)`. Its accepted event intervals are:

```
[g_j, g_(j+1))                 for j < n_targets - 1
[g_(n_targets-1), trial_stop)  for the final reach.
```

The implementation accepts only whole 20-ms bins inside these intervals and does not cross an event boundary. This is the sole reach parser allowed here. A malformed cue row, undeclared finite cue, non-monotonic cue sequence, or endpoint outside its trial fails closed.

The cited stage-0 receipt establishes that cursor position is two-dimensional, centimetre-valued, explicitly timestamped at a median 100-Hz interval, and has every recorded go cue in range. Therefore two time-stamped positions define the **net cursor displacement of the parser-defined event interval**. They do not establish that the trajectory was straight, had no correction/loop, or had settled exactly at the next go cue. The candidate is an endpoint-displacement direction, not an instantaneous-velocity direction or target annotation. It is derived from the recorded trace, not natively available metadata. Section 5 freezes dense integration only as an external semantic audit.

---

## 2. Frozen sparse endpoint label

For an accepted reach `r = (trial k, reach j)`, let `s_r = g_j`; let `e_r = g_(j+1)` except for the final declared reach, where `e_r = stop_k`.

### 2.1 Endpoint sampler

Let raw cursor position be `(t_q, p_q)`, with position `p_q` in cm and strictly increasing `t_q`. Define `P(t)` deterministically:

1. If a raw timestamp equals `t`, use that two-coordinate position.
2. Otherwise, linearly interpolate the adjacent pair `t_l < t < t_u`: `P(t) = p_l + ((t-t_l)/(t_u-t_l)) * (p_u-p_l)`.
3. Read no position sample in the open event interval other than the two brackets required for `P(s_r)` and `P(e_r)`. No trajectory summary, endpoint search, peak, path length, velocity, acceleration, or target-like value is permitted.

An endpoint is missing if it lies outside the position time range, lacks an exact sample or bracketing pair, has a nonfinite bracket/position, or has bracket width greater than **20 ms**. A reach with either missing endpoint is excluded and counted by reason. Nearest-sample substitution, extrapolation, carry-forward, imputation, and use of a later cue are prohibited.

### 2.2 Direction and short-displacement rule

```
d_r     = P(e_r) - P(s_r)             # cm
L_r     = ||d_r||_2                   # cm
theta_r = atan2(d_r[1], d_r[0])        # radians in (-pi, pi]
u_r     = [cos(theta_r), sin(theta_r)] = d_r / L_r.
```

`MIN_ENDPOINT_DISPLACEMENT_CM = 0.50` is fixed before reading any endpoint agreement, coverage, fit, or transfer statistic. If `L_r < 0.50 cm`, the reach is `short_endpoint_displacement` and has no label. A nonfinite `d_r` or `theta_r` is also excluded. This is a deterministic angular-stability guard for a 100-Hz, centimetre-scale cursor sample; it is not tuned by session, trial, or result.

Each retained neural block in reach `r` contributes only to that reach's neural response mean in §3. The direction remains one label per reach and is never a per-bin kinematic label. Repeating `u_r` once per block as a fitting row is prohibited.

---

## 3. Frozen M24, event, lag, and bin contract

Only the label basis changes. All timing constraints of the sealed RT AFC4 construction remain.

| Item | Frozen value |
| --- | --- |
| Session scope | exactly the 15 `sub-C_ses-RT-*_behavior+ecephys.nwb` sessions in the Stage-0 receipt |
| Support | chronological trial indices `[0,24)` (M24), not 24 surviving reaches |
| Forward-audit query | chronological trial indices `[24,N)`; values are scorer-only and cannot change a fit, threshold, or choice |
| Reach parser | §1 `go_cue_j → next go_cue`; final `go_cue → trial stop` |
| Raw neural bin | 20 ms |
| Neural observation | five consecutive raw neural bins, converted to one 100-ms rate block; all eligible blocks in a reach are then averaged to one per-channel reach mean-rate response |
| +40-ms event containment | retain a neural block `[l,l+5)` only when `[l,l+7)` is wholly inside the same accepted reach; this preserves the two extra 20-ms bins / +40-ms event-boundary contract |
| Fit eligibility | event metadata, cursor-position endpoint eligibility, and neural-bin availability only. `cursor_vel`, including its validity mask, is prohibited before and during every fit/eligibility decision. |
| Extra label eligibility | the event reach must meet every endpoint and short-distance rule in §2 |
| Block stride | five raw bins |
| Primary response row | for each reach and channel, the arithmetic mean of its eligible 100-ms neural-block rates; a reach with zero eligible blocks is excluded and counted by reason |
| Fit | one OLS row per retained reach per channel: `reach_mean_rate = b + a cos(theta_r) + c sin(theta_r)` |
| Possible later carrier | `[a,c,0,0]`; Stage 1 does not construct a decoder |

There is no lag estimation or sweep. `theta_r` is not shifted. The former dense-carrier `+40 ms` rule is retained solely as same-reach event containment of the five-bin neural block plus its two-bin lead span; no velocity value or validity bit is read to implement it. Event containment is computed from the go-cue/trial-stop parser alone. If a support/half partition has fewer than three retained **reach rows**, rank below three for `[1,cos(theta),sin(theta)]`, nonfinite condition, or nonfinite OLS, it is undefined and fails closed. Block-weighted rows may be reported only as an eligibility/design audit and can never supply a primary fit, score, threshold, or gate.

---

## 4. Stage 0B — endpoint constructibility and semantic gate

Stage 0B is CPU-only and begins only after root approves this addendum. It may open only the 15 public RT NWBs already opened by the cited receipt. The run must use `CUDA_VISIBLE_DEVICES=""`, caps of one or two for `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`, and `nice`. It must not import Torch, create CUDA state, construct a DataModule/decoder/optimizer, inspect query performance artifacts, or write outside `results/rt_simple_label_v1/`.

For each session and M24, report:

1. Declared reaches, accepted event reaches, endpoint-readable reaches, short reaches, endpoint-missing reaches by frozen reason, and final endpoint-labelled reaches. Endpoint coverage has accepted event reaches as denominator.
2. For every trial and session: ordered unique endpoint timestamps actually read to form final endpoint-labelled reaches; number of those unique endpoint timestamps; raw scalar coordinates consumed (`2 × unique endpoint timestamps`); derived direction count; and the number/ratio of those coordinates to the dense RT retained-row target scalars (`2 × retained dense rows`). Adjacent reaches that share the same endpoint time count that position exactly once. The report must distinguish support M24, later scorer-only rows, and their union. Dense retained-row counts are calculated only after endpoint and primary reach-row eligibility are frozen, and are report-only.
3. Endpoint-label coverage, reach-row coverage, and eligible 100-ms block coverage, including blocks lost solely because an otherwise accepted event reach lacks an endpoint label or has zero neural blocks. The block figures are audit-only and cannot become fitting weights.
4. Direction coverage: resultant length `||mean(u_r)||_2`, largest circular angular gap, and an eight-bin fixed histogram over `[-pi,pi)` with edges `-pi + q*pi/4`, `q=0..8`.
5. Rank and condition number for the primary one-row-per-reach design `[1,cos(theta),sin(theta)]`. The former block-weighted design is also reported solely as a design/eligibility audit, with its repeated-label status explicit; it is not a passing condition.
6. The dense audit in §5, with defined-pair coverage and every undefined reason.

### 4.1 Precommitted Stage-0B gate

Every M24 session must satisfy all conditions below.

| Gate | Required condition |
| --- | --- |
| Endpoint labels | at least 24 final endpoint-label reaches and coverage at least 0.80 of accepted event reaches |
| Direction design | the primary one-row-per-reach design has rank 3 and finite condition number; the block-weighted audit is report-only and cannot affect this gate |
| Dense-audit availability | at least 0.80 of final endpoint labels have a defined dense-integrated audit pair |
| Dense-audit agreement | session median endpoint-versus-dense direction cosine is at least 0.70 |

All 15 passing gives `PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU`; any failure gives `STOP_STAGE0B_ENDPOINT_SEMANTIC_OR_COVERAGE_FAILURE_NO_GPU` and identifies each failing session/condition. Failure cannot permit a different endpoint time, smoothing/interpolation, length threshold, onset detector, lag, budget, or dense-label fallback.

---

## 5. Dense-integrated direction: audit only

Only after all endpoint and primary neural reach-row eligibility have been frozen, the audit may read the existing 20-ms-binned `cursor_vel` on raw bins wholly in the same `[s_r,e_r)` interval. Let `V_r` be those bins and require all to be valid/nonempty:

```
dense_d_r = 0.020 * sum(v_b for b in V_r)
theta_dense_r = atan2(dense_d_r[1], dense_d_r[0])
cos_agreement_r = dot(d_r,dense_d_r) / (||d_r||_2 * ||dense_d_r||_2).
```

The audit pair is undefined if `V_r` is empty, a used velocity bin is invalid, `||dense_d_r||_2 < 0.50 cm`, or an input is nonfinite. Dense direction and velocity validity must never replace the endpoint label, determine any primary fit eligibility or coverage, enter OLS, enter a shuffle seed, tune a parameter, select a session, or become a model target. It has only two permitted roles: the §4 semantic check and transparent reporting after eligibility is frozen. Passing it does not reinstate an annotation-cost claim.

---

## 6. Stage 1 — minimal CPU AC4 constructibility screen

Stage 1 requires a passing Stage-0B receipt and separate root authorization. It is an all-15 within-session CPU screen, not a decoder run, outer-LOSO result, or Stage 2. Candidate/null fits use only M24 endpoint directions and one neural reach-mean response per channel/reach; no dense velocity vector or `cursor_vel` validity bit enters any fit or fit-eligibility decision.

### 6.1 Full fit and AC4 split reliability

Fit §3's one-row-per-reach model on all eligible M24 reach rows. For each retained reach, form its response for every channel by averaging only its eligible 100-ms neural-block rates. Report reach-row design rank/condition, reach/block counts, every channel `[a,c]`, and undefined channels. The block-weighted repeated-label construction may be emitted only as an explicitly non-gating eligibility/design audit; it cannot be fitted, scored, compared, or used to decide passage.

For reliability, independently fit chronological ranges `[0,12)` and `[12,24)`, sharing neither block nor reach. For a channel with finite nonzero vectors in both halves:

```
split_cosine_i = dot([a,c]_first,[a,c]_second) /
                 (||[a,c]_first||_2 * ||[a,c]_second||_2).
```

If either norm is at most `1e-12`, or a half is undefined, the channel is undefined, not zero. Report per session and pooled: defined count, median split cosine, and fraction at least `0.40`.

### 6.2 Fixed forward transfer and null

The forward audit predicts later **neural reach-mean rates** from scorer-only later endpoint directions, using M24 coefficients. Later reaches are eligible by exactly the same event/endpoint/neural-block rules as support, without reading `cursor_vel` or its validity mask. For each later-defined channel:

```
R2 = 1 - sum((rate-predicted_rate)^2) / sum((rate-mean(later_rate))^2).
```

Fit three support-only models:

* `AC4-correct`: actual sparse endpoint directions.
* `AC4-reach-shuffled`: rotate ordered M24 reach labels by the deterministic nonzero shift `1 + uint64_le(SHA256("rt-sparse-endpoint-v1:<session>:42")[:8]) mod (R-1)`, where `R` is the number of retained M24 reach rows. Rotation acts on one reach label per reach, preserves their multiset, and leaves no reach paired to itself. It uses no dense velocity value or validity bit.
* `intercept-only`: M24 per-channel mean rate.

For each defined channel record `R2(correct)-R2(shuffled)` and `R2(correct)-R2(intercept)`. The session statistic is the median over its defined channels; defined count and exclusion reasons are mandatory. Aggregate the 15 session statistics for both contrasts as equal-session mean, equal-session median, positive/zero/negative signs, and leave-largest-out mean. Leave-largest-out removes the session with the greatest absolute difference, breaking ties by earliest session name; report its identity and all ordered session values. A mean alone is never a positive result.

### 6.3 Precommitted Stage-1 gate

`PASS_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBLE_NO_GPU` requires all four conditions:

| Gate | Required condition |
| --- | --- |
| AC4 split direction | median of 15 session-median defined-channel split cosines is at least `0.50` |
| AC4 split coverage | pooled fraction of defined channels with split cosine at least `0.40` is at least `0.50` |
| Correct-pair transfer | 15-session median of session channel-median `R2(correct)-R2(shuffled)` is at least `0.01` |
| Directional value | 15-session median of session channel-median `R2(correct)-R2(intercept)` is strictly greater than `0` |

The receipt reports mean, median, signs, and leave-largest-out mean for both transfer contrasts whether it passes or fails. A pass is constructibility/conditioning only, not decoder R2 evidence and not GPU authorization. A failure is `STOP_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBILITY_FAILED_NO_GPU` and forbids post-hoc changes.

---

## 7. Isolation, receipt, and exclusions

A future immutable receipt may be created only under `results/rt_simple_label_v1/stage0b/` or `results/rt_simple_label_v1/stage1/`, in a new empty directory with result file mode `0444`. It must bind this document SHA, the Stage-0 receipt SHA, exact NWB paths/hashes/byte sizes, every frozen parameter, per-trial/per-session deduplicated endpoint timestamps, raw-coordinate counts, derived-direction counts, dense-retained-row scalar counts and their ratios, per-session coverage/design/audit/split/transfer values and exclusions, null namespace/seed/rotation, thread caps/niceness, and explicit no-Torch/no-CUDA/no-decoder/no-process-signal/no-watcher-write statements.

Forbidden: GPU/CUDA; decoder training/evaluation; a new lag/bin/window/M/endpoint/trajectory/direction/short-distance/shuffle variant; dense velocity or its validity mask as fit eligibility or label; dense integration beyond §5; a claim of native target/direction metadata, no continuous behavioural measurement, or annotation-cost reduction; hidden/formal endpoints; paper edits; watcher writes; `git`, `pip`, or `conda`.
