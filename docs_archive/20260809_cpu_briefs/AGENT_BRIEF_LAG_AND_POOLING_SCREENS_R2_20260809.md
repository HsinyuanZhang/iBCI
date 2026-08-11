# AGENT BRIEF S2: three follow-up screens after the round-1 lag and pooling result

**Date:** 2026-08-09
**Type:** implementation brief for a coding agent. Read this file in full before any action.
**Round-1 brief:** `sua_exploration/docs/AGENT_BRIEF_LAG_AND_POOLING_SCREENS_20260809.md`
**Authorization level:** code and CPU screens on source data. No GPU. No target data. No training.

---

## 0. Your task in one sentence

Round 1 gave a reading that two of its three conclusions do not support. Run three follow-up
screens that settle them, and add one metric that the round-1 pooling screen could not measure.

---

## 1. STOP conditions

All stop conditions of the round-1 brief section 2 stay in force. Read them again. In short:

- No GPU, even if both cards are idle. No training. No checkpoint forward.
- No target, minival, formal, or EvalAI file.
- No edit to any file in round-1 section 2.3. No `git commit`.
- Source scope is the 11 fold-0 source recordings of `SPINT-main/data/000954`.
- Do not stop, signal, or restart a process that you did not start.

Report the state of the GPUs and of the five round-1 PIDs before you begin.

---

## 2. What round 1 settled, and what it did not

### 2.1 Settled. Do not rerun.

`pool_loo - eb_global = -0.0096` mean and `-0.0076` median, with a matched shrinkage formula
and only the prior source changed. So a prior taken from the other source recordings is not
better than a global prior. **The prior-source lever is closed.** Report it again only if it
falls out of the new work for free.

### 2.2 Not settled: arm P was censored at the sweep edge

Arm P gave `tau* = +8 .. +10` bins in **all 11 recordings**, and the sweep stopped at `+10`.
Three facts make the reading "no lag content" unsafe:

1. All 11 recordings pile up at the edge. That is the signature of an optimum outside the
   window. A censored argmax lowers the real ratio. It does not lower the null ratio, because
   the null has no true peak. So the test is biased against a real effect.
2. `tau = +8 .. +10` means the neural bins lead the movement by 160 to 200 ms. That is the
   textbook direction and size for motor cortex. Eleven recordings agree.
3. The ratio statistic has low power here. Velocity autocorrelates over some hundreds of
   milliseconds, so the real `R2(tau)` curve is broad and flat, and its peak-to-zero ratio is
   small by construction. The null is noise, and an argmax over noise gives a larger relative
   rise. The statistic favours the null.

**The correct test removes the argmax.** See section 3.

### 2.3 Not settled: arm E contradicts its own number

Round 1 reported `frac > null q95 = 0.33 .. 0.51`. Under the null this fraction is `0.05`. So
the observation is 7 to 10 times the null rate. That number points to lag content. The stated
conclusion says the opposite.

But the number is not yet usable, because 8 null offsets cannot support a `q95`. With 8 draws
the empirical maximum sits near the 89th percentile, so a `q95` is an extrapolation.

Also, low cross-recording stability of `tau*_i` (`r = 0.12`) does **not** close arm E. At
deployment the per-channel weights across a fixed lag basis are fit again on each target
session. `tau*_i` does not need to transfer. What must hold is that `tau*_i` is estimable
inside one session at the deployment budget. Round 1 did not measure that.

### 2.4 Not settled: the pooling screen measured the wrong contrast for the big effect

Round 1 gave `raw = 0.365` and `eb_global = 0.777`. That is `+0.412` from shrinkage. The
sealed record says the deployed empirical-Bayes step moved the same statistic by about
`0.015`. Both arms in round 1 used a James-Stein formula, so **neither arm was the deployed
estimator**. The comparison against `+0.015` is void.

If the deployed formula reaches far less than `0.777` on the same plan, then the shrinkage
**form**, not the prior source, is the available upgrade. Section 5 measures this.

### 2.5 A metric defect you must fix

A split-half attachment cosine is inflatable by shrinkage. If both halves shrink toward a
common target, they agree more, and the cosine rises without any gain in accuracy. In the
limit of full shrinkage both halves equal the prior mean and the cosine is `1.0`.

So the `+0.412` of round 1 may be an artifact of the metric. Section 5 adds a metric that
shrinkage cannot inflate.

---

## 3. Deliverable SA-P2 — the population lag without an argmax

**File:** extend `SPINT-main/src/data/h1_lag_screen.py`. Keep the round-1 functions. Add new
ones. Do not change a round-1 result.

### 3.1 Widen the sweep for arm P only

Sweep `tau` over `-10 .. +20` bins, that is `-200 ms .. +400 ms`. Report the complete
`R2(tau)` curve for each recording. Say whether a peak exists inside the window, or whether
the curve still rises at `+20`.

### 3.2 The leave-one-recording-out declared-lag test

This is the primary test of SA-P2. It has no argmax on the scored recording, so it needs no
argmax null.

For each source recording `r` of the 11:

1. Pool the `R2(tau)` curves of the **other 10** recordings. Declare `tau_r` as the argmax of
   that pooled curve.
2. On recording `r`, compute `R2(tau_r)` and `R2(0)`. Do not search on recording `r`.
3. Record the paired delta `R2(tau_r) - R2(0)`, and the ratio.

Then report:

- the 11 paired deltas and the sign pattern;
- an exact two-sided sign test;
- the absolute `R2(0)` for each recording, so a reader can judge the relative size;
- the 11 declared values `tau_r`, and their spread. If all 11 folds declare the same lag, say
  so. That is a strong stability statement by itself.

### 3.3 The read rule. Write it into the module before you run.

| Condition | Reading |
|---|---|
| At least 10 of 11 paired deltas are positive, and the declared lag is stable across folds | A single frozen global lag is a candidate. Report the value. |
| The sign pattern is mixed | Arm P stops. |
| The widened curve still rises at `+20` | State it. Do not widen again in this brief. Report it as an open boundary. |

This test is source-internal leave-one-out. It screens. It proves nothing about a target
session. State that in the receipt.

---

## 4. Deliverable SA-E2 — the per-channel lag, with a usable null and a stability test

**File:** extend `SPINT-main/src/data/h1_lag_screen.py`.

### 4.1 First, state what round 1 used

Round 1 did not say whether it built the design from all blocks of a recording or only from
the `M=4` support blocks. State which it was. This changes how round 1 reads.

Then run both from now on, and report them side by side:

| Setting | Question it answers |
|---|---|
| all blocks of the recording | Does per-channel lag structure exist at all? High power. |
| the `M=4` support blocks only | Is it estimable at the deployment budget? The real question. |

### 4.2 Fix the null

1. Raise the number of circular offsets to at least 40. Pre-declare them. Keep the same rule:
   at least 1 second, not inside the tested lag set, no crossing of a trial boundary.
2. Report the null mean and the null standard deviation for each channel, next to `q95`.
3. Add a pooled rank test as the **primary** statistic, because it does not depend on a tail
   quantile. For each channel, find the rank of the real ratio inside its null draws. Under
   the null these ranks are uniform. Report the pooled rank distribution and its deviation
   from uniform.
4. Keep `frac > null q95` as a secondary statistic, and print the expected value `0.05` next
   to it.

### 4.3 Add the within-session split-half stability of `tau*_i`

This is the decisive number for arm E.

1. Split the blocks of one recording into two halves. Use **alternate blocks** as the primary
   scheme, because it removes a drift confound. Use first-half against second-half as a
   sensitivity.
2. Compute `tau*_i` on each half.
3. Report the correlation of `tau*_i` between the two halves, for each recording.
4. Compute the same correlation on circularly shifted velocity, as a null.
5. Run this for both settings of section 4.1.

### 4.4 Emit the raw array

Write `tau*` for every channel into the receipt. Round 1 dropped it during the aggregate. It
is needed to check the stability numbers.

### 4.5 The read rule. Write it into the module before you run.

| Condition | Reading |
|---|---|
| The pooled rank distribution is uniform | No per-channel lag content. Arm E stops. |
| The ranks deviate from uniform, **and** the split-half correlation of `tau*_i` at the `M=4` budget clearly exceeds its null | Per-channel lag content exists and is estimable at the deployment budget. This justifies a change of estimator form. |
| The ranks deviate from uniform, but the split-half correlation at `M=4` does not exceed its null | The structure exists but is not estimable at `M=4`. Arm E stops for the current budget. Report it as budget-limited, not absent. |

---

## 5. Deliverable SB-3 — the deployed shrinkage as a third arm, and an uninflatable metric

**File:** extend `SPINT-main/src/data/h1_pooling_screen.py`.

### 5.1 Arms. Use one plan and one recording set for all of them.

Use the fold-0 source plan of round 1: `q=16`, `lambda=100`, the 11 source recordings.

| Arm | Shrinkage |
|---|---|
| `raw` | none |
| `eb_deployed` | the deployed empirical-Bayes step, `w = tau2 / (tau2 + v)`, with the analytic covariance |
| `eb_js_global` | the round-1 James-Stein arm with a global prior |

Find the deployed implementation in the repository first. Reuse it. Do not rewrite the
formula. If you must call it through a wrapper, say so and give the source line.

`pool_loo` is settled. Include it only if it is free.

### 5.2 Two metrics. The second is the primary one.

**Metric 1, split-half attachment cosine.** Keep it, so that round 1 stays comparable. Mark
it in the receipt as **inflatable by shrinkage**, with the reason from section 2.5.

**Metric 2, cosine against a same-recording reference fit.** This is the primary metric.

1. Fit the carrier on the `M=4` support blocks with each shrinkage arm.
2. Fit a reference carrier on **all remaining blocks** of the same recording. A recording has
   about 600 blocks and the support has about 24, so the reference is a high-precision
   estimate for that session.
3. Report the per-channel cosine between each arm and the reference.

Shrinkage cannot inflate metric 2, because the reference does not move.

Limits you must record: the reference comes from later parts of the recording, so
nonstationary drift makes it an imperfect target. The drift affects all three arms in the same
way, so the paired deltas stay readable. Label metric 2 an oracle-reference diagnostic. It is
not a deployable quantity.

### 5.3 Report

For each arm and each of the 11 recordings, give metric 1 and metric 2. Then give the paired
deltas:

- `eb_js_global - eb_deployed` on metric 2 — **this is the number that decides SB-3**;
- `eb_deployed - raw` on metric 2;
- the same two on metric 1, for comparison with round 1.

Also print this sentence in the receipt: absolute cosines on the fold-0 source plan are not
comparable to the sealed `0.64` from the date-LODO plan. Only within-plan paired deltas are
readable.

### 5.4 The read rule. Write it into the module before you run.

| Condition | Reading |
|---|---|
| `eb_js_global - eb_deployed` on metric 2 is positive on most recordings and is large next to `eb_deployed - raw` | The shrinkage **form** is an available upgrade. Report it as a candidate. |
| The two arms agree on metric 2 | The round-1 `+0.412` is what shrinkage does on this plan, and the sealed `+0.015` differs for plan reasons only. No upgrade here. |
| Metric 2 shows a much smaller shrinkage gain than metric 1 | Confirm that metric 1 was inflated. Say so plainly, and read only metric 2. |

---

## 6. What you must not do

| Item | Reason |
|---|---|
| Rerun `pool_loo` as a new question | Settled negative in round 1. |
| Widen the arm-P sweep past `+20` | Out of scope. Report an open boundary instead. |
| Change `q`, `lambda`, the PCA rank, or the carrier width | The screens measure lag and shrinkage only. |
| Rewrite the deployed empirical-Bayes formula | Reuse the repository implementation. |
| Select a lag or a shrinkage arm for a launch | You report the reading. A separate review selects. |
| Invent a numeric threshold that acts as a noise floor | Report the real distribution against the matched null. |
| Read the fold-0 target into any aggregate | Mark it and exclude it. |

---

## 7. Acceptance criteria

Report each with the command output.

1. `git status` shows only your changed and new files. Prove that no file from round-1 section
   2.3 changed, with its modification time.
2. `nvidia-smi` and `ps` show no process of yours. Report the state of the five round-1 PIDs.
3. Round-1 results are unchanged. Give the SHA-256 of the round-1 receipts before and after.
4. All tests pass. Give the pytest line and its output.
5. SA-P2 ran on all 11 recordings. Give the widened curve summary and the 11 leave-one-out
   paired deltas with the sign test.
6. SA-E2 ran on all 11 recordings, in both settings of section 4.1. Give the pooled rank
   result and the split-half correlation table with its null.
7. SB-3 ran on all 11 recordings, three arms, both metrics. Give both tables.
8. The `tau*` per-channel arrays are present in the receipt.
9. No target, minival, formal, or EvalAI file was opened. State how you checked.

---

## 8. Report format

Give a short report with these parts:

- the files that you changed or created, with the line count of each;
- the nine outputs from section 7;
- **your reading against the read rules in 3.3, 4.5, and 5.4, and nothing more**. Do not
  propose a next experiment. Do not select a route;
- the answer to the question in section 4.1 about what round 1 used;
- every design choice that this brief left open, and the choice that you made;
- every point where you think this brief is wrong or unclear.

Do not fix a problem that this brief does not cover. Report it instead.
