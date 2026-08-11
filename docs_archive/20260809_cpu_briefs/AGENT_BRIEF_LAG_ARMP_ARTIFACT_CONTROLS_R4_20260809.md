# AGENT BRIEF S4: three artifact controls for the population lag

**Date:** 2026-08-09
**Type:** implementation brief for a coding agent. Read this file in full before any action.
**Earlier briefs:** the round-1, `_R2_`, and `_R3_` lag briefs in this directory.
**Authorization level:** code and CPU screens on source data. No GPU. No target data. No training.

---

## 0. Your task in one sentence

Your round-3 report says the detrended **null** peaks at the same lag with a similar absolute
R². That makes the current reading unsupported. Run the controls that decide whether the peak
is a real neural lead or an artifact of the analysis.

---

## 1. Why the round-3 reading does not hold

You asked whether a future brief should compare the real delta distribution against the null
delta distribution. The answer is stronger than "yes, for rigour". **The comparison is the
test, and without it the round-3 conclusion has no support.**

The leave-one-out test removes the selection bias in the choice of `tau`. It measures
`R2(tau_r) - R2(0)`. If the real curve and the null curve both rise from `tau=0` to `tau=17`
by the same amount, then the null delta equals the real delta. Your sign test then says only
that "R2 is higher at 17 than at 0". The null satisfies that sentence too.

So `10 of 11` and `p=0.012` do not separate the lead from the null.

### 1.1 Two mechanisms that would produce exactly this

**Mechanism A, a sample set that changes with the lag.** The earlier briefs told you to drop a
block whose shifted bins leave its own TrialNum. So **each `tau` uses a different block set**.
The blocks that a large positive `tau` drops all sit at the end of a trial. An end-of-trial
block may hold a low-speed or hold state, with small variance and poor predictability. To drop
those blocks raises R² whether or not a signal exists, and it raises the null in the same way.
This also explains a peak at a latency that is not physiologically usual.

**Mechanism B, an in-sample R².** The frozen plan reports
`source_grid_best_equal_recording_r2 = 0.0286`. Your value is about `0.080`, which is near
three times the selection criterion of the plan itself. That points to an R² measured on the
same blocks that fitted the model. With 17 columns and about 627 rows, the chance in-sample R²
is about `17/627 = 0.027`. Inflation of that kind applies to the real data and to the null in
the same way.

---

## 2. Facts. Use them. Do not re-derive them.

| ID | Fact |
|---|---|
| T1 | A trial holds 151 blocks at the median, that is 15.1 s. No trial is shorter than 2.5 s. |
| T2 | The `M=4` support holds 558 to 696 blocks, about 60 s. |
| T3 | The carrier design matrix is about `627 x 17`. It is strongly overdetermined. |
| T5 | For a trial of `L` bins, block `j` covers bins `[5j, 5j+5)`. A block is valid at lag `tau` when `5j + tau >= 0` and `5j + 5 + tau <= L`. Over `tau` in `-10 .. +40` and `L = 755`, the valid block index range is `j` in `[2, 142]`, that is 141 of 151 blocks. **The fixed set of section 4 keeps about 93% of the blocks.** |

---

## 3. Step 0 — answer one question. No compute.

State plainly, in the report:

- Is your `R2(tau)` measured **in sample** or **out of sample**?
- Which blocks fitted the model, and which blocks scored it?
- What is the exact definition of the R² that you print, next to the plan value `0.0286`?

Do this before you run anything. If the answer is "in sample", control C3 in section 6 is
mandatory whatever the other controls give.

---

## 4. Control C2 — a fixed block set across all lags. Run this first.

This is the cheapest control and it is the most likely to settle the question on its own.

### 4.1 A zero-compute diagnostic first

You already recorded the dropped block count for each lag in round 1. Before you run anything
new, take the existing `R2(tau)` curve and the existing `n_blocks(tau)` curve and report their
correlation, for each recording, for the real data and for the null.

If R² tracks the block count closely, mechanism A is confirmed at once and you can say so.

### 4.2 The fixed set

1. Define the valid block set as the blocks that stay inside their own TrialNum at **every**
   tested lag, `tau` in `-10 .. +40`. See T5.
2. Recompute every curve on that one fixed set: the raw arm, the detrended arm at degree 3,
   the real data, and every null draw.
3. Report the block count of the fixed set for each recording, and the fraction that it keeps.

### 4.3 Report

For each recording, and for the raw arm and the detrended arm:

| Quantity | variable set (round 3) | fixed set (new) |
|---|---|---|
| the complete `R2(tau)` curve | | |
| the peak location `tau*` | | |
| `R2(0)` absolute | | |
| the null peak location and the null `R2` at that peak | | |

### 4.4 The read rule. Write it into the module before you run.

| Condition | Reading |
|---|---|
| The peak disappears or moves to `tau=0` on the fixed set | Mechanism A is confirmed. **Arm P stops and the lag route closes.** Stop here. Do not run C1 or C3. |
| The peak survives on the fixed set, and the real and null peaks now differ | Continue to C1 and C3. |
| The peak survives, and the real and null peaks are still equal | Continue to C1 and C3, but state that the current evidence still does not separate the lead from the null. |

---

## 5. Control C1 — the null leave-one-out distribution. Only if C2 keeps the peak.

Run the identical leave-one-out declared-lag procedure on the null data.

1. Choose at least 8 pre-declared null offsets. Treat each null realization as if it were the
   data.
2. For each null realization, and for each recording `r`: declare `tau_r` from the pooled null
   curves of the other 10 recordings, then compute `R2(tau_r) - R2(0)` on recording `r`.
3. This gives, for each null realization, an 11-recording mean delta and a positive count.
4. Report the null distribution of the mean delta and of the positive count.
5. Report where the real mean delta and the real positive count fall inside that null
   distribution. Give the percentile.

Use the fixed block set of section 4 for both the real data and the null.

### 5.1 The read rule

| Condition | Reading |
|---|---|
| The real mean delta sits inside the bulk of the null distribution | No lag-specific content. **Arm P stops.** |
| The real mean delta sits clearly above the null distribution | The lead separates from the null. Report it as a candidate. |

Do not invent a threshold. Report the percentile and the two distributions.

---

## 6. Control C3 — an out-of-sample R². Mandatory if step 0 says "in sample".

1. Split the blocks of a recording into `K = 5` **contiguous** folds. Contiguous folds are
   required, because the block series is autocorrelated and a random split leaks.
2. Fit on `K-1` folds. Score on the held fold. Average over the folds.
3. Recompute the raw arm and the detrended arm, real and null, on the fixed block set.
4. Report the in-sample and the out-of-sample curves side by side, with the plan value
   `0.0286` printed next to them.
5. State whether the out-of-sample real curve and the out-of-sample null curve separate.

---

## 7. What you must not do

| Item | Reason |
|---|---|
| Build a story about system latency or feedback | The peak may be an artifact. Do not explain a result that is not yet established. |
| Change the detrend degree | Degree 3 was declared before round 3. Do not re-select after seeing curves. |
| Widen the sweep past `+40` bins | Settled in round 3. |
| Reopen arm E, `pool_loo`, the shrinkage form, or H-NF | All settled. |
| Change `q`, `lambda`, the PCA rank, or the carrier width | These controls change the sample set and the scoring only. |
| Run C1 or C3 if C2 kills the peak | Stop at the first decisive negative. |
| Select a lag for a launch | You report the reading. A separate review selects. |

---

## 8. Acceptance criteria

Report each with the command output.

1. `git status` shows only your changed and new files. Prove that no protected file changed,
   with its modification time.
2. `nvidia-smi` and `ps` show no process of yours. Report the state of the five round-1 PIDs.
3. The round-1, round-2, and round-3 receipts are unchanged. Give their SHA-256 before and
   after.
4. All tests pass. Give the pytest line and its output.
5. The step-0 answer is given, in plain words.
6. The zero-compute diagnostic of 4.1 is reported for all 11 recordings.
7. C2 ran on all 11 recordings, both arms, real and null. Give the table of 4.3, and the kept
   block fraction.
8. If C2 did not stop the route: C1 and C3 ran, and their tables are given.
9. If C2 stopped the route: state that C1 and C3 were **not** run, and why.
10. No target, minival, formal, or EvalAI file was opened. State how you checked.

---

## 9. Report format

Give a short report with these parts:

- the files that you changed or created, with the line count of each;
- the ten outputs from section 8;
- **your reading against the read rules in 4.4, 5.1, and 6, and nothing more**. Do not propose
  a next experiment. Do not select a route;
- every design choice that this brief left open, and the choice that you made;
- every point where you think this brief is wrong or unclear.

Do not fix a problem that this brief does not cover. Report it instead.
