# AGENT BRIEF C-FINAL: per-column residuals, and the decision that closes the screen line

**Date:** 2026-08-09
**Type:** implementation brief for a coding agent. Read this file in full before any action.
**Previous briefs:** the three `AGENT_BRIEF_CPU_QUEUE_*` files in this directory.
**Authorization level:** CPU only, on source data. No GPU. No training. No target data.

**This is the last screen in this line.** After it, the reading goes to a direction review. Do
not expect another screen brief.

---

## 0. Your task in one sentence

Report the residual R² **for each carrier column separately**, because the pooled number mixes
a coordinate that the activity path already holds with three that it cannot.

---

## 1. STOP conditions

All stop conditions of the previous briefs stay in force. In short:

- No GPU. No training. No optimizer. No backward pass.
- No target, minival, formal, or EvalAI file.
- No edit to any protected file. No `git commit`.
- Source scope is the 11 fold-0 source recordings of `SPINT-main/data/000954`.
- Forward passes stay allowed under the earlier limits: source only, forward only, no gradient,
  unchanged model state hash.
- Do not stop, signal, or restart a process that you did not start.

---

## 2. What is settled. Do not rerun any of it.

| Item | Result |
|---|---|
| C1, the same-date against cross-date reframe | done, stands |
| the rank collapse | real, and the per-column scale normalizer fixed it: first-component fraction `0.9948 -> 0.3380` for L-A and `0.9942 -> 0.3981` for L-C |
| `H-C0` as the neutral reference | done, the bias against `H-C` is about `0.006` |
| lever L-C, the noise normalization | **closed.** The normalized ratio is `106.75` against `103.94`, so it is a tie and L-C adds nothing |
| N4 | it is itself rank 1 and its pooled residual of `0.249` is mostly the `mean_rate` coordinate |

Two corrections that must stay in the record:

1. The earlier diagnosis said the screen applied the wrong normalizer family. Your audit found
   that **no normalizer at all was applied to the 4-dimensional output**, on any of the six
   forms. The direction of the diagnosis was right; the mechanism was not.
2. The H1 production pipeline uses a **single global RMS scalar**. That scalar cannot balance
   the columns. So an L-A arm cannot enter the existing pipeline unchanged. It needs the
   per-column normalizer wired into the production path, and `H-C0 = [0,0,0,0]` must be proved
   again there. Record this as a scheduling constraint, not as work to do now.

Your choice to report only the `H-C0` reference was correct. Do not repeat the `H-C` reference.

---

## 3. Why one more measurement

After the per-column normalizer fixed the rank collapse, the overlap residual did not move:
`0.293 -> 0.292` for L-A, and `0.467 -> 0.467` for L-C.

A per-column linear rescale does not change the R² of that column against the activity path,
because R² is invariant to the scale of the regressand. So either your pooled statistic is a
mean over per-column R², in which case it never felt the collapse and the earlier numbers were
always valid, or it is a stacked variance ratio, in which case the reweighting happened to
cancel.

Either way the pooled number mixes coordinates that answer different questions:

| Column | Does the activity path hold it? | Expected residual |
|---|---|---|
| `b`, the baseline rate | **Yes.** It is a linear function of the calibration activity. | low |
| the three `W` projections | **No.** The activity path never sees kinematics. | high |

If `b` has a low residual and the pooled value is a mean, then the three `W` columns are higher
than `0.292`, and the useful content of L-A is being averaged away by `b`.

The same applies to the floor. N4 is rank 1 and led by `mean_rate`. So the correct floor for a
`W` column is **not** the pooled `0.249`. It is the residual of N4's non-rate coordinates.

---

## 4. The measurements

### 4.1 M1 — state the pooled definition. Zero compute.

Read your own code and state which pooled statistic you used:

- a mean over per-column R²; or
- a stacked total-variance ratio; or
- something else, described exactly.

Give the source line.

### 4.2 M2 — per-column residual R²

For all six carrier forms, against the `H-C0` activity path, on the 11 source recordings,
report the residual R² of **each of the four columns separately**. Give a value for each
recording, plus the median across recordings.

Name each column. For L-A that is three `W` projections and `b`. For N4 that is `mean_rate`,
`Fano`, `lag-1 autocorrelation`, and `population coupling`.

### 4.3 M3 — the raw `W` columns, before the `U` projection

Report the residual R² of the **seven raw `W` columns**, before any `U[7,3]` projection, on the
same reference and the same recordings.

This separates two different failures. If the raw `W` columns hold clear residual content but
the `U`-projected columns do not, then `U` is discarding the independent content, and the
problem lies in the choice of `U`, not in the encoding form.

### 4.4 M4 — the paired comparison

For each of the 11 recordings, compute:

```
A_r = median residual R² over the three W projection columns of L-A
B_r = median residual R² over the three non-rate columns of N4
```

Report the 11 paired differences `A_r - B_r`, the sign pattern, and an exact two-sided sign
test. Report the median paired difference.

Use a sign test on the 11 recordings. Do not invent a numeric threshold.

**A caveat you must record.** The L-A `W` columns are `U`-projections and the N4 non-rate
columns are raw statistics. The comparison is a floor comparison between the non-rate parts of
two four-wide carriers. It is not a matched contrast.

### 4.5 The test-of-the-test check. Do this before you read M4.

The activity path holds the firing rate. So both of these must have a **low** residual:

- the `b` column of L-A;
- the `mean_rate` column of N4.

Also, the four columns of the H1 positive control should all be clearly above zero.

If those do not hold, the measurement is behaving in a way this brief did not predict. **Stop
and report. Do not read M4.**

---

## 5. The decision rule. Write it into the module before you run.

Three outcomes. Report which one holds. Do not select a route yourself.

| Condition | Reading |
|---|---|
| The M4 sign test favours L-A over the N4 floor | The encoding form holds independent content. L-A goes to the GPU queue for Stage 1, with the production-normalizer constraint of section 2. |
| M4 is a tie, **and** the raw `W` columns of M3 are clearly above the `U`-projected columns | The encoding form holds content, but `U[7,3]` discards it. L-A closes in its current form, and **lever L-B, the choice of `U`, is indicated.** |
| M4 is a tie, **and** M3 shows no advantage for the raw columns | **L-A closes.** The content axis is then empty, and lever L-D, the multiplicative consumption, is the only untouched structural lever. |

---

## 6. What you must not do

| Item | Reason |
|---|---|
| Rerun C1, the rank fix, the `H-C0` reference, or lever L-C | All settled. |
| Re-measure against the `H-C` activity path | `H-C0` is the neutral reference and the bias is already quantified. |
| Wire the per-column normalizer into the production pipeline | That path is frozen mid-matrix. Record the constraint only. |
| Build lever L-B or lever L-D | Out of scope. A review decides whether they start. |
| Invent a numeric threshold | Use the sign test on the 11 recordings. |
| Read M4 when the check in 4.5 fails | Stop and report instead. |
| Any backward pass, any optimizer, any training | Forbidden. |
| Select a route | You report which of the three outcomes holds. A review selects. |

---

## 7. Acceptance criteria

Report each with the command output.

1. `git status` shows only your changed and new files. Prove that no protected file changed,
   with its modification time.
2. `nvidia-smi` and `ps` show no process of yours. Report the state of the earlier PIDs.
3. Every sealed receipt and every earlier receipt in this line is unchanged. Give the SHA-256
   before and after.
4. All tests pass. Give the pytest line and its output.
5. M1 is answered, with a source line.
6. M2 gives all six forms, four columns each, per recording and median.
7. M3 gives the seven raw `W` columns.
8. The check of 4.5 is reported, and it passed or the run stopped.
9. M4 gives the 11 paired differences, the sign pattern, the sign test, and the median.
10. The model state hash is identical before and after every forward pass.
11. No target, minival, formal, or EvalAI file was opened. State how you checked.

---

## 8. Report format

Give a short report with these parts:

- the files that you changed or created, with the line count of each;
- the eleven outputs from section 7;
- **which of the three outcomes in section 5 holds, and nothing more.** Do not propose a next
  experiment. Do not select a route;
- every design choice that this brief left open, and the choice that you made;
- every point where you think this brief is wrong or unclear.

Do not fix a problem that this brief does not cover. Report it instead.
