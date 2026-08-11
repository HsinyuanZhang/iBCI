# AGENT BRIEF C-FIX2: the rank question, then a per-column normalizer

**Date:** 2026-08-09
**Type:** implementation brief for a coding agent. Read this file in full before any action.
**Previous briefs:** `AGENT_BRIEF_CPU_QUEUE_C1_C2_C4_20260809.md` and
`AGENT_BRIEF_CPU_QUEUE_FIXES_F1_F4_20260809.md`
**Authorization level:** CPU only, on source data. No GPU. No training. No target data.

---

## 0. Your task in one sentence

Answer two zero-compute questions first. They decide whether the rest of this brief applies.
Only if they say so, rebuild the two candidate carriers with a per-column normalizer.

---

## 1. STOP conditions

All stop conditions of the two previous briefs stay in force. In short:

- No GPU, even if both cards are idle. No training. No optimizer. No backward pass.
- No target, minival, formal, or EvalAI file.
- No edit to any protected file. No `git commit`.
- Source scope is the 11 fold-0 source recordings of `SPINT-main/data/000954`.
- Forward passes stay allowed under the earlier limits: source only, forward only, no gradient,
  unchanged model state hash.
- Do not stop, signal, or restart a process that you did not start.

Your C1 result stands. The overlap gate result with the `H-C0` reference stands.

---

## 2. What the last round established

Your F4 output is the important result:

```
b variance = 240.25       mean W column variance = 0.0007      ratio = 326,463
singular values = [26.5, 0.07, 0.05, 0.03]    first component fraction = 0.994
```

Every encoding carrier form is **effectively rank 1**, and that one dimension is the baseline
rate. The three `W` projection slots are negligible.

This connects to a sealed result. In the SUA attribution table, `B4 = [0,0,0,b]` reaches
`0.287273` while `Z4 = [0,0,0,0]` reaches `0.326008`. **A baseline-rate carrier is worse than a
zero carrier by `0.0387`.** So the current L-A build is not merely uninformative. Its expected
effect is negative.

Your F1 report was correct and useful. The old `U[8,4]` already placed `b` on its own column,
so the frozen compression rule reaches the same carrier. **The F1 fix was a no-op.** The
diagnosis of `b` dominance was right; the mechanism was not the projection.

### 2.1 The suspected real cause

`b` is a firing rate in Hz, with a standard deviation of about 15.5. A `W` coefficient is a
rate per kinematic unit, with a standard deviation of about 0.026. The imbalance is a matter of
units, not of information. The carrier needs the **pattern** of `W` across channels, not its
absolute size.

Two normalizer families exist in this repository:

| Lineage | Normalizer | Effect |
|---|---|---|
| SUA and RT, the AFC4 and T4 family | **per-column** train-only z-score, as in `fit_train_k4_stats` | the four columns reach a comparable scale |
| H1 CarrierID | a **single global RMS scalar**, as in `SourceRmsNormalizer` | it keeps `H-C0` exactly zero, but it does not balance the columns |

The suspicion is that you built SUA-and-RT style **content** and applied H1 style **single
scalar** normalization. A single scalar cannot remove an imbalance of `326,463` in variance.

Section 4 gives the fix. Section 3 decides whether the suspicion holds at all.

---

## 3. Step 1 — two zero-compute answers. Then stop or continue.

### 3.1 Question A — the spectra you have not yet reported

Acceptance item 8 of the previous brief asked for six carrier forms. Your F4 section reported
only the four encoding forms. Report the two that are missing:

- the singular value spectrum of the **H1 current carrier**, for each recording;
- the singular value spectrum of the **N4** carrier, for each recording.

For every carrier, also report **the number of components needed to reach 90% of the total
variance**.

### 3.2 Question B — which normalizer each carrier actually used

Read the code and report, for each of the six carrier forms, exactly which normalizer was
applied. Give the source file and line. State plainly whether it was per-column or a single
global scalar.

This checks the suspicion in 2.1. If the H1 carrier and N4 also used a single global scalar,
the whole comparison ran on globally normalized carriers and that must be recorded.

### 3.3 The decision gate. Apply it before you write any new code.

| Condition | Action |
|---|---|
| The H1 current carrier needs **one** component to reach 90% of variance | **STOP. Do not do section 4.** A rank-1 carrier is then normal for a carrier that works, and the premise of section 4 is void. Report and wait for a review. |
| The H1 current carrier needs **two or more** components to reach 90% | Continue to section 4. |

Report the number either way.

---

## 4. Step 2, conditional — rebuild with a per-column scale normalizer

Do this only if the gate in 3.3 says continue.

### 4.1 The normalizer

For each column `j` of the source descriptor matrix, compute the source standard deviation
`s_j`. Then divide column `j` by `s_j`.

**Do not subtract a mean.** Scale without centring keeps `H-C0 = [0,0,0,0]` exactly zero,
because a zero vector divided by any scalar is still zero. So this fix satisfies both
constraints at once. It is not a trade.

Rules:

- Fit `s_j` on the source recordings only. Freeze it. Record all four values in the receipt,
  so that the size imbalance stays visible.
- Apply the same frozen `s_j` to every recording. Do not refit per recording.
- Add a floor to `s_j`, as the existing RMS normalizer does, and record the floor.
- Do not invent a weighting between `b` and the `W` slots. Equal columns is the SUA and RT
  precedent. It is the defensible default.

### 4.2 Rebuild and re-measure

Rebuild `L-A` and `L-C` under the per-column normalizer, then repeat all three measurements:

| Measurement | From |
|---|---|
| the overlap residual R², against the `H-C0` activity path | F3 |
| separability, drift, and the normalized ratio at four significant figures | F2 |
| the singular value spectrum and the first-component fraction | F4 |

Keep the two controls in every table: the H1 current carrier and N4.

### 4.3 Verification checks that must pass

State each result plainly.

1. `H-C0` is still exactly `[0,0,0,0]` after the normalizer. Prove it by test.
2. The first-component fraction of the rebuilt carriers **falls**. Report the before and after
   values.
3. The reference cosine is **no longer exactly 1.000**. Report the new distribution.
4. The number of components to reach 90% of variance rises above one for the rebuilt carriers.

If check 2, 3, or 4 fails, the per-column normalizer did not remove the collapse. Report that
and stop. Do not try a third normalizer.

### 4.4 Then read the two rules again

Give your reading against:

- the overlap read rule of the first brief section 3.5, with the corrected wording of the
  second brief section 6.1;
- the L-C read rule of the first brief section 4.4, on the normalized ratio.

State clearly that the earlier L-C reading, `10411` against `7711`, is **superseded**, because
both carriers were baseline-rate carriers at that time.

---

## 5. What you must not do

| Item | Reason |
|---|---|
| Do section 4 when the gate in 3.3 says stop | The premise would be void. |
| Subtract a mean in the normalizer | It would break `H-C0 = 0` exactly. |
| Invent a weighting between `b` and the `W` slots | Equal columns is the precedent. |
| Try a third normalizer if the checks in 4.3 fail | Report and stop. |
| Change the C1 result or the `H-C0` overlap result | They stand. |
| Build lever L-B or lever L-D | Out of scope. |
| Reopen N4, H-NF, the lag route, pooling, or the shrinkage form | All closed. N4 stays a control. |
| Any backward pass, any optimizer, any training | Forbidden. |
| Select a carrier for a GPU launch | You report the reading. A separate review selects. |

---

## 6. Acceptance criteria

Report each with the command output.

1. `git status` shows only your changed and new files. Prove that no protected file changed,
   with its modification time.
2. `nvidia-smi` and `ps` show no process of yours. Report the state of the five earlier PIDs.
3. The C1 receipt and every sealed receipt are unchanged. Give their SHA-256 before and after.
4. All tests pass. Give the pytest line and its output.
5. Question A is answered: the H1 and N4 spectra, per recording, and the component count at
   90% of variance for all six forms.
6. Question B is answered: the normalizer of each of the six forms, with a source line.
7. The gate decision of 3.3 is stated, with the number that produced it.
8. If section 4 ran: the four frozen `s_j` values, the floor, the three re-measured tables, and
   the four verification checks of 4.3.
9. If section 4 did not run: state that plainly, and why.
10. The model state hash is identical before and after every forward pass.
11. No target, minival, formal, or EvalAI file was opened. State how you checked.

---

## 7. Report format

Give a short report with these parts:

- the files that you changed or created, with the line count of each;
- the eleven outputs from section 6;
- **your reading against the two rules named in 4.4, and nothing more**, and only if section 4
  ran. Do not propose a next experiment. Do not select a route;
- every design choice that this brief left open, and the choice that you made;
- every point where you think this brief is wrong or unclear.

Do not fix a problem that this brief does not cover. Report it instead.
