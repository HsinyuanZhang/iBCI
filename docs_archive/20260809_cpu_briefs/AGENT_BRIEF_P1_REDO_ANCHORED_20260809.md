# AGENT BRIEF P1-REDO: identity-limitedness, with an anchor gate and M1 included

**Date:** 2026-08-09
**Type:** implementation brief. Read in full before any action.
**Supersedes:** section 3 of `AGENT_BRIEF_GENERALIZATION_P1_P3_20260809.md`. The earlier P1 run
is void. Its receipt stays for audit.
**Authorization:** CPU only. Forward passes allowed under section 2. No GPU. No training.

---

## 0. Why the first P1 run is void

Two defects, and the second one is the brief author's error, not yours.

**Defect 1 — no anchor reproduction.** Neither arm reproduced a known sealed value.

| Task | first run `r2_full` | sealed value it should be near |
|---|---|---|
| H1 | `0.9631` | H-C fold 0 is `0.5255`; the five dates are `0.44` to `0.52` |
| M2 | **`-0.0678`** | official held-out `0.303`; local M24 about `0.227` |

A negative full-model R² on M2 means the evaluation is broken. The receipt also holds a
`behavior_scaling_factor` of `20.0` for H1 and `5.0` for M2, and H1 scored only `200` windows
out of `6331` to `11980` available. None of that is in the protocol.

The repository already holds the correct pattern. The dose-response v2 run first reproduced the
sealed CUDA metric inside a pre-frozen `1e-6` absolute tolerance, and only then ran anything
new. Do the same here.

**Defect 2 — M1 was reported as absent, but it exists.** The earlier brief forbade
`streaming_calibration_exp/logs/` and `streaming_calibration_exp/outputs/`. That ban was
written to protect a running job and it was too wide: M1 checkpoints live there. **The ban is
now corrected to write-only.** These exist:

```
streaming_calibration_exp/logs/m1_afc4_emg_full_all_source_final/.../best_ckpt/epoch_011.ckpt
streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/.../best_ckpt/epoch_019.ckpt
streaming_calibration_exp/outputs/streaming_calibration/m1_clean_selection_v1_t4_m1_f1_s42/checkpoints/best.ckpt
streaming_calibration_exp/outputs/streaming_calibration/m1_afc4_emg_*/checkpoints/best.ckpt
```

M1 code also exists: `streaming_calibration_exp/src/data/falcon_emg_afc4_features.py`,
`falcon_m1_all_source_datamodule.py`, and the `m1_afc4_emg_*` experiment configs.

**If a search cannot reach a path, report "I cannot search there". Never report "it does not
exist".**

---

## 1. STOP conditions

- No GPU. No training. No optimizer. No backward pass.
- No `git commit`, `git checkout`, `git stash`, or branch change.
- **Read is allowed** anywhere in the repository. **Write is forbidden** in
  `streaming_calibration_exp/logs/`, `streaming_calibration_exp/outputs/`,
  `SPINT-main/logs/`, `SPINT-main/pilot_artifacts/`,
  `sua_exploration/results/rt_r4_budget_response_common_q24_v1/`, and any file with mode `0444`.
- Do not stop, signal, or restart a process you did not start. Report `nvidia-smi` and `ps`.
- Open only an endpoint that a prior sealed receipt already opened. Cite that receipt.
- Mark every result **same-checkpoint forward-only diagnostic, non-routing**.

Forward passes: forward only, no gradient, `map_location="cpu"`, model state hash identical
before and after.

---

## 2. The anchor gate. Nothing else runs until it passes.

For each task, before you zero anything:

1. Choose one sealed number that a receipt already reports for that checkpoint and that
   endpoint. Name it and give the receipt path and SHA-256.
2. Reproduce it with your own evaluator, on the **complete** window set that the receipt used.
3. Freeze a tolerance **before** you look. Use `1e-6` absolute when the receipt used the same
   device and construction. State and justify a larger tolerance if the device differs.
4. Report the sealed value, your value, and the difference.

**Rules:**

- Do not introduce a behaviour scaling factor. If the sealed evaluation used one, take it from
  the sealed configuration and cite the line. If it did not, use none.
- Do not subsample windows. Use every window the receipt used.
- If the anchor does not reproduce inside the frozen tolerance, **stop for that task and report
  the gap.** Do not run the zeroed arm. Do not tune until it matches.

A task that fails the gate is reported as `ANCHOR_NOT_REPRODUCED`, with your value next to the
sealed one. That is a valid and useful outcome.

---

## 3. The measurement

For each task that passed the anchor gate:

1. Score the already-open development endpoint unchanged. This is `full`.
2. Set the identity token to exactly zero immediately before it enters the neural window. Score
   again. This is `zeroed`.
3. Report `identity_limitedness = R2(full) - R2(zeroed)` for each session and in aggregate.

Give the source file and line where you zeroed, for each architecture. H1 CarrierID, SPINT on
M2, and the M1 model are three different objects.

### 3.1 Tasks

| Task | Checkpoint | Note |
|---|---|---|
| H1 | the sealed H-C epoch-49 checkpoint | anchor against a five-date or fold-0 sealed value |
| M2 | a T4 or clean-SPINT checkpoint with a sealed local held-out value | the M24 local held-out receipt is the natural anchor |
| **M1** | one of the checkpoints listed in section 0 | **this is the task the brief exists for** |

For M1, use the audited legal endpoint only: support `[0,10)`, and a post-support held-in-calib
window. Cite the receipt that opened it. Never reuse a `query_start=0` overlap endpoint.

### 3.2 What to report

| Task | baseline R² | identity-limitedness | known carrier gain |
|---|---|---|---|
| SUA | cite the sealed value | B3 zeroing gives `-0.277`, already known | `+0.2528` |
| M2 | | | official `+0.116765` |
| H1 | | | five-date `+0.056287` |
| **M1** | | | official `-0.003825` |

Then state whether the carrier gain follows the identity-limitedness across the tasks. Report
the pattern only. Do not fit a model to four points.

---

## 4. The read rule. Write it into the module before you run.

| Condition | Reading |
|---|---|
| M1 shows a small identity-limitedness while SUA shows a large one | M1 is not an identity-limited task. The carrier has nothing to correct there by construction. |
| M1 shows a large identity-limitedness | M1 is identity-limited and the carrier still failed. That is a real negative. |
| M1 fails the anchor gate | Report `ANCHOR_NOT_REPRODUCED` with both numbers. Draw no conclusion about M1. |

---

## 5. Acceptance

Report each with command output.

1. `git status` shows only new files. No protected file changed; prove with modification times.
2. `nvidia-smi` and `ps` show no process of yours.
3. The anchor gate table: sealed value, your value, difference, frozen tolerance, pass or fail,
   for every task.
4. All tests pass.
5. The four-task table of 3.2 for every task that passed the gate.
6. The zeroing source line for each architecture.
7. Model state hashes identical before and after every forward pass.
8. Every endpoint you opened, with the sealed receipt that opened it.
9. No formal, minival, or EvalAI file opened. State how you checked.

---

## 6. Report format

- files created, with line counts;
- the nine outputs above;
- **your reading against section 4, and nothing more**. Do not propose experiments. Do not
  select a route;
- design choices this brief left open, and what you chose;
- anywhere this brief is wrong or unclear.
