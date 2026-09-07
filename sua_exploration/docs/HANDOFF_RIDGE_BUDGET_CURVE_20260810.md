# HANDOFF: a supervision-budget curve for the direct ridge comparator

**Status:** proposal and protocol. **This document authorizes no run.** It needs the usual
pre-registration and the usual review chain before any execution.
**Date:** 2026-08-10
**Written in:** ASD-STE100 Simplified Technical English.
**Compute class:** **CPU only. Zero GPU. Zero training. Zero new data scope.**
**Interference:** none. Section 2 is a hard contract. If any item in section 2 cannot be met, stop
and report instead of proceeding.

---

## 1. Why this exists

The paper reports one comparator that is higher than the method:

| View, 15 external subject-M sessions | F0-B3 | PV50 | Ridge50 | T4 |
|---|---:|---:|---:|---:|
| SUA | `-0.1961` | `0.1154` | **`0.4179`** | `0.3568` |
| Pseudo-MUA | `-0.2047` | `0.1042` | **`0.4102`** | `0.3061` |

Three facts about that row are already in the paper and are easy to miss:

1. **The supervision is not equal.** T4 receives **one direction label per rewarded trial**, that is
   50 scalars. Ridge50 receives the **dense per-bin velocity trace of all 50 calibration trials**,
   and regresses it from a causal 50-bin spike history.
2. **The paired intervals cross zero.** `T4 - Ridge50` is `[-0.1752, +0.0699]` on SUA and
   `[-0.2116, +0.0046]` on pseudo-MUA. Ridge50 is higher in 12/15 and 13/15 sessions.
3. **The same comparator family reverses on native M2.** There, the per-session direct ridge scored
   `0.1139` against T4 `0.2268`, so T4 was higher by `+0.1129`.

So the present evidence is one point on each of two different cohorts, with opposite signs, and no
axis that connects them.

**What is missing is a curve, not an experiment.** T4 already has budget points at `M=15`, `M=30`,
and `M=50` on this same external cohort. Ridge has only `M=50`. Two more closed-form ridge fits put
both methods on one supervision axis.

**The claim this would support**, if the numbers allow it: the comparison between an analytic
functional carrier and a directly supervised linear readout depends on the label budget, and the
crossing point is measurable. That converts a defensive paragraph into a characterized boundary.

**The claim this would not support:** it is not a new confirmation, not a hidden-endpoint result,
and not an upper bound on supervised recalibration. See section 7.

---

## 2. Non-interference contract

Two GPUs are busy. H1 `CI64` runs on both static partitions, and the RT `L-D` v8 watcher is armed on
GPU1 and polls for `19250113/19250119` current-CI terminal receipts. **Nothing in this work may touch
either.**

| # | Rule |
|---:|---|
| 1 | **No GPU.** Export `CUDA_VISIBLE_DEVICES=""` for every process. Create no CUDA context, even to query the device. |
| 2 | **Do not start, stop, signal, or restart any process** that this work did not start. Do not run `kill`, `pkill`, or any scheduler command. |
| 3 | **Write nowhere that an armed watcher polls.** No writes under `SPINT-main/pilot_artifacts/`, none under `h1_carrierid_date_lodo_ci/`, and none into any receipt directory bound by the v8 watcher or the CI64 runner. |
| 4 | **All output goes to one new isolated directory**, created for this work only, for example `sua_exploration/results/ridge_budget_curve_v1/`. |
| 5 | **Cap CPU threads.** Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to a small fixed value, and run under `nice`. A 50-bin by N-unit ridge solve will otherwise take every core through BLAS and starve the data loaders that feed the two training jobs. This is the most likely way to disturb a running experiment, and it is silent. |
| 6 | **Read-only on every existing artifact.** `map_location="cpu"` if any checkpoint is opened. Record any model state hash before and after and require equality. |
| 7 | **No git command.** No `commit`, `checkout`, `stash`, or branch change. No `pip` or `conda`. |
| 8 | **Open no formal, minival, or EvalAI path.** Use only endpoints that a prior sealed receipt already opened, and cite that receipt for each one. |
| 9 | If a running job finishes or fails while this work runs, **do not react to it**. Report it and continue or stop; do not take over a freed card. |

---

## 3. What already exists, and must be reused unchanged

| Asset | Value | Use |
|---|---|---|
| `T4` budget points on the external cohort | `M=15`, `M=30`, `M=50`, three seeds | The x-axis partner. Reuse the sealed numbers. Do not recompute. |
| `M=30` deltas from `T4@50` | `+0.0013` SUA, `-0.0008` pseudo-MUA | Already reported |
| `M=15` deltas | margin holds in 10/15 and 9/15 sessions | Already reported |
| `Ridge50` definition | causal 50-bin spike history, dense first-50 velocity, calibration-only standardization, fixed normalized ridge penalty, no target-session hyperparameter selection | Must be reused **exactly**, with only the fit budget changed |
| `V9` sealed external aggregate | SHA-256 `8a5ba373...aca7f4` | Read-only reference. Not modified, not superseded |

---

## 4. The experiment

**Primary.** On the same 15 external subject-M sessions, in both the SUA and the deterministic
pseudo-MUA views, fit the existing Ridge comparator at **`M=15`** and **`M=30`** calibration trials,
and score it on the **identical post-calibration query targets** already used for `Ridge50`.

**The only thing that changes is the number of calibration trials whose dense velocity enters the
ridge fit.** Everything else is held: the causal 50-bin history, the standardization rule, the
penalty rule, the query windows, the session set, and the scoring code.

This mirrors the T4 study exactly, which is a **fixed-query** budget study. Both curves therefore
share one x-axis and one query set.

**Deliverable.** One table and one figure:

| Budget | T4 SUA | Ridge SUA | T4 pMUA | Ridge pMUA |
|---|---|---|---|---|
| `M=15` | sealed | **new** | sealed | **new** |
| `M=30` | sealed | **new** | sealed | **new** |
| `M=50` | `0.3568` | `0.4179` | `0.3061` | `0.4102` |

Report paired per-session differences with the same bootstrap procedure already used for the
`M=50` row: average the three T4 seeds within a session first, because Ridge is deterministic and
not seeded, then bootstrap the 15 paired session differences. Report the intervals as
**descriptive**. They pass no gate.

---

## 5. Fairness controls. These are mandatory

If the ridge degrades at a low budget, the paper must be able to say **why**. Otherwise a reviewer
will conclude that the comparator was crippled by a penalty chosen for `M=50`.

**5.1 State the penalty semantics before running.** Read the implementation and write down what
"fixed normalized ridge penalty" normalizes by. If it is normalized by sample count or by a trace,
it adapts to the budget by construction and one arm is enough. If it is a fixed absolute value, then
declare **in advance** a second, per-budget renormalized arm as a sensitivity, and report both. Do
not decide this after seeing the scores.

**5.2 Report the conditioning at every budget**, in the style of the `H-PCF8` audit:

- number of rows and number of features;
- median regularized condition number;
- `trace(H)`, the effective degrees of freedom.

At `M=15` the design is strongly underdetermined: about `50 x N` features against roughly one third
of the `M=50` rows. A large drop in `trace(H)` would show that the frozen penalty, not the label
count, produced the fall. That distinction decides what the paper may claim.

**5.3 Keep PV50 in the table.** `T4 - PV50` is `+0.2415` SUA and `+0.2019` pseudo-MUA, with 14/15
sessions positive and intervals clear of zero. PV50 uses the same 50 direction labels **plus** dense
velocity for its affine gain. It is the closest thing to an equal-label classical comparator, and it
must stay visible next to the ridge row.

---

## 6. Read rules. Freeze all four before any run

| Outcome | What the paper may say |
|---|---|
| Ridge falls below T4 at a lower budget, in both views | The comparison depends on the label budget, and the crossing point is measured on this cohort. This is the target result. |
| Ridge falls below T4 in one view only | Report the view difference. Do not aggregate the two views into one claim. |
| Ridge stays above T4 at every budget, but `trace(H)` shows the penalty is dominating | Report both. The comparator is not crippled by us, and the honest statement is that a fixed-penalty dense-velocity ridge is stronger at every tested budget on this cohort. |
| Ridge stays above T4 at every budget with healthy conditioning | **Report it plainly as a negative.** The deployment-boundary paragraph becomes stronger, not weaker, and the paper states that a directly supervised linear readout on this cohort exceeds the carrier at every tested budget. |

**The fourth row is the reason to run this.** That fact would already be true today; it is simply
not measured. Learning it from a reviewer is far worse than learning it now. A read rule written
after the numbers appear voids the item.

---

## 7. Evidence status and governance

- The existing `Ridge50` row is already described in the paper as a **post-hoc additive reuse** that
  **does not add another independent confirmation**. The new budget points inherit exactly that
  status and may not be upgraded.
- The sealed `V9` external aggregate is not modified, not superseded, and not recomputed.
- The T4 budget curve is itself a post-hoc reuse of a completed endpoint, produced by a model trained
  under the `M=50` descriptor distribution. The joint curve therefore compares a **retrained-per-budget
  ridge** with a **fixed-descriptor-distribution T4**. This asymmetry favours the ridge at low
  budgets and **must be stated in the caption**, not buried.
- The 15 external sessions were opened by a prior sealed receipt. Cite it. Open nothing else.
- Every number produced here is development evidence.

---

## 8. Stage 2, conditional: native M2

The user asked for `M2` as well. It is a different comparator, so it is a separate stage and it may
not be merged with stage 1.

**First do a zero-compute inventory check** and report the answer before proposing anything:

1. Does a matched **T4** budget curve exist on native M2, or does M2 have only the `M=24` and `M=33`
   points?
2. Is the M2 `Ridge-W50` definition, which uses a 50-bin history at a 24-trial budget with 96 native
   channels, reproducible at other budgets without changing its semantics?

**If a matched T4 budget curve does not exist on M2, stop.** Producing new T4 budget points on M2
requires training, which requires GPU, which section 2 forbids. In that case the only honest stage-2
output is a **ridge-only** budget curve, which cannot answer the crossing question and should not be
run for that purpose.

---

## 9. What must not happen

| Item | Reason |
|---|---|
| Any GPU use, or reacting to a freed card | Section 2 |
| Changing the Ridge definition beyond the fit budget | It stops being the same comparator, and the `M=50` point stops being comparable |
| Choosing the penalty rule after seeing the scores | It is the difference between a fairness control and a constructed win |
| Recomputing, restating, or superseding the sealed T4 or `V9` numbers | They are the reference |
| Dropping a view, a session, or a budget after seeing it | The session set and both views are fixed by the existing `M=50` row |
| Calling any output a confirmation, a hidden-endpoint result, or an upper bound on supervised recalibration | Section 7 |
| Merging stage 2 into stage 1 | Different cohort, different comparator, different budget semantics |

---

## 10. Deliverables

1. One immutable receipt, mode `0444`, in the isolated output directory, recording: the reused
   Ridge implementation hash, the penalty semantics found in section 5.1, the session list and the
   receipt that opened it, per-budget rows and features, condition number, `trace(H)`, per-session
   scores, paired differences, and the bootstrap intervals.
2. The table of section 4 and one figure with both curves on one supervision axis.
3. A one-paragraph reading that follows exactly one row of section 6.
4. A note stating that no GPU was used, no process was signalled, and no watched directory was
   written.

---

## 11. References

- `bci_paper_overleaf/sections/04_experiments.tex` — `tab:external_classical`, the `T4 - Ridge50`
  intervals, the PV50 contrast, and `sec:label_budget`.
- `bci_paper_overleaf/sections/05_conclusion.tex` — the external Ridge50 paragraph and the native-M2
  Ridge-W50 paragraph.
- `CURRENT_RESULTS.md` — the 2026-08-06 external sub-M budget-curve entries and the `M30`
  true-early-start entry.
- `ACTIVE_EXPERIMENT_CONTROL_BOARD.md` — the running CI64 and RT L-D routes that section 2 protects.
- `HANDOFF_EXPERIMENT_SOFT_SPOTS_AND_PIVOT_20260810.md` §2 — the label-free comparator gap, which
  this document does **not** address. A dense-velocity ridge is not a label-free comparator.
