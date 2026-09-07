# HANDOFF: low-density event summaries on SUA, M2, and RT, with H1 as a separate module

**Status:** historical proposal plus reviewed execution record. Stages 0/0B/1 are complete. Stage 2
was subsequently authorized by the separate frozen
`RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md` and is running; that contract and
`ACTIVE_EXPERIMENT_CONTROL_BOARD.md` supersede this handoff for execution status and gates.
**Date:** 2026-08-10
**Written in:** ASD-STE100 Simplified Technical English.
**Interference:** none. Section 2 is a hard contract.

---

## 1. The idea, stated so it can fail

Two cohorts already calibrate from **one scalar label per trial**: SUA and M2 use the target
direction of each rewarded reach. Historical RT fits against dense per-bin cursor velocity, and H1
fits against dense per-bin 7-DoF kinematics. The RT candidate below is different from SUA/M2: its
M24 prefix contains `60--88` retained reaches per session, and it uses one derived direction per
retained reach, not one scalar per trial.

**The proposal has two halves.**

**Half one, testable.** The RT carrier estimator may not need dense velocity. The paper records the reason RT uses it:
*"The recorded trial-table direction takes one value and makes ordinary T4 rank-deficient."* That is
a **degenerate annotation column**, not a property of the task. The reaches physically have
directions, and the fit is already restricted to go-cue-bounded reach segments, so reach onset is
identifiable. If a per-reach direction can be derived, RT can test a low-density endpoint-summary
carrier across a second task geometry. This does **not** make RT a native simple-label task or make
its supervision protocol identical to the per-trial SUA/M2 protocol.

**Half two, a characterization, not an experiment.** H1 currently has no validated task-matched
per-trial scalar carrier under its M=4, multi-phase protocol. This is an empirical boundary of the
tested bases and calibration design, not a proof that every scalar or event summary is impossible.
Section 7 states the narrower conclusion.

**What would remove the annotation-cost claim, and it must be allowed:** if the per-reach direction
must be derived from the recorded behavior trace, then the native annotation-cost claim is void even
if accuracy holds. The production implementation may still read only sparse endpoint/bracket samples
and reduce estimator values and coordinate I/O; those are algorithmic density claims, not native labels.
Stage 0 decides this before any fit.

---

## 2. Non-interference contract

The rules of `HANDOFF_RIDGE_BUDGET_CURVE_20260810.md` §2 apply unchanged. In short: no GPU and no
CUDA context; do not start, stop, or signal any process this work did not start; do not take over a
freed card; cap `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and run under `nice`;
write only into a new isolated directory, for example
`sua_exploration/results/rt_simple_label_v1/`; nothing under `SPINT-main/pilot_artifacts/`,
`h1_carrierid_date_lodo_ci/`, or any watcher-polled receipt directory; read-only on existing
artifacts; no `git`, `pip`, or `conda`; open only endpoints a prior sealed receipt already opened and
cite each one.

---

## 3. Stage 0 — the metadata question. Zero compute. Do this first and report before anything else

**Question.** For the 15 RT sessions, does the NWB record a **per-trial target position**, or any
per-trial field from which a reach direction can be computed **without** reading the dense velocity
or position trace?

Report, per session:

- the exact trial-table column names and dtypes;
- whether a target position, target index, or goal coordinate exists per trial;
- the recorded value of the degenerate direction column, and confirmation that it is constant;
- whether the go-cue time and the reach-segment boundaries are recorded fields or are derived;
- if a cursor position trace exists, whether a single sample at go-cue would be enough to define the
  reach origin.

**Read rule, frozen now. Both branches continue; they support different claims.**

| Stage-0 outcome | What the later result may claim |
|---|---|
| A per-trial target field exists | Reach direction is available from native task metadata at **two coordinates per reach**. The accuracy claim and a native-metadata payload-density claim are available. This fact alone still does not measure human labelling effort. |
| No per-trial target field; direction must come from the dense trace | The accuracy and algorithmic supervision-density questions remain available. A positive result would show that the endpoint-summary system retains positive carrier-content evidence without using dense velocity for target-session carrier construction. **A native-label or human-annotation-cost claim is void and must not be written.** Record this explicitly in the receipt. |

Do not skip the second row. It is the honest outcome and it is not a failure.

---

## 4. Stage 1 — CPU constructibility screen. Source sessions only

Build the derived per-reach direction `theta_k` and the candidate carrier

```
r_i = b_i + a_i * cos(theta_k) + c_i * sin(theta_k)
```

which uses the same first-harmonic cosine basis as SUA/M2 `AC4`/`T4`. The Stage-2 carrier retains
`[a,c]` and pads it to `[a,c,0,0]`; it is not the complete four-component SUA/M2 descriptor. Fit it
on the same first-24 chronological trials and the same go-cue-bounded reach segments that the sealed
RT Full carrier uses. Change nothing else: same 20-ms bins, same source-fixed `+40 ms` lag, same
session set.

### 4.1 What to report

1. **Circular coverage** of the retained reach directions within each 24-trial prefix: the resultant length, the largest
   angular gap, and a histogram. Center-out has 8 discrete directions; RT should have continuous
   coverage, which is expected to be **better** conditioned, not worse. Report it either way.
2. **Design conditioning** per channel: rank and condition number of the `[cos, sin, 1]` system,
   and the degenerate-channel list.
3. **Coefficient stability**: the median normalized per-channel `[a, c]` split cosine, and the
   fraction of channels above `0.40`.
4. **Later-trial forward transfer**: fit on one half of the support, score on disjoint later trials,
   and report `delta R2` of the correct arm against a deterministic label-shuffled null.
5. **The reference point**: the sealed dense-velocity RT carrier has a chronological W-direction
   split cosine median of `0.795829`. Report the new value beside it and state clearly that the two
   are not a matched statistic, because one is a 2-D `[a,c]` fit on derived directions and the other
   is a 2-D `[w_x,w_y]` fit on dense velocity.

### 4.2 Frozen gates, modelled on the `H-PCF8` precursor that failed honestly

| Gate | Threshold |
|---|---|
| median normalized `[a,c]` split cosine | `>= 0.50` |
| channels with cosine `>= 0.40` | `>= 50%` |
| later-trial median `delta R2(full - label-shuffled)` | `>= 0.01` |
| later-trial median `delta R2(full - intercept-only)` | `> 0` |

**Read rule.** These gates decide **constructibility and conditioning only**. They do **not** predict
decoder gain, and no calibration exists that would let them. The overlap ruler is underpowered at
`n=3` with a predictor span of `0.021`, so no CPU statistic in this program currently predicts `R²`.
A pass authorizes a GPU proposal; it is not evidence of a gain.

**If the gates fail**, record the terminal status and stop. Do not add a lag sweep, a shrinkage
term, a different direction definition, or a budget change after seeing these numbers. Any of those
is a new pre-registration.

---

## 5. Stage 2 — GPU arms. Active only under the separate frozen contract

This handoff did not authorize Stage 2. A later independent review and immutable readiness chain did;
the 45-cell matrix now runs under `RT_SPARSE_ENDPOINT_STAGE2_THREE_ARM_CONTRACT_20260810.md`.
Nothing in this historical proposal may add an arm, gate, threshold, or interpretation after fold 0
was opened.

**Three freshly and independently source-trained arms, clean nested outer LOSO over all 15 sessions,
seed 42, matched in every other respect:**

| Arm | Carrier |
|---|---|
| `R-T4d` | derived per-reach direction, `[a, c, 0, 0]` padded to width 4 |
| `R-Full` | the dense-velocity carrier, freshly retrained as the matched reference |
| `R-Zero4` | zero content, width-matched |

**Do not gate against the sealed `R-Full = 0.44195`.** A fresh construction lost `0.06239` on
identical windows elsewhere in this program. The reference arm must be retrained inside this matrix.

**Primary, frozen:** `R-T4d - R-Zero4`. Report the mean, **the median**, and **the sign count out of
15**, and require all three to agree in direction before any positive statement.

**Secondary, descriptive:** `R-T4d - R-Full`. This is a bundled production-system contrast: the arms
differ in supervision sampling, response aggregation, and carrier content (`[a,c,0,0]` versus
`[w_x,w_y,||W||,b]`). It is reported whatever its sign and is not a label-cost attribution, a
non-inferiority gate, or an equivalence test.

### 5.1 The reporting rule that both previous rounds needed

Two comparison rounds in this program produced a mean and a median with opposite signs, in every
cell. **Every paired comparison in this protocol reports mean, median, sign count, and the mean after
removing the single largest session difference.** A positive mean with a negative median is not a
positive result and must not be written as one.

---

## 6. What the RT result can and cannot support

| Outcome | Statement available |
|---|---|
| `R-T4d` clears its primary `R-T4d−R-Zero4` gate | The frozen derived-endpoint carrier has positive system-level content evidence on RT. Report all three supervision-density accounting layers and the exact descriptive `R-T4d−R-Full` distribution. Do not call the systems equivalent or the difference a pure label cost. |
| `R-T4d` fails its primary gate | This frozen endpoint-T4d instantiation did not establish content gain over Zero4. It does **not** prove that RT intrinsically needs dense velocity or that every sparse summary fails. Historical dense RT remains separate task-generality evidence. |

Even a primary-gate failure does not remove evidence that the paper already has. RT's existing
dense-velocity contribution — `Full - MB4
= +0.25751` and `Full - XLSv2 = +0.28843`, both 15/15 — does not depend on this experiment.

---

## 7. H1 as its own module. Zero compute. This is a writing item

H1 currently has no validated task-matched per-trial scalar carrier. The observed constraint comes
from the dataset protocol, chosen carrier basis, phase structure, and short M=4 prefix together. It
should be stated as a measured boundary of tested designs, not as a mathematical impossibility.

### 7.1 The numbers that support it

- H1 has **8 to 15 trials per session**, and each trial is `13.0--19.6 s`.
- The `M=4` support is `558--696` blocks, about **60 s** of calibration.
- The current dense carrier design matrix is about **`627 x 17`**. Its algebraic row count comes
  entirely from using every bin; temporal correlation means `627` must not be called `627`
  independent observations.
- Four trial-level scalar rows can algebraically fit a rank-3 `[cos,sin,1]` model, but leave only one
  residual degree of freedom and cannot support the split-half reliability evidence used here. They
  are also insufficient for the tested 17-column dense basis.
- The discrete-label version was already tried and failed a pre-registered gate. The Version-B
  per-channel event carrier used the 8 native action epochs. `7/13` recordings had only
  `0.04--0.08 s` exposure for `Orient`, `Carry`, or `Orient2` in at least one half; the P4 split
  cosine was `0.1745--0.3464` against a `0.50` gate; and the forward gain against channel rate-only
  was positive in `0/5` dates. This closes that tested event/discrete configuration, not every
  possible event label. Status
  `FAIL_CONSTRUCTIBILITY_NO_GPU`, receipt `b77eef1c...b409924`.

### 7.2 The rule this supports, which is the general contribution

> **The required estimator granularity depends jointly on the number and diversity of calibration
> events, label geometry and coverage, coefficient dimension, temporal segmentation, and correlation;
> task complexity or trial count alone does not determine it.**

| Cohort | Calibration prefix | Status under the tested carrier |
|---|---:|---|
| SUA | 50 trials / 50 direction events | per-trial AC4 in use |
| M2 | 24 trials / 24 direction events | per-trial AC4 in use |
| RT | 24 trials / `60--88` retained reaches | one derived direction per reach; Stage 2 decides decoder utility |
| H1 | 4 long, multi-phase trials | no validated sparse scalar carrier; current dense carrier and failed native-event test remain distinct evidence |

Under this narrower rule H1's dense labels are not inconsistent with the low-density SUA/M2/RT
estimators: they are the supervision regime of the currently validated H1 system. A future user must
audit event count, coverage, basis rank/conditioning, temporal dependence, and forward transfer before
choosing a lower-density summary; trial count alone is not an operational test.

### 7.3 What the H1 module must not say

- Not that the method failed on H1. The carrier content is positive at `H-C - H-C0 = +0.032557`,
  4/5 dates, with an interval that touches zero; and the organizer-held system score is `0.274939`
  against `0.261492` for the paper-learning-rate SPINT reproduction.
- Not that H1 is a boundary case in the same sense as M1. On M1 the tested carrier content is
  negative, `B-C - B-C0 = -0.006525`. On H1 it is positive and small. These are different findings.
- Not that `4` trials is a limit of our method. It is the number of calibration trials the dataset
  provides.

---

## 8. Deliverables

Stages 0/0B/1 and their receipts are complete. The list below is the historical delivery contract;
the running Stage 2 matrix and terminal verifier follow the separate Stage-2 contract.

1. Stage 0: a short report answering every question in section 3, and the recorded branch of the
   stage-0 read rule. **Report and stop for a decision before stage 1.**
2. Stage 1: one immutable receipt, mode `0444`, in the isolated directory, with per-session values
   for every quantity in section 4.1, the four gate outcomes, and the terminal status.
3. A non-interference statement: no GPU, no process signalled, no watched directory written, thread
   caps used.
4. No interpretation, no recommended framing, and no paper text. Section 7 is a specification of what
   the writing must contain, not a request to write it.

---

## 9. What must not happen

| Item | Reason |
|---|---|
| Skipping stage 0, or skipping its second branch | Whether RT has native target metadata depends entirely on it; human labelling effort is not measured in either branch |
| Changing the lag, the bin size, the budget, the reach-segment rule, or the direction definition after seeing stage-1 numbers | Any change is a new pre-registration |
| Gating a fresh arm against the sealed `R-Full = 0.44195` | A fresh construction lost `0.06239` on identical windows |
| Reporting a mean without its median and sign count | Section 5.1 |
| Treating a stage-1 pass as evidence of a decoder gain | No CPU statistic in this program predicts `R²` |
| Any GPU use under this document | Stage 2 is not authorized here |
| Reopening the H1 discrete-label route inside the current program | Version-B failed its pre-registered gate. A genuinely different event form would require a new source-only protocol and must not be inferred from, or mixed into, the running RT matrix |
| Writing H1 or M1 as the same kind of boundary | Section 7.3 |

---

## 10. References

- `bci_paper_overleaf/sections/04_experiments.tex` `sec:rt_general_carrier` — the rank-deficiency
  sentence, the RT fit definition, and `tab:rt_general_carrier`.
- `CURRENT_RESULTS.md` 2026-08-09 — the H1 dataset audit: trial counts, trial durations, the
  `627 x 17` design matrix, and the Version-B event-carrier gate failure.
- `HANDOFF_RIDGE_BUDGET_CURVE_20260810.md` §2 — the non-interference contract.
- `sua_exploration/results/classical_comparator_inventory_v1/` — the mean-versus-median lesson that
  section 5.1 encodes.
- `ACTIVE_EXPERIMENT_CONTROL_BOARD.md` — the running routes that section 2 protects.
