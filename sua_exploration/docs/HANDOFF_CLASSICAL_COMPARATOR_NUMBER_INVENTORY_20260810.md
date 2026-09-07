# HANDOFF: complete number inventory for the classical-comparator comparison

**Status:** data-gathering protocol. **This document authorizes the CPU inventory directly.** There
is no launch, no GPU, no card contention, and no checkpoint write, so the GPU launch-integrity chain
does not apply. Two narrow gates remain, and both are self-executed: the section-6.2 specification
freeze, and the section-8 check-only limit.
**Date:** 2026-08-10
**Written in:** ASD-STE100 Simplified Technical English.
**Compute class:** **CPU only. Zero GPU. Zero training. Zero new data scope.**
**Purpose:** collect every number that a later analysis could reasonably need, in one frozen matrix.
**Explicit non-purpose:** this document draws **no conclusion** and proposes **no framing**. That work
happens after, in a separate document, on the complete matrix.

---

## 1. The integrity design. Read this before anything else

The plan is to gather the numbers first and decide the presentation later. That plan is legitimate
**only** under one condition, and this section makes it a mechanism rather than an intention.

> **Freeze the complete cell list before running. Report every cell in the receipt. Report every cell
> in any table that reaches the paper.**

With the full matrix frozen and published, the later writing decision is about **emphasis, ordering,
and framing** — which is normal scientific writing. Without it, the same activity becomes selective
reporting. The difference is entirely whether a cell can disappear after someone has seen it.

Three rules follow, and they are binding:

1. **No cell may be dropped after it is computed.** If a cell is impossible to compute, record the
   reason in the receipt before moving on. A missing cell must be visible.
2. **This document contains no read rules that select a winner.** It has one output rule: report
   everything. Interpretation is a separate, later, reviewed step.
3. **Numbers that are unfavourable are in scope and are named explicitly below.** Sections 6.3, 9.2,
   and 9.3 exist for that reason. If the inventory only contained favourable quantities it would not
   be an inventory.

---

## 2. Non-interference contract

H1 `CI64` occupies both static GPU partitions and the RT `L-D` v8 watcher is armed on GPU1. The rules
of `HANDOFF_RIDGE_BUDGET_CURVE_20260810.md` §2 apply unchanged and in full. The three that were
closest to being missed last time:

| # | Rule |
|---:|---|
| 1 | `CUDA_VISIBLE_DEVICES=""` for every process. No CUDA context, not even to query a device. |
| 2 | Cap `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` to a small fixed value and run under `nice`. A PCA plus ridge sweep over 15 sessions and 3 budgets will otherwise take every core through BLAS and starve the loaders feeding the two training jobs. This is the most likely and the most silent way to disturb a running experiment. |
| 3 | All output to one new isolated directory, for example `sua_exploration/results/classical_comparator_inventory_v1/`. No writes under `SPINT-main/pilot_artifacts/`, under `h1_carrierid_date_lodo_ci/`, or into any receipt directory bound by an armed watcher. |

Also: do not start, stop, or signal any process this work did not start; do not take over a card that
frees; read-only on every existing artifact; no `git`, `pip`, or `conda`; open no formal, minival, or
EvalAI path.

---

## 3. Inventory A — numbers that already exist. Collect, do not recompute

Copy these into the inventory receipt with their source path and SHA. **Do not recompute them.**

| Item | Value | Source |
|---|---|---|
| T4 external budget curve, SUA | `M=10/15/20/30/40/50` -> `0.3043 / 0.3381 / 0.3518 / 0.3582 / 0.3516 / 0.3568` | `tab:low_budget` |
| T4 external budget curve, pseudo-MUA | `0.2592 / 0.2882 / 0.2984 / 0.3053 / 0.3011 / 0.3061` | same |
| T4 deltas from `M=50` | SUA `-0.0526 / -0.0187 / -0.0051 / +0.0013 / -0.0052 / 0` | same |
| Pooled label fraction per budget | `3.49 / 5.24 / 6.98 / 10.47 / 13.96 / 17.45` % | same |
| `M=15` hierarchical intervals | `[-0.0476,0.0101]` SUA, `[-0.0511,0.0137]` pMUA; per-session margin holds 10/15 and 9/15 | `sec:label_budget` |
| `M=30` intervals | `[-0.0098,0.0126]`, `[-0.0129,0.0097]` | same |
| External classical table | F0-B3 `-0.1961/-0.2047`; PV50 `0.1154/0.1042`; Ridge50 `0.4179/0.4102`; T4 `0.3568/0.3061` | `tab:external_classical` |
| `T4 - PV50` | `+0.2415` `[0.1737,0.3005]`, `+0.2019` `[0.1262,0.2655]`, 14/15 both views | same |
| `T4 - Ridge50` | `-0.0611` `[-0.1752,+0.0699]`, `-0.1041` `[-0.2116,+0.0046]`; Ridge higher 12/15 and 13/15 | same |
| Ridge budget curve `M=15/30` | SUA `0.0726 / 0.3222`; pMUA `0.1291 / 0.3414` | `ridge_budget_curve_v1` receipt `591cb405...` |
| Native-M2 Ridge-W50 closure | direct ridge `0.1139`; legacy B3/F0 `0.1739`; T4 `0.2268`; K4 `0.2458`; 24-trial budget, 96 channels | `04_experiments.tex` |
| True-early-start replay | trials 31--50: T4 `0.3937` SUA, `0.3393` pMUA | `tab:true_early` |
| Identity cost, H1 | `102.6x` parameters, `98.9x` identity MACs | `CURRENT_RESULTS.md` |
| Identity cost, M1 | `66.53x` parameters, `50.22x` identity MACs | same |
| Amortized calibration share | about `1/85,000` of total system compute | `03_methodology.tex` |

---

## 4. Inventory B — descriptive completion of the existing ridge curve. Zero new fits

Everything here is recomputable from `ridge_budget_curve_v1/ridge_budget_curve_receipt.json`. The
existing deliverable reported means only. The inventory must carry the full descriptive set, because
the mean and the median disagree in every cell.

For each of `M in {15, 30, 50}` and each view, from `per_session_delta_r2` and `per_session_r2`:

1. mean, **median**, min, max of the paired `T4 - Ridge` difference;
2. **sign count** (sessions where T4 is higher, out of 15);
3. the existing bootstrap interval;
4. **leave-one-out-max sensitivity**: the mean after removing the single largest paired difference;
5. per-session ridge `R²`: mean, median, min, and the **count of sessions below zero**;
6. the identity of every session whose ridge `R²` is negative, at every budget.

For reference, the values already recomputed from that receipt, which the inventory must reproduce
exactly:

| cell | mean | median | T4 higher | mean after dropping the single largest |
|---|---:|---:|---:|---:|
| SUA `M=15` | `+0.2655` | `-0.0029` | 7/15 | `-0.0088` |
| SUA `M=30` | `+0.0359` | `-0.0198` | 7/15 | `-0.0738` |
| pMUA `M=15` | `+0.1591` | `-0.0342` | 7/15 | `-0.0506` |
| pMUA `M=30` | `-0.0362` | `-0.0400` | 4/15 | `-0.1136` |

Ridge per-session negatives: 2/15 at `M=15` in both views, worst `-3.776` SUA and `-2.824` pMUA;
1/15 at `M=30`, worst `-1.230` and `-0.734`.

**If any of these do not reproduce, stop and report.** They are the arithmetic basis of everything
that follows.

---

## 5. Inventory C — session covariates and the conditioning hypothesis

**Question this answers.** Are the ridge failures a property of the method at low labels, or a
property of the sample-to-feature ratio in particular sessions?

For every session, every budget, and every view, record:

- number of rewarded trials in the session, and the number used for calibration;
- eligible calibration rows, and feature dimension (`50 x N_units`);
- `N_units` for that session and view;
- **rows / features ratio**;
- median regularized condition number and `trace(H)` (already per-session in the existing receipt);
- the resulting ridge `R²`.

Then report, as a plain table with no conclusion attached, the ridge `R²` against the rows/features
ratio across all `15 x 3 x 2` cells.

**Cross-cohort extension.** Compute the same ratio for the native-M2 Ridge-W50 setting: 96 channels,
50-bin history, 24-trial budget. The external cohort medians are `2750` features on SUA (55 units)
and `1950` on pseudo-MUA (39 units), against about `3053` rows at `M=15` and `5750` at `M=30`.
Report M2's rows, features, and ratio next to them. This is the only available check on whether the
two cohorts' opposite ridge outcomes share one explanation.

---

## 6. Computation D — the ridge fairness upgrade

**This computation is expected to make the comparator stronger. Run it anyway.** Its purpose is to
remove an attack, not to win one.

### 6.1 What changes

The `M=15` design is nearly square: about `3053` rows against `2750` features. A competent
practitioner would reduce the feature dimension using the calibration block. The current comparator
does not, so a reviewer can say the baseline was crippled.

Give the ridge the **same first-30-trial block that T4's activity path already uses**, for
unsupervised purposes only:

- standardization statistics from all 30 trials;
- a PCA basis fitted on all 30 trials of neural windows, with the labelled subset then used for the
  supervised fit.

Labels stay at `M`. Only unlabelled structure comes from the 30-trial block. This matches the
information T4's activity path consumes, which `tab:low_budget` states is "the same first-30 neural
activity support" at every budget.

### 6.2 The specification freeze. Self-executed, no reviewer needed

The load-bearing thing here is the **freeze**, not a reviewer. The free choices below can move the
comparator up or down. If they are made after the scores exist, the number means nothing and anyone
can say so. If they are fixed in an immutable file before the scores exist, the number stands on its
own. That protection comes from the timestamp and the hash, not from a second person.

**Procedure.** Write one specification file in the isolated output directory. Compute its SHA-256.
Only then compute anything. Do not modify it afterwards. Deliver it with the receipt, and record its
hash inside the receipt.

**Contents, all of them:**

- the component-count rule: a fixed variance threshold or a fixed count, with the exact value, chosen
  **without reference to any target score**;
- which windows the PCA is fitted on: all 30 trials, or only the `M` labelled trials;
- the source range of the standardization statistics;
- the penalty rule, unchanged: mean-normalized Gram, `lambda = 1.0`, no target-session selection;
- the query set, unchanged: trials `[50:end]`, identical to every existing cell;
- the session set and both views, unchanged;
- the handling of a degenerate session, for example one whose component count cannot be reached, and
  what is recorded for it.

The last item exists because an underspecified rule produces a number that cannot be interpreted, and
then the work is repeated. That already happened once on the first budget curve, where the read rules
were written on means only and the outcome could not be mapped onto them.

### 6.3 Cells to report, all of them

`M in {15, 30, 50}` x `{SUA, pseudo-MUA}` x `{plain ridge, PCA ridge}`, with the full descriptive
set of section 4 for each. **The plain-ridge cells at `M=50` already exist and must be carried
through unchanged so that the two ridge variants are comparable.**

**Named unfavourable outcome, in scope:** if the PCA ridge removes the catastrophic sessions, then
the low-budget tail advantage disappears. That result is reported in the receipt with the same
prominence as any other.

---

## 7. Computation E — the PV budget curve

`PV50` is the closest thing to an equal-label classical comparator: it uses the same 50 direction
labels, plus dense velocity for a two-dimensional affine gain. It currently exists at `M=50` only.

Compute `PV` at `M in {15, 30}` under the identical fixed-query protocol, both views, full
descriptive set. This completes the classical-comparator panel so that the paper is not reporting a
budget curve for one classical method and a single point for the other.

---

## 8. Computation F — conditional: cross-session ridge on a fixed channel set

**Do the data-boundary check first, and report it before proposing any run.**

The paper's claim is cross-session deployment with a frozen decoder. Every ridge number so far is
within-session, which is the easiest condition for a per-session refit. A symmetric cross-session
comparison is well posed **only** where the channel set is stable: native M2 (96 channels) and H1
(176 persistent channels). It is not well posed on sorted SUA, where unit sets differ across days and
the ridge weight vector has no correspondence.

**Boundary check, zero compute, report before proceeding:**

1. For M2 and H1, which session pairs has a prior sealed receipt already opened? Cite each.
2. Do matched carrier cross-session numbers already exist for exactly those pairs?
3. Can the ridge be applied across a pair without changing its definition, given the channel set?

**If any answer is no, stop and report.** Do not construct a new data boundary.

**If all answers are yes**, the pre-registration must contain a symmetric read rule stating that a
competitive cross-session ridge is reported as such. It must also state plainly that a per-session
linear readout is not designed for transfer, so this measures the cost of per-session refitting when
the session changes, and is not a claim that the ridge is a poor method. Without that clause this
becomes the straw-man arm that
`HANDOFF_EXPERIMENT_SOFT_SPOTS_AND_PIVOT_20260810.md` controlling order 4 forbids.

---

## 9. Inventory G — requirements and cost, zero compute

Scores alone do not describe what each method needs. Build one table from existing receipts and the
methodology section. **It must include the rows where the comparator is better.**

### 9.1 Per new session

| Quantity | T4 + frozen decoder | Ridge | PV |
|---|---|---|---|
| Labelled trials | `M` | `M` | `M` |
| Label type | one direction scalar per trial | dense per-bin 2-D velocity | direction scalar plus dense velocity for the gain |
| Unlabelled neural trials used | 30 | 0 in the current arm; 30 in the section-6 arm | to be recorded |
| Backward passes | 0 | 0 | 0 |
| Decoder weights changed | none | all | all |
| Calibration compute | one closed-form solve | one closed-form solve | one closed-form solve |
| Cached state per session | carrier and identity token | full readout matrix | readout plus gain |

### 9.2 Structural properties. Record the honest entry, not the flattering one

- unit-count invariance: T4 with a permutation-invariant decoder handles arbitrary `N`; a ridge
  readout of shape `(50 x N) -> 2` is tied to one unit set and count;
- transferability without new labels: record what each method can do, including that the ridge
  cannot transfer at all in the sorted-unit setting, **and** that this is a design property of a
  per-session decoder rather than a defect measured by us.

### 9.3 Online cost. This one favours the comparator

Record the per-bin online cost of both. The ridge readout is roughly `50 x N` multiply-accumulates
per bin, which is **cheaper** than the streaming transformer decoder. Record the actual numbers from
`03_methodology.tex` and the cost receipts. A cost table that omits the row where we lose is not an
inventory.

---

## 10. Output rules

- One immutable receipt, mode `0444`, in the isolated directory, containing **every cell** of
  sections 4 to 9, each with its source or its recorded reason for being impossible.
- Per-session values for every new computation, not only aggregates. The existing deliverable showed
  why: the mean and the median disagreed in all four cells and only the per-session values revealed
  it.
- One table per section. **No conclusion paragraph, no recommended framing, no selected headline.**
  Interpretation is a later, separately reviewed step on the complete matrix.
- A non-interference statement: no GPU, no process signalled, no watched directory written, thread
  caps used.
- A single explicit list of any cell that could not be computed, with the reason.

---

## 11. What must not happen

| Item | Reason |
|---|---|
| Dropping, deferring, or omitting a cell after seeing it | Section 1. This is the one rule that makes the whole plan legitimate |
| Choosing the PCA component rule, the penalty, or the budget list after seeing scores | It converts a fairness control into a constructed result |
| Writing a conclusion, a headline, or a recommended framing in this deliverable | Section 10. That step is separate and comes later |
| Any GPU use, or reacting to a freed card | Section 2 |
| Changing the query set, the session set, or the ridge definition beyond what section 6 specifies | The existing `M=50` cells stop being comparable |
| Constructing a new data boundary for section 8 | Section 8 |
| Recomputing or superseding sealed T4, PV50, Ridge50, or V9 values | They are the reference |
| Presenting any output as a confirmation, a hidden-endpoint result, or an upper bound on supervised recalibration | Every number here is development evidence |

---

## 12. Order of work

1. Section 2 contract in place; isolated directory created.
2. Section 3 collection, and section 4 reproduction. **Stop and report if section 4 does not
   reproduce.** It is the arithmetic basis of everything after it.
3. Section 5 covariates.
4. Section 6.2 specification file written and hashed, then section 6 computed. No reviewer is
   required. The file must exist and be hashed before the first score is produced.
5. Section 7.
6. Section 8 boundary check only. Report the three answers and stop there.
7. Section 9 table.
8. Receipt, then hand back. No interpretation.

**The only two hard stops are step 2 and the section-8 limit.** Everything else runs straight
through.

---

## 13. References

- `HANDOFF_RIDGE_BUDGET_CURVE_20260810.md` — the §2 non-interference contract and the first budget
  curve.
- `sua_exploration/results/ridge_budget_curve_v1/` — receipt `591cb405...`, per-session values, and
  the deliverable report whose mean-only reading section 4 corrects.
- `bci_paper_overleaf/sections/04_experiments.tex` — `tab:external_classical`, `tab:low_budget`,
  `tab:true_early`, `sec:label_budget`.
- `bci_paper_overleaf/sections/03_methodology.tex` — the rate-domain cost analysis.
- `HANDOFF_EXPERIMENT_SOFT_SPOTS_AND_PIVOT_20260810.md` — controlling order 4, the straw-man rule.
- `ACTIVE_EXPERIMENT_CONTROL_BOARD.md` — the running routes that section 2 protects.
