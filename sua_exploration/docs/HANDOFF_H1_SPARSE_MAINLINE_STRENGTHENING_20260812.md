# Handoff: Strengthening the H1 Sparse Mainline

**Date:** 2026-08-12
**Author role:** review and design proposal
**Target:** `bci_paper_overleaf/paper_6pp.tex` and the H1 experiment line
**Scope:** narrative restructuring, one new theoretical argument, five new experiments
**Constraint:** no sealed result may be reopened or reinterpreted; every proposal below either
uses existing artifacts or adds new arms that do not touch sealed ones.

---

## 1. The core problem

The paper argues that the carrier converts **sparse** supervision into a cached functional
identity. But Section 4.1 of `paper_6pp.tex` declares H1 an exception:

> "H1 is the dense-covariate regime in this study. [...] We therefore evaluate H1 as the
> higher-dimensional dense-carrier and compact-consumer case and reserve scalar label-density
> accounting for the lower-dimensional tasks."

H1 is the only human dataset, the only organizer-held-out result, and the only deployment-scale
evidence in the paper. Declaring it outside the sparse mechanism means the strongest result does
not demonstrate the core claim. A reviewer will notice this immediately.

**This is a presentation problem, not an evidence problem.** A compliant H1 sparse result already
exists and is currently absent from the paper.

---

## 2. The unused asset: H-SE5

The sealed review constraint forbids only:

> "No H1 Context Full, event-tag, tag-shuffle, or `0.5165` sparse-context claim."

**H-SE5 (`0.500037`) is not on that list.** Context Full was excluded because its per-recording
deltas against H-SE5 are `-0.004610/+0.077458`, failing the both-recordings-positive clause.
H-SE5 does not have that problem.

### H-SE5 evidence, fold-0, 8,965-window strict post-support query

| Contrast | Value | Recording uniformity |
|---|---:|---|
| H-SE5 - independently trained Zero5 | **+0.028469** | both positive: `+0.02590`, `+0.03593` |
| H-SE5 - same-checkpoint label shuffle | +0.007092 | both positive: `+0.00712`, `+0.00702` |
| H-SE5 - same-checkpoint row shuffle | +0.012544 | - |
| H-SE5 - same-checkpoint zero | +0.011442 | - |
| H-SE5 - H-S (activity-only SPINT) | +0.003204 | - |
| H-SE5 - dense H-C | -0.025474 | below dense |

### H-SE5 CPU estimator audit, all 13 public recordings

| Budget | correct - label-shuffle | correct - intercept | Retained variance (median/min) |
|---|---:|---:|---:|
| M=3 | +0.0238, **13/13** | +0.0101, **13/13** | 0.718 / 0.716 |
| M=4 | +0.0253, **13/13** | +0.0121, **13/13** | 0.718 / 0.716 |

Verified by terminal receipt, structural verifier, and an independent batch-29 recomputation.

### Why this matters mechanically

H-SE5 uses a **per-channel closed-form ridge**, which is the same estimator family as paper
eq. (7)-(8). It is not the H1 population estimator. This means H1 can be folded into the paper's
main Method instead of being described separately.

---

## 3. Part A - Narrative restructuring (no new experiments)

### A1. Unify the Method around one event-level estimator

Replace the current split (per-unit OLS for T4/RT, plus a prose-only paragraph for the H1
population estimator) with a single estimator plus a task-basis table.

```latex
g\big(r_i(e)\big) = b_i + \mathbf{w}_i^\top \boldsymbol{\phi}(a_e) + \varepsilon_i(e)
\qquad
\boldsymbol{\beta}_i = \big(X^\top X + n\lambda D\big)^{-1} X^\top \mathbf{y}_i,
\quad D = \mathrm{diag}(0, I_q)
```

`lambda = 0` recovers the pseudoinverse form already in the paper.

| Task | Event unit | Native annotation `a_e` | `phi(a_e)` | `q` | `g` |
|---|---|---|---|---:|---|
| Center-out | trial | target direction `theta_m` | `[cos, sin]` | 2 | identity |
| FALCON M2 | trial | target direction `theta_m` | `[cos, sin]` | 2 | identity |
| RT | go-cue reach | endpoint pair | unit displacement direction | 2 | identity |
| **FALCON H1** | movement epoch | 7-DoF endpoint pair | `P_src * standardize(dq)` | 4 | `log1p` |

**Effect.** H1 stops being an exception and becomes the highest-dimensional instance of one
mechanism. The thesis then covers 4 datasets, 2 species, 2-DoF to 7-DoF, and sorted units to
threshold channels. This also fixes the previously flagged P0 issue that the H1 estimator has no
equation.

**Keep** the H-C population estimator, but demote it to a short paragraph describing the
dense-carrier deployment system that produced the organizer-held result. It is a variant, not the
main method.

### A2. Put H1 on the supervision-density figure

`fig_supervision_density` currently shows only the external-subject cohort and RT. Add H1:

| H1 fold-0 system | Target coordinates consumed | Pooled R2 |
|---|---:|---:|
| H-SE5 sparse endpoint carrier | 574 acquired endpoint coordinates | **0.500037** |
| Ridge v2r2 dense readout | 42,616 velocity coordinates | 0.258235 |

Ratio `74.2x` fewer coordinates at roughly twice the score, on the same query windows.

This is the strongest single supervision-efficiency point in the project. It also rescues the
orphaned Ridge v2r2 paragraph in Section 3.5, which currently sits with heavy disclaimers and no
narrative purpose.

Keep the existing caveat: ridge is a direct per-session readout, H-SE5 is source-pretrained, so
this is not architecture-matched. The figure measures supervision consumption, which is exactly
the claim being made.

### A3. Report the 13-recording constructibility audit

The paper already does this for RT (split-half cosine, correct-minus-shuffle 15/15). Do the same
for H1: 13/13 recordings positive on both controls at both budgets. This converts "one decoder
fold" into "one decoder fold plus a 13-recording estimator audit," which is much harder to
dismiss as a single lucky cell.

### A4. Report the combined sparse + compact claim

`falcon_h1_sparse_event_endpoint.yaml` sets `carrier_hidden_dim: 32`. H-SE5 runs through the
**same compact 32-wide consumer** as H-C. So there is an unreported combined result:

> On human 7-DoF data, event-level endpoint labels feeding a 58k-parameter identity path
> (102.6x smaller than the 5.97M activity-only path) match or slightly exceed the full
> activity-only SPINT identity encoder (0.5000 vs 0.4968).

Sparse supervision and 102.6x compression hold **simultaneously**. The paper currently reports
compression only for the dense H-C arm, so this combination is invisible.

### A5. Note that H1 is already the unsorted regime

The paper's pooling-linearity argument predicts survival without spike sorting. H1 is 176
threshold-crossing channels with no sorting. So H1 already confirms that prediction on human data
without needing a sorted-to-pooled comparison. One sentence, zero cost.

**Verify before using:** confirm FALCON H1 ships binned threshold crossings rather than sorted
units.

---

## 4. Part B - New theoretical argument (no new experiments)

This is the biggest missing piece and the cheapest to add. The paper counts labels but never
explains **why** the sparse route can win. There is a clean statistical reason.

### B1. Separability asymmetry between encoding and decoding

Two estimation problems are being compared on the same calibration block.

**Direct decoding readout (ridge).** One joint problem mapping `N*W` neural features to `C`
outputs. Parameters are coupled across all channels. Sample ratio:

```
rows / (N * W)
```

**Encoding carrier.** `N` independent problems, each with `q+1` parameters, all sharing one
design matrix `X`. Sample ratio per problem:

```
n_events / (q + 1)
```

**The key point: the carrier's per-parameter sample ratio does not depend on `N`. The direct
readout's degrades as `1/N`.**

### B2. Instantiate on H1 M4 — MEASURED 2026-08-12, see Section 10

An earlier draft of this section used `41 events / 5 params = 8.2` observations per parameter.
**That was wrong.** 41 is the two-recording fold-0 total, but the carrier is fitted *per session*.
The measured per-session values are below. Receipt:
`sua_exploration/results/h1_sample_complexity_audit/audit.json`,
SHA `d336aa5ab5646ef7b12777e4a846f5f6b074970fd028257ad559750af35ba6db`.

| Estimator | Parameters | Observations | Observations per parameter (median, M4) |
|---|---:|---:|---:|
| Ridge v2r2 (50 bins x 176 ch -> 7 DoF) | 8,800 per output | ~3,100 eligible rows | **0.356 (underdetermined, 13/13 sessions)** |
| H-SE5 carrier (per channel, shared design) | 5 | ~20 events | **4.0 (overdetermined)** |

Median posedness ratio **11.4x** at M4 and 11.3x at M3. The ridge is underdetermined in **all 13**
sessions. The argument survives with the corrected numbers, and the print-ready claim is that the
carrier poses an overdetermined 4-observations-per-parameter problem while the evaluated dense
readout is underdetermined by roughly 3x.

**Two framing requirements.**
1. The two estimators use different observation units (events vs bins) and different targets
   (log rate vs velocity). State this as "the two estimators pose problems with very different
   sample-to-parameter ratios on the same calibration block," never as "same data, different
   method."
2. A reviewer can object that a ridge could shrink `W` or compress channels to cut parameters.
   Scope the claim to the *evaluated* ridge implementation, matching how the paper already hedges
   every other ridge comparison.

### B2b. Supervision coordinate accounting — verified

The 13-session M3 totals reproduce the sealed `source_audit_v2r2.json` label accounting **exactly**
(199 support events, 2,786 endpoint coordinates, 30,879 dense bins, 216,153 dense coordinates; all
differences zero). Fold-0 M4 support events (41) and endpoint coordinates (574) also reproduce
exactly.

One discrepancy, understood and benign: raw eval-valid bins in the fold-0 M4 support give 6,473
dense rows / 45,311 coordinates, whereas the sealed Ridge v2r2 receipt records 6,088 / 42,616. The
difference (385 rows) is the 50-bin causal-history constraint, which drops roughly 49 leading bins
per calibration trial. **Print the sealed 42,616 figure**, which is bound to the actual ridge run
and is the more conservative of the two.

### B3. Calibration cost is one small solve

Because `X` is shared across channels, the entire H1 calibration is one `5x5` solve applied to a
`41 x 176` response matrix:

```
B_hat = (X^T X + n*lambda*D)^{-1} X^T Y,     Y in R^{41 x 176}
```

Cost is `O(n q^2 + q^2 N)`. This is a much more convincing deployment statement than the current
vague "session overhead amortizes to roughly 1/85,000 of streaming compute."

---

## 5. Part C - New experiments, ranked by cost

### Critical enabler

`SPINT-main/scripts/h1_sparse_event_endpoint_evaluate.py` loads a **sealed checkpoint** and
evaluates it with substituted carriers. The existing `zero`, `row`, and `label` interventions are
already forward-only. Therefore any experiment that changes only the **target-session carrier
fit** requires no training.

---

### C1. Forward-only event-budget sweep (nearly free) - HIGHEST VALUE

**Question.** How many movement events does the human 7-DoF carrier actually need?

**Design.** Keep the sealed H-SE5 Full checkpoint and the exact 8,965-window query fixed.
Recompute the target carrier using only the first `k` valid support events, for
`k` in `{8, 12, 16, 24, 32, 41}`. Evaluate forward-only.

- `k >= 8` respects the frozen fail-closed floor (protocol Section 5).
- Every point compares against the **same** sealed Zero5 (`0.471569`) and H-S (`0.496833`)
  references, so no new baselines are needed.
- No training, no source change, no new query boundary.

**Deliverable.** The H1 analogue of Fig. 4, on human data.

**Honest framing.** This is a deployment-time budget sweep on a fixed source-trained model, not a
matched retraining per budget. State that explicitly. It is also the deployment-relevant question:
you ship one trained model and calibrate with whatever labels the session provides.

**Outcome value in all directions.**
- Flat down to `k=8`: strong result, "8 movement events suffice on human 7-DoF."
- Degrades at low `k`: still gives a defensible operating point and an honest curve.
- Either way the paper gains a dose-response curve on its flagship dataset.

---

### C2. Forward-only conditioning-controlled diversity sweep (nearly free)

**Question.** Does performance track event **count** or event **diversity**?

**Motivation.** The paper's closing contribution (vi) asserts:

> "event diversity, label geometry, coefficient dimension, and temporal dependence jointly
> matter---not trial count alone."

This is currently **asserted, not measured**. It is the weakest-supported claim in the paper, and
it is the closing sentence of the abstract and the conclusion.

**Design.** Hold `k` fixed (e.g. `k=16`). Select event subsets with deliberately different design
conditioning `kappa(X)` — greedy maximum-spread versus greedy minimum-spread in the `q4`
coordinate space. Evaluate forward-only on the sealed checkpoint. Report R2 against `kappa(X)`.

**Deliverable.** If R2 tracks `kappa(X)` rather than `k`, claim (vi) becomes an empirical law
instead of an assertion, measured on human data. `kappa(X)` also connects directly to the
"well-conditioned and diverse events" language already in Section 4.1.

**Risk is informative.** If R2 does not track `kappa`, claim (vi) is wrong and should be softened
before a reviewer finds it.

---

### C3. Separately trained shuffle controls on H1 (2 GPU runs)

**Gap.** H-SE5's label-shuffle and row-shuffle controls are **same-checkpoint only**. Every other
dataset in the paper has **separately trained** corruption arms (TS4/LS4 on center-out, XLSv2 on
RT). Same-checkpoint controls are weaker because they do not test whether a model could have
learned to exploit shuffled content.

A reviewer comparing control tables across datasets will notice that H1's controls are weaker
than the rest of the paper.

**Design.** Train separately initialized H-SE5-LS and H-SE5-RS arms, identical source schedule and
seed, on fold-0. The frozen protocol (Section 7 of the sparse-endpoint protocol) already specifies
both arms. The sparse handoff lists them as "optional precision work, not missing implementation."

**Cost.** 2 runs x 50 epochs. Both 3090s are idle.

---

### C4. Second date for H-SE5 (2 GPU runs)

**Gap.** H-SE5 is fold-0 only (date `19250101`, 2 recordings). This is its single biggest
weakness.

**Design.** Run H-SE5 Full + Zero5 on one additional date. The five-date LODO infrastructure
already exists (`19250108`, `19250113`, `19250115`, `19250119`, `19250120`).

**Payoff.** Upgrades "positive first cell" to "positive across 2 dates / 4 recordings," which is a
legitimate multi-date generalization statement rather than a single cell.

**Cost.** 2 runs x 50 epochs.

---

### C5. M3 organizer-budget cell (3-4 GPU runs)

**Motivation.** H-SE5 passed the CPU gate at M3, and **M3 is the organizer's official calibration
budget**. A matched M3 decoder cell would let the paper say: on the human 7-DoF benchmark, three
calibration trials (roughly 13-18 movement events) suffice.

**Constraint.** The protocol forbids reusing the M4 references: "M=3 must not reuse the M=4 H-S
reference." M3 changes the query boundary, so it needs its own matched H-S and Zero5 arms.

**Cost.** 3-4 runs. Lower priority than C1-C4 because C1 already delivers a budget curve at a
fraction of the cost, using the M4 query boundary that all sealed references share.

---

## 6. Part D - Items to verify (each is a potential asset)

1. **Are H1 endpoint labels native task metadata?** The RT sparse claim is weakened because
   directions are derived from recorded position. For H1, calibration is an open-loop block where
   the subject observes prescribed movements. If `OpenLoopKinematics` during calibration is the
   task-prescribed trajectory rather than a decoded output, then **H1 is the only dataset where
   the sparse annotation is genuinely native**, which is strictly stronger than RT. Verify against
   the FALCON H1 protocol before claiming it.

2. **Confirm H1 is threshold crossings, not sorted units** (supports A5).

3. **Confirm the 58k parameter count applies to the H-SE5 arm**, not only H-C. The sparse handoff
   says the 5-wide carrier "adds 32 weights relative to the four-dimensional version," so it
   should be about 58k, but the exact number must be recomputed before printing.

---

## 7. What not to do

- **Do not try to revive Context Full or tag content.** Ten CPU design programs (TCE5, nested
  context, meta-learned basis, semantic V4, C2F5, LRT5, NLE5, PNO5, QC2F5) all plateaued at
  `+0.012` to `+0.016`, below the `+0.02` gate. That route is exhausted.
- **Do not claim H-SE5 beats dense H-C.** It is `-0.0255` below.
- **Do not reopen H1 architecture search.** CI64 is terminal negative.
- **Do not frame H-SE5 as beating SPINT.** The margin over H-S is only `+0.0032`. The load-bearing
  contrast is against the independently trained Zero5 (`+0.028469`), which is how every other
  dataset in the paper is framed.
- **Do not use `git add .`** in this dirty multi-agent worktree.

---

## 8. Recommended order

| Step | Item | Cost | Unlocks |
|---|---|---|---|
| 1 | A1 unified estimator + task-basis table | writing | H1 joins the mainline; fixes missing-equation P0 |
| 2 | B1-B3 separability argument | writing | explains *why* sparse works |
| 3 | A2-A5 figure, audit, combined claim | writing | supervision-efficiency headline |
| 4 | C1 event-budget sweep | forward-only | human-data dose-response curve |
| 5 | C2 conditioning sweep | forward-only | turns claim (vi) into a measurement |
| 6 | C3 separately trained LS/RS | 2 GPU runs | control parity with other datasets |
| 7 | C4 second date | 2 GPU runs | multi-date generalization |
| 8 | C5 M3 organizer budget | 3-4 GPU runs | organizer-budget deployment claim |

Steps 1-5 require no training and would already change the paper from "sparse method plus a
separate dense human result" to "one sparse mechanism validated from 2-DoF monkey reaching to
7-DoF human control, with a statistical account of why it works."

---

## 10. CPU screen results, 2026-08-12 — C1 and C2 executed

Both screens were built, tested, and run before any GPU time was spent. They changed two planned
claims. New files (nothing existing was modified):

- `sua_exploration/mc_maze/h1_event_budget_sweep.py` + runner + tests
- `sua_exploration/mc_maze/h1_event_conditioning_sweep.py` + runner + tests
- `sua_exploration/scripts/audit_h1_sample_complexity.py` + tests

All three screens reproduce the sealed H-SE5 forward-transfer numbers to **exactly 0.0** absolute
difference, so they are measuring the sealed estimator and not a variant.

### 10.1 C1 event-budget sweep — the sparsity-headroom hypothesis is FALSIFIED

Support window fixed at 4 trials; the later-event evaluation set is identical at every point.
Per-session M4 support is 17-23 events (median 20, total 262 over 13 sessions).

Reported on a common 13-session set. `k=20` is excluded from this table because only 9 sessions
have 20 or more support events; on a matched 9-session set `k=20` and `all` are equivalent
(`+0.014216` vs `+0.013503`), so there is no "20 beats all" effect.

| k | mean delta-intercept | median | positive | mean carrier fidelity | % of full effect |
|---:|---:|---:|---:|---:|---:|
| 8 | +0.000321 | +0.001720 | 8/13 | 0.722 | **2.7%** |
| 10 | +0.006250 | +0.004713 | 10/13 | 0.829 | 51.5% |
| 12 | +0.006997 | +0.004536 | 12/13 | 0.862 | 57.7% |
| 14 | +0.007908 | +0.008325 | 12/13 | 0.907 | 65.2% |
| 16 | +0.009747 | +0.010638 | 12/13 | 0.939 | 80.4% |
| all (~20) | +0.012127 | +0.011506 | 13/13 | 1.000 | 100% |

**The curve is monotonic with no plateau.** The H1 carrier does not saturate below its full
~20-event budget, and at 8 events it retains under 3% of the effect. Therefore:

- **Do not claim H1 has sparsity headroom.** The existing H-SE5 operating point is already the floor.
- This does not weaken the existing H-SE5 result, which already uses all ~20 events and is still
  74x sparser in target coordinates than the dense ridge.
- Cross-check worth noting: the sealed M3/M4 ratio is `0.0101/0.0121 = 83%`, better than the ~70%
  this curve predicts at a comparable event count. The reason is that M3 is *trial-aligned* whereas
  this sweep truncates mid-trial. Trial-aligned truncation is the better sparsification axis.

### 10.2 C2 conditioning sweep — diversity is real, but the condition number is the wrong statistic

The subagent reported only a pooled correlation across all session-rule cells, which confounds
within-session diversity effects with between-session baseline differences. The corrected
within-session analysis:

| Relationship | k=10 | k=14 |
|---|---:|---:|
| within-session Spearman, log10 cond vs delta-intercept | -0.17 (7/13 negative) | -0.19 (9/13 negative) |
| within-session Spearman, log10 cond vs carrier fidelity | **-0.56 (12/13)** | **-0.45 (12/13)** |

Paired within-session rule contrasts (sign test):

| Contrast | metric | k=10 | k=14 |
|---|---|---|---|
| maxspread - minspread | delta-intercept | +0.0062, 10/13, p=0.092 | **+0.0052, 11/13, p=0.023** |
| maxspread - minspread | carrier fidelity | **+0.357, 13/13, p=0.0002** | **+0.164, 13/13, p=0.0002** |
| maxspread - **first** | delta-intercept | +0.0013, 9/13, p=0.27 | +0.0028, 8/13, p=0.58 |
| first - minspread | delta-intercept | +0.0049, 10/13, p=0.092 | +0.0023, 9/13, p=0.27 |

Three conclusions:

1. **Diversity at fixed event count is real.** Spread-out event selection beats clustered selection
   on forward transfer (11/13 at k=14) and on carrier fidelity (13/13 at both k).
2. **The condition number is not the operative variable for forward transfer.** It reliably
   predicts *carrier fidelity* (12/13) but not *forward transfer* (7-9/13, weak). The paper's
   phrase "enough **well-conditioned** and diverse events" in Section 4.1 should drop
   "well-conditioned" or restrict it to carrier stability.
3. **Optimized event selection buys nothing over the chronological default.** `maxspread - first`
   is insignificant on both metrics. There is no deployment gain from choosing which events to
   annotate; taking the first k in time is already near-optimal. This closes a design idea rather
   than opening one.

### 10.3 Consequences for the paper's contribution (vi)

The paper currently closes both the abstract and the conclusion with: "event diversity, label
geometry, coefficient dimension, and temporal dependence jointly matter---not trial count alone."
The measured position is more specific and partly different:

- Event **count** is a first-order driver on H1 and should not be downplayed.
- **Diversity** at fixed count is a real but secondary effect.
- **Conditioning** is not the right scalar summary for transfer.

Revise the claim accordingly instead of restating it.

### 10.4 Effect on the GPU plan

- **C3 (separately trained LS/RS) and C4 (second date) are unaffected** and remain the best GPU
  spend. They concern control rigor and generalization, not sparsity headroom.
- **C5 (M3 organizer-budget cell) is downgraded but not dead.** C1 shows no headroom, yet the
  sealed M3 CPU audit is 83% of M4 and passes every gate 13/13 because M3 is trial-aligned. Run it
  only after C3 and C4.

---

## 11. Trial-aligned budget and evaluation-boundary decomposition, 2026-08-12

Motivation: Section 10.1 noted that the sealed M3 result is 83% of M4 while the C1 event sweep at a
comparable count gave only ~70%, and attributed the gap to trial alignment. That inference was
confounded: the sealed M3 scores on `trial_index >= 3` while C1 fixes the evaluation set at `>= 4`.
This round separates the two.

Built and run by two agents in parallel: a primary screen
(`sua_exploration/mc_maze/h1_trial_aligned_sweep.py` + runner + tests) and a deliberately
**independent** cross-check (`sua_exploration/scripts/verify_h1_trial_aligned_decomposition.py` +
tests) that re-derives the ridge solve and the R-squared from scratch and never reads the primary
implementation.

**Verification status: the two implementations produce bit-identical values** — 52 numeric cells
(13 sessions x 4 defined cells), `max|diff| = 0.000e+00` on both `median_delta_intercept` and
`median_r2_correct`. Both also reproduce `v2.forward_transfer` at budgets 3 and 4 to exactly 0.0.

Receipts:
- primary `sua_exploration/results/h1_trial_aligned_sweep/source_screen.json`, SHA `58fcb777ae00c425d4d73a1271bd94cd4c086a51ebd99346e72e67dd65ab2db6`
- independent `sua_exploration/results/h1_trial_aligned_sweep/independent_crosscheck.json`, SHA `37c642a87e83280afbb316f262281b728ae56870c0a49d01b30f7b0f36beb354`

### 11.1 Trial-aligned dose-response, evaluation set fixed at `trial_index >= 4`

| Cell (M,E) | defined | mean delta-intercept | positive | carrier fidelity | % of M4 |
|---|---:|---:|---:|---:|---:|
| (1,4) | **0/13** | undefined | - | - | - |
| (2,4) | 13/13 | +0.005973 | 11/13 | 0.834 | 49.3% |
| (3,4) | 13/13 | +0.010086 | 13/13 | 0.926 | 83.2% |
| (4,4) | 13/13 | +0.012127 | 13/13 | 1.000 | 100% |
| (3,3) sealed M3 | 13/13 | +0.010067 | 13/13 | 0.926 | - |

### 11.2 The decomposition — the evaluation boundary does not matter

| Quantity | mean | median | positive | share of total |
|---|---:|---:|---:|---:|
| `eval_effect` = Cell(3,3) - Cell(3,4) | **-0.000019** | -0.000799 | 5/13 | **0.9%** |
| `support_effect` = Cell(3,4) - Cell(4,4) | -0.002041 | -0.001595 | 3/13 | **99.1%** |
| `total` = Cell(3,3) - Cell(4,4) | -0.002060 | -0.002060 | 4/13 | 100% |

The identity `total == eval_effect + support_effect` holds to 0.0 in all 13 sessions.

**The confound hypothesis is falsified.** The evaluation boundary explains 0.9% of the M3-to-M4
gap; support size explains 99.1%. Consequence for the paper: **the sealed M3 and M4 numbers are
directly comparable despite scoring on different evaluation sets.** This validates the existing
protocol and removes a line of reviewer attack rather than opening one.

### 11.3 Why C1's fixed-k curve read lower than the trial-aligned curve

Not a trial-alignment mechanism. Movement events are contiguous by trial in all 13 sessions, so
"first M trials" and "first n_M events" are the same set. The difference is only that a fixed event
cap `k` forces every session to the same count, whereas a trial budget lets event-rich sessions
keep more (M3 spans 13-18 events, median 15). A trial-based budget therefore adapts to per-session
event density. This is a deployment note, not a mechanism claim.

### 11.4 Two new usable results

1. **The method's own fail-closed rule sets the minimum H1 calibration budget at two trials.** M1
   is undefined in 0/13 sessions because it yields only 4-6 events against the estimator's 8-event
   floor. This is a clean, quotable deployment boundary that costs nothing to state.
2. **M2 is viable and is the strongest new sparse point.** Two trials (about 30 s, ~10 events) give
   13/13 defined carriers, 11/13 positive forward transfer, carrier fidelity 0.834, and 49% of the
   M4 estimator effect. The paper can discuss a two-trial human calibration point instead of only
   the four-trial one. This is an estimator-level result; it has not been translated through the
   decoder.

### 11.5 Revised GPU priority

| Rank | Cell | Rationale | Risk |
|---|---|---|---|
| 1 | C3 separately trained LS/RS at M4 | control parity with every other dataset in the paper | low |
| 2 | C4 second date at M4 | multi-date generalization | low |
| 3 | **M2 budget cell** (new) | biggest sparsity statement available on human data: half the trials | medium; 49% of estimator effect, 11/13 |
| 4 | M3 budget cell (was C5) | organizer-facing budget, safe but incremental | low value; 83% of M4, largely redundant |

M2 displaces M3 as the interesting budget experiment. M3 is now known to be only a 17% reduction in
estimator effect, so a matched M3 decoder cell would mostly restate M4.

---

## 12. K-point within-event carrier, 2026-08-12 — NEGATIVE, with a mechanism

Question: every H1 sparse candidate ever tested reads exactly two position samples per event
(start, stop). The "midpoint" in earlier screens is `(start+stop)/2`, algebraically derived and
carrying no new information. So within-event trajectory shape had never been read. Dense H-C's
advantage is exactly the within-event profile. Does reading K samples per event recover it?

`q` was pinned at 4 and the carrier at `[176,5]` for every arm, so a positive result would have
needed no decoder change.

**Predeclared gate, frozen before the run:** mean >= +0.010, median >= +0.008, >=10/13 positive,
positive leave-largest-out. Calibrated to the observed Context Full CPU-to-decoder ratio
(CPU +0.0144 produced decoder +0.0165).

Receipts:
- `sua_exploration/results/h1_kpoint_event_carrier/source_screen.json`, SHA `bcc23a4522b9a3caa5184212e18ff994d4796fce894075cfc7d13a89f7d42be5`
- `sua_exploration/results/h1_curvature_geometry/diagnostic.json`, SHA `0ed71b6f00fbd011bd06118a1afb506ff2e8e5c853915ae0b8fe13dcb473c39b`

### 12.1 Result: no arm passes; best contrast is 5x below the gate

| Family | Contrast | mean | median | positive | gate |
|---|---|---:|---:|---:|---|
| increments | K3 - K2 | -0.001054 | -0.000686 | 1/13 | FAIL |
| increments | K5 - K2 | +0.001888 | +0.002387 | 9/13 | FAIL |
| delta_curvature | K3 - K2 | +0.001474 | +0.001891 | 8/13 | FAIL |
| delta_curvature | K5 - K2 | +0.001077 | +0.001049 | 8/13 | FAIL |

Best available gain is +0.0019 against a +0.010 requirement, and roughly an order of magnitude
below Context Full's +0.0144. Integrity: K=2 reproduces sealed H-SE5 to exactly 0.0, and every
`_flat` control collapses back to K=2 to ~1e-16, confirming the curvature decomposition is exact.

### 12.2 Mechanism 1: variance dilution at fixed q

| Arm | retained variance | loss vs K2 |
|---|---:|---:|
| K2 | 0.7190 | - |
| K3 increments | 0.6898 | -0.0293 |
| K3 delta_curvature | 0.6615 | -0.0575 |
| K5 increments | 0.6448 | -0.0742 |
| K5 delta_curvature | 0.6024 | -0.1167 |

With `q` pinned at 4, the same four components must cover a wider raw space, so they capture less of
it. Curvature competes with displacement for the same four slots. **This empirically confirms the
Section B posedness argument**: you cannot buy accuracy by adding information without paying in
either retained variance or observations-per-parameter.

### 12.3 Mechanism 2: the "curvature" is timing, not geometry

The geometry diagnostic measured 860 events, 0 rejected, 0 degenerate.

| Statistic | median | p90 | max |
|---|---:|---:|---:|
| max deviation from the straight constant-speed line / chord | 0.2681 | 0.3840 | 0.6399 |
| arc length / chord length | **1.0232** | 1.3014 | 1.7374 |
| speed coefficient of variation | **0.5691** | 0.7382 | 0.9024 |

A genuinely curved path with sagitta/chord = 0.268 would have arc/chord = **1.181**. The observed
1.023 is about 14x straighter. A synthetic straight path with uneven speed reproduces the observed
signature exactly (deviation 0.225, arc/chord 1.000, speed CV 0.735).

**H1 movement events are spatially almost straight but temporally very non-uniform.** The deviation
the K-point features measure is nearly all speed profile, not path shape.

By tag, curvature concentrates in `Carry` (0.400, arc/chord 1.313), `Shape` (0.331) and `Grasp`
(0.276); `Reach` (0.142), `Orient` (0.124) and `Orient2` (0.136) are nearly straight. `SnapTo` and
`Release` have zero parsed events. Support and later events are statistically indistinguishable
(0.2672 vs 0.2681), so this is not a support/query artifact.

### 12.4 Why this was always going to fail, and what it implies

The carrier's response is **one scalar per event**: `log1p(rate over the whole event)`. Enriching
the *predictor* with temporal detail cannot help much when the *response* has already been
temporally collapsed. The speed profile is real information (CV 0.57) but there is nowhere to
attach it.

This gives the first mechanistic account of the sparse-to-dense gap: **dense does not win because it
has more label coordinates; it wins because it has temporal resolution on both sides, predictor and
response.**

### 12.5 The follow-up this generates — sub-event splitting (hypothesis, not a recommendation)

The correct way to use within-event time is not richer features on the same events, but splitting
each event into sub-events, each contributing its own `(displacement, rate)` observation pair. This
multiplies observations instead of feature dimensions, which is the one lever C1 identified as
effective (monotonic starvation for observations, 4.0 obs/param, no plateau). It keeps `q=4` and
`[176,5]`, and costs the same acquisition reads as the K-point screen.

Arithmetic: ~20 support events per session becomes ~40 at a 2-way split (8.0 obs/param) or ~80 at a
4-way split (16 obs/param), moving the estimator out of the marginal regime entirely.

**Known risk that could kill it, stated up front:** because arc/chord is 1.02, sub-displacements
within one event are nearly parallel. Splitting therefore adds observations along nearly the same
direction, varying mainly in magnitude. It would improve estimation of tuning *gain* but not tuning
*direction*, and C2 showed directional diversity is what drives carrier fidelity. Shorter windows
also mean fewer spikes and noisier rates. Whether this nets positive depends on whether the
estimator is limited by sample count (C1 says yes) or directional coverage (C2 says yes) — both are
real, so the outcome is genuinely uncertain. That is what makes it worth a cheap CPU screen and not
worth any GPU time in advance.

---

## 9. Summary of the strongest single change

Fold H-SE5 into the paper as the H1 sparse result, unify the Method around one event-level
closed-form estimator, and add the separability argument that explains the supervision-efficiency
gap. Together these cost no GPU time and remove the paper's most visible structural weakness:
that its flagship human result does not demonstrate its core mechanism.
