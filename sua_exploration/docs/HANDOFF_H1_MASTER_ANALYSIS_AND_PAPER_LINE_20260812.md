# Master Handoff: H1 — Final Evidence Boundary and Paper Line

**Date:** 2026-08-12
**Status:** terminal H1 decision record; no additional H1 GPU experiment is required for the
current paper scope.
**Supersedes for navigation:** the H1 portions of the five companion docs listed below. Those remain
authoritative for detail and receipts; this document is the map and the decision record.

| Companion | Holds |
|---|---|
| `HANDOFF_H1_SPARSE_MAINLINE_STRENGTHENING_20260812.md` | CPU screen results, posedness argument |
| `H1_TAGFREE_POSITION_CONTEXT_CARRIER_PROTOCOL_20260812.md` | tag-free screen, frozen + terminal |
| `H1_CONTEXT_FULL_UNSEAL_MULTIDATE_GPU_PROTOCOL_20260812.md` | CTXV2 frozen GPU protocol |
| `HANDOFF_PAPER_EDITS_20260812.md` | line-level paper edits, P0-P4 |
| `HANDOFF_H1_CONTEXT_PROGRAM_STATE_20260812.md` | live process state, Stage A blocker |

---

## 1. The central problem in one paragraph

The paper's sparse-supervision claim is established by SUA/M2 and RT, not by H1. H1 is the human,
seven-degree-of-freedom stress test and supports a different, complementary result: the **dense**
H-C carrier works with a compact consumer across five development dates and improves the
organizer-held endpoint. A four-trial sparse endpoint carrier (H-SE5) was positive on one public
development date but did not replicate on a second independently implemented date. H1 therefore
supports the dense multi-phase carrier and compact-consumer claims, while defining an explicit
boundary for extreme sparse event calibration. These two scopes must never be merged.

---

## 2. Complete H1 evidence ledger

### 2.1 Fold-0 decoder results, 8,965-window strict post-support query

| System | Carrier content | pooled R2 |
|---|---|---:|
| H-C | dense 7-DoF velocity | **0.525511** |
| Context Full | endpoint + midpoint + event tag | 0.516518 |
| cross-recording transfer | Context carrier from the *other* recording | **0.519024** |
| Context tag shuffle, same-checkpoint | | 0.513960 |
| **H-SE5** | endpoint displacement only | **0.500037** |
| Context endpoint-label shuffle, same-checkpoint | | 0.499189 |
| Context row shuffle, same-checkpoint | | 0.499152 |
| H-S | activity-only SPINT, 5.97M identity path | 0.496833 |
| Context zero, same-checkpoint | | 0.484059 |
| Zero5 | independently trained null | 0.471569 |
| Ridge v2r2 | dense per-bin per-session linear readout | 0.258235 |

### 2.2 H-SE5 date 1 — positive development cell, not a standalone claim

| Contrast | Value | Recording uniformity |
|---|---:|---|
| H-SE5 - independently trained Zero5 | **+0.028469** | both positive: `+0.02590`, `+0.03593` |
| H-SE5 - same-checkpoint label shuffle | +0.007092 | both positive: `+0.00712`, `+0.00702` |
| H-SE5 - same-checkpoint row shuffle | +0.012544 | - |
| H-SE5 - H-S | +0.003204 | - |
| H-SE5 - H-C | -0.025474 | below dense |

CPU estimator audit across all 13 public recordings:

| Budget | correct - label shuffle | correct - intercept | retained variance |
|---|---:|---:|---:|
| M3 | +0.0238, **13/13** | +0.0101, **13/13** | 0.718 / 0.716 |
| M4 | +0.0253, **13/13** | +0.0121, **13/13** | 0.718 / 0.716 |

### 2.3 H-SE5 date 2 — strict replication fails

The independent outer-date (`19250108`) pair used 10 source recordings, three target recordings,
109 source support blocks, the fixed seed 42 and fixed terminal epoch 49. Full and independently
trained Zero5 were scored once on the same 13,107 strict post-support windows.

| System / contrast | pooled R2 | per-recording Full - Zero5 |
|---|---:|---|
| H-SE5 Full | `0.495670` | - |
| independently trained Zero5 | `0.518260` | - |
| **Full - Zero5** | **`-0.022590`** | `-0.042554 / +0.001088 / +0.027104` |

The predeclared replication condition (positive pooled delta and positive delta in all three
recordings) fails. An independent structural verifier passed, and an independent batch-size-29
prediction replay reproduced the pooled delta as `-0.0225895`. The paper consequence is simply:
no positive cross-date sparse H1 claim.

Immutable evidence:

- terminal receipt SHA-256 `01f8542bfd74d14e9ad033cf7fe6e353fceb16511ea3605f61bbec0867983d0c`;
- structural verifier SHA-256 `e909ab36c3e9f01821c029400068eb256d0f137a101273eca18a3b06fee2c3b6`;
- batch-size-29 replay SHA-256 `9fed4fd9061f8093e1c663ceedb80dc3a39fdf5d7d3bc7d0e47ef8e93673ce31`.

### 2.4 Context Full decomposition — measured, not inferred

| Step (M4) | vs H-SE5 | positive | increment |
|---|---:|---:|---:|
| supervised source-encoding basis, `delta` only, no tags | +0.005051 | 9/13 | - |
| add midpoint, still no tags | +0.004720 | 10/13 | **-0.000331** |
| add event tag | +0.014438 | 13/13 | **+0.009719** |

**The midpoint adds nothing alone. The event tag supplies 67% of the gain at M4, 53% at M3.**

Context Full's per-recording deltas against H-SE5 are `-0.004610 / +0.077458` — **one recording is
negative**. This is why its sealed status is `STOP_CONTEXT`.

### 2.5 Five-date compact-consumer result (dense carrier)

| Contrast | equal-date mean | positive dates | date-bootstrap 95% |
|---|---:|---:|---|
| H-C - H-S | **+0.056287** | 4/5 | `[+0.005965, +0.097195]` |
| H-C - H-C0 | +0.032557 | 4/5 | `[-0.004809, +0.079771]` touches zero |
| H-C0 - H-S | +0.023730 | 3/5 | `[-0.004455, +0.052217]` |
| CI64 - CI32 | **-0.020130** | 2/5 | width escalation fails |

### 2.6 Organizer-held endpoint

| System | held-out mean ± std | held-in | normalized latency |
|---|---:|---:|---:|
| all-source CarrierID | **0.274939 ± 0.127206** | 0.473125 | **0.113919** |
| paper-LR SPINT | 0.261492 ± 0.148717 | 0.470423 | 0.129075 |
| released-LR SPINT | 0.209917 ± 0.114232 | 0.439027 | 0.129293 |

The organizer-scored **dense** all-source CarrierID system uses the compact 58,140-parameter
identity path and is about **12% faster** than paper-LR SPINT in the organizer harness. The distinct
development-only H-SE5 sparse arm uses **58,172** identity parameters versus SPINT's 5,965,500
(**102.6x** fewer). Do not attach the organizer latency measurement to H-SE5: sparse H-SE5 has not
been submitted to that endpoint.

### 2.7 Supervision accounting

| Estimator | target supervision consumed | pooled R2 |
|---|---:|---:|
| Context Full / H-SE5, fold-0 M4 | 41 events across 2 target recordings, **574** acquired endpoint coordinates | 0.5165 / 0.5000 |
| Ridge v2r2, fold-0 M4 | 6,088 dense rows, **42,616** velocity coordinates | 0.2582 |

Ratio **74.2x** fewer coordinates. Across 13 recordings at M3: 2,786 endpoint scalars vs 216,153
raw dense scalars = **77.6x**, or **15.5x** under the conservative 100 ms-averaged counting. Report
both definitions or neither.

**Print the sealed `42,616`**, not the raw eval-bin count `45,311`. The difference is the ridge's
50-bin causal-history constraint; the sealed figure is bound to the actual ridge run.

### 2.8 Posedness — why sparse estimation is algebraically possible but not sufficient on H1

| Estimator | parameters | observations | obs/param, M4 median |
|---|---:|---:|---:|
| carrier, per channel, shared design | 5 | ~20 events | **4.0, overdetermined** |
| ridge readout, 50 bins x 176 channels | 8,800/output | ~3,100 rows | **0.356, underdetermined in 13/13** |

Ratio **11.4x**. The carrier's ratio is independent of channel count because the encoding fit is
separable across channels; the readout's degrades as `1/N` because its parameters are coupled.
This is a posedness statement, not an accuracy guarantee: date 2 shows that an overdetermined
five-parameter fit can still lack stable task-relevant content.

---

## 3. THE PAPER LINE — what to say, what not to say

### 3.1 Method section

The H-SE5 analysis may be described as a boundary experiment using a per-channel closed-form ridge
from the same estimator family as eq. (7)-(8); only `phi` differs. It must not be presented as the
H1 main method or as a successful extension of the sparse claim.

```
g(r_i(e)) = b_i + w_i^T phi(a_e) + eps,    beta_i = (X^T X + n*lambda*D)^-1 X^T y_i,  D = diag(0, I_q)
```

| Task | event | native annotation | `phi` | q | g |
|---|---|---|---|---:|---|
| Center-out | trial | direction `theta` | `[cos, sin]` | 2 | identity |
| M2 | trial | direction `theta` | `[cos, sin]` | 2 | identity |
| RT | go-cue reach | endpoint pair | unit displacement direction | 2 | identity |
| **H1** | movement epoch | 7-DoF endpoint pair | `P_src * standardize(dq)` | 4 | `log1p` |

`lambda = 0` recovers the pseudoinverse form. This table documents the tested sparse extension; the
two-date result shows that it is not a stable H1 replacement for the dense estimator.

**Do:** keep the H-C population estimator, demoted to a short paragraph as the dense-carrier
deployment variant that produced the organizer result.

### 3.2 Results — the H1 evidence split

**Main text:** report dense H-C as the H1 result: five-date development evidence, a compact
58,140-parameter identity consumer and the organizer-held improvement. This is the evidence that
supports the H1 entry in the main results table.

**Boundary or supplement:** report both H-SE5 dates together. Date 1 is `+0.028469` over an
independently trained Zero5 null (2/2 recordings positive); date 2 is `-0.022590` (2/3 positive).
State that four-trial endpoint-only calibration is not reproducible across dates. Never print the
date-1 result alone as an H1 sparse success, and do not claim that H-SE5 beats SPINT (date-1 margin
over H-S is only `+0.003204`).

The compact H-SE5 implementation remains useful as an engineering feasibility result, but the
failed content replication means it cannot establish that sparse supervision and compression both
retain accuracy on H1.

### 3.3 Supervision efficiency

If supervision accounting is retained, describe only the evaluated fold-0 systems: an H-SE5
carrier consuming 574 endpoint coordinates reaches `0.500` while the evaluated dense ridge readout
using 42,616 velocity coordinates reaches `0.258`. Pair this with the non-replication result and the
posedness explanation from Section 2.8; do not promote the fold-0 comparison into a general sparse
H1 accuracy claim.

**Do keep the caveats:** the ridge is a direct per-session readout while the carrier is
source-pretrained, so this is not architecture-matched; and the two estimators use different
observation units and different targets. Say "the two estimators pose problems with very different
sample-to-parameter ratios on the same calibration block", never "same data, different method".
Scope every ridge statement to the evaluated implementation.

### 3.4 Deployment and cost

**Do add** organizer-measured latency to `tab:consumer` as a third column: `0.113919` vs `0.129075`,
about 12% faster. Parameter and MAC ratios are our measurements of our own module; latency was
measured by the organizer's harness on hidden recordings, which makes it the only third-party
efficiency evidence in the paper.

**Do write** that each target recording is calibrated independently with one `5x5` system shared
across its 176 channels. For one recording with `n` sparse events and `p=q+1=5`, forming and solving
the multi-output ridge system costs
`O(n p^2 + n p N + p^3 + p^2 N)`. The printed fold-0 total of 41 events is the sum of the two
recording-specific calibration blocks, not one pooled `41 x 176` fit.

**Do use** the CI64 negative only to justify retaining the selected 32-wide consumer: widening the
tested compact path failed (`-0.020130`, 2/5 dates). It does not establish the stronger claim that
the consumer is generally "not capacity-limited".

### 3.5 The pooling claim

**Do write** only the dataset-supported fact: H1 provides 176-channel multi-unit spiking activity
in the public DANDI metadata. Do not call these channels threshold crossings, and do not claim a
sorted-to-pooled comparison, unless an independent dataset provenance source establishes that
acquisition detail.

### 3.6 Completed wording correction: no drift absolute

The former statement that a refitted carrier "cannot drift" has been deleted. It was unsupported,
and the only evidence bearing on it points the other way. A forward-only
cross-recording transfer scored `0.519024` against `full` at `0.516518` — swapping carriers between
the two fold-0 recordings slightly **improved** performance, on both recordings. Integrity clean:
`full` reproduced sealed `0.516518` to `3.3e-08`, query SHA identical, model state unchanged, zero
optimizer and backward calls.

**Scope caveat that must travel with that number:** `ses-19250101T111740` and `ses-19250101T112404`
are timestamps six minutes apart on the same day, i.e. consecutive blocks, not different sessions.
It shows the per-block refit earns nothing at six-minute separation; it does **not** test drift
across days.

The transfer diagnostic can remain internal; it is not needed for the final paper claim.

### 3.7 The "well-conditioned" claim in Section 4.1

Currently: usability requires "enough **well-conditioned** and diverse events".

Measured, within-session: the design condition number predicts carrier fidelity (12/13) but **not**
forward transfer (7-9/13, weak). Drop "well-conditioned" or restrict it to carrier stability.

Three proxy statistics have now each failed to predict forward transfer: design condition number,
retained variance, retained energy. Never use one as a selection criterion.

### 3.8 Contribution (vi), which closes both the abstract and the conclusion

Currently: "event diversity, label geometry, coefficient dimension, and temporal dependence jointly
matter---not trial count alone."

Measured position:
- event **count** is a first-order driver on H1, monotonic with **no plateau**; at 8 events the
  carrier retains under 3% of the effect;
- **diversity** at fixed count is real but secondary (maxspread over minspread `+0.0052`, 11/13,
  p=0.023 at k=14; 13/13 on carrier fidelity);
- **conditioning** is not the operative summary statistic.

Revise rather than restate. Do not downplay count.

### 3.9 H-SE5 cross-date terminal outcome

The independently implemented `19250108` H-SE5 cell did not replicate the positive `19250101`
result. Full minus independently trained Zero5 is `-0.022590` pooled; one recording is negative and
two are positive, while endpoint-label shuffle also exceeds Full by `0.001448`. Across both dates,
the load-bearing contrast is positive on only 1/2 dates (4/5 recordings). The predeclared gate
fails, so no positive cross-date sparse H1 claim is permitted.

Keep the two H1 stories separate. The organizer-held `0.274939` result belongs only to dense H-C;
H-SE5 was development-only and was never submitted to the organizer endpoint.

### 3.10 Standing prohibition

No H1 Context Full, event-tag, tag-shuffle, or `0.5165` claim unless the editorial prohibition is
explicitly revisited. No positive H-SE5 efficacy claim is permitted: H-SE5 may be reported only as
mixed development evidence or as a negative boundary, with both dates and the independently
trained Zero5 contrast stated together.

---

## 4. Closed — do not re-run these

| Route | Terminal status | Result |
|---|---|---|
| Tag-free position context (`[delta, midpoint]`, `[start, stop]`, `[delta, start]`, PCA variant) | `STOP_CPU_PCTX_NOT_MATERIAL` | Best `+0.004720` vs a `+0.011550` requirement. No tag-free path to Context-level performance exists. |
| K-point within-event trajectory (K=2/3/5, two feature families, flat controls) | no arm passed | Best `+0.0019` vs a `+0.010` gate. Retained variance falls `0.719 -> 0.602` as raw space widens at fixed `q`. |
| Ten earlier event-carrier design screens (TCE5, nested, meta-learned, semantic V4, C2F5, LRT5, NLE5, PNO5, QC2F5) | all below `+0.02` | Plateau at `+0.012` to `+0.016`. |
| CI64 width escalation | terminal negative | `-0.020130`, 2/5 dates. |
| H1 sparsity headroom below the full ~20-event budget | falsified | Monotonic, no plateau. |
| M1 as a positive carrier dataset | negative boundary | matched content `-0.00652`. |

Mechanistic reason the K-point route failed, worth keeping: H1 events are spatially near-straight
(arc/chord median **1.023**, where a genuinely curved path with the observed deviation would be
1.181) but temporally very non-uniform (speed CV **0.569**). The carrier's response is one scalar
per event, so enriching the predictor with temporal detail has nowhere to attach. **Dense wins not
by having more label coordinates but by having temporal resolution on both sides.**

---

## 5. Experimental closure

### 5.1 Retired

**Context Stage B — separately trained Context-LS and Context-RS, fold-0.** Both runs were stopped
before their fixed terminal epoch because they refine the already stopped Context route but cannot
provide the missing H-SE5 cross-date evidence. They have no terminal checkpoint or score and are
neither positive nor negative scientific results. The retirement receipt is
`SPINT-main/pilot_artifacts/h1_ctxv2_stage_b/STAGE_B_PRIORITY_RETIREMENT_20260812.md`.

### 5.2 Completed: H-SE5 second-date replication

The additive date-parameterized implementation passed exact fold-0 fidelity, source/target
isolation and immutable snapshot-binding audits. Full and independently trained Zero5 completed all
50 epochs. The strict date-2 terminal result is `0.495670 - 0.518260 = -0.022590`, with only 2/3
recording deltas positive. The predeclared replication condition failed and the result was
independently reproduced. This closes, rather than queues, the sparse H1 mainline.

The pre-existing `h1_ctxv2_hse5_full_19250108.yaml` and `h1_ctxv2_zero5_19250108.yaml` remain
invalid placeholders and must never be confused with this completed dated lineage.

### 5.3 Closure decision

- Do not run another H-SE5 seed or date to rescue the sign; the fixed-seed cross-date question is
  answered and post-result expansion would not repair the failed prospective claim.
- Do not resume Context Stage B, sub-event splitting, sparse budget sweeps or cross-date transfer
  for the current paper.
- Preserve H-SE5 as a documented boundary experiment and keep dense H-C as the H1 main result.
- Additional sparse H1 research would require a newly scoped program with a different estimator and
  fresh evaluation dates, not another arm appended to this sealed program.

---

## 6. Standing constraints

- **Never modify a sealed file.** Anything under `SPINT-main/src/`, `SPINT-main/configs/`,
  `SPINT-main/scripts/` predating 2026-08-12 is sealed. Import and subclass.
- `PYTHONNOUSERSITE=1` always; training needs `PYTHONPATH=<root>/SPINT-main:<root>`, both, in order.
- Interpreter `~/miniconda3/envs/spint/bin/python` (Torch 2.5.1 / CUDA 11.8).
- Only the 13 public `sub-HumanPitt-held-in-calib` NWBs. Never `held-out`, `heldout`, `minival`,
  `formal`, `private`, `evalai`, `test_ecephys`.
- Never read `acquisition/OpenLoopKinematicsVelocity` in a sparse-carrier path.
- Predeclared gates are frozen. A failed gate is a failure; do not relax thresholds or add arms
  afterwards.
- Every published number must cite its receipt SHA and its independent verifier.
- Do not `git add .` in this dirty multi-agent worktree.
- Disk at 91%, ~79 GB free; each GPU run is ~126 MB.

## 7. Facts to verify before printing

1. Is the calibration-block event tag task-script metadata known a priori, or only dataset
   annotation? (needed for Section 3.9)
2. If any stronger acquisition statement than the official "multi-unit spiking activity" wording
   is needed, obtain and cite an independent H1 provenance source first.
