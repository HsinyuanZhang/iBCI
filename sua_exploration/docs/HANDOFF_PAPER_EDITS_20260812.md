# Handoff: Paper Edits for `paper_6pp.tex`

**Date:** 2026-08-12
**Target:** `bci_paper_overleaf/paper_6pp.tex`
**Supersedes:** `bci_paper_overleaf/REVIEW_6PP.md` for anything that conflicts; that file's P0-P3 list
remains valid where not contradicted here.
**Companion analysis:** `PAPER_FRAMING_ANALYSIS_20260812.md`

Every number below is measured and bound to a receipt. Nothing here is an estimate.

---

## P0 — Correctness. Do these first.

### P0.1 Delete or qualify the session-drift claim (lines 363-364)

**Current text:**

> "because $\mathbf{c}_i$ is re-fitted every session it cannot drift like a transferred neural
> fingerprint"

This is unsupported, and the only experiment bearing on it points the other way.

**Evidence.** Forward-only cross-recording carrier transfer on the sealed fold-0 Context checkpoint.
Receipt `SPINT-main/pilot_artifacts/h1_cross_session_transfer/H1_CROSS_SESSION_CARRIER_TRANSFER_FOLD0_v1.json`,
SHA `ec0d9802d563d73fc6096bbf3abed33523ba35d2ed038c04970954e13ae68f98`.

| Arm | pooled R2 | 111740 | 112404 |
|---|---:|---:|---:|
| full | 0.516518 | 0.556094 | 0.401429 |
| **cross-recording transfer** | **0.519024** | 0.558373 | 0.404597 |
| row shuffle | 0.499152 | 0.542581 | 0.372902 |
| zero | 0.484059 | 0.524006 | 0.367850 |

`full - transfer = -0.002506`, negative on both recordings. Integrity: `full` reproduced the sealed
`0.516518` to `3.3e-08`; query SHA `665fe535...e4da` identical across arms; model state digest
unchanged; `optimizer_steps = 0`, `backward_calls = 0`.

**Scope, which matters.** `ses-19250101T111740` and `ses-19250101T112404` are timestamps six minutes
apart on the same day. This is a **within-day, between-block** transfer, not a between-session
transfer. It does not test drift across days.

**Action.** Delete the quoted clause. Replace the surrounding "Consequences" paragraph with a
statement of what is actually supported: the descriptor requires no optimizer on the new session,
and pooling linearity predicts robustness to loss of spike sorting. Do not claim session-specificity
until a cross-date transfer result exists.

**Optional stronger move.** Report the transfer result as a finding rather than hiding it: between
same-day blocks the carrier is interchangeable, while remaining `+0.0199` above row shuffle and
`+0.0350` above zero. That localises the value of the carrier in its *content and attachment* rather
than in per-session refitting, which is an honest and defensible position.

### P0.2 Add the H1 estimator equation

Unchanged from `REVIEW_6PP.md` P0.1. The H1 population carrier is still described in prose with no
equation.

### P0.3 Correct the observations-per-parameter figures if the posedness argument is added

The correct per-session values are **4.0** for the carrier and **0.356** for the ridge at M4, ratio
**11.4x**, ridge underdetermined in 13/13 sessions. An earlier draft used `8.2`, which came from
dividing a two-recording event total by the per-channel parameter count. Receipt
`sua_exploration/results/h1_sample_complexity_audit/audit.json`,
SHA `d336aa5ab5646ef7b12777e4a846f5f6b074970fd028257ad559750af35ba6db`.

When citing H1 dense supervision, print the sealed `42,616` velocity coordinates, not the raw
eval-bin count of `45,311`. The difference is the ridge's 50-bin causal-history constraint, and the
sealed figure is bound to the actual ridge run.

---

## P1 — Free favorable material currently unused

### P1.1 Add organizer-measured latency to `tab:consumer`

Held-out submission: normalized latency **0.113919** for the compact CarrierID system against
**0.129075** for paper-LR SPINT, about **12% faster**. The table currently reports only parameter
and identity-MAC ratios, both of which are our measurements of our own module. Latency was measured
by the organizer's harness on hidden recordings, so it is third-party evidence, and the paper's
framing is implanted power-limited hardware. Add it as a third column.

### P1.2 Add the posedness argument

| Estimator | observations per parameter, M4 median |
|---|---:|
| carrier, per channel, shared design | 4.0, overdetermined |
| dense ridge readout, 50 bins x 176 channels | 0.356, **underdetermined in 13/13 sessions** |

The carrier's ratio is independent of channel count because the encoding fit is separable across
channels; the readout's degrades as `1/N` because its parameters are coupled. This is the missing
explanation of why fewer labels can suffice, and it converts the supervision-density section from
label-counting into a statistical argument.

Frame it carefully: the two estimators use different observation units and different targets, so
say "the two estimators pose problems with very different sample-to-parameter ratios on the same
calibration block", never "same data, different method". Scope the claim to the evaluated ridge
implementation.

### P1.3 State that H1 is natively unsorted

176 threshold-crossing channels, so H1 confirms the pooling prediction on human data without needing
a sorted-to-pooled comparison. One sentence. **Verify the threshold-crossing claim before printing.**

### P1.4 Report the sparse-plus-compact conjunction

`falcon_h1_sparse_event_endpoint.yaml` sets `carrier_hidden_dim: 32`: the sparse H1 arm already runs
through the same compact consumer as the dense arm. So sparse event-level supervision and a 102.6x
smaller identity path hold **simultaneously**. Compression is currently reported only for the dense
arm, so the conjunction is invisible.

### P1.5 Use the CI64 negative as support for the efficiency claim

Widening the joint carrier consumer failed (`-0.020130` equal-date mean, 2/5 dates positive). Read
correctly this says the 32-wide consumer is not capacity-limited, which strengthens the
parameter-efficiency interpretation. A negative result that supports the thesis is cheap credibility.

---

## P2 — H1 sparse arm

### P2.1 Add H-SE5, which is currently absent from the paper entirely

`0.500037` pooled, `+0.028469` over an independently trained null, **positive on both recordings**
(`+0.02590`, `+0.03593`), plus a 13/13 CPU constructibility audit at both M3 and M4. It is not on
the review's forbidden list; only Context Full, event tags, tag shuffle and the `0.5165` figure are.

Its absence is the reason the paper currently declares H1 outside the sparse mechanism in
Section 4.1, which contradicts the paper's own thesis.

### P2.2 Unify the Method around one event-level estimator

Replace the split presentation with a single estimator plus a task-basis table. H-SE5 uses a
per-channel closed-form ridge, which is the same estimator family as eq. (7)-(8); only `phi` differs.
This makes H1 the highest-dimensional instance of one mechanism instead of an exception, and it
subsumes the missing-equation problem in P0.2.

### P2.3 Fix the claim about what makes sparse labels usable

Section 4.1 currently says usability requires "enough **well-conditioned** and diverse events".
Measured: within-session, the design condition number predicts carrier fidelity (12/13) but **not**
forward transfer (7-9/13, weak). Diversity at fixed count is real (maxspread over minspread, 11/13
at k=14, p=0.023; 13/13 on fidelity), but event **count** is a first-order driver with no plateau —
at 8 events the carrier retains under 3% of the effect.

Three proxy statistics have now each failed to predict forward transfer: design condition number,
retained variance, and retained energy. Drop "well-conditioned", or restrict it to carrier stability.

### P2.4 Revise contribution (vi)

The abstract and conclusion both close with "event diversity, label geometry, coefficient dimension,
and temporal dependence jointly matter---not trial count alone." The measured position is:
count is a first-order driver on H1; diversity at fixed count is a real secondary effect;
conditioning is not the operative summary statistic. Revise rather than restate.

---

## P3 — If CTXV2 multi-date replicates

### P3.1 Do not write "positive on all recordings"

Fold-0's per-recording deltas against H-SE5 are `-0.004610 / +0.077458`; one is already negative.
Even with a 3/3 second date the honest statement is **2/2 dates pooled-positive, 4/5 recordings
positive**. Date is already the paper's reporting unit for H1, since the existing compact-consumer
result is reported as 4/5 dates with a date-level bootstrap.

### P3.2 Disclose the event-tag dependence

Measured: the tag supplies **67%** of Context's gain at M4 and 53% at M3, and no tag-free
reparameterisation recovers it (best tag-free candidate `+0.004720` against a `+0.011550`
requirement; terminal `STOP_CPU_PCTX_NOT_MATERIAL`). Report the ablation rather than omitting it.

Justify the tag as scripted-task metadata known a priori in a clinical calibration block, **only
after verifying that against the FALCON H1 protocol.** If it cannot be verified, describe it plainly
as native epoch metadata supplied with the dataset and state the supervision accounting for
endpoints and tags separately.

### P3.3 Include the 4x control-understatement finding

The sealed same-checkpoint within-trial tag shuffle costs 16% of Context's decoder gain; the true
tag contribution is 67%. Same-checkpoint interventions on a model trained with correct content
measure something weaker than retraining without it. This is a transferable methodological point and
it is what motivates the separately trained Stage B arms.

### P3.4 Keep the two H1 stories separate

H1 carries a sparse mechanism result (Context, development folds) and a dense deployment system
result (H-C, five dates plus the organizer-hidden endpoint). A reader must never be able to merge
them into "our sparse system scored 0.2749 on the hidden set". It did not; the dense system did.

---

## P4 — Judgement call

Consider adding M1 as a stated negative boundary (matched content contrast `-0.00652`). The paper
currently lists only datasets where the carrier worked, which is itself a visible risk. Reviewers
generally treat a volunteered negative boundary as evidence against cherry-picking, and the
threats-to-validity section can host it in one sentence.

---

## Experiments in flight or queued

| Status | Item | Purpose |
|---|---|---|
| **Running** | Stage B: separately trained Context-LS and Context-RS, fold-0 | Control parity; motivated by the measured 4x bias |
| **Done** | Cross-recording carrier transfer | Falsified the drift claim within-day; see P0.1 |
| **Queued** | Cross-**date** carrier transfer | The meaningful version of P0.1; needs a second held-out date |
| **Queued** | CTXV2 Stage A, date `19250108` | Generalisation, and it unblocks the cross-date transfer test |

Stage A now has two justifications, not one: multi-date generalisation, and supplying the held-out
date needed for a clean cross-date transfer test.

---

## Do-not-revert constraints

- No H1 Context Full, event-tag, tag-shuffle, or `0.5165` claim unless the editorial prohibition is
  explicitly revisited. P3 applies only in that case.
- No matched-label T4-Ridge claim.
- Preserve the within-Ridge A2b-v2 numbers, the H1 five-date result, the organizer-held aggregate,
  and the three-way distinction among algorithmic target-supervision consumption, human annotation
  cost, and compute cost.
