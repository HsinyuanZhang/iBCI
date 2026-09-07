# Paper Framing Analysis: Multi-Date Context, Unused Assets, and Cheap Controls

**Date:** 2026-08-12
**Inputs:** `paper_6pp.tex`, the CTXV2 protocol, the four CPU screens run today, `CURRENT_RESULTS.md`
**Purpose:** (1) how to phrase a multi-date Context result if CTXV2 passes; (2) a review of
favorable material the paper is not using and controls worth adding.

---

## Part 1 — If CTXV2 replicates, how the paper should say it

### 1.1 The trap to avoid: "positive on all recordings"

Fold-0's per-recording deltas against H-SE5 are `-0.004610 / +0.077458`. **One recording is
negative.** Even if date `19250108` comes back 3/3 positive, the combined picture is:

- **2 of 2 dates** pooled-positive
- **4 of 5 recordings** positive

The paper must not write "positive on all recordings." It would be false and a reviewer checking
the appendix would catch it.

### 1.2 Use the date as the unit of analysis — and note this is already the paper's convention

The existing H1 compact-consumer result in the paper is reported as **4/5 dates positive** with a
date-level bootstrap interval. Date is already the reporting unit for H1, because the LODO folds
are date-based. A Context result reported as "2/2 dates pooled-positive, 4/5 recordings positive"
is therefore *consistent with how the paper already handles H1*, not a special pleading.

Recommended sentence shape:

> Across two held-out source dates the sparse event-context carrier exceeds the endpoint-only
> carrier on both dates pooled, and on 4 of the 5 individual recordings. Content and attachment
> controls are separately trained rather than same-checkpoint.

### 1.3 State the claim and the non-claim in the same breath

**The claim.** Sparse event-level supervision recovers most of the endpoint-to-dense gap
(64.7% at fold-0) with roughly 74x fewer target-supervision coordinates and no target-session
backpropagation.

**The non-claims, which must be adjacent:**
- not superior to the dense carrier (`0.5165` against `0.5255`);
- not an organizer-hidden result — the hidden endpoint remains the dense H-C system;
- not architecture-matched against the ridge reference.

### 1.4 Handle the event tag head-on, because we now know it carries 67%

The tag-free screen (`STOP_CPU_PCTX_NOT_MATERIAL`) measured that the event tag supplies **67%** of
Context's gain at M4 and 53% at M3, and that no tag-free reparameterisation recovers it. Two things
follow.

**Do not bury it.** Report the ablation. "Without the event tag the carrier falls back to
approximately the endpoint-only baseline" precisely localises where the information lives, which is
stronger science than an unexplained aggregate. A reviewer who discovers this himself will treat it
as a concealed dependency; a reviewer who reads it in the ablation table treats it as rigour.

**Justify the tag as a legitimate deployment input.** The argument is that a clinical calibration
block is scripted — the experimenter knows which phase (reach, orient, grasp, carry) is being
instructed — so the phase label is task metadata available by construction, not a new annotation of
neural data. **This argument must be verified against the FALCON H1 protocol before it is printed.**
If it cannot be verified, the tag must be described plainly as native epoch metadata supplied with
the dataset, and the supervision-accounting claim must be stated for endpoints and tags separately.

### 1.5 Carry the 4x control-understatement finding into the paper

The sealed same-checkpoint within-trial tag shuffle costs only 16% of Context's decoder gain, while
the true tag contribution is 67%. **That control understates dependence by about 4x.**

This is worth a sentence of its own because it is a transferable methodological point: a
same-checkpoint intervention on a model trained with correct content measures something weaker than
retraining without that content. It also supplies the motivation for Stage B's separately trained
arms, turning "we ran more controls" into "we ran more controls because we measured the cheap ones
to be biased."

### 1.6 Where it goes structurally

H1 then carries two separable things, and the paper must keep them separate:
- a **sparse mechanism** result (Context, two dates, development folds);
- a **dense deployment system** result (H-C, five dates plus the organizer-hidden endpoint).

Do not let a reader merge them into "our sparse system scored 0.2749 on the hidden set." It did not;
the dense system did.

---

## Part 2 — Favorable material the paper is not currently using

Ranked by value per unit of effort. Items 1-3 cost nothing but writing.

### 2.1 Organizer-measured latency — free, third-party, currently absent

The held-out submission recorded normalized latency **0.113919** for the compact CarrierID system
against **0.129075** for paper-LR SPINT, roughly **12% faster**. `tab:consumer` currently reports
only parameter and identity-MAC ratios, both of which are *our* measurements of *our* module.

Latency is measured by the organizer's harness on hidden recordings. For a paper whose framing is
implanted, power-limited hardware, a third-party-measured speed number sitting unused is the single
cheapest credibility gain available. Add a latency column to `tab:consumer`.

### 2.2 The posedness argument — explains the whole supervision-efficiency result

Measured today and verified against the sealed receipts:

| Estimator | observations per parameter, M4 median |
|---|---:|
| carrier, per channel, shared design | **4.0 (overdetermined)** |
| dense ridge readout, 50 bins x 176 channels | **0.356 (underdetermined in 13/13 sessions)** |

Ratio 11.4x. The carrier's ratio is independent of channel count; the readout's degrades as `1/N`.
This converts the supervision-density result from label-counting into a statistical explanation.
The paper currently counts labels and never says why fewer can be enough.

### 2.3 H-SE5 is absent from the paper entirely

`0.500037` pooled, `+0.028469` over an independently trained null, positive on both recordings,
with a 13/13 CPU constructibility audit at both budgets. It is not on the review's forbidden list —
only Context Full, event tags, tag shuffle and the `0.5165` figure are. Its absence is why the paper
currently declares H1 outside the sparse mechanism.

### 2.4 Sparse carrier and 102.6x compression hold simultaneously

`falcon_h1_sparse_event_endpoint.yaml` sets `carrier_hidden_dim: 32` — the sparse arm runs through
the same compact consumer as the dense arm. So the paper can say sparse supervision **and** a
102.6x smaller identity path at once. Currently compression is reported only for the dense arm, so
the conjunction is invisible.

### 2.5 The CI64 negative is a positive for the efficiency claim

Widening the joint carrier consumer *failed* (`-0.020130` equal-date mean, 2/5 dates). Read
correctly this says the 32-wide consumer is **not capacity-limited**, which strengthens the
parameter-efficiency interpretation rather than weakening it. A negative result that supports the
thesis is cheap to include and reads as honesty.

### 2.6 H1 is natively unsorted — free confirmation of the pooling prediction

The paper's pooling-linearity argument predicts survival without spike sorting. H1 is 176
threshold-crossing channels. That is the prediction confirmed on human data with no sorted-to-pooled
comparison needed. One sentence. **Verify the threshold-crossing claim before printing.**

### 2.7 M1 as a stated negative boundary

M1 is currently unmentioned; the paper lists only datasets where the carrier worked. The matched
content contrast there is `-0.00652`. Conference reviewers generally treat a volunteered negative
boundary as evidence against cherry-picking, and the paper already has a threats-to-validity
section to host it. This is a judgement call — it is an unfavorable number in exchange for
credibility — but the current all-positive dataset list is itself a visible risk.

---

## Part 3 — Controls worth running, ranked

### 3.1 Cross-session carrier transfer — HIGHEST VALUE, patches a printed unsupported claim

The paper asserts, with no experiment behind it:

> "because $\mathbf{c}_i$ is re-fitted every session it cannot drift like a transferred neural
> fingerprint" (lines 363-364)

Nothing in the evidence ledger tests this. Every existing control (zero, row shuffle, label shuffle)
corrupts the carrier *within* a session. None attaches a **real, correctly fitted carrier from a
different session**.

**Design.** Forward-only on a sealed checkpoint: score session B using session A's fitted carrier,
with everything else identical. Expected outcome is a clear drop, which would directly support the
sentence already in the paper.

**Why it is the best control available:** it is forward-only, needs no training, uses sealed
checkpoints, and it is the only proposed experiment that converts an existing *assertion* into
*evidence*. It is also a control a reviewer is likely to ask for, because "is the carrier really
session-specific, or just a good average?" is the obvious question about the whole method.

**Risk:** if the transferred carrier does *not* hurt, the method is partly a fixed prior rather than
a session-specific descriptor, and lines 363-364 must be deleted. That is precisely why it should be
run before submission rather than discovered in review.

### 3.2 Aged-carrier control — the temporal version of the same question

Apply a carrier fitted on an earlier date to a later date, sweeping the gap. Tests the drift claim
directly and produces a decay curve. Also forward-only. Weaker than 3.1 only because it needs
cross-date checkpoint compatibility.

### 3.3 Decoder-level budget curve on H1

The CPU version is done (`h1_event_budget_sweep`). The decoder version is forward-only on the sealed
H-SE5 checkpoint, using the same query and the same sealed references. Gives H1 a dose-response
curve to match the center-out figure. Note the CPU result predicts monotonic degradation with no
plateau, so this is honest reporting rather than a headline.

### 3.4 Separately trained corruption arms on H1 — Stage B

Brings H1 to control parity with center-out (TS4/LS4) and RT (XLSv2). Motivated by the measured 4x
understatement of same-checkpoint controls. Two GPU runs, infrastructure written but unreviewed and
unlaunched.

---

## Part 4 — Recommended order

| # | Action | Cost | Why first |
|---|---|---|---|
| 1 | Add latency column; add posedness argument; add H-SE5; add the compact+sparse conjunction | writing only | Largest gain per effort; no new results needed |
| 2 | Cross-session carrier transfer control | forward-only | Converts a printed assertion into evidence, and pre-empts the obvious reviewer question |
| 3 | Stage B separately trained LS/RS | 2 GPU runs | Control parity, already motivated by a measured bias |
| 4 | CTXV2 Stage A multi-date | 3 GPU runs plus a pipeline fork | Highest scientific value but highest cost and risk |

Items 1 and 2 are independent of whether CTXV2 ever runs, and should not be blocked behind it.
