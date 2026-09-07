# CEBRA Exploration — Project Vision

**What this sub-project is for, why it is shaped this way, and what it is allowed to conclude.**

Read this before the track documents. It is the only file that explains *why* the work was
partitioned as it was; everything else answers a question this file poses.

---

**2026-08-14 status correction.** This document preserves the reasoning that
generated the original queue, but its old item 6 is no longer pending.  The
source-only cross-session consistency screen is terminal: the noise-matched T4
source--source half-sample advantage is `0.741286` against a within-session
split-half ceiling of `0.745901`, leaving a frozen headroom fraction of only
`0.006188 < 0.10` (the full-sample source--source advantage is `0.854442`).
Accordingly, the projection-head consistency training arm is killed rather than
implemented.  The immutable receipt is
`sua_exploration/results/cross_session_consistency_screen/cross_session_consistency_screen_receipt.json`
(SHA-256 `521db6857880f87dc533e00e1df07a5f19abe597929b8c40ceb2ed54e420b425`).
F18 keeps continuous behaviour matching viable for other mechanisms; it does
not override this consistency-specific ceiling gate.  The active non-H1 work is
the internal routing/identity queue followed by the repaired Track-B v2
comparator on subject-M SUA/pseudo-MUA and RT.  H1 remains open at lower
priority, but is excluded from Track-B v2.

## 1. The starting position

Our paper argues that session calibration should change **what it computes**, not how it is
optimized. A source-trained permutation-invariant decoder receives, for each new session, a
closed-form per-unit encoding signature fitted on a short labelled prefix and cached as an identity
token. No parameter receives a gradient on the target session.

The claim is deliberately not "we are the most accurate". It is a **conjunction**: sparse labelled
supervision, no target-session backward pass, no channel correspondence, and a small identity path —
a combination nothing else in the literature occupies.

A conjunction claim is only as strong as the comparison set that surrounds it. That is where CEBRA
enters.

## 2. Why CEBRA specifically

Our comparator set was heavy on classical closed-form readouts — ridge, population vector, Kalman —
and on cited numbers from FALCON Table 1. All of those are either much simpler than us or evaluated
on someone else's endpoint. **None of them attacks the same problem with a learned solution.**

CEBRA does. It handles sessions with different neuron counts and produces latents that are
consistent across sessions and subjects. Stripped to its structure, CEBRA and our method are two
answers to one question:

> When the recorded unit set changes, what absorbs the change?

| | CEBRA `MultiSessionSolver` | CEBRA `UnifiedSolver` | Ours |
|---|---|---|---|
| what absorbs a changed unit set | a per-session encoder, **learned** | one shared model over **concatenated** units | an identity token, **closed-form** |
| what aligns sessions | the contrastive objective | the contrastive objective | the token's content |
| weights shared across sessions | **none** (F2b) | one model | the entire decoder |
| **can an unseen session be served?** | only by training a new encoder for it | **no** — input width is hard-wired to the sum of training sessions' unit counts | yes: a closed-form solve plus one forward pass |

*Corrected after red team.* An earlier version claimed "parameter growth per session: linear versus
zero", which is **false** against `UnifiedSolver` (13,155 shared parameters against
`MultiSessionSolver`'s 22,790 on the same two sessions). The separating axis is not parameter growth
but whether an unseen session can be served at all. See `COORDINATOR_VERIFIED_FINDINGS.md`, F2b
addendum.

That table is the whole reason this sub-project exists. It is also the reason the comparison is
delicate: CEBRA is close enough to us that a sloppy reimplementation would be mistaken for a real
result, in either direction.

## 3. The three tracks, and the logic of the split

The work divided into three genuinely independent questions, which is why three agents could run in
parallel without shared state.

**Track A — can we use CEBRA's *data*?** The cheapest possible win would be a new motor cohort on
which to repeat our claim. Judged strictly on whether a dataset can exercise *cross-session
calibration*, since a single-session recording cannot, however well curated.

**Track B — can we use CEBRA's *method*, on our data, as a comparator?** The scientifically
substantive track. It carries a second job: because adapting CEBRA to a new session requires a
target-session backward pass and ours does not, this comparator is also the **cost-of-no-backprop
measurement** the paper has always lacked. The paper asserts a backward pass is unaffordable without
ever saying what it costs.

**Track C — can our *method* borrow CEBRA's ideas?** Pure ideation, run as **two independent agents
on different models** who then reviewed each other. Two models were used deliberately: a single
brainstorm tends to produce a plausible-sounding list nobody can falsify, whereas two independent
lists plus mutual review produce *disagreements*, and disagreements are checkable.

## 4. What the comparison is allowed to conclude

Governed by `sua_exploration/docs/HANDOFF_COMPARATORS_20260812.md`, in particular:

- A number from another paper is citable only if it was computed on the same data, the same split and
  the same metric. CEBRA is **not** in FALCON Table 1, so any CEBRA number must be produced by us.
- §10: *a badly tuned reimplementation is worse than no comparison*, because a reviewer who knows the
  method will see it. This is the governing risk of Track B and it nearly materialized twice — once
  as F8, once as F15.
- Reporting rule 5: never tune a comparator where tuning helps us and leave it fixed where it does
  not.

Two structural commitments follow, and both are now enforced in code:

1. **Every arm must pass a synthetic positive control** — three sessions of differing unit counts
   built from one shared latent — before any number from it is interpreted. An arm that cannot
   recover a latent we planted is void.
2. **Every arm declares its direction of bias.** A handicapped arm may be reported, but never as
   CEBRA's best.

## 5. What actually happened

**Track A: closed, negative, and correct on its load-bearing claim.** No CEBRA dataset can exercise
cross-session calibration on a motor task. Area2_Bump is one monkey and one recording — its `session`
argument selects bump-versus-reach trial type, not a recording day — and this was verified in the
loader source. The rat hippocampus data has four subjects with differing unit counts, but it is CA1
place-cell activity on a linear track: the behavioural variables are position and direction of travel
(`[position, right, left]`), which is a different encoding problem from motor tuning, not a
center-out reach geometry.

*Corrected after audit:* an earlier version of this file said the hippocampus data has "no direction
or velocity basis". That is wrong — the loader does expose a direction label. Track A itself got this
right; the error was introduced when I compressed its audit into this summary. The dataset is
unsuitable because it is the wrong task and species for a motor-decoding claim, **not** because no
behavioural basis exists. Note also that the findings file contains **no independent verification of
any Track A claim** — the track was closed on Track A's own evidence plus a spot-check of the
Area2_Bump loader.

**Track B: the first design would have produced a false conclusion.** As originally built, the arms
scored as low as **−1.23** on a positive control where the joint fit reaches ~+0.99 (that figure is
an extreme draw; an 8-seed mean is ≈ +0.01 and the direction, not the magnitude, is what reproduces),
because CEBRA's
cross-session alignment lives entirely in cross-session positive sampling, which `adapt=True` does
not do (F8). Every dataset would have returned ~0, and we would have written "CEBRA fails on our
data" — an artefact of our own harness. The arms were rebuilt around a joint fit, plus a
frozen-source variant that required patching the vendored source, which is why the source was
vendored rather than pip-installed.

**Track C: the ideas were the smaller half of the value, but a premature verdict nearly buried the
rest.** The mutual review falsified three load-bearing claims and the coordinator audit falsified two
more — none through discussion, all by checking code or receipts.

I then concluded that CEBRA "offers no usable mechanistic value; its ideas either violate the
no-backprop constraint or are incompatible with the closed-form estimator". **A red team partially
overturned that**, and the refutation was free: **16 of the 18 numbered proposals across the two
brainstorms already carry an explicit line reading "Violates no-target-session-backprop? No"**. My
own documents contradicted the verdict I drew from them.

The framing error underneath it was asking only "can we transplant a mechanism into the **carrier**".
The carrier is closed-form and CEBRA is learned, so that graft was always unlikely. **Source training
is unconstrained by the no-backprop rule** — that rule governs only the target session — and exactly
one idea was ever explored there.

Three live mechanistic leads survive:

1. **Cross-session consistency loss at source-training time**, now in screening, and subject to F17:
   it must sit behind a projection head, not on `ŷ` where the linear read-out lives.
2. **S1 — measured, and it resolved into something better than the idea.** See F19. The intercept
   dominance that motivated S1 turns out to be a property of **H1's normalizer**, not of the carrier:
   center-out, which z-scores per coordinate, shows 0.884× standardized against H1's 32–61× under a
   global scalar. The session rate gauge is real and confirmed (`mean(b̂)/rate = 1.0025 ± 0.0154`,
   ρ = 0.928), but its magnitude across our sessions is only 2.0×. So the payoff is not the proposed
   gauge-randomizing augmentation; it is the **normalizer swap on H1**, which is simpler and directly
   targeted.
3. **Pseudo-session unit-set mixing**, the strongest idea neither brainstorm had. CEBRA's
   `UnifiedDataset` concatenates behaviourally matched units across sessions into one "pseudo-animal";
   our decoder is permutation-invariant and consumes that natively, and `key_padding_mask` is already
   plumbed through `CrossAttentionLayer`. Source-training only, deployment path unchanged.

**The red team's redirect was measured on real data and did not survive — see F18.** Its synthetic
estimate said the nearest-neighbour behaviour residual scales as `−1/d`, about 2.6% at 2-D but 37% at
H1's 7 DoF, implying behaviour-matched pairing works on center-out and RT but not H1.

On our own cohorts the picture is different and better. **Cross-session matching costs only
1.01–1.13× the within-session floor in every cohort**, and **H1's median ratio, 1.009, is the
tightest of the four**. Cross-session drift is simply not the obstacle. The `−1/d` law does hold, but
its exponent is set by the **participation ratio** rather than the nominal dimension (a nominally
100-dimensional behaviour window measures `d_eff` 6.4–15.2), and it governs **single-session sampling
density**, not cross-session distance. H1 remains hard — but because 39–79 events per session cannot
cover a 7-DoF space, which is a sampling limit with different remedies, not a drift exclusion.

A second assumption also fell, and it was mine: I briefed that center-out's ~8 discrete directions
would make matching trivially easy and that discrete keying is what the mechanisms would use.
**Discrete matching is 2.8× worse than continuous**, because dividing the reference pool by 8 costs
more sampling density than the shared label buys.

Honest status: none of the three is established. In the red team's own 8-seed synthetic, mixing
reduced the gap from 0.1069 to 0.0727 against a seed sd of 0.044–0.071, cost held-in R², and was
matched by plain gauge augmentation at no such cost. It also recorded that its headline mechanism was
false — session identity stayed 99.7% linearly decodable in every arm — and left the refutation
attached rather than deleting the claim.

## 6. What this sub-project produced that outlives it

Ranked by durability, and note that the top item is about **our** method, not CEBRA:

1. **Three competing, ranked explanations of the H1 sparse failure** (F11, F13, F14), the strongest
   being that the failure is a **query-horizon** effect: the failing recording holds 64% of the
   evaluation samples and is 3.3× longer than its neighbours, while all three share a fixed
   four-trial calibration prefix. If that holds, the remedy — re-solving mid-session — is nearly free
   for a closed-form estimator and impossible for a backprop-based one, which converts the paper's
   most awkward negative into a demonstration of its central advantage.
2. **A sharper statement of our efficiency claim.** F2b shows the real contrast is not "learned
   versus closed-form input layer" but "a full encoder per session, aligned by a loss" versus "one
   shared decoder plus a small cached descriptor". **Do not phrase this as "zero parameter growth per
   session"** — that is refutable by `UnifiedSolver`, whose parameters do not grow with session count
   either. The defensible axis is that neither CEBRA solver can serve an **unseen** session: one needs
   a new encoder trained for it, the other cannot accept it at any input width. See the F2b addendum.
3. **A reusable positive-control discipline.** The same latent-recovery test would have caught the
   earlier FA-alignment failure, which died of the identical cause.
4. **A number in the paper that needs re-checking**, unrelated to CEBRA: the H1 sparse date-2 delta
   of −0.0226 is sample-weighted; equal-weighted across recordings it is **−0.0048**.

## 7. What this sub-project has *not* produced

**No citable number.** Every arm is unrun on real data. Part A sample counts are unmeasured, the
integrity gate is wired but never executed, and no scoring arm has touched subject-M, RT, H1 or M2.

This is deliberate, but an earlier version of this sentence claimed the gating question is not "can
we build it — that is answered". **That was wrong**, and the independent audit found four further
defects that would each independently have voided any authorised number:

- the primary decoder cannot accept H1's 7-D labels at all (the sealed ridge hard-requires 2-D)
- **no code path evaluates a target *query* block** — `target_query_neural` is accepted and dead, so
  scores come from the same rows that entered the fit
- `INTEGRITY_ATOL = 1e-5` is unsatisfiable against sealed references quoted to four decimals
- the primary arm **fails its own gate at the declared scoring configuration**: the gate hard-codes
  250 iterations and passes at 0.7502, but at `SOURCE_MAX_ITERATIONS = 10000` it scores 0.6086 and
  raises "numbers from this arm are void". The gate's negative control is also seed-dependent — 2 of
  8 seeds score +0.52/+0.56 against a required `< 0.20`, which would fail the gate outright.

So "can we build it" is **not** answered. The correct statement is that the arms are structurally
sound after the F8 rebuild but the harness is not yet capable of producing a valid number. Honest
accounting of the remaining value:

- **Still worth it:** CEBRA is the only comparator covering the learned multi-session family, and
  nothing else can occupy that cell of the comparison table.
- **Worth less than it first appeared:** the cost-of-no-backprop job should not rest on CEBRA. A
  readout-only probe on *our own* architecture is a cleaner measurement, free of another
  implementation's confounds. And post-F8 we already know the honest CEBRA arm requires retraining
  the whole multi-session model per new session; that axis is close to settled in advance, leaving
  only the accuracy number genuinely open.
- **The competing use of the same effort:** items 1–6 of `HANDOFF_COMPARATORS_20260812.md` §12 cost
  **zero compute** — verifying our FALCON split, printing the Table 1 baselines, rewriting Threats to
  Validity, repositioning the contribution on the seven axes — and remain the highest-return work
  left on a six-page submission.

## 8. If work resumes, in order

*Reordered twice: after the audit demoted the query-horizon test from #1 (F14), and after the red
team reinstated a mechanistic line the earlier verdict had closed.*

1. ~~Measure T4 coordinate scales on center-out and subject-M.~~ **DONE — F19, and it produced the
   one actionable fix.** Center-out shows 9.53× raw / 0.884× standardized intercept imbalance against
   H1's 32–61×. The cause is the **normalizer family**: H1 divides by a single global scalar, which
   cannot change coordinates' scale relative to one another, while center-out applies a per-coordinate
   z-score fitted on source sessions only. **The candidate fix is to give H1 source-fitted
   per-coordinate standardization** — a like-for-like change that leaks nothing from the target
   session, with center-out as the existence proof. Requires retraining the H1 consumer, so GPU.
2. **The effective-contribution probe** on all four H-SE5 checkpoints, computed end-to-end in one
   pass, discriminating F11 from F13. This also replaces the withdrawn "7–12×" interval, which the
   audit showed ranges to 112.6× on the weakest coordinate.
3. ~~The nearest-neighbour behaviour residual between real center-out sessions.~~ **DONE — see F18.**
   The gate **passes**: cross-session matching costs 1.01–1.13× the within-session floor in all four
   cohorts, so the behaviour-matched family is viable and is *not* limited by drift. Use **continuous**
   matching, not discrete direction keys, which are 2.8× worse. Both center-out cohorts support it on
   every representation; RT supports it on the window representation; H1 is undecidable on sampling
   grounds rather than excluded.
4. ~~Recompute C2's abstention statistic with the V2 estimator.~~ **DONE.** The statistic was V2 all
   along (0.8077, reproducing the disputed 0.808; the actual V1 path gives 0.7527), so the reviewer's
   objection was a citation error, not a computation error. But the **abstention gate is dead**: the
   absolute split-half cosine is ≈ 0.06 with 4 of 13 sessions *negative*, the predicted quantity never
   goes negative while the decoder delta it exists to explain does, and ρ collapses to 0.52 under
   random splits and 0.47 against the M3 outcome — where the statistic is structurally uncomputable.
   **The paper's conclusion sentence should not be replaced.** See F12.
5. **The aggregation check.** Both H1 dates are sensitive to sample- versus equal-weighting, and on
   both the equal-weighted number is more favourable to us. Decide and state the convention before
   review, in both directions.
6. ~~The cross-session consistency arm.~~ **TERMINAL KILL.** F18 cleared the matching-feasibility
   prerequisite, but the subsequent source-only screen found that the existing T4 representation
   already sits at the split-half consistency ceiling: frozen headroom `0.006188 < 0.10`.  Under
   F17, a future implementation would have had to attach the loss behind a disposable projection
   head, but the screen says there is no material inconsistency left for that loss to repair.  Do
   not build or tune a trainer from this route without an explicitly superseding protocol.
7. **Only then** Track B, and only after fixing all six harness defects in §7 — the four listed there
   plus the F15 decoder bias and the fixed `NORMALIZED_LAMBDA_FIXED = 1.0`, which the audit showed
   costs CEBRA about 0.19 R² on our own positive control.

The within-date length ordering that survived the F14 refutation is a legitimate second-order
follow-up, but it is worth about a fifth of the date-level effect and does not merit priority.

---

## Map of the documents

| File | What it holds |
|---|---|
| `PROJECT_VISION.md` | this file — why the project exists and what it may conclude |
| `COORDINATOR_VERIFIED_FINDINGS.md` | **authoritative.** Every fact established by running code. Overrides the track documents |
| `BACKGROUND_BRIEF.md` | the shared brief given to every agent: paper, datasets, scope discipline, environment |
| `CODE_INTERFACE.md` | the comparator's public API and how to run it |
| `TRACK_A_DATASET_AUDIT.md` | CEBRA's datasets, per-dataset verdicts |
| `TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md` | the frozen comparator protocol, arms, gates, bias declarations |
| `TRACK_C_BRAINSTORM_GROK.md`, `TRACK_C_BRAINSTORM_OPUS.md` | the two independent brainstorms |
| `TRACK_C_REVIEW_GROK_REVIEWS_OPUS.md`, `TRACK_C_REVIEW_OPUS_REVIEWS_GROK.md` | the mutual reviews |

**Reading order for someone new:** this file, then the findings index, then whichever track matters.
The brainstorm documents contain claims that were later falsified — always check them against the
findings before acting on anything they say.
