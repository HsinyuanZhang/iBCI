# Causal Calibration Memory as the Second Method Contribution

Date: 2026-08-30

Status: design and execution guidance. This document does not authorize a new
training launch by itself.

Primary surface: DANDI 000688 sub-C/sub-M center-out, with honest completed-trial
boundaries, no target gradients, and M4/M10/M30 support budgets.

## 1. Executive decision

The paper should have two distinct method contributions:

1. **Functional carrier (T4):** use a small labeled support set to estimate what
   each neural unit means.
2. **Causal calibration memory (CDM):** use completed, unlabeled target trials to
   update how the current population is represented, without target optimizer,
   backward, or parameter updates.

The simplest description is:

> T4 estimates **what each unit means**. Causal calibration memory tracks
> **how the current session is expressed**.

The validated core of the second contribution is the **activity-memory state**.
The current pseudo-label carrier-update state is not a positive contribution on
DANDI and must not be presented as one. Posterior uncertainty remains useful as
a possible safety controller, but it has not yet produced an improvement beyond
activity-only CDM.

The next primary experiment is therefore:

> combine C1 prefix-robust weights with activity-only CDM, using a clean 2 x 2
> factorial comparison against T0/C1 static and T0 activity-only CDM.

Only after that experiment may a conservative posterior-controlled carrier
residual be tested.

## 2. Evidence that fixes the design

### 2.1 GitHub E02/E03 were weak M50 motivation, not deployment proof

The teammate's E01--E10 screen used one seed, the same subject, 50 calibration
trials, and eight held-out-selected development sessions.

| Cell | Mean R2 | Delta | Breadth | Interpretation |
|---|---:|---:|---:|---|
| E01 ordinary T4 | 0.6137 | -- | -- | fixed-M50 reference |
| E02 posterior mean | 0.6287 | +0.0150 vs E01 | 5/8 | weak positive motivation |
| E03 posterior plus angular reliability | 0.6367 | +0.0080 vs E02 | 5/8 | mechanism unconfirmed |

E02's descriptive 95% interval versus E01 was `[-0.0104, +0.0405]`. E03 had no
constant-width control, no reliability-row shuffle, and no carrier-error
correlation result. These cells justified a stricter experiment; they did not
establish a general posterior or uncertainty claim.

### 2.2 The budget-matched C2/C3 test rejected direct posterior consumption

The local experiment used a stronger and different question: M4/M10/M30,
within-6 and sub-C-to-sub-M external-15, a matched C1 comparator, fixed support
selection, and explicit C3 constant and q-shuffle controls.

For C2 posterior-input training versus C1:

| Surface | M4 delta | M10 delta | M30 delta |
|---|---:|---:|---:|
| external | -0.232744 | -0.222688 | -0.101237 |
| within | -0.079634 | -0.077373 | -0.054847 |

This is not a borderline null. Direct budget-matched posterior carrier
consumption damaged every surface and budget.

C3-Real recovered `+0.040755` over the collapsed C2 external-M4 row, but its
absolute external-M4 R2 was still `-0.045827`, far below C1 `0.146162`.
Moreover, the same-checkpoint q-shuffle control was at least as good as the
correct q binding:

| External budget | C3-Real minus q-shuffle |
|---|---:|
| M4 | -0.006426 |
| M10 | -0.004241 |
| M30 | -0.015858 |

Therefore:

- close **C2-V1**, defined as replacing the ordinary carrier with the posterior
  mean throughout budget-matched decoder training;
- close **C3-V1**, defined as concatenating posterior reliability as a free
  per-unit consumer feature;
- do not run seeds 43/44 for either exact system;
- do not claim that the current consumer learned the semantic q-to-unit binding.

### 2.3 Activity-only CDM is the strong positive result

The frozen activity-only CDM keeps the support carrier fixed and updates only
the B3S activity state from completed trials.

| External cell | Sealed static | Activity-only CDM | Delta | Breadth |
|---|---:|---:|---:|---:|
| M4 | 0.119684 | 0.226495 | +0.106811 | 14/15 |
| M10 | 0.295456 | 0.388809 | +0.093352 | 15/15 |

Within improvements were `+0.156670` at M4 and `+0.073769` at M10, both 6/6.
This is the clearest new short-calibration result in the current program.

### 2.4 Carrier updates reduce the activity-memory gain

Full CDM V8, which also permits gated pseudo-label carrier updates, obtained:

| External cell | Full CDM V8 | Delta vs sealed | Activity-only minus full CDM |
|---|---:|---:|---:|
| M4 | 0.191126 | +0.071442 | +0.035369 |
| M10 | 0.340083 | +0.044627 | +0.048726 |
| M30 | -- | -0.059092 | unsafe carrier-update row |

The full system remained better than sealed at M4/M10, but it was worse than
activity-only CDM and unsafe at M30. The carrier gate therefore has negative
incremental value on this surface.

### 2.5 Precision gating is a safety repair, not yet an added performance gain

Precision-CDM V2 obtained:

| External cell | Precision-CDM | Delta vs sealed | Delta vs activity-only |
|---|---:|---:|---:|
| M4 | 0.222630 | +0.102946 (14/15) | -0.003865 |
| M10 | 0.351002 | +0.055546 (12/15) | -0.037807 |

At M4, the precision gate successfully removed almost all harm from carrier
updates. At M10, it still lost about 0.038 R2 relative to activity-only. This is
a useful robustness diagnosis, but it is not evidence that posterior gating
improves the validated CDM core.

## 3. What survives from E02/E03

There is still a narrow, testable improvement space, but the posterior must
change role.

### 3.1 Posterior mean must not replace the carrier unconditionally

Let `z_ridge` be the accepted support-only fixed-ridge T4 carrier and `z_post`
the E02-style posterior carrier. A safe successor must be anchored to the
working carrier:

```text
delta_post = z_post - z_ridge
z_eff      = z_ridge + alpha_M * w(q) * delta_post
```

Requirements:

- `alpha_M = 0` exactly reconstructs the accepted carrier bytes;
- `alpha_M` is selected on source grouped-OOF data only;
- initial candidate set is `{0, 0.125, 0.25, 0.5}`; full replacement is not the
  default candidate;
- interpolation occurs in raw carrier coordinates before one frozen ordinary
  T4 normalizer;
- the decoder does not receive q as an additional free feature;
- no target label selects alpha, q mapping, checkpoint, or carrier state.

Call this successor **C2-S**: support-anchored posterior residual, not C2-V1.

### 3.2 Reliability must control a defined operation

E03's q should control only the magnitude or acceptance of the posterior
residual. It should not be concatenated and left for the consumer to interpret.

The first controller must be low capacity and source-frozen. For example,
`w(q)` may be a monotone clipped transform whose two calibration points are fit
from source-only q quantiles. A learned controller is allowed only after this
zero/low-parameter form is positive.

Mandatory controls on the same prediction records:

- `alpha=0` exact baseline;
- constant q with identical width and arithmetic;
- deterministic q-to-unit row shuffle;
- posterior residual with `w=1` to separate shrinkage from reliability;
- ordinary fixed-ridge carrier.

The q claim requires correct q binding to beat both constant and q-shuffle.

### 3.3 Posterior covariance is more credible as a state constraint

If C2-S is positive, posterior covariance may constrain future carrier changes.
The state must always remain anchored to the original support carrier:

```text
z_t = z_support + bounded_delta_t
```

It must not repeatedly update from the previous active carrier, because that
turns small pseudo-label bias into cumulative drift. M30 carrier updates remain
disabled. At M4/M10, a carrier proposal must satisfy both:

1. a support-posterior credible-region constraint; and
2. a source-calibrated prediction-consistency or group-agreement condition.

The existing credible-region test alone is insufficient: it accepted 65.4% of
M10 proposals while the resulting system still underperformed activity-only
CDM.

## 4. The second innovation: causal calibration memory

### 4.1 Paper-level definition

The second contribution should be named **Causal Calibration Memory**. CDM is
the implementation family.

It has two conceptually separate state variables:

```text
activity state A_t:
    bounded completed-trial neural activity used by the identity path

carrier state Z_t:
    support-only functional carrier plus an optional bounded posterior residual
```

The validated method is `A_t` with a frozen `Z_0`. The optional `Z_t` update is
a successor and must earn its own incremental result.

This framing is honest and useful:

- T4 provides the **functional prior**;
- Causal Calibration Memory provides **target-session state adaptation**;
- no target gradient or target parameter update is used;
- trial causality is explicit: only completed earlier trials affect the next
  trial;
- the method applies only when a legal trial/update boundary exists.

### 4.2 C1 is the training-side companion

C1 prefix-cycle training improves the weight function that consumes activity
state. It made source identity nearly prefix-invariant and improved static
external M4/M10 by `+0.0265/+0.0350`, while external M30 declined `-0.0220`.

C1 and activity-only CDM may be complementary or redundant:

- C1 changes the **weights** so short and long activity prefixes are interpreted
  consistently;
- activity-only CDM changes the **inference state** by appending completed
  trials.

This interaction has not yet been measured. It is the next experiment.

## 5. Experiment 1: C1 x activity-memory factorial

Use identical targets, support rows, query rows, trial order, metric, and
activity-memory law.

| Cell | Decoder weights | Activity state | Carrier state | Status |
|---|---|---|---|---|
| A0 | T0 | static first M | frozen ridge T4 | existing |
| A1 | C1 | static first M | frozen ridge T4 | existing |
| A2 | T0 | activity-only CDM | frozen ridge T4 | existing |
| A3 | C1 | activity-only CDM | frozen ridge T4 | new primary cell |

Report within-6 and external-15 at M4/M10/M30. The primary comparisons are:

```text
C1 static effect       = A1 - A0
activity-memory effect = A2 - A0
combined improvement   = A3 - A0
gain over best single  = A3 - max(A1, A2)
interaction            = (A3 - A1) - (A2 - A0)
```

The second-innovation stacking claim requires, on external M4 or M10:

```text
A3 - max(A1, A2) >= +0.01 R2
positive sessions >= 10/15
the other low budget >= 0
within every budget >= best-single - 0.02
```

M30 must be reported, not averaged with low budgets. Because C1 already has an
external-M30 cost, report two distinct systems:

1. **single-checkpoint scientific cell:** A3 at all budgets;
2. **budget-routed deployment policy:** A3 at M4/M10 and A0 at M30.

The budget-routed policy uses only the known support budget, never target
labels. However, because this rule was specified after seed-42 results, its
current-seed value is descriptive. A new seed or untouched evaluation surface
is required for a confirmatory claim.

Possible outcomes:

- `A3 > A2`: C1 and causal memory genuinely stack; retain the unified system.
- `A3 ~= A2`: they are redundant; use activity-only CDM as the simpler system.
- `A3 < A2`: C1 suppresses useful activity-state sensitivity; keep the two
  methods separate and do not force a combined story.

## 6. Experiment 2: CDM plus C2-S posterior residual

Run this only after Experiment 1. Use the best fixed activity system from the
factorial and never retrain the decoder for the first screen.

| Cell | Activity system | Carrier |
|---|---|---|
| P0 | best static/activity system | frozen fixed-ridge T4 |
| P1 | same | C2-S anchored posterior residual |
| P2 | same | C2-S with constant q |
| P3 | same | C2-S with unit-row-shuffled q |
| P4 | same | full posterior residual (`w=1`) diagnostic |

Select `alpha_M` and the q transform on strict source grouped-OOF records.
Freeze them before within/external scoring.

Promotion requires:

```text
P1 - P0 >= +0.01 external R2 at M4 or M10
positive sessions >= 10/15
P1 > P2 and P1 > P3 on the promoted budget
the other low budget >= 0
M30 uses P0 exactly; no posterior residual and no carrier update
```

If P1 fails, close posterior consumption and retain activity-only Causal
Calibration Memory as the second contribution. Do not train another free q
consumer.

## 7. Experiment 3: bounded carrier state, conditional only

Only a positive P1 authorizes online carrier refinement. The successor must:

- accumulate trial-level sufficient statistics in blocks, not repeatedly mutate
  the active carrier one trial at a time;
- shrink every block refit toward the frozen support initializer;
- use finalized completed trials only;
- keep activity commits independent of carrier rejection;
- retain exact no-op carrier behavior at M30;
- compare against activity-only CDM, not merely against sealed static;
- report proposal, commit, rejection, and cumulative-delta norms per session.

The carrier-update claim requires a positive incremental result over
activity-only CDM. Beating only sealed static is insufficient.

## 8. What the paper may say now

Supported now:

> A causal completed-trial activity memory substantially improves short-budget
> cross-session decoding without target gradients: +0.1068 external R2 at M4
> and +0.0934 at M10 on DANDI 000688.

Supported now, as a separate training result:

> Calibration-prefix exposure makes the activity identity path robust to short
> support and improves static M4/M10 decoding, with an external-M30 trade-off.

Not supported now:

- posterior mean is a generally better T4 carrier;
- an unconstrained reliability feature improves transfer;
- carrier updates improve activity-only CDM;
- precision gating adds performance beyond activity-only CDM;
- C1 and CDM stack;
- the complete method transfers unchanged to datasets without explicit trial
  boundaries.

## 9. Recommended execution order

1. Implement and score A3 (`C1 weights + activity-only CDM`). No training.
2. Publish the 2 x 2 factorial and interaction.
3. If A3 is at least as good as A2, run the source-only C2-S interpolation
   screen.
4. Score P0--P4 once only if the source screen is positive.
5. Do not build online posterior carrier updates unless P1 beats P0.
6. Use a new seed or untouched evaluation surface before turning the
   budget-routed policy or stacking result into a confirmatory claim.

This order concentrates work on the validated activity-memory mechanism and
gives posterior uncertainty one precise, falsifiable role without allowing the
failed C2/C3 systems to dominate the program.
