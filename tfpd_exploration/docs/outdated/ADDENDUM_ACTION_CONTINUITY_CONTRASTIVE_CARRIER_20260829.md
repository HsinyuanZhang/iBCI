# Addendum: Action-Continuity Contrastive Carrier

Date: 2026-08-29  
Status: ADDENDUM DESIGN v1 — for independent review  
Authorization: no new data access, training, scoring, GPU launch, root creation,
or modification of an already dispatched experiment  
Primary datasets: M2 and SUA/pseudo-MUA internal trial protocols  
Working name: AC3 — Action-Continuity Contrastive Carrier

---

## 0. Purpose

This addendum changes the interpretation and future ordering of the learned-gate
program after the terminal P2-prime result. It does not replace, edit, stop, or
restart any experiment that has already been dispatched.

The central revision is:

> Stop treating temporal smoothing as a way to improve carrier pseudo-labels.
> Add a new source-trained contrastive representation that uses action
> continuity and cross-session action correspondence to estimate completed-trial
> direction and credibility.

Final-output smoothing, contrastive carrier representation, and carrier action
selection are three separate experimental axes.

## 1. Documents and immutable evidence

### 1.1 Parent design documents

- `tfpd_exploration/docs/DESIGN_LEARNED_GATE_SMOOTHED_PSEUDOLABEL_20260828.md`  
  SHA256 `7d61c445e23c10a4bc0e614b9dcb972a370867c488ee93fc0a18e925973f0f02`
- `tfpd_exploration/docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md`  
  SHA256 `32508a28ec96ab0edc89e9053e941b748b6615dc8dc51176c2cc730c40deb816`

This addendum is a successor decision document. It does not silently change the
meaning of either parent document or any artifact already bound to them.

### 1.2 P2-prime terminal evidence

- `tfpd_exploration/results/learned_gate_p2prime_v1/result.json`  
  SHA256 `0b1264318c37b31fa34d655c1bf89989c6c97090c17821a092f2c4daa1971303`
- `tfpd_exploration/results/learned_gate_p2prime_v1/terminal.json`  
  SHA256 `0ab2c2fef8c0c2e880ae47e57567921e1d5dd95aae721169cc64369d3f115b36`

P2-prime is terminal, immutable, target-update-free, and explicitly
non-governing. Its verdict is:

```text
SMOOTHING_NO_CARRIER_VALUE_KEEP_OUTPUT_FILTER_ONLY
```

### 1.3 Earlier output/activity evidence

- continuity probe SHA256  
  `8afaa9109f2dbeb1fb68e1044c4e7ae275771113c4cb3daf5c5565ea77c588b1`
- accepted activity-only result SHA256  
  `48d658c866b0bcb45a9f03c6e0c84204ecb87286e940d850c846a46f67b33bb8`

These remain evidence for the final-output filter branch. They are not evidence
that smoothing improves carrier direction.

### 1.4 Literature reference

Primary reference:

> Schneider, S., Lee, J. H. & Mathis, M. W. Learnable latent embeddings for
> joint behavioural and neural analysis. Nature 617, 360–368 (2023).
> https://doi.org/10.1038/s41586-023-06031-6

Official resources:

- paper: https://www.nature.com/articles/s41586-023-06031-6
- implementation: https://github.com/AdaptiveMotorControlLab/CEBRA
- usage and multi-session training: https://cebra.ai/docs/usage.html
- multi-session conditional sampling:
  https://cebra.ai/docs/api/pytorch/distributions.html

AC3 is CEBRA-inspired. It does not claim to be the standard CEBRA model or a
reproduction of the Nature paper.

## 2. What P2-prime closed

P2-prime compared raw and smoothed completed-trial pseudo directions under the
same matched causal carrier-action problem.

| Contrast | M4 external | M10 external | Disposition |
|---|---:|---:|---|
| `O1 - O0`: smoothed-pseudo versus raw-pseudo coherent action value | -0.000236 | -0.000343 | smoothing has no carrier value |
| `P - A1`: hand gate versus smoothed activity-only | -0.001363 | -0.025537 | current hand gate has no value |

The following successors are therefore stopped:

1. longer FIR smoothing for carrier pseudo-labels;
2. a trainable EMA used only to smooth carrier pseudo-labels;
3. TCN/GRU output smoothers whose carrier claim depends only on smoother
   velocity trajectories;
4. threshold sweeps around the failed hand gate;
5. counting final-output smoothing gain as learned-carrier gain.

An already dispatched experiment in one of these categories is allowed to
finish unchanged. Its role is negative-control or replication evidence. It does
not authorize another expansion.

## 3. What P2-prime did not close

Two large matched opportunities remain.

| Contrast | M4 external | M10 external | Meaning |
|---|---:|---:|---|
| `O1 - A1` | +0.0645 | +0.0602 | coherent carrier admission still has action value |
| `O2 - O1` | +0.0922 | +0.0320 | pseudo-direction quality remains a bottleneck |

P2-prime did not fit a deployable learned policy. KC4 remains pending in that
experiment. It also did not train a source-supervised representation that
aligns similar actions across sessions.

The remaining questions are:

1. Can source-only features identify a useful fraction of coherent carrier
   action value?
2. Can action-continuity contrastive learning reduce completed-trial direction
   error across held-out source sessions?
3. Does a better direction representation improve future carrier utility, not
   merely representation quality?

## 4. Separate the three mechanisms

### 4.1 Final-output filter

Purpose: remove decoder jitter from the emitted velocity stream.

The activity-only matched P2-prime result is:

| Budget | External delta | Positive external sessions | Within delta | Positive within sessions |
|---|---:|---:|---:|---:|
| M4 | +0.019409 | 12/15 | -0.043985 | 0/6 |
| M10 | +0.023361 | 12/15 | -0.048432 | 0/6 |
| M30 | +0.038512 | 15/15 | -0.038780 | 0/6 |

This is an external-denoising result with a systematic within penalty. Its
successor must address conditional bypass and lag. It is not a carrier result.

### 4.2 Contrastive carrier representation

Purpose: estimate a completed trial's direction and credibility in a
cross-session-consistent representation.

This is the new AC3 route.

### 4.3 Carrier action gate

Purpose: use deployable, label-free evidence to accept or reject a proposed
carrier update.

The first learned gate remains logistic or a very small MLP. A large network is
not justified until a small gate realizes positive source-held-session utility.

## 5. Change-control rule for dispatched experiments

Every experiment already dispatched before this addendum follows these rules:

1. allow it to reach a natural terminal or failure;
2. do not change its work order, source bytes, arguments, thresholds, root, or
   metric after launch;
3. do not inject AC3 code into its process;
4. do not reuse its result root for AC3;
5. do not relabel a smoothing result as a contrastive-representation result;
6. do not expand seeds/folds automatically after terminal;
7. preserve its original interpretation and add the revised interpretation in
   a successor document or summary.

Specific disposition:

| Existing experiment class | Action now | Future role |
|---|---|---|
| fixed final-output filter | finish unchanged | output-filter baseline |
| adaptive/learned output filter | finish one dispatched pilot | conditional output-filter evidence |
| smoothed-pseudo carrier experiment | finish unchanged | negative control |
| raw-pseudo learned gate | retain | live carrier-policy candidate |
| output-filter-integrated training J2/J3 | finish dispatched pilot only | output regularization; no carrier claim |
| P2-prime oracle rows | immutable | AC3 upper bounds and controls |

## 6. AC3 scientific hypothesis

Neural coordinates, unit identity, firing scale, and decoder errors change
across sessions. Local action continuity and the geometry of movement direction
are more likely to be shared.

AC3 uses this shared structure to learn:

```text
completed-trial observation
    -> shared contrastive encoder
    -> action-continuous latent z
    -> direction mean mu
    -> credibility kappa / group consistency
    -> uncertainty-aware carrier admission
```

The claim is not that temporal neighbors should always have identical outputs.
The claim is that positives and negatives can shape a representation that is
locally continuous, cross-session aligned for similar actions, and separated
for different actions.

## 7. Why this is not another smoothing method

A low-pass filter only reduces high-frequency variation in the original output
coordinates. It does not distinguish decoder noise from a true action change.

Contrastive action-continuity learning uses both attraction and separation:

```text
pull together:
    local states whose action is continuous
    cross-session states with similar source action
    complementary-group views of the same action

push apart:
    directionally different actions
    especially speed/phase-matched opposite directions
```

The negatives prevent simple temporal collapse. A CEBRA-style objective may
therefore change cross-session representation geometry even when low-pass
smoothing does not change carrier utility.

## 8. Input contract

### 8.1 Primary v1 input

The first AC3 screen uses completed-trial trajectories already produced by the
accepted frozen decoder under complementary-unit-group views:

```text
trajectory[s, trial, group] = [v_1, ..., v_T],  v_t in R^2
```

This is preferred to raw fixed-width neural vectors because it gives a shared
input dimension across sessions and does not require a new target-session
encoder.

### 8.2 Permitted fixed summaries

A low-cost encoder may consume a predeclared summary containing:

- integrated displacement;
- early/middle/late displacement;
- mean and peak speed;
- endpoint direction;
- speed-weighted direction;
- path curvature;
- duration and valid-bin count;
- complementary-group disagreement;
- output dispersion.

All summaries must be causal with respect to the completed trial and reset at
the trial boundary.

### 8.3 Forbidden v1 inputs

- target behavior labels;
- future trials;
- session ID as a learnable feature;
- recording date or absolute timestamp as a learnable feature;
- external R2 or target error;
- target-fitted normalizers;
- post-trial data outside the declared completed-trial capability;
- a per-target-session trained encoder.

## 9. Contrastive sampling contract

Let `z_(s,g,t) = E_psi(h_(s,g,t))`.

### 9.1 C-Time positives

Use temporal neighbors inside one uninterrupted source trial:

```text
(s, g, t) <-> (s, g, t + delta)
```

Constraints:

- never cross a trial/session/gap reset;
- use a fixed delta distribution;
- exclude ambiguous or invalid rows;
- disclose whether source behavior changed sharply inside the positive pair.

This arm tests pure temporal/action continuity without behavior labels in the
pair construction.

### 9.2 C-Action positives

Use source behavior to align similar action states across sessions:

```text
s1 != s2
distance(u_1, u_2) <= epsilon_action
```

For direction, use circular coordinates:

```text
u_theta = [cos(theta), sin(theta)]
```

Near-zero-speed rows have undefined direction. They must be excluded or assigned
to one predeclared rest condition.

### 9.3 Cross-group positives

Complementary groups observing the same completed trial are auxiliary positive
views. They cannot be the only positives because a same-trial shortcut could
encode time, duration, or intensity instead of action.

### 9.4 Hard negatives

Primary hard negatives match speed and approximate trial phase while changing
direction, especially opposite-direction pairs.

Random negatives are allowed only as an additional pool.

### 9.5 Shuffle control

Direction-label or action-index shuffling must destroy the action-conditioned
advantage. A contrastive result that survives the shuffle is rejected as a
shortcut or non-action representation.

## 10. AC3-0: frozen-representation screen

AC3-0 is the first new stage. It does not retrain Cell-D and does not score an
external target.

### 10.1 Matrix

| Row | Representation | Pair construction | Purpose |
|---|---|---|---|
| R0 | raw `atan2` pseudo direction | none | accepted raw baseline |
| R1 | P2-prime smoothed pseudo direction | none | smoothing negative control |
| R2 | ordinary supervised circular MLP | source direction labels | extra-head control |
| R3 | C-Time | temporal neighbors | pure continuity test |
| R4 | C-Action | cross-session action-near pairs | primary contrastive method |
| R5 | C-Hybrid | temporal + action pairs | conditional combined method |
| RS | shuffled C-Action | shuffled labels | negative control |
| O2 | true completed-trial direction | leakage oracle | upper bound |

### 10.2 Split discipline

- outer unit: source session;
- inner unit: source session, never random windows;
- all rows/bins from one trial remain in one split;
- decoder outputs used to fit the encoder must be source-only;
- temperature, embedding dimension, action threshold, and loss weights are
  selected inside grouped source folds only;
- no external-15 label or score selects any AC3 component.

### 10.3 Primary metrics

1. circular direction error;
2. direction snap mismatch;
3. cross-session direction retrieval;
4. cross-group latent dispersion;
5. correct-minus-shuffle;
6. source-OOF credibility calibration;
7. coherent matched carrier utility under exact replay;
8. recovered fraction of `O2 - O1`;
9. resource cost and deterministic embedding digest.

### 10.4 AC3-0 gates

At least one of R3/R4/R5 must satisfy all representation gates:

- source grouped-OOF circular error improves over R0 by at least `0.10 rad`;
- snap mismatch improves by at least `10` percentage points;
- RS degrades as expected;
- the effect is not carried by one source session;
- all target update counts remain zero.

To claim contrastive value beyond an ordinary head, R4 or R5 must beat R2 by at
least one of:

- `0.05 rad` circular error; or
- `+0.005` coherent matched carrier utility.

Downstream advance additionally requires:

```text
corrected-pseudo coherent utility - raw-pseudo coherent utility >= +0.01 R2
```

If representation metrics improve but carrier utility does not, AC3 stops.

## 11. AC3-1: small learned carrier gate

AC3-1 starts only after AC3-0 passes.

### 11.1 Candidate deployable features

- direction concentration or calibrated confidence;
- temporal latent continuity residual;
- complementary-group latent dispersion;
- circular direction disagreement;
- distance to source direction prototypes;
- initial carrier posterior uncertainty;
- proposal departure;
- carrier age;
- activity-memory progress;
- valid-unit/group fractions.

### 11.2 First model

Use logistic regression. A one-hidden-layer MLP is allowed only if logistic
regression has positive held-session utility and a source-only nonlinear oracle
shows additional recoverable value.

### 11.3 Labels and objective

Gate labels come from source-only future utility under a matched causal replay.
The proposal trial and the future utility evaluation rows must be disjoint.

The primary model-selection objective is realized held-session future R2, not
AUC and not beta-distance classification accuracy.

### 11.4 AC3-1 gate

Advance only if the source held-session learned policy:

- improves realized R2 by at least `+0.01` over its matched reject-all/raw-pseudo
  baseline;
- recovers at least 25% of coherent raw-pseudo opportunity;
- has no material worst-session failure under the predeclared safety rule;
- remains deterministic and target-update-free.

The approximate 25% targets from P2-prime are:

- M4: `+0.016`;
- M10: `+0.015`.

## 12. AC3-2: conditional GPU training pilot

AC3-2 is not authorized by this addendum. It becomes reviewable only after
AC3-0 and AC3-1 pass.

### 12.1 Pilot arms

| Arm | Decoder training | Auxiliary representation | Gate |
|---|---|---|---|
| G0 | matched baseline retraining | none | matched baseline |
| G1 | matched retraining | ordinary direction loss | same small gate |
| G2 | matched retraining | C-Time | same small gate |
| G3 | matched retraining | C-Action | same small gate |
| G4 | matched retraining | C-Hybrid | same small gate |

The first GPU pilot runs G0 and only one source-selected G-star arm, on one fold
and one seed. It does not launch all five arms.

### 12.2 Joint loss

```text
L_total =
    L_velocity
    + lambda_contrast * L_contrast
    + lambda_direction * L_direction
```

`L_velocity` remains governing. Contrastive and direction losses are auxiliary.
Loss weights are source-selected and frozen before external evaluation.

### 12.3 GPU pilot safety

- matched source exposures and optimizer-step count;
- matched seed and initialization policy;
- raw velocity validation retained;
- no target labels or target gradients;
- exact checkpoint selection rule;
- no final-output filter in the carrier-attribution pilot;
- separate receipt for every loss and representation digest.

## 13. AC3-3: matched external factorial

AC3-3 is allowed only after a separate work order and root audit.

The required factorial is:

| Carrier identity/action | Final output |
|---|---|
| raw pseudo, reject-all | raw |
| raw pseudo, learned gate | raw |
| AC3 direction, learned gate | raw |
| raw pseudo, reject-all | fixed output filter |
| raw pseudo, learned gate | fixed output filter |
| AC3 direction, learned gate | fixed output filter |

Primary estimands:

```text
AC3 carrier gain =
    AC3 gate with raw output
    - raw-pseudo gate with raw output

final-output filter gain =
    filtered output with fixed carrier route
    - raw output with the same carrier route

interaction =
    whether the two effects compose
```

Never report only `AC3 + filter - raw reject-all`; that contrast cannot identify
which mechanism produced the gain.

## 14. Statistical status of external-15

The external-15 roster has already informed the motivation of this successor
through output-filter, P2-prime, and oracle results. It is therefore not a
completely untouched confirmatory set for AC3.

Rules:

1. select AC3 architecture and hyperparameters only on grouped source folds;
2. freeze one successor before any AC3 external score;
3. run external-15 once;
4. describe it as a matched successor evaluation on a previously characterized
   roster;
5. do not return to external-15 to retune AC3;
6. use SUA, another independent cohort, or an official evaluation for stronger
   confirmation.

## 15. Cross-dataset scope

### 15.1 M2 and SUA/pseudo-MUA

Primary scope. The output is two-dimensional velocity, action direction has a
natural circular geometry, and internal protocols expose trial boundaries.

Each neural view retains a separate input authority. A representation trained
on one view cannot silently claim transfer to another.

### 15.2 H1

Conditional breadth only. H1 can use action-continuity contrastive learning if
the internal protocol exposes honest segments and direction/task labels, but
its multi-DoF action requires a predeclared behavior geometry rather than the
M2 `[cos(theta), sin(theta)]` label.

### 15.3 M1

Conditional breadth only. M1 EMG and action structure differ from M2, and its
previous latent-space results were weak. AC3 transfers only after M2/SUA proves
downstream carrier or adaptation utility. A generic low-dimensional embedding
is not sufficient evidence.

### 15.4 Formal FALCON deployment

Trial-aware AC3 state is not automatically deployable where the evaluator does
not expose trial boundaries. A separate causal segmentation/change-point
component and audit are required. Internal trial-aware and formal stream-aware
claims remain separate.

## 16. Shortcut and leakage controls

AC3 fails if any of the following is possible:

1. session ID predicts the contrastive target;
2. train and validation contain bins from the same trial;
3. same-trial cross-group pairs are the only positives;
4. time proximity alone explains an action-conditioned result;
5. speed or phase explains direction retrieval;
6. near-zero-speed undefined directions enter the direction loss silently;
7. target labels select the encoder or gate;
8. target unlabeled adaptation changes encoder parameters without a separate
   authorization;
9. shuffled labels retain the claimed advantage;
10. representation quality improves without downstream carrier utility.

Required controls include speed/phase-matched direction negatives, direction
label shuffling, C-Time, ordinary supervised R2, and source-session grouping.

## 17. Resource and scheduling rule

This addendum does not preempt existing CPU or GPU work.

### 17.1 While dispatched jobs are running

- allow them to run unchanged;
- AC3 document/tests may use CPU/synthetic work only if they do not read active
  result roots or contend with their assigned resources;
- do not use the same canonical result root;
- do not change shared source files imported by an active frozen closure;
- wait for natural terminal evidence before binding a predecessor.

### 17.2 New compute ordering

1. AC3-0 frozen/small representation screen;
2. AC3-1 small gate;
3. one AC3-2 G0/G-star pilot;
4. one frozen AC3-3 external score;
5. independent breadth/confirmation;
6. only then expand folds and seeds.

No full GPU matrix starts before AC3-0 and AC3-1 pass.

## 18. Receipt requirements

Every AC3 stage records:

- parent workorder and evidence SHAs;
- source session/fold identities;
- exact trial and row membership digests;
- encoder architecture and parameter digest;
- contrastive sampler and seed;
- positive/negative construction counts;
- temporal, action, cross-session, cross-group, and hard-negative counts;
- invalid/rest row counts;
- embedding dimension, temperature, and normalization;
- embedding and direction prediction digests;
- circular direction error and snap mismatch per session;
- shuffled-control results;
- cross-session retrieval and cross-group dispersion;
- gate features, decisions, and causal state transitions;
- future-utility row separation proof;
- raw and corrected coherent carrier utility;
- target optimizer/backward/update counts;
- model/checkpoint state-before/state-after;
- resource and determinism evidence;
- attempt-before-data/model and atomic terminal/failure lineage.

## 19. Stop conditions

Stop AC3 if any condition holds:

1. no contrastive arm improves R0 circular error by `0.10 rad`;
2. snap mismatch does not improve by `10` percentage points;
3. R4/R5 does not beat ordinary supervised R2 by the declared contrastive gate;
4. shuffled control does not degrade;
5. representation gain is driven by one source session;
6. corrected direction does not improve coherent carrier utility by `+0.01`;
7. the small learned gate has null/negative held-session realized utility;
8. a larger model is required before a small model shows any downstream value;
9. governing velocity performance degrades materially in the GPU pilot;
10. AC3 requires target parameter updates or target-selected hyperparameters;
11. trial/reset chronology cannot be proved;
12. external improvement cannot be separated from final-output filtering.

Do not respond to a STOP by increasing embedding size, adding a recurrent
network, enlarging the hyperparameter grid, or repeatedly scoring external-15.

## 20. Revised execution order

```text
CURRENTLY DISPATCHED JOBS
    -> finish unchanged
    -> preserve original terminal/failure evidence

FINAL-OUTPUT BRANCH
    -> fixed/adaptive output-filter interpretation
    -> solve external gain versus within harm

CARRIER BRANCH
    -> P2-prime immutable baselines and oracle bounds
    -> AC3-0 frozen action-continuity representation
    -> AC3-1 small learned carrier gate
    -> AC3-2 one matched GPU pilot
    -> AC3-3 one external factorial score
    -> independent breadth/official confirmation
```

## 21. Independent-review questions

1. Is the primary AC3 v1 input best defined as a decoded trajectory, a fixed
   trajectory summary, or a shared permutation-invariant neural-set feature?
2. Does C-Time test action continuity cleanly enough, or must positives also
   satisfy a source-behavior continuity threshold?
3. Should cross-group same-trial pairs be auxiliary positives or only a
   consistency regularizer?
4. What speed/phase matching rule gives the strongest direction hard negative
   without creating sparse sampling?
5. Is `0.10 rad` plus `10` percentage points an appropriate constructibility
   threshold before downstream utility?
6. Should credibility be a von Mises concentration, source conformal score, or
   only group/prototype dispersion in v1?
7. Is recovering 25% of the P2-prime coherent opportunity sufficient to justify
   a GPU training pilot?
8. Which independent dataset should serve as the first stronger confirmation:
   SUA, another held cohort, or official evaluation?

## 22. Current disposition

```text
GO:
    independent review of this addendum
    no-data/synthetic AC3 contract work
    AC3-0 only after exact source authority and predecessor review

HOLD:
    AC3-1 until AC3-0 passes
    AC3-2 GPU training until AC3-0 and AC3-1 pass
    AC3-3 external score until a separate frozen authority exists

STOP:
    further pseudo-label smoothing as the carrier mechanism
    attribution of final-output smoothing gain to carrier learning
    modification of any already dispatched experiment
```

Final principle:

> Use action continuity as a source-only correspondence signal to learn a
> cross-session direction representation. Require that the representation
> improve causal future carrier utility before paying for end-to-end training.

---

## 23. Operator decision on the review (2026-08-29) — AMENDMENTS BINDING

> **GO AC3-0 with R-GE mandatory; HOLD AC3-1; NO authorization for AC3-2 until
> AC3-0/AC3-1 pass under the amended gates.**

Six amendments, binding on all successor work orders:

1. **R-GE is added as row R0.5 and is THE formal primary baseline** — the
   zero-learned-parameter group ensemble: circular mean of the four
   complementary-group view directions, dispersion (ρ_GE) as credibility.
2. **The contrastive-claim gate strengthens to beat max(R2, R-GE)** — not merely
   the raw direction baseline — by the declared margins (0.05 rad or +0.005
   coherent utility).
3. **The E7-mirror outcome is formally pre-registered**: R2 (supervised head)
   improves direction and carrier utility; R4/R5 fail to beat R2; disposition =
   keep the supervised direction route, terminate the contrastive method claim.
   This outcome is a SUCCESS of the route, not of the contrastive mechanism,
   and must be reported as exactly that.
4. **ρ_GE credibility calibration is a REQUIRED AC3-0 output** (calibration
   curve of group dispersion vs true direction error), not an incidental
   diagnostic — it is the zero-learning credibility candidate for AC3-1.
5. **AC3-2 remains conditional** on the full AC3-0 + AC3-1 chain; no
   representation-metric improvement alone authorizes GPU training.
6. **AC3-3 retains the 2×3 carrier × filter factorial** as the unified
   mechanism decomposition closing both branches.
