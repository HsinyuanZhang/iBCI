# H1 Sparse Event Endpoint Carrier — Design Screen and Status

**Date:** 2026-08-11; terminal update 2026-08-12
**Status:** Ten CPU-only design rounds completed and sealed below the frozen +0.02 GPU gate. The
matched H-SE5/Zero5 M4 first cell completed with a positive carrier-content result
(`Full−Zero5 = +0.028469`), while the gain over sealed H-S was only `+0.003204`. This new decoder
translation evidence reopens exactly one fixed `ser_context_q4` Full cell under the addendum below;
no nested/meta/estimator carrier arm is queued.

---

## What was tried

Four of the ten CPU rounds were direct carrier-design screens, testing distinct approaches to
improving label utilization on H1 (176 channels, 3-4 trials, 17-23 movement events). The remaining
rounds tested estimator, transfer, nonlinear-label, nuisance, and low-rank hypotheses below.

### Round 1 — Fixed carrier design matrix

Tested seven designs: global PCA endpoint, source-supervised endpoint, delta + midpoint + tag, tag × delta, tag × delta + tag × midpoint, duration-weighted ridge, and q7 dimension ceiling diagnostic.

Best design: source-encoding [delta, midpoint, tag].

- vs global H-SE5 mean: +0.0154 (M3), +0.0144 (M4), 13/13 positive
- vs intercept: +0.031 (M3), +0.035 (M4)
- vs label-shuffle: +0.048 (M3), +0.050 (M4)
- vs tag-shuffle: +0.019 (M3), +0.020 (M4)

Negative findings: q4→q7 gives no stable improvement; duration weighting gives no improvement.

### Round 2 — Nested source-only context selection

Tested start/midpoint/stop states, ridge λ sweeps (0.1/1/3/10), target ridge (1/3/10), response z-score vs center-only, source coefficient equalization. Configuration selected per outer date using only other source dates.

- vs global H-SE5 mean: +0.0142 (M3), +0.0143 (M4), 13/13 positive
- vs intercept: +0.031 (M3), +0.033 (M4)
- vs tag-shuffle: +0.018 (M3), +0.020 (M4)

Nested tuning did not amplify the effect.

### Round 3 — Source-only meta-learned label basis

Optimized a label basis directly on source sessions: does a carrier fitted on the first M events predict later-event neural responses? Used source-only CPU backward; deployment remains closed-form 5-D ridge with zero target optimizer/backward steps.

- vs global H-SE5 mean: +0.0161 (M3), +0.0123 (M4), 13/13 positive
- vs intercept: +0.030 (M3), +0.032 (M4)
- vs tag-shuffle: +0.022 (M3), +0.023 (M4)

Source meta-objective decreased clearly on all six outer dates, but target outer-date gain did not break +0.02.

### Round 4 — Endpoint label semantics

Tested whether an event firing-rate response should be paired with endpoint mean velocity rather
than displacement, without adding any position samples or widening the carrier.  Arms were exact
H-SE5 displacement, PCA mean velocity, source-encoding mean velocity, and source-encoding
source-normalized direction plus log speed.

- `ser_mean_velocity_q4` vs H-SE5 mean/median: `+0.00518/+0.00592` at M3 and
  `+0.00583/+0.00469` at M4;
- `ser_direction_speed_q4`: `+0.00475/+0.00613` at M3 and `+0.00484/+0.00329` at M4;
- both source-encoding velocity arms retained positive shuffle/intercept margins, but neither
  reached the material gate;
- unsupervised PCA mean velocity was slightly negative relative to displacement at both budgets.

Thus rate-versus-displacement semantics explains only a small part of the plateau.  The screen is
sealed `STOP_CPU_SEMANTIC_CANDIDATE_NOT_MATERIAL`; it authorizes no GPU arm.

---

## What the completed CPU program establishes

1. Tag and endpoint-state context carry real functional information (tag-shuffle is positive on 13/13 in the applicable rounds).
2. Context carrier improves correct−intercept from ~+0.010 (global H-SE5) to ~+0.030–0.035.
3. But the incremental gain over the global endpoint carrier is stable at +0.012–0.016, below the frozen +0.02 GPU gate.
4. Source objective optimization (meta-learning) does not transfer to target gain improvement.
5. The observed plateau was not materially removed by basis changes, PCA dimensionality, ridge
   regularization, event exposure weighting, feature normalization, Poisson count/exposure fitting,
   trial/population nuisance removal, empirical-Bayes shrinkage, low-rank sharing, or affine/quadratic
   source-to-future correction.
6. Replacing displacement with endpoint mean velocity/direction-speed yields only about +0.005 in
   the CPU forward-transfer diagnostic, so label dimensional semantics is not the missing material
   effect either.

The observed bottleneck is a **five-parameter, linear event-regression plateau under the tested
label maps and source-transfer operators**. The 17--23 event observations imply low per-channel
precision, and cross-date drift can limit transfer, but the ten screens do **not** prove an
information-theoretic ceiling. Outside the sealed ten-screen program, a source-selected global
neural latency and endpoint-sparse temporal structure remain untested; neither is authorized for
GPU work by the completed screens.

---

## Receipts

| Round | Path | SHA-256 | Verifier |
|---|---|---|---|
| 1 | `h1_event_carrier_design_screen_v1/source_screen.json` | `74bbc014...` | PASS |
| 2 | `h1_event_carrier_nested_context_v2/source_screen.json` | `2c273cd7...` | PASS |
| 3 | `h1_event_carrier_meta_basis_v3/source_screen.json` | `e52a7cdd...` | PASS |
| 4 | `h1_event_carrier_semantic_v4/source_screen.json` | `5a96b204...` | PASS |

All four verifiers checked the applicable source/input/code hashes, exact H-SE5 baseline reproduction
(78 control quantities, max difference 0.0), aggregate/gate arithmetic, and terminal STOP status;
the nested/meta verifiers additionally checked configuration selection and source meta-loss.
Combined event-carrier tests: 23 passed.

---

## Residual hypotheses (brainstorm, not committed)

The completed program has already tested the empirical-Bayes, likelihood, nuisance, low-rank,
nonlinear-label, and source-to-future correction ideas that earlier versions listed here. Two
conceptually distinct routes remain outside the frozen program:

1. **Fixed source-selected neural latency.** Shift the spike-count interval relative to the native
   event by one global latency selected entirely on source dates while leaving endpoint labels
   unchanged. Per-channel latency is excluded at the M3/M4 budget.
2. **Within-event temporal regression.** Split each event into K sub-intervals to provide more
   observations. This directly addresses sample count, but reads the within-event position
   trajectory, weakens the endpoint-only sparse-label claim, and is therefore a different scope.

For completeness, the other previously proposed axes are now closed:

3. **Endpoint mean velocity — tested and stopped.** The fitted neural response is an event
   firing *rate*, whereas H-SE5 uses event *displacement*.  V4 tested `Delta q / Delta t` and
   direction plus log speed with the same endpoint/timestamp budget.  The best gain was only about
   `+0.005`, so this is no longer an untested rescue axis.
4. **Count/exposure likelihood and within-trial contrasts — tested and stopped in estimator V5r2.**
5. **Channel-correspondent shrinkage — tested and stopped in estimator V5r2.**
6. **Reduced-rank multi-output fitting — tested and stopped in LRT5r2.**
7. **Support-only population/trial nuisance removal — tested and stopped in PNO5.**
8. **Affine/quadratic source-to-future correction — tested and stopped in C2F5/QC2F5.**

Robust likelihood variants, bootstrap bagging, learned set estimators, confidence as a sixth input,
per-tag target regressions, and within-event dense trajectories remain lower-priority ideas. They do
not justify extending the current GPU program after the frozen CPU gate was missed.

### Completed V5 estimator screen

V5 was deliberately not another endpoint basis, tag encoding, or decoder fusion.  Its candidate
matrix was fixed before its outer-date results were read:

1. a Poisson count model with event duration as an exposure offset;
2. a within-trial contrast fit that removes trial-rate nuisance before estimating the shared
   endpoint slopes and restores a single support intercept;
3. source-only empirical-Bayes/James--Stein coefficient shrinkage, with every prior and shrinkage
   hyperparameter selected without the outer date.

All arms emit the same five deployment values per channel and use the same native endpoint
observations as H-SE5.  The screen is CPU-only, uses M3/M4 source-date LODO, and reproduced the
H-SE5 reference in 78/78 comparisons with maximum absolute difference `2.22e-15`.  A candidate was
called material only if, at **both** budgets,
its mean delta versus H-SE5 is at least `+0.02`, its median is at least `+0.01`, at least 10/13
recordings are positive, the leave-largest-absolute-recording-out mean is positive, and the
matched endpoint-label-shuffle/intercept/attachment controls remain positive.

| Estimator | M3 delta vs H-SE5, mean / median / signs | M4 delta vs H-SE5, mean / median / signs | Decision |
|---|---:|---:|---|
| Poisson count/exposure IRLS | `-0.10280 / -0.10571 / 0/13` | `-0.07918 / -0.08191 / 0/13` | STOP |
| Within-trial contrast ridge | `-0.00112 / -0.00116 / 5/13` | `+0.00045 / +0.00140 / 7/13` | STOP |
| Channel-correspondent EB shrinkage | `+0.00969 / +0.01169 / 11/13` | `+0.00557 / +0.00518 / 10/13` | below material gate; STOP |

The authoritative immutable receipt is
`sua_exploration/results/h1_event_carrier_estimator_v5/source_screen_v2.json`, SHA-256
`6fa4405e29abc478a64f1c03619f89f1bcf31e1a77db314be014b241764d2645`, mode `0444`; its independent
verifier passed.  Poisson correct and label-shuffle fits converged on 13/13 recordings at both
budgets.  The original `source_screen.json` receipt, SHA-256 `a50ff370...eeb86`, is preserved but
invalidated: root audit found that its IRLS working response subtracted the duration offset twice,
and its EB variance used the wrong ridge covariance.  V5r2 corrects the exposure equation, uses the
frequentist ridge sandwich covariance and effective residual degrees of freedom, and adds synthetic
tests that would fail under the old implementation.  Only V5r2 may be cited.

The mechanistic result is narrower than “all estimator improvements fail.”  A properly converged
Poisson log-rate model is materially worse than H-SE5 on this endpoint; removing trial-wide rate
offsets is indistinguishable from ordinary ridge; channel-correspondent EB provides a small,
pairing-dependent variance reduction but not the required cross-budget effect.  None authorizes a
GPU follow-up.

The older H1 empirical-Bayes and reduced-rank receipts do not make this screen redundant.  They
operated on the dense population-decoder carrier, not on the sparse endpoint event regression, and
the corresponding GPU path was later shown to inject an approximately `1e-9`-scale identity
residual.  They do impose a control requirement: a source prior cannot count as functional content
unless the correctly paired target estimate beats both its label-shuffled refit and a row-attachment
shuffle.  A mostly static channel-index lookup is not a positive sparse-carrier result.

### Completed CPU screens: C2F5 future correction and LRT5 low-rank sharing

C2F5 asked whether source
sessions contain a transferable, signed correction from the low-budget support coefficient to the
coefficient that predicts later events.  For each outer date, a tiny affine/ridge operator is fit
only on non-outer source recordings:

```text
input_i = [H-SE5 support coefficient_i, analytic support fit-quality statistics_i]
B_corrected_i = B_support_i + F_source(input_i)
```

The operator is shared across all channels and receives no channel/electrode index.  Its teacher
coefficients and any inner-model selection use later events from source recordings only.  On the
outer-date recording it receives the chronological M3/M4 support statistics once and never reads
the later labels until scoring.  The descriptor remains `[w1,w2,w3,w4,b]`; target deployment uses
one small forward matrix and no optimizer or backward step.

C2F5 is distinct from V3, which learned a linear map of the endpoint/tag label space, and from V5
EB, which shrank each target coefficient toward a source channel prior.  It is also more specific
than the earlier SUA cross-budget reliability audit: that program predicted a scalar error
magnitude and did not implement a signed future-coefficient correction.  The earlier non-transfer
result remains an adverse prior, so C2F5 uses a small shared operator rather than a free MLP.

The frozen controls were raw H-SE5, endpoint-label-shuffled support passed through the same operator,
row-shuffled corrected rows, a quality-only/no-`B_support` correction input, and intercept-only.  At M3 and
M4 the unchanged material gate requires mean/median C2F5-minus-H-SE5 of at least `+0.02/+0.01`, at
least 10/13 positive recordings, positive leave-largest-absolute-out mean, and robust wins over all
four controls.  C2F5 improved over H-SE5 by `+0.00972/+0.01073` mean/median at M3 and
`+0.00756/+0.00844` at M4, with 11/13 positive at both budgets.  Correct−label-shuffle remained
positive, but the mean gain was below `+0.02`; its sealed outcome is
`STOP_CPU_C2F5_NOT_MATERIAL`.  Receipt SHA-256:
`4096221bbfaf0c7c3c39e50a59e47073f48d60d6d8fcc03ea289eea413d829e8`.

LRT5 separately used a source-frozen low-rank channel subspace and a target-pairing-conditioned
shared coefficient fit.  It improved over H-SE5 by `+0.01165/+0.01272` at M3 and
`+0.00683/+0.00786` at M4, also below the mean gate.  A root audit corrected only the static-prior
control's intercept to `mean(Y)-mean(Z)@W`; after that fair alignment, LRT5 exceeded the prior by
only `+0.00027` at M3 and `+0.00198` at M4.  The authoritative r2 receipt SHA-256 is
`cf4a8a6256a9743a8cc0ee26eef62debae28a4a51db48e8ea485063a32ed4729`; r1 is superseded only for
that control.  Its sealed outcome is `STOP_CPU_LRT5_NOT_MATERIAL`.

NLE5 nonlinear sparse-label embedding then tested the six event tags actually present in every
source/target scoring pool.  It was worse than H-SE5 by `−0.00899/−0.00857` mean/median at M3 and
`−0.00806/−0.00687` at M4, although correct labels and tags still beat their shuffles.  This is a
representation-quality failure, not an absence-of-information result.  Receipt SHA-256:
`45f346decbda51742235efd63cf2feecbdb2650c2a2a141aaa1769117abef3d9`.

PNO5 support-only population/trial nuisance removal was indistinguishable from H-SE5: mean delta
`+0.00003` at M3 and `−0.00020` at M4, with 7/13 positive at both budgets.  It also failed the
nuisance-row-shuffle mechanism control.  Receipt SHA-256:
`0ad94ca988b92a23b69e5be2d0f3ce42aca41c4d2e4536d3f0ef123cf7fa4f52`.

QC2F5 finally tested a frozen degree-two, channel-shared support-to-future correction because linear
C2F5 had a small consistent gain.  It reproduced H-SE5 and linear C2F5 exactly, but improved over
H-SE5 by only `+0.01104/+0.01439` mean/median at M3 and `+0.00962/+0.00892` at M4.  More decisively,
its increment over linear C2F5 was only `+0.00132` at M3 and `+0.00206` at M4, with 7/13 and 8/13
positive.  Receipt SHA-256:
`ef8b216c90046ca465c4bd687ea0737eb261c18caa1d787a657feb84a24fa51f`.

An additive correction receipt, SHA-256
`5d66a23179bd16adba96a2f224dd4b8d9e80ba389426ebbfed1f9d53cd22e180`, fixes only 12 nested
tie-break description strings: the predeclared and executed rule was strongest-ridge-on-tie, and
all stored selected lambdas already match it.  Metrics and gates did not change.  Source-only
covariate-shift weighting and a source-frozen state-space posterior remain brainstorm ideas, not
running experiments; the bounded estimator search is now stopped.

### Completed CPU screen: V4 semantic alignment

The first follow-up stayed endpoint-sparse and five-wide.  Four arms were frozen before reading
their outer-date results:

1. `pca_delta_q4` — exact H-SE5 reproduction;
2. `pca_mean_velocity_q4` — PCA of `Delta q / Delta t`;
3. `ser_mean_velocity_q4` — source-encoding basis of `Delta q / Delta t`;
4. `ser_direction_speed_q4` — source-encoding basis of unit mean-velocity direction plus log speed.

All maps were fit LODO on non-outer-date source recordings.  Target fitting remained the same
five-coefficient ridge, M3/M4 chronological support, with correct, endpoint-label-shuffle, and
intercept controls.  Acquisition position counts remain unchanged; event start/stop timestamps are
already required by H-SE5.  GPU entry retains the existing material gate at **both** M3 and M4:
mean delta versus H-SE5 at least `+0.02`, median at least `+0.01`, at least 10/13 positive, positive
leave-largest-absolute-session-out mean, and robust positive margins over shuffle and intercept.
No arm passed.  The current H-SE5/Zero5 pair continued unchanged throughout the sub-second V4
screen; V4 does not authorize another GPU arm.

Two-sentence pitch: H-SE5 may be sample-limited partly because it regresses event firing rates on
displacements with inconsistent duration semantics.  A duration-normalized endpoint label tests
the same generalized-T4 principle with no denser annotation, no wider online state, and no target
backpropagation.

The first validation stage is complete and negative under its material gate. Count/exposure was
subsequently tested and stopped in corrected V5r2. Global latency remains a distinct CPU hypothesis,
but it does not justify a decoder run under the frozen program. This prevents a sequence of small
CPU positives from turning into uncontrolled GPU search.

No new GPU arm is queued. H-SE5 did translate into a positive decoder-level content margin versus
the independently trained Zero5 arm, but the ten candidate refinements still missed their frozen
CPU entry gate. V4 therefore remains a CPU mechanism audit and does not justify another decoder
variant.

---

## Terminal decision

The matched M4 fold-0 seed-42 pair and its independent terminal audit are complete:

- H-SE5 Full: `0.500037`; independently trained Zero5: `0.471569`; delta `+0.028469`;
- same-checkpoint label shuffle: `0.492945`; correct-label margin `+0.007092`;
- both target recordings are positive for Full−Zero5 (`+0.02590`, `+0.03593`) and
  Full−label-shuffle (`+0.00712`, `+0.00702`);
- sealed H-S: `0.496833`, so H-SE5 exceeds it by only `+0.003204`;
- sealed dense H-C: `0.525511`, so H-SE5 remains lower by `0.025474`.

This is a positive mechanism result: sparse endpoint content survives the compact consumer and
correct pairing matters. It is not a broad accuracy-upgrade claim, because the margin over H-S is
small and the result covers one fold and one seed. C2F5, LRT5, NLE5, PNO5, QC2F5, and the earlier
tag/context designs remain closed below their CPU gate; no additional carrier GPU arm follows from
this program.

The immutable terminal receipt, independent structural audit, independent prediction recomputation,
and final program completion have SHA-256 prefixes `f53f527d`, `f2891d0a`, `2ee6bcdb`, and
`4ed0fdc7`, respectively.

### One changed-evidence exception

The old `+0.02` CPU gate preceded any observation of how sparse carrier content translates through
the compact consumer. H-SE5's positive decoder content margin changes that premise. It does not
validate the proposed proportional extrapolation, because CPU neural forward-transfer R² and GPU
behavioral decoder R² are different quantities, but it makes one matched decoder cell informative.

The sole reopened candidate is fixed `ser_context_q4`: M4 mean/median improvement over H-SE5 is
`+0.014438/+0.013464`, 13/13 recordings are positive, and the decoder-facing carrier remains five
wide. It is simpler and slightly stronger at M4 than the nested or meta variants. Only Context Full
is trained; a literal-zero-boundary parity audit permits reuse of sealed H-SE5 Zero5 as the common
independently trained null. The material gate is fixed at Context-minus-H-SE5 `>= +0.010` pooled
with both target recordings positive and all content/attachment controls positive. A smaller
same-sign gain is recorded as nonmaterial and does not trigger another experiment.

The final supplemental CPU evidence index is
`sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v2.json`,
SHA-256 `f284d53e2f956518d367fc1487777de17186955c589a4879be9a8e79d5e93a04`, mode `0444`; its verifier
and focused test pass and it authorizes zero new GPU arms. The preserved closure v1
(`39c2b02d...67786a3`) is superseded only for nomenclature and omitted estimator-V5 coverage; no
historical receipt or result was rewritten.
