# H1 Sparse-Event Endpoint Carrier — Frozen Development Protocol

**Status:** frozen before the source-only functional-transfer audit; no GPU result exists yet  
**Date:** 2026-08-11  
**Scope:** the 13 public `sub-HumanPitt-held-in-calib` recordings only  
**Candidate name:** `H-SE4` (H1 sparse-event, four-dimensional carrier)

## 1. Question

Can H1 calibration replace the current dense, per-bin 7-DoF kinematic carrier with a
per-movement-event descriptor while preserving useful target-session functional identity?

The candidate does **not** fit an eight-phase mean neural profile and does not require all eight
phase categories to occur. Every valid native movement epoch is one observation. Its target is the
7-DoF endpoint displacement of that event:

\[
\Delta q_e = q(t_e^{\mathrm{stop}})-q(t_e^{\mathrm{start}}).
\]

The neural response is the per-channel event rate. A source-frozen low-dimensional displacement
basis maps `Delta q_e` to three coordinates `z_e`; each target-session channel is then fitted by
closed-form ridge regression:

\[
\log(1+r_{i,e}) = b_i + w_i^\top z_e + \epsilon_{i,e}.
\]

The deployed side feature is `[w_i1, w_i2, w_i3, b_i]`, preserving `side_dim=4`. Calibration uses
no neural-network backward pass and stores only one four-value row per channel after fitting.

## 2. Data and no-leakage boundary

- Input allowlist: the exact 13 public held-in-calibration H1 NWBs already enumerated by
  `src.data.h1_m4_eb_pilot.H1_HELDIN_SESSIONS`.
- Rejected before opening: any path containing `held-out`, `heldout`, `minival`, `formal`, `private`,
  `evalai`, or `test_ecephys`.
- The CPU audit uses calendar-date leave-one-date-out (LODO). Displacement normalization and the
  three-dimensional basis are fitted only on non-outer-date recordings.
- Primary deployment budget: the first three chronological eval-valid `TrialNum` trials (`M=3`).
  This matches the organizer-facing H1 calibration budget.
- Companion development budget: the first four trials (`M=4`), matching the existing dense
  CarrierID development protocol.
- Later trials are used only for the CPU forward-transfer diagnostic. They never enter the carrier
  fit for that session.

## 3. Native event and endpoint contract

The frozen movement tag allowlist is:

`Reach`, `Orient`, `SnapTo`, `Shape`, `Grasp`, `Carry`, `Orient2`, `Release`.

`Presentation*`, `Intertrial`, and unknown tags are not movement observations. A movement epoch is
valid when:

1. it has finite, ordered native start/stop times;
2. the sampled interval belongs to exactly one finite `TrialNum` value (epochs crossing a trial
   boundary are excluded rather than clipped into a synthetic partial event);
3. that trial is eval-valid and the epoch contains at least five eval-valid 20-ms samples;
4. the native `OpenLoopKinematics` start and stop positions can be linearly interpolated using
   brackets no wider than one 20-ms sample;
5. the resulting 7-D displacement and event neural rates are finite.

Missing phase categories do not invalidate a trial or session. No phase one-hot vector or
phase-by-velocity interaction is present.

The endpoint source is the native `OpenLoopKinematics` position series, whose declared dimensions
are `tx,ty,tz,rx,g1,g2,g3`. `OpenLoopKinematicsVelocity` is not read by this candidate. Therefore a
successful audit supports a genuinely endpoint-sparse input contract rather than a displacement
reconstructed by integrating the dense velocity trace.

## 4. Source-frozen displacement basis

For each outer date and budget separately:

1. pool valid support-event displacements from non-outer recordings;
2. fit a seven-dimensional mean and componentwise standard deviation with a `1e-6` floor;
3. apply SVD to the standardized, centered displacement matrix;
4. retain exactly three components and canonicalize each sign by its largest-magnitude loading;
5. divide projected scores by their source standard deviation, again with a `1e-6` floor.

The target date is projection-only. The three-dimensional width is fixed by the four-dimensional
hardware-facing interface; it is not selected from target results. The receipt reports explained
variance so loss from this width is visible.

## 5. Closed-form target-session estimator

For one session and budget `M`, concatenate all valid events in the first `M` trials. Let
`X=[1,z]` and `Y=log1p(event_rate)` with shape `[events,176]`. Fit

\[
\hat B=(X^TX+n\lambda D)^{-1}X^TY,
\]

where `D=diag(0,1,1,1)`, `n` is the event count, and `lambda=0.1`. The intercept is not penalized.
The fixed normalization makes this penalty invariant to event count and displacement scale. The
carrier is `B.T`, ordered `[w1,w2,w3,b]`.

The implementation must fail closed if there are fewer than eight valid support events, if the
four-column design is rank deficient, or if any coefficient is nonfinite.

## 6. CPU source-only audit

For every recording and both `M=3` and `M=4`, fit the carrier on the support trials and predict
event log-rates in all later trials. Report per-session median channel `R²` differences for:

- correct endpoint pairing minus an intercept-only rate model;
- correct endpoint pairing minus a deterministic within-trial cyclic endpoint-label shuffle.

Also report:

- valid/excluded event counts and exclusion reasons;
- valid events per support trial and endpoint scalar accounting;
- raw 7-D and retained 3-D label counts;
- LODO source-basis explained variance and condition/rank checks;
- per-channel coefficient stability from trial splits where both fits are defined;
- mean, median, sign count, and leave-largest-absolute-session-out mean for paired deltas.

### Frozen GPU entrance gate

An `H-SE4` GPU arm may be launched only if one of the two budgets satisfies all of:

1. all 13 recordings have a defined carrier with at least eight support events and rank four;
2. every LODO source basis retains at least 50% of standardized displacement variance in three
   components, and the median retained variance is at least 65%;
3. correct-minus-shuffled forward-transfer delta has positive mean and median, at least 8/13
   positive sessions, and positive leave-largest-out mean;
4. correct-minus-intercept forward-transfer delta has positive median and at least 7/13 positive
   sessions.

`M=3` is preferred. If only `M=4` passes, the candidate is development-only and must not be
described as organizer-budget compatible.

## 7. GPU matrix after a passing CPU gate

The candidate is a new data arm; it does not overwrite the sealed dense CarrierID implementation.
The initial matched matrix is:

| Arm | Carrier | Purpose |
|---|---|---|
| `H-S` | existing SPINT identity | matched baseline |
| `H-C0` | four zeros | compact-consumer control |
| `H-Dense` | existing dense CarrierID | accuracy/reference-information ceiling |
| `H-SE4` | correct sparse endpoint carrier | candidate |
| `H-SE4-LS` | within-trial cyclic endpoint-label shuffle, separately trained | label-content control |
| `H-SE4-RS` | deterministic channel-row shuffle, separately trained | attachment control |

All arms use identical source/date splits, seeds, epoch rule, neural windows, decoder architecture,
and query windows. The first GPU cell is one fold and seed 42. Expansion is allowed only if
`H-SE4-H-C0 > 0`, `H-SE4-H-SE4-LS > 0`, and `H-SE4-H-SE4-RS > 0` in that cell. The full result must
report mean, median, date/session sign count, and paired uncertainty; a single mean is insufficient.

## 8. Label-information claims

The receipt counts endpoint access before projection:

- acquisition-level count: two 7-D position endpoints per valid event;
- derived-label count: one 7-D displacement per valid event;
- model-input count: one 3-D projected displacement per valid event.

PCA does not turn a 7-D acquisition into a three-scalar annotation. The fair comparison with the
dense carrier is the acquisition-level count versus the number of per-bin 7-D velocity values read.
Neural exposure is reported separately and is not described as reduced.

## 9. Required implementation evidence

Before scientific interpretation, the implementation must have tests covering:

- native endpoint interpolation and 7-D conversion/offset handling;
- exclusion of cross-trial and undersampled epochs without requiring all phase names;
- first-three/first-four support boundaries and later-trial isolation;
- source-only PCA fitting and sign canonicalization;
- normalized closed-form ridge and exact carrier column order;
- deterministic label and row shuffles with no fixed points;
- forbidden-path rejection;
- deterministic receipt hashes and independent receipt verification.

This protocol is a development contract, not evidence that `H-SE4` improves decoder R².
