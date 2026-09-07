# M1 Functional Carrier Memory: corrected experiment plan

**Date:** 2026-09-02  
**Status:** design and execution guidance; no GPU launch is authorized by this document  
**Primary scope:** FALCON M1 development surfaces, followed by at most one pre-registered official evaluation  
**Out of scope:** new DANDI 000688 exploration, PACD P1/P2 expansion, target backpropagation, and post-result carrier search

## 1. Executive decision

The latest M1 T0/C1 experiment did **not** use a T4-like functional carrier. It tested only:

1. a fixed M10 B3S calibration-activity identity (`T0`) versus an activity-prefix cycle (`C1`); and
2. static M10 deployment versus activity-only causal FIFO deployment (`CDM-A`).

Consequently, its receipts remain valid, but only for the **carrier-absent activity path**. They cannot reject any of the following:

- an M1 EMG-derived T4-like carrier;
- carrier-quality cycling during source training;
- an interaction between a functional carrier and causal activity memory.

The corrected M1 programme is called **Functional Carrier Memory (FCM-M1)**. It has three separately measurable parts:

- **EMG-Syn3:** a four-coordinate, source-defined nonnegative-synergy functional carrier fitted from target support EMG and neural activity;
- **CQC:** Carrier-Quality Cycling during source training;
- **CDM-A:** activity-only causal deployment memory, with the carrier held fixed unless a later experiment explicitly studies carrier updates.

The first scientific objective is not to claim that all three help. It is to determine which of the three terms—carrier content, carrier-quality robustness, or activity memory—produces a reproducible M1 gain and whether their interaction is positive.

## 2. Exact correction to the latest M1 interpretation

### 2.1 What the latest route actually consumed

The 50-epoch T0/C1 route uses the ordinary `FalconLitModule` API:

```python
forward(x, calib_trialized_neural_features=calib)
```

Its training intervention changes only the chronological prefix of `calib_trialized_neural_features` according to `(10, 5, 2)`. Its Phase-3 deployments are only:

- `static_m10`; and
- `cdm_activity_fifo_m10`.

Neither the training route nor the scorer supplies per-unit `[N,4]` side features. Therefore the following earlier shorthand is incorrect and must not be repeated:

> “M1 carrier plus C1/CDM was null.”

The correct statement is:

> “In a carrier-absent M1 decoder, activity-prefix cycling did not improve the 50-epoch result, while activity-only causal memory produced a small positive local held-out increment over the corresponding static deployment.”

### 2.2 What the completed numbers still tell us

At the 50-epoch Phase-3 endpoint:

| comparison | held-in delta | local held-out-fold delta | interpretation |
|---|---:|---:|---|
| C1 static − T0 static | `+0.001534` | `−0.005906` | activity-prefix cycling alone is null/negative |
| CDM-A − static, T0 | `−0.109238` | `+0.015614` | small cross-session-positive activity-memory signal, with a large held-in cost |
| CDM-A − static, C1 | `−0.115698` | `+0.011107` | the signal survives C1 but is smaller |

These are development results, not a formal benchmark verdict. The 50-epoch recipe also transferred worse than the older 20-epoch recipe, so it must not become the default training horizon for the corrected experiment.

### 2.3 Why existing T0/C1 checkpoints cannot simply be rescored with a carrier

The existing M1 EMG carrier path uses `StreamingCalibrationLitModule` with `side_dim=4`. Its dataset returns:

```text
(neural, target, calibration_activity, session, side_features)
```

and its student consumes aligned `side_features` during training and inference. The latest T0/C1 checkpoints were built with the ordinary carrier-absent model. Adding a carrier only at evaluation would either be ignored or require new, untrained parameters. It would not be an honest evaluation.

Therefore a carrier-aware comparison requires matched retraining. Existing T0/C1 results remain useful as an activity-only reference, not as the matched zero-carrier control for the new topology.

## 3. M1 EMG-derived T4-like carrier

### 3.0 Preferred successor: EMG-Syn3

The preferred new carrier hypothesis is **EMG-Syn3**. It keeps the existing
four-value AFC4 interface but replaces the signed PCA task basis with a
source-frozen, three-component nonnegative EMG synergy basis.

For source movement-window 20-ms EMG bins, fit a deterministic rank-3
nonnegative factorization and freeze the source dictionary. For a target
support bin, infer nonnegative synergy activation

```text
z(t) = NNLS(EMG(t); source_dictionary) in R^3_{≥0}.
```

Then fit, independently for every neural channel, on target support `[0,10)`:

```text
r_i(t) = b_i + W_i^T z(t) + epsilon,
```

with ridge `lambda=1` applied to `W_i` and not to `b_i`. The descriptor remains

```text
EMG-Syn3_i = [W_i1, W_i2, W_i3, b_i].
```

This describes how a unit encodes the low-dimensional EMG field that the M1
decoder must predict. It is not a direction T4, a 16-D DirectRidge readout, or
a target-learned latent representation.

Four implementation corrections are mandatory:

1. **Prove nonnegativity first.** The physical signal may be EMG, but the stored
   `preprocessed_emg` tensor is not automatically guaranteed to be a rectified
   nonnegative envelope. Stage 0 must record its minimum and preprocessing law.
   If it is signed, a source-frozen rectification/envelope operator must be
   specified before NNMF; target-dependent shifting is forbidden.
2. **Do not copy PCA centering.** Mean subtraction destroys nonnegativity.
   EMG-Syn3 may use a source-frozen positive scale normalization, but not the
   signed `(x-mean)/scale` PCA transform. Target activations are obtained by a
   frozen-dictionary nonnegative solve, not by an ordinary dot-product
   projection.
3. **Remove NNMF indeterminacy.** Freeze solver, initialization, seed,
   convergence tolerance, iteration cap, component normalization and component
   ordering. Order components by a source-only rule and bind dictionary and
   transform hashes in every receipt.
4. **Disclose the dense-label budget.** Fitting on 20-ms bins uses many aligned
   EMG observations inside ten trials. It is still chronological M10 support,
   but it is not “ten scalar labels.” Record valid bin count, duration, trial
   count, alignment/lag law and effective design condition. No bin may cross a
   trial or support/query boundary.

`q=3` is predeclared because it preserves the fixed four-coordinate interface
and matches the existing M1 low-dimensional feasibility evidence. The PCA
explained-energy number is supportive geometry, not a proof that NNMF rank 3 is
optimal. No target score may select rank.

### 3.0.1 Correct injection law

The existing B3S side path concatenates carrier features with pooled activity
before `post_pool`, producing one entangled identity. That is not the desired
EMG-Syn3 system.

The successor must preserve the complete activity identity `E^A` and inject a
separate carrier projection after the decoder's window projection:

```text
u_i = fc_in(x_i + E_i^A) + P(EMG-Syn3_i).
```

The first implementation uses `P = Linear(4, model_dim, bias=False)` only. Its
weights are initialized to exact zero, so all carrier-aware arms share the same
initial active model state and `Zero4` has an exact zero contribution. An MLP,
gate or FiLM adapter is not part of the first experiment; it is considered only
after a linear EMG-Syn3 carrier passes its content controls.

The carrier branch must follow the exact same unit correspondence, permutation
and whole-unit dropout mask as the neural/activity branch. Operationally, the
same per-unit mask is applied to the carrier input before `P`; a dropped unit
must not retain an unmasked carrier token. Otherwise the model could recover a
removed unit through `P(carrier)`, invalidating the activity-dropout and
identity-memory comparison. Receipts must prove equal masks and aligned source
unit order on both branches.

This injection has three important properties:

- `E^A` cannot be replaced by the carrier;
- C1 and CDM-A can change `E^A` without changing the carrier;
- carrier content can be ablated by replacing only the input to `P`.

The model receipt must separately hash `E^A`, the raw/normalized carrier,
`P(carrier)`, and the resulting prediction.

### 3.1 Reuse the existing analytic construction

Do not invent a learned target encoder. Reuse the existing M1 source-frozen
EMG-AFC4 loading, chronology, normalization-authority and control scaffold.
EMG-Syn3 changes only the source task-basis estimator and the explicit
carrier-injection site:

1. On source sessions only, validate the EMG signal view and fit the frozen
   rank-3 nonnegative synergy dictionary.
2. Resolve component scale and order from source data only.
3. For each target session and unit, use chronological support trials to fit neural firing rate as an affine function of the three EMG scores.
4. Store the four per-unit coefficients

```text
[w1, w2, w3, b]
```

as the M1 T4-like carrier.

The pre-existing PCA-AFC4 remains the matched basis-family reference. It is not
silently relabeled as EMG-Syn3. If EMG-Syn3 becomes positive, a matched PCA3
arm is required before attributing the gain specifically to nonnegative
synergies.

This is the correct analogue of T4 for M1: T4 is a circular two-dimensional task-basis special case, while M1 uses a source-defined three-dimensional EMG task basis. The target performs only closed-form/convex support fitting and no backpropagation.

### 3.2 Required controls

The complete mechanism claim eventually requires:

- `Zero4`: exact zero carrier in the same width and topology;
- `Syn3`: correctly attached EMG-Syn3;
- `RS4`: row-shuffled carrier, breaking unit attachment;
- `LS4`: support-label/EMG-row derangement, breaking the neural–EMG relationship;
- `B4`: baseline-rate coordinate only, when used as a mechanism diagnostic.

The primary content contrast is `Syn3 − Zero4`. The attachment and label-content claims require `Syn3 − RS4` and `Syn3 − LS4`; `Syn3` beating only the ordinary carrier-absent SPINT baseline is not sufficient. A matched `PCA3` arm is additionally required before attributing any gain specifically to nonnegative synergy rather than to the new injection topology.

### 3.3 Historical evidence and why this route is not already proven

The repository contains an authoritative one-fold, seed-42 compact EMG-AFC4 result:

```text
Full − Zero4 = −0.00652463 R²
```

This makes **static carrier content alone a weak prior**, not a new success. It must be disclosed before the new run. Reopening one carrier pilot is justified only by two pre-result changes to the carrier estimand:

1. an explicitly separated carrier projection rather than an entangled post-pool identity; and
2. bin-level nonnegative-synergy encoding rather than the old trial-mean signed-PCA encoding.

Carrier-aware CDM-A is a separately unmeasured deployment interaction, but it is not evidence that carrier content exists. CQC is even more downstream: it cannot justify reopening the carrier branch and remains forbidden until a static Syn3 content contrast passes its predeclared gate.

If neither interaction is positive, the M1 functional-carrier branch stops. We must not respond by sweeping carrier width, PCA rank, nonlinear heads, or target-specific bases.

## 4. Carrier-Quality Cycling (CQC)

### 4.1 The intervention

CQC trains one carrier-aware decoder against deterministic, chronological carrier estimates of different quality. It is the carrier-side analogue of activity-prefix cycling.

EMG-Syn3 uses valid 20-ms support bins, not one trial-mean row per trial. Even two trials may contain enough rows for the four-column design `[1,z1,z2,z3]` to be algebraically full rank. Therefore `M=2` must not be rejected merely because it contains fewer than four trials.

The real low-budget risk is insufficient **independent EMG-synergy coverage**: highly autocorrelated bins from two trials may occupy only a narrow region of the synergy field even when the numerical matrix rank is four. The source-only coverage audit must record valid bin count and duration, design rank/condition, smallest normalized Gram eigenvalue, synergy-coordinate dispersion, trial-stratified occupancy, and split-half coefficient stability.

Use the predeclared carrier cycle:

```text
(10, 6, 4)
```

The conservative `(10,6,4)` carrier cycle is retained for the first CQC experiment because it preserves multiple trial contexts and avoids making the first GPU cell depend on an unproven two-trial coverage regime—not because `M=2` is algebraically unidentifiable. Stage 0 must still report M2 as a diagnostic. A failed M4 or M6 coverage contract is a typed unavailable cell; it is not repaired after seeing R².

### 4.2 Two cycles must remain distinguishable

There are two different robustness interventions:

- **A-Cycle:** vary only the B3S calibration-activity prefix;
- **CQC:** vary only the EMG-AFC4 carrier estimate;

and one explicit combination:

- **Joint-Cycle:** vary both under a predeclared matched budget schedule.

They are not synonyms. The receipt must record both activity budget and carrier budget at every optimizer step. This prevents a positive result from being attributed to “calibration robustness” when only one input changed.

### 4.3 What CQC is intended to improve

CQC is not an online carrier updater. Its purpose is to train the decoder so that a noisy or low-budget but honest carrier does not create a large distribution shift. The carrier remains a closed-form support estimate at deployment.

The required mechanism readouts are:

- prediction sensitivity to carrier budget on source-grouped OOF sessions;
- `Syn3 − Zero4`, `Syn3 − RS4`, and `Syn3 − LS4` under the same checkpoint rule;
- cross-session R², not only source/train R²;
- carrier coefficient dispersion and condition number as diagnostics, never as checkpoint selectors.

## 5. CDM-A on M1

### 5.1 Definition

`CDM-A` is activity-only causal memory:

- initialize B3S activity identity from the selected support prefix;
- after each completed query trial, append only that trial's neural activity to a bounded FIFO;
- use the updated activity state starting from the next trial;
- do not update model weights;
- do not update the functional carrier in this experiment.

The last rule is essential. It cleanly separates activity memory from carrier updating and preserves the interpretation of CQC.

### 5.2 Why it remains worth testing

The latest carrier-absent result gives a small local held-out signal (`+0.0156` for T0 and `+0.0111` for C1), so CDM-A is not a blind hypothesis. However, its held-in penalty is large. The carrier-aware experiment must therefore report both surfaces and must not call CDM-A universally beneficial based on a held-out mean alone.

### 5.3 Future carrier memory is a separate experiment

Continuous carrier updating is deliberately excluded from the first FCM-M1 matrix. M1 support EMG is available, but query-time EMG availability, trial boundaries, and label legality must be specified independently. Any online EMG-carrier update would change the deployment information contract and needs its own controls and receipts.

## 6. Minimal experiment matrix

### Stage 0 — CPU constructibility and frozen-weight diagnostics

No GPU training starts before all items pass:

1. Build M1 EMG-Syn3 at `M=10` for every development fold; additionally audit `M=6,4,2` constructibility without making those lower-budget carriers part of the first GPU cell.
2. Record valid bins/duration, synergy coverage, rank, condition number, normalized-Gram spectrum, coefficient norm, source-normalized carrier SHA, and query-isolation evidence.
3. Materialize `Zero4`, `Syn3`, `RS4`, `LS4`, `B4`, and the matched legacy `PCA3` reference from the same source scope.
4. Prove that the existing T0/C1 checkpoint has no active carrier interface and therefore is not used for carrier scoring.
5. Reproduce the existing activity-only static/CDM-A score as a reference, without opening a new formal surface.
6. Report trial-stratified split-half correlation/stability of the fitted `W` coefficients and place it beside the old trial-mean PCA-AFC4 reliability evidence.

Stage 0 is a constructibility and disclosure gate, not a decoder-performance gate. Split-half instability or a poor comparison with trial-mean PCA-AFC4 must be visible in the receipt but does not by itself kill the predeclared one-fold GPU pilot; only a validity failure such as nonqualified signal view, nonfinite/undefined fit, source/target leakage or chronology violation blocks it. It should take hours, not days.

### Stage 1 — One-fold matched carrier-aware pilot

Use fold 0, seed 42, the strict source-only LOSO endpoint with support `[0,10)` and report query `[10,210)`, and the prior carrier-aware pilot's matched recipe: task-only loss, Adam `lr=1e-4`, weight decay 0, exactly 12 epochs, and fixed-last `epoch_011`. The left-out query is evaluated once after training and is never used for checkpoint selection. Do not use the failed 50-epoch horizon as default.

Train exactly these three first-stage arms:

| arm | carrier | activity training | purpose |
|---|---|---|---|
| `Z-Fix` | Zero4 | fixed M10 | topology-matched baseline |
| `S-Fix` | EMG-Syn3 M10 | fixed M10 | static Syn3 content effect |
| `S-Acyc` | EMG-Syn3 M10 | cycle `(10,5,2)` | activity-cycle effect with the carrier frozen |

Every trained checkpoint is evaluated under two deployment modes:

- `Static`: fixed M10 activity identity;
- `CDM-A`: causal activity FIFO, with the carrier fixed at the predeclared M10 support estimate.

This produces six first-stage cells without training six models. CQC is not silently included in this stage.

`S-Fix` is intentionally a bundled successor: relative to the historical PCA-AFC4 route it changes the nonnegative basis, bin-level encoding fit and independent carrier injection. Because `Z-Fix` matches the new topology and training contract, `S-Fix − Z-Fix` is a valid test of the **new carrier-content bundle**. It is not, by itself, evidence that NNMF is better than PCA or that bin-level fitting is better than trial-mean fitting. Matched PCA3 can isolate basis family; this matrix has no matched trial-mean Syn3 arm, so any bin-level superiority claim remains forbidden and the bundle must be disclosed even if positive.

### Stage 2 — Mechanism controls only for a positive carrier cell

If and only if `S-Fix` beats `Z-Fix`, train `S-RS4`, `S-LS4`, `S-B4`, and `S-PCA3` under the exact Stage-1 fold/seed/source split/initialization/optimizer/12-epoch/fixed-last contract. Their definitions and conditional launch rule must be present in the work order **before any Stage-1 result is opened**. Inference-time substitution is descriptive only and cannot satisfy the mechanism controls because `P` is learned jointly with the carrier distribution.

The carrier claim survives only if Syn3 beats all relevant matched-training mechanism controls. A Syn3 gain that disappears against B4 is a baseline-rate result, not an EMG-synergy result. A Syn3 gain that does not beat matched PCA3 is a functional-carrier/injection result, not an NNMF-specific result.

### Stage 3 — Conditional Carrier-Quality Cycling

CQC starts only if `S-Fix − Z-Fix` passes the predeclared static-content gate. A positive Syn3×CDM-A interaction with nonpositive static content does not open CQC; it is handled under Outcome C as an activity-memory result. Add a source-grouped 2×2 cycle factorial around the frozen Syn3 system:

| arm | activity budget during training | carrier budget during training |
|---|---|---|
| `S-Fix` | fixed M10 | fixed M10 |
| `S-Acyc` | `(10,5,2)` | fixed M10 |
| `S-Ccyc` | fixed M10 | `(10,6,4)` |
| `S-Jcyc` | `(10,5,2)` | `(10,6,4)` |

`S-Fix` and `S-Acyc` are reused from Stage 1; only two additional models are trained. If `S-Fix − Z-Fix` fails, `S-Acyc` is descriptive only: it cannot support the statement “C1 works when a carrier is present,” cannot open Stage 3, and cannot rescue the carrier route. If static content passes and `S-Acyc` is positive, add `Z-Acyc` as the predeclared conditional zero-carrier topology control before claiming a carrier-by-activity-cycle interaction.

### Stage 4 — Expansion

Expand to all development folds only if at least one of the following predeclared effects is positive and practically nontrivial on the one-fold pilot:

- `S-Fix − Z-Fix`: carrier content;
- `S-Acyc − S-Fix`: activity-cycle effect with fixed carrier;
- `S-Ccyc − S-Fix`: carrier-quality cycling;
- the registered 2×2 interaction in `S-Jcyc`;
- `CDM-A − Static` within the same carrier-aware arm.

The one-fold static-content gate remains the previously registered Version-B threshold: `S-Fix − Z-Fix >= +0.03 R²`, with no validity failure. It is not relaxed merely because the basis and injection changed; no new source-only MDE has been sealed that would justify a lower threshold. At the all-development-fold stage, require equal-session mean `>= +0.01`, at least three of four fold targets positive, and no session with `delta < −0.05`. These are practical screening rules, not significance claims. Full-fold reporting must include every paired session delta, equal-session mean, and dispersion.

Only one frozen winner may proceed to a formal/official M1 evaluation. The official surface must never select the arm, epoch, carrier basis, cycle, or CDM policy.

## 7. Attribution table

The following contrasts answer different questions and must not be collapsed:

| contrast | scientific question |
|---|---|
| `S-Fix − Z-Fix` | Does EMG-Syn3 carrier content help? |
| `S-Acyc − S-Fix` | Does activity cycling help when Syn3 is fixed? |
| `S-Ccyc − S-Fix` | Does carrier-quality cycling help without activity cycling? |
| registered `S-Jcyc` interaction | Are activity and carrier quality robustness complementary? |
| `CDM-A − Static` | Does causal activity memory help at deployment? |
| interaction of `CDM-A` with `Syn3` | Does a functional carrier make activity memory more useful or safer? |
| `Syn3 − RS4/LS4/B4/PCA3` | Is the gain attributable to attached nonnegative EMG synergy content? |

This table is the core of the paper story. A system-level win without these contrasts is useful engineering evidence but not a functional-carrier mechanism claim.

### 7.1 Comparator hierarchy

Comparators serve different roles and must remain on matched scoring surfaces:

1. `Z-Fix` is the primary topology- and training-matched attribution control.
2. Matched `PCA3`, `RS4`, `LS4`, and `B4` identify the carrier mechanism.
3. The latest carrier-absent T0/C1/CDM-A cells are descriptive activity-path references because their model topology and training recipe differ.
4. Original SPINT and the historical official M1 Original/T4/D4 values are benchmark context only. They may be compared numerically only after the frozen FCM-M1 candidate is evaluated on the identical official surface.
5. M10 DirectRidge is a label-cost comparator and must disclose that EMG-Syn3 uses dense 20-ms EMG observations inside ten trials.

## 8. Expected outcomes and stop rules

### Outcome A — carrier and memory are complementary

If `S-Fix > Z-Fix` and `CDM-A > Static`, with a positive interaction, the paper story is:

> A source-defined EMG functional carrier supplies stable per-unit task identity, while causal activity memory tracks session state online; the two information channels are complementary.

### Outcome B — CQC is the main gain

If static carrier content is positive and `S-Ccyc` or `S-Jcyc` adds a further gain, the story is:

> The carrier is useful only when the decoder is trained against realistic support-estimation quality; robustness to the estimator, rather than a richer carrier head, is the missing component.

### Outcome C — only CDM-A helps

If the static Syn3 content gate fails but CDM-A remains positive—including a descriptive positive Syn3×CDM-A interaction—retain CDM-A as the M1 contribution and close EMG-Syn3/CQC. Do not relabel an activity-memory-conditioned effect as carrier-content evidence.

### Outcome D — all corrected cells are null

Stop the M1 calibration adaptation branch. Report the boundary honestly: M1 identity reliance is strong, but the tested analytic carrier and bounded causal activity memory do not close the cross-session gap under the available support budget.

## 9. M2 transfer after M1

M2 already has a natural circular T4 carrier. If the M1 matrix identifies a positive general mechanism, transfer only that mechanism to M2:

- positive CQC → cycle the quality of the ordinary M2 T4 estimate while keeping the activity cycle separately controlled;
- positive CDM-A → use the already reviewed activity-memory state machine;
- positive carrier-memory interaction → test the same two-factor decomposition, not a monolithic combined arm.

M2 should retain its official baselines and M4/M10/M30 budget surfaces. M1 EMG-synergy details do not transfer; only the abstract functional-carrier contract transfers.

## 10. Resource and isolation rules

1. No new DANDI P1/P2/PACD scoring or successor job is launched. A nearly completed existing DANDI producer may terminalize for archival integrity, after which the route is frozen.
2. Stage 0 is CPU-only and must not open formal M1 data.
3. Stage 1 uses one GPU and one fold. It must not share a result root, CUDA device, CPU affinity, log, checkpoint directory, or source authority with another live job.
4. Another GPU job is treated as immutable external state: no signals, restarts, affinity changes, priority changes, or root writes.
5. A short smoke checks only legality, finite gradients, carrier consumption, and runtime. It does not select the scientific arm.
6. Do not extend training to 50 epochs merely because loss continues to decline. The completed 50-epoch run shows that lower source loss can coincide with worse cross-session transfer.

## 11. Immediate execution order

1. Freeze the latest M1 T0/C1 result under the corrected label: **carrier-absent activity experiment**.
2. Implement and audit source-frozen EMG-Syn3 at M10; audit M6/M4 as future-CQC constructibility and M2 as a coverage diagnostic without an automatic algebraic rejection.
3. Build a carrier-aware one-fold smoke that proves the `[N,4]` tensor changes predictions and receives gradients where intended.
4. Run the three Stage-1 arms with matched initialization, source data, optimizer steps, checkpoint rule, and seed.
5. Score Static and CDM-A from every checkpoint; apply the attribution table before any expansion.
6. Run mechanism controls and the two additional CQC models only if their upstream gates pass.
7. Expand only the surviving mechanism to all M1 development folds.
8. Freeze one candidate, then decide whether a single official M1 evaluation is justified.
9. Transfer only the surviving abstract mechanism to M2.

## 12. Paper-safe current wording

The following wording is accurate now:

> Our initial M1 calibration-prefix experiment varied only the activity-derived identity path and did not instantiate a functional carrier. Activity-prefix cycling did not improve the evaluated 50-epoch model, whereas a causal activity-only memory produced a small positive cross-session development increment. We therefore treat those results as an activity-path ablation, not as evidence against functional carriers. The corrected evaluation separately tests an EMG-derived analytic carrier, carrier-quality cycling, and causal activity memory under a matched carrier-aware topology.

Do not claim an M1 functional-carrier improvement until `Syn3` passes the matched Zero4 and mechanism controls on a predeclared cross-session surface.

## 13. Implementation ownership and review boundary

The successor should be additive. It must not edit the completed M1 T0/C1
receipts, checkpoints, scorers or result roots. A recommended route-local layout
is:

```text
tfpd_exploration/src/m1_emg_syn3_fcm_v1/
  plan.py               # frozen fold, budgets, recipe, controls and roots
  syn3.py               # source dictionary, NNLS transform and ridge carrier
  model.py              # independent post-fc_in carrier projection seam
  data.py               # strict source-only support/query adapter
  training.py           # Z-Fix / S-Fix / S-Acyc dispatch
  score.py              # Static / CDM-A paired scorer
  receipts.py           # typed source/carrier/model/score evidence
tfpd_exploration/scripts/run_m1_emg_syn3_stage0.py
tfpd_exploration/scripts/run_m1_emg_syn3_pilot.py
tfpd_exploration/tests/test_m1_emg_syn3_fcm_v1.py
```

Shared production modules are changed only if an independent seam audit proves
that additive composition cannot implement the exact injection law. Any shared
seam must be backward-compatible and must retain byte-semantic behavior for the
ordinary carrier-absent model.

The following tests are required before a GPU capability can be issued:

1. signed input rejects NNMF; qualified nonnegative input passes;
2. source dictionary is deterministic and target data cannot alter it;
3. NNLS activations are finite/nonnegative and reconstruct under a frozen rule;
4. support bins are exact `[0,10)`, time aligned, and query-disjoint;
5. Zero4 yields exact zero `P(carrier)` and identical initialized predictions;
6. Syn3 changes predictions after a controlled nonzero projection;
7. carrier and activity share exact unit order, permutation and dropout mask;
8. C1 changes only the activity prefix; its carrier SHA remains constant;
9. CDM-A changes only next-trial activity state; model/carrier states remain exact;
10. no target optimizer/backward/update occurs;
11. fixed-last epoch selection cannot read the left-out query;
12. dry CLI imports no Torch, opens no data, initializes no CUDA and writes no result root.

The Stage-0 review boundary is reached only when the CPU audit publishes the
source signal-view authority, dictionary/transform/carrier digests, all-fold
constructibility table, and explicit pass/fail decision without a decoder R².
Only a separately reviewed pilot work order may then authorize one GPU.
