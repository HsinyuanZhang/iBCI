# B2 — M2 source-training carrier-reliance corruption contract

**Date:** 2026-08-13
**Status:** root-audited design/CPU-helper candidate v2; **no training, inference, formal test, or launch is authorized**
**Scope:** one matched FALCON M2 internal-development experiment; no network fusion path

## 1. Audit disposition

Historical P3 does **not** answer this experiment.

- `carrier_perf_program/src/carrier_perf/p3_corruption.py` says that it defines only a sampler and
  "does not train". Its M grid is sampled into a receipt but never refits a carrier at that M.
- `configs/experiment/p3_carrier_corruption_m2_t4.yaml` places `corruption_config` under `data`, while
  `FalconDataModule.__init__` has no `corruption_config` argument. No streaming runtime imports the
  isolated P3 sampler.
- `carrier_perf_program/results/p3_stage_a_baseline_receipt_v1/receipt.json` only verifies historical
  SUA RS4/LS4 aggregates. It records `no_gpu=true` and `no_inference_rerun=true`.
- No `carrier_corruption_aug_v1` training aggregate exists in the workspace.
- The later unlaunched `b2_activity_path_dropout_p025_m2_t4.yaml` zeros pooled activity and therefore
  forces **more** carrier reliance. The present question is whether source training can reduce
  **blind** reliance on wrong carrier content. That scaffold is not a substitute.

Historical SUA RS4/LS4 below Z4 and any B1 result are hypothesis-generating only. They are not an M2
effect estimate, a threshold source, an arm selector, or an execution gate.

## 2. Claim and independence from B1

The claim under test is:

> Source-only exposure to wrong or absent standardized T4 content teaches the unchanged B3S consumer
> to fall back toward its activity-derived identity when carrier content is wrong, without erasing the
> advantage of correct T4 content.

B2's mechanism is **logically independent of B1**, but its execution is operationally downstream of
B1. B1 asks whether identity distillation selectively suppresses carrier content; B2 changes the
source input distribution and asks whether the consumer blindly follows bad content. A B1 null does
not kill B2. However, B2 cannot modify the now hash-locked B1 streaming runtime, and a positive B1
would change which loss is the selected mainline substrate. Therefore B2 remains CPU-contract-only
until B1 closes. At that point its loss is frozen once to the selected B1 substrate; it is never
chosen using a B2 target score. The current candidate is
`task_plus_y_plus_E, lambda_y=1.0, lambda_E=0.1`; if B1 selects `task_plus_y`, this document must be
versioned before any B2 preflight rather than silently relabelling a result.

## 3. Frozen intervention

Architecture, decoder, teacher, optimizer, support/query policy, and tensor width are identical in the
two training arms. The only change is made after the ordinary source-session T4 fit and source-only
z-score, immediately before the existing B3S side-feature consumer.

| training arm | per-source-session, per-epoch standardized carrier |
|---|---|
| `clean_t4` | T4 with probability 1.0 |
| `reliance_corruption` | T4 0.50; complete-row-deranged RS4 0.25; all-zero Z4 0.25 |

Carrier identity must not vary across query windows from the same recording. For each run seed and
source session, the 12 logical epochs receive one deterministic permutation of the exact multiset
`{T4 x 6, RS4 x 3, Z4 x 3}`. Every batch and every example from that source session therefore sees
the same carrier state throughout an epoch. The schedule key is only
`(run seed, source session, logical epoch)`; it is independent of batch order, worker count, rank,
resume boundary, and example index. RS4 uses one complete row derangement fixed by
`(run seed, source session)` across the run, permutes all four coordinates together, and leaves
neural activity and calibration tensors in their original channel order. The exact schedule and
row-permutation hashes are logged but never select a model.

The 6/3/3 counts are design constants: half of session-epochs preserve the exact deployment input,
while the corrupted half is split equally between confidently wrong content and omission. This
balanced schedule replaces independent per-example probability draws; it prevents accidental
frequency imbalance and prevents a session identity from changing between windows in one epoch.
There is no probability sweep.

LS4 is intentionally absent from source training. Its forward-only mechanism score therefore checks
whether any reliance change transfers to a distinct, unseen wrong-content construction rather than
only adapting to the exposed row-permutation operation. Source-only prelaunch checks may verify
constructibility and realized draw frequencies, but no source or target accuracy selects these
probabilities.

`M=33` is fixed for activity calibration and T4 in both arms. **There is no M sampling.** Small-M
sampling would mix reliance calibration with estimator-noise robustness (the separate B3 question).
There is also no Gaussian carrier noise and no activity-path dropout. Those would add mechanisms to
the one experiment.

Training corruption is disabled in validation, test, export, and deployment. The deployed model and
input remain ordinary T4; no fusion, confidence flag, extra input, or alternative network path is
added.

## 4. Matched forward diagnostics

Every stored checkpoint is forward-scored on the same left-out M2 development session and identical
query windows under four carrier views. No weight update occurs.

| view | construction |
|---|---|
| `T4` | ordinary correct `[a,c,m,b]`, standardized by source-training-session statistics |
| `Z4` | ordinary T4 fit and same normalizer, then `zeros_like` |
| `RS4` | ordinary standardized T4 followed by a fixed complete row derangement; neural/calibration order unchanged |
| `LS4` | same M33 support, but finite calibration target directions are exactly deranged before the T4 fit; same source normalizer |

The RS4/LS4 evaluation schedule is fixed by session name and control seed `20260813`, and is identical
across training arms, training seeds, folds, and epochs. LS4 preserves the finite direction-label
multiset, keeps centre/missing trials missing, changes every finite label, and fails closed when a
complete derangement is impossible. A fold with ineligible LS4 is not silently reduced to three views;
the whole matrix stops before launch.

T4 is the deployment outcome. Z4/RS4/LS4 are mechanism diagnostics, not deployable alternatives.

## 5. Primary estimand: reliance rather than generic regularization

Let `R[a,f,s,v]` be the unweighted mean validation R² across logical epochs 5--12 for training arm
`a`, fold/session `f`, seed `s`, and evaluation view `v`.

For wrong-content view `w` in `{RS4, LS4}`:

```text
theta_w(f,s) = [R(corrupt,f,s,w) - R(corrupt,f,s,Z4)]
             - [R(clean,  f,s,w) - R(clean,  f,s,Z4)]

theta_reliance(f,s) = 0.5 * [theta_RS4(f,s) + theta_LS4(f,s)]
```

Equivalently, `theta_reliance` is the clean-minus-corrupt change in the average penalty
`0.5*((Z4-RS4)+(Z4-LS4))`. Positive values mean wrong content moved toward omission rather than
continuing to drag performance below omission.

This is the primary because a generic `+c` lift to T4, Z4, RS4, and LS4 cancels exactly. Therefore a
mean improvement without a positive reliance interaction must be reported as generic regularization,
not calibrated reliance.

Two anti-triviality outcomes are mandatory:

```text
content_retained(f,s) = R(corrupt,f,s,T4) - R(corrupt,f,s,Z4)
deployment_delta(f,s) = R(corrupt,f,s,T4) - R(clean,f,s,T4)
```

If wrong carriers become harmless only because the model ignores all carrier content, B2 fails.

## 6. Frozen cells, epochs, and cost

| item | frozen value |
|---|---|
| model | ordinary B3S, `side_dim=4`, no fusion addition |
| task | FALCON M2 internal development |
| folds | LOSO `{4,5,6}` = `2020-10-27 Run1`, `2020-10-27 Run2`, `2020-10-28 Run1` |
| routing seed | `{42}` across all three folds |
| confirmation seeds | `{43,44}` across the same three folds, only after the frozen routing gate |
| training arms | `{clean_t4, reliance_corruption}` |
| epochs | 12 fixed, no early stopping or validation selection |
| reported cell score | unweighted mean of forward scores at logical epochs 5--12 |
| Stage P training cells | `2 x 3 folds x 1 seed = 6` |
| Stage F training cells | `2 x 3 folds x 2 seeds = 12`, only after Stage P passes |
| maximum training cells | `18` |
| run-level diagnostic bundles | `18 x 4 views = 72` |
| checkpoint-session forwards | `18 x 4 x 8 epochs = 576` |

An on-disk, same-shape M33 B3S/T4 source-cost program provides a planning ruler, not execution
evidence for B2. Its 12-epoch fold-4/5/6 seed-42 cells recorded `38,832/37,908/36,756` training batches,
`2.7241e15/2.6592e15/2.5784e15` source-training MAC, and `22.70/21.81/19.61` minutes wall time. If all
18 B2 cells are fresh and exhibit the same runtime, the matrix is approximately `4.777e16` training
MAC plus `3.474e14` in-training-validation MAC, or about **6.41 serial device-hours** before artifact
and checkpoint-load overhead. This is an order-of-magnitude budget, not a runtime guarantee; B2's
future preflight must bind hardware and remeasure one default-off cell. Historical full-query batched
forward timings imply only about five minutes of kernel time for the 576 diagnostic forwards, but
repeated checkpoint loading and data setup are excluded from that figure.

Folds 4--6 avoid B1's frozen folds 0--3, but they are still M2 development sessions, not a new formal
test. Existing clean checkpoints may replace a clean cell only after exact proof of teacher bytes,
source split, query windows, source normalizer, seed, fold, M33, objective, optimizer, 12-epoch
retention, and source identity. No reuse is assumed in the cost.

Stage P is a predeclared routing screen, not part of the confirmatory endpoint. It runs both training
arms for seed 42 on all three fixed folds. If its frozen mechanism, content-retention, and deployment
gates fail, B2 stops after six cells. If they pass, Stage F runs seeds 43 and 44 on those same three
folds. The confirmatory endpoint uses Stage F only; the combined three-seed matrix is labelled
descriptive sensitivity and never receives the terminal gate. There is no staged probability or fold
search, and no partial result may replace a fold or change the intervention.

## 7. Inference and frozen gates

The generalization unit is the left-out development session, not a query window or checkpoint. Seeds
measure optimization variability and epochs 5--12 define one run-level score; neither is an
independent session replicate.

Stage P reports its three seed-42 session interactions and applies the same practical/anti-triviality
conditions below only to decide whether Stage F is worth running. Stage F reports six paired values,
three session means, two seed means, and an unweighted grand mean. A crossed session/seed bootstrap
interval is descriptive only. No `p <= 0.05` gate is used: three sessions or two confirmation seeds
cannot support the commonly misused exact two-sided sign/Wilcoxon gate. Stage-P scores are never
pooled into confirmatory inference.

Launch Stage F, and then call the mechanism confirmatory-positive, only if the relevant stage satisfies
all conditions:

1. grand-mean `theta_reliance >= +0.03`;
2. grand-mean `theta_RS4 > 0` and `theta_LS4 > 0`;
3. all three session means and all three seed means of `theta_reliance` are positive;
4. augmented `T4-Z4 >= +0.03` and all three session means are positive;
5. grand-mean deployment `T4_corrupt-T4_clean >= -0.03`.

The `+0.03` practical floor and symmetric `-0.03` deployment margin are prospective engineering
margins frozen before any B2 score. They are not estimated from P3 or B1 accuracy results.

Interpretation is fail-closed:

- mechanism gate fails, even if T4 rises: **generic regularization or null; stop**;
- mechanism passes but correct content disappears or deployment is inferior: **trivial carrier
  suppression/harm; stop**;
- Stage P passes: **launch the two predeclared confirmation seeds**, with no other change;
- Stage F passes: **development GO for one separately contracted external confirmation**, not a
  formal claim;
- no post-hoc probability, activity-corruption, Gaussian-noise, M-grid, loss, epoch, seed, or fold rescue.

## 8. Label and isolation ledger

| phase | neural activity | direction/behaviour labels | permitted use |
|---|---|---|---|
| source training sessions | source calibration and query activity | calibration target direction for T4; source query velocity for task loss; inherited teacher targets | gradients under the frozen objective; source-only T4 normalizer |
| left-out M2 development session support | first 33 calibration trials | support target direction only | fit T4/LS4 diagnostics; no weight update and no normalizer/probability/epoch selection |
| left-out M2 development query | fixed query activity | query velocity | scoring only after predictions are fixed |
| inherited teacher | existing M2 teacher provenance | may historically include the development session | identical in both arms; disclosed confound, so inference is internal-development only |
| formal/EvalAI held-out | none | none | excluded; `include_heldout_in_fit=false`, `include_heldout_in_test=false`; no submission |

No target query label may affect the corruption draw, carrier/normalizer fit, gradient, checkpoint,
epoch, fold, probability, M, or gate definition.

## 9. Implementation and prelaunch gates

The isolated default-off helpers are:

- `sua_exploration/b2_carrier_reliance/corruption.py`: exact T4/RS4/Z4 training transforms and LS4
  finite-label derangement;
- `sua_exploration/b2_carrier_reliance/contract.py`: exact 18-cell/72-score matrix and preregistered
  aggregate;
- `sua_exploration/b2_carrier_reliance/tests/test_contract.py`: exact-null/control and estimand tests.

They contain no trainer, launcher, data reader, artifact writer, or runtime hook. This keeps the
B1-bound streaming tree unchanged.

Runtime implementation remains **NO-GO** until an independent prelaunch review proves all of:

1. default-off `p=0` path is bitwise output/RNG identical and checkpoint-compatible;
2. corruption occurs after source-only standardization and only during source training;
3. all four views share query windows, activity/calibration tensors, normalizer, and checkpoint;
4. RS4/LS4 are complete nonidentity interventions and Z4 is post-normalizer matched;
5. the per-session/per-epoch 6/3/3 schedule and fixed per-session RS4 mapping are deterministic across
   resume, rank, batch order, and data-worker counts; no carrier state varies across examples/windows
   from one session within an epoch;
6. the source manifest does not overwrite or silently invalidate A2/A11/B1 bound artifacts;
7. Hydra composition, CPU real-format batch construction, epoch retention, and no-heldout guards pass;
8. no existing P3 result satisfying this exact estimand is discovered.

This contract itself is not an official preflight or receipt and authorizes no execution.

## 10. Root-audit correction record

Candidate v1 proposed independent per-example corruption keyed by batch and example index and one
unstaged 18-cell matrix. That construction was rejected before runtime implementation: T4 is a
session identity, so changing it between windows would train on an unphysical high-frequency noise
process, and an all-at-once matrix would ignore the program's early-kill discipline. Candidate v2
uses an exact session-epoch schedule and a six-cell routing stage followed by an independently scored
12-cell confirmation. No v1 GPU result exists.
