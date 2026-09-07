# Handoff: Teacher-Free Population-Robustness Round

Date: 2026-08-17

## Goal

Improve external subject transfer. Performance is primary; mechanism ablations must stay small.

Test whether the main missing ingredients in Arm A are stochastic unit-subset training and the
teacher's 64-head attention geometry. Do not use a teacher checkpoint or run another pretraining
stage in this round.

## Correct provenance

- A2 uses the 000128/sub-Jenkins teacher checkpoint at epoch 83, SHA `9b4a94ca...`.
- TFAP used the correct teacher dataset, but did not reproduce the teacher training regime.
- A2's B3S identity encoder is fresh. The inherited component is the SPINT decoder.
- The active deployment decoder is not larger than Arm A. The important known differences are:
  `num_heads=64` versus `2`, and `dynamic_dropout=true` versus `false`.
- The 000953/epoch-34 M2 run is a different lineage and must not be cited as the A2 teacher.

## Existing baseline

Use sealed Arm A as the only baseline:

- training receipt: `results/admission_arms_v1/armA_direct_t4_48/terminal_receipt.json`
  (SHA `6cf7d317...`)
- final-four SWA: `results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt`
  (SHA `920eb4c9...`)
- within R2: `0.516273`
- external R2: `0.161050`

## New cells

Launch these two seed-42 cells first, in parallel when two GPUs are available.

### Cell D: dynamic dropout only

- `num_heads=2`
- `dynamic_dropout=true`
- `dynamic_dropout_low=0.0`
- `dynamic_dropout_high=1.0`

### Cell DH: performance arm

- `num_heads=64`
- `dynamic_dropout=true`
- `dynamic_dropout_low=0.0`
- `dynamic_dropout_high=1.0`

Both cells must otherwise be exact Arm A replicas:

- standard initialization; no teacher weights, logits, feature matching, or distillation
- same canonical initial tensor state as Arm A, loaded with `strict=True`
- prove all initialized parameter tensors are bitwise equal before training
- strict-27 source roster, M30, normalized T4, task-only loss
- 48 epochs, same batch order, optimizer, warmup-plus-cosine schedule, and seed 42
- same epoch 44--47 final-four SWA
- no target-based selection, no formal/organizer-held data

Implement this additively in new route-owned files. Do not edit the existing sealed Arm A runner,
`spintshape_module.py`, shared `spint.py` / `streaming_spint.py`, or any existing artifact.

Do not launch the 64-head/no-dynamic cell yet. Return D and DH results first. That cell is a
conditional mechanism ablation, not the current performance priority.

## Required implementation checks

- Head count and dynamic-dropout configuration must be explicit in launch and terminal receipts.
- The 2-head and 64-head models must have identical state-dict keys/shapes and equal active
  parameter counts.
- Changing head count must not consume extra RNG or alter initial tensor bytes.
- Dynamic dropout must be active only during training and disabled during validation/scoring.
- Use the existing implementation exactly for this first test: one probability sampled uniformly
  from `[0,1)` per forward, followed by PyTorch dropout on the unit mask.
- Instrument the probability already sampled by the model. Do not draw a second random number for
  logging.

## Required diagnostics

Record per epoch:

- sampled dropout-probability summary
- realized retained-unit fraction and retained-unit count
- all-zero-population example count
- token norm before and after dropout
- `fc_in` input norm, global gradient norm, and finiteness
- training loss and the existing source-side validation metrics

For DH also record per-head attention entropy and a head-diversity summary. Diagnostics must not
change the forward result or RNG sequence.

## Scoring and decision

After both source runs are terminal, score once with the existing matched scorer on the same six
within and fifteen external development sessions. Use native T4 as the primary result. Keep zero,
wrong-pair, and destroyed-activity modes as diagnostics.

For each new cell report paired deltas versus Arm A:

- external mean, median, 15 session deltas, positive-session count, and fixed-seed bootstrap CI
- within mean, median, six session deltas, and positive-session count

A cell is a performance candidate only if all three conditions hold:

- external mean delta versus Arm A is at least `+0.03`
- at least `10/15` external session deltas are positive
- within mean delta versus Arm A is at least `-0.03`

Do not compare directly against A2's `0.341` as an adoption gate until A2 is rescored under the
same deployable-model estimator. It may be shown only as a historical reference.

## After the result

- If D passes and DH does not materially exceed D, stop: dynamic dropout is the surviving method.
- If DH passes and its external mean exceeds D by at least `+0.01`, propose the
  64-head/no-dynamic cell to isolate the head contribution; do not launch it automatically.
- If both fail, stop this round. The next hypothesis is random calibration / identity-token
  variability or decoder pretraining, not model width.

If dynamic dropout is positive, do not claim that an old SPINT switch is the paper innovation.
Use it as evidence for a new teacher-free method: explicit population thinning plus prediction
consistency across nested unit subsets, with subset size separated from inverse-dropout gain.

## Do not do

- no teacher checkpoint initialization
- no 000128 or 000953 pretraining
- no width scaling
- no extra seeds before the seed-42 decision
- no target-guided hyperparameter or checkpoint selection
- no automatic 64-head/no-dynamic launch
- no claim that A2 inherited a B3S identity encoder
