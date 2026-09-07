# Work Order: H1 Calibration-Profile FiLM V1

Date: 2026-09-04  
Status: `AUTHORIZED_SOURCE_ONLY_GPU0_ONCE`

Implement and execute the frozen design in
`DESIGN_H1_CALIBRATION_PROFILE_FILM_V1_20260904.md` exactly once on logical
GPU0.  Do not query, signal, allocate on, or otherwise touch GPU1.

## Required execution

- Five H1 held-in date-LODO folds in fixed order.
- Load each sealed epoch-49 C1 checkpoint through the reviewed C1 loader.
- Load the matching sealed LP-R3 branch checkpoint from
  `h1_support_resampled_postpool_v1`.
- Construct the H1 M3 task profile exactly as specified.
- Train only independent, initially identical EP-FILM and LP-FILM modules.
- Fixed seed 42, 12 epochs, Adam `lr=3e-4`, `weight_decay=0`, batch 32,
  source query stride 4.
- Use the exact same M3 support choice for both FiLM arms at every paired step.
- Seal and strict-reload the two FiLM state dictionaries before opening the
  outer date.
- Score EP-ZERO, EP-FILM, LP-ZERO, and LP-FILM on the exact prior target rows.
- Reproduce the sealed EP-ZERO and LP-ZERO prediction SHA/R2 anchors exactly.
- Publish the ten FiLM checkpoints (two per fold), five fold receipts, score, and terminal
  with SHA sidecars and mode 0444.

## Forbidden

- formal held-out or EvalAI access;
- target gradients, optimizer steps, or model updates;
- checkpoint, epoch, learning-rate, seed, profile-mask, threshold, or pooling
  selection after seeing outer-date results;
- changes to C1, LP-R3, the scheduled 08:00 LP-R3 task, or their immutable
  result roots;
- any pre-fusion control beyond the two declared early arms;
- any automatic all-source training, Docker build, or submission.

## Launch gates

Before the attempt:

1. design and work-order bytes are hashed;
2. the sealed LP-R3 score and terminal bodies match their literal SHA256;
3. the canonical result root does not exist;
4. `CUDA_VISIBLE_DEVICES=0`, exactly one CUDA device is visible, and its UUID
   is `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`;
5. the 08:00 LP-R3 systemd timer remains active and is not modified.

The run must fail closed with an immutable failure receipt on a Python
exception.  A failed canonical root is never overwritten or retried; repair is
an additive successor.

## Decision

Use only the gates and labels in Design section 7.  Report both FiLM increments,
the zero-pooling effect, the FiLM-pooling effect, and their interaction.  Do not
describe M2 as late pooling and do not call a late-only result pooling-invariant.
