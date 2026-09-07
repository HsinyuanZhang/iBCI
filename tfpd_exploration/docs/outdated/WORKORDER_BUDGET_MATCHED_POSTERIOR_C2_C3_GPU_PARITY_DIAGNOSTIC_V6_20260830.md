# C2/C3 GPU parity diagnostic V6

## Purpose

Measure the exact CPU/GPU difference that caused the V5 parity failure before
authorizing any accelerated full matrix. This route is diagnostic only. It
runs the first within-session C2/M4 cell, once on CPU and twice on GPU1, then
stops and publishes the complete comparison.

## Immutable predecessor

V6 must descriptor-validate the exact four-leaf V5 failed result before it
reserves its own root. V5 failed during parity, before the 252-cell matrix,
without target optimization, backward, or update.

## Numerical contract

- CPU logical batch: 32.
- GPU candidates: 1024, 512, 128.
- TF32 disabled; deterministic algorithms enabled.
- Two GPU predictions must be measured separately.
- Persist prediction SHA-256, repeated-GPU bitwise equality, maximum absolute
  CPU/GPU prediction difference, CPU R2, GPU R2, absolute R2 difference,
  batch fallback history, peak CUDA memory, and wall time.
- The historical strict tolerances remain 2e-6 prediction and 2e-7 R2. A
  strict failure is a valid V6 diagnostic terminal, not a reason to discard
  the measured evidence.
- V6 never runs the full matrix and never changes target/model state.

## Drift and restart policy

Execution and review closure are disjoint.

- Model code, parser/data selection, preprocessing, inference, metric, parity
  computation, runtime launcher, or numerical policy drift is execution drift
  and fails closed.
- Work-order text, tests, comments, and review checklists are review-only.
  Their drift is recorded as `ACCEPTED_NON_NUMERIC_DRIFT`, with
  `numerical_acceptance_affected=false` and `restart_required=false`.
- A review-only edit must never stop or restart numerical execution.
- If a future audit finds a path was unnecessarily included in execution
  closure, a successor must reclassify that path before launch. It must not
  relaunch an identical numerical job merely to accept non-numeric drift.

## Successor decision

V6 does not choose a new tolerance. After V6 terminalizes, a separately named
engineering-equivalence successor may predeclare a relaxed tolerance only if
GPU repeats are bitwise equal and the observed CPU/GPU R2 difference is far
smaller than every scientific decision margin. Otherwise GPU acceleration is
not authorized.

