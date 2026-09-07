# DANDI 000688 CP-FiLM + Post-Pool Co-Adaptation V1

Status: `FROZEN_SOURCE_ONLY_SUCCESSOR__GPU0_AUTHORIZED__FORMAL_TEST_FORBIDDEN`

## Question

The frozen-head CP-FiLM screen left roughly half of the low/high-speed profile
variance unexplained by `[T4, frozen pre-pool activity mean]`, but the real
profile did not beat an empty-profile capacity control. Does the profile become
useful when the small B3S `post_pool` MLP is allowed to co-adapt with FiLM while
the activity `pre_pool` and full decoder remain frozen?

This is an operator-capacity successor. It is not the exact M2 recipe: the M2
positive route froze its base path. A positive result here would show that the
DANDI profile needs downstream co-adaptation, not that frozen-path CP-FiLM
transfers unchanged.

## Frozen substrate and scope

- DANDI 000688 `sub-C`, center-out, sorted SUA.
- Exact 27 train / 6 validation / 6 formal-test-name strict manifest.
- Parent checkpoints: seed-matched final epoch M30/T4@30 B3S mainline.
- Activity support: first 30 chronological rewarded trials.
- T4 support: the same first 30 rewarded trials.
- Query windows: rewarded trials 50 onward only.
- Profile: the already frozen M10/M30 low/high-speed robust-z profile, active
  columns `[high-low mean, logmean difference]` and mask `[1,1,0,0]`.
- No target update, no continual memory, no formal-test path resolution, and no
  EvalAI submission.

The complete three-seed predecessor result is
`RESULT_DANDI_000688_CALIBRATION_PROFILE_FILM_V1_20260904.md`. Its grand mean
deltas were CP10 `+0.000179`, CP30 `+0.000887`, and EMPTY `+0.003680`; hence V1
is a profile-semantic null.

## Operator and trainable surface

For each arm, copy the seed-matched native `post_pool` MLP exactly and attach a
separate zero-initialized FiLM head:

```
h = mean_i pre_pool(activity_i)                 # frozen
[gamma, beta] = film([T4, profile])             # trainable
h_film = (1 + gamma) * h + beta
identity = post_pool_arm([h_film, T4])           # trainable copy
prediction = frozen_decoder(query, identity)
```

The frozen native branch retains its original post-pool MLP. The four
experimental arms are:

- `CP10`: real first-10 profile;
- `CP30`: real first-30 profile;
- `SHUFFLE10`: frozen nonidentity unit-row permutation of the M10 profile;
- `EMPTY`: exact positive-zero profile with T4 still present in the FiLM
  context.

Each arm trains exactly its 1,224-parameter FiLM head and its own 11,826-
parameter post-pool copy: 13,050 parameters per arm. The frozen pre-pool and
decoder must remain byte-stable. `EMPTY` has the same trainable surface as the
real-profile arms and is the governing capacity/co-adaptation control.

Before the first update, every arm's identity and prediction must be bitwise
equal to native.

## Training and checkpoint law

- 12 epochs, Adam `3e-4`, no weight decay, batch 32.
- Same deterministic Q50 source windows and 1,024-window-per-session cap as V1.
- Same resident input batch is used by all four arms before advancing.
- Last-bin task MSE; no teacher loss and no decoder/base loss.
- Frozen modules remain in evaluation mode; no dropout.
- Save every epoch's arm state in memory for audit.
- The governing state is the coordinate-wise float64 mean of epochs 9--12,
  cast back to each original tensor dtype. This last-four parameter average is
  frozen before validation R2 is observed.
- Epoch-12 states are scored only as non-governing diagnostics. No best-epoch
  selection is allowed.

Parameter averaging is used to avoid letting the final deterministic window
sample dominate a small adaptation head. It is not evidence that the V1 final
checkpoint was defective.

## Decision law

Seed 42 runs first. Primary values use the last-four averaged states.

Profile utility requires both:

1. `CP - NATIVE` mean delta greater than zero; and
2. `CP - EMPTY` mean paired delta at least `+0.002` with at least 4/6 sessions
   positive.

A solid signal additionally requires `CP - NATIVE >= +0.010`.

Seeds 43/44 are authorized only if CP10 or CP30 meets profile utility at seed
42. If neither does, the co-adaptation route stops after seed42. A shared gain
of CP and EMPTY is attributed to source post-pool adaptation, not to the
calibration profile.

## Required evidence

- Exact design/workorder/manifest/teacher/parent-checkpoint hashes.
- Attempt before DataModule, checkpoint, Torch CUDA, or GPU access.
- Source and validation session lists; formal-test names only.
- M10/M30 profile, activity30, T4, and Q50 digests.
- GPU0 UUID and isolated `CUDA_VISIBLE_DEVICES=0`.
- Bitwise zero-init identity/prediction parity for every arm.
- Trainable-name lists and exact per-arm parameter count.
- Frozen pre-pool/decoder/native-substrate state digest before and after.
- Per-epoch loss, state digest, gradient-step count, and last-four averaging
  provenance.
- Six-session R2 rows for native, averaged arms, and final-epoch diagnostics.
- Paired CP-minus-native, CP-minus-empty, and CP-minus-shuffle summaries.
- `formal_test_files_opened=false`, target update count zero, EvalAI push false.

## Stop conditions

- Any formal-test path resolution or target-session update.
- Any failure of zero-init parity or frozen-substrate stability.
- Any mismatch in trainable parameter names/counts across arms.
- Any nonfinite loss, parameter, identity, prediction, or R2.
- Neither real-profile arm meets the seed42 profile-utility law.

