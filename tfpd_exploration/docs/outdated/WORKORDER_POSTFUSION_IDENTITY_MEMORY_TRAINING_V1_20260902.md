# Work Order — Post-Fusion Identity Memory Training V1 (2026-09-02)

## 1. Authority and scope

This work order authorizes an additive, no-data implementation and one bounded
matched GPU smoke for the M2 Post-Fusion Identity Memory hypothesis. It does
not yet authorize the full T0-pf/C1-pf pair or target scoring. A future matched
cell uses the same fixed 12-epoch exposure as the exploratory post-fusion
screen; this is a comparison contract, not a universal convergence claim.

The governing design is:

`tfpd_exploration/docs/DESIGN_POSTFUSION_IDENTITY_MEMORY_TRAINING_V1_20260902.md`

with SHA-256:

`29afc5991ca6f287c0a9de0f5a20f737bd2823a9131fd4501752e9775925d4e2`

The implementation must be additive. It must not edit or monkeypatch the
sealed `pit_m2_v1`, `m2_postfusion_probe_v1`, `m2_kcurve_v1`,
`m2_memory_law_scan_v1`, streaming model, data loader, checkpoint, or result
receipts. A narrow shared seam is allowed only if independent review proves it
is backward-compatible and the existing route remains regression-pinned.

## 2. Scientific question

The current pre-fusion identity law is:

```text
rho(mean_i(phi(trial_i)), T4)
```

The proposed post-fusion identity law is:

```text
mean_i(rho(phi(trial_i), T4))
```

These are different set-function placements. Neither is claimed to be a
strict superset of the other. The test asks whether explicit source training
of the post-fusion placement creates a more transferable continual identity
representation than a matched pre-fusion control.

## 3. Frozen context and applicability boundary

1. Frozen-weight post-fusion is already harmful on the local M2 external
   surface: `-0.0826452829` equal-session R2 versus pooled.
2. The sealed local practical anchor is `0.2990573` external and `0.6540862`
   within for the existing B30/D-opt-k4 activity-memory row. It is context,
   not the matched causal control.
3. A0 V2 showed true-trial continual activity minus static `+0.00896` mean
   with `5/6` positives, but fixed phase-0 chunks minus true-trial
   `-0.0321648` mean, `-0.0677556` worst, and `1/6` positives. Therefore this
   route uses known completed-trial boundaries and makes no trial-free or
   official-deployment claim.
4. The V1 scoring support configuration is frozen now as `B=30`, D-opt `k=4`.
   The continual deployment law is uniform `UNCAPPED`. The incomplete
   `m2_kcurve_ext_v1` attempt cannot select this experiment after the fact.

## 4. Frozen arms

Exactly two training arms are permitted:

- `T0_PF`: native pre-fusion B3S identity arithmetic;
- `C1_PF`: post-fusion per-trial readout followed by an arrival-order float32
  mean.

Both arms must start from the exact same initial student state and consume the
same source data, batch order, target path, side tensor, optimizer law,
learning rate, number of optimizer steps, pool selections, Python/NumPy/Torch
RNG states, and dropout decisions.

No PIT cycle, carrier update, learned gate, EMA, recency weighting, chunking,
boundary inference, target backpropagation, or third arm is permitted.

## 5. Identity adapter contract

The route-owned adapter may change only identity arithmetic. It must reuse the
same `pre_pool` and `post_pool` parameter objects and state keys.

### 5.1 T0 conditional no-op

For identical selected `calib_trials`, lengths/masks, and T4 side tensor,
`T0_PF` must be bitwise equal to native B3S in identity, loss, gradients, and
post-update materialized state. With the controller fixed to all available
members, this equivalence must be demonstrated on the real Cell-D architecture
in a CPU/no-CUDA test.

The route must not claim that random or cyclic pool exposure reproduces the
entire historical sealed training trajectory.

### 5.2 C1 post-fusion law

For each selected trial in fixed arrival order:

1. apply the existing `pre_pool`;
2. concatenate the same frozen T4 tensor;
3. apply the existing `post_pool`;
4. accumulate a float32 sequential sum; and
5. divide once by the exact selected member count.

The adapter adds no trainable parameters. T0 and C1 parameter names, shapes,
ordering, count, initial state digest, and optimizer parameter groups must be
identical. Empty pools, duplicate indices, out-of-range indices, inconsistent
lengths, nonfinite values, or a state-key mismatch fail closed.

## 6. Source-only pool-exposure controller

The controller is deterministic and shared by both arms. The selected pool
size follows the fixed optimizer-step cycle:

```text
(30, 10, 4), repeating in this order
```

For a selected size `m`, the controller uses the first `m` chronological
calibration members supplied by the sealed source training record. It does not
use behavior values, target/external data, R2, or trial directions to choose
members. If a source record has fewer than 30 valid calibration members, the
route fails closed rather than changing the cycle.

Each step receipt must bind:

- global optimizer-step index and cycle position;
- pool size and ordered member-index digest;
- selected calibration tensor/mask digest;
- T4 tensor digest and proof that it is shared between arms;
- controller RNG before/after digest, even if the V1 controller consumes no
  random value;
- model/dropout RNG before/after digest; and
- exact T0/C1 batch identity.

The controller is a training exposure intervention. It must never silently
change evaluation support selection or continual-memory law.

## 7. Training and checkpoint-selection contract

The implementation composes the reviewed PIT-M2 training discipline:

- dataset/config/checkpoint/teacher/normalizer authority unchanged;
- seed `42`;
- batch size `32`;
- exactly `12` completed epochs for a future full cell;
- mandatory source-only learning-curve milestones after epochs `3`, `6`, and
  `12`;
- Adam at constant learning rate `1e-4`;
- decoder frozen and student-only optimizer;
- zero target/external gradients and zero target-driven updates;
- source-only validation and checkpoint selection.

Before a full-cell work order can be issued, the selector must prove:

1. training and validation session groups are disjoint;
2. the selection metric is an equal mean over source validation session
   groups;
3. every epoch records that grouped metric and controller-law digest, and the
   final checkpoint is selected over the complete 12-epoch trajectory;
4. T0 and C1 use the same fixed selection rule independently; and
5. no target/external label, row, metric, or sealed score participates in
   checkpoint selection.

If inherited `val_heldin/r2_mean` cannot prove these properties, a route-owned
grouped-OOF evaluator is required.

The epoch-12 source slope is reported descriptively. If it remains positive,
the receipt records a possible horizon limitation, but V1 neither extends the
run nor uses target/external metrics to decide whether to extend it. Any longer
horizon requires a separately preregistered successor.

## 8. No-data and CPU admission tests

The additive implementation must pass all of the following before GPU smoke:

1. T0 native bitwise equivalence on genuine Cell-D tensors;
2. C1 permutation invariance under joint trial/mask permutation and exact
   sensitivity to an unpaired permutation;
3. C1 sequential-mean equality to the frozen post-fusion probe primitive;
4. exact same parameter objects/keys/count and optimizer ordering across arms;
5. exact `(30,10,4)` controller cycle and rejection of shorter records;
6. paired RNG/dropout/batch/T4/pool evidence;
7. nonzero finite encoder gradients and finite losses for both arms;
8. no target data or CUDA import during dry construction;
9. attempt-first, terminal-xor-failure, `0444` body/sidecar receipts; and
10. inert public CLI that cannot mint a live capability.

## 9. Bounded matched GPU smoke

Only after Section 8 and an independent closure audit pass may root issue one
smoke capability per arm. The smoke order is `T0_PF` then `C1_PF`; they run
serially on one physical GPU and must not overlap another experiment on that
GPU.

Each smoke uses the same fixed source sessions, batch stream, initialization,
controller cycle, and optimizer-step count. The exact smoke step count must be
frozen in the implementation plan before launch and be long enough to cover at
least four complete `(30,10,4)` cycles.

Smoke gates:

- T0 and C1 finite loss and finite gradients;
- no decoder or forbidden parameter update;
- exact paired controller/batch/T4/RNG evidence;
- T0 native no-op anchor passes;
- peak VRAM leaves at least 4 GiB physical headroom;
- no OOM/CUDA warning/error;
- C1 throughput and wall-time multiplier relative to T0 are recorded;
- source grouped validation does not regress by more than the pre-registered
  smoke safety tolerance; smoke is feasibility/no-harm only and cannot support
  the paper claim.

Any failure ends this work order. No full training retry is authorized here.

## 10. Future full-cell scoring contract

A later full-cell work order, if smoke passes, must train the matched pair and
score the same two checkpoints in this mandatory 2 x 2 table:

| checkpoint | fixed B30/D-opt-k4 support | uniform UNCAPPED true-trial memory |
|---|---:|---:|
| `T0_PF` | required | required |
| `C1_PF` | required | required |

It must report:

```text
placement_fixed     = C1_fixed - T0_fixed
memory_T0           = T0_continual - T0_fixed
memory_C1           = C1_continual - C1_fixed
placement_x_memory  = memory_C1 - memory_T0
primary_delta_train = C1_continual - T0_continual
```

The promotion gate is `primary_delta_train >= +0.01` external with at least
`4/6` positive sessions. Below `+0.005` is null. Within gain with external loss
is the PIT-M2 overfit signature. Recovery of the frozen-weight OOD loss without
a positive matched effect is null. The axis stops after this matched pair
regardless of outcome.

## 11. Resource isolation

This route must bind one explicitly assigned physical GPU by UUID, CVD, PID,
CPU affinity, and root. It must not enumerate, query, initialize, schedule,
signal, or modify the GPU/process/root assigned to the concurrent M1 work.
Multi-process sharing of one GPU is not authorized for the first smoke because
C1's post-pool multiplier and peak memory are not yet measured.
