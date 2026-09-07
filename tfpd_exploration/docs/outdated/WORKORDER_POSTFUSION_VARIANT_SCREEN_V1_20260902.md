# Work Order — Post-Fusion Variant Screen V1 (2026-09-02)

## 1. Purpose

This work order authorizes a fast, exploratory, source-only M2 screen of three
post-fusion identity variants. It deliberately omits a newly trained pre-fusion
T0 arm. Its result may select a post-fusion candidate for later confirmatory
training, but it cannot by itself establish a causal improvement over the
training recipe or support the final paper claim.

The reviewed design is:

`tfpd_exploration/docs/DESIGN_POSTFUSION_IDENTITY_MEMORY_TRAINING_V1_20260902.md`

The current design SHA-256 is:

`29afc5991ca6f287c0a9de0f5a20f737bd2823a9131fd4501752e9775925d4e2`

## 2. Frozen screen arms

All arms reuse the same sealed B3S `pre_pool` and `post_pool` parameter objects
and start from byte-identical student weights.

### PF-MEAN

```text
h_pf = mean_i(post_pool(concat(pre_pool(trial_i), T4)))
h = h_pf
```

No additional trainable parameter.

### PF-R1

```text
h_native = post_pool(concat(mean_i(pre_pool(trial_i)), T4))
h_pf     = mean_i(post_pool(concat(pre_pool(trial_i), T4)))
h        = h_native + tanh(alpha) * (h_pf - h_native)
```

`alpha` is one trainable float32 scalar initialized to exact positive zero.

### PF-R50

The same residual law, but `alpha` is a trainable float32 vector of length 50,
initialized to exact positive zeros and applied elementwise.

PF-R1 and PF-R50 must be bitwise identical to native B3S at initialization.
Their first-step receipts must separately prove nonzero finite gradient on
`alpha`; after `alpha` moves, both native and post-fusion branches must remain
gradient-connected. The decoder remains frozen.

No attention, learned trial weights, PIT cycle, carrier update, EMA, chunk,
boundary inference, or fourth arm is permitted.

## 3. Shared training input law

The three arms consume one source-only DataLoader and the exact same batch in
each optimizer step. The calibration member count uses the deterministic cycle

```text
(30, 10, 4), repeating by global optimizer step
```

with the first `m` chronological members. The same selected calibration tensor,
mask/lengths, T4 tensor, target, batch order, Python/NumPy/Torch RNG snapshot,
and dropout decision are supplied to all three arms. If any record has fewer
than 30 valid members, the route fails closed.

The controller has a private receipt-bound generator even though V1 consumes no
random draw. No target/external value, R2, direction, or post-hoc metric may
select a member or pool size.

## 4. Fixed 12-epoch screen law

- dataset/config/checkpoint/teacher/normalizer: exact PIT-M2 sealed authority;
- seed: `42`;
- all three arms: exactly `12` completed epochs, with mandatory milestones
  after epochs `3`, `6`, and `12`;
- batch size: `32`;
- optimizer: independent Adam per arm, constant learning rate `1e-4`;
- decoder: frozen;
- student identity path: trainable;
- DataLoader construction and source materialization: exactly once;
- optimizer steps and examples seen: identical across all arms;
- target/external gradients and updates: zero.

Live CPU admission established that the inherited PIT-M2 source monitor is not
grouped OOF: its seven `val_heldin` session names equal the seven training
session names, and all `1,011` monitor `(session, window_start)` coordinates
are contained in the `115,911` training coordinates. V1 therefore retains the
inherited monitor only under the honest name `source_heldin_in_sample_monitor`.
It is the equal mean of the seven per-session R2 values and is computed on the
same rows for every arm. It may select the best epoch independently within
each arm and provide a descriptive source ranking; it is not evidence of
source-session generalization and is not called validation, grouped OOF, or
held-out performance.

The descriptive source winner is the highest best-epoch monitor value through
epoch 12. If values differ by at most `0.002`, choose the arm with fewer added
parameters in the fixed order `PF-MEAN`, `PF-R1`, `PF-R50`. This winner does
not suppress evaluation of the other two preregistered arms.

Twelve epochs is the frozen common exposure for this fast screen, not a claim
that every post-fusion parameterization has converged. The terminal must report
the epoch-3/6/12 source curves and whether the epoch-12 source slope remains
positive. A positive endpoint slope is recorded as
`possible_horizon_limitation=true`; it does not authorize an extension or
invalidate the matched 12-epoch comparison. No external/target value may
choose the training horizon or any checkpoint.

The sealed pre-fusion score may be recorded only as context. It is not a matched
causal control, cannot select the winner, and cannot be used to claim a new
external improvement from this screen.

## 5. Single-process GPU utilization plan

The implementation uses one process and one physical GPU:

1. materialize one source batch;
2. transfer it to the selected GPU once;
3. execute PF-MEAN, PF-R1, and PF-R50 forward/backward/step sequentially on
   that same resident batch;
4. advance the DataLoader only after all three arm steps complete.

This raises useful GPU work per data-loading event without CUDA streams,
multiple processes, nondeterministic overlap, or duplicated source parsing.
Model/optimizer states remain independent. AMP, `torch.compile`, CUDA graphs,
and batch-size changes are forbidden in V1 so the screen does not mix numerical
or optimization changes with placement. All three independent model/optimizer
states remain resident and receive exactly the same 12-epoch batch exposure.

The selected GPU must retain at least 4 GiB physical VRAM headroom. Another
physical GPU and its M1 process/root must not be enumerated, queried,
initialized, scheduled, signaled, or modified.

## 6. Mandatory no-CUDA and CPU tests

Before a live capability can be issued:

1. genuine Cell-D PF-MEAN arithmetic matches the frozen post-fusion probe;
2. PF-R1/PF-R50 are bitwise native at zero initialization;
3. all arms have the expected exact parameter topology and independent state;
4. first-step alpha gradients are finite/nonzero for both residual arms;
5. the `(30,10,4)` cycle and shared batch/T4/pool/RNG evidence are exact;
6. an arm-order permutation preserves each arm's update under replayed RNG;
7. one DataLoader/materialization feeds all three arms;
8. the exact same seven-session roster and exact `1,011/1,011` monitor-window
   overlap with training are descriptor-proved and honestly named in-sample;
9. public dry CLI imports neither Torch nor CUDA and cannot mint capability;
10. attempt-first, terminal-xor-failure, sidecar/mode/topology, closure and
    one-shot capability adversarials pass.

## 7. Live smoke and continuation gate

The live run begins with exactly 12 paired optimizer steps, covering four full
pool cycles. It records per-arm loss, gradient, state change, step time, GPU
utilization samples, and peak allocated/reserved memory.

The process may continue directly into the twelve common source epochs only if:

- every arm is finite and gradient-connected;
- PF-R1/PF-R50 alpha moves away from exact zero;
- no forbidden decoder/teacher/target update occurs;
- paired batch/T4/pool/RNG/dropout evidence is exact;
- no OOM, CUDA error, or nondeterministic-operation warning occurs;
- at least 4 GiB VRAM remains free; and
- projected time for the twelve-epoch three-arm screen stays within a
  180-minute hard cap.

Otherwise it publishes a typed failure and stops. No automatic retry is
authorized.

## 8. Screen terminal and next decision

The terminal reports all three 12-epoch source-monitor curves, one independently
selected source-best checkpoint per arm, the descriptive source winner,
tie-break, endpoint-slope limitation flags, throughput, GPU peak, the exact
monitor/training overlap, and explicit limitations
`matched_prefusion_control_trained=false`.

- If no arm is finite/non-harmful on the source monitor, record that fact; it
  does not hide any preregistered arm from the one-shot evaluation.
- No additional training is authorized in this V1 screen.
- After the epoch-12 source terminal and an independent receipt audit, the
  three preregistered source-best checkpoints may each be scored once under
  the same fixed and UNCAPPED local memory regimes. All three are evaluated;
  the in-sample source ranking cannot hide an arm. External results are
  evaluation-only and cannot alter training or checkpoints.
- A newly trained matched pre-fusion T0 is deferred until the winner has a
  positive external signal; it remains mandatory before a final causal or
  paper claim.
