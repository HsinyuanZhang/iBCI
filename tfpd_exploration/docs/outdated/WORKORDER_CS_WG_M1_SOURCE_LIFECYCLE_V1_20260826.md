# Work Order: CS-WG M1 Source Lifecycle V1

Date: 2026-08-26  
Status: additive no-data implementation and source-only smoke specification  
Scientific role: first M1 performance system; H1 remains blocked

## 1. Objective

Build the smallest auditable source-training route for Cross-Session
Worst-Group SPINT (CS-WG) on M1.

This route changes the sampler and source-training loss only. It must use the
exact existing M1 SPINT inference graph. It must not add a network layer,
change a parameter, change the calibration path, or open a target surface
during source training.

The implementation stage is no-data and no-CUDA. A later root review may
authorize exactly one 100-step source-only GPU smoke. Full training needs a
separate decision after that smoke.

This work order accepts exactly the re-frozen Stage-0 authority:

```text
Stage-0 work order SHA256:
225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d

Stage-0 explicit 31-path closure SHA256:
dd1fc152d4f7f900d6707bcea46136e1bde7cf781ca2f99dc12991296be7bb51
```

Any Stage-0 byte or closure drift supersedes this work order before source
resolution, root reservation, model construction, or CUDA.

## 2. Frozen performance cell

The system literals are:

- task: M1;
- seed: 42;
- decoder window: 100;
- neural units: 64;
- calibration: chronological M10;
- calibration tensor per row: `[10,1024,64]`;
- raw behavior outputs: 16;
- loss surface: valid final bin only;
- live parameters after lazy materialization: 15,007,496;
- optimizer: Adam;
- learning rate: `1e-5`;
- weight decay: `0.0`;
- scheduler: none;
- epoch budget: 20;
- total windows per optimizer step: 32;
- task-norm quantiles: 4;
- CS-WG `lambda=1.0`;
- CS-WG `tau=0.01`;
- no AMP, TF32, compile, target gradients, or target updates.

For `lambda=1`, the centered objective is exactly a smooth maximum up to a
constant:

```text
loss = tau * logsumexp(L_session / tau) - tau * log(num_sessions)
```

Its gradient weights are the softmax of current session losses and are always
non-negative. `tau=0.01` is frozen before any CS-WG result and is on the scale
of the existing source-only M1 training losses. There is no lambda/tau grid,
hidden inner search, or retry with another penalty after a negative result.

The matched ERM arm must start from the same seed and use the same graph,
optimizer, number of B32 steps, epoch budget, checkpoint rule, and untouched
outer-session scorer. Only the CS-WG episode sampler and complete objective
may differ.

## 3. Exact folds and data boundary

The held-in development sessions are:

```text
20120924
20120926
20120927
20120928
```

Each of four outer folds trains on three sessions and leaves the fourth
untouched. The three official held-out M1 sessions are forbidden during this
development stage.

The route-owned source provider must support all four outer folds, including
the missing historical fold whose outer target is `20120928`. It may reuse
closure-bound parsing primitives, but it must not mutate the existing shared
M1 datamodule or its fold table.

Before any model or CUDA construction, source preparation must prove:

1. exactly the three run-spec source files are selected;
2. the outer held-in target is absent;
3. minival, held-out, formal, EvalAI, and test files are absent;
4. the query window is `[100,64]`;
5. every row retains its own `[10,1024,64]` B3S calibration tensor;
6. valid final-bin raw labels are `[16]` and source-only;
7. session and sample identity are stable through batching;
8. no unit padding, unit mask, time truncation, or calibration sharing is used.

Quantile boundaries and task strata are fitted from the three source sessions
only. If the physical source pools cannot satisfy the frozen Stage-0 common
stratum contract, preparation must stop before CUDA and report the exact
missing-stratum topology. It must not invent a fallback in the live route.

## 4. Training-step contract

The baseline step count for a fold and epoch is:

```text
sum_s floor(valid_source_windows[s] / 32)
```

Both CS-WG and matched ERM use exactly that many optimizer steps and exactly
32 windows per step.

For CS-WG, each outer-fold step uses the deterministic rotating `11/11/10`
quota from Stage 0. It constructs three source microbatches, concatenates them,
and calls the unchanged M1 model exactly once. It then slices the one output
by the bound ownership table and computes three differentiable current-step
session losses. Stale, detached, previous-step, or single-session loss tables
are forbidden.

For matched ERM, use the exact historical source-only SessionBatchSampler
semantics and one B32 single-session forward per step. The total number of B32
steps must equal the paired CS-WG run.

Both routes preserve the existing M1 dynamic dropout behavior. Neither route
may run three B32 forwards, increase the total batch to 96, or cache a session
loss across steps.

## 5. Checkpoint rule

Preserve the source-only fixed-epoch callback semantics:

- train for exactly 20 epochs;
- monitor source `train/loss` only;
- save the best source-loss checkpoint and the last checkpoint;
- no early stopping;
- no validation, target, minival, formal, or EvalAI checkpoint selection;
- no SWA in this route.

The paired performance scorer will use the predeclared best source-loss
checkpoint for each system. Every receipt must also identify the last
checkpoint so a later audit can detect selection drift.

## 6. Additive ownership

The worker may add only:

```text
tfpd_exploration/src/cross_session_worst_group_v1/source_lifecycle.py
tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py
tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_v1.py
tfpd_exploration/tests/test_cross_session_worst_group_m1_source_v1.py
```

Small backward-compatible additions to the existing additive CS-WG package are
allowed only when required to expose an already-tested Stage-0 primitive.

Do not edit the shared M1 model, datamodule, Lightning module, trainer,
configuration, checkpoint, scorer, results, or another experiment route.

## 7. Artifact lifecycle

Use separate immutable roots for smoke and full runs. The canonical smoke root
is:

```text
tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v1
```

Full roots are spec-scoped by outer fold and system. They must not share files
or directories with historical M1 runs.

The source lifecycle is:

```text
attempt -> launch -> source_authority -> smoke or epoch receipts
        -> checkpoint manifest -> terminal
```

On failure it is:

```text
attempt -> optional launch/source_authority -> failure
```

Every published body and canonical sidecar is immutable mode 0444. Downstream
links consume descriptor-reloaded body SHAs. Publication is transactional and
must reject symlinks, hard-link aliases, parent replacement, stale closure,
wrong run spec, partial topology, and an existing canonical root.

Attempt publication occurs before source resolution or model/CUDA work.
Source authority occurs after exact source preparation and before the first
optimizer step. Target/minival/held-out/formal/EvalAI flags remain false in
every receipt.

## 8. Dynamic GPU contract

Do not hard-code GPU0, GPU1, 3090, or a remote 5070Ti into scientific code.
Root chooses one currently idle device at authorization time and binds its
exact UUID, PCI bus, name, compute capability, total bytes, Torch/CUDA/cuDNN
versions, and visible-device mapping into the launch and source authority.

The physical route requires:

- exactly one visible GPU;
- TF32 disabled for matmul and cuDNN;
- no AMP or compile;
- OMP, MKL, OpenBLAS, and NumExpr threads equal to 1;
- no signal, priority, affinity, environment, or file mutation affecting
  Z1, Z4, Z6, P4, or another user process.

The route restores process-local numerical flags on every normal or failed
close path.

## 9. Source-only smoke

The first authorized smoke uses outer target `20120924`, trains only on
`20120926/20120927/20120928`, and runs exactly 100 CS-WG optimizer steps.

It must prove:

- exact current Stage-0 closure and work-order binding;
- source-only file topology and label authority;
- exact M1 graph, parameter count, and `[B,100,16]` output;
- one concatenated B32 forward per step;
- exact rotating `11/11/10` quotas;
- finite objective, model, gradients, and Adam state;
- every initialized trainable parameter either receives a finite gradient or
  is covered by an exact predeclared architectural exclusion;
- non-negative per-session objective derivatives;
- no detached or stale session loss;
- no target/minival/held-out/formal/EvalAI access;
- zero target backward, optimizer, and update counts;
- synchronized elapsed time, steps/s, samples/s, current and peak CUDA memory;
- model state changes after optimization and reloads from the smoke checkpoint.

The smoke is constructibility and resource evidence only. It does not
authorize a scientific claim or H1.

## 10. Mandatory no-data tests

At minimum, test:

1. exact Stage-0 closure and final work-order binding;
2. all four fold topologies, including fold 3;
3. source-only path admission and every forbidden surface;
4. `[10,1024,64]` calibration preservation and rejection of `[10,100,64]`;
5. one-forward B32 ownership and rotating quotas;
6. exact step-count parity between CS-WG and matched ERM;
7. fixed `lambda=1.0`, `tau=0.01`, and no grid/search interface;
8. smooth-max algebra and non-negative gradients;
9. deterministic episode replay without host RNG mutation;
10. checkpoint and terminal topology, including best and last roles;
11. attempt-before-source/model/CUDA ordering;
12. failure receipts before and after source authority;
13. descriptor, sidecar, closure, identity, root, and topology adversaries;
14. dynamic device profile validation with no physical CUDA in tests;
15. static dry CLI imports no Torch and writes nothing;
16. a real CPU one-step M1 graph test with finite backward and Adam update;
17. unchanged inference output for a fixed state and input when called through
    the route-owned wrapper versus the exact baseline graph.

## 11. Stop conditions

Stop before any GPU or source-data action if the no-data candidate is not fully
green or its closure is unstable.

Stop the smoke before CUDA if source strata, shapes, folds, files, or labels do
not satisfy the frozen contract.

Do not launch full training from this work order alone. Root must independently
audit the frozen bytes, current source authority, idle device, fresh root, and
the exact 100-step smoke capability. Luna performs read-only monitoring after
launch and reports natural events or a 30-minute aggregate; it never retries,
edits, signals, or intervenes.

H1 remains blocked until the complete four-fold M1 CS-WG versus matched-ERM
development result is positive under the governing gate.
