# Work Order: TF-SR Throughput Equivalence Audit

Date: 2026-08-19  
Scientific design and acceptance: root/Sol  
Implementation: Terra  
Runtime monitoring: Luna  
Status: implementation and no-data CPU tests may proceed; GPU benchmark requires root review

## 1. Purpose

The live seed-42 Phase-D v2 run measured 5.910276663 steps/s at batch 32 and therefore projects
275503.1774 seconds (about 76.5 hours) for 48 x 33925 optimizer steps. This is acceptable for the
single already-running architecture test, but it is not acceptable as the default cost of later
seeds or ablations.

This work order measures engineering-only ways to accelerate the exact TF-SR computation. It must
not modify, stop, attach to, inspect tensors from, or otherwise affect the live seed-42 process. It
must not create a scientific result, alter the frozen Phase-D model, or authorize a replacement
training run.

## 2. Fixed evidence

The benchmark binds these exact immutable facts:

```text
model.py SHA                         3d4a3a8d4e2a68e9933e84274f2a62308a6eeb5a6978671b53e72af442148fc4
native causal activity dependency    tfpd_exploration/src/tfpd/bilinear_readin.py
native activity dependency SHA       2576e91baa2ecbae7d9b5734d2aad6618563fe5cd8dbb5cf0f15358caf244ac0
live Phase-D v2 throughput body      tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v2/throughput100.json
live throughput body SHA             60b12c8f79d58ee50af4dd60255de3bf77a69ced4c7ebafe758710225b2da419
measured baseline                    5.910276663093795 steps/s
measured 100-step wall               16.91968171717599 s
projected 48-epoch wall              275503.17740077665 s
physical benchmark GPU               GPU0
GPU0 UUID                            GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
GPU0 BDF                             00000000:01:00.0
GPU0 model                           NVIDIA GeForce RTX 3090
GPU0 nvidia-smi nominal memory       24576 MiB
GPU0 Torch total_memory              25435111424 bytes (24256 MiB after floor division by 2^20)
```

The benchmark may read the throughput JSON and its 0444 sidecar. It must not read any training
checkpoint, epoch state, optimizer state, NWB, cache, target, validation, or formal-test asset.

## 3. Allowed files

Additive files only:

```text
tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark.py
tfpd_exploration/scripts/benchmark_tfsr_b3st4_ddrop_throughput.py
tfpd_exploration/tests/test_tfsr_b3st4_ddrop_v1_throughput_benchmark.py
```

Do not edit `model.py`, `train.py`, `score.py`, their CLIs/tests, either Phase-D work order, the
Phase-E work order, or any existing result.

## 4. Benchmark matrix

Use deterministic synthetic finite tensors with window 50, calibration 30 x 100, normalized T4
width 4, two output channels, and representative N=128. No dataset builder is permitted.

Measure the following in order, stopping a larger batch after a cleanly recorded CUDA OOM:

1. exact production-contract eager step, batch 32;
2. mathematical eager step, batch 32, using the same model forward/loss/backward/Adam but deferring
   the post-step Python receipt `.item()` conversions until after timing;
3. benchmark-local validation-hoisted eager step, batch 32;
4. exact production-contract eager step, batch 64;
5. exact production-contract eager step, batch 128;
6. `torch.compile` mathematical step, batch 32;
7. `torch.compile` mathematical step at the fastest non-OOM eager batch;
8. a benchmark-local GRUCell-equivalent candidate at batch 32 only.

`torch.compile` is an optional engineering implementation on this installed Torch/Triton runtime.
If and only if a compiled cell raises a non-OOM compiler/import/runtime exception, record that cell
as `COMPILE_UNAVAILABLE` with the exact exception class, exception-repr SHA-256, failure stage, and
post-error CUDA cleanup proof.  A batch-32 compile-unavailable cell skips the compiled-fastest cell
with an exact reason and proceeds to the final GRUCell diagnostic.  Do not catch or downgrade such
an exception from any eager production, eager mathematical, validation-hoisted, or GRUCell cell.

Every measured training step is forward + the exact dense valid-bin MSE + backward + Adam step +
zero_grad. Use the frozen Adam configuration and deterministic seed 42. Record warm-up separately;
time at least 20 post-warm-up steps. Set CPU thread pools to one.

The production-contract baseline must include the same per-step loss/dropout/survivor/gain extraction
and finite checks used by `TorchTrainingBackend.train_step`. The mathematical-step cell may defer
only receipt conversion/synchronization.  Its one-step comparison uses the same initial state and
batch and must satisfy the fixed FP32 numerical contract below before it may be timed.

The first reviewed GPU attempt exposed that two independent CUDA backward executions are not
bitwise deterministic even when their forward and loss are bitwise equal.  Five bounded root
diagnostics observed exactly zero forward/loss difference, gradient maxima 2.53e-7 to 2.83e-7,
post-step model maxima 4.23e-6 to 1.17e-5, and Adam-state maxima 2.61e-8 to 2.84e-8.  A separate
deterministic-algorithm probe with `CUBLAS_WORKSPACE_CONFIG=:4096:8` produced zero in all five fields,
confirming that the two code paths have the same operations; deterministic mode is not used for
timing because it would change the live production runtime.  Freeze this pre-timing acceptance:

```text
forward_max_abs          == 0
loss_abs                 == 0
gradient_max_abs         <= 1e-6
model_state_max_abs      <= 2e-5
optimizer_state_max_abs  <= 1e-7
```

The receipt must preserve the raw differences, whether all five were bitwise exact, the exact fixed
tolerance map, and whether the tolerance contract passed.  These tolerances authorize only timing
of the deferred-receipt engineering cell; they do not authorize a training successor.

The validation-hoisted cell exists because the frozen implementation repeatedly performs full CUDA
finite scans and `.item()` synchronizations in `NormalizedT4Batch.validate`,
`TFSRDecoder._validate`, `B3S.forward`, dense-loss validation, and step receipt extraction. Implement
it only inside the benchmark module as an explicit call-through of the same frozen submodules and
operations after one full typed-input validation outside the timed loop. It must not monkeypatch or
edit `model.py`. It must report forward, loss, gradient, post-step model-state, and Adam-state
differences from the production-contract step. A speedup without accepted equivalence is diagnostic
only and cannot authorize a training successor.

The GRUCell candidate is engineering evidence only. Copy every GRU tensor from the baseline by an
explicit name map and report maximum absolute forward, loss, and gradient differences on the same
initial state and same synthetic batch. It is eligible for later scientific review only if root
accepts its numerical equivalence; this work order never authorizes it as a model replacement.

## 5. Required measurements

For every attempted cell record:

- batch, N, dtype, eager/compiled/candidate label;
- warm-up count and measured step count;
- total and median step wall time;
- samples/s and steps/s;
- peak allocated and reserved CUDA bytes;
- process RSS;
- forward/loss/gradient finite checks;
- model and optimizer finite checks after measurement;
- whether CUDA OOM occurred, with no automatic parameter or batch fallback;
- whether the optional compiled implementation was unavailable, with its exception class/SHA and
  failure stage but no fabricated timing/equivalence evidence;
- exact GPU UUID/BDF/name/memory, Torch/CUDA/cuDNN versions;
- source closure at launch and final.

Also report projected 48-epoch time using 33925 steps/epoch. Label every projection
`ENGINEERING_ESTIMATE_NOT_A_TRAINING_RESULT`.

## 6. Correctness and isolation gates

Before GPU execution, synthetic CPU tests must prove:

1. zero-argument CLI is dry: no Torch import, CUDA call, write, or data path;
2. one execution flag is rejected before Torch import;
3. only paired explicit flags may enter the benchmark;
4. the three allowed files, frozen model, its imported native causal-activity implementation, and
   the frozen throughput receipt form an explicit, non-globbed closure;
5. the physical route accepts only `CUDA_VISIBLE_DEVICES=0` and logical `cuda:0` with the exact GPU0
   UUID/BDF/name.  It freezes the two memory authorities separately: `nvidia-smi` must report the
   nominal 24576 MiB field, while Torch `get_device_properties(0).total_memory` must report exactly
   25435111424 bytes.  These values are not interchangeable and must not be coerced to one shared
   MiB field;
6. synthetic inputs use a typed normalized-T4 capability and never a bare tensor;
7. every variant starts from the same exact model state and its own fresh Adam state;
8. measured variants cannot reuse warmed model or optimizer state;
9. OOM cleanup resets CUDA peak statistics and does not silently retry with a different batch;
10. launch/final closure drift prevents receipt publication;
11. output publication is transactional O_EXCL + fsync + 0444 body/sidecar;
12. the receipt schema says `ENGINEERING_ONLY`, `scientific_result=false`, `data_opened=false`,
    `checkpoint_opened=false`, `target_or_formal=false`, and `authorizes_training=false`.
13. the mathematical-step cell passes the fixed one-step FP32 equivalence contract above before it
    may be timed; its raw differences, bitwise-exact flag, tolerance map, and tolerance-pass flag are
    all receipt fields;
14. the validation-hoisted and GRUCell candidates carry explicit numerical-difference evidence and
    are never silently substituted for the frozen production graph.
15. receipt validation reconstructs the exact attempted matrix order and OOM stop semantics: batch
    128 is forbidden after a batch-64 eager OOM, the compiled-fastest cell is forbidden after a
    compiled-batch-32 OOM, and every stop marker must agree with the actual attempted cell statuses.
16. a compiled-batch-32 `COMPILE_UNAVAILABLE` record likewise forbids compiled-fastest, carries exact
    post-error cleanup evidence, and does not prevent the required final GRUCell diagnostic; the same
    exception from a non-compiled cell remains a hard failure and cannot enter a receipt.

## 7. Canonical engineering output

After root reviews the implementation, exactly one GPU0 benchmark may publish:

```text
tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v1/receipt.json
tfpd_exploration/results/tfsr_b3st4_ddrop_throughput_engineering_v1/receipt.json.sha256
```

The pair must be fresh before launch. No result under this root may be used as a scientific score or
as permission to change the running seed-42 job.

## 8. Decision rule

- Continue the current live seed-42 run unchanged in every outcome.
- Do not launch seed 43/44 from the current slow implementation.
- A later optimized training successor requires a separate scientific work order and a new matched
  baseline if batch size, floating-point mode, optimizer-step count, or numerical trajectory changes.
- Prefer an implementation-only acceleration only when model state, forward, loss, gradients,
  optimizer semantics, sample exposure, and receipt lineage are independently accepted as
  equivalent. Throughput alone cannot establish equivalence.
