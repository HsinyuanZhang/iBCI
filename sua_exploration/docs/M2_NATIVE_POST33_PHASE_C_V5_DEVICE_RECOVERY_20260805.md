# Native-M2 Phase-C v5 device recovery

**Status: engineering repair verified; no formal rerun authorization.**

## Incident mechanism

The r6d Stage-A SPINT workers reached their post-`Trainer.test` deployment
microbenchmark, where the v4 evaluator did only `model.eval()` and then passed
a CPU B=1 neural window to `DeploymentProfilerV4`.  The profiler transfers
that window to CUDA before calling `model.cached_online_forward`.

The installed Lightning version is 2.4.0.  Its
`lightning.pytorch.strategies.strategy.Strategy.teardown()` calls
`self.lightning_module.cpu()`: registered parameters and buffers move to CPU.
Phase-C's cached identities are plain tensor attributes, so that operation
does not migrate them.  The evaluation loop also restores the prior
`nn.Module.training` state after test.  The concrete paths are:

| Arm | Deployment module | Plain cached identity | v4 post-test B=1 input |
| --- | --- | --- | --- |
| SPINT | `model.net` | `model._cached_identity` | profiler CPU input → CUDA |
| T4 | `model.student` | `model._cached_outer_identity` | profiler CPU input → CUDA |

Thus the observed post-test state is registered deployment state on CPU,
plain cache on CUDA, and `training=True`.  Calling `model.eval()` fixes only
the mode, not the device split.

## v5 repair boundary

The new score-free helper is
`sua_exploration/mc_maze/m2_native_post33_deployment_device_prep_v5.py`
(SHA-256 `3e476e11297e32632fe411cf0723afedd3ed81a03964b7f5abc8eb72e7e4d146`).
Immediately after `Trainer.test`, it:

1. captures canonical cached-identity bytes/hash and byte count;
2. explicitly moves the arm's deployment module and plain cache to one CUDA
   device, then forces owner and deployment module into eval mode;
3. verifies all deployment parameters, buffers, cache, and the actual B=1
   tensor supplied by the profiler share that device;
4. wraps every profiler forward with pre/post device/mode/cache-byte checks;
5. rechecks the full cached-identity hash after profiling.

It contains no dataset loading, target labels, chronology mutation, checkpoint
selection, score, or R² logic.  Hashing is intentionally outside timed
repeats, so the benchmark's measured calls do not include a CUDA-to-CPU hash
copy.

The append-only full evaluator copies are:

- `SPINT-main/src/evaluate_post33_phase_c_v5.py` — SHA-256
  `0958958284ac4832e7d1afccda36fe55bb7476082da314c24b6e50cd9ea67c03`
- `streaming_calibration_exp/src/evaluate_post33_phase_c_v5.py` — SHA-256
  `9c08a092201f53c73167f84d3371024e405eed2a3f4a4db3fd7cb2ca5fd21b69`

Their protected `main()` bodies are AST-identical to v4 before and after the
single replacement block.  The only executable replacement is:

```text
neural_cpu = deployment_neural_window()
prepared = prepare(... arm=spint|t4, neural_cpu)
profiler.benchmark_online_b1(prepared.bind_checked_forward(cached_online_forward), neural_cpu)
prepared.assert_ready()
```

The new v5 evaluator files still deliberately reference the same Phase-C v4
contracts while they are review artifacts.  They are **not** a license to run
against r6d or to construct a formal endpoint execution.

## Verification performed

- Focused CPU unit suite: 8 passed with third-party pytest autoload disabled
  because the host-wide `dandi` pytest plugin is incompatible with its current
  Click version.
- Synthetic real-CUDA B=1 smoke on local GPU0: both SPINT and T4 used a real
  Lightning `Trainer.test`, reproduced the post-test CPU/CUDA/training split,
  then ran the unchanged `DeploymentProfilerV4` 5 warmups plus 20 timed calls
  through the v5 checked callable.  The smoke uses only synthetic tensors; it
  explicitly reports no formal endpoint or R² access.
- `test_m2_native_post33_phase_c_v5_evaluator_exact_diff.py` prevents an
  unreviewed change elsewhere in either evaluator's protected `try/finally`
  body.

## r6d disposition

r6d is retired, not repaired in place.  The append-only retirement receipt is
[`r6d_engineering_failure_retirement.json`](../results/m2_native_post33_phase_c_v5_r7_device_recovery_20260805/r6d_engineering_failure_retirement.json)
(SHA-256 `d273e80462a5ca8453872928c3487757813b9d137cb2d6fdeb513ce04c93812f`,
mode `0444`), with a read-only seal reference beside it.  The exact live
signer received a controlled SIGINT, briefly became a zombie, and was reaped
by its verified tmux parent after a SIGCHLD request.  The retirement audit
opened neither failure status/log content nor selector/endpoint/R² content and
confirmed no Stage-A decision, Stage-B/full-opening authorization, conditional
acknowledgement, live worker, or GPU compute process.

## Authorization gap

No r7 cell root, selector, nonce, signer key, program receipt, authorization,
or GPU launch was created by this recovery work.  Before any formal rerun,
the following must be independently reviewed and deliberately created:

1. a fresh r7 root and fresh per-capability nonces/selection state, disjoint
   from r6d;
2. an r7 program/source closure sealing the two v5 evaluators and device helper;
3. a fresh signer/public-anchor and matching r7 authorization chain;
4. explicit owner GO for formal GPU execution and held-out endpoint access.

This document itself is a recovery handoff, not any of those authorizations.
