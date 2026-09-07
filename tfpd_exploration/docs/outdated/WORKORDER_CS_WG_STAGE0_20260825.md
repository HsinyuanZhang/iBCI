# Work Order: CS-WG Stage 0

Date: 2026-08-25  
Status: design and additive CPU/synthetic implementation specification  
Scientific role: first M1 performance-oriented GPU candidate after review

## 1. Objective

Prepare the source-training core for **Cross-Session Worst-Group SPINT
(CS-WG)** while preserving the exact M1 baseline inference graph.

This Stage 0 work is CPU/synthetic only. It does not authorize source data,
target data, checkpoint tensors, result roots, CUDA/GPU, training, scoring, or a
launch.

## 2. Frozen graph and evaluation contract

The future performance cell must retain:

- M1 W=100;
- M1 neural unit axis N=64 for every one of the four held-in source sessions;
- M10 calibration with a separate B3S trial-time axis of 1024, so each
  per-row calibration tensor is `[10,1024,64]`; W=100 is the decoder window
  and must not be reused as the B3S trial length;
- 16 raw behavior outputs;
- final-bin-only raw-output MSE;
- current B3S identity path;
- exact source/target folds;
- 15,007,496 live parameters after `LazyLinear(1024)` materialization;
- baseline inference topology and latency;
- no target fine-tuning or target backpropagation.

Stage 0 must not edit or fork the baseline model. It should implement only the
sampler/loss/receipt helpers needed by a later route-owned training lifecycle.

"Exact source/target folds" has one unambiguous development meaning here. M1
has four held-in source sessions (`20120924`, `20120926`, `20120927`, and
`20120928`). The first performance screen is four-fold held-in LOSO: each fold
trains only on three sessions and evaluates the untouched fourth session. The
three official held-out sessions are not opened for strata construction,
hyperparameter selection, checkpoint selection, or the four-fold development
gate. Only after a positive four-fold result may one final all-four-source
model be trained and evaluated on the official held-out surface once.

Each outer held-in fold needs a matched ordinary-ERM reference trained from
scratch on the same three source sessions. It must use the same initialization
seed, optimizer, LR schedule, epoch budget, total B32 contract, checkpoint/SWA
rule, and untouched fourth-session scorer as CS-WG; only the predeclared CS-WG
episode sampler and complete robust objective differ. A historical checkpoint
that trained on the outer target session is not a valid comparator. This
matched ERM arm is the minimum performance reference, not a request to split
CS-WG into sampler/loss ablations before a positive result.

## 3. Changed training semantics

For each source session, assign every valid final-bin target to a source-only
task stratum:

```text
active_flag
raw_output_norm_quantile
dominant_coordinate = argmax(abs(y))
```

Quantile boundaries are fitted from source-training labels only and frozen in a
typed authority. Target labels never construct or select strata.

Build balanced episodes with equal represented task strata per source session.
Compute exact per-session dense valid-bin MSE after task-state equalization.

One robust optimization step must contain differentiable losses from at least
two distinct source sessions. The existing `SessionBatchSampler` emits a
single-session batch, so a later route must explicitly assemble an episode as a
mapping from source session to one balanced microbatch. It is invalid to apply
the worst-group formula to one session at a time, or to combine stale/detached
losses from earlier optimizer steps.

Keep total windows per optimizer step equal to the baseline B32 contract. For
an outer fold with three source sessions, use deterministic rotating quotas
`11/11/10`; the session receiving 10 rotates every step. Concatenate the three
microbatches and run the unchanged M1 graph once, then segment its outputs by
the bound session indices to compute the three differentiable losses. Do not
run three full B32 forwards or silently triple the effective batch/MAC. For the
later all-four-source final model, use exactly 8 windows per source session.
Tests must prove that mixed-session batching changes no per-sample model input:
each row keeps its own calibration tensor and session identity.

The complete objective is:

```text
mean_loss = mean_s L_session[s]
robust    = tau * (
    logsumexp((L_session[s] - mean_loss) / tau) - log(num_sessions)
)
loss      = mean_loss + lambda * robust
```

Freeze `0 <= lambda <= 1` and `tau > 0`.  Values above one are forbidden
because they can give a low-loss session a negative loss derivative.  The
`-log(num_sessions)` term is mandatory so equal session losses reduce exactly
to the ordinary mean without a fold-size-dependent constant.  These are
immutable run-spec values. Later selection is
source-only nested leave-one-session-out inside each outer fold and is outside
Stage 0. No outer held-in target or official held-out session may select these
values.

## 4. Stage-0 ownership

Create only additive files under:

```text
tfpd_exploration/src/cross_session_worst_group_v1/
tfpd_exploration/scripts/run_cross_session_worst_group_stage0.py
tfpd_exploration/tests/test_cross_session_worst_group_stage0.py
```

Do not edit shared M1 model, datamodule, trainer, config, scorer, result, or
checkpoint files. Preserve other agents' and the user's work.

## 5. Mandatory tests

Cover at least:

1. source-only quantile fitting and typed authority;
2. target/caller-supplied quantile rejection;
3. deterministic active/norm/dominant-coordinate stratum assignment;
4. exact equal-stratum/equal-session weighting;
5. missing-stratum fail-closed or explicitly declared deterministic fallback;
6. manual equality of per-session MSE and the complete objective;
7. lambda=0 reduces exactly to balanced mean loss;
8. equal session losses give an exactly zero centered robust term; increasing
   any one session loss cannot reduce the complete objective anywhere in the
   frozen `0 <= lambda <= 1` range;
9. one objective contains differentiable contributions from at least two
   distinct sessions and gradients reach every session loss;
10. a single-session pseudo-episode and detached/stale loss table are rejected;
11. session permutation invariance;
12. duplicated windows cannot change session weight;
13. valid-bin and final-bin-only enforcement;
14. no model parameters or inference module in the Stage-0 package;
15. static proof that the future route must validate the exact baseline model
    parameter count and topology before training;
16. dry CLI imports no Torch and performs no write or launch.
17. an outer-fold episode uses total B32 with rotating `11/11/10` quotas, one
    concatenated forward, exact per-row calibration ownership, and no
    three-times-B32 substitution; the final four-source quota is `8/8/8/8`.
18. the future run spec distinguishes the complete CS-WG system from its
    matched same-fold ordinary-ERM performance reference and forbids using a
    checkpoint that trained on the held-in outer target as that reference.
19. a real CPU instance of the closure-bound M1 `SpintModel` receives
    `x=[1,100,64]` and
    `calib_trialized_neural_features=[1,10,1024,64]`, emits `[1,100,16]`,
    materializes `fc_id_in[0].weight` as `[1024,1024]`, and has exactly
    15,007,496 trainable parameters without initializing CUDA. The test must
    also demonstrate that substituting calibration length 100 would produce
    the wrong 14,061,320-parameter graph and is therefore rejected by the
    contract.

## 6. Required worker handoff

Return file SHAs, an explicit closure, exact tests, py_compile/whitespace proof,
and all remaining source-smoke/GPU requirements. Stop before data or CUDA.
