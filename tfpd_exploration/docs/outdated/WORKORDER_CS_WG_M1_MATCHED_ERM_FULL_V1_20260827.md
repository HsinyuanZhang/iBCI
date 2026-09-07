# CS-WG M1 matched-ERM full-training v1

## Purpose and scope

Build one additive, source-only, same-fold ordinary-ERM reference for the
accepted CS-WG fold whose held-in outer target is `20120924`.  This is a
matched reference, not a new model, parser, sampler, or hyperparameter search.
The only optimization-objective difference from the completed CS-WG full run
is `system=MATCHED_ERM` and `lambda=0.0`; `tau=0.01` remains bound as the
paired run-spec literal.

This work order covers a no-data/no-CUDA candidate only.  It does not reserve
a result root, issue a capability, read an NWB, load checkpoint tensors, or
run a smoke/full job.

## Frozen scientific contract

- Outer target is `20120924`, never opened by source preparation or training.
- Exact ordered source sessions are `20120926`, `20120927`, `20120928`.
- Reuse the accepted V6 common-stratum source constructibility predecessor as
  source-topology evidence only: 16 exact `0444` body/sidecar leaves,
  terminal `dab4c9cfaad593c76d14b9d300642da31d7545248d4daf22bf2691c2b8cceb01`,
  identity `cd8c387317e27865f2b1bbc2b6594aa7d258dd10c9eb472493df0029fdda8879`,
  historical closure `4aa89da216fdf401e6c129a12af28cbb45e9f858d5f4db092fb24b1d77c6947c`.
- Preserve the exact M1 graph: W=100, U=64, raw output=16, M10
  calibration `[10,1024,64]`, 15,007,496 live parameters after materialization,
  dynamic dropout, B32 11/11/10 mixed-session episodes, one concatenated
  forward/step, final-bin raw-output MSE, seed 42, Adam `lr=1e-5`,
  weight-decay 0, no scheduler, 20 epochs.
- Checkpoint selection is first strict minimum source-train epoch mean plus
  last; strict reload/state links are required.  SWA, SWA manifest, and all
  SWA claims are forbidden.
- Exact full objective is the existing complete three-session objective with
  `lambda=0.0`.  Its per-session-loss autograd reference is the uniform
  arithmetic-mean derivative `(1/3,1/3,1/3)`, not the CS-WG softmax reference.
  Fixed observer tolerances are sum/reference `2e-6` and minimum `-2e-6`.
- Source-only remains exact: target optimizer/backward/update is 0; held-out,
  mini-validation, formal, and EvalAI surfaces remain unopened.

## Implementation boundary

New additive ownership is the `cross_session_worst_group_matched_erm_full_v1`
package, its dry CLI, this work order, and its focused test.  Minimal shared
backward-compatible changes may only:

1. take `lambda`/`tau` from the typed `identity.spec.stage0_spec` in the one
   shared optimizer loop while preserving no-argument CS-WG behavior; and
2. allow a typed explicit full-route closure extension and objective-specific
   derivative evidence without changing the historical CS-WG default path.

No optimizer loop is copied.  The route composes the reviewed V6-bound source
provider, reader/cache, M1 materializer, common-stratum fallback, and shared
full runner.

## Future immutable lifecycle

The prospective root is
`tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/fold_20120924_matched_erm`.
At an authorized future launch it must be fresh and non-symlink.  The order is
exactly capability/predecessor validation, reserve, attempt, launch, source
authority, 20 epoch receipts, training, best/last checkpoint pairs, manifest,
terminal; otherwise an honest pre-terminal failure.  Current closure and V6
predecessor must be revalidated before terminal.

## Acceptance tests

Focused synthetic tests must prove: CS-WG default spec/objective behavior is
unchanged; ERM derives lambda/tau from its typed spec; no second optimizer
loop; the exact V6 graph precedes capability/reservation; full temporary
lifecycle has attempt-before-prepare, exactly 20 epoch receipts, best/last
strict links, no SWA, and target updates 0; uniform derivative evidence rejects
nonuniform/wrong-system values; closure/workorder drift and device/root order
fail closed; dry CLI remains inert and imports no Torch.
