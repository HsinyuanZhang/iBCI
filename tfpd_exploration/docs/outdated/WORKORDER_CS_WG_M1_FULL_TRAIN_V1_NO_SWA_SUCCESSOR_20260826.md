# CS-WG M1 Full Training V1: V6-Bound No-SWA Successor

## Scope

This is an additive, source-only full-training candidate for the accepted
CS-WG M1 V6 100-step smoke.  It is code plus synthetic/no-CUDA tests only.
It may not open a source, target, held-out, minival, formal, EvalAI, or
checkpoint-tensor surface; initialize CUDA; reserve a root; mint a capability;
or launch.  Historical V1--V6 receipt roots are immutable evidence.

The candidate owns a new full lifecycle and composes the reviewed V1 physical
runner through its typed shared training plan.  It does not copy an optimizer
loop, mutate a model/data/parser/core module, or alter the V6 scientific cell.

## Accepted V6 predecessor

The completed predecessor is exactly the 16-leaf body/sidecar graph at:

```text
tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6
```

The successor must hold and no-follow descriptor-read its root.  It requires
the exact terminal body SHA-256:

```text
dab4c9cfaad593c76d14b9d300642da31d7545248d4daf22bf2691c2b8cceb01
```

The terminal must reconstruct the exact accepted V6 identity:

```text
cd8c387317e27865f2b1bbc2b6594aa7d258dd10c9eb472493df0029fdda8879
```

and both launch/final closure fields must equal:

```text
4aa89da216fdf401e6c129a12af28cbb45e9f858d5f4db092fb24b1d77c6947c
```

The loader must reject extra/failure leaves, non-0444/nonregular/hard-linked
leaves, wrong canonical basename sidecars, terminal-link substitutions, V6
identity/closure drift, or a body/sidecar substitution after opening.  The
exact terminal binds the remaining six pairs: attempt, launch,
source_authority, smoke, best checkpoint, last checkpoint, and checkpoint
manifest.  The predecessor is historical evidence; do not rebuild its old
current closure.

## Frozen full cell

The initial full candidate is the V6 smoke fold only:

```text
system: CS_WG
outer held-in target: 20120924 (unopened)
source sessions: 20120926, 20120927, 20120928
seed: 42
```

`source_lifecycle.build_fold_route_specs("20120924")` is the only fold-pair
authority.  The candidate binds the exact companion MATCHED_ERM run spec and
proves graph/seed/B32/epoch/optimizer/checkpoint equality, but it does not
silently train or select that companion under this single-root lifecycle.

The physical graph, parser, V3 deterministic common-stratum fallback, V4
digest cache, V5 derivative observer semantics, and V6 audit-spec-to-smoke
preparation law remain unchanged except that the exact prepared rows are
rebound to the typed full spec.  No target label, target gradient, target
optimizer/update, refit, validation, held-out/minival/formal/EvalAI access is
permitted.

## Exact training and checkpoint rule

The closure-bound M1 source-decoder authority fixes:

- 20 epochs;
- Adam, `lr=1e-5`, `weight_decay=0`;
- no scheduler, AMP, TF32, or compile;
- one mixed-session concatenated B32 forward, `11/11/10` rotating quota, and
  final-bin raw-output MSE/complete CS-WG objective per logical step;
- `steps_per_epoch = sum_s floor(valid_source_windows[s] / 32)` from the
  source authority, with epoch-local deterministic episode indices because
  the frozen source decoder config has `reshuffle_train_sampler_each_epoch:
  false`;
- a source-only `train/loss` monitor equal to the arithmetic mean of the
  complete returned source objective over the epoch; ties keep the first
  strictly smaller epoch, matching the existing source runner's strict
  minimum rule.

There is **no SWA** in this route.  The old
`M1_CHECKPOINT_SWA_RULE_LITERAL` is an unresolved historical placeholder and
is not consumed.  `swa.pt`, SWA receipts/proofs, SWA manifest fields, or any
claim that SWA was used are forbidden and fail validation.

At every one of epochs 0--19 the lifecycle publishes one immutable epoch
receipt after that epoch's final optimizer boundary.  It publishes exactly two
checkpoint bodies after the run: `checkpoint_best_source_train_loss.pt` and
`checkpoint_last.pt`.  Both are strict-loaded into a fresh exact graph and
rehashed before the checkpoint manifest and terminal.  The manifest chooses
only the first minimum epoch-mean source loss, records the final last state,
and contains no target/validation/SWA selection field.

## Lifecycle and runtime

The prospective root is:

```text
tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/fold_20120924_cswg
```

The root is never probed, reserved, or created in this implementation stage.
At a later root-reviewed execution it is:

```text
attempt -> launch -> source_authority -> epoch_00 ... epoch_19
        -> checkpoint best/last -> checkpoint_manifest -> terminal
```

Failure publishes only the honestly reached prefix plus `failure.json`; it
never publishes terminal.  Attempt is durable before source resolution or
model/CUDA construction.  The selected dynamic single-visible-device profile
is descriptor-bound before reservation and revalidated before terminal.  The
reviewed runtime disables TF32 before model construction, records pre/post
state, measures finite resource/throughput/model/Adam/gradient evidence, and
restores both TF32 flags and Python/NumPy/Torch RNG state on every close path.

## Required no-data coverage

Tests must prove the shared runner has one optimizer-loop implementation and
the old smoke facade/schema remains unchanged; full plans are exactly
20 epochs/no-SWA and reject an ambiguous cardinality/SWA claim; epoch receipt
order/cardinality/digest chain; source-loss first-minimum checkpoint selection;
best/last strict state links; V6 descriptor/topology/link/tamper rejection;
attempt-before-prepare; failure progress; dynamic profile/TF32 restoration
seams; source-only flags; fresh-root/collision rejection; no Torch/CUDA in the
public dry CLI; and no current V6 closure rebuild.

The candidate stops at a no-data/no-CUDA review boundary.  Root alone may
review, mint, reserve, or launch it.
