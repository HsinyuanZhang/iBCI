# PACD Full V1 P0 Failure Record

Date: 2026-08-31

## Outcome

The first PACD full-training V1 P0 attempt terminated fail-closed before the
first optimizer update. This is an implementation-diagnostic failure, not a
PACD performance result and not evidence for or against calibration dropout.

Canonical immutable root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p0_fullfull_seed42
```

Exact body SHA-256 values:

```text
attempt.json          803e32bedcf7f9b2129b79ff7769733ece0e9fc164bdc07e0a88eca2f3d77073
launch.json           de1e8a75b56f5ebc82e82e886d69d46ff82f2c5887a317c9b108389ff8c2392c
source_authority.json d0524318304bad0f38742a85938e7801ec98d247ba4bd57fb55917921e06e124
failure.json          65b125b982506c82a9daa9eb32d88fed1afca7355155abeab2c224f31b4c3d08
```

Every body and sidecar is regular, non-symlink, and mode `0444`. There is no
epoch receipt, checkpoint, SWA, manifest, or terminal.

## What passed before failure

- accepted PACD V2 source-smoke lineage;
- execution closure
  `c6e4a4811c58ddf4530e200dd6ac704828b4a0f079b5bf1e303127257a380f8e`
  at launch and failure;
- GPU0 identity/idle preflight and post-attempt CUDA binding;
- strict source-27 materialization with no target surface;
- exact behavior and T4 normalizer authority;
- canonical initial-state deserialization, strict state equality, and exact
  positive-zero side block;
- full dataset-window and sampler-batch order digest construction;
- the first real paired forward and anchor half-backward.

Progress at failure:

```text
epochs published      0
checkpoints published 0
SWA published         false
optimizer updates     0 (failure occurred before optimizer.step)
target access         false
```

## Exact cause

The optional sentinel diagnostic asked `torch.autograd.grad` to traverse the
decoder parameter list. Cell-D intentionally retains two inactive
`UninitializedParameter` values in a lazy branch not used by the ordinary
coupled forward. The production finiteness check already skipped and counted
those inactive values, but the newly added short-branch norm/cosine diagnostic
did not apply the same filter. Torch therefore raised `ValueError` before the
short half-backward and before the optimizer update.

The paired loss, RNG replay, dropout mask replay, Cell-D forward, and Adam rule
were not the failing components.

## Disposition

The V1 root is final and will not be overwritten or retried. A science-identical
V2 successor must:

1. bind this exact failure graph through held descriptors;
2. exclude only inactive `UninitializedParameter` values from evidence-only
   branch-gradient traversal and record the skipped count;
3. prove on a real CPU Cell-D graph that evidence-on and evidence-off produce
   the same post-update state;
4. start again from the canonical initial state in a fresh V2 root.

The successor work order is
`WORKORDER_PACD_MATCHED_FULL_TRAINING_V2_20260831.md`.

The concurrently running Stage-P process on physical GPU1 was not signaled,
restarted, edited, or moved. PACD used only GPU0, and GPU0 returned to idle
after the fail-closed exit.
