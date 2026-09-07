# Work Order: PACD Matched Full Training V2

Date: 2026-08-31

Status: authorized for one bounded implementation repair and no-data/no-CUDA
review. Live launch is separate and may occur only after independent root
audit. This document does not authorize target access or scoring.

## 1. Purpose

V2 is a science-identical successor to PACD matched full-training V1. It fixes
only the diagnostic traversal of inactive Cell-D lazy parameters. PACD loss,
two-forward RNG replay, dropout mask, data, model, optimizer, LR, epoch/step
budget, checkpoint rule, SWA rule, and source-only discipline remain unchanged.

The failed V1 P0 attempt reached the first real paired forward but failed
before its first optimizer update. `autograd.grad` received an inactive
`torch.nn.parameter.UninitializedParameter` while collecting optional
short-branch sentinel evidence. The ordinary paired operator and model
finiteness checker already treat those two inactive lazy parameters as
untrained topology. V2 applies the same rule to evidence-only autograd: only
materialized parameters enter branch norm/cosine traversal, and the receipt
records materialized/skipped counts.

## 2. Exact V1 failure predecessor

Before V2 result-root reservation, Torch import, source resolution, checkpoint
deserialization, or CUDA initialization, validate through one held directory
FD with `O_NOFOLLOW` the exact V1 P0 failure root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p0_fullfull_seed42
```

It contains exactly these eight regular, non-symlink, mode-`0444` leaves and
no extras:

```text
attempt.json
attempt.json.sha256
launch.json
launch.json.sha256
source_authority.json
source_authority.json.sha256
failure.json
failure.json.sha256
```

Exact body SHA-256 literals:

```text
attempt.json          803e32bedcf7f9b2129b79ff7769733ece0e9fc164bdc07e0a88eca2f3d77073
launch.json           de1e8a75b56f5ebc82e82e886d69d46ff82f2c5887a317c9b108389ff8c2392c
source_authority.json d0524318304bad0f38742a85938e7801ec98d247ba4bd57fb55917921e06e124
failure.json          65b125b982506c82a9daa9eb32d88fed1afca7355155abeab2c224f31b4c3d08
```

The validator exact-checks V1 arm P0, accepted PACD V2 smoke lineage, closure
`c6e4a4811c58ddf4530e200dd6ac704828b4a0f079b5bf1e303127257a380f8e`,
zero epoch/checkpoint/SWA publication, source-only/no-target facts, and failure
class/detail identifying evidence-only uninitialized-parameter autograd. It is
revalidated before V2 terminal or failure. V2 never edits this graph.

## 3. Unchanged full-training contract

All requirements of
`WORKORDER_PACD_MATCHED_FULL_TRAINING_V1_20260831.md` remain normative except
the V2 root identity and explicit failure predecessor above. In particular:

```text
seed 42; batch 32; workers 0
48 epochs x 33,925 updates = 1,628,400 updates per arm
P0 M30/M30; P1 M30/M4; P2 M30/M10
checkpoints 44-47; inherited final-four SWA
physical GPU0 / CVD=0 only
```

The canonical V2 roots are:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p0_fullfull_seed42
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p1_m4_seed42
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p2_m10_seed42
```

V2 must compose the V1 lifecycle/runner through a typed immutable profile or
equivalent narrow seam. It must not copy the 48-epoch loop or paired operator.
The V1 public default remains its historical V1 root/lineage behavior.

## 4. Required repair evidence

Before launch, no-CUDA tests must prove:

1. the exact eight-leaf V1 failure graph validates and extra/body/sidecar/
   mode/symlink/semantic drift fails closed;
2. a real CPU Cell-D-shaped forward with the two inactive lazy parameters can
   collect sentinel branch norms/cosines without materializing or traversing
   them;
3. evidence-off and repaired evidence-on produce bitwise-identical model state
   after the same paired update;
4. the evidence records exact materialized/skipped counts independently for
   encoder and decoder parameter lists;
5. a synthetic V2 execute-success lifecycle reaches terminal, and a failure
   remains exclusive with terminal;
6. V2 capability binds exact arm/root, V1 failure predecessor, current V2
   closure, static GPU0 profile, and production runtime factory;
7. public CLI is inert and cannot mint execution authority.

## 5. Launch rule

After independent root audit, only P0 V2 may launch first. It starts from the
canonical initial state; no V1 optimizer update is reused because V1 completed
none. P1/P2 remain pending a valid P0 lifecycle. No retry is permitted inside
either V1 or V2 canonical root.
