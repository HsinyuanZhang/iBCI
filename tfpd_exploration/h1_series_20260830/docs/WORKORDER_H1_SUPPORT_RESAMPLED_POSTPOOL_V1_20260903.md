# Work Order: H1 Support-Resampled Post-Pool V1

Date: 2026-09-03  
Status: `AUTHORIZED_SOURCE_ONLY_GPU0_THREE_ARM_12_EPOCH`

Execute `DESIGN_H1_SUPPORT_RESAMPLED_POSTPOOL_V1_20260903.md` exactly once.

## Fixed authority and scope

- predecessor: the five sealed C1 epoch-49 date-LODO checkpoints;
- negative-result predecessor: H1 M3 joint-postpool terminal SHA256
  `5decf64d428c8c88eef72138c46c51210931376f4dbc770f497b2249546bdd81`;
- train arms: `LP-F3`, `LP-R3`, `SRPD`; anchor: `FROZEN-C1`;
- operator, support law, loss, distillation weight, optimizer, 12 epochs,
  batching, seed, folds, gates, and stop rule are exactly those in the design;
- only physical GPU0 may be queried or used.  GPU1 must not be queried,
  signalled, reset, reprioritized, or otherwise touched;
- formal held-out and EvalAI remain closed;
- target-date optimizer/backward/model-update counts are zero.

## One-GPU execution

For each fold, load one C1 base and create three byte-identical branch-training
clones resident on logical `cuda:0`.  Build one source roster and one query
batch stream.  Execute the three arm updates sequentially on each resident
batch; LP-R3 and SRPD consume the same support-bank selection.  Seal each
branch-only epoch-12 checkpoint before opening the outer date.

Fresh result root:

`tfpd_exploration/h1_series_20260830/results/h1_support_resampled_postpool_v1/`

Publish immutable attempt, per-fold training/checkpoint/fold receipts, score,
and terminal.  A failure creates failure evidence and the root is not retried
or overwritten.  PASS requires a new all-source/package work order.  FAIL
closes this late-pool+C1 H1 branch.

