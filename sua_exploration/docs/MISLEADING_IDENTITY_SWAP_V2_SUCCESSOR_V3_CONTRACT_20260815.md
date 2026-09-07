# Misleading-identity swap-v2: Stage-P successor v3

**Status:** successor-only development execution contract.  It supersedes the
failed v2 launch path but does not alter the scientific intervention, cells,
epoch window, data scope, target-query policy, or decision gates.

## Why a successor is required

The v2 canonical attempt wrote the immutable source-only `clean_z4` launch
receipt
`checkpoints/misleading_identity_swap_v2_stage_p_seed42/clean_z4/launch_receipt.json`
with byte SHA
`1dfb05e1155fbfc7d0f39f38081e2ef19e5f5fd6612d76b7e5f0924ce0d06f73`.
Lightning 2.4 entered its pre-fit validation sanity loop before epoch zero,
but the v2 wrapper recognised only `trainer.validating`, misclassified that
loop as stand-alone scoring, and refused an absent checkpoint epoch.  It
produced no epoch checkpoint, score, target access, formal access, or terminal
receipt.

That launch receipt and its v2 root remain immutable evidence.  They must not
be deleted, repaired, copied, or reused.  The former v2 official preflight
`b117fa4bf24c48d5cf5b1f08a390ef2ed9666ef50c3e5336412146166c831bdf` is
not an authority for this successor: its closure does not include the phase
fix or the production source-only sanity regression.

## Sole behavioural repair

During ordinary `Trainer.fit`, both `trainer.sanity_checking` and
`trainer.validating` are internal source validation.  They must use
`eval_clean` at the current training epoch and may never enter
`eval_swapped_diagnostic`.  Stand-alone checkpoint scoring continues to
require an explicit checkpoint epoch.  Training itself remains `train_clean`
or `train_swap` according to the predeclared cell.

Before successor preflight mint, the exact production
`MisleadingIdentitySwapV2LitModule` must complete a CPU-only `Trainer.fit`
source-only regression for all four cells (`clean_z4`, `clean_t4`, `swap_z4`,
`swap_t4`) through the epoch-zero train/validation boundary.  It uses strict
sub-C source and development-validation sessions only, never resolves a
formal session, target session, target authority, target query, target score,
or GPU.

## Frozen scientific design retained verbatim

- Cells remain clean/swap × Z4/T4, seed 42, B3S, task-only, 12 epochs.
- All four consume byte-identical strict27 M30 matching-authority bytes and
  the same immutable initial-state bytes.
- The scoring epoch window remains 5--12.
- Support is chronological rewarded trial 0--29; query is strictly after
  rewarded trial 30; source normalizers remain strict27 source-only.
- Normal clean scoring and the swapped-input diagnostic remain distinct.
- Target authorities are built once per development domain, byte-shared across
  all four cells/T4/Z4, use no target optimizer or backward steps, and are not
  opened before every source terminal exists.
- The sixteen score slots remain four cells × within/external development ×
  clean/swapped diagnostic.  The aggregate is fresh, immutable, and requires
  all sixteen exact pairs.
- The external clean-input T4 absolute lift remains the primary adoption gate;
  within clean-input T4 remains the sibling preservation gate.  Carrier
  interaction and swapped-input robustness are reported only as non-rescuing
  diagnostics.  No intermediate score may select or stop a cell.

## Successor execution boundary

The only valid successor authority is the new canonical v3 preflight under
`results/misleading_identity_swap_v2_source_authority_dev/official_cpu_preflight_v3.json`.
It binds this contract, the repaired implementation closure, the preserved
invalid v2 attempt, exact strict27 authority/lineage/initial state, sealed A2
parents, four new canonical cell roots under
`checkpoints/misleading_identity_swap_v2_stage_p_successor_v3_seed42`, and
the fresh score/aggregate root
`results/misleading_identity_swap_v2_stage_p_successor_v3`.

Only after the v3 preflight, source-only regression, root review, and a fresh
GPU0 launch may the full fixed order run at concurrency one:

```text
clean_z4 -> clean_t4 -> swap_z4 -> swap_t4
```

The child must set `PYTHONNOUSERSITE=1`, start with empty `PYTHONPATH`, bind
physical GPU0 through `CUDA_VISIBLE_DEVICES=0`, and record the isolated Torch
identity.  GPU1 is reserved for concurrent CEBRA work.  Target authority,
target scoring, and aggregation are separate post-terminal operations and may
not start before all four source terminals validate against the v3 preflight.

Formal sub-C sessions remain sealed throughout.  This is a bounded development
result, not a formal result or a universal causal claim about carrier or
NeuronID.
