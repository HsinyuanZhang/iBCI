# Work Order — M2 Post-Fusion Checkpoint Score V1 (2026-09-02)

Status: pre-registered while the source-only Post-Fusion screen is running.
This file is not part of that live screen's execution closure. No scoring
capability may be issued until the successful screen terminal and all three
checkpoint bodies have been independently validated and bound by SHA-256.

## 1. Purpose and scope

Score every checkpoint produced by the fixed 12-epoch Post-Fusion variant
screen. This is an exploratory candidate-selection score, not a matched causal
estimate: the screen intentionally omitted a newly trained pre-fusion T0.

The score is the narrow mechanism test already frozen by
`DESIGN_POSTFUSION_IDENTITY_MEMORY_TRAINING_V1_20260902.md`:

```text
B = 30 source pool
k = 4 D-opt support rows
surfaces = {external_post30_local, within_post30}
arms = {PF-MEAN, PF-R1, PF-R50}
memory laws = {FIXED30, UNCAPPED}
```

V1 must not expand this table to chronological M10/M30, official-query, EMA,
learned eviction, carrier update, or a newly selected support law. Those would
change the scientific question and require a separately registered successor.

Expected score topology:

```text
3 arms x 2 memory laws x (6 external + 7 within sessions) = 78 rows
```

## 2. Producer binding

Before any model, checkpoint tensor, source data, target data, Torch, or CUDA
access, a held-descriptor validator must bind the successful screen graph:

- exact screen attempt, launch, source-authority, screen, and terminal bodies;
- exact screen implementation-closure SHA-256;
- exact successful status and terminal/failure exclusion;
- exact three binary checkpoint names, body SHA-256 sidecars, `0444` modes,
  regular-file and no-symlink topology;
- exact per-arm selected source epoch and student-state digest;
- exact disclosure that the source selector was the seven-session
  `source_heldin_in_sample_monitor`, not validation or grouped OOF;
- exact evidence that all three checkpoints were retained independently and
  that no external score selected their epochs.

The literal body digests are deliberately not guessed in this work order.
They must be added to the route binding only after a successful natural
terminal is independently audited. A failed or incomplete screen cannot mint
a scoring capability.

## 3. Model reconstruction and freeze law

For each arm, construct the exact sealed PIT-M2 stack, install the matching
Post-Fusion adapter before loading the checkpoint, and then strict-load the
checkpoint model state:

```text
PF-MEAN -> PostFusionIdentityAdapter(..., "PF-MEAN")
PF-R1   -> PostFusionIdentityAdapter(..., "PF-R1")
PF-R50  -> PostFusionIdentityAdapter(..., "PF-R50")
```

All student, adapter, teacher, and decoder parameters are frozen after load.
Scoring uses `eval()` plus inference/no-grad mode, constructs no optimizer,
performs no backward pass, and reports zero parameter updates. The strict
loaded state digest must match the producer checkpoint descriptor before and
after scoring.

The existing native streaming delegation on the adapter is not the trained
Post-Fusion per-trial readout and must not be used to construct memory members.
For every selected or completed trial activity row, the member identity is
computed by the trained public adapter itself:

```text
adapter.forward_batch(
    activity[None, None, :, :],
    side_features=frozen_T4[None, :, :],
)
```

This preserves the learned PF-MEAN/PF-R1/PF-R50 arithmetic. It must not call
the historical probe's native `per_trial_identity(adapter, ...)` helper.

## 4. Shared input authority

Materialize each `(surface, session)` exactly once and share that input record
across all six `(arm, memory-law)` rows. Reuse the sealed M2 Post-Fusion probe
authorities for:

- the exact D-opt-4 support indices inside B30;
- the fixed-ridge T4 carrier and raw/normalized T4 digests;
- selected trialized B3S activity rows;
- ordered completed-query trial rows and causal commit order;
- governed scored window starts, targets, validity masks, normalizer, and
  last-bin metric law;
- the exact arrival-order float32 sequential sum/divide operation.

The input authority records session/surface/support/T4/activity/query/window/
target/mask digests and is referenced—not rematerialized—by every result row.
The two surfaces remain distinct:

- `external_post30_local`: six held-out-dataset sessions; primary practical
  transfer surface for this local mechanism experiment;
- `within_post30`: seven source-dataset post-30 sessions; diagnostic surface
  for capacity competition and placement-by-memory interaction.

Neither surface may be renamed `external_official_query`, and no official
submission claim is authorized by this score.

## 5. Memory laws

Both laws begin with the same four immutable D-opt support identities. Support
members are never evicted.

### FIXED30

Use the reviewed `PostFusionUniformIdentityPool(capacity=30)` law. The pool may
hold the four support identities plus at most 26 completed-query identities;
afterward it evicts only the oldest completed identity. Decode each governed
query before committing that query's completed-trial identity.

### UNCAPPED

Use the same reviewed pool with `capacity=None`. It appends every completed
query identity in causal order and never evicts a member. Decode-before-commit
is identical to FIXED30.

On external sessions where fewer than 27 completed-query identities can enter
the pool before the final governed prediction, FIXED30 and UNCAPPED are
expected to be prediction-bitwise identical. That is a required parity/safety
check, not a missing gain. Their informative separation is expected primarily
on `within_post30`.

## 6. Metrics, rows, and contrasts

Every row records:

- arm, memory law, surface, session, and input-authority key;
- prediction and target SHA-256;
- exact scored-window count;
- variance-weighted last-bin R2;
- initial/final member counts, completed commits, evictions, and causal trace
  digest;
- loaded model-state digest before/after and zero-update facts.

Report equal-session means and all paired per-session deltas. Required
contrasts are:

1. `UNCAPPED - FIXED30` within each arm and surface;
2. `PF-R1 - PF-MEAN` and `PF-R50 - PF-MEAN` within each memory law and surface;
3. each Post-Fusion row versus the exact sealed POOLED k4 comparator rows as a
   practical historical reference, clearly labeled unmatched-training context.

For every contrast report mean delta, median delta, positive-session count,
and the complete paired session table. Add a descriptive session bootstrap:

```text
resampling unit = session
seed = 42
resamples = 10,000
interval = ordinary percentile [2.5%, 97.5%]
```

Bootstrap intervals are descriptive and cannot select an epoch, support,
memory law, or unregistered variant.

## 7. Exploratory decision rule

All three checkpoints are scored exactly once regardless of their in-sample
source ranking. The practical nomination gate is evaluated on
`external_post30_local / UNCAPPED` against the sealed POOLED k4 per-session
rows:

```text
equal-session mean delta >= +0.010
and positive sessions >= 4/6
```

If multiple arms pass, nominate the largest external mean delta; an exact tie
uses the frozen complexity order `PF-MEAN`, `PF-R1`, `PF-R50`. This nominates
one arm for a future matched pre-fusion T0 confirmation; it is not a final
paper claim. Delta below `+0.005` is null. Within gain accompanied by external
loss is the pre-registered PIT-M2 overfit signature. If no arm passes, stop the
Post-Fusion architecture axis; do not expand epochs or search M10/M30 after
seeing the score.

## 8. Lifecycle and runtime

The first implementation should be CPU-only because the reviewed frozen
identity decode primitive is CPU-specific:

```text
CUDA_VISIBLE_DEVICES=''
cuda_initialized=false
```

Run a one-session/all-three-arm/two-law smoke first, record wall time, and
project the full 78-row score. If the projection exceeds two hours, stop before
full scoring and design a separately tested GPU-safe decoder with explicit
CPU/GPU prediction and R2 parity. Do not silently edit the CPU primitive.

The immutable result topology is:

```text
attempt -> launch -> input_authority -> score -> terminal XOR failure
```

Attempt publication precedes screen-graph validation, checkpoint loading, data
materialization, and Torch import. The capability is opaque, root-bound,
closure-bound, producer-bound, one-shot, and revalidated before terminal. The
terminal binds all four bodies, the 78-row cardinality, all summary/contrast
recomputations, runtime/resource evidence, `target_updates=0`, and the explicit
limitations:

```text
matched_prefusion_control_trained=false
official_submission_surface_evaluated=false
exploratory_candidate_selection_only=true
```

No automatic retry is authorized.

## 9. Mandatory tests before scoring

1. strict reconstruction/load for all three adapters and exact alpha topology;
2. one real-shape per-trial identity parity test for each trained adapter;
3. FIXED30 and UNCAPPED causal decode-before-commit traces;
4. external no-eviction parity where the capacity census permits it;
5. support-never-evicted, query reorder, early commit, and T4 drift rejection;
6. one materialization per surface/session shared by six result rows;
7. exact 78-row order/cardinality and independent summary recomputation;
8. session-bootstrap determinism and window-resampling rejection;
9. held producer graph/checkpoint body/mode/sidecar/symlink/extra-leaf
   adversarials;
10. attempt-first, one-shot capability, terminal/failure XOR, closure drift,
    zero-gradient/update, no-CUDA, and inert public CLI tests.

