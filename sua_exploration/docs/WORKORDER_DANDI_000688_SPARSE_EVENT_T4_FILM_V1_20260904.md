# Work Order: DANDI 000688 Sparse-Event T4 Estimation + Conditional FiLM V1

Date: 2026-09-04
Status: `EXECUTION_AUTHORITY__SOURCE_TRAIN_VAL_ONLY__CPU_AND_GPU0_GPU1__24H_BOUND`
Route: `dandi688_sparse_event_t4_v1`

## 1. Governing authority

The sole scientific design authority is:

```text
sua_exploration/docs/DESIGN_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md
SHA256 56982085d4cc7c4e06d29d79a3701e8e6f6d93d08955ceff4c09737bef956705
```

This work order authorizes implementation and execution of the design's CPU
Stage 0, frozen-parent Stage 1, conditional matched Stage 2, and only the
corresponding conditionally admitted pseudo-MUA Stage 3. It does not authorize
test-split NWB access, external-15 scoring, EvalAI submission, or an unlisted
successor arm.

The 24-hour bound is an operational completion target, not permission to skip
a scientific gate, overwrite a failed root, shorten 12 epochs, or change a
frozen estimator/profile after observing a decoder score.

## 2. Frozen input authorities

| Input | Relative path | SHA256 |
|---|---|---|
| Strict 27/6/6 manifest | `sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` | `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9` |
| Teacher | `sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt` | `9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d` |
| Parent seed 42, final epoch | `sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt` | `2a50fb2072f0fc76057c22639615a37a21cfa9b83c282cd7fb5e5482b40a199d` |
| Parent seed 43, final epoch | `sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s43/epoch_ckpts/epoch_011.ckpt` | `2126ee843d1b90ce2273f52be81f414f1cd9bc9241f965203445a86981fa7ec5` |
| Parent seed 44, final epoch | `sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_t4_dandi688_co_s44/epoch_ckpts/epoch_011.ckpt` | `654b5423e3f4450d9cf6f4135ddbe1d425e3a2f5e114948d4fb2792f029aa5a7` |

`run_metadata.best_checkpoint` is forbidden. Parent checkpoints are read-only
byte authorities even though their existing filesystem mode is not an
immutable-artifact mode; no code may chmod, rewrite, move, or replace them.

## 3. Authorized hardware and environment

The executor may use both local GPUs, without delayed scheduling:

| Physical GPU | UUID | Model |
|---:|---|---|
| 0 | `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` | NVIDIA GeForce RTX 3090 |
| 1 | `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` | NVIDIA GeForce RTX 3090 |

Each process must attest its physical UUID before CUDA model construction and
publish the physical/logical mapping. A paired cell must remain on one GPU;
seeds may run concurrently on different GPUs. CPU Stage 0, receipt validation,
data preparation, and scoring may overlap GPU execution when they do not read
or mutate a live cell's private files.

Required deterministic process environment:

```text
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Use `/home/xinyuan/miniconda3/envs/spint/bin/python`. CUDA visibility is set
per child to exactly one authorized physical device.

## 4. Namespace and immutable lifecycle

Canonical parent:

```text
sua_exploration/results/dandi688_sparse_event_t4_v1/
```

Authorized fresh subroots are:

```text
stage0/
stage1/sua/seed42/
stage1/sua/seed43/
stage1/sua/seed44/
stage2/estimator/sua/seed42/
stage2/estimator/sua/seed43/
stage2/estimator/sua/seed44/
stage2/film/sua/seed42/
stage2/film/sua/seed43/
stage2/film/sua/seed44/
stage3/estimator/pseudo_mua/seed42..44/
stage3/film/pseudo_mua/seed42..44/
aggregate/
```

Every executable subroot must be absent at admission. It publishes an immutable
`attempt.json` plus SHA sidecar before importing Torch for a GPU stage,
constructing a DataModule, opening an NWB, reading a checkpoint, or initializing CUDA. A
successful root terminates in `terminal.json`; an exception terminates in
`failure.json` with the exact already-published prefix. No root may be retried,
deleted, or overwritten. A correction requires an additive version/root.

Stage 0 is the only import exception: after its attempt, reviewed raw-rate
primitives may transitively import CPU Torch/Lightning modules. Stage 0 may not
construct a model, load a checkpoint, or initialize CUDA; its attempt must
still precede every such transitive import and every NWB read.

## 5. Stage 0 required sequence

1. Verify design/work-order/manifest bytes and prove the exact train/validation
   allowlist before opening any NWB.
2. Publish `attempt.json` with decoder score, test, external, and GPU access all
   false.
3. Materialize `WHOLE-T4`, `POST700-T4`, `T4^H`, `T4^R`, and raw
   `[a_R,c_R,m_R,b_R-b_H]` through the exact route-local contract in design
   Section 22. No dense behavior object may enter the materializer signature.
4. Run every geometry, label-horizon, rank, era, namespace-separation,
   pseudo-MUA order, and descriptor-reference check in design Stage 0.
5. Apply the predeclared two-part per-column reliability law exactly. No
   threshold/window revision is permitted after reading these statistics.
6. Publish the four-bit retention mask and independent estimator/FiLM decisions.
   If a column fails, all later arms receive exact positive zero for it. If all
   columns fail, no FiLM GPU root may be admitted.
7. Publish `stage0.json` and `terminal.json`, then independently rehash the full
   root before any Stage-1/2 capability is issued.

Stage-0 source trials 50-109 may be read only inside the receipt-separated
reliability-audit namespace. They never enter candidate tensors, training, or
scoring.

## 6. Stage 1 authorized sequence

The estimator screen is descriptive and non-gating. For every seed, score the
three no-update rows `WHOLE-PARENTNORM`, `WHOLE-M10NORM`, and
`POST700-M10NORM` on identical Q50 validation windows.

If at least one Stage-0 profile column survives, train the four FiLM heads
`EMPTY`, `PHASE-R`, `SE-T4`, and `ROW-SHUFFLE` together for exactly 12 epochs,
with the parent/base fully frozen as specified by the design. The zero/native
identity and prediction sentinel must pass before the first optimizer step.

All three seeds are authorized without a seed-42 allocation gate. Seeds 42 and
43 should occupy different GPUs concurrently; seed 44 starts immediately on
the first released GPU. Save all epoch states and construct the governing
epoch-8-through-11 float64 parameter average. Only the frozen Stage-1 FiLM
opening predicate may admit Stage-2 FiLM.

## 7. Stage 2 authorized sequence

The matched estimator branch opens directly if Stage-0 descriptor step 9
passes. For every seed, train paired `WHOLE-T4@M10` and `POST700-T4@M10` arms
for exactly 12 epochs. Their source-fitted M10 normalizers are separate and
bound; all other data, model, optimizer, target, batch, and RNG laws are paired.

The matched FiLM branch opens only if Stage 0 retains at least one column and
the three-seed Stage-1 FiLM opening predicate passes. Train the five-arm matrix
from the matching seed's held final-epoch parent for exactly 12 epochs under
the design's common base trainability policy. `WHOLE-NATIVE` has no FiLM head;
the other four heads have identical rank, initialization, and optimizer steps.

Every trained arm saves all 12 epoch states. Governing states are fixed
epoch-8-through-11 averages, never validation-selected epochs. Score all six
validation sessions and compute the exact estimator, semantic, baseline,
attachment, product, and capacity/global-correction contrasts.

## 8. Conditional Stage 3

Pseudo-MUA estimator replication runs only if the SUA matched estimator gate
passes. Pseudo-MUA FiLM replication runs only if the SUA matched semantic and
attachment gates pass. It must pool SUA interval rates by electrode before
refitting every carrier/profile and must fit view-specific normalizers.

Stage 3 may use both GPUs with the same per-seed allocation and exactly 12
epochs. A failed SUA branch creates no corresponding pseudo-MUA training root.

## 9. Fixed training/scoring law

- Seeds: `42,43,44`, paired within seed.
- Epochs: exactly `12` for every trained cell.
- Batch size: `32` unless a pre-model deterministic smoke proves the resident
  paired matrix cannot fit; a smaller common batch requires a failure receipt
  and additive successor, not an in-place change.
- Activity support: chronological M30.
- Candidate labels/profile horizon: chronological M10.
- Query: Q50 only.
- Loss: unchanged `task_only` last-bin MSE.
- Parent training policy: LR `1e-4`, decoder trainable, no early stopping.
- FiLM head rank: `8`; head LR: `3e-4`, matching the frozen CP-FiLM V1 head
  recipe. It must be identical across the four FiLM arms. In Stage 2, shared
  base parameters retain LR `1e-4`; the head uses its explicit `3e-4`
  optimizer group in every FiLM arm.
- Teacher diagnostics use the exact teacher authority but never enter the
  `task_only` training loss.
- No validation argmax, hyperparameter sweep, window sweep, arm insertion, or
  early stopping.
- No target/model update at validation/scoring time.

## 10. Required final evidence

The aggregate receipt/result document must include:

- all body and closure hashes;
- exact source roster and proof that no test/external file was opened;
- Stage-0 per-session/per-era/per-column statistics and four-bit mask;
- all normalizer, carrier, profile, activity, query, prediction, and target
  digests needed to prove matched comparisons;
- per-epoch loss/step/state evidence and epoch-average construction;
- per-seed/per-session R2 for every executed arm;
- deterministic 10,000-draw paired session bootstrap;
- exact design gates and outcome classification;
- GPU UUID/runtime/utilization summary;
- explicit list of branches not executed because a prerequisite gate failed.

Completion means the authorized causal graph has reached terminal receipts and
the final result document honestly reports every opened or closed branch. It
does not mean that a scientific gate must be positive.
