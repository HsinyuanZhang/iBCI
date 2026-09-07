# Work Order: PACD Matched Full Training V1

Date: 2026-08-31

Status: authorized for implementation and no-data/no-CUDA review only. No
full GPU launch, source-data access, checkpoint deserialization, canonical
full-result reservation, target scoring, or target access is authorized by
this work order.

## 1. Objective

Build the source-only full-training successor of the accepted PACD V2 smoke.
The mandatory seed-42 cells are:

```text
P0-FullFull   anchor M30 + short M30
P1-PACD-M4   anchor M30 + short M4
P2-PACD-M10  anchor M30 + short M10
```

Each cell starts from the same immutable canonical Cell-D state, uses the same
strict 27-session source roster and batch order, and performs the same number
of optimizer updates. The only scientific difference is the short branch's
chronological B3S calibration-activity prefix. Ordinary M30 T4 remains fixed
in both branches.

The canonical state has artifact SHA-256
`b0a340fe4d09eac1f2b498658d39a8a87e87f52304cad2539753ae9e040fcbd4`
and strict-loaded model-state SHA-256
`65bacb85447df40ea5e03cffed03b1763d1964bfba50cbbe3d21d1614c07f2a3`.
Both values, the strict-load equality proof, and the initial positive-zero
side block must be checked and recorded independently by every arm.

The full runner must call the reviewed PACD paired-step operator. It must not
replace it with CAL-AUG's single-view prefix hook, copy a second PACD operator,
introduce a consistency loss, or modify the sealed Cell-D model.

## 2. Accepted PACD V2 predecessor

Before reserving a full-result root, importing Torch, resolving source data,
loading checkpoint tensors, or initializing CUDA, the route must validate the
accepted V2 source-smoke graph through a held directory descriptor with
`O_NOFOLLOW`.

Canonical predecessor root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42
```

Exact topology is four regular, non-symlink, mode-`0444` leaves and no extras:

```text
attempt.json
attempt.json.sha256
terminal.json
terminal.json.sha256
```

Exact body SHA-256 literals:

```text
attempt.json   ef2ebde24864c8105e47b6ef1c925149b128fe9531577dbf58fba227ea11eb3a
terminal.json  a04be949665a5c57091ba2192793401f43950735142fdb8fac9e2ec6829f9dfe
```

The V2 terminal must bind attempt SHA above, status
`PACD_SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE`, launch/final closure
`1b7ebf98a01e197718588eb9702949304f68b527ba08660ae88dd62857d11339`,
the exact V1 failed predecessor, all three arms with eight optimizer steps,
P0 prediction/identity equality on every step, paired mask/RNG equality,
29 materialized finite parameters, two inactive lazy parameters, and the
source-only/no-target facts.

The full route revalidates this held graph before every terminal or failure.
It never edits, renames, deletes, chmods, or adds a predecessor leaf.

## 3. Immutable scientific budget

Every cell uses:

```text
seed                  42
epochs                48
optimizer steps/epoch 33,925
total optimizer steps 1,628,400
logical batch size    32
num_workers           0 while another repository GPU job is active
optimizer             inherited Cell-D Adam constructor
LR schedule           inherited 48-epoch warmup/cosine schedule
checkpoint epochs     44, 45, 46, 47
SWA                    exact inherited final-four rule
```

The accepted eight-step V2 smoke measured 5.876, 10.077, and 9.148 paired
optimizer steps/s for P0, P1, and P2. Applied mechanically to 1,628,400 steps,
these give descriptive first projections of 76.98 h, 44.89 h, and 49.45 h
(171.31 h sequentially). These are planning estimates, not acceptance results.
Each attempt records the precursor values and projection. Epoch 0 records a
synchronized observed throughput and replaces the estimate for monitoring;
it cannot select, stop, or retry an arm merely for being faster or slower.

Free memory is not authority to change batch size. Any future physical
microbatch or batch-size acceleration is a separate numeric-equivalence
successor and must apply identically to P0, P1, and P2.

For each unique source query and update, the runner performs:

```text
loss = 0.5 * task_loss(M30 anchor) + 0.5 * task_loss(short view)
```

Both forwards use the same query, behavior target, ordinary M30 T4, unit
order, valid mask, realized Cell-D whole-unit dropout probability, and exact
unit mask. The post-pair RNG state equals one ordinary forward. Backward and
the single Adam update consume no route-visible RNG.

`SessionBatchSampler.batched_indices` is deterministic and static for this
route. Every arm must publish one canonical digest over the complete nested
integer batch-index sequence and exact dataset `window_indices` identity.
The runner rechecks the sampler digest before and after every epoch and writes
it into the epoch receipt. P0, P1, and P2 terminals must expose equal sampler
and dataset-order digests. A session-name-only chain or an eight-batch prefix
is not sufficient evidence for the full-run matched-order claim.

P0 must enforce bitwise-equal prediction and identity digests on every step.
Every materialized parameter and Adam state must remain finite. Inactive Torch
`UninitializedParameter` objects are counted explicitly and are never treated
as trained state.

## 4. Full-result identities

The only canonical roots are:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p0_fullfull_seed42
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p1_m4_seed42
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p2_m10_seed42
```

Each cell is an independent fresh transactional graph. Alternate paths,
existing roots, symlink roots, and symlinked parents are rejected before
publication. Each attempt binds one arm and cannot be reused for another.

The attempt is immutable and published before Torch/source/checkpoint/CUDA.
After attempt, the route publishes a launch/source-authority pair, exactly 48
epoch receipts, four final-window checkpoints with sidecars, one final-four
SWA with sidecar, one manifest pair, and exactly one terminal pair. On any
error it publishes one failure pair instead of terminal; it does not retry.

The full route must use descriptor reloads and exact SHA links for downstream
edges. A terminal binds all epoch receipts, checkpoint bodies, SWA, manifest,
attempt, source authority, arm identity, launch/final closure, and the accepted
V2 predecessor.

## 5. Epoch evidence without receipt explosion

All per-step PACD invariants are hard checks. A failed check stops the cell.
The runner must not serialize 1,628,400 full step dictionaries into the
terminal. Instead, each epoch receipt contains:

- exact optimizer-step count and cumulative count;
- mean/min/max anchor, short, and combined task loss;
- mean/min/max anchor and combined encoder/decoder gradient norms from every
  ordinary paired update;
- independent short-branch encoder/decoder gradient norms and branch-gradient
  cosine only at the four fixed sentinel steps;
- valid-bin count range;
- dropout probability summary and a digest chain over every paired mask row;
- RNG-transition digest chain and zero violation counts;
- calibration-prefix digest chain and zero mutation count;
- P0 prediction/identity mismatch counts, required to be zero;
- materialized/lazy-parameter count ranges and finite-state result;
- Adam finite-state result;
- first, last, and fixed mid-epoch sentinel rows with detailed evidence;
- exact model and optimizer state digests;
- wall time, synchronized throughput, and CUDA current/peak memory.

The fixed sentinel indices are `0`, `1`, `16962`, and `33924`. They are
diagnostic only. Independent short-branch gradient geometry requires an extra
read-only autograd traversal, so it is deliberately limited to these
sentinels; running it on all 1,628,400 steps would change the measured cost
without changing the optimizer update. Every step still executes the hard
paired checks and records the natural anchor and combined gradient evidence.

## 6. Checkpoint and SWA contract

Only epochs 44-47 produce checkpoints. Every checkpoint contains arm, epoch,
state dict, model-state digest, optimizer-state digest, predecessor identity,
and current implementation identity. It is written once, sealed `0444`, and
sidecar verified before entering the manifest.

The SWA uses the same inherited final-four construction as the accepted
Cell-D/CAL-AUG producer. It must be strict-loaded into a fresh Cell-D graph and
must pass finite, eval/no-dropout, repeated-forward equality, and state-before
equals state-after checks. The terminal must not call a different arithmetic
rule silently.

Checkpoint selection is predeclared final-four only. Source loss, source R2,
within R2, external R2, or target labels cannot select an epoch, retry, arm,
or SWA.

## 7. Source-only and resource isolation

The full trainer opens only the exact source roster. Validation, test,
within-development, external sub-M, formal, and organizer-held paths remain
unresolved. Target backward, optimizer, model update, and checkpoint selection
counts are zero.

The source authority exact-checks, rather than prefix-checks, the source
behavior-normalizer semantic SHA-256
`f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391`
and T4 side-normalizer semantic SHA-256
`293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0`.

No PACD full cell may start while its selected GPU has another compute process.
For this V1 route the selected device is deliberately fixed to physical GPU0,
UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`, with
`CUDA_VISIBLE_DEVICES=0` and logical `cuda:0`. GPU1 and the active Stage-P
experiment remain outside scope. Each attempt seals the exact physical index,
UUID, PCI identity, `CUDA_VISIBLE_DEVICES`, logical device, driver, and TF32
state. It may not share a GPU, fall back to CPU, or move to another GPU after
attempt. A future dynamic-device route is a separately reviewed successor.

Luna Max may watch processes, GPU counters, logs, and immutable receipt
topology only. It may not signal, retry, edit, score, or deserialize tensors.

## 8. Implementation closure and review drift

The execution closure includes this work order, PACD paired operator, full
lifecycle/runner/CLI, and every executed loader/model/checkpoint/SWA helper.
Tests, design prose, result summaries, and reviewer comments are review-only
evidence and are not part of the launch/final execution closure. Editing a
review-only file cannot restart or invalidate a numerically unaffected live
cell.

Changes to model, data records, paired loss, dropout/RNG law, optimizer,
schedule, batch order, checkpoint/SWA rule, or executable source are execution
drift and fail closed.

## 9. Required no-data review

Before any future launch, focused CPU/no-CUDA tests must prove:

1. the exact V2 predecessor graph and semantic terminal validate through held
   descriptors, and topology/body/sidecar/mode/symlink drift rejects;
2. attempt publication occurs before runtime/data/model access;
3. P0/P1/P2 exact roots and arm identities cannot be exchanged;
4. a small synthetic multi-epoch lifecycle produces exact epoch/checkpoint/
   SWA/manifest/terminal topology;
5. failure is exclusive with terminal and cannot overwrite a prior artifact;
6. the full runner delegates to the reviewed PACD paired step;
7. step and epoch counts, LR, batch order, P0 equality, mask/RNG chains,
   sentinel rows, finite checks, and cumulative counts reject drift;
8. final-four checkpoint/SWA loading uses the inherited arithmetic rule;
9. exact initial-state load, full normalizer SHAs, full sampler/dataset-order
   digests, source-only/no-target, and selected-device evidence cannot be
   forged;
10. dry CLI imports no Torch, opens no data/checkpoint/result root, initializes
    no CUDA, and cannot execute without an opaque root authorization.

## 10. Launch boundary

This work order ends at a frozen implementation candidate. A full training
launch is a separate, explicit decision after independent root audit of live
bytes, accepted V2 lineage, dynamic device state, projected duration, and
fresh canonical roots.

P0 should launch first when authorized. A P0 lifecycle/infrastructure failure
stops the family. P1 and P2 remain mandatory scientific cells after a valid P0;
a noisy smoke loss cannot select either one out.
