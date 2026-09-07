# Work Order: TF-SR Phase-D Successor v2 after the Epoch-Proof Digest Failure

Owner of design, review, and launch authorization: root/Sol  
Implementation owner: Terra  
Runtime monitor after launch: Luna, read-only every 10 minutes  
Status: code repair and no-data tests are authorized; GPU/data launch is not authorized by this document

## 1. Predecessor failure is immutable evidence

The first Phase-D attempt is terminally failed and must never be deleted, edited, copied into a new
root, or described as an epoch completion:

```text
root     tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v1/
failure  failure.json
SHA-256 997be82b3fdaf3774fba9c1428fd9526ba7dd56e6cbf3a2734fd917ef4c5f0a4
status   optimizer_steps_completed=33924; no epoch/checkpoint/SWA/terminal
```

All four v1 body/sidecar pairs are regular, non-symlink, mode `0444`, and exact. The failure occurred
after the first epoch's final optimizer update entered the boundary-proof path, while serializing the
Adam optimizer state:

```text
_optimizer_digest
  -> _jsonable_optimizer
  -> _tensor_digest
  -> scalar_float_tensor.view(torch.uint8)
RuntimeError: self.dim() cannot be 0 to view Float as Byte
```

Because no checkpoint was durably published and the process state is gone, v2 starts from the exact
canonical seed-42 initial state. It may not infer or reconstruct a resume point from the v1 counter.

## 2. Scientific treatment is unchanged

This is a receipt/provenance implementation repair only. The following remain bit-for-bit or
semantically identical to the reviewed v1 treatment:

- model graph and initialization;
- strict-27 source roster and M30 normalized T4 authority;
- seed 42, batch 32, 48 epochs, 33,925 steps per epoch;
- Adam, warmup-cosine schedule, loss, dropout law, and data order;
- checkpoint epochs 44--47 and final-four SWA;
- physical GPU1 / logical `cuda:0` device contract;
- no validation, target, formal, scoring, or target update.

Do not change `model.py`, `contract.py`, `source_smoke.py`, any sealed receipt, or the main scientific
handoff. Do not use the repair as an opportunity to change performance behavior.

## 3. Exact repair

`_tensor_digest` must continue to hash the sorted tensor key, dtype string, original shape, and raw
contiguous bytes. Before reinterpreting storage as `uint8`, reshape the detached CPU-contiguous tensor
to a one-dimensional element vector. This makes 0-D tensors legal while preserving byte order and the
existing digest for every non-scalar tensor. Empty, scalar, vector, matrix, integer, floating, and
boolean tensors must be covered by tests.

The Adam optimizer-state digest must be tested after a real CPU optimizer step, including its scalar
`step` tensor and non-scalar moment buffers. No full model/optimizer scan may move back onto ordinary
steps: proof remains requested exactly once at each epoch boundary and never during the first 100
throughput steps.

## 4. Successor topology and lineage

The only legal successor root is fresh and currently absent:

```text
tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_train_v2/
```

The production identity, attempt, launch, every epoch/checkpoint, SWA, failure, and terminal receipt
must bind:

- the canonical v1 failure path and exact body SHA above;
- a declaration that v1 supplied zero accepted checkpoints and is not resumed;
- this work order and its exact SHA;
- the complete current Phase-D implementation closure;
- the unchanged accepted Phase-C source-smoke authority.

The v2 loader must reject the v1 root as a canonical output alias. The v1 failure is an input lineage
authority only. Any existing v2 body, sidecar, symlink, partial, or directory alias fails before source
resolution or CUDA initialization.

## 5. Required no-data tests

Before root review, isolated tests must prove:

1. scalar and non-scalar `_tensor_digest` behavior and non-scalar backward compatibility;
2. deterministic Adam state digest after a real CPU step;
3. the exact epoch-boundary proof path completes with Adam scalar state;
4. ordinary steps request no proof and the first 100 steps contain no full-state scan;
5. v1 failure body/sidecar/path/mode/SHA and zero-checkpoint topology are exact;
6. missing, copied, symlinked, mutable, malformed, or wrong-SHA predecessor evidence fails closed;
7. every v2 receipt binds predecessor lineage and current launch/final closure;
8. canonical v2 output freshness is checked before source/device work;
9. dry CLI remains no-data/no-CUDA/no-write, and either execution flag alone fails early;
10. the complete mock lifecycle still produces exactly one terminal or one failure topology.

Run a small real CPU optimizer-boundary regression. Do not open NWB files and do not run CUDA during
implementation or review.

## 6. Launch gate

Terra reports a frozen patch, exact SHAs, focused tests, dry plan, predecessor verification, and v2
root freshness. Root independently reviews and reruns the boundary regression. Only root may then
mint/accept a successor preflight and authorize the exact two-flag v2 launch. After launch, Terra stops
editing the bound closure and Luna resumes read-only checks every 10 minutes. Luna never repairs,
restarts, stops, or mutates the run.
