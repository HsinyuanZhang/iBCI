# Work Order — M2 A0 Deterministic-cuBLAS Successor V2 (2026-09-02)

## 1. Purpose and scope

This work order authorizes one narrow successor to the failed M2 A0 V1
external attempt. The failure occurred before a completed model forward because
PyTorch deterministic algorithms require a pre-process cuBLAS workspace
configuration on CUDA >= 10.2.

V2 changes only the launch/environment and lineage contracts. It must preserve
the exact V1 science: M10 chronological carrier `[0..9]`, sealed first-30 cubic
B3S activity seed, true-trial cubic commits, raw 100-bin chunk phase origin,
W50 governed decoding, four arms, frozen batch size 32, model/checkpoint,
normalizer, query rows, metrics, gates, and zero-update/zero-gradient laws.

## 2. Exact V1 failure predecessor

The V1 external root is immutable and must never be deleted, rewritten, or
reused:

`tfpd_exploration/results/m2_a0_chunk_noninferiority_v1/external`

Its exact bodies are:

- `attempt.json`:
  `a417a5d9b33e0d5edf7aa415c26b4012808e3a939202645aef2ff840ef6e2cd0`
- `launch.json`:
  `33942ddcf9d8f856926537fb7ced39f3511e3e569bffda415f5ef1f8623373fc`
- `failure.json`:
  `ca1fd1b8077285c9010de4c7a772620051e64d35f4fdcd23f582faaf6e173093`
- V1 execution closure:
  `35890d88e27de8809b87baaacfe43bb004a15775ac7ea99916771996c957b5d4`

The held-descriptor validator must require the exact six-leaf topology,
regular `0444` single-link bodies and sidecars, canonical sidecar digests,
attempt/launch/failure links, `status=FAIL_CLOSED`, `stage=replay`,
`parameter_updates=0`, `target_gradients=0`, and the deterministic-cuBLAS
missing-workspace error semantics. No terminal may exist.

## 3. V2 roots and lineage

V2 uses fresh roots only:

- `tfpd_exploration/results/m2_a0_chunk_noninferiority_v2/external`
- `tfpd_exploration/results/m2_a0_chunk_noninferiority_v2/within`
- `tfpd_exploration/results/m2_a0_chunk_noninferiority_v2/aggregate`

Every V2 shard capability and terminal/failure must bind both:

1. the exact accepted historical C-Pre V2 witness from
   `WORKORDER_M2_A0_HISTORICAL_CPRE_BINDING_V1_20260902.md`; and
2. the exact V1 external failure predecessor above.

V2 must have a distinct route/root identity. It may reuse the reviewed V1
physical/science implementation through a typed immutable execution profile;
it must not copy the replay loop or mutate global plan constants.

## 4. Deterministic CUDA environment

Before Python/Torch/CUDA/model/data/checkpoint access, the root-only V2 issuer
must exact-validate:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
```

The launch receipt must record this mapping. The terminal/failure path must
revalidate it. The pinned physical device remains GPU0 only, canonical UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`; GPU1 remains forbidden and must
not be enumerated, queried, initialized, or used.

## 5. Tests and launch order

Before live V2 admission, no-CUDA tests must prove:

- missing or altered `CUBLAS_WORKSPACE_CONFIG` rejects before root reservation;
- the exact V1 failure graph is accepted and any body/sidecar/mode/topology,
  semantic, or closure drift rejects;
- V1 and V2 roots cannot alias;
- V1 science inputs and V2 science inputs are byte/shape/digest identical;
- V2 current closure is separate from the historical V1 closure;
- public CLI remains inert and cannot mint a capability.

Live order is fixed: run V2 external alone on GPU0; audit its terminal, rows,
GPU peak, and throughput; only then decide whether to run V2 within. No
concurrent second M2 CUDA process is authorized by this work order.

