# M2 AJPF V3 score incident: cross-device bitwise sentinel

Date: 2026-09-03 HKT  
V3 status: `FAIL_CLOSED_BEFORE_SENTINEL_RECEIPT_AND_AJPF_ROWS`  
AJPF held-out result: **not evaluated**

## What happened

AJPF V3 reused the sealed, successful V2 twelve-epoch checkpoints and launched a
fresh CPU-only score attempt.  It published only `attempt.json` and
`v2_authority.json`, then stopped before `sentinel_authority.json`,
`input_authority.json`, any new AJPF score row, either preregistered gate, or a
terminal receipt.

Immutable V3 bodies:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `3163e7dab31c83a77bd69830db753ea87828fb9f8119dc28ab6a8637215cdc32` |
| `v2_authority.json` | `a142f60a607f10206a50c2cb4d9021d08ffbec4007381e382c81030f7282e43b` |
| `failure.json` | `a1ee808841e699842d361a889d9945d35340efad97c11ca87d97dd3a4f2b69b1` |

The failure was:

```text
RuntimeError: V3 13/13 sentinel authority failed
```

## Bounded read-only reproduction

After the formal process exited, the exact V3 `detailed_sentinel` computation
was rerun read-only under the same frozen CPU environment.  It compared the
five declared fields for every one of the 13 materialized records.

Observed result:

| Field | Exact matches |
|---|---:|
| target SHA | 13/13 |
| governed query-start SHA | 13/13 |
| window count | 13/13 |
| prediction SHA | 0/13 |
| scalar R2, bitwise equality | 0/13 |

Despite the bitwise failures, the largest absolute R2 difference was only
approximately `1.32e-7`.  Thus the session roster, targets, governed windows,
and window counts are exactly the same; only floating-point predictions differ
at a numerically negligible level.

## Root cause

The sealed historical POOLED authority
`m2_precision_cdm_v2_screen_v1/score.json` (body SHA
`455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6`)
records an NVIDIA RTX 3090 CUDA execution with batch size 1024 and deterministic
cuBLAS settings.  AJPF V3 deliberately runs the new scorer on CPU.  Requiring a
CPU reconstruction to have the same prediction bytes and scalar R2 bits as the
historical CUDA execution is therefore a cross-device bitwise contract, not a
scientific same-input test.

This incident does **not** show that AJPF is worse.  No AJPF held-out row or
promotion gate was evaluated.

## Permitted successor

A final score-only successor may not retrain or modify the sealed V2
checkpoints, historical comparator, AJPF rows, or preregistered gates.  It may:

1. bind this V3 failure graph, the V2 successful training graph, and the
   historical GPU POOLED score graph exactly;
2. publish its attempt before opening Torch, data, targets, or checkpoints;
3. verify exact equality of the 13 record keys, target SHA, governed-start SHA,
   and window count;
4. explicitly disclose the historical-GPU versus current-CPU execution law and
   publish all 13 paired R2 values and absolute differences;
5. require a preregistered numerical bridge of maximum absolute R2 difference
   at most `2e-7` across all 13 records;
6. compute the unchanged 78 AJPF CPU rows and unchanged gates only after that
   bridge passes, using the sealed historical POOLED R2 values as the practical
   comparator.

Prediction-SHA equality must remain reported as 0/13; it must not be relabeled
as exact.  The numerical bridge is a new, explicit cross-device validation
contract justified by the observed device mismatch, not a rewrite of V3.
