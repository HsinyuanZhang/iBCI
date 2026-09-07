# M2 AJPF V1 formal-run incident

Date: 2026-09-03 HKT  
Status: `FAIL_CLOSED_BEFORE_SOURCE_AUTHORITY_AND_FIRST_UPDATE`  
Scientific result: **not evaluated**

## What happened

The independently reviewed AJPF V1 implementation was admitted for one GPU0-only formal run.  It published an immutable attempt and launch receipt, loaded the source-only PIT/Selected-T4 preparation path, and then stopped before publishing `source_authority.json`, before the 12-step smoke, and before any optimizer step.

The first exception was:

```text
AttributeError: module
'tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.physical'
has no attribute '_state_digest'
```

The exact defect is a private-helper reference in
`m2_anchored_joint_postfusion_v1/runner.py::_student_state_sha256`.  The called module exposes the screen-compatible helper as `_student_state_sha`, while the existing arbitrary-module implementation is `pit_m2_v1.trainer::_state_digest`.  The two use the same canonical tensor framing for the ASCII state keys/shapes/dtypes used here; a successor must nevertheless prove the selected student literal on a real source-only strict load rather than infer equivalence.

This is an implementation/admission failure, not evidence for or against AJPF, post-fusion, J-R1, or J-MEAN.

## Immutable V1 failure graph

Root:

```text
tfpd_exploration/results/m2_anchored_joint_postfusion_v1/training
```

The exact topology is three immutable body/sidecar pairs only:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `616e2e7b25c6d4abaef3dec487f955f1fdacd99a986121df831b4ed9b125f526` |
| `launch.json` | `129f59866935cf9caf17664989ff1ee236f14af0ae6aa797aeb310aaa87a0c6f` |
| `failure.json` | `dfebd25532f241c16131679491db19e2247fb9b310d64cac87d8be4f9fc6b269` |

All six leaves are regular, mode `0444`, and sidecar-bound.  The failure receipt records the exact published prefix `[attempt.json, launch.json]`, `target_access=false`, and the `AttributeError` above.  There is no source-authority receipt, smoke receipt, epoch receipt, checkpoint directory, manifest, terminal, score root, R², or gate result.

The launch receipt attests only logical GPU0, canonical UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`, the frozen environment, and
`target_access=false`.  GPU1 was not exposed or queried by the AJPF process.

## Successor rule

V1 is immutable and must never be overwritten or retried.  A V2 successor may change only the state-digest call and the minimum route/lifecycle identity needed to bind this exact V1 failure graph.  It must preserve the V1 science contract, three arms, paired mask/RNG law, fixed 12 epochs, optimizer settings, source-only training data, selected-support4 carrier, CPU 13-session scorer, and both preregistered external gates.

Before V2 launch, it must pass:

1. a real source-only CPU strict-load regression proving the selected student state digest equals `2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20`;
2. exact held-FD/no-follow validation of the six-leaf V1 failure graph;
3. the full no-CUDA suite, explicit closure rehash, independent root audit, and GPU0-only preflight.

