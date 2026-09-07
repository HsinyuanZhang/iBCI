# DANDI 000688 sparse-event T4 / FiLM Stage-1 V2 import incident and additive V3 authority

Status: `FROZEN_ADDITIVE_SUCCESSOR_AUTHORITY__SOURCE_ONLY`

Date: 2026-09-04 (Asia/Hong_Kong)

This document records a launch-environment import-path omission in Stage-1 V2
and authorizes a science-neutral additive V3 successor. It does not change the
frozen experiment design, model, data, optimizer, seed, arm, scoring, gate, or
twelve-epoch law. It does not authorize test/external data or mutation of any
earlier result root.

## 1. Parent authority chain

- Design SHA-256:
  `56982085d4cc7c4e06d29d79a3701e8e6f6d93d08955ceff4c09737bef956705`
- Work-order SHA-256:
  `18cc4dde507a41b3853cb6b5a6bd6f9f5c3d47b0837e26949433803ce427c6d2`
- Stage-0 mask-authority SHA-256:
  `7a5a4cf505a31e12b423059e8a2df0de7ff534d6301ed3409a8064584088bb54`
- Stage-1 V1 incident / V2 authority SHA-256:
  `baf7e989c5439239a1214aa083fa06ad6d825b32fc1a9b45dafeb679b853841c`

The V1 incident's one-template/deep-copy correction remains mandatory for
Stage-1 V3 and Stage-2 FiLM.

## 2. Immutable V2 failure witnesses

Each root below must contain exactly `attempt.json`, its sidecar,
`failure.json`, and its sidecar. Every leaf must be a regular file with mode
`0444` and link count one.

### Seed 42

Root:
`sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v2/sua/seed42`

- attempt body SHA-256:
  `9eff68b6cdba7a97ba660d9eff138673bb4eb360680103f1a20125ce760d8fd1`
- failure body SHA-256:
  `eefbb33d2b86e87934d1940bbc91415f0d1c2fe53dcd974ba4e2427f82e140fd`

### Seed 43

Root:
`sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v2/sua/seed43`

- attempt body SHA-256:
  `37c0ccab9835254d06df636174f26308a9fff5fa141f28571e878b3cad5b1abb`
- failure body SHA-256:
  `0f5be1c791cbba11f92c1462aab4edb4d7551ee3835992f503b5cad56250332a`

Both failures have `exception_type=ModuleNotFoundError`, exact message
`No module named 'src.models'`, and exact published prefix `attempt.json`,
`attempt.json.sha256`. They occurred after source-only CPU materialization and
before checkpoint loading, model construction, CUDA initialization, optimizer
construction, any optimizer step, or validation prediction. No test or
external file was opened.

## 3. Root cause and sole correction

The V2 launch supplied:

```text
PYTHONPATH=/home/xinyuan/Work_host/SPINT/sua_exploration:/home/xinyuan/Work_host/SPINT
```

The frozen student implementation is imported as `src.models...` and its
package root is:

```text
/home/xinyuan/Work_host/SPINT/streaming_calibration_exp
```

The only newly authorized change is the exact launch path:

```text
PYTHONPATH=/home/xinyuan/Work_host/SPINT/sua_exploration:/home/xinyuan/Work_host/SPINT/streaming_calibration_exp:/home/xinyuan/Work_host/SPINT
```

The executor must fail closed if the process does not carry this exact value.
No source file, checkpoint byte, training law, or scientific input changes.

## 4. Additive V3 roots

Fresh execution is authorized only under:

```text
sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v3/sua/seed42/
sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v3/sua/seed43/
sua_exploration/results/dandi688_sparse_event_t4_v1/stage1_v3/sua/seed44/
```

Before root reservation, held-file admission must validate the complete V1
incident/V2 authority, the two V1 failure graphs, this document, and the two V2
failure graphs. Every V3 attempt must bind the complete chain. The three V3
roots remain one-time immutable roots. No V2 root may be reused or deleted.

All later Stage-1 aggregation and Stage-2 FiLM admission must bind the V3
terminal bodies, not V2 terminal paths.

