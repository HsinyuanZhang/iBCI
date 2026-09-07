# Work Order — M2 Post-Fusion Checkpoint Score V2 Import Recovery (2026-09-02)

Status: authorized narrow successor to the immutable V1 pre-data failure.

## 1. Scope

Recover the already reviewed M2 Post-Fusion checkpoint score without changing
its scientific experiment. V2 must retain V1's exact:

- three checkpoints: `PF-MEAN`, `PF-R1`, and `PF-R50`;
- B30 / D-opt-k4 support law;
- `FIXED30` and `UNCAPPED` activity-memory laws;
- six `external_post30_local` plus seven `within_post30` sessions;
- 78-row order, last-bin variance-weighted R2, POOLED comparator, contrasts,
  bootstrap, nomination thresholds, CPU batch size, and two-hour projection
  stop;
- zero-gradient, zero-update, no-target-backprop contract.

V2 may repair only the legacy Python package-resolution seam described below.
It may not retrain, edit checkpoint tensors, change a target input, search a
hyperparameter, add a row, change a metric, or introduce a new scientific arm.

## 2. Immutable failed predecessor

V1 is a valid fail-closed predecessor at:

```text
tfpd_exploration/results/m2_postfusion_checkpoint_score_v1
```

V2 must descriptor-read this directory through a held directory file
descriptor with no symlink following and validate the exact six-leaf topology,
regular files, mode `0444`, link count one, canonical sidecar basenames, body
digests, and absence of extra leaves. Exact bodies:

```text
attempt.json  dc0ef024e657773cd58833288af524e5d2fb39af9ac7e496739133f6372f72d5
launch.json   aebce91e6a441db0c7a053e475f218b55019419abd9c4684130c760852a54fe3
failure.json  9b5dca51bea287d6620847a5e6877d0a7d59add8412027dae008b8225fdf6c04
```

The semantic validator must require that V1 reached strict load of all three
checkpoints, attempted but did not complete target materialization, produced
zero rows and no terminal, did not initialize CUDA, and recorded error digest:

```text
f96a882aa286e4363a3142459c0ffe1c30793d3864dea9efff65f3bfdedc1d85
```

No V1 leaf may be modified, removed, renamed, or used as a retry root.

## 3. Exact diagnosed defect

The failure occurs before held-out dataset construction and before R2. The
qualified import of `tfpd_exploration.src.cdm_p1_m2_local_v1` eagerly reaches
legacy imports such as:

```text
from src.causal_dual_memory_cell_d_v1 import core
```

The production process exposes the qualified repository package but does not
install `tfpd_exploration/src` as a top-level `src` package. The resulting
exception is:

```text
ModuleNotFoundError: No module named 'src.causal_dual_memory_cell_d_v1'
```

This is a namespace compatibility defect, not a data, checkpoint, CUDA, model,
memory-law, or metric defect.

## 4. Permitted repair

Implement one reviewed, deterministic import compatibility seam that resolves
the legacy replay and all of its transitive route dependencies from the exact
repository tree in both supported package contexts. The repair must:

1. avoid a second DataModule, model reconstruction, or target materialization;
2. avoid copying or changing the replay/science implementation;
3. avoid source-text rewriting at runtime;
4. avoid leaving a fabricated or partially populated top-level `src` package
   in `sys.modules`;
5. reject a preoccupied incompatible namespace rather than silently replacing
   it;
6. bind every actually imported route dependency into the V2 explicit closure;
7. preserve the original V1 import behavior for existing callers, with a
   regression for both legacy and qualified package contexts.

A package-relative/qualified compatibility import inside the historical module
is permitted if it is byte-reviewed, backward compatible, and bound only as
current V2 closure authority. A global monkeypatch, broad `sys.path` mutation,
or persistent `sys.modules` substitution is not permitted.

## 5. V2 lifecycle

Use a new canonical root and schema:

```text
tfpd_exploration/results/m2_postfusion_checkpoint_score_v2
m2_postfusion_checkpoint_score_v2
```

The opaque one-shot capability must bind the V1 failed graph, the unchanged
successful screen graph, the unchanged POOLED comparator, exact environment,
fresh V2 root identity, and current V2 closure. Validation occurs before root
reservation and again before terminal or failure.

Attempt publication precedes predecessor reads, checkpoint access, Torch import,
and target materialization. The V2 failure receipt must include the exception
class and a bounded diagnostic message in addition to its digest; it must not
again reduce a recoverable engineering failure to an opaque hash.

No automatic retry is authorized. After code freeze and independent audit,
root may issue exactly one reviewed V2 attempt.

## 6. Runtime policy

Keep the scorer CPU-only first:

```text
CUDA_VISIBLE_DEVICES=''
cuda_initialized=false
```

The same one-external-session smoke projects the remaining 13-session work. If
the projection is at most 7200 seconds, continue the same attempt. If it exceeds
7200 seconds, fail closed and separately review a GPU0 scorer with CPU/GPU
prediction and R2 parity. GPU1 is outside this work order and must not be
queried, initialized, scheduled, signaled, or monitored.

## 7. Mandatory verification before live V2

1. exact held-FD V1 six-leaf predecessor validation and adversarial tests;
2. clean-process qualified replay import with no top-level `src` pollution;
3. clean-process legacy import regression, or an exact proof that the legacy
   caller context is unchanged;
4. direct call through the repaired replay seam reaches held-out construction
   without the diagnosed `ModuleNotFoundError`;
5. existing Post-Fusion scorer focused suite remains passing;
6. attempt-first, fresh-root, one-shot capability, terminal/failure XOR,
   closure/environment/predecessor drift, and diagnostic failure-receipt tests;
7. inert public CLI and no-Torch/no-CUDA dry import;
8. no target R2 is computed during the no-data code-review phase.

Only a successful V2 score terminal with exactly 78 validated rows can support
an effect-size or nomination statement.
