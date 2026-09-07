# Work Order: CDM-D Matched Score V6 Reserved-Root Successor

Date: 2026-08-26  
Status: implementation and CPU/no-data/no-CUDA test authority only  
Scientific role: execute the unchanged V5 12-cell performance matrix after repairing one lifecycle ordering bug

## 1. Scope

Build a narrow V6 successor for the accepted CDM-D matched scorer. Do not
change the scientific experiment, target sessions, supports, query windows,
model, weights, normalizers, metric, transition rule, gates, or cell order.

The only production defect to repair is:

```text
reserve fresh score root
-> shared lifecycle asserts that the same root is absent
-> fail before attempt publication
```

The correct order is:

```text
validate prospective freshness
-> reserve and hold the exact score root
-> validate its named identity, parent identity, topology, and emptiness
-> publish immutable attempt
-> prepare model/data/CUDA
```

No target, checkpoint tensor, or CUDA action may occur before the attempt.

## 2. Historical V5 launch boundary

Preserve the existing V5 authority and empty score root. Do not delete,
rename, populate, or reuse either root.

V5 authority:

```text
root:
  tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v5

official_preflight.json:
  8065b303e3f4608e586b119928336d7bd59686e6f18b01d5213cc23f8c8e2f6b

root_authorization.json:
  7d4ddc1775dc4cf4c9c3c74c4d1f44b3c7748043266e6de694025e676c112df3

identity:
  f2eae715b44fdd6e3c8b1e74450cad128045905d63a4d805ecb5ad14efb01ffd

V5 implementation closure:
  8573cb21c00adaf86e00a756e3803fd7e78bed5636e6f5775868cd90908f5946
```

Failed V5 score reservation:

```text
root:
  tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v5

directory device/inode/mode:
  64512 / 28863194 / 0755

exact child topology:
  empty

attempt published:
  false

target or checkpoint opened:
  false

CUDA initialized:
  false

GPU1 post-failure:
  idle, 23 MiB, no compute process

launch log SHA-256:
  a5d338f679af2021c47c5b73fe9201e7e2650f0f1dbabc069f3c36f7b8f9f533

failure:
  V5 prospective root already exists after V5 itself reserved it
```

V6 must descriptor-check the two V5 authority pairs and the exact empty V5
score directory before it reserves any V6 output. This is historical lineage,
not a reusable execution authority.

## 3. Accepted scientific predecessor

Retain the exact V5 strict-27 source gate and the corrected semantic
interpretation from the V5 score work order:

```text
source-gate terminal:
  e2e07244e1021137ba806f8f0156c77474b18608e747ef413e38b5c51a1976ff

source-gate binding:
  e08abc3214fbfeeee2444b9f3b1a227c6cd1185763aeaee99436e1fd853a1e56

pass / total / minimum:
  M30 27 / 27 / 14
  M10 27 / 27 / 14
  M4  26 / 27 / 14
```

The V6 identity and authorization must reload this exact predecessor. Do not
derive a new source result or rerun the source gate.

## 4. Implementation boundary

Prefer one small backward-compatible shared-lifecycle hook that validates an
already-reserved `ArtifactRoot`. It must exact-check:

- the held artifact type;
- the route-specific canonical path;
- the route-specific topology;
- named directory identity;
- parent directory identity;
- empty topology before attempt;
- no symlink, inode replacement, or extra leaf.

The hook must receive the actual held artifact, not re-open a caller-provided
path and not merely assert that a directory exists. V1/V5 default receipt and
science behavior must remain unchanged. A V6 caller may not replace the hook
with an arbitrary mapping, environment value, monkeypatch, or `sys.modules`
substitution.

Create an additive V6 package, CLI, and focused test. Use fresh roots:

```text
tfpd_exploration/results/causal_dual_memory_cell_d_score_authority_v6
tfpd_exploration/results/causal_dual_memory_cell_d_matched_score_v6
```

V6 may inherit or compose V5 plan/score/physical code only through exact typed
seams. Do not copy the evaluator or lifecycle.

## 5. Unchanged experiment

Run exactly the V5 matrix:

```text
M30 -> M10 -> M4
for each budget:
  within-6 sealed Cell-D
  within-6 CDM-D
  external-15 sealed Cell-D
  external-15 CDM-D
```

Always complete all 12 cells unless integrity or execution fails. A negative
performance delta is a valid result and must not skip a later budget.

All V5 science rules remain exact:

- common post-first30 query pool;
- M4 D-optimal four, M10 chronological ten, M30 chronological thirty;
- fixed-ridge-by-trial initial carrier, normalized lambda 0.1;
- FIFO capacities M4=26, M10=20, M30=0;
- `IndependentActivityCausalDualMemory` and `commit_independent()`;
- predict trial j before committing trial j;
- target labels used only for final last-bin R2;
- same sealed Cell-D SWA, normalizers, inputs, masks, targets, and model state;
- zero target gradient, backward, optimizer, update, and dropout calls;
- complete transition/resource evidence and atomic score+terminal publication.

Keep the V5 external gates unchanged.

## 6. Required tests

Add exact no-data tests for:

1. the historical V5 authority pair and empty-score-root lineage;
2. prospective V6 freshness before reservation;
3. exact held V6 artifact identity/topology/emptiness after reservation;
4. attempt publication before prepare/materialize/CUDA-reachable action;
5. named-root replacement, parent replacement, symlink, extra leaf, preexisting
   attempt, and wrong topology rejection;
6. a complete synthetic V6 lifecycle reaching atomic score+terminal;
7. failure publication after a post-attempt prepare failure;
8. unchanged complete M30/M10/M4 matrix and V5 science payloads;
9. unchanged V1 and V5 regressions.

Use no user site, no CUDA, disabled bytecode/plugin autoload, and one CPU thread.

## 7. Stop boundary

Terra must stop after frozen code, tests, dry CLI, explicit no-glob closure,
and fresh-root checks. Terra may not read NWB/checkpoint tensors, initialize
CUDA/GPU, mint V6 authority, reserve the V6 score root, or launch.

Root will independently audit the historical V5 boundary, current bytes,
accepted source gate, fixed evaluation metadata, device availability, and both
fresh V6 roots before a separate launch decision. A long V6 run, if launched,
will be handed to Luna for read-only 30-minute monitoring.
