# Work Order: M2 APFG Same-Surface Target Score V2

Date: 2026-09-02  
Status: authorized once for implementation, no-data audit, and—only after independent review—one CPU-only canonical execution  
Scope: target-only successor to the immutable failed APFG V1 attempt; no source retraining and no GPU access

## 1. Purpose

APFG V1 completed its source-only scalar fit and passed its pre-registered source safety gate, but failed before publishing target scores because it compared CPU-produced `APFG-ZERO` prediction bytes with a historical POOLED prediction produced on CUDA. CPU and CUDA floating-point execution are not a bitwise-equivalent numerical surface even when the checkpoint, inputs, batch size, and TF32 settings agree.

V2 shall repair only that control error. It shall evaluate the original native POOLED operator, APFG-ZERO, and APFG-LEARNED in the same CPU process over the same rematerialized target inputs. The exact control is native POOLED versus APFG-ZERO on that same numerical surface. The historical CUDA POOLED rows remain immutable provenance and descriptive context; their prediction SHA is not a cross-device hard gate.

This work order does not authorize changing APFG science, refitting alpha, opening GPU0 or GPU1, overwriting V1, or relaxing same-input requirements.

## 2. Immutable V1 predecessor

The predecessor root is exactly:

`tfpd_exploration/results/m2_anchored_postfusion_gate_v1`

It contains exactly six immutable body/sidecar pairs (12 leaves), with body SHA-256 values:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `e546cb34f3efc32b8d6056e5eb6dc74877a9a548543d51a1040d149f1e5407a7` |
| `launch.json` | `9747223a8a5afed21e9ff59df3d0c44bea85c9a244c6095bf5afa5ace7c9dfef` |
| `source_authority.json` | `ae1f02c97d507c28fb35165ab616100156e279fa855cb3976d61ab9ff55c0ccf` |
| `alpha_selection.json` | `c60a9e29928a8f559881745a727d76716b2b458c1aa3d892772f6b8b65eef908` |
| `input_authority.json` | `41185bceb1307cebc1afb2d7672c55bc71d601ba028a4c0d694310d8af366e24` |
| `failure.json` | `6e979a4b8abc9761cf6dd9b521bfc7c4a08422ba9d775530bb0cecfc72a23f1d` |

The V2 descriptor validator must use a held directory FD with `O_DIRECTORY|O_NOFOLLOW`, body-relative no-follow opens, exact `0444`, `nlink==1`, exact sidecar basename/digest, exact topology, and no extra leaves. It must validate semantic links rather than SHA literals alone.

Required V1 semantics include:

- attempt closure `879ef63ce91da2c084285e72a34a7ba0d3a443246720f6bb0afb9e67e6d70934`;
- source safety gate passed;
- selected epoch `11`;
- source validation delta versus zero `0.002202600400827759`;
- validation per-session deltas `0.0026953303948594742` and `0.0017098704067960435`;
- refit alpha `-0.20759029686450958`;
- selected checkpoint body `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`;
- selected student state after strict load `2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20`;
- failure class `tfpd_exploration.src.m2_anchored_postfusion_gate_v1.scoring.ScoringError`;
- diagnostic message `APFG zero POOLED prediction mismatch`;
- error SHA `59297c1eebb9e534872f780b192d5e45bf3f24a2fb7f84cee38533b894bc559b`;
- target input materialization completed, but no `score.json` or `terminal.json` was published.

The historical POOLED comparator remains bound to the accepted immutable `m2_precision_cdm_v2_screen_v1/score.json` authority already validated by V1. Its receipt records CUDA, batch size 1024, TF32 disabled, and is not a CPU bitwise oracle.

## 3. Frozen V2 method

V2 performs no training and no optimizer construction.

After publishing `attempt.json`, it shall:

1. Revalidate the complete V1 failure graph.
2. Strict-load one selected-T4 POOLED model on CPU and verify the checkpoint and post-load student-state digests above.
3. Rematerialize the same 13 target session records exactly once through the reviewed PIT/DataModule and G00m materialization route.
4. Validate every rematerialized public input record against the immutable V1 `input_authority.json`, including surface/session roster, query starts, target, window count, selected support, activity authority, carrier/T4, and normalizer evidence.
5. Create three fully frozen/eval/no-dropout views from the one strict-loaded state:
   - `NATIVE-POOLED`: unwrapped historical native B3S operator;
   - `APFG-ZERO`: APFG adapter with exact IEEE `+0.0` alpha and alpha training disabled;
   - `APFG-LEARNED`: APFG adapter with exact V1 refit alpha `-0.20759029686450958` and alpha training disabled.
6. Score CPU only, decode batch size 1024, in canonical surface/session order, decode before each completed-trial commit.

Memory laws:

- `NATIVE-POOLED` is evaluated under historical `FIXED30` only.
- `APFG-ZERO` and `APFG-LEARNED` are evaluated under both `FIXED30` and `UNCAPPED`.
- Initial activity is the same B30/D-opt-k4 selected support and selected-support4 fixed-ridge carrier used by historical `m4_activity_only`.
- Activity representation is POOLED/G00m linear. PIT cubic calibration arrays are forbidden.
- Support trials are immutable. `FIXED30` evicts only the oldest completed query activity after total cardinality exceeds 30. `UNCAPPED` never evicts.

Expected row count is 65: 13 native controls plus 13 sessions × 2 APFG systems × 2 memory laws.

## 4. Governing controls and contrasts

### 4.1 Same-surface hard sentinel

For all 13 sessions, `NATIVE-POOLED|FIXED30` and `APFG-ZERO|FIXED30` must match exactly in:

- prediction SHA-256;
- target SHA-256;
- query-start SHA-256;
- window count;
- R2;
- ordered support/carrier/activity authority;
- zero parameter and target updates.

Each model must independently have identical before/after state digests. Whole-model state digests must **not** be compared across `NATIVE-POOLED` and `APFG-ZERO`, because the latter has a wrapper key prefix and one additional scalar parameter. Instead, V2 must prove that the native B3S sub-state in all three views is identical to the strict-loaded selected student state `2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20`, while separately recording exact positive-zero alpha for ZERO and the frozen V1 alpha for LEARNED.

Any prediction/input/R2 mismatch or any within-model mutation is a terminal failure. This sentinel, not the historical CUDA prediction SHA, proves that alpha `+0.0` is an operational no-op.

### 4.2 Required paired contrasts

Report paired session deltas for both surfaces:

1. `APFG-ZERO|UNCAPPED - APFG-ZERO|FIXED30` (memory-only effect);
2. `APFG-LEARNED|FIXED30 - APFG-ZERO|FIXED30` (bounded gate effect);
3. `APFG-LEARNED|UNCAPPED - APFG-ZERO|UNCAPPED` (uncapped gate effect);
4. `APFG-LEARNED|UNCAPPED - NATIVE-POOLED|FIXED30` (total deployed effect).

For each, publish mean, median, positive count, all per-session deltas, and a deterministic 10,000-draw session bootstrap using seed 42. Historical CUDA POOLED versus same-process CPU native POOLED may be reported only as a descriptive numerical-surface comparison.

### 4.3 Sole promotion gate

The only promotion gate is the external total deployed effect:

- mean paired delta `>= +0.010`; and
- at least `4/6` external sessions strictly positive.

Within-session numbers, source validation, bootstrap intervals, memory-only effects, and historical cross-device differences cannot independently promote the method.

## 5. Lifecycle and resource law

Canonical result root:

`tfpd_exploration/results/m2_anchored_postfusion_gate_v2_same_surface_control`

Execution is CPU-only. Exact environment must include:

```text
CUDA_VISIBLE_DEVICES=
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONPATH=/home/xinyuan/Work_host/SPINT
```

The public CLI is inert. A root-owned opaque one-shot capability must bind the canonical fresh result root, parent inode, exact environment, V1 predecessor graph, historical POOLED authority, and current explicit/no-glob implementation closure. Attempt publication precedes checkpoint, DataModule, target, or Torch runtime work. Final and failure paths revalidate the predecessor, environment, closure, and reserved root identity.

Success publishes immutable body/sidecar pairs for at least:

- `attempt.json`;
- `predecessor_authority.json`;
- `input_authority.json`;
- `score.json`;
- `terminal.json`.

Failure publishes `failure.json` while preserving the honest immutable prefix; terminal and failure are mutually exclusive. No automatic retry is authorized.

This work order authorizes exactly one canonical V2 CPU execution only after focused no-CUDA tests, compile, inert dry-run, closure recomputation twice, and independent audit pass. It never authorizes GPU0 or GPU1 access and must not query or interfere with the concurrent M1 job.

## 6. Interpretation

Possible outcomes are pre-registered:

- gate passes: APFG is a meaningful PF improvement over the strongest same-input POOLED control;
- gate misses but learned gate is positive: report a bounded residual signal, not a promoted method;
- same-surface ZERO sentinel fails: implementation/control failure, no scientific conclusion;
- learned arm is null or negative: close scalar APFG as insufficient, preserving the source-only positive result as non-transfer evidence.

No result may be called an improvement merely because it beats PF-MEAN, PF-R1, or PF-R50. The governing comparator is same-process native POOLED under `FIXED30`.
