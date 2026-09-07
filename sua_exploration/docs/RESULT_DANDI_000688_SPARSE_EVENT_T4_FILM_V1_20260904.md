# Result: DANDI 000688 Sparse-Event T4 Estimation + Conditional FiLM V1

Date: 2026-09-04  
Status: `IN_PROGRESS__STAGE2_RUNNING__NOT_A_FINAL_SCIENTIFIC_RECEIPT`  
Route: `dandi688_sparse_event_t4_v1`

## 1. Governing authorities

| Authority | SHA256 |
|---|---|
| `DESIGN_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md` | `56982085d4cc7c4e06d29d79a3701e8e6f6d93d08955ceff4c09737bef956705` |
| `WORKORDER_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md` | `18cc4dde507a41b3853cb6b5a6bd6f9f5c3d47b0837e26949433803ce427c6d2` |
| strict 27/6/6 source manifest | `4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9` |

This experiment is source-train/source-validation only. It does not authorize
or report formal-test, external-15, EvalAI, or target-session update results.

## 2. Causal graph status

| Stage/branch | Status | Governing evidence |
|---|---|---|
| Stage 0 constructibility/reliability | complete, both routes open | `stage0/terminal.json` |
| Stage 0 descriptive supplement | complete | `stage0_supplement_v1/terminal.json` |
| Stage 1 frozen-parent/OOD | complete, FiLM opening predicate open | `stage1_v3/aggregate.json` |
| Stage 2 matched SUA estimator | seed 42/43 complete; seed 44 pending | additive V1/V3 roots |
| Stage 2 matched SUA FiLM | seed 42/43 running; seed 44 pending | additive V2 roots |
| Stage 2 final aggregate | pending | `stage2_v3/aggregate.json` not yet published |
| Stage 3 pseudo-MUA estimator | conditional/pending | runs iff matched SUA estimator gate passes |
| Stage 3 pseudo-MUA FiLM | conditional/pending | runs iff matched SUA FiLM gate passes |
| Final result receipt | pending | `aggregate/stage3_pseudo_mua.json` not yet published |

Failed/interrupted earlier immutable roots remain part of the audit trail and
were not deleted, retried in place, or overwritten. Their successors are
governed by the corresponding additive incident documents and held admission
receipts.

## 3. Stage 0: information and constructibility gate

### 3.1 Closure hashes

| Body | SHA256 |
|---|---|
| `stage0/attempt.json` | `1218e6e7be67d2407f712714c0d7eb1b8e54625b87b83eef20b4d01233dbd533` |
| `stage0/stage0.json` | `99ee1c3c64520d503c84ee2be608ae3a2aa9f6279602120de00ded8f56006af8` |
| `stage0/terminal.json` | `bc39dd46efed12f4f27353e335a24903a3b98472944567c7139c6650c6dc2caf` |
| `stage0_supplement_v1/attempt.json` | `1cf77b38e08b4c5ab4407c5ad1611170b5b2eb1d28df50155f4676900f48fcd4` |
| `stage0_supplement_v1/supplement.json` | `02e439dd71c320147edd105bfa220645ec64b722f45f754f410826227c755b4b` |
| `stage0_supplement_v1/terminal.json` | `8167122b71ba430f09b9b8dcdb1488d3c82d3fd66f5b7d7c0fbda9208d031e43` |

The Stage-0 attempt records the exact 27 train, 6 validation, and 6 unopened
test session rosters. It also records `test_opened=false`,
`external_opened=false`, `decoder_opened=false`, and `gpu_opened=false`.
The supplement records `dense_velocity_scalars_consumed=0` and
`mask_or_gate_mutated=false`.

### 3.2 Era geometry

| Era | Train | Validation |
|---|---:|---:|
| no delay | 9 | 0 |
| short delay | 5 | 0 |
| long delay | 13 | 6 |

All route geometry, first-M10 label horizon, rank, interval containment,
candidate/audit namespace separation, and pseudo-MUA pool-then-refit checks
closed successfully in the held Stage-0 receipts.

### 3.3 Reliability

All four profile columns survived both predeclared gates:

| Column | split-half Fisher-z aggregate r | deployment-reference Fisher-z aggregate r | retained |
|---|---:|---:|---|
| `a_R` | 0.856230 | 0.865232 | yes |
| `c_R` | 0.842975 | 0.864951 | yes |
| `m_R` | 0.877809 | 0.875841 | yes |
| `delta_b` | 0.933743 | 0.916660 | yes |

Frozen retention mask: `[true, true, true, true]`. Both estimator and FiLM
routes opened before any decoder number was read. The source-only descriptor
reference check also found POST700 higher than WHOLE for aggregate `a` and `c`,
as required for estimator admission.

## 4. Stage 1: frozen-parent deployment-OOD evidence

The immutable Stage-1 V3 aggregate SHA256 is
`e41e4c1e9195ed0a3a678ab464898c3faf356238d2fff0804a8347c5d3f14382`.
These results are explicitly OOD screens and do not substitute for matched
Stage-2 efficacy.

| Contrast | mean delta R2 | bootstrap 95% lower | positive validation sessions | Interpretation |
|---|---:|---:|---:|---|
| `POST700-M10NORM - WHOLE-M10NORM` | +0.059651 | +0.028921 | 6/6 | estimator OOD screen positive |
| `SE-T4 - EMPTY` | +0.063481 | +0.032256 | 5/6 | semantic OOD screen positive |
| `SE-T4 - ROW-SHUFFLE` | +0.059860 | +0.033993 | 6/6 | row attachment OOD screen positive |
| `SE-T4 - PHASE-R` | +0.000147 | -0.002050 | 2/6 | retained `delta_b` adds no detected OOD value |

The frozen Stage-1 opening predicate is `OPEN`: semantic mean positive,
attachment mean positive, and 5/6 semantic validation sessions positive.
Stage 1 therefore authorized, but did not prejudge, the matched FiLM branch.

## 5. Stage 2: matched SUA training

Every executed arm uses 12 epochs and the fixed float64 parameter average of
zero-based epochs 8--11. There is no validation-selected checkpoint.

### 5.1 Estimator branch, incomplete three-seed view

| Seed | mean `POST700-T4 - WHOLE-T4` | positive sessions | Terminal SHA256 |
|---:|---:|---:|---|
| 42 | +0.057868 | 4/6 | `fecc7ad5fe0c0dd517918510b18c8e7a2a5c7b6ff4f96c34744c34a07889e931` |
| 43 | +0.110162 | 6/6 | `421e75ef86ad88e841573a9b004cafb235dcf266038d58579de208970b803797` |
| 44 | pending | pending | pending additive V3 terminal |

The first two seeds are descriptive partial evidence only. The design requires
the three-seed, seed-averaged six-session gate, including every-seed
nonnegative mean and a positive paired-session bootstrap lower bound.

### 5.2 FiLM branch

Seeds 42 and 43 are currently running as coordinated five-arm cells on physical
GPU0. Seed 44 starts from the first released slot. No Stage-2 FiLM score or
scientific conclusion is recorded here until its immutable three-seed
aggregate exists.

The implementation distinguishes two decisions:

- the FiLM-route gate uses the semantic and row-attachment requirements;
- the retained `delta_b` baseline claim has a separate `D_baseline > 0` gate.

This implements the frozen Outcome-E rule: a positive movement-window profile
with a null baseline contrast is not misclassified as a total FiLM failure.

## 6. Stage 3: conditional pseudo-MUA replication

Pending. A branch is instantiated only when its corresponding matched SUA gate
passes. Each admitted pseudo-MUA branch must refit T4/profile after electrode
pooling, use pseudo-MUA-specific source normalizers, run all three paired seeds
for 12 epochs, and report within-view effects. A non-admitted branch must have
no training root and must be listed explicitly by the final aggregate.

## 7. GPU/runtime evidence

Pending final closure. Every completed seed terminal binds its physical GPU
UUID and logical mapping. A read-only 10-second `nvidia-smi dmon` series for
physical GPU0 began at `2026-09-04 22:48:29 +08:00`; final utilization
statistics will explicitly describe that partial-runtime sampling interval and
will not be presented as coverage of the earlier run.

## 8. Final outcome and claim boundary

Pending the Stage-2 V3 aggregate and every conditionally admitted Stage-3
terminal. Regardless of sign, the final interpretation will keep these claims
separate:

1. POST700 versus WHOLE is a sparse carrier/rate-estimator result, not FiLM.
2. SE-T4 versus EMPTY is the FiLM profile-content result.
3. SE-T4 versus ROW-SHUFFLE is row-attachment evidence.
4. SE-T4 versus PHASE-R is the hold-to-movement `delta_b` submechanism.
5. pseudo-MUA is a conditional replication, never a prerequisite for the SUA
   conclusion and never evidence that pseudo-MUA exceeds SUA.
6. No source-development result is an EvalAI, external-15, or formal-test
   result.

## 9. Completion checklist

- [x] exact design/work-order/manifest authority bound;
- [x] Stage-0 source roster and no-test/no-external evidence;
- [x] Stage-0 per-session/per-era/per-column evidence and frozen mask;
- [x] Stage-1 three-seed frozen-parent/OOD closure;
- [ ] Stage-2 estimator seed 44 terminal;
- [ ] Stage-2 FiLM seed 42/43/44 terminals;
- [ ] Stage-2 V3 aggregate and exact gate/outcome classification;
- [ ] all conditionally admitted Stage-3 pseudo-MUA terminals;
- [ ] Stage-3/final aggregate with non-admitted branches listed;
- [ ] per-seed/per-session absolute R2 table for every executed arm;
- [ ] terminal/body/sidecar hash audit for all successful and held-failure roots;
- [ ] GPU UUID/runtime/utilization summary;
- [ ] final tests and clean completion audit.

