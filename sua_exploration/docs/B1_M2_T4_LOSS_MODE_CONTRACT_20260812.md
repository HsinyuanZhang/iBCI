# B1 — M2 carrier-content × identity-distillation contract

**Date:** 2026-08-13
**Status:** independent-audit-fix candidate v8, **not yet minted as an official
preflight. Authorizes no GPU run.** The v7 preflight is quarantined for launch
purposes: a direct scorer invocation did not prove that `--out` was the exact
`future_score_receipt` reserved in the immutable launch contract. The successor
requires the scorer to reopen the post-training binding and future launch
contract, require that exact resolved path, and recheck that both its body and
sidecar remain absent before any artifact audit, Torch/Lightning import, data
open, or forward pass. A fresh official preflight must be regenerated from the
reviewed live B1 tree before any cell can run. The T4-only v1 sweep and its
unattainable exact-p gate are withdrawn before launch. The reviewed implementation
now includes an additive all-epoch scorer and immutable launch
plan, explicit post-training binder, and CPU-only real-checkpoint construction smoke,
but the runner remains deliberately **GPU-refusing** pending independent
review. A best-validation checkpoint must never replace the fixed
epoch-5--12 estimator.
**Screen ID:** `m2_carrier_distillation_interaction_v2`
The immutable CPU preflight binds an explicit entrypoint/config allowlist plus
the recursive local-Python import closure, including every eagerly imported
vendored streaming dependency. It intentionally excludes unrelated experiments,
tests, mutable documents, logs, and results. A later change or omission inside
that runtime/configuration closure invalidates the preflight; a new unimported
file does not.

**Scope note (collision avoidance):** The active additive scaffolding is under
`streaming_calibration_exp/` (`b1_m2_matched_z4_datamodule.py`, four `b1_v2_*`
configs, `preflight_b1_m2_factorial.py`, `run_b1_m2_factorial.py`,
`aggregate_b1_m2_factorial.py`, and tests). It changes neither the A2-v2
bound files nor shared trainer/model code. The scorer binds the full science
configuration, source/split/normalizer provenance, teacher byte hash, and
every stored epoch-004--011 checkpoint. Because Hydra checkpoints live in the
explicit `hydra.run.dir` while `train.py` metadata lives in a separately named
artifact directory, a post-training immutable path binding must link those
two directories before scoring. Each future receipt also binds the absolute Python
interpreter, `src/train.py` path, and `streaming_calibration_exp` working directory;
relative teacher paths are resolved only against that bound working directory. The
runner still refuses `--launch`
until independent review explicitly changes that status.

**One-cell executor boundary.** `scripts/execute_b1_m2_factorial_cell.py` is
an additive, default-dry-run executor for a *future* independently reviewed
run. It accepts exactly one predeclared `(stage, fold, seed, carrier,
loss-mode)` cell and exactly one explicit CUDA index. Before any subprocess it
verifies the immutable official preflight against live source bindings and
writes an O_EXCL pre-execution contract. That contract reserves every immutable
output path, including post-training binding and score body/sidecars, before a
subprocess can start. It then records distinct immutable
start and completion receipts; the latter carries the actual subprocess exit
code, exact invoked command, absolute working directory, cell, CUDA device, and
the controlled environment fields. This prevents a pre-execution receipt that says `training_started=false`
from being misused as execution evidence. A successful post-training binder
also requires the linked start and completion receipts, committed execution log,
and a zero exit code before it will bind artifacts. The scorer independently
reopens and verifies the same launch→start→completion chain rather than trusting
the binding alone. Every launch contract must contain non-null post-training-binding,
score, start, completion, and log paths; the legacy runner can no longer prepare launch receipts. The executor itself does not
authorize a GPU allocation, and this document does not mint an official
preflight.

---

## Claim under test (must be able to fail)

On FALCON M2 development **internal-LOSO validation**, removing the identity-distillation term increases
the **carrier content gain** `(T4 - Z4)`. This interaction, rather than a T4-only improvement,
is required before claiming that an activity-only teacher suppresses functional carrier content.

---

## Frozen staged matrix

| Carrier content | Loss mode | `lambda_y` | `lambda_E` | Seeds | Fold |
|---|---|---:|---:|---|---|
| T4 | `task_plus_y` | `1.0` | `0.0` | 42, 43, 44 | M2 internal LOSO fold 0 |
| Z4 | `task_plus_y` | `1.0` | `0.0` | 42, 43, 44 | same |
| T4 | `task_plus_y_plus_E` | `1.0` | `0.1` | 42, 43, 44 | same |
| Z4 | `task_plus_y_plus_E` | `1.0` | `0.1` | 42, 43, 44 | same |

**Stage P (pilot):** `2 carrier arms x 2 loss modes x 3 seeds x fold 0 = 12`
fresh cells. This is a **single fixed-LOSO-session, three-seed descriptive
pilot**, not session-level inference.

**Stage F (predeclared expansion):** folds `{1,2,3}` are frozen before Stage-P
results: `2 x 2 x 3 x 3 = 36` additional fresh cells. Stage F can occur only
if Stage P's immutable aggregate simultaneously has mean interaction
`>= +0.03` and all three seed interactions `> 0`; otherwise B1 stops. **Stage P
is routing only and is never included in the terminal test.** Stage F is the
confirmatory primary: its predeclared folds `{1,2,3}` form exactly three fresh
LOSO sessions × three seeds (36 fresh cells). `P+F` may be displayed only as a
labeled descriptive sensitivity, never as a terminal or confirmatory result.
No result can choose alternative folds, parameters,
or an additional arm.

An existing checkpoint may replace a cell only if a preflight proves exact
protocol/config/data/epoch/seed identity and binds its SHA; otherwise train
fresh. `task_only` is not part of this factorial.

### Frozen protocol

| Parameter | Frozen value |
|---|---|
| Variant | B3S |
| Side feature group | `t4` for both arms; Z4 uses the B1 wrapper to apply `mask_standardized_t4` only after the ordinary T4 fit and source normalizer |
| Experiment base | `streaming_calibration_exp/configs/experiment/b3s_t4_m2_loso_internal.yaml` semantics |
| Calibration | chronological first `33` trials, `random_calibration=false` |
| Epochs | `12`, no early stopping |
| Loss overrides | `gate2_matrix.py::loss_overrides` |
| Stage P validation | one M2 internal-LOSO fold-0 session |
| Stage F validation | predeclared M2 internal-LOSO folds 1, 2, 3 |
| Formal EvalAI held-out | not opened in this matrix |

---

## Frozen acceptance rule

Primary contrast (per seed and session):

```text
interaction = (T4 - Z4)_task_plus_y - (T4 - Z4)_task_plus_y_plus_E
```

| Gate | Condition |
|---|---|
| Stage-P practical gate | mean interaction `>= +0.03` |
| Stage-P seed consistency | all three fold-0 seed-level interactions `> 0` |
| Stage-P interpretation | descriptive only; it cannot support a session claim |
| Stage-F trigger | both Stage-P conditions above, from its immutable aggregate |
| Final inference | crossed 3 predeclared Stage-F LOSO sessions × 3 seeds; Stage P is routing-only |
| Final terminal rule | **Stage F only:** mean interaction `>= +0.03`, all 3 session means `>0`, all 3 seed means `>0`; two-way bootstrap CI is descriptive only |
| P+F report | explicitly labeled descriptive sensitivity only; it never receives the terminal rule |
| Matrix integrity | all four arms match session/query/normalizer/epoch policies |

Kill criterion: if Stage P does not pass its two practical/seed rules, stop
B1 and drop the claim that identity distillation specifically suppresses
carrier content. This outcome does **not** foreclose B2 carrier forcing or B3
estimator-noise augmentation; those test independent mechanisms.

Secondary descriptive: report T4 and Z4 main effects of removing `lambda_E`. A generic improvement
shared by T4 and Z4 is training-objective evidence, not carrier evidence. Only after a positive
full P+F interaction may a separately contracted `task_only` arm ask whether `lambda_y` is also
harmful.

---

## Outcome authorization map

| Outcome | Authorizes | Forecloses |
|---|---|---|
| Stage-P practical/seed pass | **Only** predeclared Stage F folds 1--3 | — |
| Stage-P fail/uncertain | B2/B3 remain independently testable | B1 carrier-specific hypothesis |
| Full P+F interaction pass | Distillation-confound claim; optional separately contracted task-only stage | — |
| Both T4 and Z4 improve equally without E loss | Generic objective improvement only | Carrier-specific interpretation |
| All four arms tie within uncertainty | Loss mode not the limiter | B1 as root cause |

---

## Data isolation boundary

M2 internal development validation sessions only. No EvalAI submission endpoint scoring in this matrix.
Support-trial direction labels are read only for the T4 carrier (including the
matched Z4 pre-mask); query behavior is loaded only for validation scoring. It is
not used for gradients, carrier/normalizer fitting, or checkpoint selection.
During B1 there are no target-session weight updates. The inherited frozen teacher
was historically pretrained with the B1 validation session, and no clean-teacher
target exclusion is claimed; the whole matrix is therefore **internal development
only**, not an external held-out test.

---

## GPU cost

| Item | Count |
|---|---:|
| Stage P training cells | 12, fewer only with exact checkpoint reuse |
| Stage F training cells | 36 only after immutable Stage-P gate pass |
| Terminal confirmation | 36 Stage-F cells; no estimate is a result and no best-val reuse is legal |
| Descriptive sensitivity | 48 cells including the routing pilot; never terminal |

---

## Implementation bindings

| Artifact | Path |
|---|---|
| Matched Z4 | `streaming_calibration_exp/src/data/b1_m2_matched_z4_datamodule.py` — ordinary T4 label fit/normalizer, then post-standardization zero mask |
| Four configs | `streaming_calibration_exp/configs/experiment/b1_v2_m2_{t4,z4}_{task_plus_y,task_plus_y_plus_E}.yaml` |
| Fixed epoch retention | `streaming_calibration_exp/configs/callbacks/b1_epoch_window.yaml` |
| Preflight / executor / binder / scorer / core / aggregate | `streaming_calibration_exp/scripts/preflight_b1_m2_factorial.py`, `scripts/execute_b1_m2_factorial_cell.py`, `scripts/bind_b1_m2_factorial_post_training.py`, `scripts/score_b1_m2_factorial_epochs.py`, `src/metrics/b1_m2_factorial.py`, `scripts/aggregate_b1_m2_factorial.py` |
| Loss machinery reference | `streaming_calibration_exp/src/models/streaming_calibration_module.py` and `src/metrics/gate2_matrix.py::loss_overrides` |
| Runtime closure | Exact equality for the reviewed B1 entrypoints, selected Hydra config closure, and recursive local imports (including `src/data/validation_protocol.py` and imported vendored `third_party/` files). The artifact's broad full-tree `source_manifest.json` remains byte-hashed; unrelated extra rows are recorded with an extras digest/list but do not constitute scientific drift. Missing or changed required rows fail closed. |
| Launch state | **NO-GO:** `run_b1_m2_factorial.py --launch` always refuses and has no receipt-preparation bypass; only the explicit one-cell executor can create a future O_EXCL launch/start/completion/log chain, followed by the mandatory post-training binding and independent scorer verification |

**This document authorizes nothing.** No launch until the v4 configs,
immutable preflight, all-epoch scorer, full artifact binding, real-format CPU strict-load smoke, and interaction
aggregator pass independent review.
