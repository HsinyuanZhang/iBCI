# Code Interface

The comparator's public surface. Signatures verified by introspection on 2026-08-13.

Design rules: everything is CPU-only by default, no scoring path can run without passing two gates,
and every dataset binding goes through a sealed loader rather than a glob.

---

## Running it

```bash
source cebra_exploration/scripts/cebra_env.sh     # sets CEBRA_PY, PYTHONPATH, PYTHONNOUSERSITE
cd cebra_exploration

CUDA_VISIBLE_DEVICES= "$CEBRA_PY" -s scripts/run_cebra_comparator.py --dry-run
CUDA_VISIBLE_DEVICES= "$CEBRA_PY" -s scripts/run_cebra_comparator.py --positive-control
CUDA_VISIBLE_DEVICES= "$CEBRA_PY" -s -m pytest tests/test_cebra_comparator.py -q
```

| Flag | Effect |
|---|---|
| `--dataset {subject_m,rt,falcon_h1,falcon_m2}` | restrict to one dataset |
| `--view {sua,pseudo_mua}` | subject-M view |
| `--dry-run` | resolve paths and report readiness without opening neural arrays. **Start here.** |
| `--audit-only` | Part A constructibility audit, no fitting |
| `--positive-control` | run the required latent-recovery gate over all arms |
| `--write-receipt` | emit the immutable JSON receipt |
| `--authorise-scoring` | **refused.** Deliberately inert; scoring needs coordinator authorisation |

`PYTHONNOUSERSITE=1` is mandatory — a user-site `torch 2.12.0+cu130` otherwise shadows the env and
reports `cuda False`.

## Arms

`ARMS` in declaration order. Each carries a bias declaration in `ARM_BIAS`; none may be reported
without it.

| Arm | Status on the gate | Bias |
|---|---|---|
| **`cebra_joint_behavior`** (`PRIMARY_ARM`) | passes, target ≈ source | favours CEBRA on accuracy (target prefix is inside the fit); favours us honestly on cost |
| `cebra_frozen_source_adapt` | passes | the fairest analogue of us — source frozen, cross-session sampling kept |
| `cebra_adapt_unaligned` (`NEGATIVE_CONTROL_ARM`) | **must fail** | declared negative control. Never presentable as CEBRA's best |
| `cebra_no_adapt` | `CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH` | n/a |
| `cebra_joint_time` | `CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY` | n/a |
| `cebra_joint_time_query_unlabelled` | same | n/a |

The two `*_time` arms are undefined by construction, not unimplemented: sklearn multi-session CEBRA
raises `No label: labels are needed for alignment in the multisession implementation`, and supplying
time indices would invent a cross-session clock correspondence.

Decoders: `PRIMARY_DECODER = "linear_ridge"`, `SECONDARY_DECODER = "knn"`. **See finding F15** — the
linear-primary choice can understate CEBRA the better it is trained, and must be revisited before
scoring.

## The two gates

Both must pass before any number is interpreted.

```python
run_positive_control_gate(*, arms=None, max_iterations=250, max_adapt_iterations=250) -> dict
```
Builds three sessions of `POSITIVE_CONTROL_N_CHANNELS = (24, 31, 37)` from one shared 2-D latent,
fits a readout on **source embeddings only**, and requires target R² ≥ `0.70`
(`POSITIVE_CONTROL_MIN_TARGET_R2`) while the negative control must stay below `0.20`
(`POSITIVE_CONTROL_UNALIGNED_MAX_TARGET_R2`). Asserting both directions is what gives the gate teeth:
it proves the harness can detect the F8 failure rather than merely not exhibiting it.

```python
integrity_gate(dataset, *, reproduced=None, view=None) -> dict
```
Reproduces the dataset's sealed reference from `SEALED_REFERENCES` within `INTEGRITY_ATOL = 1e-5`.
Currently `HOOK_WIRED_NOT_EXECUTED`.

## Dataset discovery — sealed loaders only

```python
discover_subject_m_sessions(data_dir) -> dict[str, Path]
discover_rt_sessions(data_dir)        -> dict[str, Path]   # exactly 15, sub-C_ses-RT-* only
discover_h1_sessions(data_dir)        -> dict[str, Path]   # via index_heldin_calib
discover_m2_sessions(data_dir)        -> dict[str, Path]
```

Enforced constraints, each corresponding to a past incident:

- **RT** resolves under `dandi_000688/sub-C` and **refuses** any path containing `000129` or
  `sub-Indy`, then routes names through `rt_classical_comparators.session_name_from_nwb_path` and
  requires exactly `RT_EXPECTED_FOLDS = 15` (finding F6/F10).
- **H1** goes through `h1_sparse_event_endpoint.index_heldin_calib`, never a glob. A previous agent's
  raw glob of `*held-out-calib*.nwb` had its entire receipt quarantined.
- **M2** allows `sub-MonkeyN-held-out-calib`, which is in scope for M2 and only for M2.

## Leak and budget guards

```python
refuse_target_query_labels(labels_query)          # raises LabelLeakError
assert_matched_budget(dataset, n_trials_used, *, requested=None) -> int
```

The joint fit puts the target session inside training, which creates a leak surface the original
adapt-based design did not have. Only the matched calibration prefix may carry labels — subject-M 50
trials, RT 24, H1 4 — and `QUERY_ACTIVITY_IN_PRIMARY = False` records that unlabelled target query
activity does not participate in the primary arm.

## Core fitting

```python
fit_source_cebra(neural_sessions, label_sessions, *, max_iterations, output_dimension, device)
fit_joint_cebra(neural_sessions, label_sessions, *, max_iterations, output_dimension, ...)
run_arm_on_synthetic_fold(*, arm, neural_sessions, label_sessions, target_index, dataset, ...)
make_shared_latent_sessions(*, n_channels=(24,31,37), n_samples=240, latent_dim=2, noise=0.05, ...)
```

`fit_joint_cebra` reaches the patched `freeze_sessions` / `init_from` kwargs in the vendored source.

## Vendored CEBRA patch

`third_party/cebra/cebra/integrations/sklearn/cebra.py`, recorded in
`third_party/CEBRA_PROVENANCE.txt`. Two kwargs added to `fit` / `partial_fit` / `_prepare_fit`:

- `init_from` — copy encoder `state_dict`s from a fitted multi-session estimator into the first N
  sessions of a larger joint fit; the target session stays randomly initialized.
- `freeze_sessions` — set `requires_grad=False` on those encoders and build Adam over the remaining
  parameters only.

Both default to `None`, so upstream behaviour is unchanged, and neither combines with `adapt=True`.
This is the minimum change that trains only the target encoder **while keeping cross-session positive
sampling**, which finding F2b shows is the only thing that aligns sessions — there is no shared trunk
to freeze.

## Receipts

`IMPLEMENTATION_BINDING` points at the authoritative source modules rather than at copies, so the
audit trail survives. Receipts carry input SHAs, resolved config, environment fingerprint and the
gate outcomes.

## Constants worth knowing

| Name | Value | Note |
|---|---|---|
| `MODEL_ARCHITECTURE` | `offset1-model` | |
| `OUTPUT_DIMENSION` | `8` | `D_GRID = (3, 8, 16)` available |
| `SOURCE_MAX_ITERATIONS` / `MAX_ADAPT_ITERATIONS` | `10000` / `500` | see F15 before fixing these |
| `SUBM_CALIBRATION_TRIALS` | `50` | budgets `(15, 30, 50)` |
| `RT_CALIBRATION_TRIALS` | `24` | |
| `DEFAULT_DEVICE` | `cpu` | `require_cpu_device()` enforces it |
