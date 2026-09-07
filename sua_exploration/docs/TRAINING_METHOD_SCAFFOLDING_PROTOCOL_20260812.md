# Training-method scaffolding protocol (B1–B4)

**Date:** 2026-08-12  
**Status:** pre-registered scaffolding only. **Authorizes no GPU run and no training.**  
**Scope:** additive, default-off hooks for Category B training-method experiments in `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` §3a items B1–B4.

---

## 0. Non-authorization statement

This document registers interfaces, configs, tests, and aggregators. It does **not** authorize:

- any GPU allocation or training run,
- modification of sealed checkpoints, receipts, or result trees,
- changes to default hyperparameters on existing configs.

Shell entry point `streaming_calibration_exp/scripts/run_training_method_scaffolding_sweep.sh` refuses to start without `--i-have-authorization` and still exits non-zero because this host lacks GPU authorization.

---

## 1. Frozen exact-null guarantee

Every new flag defaults to **disabled**. With all flags at defaults, the training step must be **bitwise identical** to the pre-scaffolding code path.

| Flag | Default | Disabled semantics |
|---|---|---|
| `activity_path_dropout_p` | `0.0` | `build_encoder` returns plain `SideFeatureEarlyPoolEncoder`, not the subclass |
| `carrier_noise_scale` | `0.0` | `model_step` skips carrier perturbation entirely |
| `correspondence_breaking_mode` | `"none"` | `model_step` skips correspondence breaking entirely |
| `correspondence_breaking_seed` | `0` | ignored when mode is `none` |
| `correspondence_breaking_keep_fraction` | `0.75` | ignored when mode is `none` |

**Verification command (reviewer-facing):**

```bash
cd streaming_calibration_exp
CUDA_VISIBLE_DEVICES="" pytest tests/test_training_method_scaffolding.py::test_exact_null_forward_and_training_step_bitwise_identical -q
```

The test asserts real tensor values for forward outputs, training-step loss, and **unperturbed CPU RNG state** after constructing modules with scaffolding kwargs at defaults (see `DECODER_SIDE_DESIGN_SPACE_20260809.md` §6.1).

---

## 2. B1 — loss-mode sweep on the carrier arm

### Hypothesis

`lambda_E` identity distillation pulls the carrier-aware student toward an **activity-only teacher** identity, regularizing back to a phase-blind solution. M2/streaming defaults to `task_plus_y_plus_E` while SUA defaults to `task_only`; this co-occurrence may cap carrier value on M2.

### Provenance of the original R1 loss-mode selection

**Finding:** the original Gate-2 R1 loss ablation (`b3_d64_anchor`, `b3_d64_task_only`, `b3_d64_task_plus_y`) used variant **B3** (`streaming_b3.yaml`, `side_dim=0`) — a **carrier-free** early-pool encoder. The anchor comparison role is `loss_ablation_reference` with `loss_mode: task_plus_y_plus_E`. `gate2_matrix.py::evaluate_r1` selects the winning mode from these B3-only rows. No receipt in the repository records a loss-mode selection on a **carrier-present** (B3S/T4) student.

**Conclusion:** the historical selection was performed on a carrier-free student. Re-testing on the M2 T4 carrier arm is therefore **not** a replay of a sealed decision; it is a new estimand.

### Parameterization

Three unlaunched Hydra configs (M2 T4 / B3S / `side_dim=4`):

- `configs/experiment/b1_carrier_loss_mode_task_only_m2_t4.yaml`
- `configs/experiment/b1_carrier_loss_mode_task_plus_y_m2_t4.yaml`
- `configs/experiment/b1_carrier_loss_mode_task_plus_y_plus_E_m2_t4.yaml`

Loss overrides follow `gate2_matrix.py::loss_overrides`.

### Aggregator

`src/metrics/b1_carrier_loss_mode_matrix.py::aggregate_b1_carrier_loss_mode` — fail-closed unless all three cells are present with finite `R2` and `delta_fixed_B0`. CLI: `scripts/aggregate_b1_carrier_loss_mode_sweep.py`.

### Authorization required later (not granted here)

- 3 cells × agreed seed set × M2 LOSO protocol matching the T4 mainline
- Matrix rows tagged `b1_carrier_loss_mode`
- Kill criterion: if `task_only` does not beat `task_plus_y_plus_E` on the carrier arm, drop the hypothesis

---

## 3. B2 — activity-path dropout (carrier forcing)

### Hypothesis

Zeroing the pooled activity vector before side-feature concat forces the consumer to rely on the carrier, addressing §1.5 consumer under-use.

### Parameterization

| Parameter | Default | Active range |
|---|---|---|
| `activity_path_dropout_p` | `0.0` | `[0, 1]` Bernoulli per batch element |

Implementation: `activity_path_dropout.py`, `activity_path_dropout_encoder.py` (`ActivityPathDropoutSideFeatureEarlyPoolEncoder`). Mask applies to pooled `mean_feat` only; side features are untouched.

Example config: `configs/experiment/b2_activity_path_dropout_p025_m2_t4.yaml`.

---

## 4. B3 — carrier estimator-noise augmentation

### Hypothesis

Noise drawn from the analytic small-`M` OLS covariance `σ²(X'X)⁻¹` trains robustness to deployment estimation error (targets the M10–M50 budget gap).

### Parameterization

| Parameter | Default | Notes |
|---|---|---|
| `carrier_noise_scale` | `0.0` | multiplier on Cholesky draw |
| `set_carrier_noise_cholesky()` | unset | per-unit `[2,2]` Cholesky of `[a,c]` block; required when scale > 0 |

Only `[a, c]` are perturbed; `m` is **recomputed** as `hypot(a, c)`; `b` is never independently perturbed. Covariance helper: `carrier_noise_augmentation.py::ac_block_covariance_from_design`.

Example config: `configs/experiment/b3_carrier_noise_scale1_m2_t4.yaml` (requires datamodule wiring for Cholesky at launch time).

---

## 5. B4 — correspondence-breaking augmentation

### Hypothesis

Training under broken correspondence (permutation, subsetting, pseudo-electrode pooling) should improve robustness where deployment stress breaks unit alignment (counterpart to A2/A3).

### Parameterization

| Parameter | Default | Active values |
|---|---|---|
| `correspondence_breaking_mode` | `"none"` | `"permute"`, `"subset"`, `"electrode_pool"` |
| `correspondence_breaking_seed` | `0` | recorded in `CorrespondenceBreakingReceipt.resolved_seed` |
| `correspondence_breaking_keep_fraction` | `0.75` | used by `"subset"` only |

Permutation/subsetting/pooling applied **consistently** to query neural, calibration, and side features. Seeding follows `deterministic_nonidentity_row_permutation` salt pattern.

Example config: `configs/experiment/b4_correspondence_breaking_permute_m2_t4.yaml`.

---

## 6. What each experiment needs before launch (not granted)

1. Explicit GPU authorization and `CUDA_VISIBLE_DEVICES` allocation  
2. Protocol sign-off referencing this document revision  
3. For B3: datamodule hook supplying per-session `[a,c]` Cholesky factors  
4. For B1: populated `gate2_revised_matrix.csv` rows for all three loss modes  
5. No change to sealed artifacts under `sua_exploration/results/`, `SPINT-main/pilot_artifacts/`, or any `logs/` tree  

---

## 7. Reviewer checklist

- [ ] `pytest tests/test_training_method_scaffolding.py` passes on CPU with `CUDA_VISIBLE_DEVICES=""`  
- [ ] Exact-null test quoted in CI or review notes  
- [ ] No default config or signature breakage  
- [ ] Shell entry point refuses without `--i-have-authorization`  
