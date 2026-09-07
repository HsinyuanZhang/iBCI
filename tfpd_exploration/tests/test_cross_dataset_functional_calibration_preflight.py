"""P preflight tests: frozen normalizer, live consumer §6, disposable launcher dry-path.

CPU only. Does not start CUDA training or overwrite Stage0 / revision roots.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import torch

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import basis as p_basis
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import carrier_solver as solver
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as bank


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
_CLI_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": str(ROOT),
}
_CONSUMER_ATOL = 1.0e-6
_CONSUMER_RTOL = 1.0e-5
_F64_ATOL = 1.0e-8
_F64_RTOL = 1.0e-8


def _interior_dictionary_emg(seed: int = 21):
    torch.manual_seed(seed)
    raw = torch.abs(torch.randn(3, 16, dtype=torch.float64)) + 0.35
    scale = torch.ones(16, dtype=torch.float64)
    emg = torch.abs(torch.randn(12, 16, dtype=torch.float64)) + 0.25
    return raw, scale, emg


def test_encode_rejects_nonfinite_and_subfloor_scale() -> None:
    raw, scale, emg = _interior_dictionary_emg()
    module = p_basis.RowNormalizedNNMFBasis(dictionary=raw, scale=scale, trainable=False)
    bad_emg = emg.clone()
    bad_emg[0, 0] = float("nan")
    with pytest.raises(p_basis.BasisError, match="finite"):
        module.encode(bad_emg)
    tiny = scale.clone()
    tiny[3] = 1.0e-9
    module_tiny = p_basis.RowNormalizedNNMFBasis(dictionary=raw, scale=tiny, trainable=False)
    with pytest.raises(p_basis.BasisError, match="scale"):
        module_tiny.encode(emg)
    inf_scale = scale.clone()
    inf_scale[0] = float("inf")
    module_inf = p_basis.RowNormalizedNNMFBasis(dictionary=raw, scale=inf_scale, trainable=False)
    with pytest.raises(p_basis.BasisError, match="finite"):
        module_inf.encode(emg)


def test_nnls_support_freeze_kkt_and_edge_cases() -> None:
    raw, scale, emg = _interior_dictionary_emg(22)
    module = p_basis.RowNormalizedNNMFBasis(dictionary=raw, scale=scale, trainable=False)
    scaled = torch.relu(emg) / scale
    dictionary = module.constrained_dictionary()
    z = module.encode(emg)
    report = p_basis.nnls_contract_report(dictionary, scaled, z)
    assert report["support_eps"] == 1.0e-12
    assert report["scipy_forward_max_abs"] <= 1.0e-8
    assert bool(report["nonnegative"])
    assert float(report["kkt_residual"]) <= 1.0e-6
    assert report["ridge_repair_used"] is False

    zero = torch.zeros_like(emg)
    z_zero = module.encode(zero)
    zero_report = p_basis.nnls_contract_report(dictionary, torch.zeros_like(scaled), z_zero)
    assert torch.equal(z_zero, torch.zeros_like(z_zero))
    assert zero_report["empty_active_set"] is True
    assert zero_report["ridge_repair_used"] is False

    near = torch.zeros(1, 16, dtype=torch.float64)
    near[0] = dictionary[0] * 1.0e-11
    z_near = p_basis.nnls_with_active_set(dictionary, near)
    near_report = p_basis.nnls_contract_report(dictionary, near, z_near)
    assert near_report["near_boundary_disclosed"] is True
    assert near_report["ridge_repair_used"] is False

    degenerate = dictionary.clone()
    degenerate[1] = degenerate[0]
    with pytest.raises(p_basis.BasisError):
        p_basis.nnls_with_active_set(degenerate, scaled[:2])


def test_solve_ridge_follows_input_dtype_adapter_forces_float64_outside_autocast() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    z32 = torch.randn(10, 3, dtype=torch.float32)
    rates32 = torch.randn(10, 5, dtype=torch.float32)
    native = solver.solve_ridge(z32, rates32, ridge_lambda=1.0)
    assert native.dtype == torch.float32

    z64 = z32.to(torch.float64)
    rates64 = rates32.to(torch.float64)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        forced = model_adapter.solve_carrier_float64(z64, rates64, ridge_lambda=1.0)
        leaked = solver.solve_ridge(z32, rates32, ridge_lambda=1.0)
    assert forced.dtype == torch.float64
    assert leaked.dtype != torch.float64 or bool(torch.is_autocast_enabled()) is False
    with torch.autocast(device_type="cpu", dtype=torch.float32):
        forced_fp32_ctx = model_adapter.solve_carrier_float64(z32, rates32, ridge_lambda=1.0)
    assert forced_fp32_ctx.dtype == torch.float64


def test_frozen_normalizer_source_pooled_parity() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    assert frozen.name == "SOURCE_INITIAL_DICTIONARY_FROZEN_NORMALIZER_V1"
    assert frozen.mu0.shape == (4,)
    assert frozen.sigma0.shape == (4,)
    assert np.array_equal(frozen.mu0, frozen.shared_mu0_for("P-FIX"))
    assert np.array_equal(frozen.sigma0, frozen.shared_sigma0_for("P-CA"))
    assert frozen.updated_per_step is False
    assert frozen.refit_on_target is False
    assert frozen.parent_carrier_parity_passed is True
    assert frozen.d0_zero_count >= 0
    assert frozen.d0_positive_count + frozen.d0_zero_count == 3 * 16
    assert frozen.relu_lock is True
    receipt = frozen.receipt()
    assert receipt["array_sha256"]
    assert receipt["input_sha256"]
    assert receipt["code_sha256"]


def test_zero_basis_perturbation_reproduces_parent_prediction() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.zero_perturbation_parent_replay()
    assert report["carrier_atol"] == _F64_ATOL
    assert report["carrier_rtol"] == _F64_RTOL
    assert report["carrier_parity_passed"] is True
    assert report["prediction_atol"] == _CONSUMER_ATOL
    assert report["prediction_rtol"] == _CONSUMER_RTOL
    assert report["prediction_parity_passed"] is True


def test_coefficient_recovery_intercept_unpenalized_label_permutation() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    rng = np.random.default_rng(33)
    z = rng.normal(size=(48, 3))
    weights = rng.normal(size=(7, 3))
    intercepts = rng.normal(size=(7,))
    rates = z @ weights.T + intercepts
    recovered = solver.solve_ridge(
        torch.as_tensor(z, dtype=torch.float64),
        torch.as_tensor(rates, dtype=torch.float64),
        ridge_lambda=1.0,
    )
    expected = bank.carrier_from_encoding(*bank.fit_all_units(z, rates))
    assert torch.allclose(recovered, torch.as_tensor(expected), atol=1e-10, rtol=1e-10)

    offset = 3.0
    shifted = solver.solve_ridge(
        torch.as_tensor(z, dtype=torch.float64),
        torch.as_tensor(rates + offset, dtype=torch.float64),
        ridge_lambda=1.0,
    )
    assert torch.allclose(shifted[:, :3], recovered[:, :3], atol=1e-10, rtol=1e-10)
    assert torch.allclose(shifted[:, 3], recovered[:, 3] + offset, atol=1e-10, rtol=1e-10)

    perm = rng.permutation(7)
    shuffled = solver.solve_ridge(
        torch.as_tensor(z, dtype=torch.float64),
        torch.as_tensor(rates[:, perm], dtype=torch.float64),
        ridge_lambda=1.0,
    )
    assert torch.allclose(shuffled, recovered[list(perm)])
    assert shuffled.shape[0] == recovered.shape[0]
    assert model_adapter.label_count(recovered) == model_adapter.label_count(shuffled) == 7


def test_solve_grads_finite_nonzero_and_source_query_loss_dictionary_grad() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    raw, scale, emg = _interior_dictionary_emg(40)
    rates = torch.randn(emg.shape[0], 6, dtype=torch.float64)
    module = p_basis.RowNormalizedNNMFBasis(dictionary=raw.clone(), scale=scale, trainable=True)

    def chain(dictionary):
        z_local = p_basis.encode_from_raw(dictionary, scale, emg)
        return solver.solve_ridge(z_local, rates, ridge_lambda=1.0)

    interior = module.raw_dictionary.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(chain, (interior,), eps=1e-6, atol=1e-7, rtol=1e-5)

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    grad_report = pair.source_query_loss_dictionary_grad()
    assert grad_report["finite"] is True
    assert grad_report["nonzero"] is True
    assert grad_report["not_just_carrier_sum"] is True


def test_query_history_disjointness() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.query_history_disjointness()
    assert report["support_carrier_unchanged"] is True
    assert report["forbidden_query_labels_mutated"] is True


def test_simultaneous_unit_activity_carrier_permutation() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.unit_permutation_invariance()
    assert report["passed"] is True
    assert report["max_abs"] <= _CONSUMER_ATOL or report["rel"] <= _CONSUMER_RTOL


def test_dropout_masks_synchronize_neural_activity_carrier() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.dropout_mask_sync()
    assert report["shared_mask"] is True
    assert report["passed"] is True


def test_native_16d_signed_emg_unchanged() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.native_output_contract()
    assert report["output_dim"] == 16
    assert report["hidden_output_projection"] is False
    assert report["signed_target_view"] is True


def test_target_fit_zero_optimizer_steps_source_hashes_unchanged() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    before = pair.source_learned_state_hashes()
    report = pair.target_fit_no_backward()
    after = pair.source_learned_state_hashes()
    assert report["optimizer_steps"] == 0
    assert report["backward_steps"] == 0
    assert before == after


def test_full_new_stage_resume_not_blocked_by_old_sfix() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.new_stage_resume_roundtrip()
    required = {"model", "basis", "optimizer", "rng", "sampler", "normalizer"}
    assert required.issubset(set(report["saved_fields"]))
    assert report["roundtrip_ok"] is True
    assert report["old_sfix_missing_fields_block"] is False


def test_pfix_pca_initial_consumer_weights_byte_identical() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import normalizer

    frozen = normalizer.materialize(ROOT)
    pair = model_adapter.build_p_pair(ROOT, frozen)
    report = pair.initial_allowlist()
    assert report["consumer_weights_byte_identical"] is True
    assert report["p_ca_trains_raw_dictionary"] is True
    assert report["p_fix_trains_raw_dictionary"] is False
    assert report["shared_allowlist"] == ["decoder", "id_encoder", "carrier_projection_weight"]
    assert report["raw_16d_residual_capacity"] is True


def test_disposable_profile_launcher_dry_path() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    spec = model_adapter.disposable_profile_spec()
    assert spec["steps_per_arm"] == 100
    assert spec["effective_batch"] == 32
    assert spec["source_sessions"] == list(plan.M1_FOLD0_SOURCES)
    assert spec["query"] == [plan.M1_QUERY_START, plan.M1_QUERY_STOP_EXCLUSIVE]
    assert spec["outer_scoring"] is False
    assert spec["inherits_smoke_rng"] is False
    assert spec["auto_extend_to_12_epochs"] is False
    assert spec["budget_minutes"] == 30
    assert spec["timers"] == [
        "support_read",
        "scipy_nnls_cpu_gpu",
        "active_set_solve",
        "ridge",
        "consumer_fwd_bwd",
        "checkpoint",
    ]
    dry = model_adapter.launch_disposable_profile(ROOT, dry_run=True)
    assert dry["gpu_work_started"] is False
    assert dry["cuda_training_started"] is False
    assert dry["status"] == "DRY"


def test_disposable_profile_requires_env_and_nousersite() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import model_adapter

    env = {**_CLI_ENV}
    env.pop("CDF_DISPOSABLE_PROFILE", None)
    completed = subprocess.run(
        [PYTHON, str(CLI), "--disposable-profile"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
    )
    assert completed.returncode != 0
    env_ok = {**_CLI_ENV, "CDF_DISPOSABLE_PROFILE": "1"}
    env_ok["PYTHONNOUSERSITE"] = "0"
    completed_nouser = subprocess.run(
        [PYTHON, str(CLI), "--disposable-profile"],
        capture_output=True,
        text=True,
        env=env_ok,
        cwd=str(ROOT),
    )
    assert completed_nouser.returncode != 0
    with pytest.raises(model_adapter.AdapterError):
        model_adapter.launch_disposable_profile(ROOT, dry_run=False, environ={"PYTHONNOUSERSITE": "1"})


def test_execute_gpu_still_rejected_for_formal_pilots() -> None:
    completed = subprocess.run(
        [PYTHON, str(CLI), "--execute-gpu"],
        capture_output=True,
        text=True,
        env=_CLI_ENV,
        cwd=str(ROOT),
    )
    assert completed.returncode != 0
    assert "CPU-only" in completed.stderr


def test_stage0_terminal_remains_byte_identical() -> None:
    path = ROOT / plan.STAGE0_ROOT_RELATIVE / "terminal.json"
    assert plan.sha256_bytes(path.read_bytes()) == plan.STAGE0_TERMINAL_SHA256


def test_preflight_root_deliverables() -> None:
    from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import execute_preflight

    hashes, terminal = execute_preflight.execute(ROOT)
    root = ROOT / plan.P_PREFLIGHT_ROOT_RELATIVE
    for name in (
        "normalizer.json",
        "consumer_tests.json",
        "STAGE1_IMPLEMENTATION_SPEC.md",
        "HANDOFF_FOR_ASTRA_REVIEW.md",
        "terminal.json",
    ):
        assert (root / name).is_file()
        assert name in hashes
    stage0 = json.loads((ROOT / plan.STAGE0_ROOT_RELATIVE / "terminal.json").read_text(encoding="utf-8"))
    assert plan.sha256_bytes((ROOT / plan.STAGE0_ROOT_RELATIVE / "terminal.json").read_bytes()) == (
        plan.STAGE0_TERMINAL_SHA256
    )
    assert stage0["gpu_work_started"] is False
    body = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert body["gpu_work_started"] is False
    assert body["gpu_eligible"] is False
    assert terminal == hashes["terminal.json"]
    spec = (root / "STAGE1_IMPLEMENTATION_SPEC.md").read_text(encoding="utf-8")
    assert "gpu_eligible=false" in spec.lower() or "gpu_eligible` = false" in spec.lower() or "`gpu_eligible=false`" in spec
    handoff = (root / "HANDOFF_FOR_ASTRA_REVIEW.md").read_text(encoding="utf-8")
    assert "PREFLIGHT_CPU_READY" in handoff or "BLOCKER" in handoff
