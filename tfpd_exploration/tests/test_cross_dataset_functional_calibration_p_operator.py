"""P-operator revision tests. CPU / synthetic only. No NWB, no decoder R2, no CUDA."""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pytest
import torch

from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import basis as p_basis
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import carrier_solver as solver
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import parent_audit
from tfpd_exploration.src.cross_dataset_functional_calibration_v1 import plan
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as bank


def _repo():
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def test_torch_ridge_matches_numpy_bank() -> None:
    rng = np.random.default_rng(4)
    z = rng.normal(size=(40, 3))
    rates = z @ np.array([[0.2, -0.1, 0.3], [0.0, 0.4, -0.2]]).T + 0.5
    rates = np.column_stack((rates, rng.normal(size=(40, 2))))
    weights, intercepts = bank.fit_all_units(z, rates)
    expected = bank.carrier_from_encoding(weights, intercepts)
    got = solver.solve_ridge(
        torch.as_tensor(z, dtype=torch.float64),
        torch.as_tensor(rates, dtype=torch.float64),
        ridge_lambda=1.0,
    )
    assert got.shape == expected.shape
    assert torch.allclose(got, torch.as_tensor(expected), atol=1e-10, rtol=1e-10)


def test_intercept_unpenalized_and_label_permutation() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(size=(30, 3))
    rates = rng.normal(size=(30, 6))
    carrier = solver.solve_ridge(
        torch.as_tensor(z, dtype=torch.float64),
        torch.as_tensor(rates, dtype=torch.float64),
        ridge_lambda=1.0,
    )
    perm = rng.permutation(6)
    shuffled = solver.solve_ridge(
        torch.as_tensor(z, dtype=torch.float64),
        torch.as_tensor(rates[:, perm], dtype=torch.float64),
        ridge_lambda=1.0,
    )
    assert torch.allclose(shuffled, carrier[list(perm)])
    assert shuffled.shape == carrier.shape


def test_ridge_gradients_finite_and_nonzero() -> None:
    torch.manual_seed(6)
    z = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
    rates = torch.randn(20, 5, dtype=torch.float64)
    carrier = solver.solve_ridge(z, rates, ridge_lambda=1.0)
    loss = carrier.pow(2).sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.norm()) > 0.0


def test_ridge_gradcheck_small_float64() -> None:
    torch.manual_seed(7)
    z = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    rates = torch.randn(8, 2, dtype=torch.float64)

    def fn(scores):
        return solver.solve_ridge(scores, rates, ridge_lambda=1.0)

    assert torch.autograd.gradcheck(fn, (z,), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_f_eta_gauge_and_nnls_parity() -> None:
    rng = np.random.default_rng(8)
    dictionary = np.abs(rng.normal(size=(3, 16)))
    dictionary = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)
    scale = np.maximum(rng.random(16), 0.1)
    emg = np.abs(rng.normal(size=(25, 16)))
    expected = bank.nnls_activations(emg / scale, dictionary)
    module = p_basis.RowNormalizedNNMFBasis(
        dictionary=torch.as_tensor(dictionary, dtype=torch.float64),
        scale=torch.as_tensor(scale, dtype=torch.float64),
        trainable=True,
    )
    got = module.encode(torch.as_tensor(emg, dtype=torch.float64))
    assert got.shape == (25, 3)
    assert torch.allclose(got, torch.as_tensor(expected), atol=1e-8, rtol=1e-8)
    rescaled = dictionary * np.array([1.7, 0.4, 2.2])[:, None]
    module_b = p_basis.RowNormalizedNNMFBasis(
        dictionary=torch.as_tensor(rescaled, dtype=torch.float64),
        scale=torch.as_tensor(scale, dtype=torch.float64),
        trainable=False,
    )
    got_b = module_b.encode(torch.as_tensor(emg, dtype=torch.float64))
    assert torch.allclose(got, got_b, atol=1e-8, rtol=1e-8)


def test_zero_eta_matches_bank_carrier() -> None:
    rng = np.random.default_rng(9)
    dictionary = np.abs(rng.normal(size=(3, 16)))
    dictionary = dictionary / np.linalg.norm(dictionary, axis=1, keepdims=True)
    scale = np.maximum(rng.random(16), 0.1)
    emg = np.abs(rng.normal(size=(40, 16)))
    rates = rng.normal(size=(40, 7))
    z = bank.nnls_activations(emg / scale, dictionary)
    weights, intercepts = bank.fit_all_units(z, rates)
    expected = bank.carrier_from_encoding(weights, intercepts)
    module = p_basis.RowNormalizedNNMFBasis(
        dictionary=torch.as_tensor(dictionary, dtype=torch.float64),
        scale=torch.as_tensor(scale, dtype=torch.float64),
        trainable=True,
    )
    z_t = module.encode(torch.as_tensor(emg, dtype=torch.float64))
    got = solver.solve_ridge(z_t, torch.as_tensor(rates, dtype=torch.float64), ridge_lambda=1.0)
    assert torch.allclose(got, torch.as_tensor(expected), atol=1e-8, rtol=1e-8)


def test_basis_gradient_survives_solve() -> None:
    torch.manual_seed(10)
    raw = torch.abs(torch.randn(3, 16, dtype=torch.float64))
    scale = torch.ones(16, dtype=torch.float64)
    emg = torch.abs(torch.randn(16, 16, dtype=torch.float64))
    rates = torch.randn(16, 4, dtype=torch.float64)
    module = p_basis.RowNormalizedNNMFBasis(dictionary=raw, scale=scale, trainable=True)
    z = module.encode(emg)
    carrier = solver.solve_ridge(z, rates, ridge_lambda=1.0)
    carrier.sum().backward()
    grad = module.raw_dictionary.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.norm()) > 0.0


def test_parent_audit_binds_s_fix_and_forbids_allsource() -> None:
    report = parent_audit.audit(_repo())
    assert report["parent_bytes"] == plan.S_FIX_EPOCH011_SHA256
    assert report["parent_relative"] == plan.S_FIX_EPOCH011_RELATIVE
    assert report["teacher_bytes"] == plan.FOLD0_SOURCE_TEACHER_SHA256
    assert report["outer_left_out"] == "ses-20120924"
    assert "20120924" not in "".join(report["teacher_train_sessions"])
    assert report["allsource_forbidden"] is True
    assert report["z_fix_rejected"] is True
    assert report["claim"] == "CLEAN_OUTER_SESSION_FILE_EXCLUSION"
    assert report["gpu_eligible"] is False
    assert report["full_ckpt_schema"] is False


def test_named_revision_does_not_import_cuda() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.cuda.is_available() is False or not torch.cuda.is_initialized()


def test_revision_document_hash_matches_disk() -> None:
    path = _repo() / plan.P_REVISION_RELATIVE
    assert path.is_file()
    assert plan.sha256_bytes(path.read_bytes()) == plan.P_REVISION_SHA256
