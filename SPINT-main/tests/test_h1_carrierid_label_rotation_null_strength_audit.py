"""Focused non-data contracts for H1 label-rotation null-strength audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/h1_carrierid_label_rotation_null_strength_audit.py"
SPEC = importlib.util.spec_from_file_location("h1_label_rotation_null_strength_audit_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_exact_rotation_shift_is_deterministic_nonidentity_and_in_range():
    first = AUDIT.rotation_shift(session_name="ses-19250113T120811", trial_number=7.0, block_count=23)
    second = AUDIT.rotation_shift(session_name="ses-19250113T120811", trial_number=7.0, block_count=23)
    assert first == second
    assert 1 <= first < 23
    with pytest.raises(AUDIT.LabelRotationAuditError, match="at least two"):
        AUDIT.rotation_shift(session_name="ses-19250113T120811", trial_number=7.0, block_count=1)


def test_common_linear_and_orthogonal_recovery_are_cross_session_not_resubstitution():
    rng = np.random.default_rng(123)
    angle = 0.73
    q = np.array([[np.cos(angle), -np.sin(angle), 0.0, 0.0], [np.sin(angle), np.cos(angle), 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0]])
    sessions = {}
    for index in range(4):
        label = rng.normal(size=(32, 4))
        full = label @ q
        sessions[f"source-{index}"] = {"label": [label], "full": [full]}
    result = AUDIT._common_inverse_audit(sessions)
    assert result["strict_common_linear_recoverable"] is True
    assert result["strict_common_orthogonal_recoverable"] is True
    assert result["summary"]["common_orthogonal"]["frobenius_cosine"]["median"] > 0.999999


def test_normalization_preserves_pairwise_carrier_cosine_and_pearson():
    rng = np.random.default_rng(5)
    full = rng.normal(size=(176, 4))
    label = rng.normal(size=(176, 4))
    metrics = AUDIT._entry_metrics(full_raw=full, label_raw=label, denominator=7.3)
    assert metrics["raw"]["frobenius_cosine"] == pytest.approx(metrics["source_rms_normalized"]["frobenius_cosine"], abs=1e-14)
    assert metrics["raw"]["flattened_pearson"] == pytest.approx(metrics["source_rms_normalized"]["flattened_pearson"], abs=1e-14)
    assert metrics["source_rms_normalized"]["full_rms"] == pytest.approx(metrics["raw"]["full_rms"] / 7.3)


def test_source_scope_has_no_target_or_gpu_execution_route():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "load_outer_date_target_records" not in source
    assert "H1CarrierIdDateLodoStrictTargetDataset" not in source
    assert "torch.cuda" not in source
    assert "Trainer(" not in source
    assert "--run-source-only-audit" in source


def test_common_inverse_rejects_rank_deficient_source_carriers():
    repeated = np.ones((12, 4))
    sessions = {
        "a": {"label": [repeated], "full": [repeated]},
        "b": {"label": [repeated], "full": [repeated]},
    }
    with pytest.raises(AUDIT.LabelRotationAuditError, match="rank deficient"):
        AUDIT._common_inverse_audit(sessions)
