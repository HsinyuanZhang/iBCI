"""Pure CPU contracts for the frozen M1 decoder-latent feasibility audit."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m1_decoder_latent_alignment_oracle.py"


def module():
    spec = importlib.util.spec_from_file_location("m1_decoder_latent_oracle", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_canonical_target_is_exact_teacher_identity_path_and_has_no_query_argument():
    m = module()
    torch.manual_seed(0)
    fc_in = nn.Sequential(nn.Linear(5, 7), nn.ReLU(), nn.Linear(7, 7))
    fc_out = nn.Sequential(nn.Linear(7, 5))
    calib = torch.randn(2, 10, 5, 3)
    actual = m.canonical_identity_target(calib, fc_in, fc_out)
    manual = fc_out(fc_in(calib.permute(0, 1, 3, 2)).mean(dim=1))
    assert actual.shape == (2, 3, 5)
    assert torch.equal(actual, manual)
    assert tuple(m.canonical_identity_target.__code__.co_varnames[:3]) == ("calib", "fc_id_in", "fc_id_out")


def test_rank_is_strictly_limited_to_one_through_three_and_reconstructs():
    m = module()
    matrix = np.arange(24.0).reshape(6, 4)
    a, b, reconstructed = m.low_rank_decoder_residual(matrix, 2)
    assert a.shape == (6, 2) and b.shape == (4, 2) and reconstructed.shape == (6, 4)
    assert np.linalg.matrix_rank(reconstructed) <= 2
    with pytest.raises(ValueError, match="one of"):
        m.low_rank_decoder_residual(matrix, 4)
    with pytest.raises(ValueError, match="frozen"):
        m.validate_rank_lambda_grid((1, 2), (0.1,))


def test_label_shuffle_preserves_m10_multiset_and_is_nonidentity():
    m = module()
    labels = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    first = m.deterministic_label_shuffle(labels, session_name="s", seed=42)
    second = m.deterministic_label_shuffle(labels, session_name="s", seed=42)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(first, labels)


def test_orthogonal_only_has_no_linear_training_rate_component():
    m = module()
    rate = np.column_stack([np.arange(8.0), np.arange(8.0) ** 2])
    label = np.column_stack([2.0 * rate[:, 0] + 1.0, -rate[:, 1] + 3.0])
    fitted = m.fit_orthogonalizer(rate, label, ridge=0.0)
    residual = m.orthogonal_only(fitted, rate, label)
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-10)


def test_nested_selection_rejects_outer_heldout_session():
    m = module()
    records = [
        {"rank": 1, "lambda": 0.1, "inner_validation_sessions": ["a", "b"], "mean_identity_mse": 1.0},
        {"rank": 2, "lambda": 0.1, "inner_validation_sessions": ["a", "heldout"], "mean_identity_mse": 0.0},
    ]
    with pytest.raises(ValueError, match="non-train"):
        m.nested_train_only_choice(records, ["a", "b"])
    selected = m.nested_train_only_choice(records[:1], ["a", "b"])
    assert selected["rank"] == 1


def test_support_features_reject_post_support_and_missing_condition_semantics():
    m = module()
    sums = np.ones((10, 3))
    lengths = np.arange(1, 11, dtype=float)
    labels = np.asarray([1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    assert m.support_condition_features(sums, lengths, labels).shape == (3, 4)
    with pytest.raises(ValueError, match="M=10"):
        m.support_condition_features(np.ones((11, 3)), np.ones(11), np.ones(11, dtype=int))
    with pytest.raises(ValueError, match="levels"):
        m.support_condition_features(sums, lengths, np.ones(10, dtype=int))


def test_run_refuses_overwrite(tmp_path: Path):
    m = module()
    out = tmp_path / "receipt"
    out.mkdir()
    with pytest.raises(FileExistsError):
        m.run(out)


def test_receipt_writer_is_separate_from_oracle_fit_contract():
    source = (ROOT / "sua_exploration/scripts/write_m1_decoder_latent_alignment_oracle_receipt.py").read_text()
    assert "formal_oracle_status\": \"blocked_pending_root_review" in source
    assert "rank_or_lambda_fit" in source
    assert "GPU_launch" in source
