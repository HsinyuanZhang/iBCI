from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tfpd_exploration.src.m2_same_query_comparator_v1 import core, plan, physical


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cell_matrix_is_complete_and_distinguishes_supervision() -> None:
    names = [cell.name for cell in core.CELL_SPECS]
    assert len(names) == len(set(names)) == 21
    assert names[:4] == [
        "spint_chronological_m30",
        "spint_chronological_m10",
        "spint_chronological_m4",
        "spint_matched_doptimal_m4",
    ]
    assert [cell.name for cell in core.CELL_SPECS if cell.source == "reused_parent"] == [
        "t4_ridge_static_m30",
        "t4_ridge_static_m10",
        "t4_ridge_activity30_m10",
        "t4_ridge_static_m4",
        "t4_ridge_activity30_m4",
    ]
    dense = {cell.family: cell.supervision for cell in core.CELL_SPECS if cell.budget == 30}
    assert dense["direct_ridge"] == "dense_velocity_labels"
    assert dense["pv"] == "trial_direction_plus_dense_velocity_labels"


def test_dense_support_rows_never_bridge_omitted_trials() -> None:
    starts = np.arange(35, dtype=np.int64) * 60
    selected = np.asarray([1, 7, 16, 29], dtype=np.int64)
    rows = core.dense_support_target_bins(starts, total_bins=35 * 60, selected_indices=selected)
    expected = np.concatenate([np.arange(index * 60 + 49, (index + 1) * 60) for index in selected])
    assert np.array_equal(rows, expected)
    for endpoint in rows:
        start = endpoint - 49
        assert any(index * 60 <= start <= endpoint < (index + 1) * 60 for index in selected)
    assert not np.any((rows >= 2 * 60) & (rows < 7 * 60))


def test_dense_support_skips_short_trials_without_crossing_and_rejects_duplicate_selection() -> None:
    starts = np.arange(35, dtype=np.int64) * 60
    with pytest.raises(core.ComparatorError):
        core.dense_support_target_bins(starts, 35 * 60, [1, 1, 2, 3])
    short = starts.copy()
    short[3:] -= 20
    rows = core.dense_support_target_bins(short, int(short[-1] + 60), [2, 4, 5, 6])
    expected = np.concatenate(
        [np.arange(int(short[index] + 49), int(short[index + 1])) for index in [4, 5, 6]]
    )
    assert np.array_equal(rows, expected)
    assert not np.any((rows >= short[2]) & (rows < short[3]))


def test_dense_support_rejects_when_every_selected_trial_is_short() -> None:
    starts = np.arange(35, dtype=np.int64) * 40
    with pytest.raises(core.ComparatorError, match="no within-trial W50 endpoint"):
        core.dense_support_target_bins(starts, 35 * 40, [1, 2, 3, 4])


def test_materialized_w50_windows_are_exact() -> None:
    neural = np.arange(200 * 96, dtype=np.float32).reshape(200, 96)
    endpoints = np.asarray([49, 77, 199], dtype=np.int64)
    features = core.materialize_windows(neural, endpoints)
    assert features.shape == (3, 50 * 96)
    assert np.array_equal(features[0], neural[:50].reshape(-1))
    assert np.array_equal(features[1], neural[28:78].reshape(-1))
    assert np.array_equal(features[2], neural[150:200].reshape(-1))


def test_parent_score_is_exact_and_complete() -> None:
    indexed, digest = physical._load_parent(REPO_ROOT)
    assert digest == plan.PARENT_SCORE_SHA256
    assert len(indexed) == 65
    surfaces = {key[0] for key in indexed}
    assert surfaces == {"within_post30", "external_official_query"}


def test_reused_parent_row_requires_current_query_and_target_identity() -> None:
    indexed, _ = physical._load_parent(REPO_ROOT)
    key = next(key for key in indexed if key[2] == "ridge_static_m30")
    parent_row = indexed[key]
    cell = next(cell for cell in core.CELL_SPECS if cell.name == "t4_ridge_static_m30")

    # Reconstruct arrays only through their expected digest domain by replacing
    # the one row in a synthetic parent. A digest mismatch must fail before R2
    # is reused.
    starts = np.arange(int(parent_row["window_count"]), dtype=np.int64)
    target = np.zeros((starts.size, 2), dtype=np.float32)
    with pytest.raises(core.ComparatorError, match="query start digest mismatch"):
        physical._reused_parent_row(
            cell=cell,
            surface=key[0],
            session=key[1],
            starts=starts,
            target=target,
            parent=indexed,
        )


def test_atomic_publication_writes_only_canonical_pair(tmp_path: Path) -> None:
    root = tmp_path / "result"
    payload = {"schema": plan.SCHEMA, "status": "TERMINAL", "value": 3}
    physical._publish_atomic(root, payload)
    assert sorted(path.name for path in root.iterdir()) == ["score.json", "score.json.sha256"]
    assert (root.stat().st_mode & 0o777) == 0o555
    assert ((root / "score.json").stat().st_mode & 0o777) == 0o444
    assert ((root / "score.json.sha256").stat().st_mode & 0o777) == 0o444
    body = (root / "score.json").read_bytes()
    import hashlib

    assert (root / "score.json.sha256").read_text() == f"{hashlib.sha256(body).hexdigest()}  score.json\n"
    with pytest.raises(core.ComparatorError, match="already exists"):
        physical._publish_atomic(root, payload)


def test_equal_session_summary_and_pairing() -> None:
    candidate = {"b": 0.3, "a": 0.2, "c": -0.1}
    reference = {"a": 0.1, "b": 0.4, "c": -0.3}
    summary = core.summarize_sessions(candidate)
    assert summary["equal_session_mean"] == pytest.approx((0.2 + 0.3 - 0.1) / 3)
    contrast = core.paired_contrast(candidate, reference)
    assert contrast["positive_sessions"] == 2
    assert contrast["candidate_minus_reference_mean"] == pytest.approx((0.1 - 0.1 + 0.2) / 3)


def test_dry_cli_is_torch_free_and_inert() -> None:
    command = [sys.executable, str(REPO_ROOT / "tfpd_exploration/scripts/run_m2_same_query_comparator_v1.py")]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH"
    assert len(payload["cells"]) == 21
    script = (
        "import json,runpy,sys; "
        f"runpy.run_path({str(REPO_ROOT / 'tfpd_exploration/scripts/run_m2_same_query_comparator_v1.py')!r}, run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, check=True, capture_output=True, text=True)


def test_cli_bootstrap_supports_legacy_mc_maze_population_vector_import() -> None:
    cli = REPO_ROOT / "tfpd_exploration/scripts/run_m2_same_query_comparator_v1.py"
    script = (
        "import runpy,sys; "
        f"runpy.run_path({str(cli)!r}, run_name='__main__'); "
        "import sua_exploration.mc_maze.population_vector_comparator; "
        "import mc_maze.unit_side_features; "
        "import torch; assert torch.cuda.is_initialized() is False"
    )
    subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )


def test_real_spint_cached_identity_reduction_is_exact_on_cpu() -> None:
    import torch

    from streaming_calibration_exp.src.models.components.spint import SpintModel

    torch.manual_seed(42)
    model = SpintModel(
        model_dim=8,
        num_covariates=2,
        window_size=50,
        num_heads=2,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    ).eval()
    neural = torch.randn(3, 50, 96)
    calibration = torch.randn(1, 4, 100, 96)
    with torch.inference_mode():
        direct = model(neural, calib_trialized_neural_features=calibration)
        identity = physical._teacher_identity(torch, model, calibration)
        cached = physical._manual_teacher_decode(model, neural, identity)
    assert torch.equal(direct, cached)
    assert torch.cuda.is_initialized() is False
