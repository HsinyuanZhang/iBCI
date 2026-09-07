import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from sua_exploration.mc_maze import subm_v9_m30_true_early_start as subject


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "sua_exploration/scripts/run_subm_v9_m30_true_early_start.py"
FINALIZER_PATH = REPO_ROOT / "sua_exploration/scripts/finalize_subm_v9_m30_true_early_start.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("subm_m30_early_runner_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finalizer_module():
    spec = importlib.util.spec_from_file_location("subm_m30_early_finalizer_test", FINALIZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seal_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    os.chmod(path, 0o444)
    return hashlib.sha256(raw).hexdigest()


def _seal_npz(path: Path, prediction: np.ndarray, target: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, predictions=prediction, targets=target)
    os.chmod(path, 0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trials():
    # Every trial has six candidate 50-bin windows.  The support boundary is
    # trial 30's exclusive stop, 3000.
    return [
        {"start": index * 100, "stop": index * 100 + 55}
        for index in range(35)
    ]


def test_true_early_query_requires_exact_trials_after_first_30():
    trials = _trials()
    starts = subject.expected_valid_starts_after_trial_prefix(trials)
    audit = subject.audit_true_early_start_query(trials, starts)
    assert audit.support_trials == 30
    assert audit.support_end_bin_exclusive == 2_955
    assert audit.first_query_trial_start_bin == 3_000
    assert audit.observed_query_window_count == 30
    assert audit.query_history_fully_after_support is True
    assert audit.observed_equals_trials_after_support is True


def test_true_early_query_rejects_leaky_or_missing_start():
    trials = _trials()
    starts = subject.expected_valid_starts_after_trial_prefix(trials)
    with pytest.raises(subject.EarlyStartError, match="valid_starts differs"):
        subject.audit_true_early_start_query(trials, starts[1:])
    with pytest.raises(subject.EarlyStartError, match="valid_starts differs"):
        subject.audit_true_early_start_query(trials, np.concatenate(([2_954], starts)))


def test_post50_set_must_be_an_exact_early_start_suffix():
    early_starts = np.arange(300, 390, dtype=np.int64)
    late_starts = np.arange(350, 390, dtype=np.int64)
    mask = subject.post50_suffix_mask(early_starts, late_starts)
    assert mask.dtype == bool
    assert mask.sum() == 40
    assert np.array_equal(early_starts[mask], late_starts)
    with pytest.raises(subject.EarlyStartError, match="exact.*suffix"):
        subject.post50_suffix_mask(early_starts, np.arange(349, 389, dtype=np.int64))


def test_dataset_target_materialization_matches_last_history_bin():
    class Record:
        neural = np.zeros((80, 4), dtype=np.float32)
        behavior = np.arange(160, dtype=np.float32).reshape(80, 2)
        valid_starts = np.asarray([0, 10, 30], dtype=np.int64)

    target = subject.query_targets_from_record(Record())
    assert np.array_equal(target, Record.behavior[np.asarray([49, 59, 79])])
    assert target.dtype == np.float32


def test_expected_cell_count_is_fixed_three_arm_matrix():
    assert subject.expected_cells_for_session_count(15) == 270
    assert subject.expected_cells_for_session_count(1) == 18
    with pytest.raises(subject.EarlyStartError):
        subject.expected_cells_for_session_count(16)


def test_runner_contract_pins_budget_audit_and_three_arm_270_topology():
    runner = _runner_module()
    assert "sua_exploration/mc_maze/subm_v9_t4_label_budget.py" in runner.SOURCE_RELATIVE_PATHS
    assert tuple(runner.early.ARMS) == ("shared_t4", "shared_zero4", "shared_ts4")
    assert runner.early.EXPECTED_CELLS == 270
    assert runner.early.expected_cells_for_session_count(15) == 270


def test_runner_receipt_comparison_accepts_only_canonical_json_roundtrip_normalization():
    runner = _runner_module()
    prepared = {
        "design": {"present_direction_indices": (0, 1, 2), "direction_counts": {0: 4, 1: 5}},
        "nested": {"tuple": ("a", "b")},
    }
    receipt = {
        "design": {"present_direction_indices": [0, 1, 2], "direction_counts": {"0": 4, "1": 5}},
        "nested": {"tuple": ["a", "b"]},
    }
    assert runner.canonical_json_equivalent(prepared, receipt) is True
    receipt["design"]["direction_counts"]["1"] = 6
    assert runner.canonical_json_equivalent(prepared, receipt) is False


def test_runner_manifest_race_waits_for_0444_then_requires_exact_payload(tmp_path, monkeypatch):
    runner = _runner_module()
    path = tmp_path / "run_manifest.json"
    payload = {"schema": "test", "value": [1, 2, 3]}
    path.write_bytes(runner.canonical_bytes(payload))
    os.chmod(path, 0o600)
    calls = []

    def seal_after_one_wait(_seconds):
        calls.append(_seconds)
        os.chmod(path, 0o444)

    monkeypatch.setattr(runner.time, "sleep", seal_after_one_wait)
    runner.wait_for_exact_sealed_json(path, payload, attempts=2)
    assert calls == [0.05]
    with pytest.raises(RuntimeError, match="output manifest drift"):
        runner.wait_for_exact_sealed_json(path, {"schema": "wrong"}, attempts=1)


def test_runner_has_no_metric_optimizer_or_backward_call_sites():
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    assert "R2Score" not in calls
    assert "recompute_torchmetrics_r2_cpu" not in calls
    assert "backward" not in calls
    assert "step" not in calls
    assert "zero_grad" not in calls


def test_runner_refuses_preflight_slice(monkeypatch):
    runner = _runner_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--mode", "preflight",
            "--nwb-root", "/tmp/nwb",
            "--input-manifest", "/tmp/input.json",
            "--reference-v9-root", "/tmp/v9",
            "--preflight-receipt", "/tmp/receipt.json",
            "--session-offset", "1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.parse_args()
    assert error.value.code == 2


def test_finalizer_keeps_three_noninterchangeable_latency_endpoints_and_validates_commit_binding():
    source = FINALIZER_PATH.read_text(encoding="utf-8")
    assert '"all_post30"' in source
    assert '"new_early_only"' in source
    assert '"post50_suffix"' in source
    assert "r2_new_early_only_trials31_to_50" in source
    for required_commit_field in (
        "prediction_sha256",
        "artifact_bytes",
        "checkpoint_sha256",
        "checkpoint_path",
        "runtime_contract_sha256",
    ):
        assert required_commit_field in source


def test_finalizer_accepts_exact_full_270_contract_without_opening_metric(tmp_path):
    """Exercise runner-commit/finalizer-contract field compatibility only."""
    finalizer = _finalizer_module()
    root = tmp_path / "m30"
    reference = tmp_path / "v9"
    reference_manifest_sha = _seal_json(reference / "run_manifest.json", {"reference": "v9"})
    target = np.arange(8, dtype=np.float32).reshape(4, 2)
    prediction = target / 2.0
    checkpoint_sha = "a" * 64
    checkpoint_path = "/frozen/checkpoint.ckpt"
    cohort = []
    for session_index in range(subject.EXPECTED_SESSIONS):
        asset_id = f"asset{session_index:02d}"
        session_id = f"session{session_index:02d}"
        views = {}
        for view in subject.VIEWS:
            views[view] = {
                "query": {
                    "observed_query_window_count": 4,
                    "post50_suffix_window_count": 2,
                    "target_sha256": finalizer.sha_array(target),
                }
            }
            for arm in subject.ARMS:
                for seed in subject.SEEDS:
                    ref_artifact = reference / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"
                    _seal_npz(ref_artifact, prediction[-2:], target[-2:])
                    _seal_json(
                        reference / "commits" / asset_id / view / arm / f"seed_{seed}.json",
                        {"evidence": {"checkpoint_sha256": checkpoint_sha, "checkpoint_path": checkpoint_path}},
                    )
                    artifact = root / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"
                    artifact_sha = _seal_npz(artifact, prediction, target)
                    _seal_json(
                        root / "commits" / asset_id / view / arm / f"seed_{seed}.json",
                        {
                            "schema": subject.SCHEMA,
                            "metric_computed": False,
                            "backward_called": False,
                            "asset_id": asset_id,
                            "session_id": session_id,
                            "view": view,
                            "arm": arm,
                            "seed": seed,
                            "artifact": str(artifact.relative_to(root)),
                            "artifact_bytes": artifact.stat().st_size,
                            "artifact_sha256": artifact_sha,
                            "query_window_count": 4,
                            "query_target_sha256": finalizer.sha_array(target),
                            "prediction_sha256": finalizer.sha_array(prediction),
                            "checkpoint_sha256": checkpoint_sha,
                            "checkpoint_path": checkpoint_path,
                            "runtime_contract_sha256": "contract",
                        },
                    )
        cohort.append({"asset_id": asset_id, "session_id": session_id, "views": views})
    receipt_sha = _seal_json(
        root / "preflight" / "preflight_receipt.json",
        {
            "schema": subject.SCHEMA,
            "protocol": {"expected_full_cells": subject.EXPECTED_CELLS},
            "metric_computed": False,
            "backward_called": False,
            "cohort": cohort,
        },
    )
    _seal_json(
        root / "run_manifest.json",
        {
            "schema": subject.SCHEMA,
            "status": "FORWARD_ONLY_NO_METRIC",
            "protocol": {"expected_full_cells": subject.EXPECTED_CELLS},
            "preflight_receipt": str(root / "preflight" / "preflight_receipt.json"),
            "preflight_receipt_sha256": receipt_sha,
            "reference_v9_manifest_sha256": reference_manifest_sha,
            "runtime_contract_sha256": "contract",
        },
    )
    _manifest, _receipt, cells = finalizer.validate_complete_topology(root, reference)
    assert len(cells) == subject.EXPECTED_CELLS
    assert all(cell["new_early_only_window_count"] == 2 for cell in cells)
