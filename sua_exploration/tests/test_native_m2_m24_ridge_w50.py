"""Focused synthetic contracts for the independent Native-M2 M24 ridge arm."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import csv
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import native_m2_m24_ridge_w50 as ridge


ROOT = Path(__file__).resolve().parents[2]


def _runner_module():
    path = ROOT / "sua_exploration/scripts/run_native_m2_m24_ridge_w50.py"
    spec = importlib.util.spec_from_file_location("native_m2_m24_ridge_w50_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finalizer_module():
    path = ROOT / "sua_exploration/scripts/finalize_native_m2_m24_ridge_w50.py"
    spec = importlib.util.spec_from_file_location("native_m2_m24_ridge_w50_finalizer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_raw_session() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 26 fixed-length trials gives M24 support plus exactly two query trials.
    bins_per_trial = 60
    time = 26 * bins_per_trial
    neural = np.arange(time * ridge.CHANNELS, dtype=np.float32).reshape(time, ridge.CHANNELS)
    velocity = np.column_stack((np.arange(time), -np.arange(time))).astype(np.float32)
    trial_change = np.zeros(time, dtype=bool)
    trial_change[::bins_per_trial] = True
    eval_mask = np.ones(time, dtype=bool)
    return neural, velocity, trial_change, eval_mask


def _strict_reference_manifest() -> dict:
    audit = {}
    for session, expected in ridge.EXPECTED_HELDOUT_LAYOUT.items():
        audit[session] = {
            "support_trials": ridge.CALIBRATION_TRIALS,
            "query_start_trial": ridge.CALIBRATION_TRIALS,
            "query_trials": expected["total_trials"] - ridge.CALIBRATION_TRIALS,
            "window_size": ridge.WINDOW_BINS,
            "full_window_disjoint": True,
            "raw_query_start_bin": expected["support_boundary_raw_bin"],
            "minimum_window_start_padded_bin": expected["support_boundary_raw_bin"] + ridge.WINDOW_BINS - 1,
            "eligible_windows": expected["query_rows"],
        }
    return {
        "heldout_evaluated_in_fit": False,
        "heldout_evaluated_in_test": True,
        "query_start_trial": ridge.CALIBRATION_TRIALS,
        "heldout_query_window_audit": audit,
    }


def test_raw_w50_layout_is_strictly_disjoint_without_left_padding() -> None:
    neural, velocity, trial_change, eval_mask = _synthetic_raw_session()
    layout = ridge.chronological_m24_layout("synthetic", neural, velocity, trial_change, eval_mask)
    boundary = 24 * 60

    assert layout.support_boundary_raw_bin == boundary
    assert layout.support_target_bins[0] == ridge.WINDOW_BINS - 1
    assert layout.support_target_bins[-1] == boundary - 1
    assert layout.query_target_bins[0] == boundary + ridge.WINDOW_BINS - 1
    assert layout.query_target_bins[0] - ridge.WINDOW_BINS + 1 == boundary
    assert layout.support_target_bins[-1] < layout.query_target_bins[0] - ridge.WINDOW_BINS + 1

    query_feature = ridge.materialize_raw_w50_features(neural, layout.query_target_bins[:1])
    np.testing.assert_array_equal(query_feature.reshape(ridge.WINDOW_BINS, ridge.CHANNELS), neural[boundary : boundary + ridge.WINDOW_BINS])
    np.testing.assert_array_equal(
        ridge.velocity_targets_at_bins(velocity, layout.query_target_bins[:1]),
        velocity[[boundary + ridge.WINDOW_BINS - 1]],
    )


def test_dual_solution_has_ridge_normal_equation_and_compiles_exactly() -> None:
    rng = np.random.default_rng(20260806)
    # The actual W50 feature dimension is retained, while the row count stays
    # small enough for a fast synthetic test of the dual system.
    x = rng.normal(size=(31, ridge.FEATURE_DIM)).astype(np.float32)
    y = rng.normal(size=(31, ridge.OUTPUT_DIM)).astype(np.float32)
    fitted = ridge.fit_dual_ridge_w50(x, y, normalized_lambda=1.0, device="cpu")
    compiled = ridge.compile_raw_ridge(fitted)

    z = (x.astype(np.float64) - fitted.feature_mean.astype(np.float64)) / fitted.feature_scale.astype(np.float64)
    yc = y.astype(np.float64) - fitted.target_mean.astype(np.float64)
    w = fitted.standardized_weights.astype(np.float64)
    gradient = z.T @ (z @ w - yc) / float(x.shape[0]) + w
    np.testing.assert_allclose(gradient, 0.0, rtol=0.0, atol=3.0e-5)

    normalized_prediction = z @ w + fitted.target_mean.astype(np.float64)
    compiled_prediction = ridge.predict_compiled_raw_ridge(x, compiled).astype(np.float64)
    np.testing.assert_allclose(compiled_prediction, normalized_prediction, rtol=3.0e-5, atol=3.0e-5)


def test_dual_ridge_normalizes_only_support_and_preserves_constant_intercept() -> None:
    x = np.full((8, ridge.FEATURE_DIM), 3.0, dtype=np.float32)
    y = np.tile(np.asarray([[2.5, -4.5]], dtype=np.float32), (8, 1))
    fitted = ridge.fit_dual_ridge_w50(x, y)
    compiled = ridge.compile_raw_ridge(fitted)
    prediction = ridge.predict_compiled_raw_ridge(x, compiled)

    np.testing.assert_allclose(prediction, y, atol=1.0e-6)
    assert fitted.normalized_lambda == ridge.RIDGE_NORMALIZED_LAMBDA
    assert np.all(fitted.feature_scale == 1.0)


def test_reference_manifest_binds_exact_six_session_m24_layout() -> None:
    manifest = _strict_reference_manifest()
    audit = ridge.validate_reference_split_manifest(manifest)
    assert set(audit) == set(ridge.EXPECTED_HELDOUT_SESSIONS)

    bad = _strict_reference_manifest()
    bad["heldout_query_window_audit"]["ses-2020-10-30-Run1"]["eligible_windows"] += 1
    with pytest.raises(ridge.NativeM2RidgeError, match="strict M24 reference query layout drift"):
        ridge.validate_reference_split_manifest(bad)


def test_profile_is_compiled_w50_not_runtime_normalizer_state() -> None:
    profile = ridge.compiled_streaming_profile()
    assert profile["feature_dim"] == 4800
    assert profile["online_weight_bytes"] == 38_408
    assert profile["online_history_bytes"] == 19_200
    assert profile["online_total_state_bytes"] == 57_608
    assert profile["macs_per_output"] == 9_600
    assert profile["macs_per_second_at_50hz"] == 480_000


def test_torchmetrics_151_r2_primitive_is_pinned_and_separate() -> None:
    torchmetrics = pytest.importorskip("torchmetrics")
    assert torchmetrics.__version__ == "1.5.1"
    target = np.asarray([[1.0, -1.0], [3.0, 2.0], [5.0, 7.0]], dtype=np.float32)
    assert ridge.score_r2_variance_weighted_torchmetrics151(target, target) == pytest.approx(1.0)


def test_runner_has_no_final_metric_path_and_has_shard_argument() -> None:
    runner = (ROOT / "sua_exploration/scripts/run_native_m2_m24_ridge_w50.py").read_text(encoding="utf-8")
    assert 'choices=("preflight", "forward")' in runner
    assert "--sessions" in runner
    assert "metric_computed\": False" in runner
    assert "score_r2_variance_weighted_torchmetrics151(" not in runner
    assert "R2Score(" not in runner


def test_forward_rejects_any_receipt_vs_recomputed_payload_drift(tmp_path: Path) -> None:
    runner = _runner_module()
    manifest_path = tmp_path / "reference_split_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    sessions = ("ses-2020-10-30-Run1",)
    expected = {
        "program_id": ridge.PROGRAM_ID,
        "mode": "preflight",
        "metric_computed": False,
        "predictions_generated": False,
        "selected_sessions": list(sessions),
        "reference_split_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "implementation": {"numerical_core_sha256": "core-a", "runner_sha256": "runner-a"},
        "session_source_receipts": {sessions[0]: {"source_sha256": "source-a"}},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(expected), encoding="utf-8")
    actual = runner.validate_preflight_receipt(
        receipt_path,
        reference_manifest_path=manifest_path,
        sessions=sessions,
        expected_payload=expected,
    )
    assert actual == expected

    drifted = dict(expected)
    drifted["implementation"] = dict(expected["implementation"])
    drifted["implementation"]["runner_sha256"] = "changed-code"
    receipt_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ridge.NativeM2RidgeError, match="immediately recomputed source/code audit"):
        runner.validate_preflight_receipt(
            receipt_path,
            reference_manifest_path=manifest_path,
            sessions=sessions,
            expected_payload=expected,
        )


def test_finalizer_binds_aggregate_and_both_completed_reference_csvs(tmp_path: Path) -> None:
    finalizer = _finalizer_module()
    k4 = {session: 0.2 + 0.01 * index for index, session in enumerate(ridge.EXPECTED_HELDOUT_SESSIONS)}
    t4 = {session: value - 0.02 for session, value in k4.items()}

    def write_metrics(path: Path, values: dict[str, float]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "session", "M", "R2_variance_weighted"])
            writer.writeheader()
            for session, value in values.items():
                writer.writerow({"split": "test_heldout", "session": session, "M": 24, "R2_variance_weighted": value})

    k4_path, t4_path = tmp_path / "k4.csv", tmp_path / "t4.csv"
    write_metrics(k4_path, k4)
    write_metrics(t4_path, t4)
    per_session = {session: k4[session] - t4[session] for session in ridge.EXPECTED_HELDOUT_SESSIONS}
    aggregate = {
        "formal_heldout_evaluated": True,
        "cell": {"task": "m2", "calibration_trials": 24, "window_size": 50},
        "arms": {"k4": {"mean_r2": float(np.mean(list(k4.values())))}, "t4": {"mean_r2": float(np.mean(list(t4.values())))}}
        ,
        "paired_deltas_r2": {"K4_minus_T4": {"mean": float(np.mean(list(per_session.values()))), "per_session": per_session}},
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    binding = finalizer.validate_existing_m24_reference(
        aggregate_path,
        k4_metrics=k4_path,
        t4_metrics=t4_path,
        k4_scores=finalizer.read_completed_arm_scores(k4_path, arm="K4"),
        t4_scores=finalizer.read_completed_arm_scores(t4_path, arm="T4"),
    )
    assert binding["aggregate_k4_minus_t4_verified"] is True

    aggregate["paired_deltas_r2"]["K4_minus_T4"]["per_session"][ridge.EXPECTED_HELDOUT_SESSIONS[0]] += 0.1
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    with pytest.raises(ridge.NativeM2RidgeError, match="aggregate K4−T4"):
        finalizer.validate_existing_m24_reference(
            aggregate_path,
            k4_metrics=k4_path,
            t4_metrics=t4_path,
            k4_scores=k4,
            t4_scores=t4,
        )


def test_finalizer_rejects_prediction_artifact_hash_drift_across_two_shards(tmp_path: Path) -> None:
    finalizer = _finalizer_module()
    sessions = list(ridge.EXPECTED_HELDOUT_SESSIONS)
    commits: list[Path] = []
    first_artifact: Path | None = None
    for shard_index, shard_sessions in enumerate((sessions[:3], sessions[3:])):
        shard = tmp_path / f"shard_{shard_index}"
        receipt = {
            "program_id": ridge.PROGRAM_ID,
            "mode": "preflight",
            "metric_computed": False,
            "predictions_generated": False,
            "reference_split_manifest": {"path": "/frozen/manifest.json", "sha256": "a" * 64},
            "implementation": {"numerical_core_sha256": "b" * 64, "runner_sha256": "c" * 64},
            "contract": {"window_bins": 50, "normalized_lambda": 1.0},
        }
        receipt_path = shard / "preflight.json"
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        rows = {}
        for session in shard_sessions:
            count = ridge.EXPECTED_HELDOUT_LAYOUT[session]["query_rows"]
            bins = np.arange(count, dtype=np.int64)
            artifact = shard / f"{session}.npz"
            np.savez_compressed(
                artifact,
                predictions=np.zeros((count, 2), dtype=np.float32),
                targets=np.zeros((count, 2), dtype=np.float32),
                query_target_bins=bins,
            )
            if first_artifact is None:
                first_artifact = artifact
            rows[session] = {
                "layout": {"query_rows": count, "query_target_bins_sha256": ridge.sha256_array_int64(bins)},
                "prediction_target_artifact": {"path": str(artifact), "sha256": finalizer.sha256_file(artifact)},
            }
        commit = {
            "program_id": ridge.PROGRAM_ID,
            "mode": "forward_only_unscored",
            "metric_computed": False,
            "selection_performed": False,
            "preflight_receipt": {"path": str(receipt_path), "sha256": finalizer.sha256_file(receipt_path)},
            "preflight_contract_sha256": hashlib.sha256(finalizer.canonical_bytes(receipt)).hexdigest(),
            "sessions": rows,
        }
        commit_path = shard / "forward_commit.json"
        commit_path.write_text(json.dumps(commit), encoding="utf-8")
        commits.append(commit_path)

    loaded, _binding = finalizer.load_two_shard_predictions(commits)
    assert set(loaded) == set(ridge.EXPECTED_HELDOUT_SESSIONS)
    assert first_artifact is not None
    with first_artifact.open("ab") as handle:
        handle.write(b"hash-drift")
    with pytest.raises(ridge.NativeM2RidgeError, match="prediction artifact hash drift"):
        finalizer.load_two_shard_predictions(commits)
