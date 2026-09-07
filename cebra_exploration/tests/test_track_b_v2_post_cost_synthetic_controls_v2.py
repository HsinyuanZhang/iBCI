"""No-GPU/no-target adversarial tests for fresh v2 synthetic controls."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
VENDOR = ROOT / "cebra_exploration/third_party/cebra"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_actual_cpu_route as route  # noqa: E402
import track_b_v2_post_cost_synthetic_controls_v2 as controls  # noqa: E402


RUNNER = ROOT / "cebra_exploration/scripts/run_track_b_v2_post_cost_synthetic_controls_v2.py"
ALIGNMENT = {"offset_left": 5, "offset_right": 5, "valid_slice_start": 5, "valid_slice_stop": -5}


def _runner(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{SRC}:{VENDOR}"
    return subprocess.run([sys.executable, str(RUNNER), *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)


def test_exact_four_arm_lifecycles_and_full_readout_topology() -> None:
    assert controls.ARMS == (
        "cebra_joint_behavior", "cebra_frozen_source_adapt",
        route.UNALIGNED_ARM, route.DERANGED_ARM,
    )
    assert controls.FIT_CALLS_PER_ARM == {
        "cebra_joint_behavior": 1,
        "cebra_frozen_source_adapt": 2,
        route.UNALIGNED_ARM: 3,
        route.DERANGED_ARM: 1,
    }
    assert controls.FIT_LIFECYCLE["cebra_frozen_source_adapt"] == (
        ("frozen_source__source_multisession_fit", 10_000),
        ("frozen_source__28_session_init_freeze_target_fit", 500),
    )
    assert controls.EXPECTED_ARM_RUNS == 32
    assert controls.EXPECTED_FIT_CALLS == 56
    assert controls.EXPECTED_MEASUREMENTS == 192
    assert controls.READOUT_ROUTES == (
        "source_only_consumer_mechanism_alignment",
        "target_support_only_standard_cebra_accuracy",
        "source_plus_target_support_hybrid_sensitivity",
    )
    fold = controls.synthetic_fold()
    authority = controls.synthetic_data_authority(fold)
    assert len(fold.peer_neural) == len(fold.peer_auxiliary) == 27
    assert authority["generator"]["peer_session_count"] == 27
    assert authority["generator"]["joint_session_model_count"] == 28


def test_live_preflight_binds_cost_and_recursive_runtime_closure() -> None:
    payload = controls.build_preflight()
    assert set(payload) == controls.PREFLIGHT_KEYS
    assert payload["canonical_cost_gate"]["canonical_body_sha256"] == controls.EXPECTED_COST_SHA256
    assert payload["expected_fit_call_count"] == 56
    closure = payload["implementation_closure"]
    cebra_files = closure["recursive_runtime"]["vendored_cebra_full_python_runtime"]["files"]
    tm_files = closure["recursive_runtime"]["torchmetrics151_full_python_runtime"]["files"]
    assert {row["relative_path"] for row in cebra_files} >= {
        "solver/multi_session.py", "data/multi_session.py", "integrations/sklearn/cebra.py",
    }
    assert "regression/r2.py" in {row["relative_path"] for row in tm_files}
    assert closure["fixed_files"]["vendored_cebra_provenance"]["path"].endswith("CEBRA_PROVENANCE.txt")
    assert closure["fixed_files"]["v2_focused_tests"]["path"] == str(Path(__file__).absolute())
    assert payload["synthetic_arrays_built"] is payload["gpu_used"] is False


def test_default_runner_no_gpu_data_or_write_and_dual_flags() -> None:
    assert not os.path.lexists(controls.CANONICAL_OUTPUT)
    completed = _runner()
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == controls.STATUS_PREFLIGHT
    assert payload["torch_imported"] is payload["cebra_imported"] is False
    assert payload["target_data_opened"] is payload["NWB_or_NPZ_opened"] is False
    assert not os.path.lexists(controls.CANONICAL_OUTPUT)
    for flag in ("--execute", "--i-have-authorization"):
        rejected = _runner(flag)
        assert rejected.returncode != 0 and "requires both" in rejected.stderr


def test_runner_no_override_surface_and_transactional_writer() -> None:
    text = RUNNER.read_text().lower()
    for forbidden in ("--target", "--formal", "--nwb", "--npz", "--seed", "--geometry",
                      "--iterations", "--threshold", "--output"):
        assert forbidden not in text
    assert "write_immutable_pair" in text
    assert "write_immutable_receipt" not in text
    assert text.index("controls.assert_output_fresh()") < text.index("    import torch")
    assert text.rindex("controls.assert_output_fresh()") < text.index("write_immutable_pair")


def test_output_conflicts_fail_before_cost_closure_or_synthetic_data(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for body_exists, side_exists in ((True, False), (False, True), (True, True)):
        path = tmp_path / f"case-{body_exists}-{side_exists}" / "receipt.json"
        path.parent.mkdir()
        if body_exists:
            path.write_text("poison")
        if side_exists:
            controls._sidecar(path).write_text("poison")
        monkeypatch.setattr(controls, "CANONICAL_OUTPUT", path)
        monkeypatch.setattr(controls, "validate_cost_pair",
                            lambda: (_ for _ in ()).throw(AssertionError("cost reached")))
        monkeypatch.setattr(controls, "implementation_closure",
                            lambda: (_ for _ in ()).throw(AssertionError("closure reached")))
        with pytest.raises(controls.PostCostSyntheticControlV2Error, match="fresh"):
            controls.build_preflight()


def test_each_block_cropped_before_route_concatenation_and_all_six_routes_scored() -> None:
    torch = pytest.importorskip("torch")
    torchmetrics = pytest.importorskip("torchmetrics")
    fold = controls.synthetic_fold()
    run = route.EmbeddingRun(
        tuple(np.asarray(x[:, :8], dtype=np.float64) for x in fold.peer_neural),
        np.asarray(fold.held_support_neural[:, :8], dtype=np.float64),
        np.asarray(fold.held_query_neural[:, :8], dtype=np.float64),
        ({"label": "fake"},), ALIGNMENT,
    )
    rows, alignment = controls.score_embedding_run(
        run=run, fold=fold, seed=0, arm="cebra_joint_behavior",
        torch=torch, torchmetrics=torchmetrics,
    )
    assert len(rows) == 6
    assert {(row["readout_route"], row["decoder"]) for row in rows} == {
        (readout, decoder) for readout in controls.READOUT_ROUTES for decoder in controls.DECODERS}
    assert all(set(row) == controls.MEASUREMENT_KEYS for row in rows)
    assert [proof["valid_row_count"] for proof in alignment["source_block_crops"]] == [86] * 27
    assert alignment["held_support_crop"]["valid_row_count"] == 38
    assert alignment["held_query_crop"]["valid_row_count"] == 38
    assert all(set(proof) == controls.CROP_PROOF_KEYS for proof in alignment["source_block_crops"])
    assert alignment["concatenate_then_crop_used"] is False
    headline = [row for row in rows if row["readout_route"] == controls.READOUT_ROUTES[1]]
    assert all(row["readout_fit_row_count"] == 38 for row in headline)
    hybrid = [row for row in rows if row["readout_route"] == controls.READOUT_ROUTES[2]]
    assert all(row["readout_fit_row_count"] == 2360 for row in hybrid)
    assert all(row["query_neural_or_auxiliary_in_fit"] is False for row in rows)


def test_short_block_and_concatenate_then_crop_geometry_fail_closed() -> None:
    with pytest.raises(route.TrackBV2ActualCpuError):
        controls._crop_block(np.zeros((9, 8)), np.zeros((9, 2)), ALIGNMENT, role="short")
    # Cropping a concatenated 27x96 block keeps 2582, not the required 27x86.
    x = np.zeros((2592, 8))
    y = np.zeros((2592, 2))
    cropped_x, _, _ = controls._crop_block(x, y, ALIGNMENT, role="poison_concatenation")
    assert cropped_x.shape[0] == 2582
    assert cropped_x.shape[0] != 27 * (96 - 10)


def test_frozen_positive_uses_distinct_source_then_init_freeze_target_lifecycle() -> None:
    text = Path(controls.__file__).read_text()
    assert '"frozen_source__source_multisession_fit"' in text
    assert '"frozen_source__28_session_init_freeze_target_fit"' in text
    assert "freeze_sessions=list(range(len(peers_x)))" in text
    assert "init_from=source_estimator" in text
    assert "len(joint_estimator.model_) == len(peers_x) + 1" in text


class _FakeBackend:
    identity = {"backend": "synthetic_test", "actual_cebra": False}

    def __init__(self, **_: object) -> None:
        pass

    def run_arm(self, *, arm: str, fold: route.SourcePseudoTargetFold,
                support_auxiliary: np.ndarray, seed: int) -> route.EmbeddingRun:
        del support_auxiliary
        return route.EmbeddingRun(
            tuple(np.zeros((x.shape[0], 8)) for x in fold.peer_neural),
            np.zeros((fold.held_support_neural.shape[0], 8)),
            np.zeros((fold.held_query_neural.shape[0], 8)),
            tuple({"label": label, "iterations": iterations, "requested_seed": seed}
                  for label, iterations in controls.FIT_LIFECYCLE[arm]),
            ALIGNMENT,
        )


def _fake_score(*, seed: int, arm: str, **_: object) -> tuple[list[dict], dict]:
    rows = []
    arm_value = {"cebra_joint_behavior": .8, "cebra_frozen_source_adapt": .7,
                 route.UNALIGNED_ARM: .1, route.DERANGED_ARM: -.3}[arm]
    for readout in controls.READOUT_ROUTES:
        for decoder in controls.DECODERS:
            metric = {key: None for key in controls.METRIC_KEYS}
            metric.update({"implementation": "torchmetrics.regression.R2Score", "torchmetrics_version": "1.5.1",
                           "multioutput": "variance_weighted", "dtype": "float32", "device": "cpu",
                           "update_call_count": 1, "compute_call_count": 1,
                           "prediction_shape": [38, 2], "prediction_float32_sha256": "1" * 64,
                           "target_float32_sha256": "2" * 64})
            row = {key: None for key in controls.MEASUREMENT_KEYS}
            row.update({"seed": seed, "arm": arm,
                        "arm_role": "positive_aligned" if arm in route.POSITIVE_ARMS else
                                    "ordinary_negative_diagnostic" if arm == route.UNALIGNED_ARM else
                                    "deranged_support_hard_null",
                        "readout_route": readout, "readout_role": "test", "decoder": decoder,
                        "target_query_r2": arm_value + seed * .001, "geometry": controls.GEOMETRY.as_dict(),
                        "fit_call_count_for_arm": controls.FIT_CALLS_PER_ARM[arm],
                        "linear_ridge_normalized_lambda": .01 if decoder == "linear_ridge" else None,
                        "cosine_knn_k": 3 if decoder == "knn_cosine_k3" else None,
                        "readout_fit_embedding_float64_sha256": "3" * 64,
                        "readout_fit_auxiliary_float64_sha256": "4" * 64, "readout_fit_row_count": 38,
                        "query_valid_embedding_float64_sha256": "5" * 64,
                        "query_valid_auxiliary_float64_sha256": "6" * 64,
                        "query_neural_or_auxiliary_in_fit": False, "metric_runtime": metric})
            rows.append(row)
    alignment = {"source_block_crops": [], "held_support_crop": {}, "held_query_crop": {},
                 "source_blocks_cropped_independently_before_concatenation": True,
                 "support_and_query_cropped_independently": True, "concatenate_then_crop_used": False,
                 "padded_edge_rows_used": 0, "query_neural_or_auxiliary_in_fit": False}
    return rows, alignment


def test_fake_full_execution_builds_exact_32_56_192_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = controls.build_preflight()
    closure = preflight["implementation_closure"]
    monkeypatch.setattr(controls, "VendoredCebraGpuControlBackendV2", _FakeBackend)
    monkeypatch.setattr(controls, "score_embedding_run", _fake_score)
    payload = controls.execute_controls(
        preflight=preflight, torch=object(), torchmetrics=object(), cebra=object(),
        cuda_identity={"logical_device": "test"}, launch_closure=closure,
    )
    controls.validate_measurement_receipt(payload)
    assert set(payload) == controls.RECEIPT_KEYS
    assert payload["arm_run_count"] == 32
    assert payload["cebra_fit_call_count"] == 56
    assert payload["decoder_measurement_count"] == 192
    assert payload["invalid_v1_attempt"]["receipt_minted"] is False
    assert payload["threshold_frozen"] is payload["target_execution_authorized"] is False


@pytest.mark.parametrize("mutation", (
    "extra_top_key", "missing_frozen", "fit_count", "missing_readout", "query_leak",
    "metric_extra_key", "threshold", "target_open", "closure", "self_sha",
))
def test_receipt_adversarial_tamper_fails_closed(
        monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    preflight = controls.build_preflight()
    closure = preflight["implementation_closure"]
    monkeypatch.setattr(controls, "VendoredCebraGpuControlBackendV2", _FakeBackend)
    monkeypatch.setattr(controls, "score_embedding_run", _fake_score)
    payload = controls.execute_controls(
        preflight=preflight, torch=object(), torchmetrics=object(), cebra=object(),
        cuda_identity={}, launch_closure=closure)
    if mutation == "extra_top_key":
        payload["poison"] = True
    elif mutation == "missing_frozen":
        payload["arms"].remove("cebra_frozen_source_adapt")
    elif mutation == "fit_count":
        payload["cebra_fit_call_count"] = 55
    elif mutation == "missing_readout":
        payload["measurements"].pop()
    elif mutation == "query_leak":
        payload["measurements"][0]["query_neural_or_auxiliary_in_fit"] = True
    elif mutation == "metric_extra_key":
        payload["measurements"][0]["metric_runtime"]["poison"] = True
    elif mutation == "threshold":
        payload["threshold_frozen"] = True
    elif mutation == "target_open":
        payload["target_data_opened"] = True
    elif mutation == "closure":
        payload["implementation_closure_at_final"] = {}
    elif mutation == "self_sha":
        payload["receipt_payload_sha256"] = "0" * 64
    if mutation != "self_sha":
        payload.pop("receipt_payload_sha256", None)
        payload["receipt_payload_sha256"] = controls._sha_json(payload)
    with pytest.raises(controls.PostCostSyntheticControlV2Error):
        controls.validate_measurement_receipt(payload)


def test_no_target_formal_nwb_loader_or_authority_minter_surface() -> None:
    text = Path(controls.__file__).read_text().lower()
    for forbidden in ("pynwb", "h5py", "np.load", "open_canonical_verified_asset_after_authority"):
        assert forbidden not in text
    assert '"target_execution_authorized": true' not in text
    assert '"threshold_frozen": true' not in text
