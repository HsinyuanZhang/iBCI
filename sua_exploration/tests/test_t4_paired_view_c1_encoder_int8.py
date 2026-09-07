from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "sua_exploration",
    ROOT / "sua_exploration/scripts",
    ROOT / "software-to-hardware",
    ROOT / "streaming_calibration_exp",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load(name: str, filename: str):
    path = ROOT / "sua_exploration/scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


writer = _load(
    "c1_int8_writer_test", "write_t4_paired_view_c1_encoder_int8_prelaunch.py"
)
evaluator = _load(
    "c1_int8_evaluator_test", "eval_t4_paired_view_c1_encoder_int8.py"
)
aggregate = _load(
    "c1_int8_aggregate_test", "aggregate_t4_paired_view_c1_encoder_int8.py"
)


def test_prelaunch_is_independent_dual_view_encoder_only_and_binds_seed44() -> None:
    receipt = writer.build_receipt()
    assert receipt["status"] == "authorized_for_c1_shared_encoder_ptq"
    assert receipt["ordinary_t4_int8_selection_receipt_reused"] is False
    assert receipt["c2_authorized"] is False
    assert receipt["formal_test_files_opened"] is False
    assert receipt["strict_manifest"]["counts"] == [27, 6, 6]
    assert receipt["strict_manifest"]["formal_paths_resolved"] is False
    protocol = receipt["frozen_protocol"]
    assert protocol["views"] == ["sua", "pseudo_mua"]
    assert protocol["scale_fit_records"] == 54
    assert protocol["scale_fit_view_weight"] == {"sua": 0.5, "pseudo_mua": 0.5}
    assert protocol["scale_statistic_equal_mass_per_view"] is True
    assert protocol["signed_t4_included_in_post0_input_scale_fit"] is True
    assert protocol["decoder_precision"] == "FP32"
    assert protocol["deployment_checkpoint"] == "epoch_011.ckpt"
    assert (
        receipt["seed_artifacts"]["44"]["checkpoint"]["sha256"]
        == "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6"
    )
    assert receipt["conditional_qat"]["run_all_three_seeds_if_triggered"] is True
    assert receipt["conditional_qat"]["mixed_ptq_qat_seed_aggregate_forbidden"] is True


def test_written_receipt_validates_and_seed_checkpoint_is_epoch11(tmp_path: Path) -> None:
    receipt = writer.build_receipt()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    validated = writer.validate_receipt(path)
    assert validated["c1_positive_trigger"]["c1_pass"] is True
    seed = writer.validate_receipt(path, seed=42)
    assert seed["_validated_seed"]["checkpoint"].endswith("epoch_011.ckpt")


def _weights():
    from b3_hw_golden import B3Weights

    rng = np.random.default_rng(2)
    return B3Weights(
        pre_w=rng.normal(0, 0.02, (64, 100)).astype(np.float32),
        pre_b=np.zeros(64, dtype=np.float32),
        post0_w=rng.normal(0, 0.02, (64, 68)).astype(np.float32),
        post0_b=np.zeros(64, dtype=np.float32),
        post1_w=rng.normal(0, 0.02, (64, 64)).astype(np.float32),
        post1_b=np.zeros(64, dtype=np.float32),
        post2_w=rng.normal(0, 0.02, (50, 64)).astype(np.float32),
        post2_b=np.zeros(50, dtype=np.float32),
    )


def test_scale_fit_requires_and_records_equal_27x2_views() -> None:
    rng = np.random.default_rng(3)
    calibs = [rng.integers(0, 3, (30, 100, 1)).astype(np.float32) for _ in range(54)]
    sides = [rng.normal(0, 0.3, (1, 4)).astype(np.float32) for _ in range(54)]
    views = ["sua"] * 27 + ["pseudo_mua"] * 27
    names = [f"source-{index % 27}::{view}" for index, view in enumerate(views)]
    scales, rows = evaluator.select_dual_view_source_scales(
        _weights(), calibs, sides, views, names
    )
    assert len(rows) == len(evaluator.SCALE_CANDIDATES)
    assert len(scales.source_sessions) == 54
    assert set(rows[0]["train_identity_rmse_by_view"]) == {"sua", "pseudo_mua"}
    bad_views = ["sua"] * 28 + ["pseudo_mua"] * 26
    with pytest.raises(ValueError, match="equally weighted"):
        evaluator.select_dual_view_source_scales(
            _weights(), calibs, sides, bad_views, names
        )


def test_equal_mass_scale_statistic_preserves_each_view_and_extrema() -> None:
    left = np.arange(1000, dtype=np.float64)
    right = -np.arange(10, dtype=np.float64) * 100.0
    paired = evaluator._deterministic_equal_mass_pair(left, right, cap=10)
    assert paired.shape == (20,)
    assert np.max(np.abs(paired[:10])) == 999.0
    assert np.max(np.abs(paired[10:])) == 900.0


def test_metadata_validation_fails_closed_on_formal_or_nonpaired() -> None:
    metadata_path = (
        ROOT
        / "sua_exploration/results/"
        "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/"
        "shared_t4_s42/run_metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    evaluator._metadata_checks(metadata, 42)
    bad = {**metadata, "held_out_test_evaluated": True}
    with pytest.raises(ValueError, match="formal_sealed"):
        evaluator._metadata_checks(bad, 42)
    bad = {**metadata, "training_kind": "single_view"}
    with pytest.raises(ValueError, match="paired"):
        evaluator._metadata_checks(bad, 42)


def test_paired_stats_keep_seed_and_session_axes() -> None:
    matrix = np.asarray(
        [
            [-0.005, -0.004, -0.003, -0.002, -0.001, 0.0],
            [-0.006, -0.005, -0.004, -0.003, -0.002, -0.001],
            [-0.007, -0.006, -0.005, -0.004, -0.003, -0.002],
        ]
    )
    row = aggregate.summarize_delta_matrix(matrix, bootstrap_seed=9)
    assert row["overall_mean_ge_minus_0p01"] is True
    assert row["all_seed_means_ge_minus_0p01"] is True
    assert len(row["seed_mean_delta_r2"]) == 3
    assert len(row["session_mean_delta_r2"]) == 6
    assert len(row["cell_bootstrap_95_interval"]) == 2


def test_runner_is_seed_scoped_decoder_fp32_and_no_formal_argument() -> None:
    runner = (
        ROOT / "sua_exploration/scripts/run_t4_paired_view_c1_encoder_int8_ptq.sh"
    ).read_text()
    assert "epoch_011.ckpt" in runner
    assert "shared_t4_s${seed}" in runner
    assert "CUDA_VISIBLE_DEVICES" in runner
    assert "formal" not in runner.lower()
    source = (ROOT / "sua_exploration/scripts/eval_t4_paired_view_c1_encoder_int8.py").read_text()
    assert "decoder_quantized_in_this_run\": False" in source
    assert "validate_selection" not in source
