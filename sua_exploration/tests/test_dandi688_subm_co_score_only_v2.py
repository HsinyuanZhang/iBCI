"""Score-free and tiny-synthetic regressions for the external sub-M v2 runner.

No test in this file opens a checkpoint, normalizer or NWB, runs a model
forward, or uses a GPU.  The only TorchMetrics calls operate on five-row,
in-memory synthetic arrays in order to prove that aggregate verification does
not trust a sealed metric JSON scalar.  The source snapshot deliberately
includes this file, so its existence is required for v2's stored-prelaunch
closure.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze import subm_co_score_only_v2 as score_only_v2  # noqa: E402


def test_v2_dry_run_is_static_and_reports_unpinned_parity_blocker() -> None:
    plan = score_only_v2.build_dry_run_plan_v2(ROOT)
    assert plan["status"] == "NOT_AUTHORIZED_FOR_SCORING"
    assert plan["frozen_N"] == 15
    assert plan["scorer_adapter_parity_receipt_pin_configured"] is False
    assert plan["root_ed25519_key_pin_configured"] is False
    assert plan["no_checkpoint_or_nwb_opened"] is True
    counters = plan["audit_counters"]
    assert counters["checkpoint_load_calls"] == 0
    assert counters["model_forward_calls"] == 0
    assert counters["r2_update_calls"] == 0
    assert counters["r2_compute_calls"] == 0
    assert counters["normalizer_hash_checks"] == 0


def test_v2_prelaunch_bundle_is_write_once_and_carries_the_four_file_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "v2-prelaunch"
    result = score_only_v2.write_prelaunch_artifacts_v2(output, ROOT)
    assert result["status"] == "NOT_AUTHORIZED_FOR_SCORING"
    stored = score_only_v2.load_stored_prelaunch_bundle(output, ROOT)
    assert set(stored["runner_source_snapshot"]) == set(score_only_v2.V2_SOURCE_RELATIVE_PATHS)
    assert stored["draft"]["scorer_adapter_parity_requirement"]["receipt_pin_configured"] is False
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


def _cpu_binding() -> dict[str, object]:
    return {
        "torch_device": "cpu",
        "kind": "cpu",
        "matrix_torch_device": "cpu",
        "host": None,
        "physical_gpu_uuid": None,
        "physical_gpu_pci_bus_id": None,
        "cuda_visible_devices": None,
        "all_180_cells_same_device": True,
    }


def test_v1_six_pins_and_v2_device_cli_remain_explicit() -> None:
    assert score_only_v2.verify_v1_immutable(ROOT) == {
        key: pin.sha256 for key, pin in score_only_v2.V1_IMMUTABLE_PINS.items()
    }
    entry_source = (ROOT / "sua_exploration/scripts/run_dandi688_subm_co_score_only_v2.py").read_text(encoding="utf-8")
    assert '"--device"' in entry_source


def test_cuda_binding_normalizes_torch_bare_uuid_and_rejects_device_mismatch_without_gpu() -> None:
    binding = {
        "torch_device": "cuda:0",
        "kind": "cuda",
        "matrix_torch_device": "cuda:0",
        "host": "scoring-host-01",
        "physical_gpu_uuid": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
        "physical_gpu_pci_bus_id": "00000000:01:00.0",
        "cuda_visible_devices": "0",
        "all_180_cells_same_device": True,
    }
    parsed = score_only_v2.validate_execution_device_binding(
        binding,
        requested_device="cuda:0",
        runtime_identity={
            "torch_device": "cuda:0",
            "host": "scoring-host-01",
            # Torch 2.5 exposes this bare spelling; runtime validation must
            # normalize it before comparison with the pinned GPU- form.
            "physical_gpu_uuid": "ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
            "physical_gpu_pci_bus_id": "00000000:01:00.0",
            "cuda_visible_devices": "0",
        },
    )
    assert parsed.physical_gpu_uuid == binding["physical_gpu_uuid"]
    bare_authorization = dict(binding)
    bare_authorization["physical_gpu_uuid"] = "ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    assert score_only_v2.execution_device_binding_from_mapping(bare_authorization).physical_gpu_uuid == binding["physical_gpu_uuid"]
    with pytest.raises(score_only_v2.AuthorizationError, match="requested --device differs"):
        score_only_v2.validate_execution_device_binding(binding, requested_device="cpu")
    with pytest.raises(score_only_v2.AuthorizationError, match="physical GPU UUID differs"):
        score_only_v2.validate_execution_device_binding(
            binding,
            requested_device="cuda:0",
            runtime_identity={
                "torch_device": "cuda:0",
                "host": "scoring-host-01",
                "physical_gpu_uuid": "GPU-wrong",
                "physical_gpu_pci_bus_id": "00000000:01:00.0",
                "cuda_visible_devices": "0",
            },
        )
    assert score_only_v2.validate_execution_device_binding(_cpu_binding(), requested_device="cpu").kind == "cpu"


def test_full_ts4_vector_is_exact_and_not_hash_only() -> None:
    receipt = score_only_v2.realized_ts4_permutation(11, 42)
    assert receipt["vector"] == np.random.RandomState(42).permutation(11).tolist()
    assert len(receipt["vector"]) == 11
    assert score_only_v2.validate_realized_ts4_permutation(receipt, n_channels=11, seed=42) == receipt
    tampered = dict(receipt)
    tampered["vector"] = list(receipt["vector"])
    tampered["vector"][0], tampered["vector"][1] = tampered["vector"][1], tampered["vector"][0]
    with pytest.raises(score_only_v2.ScoreOnlyContractError, match="vector differs"):
        score_only_v2.validate_realized_ts4_permutation(tampered, n_channels=11, seed=42)


def test_free_form_signature_and_unpinned_subc_parity_receipt_cannot_authorize() -> None:
    with pytest.raises(score_only_v2.AuthorizationError, match="envelope object"):
        score_only_v2._validate_signature_envelope("root said yes", {})
    with pytest.raises(score_only_v2.AuthorizationError, match="CONSUMED_SUBC"):
        score_only_v2._verify_adapter_parity_receipt({}, ROOT)


def _synthetic_arrays() -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [3.0, -0.5], [4.0, 2.0]],
        dtype=np.float32,
    )
    predictions = np.asarray(
        [[0.1, 0.8], [1.2, -0.9], [1.8, 0.7], [3.1, -0.6], [3.7, 1.9]],
        dtype=np.float32,
    )
    return predictions, targets


def _write_tiny_full_matrix(
    tmp_path: Path,
    *,
    tamper_first: str | None = None,
) -> tuple[Path, score_only_v2.ScoreContract, dict[str, int]]:
    """Build a self-consistent 180-cell *synthetic* seal for verifier tests."""

    contract = score_only_v2.validate_authority_chain_v2(ROOT)
    output = tmp_path / "synthetic-seal"
    device = score_only_v2.execution_device_binding_from_mapping(_cpu_binding())
    predictions, targets = _synthetic_arrays()
    exact_r2 = score_only_v2.recompute_torchmetrics_r2_cpu(predictions, targets)
    expected_counts = {session.asset_id: predictions.shape[0] for session in contract.frozen_sessions}
    records: list[dict[str, object]] = []
    first = True
    for session in contract.frozen_sessions:
        for view in score_only_v2.VIEWS:
            for arm in score_only_v2.ARMS:
                for seed in score_only_v2.SEEDS:
                    terminal = contract.checkpoint_by_key()[(arm, seed)]
                    prefix = f"sealed/sessions/{session.asset_id}/{view}/{arm}/seed_{seed}"
                    prediction_relative = f"{prefix}/predictions_targets.npz"
                    metric_relative = f"{prefix}/metric.json"
                    if first and tamper_first == "canonical_path":
                        metric_relative = f"{prefix}/not_canonical_metric.json"
                    write_predictions = predictions.astype(np.float64) if first and tamper_first == "dtype" else predictions
                    prediction_path = score_only_v2._safe_relative_path(output, prediction_relative, label="synthetic prediction")
                    metric_path = score_only_v2._safe_relative_path(output, metric_relative, label="synthetic metric")
                    prediction_sha = score_only_v2._write_immutable_npz_exclusive(
                        prediction_path,
                        predictions=write_predictions,
                        targets=targets,
                    )
                    normalizer = contract.normalizers[view].as_dict()
                    if first and tamper_first == "normalizer":
                        normalizer = dict(normalizer)
                        # Preserve the side-feature semantic hash so this
                        # proves the verifier compares the *full* normalizer
                        # dictionary, not only that one semantic field.
                        normalizer["behavior"] = dict(normalizer["behavior"])
                        normalizer["behavior"]["sha256"] = "0" * 64
                    metric = {
                        "schema_version": score_only_v2.V2_SCHEMA_VERSION,
                        "kind": score_only_v2.V2_METRIC_KIND,
                        "sealed": True,
                        "metric_status": "FINITE_R2",
                        "scope_id": score_only_v2.SCOPE_ID,
                        "asset_id": session.asset_id,
                        "session_id": "wrong-session-id" if first and tamper_first == "session" else session.session_id,
                        "frozen_path": session.frozen_path,
                        "view": view,
                        "arm": arm,
                        "seed": seed,
                        "checkpoint_sha256": terminal.checkpoint.sha256,
                        "source_normalizer": normalizer,
                        "execution_device": device.as_dict(),
                        "evaluation": {"query_window_count": predictions.shape[0] + (1 if first and tamper_first == "count" else 0)},
                        "metric": {
                            "name": "torchmetrics.regression.R2Score",
                            "multioutput": score_only_v2.R2_MULTI_OUTPUT,
                            "r2": exact_r2 + (0.01 if first and tamper_first == "r2" else 0.0),
                            "aggregate_recompute_atol": score_only_v2.R2_RECOMPUTE_ATOL,
                        },
                        "prediction_target_bundle": {
                            "relative_path": prediction_relative,
                            "sha256": prediction_sha,
                            "shape": list(predictions.shape),
                        },
                        "ts4_realized_permutation": None if arm == "shared_t4" else score_only_v2.realized_ts4_permutation(7, seed),
                    }
                    metric_sha = score_only_v2._write_immutable_json_exclusive(metric_path, metric)
                    records.append(
                        {
                            "asset_id": session.asset_id,
                            "session_id": session.session_id,
                            "view": view,
                            "arm": arm,
                            "seed": seed,
                            "metric_relative_path": metric_relative,
                            "metric_sha256": metric_sha,
                            "prediction_relative_path": prediction_relative,
                            "prediction_sha256": prediction_sha,
                            "execution_device": device.as_dict(),
                        }
                    )
                    first = False
    seal = {
        "schema_version": score_only_v2.V2_SCHEMA_VERSION,
        "kind": score_only_v2.V2_SEAL_KIND,
        "status": "SEALED_PER_SESSION_PREDICTIONS_AND_METRICS",
        "scope_id": score_only_v2.SCOPE_ID,
        "expected_metric_cell_count": score_only_v2.EXPECTED_SEALED_METRIC_COUNT,
        "execution_device": device.as_dict(),
        "one_device_for_all_180_cells": True,
        "records": sorted(records, key=lambda item: (item["asset_id"], item["view"], item["arm"], item["seed"])),
        "aggregate_status": "NOT_OPENED",
    }
    score_only_v2._write_immutable_json_exclusive(
        score_only_v2._safe_relative_path(output, "sealed/seal_manifest.json", label="synthetic seal"),
        seal,
    )
    return output, contract, expected_counts


def test_writer_derives_exact_preflight_counts_before_it_can_seal(tmp_path: Path) -> None:
    contract = score_only_v2.validate_authority_chain_v2(ROOT)
    counts = score_only_v2.expected_query_window_counts_from_preflight(contract, ROOT)
    assert sum(counts.values()) == 708_795
    writer = score_only_v2.SealedOutputWriterV2(
        tmp_path / "writer",
        contract,
        authorization_sha256="0" * 64,
        execution_device=score_only_v2.execution_device_binding_from_mapping(_cpu_binding()),
        authority_root=ROOT,
    )
    predictions, targets = _synthetic_arrays()
    session = contract.frozen_sessions[0]
    with pytest.raises(score_only_v2.ScoreOnlyContractError, match="pinned v2 preflight ledger"):
        writer.write_session_result(
            session=session,
            view="sua",
            arm="shared_t4",
            seed=42,
            checkpoint_sha256=contract.checkpoint_by_key()[("shared_t4", 42)].checkpoint.sha256,
            normalizer=contract.normalizers["sua"],
            predictions=predictions,
            targets=targets,
            query_window_count=predictions.shape[0],
            ts4_permutation=None,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torchmetrics") is None,
    reason="TorchMetrics is required for the writer's saved-float32 R2 check",
)
def test_writer_reports_r2_recomputed_from_reopened_saved_float32_npz(tmp_path: Path) -> None:
    contract = score_only_v2.validate_authority_chain_v2(ROOT)
    session = contract.frozen_sessions[0]
    count = score_only_v2.expected_query_window_counts_from_preflight(contract, ROOT)[session.asset_id]
    predictions, targets = _synthetic_arrays()
    predictions = np.resize(predictions, (count, 2)).astype(np.float32, copy=False)
    targets = np.resize(targets, (count, 2)).astype(np.float32, copy=False)
    output = tmp_path / "writer-float32"
    writer = score_only_v2.SealedOutputWriterV2(
        output,
        contract,
        authorization_sha256="1" * 64,
        execution_device=score_only_v2.execution_device_binding_from_mapping(_cpu_binding()),
        authority_root=ROOT,
    )
    writer.write_session_result(
        session=session,
        view="sua",
        arm="shared_t4",
        seed=42,
        checkpoint_sha256=contract.checkpoint_by_key()[("shared_t4", 42)].checkpoint.sha256,
        normalizer=contract.normalizers["sua"],
        predictions=predictions,
        targets=targets,
        query_window_count=count,
        ts4_permutation=None,
    )
    metric_path = next(output.rglob("metric.json"))
    prediction_path = next(output.rglob("*.npz"))
    metric = score_only_v2._load_json(metric_path, "writer synthetic metric")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        assert loaded["predictions"].dtype == np.float32
        assert loaded["targets"].dtype == np.float32
        expected_r2 = score_only_v2.recompute_torchmetrics_r2_cpu(loaded["predictions"], loaded["targets"])
    assert metric["metric"]["r2"] == pytest.approx(expected_r2, abs=0.0)
    assert stat.S_IMODE(metric_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(prediction_path.stat().st_mode) == 0o444


@pytest.mark.skipif(
    importlib.util.find_spec("torchmetrics") is None,
    reason="TorchMetrics is required for mandated exact synthetic R2 recomputation",
)
def test_all_180_sealed_npzs_are_recomputed_and_npz_tampering_is_hash_rejected(tmp_path: Path) -> None:
    output, contract, synthetic_counts = _write_tiny_full_matrix(tmp_path)
    grids, evidence = score_only_v2._read_and_verify_sealed_metric_grid_v2(
        output,
        contract,
        expected_query_window_counts=synthetic_counts,
    )
    assert evidence["r2_recomputation"]["cells_recomputed"] == 180
    assert grids["sua"]["shared_t4"].shape == (3, 15)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in output.rglob("*.json"))
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in output.rglob("*.npz"))
    # The public entry has no synthetic-count bypass: it derives the real
    # pinned ledger counts and rejects this tiny test matrix.
    with pytest.raises(score_only_v2.ScoreOnlyContractError, match="query-window count differs"):
        score_only_v2.read_and_verify_sealed_metric_grid_v2(output, contract, root=ROOT)
    prediction_path = next(output.rglob("*.npz"))
    os.chmod(prediction_path, 0o644)
    with prediction_path.open("ab") as handle:
        handle.write(b"synthetic-tamper")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(prediction_path, 0o444)
    with pytest.raises(score_only_v2.ScoreOnlyContractError, match="prediction hash drift"):
        score_only_v2._read_and_verify_sealed_metric_grid_v2(
            output,
            contract,
            expected_query_window_counts=synthetic_counts,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torchmetrics") is None,
    reason="TorchMetrics is required for mandated exact synthetic R2 recomputation",
)
def test_self_consistent_metric_json_r2_lie_is_recomputed_and_rejected(tmp_path: Path) -> None:
    output, contract, synthetic_counts = _write_tiny_full_matrix(tmp_path, tamper_first="r2")
    with pytest.raises(score_only_v2.ScoreOnlyContractError, match="differs from CPU TorchMetrics recomputation"):
        score_only_v2._read_and_verify_sealed_metric_grid_v2(
            output,
            contract,
            expected_query_window_counts=synthetic_counts,
        )


@pytest.mark.parametrize(
    ("tamper_first", "message"),
    [
        ("normalizer", "source normalizer full binding"),
        ("session", "session ID differs"),
        ("count", "query-window count differs"),
        ("canonical_path", "canonical path drift"),
        ("dtype", "exact float32 dtype"),
    ],
)
def test_self_consistent_provenance_count_path_and_dtype_tampering_fails(
    tmp_path: Path,
    tamper_first: str,
    message: str,
) -> None:
    output, contract, synthetic_counts = _write_tiny_full_matrix(tmp_path, tamper_first=tamper_first)
    with pytest.raises(score_only_v2.ScoreOnlyContractError, match=message):
        score_only_v2._read_and_verify_sealed_metric_grid_v2(
            output,
            contract,
            expected_query_window_counts=synthetic_counts,
        )
