"""Focused synthetic tests for the production V9 runtime ledger.

They never open an NWB or a real checkpoint.  The real-data smoke is run by
the explicit CLI after this module has passed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime


def _contract() -> runtime.RuntimeContract:
    cohort = tuple(
        runtime.CohortSession(
            asset_id=f"asset-{index:02d}",
            session_id=f"session-{index:02d}",
            frozen_path=f"sub-M/session-{index:02d}.nwb",
            nwb_sha256="a" * 64,
            nwb_bytes=100 + index,
            query_window_count=3,
        )
        for index in range(runtime.EXPECTED_SESSION_COUNT)
    )
    checkpoints = tuple(
        runtime.CheckpointSpec(
            arm=arm,
            seed=seed,
            path=f"/fake/{arm}-{seed}.ckpt",
            sha256=("b" if arm == "shared_t4" else "c" if arm == "shared_zero4" else "d") * 64,
            bytes=10,
            closure_path=f"/fake/{arm}-{seed}.json",
            closure_sha256="e" * 64,
            closure_bytes=11,
        )
        for arm in runtime.ARMS
        for seed in runtime.SEEDS
    )
    return runtime.RuntimeContract(
        cohort=cohort,
        checkpoints=checkpoints,
        teacher_path="/fake/teacher.ckpt",
        teacher_sha256="f" * 64,
        teacher_bytes=12,
        behavior_normalizer_paths={view: f"/fake/{view}-behavior.npz" for view in runtime.VIEWS},
        behavior_normalizer_sha256={view: "1" * 64 for view in runtime.VIEWS},
        side_normalizer_paths={view: f"/fake/{view}-side.npz" for view in runtime.VIEWS},
        side_normalizer_sha256={view: "2" * 64 for view in runtime.VIEWS},
        cohort_receipt_sha256="3" * 64,
        scope_manifest_sha256="4" * 64,
        query_map_sha256=runtime.canonical_sha256({row.asset_id: row.query_window_count for row in cohort}),
    )


def _arrays(key: runtime.CellKey) -> tuple[np.ndarray, np.ndarray]:
    target = np.ascontiguousarray([[0.0, 1.0], [1.0, 3.0], [2.0, 7.0]], dtype=np.float32)
    if key.arm == "shared_t4":
        prediction = target.copy()
    elif key.arm == "shared_zero4":
        prediction = np.zeros_like(target)
    else:
        prediction = np.ascontiguousarray(target * np.float32(0.5))
    return prediction, target


def test_input_manifest_round_trip_is_canonical_and_contract_bound(tmp_path: Path) -> None:
    contract = _contract()
    path = tmp_path / "input_manifest.json"
    digest = runtime.write_input_manifest(path, contract)
    reopened, reopened_digest = runtime.load_input_manifest(path, require_frozen_cohort=False)
    assert reopened == contract
    assert reopened_digest == digest
    assert path.stat().st_mode & 0o777 == 0o444


def test_input_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    contract = _contract()
    path = tmp_path / "input_manifest.json"
    runtime.write_input_manifest(path, contract)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    path.chmod(0o644)
    try:
        path.write_bytes(runtime.canonical_bytes(payload))
    finally:
        path.chmod(0o444)
    with pytest.raises(runtime.V9RuntimeError, match="contract SHA"):
        runtime.load_input_manifest(path, require_frozen_cohort=False)


def test_phase_a_resume_reopens_exact_arrays_and_finalizer_requires_all_270(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _contract()
    input_manifest = tmp_path / "input_manifest.json"
    input_sha = runtime.write_input_manifest(input_manifest, contract)
    monkeypatch.setattr(runtime, "load_input_manifest", lambda _path: (contract, input_sha))
    output = tmp_path / "output"
    runtime.initialize_output(
        output, contract, input_manifest_path=input_manifest, input_manifest_sha256=input_sha,
        execution_device={"device": "cpu", "torch_version": "test", "cuda_version": None,
                          "tf32_disabled": True, "deterministic_algorithms": True, "batch_size": 128},
    )
    first = contract.expected_keys[0]
    prediction, target = _arrays(first)
    runtime.write_cell_artifact(
        output_root=output,
        contract=contract,
        key=first,
        predictions=prediction,
        targets=target,
        evidence={"forward_only": True},
    )
    reopened_prediction, reopened_target = runtime._validate_committed_cell(
        output_root=output, contract=contract, key=first
    )
    assert reopened_prediction.tobytes() == prediction.tobytes()
    assert reopened_target.tobytes() == target.tobytes()
    with pytest.raises(runtime.V9RuntimeError, match="incomplete"):
        runtime.finalize_artifacts(output_root=output, contract=contract)


def test_resume_rejects_changed_execution_device(tmp_path: Path) -> None:
    contract = _contract()
    input_manifest = tmp_path / "input_manifest.json"
    input_sha = runtime.write_input_manifest(input_manifest, contract)
    output = tmp_path / "output"
    cpu_binding = {"device": "cpu", "torch_version": "test", "cuda_version": None,
                   "tf32_disabled": True, "deterministic_algorithms": True, "batch_size": 128}
    runtime.initialize_output(
        output, contract, input_manifest_path=input_manifest, input_manifest_sha256=input_sha,
        execution_device=cpu_binding,
    )
    gpu_binding = dict(cpu_binding, device="cuda:0")
    with pytest.raises(runtime.V9RuntimeError, match="output contract drift"):
        runtime.initialize_output(
            output, contract, input_manifest_path=input_manifest, input_manifest_sha256=input_sha,
            execution_device=gpu_binding,
        )


def test_full_270_finalizer_emits_paired_descriptor_contrasts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _contract()
    input_manifest = tmp_path / "input_manifest.json"
    input_sha = runtime.write_input_manifest(input_manifest, contract)
    monkeypatch.setattr(runtime, "load_input_manifest", lambda _path: (contract, input_sha))
    output = tmp_path / "output"
    runtime.initialize_output(
        output, contract, input_manifest_path=input_manifest, input_manifest_sha256=input_sha,
        execution_device={"device": "cpu", "torch_version": "test", "cuda_version": None,
                          "tf32_disabled": True, "deterministic_algorithms": True, "batch_size": 128},
    )
    for key in contract.expected_keys:
        prediction, target = _arrays(key)
        runtime.write_cell_artifact(
            output_root=output,
            contract=contract,
            key=key,
            predictions=prediction,
            targets=target,
            evidence={"forward_only": True},
        )
    result = runtime.finalize_artifacts(output_root=output, contract=contract)
    aggregate = runtime._read_json(Path(result["aggregate_path"]))
    assert aggregate["verified_cell_count"] == 270
    assert set(aggregate["paired_contrasts"]) == {
        "shared_t4_minus_shared_zero4",
        "shared_t4_minus_shared_ts4",
    }
    assert aggregate["paired_contrasts"]["shared_t4_minus_shared_zero4"]["sua"]["42"]["positive_session_count"] == 15


def test_frozen_metric_reproduction_matches_local_torchmetrics_within_validated_tolerance() -> None:
    torchmetrics = pytest.importorskip("torchmetrics.regression")
    import torch

    rng = np.random.default_rng(688)
    prediction = np.ascontiguousarray(rng.normal(size=(61, 2)).astype(np.float32))
    target = np.ascontiguousarray(rng.normal(size=(61, 2)).astype(np.float32))
    metric = torchmetrics.R2Score(multioutput="variance_weighted")
    metric.update(torch.as_tensor(prediction), torch.as_tensor(target))
    expected = float(metric.compute().item())
    observed = runtime.validated_numpy_torchmetrics_151_approximation(prediction, target)
    assert abs(observed - expected) <= 2.0e-6


def test_runtime_source_uses_v3r2_query_window_count_and_never_metric_in_collector() -> None:
    source = (Path(__file__).resolve().parents[1] / "mc_maze/subm_co_three_arm_v9_runtime.py").read_text(encoding="utf-8")
    assert 'bridge_trace["query_window_count"]' in source
    assert 'bridge_trace["query_count"]' not in source
    collector = source[source.index("def collect_forward_predictions"):source.index("def _write_npz_exclusive")]
    assert "R2Score" not in collector
    assert "raw_prediction[:, -1:, :] / 5.0" in collector
