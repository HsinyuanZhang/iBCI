"""CPU/synthetic contracts for the isolated H1 all-source H-C candidate."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.h1_carrierid_all_source_official_launcher import (
    compose_candidate,
    prepare_launch,
    validate_candidate_config,
)
from scripts.h1_carrierid_all_source_official_package import validate_checkpoint_metadata
from scripts.h1_carrierid_all_source_official_package import H1CarrierIdPackageError
from src.data.h1_carrierid_all_source_official import (
    ALL_SOURCE_ASSET_FILE,
    ALL_SOURCE_ASSET_SCHEMA,
    ALL_SOURCE_EPOCHS,
    ALL_SOURCE_PROTOCOL,
    AllSourceNormalizer,
    H1AllSourceSchedule,
    H1CarrierIdAllSourceDataModule,
    H1CarrierIdAllSourceError,
    H1_HELDIN_SESSIONS,
    assert_deployment_calibration_path,
    fit_all_source_plan,
    load_all_source_assets,
)
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, NORMALIZER_FORMULA, canonical_sha256, sha256_file
from src.models.h1_carrierid_all_source_official_module import ALL_SOURCE_CHECKPOINT_SCHEMA
from third_party.falcon_challenge.h1_carrierid_all_source_decoder import (
    H1_ALL_SOURCE_PAYLOAD_SCHEMA,
    H1CarrierIdPayloadError,
    validate_carrier_payload,
)


class _SyntheticRecord:
    def __init__(self, name: str, index: int) -> None:
        self.session_name = name
        self.input_sha256 = f"{index + 1:064x}"
        self.num_neurons = 176
        self.trial_values = tuple(float(value) for value in range(5))
        rng = np.random.default_rng(1000 + index)
        self._trials = {}
        for trial in self.trial_values:
            rates = rng.normal(loc=4.0 + 0.1 * index + trial, scale=1.0, size=(4, 176))
            velocity = rates[:, :7] * (0.02 + 0.001 * index) + rng.normal(scale=0.01, size=(4, 7))
            self._trials[trial] = SimpleNamespace(rates=rates, velocity=velocity)

    def blocks_for(self, value: float):
        return self._trials[float(value)]

    def eval_trial_neural(self, value: float) -> np.ndarray:
        del value
        return np.zeros((4, 176), dtype=np.float32)


def _records():
    return {name: _SyntheticRecord(name, index) for index, name in enumerate(H1_HELDIN_SESSIONS)}


def _asset_fixture(tmp_path: Path):
    plan = fit_all_source_plan(_records())
    cache_sha = "c" * 64
    normalizer_core = {
        "formula": NORMALIZER_FORMULA,
        "floor": NORMALIZER_FLOOR,
        "s_src": 0.5,
        "source_cache_sha256": cache_sha,
        "entries": 26,
        "rows": 176,
        "dims": 4,
    }
    normalizer = AllSourceNormalizer(0.5, cache_sha, 26, 176, 4, canonical_sha256(normalizer_core))
    arrays = tmp_path / ALL_SOURCE_ASSET_FILE
    np.savez(
        arrays, mean=plan.mean, scale=plan.scale, pcs=plan.pcs, U=plan.U, mu=plan.mu,
        tau2=np.asarray(plan.tau2), q=np.asarray(plan.q), ridge_lambda=np.asarray(plan.ridge_lambda),
        s_src=np.asarray(normalizer.s_src),
    )
    arrays.chmod(0o444)
    body = {
        "schema": ALL_SOURCE_ASSET_SCHEMA,
        "status": "PASS_ALL_PUBLIC_HELDIN_ASSETS_FROZEN_NO_GPU_NO_FORMAL",
        "protocol": ALL_SOURCE_PROTOCOL,
        "source_sessions": list(H1_HELDIN_SESSIONS),
        "source_files": [
            {"session": name, "path": f"/public/{name}.nwb", "sha256": plan.source_input_sha256[index]}
            for index, name in enumerate(H1_HELDIN_SESSIONS)
        ],
        "plan": plan.manifest(),
        "normalizer": normalizer.manifest(),
        "asset_file": arrays.name,
        "asset_file_sha256": sha256_file(arrays),
        "carrier_cache_entries": 26,
        "scope": {
            "held_in_calibration_recordings_opened": 13,
            "minival_recordings_opened": 0,
            "held_out_query_recordings_opened": 0,
            "formal_test_labels_opened": 0,
            "evalai_accessed": False,
            "trainer_constructed": False,
            "cuda_used": False,
        },
    }
    manifest = tmp_path / "H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"
    manifest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest.chmod(0o444)
    return manifest, plan, normalizer


def test_all_source_plan_is_deterministic_q16_ridge100_and_uses_all_13_sessions() -> None:
    first = fit_all_source_plan(_records())
    second = fit_all_source_plan(_records())
    assert first.source_sessions == H1_HELDIN_SESSIONS
    assert first.q == 16 and first.ridge_lambda == 100.0
    assert first.mean.shape == first.scale.shape == (176,)
    assert first.pcs.shape == (16, 176)
    assert first.U.shape == (7, 4)
    assert first.mu.shape == (4,)
    assert first.tau2 > 0.0
    assert first.transform_sha256 == second.transform_sha256
    assert np.array_equal(first.pcs, second.pcs)
    assert np.array_equal(first.U, second.U)


def test_all_source_asset_loader_binds_arrays_normalizer_and_formal_boundary(tmp_path: Path) -> None:
    manifest, plan, normalizer = _asset_fixture(tmp_path)
    loaded = load_all_source_assets(manifest)
    assert loaded.plan.transform_sha256 == plan.transform_sha256
    assert loaded.normalizer.normalizer_sha256 == normalizer.normalizer_sha256
    assert loaded.manifest_sha256 == sha256_file(manifest)
    (tmp_path / ALL_SOURCE_ASSET_FILE).chmod(0o644)
    with pytest.raises(H1CarrierIdAllSourceError, match="mutable"):
        load_all_source_assets(manifest)


def test_all_source_config_and_prepare_only_launcher_are_fixed_and_no_evalai(tmp_path: Path) -> None:
    manifest, _plan, _normalizer = _asset_fixture(tmp_path)
    row = validate_candidate_config(compose_candidate(manifest), asset_manifest=manifest)
    assert row["source_sessions"] == list(H1_HELDIN_SESSIONS)
    assert row["epochs"] == 50 and row["checkpoint_epoch_zero_based"] == 49
    receipt = tmp_path / "launch.json"
    result = prepare_launch(asset_manifest=manifest, output=receipt)
    assert result["status"] == "PASS_ALL_SOURCE_HC_PREPARED_NOT_LAUNCHED"
    assert result["scope"]["trainer_constructed_or_launched"] is False
    assert result["scope"]["formal_test_labels_opened"] == 0
    assert result["scope"]["evalai_submission_authorized"] is False
    assert receipt.stat().st_mode & 0o777 == 0o444


def test_all_source_datamodule_is_fit_only_and_fixed() -> None:
    common = dict(task="h1", data_dir="/tmp/000954", asset_manifest_path="/tmp/assets.json")
    with pytest.raises(H1CarrierIdAllSourceError, match="fixed data"):
        H1CarrierIdAllSourceDataModule(**common, fixed_epochs=49)
    module = H1CarrierIdAllSourceDataModule(**common)
    with pytest.raises(RuntimeError, match="local test"):
        module.test_dataloader()
    with pytest.raises(RuntimeError, match="formal/query"):
        module.predict_dataloader()


def test_schedule_covers_exactly_50_epochs_and_keeps_session_batches() -> None:
    windows = []
    starts = {}
    for name in H1_HELDIN_SESSIONS:
        windows.extend((name, index) for index in range(32))
        starts[name] = (0, 1)
    dataset = SimpleNamespace(
        window_indices=windows,
        cache=SimpleNamespace(starts_by_session=starts),
    )
    schedule = H1AllSourceSchedule(dataset)
    assert len(schedule) == 13
    for _ in range(ALL_SOURCE_EPOCHS):
        batches = list(iter(schedule))
        assert len(batches) == 13
        assert all(len({dataset.window_indices[index][0] for index, _start in batch}) == 1 for batch in batches)
    with pytest.raises(RuntimeError, match="epoch 49"):
        list(iter(schedule))


def test_payload_requires_matching_per_dataset_carriers_and_zero_target_updates() -> None:
    payload = {
        "carrier_payload_schema": H1_ALL_SOURCE_PAYLOAD_SCHEMA,
        "task": "h1",
        "calib_trial_features": {"dataset": np.zeros((4, 1024, 176), np.float32)},
        "calib_carriers": {"dataset": np.zeros((176, 4), np.float32)},
        "deployment_contract": {
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "formal_test_labels_packaged": 0,
            "query_labels_read": 0,
        },
    }
    result = validate_carrier_payload(payload, expected_task="h1")
    assert result["dataset"].shape == (176, 4)
    payload["deployment_contract"]["target_backward_steps"] = 1
    with pytest.raises(H1CarrierIdPayloadError, match="backpropagation"):
        validate_carrier_payload(payload, expected_task="h1")


def test_package_checkpoint_contract_binds_assets_and_rejects_target_training() -> None:
    row = {
        "schema": ALL_SOURCE_CHECKPOINT_SCHEMA,
        "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_no_formal_selection",
        "checkpoint_warm_start": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "formal_test_labels_opened": 0,
        "asset_manifest_sha256": "a" * 64,
    }
    assert validate_checkpoint_metadata(
        {"h1_carrierid_all_source_official": row}, asset_manifest_sha256="a" * 64
    ) == row
    row["target_optimizer_steps"] = 1
    with pytest.raises(H1CarrierIdPackageError, match="target-session"):
        validate_checkpoint_metadata(
            {"h1_carrierid_all_source_official": row}, asset_manifest_sha256="a" * 64
        )


def test_deployment_path_guard_accepts_only_explicit_calibration_nwb(tmp_path: Path) -> None:
    calibration = tmp_path / "sub-HumanPitt-held-out-calib_ses-19250126T113454.nwb"
    calibration.touch()
    assert assert_deployment_calibration_path(calibration) == calibration.resolve()
    query = tmp_path / "sub-HumanPitt-held-out-query_ses-19250126T113454.nwb"
    query.touch()
    with pytest.raises(H1CarrierIdAllSourceError, match="calibration NWBs only"):
        assert_deployment_calibration_path(query)


def test_new_path_does_not_modify_or_invoke_original_exporter_and_has_no_gpu_executor() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/h1_carrierid_all_source_official_launcher.py").read_text(encoding="utf-8")
    package = (root / "scripts/h1_carrierid_all_source_official_package.py").read_text(encoding="utf-8")
    runtime = (root / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py").read_text(encoding="utf-8")
    assert "import subprocess" not in launcher
    assert "--execute" not in launcher
    assert "evalai_api" not in launcher.lower()
    assert "requests." not in launcher and "docker " not in launcher
    assert "spint_decoder.main" not in package
    assert "class H1CarrierIdAllSourceDecoder(SpintDecoder)" in runtime
    assert "carrier=carrier.unsqueeze(0)" in runtime
