"""Focused contracts for the nonce-gap recovery package finalizer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

# The recovery module imports torch transitively.  Tests explicitly model the
# required process-start boundary before importing it or torch itself.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from falcon_challenge.config import FalconConfig, FalconTask
from scripts import h1_carrierid_all_source_official_recovery_package as recovery
from src.h1_m4_cce_contract import sha256_file, write_immutable_json
from src.data.h1_m4_eb_pilot import (
    FrozenEBPlan,
    H1PilotRecord,
    PilotDataError,
    TrialBlocks,
    fit_deployment_carrier,
    fit_frozen_carrier,
    interpolate_identity,
    interpolate_trial_identity,
)
from src.data.h1_carrierid_all_source_deployment import load_deployment_calibration_record_v5


def test_recovery_v1_is_immutable_hash_bound_and_explicitly_unbound() -> None:
    evidence = recovery.validate_recovery_audit()
    path = Path(evidence["path"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert evidence["sha256"] == "50c26bc58be52fd663dbeba6239c7415c1093927c82ccebb4f354409adee5fd3"
    assert evidence["audit_sha256"] == "c13aa11625ab13c3ad8a0fb07e8d06d569c37f04501639470c466b3c0b1962ea"
    gap = evidence["body"]["provenance_gap"]
    assert gap["execution_receipt_present"] is False
    assert gap["nonce_bound_launch_verified"] is False
    assert gap["nonce_reconstructed"] is False
    terminal = evidence["body"]["terminal_checkpoint"]
    assert terminal["epoch"] == 49 and terminal["epochs_completed"] == 50
    assert terminal["global_step"] == 206650
    assert terminal["optimizer_states"] == 1 and terminal["lr_schedulers"] == 0
    assert terminal["state_finite"] is True


def test_exact_allowlist_is_13_plus_14_and_does_not_load_nwb(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_loader(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("allowlist validation must not load calibration NWB bytes")

    monkeypatch.setattr(recovery.official_package, "load_deployment_calibration_record", fail_loader)
    heldin, heldout = recovery._expected_calibration_paths()
    result = recovery.validate_exact_calibration_allowlist(heldin + heldout)
    assert result["counts"] == {"heldin": 13, "heldout": 14, "total": 27}
    assert len(result["dataset_tags"]) == 27
    with pytest.raises(recovery.RecoveryPackageError, match="exactly 27"):
        recovery.validate_exact_calibration_allowlist(heldin)
    with pytest.raises(recovery.RecoveryPackageError, match="order/allowlist"):
        recovery.validate_exact_calibration_allowlist(heldout + heldin)


def _fake_preflight(*, checkpoint: Path, asset_manifest: Path,
                    calibration_files: tuple[Path, ...], output: Path) -> dict[str, Any]:
    body = {
        "schema": recovery.official_package.PACKAGE_PREFLIGHT_SCHEMA,
        "status": recovery.official_package.PACKAGE_PREFLIGHT_STATUS,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "asset_manifest": {"path": str(asset_manifest.resolve()), "sha256": sha256_file(asset_manifest)},
        "calibration_files_indexed_only": [
            {"path": str(path), "dataset_tag": FalconConfig(task=FalconTask.h1).hash_dataset(path.stem)}
            for path in calibration_files
        ],
        "scope": {
            "calibration_recording_bytes_opened": 0,
            "query_recordings_opened": 0,
            "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "payload_exported": False,
            "docker_built_or_pushed": False,
            "evalai_accessed_or_submitted": False,
        },
        "runtime_class": "third_party.falcon_challenge.h1_carrierid_all_source_decoder.H1CarrierIdAllSourceDecoder",
        "payload_schema": recovery.H1_ALL_SOURCE_PAYLOAD_SCHEMA,
        "code_sha256": {
            "package": sha256_file(Path(recovery.official_package.__file__).resolve()),
            "runtime": sha256_file(recovery.ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py"),
            "carrier_model": sha256_file(recovery.ROOT / "src/models/components/h1_carrierid_spint.py"),
            "carrier_data": sha256_file(recovery.ROOT / "src/data/h1_carrierid_all_source_official.py"),
            "carrier_estimator": sha256_file(recovery.ROOT / "src/data/h1_m4_eb_pilot.py"),
            "deployment_data": sha256_file(recovery.ROOT / "src/data/h1_carrierid_all_source_deployment.py"),
        },
    }
    write_immutable_json(output, body)
    return body


def test_preflight_uses_staging_and_new_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recovery, "RECOVERY_ROOT", tmp_path)
    monkeypatch.setattr(recovery.official_package, "prepare_package_preflight", _fake_preflight)
    output = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PREFLIGHT_TEST.json"
    result = recovery.prepare_recovery_package_preflight(output=output)
    assert Path(result["output"]) == output.resolve()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert result["output_sha256"] == sha256_file(output)
    assert not list(tmp_path.glob(".recovery_preflight_*"))
    assert len(result["calibration_files_indexed_only"]) == 27
    assert result["recovery_mode"]["provenance_gap"]["nonce_reconstructed"] is False


def _synthetic_preflight_body() -> dict[str, Any]:
    recovery_body = recovery.validate_recovery_audit()["body"]
    terminal = recovery_body["terminal_checkpoint"]
    assets = recovery_body["asset_evidence"]
    allowlist = recovery.validate_exact_calibration_allowlist(recovery._default_calibration_files())
    return {
        "schema": recovery.official_package.PACKAGE_PREFLIGHT_SCHEMA,
        "status": recovery.official_package.PACKAGE_PREFLIGHT_STATUS,
        "checkpoint": {"path": str(Path(terminal["path"]).resolve()), "sha256": terminal["sha256"]},
        "asset_manifest": {"path": str(Path(assets["path"]).resolve()), "sha256": assets["sha256"]},
        "calibration_files_indexed_only": [
            {"path": path, "dataset_tag": tag}
            for path, tag in zip(allowlist["files"], allowlist["dataset_tags"])
        ],
        "runtime_class": "third_party.falcon_challenge.h1_carrierid_all_source_decoder.H1CarrierIdAllSourceDecoder",
        "payload_schema": recovery.H1_ALL_SOURCE_PAYLOAD_SCHEMA,
        "code_sha256": {
            "package": sha256_file(Path(recovery.official_package.__file__).resolve()),
            "runtime": sha256_file(recovery.ROOT / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py"),
            "carrier_model": sha256_file(recovery.ROOT / "src/models/components/h1_carrierid_spint.py"),
            "carrier_data": sha256_file(recovery.ROOT / "src/data/h1_carrierid_all_source_official.py"),
            "carrier_estimator": sha256_file(recovery.ROOT / "src/data/h1_m4_eb_pilot.py"),
            "deployment_data": sha256_file(recovery.ROOT / "src/data/h1_carrierid_all_source_deployment.py"),
        },
        "scope": dict(recovery._PACKAGE_PREFLIGHT_SCOPE_ZERO),
    }


def test_final_audit_code_validator_binds_deployment_data() -> None:
    body = {"code_sha256": recovery._current_recovery_package_code_sha256()}
    observed = recovery.validate_recovery_package_audit_code(body)
    assert observed["deployment_data"] == sha256_file(
        recovery.ROOT / "src/data/h1_carrierid_all_source_deployment.py"
    )
    body["code_sha256"]["deployment_data"] = "0" * 64
    with pytest.raises(recovery.RecoveryPackageError, match="deployment_data"):
        recovery.validate_recovery_package_audit_code(body)


def test_final_audit_writer_emits_deployment_data_code_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The append-only v5 final audit must bind the deployment loader code."""

    monkeypatch.setattr(recovery, "RECOVERY_ROOT", tmp_path)
    preflight = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PREFLIGHT_TEST.json"
    write_immutable_json(preflight, {"schema": recovery.official_package.PACKAGE_PREFLIGHT_SCHEMA})
    monkeypatch.setattr(recovery, "_validate_package_preflight_body", lambda *_args, **_kwargs: {})
    output = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_AUDIT_TEST.json"
    result = recovery.write_recovery_package_audit(preflight=preflight, output=output)
    assert result["output"] == str(output.resolve())
    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["code_sha256"]["deployment_data"] == sha256_file(
        recovery.ROOT / "src/data/h1_carrierid_all_source_deployment.py"
    )
    assert body["scope"] == {
        "source_only": True,
        "calibration_files_indexed": 27,
        "query_recordings_opened": 0,
        "formal_test_labels_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "cuda_used": False,
        "evalai_accessed_or_submitted": False,
    }
    recovery.validate_recovery_package_audit_code(body)


@pytest.mark.parametrize("mutation", ("status", "checkpoint_sha", "asset_sha", "index", "scope", "code_sha"))
def test_export_and_final_audit_reject_mutated_preflight_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str,
) -> None:
    """Every downstream boundary must revalidate the immutable preflight body."""

    monkeypatch.setattr(recovery, "RECOVERY_ROOT", tmp_path)
    body = _synthetic_preflight_body()
    if mutation == "status":
        body["status"] = "PASS_TAMPERED"
    elif mutation == "checkpoint_sha":
        body["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "asset_sha":
        body["asset_manifest"]["sha256"] = "0" * 64
    elif mutation == "index":
        body["calibration_files_indexed_only"][0], body["calibration_files_indexed_only"][1] = (
            body["calibration_files_indexed_only"][1], body["calibration_files_indexed_only"][0]
        )
    elif mutation == "scope":
        body["scope"]["docker_built_or_pushed"] = True
    else:
        body["code_sha256"]["package"] = "0" * 64
    preflight = tmp_path / f"H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PREFLIGHT_MUTATED_{mutation}.json"
    write_immutable_json(preflight, body)

    def exporter_must_not_run(**_kwargs: Any) -> None:
        raise AssertionError("mutated preflight reached payload exporter")

    monkeypatch.setattr(recovery.official_package, "export_payload", exporter_must_not_run)
    with pytest.raises(recovery.RecoveryPackageError):
        recovery.export_recovery_payload(
            preflight=preflight,
            output=tmp_path / f"H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_{mutation}.pkl",
            allow_export=True,
        )
    with pytest.raises(recovery.RecoveryPackageError):
        recovery.write_recovery_package_audit(
            preflight=preflight,
            output=tmp_path / f"H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_AUDIT_{mutation}.json",
        )


def test_payload_validator_recomputes_actual_calibration_input_sha(tmp_path: Path) -> None:
    payload_path, payload = _synthetic_payload(
        tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_INPUT_SHA.pkl"
    )
    payload["calibration_receipts"][0]["input_sha256"] = "0" * 64
    payload_path.chmod(0o644)
    with payload_path.open("wb") as handle:
        import pickle
        pickle.dump(payload, handle)
    payload_path.chmod(0o444)
    with pytest.raises(recovery.RecoveryPackageError, match="calibration input SHA drift"):
        recovery.validate_recovery_payload(payload_path)


def _synthetic_deployment_record_and_plan() -> tuple[H1PilotRecord, FrozenEBPlan]:
    rng = np.random.default_rng(20260810)
    trials = []
    for trial_number in (1.0, 2.0, 3.0, 4.0, 5.0):
        rates = rng.normal(size=(8, 176)).astype(np.float64)
        velocity = rng.normal(size=(8, 7)).astype(np.float64)
        trials.append(TrialBlocks(trial_number, rates, velocity, np.zeros((8, 5), dtype=np.int64)))
    record = H1PilotRecord(
        "ses-19250101T000000", "19250101", Path("/tmp/synthetic_calib.nwb"), "a" * 64,
        np.zeros((8, 176), dtype=np.float32), np.zeros((8, 7), dtype=np.float32),
        np.zeros(8, dtype=bool), np.ones(8, dtype=bool), np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64),
        (1.0, 2.0, 3.0, 4.0, 5.0), tuple(trials),
    )
    pcs = np.zeros((16, 176), dtype=np.float64)
    pcs[:, :16] = np.eye(16, dtype=np.float64)
    plan = FrozenEBPlan(
        outer_date="SYNTHETIC", source_sessions=(record.session_name,), source_input_sha256=(record.input_sha256,),
        mean=np.zeros(176, dtype=np.float64), scale=np.ones(176, dtype=np.float64), pcs=pcs,
        q=16, ridge_lambda=100.0,
        U=np.vstack((np.eye(4, dtype=np.float64), np.zeros((3, 4), dtype=np.float64))),
        mu=np.zeros(4, dtype=np.float64),
        tau2=1.0, raw_plan_sha256="", raw_receipt_sha256="", eb_receipt_sha256="", transform_sha256="",
    )
    return record, plan


def test_deployment_m4_reproduces_historical_fit_and_rejects_padding() -> None:
    record, plan = _synthetic_deployment_record_and_plan()
    historical = fit_frozen_carrier(record, plan, (1.0, 2.0, 3.0, 4.0))
    deployment = fit_deployment_carrier(record, plan, (1.0, 2.0, 3.0, 4.0))
    for key in ("carrier", "raw_carrier", "raw_rows", "beta", "G", "sigma2", "projected_variance", "weight"):
        np.testing.assert_array_equal(deployment[key], historical[key])
    reduced = fit_deployment_carrier(record, plan, (1.0, 2.0, 3.0))
    assert reduced["support_m"] == 3 and reduced["support_trial_numbers"] == [1.0, 2.0, 3.0]
    with pytest.raises(PilotDataError, match="padding/duplication"):
        fit_deployment_carrier(record, plan, (1.0, 1.0, 2.0))
    with pytest.raises(PilotDataError, match="three or four"):
        fit_deployment_carrier(record, plan, (1.0, 2.0, 3.0, 4.0, 5.0))


def test_deployment_m4_identity_reproduces_historical_cubic_construction() -> None:
    path = recovery._default_calibration_files()[0]
    record = load_deployment_calibration_record_v5(path)
    values = tuple(record.trial_values[:4])
    historical = interpolate_identity(record, values)
    deployment = np.stack([interpolate_trial_identity(record, value) for value in values], axis=0)
    np.testing.assert_array_equal(deployment, historical)


def _synthetic_payload(path: Path) -> tuple[Path, dict[str, Any]]:
    evidence = recovery.validate_recovery_audit()
    body = evidence["body"]
    allowlist = recovery.validate_exact_calibration_allowlist(recovery._default_calibration_files())
    tags = allowlist["dataset_tags"]
    features = {
        tag: np.zeros((4 if source_path in allowlist["heldin"] else 3, 1024, 176), dtype=np.float32)
        for source_path, tag in zip(allowlist["files"], tags)
    }
    carriers = {tag: np.zeros((176, 4), dtype=np.float32) for tag in tags}
    receipts = []
    for source_path, tag in zip(allowlist["files"], tags):
        identity_sha = __import__("hashlib").sha256(np.ascontiguousarray(features[tag]).tobytes()).hexdigest()
        carrier_sha = __import__("hashlib").sha256(np.ascontiguousarray(carriers[tag]).tobytes()).hexdigest()
        support_m = int(features[tag].shape[0])
        support_values = tuple(
            recovery.load_deployment_calibration_record(Path(source_path)).trial_values[:support_m]
        )
        receipts.append({
            "path": source_path, "dataset_tag": tag, "input_sha256": sha256_file(Path(source_path)),
            "support_m": support_m,
            "support_trial_numbers": list(support_values),
            "identity_sha256": identity_sha, "carrier_sha256": carrier_sha,
        })

    decoder = torch.nn.Linear(1, 1, bias=False).eval()
    with torch.no_grad():
        decoder.weight.fill_(1.0)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    metadata = body["terminal_checkpoint"]["metadata"]
    candidate = body["prepared_launch"]["candidate"]
    payload = {
        "decoder": decoder, "task": FalconTask.h1, "window_size": 700,
        "behavior_scaling_factor": 20.0, "interpolate_trials": True,
        "interpolate_trials_kind": "cubic", "calib_trial_features": features,
        "calib_carriers": carriers, "smooth_calibration": False,
        "carrier_payload_schema": recovery.H1_ALL_SOURCE_PAYLOAD_SCHEMA,
        "carrier_asset_manifest_sha256": body["asset_evidence"]["sha256"],
        "carrier_transform_sha256": candidate["transform_sha256"],
        "carrier_normalizer_sha256": candidate["normalizer_sha256"],
        "source_checkpoint_metadata": dict(metadata), "calibration_receipts": receipts,
        "deployment_contract": {
            "carrier_fit": "forward_closed_form_from_each_explicit_calibration_recording",
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "formal_test_labels_packaged": 0, "query_labels_read": 0,
            "model_weights_frozen_at_runtime": True,
        },
    }
    with path.open("wb") as handle:
        import pickle
        pickle.dump(payload, handle)
    path.chmod(0o444)
    return path, payload


def test_payload_validator_binds_27_shapes_hashes_and_finite_state(tmp_path: Path) -> None:
    payload_path, _payload = _synthetic_payload(tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_TEST.pkl")
    result = recovery.validate_recovery_payload(payload_path)
    assert result["dataset_count"] == 27
    assert len(result["calibration_receipts"]) == 27
    assert result["shape"] == {"carrier": [176, 4], "identity": "per_record_m_3_or_4"}
    assert result["identity_shapes"][result["dataset_tags"][0]] == [4, 1024, 176]
    assert result["identity_shapes"][result["dataset_tags"][-1]] == [3, 1024, 176]
    assert result["finite"] is True


def test_payload_validator_rejects_padded_or_duplicate_support_values(tmp_path: Path) -> None:
    payload_path, payload = _synthetic_payload(
        tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_DUPLICATE_SUPPORT.pkl"
    )
    payload["calibration_receipts"][-1]["support_trial_numbers"][-1] = payload["calibration_receipts"][-1]["support_trial_numbers"][0]
    payload_path.chmod(0o644)
    with payload_path.open("wb") as handle:
        import pickle
        pickle.dump(payload, handle)
    payload_path.chmod(0o444)
    with pytest.raises(recovery.RecoveryPackageError, match="padding/duplicates"):
        recovery.validate_recovery_payload(payload_path)


def test_runtime_payload_validator_accepts_mixed_m_and_rejects_duplicate_support() -> None:
    runtime = recovery.validate_carrier_payload
    payload = {
        "carrier_payload_schema": recovery.H1_ALL_SOURCE_PAYLOAD_SCHEMA,
        "task": FalconTask.h1,
        "calib_trial_features": {"tag3": np.zeros((3, 1024, 176), dtype=np.float32)},
        "calib_carriers": {"tag3": np.zeros((176, 4), dtype=np.float32)},
        "calibration_receipts": [{"dataset_tag": "tag3", "support_m": 3,
                                  "support_trial_numbers": [1.0, 2.0, 3.0]}],
        "deployment_contract": {"target_optimizer_steps": 0, "target_backward_steps": 0,
                                 "formal_test_labels_packaged": 0, "query_labels_read": 0},
    }
    assert runtime(payload, expected_task=FalconTask.h1)["tag3"].shape == (176, 4)
    payload["calibration_receipts"][0]["support_trial_numbers"][-1] = 1.0
    with pytest.raises(recovery.H1CarrierIdPayloadError, match="duplicates"):
        runtime(payload, expected_task=FalconTask.h1)


def test_smoke_validator_requires_27_rows_and_explicit_no_query_scope(tmp_path: Path) -> None:
    payload_path, _ = _synthetic_payload(tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_TEST.pkl")
    payload_result = recovery.validate_recovery_payload(payload_path)
    allowlist = recovery.validate_exact_calibration_allowlist(recovery._default_calibration_files())
    rows = [
        {"split": split, "path": str(path), "output_shape": [1, 7], "finite": True}
        for split, paths in (("heldin", allowlist["heldin"]), ("heldout", allowlist["heldout"]))
        for path in paths
    ]
    body = {
        "schema": recovery.RECOVERY_SMOKE_SCHEMA,
        "status": "PASS_RECOVERY_CPU_SMOKE_NO_QUERY_OPENED",
        "payload": payload_result, "rows": rows,
        "counts": {"heldin": 13, "heldout": 14, "total": 27},
        "scope": {"query_recordings_opened": 0, "formal_test_labels_opened": 0,
                  "target_optimizer_steps": 0, "target_backward_steps": 0,
                  "cuda_used": False, "evalai_accessed_or_submitted": False},
        "submission_authorized": False,
    }
    smoke_path = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_SMOKE_TEST.json"
    write_immutable_json(smoke_path, body)
    result = recovery.validate_recovery_smoke(smoke_path, payload_sha256=payload_result["sha256"])
    assert result["counts"] == {"heldin": 13, "heldout": 14, "total": 27}
    assert len(result["rows"]) == 27
    bad = dict(body)
    bad["rows"] = rows[:-1]
    bad_path = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_SMOKE_BAD.json"
    write_immutable_json(bad_path, bad)
    with pytest.raises(recovery.RecoveryPackageError, match="27 rows"):
        recovery.validate_recovery_smoke(bad_path)


def test_export_is_disabled_without_explicit_allow_flag(tmp_path: Path) -> None:
    output = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD_TEST.pkl"
    with pytest.raises(recovery.RecoveryPackageError, match="disabled"):
        recovery.export_recovery_payload(preflight=recovery.DEFAULT_PREFLIGHT, output=output)
    assert not output.exists()


def test_official_checkpoint_loader_does_not_forward_weights_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    def signature_guard(cls: Any, checkpoint_path: Path, *, map_location: str) -> object:
        calls["checkpoint"] = checkpoint_path
        calls["map_location"] = map_location
        return object()

    monkeypatch.setattr(
        recovery.official_package.H1CarrierIdAllSourceLitModule,
        "load_from_checkpoint",
        classmethod(signature_guard),
    )
    checkpoint = tmp_path / "epoch_049.ckpt"
    model = recovery.official_package._load_checkpoint_model(checkpoint)
    assert model is not None
    assert calls == {"checkpoint": checkpoint, "map_location": "cpu"}


def test_cli_refuses_missing_cuda_visibility_before_runtime_import() -> None:
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    command = [sys.executable, str(Path(recovery.__file__).resolve()), "preflight", "--help"]
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "CUDA_VISIBLE_DEVICES=''" in (completed.stdout + completed.stderr)


def test_smoke_cli_routes_append_only_v4_defaults_without_running_runtime(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Path] = {}

    def fake_smoke(*, payload: Path, output: Path) -> dict[str, Any]:
        calls["payload"] = payload
        calls["output"] = output
        return {"status": "TEST_ONLY"}

    monkeypatch.setattr(recovery, "run_cpu_smoke", fake_smoke)
    monkeypatch.setattr(sys, "argv", [str(recovery.__file__), "smoke"])
    recovery.main()
    assert calls == {"payload": recovery.DEFAULT_PAYLOAD, "output": recovery.DEFAULT_SMOKE}
    assert json.loads(capsys.readouterr().out)["status"] == "TEST_ONLY"


def test_preflight_cli_reserves_final_audit_name_for_full_audit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Path] = {}

    def fake_preflight(*, output: Path) -> dict[str, Any]:
        calls["preflight"] = output
        return {"status": "PREFLIGHT_ONLY_TEST"}

    def fake_audit(*, preflight: Path, output: Path) -> dict[str, Any]:
        calls["audit_preflight"] = preflight
        calls["audit_output"] = output
        return {"status": "AUDIT_ONLY_TEST"}

    monkeypatch.setattr(recovery, "prepare_recovery_package_preflight", fake_preflight)
    monkeypatch.setattr(recovery, "write_recovery_package_audit", fake_audit)
    monkeypatch.setattr(sys, "argv", [str(recovery.__file__), "preflight"])
    recovery.main()
    assert calls == {
        "preflight": recovery.DEFAULT_PREFLIGHT,
        "audit_preflight": recovery.DEFAULT_PREFLIGHT,
        "audit_output": recovery.DEFAULT_PREFLIGHT_AUDIT,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "AUDIT_ONLY_TEST"


def test_smoke_cli_help_is_available_with_cuda_hidden() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, str(Path(recovery.__file__).resolve()), "smoke", "--help"]
    completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    assert completed.returncode == 0
    assert "--payload" in completed.stdout and "--output" in completed.stdout
