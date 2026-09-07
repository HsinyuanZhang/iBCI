"""Offline contracts for the H1 all-source recovery submission path."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import h1_carrierid_all_source_evalai_submit as submit
from scripts import h1_carrierid_all_source_submission_receipt as receipt


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recovery_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_PAYLOAD.pkl"
    payload.write_bytes(b"synthetic all-source decoder bytes")
    root_audit = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_AUDIT.json"
    root_audit.write_text(json.dumps({"schema": "root-recovery-audit-v1"}) + "\n", encoding="utf-8")
    payload_sha = _sha(payload)
    root_sha = _sha(root_audit)
    repo_root = Path(receipt.__file__).resolve().parents[1]
    preflight = tmp_path / "H1_CARRIERID_ALL_SOURCE_PACKAGE_PREFLIGHT.json"
    preflight.write_text(
        json.dumps(
            {
                "schema": receipt.PACKAGE_PREFLIGHT_SCHEMA,
                "status": receipt.PACKAGE_PREFLIGHT_STATUS,
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
                "code_sha256": {
                    "package": receipt.sha256_file(repo_root / "scripts/h1_carrierid_all_source_official_package.py"),
                    "runtime": receipt.sha256_file(repo_root / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py"),
                    "carrier_model": receipt.sha256_file(repo_root / "src/models/components/h1_carrierid_spint.py"),
                    "carrier_data": receipt.sha256_file(repo_root / "src/data/h1_carrierid_all_source_official.py"),
                    "carrier_estimator": receipt.sha256_file(repo_root / "src/data/h1_m4_eb_pilot.py"),
                    "deployment_data": receipt.sha256_file(repo_root / "src/data/h1_carrierid_all_source_deployment.py"),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    final = tmp_path / "H1_CARRIERID_ALL_SOURCE_RECOVERY_PACKAGE_AUDIT.json"
    final.write_text(
        json.dumps(
            {
                "schema": receipt.RECOVERY_PACKAGE_SCHEMA,
                "status": "PASS_RECOVERY_PACKAGE_WITH_PROVENANCE_GAP_NO_SUBMISSION",
                "provenance_gap": {
                    "execution_receipt_present": False,
                    "nonce_bound_launch_verified": False,
                    "nonce_reconstructed": False,
                    "interpretation": "legacy launch lacked a nonce; no nonce was reconstructed",
                },
                "scope": {
                    "source_only": True,
                    "calibration_files_indexed": 27,
                    "query_recordings_opened": 0,
                    "formal_test_labels_opened": 0,
                    "target_optimizer_steps": 0,
                    "target_backward_steps": 0,
                    "cuda_used": False,
                    "evalai_accessed_or_submitted": False,
                },
                "calibration_allowlist": {
                    "counts": {"heldin": 13, "heldout": 14, "total": 27},
                    "heldin": [f"/public/heldin-{index:02d}.nwb" for index in range(13)],
                    "heldout": [f"/public/heldout-{index:02d}.nwb" for index in range(14)],
                    "files": [f"/public/heldin-{index:02d}.nwb" for index in range(13)]
                    + [f"/public/heldout-{index:02d}.nwb" for index in range(14)],
                    "dataset_tags": [f"S{index}" for index in range(27)],
                },
                "terminal_checkpoint": {"epoch": 49, "epochs_completed": 50},
                "recovery_audit": {"path": str(root_audit), "sha256": root_sha},
                "payload": {
                    "path": str(payload), "sha256": payload_sha, "schema": receipt.PAYLOAD_SCHEMA,
                    "dataset_count": 27,
                    "shape": {"carrier": [176, 4], "identity": "per_record_m_3_or_4"},
                    "identity_shapes": {
                        f"S{index}": [4 if index < 13 else 3, 1024, 176] for index in range(27)
                    },
                    "calibration_receipts": [
                        {
                            "path": f"/public/heldin-{index:02d}.nwb" if index < 13 else f"/public/heldout-{index - 13:02d}.nwb",
                            "dataset_tag": f"S{index}",
                            "input_sha256": "a" * 64,
                            "support_m": 4 if index < 13 else 3,
                            "support_trial_numbers": [float(value) for value in range(1, (5 if index < 13 else 4))],
                            "identity_shape": [4 if index < 13 else 3, 1024, 176],
                            "carrier_shape": [176, 4],
                        }
                        for index in range(27)
                    ],
                },
                "smoke": {
                    "schema": receipt.RECOVERY_SMOKE_SCHEMA,
                    "status": "PASS_RECOVERY_CPU_SMOKE_NO_QUERY_OPENED",
                    "counts": {"heldin": 13, "heldout": 14, "total": 27},
                    "rows": [
                        {"split": "heldin", "path": f"/public/heldin-{index:02d}.nwb", "output_shape": [1, 7], "finite": True}
                        for index in range(13)
                    ]
                    + [
                        {"split": "heldout", "path": f"/public/heldout-{index:02d}.nwb", "output_shape": [1, 7], "finite": True}
                        for index in range(14)
                    ],
                    "scope": {
                        "query_recordings_opened": 0,
                        "formal_test_labels_opened": 0,
                        "target_optimizer_steps": 0,
                        "target_backward_steps": 0,
                        "cuda_used": False,
                        "evalai_accessed_or_submitted": False,
                    },
                    "payload": {"path": str(payload), "sha256": payload_sha},
                    "submission_authorized": False,
                },
                "submission": {"authorized": False, "requires_independent_human_review": True},
                "package_preflight": {"path": str(preflight), "sha256": _sha(preflight)},
                "code_sha256": {
                    "recovery_package": receipt.sha256_file(repo_root / "scripts/h1_carrierid_all_source_official_recovery_package.py"),
                    "official_package": receipt.sha256_file(repo_root / "scripts/h1_carrierid_all_source_official_package.py"),
                    "runtime": receipt.sha256_file(repo_root / "third_party/falcon_challenge/h1_carrierid_all_source_decoder.py"),
                    "carrier_model": receipt.sha256_file(repo_root / "src/models/components/h1_carrierid_spint.py"),
                    "carrier_data": receipt.sha256_file(repo_root / "src/data/h1_carrierid_all_source_official.py"),
                    "carrier_estimator": receipt.sha256_file(repo_root / "src/data/h1_m4_eb_pilot.py"),
                    "deployment_data": receipt.sha256_file(repo_root / "src/data/h1_carrierid_all_source_deployment.py"),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return final, payload, payload_sha


def test_recovery_package_requires_unbound_gap_and_matches_payload_bytes(tmp_path: Path) -> None:
    audit, payload, payload_sha = _recovery_fixture(tmp_path)
    result = receipt.validate_recovery_package(audit, payload_path=payload, expected_payload_sha256=payload_sha)
    assert result["payload"]["sha256"] == payload_sha
    body = json.loads(audit.read_text(encoding="utf-8"))
    body["provenance_gap"]["nonce_reconstructed"] = True
    audit.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(receipt.SubmissionReceiptError, match="nonce_reconstructed"):
        receipt.validate_recovery_package(audit)

    # A preflight-only recovery artifact is never submission-ready, even when
    # its payload/smoke bytes happen to look valid.
    body = json.loads(audit.read_text(encoding="utf-8"))
    body["status"] = "PASS_RECOVERY_PACKAGE_PREFLIGHT_ONLY_WITH_PROVENANCE_GAP"
    audit.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(receipt.SubmissionReceiptError, match="passing final audit"):
        receipt.validate_recovery_package(audit)


def test_recovery_package_rehashes_v5_wrapper_and_preflight_code(tmp_path: Path) -> None:
    audit, _payload, _payload_sha = _recovery_fixture(tmp_path)
    body = json.loads(audit.read_text(encoding="utf-8"))
    body["code_sha256"]["recovery_package"] = "0" * 64
    audit.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(receipt.SubmissionReceiptError, match="recovery code SHA mismatch"):
        receipt.validate_recovery_package(audit)

    # Rebuild the final-audit bytes with a fresh declared preflight SHA, then
    # mutate the sealed preflight's carrier-model code binding.
    audit, _payload, _payload_sha = _recovery_fixture(tmp_path / "second")
    preflight = Path(json.loads(audit.read_text(encoding="utf-8"))["package_preflight"]["path"])
    preflight_body = json.loads(preflight.read_text(encoding="utf-8"))
    preflight_body["code_sha256"]["carrier_model"] = "f" * 64
    preflight.write_text(json.dumps(preflight_body), encoding="utf-8")
    body = json.loads(audit.read_text(encoding="utf-8"))
    body["package_preflight"]["sha256"] = _sha(preflight)
    audit.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(receipt.SubmissionReceiptError, match="preflight code SHA mismatch"):
        receipt.validate_recovery_package(audit)


def test_core_writer_shaped_v5_final_audit_binds_scope_trials_and_code_closure(tmp_path: Path) -> None:
    """Keep the submission gate aligned with the core writer's full v5 body."""

    audit, _payload, _payload_sha = _recovery_fixture(tmp_path)
    result = receipt.validate_recovery_package(audit)
    body = result["body"]
    assert body["schema"] == receipt.RECOVERY_PACKAGE_SCHEMA
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
    assert set(body["code_sha256"]) == {
        "recovery_package", "official_package", "runtime", "carrier_model",
        "carrier_data", "carrier_estimator", "deployment_data",
    }
    receipts = body["payload"]["calibration_receipts"]
    assert len(receipts) == 27
    assert [row["support_m"] for row in receipts] == [4] * 13 + [3] * 14
    assert all(len(row["support_trial_numbers"]) == row["support_m"] for row in receipts)
    assert result["stability_audit"]["sha256"] == receipt.STABILITY_AUDIT_SHA256

    body["scope"]["cuda_used"] = True
    audit.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(receipt.SubmissionReceiptError, match="cuda_used"):
        receipt.validate_recovery_package(audit)


def test_recovery_smoke_rows_are_bound_to_allowlist_order(tmp_path: Path) -> None:
    audit, _payload, _payload_sha = _recovery_fixture(tmp_path)
    body = json.loads(audit.read_text(encoding="utf-8"))
    body["smoke"]["rows"][0]["path"] = "/public/wrong-calibration.nwb"
    audit.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(receipt.SubmissionReceiptError, match="smoke row 0 path/split drift"):
        receipt.validate_recovery_package(audit)


def test_offline_prepare_and_finalize_carry_hashes_metadata_and_identity(tmp_path: Path) -> None:
    audit, payload, payload_sha = _recovery_fixture(tmp_path)
    image_id = "1" * 64
    prepare_path = tmp_path / "prepare.json"
    prepared = receipt.prepare_receipt(
        recovery_package_audit=audit,
        payload_path=payload,
        output=prepare_path,
        image_id=image_id,
        embedded_decoder_sha256=payload_sha,
        image_tag="spint_h1_all_source:epoch49",
        quota={"today": 0, "month": 0, "total": 0, "active": 0},
    )
    assert prepared["schema"] == receipt.PREPARE_SCHEMA
    assert prepared["recovery_package_final_audit"]["sha256"] == _sha(audit)
    assert prepared["recovery_payload"]["sha256"] == payload_sha
    assert prepared["challenge"] == {
        "challenge_id": 2319,
        "phase_id": 4599,
        "phase_slug": "few-shot-test-2319",
        "team_id": 41975,
        "visibility": "private",
    }
    assert {row["name"]: row["value"] for row in prepared["submission_attributes"]} == {
        "IsHeldOutZeroShot": False,
        "IsTestTimeAdaptive": True,
        "IsPretrained": False,
    }
    assert prepared["provenance_gap"]["nonce_bound_launch_verified"] is False
    assert prepare_path.stat().st_mode & 0o777 == 0o444

    final_path = tmp_path / "final.json"
    final = receipt.finalize_receipt(
        preparation_receipt=prepare_path,
        output=final_path,
        submission_id="123",
        timestamp="2026-08-10T00:00:00Z",
        status="finished",
        metrics={"Held Out R2 Mean": 0.31},
    )
    assert final["schema"] == receipt.FINAL_SCHEMA
    assert final["submission"]["id"] == 123
    assert final["provenance_gap"]["execution_receipt_present"] is False
    assert final_path.stat().st_mode & 0o777 == 0o444

    # Preparation/finalization outputs are append-only; a second publication
    # cannot replace the first inode.
    with pytest.raises(FileExistsError, match="overwrite"):
        receipt.prepare_receipt(
            recovery_package_audit=audit,
            payload_path=payload,
            output=prepare_path,
            image_id=image_id,
            embedded_decoder_sha256=payload_sha,
            image_tag="spint_h1_all_source:epoch49",
        )


class _FakeImage:
    id = "sha256:" + "2" * 64

    def __init__(self, payload_sha: str, audit_sha: str) -> None:
        self.attrs = {
            "Size": 100,
            "Config": {
                "Env": [
                    "TASK=h1",
                    "BATCH_SIZE=8",
                    "PHASE=test",
                    "MODEL_FILE=h1_carrierid_all_source_payload.pkl",
                    f"RECOVERY_AUDIT_SHA256={audit_sha}",
                    f"RECOVERY_PAYLOAD_SHA256={payload_sha}",
                    f"RECOVERY_STABILITY_SHA256={receipt.STABILITY_AUDIT_SHA256}",
                    f"DECODER_SHA256={payload_sha}",
                ]
            },
        }


class _FakeContainers:
    def __init__(self, payload_sha: str) -> None:
        self.payload_sha = payload_sha
        self.commands: list[tuple[object, dict[str, object]]] = []

    def run(self, *_args: object, **_kwargs: object) -> bytes:
        self.commands.append((_args, dict(_kwargs)))
        command = _kwargs.get("command", ())
        if "-c" in command:
            return b"H1_ALL_SOURCE_RUNTIME_SMOKE_PASS\n"
        return f"{self.payload_sha}  /data/decoder.pkl\n".encode()


class _FakeImages:
    def __init__(self, image: _FakeImage) -> None:
        self.image = image

    def get(self, _tag: str) -> _FakeImage:
        return self.image


class _FakeDocker:
    def __init__(self, image: _FakeImage, payload_sha: str) -> None:
        self.images = _FakeImages(image)
        self.containers = _FakeContainers(payload_sha)

    def ping(self) -> bool:
        return True


def test_read_only_evalai_preflight_binds_recovery_docker_and_quota_without_network(tmp_path: Path) -> None:
    audit, payload, payload_sha = _recovery_fixture(tmp_path)
    audit_sha = _sha(audit)
    image = _FakeImage(payload_sha, audit_sha)
    phase = {
        "id": 4599,
        "challenge": 2319,
        "is_active": True,
        "is_submission_paused": False,
        "max_submissions_per_day": 6,
        "max_submissions_per_month": 50,
        "max_submissions": 100,
        "max_concurrent_submissions_allowed": 3,
    }
    report, _client, _image = submit.preflight(
        recovery_package_audit=audit,
        payload_path=payload,
        image_tag="spint_h1_all_source:epoch49",
        embedded_decoder_sha256=payload_sha,
        client=_FakeDocker(image, payload_sha),
        phase=phase,
        challenge={"id": 2319, "max_docker_image_size": 1000},
        submissions={"count": 0, "results": []},
    )
    assert report["schema"] == submit.PREFLIGHT_SCHEMA
    assert report["image_id"] == "sha256:" + "2" * 64
    assert report["embedded_decoder_sha256"] == payload_sha
    assert report["challenge_id"] == 2319 and report["phase_id"] == 4599 and report["team_id"] == 41975
    assert report["quota"]["max_per_day"] == 6
    assert report["submission_attributes"][1]["value"] is True
    assert report["external_action"]["evalai_push_performed_by_this_tool"] is False
    assert report["external_action"]["evalai_get_performed_by_this_tool"] is False
    assert report["runtime_smoke"]["status"] == "PASS_H1_ALL_SOURCE_RUNTIME_IMPORT_INIT"

    # The state writer uses the same no-replacement publication discipline.
    state_path = tmp_path / "push_state.json"
    submit._write_state(state_path, {"schema": submit.STATE_SCHEMA, "pushed": True})
    assert state_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError, match="overwrite"):
        submit._write_state(state_path, {"schema": submit.STATE_SCHEMA, "pushed": False})


def test_runtime_init_smoke_checks_frozen_clf_and_complete_v5_carrier_roster() -> None:
    payload_sha = "a" * 64
    client = _FakeDocker(_FakeImage(payload_sha, "b" * 64), payload_sha)
    result = submit._runtime_init_smoke(client, "sha256:" + "2" * 64)
    assert result["status"] == "PASS_H1_ALL_SOURCE_RUNTIME_IMPORT_INIT"
    command = str(client.containers.commands[-1][1]["command"])
    assert "d.clf" in command and "d.local_clf" not in command
    assert "len(d.calib_carriers) == 27" in command
    assert "len(d.calib_trial_features) == 27" in command
    assert "set(d.calib_carriers) == set(d.calib_trial_features)" in command


def test_dedicated_dockerfile_isolated_from_baseline_and_declares_recovery_bindings() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = root / "third_party/falcon_challenge/h1_carrierid_all_source_sample.Dockerfile"
    source = dockerfile.read_text(encoding="utf-8")
    assert "h1_carrierid_all_source_sample.py" in source
    assert "ARG BATCH_SIZE=8" in source
    assert "ARG RECOVERY_AUDIT_SHA256" in source
    assert "ARG RECOVERY_PAYLOAD_SHA256" in source
    assert "ARG RECOVERY_STABILITY_SHA256" in source
    assert "ARG DECODER_SHA256" in source
    assert "sha256sum /data/decoder.pkl" in source
    assert "test \"$observed\" = \"$RECOVERY_PAYLOAD_SHA256\"" in source
    assert "test \"$observed\" = \"$DECODER_SHA256\"" in source
    assert "RECOVERY_STABILITY_SHA256=${RECOVERY_STABILITY_SHA256}" in source
    assert receipt.STABILITY_AUDIT_SHA256 in source
    assert "--phase $PHASE" in source and "--batch-size $BATCH_SIZE" in source
    assert (root / "third_party/falcon_challenge/spint_sample.Dockerfile").is_file()
