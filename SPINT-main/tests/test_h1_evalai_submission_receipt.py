from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import scripts.h1_evalai_submission_receipt as receipt


VARIANT = "released_code_lr_5e-5"
ROLE = "implementation_primary"


def _source_map() -> dict[str, str]:
    root = Path(receipt.__file__).resolve().parents[1]
    names = (
        "scripts/h1_terminal_package.py",
        "scripts/h1_baseline_eval.py",
        "src/data/h1_baseline_datamodule.py",
        "src/models/falcon_module.py",
        "src/models/components/spint.py",
        "third_party/falcon_challenge/spint_decoder.py",
        "third_party/falcon_challenge/spint_sample.py",
        "third_party/falcon_challenge/spint_sample.Dockerfile",
    )
    return {name: receipt.sha256_file(root / name) for name in names}


def _manifest() -> dict[str, object]:
    rows = [
        {
            "role": "heldin_calib" if i < 13 else "public_heldout_calib",
            "session": f"session-{i:02d}",
            "path": f"/public/h1/calibration/session-{i:02d}.nwb",
            "sha256": "a" * 64,
            "size_bytes": 17,
        }
        for i in range(27)
    ]
    return {
        "schema": "spint_h1_terminal_public_calibration_allowlist_v1",
        "heldin_calibration_recordings": 13,
        "public_heldout_calibration_recordings": 14,
        "minival_included": False,
        "private_test_opened": False,
        "files": rows,
    }


def _package_receipt(tmp_path: Path, *, role: str = ROLE, package_bytes: bytes = b"decoder") -> Path:
    package_path = tmp_path / "spint_h1_released_code_lr_5e-5.pkl"
    package_path.write_bytes(package_bytes)
    checkpoint_path = tmp_path / "run" / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt"
    config_path = tmp_path / "run" / ".hydra" / "config.yaml"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"fixed checkpoint")
    config_path.write_bytes(b"resolved config")
    manifest = _manifest()
    package = {
        "schema": receipt.PACKAGE_SCHEMA,
        "status": "PASS_H1_TERMINAL_PACKAGE_ALLOWLISTED",
        "protocol": receipt.EXPECTED_PROTOCOL,
        "protocol_variant": VARIANT,
        "comparison_role": role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": receipt.sha256_file(checkpoint_path),
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "global_step": 123,
        "checkpoint_optimizer_lr": 5.0e-5,
        "config_path": str(config_path),
        "config_sha256": receipt.sha256_file(config_path),
        "package_path": str(package_path),
        "package_sha256": receipt.sha256_file(package_path),
        "input_manifest": manifest,
        "input_manifest_sha256": receipt.canonical_sha256(manifest),
        "payload_audit": {"neural_calibration_features_only": True, "feature_count": 27},
        "source_sha256": _source_map(),
        "selection_guards": {
            "validation_epoch_selection": False,
            "minival_used_for_epoch_selection": False,
            "minival_used_for_variant_selection": False,
            "minival_used_for_package_selection": False,
            "private_test_used_for_epoch_selection": False,
            "private_test_used_for_variant_selection": False,
            "private_test_used_for_package_selection": False,
            "private_test_opened": False,
            "evalai_submission_performed": False,
        },
        "calibration_boundary": {
            "public_heldout_calibration_is_allowed_for_neural_only_identity_features": True,
            "dense_behavior_or_covariate_targets_serialized": False,
            "formal_private_test_labels_opened": False,
        },
    }
    path = tmp_path / "terminal_package_receipt.json"
    path.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _prepare(tmp_path: Path, **overrides: object) -> tuple[Path, dict[str, object]]:
    package_receipt = _package_receipt(tmp_path)
    output = tmp_path / "prepare.json"
    kwargs: dict[str, object] = {
        "package_receipt": package_receipt,
        "output": output,
        "image_id": "1" * 64,
        "image_digest": "2" * 64,
        "image_tag": "spint_h1:epoch050-primary",
        "variant": VARIANT,
        "role": ROLE,
    }
    kwargs.update(overrides)
    value = receipt.prepare_receipt(**kwargs)
    return output, value


def test_prepare_binds_package_source_runtime_and_challenge_and_is_immutable(tmp_path: Path):
    output, value = _prepare(tmp_path)
    assert value["schema"] == receipt.PREPARE_SCHEMA
    assert value["protocol_variant"] == VARIANT
    assert value["comparison_role"] == ROLE
    assert value["package"]["sha256"] == receipt.sha256_file(value["package"]["path"])
    assert value["runtime"] == {
        "TASK": "h1",
        "BATCH_SIZE": 8,
        "MODEL_FILE": "spint_h1_released_code_lr_5e-5.pkl",
    }
    assert value["challenge"]["challenge_id"] == 2319
    assert value["challenge"]["phase_id"] == 4599
    assert value["challenge"]["phase_slug"] == "few-shot-test-2319"
    assert value["image"]["id"].startswith("sha256:")
    assert value["image"]["digest"].startswith("sha256:")
    assert value["external_action"]["evalai_push_performed_by_this_tool"] is False
    assert value["selection_guards"]["minival_used_for_package_selection"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="overwrite"):
        _prepare(tmp_path)


def test_variant_and_role_cannot_be_swapped(tmp_path: Path):
    package_receipt = _package_receipt(tmp_path, role="appendix_paper_reference")
    with pytest.raises(ValueError, match="variant/role mismatch"):
        receipt.prepare_receipt(
            package_receipt=package_receipt,
            output=tmp_path / "bad.json",
            image_id="1" * 64,
            image_digest="2" * 64,
            image_tag="spint_h1:bad",
            variant=VARIANT,
        )


def test_package_sha_must_match_package_receipt(tmp_path: Path):
    package_receipt = _package_receipt(tmp_path)
    payload = json.loads(package_receipt.read_text(encoding="utf-8"))
    payload["package_sha256"] = "d" * 64
    package_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="package SHA mismatch"):
        receipt.prepare_receipt(
            package_receipt=package_receipt,
            output=tmp_path / "bad.json",
            image_id="1" * 64,
            image_digest="2" * 64,
            image_tag="spint_h1:bad",
            variant=VARIANT,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("checkpoint_sha256", "d" * 64, "checkpoint SHA"),
        ("config_sha256", "d" * 64, "config SHA"),
        ("checkpoint_epoch_zero_based", 48, "epoch 49"),
        ("epochs_completed", 49, "50 completed"),
        ("global_step", 0, "global_step"),
        ("checkpoint_optimizer_lr", 1.0e-5, "optimizer LR"),
    ],
)
def test_checkpoint_config_and_optimizer_bindings_are_verified(
    tmp_path: Path, field: str, value: object, message: str
):
    package_receipt = _package_receipt(tmp_path)
    payload = json.loads(package_receipt.read_text(encoding="utf-8"))
    payload[field] = value
    package_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        receipt.prepare_receipt(
            package_receipt=package_receipt,
            output=tmp_path / "bad.json",
            image_id="1" * 64,
            image_digest="2" * 64,
            image_tag="spint_h1:bad",
            variant=VARIANT,
        )


def test_payload_and_calibration_audits_are_hard_gates(tmp_path: Path):
    package_receipt = _package_receipt(tmp_path)
    payload = json.loads(package_receipt.read_text(encoding="utf-8"))
    payload["payload_audit"]["feature_count"] = 26
    package_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature_count"):
        receipt.prepare_receipt(
            package_receipt=package_receipt,
            output=tmp_path / "bad_payload.json",
            image_id="1" * 64,
            image_digest="2" * 64,
            image_tag="spint_h1:bad",
            variant=VARIANT,
        )

    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    package_receipt = _package_receipt(calibration_dir)
    payload = json.loads(package_receipt.read_text(encoding="utf-8"))
    payload["calibration_boundary"]["formal_private_test_labels_opened"] = True
    package_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration boundary guard"):
        receipt.prepare_receipt(
            package_receipt=package_receipt,
            output=tmp_path / "bad_calibration.json",
            image_id="1" * 64,
            image_digest="2" * 64,
            image_tag="spint_h1:bad",
            variant=VARIANT,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task", "m2", "TASK"),
        ("batch_size", 4, "BATCH_SIZE"),
        ("model_file", "spint_m2.pkl", "MODEL_FILE"),
        ("challenge_id", 999, "challenge_id"),
        ("phase_id", 4598, "phase_id"),
        ("phase_slug", "few-shot-minival-2319", "phase_slug"),
    ],
)
def test_runtime_and_phase_bindings_are_fail_closed(tmp_path: Path, field: str, value: object, message: str):
    with pytest.raises(ValueError, match=message):
        _prepare(tmp_path, **{field: value})


def test_true_selection_guard_rejected(tmp_path: Path):
    package_receipt = _package_receipt(tmp_path)
    payload = json.loads(package_receipt.read_text(encoding="utf-8"))
    payload["selection_guards"]["minival_used_for_variant_selection"] = True
    package_receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection/private guard is true"):
        receipt.prepare_receipt(
            package_receipt=package_receipt,
            output=tmp_path / "bad.json",
            image_id="1" * 64,
            image_digest="2" * 64,
            image_tag="spint_h1:bad",
            variant=VARIANT,
        )


def test_finalize_records_external_submission_and_metric_and_refuses_overwrite(tmp_path: Path):
    preparation_path, preparation = _prepare(tmp_path)
    output = tmp_path / "final.json"
    final = receipt.finalize_receipt(
        preparation_receipt=preparation_path,
        output=output,
        submission_id="123456",
        timestamp="2026-08-07T04:00:00Z",
        status="finished",
        metrics={"Held Out R2 Mean": 0.29, "Normalized Latency": 0.11},
    )
    assert final["schema"] == receipt.FINAL_SCHEMA
    assert final["submission"]["id"] == 123456
    assert final["submission"]["status"] == "finished"
    assert final["submission"]["metrics"]["Held Out R2 Mean"] == 0.29
    assert final["external_action"]["evalai_push_performed_by_this_tool"] is False
    assert final["external_action"]["submission_recorded_from_external_api_response"] is True
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="overwrite"):
        receipt.finalize_receipt(
            preparation_receipt=preparation_path,
            output=output,
            submission_id=123457,
            timestamp="2026-08-07T04:01:00Z",
            status="finished",
            metrics={"r2": 0.1},
        )


def test_finalize_rechecks_package_and_requires_metric_for_success(tmp_path: Path):
    preparation_path, _ = _prepare(tmp_path)
    with pytest.raises(ValueError, match="at least one metric"):
        receipt.finalize_receipt(
            preparation_receipt=preparation_path,
            output=tmp_path / "missing_metric.json",
            submission_id=123456,
            timestamp="2026-08-07T04:00:00Z",
            status="finished",
            metrics={},
        )
    package_receipt_path = tmp_path / "terminal_package_receipt.json"
    package_payload = json.loads(package_receipt_path.read_text(encoding="utf-8"))
    package_payload["package_path"] = str(tmp_path / "changed.pkl")
    package_receipt_path.write_text(json.dumps(package_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="package receipt changed"):
        receipt.finalize_receipt(
            preparation_receipt=preparation_path,
            output=tmp_path / "changed.json",
            submission_id=123456,
            timestamp="2026-08-07T04:00:00Z",
            status="submitted",
            metrics={},
        )


def test_cli_prepare_and_finalize_do_not_need_evalai_or_private_data(tmp_path: Path):
    package_receipt = _package_receipt(tmp_path)
    prepared = tmp_path / "cli_prepare.json"
    assert (
        receipt.main(
            [
                "prepare",
                "--package-receipt",
                str(package_receipt),
                "--output",
                str(prepared),
                "--image-id",
                "1" * 64,
                "--image-digest",
                "2" * 64,
                "--image-tag",
                "spint_h1:cli",
                "--variant",
                VARIANT,
            ]
        )
        == 0
    )
    final = tmp_path / "cli_final.json"
    assert (
        receipt.main(
            [
                "finalize",
                "--preparation-receipt",
                str(prepared),
                "--output",
                str(final),
                "--submission-id",
                "123456",
                "--timestamp",
                "2026-08-07T04:00:00Z",
                "--status",
                "submitted",
                "--metric",
                "placeholder=0.0",
            ]
        )
        == 0
    )
    assert json.loads(final.read_text(encoding="utf-8"))["submission"]["id"] == 123456
