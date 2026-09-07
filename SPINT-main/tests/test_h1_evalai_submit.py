from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import h1_evalai_submit as submit


def _package_receipt(tmp_path: Path, variant: str = "released_code_lr_5e-5") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = submit.CANDIDATES[variant]
    package = tmp_path / candidate["package_basename"]
    package.write_bytes(b"h1 decoder")
    receipt = {
        "schema": submit.PACKAGE_SCHEMA,
        "status": submit.PACKAGE_STATUS,
        "protocol": submit.PACKAGE_PROTOCOL,
        "protocol_variant": variant,
        "comparison_role": candidate["role"],
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": 50,
        "checkpoint_sha256": "a" * 64,
        "package_path": str(package),
        "package_sha256": submit.sha256_file(package),
        "selection_guards": {
            "validation_epoch_selection": False,
            "minival_used_for_epoch_selection": False,
            "private_test_opened": False,
            "evalai_submission_performed": False,
        },
        "input_manifest": {
            "heldin_calibration_recordings": 13,
            "public_heldout_calibration_recordings": 14,
            "minival_included": False,
            "private_test_opened": False,
        },
    }
    path = tmp_path / "terminal_package.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_package_receipt_binds_exact_variant_filename_and_bytes(tmp_path: Path) -> None:
    receipt = _package_receipt(tmp_path)
    value = submit.validate_package_receipt(receipt, "released_code_lr_5e-5")
    assert value["package_sha256"] == submit.sha256_file(value["package_path"])
    assert value["role"] == "implementation_primary"

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["package_path"] = str(tmp_path / "wrong.pkl")
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="path/basename"):
        submit.validate_package_receipt(receipt, "released_code_lr_5e-5")


def test_package_receipt_rejects_role_and_selection_drift(tmp_path: Path) -> None:
    receipt = _package_receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["comparison_role"] = "appendix_paper_reference"
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison_role"):
        submit.validate_package_receipt(receipt, "released_code_lr_5e-5")

    receipt = _package_receipt(tmp_path / "second")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["selection_guards"]["private_test_opened"] = True
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="guards"):
        submit.validate_package_receipt(receipt, "released_code_lr_5e-5")


def test_required_h1_submission_metadata_is_explicit() -> None:
    assert {row["name"] for row in submit.SUBMISSION_ATTRIBUTES} == {
        "IsHeldOutZeroShot",
        "IsTestTimeAdaptive",
        "IsPretrained",
    }
    assert all(row["required"] is True and row["value"] is False for row in submit.SUBMISSION_ATTRIBUTES)


def test_quota_counts_current_utc_day_and_rejects_full_limit() -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    phase = {
        "max_submissions_per_day": 6,
        "max_submissions_per_month": 50,
        "max_submissions": 100,
        "max_concurrent_submissions_allowed": 3,
    }
    listing = {"count": 1, "results": [{"submitted_at": now, "status": "finished"}]}
    assert submit._quota(phase, listing)["today"] == 1
    listing = {
        "count": 6,
        "results": [{"submitted_at": now, "status": "finished"} for _ in range(6)],
    }
    with pytest.raises(RuntimeError, match="daily"):
        submit._quota(phase, listing)


def test_registry_binding_accepts_manifest_or_config_identity_and_rejects_neither() -> None:
    local = "sha256:" + "1" * 64
    manifest = "sha256:" + "2" * 64
    config = "sha256:" + "3" * 64
    assert submit._registry_local_binding(local, local, config) == (
        "registry_manifest_digest_equals_local_image_id"
    )
    assert submit._registry_local_binding(local, manifest, local) == (
        "registry_config_digest_equals_local_image_id"
    )
    with pytest.raises(RuntimeError, match="do not bind"):
        submit._registry_local_binding(local, manifest, config)
