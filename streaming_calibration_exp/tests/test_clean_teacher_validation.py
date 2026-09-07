from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from src.utils import clean_teacher_validation as validation


SESSIONS = [
    "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2", "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
]


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    rows = []
    for role in ("heldin_calib", "heldin_minival"):
        for session in SESSIONS:
            path = (tmp_path / f"{role}_{session}.nwb").resolve()
            rows.append({"role": role, "session": session, "path": str(path), "size_bytes": 1, "sha256": "a" * 64})
    payload = {
        "schema_version": 1, "protocol": "m2_clean_teacher_v1", "task": "m2",
        "calibration_n_trials": 24, "expected_heldin_sessions": SESSIONS,
        "forbidden_heldout_sessions": ["ses-2020-10-30-Run1"],
        "include_heldout_in_fit": False, "include_heldout_in_test": False, "files": rows,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path, payload


def _checkpoint(tmp_path: Path, *, bad_monitor: bool = False, provenance: bool = True) -> tuple[Path, dict]:
    manifest_path, manifest = _manifest(tmp_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    accesses = sorted(row["path"] for row in manifest["files"])
    prov = {
        "protocol": "m2_clean_teacher_v1", "input_manifest_path": str(manifest_path.resolve()),
        "input_manifest_sha256": manifest_sha, "runtime_manifest": manifest,
        "runtime_manifest_sha256": validation.canonical_json_sha256(manifest),
        "runtime_manifest_equals_external": True, "task": "m2", "calibration_n_trials": 24,
        "expected_heldin_sessions": SESSIONS, "include_heldout_in_fit": False,
        "include_heldout_in_test": False, "heldout_dataset_created": False,
        "accessed_input_paths": accesses, "selection_metric": "val_heldin/r2_mean",
        "selection_mode": "max", "epoch": 7, "selection_score": 0.42,
    }
    monitor = "val_heldout/r2_mean" if bad_monitor else "val_heldin/r2_mean"
    payload = {
        "epoch": 7,
        "state_dict": {},
        "callbacks": {
            "ModelCheckpoint{'monitor': 'x'}": {"monitor": monitor, "mode": "max"},
            "EarlyStopping{'monitor': 'x'}": {"monitor": monitor, "mode": "max"},
        },
    }
    if provenance:
        payload["clean_teacher_provenance_v1"] = prov
    path = tmp_path / ("teacher_bad.ckpt" if bad_monitor else "teacher.ckpt")
    torch.save(payload, path)
    return path, prov


def _receipt(checkpoint: Path, facts: dict, tmp_path: Path) -> Path:
    provenance = facts["provenance"]
    payload = {
        "schema_version": 1, "protocol": "m2_clean_teacher_v1",
        "selected_checkpoint": {
            "path": facts["checkpoint_path"], "sha256": facts["checkpoint_sha256"],
            "epoch": facts["checkpoint_epoch"], "selection_score": facts["selection_score"],
        },
        "checkpoint_provenance_sha256": facts["provenance_sha256"],
        "callback_contract": facts["callback_contract"],
        "input_manifest": {
            "path": provenance["input_manifest_path"], "sha256": provenance["input_manifest_sha256"],
            "runtime_manifest_sha256": provenance["runtime_manifest_sha256"],
        },
        "resolved_config": {"path": str((tmp_path / "config.yaml").resolve()), "sha256": "b" * 64},
        "source_manifest": {"git_revision": "synthetic", "files": []},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_validator_accepts_only_a_receipted_clean_teacher(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path)
    facts = validation.validate_checkpoint_provenance(checkpoint)
    receipt = _receipt(checkpoint, facts, tmp_path)
    verified = validation.validate_clean_teacher_receipt(checkpoint, receipt)
    assert verified["checkpoint_epoch"] == 7
    assert verified["callback_contract"]["model_checkpoint_count"] == 1


def test_validator_rejects_checkpoint_without_embedded_provenance(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path, provenance=False)
    with pytest.raises(ValueError, match="lacks clean_teacher_provenance"):
        validation.validate_checkpoint_provenance(checkpoint)


def test_validator_rejects_heldout_monitored_checkpoint(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path, bad_monitor=True)
    with pytest.raises(ValueError, match="not held-in/max"):
        validation.validate_checkpoint_provenance(checkpoint)


def test_validator_rejects_missing_or_mismatched_receipt(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path)
    facts = validation.validate_checkpoint_provenance(checkpoint)
    with pytest.raises(FileNotFoundError, match="receipt"):
        validation.validate_clean_teacher_receipt(checkpoint, tmp_path / "missing.json")
    receipt = _receipt(checkpoint, facts, tmp_path)
    data = json.loads(receipt.read_text())
    data["selected_checkpoint"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(data) + "\n")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        validation.validate_clean_teacher_receipt(checkpoint, receipt)


def test_validator_rejects_legacy_epoch34_hash(monkeypatch, tmp_path):
    checkpoint, _ = _checkpoint(tmp_path)
    real_sha = validation.sha256_file

    def old_for_checkpoint(path: Path) -> str:
        if Path(path).resolve() == checkpoint.resolve():
            return validation.LEGACY_HELDOUT_SELECTED_TEACHER_SHA256
        return real_sha(path)

    monkeypatch.setattr(validation, "sha256_file", old_for_checkpoint)
    with pytest.raises(ValueError, match="epoch_034"):
        validation.validate_checkpoint_provenance(checkpoint)


@pytest.mark.parametrize("kind", ["missing_receipt", "mismatched_receipt"])
def test_streaming_setup_rejects_bad_teacher_before_deserialization(monkeypatch, tmp_path, kind):
    from src.models.falcon_module import FalconLitModule
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    checkpoint, _ = _checkpoint(tmp_path)
    if kind == "missing_receipt":
        receipt = tmp_path / "absent.json"
    else:
        facts = validation.validate_checkpoint_provenance(checkpoint)
        receipt = _receipt(checkpoint, facts, tmp_path)
        payload = json.loads(receipt.read_text())
        payload["selected_checkpoint"]["sha256"] = "0" * 64
        receipt.write_text(json.dumps(payload) + "\n")
    called = False

    def must_not_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("teacher deserialization must not happen after validator failure")

    monkeypatch.setattr(FalconLitModule, "load_from_checkpoint", must_not_load)
    module = StreamingCalibrationLitModule(
        task="m2", variant="B3S", teacher_ckpt_path=str(checkpoint), window_size=50,
        teacher_receipt_path=str(receipt), require_clean_teacher_receipt=True,
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        module.setup("fit")
    assert called is False


def test_streaming_setup_rejects_legacy_teacher_before_deserialization(monkeypatch, tmp_path):
    from src.models.falcon_module import FalconLitModule
    from src.models.streaming_calibration_module import StreamingCalibrationLitModule

    old = Path(__file__).resolve().parents[2] / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
    assert old.is_file()
    called = False

    def must_not_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("legacy teacher must be rejected before deserialization")

    monkeypatch.setattr(FalconLitModule, "load_from_checkpoint", must_not_load)
    module = StreamingCalibrationLitModule(
        task="m2", variant="B3S", teacher_ckpt_path=str(old), window_size=50,
        teacher_receipt_path=str(tmp_path / "irrelevant.json"), require_clean_teacher_receipt=True,
    )
    with pytest.raises(ValueError, match="epoch_034"):
        module.setup("fit")
    assert called is False


class _SelectionToy(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def training_step(self, batch, batch_idx):
        return self.weight * 0

    def validation_step(self, batch, batch_idx):
        heldin = torch.tensor(0.2 + 0.2 * self.current_epoch)
        heldout = torch.tensor(0.9 - 0.8 * self.current_epoch)
        self.log("val_heldin/r2_mean", heldin)
        self.log("val_heldout/r2_mean", heldout)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_model_checkpoint_selects_heldin_even_when_heldout_prefers_another_epoch(tmp_path):
    callback = ModelCheckpoint(
        dirpath=tmp_path / "clean_teacher_best", filename="epoch_{epoch:03d}",
        monitor="val_heldin/r2_mean", mode="max", save_top_k=1,
    )
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=2, logger=False,
        enable_model_summary=False, enable_progress_bar=False,
        limit_train_batches=1, limit_val_batches=1, callbacks=[callback],
    )
    trainer.fit(_SelectionToy(), train_dataloaders=loader, val_dataloaders=loader)
    assert callback.best_model_path.endswith("epoch=001.ckpt")
    assert callback.best_model_score.item() == pytest.approx(0.4)


def test_finalizer_roundtrip_and_manifest_drift_rejection(tmp_path):
    checkpoint, provenance = _checkpoint(tmp_path)
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text("clean_teacher_mode: true\n")
    trace = tmp_path / "preflight.strace"
    trace.write_text("\n".join(
        f'openat(AT_FDCWD, "{row["path"]}", O_RDONLY) = 3'
        for row in provenance["runtime_manifest"]["files"]
    ) + "\n")
    receipt = tmp_path / "receipt.json"
    script = Path(__file__).resolve().parents[2] / "sua_exploration/scripts/finalize_m2_clean_teacher_receipt.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run([
        sys.executable, str(script), "--checkpoint", str(checkpoint),
        "--resolved-config", str(resolved), "--output", str(receipt),
        "--strace-path", str(trace), "--strace-sha256", validation.sha256_file(trace),
    ], check=True, env=env, capture_output=True, text=True)
    validation.validate_clean_teacher_receipt(checkpoint, receipt)
    manifest_path = Path(provenance["input_manifest_path"])
    manifest_path.write_text("{}\n")
    with pytest.raises(ValueError, match="manifest"):
        validation.validate_clean_teacher_receipt(checkpoint, receipt)
