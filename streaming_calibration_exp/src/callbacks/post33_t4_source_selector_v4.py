"""Owned, explicit 12-epoch source-only selector for Phase-C T4."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule


EXPECTED_EPOCHS = tuple(range(12))


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        cursor = 0
        while cursor < len(data):
            cursor += os.write(fd, data[cursor:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_metadata(path: Path) -> dict[str, Any]:
    """Bind one T4 fit-end artifact without permitting a symlink alias."""
    if path.is_symlink():
        raise RuntimeError(f"T4 selector artifact symlink is forbidden: {path}")
    canonical = path.resolve(strict=True)
    if not canonical.is_file() or str(path) != str(canonical):
        raise RuntimeError(f"T4 selector artifact is not canonical regular file: {path}")
    digest = hashlib.sha256()
    with canonical.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "canonical_path": str(canonical),
        "size_bytes": canonical.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def explicit_t4_max_then_earlier(records: list[dict[str, Any]]) -> dict[str, Any]:
    epochs = [record.get("epoch") for record in records]
    if tuple(sorted(epochs)) != EXPECTED_EPOCHS or len(set(epochs)) != 12:
        raise ValueError("T4 selector requires exactly one record for epochs 0..11")
    return max(records, key=lambda row: (float(row["metric_value"]), -int(row["epoch"])))


class Post33T4SourceSelectorV4(Callback):
    def __init__(
        self,
        *,
        checkpoint_dir: str,
        selector_records_path: str,
        deployment_constants_path: str,
        cell_owner_path: str,
        owner_token: str,
        fold: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.selector_records_path = Path(selector_records_path).resolve()
        self.deployment_constants_path = Path(deployment_constants_path).resolve()
        self.cell_owner_path = Path(cell_owner_path).resolve()
        self.owner_token = owner_token
        self.fold = fold
        self.seed = seed
        self.records: list[dict[str, Any]] = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        owner = json.loads(self.cell_owner_path.read_text(encoding="utf-8"))
        identity = (
            owner.get("schema"),
            owner.get("phase_id"),
            owner.get("arm"),
            owner.get("fold"),
            owner.get("seed"),
            owner.get("owner_token"),
        )
        expected = (
            "m2_post33_phase_c_cell_ownership_v4",
            "PHASE_C_V4",
            "t4",
            self.fold,
            self.seed,
            self.owner_token,
        )
        if identity != expected:
            raise PermissionError("T4 selector ownership/cell mismatch")
        cell = self.cell_owner_path.parent.parent.resolve()
        if self.checkpoint_dir != (cell / "run/checkpoints").resolve():
            raise ValueError("T4 checkpoint_dir is outside the owned canonical cell")
        if self.selector_records_path != (cell / "run/selector_records.json").resolve():
            raise ValueError("T4 selector record path is outside the owned canonical cell")
        if self.deployment_constants_path != (cell / "run/deployment_constants.json").resolve():
            raise ValueError("T4 deployment constants path is outside the owned canonical cell")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        source_records = getattr(pl_module, "source_selector_records", None)
        if not isinstance(source_records, list) or not source_records:
            raise RuntimeError("T4 v4 module did not expose a source selector record")
        epoch = int(trainer.current_epoch)
        record = dict(source_records[-1])
        if record.get("epoch") != epoch:
            raise RuntimeError("T4 module/callback epoch mismatch")
        if any(existing["epoch"] == epoch for existing in self.records):
            raise RuntimeError(f"duplicate T4 selector epoch {epoch}")
        checkpoint = (self.checkpoint_dir / f"epoch_{epoch:03d}.ckpt").resolve()
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        trainer.save_checkpoint(str(checkpoint), weights_only=False)
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise RuntimeError("T4 checkpoint is not a canonical regular file")
        record["checkpoint_path"] = str(checkpoint)
        self.records.append(record)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        selected = explicit_t4_max_then_earlier(self.records)
        datamodule = trainer.datamodule
        access = getattr(datamodule, "phase_c_stage_access_evidence", None)
        expected_access = {
            "stage": "fit", "source_files_opened": 12,
            "outer_calibration_files_opened": 0, "formal_files_opened": 0,
            "outer_directional_label_accesses": 0,
            "outer_descriptor_fit_invocations": 0, "outer_calibration_claims": 0,
            "outer_query_batch_calls": 0, "scorer_calls": 0,
        }
        if access != expected_access:
            raise RuntimeError("T4 source fit touched outer/evaluator state")
        normalization = getattr(datamodule, "native_t4_normalization", None)
        if not isinstance(normalization, dict):
            raise RuntimeError("T4 training-derived normalizer is missing")
        mean = [float(value) for value in normalization["mean"]]
        std = [float(value) for value in normalization["std"]]
        if len(mean) != 4 or len(std) != 4 or not all(map(math.isfinite, mean + std)) or min(std) <= 0:
            raise RuntimeError("T4 training-derived normalizer is invalid")
        _write_json_exclusive(
            self.deployment_constants_path,
            {
                "schema": "m2_post33_phase_c_deployment_constants_v4",
                "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
                "phase_id": "PHASE_C_V4", "arm": "t4",
                "fold": self.fold, "seed": self.seed,
                "t4_normalizer": {
                    "mean": mean, "std": std,
                    "source_sessions": list(datamodule.source_session_names),
                    "support_trials": 33,
                },
                "source_stage_access_evidence": access,
            },
        )
        _write_json_exclusive(
            self.selector_records_path,
            {
                "schema": "m2_post33_t4_source_selector_records_v4",
                "policy": "max_finite_equal_session_mean_then_earlier_epoch",
                "selected_epoch": selected["epoch"],
                "selected_checkpoint_path": selected["checkpoint_path"],
                # Hash only the selected checkpoint and the fit-derived
                # normalizer/constants.  These are the two training outputs
                # the outer evaluator subsequently consumes.
                "selected_checkpoint": _file_metadata(Path(selected["checkpoint_path"])),
                "deployment_constants": _file_metadata(self.deployment_constants_path),
                "records": self.records,
            },
        )
