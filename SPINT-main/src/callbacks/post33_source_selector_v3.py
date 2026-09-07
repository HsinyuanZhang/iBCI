"""Explicit all-epoch source-only selector for M2 post-33 Phase-B v3."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule


EXPECTED_EPOCHS = tuple(range(35))


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


def explicit_max_then_earlier(records: list[dict[str, Any]]) -> dict[str, Any]:
    epochs = [record.get("epoch") for record in records]
    if tuple(sorted(epochs)) != EXPECTED_EPOCHS or len(set(epochs)) != 35:
        raise ValueError("selector requires exactly one record for every epoch 0..34")
    return max(records, key=lambda row: (float(row["metric_value"]), -int(row["epoch"])))


class Post33SourceSelectorV3(Callback):
    """Save epochs 0..34 and select with an explicit deterministic policy.

    A runner must first create ``ownership.json`` via O_EXCL.  The callback
    refuses any checkpoint directory outside that already-owned cell.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        selector_records_path: str,
        cell_owner_path: str,
        owner_token: str,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.selector_records_path = Path(selector_records_path).resolve()
        self.cell_owner_path = Path(cell_owner_path).resolve()
        self.owner_token = owner_token
        self.records: list[dict[str, Any]] = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        if not self.cell_owner_path.is_file():
            raise FileNotFoundError("cell ownership must exist before callback setup")
        owner = json.loads(self.cell_owner_path.read_text(encoding="utf-8"))
        if owner.get("owner_token") != self.owner_token:
            raise PermissionError("cell owner token mismatch")
        cell_dir = self.cell_owner_path.parent
        if self.checkpoint_dir != (cell_dir / "checkpoints").resolve():
            raise ValueError("checkpoint_dir must be the deterministic owned-cell checkpoint path")
        if self.selector_records_path != (cell_dir / "selector_records.json").resolve():
            raise ValueError("selector_records_path must be deterministic within the owned cell")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        source_records = getattr(pl_module, "source_selector_records", None)
        if not isinstance(source_records, list) or not source_records:
            raise RuntimeError("v3 source module did not expose a selector record")
        epoch = int(trainer.current_epoch)
        record = dict(source_records[-1])
        if record.get("epoch") != epoch:
            raise RuntimeError("module/callback selector epoch mismatch")
        if any(existing["epoch"] == epoch for existing in self.records):
            raise RuntimeError(f"duplicate validation record for epoch {epoch}")
        checkpoint = (self.checkpoint_dir / f"epoch_{epoch:03d}.ckpt").resolve()
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        trainer.save_checkpoint(str(checkpoint), weights_only=False)
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise RuntimeError("checkpoint write did not produce a canonical regular file")
        record["checkpoint_path"] = str(checkpoint)
        self.records.append(record)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        selected = explicit_max_then_earlier(self.records)
        payload = {
            "schema": "m2_post33_source_selector_records_v3",
            "policy": "max_finite_equal_session_mean_then_earlier_epoch",
            "selected_epoch": selected["epoch"],
            "selected_checkpoint_path": selected["checkpoint_path"],
            "records": self.records,
        }
        _write_json_exclusive(self.selector_records_path, payload)

