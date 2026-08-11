"""Fit-end provenance receipt for clean RT nested-LOSO selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite RT selection receipt: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RtNestedSelectionReceipt(Callback):
    """Write selected-checkpoint identity before the generic export phase.

    The callback is RT-specific and is instantiated only by the clean nested
    experiment config.  It does not alter optimization or checkpoint policy;
    it records the exact ModelCheckpoint choice, split-manifest bytes, and
    Hydra config bytes so the later one-shot evaluator cannot be pointed at a
    hand-picked compatible checkpoint.
    """

    def __init__(
        self,
        output_path: str,
        split_manifest_path: str,
        config_path: str,
        run_id: str,
        arm: str,
        outer_loso_fold: int,
        seed: int,
        monitor: str = "val_heldin/r2_mean",
    ) -> None:
        super().__init__()
        self.output_path = Path(output_path)
        self.split_manifest_path = Path(split_manifest_path)
        self.config_path = Path(config_path)
        self.run_id = str(run_id)
        self.arm = str(arm)
        self.outer_loso_fold = int(outer_loso_fold)
        self.seed = int(seed)
        self.monitor = str(monitor)

    @staticmethod
    def _find_checkpoint(trainer: Trainer, monitor: str) -> ModelCheckpoint:
        candidates = [
            callback
            for callback in trainer.callbacks
            if isinstance(callback, ModelCheckpoint)
            and str(getattr(callback, "monitor", "")) == monitor
            and bool(getattr(callback, "best_model_path", ""))
        ]
        if not candidates:
            raise RuntimeError(
                f"Clean RT fit did not produce a ModelCheckpoint selected by {monitor!r}"
            )
        # The clean config has one every-epoch best callback and one periodic
        # callback.  Prefer the callback with the smallest every_n_epochs value
        # so the receipt cannot silently record the periodic snapshot.
        return min(candidates, key=lambda cb: int(getattr(cb, "every_n_epochs", 1) or 1))

    def on_fit_end(self, trainer: Trainer, pl_module: Any) -> None:
        if not trainer.is_global_zero:
            return
        callback = self._find_checkpoint(trainer, self.monitor)
        checkpoint_path = Path(callback.best_model_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RT selected checkpoint is missing: {checkpoint_path}")
        score = callback.best_model_score
        if score is None:
            raise RuntimeError("RT selected ModelCheckpoint has no best_model_score")
        selected_metric_value = float(score.item() if hasattr(score, "item") else score)
        if not torch.isfinite(torch.tensor(selected_metric_value)):
            raise RuntimeError("RT selected checkpoint score is non-finite")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("RT selected checkpoint payload is not a mapping")
        epoch = payload.get("epoch")
        global_step = payload.get("global_step")
        if not isinstance(epoch, int) or not isinstance(global_step, int):
            raise RuntimeError("RT selected checkpoint lacks integer epoch/global_step metadata")

        split_manifest = {}
        datamodule = trainer.datamodule
        if datamodule is None or not hasattr(datamodule, "get_split_manifest"):
            raise RuntimeError("RT clean selection callback requires a manifest-producing DataModule")
        split_manifest = dict(datamodule.get_split_manifest())
        if split_manifest.get("validation_protocol") != "nested_loso":
            raise RuntimeError("RT selection receipt refused a non-nested split manifest")
        if split_manifest.get("requested_side_feature_group") != self.arm:
            raise RuntimeError("RT selection arm disagrees with split manifest")
        if int(split_manifest.get("outer_loso_fold", -1)) != self.outer_loso_fold:
            raise RuntimeError("RT selection fold disagrees with split manifest")
        if split_manifest.get("nested_selection", {}).get("checkpoint_metric") != self.monitor:
            raise RuntimeError("RT selection metric disagrees with split manifest")
        split_bytes = json.dumps(split_manifest, indent=2) + "\n"
        self.split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.split_manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite RT fit split manifest: {self.split_manifest_path}")
        self.split_manifest_path.write_text(split_bytes, encoding="utf-8")

        if not self.config_path.is_file():
            raise FileNotFoundError(f"Hydra config for RT selection receipt is missing: {self.config_path}")
        config_sha256 = _sha256(self.config_path)
        split_sha256 = hashlib.sha256(split_bytes.encode("utf-8")).hexdigest()
        run_dir = self.output_path.resolve().parent
        receipt = {
            "schema": "rt_clean_nested_loso_selection_receipt_v1",
            "status": "PASS_FIT_INNER_SELECTION_ONLY",
            "selection_receipt_path": str(self.output_path.resolve()),
            "run_id": self.run_id,
            "run_dir": str(run_dir),
            "arm": self.arm,
            "outer_loso_fold": self.outer_loso_fold,
            "seed": self.seed,
            "selected_by_metric": self.monitor,
            "selected_metric_scope": "inner_validation_session_only",
            "selected_metric_value": selected_metric_value,
            "selected_epoch": epoch,
            "selected_global_step": global_step,
            "best_model_path": str(checkpoint_path),
            "best_model_sha256": _sha256(checkpoint_path),
            "config_path": str(self.config_path.resolve()),
            "config_sha256": config_sha256,
            "split_manifest_path": str(self.split_manifest_path.resolve()),
            "split_manifest_sha256": split_sha256,
            "formal_heldout_opened": False,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
        }
        _write_exclusive(self.output_path, receipt)
