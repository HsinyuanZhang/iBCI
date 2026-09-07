"""Embed fail-closed clean-teacher provenance in every selected checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lightning.pytorch import Callback, Trainer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CleanTeacherProvenanceCallback(Callback):
    def __init__(self, input_manifest_path: str) -> None:
        super().__init__()
        self.input_manifest_path = str(Path(input_manifest_path).resolve())

    def on_save_checkpoint(self, trainer: Trainer, pl_module, checkpoint: dict) -> None:
        manifest = Path(self.input_manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"missing clean teacher input manifest {manifest}")
        data = trainer.datamodule
        if data is None or not getattr(data.hparams, "clean_teacher", False):
            raise ValueError("clean teacher provenance callback requires clean datamodule")
        if getattr(data, "val_heldout_dataset", None) is not None or getattr(data, "val_heldout_batch_sampler", None) is not None:
            raise ValueError("clean teacher checkpoint attempted with held-out loader")
        runtime_manifest = data.verified_clean_teacher_input_manifest()
        runtime_manifest_sha256 = getattr(data, "_clean_teacher_runtime_manifest_sha256", None)
        external_manifest_sha256 = getattr(data, "_clean_teacher_external_manifest_sha256", None)
        external_manifest_path = getattr(data, "_clean_teacher_external_manifest_path", None)
        if not isinstance(runtime_manifest, dict) or not runtime_manifest_sha256:
            raise ValueError("clean teacher checkpoint missing cached runtime input manifest")
        try:
            external_manifest = json.loads(manifest.read_text())
        except json.JSONDecodeError as error:
            raise ValueError("clean teacher external manifest is invalid JSON") from error
        # This is deliberately full-object equality, not a comparison of a
        # pre-fit SHA.  It catches changed data roots, roles, paths, sizes and
        # content hashes before a checkpoint can be emitted.
        if external_manifest != runtime_manifest:
            raise ValueError("clean teacher external manifest drifted after runtime validation")
        if external_manifest_path != str(manifest.resolve()):
            raise ValueError("clean teacher callback manifest path differs from datamodule contract")
        if external_manifest_sha256 != _sha256(manifest):
            raise ValueError("clean teacher external manifest bytes drifted after runtime validation")
        if runtime_manifest_sha256 != hashlib.sha256(
            (json.dumps(runtime_manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest():
            raise ValueError("clean teacher cached runtime manifest hash is inconsistent")

        accessed = sorted(set(getattr(data, "_clean_accessed_paths", [])))
        expected = sorted(
            str(path) for path in (list(getattr(data, "_clean_calib_files", [])) + list(getattr(data, "_clean_minival_files", [])))
        )
        if accessed != expected:
            raise ValueError("clean teacher checkpoint has incomplete or unexpected input access")
        checkpoint["clean_teacher_provenance_v1"] = {
            "protocol": "m2_clean_teacher_v1",
            "input_manifest_path": str(manifest),
            "input_manifest_sha256": _sha256(manifest),
            "runtime_manifest": runtime_manifest,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "runtime_manifest_equals_external": True,
            "task": "m2",
            "calibration_n_trials": int(data.hparams.calibration_n_trials),
            "expected_heldin_sessions": list(data.hparams.expected_heldin_sessions),
            "include_heldout_in_fit": False,
            "include_heldout_in_test": False,
            "heldout_dataset_created": False,
            "accessed_input_paths": accessed,
            "selection_metric": "val_heldin/r2_mean",
            "selection_mode": "max",
            "epoch": int(checkpoint.get("epoch", -1)),
            "selection_score": float(trainer.callback_metrics.get("val_heldin/r2_mean", float("nan"))),
        }
