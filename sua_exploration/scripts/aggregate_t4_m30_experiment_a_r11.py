#!/usr/bin/env python3
"""Post-run compatibility repair for the r10 Experiment A aggregate.

The frozen r10 aggregator correctly requires new component-arm metadata to
explicitly record that the logit-residual path is disabled.  It accidentally
applies that new-schema requirement to the older, SHA-qualified T4/TS4
references, whose metadata predates the field.  This adapter changes only the
reference loader: a missing historical field is accepted as disabled after the
result artifact has matched the hash frozen in the r10 receipt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import aggregate_t4_m30_experiment_a_r5 as implementation
from t4_m30_experiment_a_r10_authorization import RECEIPT, require_claim


_ORIGINAL_LOAD = implementation.load


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_qualified_historical_reference(
    path: Path,
    arm: str,
    seed: int,
    ref_hashes: dict,
) -> tuple[np.ndarray, dict]:
    _need(path.is_file(), f"missing artifact {path}")
    _need(_sha256(path) == ref_hashes[arm][str(seed)], f"unqualified reference {path}")

    result = json.loads(path.read_text())
    protocol = result.get("protocol", {})
    _need(
        (
            result.get("variant"),
            result.get("seed"),
            result.get("signal_view"),
            result.get("epoch_list"),
        )
        == ("B3S", seed, "sua", list(implementation.EPOCHS)),
        f"artifact identity {path}",
    )
    _need(
        (
            protocol.get("calibration_n"),
            protocol.get("pool_size"),
            protocol.get("total_epochs"),
            protocol.get("burn_in_epochs"),
        )
        == (30, 30, 12, 4),
        f"protocol drift {path}",
    )
    _need(
        result.get("no_test_files_evaluated") is True
        and result.get("uses_backward_gradients") is False,
        f"test/gradient drift {path}",
    )

    metadata_path = Path(result["run_metadata_path"])
    _need(
        metadata_path.is_file()
        and _sha256(metadata_path) == result.get("run_metadata_sha256"),
        f"metadata binding {path}",
    )
    metadata = json.loads(metadata_path.read_text())
    side = metadata.get("side_features", {})
    _need(
        (
            metadata.get("status"),
            metadata.get("variant"),
            metadata.get("seed"),
            metadata.get("task"),
            tuple(metadata.get("split_counts", [])),
            metadata.get("max_units_exclusive"),
            metadata.get("signal_view"),
            side.get("group"),
            side.get("side_dim"),
            side.get("pool_size"),
            side.get("feature_version"),
            metadata.get("held_out_test_evaluated"),
            metadata.get("teacher_sha256"),
            metadata.get("train_val_manifest_sha256"),
        )
        == (
            "completed",
            "B3S",
            seed,
            "CO",
            (27, 6, 6),
            100,
            "sua",
            arm,
            4,
            30,
            1,
            False,
            "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d",
            "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9",
        ),
        f"metadata drift {path}",
    )
    _need(
        metadata.get("teacher_checkpoint")
        == "/home/xinyuan/Work_host/SPINT/sua_exploration/checkpoints/teacher_mc_maze/"
        "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
        and metadata.get("train_val_manifest")
        == "/home/xinyuan/Work_host/SPINT/sua_exploration/configs/"
        "subc_co_27_6_strict_train_val_manifest.json",
        f"metadata path drift {path}",
    )
    _need(
        side.get("normalization_sha256")
        == "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0",
        f"normalizer drift {path}",
    )

    training = metadata.get("training", {})
    _need(
        (
            training.get("calibration_n_trials"),
            training.get("max_epochs"),
            training.get("no_early_stopping"),
            training.get("checkpoint_every_epoch"),
            training.get("loss_mode"),
            training.get("learning_rate"),
            training.get("batch_size"),
            training.get("freeze_decoder"),
            training.get("identity_mode"),
            training.get("deterministic"),
        )
        == (30, 12, True, True, "task_only", 1e-4, 32, False, "calibrated", True),
        f"training drift {path}",
    )
    residual = metadata.get("t4_logit_residual")
    _need(
        residual is None
        or (isinstance(residual, dict) and residual.get("enabled") is False),
        f"historical reference residual drift {path}",
    )

    rows = []
    for epoch in implementation.EPOCHS:
        per_session = result["per_epoch"][str(epoch)]["per_session_r2"]
        _need(
            sorted(per_session) == list(implementation.SESSIONS),
            f"session drift {path}",
        )
        rows.append([per_session[session] for session in implementation.SESSIONS])
    values = np.asarray(rows, dtype=float)
    _need(np.isfinite(values).all(), f"NaN result {path}")
    evidence = {
        "artifact_sha256": _sha256(path),
        "run_metadata_sha256": _sha256(metadata_path),
        "historical_schema_compatibility": (
            "SHA-qualified pre-logit-residual-schema reference; missing field means disabled"
        ),
    }
    return values, evidence


def load(
    path: Path,
    arm: str,
    seed: int,
    *,
    new: bool,
    status_dir: Path | None,
    train_sha: str,
    ref_hashes: dict,
) -> tuple[np.ndarray, dict]:
    if new:
        return _ORIGINAL_LOAD(
            path,
            arm,
            seed,
            new=True,
            status_dir=status_dir,
            train_sha=train_sha,
            ref_hashes=ref_hashes,
        )
    return _load_qualified_historical_reference(path, arm, seed, ref_hashes)


implementation.RECEIPT = RECEIPT
implementation.require_claim = require_claim
implementation.load = load
main = implementation.main


if __name__ == "__main__":
    main()
