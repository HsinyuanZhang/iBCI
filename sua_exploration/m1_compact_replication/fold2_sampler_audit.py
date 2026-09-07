#!/usr/bin/env python3
"""Source-only fold-2 sampler cardinality audit.

This is a CPU-only, pre-launch receipt.  It binds the exact fold-2 source
sessions and sampler order to the staged receipt, and derives the terminal
optimizer step from that receipt plus the fixed 12-epoch trainer policy.  It
never resolves the left-out target session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from sua_exploration.m1_compact_replication import fold1_runner as base


ROOT = base.ROOT
RECEIPT = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F2_S42_SOURCE_SAMPLER_AUDIT_v1.json"
)
SCHEMA = "m1_compact_b3s_f2_s42_source_sampler_audit_v1"
STATUS = "PASS_M1_COMPACT_B3S_F2_S42_SOURCE_SAMPLER_AUDITED_NOT_LAUNCHED"
SOURCES = ("ses-20120924", "ses-20120926", "ses-20120928")
TARGET = "ses-20120927"
EQUIVALENCE = "source_only_fit_reconstructed_prelaunch"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} is not an object")
    return value


def read_immutable(path: Path = RECEIPT) -> dict[str, Any]:
    value = read(path, "fold-2 sampler audit")
    need(path.stat().st_mode & 0o777 == 0o444, "fold-2 sampler audit is mutable")
    need(value.get("schema") == SCHEMA and value.get("status") == STATUS, "fold-2 sampler audit schema/status drift")
    need(value.get("canonical_content_sha256") == canonical({k: v for k, v in value.items() if k != "canonical_content_sha256"}), "fold-2 sampler audit canonical drift")
    return value


def _immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        temporary.replace(path)
        return sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _datamodule_kwargs() -> dict[str, Any]:
    return {
        "task": "m1",
        "data_dir": str((ROOT / "SPINT-main/data/000941").resolve()),
        "source_session_names": list(SOURCES),
        "heldin_session_names": list(SOURCES),
        "batch_size": 32,
        "window_size": 100,
        "calibration_n_trials": 10,
        "random_calibration": False,
        "smooth_calibration": False,
        "max_trial_length": 1024,
        "standardize_covariates": False,
        "use_intertrials": True,
        "use_calib_intertrials": False,
        "trial_feature_type": "raw",
        "remove_still_times": False,
        "remove_calib_still_times": False,
        "use_calib_active_segments": False,
        "calib_n_active_segments": 1,
        "interpolate_trials": True,
        "interpolate_trials_kind": "cubic",
        "pad_value": -1.0,
        "validation_protocol": "loso",
        "loso_fold": 2,
        "rotation_id": 0,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "query_start_trial": 0,
        "heldin_query_start_trial": 10,
        "heldin_query_end_trial": 210,
        "allow_empty_heldout_query": False,
        "num_workers": 0,
        "pin_memory": False,
        "sampler_seed": 42,
        "balance_session_batches": False,
        "reshuffle_train_sampler_each_epoch": False,
        "side_feature_shuffle_seed": 42,
        "afc4_arm": "none",
    }


def derive_sampler_cardinality(
    counts: Mapping[str, int],
    *,
    batch_size: int,
    scored_windows: int,
    dataset_windows: int,
    epochs: int = 12,
) -> dict[str, Any]:
    """Return the auditable fold-2 batch/step derivation."""
    need(set(counts) == set(SOURCES), "fold-2 sampler session set drift")
    normalized = {str(name): int(value) for name, value in counts.items()}
    need(all(value > 0 for value in normalized.values()), "fold-2 sampler contains nonpositive count")
    batches = int(sum(normalized.values()))
    need(batch_size > 0 and epochs > 0, "fold-2 sampler batch/epoch policy invalid")
    need(scored_windows == batches * int(batch_size), "fold-2 sampler windows do not equal full batches")
    need(dataset_windows >= scored_windows, "fold-2 dataset window count below scored windows")
    return {
        "source_batch_counts": normalized,
        "source_batches_per_epoch": batches,
        "batch_size": int(batch_size),
        "source_scored_windows_per_epoch": int(scored_windows),
        "dataset_windows": int(dataset_windows),
        "dropped_windows_per_epoch": int(dataset_windows - scored_windows),
        "epochs": int(epochs),
        "expected_global_step": batches * int(epochs),
        "derivation": "sum(source_batch_counts) * frozen max_epochs=12; no fold-0/fold-1 literal",
    }


def build_audit(staged_receipt: Path, staged_body: Mapping[str, Any]) -> dict[str, Any]:
    import sys
    stream_root = ROOT / "streaming_calibration_exp"
    if str(stream_root) not in sys.path:
        sys.path.insert(0, str(stream_root))
    from src.data.m1_version_b_source_loso_datamodule import M1VersionBSourceOnlyFitDataModule

    need(staged_receipt.is_file() and staged_receipt.stat().st_mode & 0o777 == 0o444, "fold-2 staged receipt missing/mutable")
    need(staged_body.get("scope", {}).get("source_sessions") == list(SOURCES), "fold-2 staged source scope drift")
    need(staged_body.get("scope", {}).get("outer_target") == TARGET, "fold-2 staged target scope drift")
    kwargs = _datamodule_kwargs()
    dm = M1VersionBSourceOnlyFitDataModule(**kwargs)
    dm.setup("fit")
    counts = dm._sampler_batch_counts(dm.train_batch_sampler)
    sampler_sha, scored_windows = dm._sampler_sha256(dm.train_batch_sampler)
    batch_size = int(dm.batch_size_per_device)
    dataset_windows = int(len(dm.train_dataset))
    cardinality = derive_sampler_cardinality(
        counts,
        batch_size=batch_size,
        scored_windows=int(scored_windows),
        dataset_windows=dataset_windows,
    )
    source_inventory = staged_body["inventory"]
    body = {
        "schema": SCHEMA,
        "status": STATUS,
        "staged_receipt": {"path": str(staged_receipt.resolve()), "sha256": sha(staged_receipt)},
        "scope": {"task": "m1", "fold": 2, "seed": 42, "source_sessions": list(SOURCES), "outer_target": TARGET, "target_opened": False, "target_values_used": False},
        "source_inventory": source_inventory,
        "sampler": {
            "source_batch_counts": cardinality["source_batch_counts"],
            "source_batches_per_epoch": cardinality["source_batches_per_epoch"],
            "batch_size": cardinality["batch_size"],
            "source_scored_windows_per_epoch": cardinality["source_scored_windows_per_epoch"],
            "dataset_windows": cardinality["dataset_windows"],
            "dropped_windows_per_epoch": cardinality["dropped_windows_per_epoch"],
            "sampler_sha256": sampler_sha,
        },
        "trainer": {"epochs": cardinality["epochs"], "expected_global_step": cardinality["expected_global_step"], "derivation": cardinality["derivation"]},
        "equivalence": {"class": "M1VersionBSourceOnlyFitDataModule", "fit_stage": "fit", "target_path_resolved": False, "validation_sessions": [], "calibration_n_trials": 10, "chronological_source_fit": True},
        "launch_policy": {"prepared_only": True, "fold1_gate_required": True, "launch_authorized": False, "target_metrics_read_by_runner": False},
    }
    return body


def prepare(staged_receipt: Path, staged_body: Mapping[str, Any]) -> dict[str, Any]:
    if RECEIPT.exists():
        return read_immutable()
    body = build_audit(staged_receipt.resolve(), staged_body)
    _immutable(RECEIPT, body)
    return read_immutable()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-receipt", type=Path, required=True)
    args = parser.parse_args()
    staged = read(args.staged_receipt.resolve(), "fold-2 staged receipt")
    body = prepare(args.staged_receipt, staged)
    print(json.dumps({"status": body["status"], "path": str(RECEIPT.resolve()), "sha256": sha(RECEIPT), "expected_global_step": body["trainer"]["expected_global_step"]}, sort_keys=True))


if __name__ == "__main__":
    main()
