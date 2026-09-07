"""A11 B0 convergence hygiene: exact-source, CPU-only validation replay.

This is intentionally a narrow evaluator for the historical SUA B0 mainline:

* the only accepted checkpoints are B0 seed 42/43/44 epoch_000..011;
* it uses the exact committed source revision that produced the original B0
  epoch-window measurements, not the subsequently evolved working tree;
* it permits NWB reads from exactly the six recorded development-validation
  sessions.  The training-session normalizer is loaded from its already
  materialized, source-fingerprinted cache; no training or formal-test NWB is
  opened during replay;
* it explicitly forces CPU inference, disables gradients, and hashes model
  state before/after every checkpoint score;
* receipts are created with O_EXCL and made read-only.  A companion SHA-256
  sidecar binds the immutable JSON payload.

The historical B0 run was a fixed 12-epoch experiment.  The evaluator reports
all twelve forward-validation values for each seed and separately computes the
predeclared trailing epoch-5..12 mean.  It never resumes, trains, or evaluates
the six formal-test sessions.

The source revision is materialized into a temporary compatibility package from
Git blobs.  This is deliberate: the active code has materially changed since
the B0 run, so a superficially similar evaluation through the active imports
would not be a provenance-preserving replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PROGRAM_ID = "a11_b0_convergence_full_access_v1"
HISTORICAL_SOURCE_COMMIT = "df69a3ddfcc2281ab10d4383c870233adfdb3abf"
SEEDS: tuple[int, ...] = (42, 43, 44)
LIGHTNING_EPOCHS: tuple[int, ...] = tuple(range(12))
PROTOCOL_EPOCHS: tuple[int, ...] = tuple(range(5, 13))
CALIBRATION_N = 30
POOL_SIZE = 30
BIN_SIZE_MS = 20
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
PAD_VALUE = -1.0
BEHAVIOR_SCALING_FACTOR = 5.0
TASK = "CO"
VARIANT = "B0"
SIGNAL_VIEW = "sua"
MAX_UNITS_EXCLUSIVE = 100
SPLIT_COUNTS = (27, 6, 6)


# These are Git blob object IDs, not live working-tree hashes.  Pinning both the
# commit and the blobs catches a rewritten or malformed local Git object before
# a forward pass can begin.
HISTORICAL_RUNTIME_BLOBS: Mapping[str, str] = {
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py": "aeeb86e9e49be2918b509aba2bc4d80d63fd54af",
    "sua_exploration/scripts/eval_adaptation_dandi688.py": "98e250a7cdeb5ad2ad3cc2dafcdd6f7343ad1fb5",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py": "26301d27429e62523800747f8d12309fc8ef7165",
    "sua_exploration/mc_maze/datamodule.py": "e9e6e91e77cbfafad6b5459a0347e1dc138eaecb",
    "sua_exploration/mc_maze/multisession_datamodule.py": "fc918ec6632fe9747befdd08833cdbca183df6c7",
    "streaming_calibration_exp/src/models/falcon_module.py": "980647ab89052bc91fb4a9f70ae9b71f8439459d",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py": "5a67fbec4dad77037ce17dfc7aba3805c81c6f91",
    "streaming_calibration_exp/src/models/components/neuron_dropout.py": "59b9cab42bca192a4cff6340606acf6995c698bd",
    "streaming_calibration_exp/src/models/components/spint.py": "8d37f70ca522aba761fdab176ff361eee250852d",
    "streaming_calibration_exp/src/models/components/streaming_spint.py": "57b4c17a997d43cb22bb097290c1156b624537f0",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py": "883c842dbefc006a18779c8ddfb5a77297f399b6",
}

EXPECTED_HPARAMS: Mapping[str, Any] = {
    "task": "mc_maze",
    "variant": VARIANT,
    "window_size": WINDOW_SIZE,
    "trial_length": TRIAL_LENGTH,
    "id_hidden_dim": 128,
    "hidden_dim": 64,
    "num_emas": 4,
    "num_filters": 4,
    "kernel_size": 5,
    "learnable_ema_alpha": False,
    "sparsity_k": 16,
    "pad_value": PAD_VALUE,
    "freeze_decoder": False,
    "encoder_warmstart_path": None,
    "freeze_encoder_base": False,
    "tune_encoder_fusion": False,
    "fusion_mean_lr_scale": 1.0,
    "loss_mode": "task_only",
    "lambda_y": 1.0,
    "lambda_E": 0.1,
    "decode_last_timestep_only": True,
    "predict_scaled_behavior": True,
    "behavior_scaling_factor": BEHAVIOR_SCALING_FACTOR,
    "compile": False,
    "neuron_dropout_mode": "none",
    "identity_mode": "calibrated",
    "fixed_slot_count": 0,
    "side_dim": 0,
    "electrode_embed_dim": 0,
    "num_electrodes": 0,
}


class AuthorityError(RuntimeError):
    """A missing or mismatched authority that must block evaluation."""


@dataclass(frozen=True)
class RunAuthority:
    seed: int
    run_dir: Path
    metadata_path: Path
    hparams_path: Path
    metadata: Mapping[str, Any]
    epoch_checkpoints: Mapping[int, Path]
    checkpoint_sha256: Mapping[int, str]
    metadata_sha256: str
    hparams_sha256: str


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(_canonical_json_bytes(payload))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_fingerprint(path: Path) -> dict[str, int | str]:
    """Match the historical normalizer-cache key's file-stat fingerprint exactly."""
    details = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(details.st_size),
        "mtime_ns": int(details.st_mtime_ns),
    }


def behavior_stats_cache_key(train_paths: Sequence[Path], bin_size_ms: int = BIN_SIZE_MS) -> str:
    """Reproduce the historical ``_behavior_stats_cache_path`` hash contract."""
    payload = {
        "cache_format_version": 1,
        "kind": "behavior_stats",
        "bin_size_ms": int(bin_size_ms),
        "train_sources": [_source_fingerprint(path) for path in train_paths],
    }
    return sha256_json(payload)


def normalizer_cache_path(cache_dir: Path, train_paths: Sequence[Path]) -> Path:
    return cache_dir / "behavior_stats" / f"{behavior_stats_cache_key(train_paths)[:20]}.npz"


def _run_git(repo_root: Path, args: Sequence[str], *, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=text,
    )


def _require_git_source(repo_root: Path) -> dict[str, str]:
    probe = _run_git(repo_root, ["cat-file", "-e", f"{HISTORICAL_SOURCE_COMMIT}^{{commit}}"])
    if probe.returncode != 0:
        raise AuthorityError(
            f"historical source commit {HISTORICAL_SOURCE_COMMIT} is unavailable: "
            f"{probe.stderr.strip()}"
        )
    observed: dict[str, str] = {}
    for relative_path, expected_blob in HISTORICAL_RUNTIME_BLOBS.items():
        result = _run_git(repo_root, ["rev-parse", f"{HISTORICAL_SOURCE_COMMIT}:{relative_path}"])
        if result.returncode != 0:
            raise AuthorityError(
                f"historical source file is unavailable: {relative_path}: {result.stderr.strip()}"
            )
        actual_blob = result.stdout.strip()
        if actual_blob != expected_blob:
            raise AuthorityError(
                f"historical source blob mismatch for {relative_path}: expected "
                f"{expected_blob}, found {actual_blob}"
            )
        observed[relative_path] = actual_blob
    return observed


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AuthorityError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str, expected_length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise AuthorityError(f"{name} must be a list")
    if expected_length is not None and len(value) != expected_length:
        raise AuthorityError(
            f"{name} must contain {expected_length} entries, found {len(value)}"
        )
    return value


def _require_same(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise AuthorityError(f"{name} must be {expected!r}, found {value!r}")


def _assert_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AuthorityError(f"{label} escapes its allowed root: {resolved}") from exc
    return resolved


def _run_dir(repo_root: Path, seed: int) -> Path:
    return repo_root / "sua_exploration" / "checkpoints" / (
        f"sua_spint_t4_mainline_fp32_v1_b0_dandi688_co_s{seed}"
    )


def _load_hparams(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AuthorityError(f"cannot read checkpoint hparams {path}: {exc}") from exc
    return _require_mapping(payload, f"hparams {path}")


def _validate_hparams(hparams: Mapping[str, Any], seed: int) -> None:
    for key, expected in EXPECTED_HPARAMS.items():
        _require_same(hparams.get(key), expected, f"seed {seed} hparams.{key}")


def _validate_run_metadata(
    *,
    metadata: Mapping[str, Any],
    seed: int,
    run_dir: Path,
    repo_root: Path,
) -> None:
    _require_same(metadata.get("status"), "completed", f"seed {seed} metadata.status")
    _require_same(metadata.get("seed"), seed, f"seed {seed} metadata.seed")
    _require_same(metadata.get("variant"), VARIANT, f"seed {seed} metadata.variant")
    _require_same(metadata.get("task"), TASK, f"seed {seed} metadata.task")
    _require_same(metadata.get("signal_view"), SIGNAL_VIEW, f"seed {seed} metadata.signal_view")
    _require_same(
        metadata.get("max_units_exclusive"),
        MAX_UNITS_EXCLUSIVE,
        f"seed {seed} metadata.max_units_exclusive",
    )
    _require_same(
        tuple(_require_list(metadata.get("split_counts"), f"seed {seed} split_counts", 3)),
        SPLIT_COUNTS,
        f"seed {seed} metadata.split_counts",
    )
    _require_same(
        metadata.get("held_out_test_evaluated"),
        False,
        f"seed {seed} metadata.held_out_test_evaluated",
    )
    training = _require_mapping(metadata.get("training"), f"seed {seed} metadata.training")
    expected_training = {
        "max_epochs": 12,
        "no_early_stopping": True,
        "checkpoint_every_epoch": True,
        "calibration_n_trials": CALIBRATION_N,
        "bin_size_ms": BIN_SIZE_MS,
        "window_size": WINDOW_SIZE,
        "trial_length": TRIAL_LENGTH,
        "behavior_scaling_factor": BEHAVIOR_SCALING_FACTOR,
        "decode_last_timestep_only": True,
        "loss_mode": "task_only",
        "identity_mode": "calibrated",
    }
    for key, expected in expected_training.items():
        _require_same(training.get(key), expected, f"seed {seed} metadata.training.{key}")

    data_dir = _assert_under(
        Path(str(metadata.get("data_dir", ""))),
        repo_root / "sua_exploration" / "data",
        f"seed {seed} data_dir",
    )
    expected_data_dir = repo_root / "sua_exploration" / "data" / "dandi_000688" / "sub-C"
    _require_same(data_dir, expected_data_dir.resolve(), f"seed {seed} metadata.data_dir")
    if not data_dir.is_dir():
        raise AuthorityError(f"seed {seed} data directory does not exist: {data_dir}")

    expected_manifest = (
        repo_root / "sua_exploration" / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
    ).resolve()
    _require_same(
        Path(str(metadata.get("train_val_manifest", ""))).resolve(),
        expected_manifest,
        f"seed {seed} metadata.train_val_manifest",
    )
    expected_cache_dir = (repo_root / "sua_exploration" / "cache" / "dandi688_subc_co_v1").resolve()
    _require_same(
        Path(str(metadata.get("cache_dir", ""))).resolve(),
        expected_cache_dir,
        f"seed {seed} metadata.cache_dir",
    )
    teacher_path = _assert_under(
        Path(str(metadata.get("teacher_checkpoint", ""))),
        repo_root / "sua_exploration" / "checkpoints",
        f"seed {seed} teacher checkpoint",
    )
    if not teacher_path.is_file():
        raise AuthorityError(f"seed {seed} teacher checkpoint is missing: {teacher_path}")
    expected_teacher_sha = metadata.get("teacher_sha256")
    if not isinstance(expected_teacher_sha, str) or len(expected_teacher_sha) != 64:
        raise AuthorityError(f"seed {seed} metadata.teacher_sha256 is invalid")
    observed_teacher_sha = sha256_file(teacher_path)
    if observed_teacher_sha != expected_teacher_sha:
        raise AuthorityError(
            f"seed {seed} teacher checkpoint SHA-256 mismatch: expected "
            f"{expected_teacher_sha}, found {observed_teacher_sha}"
        )

    session_files = _require_mapping(metadata.get("session_files"), f"seed {seed} session_files")
    _require_same(
        _require_list(session_files.get("test"), f"seed {seed} session_files.test"),
        [],
        f"seed {seed} session_files.test",
    )
    train_files = _require_list(session_files.get("train"), f"seed {seed} session_files.train", 27)
    val_files = _require_list(session_files.get("val"), f"seed {seed} session_files.val", 6)
    for split, paths in (("train", train_files), ("val", val_files)):
        for index, raw_path in enumerate(paths):
            path = _assert_under(Path(str(raw_path)), data_dir, f"seed {seed} {split}[{index}]")
            if path.parent != data_dir or not path.is_file():
                raise AuthorityError(f"seed {seed} {split}[{index}] is missing: {path}")
            if not path.name.endswith("_behavior+ecephys.nwb"):
                raise AuthorityError(f"seed {seed} {split}[{index}] is not an expected NWB name: {path}")

    session_splits = _require_mapping(metadata.get("session_splits"), f"seed {seed} session_splits")
    expected_train_names = [Path(str(path)).name.replace("_behavior+ecephys.nwb", "") for path in train_files]
    expected_val_names = [Path(str(path)).name.replace("_behavior+ecephys.nwb", "") for path in val_files]
    _require_same(
        session_splits.get("train"), expected_train_names, f"seed {seed} session_splits.train"
    )
    _require_same(session_splits.get("val"), expected_val_names, f"seed {seed} session_splits.val")

    raw_epoch_paths = _require_list(metadata.get("epoch_checkpoints"), f"seed {seed} epoch_checkpoints", 12)
    expected_epoch_paths = [
        str((run_dir / "epoch_ckpts" / f"epoch_{epoch:03d}.ckpt").resolve())
        for epoch in LIGHTNING_EPOCHS
    ]
    _require_same(raw_epoch_paths, expected_epoch_paths, f"seed {seed} metadata.epoch_checkpoints")


def audit_run(repo_root: Path, seed: int) -> RunAuthority:
    run_dir = _run_dir(repo_root, seed).resolve()
    if not run_dir.is_dir():
        raise AuthorityError(f"seed {seed} run directory is missing: {run_dir}")
    metadata_path = run_dir / "run_metadata.json"
    hparams_path = run_dir / "lightning_logs" / "version_0" / "hparams.yaml"
    if not metadata_path.is_file():
        raise AuthorityError(f"seed {seed} run metadata is missing: {metadata_path}")
    if not hparams_path.is_file():
        raise AuthorityError(f"seed {seed} checkpoint hparams are missing: {hparams_path}")
    try:
        metadata = _require_mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")), f"seed {seed} run metadata"
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"cannot read seed {seed} run metadata: {exc}") from exc
    _validate_run_metadata(metadata=metadata, seed=seed, run_dir=run_dir, repo_root=repo_root)
    hparams = _load_hparams(hparams_path)
    _validate_hparams(hparams, seed)

    checkpoint_paths: dict[int, Path] = {}
    checkpoint_hashes: dict[int, str] = {}
    for epoch in LIGHTNING_EPOCHS:
        checkpoint_path = run_dir / "epoch_ckpts" / f"epoch_{epoch:03d}.ckpt"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise AuthorityError(f"seed {seed} checkpoint is missing or empty: {checkpoint_path}")
        checkpoint_paths[epoch] = checkpoint_path.resolve()
        checkpoint_hashes[epoch] = sha256_file(checkpoint_path)

    return RunAuthority(
        seed=seed,
        run_dir=run_dir,
        metadata_path=metadata_path.resolve(),
        hparams_path=hparams_path.resolve(),
        metadata=metadata,
        epoch_checkpoints=checkpoint_paths,
        checkpoint_sha256=checkpoint_hashes,
        metadata_sha256=sha256_file(metadata_path),
        hparams_sha256=sha256_file(hparams_path),
    )


def query_contract() -> dict[str, Any]:
    return {
        "signal_view": SIGNAL_VIEW,
        "bin_size_ms": BIN_SIZE_MS,
        "trial_result_filter": "R",
        "calibration_selection_mode": "first",
        "calibration_n_trials": CALIBRATION_N,
        "excluded_prefix_pool_trials": POOL_SIZE,
        "query_trial_rule": "usable_rewarded_trials[30:]",
        "query_is_disjoint_from_calibration": True,
        "window_size_bins": WINDOW_SIZE,
        "max_trial_length_bins": TRIAL_LENGTH,
        "calibration_pad_value": PAD_VALUE,
        "calibration_interpolate_trials": True,
        "behavior_normalization": "frozen_train_only_cached_mean_std",
        "metric": {
            "name": "torchmetrics.regression.R2Score",
            "multioutput": "variance_weighted",
            "prediction": "decoder_output_last_bin / 5.0",
            "target": "normalized_cursor_velocity_last_bin",
        },
    }


def _validate_normalizer(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuthorityError(
            "original train-only normalizer cache is missing; refusing to recompute it from "
            f"NWB files: {path}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float32)
            std = np.asarray(archive["std"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as exc:
        raise AuthorityError(f"original train-only normalizer cache is unreadable: {path}: {exc}") from exc
    if mean.shape != (2,) or std.shape != (2,):
        raise AuthorityError(
            f"normalizer must contain two velocity dimensions, found mean={mean.shape}, std={std.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise AuthorityError("normalizer mean/std are not finite positive train-only statistics")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "array_contract_sha256": sha256_json(
            {"mean": mean.astype("<f4").tolist(), "std": std.astype("<f4").tolist()}
        ),
    }


def build_preflight(repo_root: Path) -> dict[str, Any]:
    """Read-only audit of every authority needed for a CPU forward replay."""
    repo_root = repo_root.resolve()
    historical_blobs = _require_git_source(repo_root)
    runs = [audit_run(repo_root, seed) for seed in SEEDS]
    baseline = runs[0].metadata
    baseline_data_dir = Path(str(baseline["data_dir"])).resolve()
    baseline_cache_dir = Path(str(baseline["cache_dir"])).resolve()
    baseline_manifest = Path(str(baseline["train_val_manifest"])).resolve()
    if not baseline_manifest.is_file():
        raise AuthorityError(f"frozen train/validation manifest is missing: {baseline_manifest}")
    manifest_sha256 = sha256_file(baseline_manifest)
    expected_manifest_sha256 = baseline.get("train_val_manifest_sha256")
    if manifest_sha256 != expected_manifest_sha256:
        raise AuthorityError(
            "frozen train/validation manifest SHA-256 differs from the B0 run metadata: "
            f"expected {expected_manifest_sha256}, found {manifest_sha256}"
        )

    val_paths = [Path(str(path)).resolve() for path in baseline["session_files"]["val"]]
    train_paths = [Path(str(path)).resolve() for path in baseline["session_files"]["train"]]
    val_names = [path.name.replace("_behavior+ecephys.nwb", "") for path in val_paths]
    for run in runs[1:]:
        metadata = run.metadata
        for key in ("data_dir", "cache_dir", "train_val_manifest", "teacher_checkpoint"):
            if metadata.get(key) != baseline.get(key):
                raise AuthorityError(f"seed {run.seed} metadata.{key} drifted from seed {runs[0].seed}")
        if metadata["session_files"]["train"] != baseline["session_files"]["train"]:
            raise AuthorityError(f"seed {run.seed} train session roster drifted")
        if metadata["session_files"]["val"] != baseline["session_files"]["val"]:
            raise AuthorityError(f"seed {run.seed} validation session roster drifted")
        if metadata.get("train_val_manifest_sha256") != expected_manifest_sha256:
            raise AuthorityError(f"seed {run.seed} manifest authority hash drifted")
        if metadata.get("teacher_sha256") != baseline.get("teacher_sha256"):
            raise AuthorityError(f"seed {run.seed} teacher authority hash drifted")

    normalizer_path = normalizer_cache_path(baseline_cache_dir, train_paths)
    normalizer = _validate_normalizer(normalizer_path)
    # The cache's input identity records file stats only.  No training NWB is opened.
    train_fingerprints = [_source_fingerprint(path) for path in train_paths]
    val_fingerprints = [_source_fingerprint(path) for path in val_paths]
    for path in val_paths:
        if path.parent != baseline_data_dir or not path.is_file():
            raise AuthorityError(f"allowed validation NWB is absent: {path}")

    q_contract = query_contract()
    q_contract_sha256 = sha256_json(q_contract)
    run_receipts = {
        str(run.seed): {
            "run_dir": str(run.run_dir),
            "run_metadata_path": str(run.metadata_path),
            "run_metadata_sha256": run.metadata_sha256,
            "hparams_path": str(run.hparams_path),
            "hparams_sha256": run.hparams_sha256,
            "epoch_checkpoints": {
                f"epoch_{epoch:03d}": {
                    "path": str(run.epoch_checkpoints[epoch]),
                    "sha256": run.checkpoint_sha256[epoch],
                }
                for epoch in LIGHTNING_EPOCHS
            },
        }
        for run in runs
    }
    authority_core = {
        "program_id": PROGRAM_ID,
        "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
        "historical_source_blobs": historical_blobs,
        "manifest_sha256": manifest_sha256,
        "normalizer": normalizer,
        "validation_source_fingerprints": val_fingerprints,
        "runs": run_receipts,
        "query_contract_sha256": q_contract_sha256,
    }
    return {
        "program_id": PROGRAM_ID,
        "status": "preflight_passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_source": {
            "commit": HISTORICAL_SOURCE_COMMIT,
            "blobs": historical_blobs,
            "why_pinned": (
                "The active tree changed after the B0 run. The replay materializes these "
                "historical Git blobs into a temporary compatibility package."
            ),
        },
        "source_configuration_authority": {
            "variant": VARIANT,
            "task": TASK,
            "expected_hparams": dict(EXPECTED_HPARAMS),
            "run_receipts": run_receipts,
        },
        "data_authority": {
            "data_dir": str(baseline_data_dir),
            "strict_manifest_path": str(baseline_manifest),
            "strict_manifest_sha256": manifest_sha256,
            "allowed_validation_sessions": val_names,
            "allowed_validation_nwb_paths": [str(path) for path in val_paths],
            "allowed_validation_source_fingerprints": val_fingerprints,
            "training_normalizer_source_fingerprints": train_fingerprints,
            "formal_test_nwb_access": "forbidden",
            "training_nwb_access_during_replay": "forbidden",
            "normalizer": normalizer,
        },
        "query_and_metric_contract": {
            "query_contract": q_contract,
            "query_contract_sha256": q_contract_sha256,
            "full_curve_lightning_epochs": list(LIGHTNING_EPOCHS),
            "full_curve_human_epochs": list(range(1, 13)),
            "predeclared_epoch_window_human_epochs": list(PROTOCOL_EPOCHS),
            "predeclared_epoch_window_checkpoint_names": [
                f"epoch_{epoch:03d}.ckpt" for epoch in range(4, 12)
            ],
        },
        "execution_contract": {
            "device": "cpu",
            "cuda_must_be_invisible": True,
            "torch_grad_enabled": False,
            "allowed_nwb_open_paths": [str(path) for path in val_paths],
            "forward_only": True,
            "weight_updates": False,
            "backward_gradients": False,
            "state_hash_before_after_required": True,
        },
        "authority_fingerprint_sha256": sha256_json(authority_core),
        "_runtime": {
            "runs": runs,
            "normalizer_path": normalizer_path,
            "val_paths": val_paths,
            "data_dir": baseline_data_dir,
            "cache_dir": baseline_cache_dir,
        },
    }


# This text becomes a file in the temporary exact-source package.  It uses the
# historical implementation's loading/data/metric primitives but has a stricter
# data guard than the historical generic evaluator: it never resolves or counts
# train/test NWBs, and it consumes the cached original normalizer directly.
COMPATIBILITY_DRIVER_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


RESULT_MARKER = "A11_RESULT_JSON="


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def tensor_state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    config = read_config(sys.argv[1])
    source_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(source_root / "sua_exploration" / "scripts"))
    sys.path.insert(0, str(source_root / "sua_exploration"))
    sys.path.insert(0, str(source_root / "streaming_calibration_exp"))
    # Must be enforced before importing model code.  The outer process provides
    # CUDA_VISIBLE_DEVICES="", and the check is a fail-closed backstop.
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        raise RuntimeError("A11 CPU replay requires CUDA_VISIBLE_DEVICES to be empty")
    # Do not call torch.cuda.is_available(): even that probe can initialize a
    # CUDA runtime on some installations.  The empty visibility mask above is
    # the process-level non-GPU contract; all tensors/models below are checked
    # explicitly for CPU placement.
    device = torch.device("cpu")
    if device.type != "cpu":
        raise RuntimeError("A11 CPU replay selected a non-CPU device")
    torch.set_grad_enabled(False)
    if torch.is_grad_enabled():
        raise RuntimeError("A11 CPU replay could not disable autograd")

    import eval_adaptation_dandi688 as adaptation
    import mc_maze.multisession_datamodule as multisession
    from dandi688_gradient_free_protocol import select_calibration_trial_indices
    from eval_adaptation_dandi688 import (
        build_calib_trials_for_indices,
        eval_r2,
        load_session_with_trials,
        make_subset_dataset,
    )
    from select_gradient_free_protocol_dandi688 import load_frozen_model

    allowed_paths = {str(Path(path).resolve()) for path in config["allowed_validation_nwb_paths"]}
    opened_paths = []
    original_eval_nwb_io = adaptation.NWBHDF5IO
    original_multi_nwb_io = multisession.NWBHDF5IO

    class GuardedNWBHDF5IO:
        def __init__(self, path, *args, **kwargs):
            canonical = str(Path(path).resolve())
            if canonical not in allowed_paths:
                raise RuntimeError("A11 data-scope violation: attempted NWB open outside the six validation files: " + canonical)
            opened_paths.append(canonical)
            self._wrapped = original_eval_nwb_io(path, *args, **kwargs)

        def __enter__(self):
            return self._wrapped.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._wrapped.__exit__(exc_type, exc_value, traceback)

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    # Both historical modules receive the guard.  Normalizer fallback would hit
    # multisession and fail rather than opening a train NWB.
    adaptation.NWBHDF5IO = GuardedNWBHDF5IO
    multisession.NWBHDF5IO = GuardedNWBHDF5IO

    normalizer_path = Path(config["normalizer_cache_path"])
    with np.load(normalizer_path, allow_pickle=False) as normalizer:
        mean = normalizer["mean"].astype(np.float32, copy=False)
        std = normalizer["std"].astype(np.float32, copy=False)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError("A11 normalizer cache failed shape/finiteness validation")

    records = []
    for raw_path in config["allowed_validation_nwb_paths"]:
        path = Path(raw_path).resolve()
        rec = load_session_with_trials(
            path,
            config["bin_size_ms"],
            config["window_size"],
            config["pool_size"],
            config["trial_length"],
            config["pad_value"],
            mean,
            std,
            trial_result_filter="R",
            cache_dir=Path(config["private_cache_dir"]),
            signal_view=config["signal_view"],
        )
        if rec["name"] not in config["allowed_validation_sessions"]:
            raise RuntimeError("A11 record name escaped the allowed validation roster: " + repr(rec["name"]))
        if rec["signal_view"] != "sua":
            raise RuntimeError("A11 record signal view drifted from SUA")
        selection = select_calibration_trial_indices(
            rec["trials"], config["calibration_n"], config["pool_size"], "first"
        )
        expected_selection = list(range(config["calibration_n"]))
        if selection != expected_selection:
            raise RuntimeError("A11 fixed-first calibration selection drifted")
        rec["calib_trials"] = build_calib_trials_for_indices(
            rec, selection, config["calibration_n"]
        )
        query_trials = rec["trials"][config["pool_size"]:]
        if not query_trials:
            raise RuntimeError("A11 query is empty after the fixed support pool")
        dataset = make_subset_dataset(rec, query_trials, rec["name"])
        if not len(dataset):
            raise RuntimeError("A11 query has zero scored windows")
        support_payload = [
            {
                "usable_trial_list_index": int(index),
                "original_trial_index": int(rec["trials"][index]["trial_index"]),
            }
            for index in selection
        ]
        query_payload = [
            {
                "original_trial_index": int(trial["trial_index"]),
                "start": int(trial["start"]),
                "stop": int(trial["stop"]),
                "target_dir": trial.get("target_dir"),
                "target_id": trial.get("target_id"),
            }
            for trial in query_trials
        ]
        starts = np.asarray(dataset.valid_starts, dtype="<i8")
        records.append(
            {
                "name": rec["name"],
                "dataset": dataset,
                "source_unit_count": int(rec["source_unit_count"]),
                "support_trial_count": len(support_payload),
                "support_trial_sha256": sha256_json(support_payload),
                "query_trial_count": len(query_payload),
                "query_trial_sha256": sha256_json(query_payload),
                "scored_window_count": int(len(dataset)),
                "scored_window_start_sha256": sha256_bytes(starts.tobytes()),
            }
        )

    result = {
        "device": device.type,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_grad_enabled": bool(torch.is_grad_enabled()),
        "normalizer_arrays_sha256": sha256_json(
            {"mean": mean.astype("<f4").tolist(), "std": std.astype("<f4").tolist()}
        ),
        "query_windows": {
            item["name"]: {
                key: value
                for key, value in item.items()
                if key != "dataset"
            }
            for item in records
        },
        "per_seed": {},
    }
    for run in config["runs"]:
        seed = int(run["seed"])
        per_epoch = {}
        for epoch_text, checkpoint_path in run["epoch_checkpoints"].items():
            epoch = int(epoch_text)
            model = load_frozen_model(
                Path(checkpoint_path),
                Path(config["teacher_checkpoint"]),
                config["variant"],
                device,
            )
            if model.training:
                raise RuntimeError("A11 frozen model unexpectedly remained in training mode")
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("A11 frozen model has a trainable parameter")
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise RuntimeError("A11 frozen model parameter escaped CPU")
            before = tensor_state_sha256(model)
            per_session = {}
            with torch.no_grad():
                for item in records:
                    per_session[item["name"]] = float(eval_r2(model, item["dataset"], device))
            after = tensor_state_sha256(model)
            if before != after:
                raise RuntimeError("A11 forward-only invariant failed: checkpoint state changed during evaluation")
            per_epoch[str(epoch)] = {
                "per_session_r2": per_session,
                "mean_r2": float(sum(per_session.values()) / len(per_session)),
                "model_state_sha256_before": before,
                "model_state_sha256_after": after,
            }
            del model
        result["per_seed"][str(seed)] = {"per_epoch": per_epoch}

    if not set(opened_paths).issubset(allowed_paths):
        raise RuntimeError("A11 data-scope violation was not caught by the guarded NWB reader")
    result["opened_nwb_paths"] = sorted(set(opened_paths))
    result["opened_nwb_path_count"] = len(result["opened_nwb_paths"])
    print(RESULT_MARKER + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
'''


def _materialize_historical_runtime(repo_root: Path, target_root: Path) -> dict[str, str]:
    """Write only the import closure needed by the historical B0 forward path."""
    source_sha256: dict[str, str] = {}
    for relative_path, expected_blob in HISTORICAL_RUNTIME_BLOBS.items():
        result = _run_git(
            repo_root,
            ["show", f"{HISTORICAL_SOURCE_COMMIT}:{relative_path}"],
            text=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise AuthorityError(f"cannot materialize historical source {relative_path}: {stderr}")
        target = target_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
        observed_blob = _run_git(
            repo_root, ["hash-object", str(target)], text=True
        ).stdout.strip()
        # Git's local hash-object default is SHA-1 for this repository.  It is
        # checked against the pinned source blob as an additional materialization check.
        if observed_blob != expected_blob:
            raise AuthorityError(
                f"materialized historical source hash mismatch for {relative_path}: "
                f"expected {expected_blob}, found {observed_blob}"
            )
        source_sha256[relative_path] = sha256_file(target)

    # Namespace packages work without these files on modern Python, but explicit
    # package markers make the temporary import surface deterministic.
    for relative_dir in (
        "sua_exploration/mc_maze",
        "streaming_calibration_exp/src",
        "streaming_calibration_exp/src/models",
        "streaming_calibration_exp/src/models/components",
    ):
        init_path = target_root / relative_dir / "__init__.py"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.touch(exist_ok=True)
    driver_path = target_root / "a11_compatibility_driver.py"
    driver_path.write_text(textwrap.dedent(COMPATIBILITY_DRIVER_SOURCE), encoding="utf-8")
    source_sha256["a11_compatibility_driver.py"] = sha256_file(driver_path)
    return source_sha256


def _copy_normalizer_into_private_cache(
    *,
    normalizer_path: Path,
    private_cache_dir: Path,
    train_paths: Sequence[Path],
) -> Path:
    expected_target = normalizer_cache_path(private_cache_dir, train_paths)
    expected_target.parent.mkdir(parents=True, exist_ok=True)
    if expected_target.exists():
        raise AuthorityError(f"private normalizer target unexpectedly exists: {expected_target}")
    # The source normalizer has already passed its numerical and SHA checks.  The
    # private cache prevents the historical function from ever attempting to build
    # a normalizer by opening a train NWB.
    expected_target.write_bytes(normalizer_path.read_bytes())
    if sha256_file(expected_target) != sha256_file(normalizer_path):
        raise AuthorityError("private normalizer copy SHA-256 mismatch")
    return expected_target


def _driver_config(preflight: Mapping[str, Any], private_cache_dir: Path) -> dict[str, Any]:
    runtime = _require_mapping(preflight["_runtime"], "preflight runtime")
    runs = runtime["runs"]
    if not isinstance(runs, list) or not all(isinstance(run, RunAuthority) for run in runs):
        raise AuthorityError("internal preflight run authority is malformed")
    normalizer_target = _copy_normalizer_into_private_cache(
        normalizer_path=Path(runtime["normalizer_path"]),
        private_cache_dir=private_cache_dir,
        train_paths=[Path(str(path)) for path in runs[0].metadata["session_files"]["train"]],
    )
    return {
        "allowed_validation_nwb_paths": preflight["data_authority"]["allowed_validation_nwb_paths"],
        "allowed_validation_sessions": preflight["data_authority"]["allowed_validation_sessions"],
        "normalizer_cache_path": str(normalizer_target),
        "private_cache_dir": str(private_cache_dir),
        "teacher_checkpoint": runs[0].metadata["teacher_checkpoint"],
        "variant": VARIANT,
        "signal_view": SIGNAL_VIEW,
        "bin_size_ms": BIN_SIZE_MS,
        "window_size": WINDOW_SIZE,
        "pool_size": POOL_SIZE,
        "calibration_n": CALIBRATION_N,
        "trial_length": TRIAL_LENGTH,
        "pad_value": PAD_VALUE,
        "runs": [
            {
                "seed": run.seed,
                "epoch_checkpoints": {
                    str(epoch): str(run.epoch_checkpoints[epoch]) for epoch in LIGHTNING_EPOCHS
                },
            }
            for run in runs
        ],
    }


def _run_driver(repo_root: Path, preflight: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Run the exact-source driver and return its only machine-readable payload."""
    with tempfile.TemporaryDirectory(prefix="a11_b0_exact_source_") as temporary_dir:
        root = Path(temporary_dir)
        materialized_root = root / "historical_source"
        source_sha256 = _materialize_historical_runtime(repo_root, materialized_root)
        private_cache_dir = root / "private_validation_cache"
        config = _driver_config(preflight, private_cache_dir)
        selected_runs: list[dict[str, Any]] = []
        for run in config["runs"]:
            permitted_epochs = plan["epochs_by_seed"][str(run["seed"])]
            selected_runs.append(
                {
                    "seed": run["seed"],
                    "epoch_checkpoints": {
                        epoch: checkpoint
                        for epoch, checkpoint in run["epoch_checkpoints"].items()
                        if int(epoch) in permitted_epochs
                    },
                }
            )
        config["runs"] = selected_runs
        config_path = root / "driver_config.json"
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONPATH"] = ""
        completed = subprocess.run(
            [sys.executable, str(materialized_root / "a11_compatibility_driver.py"), str(config_path)],
            cwd=str(repo_root),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        marker_lines = [
            line[len("A11_RESULT_JSON=") :]
            for line in completed.stdout.splitlines()
            if line.startswith("A11_RESULT_JSON=")
        ]
        if completed.returncode != 0:
            raise AuthorityError(
                "historical CPU driver failed with exit code "
                f"{completed.returncode}. stdout tail: {completed.stdout[-4000:]!r}; "
                f"stderr tail: {completed.stderr[-4000:]!r}"
            )
        if len(marker_lines) != 1:
            raise AuthorityError(
                "historical CPU driver did not emit exactly one result marker. "
                f"stdout tail: {completed.stdout[-4000:]!r}; stderr tail: {completed.stderr[-4000:]!r}"
            )
        try:
            payload = json.loads(marker_lines[0])
        except json.JSONDecodeError as exc:
            raise AuthorityError(f"historical CPU driver emitted malformed JSON: {exc}") from exc
        if payload.get("device") != "cpu" or payload.get("torch_grad_enabled") is not False:
            raise AuthorityError("historical CPU driver violated its CPU/no-grad execution contract")
        allowed = set(preflight["data_authority"]["allowed_validation_nwb_paths"])
        opened = set(payload.get("opened_nwb_paths", []))
        if not opened.issubset(allowed):
            raise AuthorityError("historical CPU driver reported an NWB open outside the validation allowlist")
        if not opened:
            raise AuthorityError("historical CPU driver reported no validation NWB opens")
        return {
            "driver_payload": payload,
            "materialized_source_sha256": source_sha256,
            "driver_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "driver_stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        }


def _full_plan() -> dict[str, Any]:
    return {
        "kind": "scientific_full_curve",
        "science_claim_allowed": True,
        "epochs_by_seed": {str(seed): list(LIGHTNING_EPOCHS) for seed in SEEDS},
    }


def _smoke_plan() -> dict[str, Any]:
    return {
        "kind": "non_scientific_cpu_forward_smoke",
        "science_claim_allowed": False,
        "epochs_by_seed": {"42": [0], "43": [], "44": []},
    }


def summarize_full_curve(driver_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Derive descriptive curves without changing the predeclared score rule."""
    per_seed = _require_mapping(driver_payload.get("per_seed"), "driver per_seed")
    expected_seed_keys = {str(seed) for seed in SEEDS}
    if set(per_seed) != expected_seed_keys:
        raise AuthorityError(f"full curve requires seeds {sorted(expected_seed_keys)}, got {sorted(per_seed)}")
    per_seed_summary: dict[str, Any] = {}
    aggregate_values: dict[int, list[float]] = {epoch: [] for epoch in LIGHTNING_EPOCHS}
    late_minus_early_values: list[float] = []
    for seed in SEEDS:
        seed_data = _require_mapping(per_seed[str(seed)], f"seed {seed} driver result")
        epochs = _require_mapping(seed_data.get("per_epoch"), f"seed {seed} per_epoch")
        expected_epoch_keys = {str(epoch) for epoch in LIGHTNING_EPOCHS}
        if set(epochs) != expected_epoch_keys:
            raise AuthorityError(
                f"seed {seed} full curve requires epochs {sorted(expected_epoch_keys)}, got {sorted(epochs)}"
            )
        means: dict[int, float] = {}
        for epoch in LIGHTNING_EPOCHS:
            record = _require_mapping(epochs[str(epoch)], f"seed {seed} epoch {epoch}")
            mean = float(record.get("mean_r2"))
            if not np.isfinite(mean):
                raise AuthorityError(f"seed {seed} epoch {epoch} has a non-finite validation R2")
            if record.get("model_state_sha256_before") != record.get("model_state_sha256_after"):
                raise AuthorityError(f"seed {seed} epoch {epoch} mutated during forward evaluation")
            means[epoch] = mean
            aggregate_values[epoch].append(mean)
        window_values = [means[human_epoch - 1] for human_epoch in PROTOCOL_EPOCHS]
        early_mean = float(np.mean([means[human_epoch - 1] for human_epoch in range(5, 9)]))
        late_mean = float(np.mean([means[human_epoch - 1] for human_epoch in range(9, 13)]))
        late_minus_early = late_mean - early_mean
        late_human_epochs = np.arange(9, 13, dtype=np.float64)
        late_values = np.asarray([means[human_epoch - 1] for human_epoch in range(9, 13)], dtype=np.float64)
        late_slope = float(np.polyfit(late_human_epochs, late_values, deg=1)[0])
        best_human_epoch = int(max(range(1, 13), key=lambda human_epoch: means[human_epoch - 1]))
        late_minus_early_values.append(late_minus_early)
        per_seed_summary[str(seed)] = {
            "mean_validation_r2_by_lightning_epoch": {str(epoch): means[epoch] for epoch in LIGHTNING_EPOCHS},
            "mean_validation_r2_by_human_epoch": {
                str(epoch + 1): means[epoch] for epoch in LIGHTNING_EPOCHS
            },
            "epoch_5_to_12_mean_validation_r2": float(np.mean(window_values)),
            "epoch_5_to_12_values": window_values,
            "mean_human_epochs_5_to_8": early_mean,
            "mean_human_epochs_9_to_12": late_mean,
            "late_minus_early": late_minus_early,
            "last_four_epoch_ols_slope": late_slope,
            "best_human_epoch": best_human_epoch,
        }
    aggregate_by_epoch = {
        str(epoch + 1): {
            "mean_over_seeds": float(np.mean(values)),
            "sample_std_over_seeds": float(np.std(values, ddof=1)),
            "values_by_seed": values,
        }
        for epoch, values in aggregate_values.items()
    }
    mean_late_minus_early = float(np.mean(late_minus_early_values))
    extension_candidate = bool(
        mean_late_minus_early >= 0.03 and all(value > 0.0 for value in late_minus_early_values)
    )
    return {
        "per_seed": per_seed_summary,
        "aggregate_by_human_epoch": aggregate_by_epoch,
        "three_seed_mean_of_epoch_5_to_12_scores": float(
            np.mean(
                [
                    per_seed_summary[str(seed)]["epoch_5_to_12_mean_validation_r2"]
                    for seed in SEEDS
                ]
            )
        ),
        "epoch_5_to_12_average_supported": True,
        "extension_candidate_diagnostic": {
            "rule_frozen_before_real_full_curve": True,
            "mean_late_minus_early": mean_late_minus_early,
            "per_seed_late_minus_early": {
                str(seed): late_minus_early_values[index] for index, seed in enumerate(SEEDS)
            },
            "threshold": 0.03,
            "requires_all_three_seed_deltas_positive": True,
            "candidate_flag": extension_candidate,
            "authorizes_training_or_continuation": False,
            "interpretation": (
                "A descriptive flag that the stored curve was still improving late; it never "
                "authorizes checkpoint selection, resumed training, or a scientific claim."
            ),
        },
    }


def _public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in preflight.items() if key != "_runtime"}


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path, str]:
    """Create a receipt once; the evaluator never overwrites an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    digest = sha256_bytes(data)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except FileExistsError as exc:
        raise AuthorityError(f"immutable receipt already exists and will not be overwritten: {path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar = Path(f"{path}.sha256")
    sidecar_data = f"{digest}  {path.name}\n".encode("ascii")
    try:
        fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except FileExistsError as exc:
        raise AuthorityError(
            f"immutable receipt digest sidecar already exists and will not be overwritten: {sidecar}"
        ) from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(sidecar_data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return path, sidecar, digest


def _output_path(repo_root: Path, mode: str, authority_fingerprint: str, out_dir: Path | None) -> Path:
    destination_dir = (
        out_dir.resolve()
        if out_dir is not None
        else repo_root / "sua_exploration" / "results" / PROGRAM_ID
    )
    return destination_dir / f"{PROGRAM_ID}_{mode}_{authority_fingerprint[:16]}.json"


def _failure_receipt(
    repo_root: Path,
    out_dir: Path | None,
    mode: str,
    error: Exception,
) -> tuple[Path, Path, str]:
    fingerprint = sha256_json(
        {"program_id": PROGRAM_ID, "mode": mode, "error": f"{type(error).__name__}: {error}"}
    )
    payload = {
        "program_id": PROGRAM_ID,
        "status": "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "error_type": type(error).__name__,
        "error": str(error),
        "forward_evaluation_started": False,
        "formal_test_nwb_access": False,
    }
    return write_immutable_json(_output_path(repo_root, f"{mode}_blocked", fingerprint, out_dir), payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full-cpu-forward",
        action="store_true",
        help="Run all 36 pinned checkpoints on CPU and create the scientific full-curve receipt.",
    )
    mode.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="Run only seed 42 epoch_000 as a non-scientific CPU/guard smoke receipt.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Additive receipt directory; defaults to sua_exploration/results/a11_b0_convergence_full_access_v1.",
    )
    args = parser.parse_args(argv)
    repo_root = project_root()
    selected_mode = "preflight"
    if args.full_cpu_forward:
        selected_mode = "full_cpu_forward"
    elif args.cpu_smoke:
        selected_mode = "cpu_smoke"

    try:
        preflight = build_preflight(repo_root)
        authority_fingerprint = str(preflight["authority_fingerprint_sha256"])
        output_path = _output_path(repo_root, selected_mode, authority_fingerprint, args.out_dir)
        if selected_mode == "preflight":
            payload = _public_preflight(preflight)
            payload["receipt_kind"] = "read_only_authority_audit"
            payload["forward_evaluation_started"] = False
        else:
            plan = _full_plan() if selected_mode == "full_cpu_forward" else _smoke_plan()
            driver_result = _run_driver(repo_root, preflight, plan)
            payload = _public_preflight(preflight)
            payload.update(
                {
                    "status": "completed",
                    "receipt_kind": plan["kind"],
                    "science_claim_allowed": plan["science_claim_allowed"],
                    "forward_evaluation_started": True,
                    "execution_plan": plan,
                    "metric_implementation": {
                        "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
                        "historical_runtime_blobs": dict(HISTORICAL_RUNTIME_BLOBS),
                        "materialized_source_sha256": driver_result["materialized_source_sha256"],
                        "compatibility_driver_sha256": driver_result["materialized_source_sha256"][
                            "a11_compatibility_driver.py"
                        ],
                        "outer_evaluator_sha256": sha256_file(Path(__file__).resolve()),
                    },
                    "cpu_forward_result": driver_result["driver_payload"],
                    "driver_stdout_sha256": driver_result["driver_stdout_sha256"],
                    "driver_stderr_sha256": driver_result["driver_stderr_sha256"],
                }
            )
            if selected_mode == "full_cpu_forward":
                payload["convergence_summary"] = summarize_full_curve(driver_result["driver_payload"])
            else:
                payload["convergence_summary"] = {
                    "epoch_5_to_12_average_supported": False,
                    "continuation_beyond_epoch_12_scientifically_warranted": False,
                    "reason": "A one-checkpoint smoke test is non-scientific and cannot support convergence claims.",
                }
        receipt, sidecar, digest = write_immutable_json(output_path, payload)
        print(f"A11 receipt: {receipt}")
        print(f"A11 receipt SHA-256: {digest}")
        print(f"A11 receipt sidecar: {sidecar}")
        return 0
    except Exception as exc:
        try:
            receipt, sidecar, digest = _failure_receipt(repo_root, args.out_dir, selected_mode, exc)
        except Exception as receipt_error:
            print(
                f"A11 blocked before a receipt could be written: {type(exc).__name__}: {exc}; "
                f"receipt error: {type(receipt_error).__name__}: {receipt_error}",
                file=sys.stderr,
            )
            return 2
        print(f"A11 blocked receipt: {receipt}", file=sys.stderr)
        print(f"A11 blocked receipt SHA-256: {digest}", file=sys.stderr)
        print(f"A11 blocked receipt sidecar: {sidecar}", file=sys.stderr)
        print(f"A11 block reason: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
