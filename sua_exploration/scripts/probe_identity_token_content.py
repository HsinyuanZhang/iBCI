#!/usr/bin/env python3
"""Run the root-audited A4-v2 identity-token content probe on CPU only.

The executable contract is deliberately narrower than the superseded v1
scaffold:

* paired AC4/Z4 component-attribution v10 checkpoints of one seed only;
* the v10 pre-declared fixed epoch window, never an argmax ``best_checkpoint``;
* checkpoint-authoritative M=30 activity support, carrier target support,
  train roster, validation roster, descriptor contract, and normalizer; and
* six-development-session LOSO ridge fits, with target rows permuted only
  inside each training session for the primary pairing-permutation null.

It never resolves a formal-test NWB path, performs backward work, mutates a
checkpoint, or exposes CUDA.  The output receipt is write-once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Must precede Torch and the scientific stack.  The command also verifies this
# runtime state after import, rather than treating it as a documentation claim.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
SCRIPTS_ROOT = SUA_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))

from eval_adaptation_dandi688 import (  # noqa: E402
    DEFAULT_TEACHER,
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    attach_side_features,
    checkpoint_architecture_kwargs,
    load_session_with_trials,
)
from mc_maze.identity_token_content_probe import (  # noqa: E402
    ACTIVITY_CALIBRATION_N,
    CARRIER_TARGET_POOL_N,
    PAIRING_PERMUTATION_VERSION,
    PROBE_NAME,
    SCHEMA_VERSION,
    ProbeHyperparameters,
    aggregate_fixed_epoch_window,
    array_sha256,
    assert_sessions_not_sealed,
    canonical_json_bytes,
    fit_session_loso_probe_suite,
    sha256_bytes,
)
from mc_maze.multisession_datamodule import (  # noqa: E402
    fit_behavior_stats,
    load_frozen_train_val_manifest,
    session_name_from_path,
)
from mc_maze.unit_side_features import (  # noqa: E402
    _compute_tuning_features_uncached,
    fit_side_feature_stats,
    side_feature_stats_sha256,
)
from select_gradient_free_protocol_dandi688 import load_frozen_model  # noqa: E402


EXPECTED_V10_VARIANT = "B3S"
EXPECTED_V10_SIGNAL_VIEW = "sua"
EXPECTED_V10_TASK = "CO"
EXPECTED_V10_SPLIT_COUNTS = [27, 6, 6]
EXPECTED_V10_SIDES = {"ac4": "carrier", "z4": "activity_only"}
EXPECTED_DESCRIPTOR_CONTRACT = {
    "ordinary_t4_normalizer_reused": True,
    "mask_applied_after_standardization": True,
}
EXPECTED_EPOCH_WINDOW = tuple(range(5, 13))
FROZEN_PROBE_HYPERPARAMETERS = ProbeHyperparameters(
    normalized_ridge_lambda=1.0,
    random_seed=42,
    modulation_eps=1.0e-6,
    min_session_units=15,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes_exclusive(path: Path, payload: bytes, *, label: str) -> None:
    """Publish one immutable artifact without a check-then-write race."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing {label}: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        # Operate through the still-open exclusive descriptor so a path race
        # cannot redirect the permission change to a different artifact.
        os.fchmod(handle.fileno(), 0o444)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_frozen_probe_hyperparameters(hyperparameters: ProbeHyperparameters) -> None:
    """Reject programmatic as well as CLI drift from the pre-declared A4 fit/null settings."""
    _require(
        hyperparameters.as_dict() == FROZEN_PROBE_HYPERPARAMETERS.as_dict(),
        "A4 frozen probe hyperparameters drift",
    )


def _checkpoint_metadata_path(checkpoint: Path) -> Path:
    # v10 epoch checkpoints live ``<run>/epoch_ckpts/epoch_*.ckpt``.  Requiring
    # this structure rejects arbitrary standalone checkpoint substitution.
    path = checkpoint.expanduser().resolve()
    _require(path.parent.name == "epoch_ckpts", f"checkpoint must be a v10 epoch checkpoint: {path}")
    metadata = path.parent.parent / "run_metadata.json"
    _require(metadata.is_file(), f"checkpoint run_metadata is missing: {metadata}")
    return metadata


def _epoch_from_checkpoint_path(path: Path) -> int:
    name = path.name
    if not name.startswith("epoch_") or not name.endswith(".ckpt"):
        raise ValueError(f"unexpected epoch checkpoint name: {path}")
    try:
        return int(name[len("epoch_") : -len(".ckpt")]) + 1
    except ValueError as exc:
        raise ValueError(f"unexpected epoch checkpoint name: {path}") from exc


def _canonical_epoch_checkpoints(metadata: Mapping[str, Any]) -> dict[int, Path]:
    paths = metadata.get("epoch_checkpoints")
    if not isinstance(paths, list):
        raise ValueError("run_metadata.epoch_checkpoints must be a list")
    found: dict[int, Path] = {}
    for item in paths:
        path = Path(str(item)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"listed epoch checkpoint missing: {path}")
        epoch = _epoch_from_checkpoint_path(path)
        if epoch in found:
            raise ValueError(f"duplicate epoch {epoch} in run_metadata")
        found[epoch] = path
    if tuple(sorted(found)) != tuple(range(1, 13)):
        raise ValueError("v10 run_metadata must bind exactly epochs 1..12")
    return {epoch: found[epoch] for epoch in EXPECTED_EPOCH_WINDOW}


def _validate_v10_metadata(
    metadata: Mapping[str, Any],
    *,
    metadata_path: Path,
    expected_arm: str,
    expected_seed: int | None,
) -> dict[str, Any]:
    side = metadata.get("side_features") or {}
    training = metadata.get("training") or {}
    _require(metadata.get("status") == "completed", f"{metadata_path}: run is not completed")
    _require(metadata.get("variant") == EXPECTED_V10_VARIANT, f"{metadata_path}: variant drift")
    _require(metadata.get("task") == EXPECTED_V10_TASK, f"{metadata_path}: task drift")
    _require(metadata.get("signal_view") == EXPECTED_V10_SIGNAL_VIEW, f"{metadata_path}: signal view drift")
    _require(metadata.get("split_counts") == EXPECTED_V10_SPLIT_COUNTS, f"{metadata_path}: split counts drift")
    _require(metadata.get("seed") == expected_seed, f"{metadata_path}: paired seed drift")
    _require(side.get("group") == expected_arm, f"{metadata_path}: expected arm {expected_arm}")
    _require(side.get("side_dim") == 4, f"{metadata_path}: side dimension must be four")
    _require(side.get("pool_size") == ACTIVITY_CALIBRATION_N, f"{metadata_path}: M=30 side pool required")
    _require(
        side.get("normalization_base_feature_group") == "t4",
        f"{metadata_path}: normalizer substrate must be ordinary t4",
    )
    _require(
        side.get("descriptor_contract") == EXPECTED_DESCRIPTOR_CONTRACT,
        f"{metadata_path}: descriptor contract drift",
    )
    _require(training.get("calibration_n_trials") == ACTIVITY_CALIBRATION_N, f"{metadata_path}: activity M=30 required")
    _require(training.get("window_size") == WINDOW_SIZE, f"{metadata_path}: window size drift")
    _require(training.get("trial_length") == TRIAL_LENGTH, f"{metadata_path}: trial length drift")
    _require(training.get("max_epochs") == 12, f"{metadata_path}: expected 12 epochs")
    _require(training.get("no_early_stopping") is True, f"{metadata_path}: fixed epoch rule required")
    _require(metadata.get("held_out_test_evaluated") is False, f"{metadata_path}: formal test was evaluated")
    _require((metadata.get("session_files") or {}).get("test") == [], f"{metadata_path}: test files listed")
    _require(isinstance(metadata.get("session_splits"), dict), f"{metadata_path}: missing roster")
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "seed": int(metadata["seed"]),
        "arm": expected_arm,
        "side_group": str(side["group"]),
        "pool_size": int(side["pool_size"]),
        "normalization_sha256": str(side["normalization_sha256"]),
        "descriptor_contract": dict(side["descriptor_contract"]),
        "train_val_manifest": str(metadata.get("train_val_manifest", "")),
        "train_val_manifest_sha256": str(metadata.get("train_val_manifest_sha256", "")),
        "teacher_checkpoint": str(metadata.get("teacher_checkpoint", "")),
        "teacher_sha256": str(metadata.get("teacher_sha256", "")),
        "session_splits": metadata["session_splits"],
    }


def validate_paired_v10_inputs(ac4_checkpoint: Path, z4_checkpoint: Path) -> dict[str, Any]:
    """Fail closed unless paired v10 metadata and fixed epoch rules are exact."""
    ac4_metadata_path = _checkpoint_metadata_path(ac4_checkpoint)
    z4_metadata_path = _checkpoint_metadata_path(z4_checkpoint)
    ac4_metadata = _read_json(ac4_metadata_path)
    z4_metadata = _read_json(z4_metadata_path)
    seed = ac4_metadata.get("seed")
    _require(isinstance(seed, int), f"{ac4_metadata_path}: integer seed required")
    ac4 = _validate_v10_metadata(
        ac4_metadata,
        metadata_path=ac4_metadata_path,
        expected_arm="ac4",
        expected_seed=seed,
    )
    z4 = _validate_v10_metadata(
        z4_metadata,
        metadata_path=z4_metadata_path,
        expected_arm="z4",
        expected_seed=seed,
    )
    for key in (
        "normalization_sha256",
        "descriptor_contract",
        "train_val_manifest",
        "train_val_manifest_sha256",
        "teacher_checkpoint",
        "teacher_sha256",
        "session_splits",
    ):
        _require(ac4[key] == z4[key], f"paired AC4/Z4 metadata drift: {key}")
    _require(
        Path(ac4["train_val_manifest"]).expanduser().resolve().is_file(),
        "checkpoint-authoritative strict manifest is absent",
    )
    manifest_path = Path(ac4["train_val_manifest"]).expanduser().resolve()
    _require(sha256_file(manifest_path) == ac4["train_val_manifest_sha256"], "strict manifest SHA drift")
    manifest = _read_json(manifest_path)
    _require(manifest.get("session_splits") == ac4["session_splits"], "manifest roster drift")
    val_sessions = list(manifest["session_splits"].get("val", []))
    train_sessions = list(manifest["session_splits"].get("train", []))
    _require(len(train_sessions) == 27 and len(val_sessions) == 6, "v10 must use the 27/6 development roster")
    assert_sessions_not_sealed(val_sessions)
    _require(set(val_sessions).isdisjoint(manifest["session_splits"].get("test", [])), "validation/formal overlap")

    ac4_epochs = _canonical_epoch_checkpoints(ac4_metadata)
    z4_epochs = _canonical_epoch_checkpoints(z4_metadata)
    # The explicit CLI checkpoint names must be epoch-5 representatives; this
    # prevents a caller from silently putting the receipt under an unrelated run.
    _require(ac4_checkpoint.expanduser().resolve() == ac4_epochs[5], "--ac4-checkpoint must be canonical v10 epoch 5")
    _require(z4_checkpoint.expanduser().resolve() == z4_epochs[5], "--z4-checkpoint must be canonical v10 epoch 5")
    return {
        "seed": seed,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "train_sessions": train_sessions,
        "validation_sessions": val_sessions,
        "formal_test_sessions": list(manifest["session_splits"].get("test", [])),
        "teacher_path": Path(ac4["teacher_checkpoint"]).expanduser().resolve(),
        "teacher_sha256": ac4["teacher_sha256"],
        "normalization_sha256": ac4["normalization_sha256"],
        "descriptor_contract": ac4["descriptor_contract"],
        "arms": {
            "ac4": {**ac4, "epoch_checkpoints": ac4_epochs},
            "z4": {**z4, "epoch_checkpoints": z4_epochs},
        },
    }


def _load_train_and_validation_paths(manifest_path: Path) -> tuple[list[Path], list[Path]]:
    data_dir = SUA_ROOT / "data" / "dandi_000688" / "sub-C"
    train_paths, validation_paths, formal_names = load_frozen_train_val_manifest(manifest_path, data_dir)
    names = [session_name_from_path(path) for path in validation_paths]
    assert_sessions_not_sealed(names)
    _require(not (set(names) & set(formal_names)), "resolved validation paths include sealed sessions")
    return train_paths, validation_paths


def _checkpoint_authoritative_normalizers(
    *,
    train_paths: Sequence[Path],
    cache_dir: Path | None,
    expected_side_normalizer_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover both train-only normalizers and prove the side normalizer matches metadata."""
    behavior_mean, behavior_std = fit_behavior_stats(train_paths, bin_size_ms=20, cache_dir=cache_dir)
    side_mean, side_std = fit_side_feature_stats(
        train_paths,
        feature_group="t4",
        pool_size=ACTIVITY_CALIBRATION_N,
        cache_dir=cache_dir,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view=EXPECTED_V10_SIGNAL_VIEW,
    )
    observed = side_feature_stats_sha256(side_mean, side_std)
    _require(observed == expected_side_normalizer_sha256, "checkpoint-authoritative T4 normalizer SHA mismatch")
    provenance = {
        "behavior_normalizer": {
            "fit_scope": "checkpoint_train_roster_only",
            "train_session_count": len(train_paths),
            "mean_digest_sha256": array_sha256(behavior_mean),
            "std_digest_sha256": array_sha256(behavior_std),
        },
        "side_normalizer": {
            "fit_scope": "checkpoint_train_roster_only",
            "base_feature_group": "t4",
            "pool_size": ACTIVITY_CALIBRATION_N,
            "sha256": observed,
            "metadata_sha256": expected_side_normalizer_sha256,
            "mean_digest_sha256": array_sha256(side_mean),
            "std_digest_sha256": array_sha256(side_std),
        },
    }
    return behavior_mean, behavior_std, side_mean, side_std, provenance


def _load_raw_carrier_target(nwb_path: Path) -> np.ndarray:
    """Raw T4 target fitted from the exact same M=30 rewarded support."""
    raw, _metadata = _compute_tuning_features_uncached(
        nwb_path,
        feature_group="t4",
        pool_size=CARRIER_TARGET_POOL_N,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view=EXPECTED_V10_SIGNAL_VIEW,
    )
    return np.asarray(raw, dtype=np.float64)


def _validate_activity_calibration_shape(
    calibration_trials: np.ndarray,
    *,
    session: str,
    n_units: int,
) -> tuple[int, int, int]:
    """Require the loader/encoder calibration layout ``[M, T, N]`` exactly.

    ``load_session_with_trials`` delegates to ``_build_calib_trials``, whose
    canonical output is ``[calibration_trial, time_bin, unit]``.  The
    downstream identity encoder consumes the batch-expanded form
    ``[B, M, T, N]``.  Checking all three unbatched axes here catches a stale
    cache or a transposed tensor before either arm can be evaluated.
    """
    values = np.asarray(calibration_trials)
    expected = (ACTIVITY_CALIBRATION_N, TRIAL_LENGTH, int(n_units))
    observed = tuple(int(item) for item in values.shape)
    _require(
        values.ndim == 3 and observed == expected,
        f"{session}: activity calibration must be [M,T,N]={expected}, got {observed}",
    )
    _require(
        np.issubdtype(values.dtype, np.number) and np.isfinite(values).all(),
        f"{session}: activity calibration contains non-finite or non-numeric values",
    )
    return observed


def _prepare_validation_inputs(
    *,
    validation_paths: Sequence[Path],
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    side_mean: np.ndarray,
    side_std: np.ndarray,
    cache_dir: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    """Open exactly six declared development NWBs and construct M=30 inputs/targets."""
    records: dict[str, dict[str, Any]] = {}
    carrier_targets: dict[str, np.ndarray] = {}
    input_digests: dict[str, Any] = {}
    for nwb_path in validation_paths:
        session = session_name_from_path(nwb_path)
        assert_sessions_not_sealed([session])
        rec = load_session_with_trials(
            nwb_path,
            bin_size_ms=20,
            window_size=WINDOW_SIZE,
            calib_n=ACTIVITY_CALIBRATION_N,
            max_trial_length=TRIAL_LENGTH,
            pad_value=PAD_VALUE,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            trial_result_filter="R",
            signal_view=EXPECTED_V10_SIGNAL_VIEW,
            cache_dir=cache_dir,
        )
        # Attach ordinary raw T4 transformed into each arm later.  Both arms
        # share this exact raw-source / normalizer substrate.
        rec = attach_side_features(
            rec,
            nwb_path,
            side_feature_group="t4",
            waveform_feature_group="t4",
            pool_size=ACTIVITY_CALIBRATION_N,
            permutation_seed=None,
            mean=side_mean,
            std=side_std,
            cache_dir=cache_dir,
        )
        raw_carrier = _load_raw_carrier_target(nwb_path)
        n_units = int(rec["n_units"])
        _require(
            raw_carrier.ndim == 2 and raw_carrier.shape == (n_units, 4),
            f"{session}: raw carrier must be [N,4]=({n_units}, 4), got {raw_carrier.shape}",
        )
        _validate_activity_calibration_shape(
            rec["calib_trials"],
            session=session,
            n_units=n_units,
        )
        records[session] = rec
        carrier_targets[session] = raw_carrier
        input_digests[session] = {
            "n_units": int(rec["n_units"]),
            "activity_calibration_shape": [int(item) for item in rec["calib_trials"].shape],
            "activity_calibration_digest_sha256": array_sha256(rec["calib_trials"]),
            "raw_carrier_target_shape": [int(item) for item in raw_carrier.shape],
            "raw_carrier_target_digest_sha256": array_sha256(raw_carrier),
            "ordinary_standardized_t4_digest_sha256": array_sha256(rec["side_features"]),
        }
    return records, carrier_targets, input_digests


def _side_for_arm(ordinary_standardized_t4: np.ndarray, arm: str) -> np.ndarray:
    values = np.asarray(ordinary_standardized_t4, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"ordinary standardized T4 must be [N,4], got {values.shape}")
    if arm == "ac4":
        result = values.copy()
        result[:, 2:] = 0.0
        return result
    if arm == "z4":
        return np.zeros_like(values)
    raise ValueError(f"unsupported A4 arm: {arm}")


def _compute_identity_tokens(model: Any, rec: Mapping[str, Any], side_features: np.ndarray) -> np.ndarray:
    calibration = torch.from_numpy(np.asarray(rec["calib_trials"], dtype=np.float32)).unsqueeze(0)
    side = torch.from_numpy(np.asarray(side_features, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        identity = model.student.compute_identity(calibration, side_features=side)
    values = identity.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
    if values.ndim != 2:
        raise RuntimeError(f"unexpected identity-token shape {values.shape}")
    return values


def _load_cpu_frozen_model(checkpoint: Path, teacher: Path) -> Any:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must remain empty")
    _require(not torch.cuda.is_available(), "A4 runner refuses a CUDA-visible runtime")
    model = load_frozen_model(
        checkpoint,
        teacher,
        variant=EXPECTED_V10_VARIANT,
        device=torch.device("cpu"),
        identity_mode="calibrated",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _run_arm_epoch_window(
    *,
    arm: str,
    arm_binding: Mapping[str, Any],
    teacher: Path,
    sessions: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    carrier_targets: Mapping[str, np.ndarray],
    hyperparameters: ProbeHyperparameters,
) -> dict[str, Any]:
    epoch_results: dict[int, dict[str, Any]] = {}
    checkpoint_evidence: dict[str, Any] = {}
    side_digest_by_session: dict[str, str] = {}
    for session in sessions:
        side_digest_by_session[session] = array_sha256(
            _side_for_arm(np.asarray(records[session]["side_features"]), arm)
        )
    for epoch, checkpoint in arm_binding["epoch_checkpoints"].items():
        model = _load_cpu_frozen_model(checkpoint, teacher)
        identity_tokens: dict[str, np.ndarray] = {}
        for session in sessions:
            side = _side_for_arm(np.asarray(records[session]["side_features"]), arm)
            identity = _compute_identity_tokens(model, records[session], side)
            _require(
                identity.shape[0] == carrier_targets[session].shape[0],
                f"{arm}/epoch{epoch}/{session}: token/target unit count drift",
            )
            identity_tokens[session] = identity
        epoch_results[epoch] = fit_session_loso_probe_suite(
            identity_tokens,
            carrier_targets,
            sessions=sessions,
            hyperparameters=hyperparameters,
        )
        checkpoint_evidence[str(epoch)] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "identity_token_digests_by_session": {
                session: array_sha256(identity_tokens[session]) for session in sessions
            },
            "architecture_kwargs": checkpoint_architecture_kwargs(
                torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            ),
        }
        del model
    aggregated = aggregate_fixed_epoch_window(epoch_results, sessions=sessions)
    return {
        "arm": arm,
        "arm_kind": EXPECTED_V10_SIDES[arm],
        "checkpoint_metadata": {
            key: arm_binding[key]
            for key in (
                "metadata_path",
                "metadata_sha256",
                "seed",
                "side_group",
                "pool_size",
                "normalization_sha256",
                "descriptor_contract",
            )
        },
        "epoch_window": list(EXPECTED_EPOCH_WINDOW),
        "epoch_checkpoints": checkpoint_evidence,
        "side_input_digests_by_session": side_digest_by_session,
        "per_epoch_session_loso": {str(epoch): epoch_results[epoch] for epoch in EXPECTED_EPOCH_WINDOW},
        "session_loso": aggregated,
    }


def run_paired_v10_preflight(
    *,
    ac4_checkpoint: Path,
    z4_checkpoint: Path,
    cache_dir: Path | None,
    hyperparameters: ProbeHyperparameters,
) -> dict[str, Any]:
    _require_frozen_probe_hyperparameters(hyperparameters)
    binding = validate_paired_v10_inputs(ac4_checkpoint, z4_checkpoint)
    teacher = binding["teacher_path"]
    _require(teacher.is_file(), f"checkpoint-authoritative teacher absent: {teacher}")
    _require(sha256_file(teacher) == binding["teacher_sha256"], "teacher checkpoint SHA drift")
    train_paths, validation_paths = _load_train_and_validation_paths(binding["manifest_path"])
    sessions = [session_name_from_path(path) for path in validation_paths]
    _require(sessions == binding["validation_sessions"], "checkpoint validation roster/order drift")
    assert_sessions_not_sealed(sessions)
    behavior_mean, behavior_std, side_mean, side_std, normalizer_provenance = _checkpoint_authoritative_normalizers(
        train_paths=train_paths,
        cache_dir=cache_dir,
        expected_side_normalizer_sha256=binding["normalization_sha256"],
    )
    records, carrier_targets, input_digests = _prepare_validation_inputs(
        validation_paths=validation_paths,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        side_mean=side_mean,
        side_std=side_std,
        cache_dir=cache_dir,
    )
    ac4 = _run_arm_epoch_window(
        arm="ac4",
        arm_binding=binding["arms"]["ac4"],
        teacher=teacher,
        sessions=sessions,
        records=records,
        carrier_targets=carrier_targets,
        hyperparameters=hyperparameters,
    )
    z4 = _run_arm_epoch_window(
        arm="z4",
        arm_binding=binding["arms"]["z4"],
        teacher=teacher,
        sessions=sessions,
        records=records,
        carrier_targets=carrier_targets,
        hyperparameters=hyperparameters,
    )
    # This is an especially important paired check: null seeds and carrier row
    # digests must match across arms, before an aggregate may compare them.
    _require(
        ac4["session_loso"]["pairing_permutation"] == z4["session_loso"]["pairing_permutation"],
        "AC4/Z4 pairing-permutation manifest drift",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_name": PROBE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm_pair": {"carrier": "ac4", "activity_only": "z4"},
        "seed": binding["seed"],
        "sessions": sessions,
        "sealed_test_sessions": binding["formal_test_sessions"],
        "sealed_test_sessions_opened": False,
        "execution_scope": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "no_training_no_backward": True,
            "checkpoint_mutated": False,
            "opened_nwb_sessions": sessions,
            "opened_nwb_session_count": len(sessions),
        },
        "checkpoint_binding": {
            "run_family": "sua_t4_m30_component_attribution_v10",
            "fixed_epoch_rule": "pre_declared_unweighted_epoch_window_5_to_12",
            "fixed_epoch_window": list(EXPECTED_EPOCH_WINDOW),
            "manifest_path": str(binding["manifest_path"]),
            "manifest_sha256": binding["manifest_sha256"],
            "teacher_path": str(teacher),
            "teacher_sha256": binding["teacher_sha256"],
            "normalization_sha256": binding["normalization_sha256"],
            "descriptor_contract": binding["descriptor_contract"],
            "train_roster": binding["train_sessions"],
            "validation_roster": sessions,
        },
        "protocol": {
            "activity_calibration_n": ACTIVITY_CALIBRATION_N,
            "carrier_target_pool_n": CARRIER_TARGET_POOL_N,
            "same_support_for_identity_and_raw_carrier_target": True,
            "window_size": WINDOW_SIZE,
            "trial_length": TRIAL_LENGTH,
            "cpu_only": True,
            "no_training_no_backward": True,
            "map_location": "cpu",
            "estimator": "session_loso_global_ridge",
            "pairing_permutation_version": PAIRING_PERMUTATION_VERSION,
        },
        "probe_hyperparameters": hyperparameters.as_dict(),
        "normalizer_provenance": normalizer_provenance,
        "shared_validation_input_digests": input_digests,
        "arms": {"ac4": ac4, "z4": z4},
    }


def write_receipt(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["receipt_body_sha256"] = sha256_bytes(canonical_json_bytes(body))
    _write_bytes_exclusive(
        output_path,
        (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        label="A4 receipt",
    )
    sidecar_path = output_path.with_name(output_path.name + ".sha256")
    _write_bytes_exclusive(
        sidecar_path,
        f"{sha256_file(output_path)}  {output_path.name}\n".encode("ascii"),
        label="A4 receipt SHA sidecar",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ac4-checkpoint", required=True, type=Path, help="canonical v10 AC4 epoch_004.ckpt")
    parser.add_argument("--z4-checkpoint", required=True, type=Path, help="canonical v10 Z4 epoch_004.ckpt")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    receipt = run_paired_v10_preflight(
        ac4_checkpoint=args.ac4_checkpoint.expanduser().resolve(),
        z4_checkpoint=args.z4_checkpoint.expanduser().resolve(),
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
        hyperparameters=FROZEN_PROBE_HYPERPARAMETERS,
    )
    write_receipt(receipt, args.out)
    print(json.dumps({
        "output": str(args.out.expanduser().resolve()),
        "sessions": receipt["sessions"],
        "ac4_phase": receipt["arms"]["ac4"]["session_loso"]["pooled"]["probes"]["phase_unit"],
        "z4_phase": receipt["arms"]["z4"]["session_loso"]["pooled"]["probes"]["phase_unit"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
