"""Frozen-checkpoint M2 movement-window T4 ablation.

The governing MOVE carrier uses raw calibration bins [5, 30), i.e. 100--600 ms
after each trial boundary.  This window was frozen from the seven held-in
sessions' source-only speed geometry before any decoder score was observed: the
aggregate movement peak is bin 14 and has returned close to baseline by bin 30.
Deployment construction uses only trial boundaries, raw neural counts and the
same sparse target-direction labels as ordinary T4.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from streaming_calibration_exp.src.data.falcon_t4_features import t4_from_trial_sums
from tfpd_exploration.src.m2_hold_film_probe_v1 import core as probe_core
from tfpd_exploration.src.m2_hold_film_probe_v1 import plan as probe_plan
from tfpd_exploration.src.m2_hold_film_probe_v1.physical import (
    _atomic_json,
    install_film,
    score_arm,
)
from tfpd_exploration.src.m2_means_squeeze_v1 import plan as squeeze_plan


SCHEMA = "m2_movement_t4_ablation_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_movement_t4_ablation_v2"
HORIZON = 33
START_BIN = 5
STOP_BIN = 30
BIN_MS = 20
GPU_INDEX = 0


class MovementT4Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MovementT4Error(message)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if np.std(first) <= 1.0e-12 or np.std(second) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _trial_window_arrays(
    dataset: Any,
    session: str,
    *,
    start_bin: int = START_BIN,
    stop_bin: int = STOP_BIN,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neural = np.asarray(dataset.calib_neural[session], dtype=np.float32)
    starts = np.flatnonzero(np.asarray(dataset.calib_trial_change[session], dtype=bool))
    ends = np.r_[starts[1:], neural.shape[0]]
    angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float32)
    _require(starts.size >= horizon and angles.size >= horizon, f"{session} lacks M{horizon}")
    _require(0 <= start_bin < stop_bin, "invalid movement window")
    sums: list[np.ndarray] = []
    lengths: list[int] = []
    for trial, (start, end) in enumerate(zip(starts[:horizon], ends[:horizon])):
        available = int(end - start)
        _require(available >= stop_bin, f"{session} trial {trial} has only {available} bins")
        chunk = neural[int(start) + start_bin : int(start) + stop_bin]
        _require(chunk.shape == (stop_bin - start_bin, probe_plan.CHANNELS), "window shape drift")
        sums.append(chunk.sum(axis=0, dtype=np.float64))
        lengths.append(stop_bin - start_bin)
    return (
        np.ascontiguousarray(sums, dtype=np.float32),
        np.asarray(lengths, dtype=np.int64),
        np.ascontiguousarray(angles[:horizon], dtype=np.float32),
    )


def _raw_t4(dataset: Any, session: str, indices: np.ndarray | None = None) -> np.ndarray:
    sums, lengths, angles = _trial_window_arrays(dataset, session)
    if indices is not None:
        sums, lengths, angles = sums[indices], lengths[indices], angles[indices]
    return t4_from_trial_sums(sums, lengths, angles, source=f"{session}:bins[{START_BIN}:{STOP_BIN}]")


def _fit_source_normalizer(train_dataset: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sessions = sorted(train_dataset.calib_trialized_neural_features)
    _require(len(sessions) == probe_plan.EXPECTED_WITHIN_SESSIONS, "held-in session count drift")
    rows = {session: _raw_t4(train_dataset, session) for session in sessions}
    joined = np.concatenate([rows[session] for session in sessions], axis=0)
    mean = joined.mean(axis=0).astype(np.float32)
    std = joined.std(axis=0).astype(np.float32)
    std[std <= 1.0e-6] = 1.0
    return mean, std, {
        "train_sessions": sessions,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "raw_t4_sha256": {session: probe_core.array_sha256(rows[session]) for session in sessions},
    }


class _CarrierOverride:
    def __init__(self, base: Any, side: dict[str, np.ndarray]):
        self._base = base
        self._side = side

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _native_t4_side_features(self, session_name: str, start_trial_idx: int, calib_n_trials: int) -> np.ndarray:
        _require(int(start_trial_idx) == 0 and int(calib_n_trials) == HORIZON, "MOVE-T4 supports first M33 only")
        return self._side[session_name]


def _override(dataset: Any, mean: np.ndarray, std: np.ndarray) -> tuple[_CarrierOverride, dict[str, str]]:
    side: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for session in sorted(dataset.calib_trialized_neural_features):
        raw = _raw_t4(dataset, session)
        values = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
        _require(values.shape == (probe_plan.CHANNELS, 4), "MOVE-T4 side shape drift")
        side[session] = values
        hashes[session] = probe_core.array_sha256(values)
    return _CarrierOverride(dataset, side), hashes


def _reliability(train_dataset: Any) -> dict[str, Any]:
    records: dict[str, Any] = {}
    names = ("a", "c", "m", "b")
    for session in sorted(train_dataset.calib_trialized_neural_features):
        angles = np.asarray(train_dataset.calib_trial_target_angles[session], dtype=np.float32)[:HORIZON]
        directional = np.flatnonzero(np.isfinite(angles))
        _require(directional.size >= 6, f"{session} lacks directional split-half support")
        even = directional[::2]
        odd = directional[1::2]
        first, second = _raw_t4(train_dataset, session, even), _raw_t4(train_dataset, session, odd)
        records[session] = {name: _pearson(first[:, i], second[:, i]) for i, name in enumerate(names)}
    return {
        "per_session": records,
        "equal_session_mean": {
            name: float(np.nanmean([records[session][name] for session in records])) for name in names
        },
    }


def execute(repo_root: Path, *, batch_size: int = 1024) -> dict[str, Any]:
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "PYTHONNOUSERSITE=1 required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "CUDA_VISIBLE_DEVICES must be 0")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    root = repo_root / RESULT_ROOT_RELATIVE
    _require(not root.exists(), f"immutable result root exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    _atomic_json(root / "attempt.json", {"schema": SCHEMA + "_attempt", "status": "STARTED"})
    _require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == probe_plan.CHECKPOINT_SHA256, "checkpoint drift")
    _require(metadata["normalization_sha256"] == probe_plan.NORMALIZATION_SHA256, "normalizer drift")
    student = model.student.to(device)
    datasets = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    _require(all(value is not None for value in datasets.values()), "dataset missing")

    mean, std, normalizer = _fit_source_normalizer(datasets["within_post30"])
    move_train, train_hashes = _override(datasets["within_post30"], mean, std)
    move_external, external_hashes = _override(datasets["external_official_query"], mean, std)
    move_datasets = {"within_post30": move_train, "external_official_query": move_external}

    # Zero-init FiLM is used only as an algebraically identical 8-D-side wrapper
    # so score_arm can share the exact scorer with the content ablation.
    install_film(student, device, film_input="t4_plus_contrast", seed=42)
    zero = np.zeros(4, dtype=np.float32)
    whole = score_arm(
        student=student, datasets=datasets, device=device, batch_size=batch_size,
        shuffle=False, arm="WHOLE-T4", horizon=HORIZON, t4_mode="native", contrast_mask=zero,
    )
    whole_mean = float(whole["summaries"]["external_official_query"]["equal_session_mean"])
    _require(abs(whole_mean - squeeze_plan.P0_M33_EXTERNAL) <= 1.0e-6, "WHOLE-T4 anchor drift")
    move = score_arm(
        student=student, datasets=move_datasets, device=device, batch_size=batch_size,
        shuffle=False, arm="MOVE-T4", horizon=HORIZON, t4_mode="native", contrast_mask=zero,
    )

    contrasts = {
        surface: probe_core.paired_contrast(
            move["session_maps"][surface], whole["session_maps"][surface]
        )
        for surface in datasets
    }
    result = {
        "schema": SCHEMA,
        "status": "TERMINAL",
        "window": {
            "start_bin_inclusive": START_BIN,
            "stop_bin_exclusive": STOP_BIN,
            "start_ms": START_BIN * BIN_MS,
            "stop_ms": STOP_BIN * BIN_MS,
            "selection_authority": "seven-held-in-source-only speed peak at bin14; near-baseline by bin30; frozen before decoder score",
            "deployment_inputs": ["raw calibration neural counts", "trial boundaries", "sparse target direction"],
            "dense_velocity_used_at_deployment": False,
        },
        "normalizer": normalizer,
        "reliability": _reliability(datasets["within_post30"]),
        "whole": {"summaries": whole["summaries"]},
        "move": {"summaries": move["summaries"]},
        "move_minus_whole": contrasts,
        "side_sha256": {"within_post30": train_hashes, "external_official_query": external_hashes},
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "old_normalization_sha256": metadata["normalization_sha256"],
        "elapsed_seconds": float(time.monotonic() - started),
        "evalai_push": False,
    }
    _atomic_json(root / "score.json", result)
    _atomic_json(root / "terminal.json", {"schema": SCHEMA, "status": "TERMINAL", "evalai_push": False})
    return result


def publish_failure(repo_root: Path, exc: BaseException) -> None:
    root = repo_root / RESULT_ROOT_RELATIVE
    root.mkdir(parents=True, exist_ok=True)
    path = root / "failure.json"
    if not path.exists():
        _atomic_json(path, {"schema": SCHEMA + "_failure", "status": "FAILED", "type": type(exc).__name__, "message": str(exc)})
