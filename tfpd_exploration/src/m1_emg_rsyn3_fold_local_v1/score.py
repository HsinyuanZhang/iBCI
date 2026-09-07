"""Paired Static / CDM-A scoring for the fold-local Stage-1 student."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from tfpd_exploration.src.m1_h1_activity_headroom_v1 import core as headroom_core
from tfpd_exploration.src.m1_heldin_heldout_gap_v1.physical import variance_weighted_last_bin_r2

from . import plan


class ScoreError(RuntimeError):
    """Fail closed for Stage-1 scoring."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


CDM_A_ARM = headroom_core.ActivityArm.ROLLING_FIXED_M
BATCH = 32
OUTPUTS = 16


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = plan.canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)})
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _cell_result(prediction: np.ndarray, target: np.ndarray, extra: Mapping[str, object]) -> dict[str, object]:
    _require(prediction.shape == target.shape and prediction.shape[1] == OUTPUTS, "score topology")
    r2 = variance_weighted_last_bin_r2(prediction, target)
    return {
        "metric": "variance_weighted_last_bin_r2",
        "last_bin_only": True,
        "eval_mode": True,
        "no_grad": True,
        "governing_r2": r2,
        "n_windows": int(prediction.shape[0]),
        "prediction_sha256": _array_digest(prediction),
        "target_sha256": _array_digest(target),
        "target_optimizer_steps": 0,
        "target_labels_used_only_for_metric": True,
        "decoder_r2_computed": True,
        **dict(extra),
    }


def output_trial_indices(dataset: Any, session_id: str) -> tuple[int, ...]:
    starts = np.asarray(dataset.trial_start_indices[session_id], dtype=np.int64)
    result: list[int] = []
    for session, start in dataset.window_indices:
        _require(session == session_id, "score session drift")
        result.append(int(np.searchsorted(starts, int(start) + plan.WINDOW_SIZE - 1, side="right") - 1))
    _require(len(result) == len(dataset), "window/trial map drift")
    return tuple(result)


def _expand_carrier(carrier: np.ndarray, batch: int) -> Any:
    import torch

    tensor = torch.as_tensor(np.ascontiguousarray(carrier), dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == 1 and batch > 1:
        tensor = tensor.expand(batch, -1, -1)
    return tensor


def score_static(model: Any, dataset: Any, *, session_id: str, carrier: np.ndarray, device: str) -> dict[str, object]:
    import torch

    session_trials = np.ascontiguousarray(
        dataset.calib_trialized_neural_features[session_id][: plan.SUPPORT_TRIALS], dtype=np.float32,
    )
    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    forwards = 0
    model.eval()
    with torch.no_grad():
        for start_row in range(0, len(dataset), BATCH):
            rows = tuple(range(start_row, min(start_row + BATCH, len(dataset))))
            xs, ys = [], []
            for row in rows:
                _session, start = dataset.window_indices[row]
                end = int(start) + plan.WINDOW_SIZE
                xs.append(dataset.neural_data[session_id][int(start):end])
                ys.append(dataset.covariate_data[session_id][end - 1])
            x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
            calibration = np.ascontiguousarray(np.stack([session_trials] * len(rows)), dtype=np.float32)
            carrier_b = _expand_carrier(carrier, len(rows)).to(device)
            output, _identity = model(
                torch.as_tensor(x, dtype=torch.float32, device=device),
                calib_trials=torch.as_tensor(calibration, dtype=torch.float32, device=device),
                side_features=None,
                carrier=carrier_b,
            )
            prediction[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(
                output[:, -1, :].detach().cpu().numpy(), dtype=np.float32,
            )
            target[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(np.stack(ys), dtype=np.float32)
            forwards += 1
    return _cell_result(prediction, target, {
        "forward_path": "static_m10_repeated_calibration",
        "forward_batches": forwards,
        "unique_activity_states": 1,
        "activity_cardinality": plan.SUPPORT_TRIALS,
        "carrier_held_fixed": True,
    })


def score_cdm_fifo(model: Any, dataset: Any, *, session_id: str, carrier: np.ndarray, device: str) -> dict[str, object]:
    import torch

    trials = np.ascontiguousarray(dataset.calib_trialized_neural_features[session_id], dtype=np.float32)
    output_trials = output_trial_indices(dataset, session_id)
    selections = tuple(
        headroom_core.selection_for_output_trial(
            CDM_A_ARM,
            output_trial_index=trial,
            total_trials=int(trials.shape[0]),
            support_trials=plan.SUPPORT_TRIALS,
        )
        for trial in output_trials
    )
    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    forwards = 0
    model.eval()
    with torch.no_grad():
        for selection, rows in headroom_core.grouped_indices(selections):
            selected = np.ascontiguousarray(trials[np.asarray(selection, dtype=np.int64)], dtype=np.float32)
            identity = model.compute_identity(
                torch.as_tensor(selected[None, ...], dtype=torch.float32, device=device),
                side_features=None,
            )
            for offset in range(0, len(rows), BATCH):
                batch_rows = rows[offset:offset + BATCH]
                xs, ys = [], []
                for row in batch_rows:
                    _session, start = dataset.window_indices[row]
                    end = int(start) + plan.WINDOW_SIZE
                    xs.append(dataset.neural_data[session_id][int(start):end])
                    ys.append(dataset.covariate_data[session_id][end - 1])
                x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
                identity_b = identity.expand(len(batch_rows), -1, -1) if identity.shape[0] == 1 else identity
                carrier_b = _expand_carrier(carrier, len(batch_rows)).to(device)
                output = model.decode_with_identity(
                    torch.as_tensor(x, dtype=torch.float32, device=device),
                    identity_b,
                    carrier=carrier_b,
                )
                prediction[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(
                    output[:, -1, :].detach().cpu().numpy(), dtype=np.float32,
                )
                target[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(
                    np.stack(ys), dtype=np.float32,
                )
                forwards += 1
    return _cell_result(prediction, target, {
        "forward_path": "cdm_activity_fifo_b3_identity",
        "forward_batches": forwards,
        "unique_activity_states": len(set(selections)),
        "activity_cardinality_min": min(map(len, selections)),
        "activity_cardinality_max": max(map(len, selections)),
        "causal": True,
        "label_free": True,
        "carrier_held_fixed": True,
        "cdm_arm": CDM_A_ARM.value,
    })
