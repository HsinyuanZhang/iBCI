"""Five-fold cross-record frozen-weight H1 C1 screen."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import anchored_identity, array_sha256, identity_states, query_member, require, stream_window
from .plan import ARM_ORDER, BATCH_SIZE, CHUNK_LENGTH, DATE_ORDER, FAMILIES, G_VALUES, SCHEMA, arm_name, decide_oof, select_source_arm


DATA_RELATIVE = "SPINT-main/data/000954"
ARTIFACT_RELATIVE = "tfpd_exploration/h1_series_20260830/artifacts/h1_c1_date_lodo_epoch49_v1"


def _minival_path(data_root: Path, session: str) -> Path:
    directory = (data_root / "sub-HumanPitt-held-in-minival").resolve()
    require(directory.is_dir(), "held-in-minival directory missing")
    matches = tuple(directory.glob(f"*_{session}.nwb"))
    require(len(matches) == 1 and matches[0].is_file() and not matches[0].is_symlink(), f"{session}: minival path drift")
    return matches[0].resolve()


def _load_minival(data_root: Path, session: str) -> dict[str, Any]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    path = _minival_path(data_root, session)
    neural, target, trial_change, eval_mask = load_nwb(path, FalconTask.h1)
    neural = np.ascontiguousarray(neural, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    changes = np.asarray(trial_change, dtype=bool).reshape(-1)
    require(neural.ndim == target.ndim == 2 and neural.shape[1] == 176 and target.shape[1] == 7, "minival geometry drift")
    require(neural.shape[0] == target.shape[0] == mask.shape[0] == changes.shape[0], "minival aligned length drift")
    require(neural.shape[0] > CHUNK_LENGTH and np.isfinite(neural).all() and np.isfinite(target).all(), "minival values drift")
    endpoints = np.flatnonzero(mask).astype(np.int64)
    require(endpoints.size > 0, "minival eval surface empty")
    return {
        "path": path,
        "path_sha256": _sha256_file(path),
        "neural": neural,
        "target": target,
        "mask": mask,
        "trial_change": changes,
        "endpoints": endpoints,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_receipt(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    sidecar = path.with_name(path.name + ".sha256")
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(sidecar, 0o444)
    return digest


def _support(record: Any, plan: Any, s_src: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from src.data.h1_m4_eb_pilot import fit_deployment_carrier, interpolate_trial_identity
    from src.h1_m4_cce_contract import NORMALIZER_FLOOR

    values = tuple(float(value) for value in record.trial_values[:3])
    require(len(values) == 3 and len(set(values)) == 3, "calibration M3 support drift")
    activity = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, value) for value in values]), dtype=np.float32)
    carrier = np.ascontiguousarray(
        fit_deployment_carrier(record, plan, values)["carrier"] / max(float(s_src), NORMALIZER_FLOOR),
        dtype=np.float32,
    )
    return activity, carrier, {
        "support_trials": list(values),
        "activity_sha256": array_sha256(activity),
        "carrier_sha256": array_sha256(carrier),
        "calibration_input_sha256": record.input_sha256,
    }


def _decode(net: Any, neural: np.ndarray, endpoints: np.ndarray, identity: Any) -> np.ndarray:
    import torch
    from m1_h1_activity_headroom_v1.core import forward_with_cached_identity

    chosen_endpoints = np.asarray(endpoints, dtype=np.int64).reshape(-1)
    require(chosen_endpoints.size > 0, "decode surface is empty")
    output = np.empty((chosen_endpoints.size, 7), dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, chosen_endpoints.size, BATCH_SIZE):
            batch_endpoints = chosen_endpoints[offset : offset + BATCH_SIZE]
            windows = np.stack([stream_window(neural, int(endpoint)) for endpoint in batch_endpoints])
            value = forward_with_cached_identity(net, windows, identity)[:, -1, :]
            output[offset : offset + len(batch_endpoints)] = value.detach().cpu().numpy() / 20.0
    return output


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    from m1_h1_activity_headroom_v1.core import variance_weighted_r2
    return variance_weighted_r2(prediction, target)


def _score_session(*, net: Any, record: Any, minival: Mapping[str, Any], plan: Any, s_src: float, device: str) -> dict[str, Any]:
    activity, carrier, support_receipt = _support(record, plan, s_src)
    member = query_member(minival["neural"])
    states = identity_states(net, activity, member, carrier, device=device)
    endpoints = np.asarray(minival["endpoints"], dtype=np.int64)
    targets = np.asarray(minival["target"], dtype=np.float32)[endpoints]
    after = endpoints >= CHUNK_LENGTH
    static_prediction = _decode(net, minival["neural"], endpoints, states["static"])
    predictions: dict[str, np.ndarray] = {"A-STATIC": static_prediction}
    for family in FAMILIES:
        candidate = states["native" if family == "RN" else "post"]
        for gate in G_VALUES:
            name = arm_name(family, gate)
            post_identity = anchored_identity(states["static"], candidate, gate)
            if gate == 0.0:
                predictions[name] = static_prediction
                continue
            prediction = static_prediction.copy()
            if bool(after.any()):
                prediction[after] = _decode(net, minival["neural"], endpoints[after], post_identity)
            predictions[name] = prediction
    rows: dict[str, Any] = {}
    for name in ARM_ORDER:
        prediction = predictions[name]
        full = _r2(prediction, targets)
        post = _r2(prediction[after], targets[after]) if int(after.sum()) >= 2 else None
        rows[name] = {
            "r2": full,
            "post_commit_r2": post,
            "prediction_sha256": array_sha256(prediction),
            "n_windows": int(len(endpoints)),
            "n_post_commit_windows": int(after.sum()),
        }
    require(rows["RN-G000"]["prediction_sha256"] == rows["A-STATIC"]["prediction_sha256"], "RN +0 static sentinel failed")
    require(rows["RP-G000"]["prediction_sha256"] == rows["A-STATIC"]["prediction_sha256"], "RP +0 static sentinel failed")
    return {
        "session": record.session_name,
        "date": record.date,
        "support": support_receipt,
        "minival_input_sha256": minival["path_sha256"],
        "query_neural_sha256": array_sha256(minival["neural"]),
        "target_sha256": array_sha256(targets),
        "endpoint_sha256": array_sha256(endpoints),
        "first_commit_exclusive": CHUNK_LENGTH,
        "query_member_sha256": array_sha256(member),
        "identity_sha256": {key: array_sha256(value.detach().cpu().numpy()) for key, value in states.items()},
        "arms": rows,
    }


def _date_means(session_rows: list[Mapping[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[str(row["date"])].append(row)
    result: dict[str, float] = {}
    for arm in ARM_ORDER:
        per_date = [sum(float(row["arms"][arm]["r2"]) for row in rows) / len(rows) for rows in grouped.values()]
        result[arm] = sum(per_date) / len(per_date)
    return result


def run_fold(repo_root: Path, outer_date: str, *, device: str, receipt_root: Path | None = None) -> dict[str, Any]:
    from h1_causal_activity_completion_v1.stage1 import ARTIFACT_RELATIVE as C1_ARTIFACT, C1_AUTHORITIES, _load_model, _load_plan
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve()
    require(outer_date in DATE_ORDER, "unknown outer date")
    directory = root / C1_ARTIFACT / outer_date
    authority = C1_AUTHORITIES[outer_date]
    plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    model, state_before, model_receipt = _load_model(directory, authority, outer_date, device)
    net = getattr(model, "net", model)
    data_root = root / DATA_RELATIVE
    indexed = index_heldin_calib(data_root)
    source_names = tuple(plan.source_sessions)
    require(source_names and all(name in indexed and not name.startswith(f"ses-{outer_date}") for name in source_names), "fold source roster drift")
    source_rows: list[dict[str, Any]] = []
    for session in source_names:
        record = load_record(indexed[session])
        minival = _load_minival(data_root, session)
        source_rows.append(_score_session(net=net, record=record, minival=minival, plan=plan, s_src=s_src, device=device))
    source_mean = _date_means(source_rows)
    selection = select_source_arm(source_mean)
    selection_sha256 = None
    if receipt_root is not None:
        selection_sha256 = _publish_receipt(
            Path(receipt_root) / f"selection_{outer_date}.json",
            {
                "schema": f"{SCHEMA}_selection",
                "status": "SELECTED_BEFORE_OUTER_DATE_OPEN",
                "outer_date": outer_date,
                "source_sessions": list(source_names),
                "source_arm_equal_date_mean_r2": source_mean,
                "selection": selection,
                "outer_date_calib_opened": False,
                "outer_date_minival_opened": False,
                "target_optimizer_steps": 0,
                "target_model_updates": 0,
            },
        )
    target_records = load_outer_date_target_records(data_root, outer_date=outer_date)
    target_rows: list[dict[str, Any]] = []
    for session, record in target_records.items():
        minival = _load_minival(data_root, session)
        target_rows.append(_score_session(net=net, record=record, minival=minival, plan=plan, s_src=s_src, device=device))
    selected = str(selection["arm"])
    target_static = sum(float(row["arms"]["A-STATIC"]["r2"]) for row in target_rows) / len(target_rows)
    target_selected = sum(float(row["arms"][selected]["r2"]) for row in target_rows) / len(target_rows)
    state_after = state_hash(model.state_dict())
    require(state_before == state_after, "frozen C1 model state changed")
    fold = {
        "outer_date": outer_date,
        "source_dates": sorted({str(row["date"]) for row in source_rows}),
        "source_sessions": list(source_names),
        "source_arm_equal_date_mean_r2": source_mean,
        "selection": selection,
        "selection_sha256": selection_sha256,
        "target_sessions": list(target_records),
        "target_static_equal_recording_r2": target_static,
        "target_selected_equal_recording_r2": target_selected,
        "selected_delta_vs_static": target_selected - target_static,
        "target_descriptive_rows": target_rows,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    if receipt_root is not None:
        fold["fold_sha256"] = _publish_receipt(Path(receipt_root) / f"fold_{outer_date}.json", fold)
    return fold


def run(repo_root: Path, *, device: str = "cuda:0", receipt_root: Path | None = None) -> dict[str, Any]:
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1, "screen requires isolated logical GPU0")
    folds = [run_fold(repo_root, date, device=device, receipt_root=receipt_root) for date in DATE_ORDER]
    decision = decide_oof(folds)
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_CROSS_RECORD_ANCHORED_POSTPOOL_SOURCE_OOF",
        "date_order": list(DATE_ORDER),
        "arm_order": list(ARM_ORDER),
        "folds": folds,
        "decision": decision,
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "run_fold")
