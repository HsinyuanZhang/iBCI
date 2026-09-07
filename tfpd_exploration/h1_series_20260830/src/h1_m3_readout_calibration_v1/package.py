"""Build the H1 exactly-M3 readout-calibrated EvalAI payload.

The builder deliberately imports the sealed all-source deployment producer
from a pinned checkout.  That producer is the authority that generated the
existing official C1 submission; current evolving H1 source files are not
silently substituted for it.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .core import ReadoutMap, fit


SCHEMA = "h1_m3_readout_calibration_evalai_package_v1"
STATUS = "COMPLETE_H1_M3RC_EVALAI_PACKAGE_LOCAL_NO_SUBMISSION"
LEGACY_HEAD = "5dd9bb4a7377a5431b7dbac4f1378e529130eb1a"
CHECKPOINT_SHA256 = "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06"
MODEL_STATE_SHA256 = "bdaf7dbcbae75ea307f20356aaf80066586f7d9afa273712a5e34708b903eb85"
SOURCE_AUTHORITY_SHA256 = "8ea4bb1174c00ab713843cd7561562d43f81509eaaea6ea12ee80cd4eba95de7"
SOURCE_SELECTION_SHA256 = "3a9f59f75da95aab056850f70e4b5c47afc093125e02e789f0c2f7db59ce6067"
TRANSFORM_SHA256 = "1c566312152d0203b282fd62a415694d9ddf5845a0208c291e4240e2b9b3ccd7"
SOURCE_NORMALIZER = 6.8260113140959355e-06
READOUT_FAMILY = "MAT7"
READOUT_RIDGE = 0.0
EXPECTED_ARRAY_SHA256 = {
    "mean": "f2958a2f71baf0f9d556bc4fbbbad3719a6e90fc51a725bd83b3ce7d6e81a8bd",
    "scale": "afb39a02e02ca0983e82f25878c78fea0d47a36c9b831aa718235c68a6c7e139",
    "pcs": "20a47d5703eab947adc39b079f293612de8c6fd8f80944aad0efa07ff10da5ff",
    "U": "4dbe4443b78a34809c8b252fcd58dbafd4ebad59dece867042e876d093c632b3",
    "mu": "6c6daed122f958114752ed981cc6724bdc8f8a07daaacb238749d927350bb67f",
}


class PackageError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _publish(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(data).hexdigest()
    os.chmod(path, 0o444)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(side, 0o444)
    return digest


def _publish_json(path: Path, value: Mapping[str, Any]) -> str:
    return _publish(path, _canonical(value))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head(path: Path) -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _mapping_payload(model: ReadoutMap) -> dict[str, Any]:
    return {
        "family": model.family,
        "ridge": model.ridge,
        "p_mean": model.p_mean,
        "p_scale": model.p_scale,
        "y_mean": model.y_mean,
        "y_scale": model.y_scale,
        "weight": model.weight,
        "intercept": model.intercept,
    }


def _mapping_public(model: ReadoutMap, array_sha256: Any) -> dict[str, Any]:
    return {
        "family": model.family,
        "ridge": model.ridge,
        "array_sha256": {
            name: array_sha256(getattr(model, name))
            for name in ("p_mean", "p_scale", "y_mean", "y_scale", "weight", "intercept")
        },
    }


def _load_heldout(path: Path) -> Any:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO
    from src.data.h1_m4_eb_pilot import (
        EXPECTED_NEURONS,
        H1PilotRecord,
        _trial_blocks,
        session_date,
        session_from_path,
    )
    from src.h1_m4_cce_contract import sha256_file

    resolved = path.resolve()
    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64 = np.asarray(neural, np.float64)
    targets64 = np.asarray(velocity, np.float64)
    spikes = spikes64.astype(np.float32)
    targets = targets64.astype(np.float32)
    changes = np.asarray(trial_change, bool).reshape(-1)
    mask = np.asarray(eval_mask, bool).reshape(-1)
    require(spikes.ndim == 2 and spikes.shape[1] == EXPECTED_NEURONS, "held-out neural shape drift")
    require(targets.shape == (len(spikes), 7), "held-out target shape drift")
    ordered = trial_num[mask & np.isfinite(trial_num)]
    require(ordered.size > 0 and not np.any(np.diff(ordered) < 0), "held-out TrialNum order drift")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    require(len(values) == 3, "official H1 held-out calibration must contain exactly three trials")
    trials = tuple(_trial_blocks(value, spikes64, targets64, mask, trial_num) for value in values)
    name = session_from_path(resolved)
    return H1PilotRecord(
        name,
        session_date(name),
        resolved,
        sha256_file(resolved),
        spikes,
        targets,
        changes,
        mask,
        trial_num,
        tuple(values),
        trials,
    )


def execute(
    *,
    repo_root: Path,
    legacy_root: Path,
    result_root: Path,
    package_path: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Build one frozen 27-session package; never opens hidden evaluation data."""
    import torch
    from h1_cross_record_postpool_v1.evaluate import _load_minival
    from h1_m3_crossrecord_joint_v1.evaluate import _predict, _session_row
    from h1_m3_readout_calibration_v1.evaluate import _base_pair, _calibration_row, _select_source
    from src.data.h1_cal_aug_all_source_heldout_v1 import H1_HELDOUT_SESSIONS, index_heldout_calib
    from src.data.h1_m4_eb_pilot import (
        array_sha256,
        fit_deployment_carrier,
        fit_frozen_carrier,
        index_heldin_calib,
        interpolate_trial_identity,
        load_record,
    )
    from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
        HELDIN_SESSION_TO_FALCON_KEY,
        HELDOUT_SESSION_TO_FALCON_KEY,
    )
    from src.h1_cal_aug_all_source_m3_deployment_v1_exec import _load_all_source_records
    from src.h1_hc_date_lodo_regen_v1 import _legal_starts, _make_final_plan, _new_model, model_config
    from src.h1_m4_cce_contract import state_hash

    root = repo_root.resolve()
    legacy = legacy_root.resolve()
    result = result_root.resolve()
    package_target = package_path.resolve()
    require(_git_head(legacy) == LEGACY_HEAD, "sealed all-source producer checkout drift")
    require(not result.exists(), "H1-M3RC package result root is not fresh")
    require(not package_target.exists(), "H1-M3RC package target is not fresh")
    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "H1-M3RC package build requires isolated logical GPU0")

    closure_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("core.py").resolve(),
        root / "SPINT-main/third_party/falcon_challenge/h1_m3rc_spint_decoder.py",
        root / "SPINT-main/third_party/falcon_challenge/h1_m3rc_spint_sample.py",
        root / "SPINT-main/third_party/falcon_challenge/h1_m3rc_spint_sample.Dockerfile",
        root / "tfpd_exploration/h1_series_20260830/docs/DESIGN_H1_M3_READOUT_CALIBRATION_V1_20260903.md",
        root / "tfpd_exploration/h1_series_20260830/docs/RESULT_H1_M3_READOUT_CALIBRATION_V1_20260903.md",
        root / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_M3RC_EVALAI_PACKAGE_V1_20260903.md",
        legacy / "SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1_exec.py",
        legacy / "SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1_contract.py",
        legacy / "SPINT-main/src/h1_hc_date_lodo_regen_v1.py",
        legacy / "SPINT-main/src/data/h1_m4_eb_pilot.py",
        root / "tfpd_exploration/h1_series_20260830/artifacts/h1_c1_all_source_epoch49_v1/epoch_049.ckpt",
    )
    closure = {str(path): _sha(path) for path in closure_paths}
    result.mkdir(parents=True)
    attempt = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_H1_DATA_AND_CUDA",
        "created_at_utc": _utc(),
        "official_calibration_trials": 3,
        "hidden_test_opened": False,
        "evalai_submissions": 0,
        "closure": closure,
    }
    attempt_sha = _publish_json(result / "attempt.json", attempt)
    try:
        data = root / "SPINT-main/data/000954"
        checkpoint_path = root / "tfpd_exploration/h1_series_20260830/artifacts/h1_c1_all_source_epoch49_v1/epoch_049.ckpt"
        require(_sha(checkpoint_path) == CHECKPOINT_SHA256, "all-source C1 checkpoint drift")
        records = _load_all_source_records(data)
        plan = _make_final_plan(
            records,
            "h1_all_source_13",
            {"q": 12, "lambda": 10.0},
            SOURCE_SELECTION_SHA256,
        )
        actual_array = {name: array_sha256(getattr(plan, name)) for name in EXPECTED_ARRAY_SHA256}
        require(actual_array == EXPECTED_ARRAY_SHA256, "sealed all-source plan array drift")
        require(plan.transform_sha256 == TRANSFORM_SHA256, "sealed all-source transform drift")
        source_carriers = []
        for name, record in records.items():
            for start in _legal_starts(record):
                support = tuple(float(x) for x in record.trial_values[start : start + 4])
                source_carriers.append(np.asarray(fit_frozen_carrier(record, plan, support)["carrier"], np.float64))
        s_src = float(np.sqrt(np.mean(np.square(np.stack(source_carriers), dtype=np.float64), dtype=np.float64)))
        require(s_src == SOURCE_NORMALIZER, "sealed source normalizer drift")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = _new_model(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        require(state_hash(model.state_dict()) == MODEL_STATE_SHA256, "all-source C1 state drift")
        heldin_paths = index_heldin_calib(data)
        source_pairs = []
        calibration_by_session: dict[str, dict[str, Any]] = {}
        for session in plan.source_sessions:
            record = load_record(heldin_paths[session])
            calibration = _calibration_row(record, plan, s_src)
            calibration_by_session[session] = calibration
            query = _session_row(
                record=record,
                minival=_load_minival(data, session),
                plan=plan,
                s_src=s_src,
            )
            source_pairs.append(_base_pair(model, calibration, query, device=device))
        selected, candidates = _select_source(source_pairs)
        require(selected["family"] == READOUT_FAMILY and float(selected["ridge"]) == READOUT_RIDGE,
                "all-source readout selection drift")
        selection_body = {
            "schema": f"{SCHEMA}_selection",
            "status": "SOURCE_SELECTED_BEFORE_HELDOUT_CALIBRATION_OPEN",
            "source_sessions": list(plan.source_sessions),
            "selected": selected,
            "candidate_rows": candidates,
            "heldout_calibration_opened": False,
            "hidden_test_opened": False,
        }
        readout_selection_sha = _publish_json(result / "readout_selection.json", selection_body)

        heldout_paths = index_heldout_calib(data)
        key_rows = HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY
        sessions: dict[str, dict[str, Any]] = {}
        public_rows = []
        source_pair_by_session = {str(pair["session"]): pair for pair in source_pairs}
        for session, key in key_rows:
            heldout = session in H1_HELDOUT_SESSIONS
            record = _load_heldout(heldout_paths[session]) if heldout else load_record(heldin_paths[session])
            values = tuple(float(value) for value in record.trial_values[:3])
            require(len(values) == 3, f"{session}: exactly-M3 support drift")
            calibration = calibration_by_session.get(session)
            if calibration is None:
                calibration = _calibration_row(record, plan, s_src)
            if session in source_pair_by_session:
                prediction = np.ascontiguousarray(source_pair_by_session[session]["calibration_prediction"], dtype=np.float32)
                target = np.ascontiguousarray(source_pair_by_session[session]["calibration_target"], dtype=np.float32)
            else:
                prediction, _ = _predict(model, calibration, device=device, alpha=None)
                target = np.ascontiguousarray(calibration["target_stream"][calibration["endpoints"]], dtype=np.float32)
            mapping = fit(prediction, target, family=READOUT_FAMILY, ridge=READOUT_RIDGE)
            identity = np.ascontiguousarray(
                np.stack([interpolate_trial_identity(record, value) for value in values]), dtype=np.float32
            )
            fitted = fit_deployment_carrier(record, plan, values)
            carrier = np.ascontiguousarray(np.asarray(fitted["carrier"], np.float64) / s_src, dtype=np.float32)
            require(identity.shape == (3, 1024, 176) and carrier.shape == (176, 4),
                    f"{session}: package support shape drift")
            sessions[key] = {
                "session": session,
                "identity": identity,
                "carrier": carrier,
                "calibration_trials": list(values),
                "readout_calibration_trials": list(values),
                "readout": _mapping_payload(mapping),
            }
            public_rows.append({
                "session": session,
                "falcon_key": key,
                "scope": "held-out-calib" if heldout else "held-in-calib",
                "nwb_sha256": record.input_sha256,
                "calibration_trials": list(values),
                "calibration_bins": int(target.shape[0]),
                "identity_sha256": array_sha256(identity),
                "carrier_sha256": array_sha256(carrier),
                "calibration_prediction_sha256": array_sha256(prediction),
                "calibration_target_sha256": array_sha256(target),
                "readout": _mapping_public(mapping, array_sha256),
            })
        require(len(sessions) == 27 and len(public_rows) == 27, "27-session package roster drift")
        calibration_sha = _publish_json(result / "calibration_authority.json", {
            "schema": f"{SCHEMA}_calibration_authority",
            "status": "PASS_EXACTLY_M3_27_SESSION_CALIBRATION",
            "source_selection_sha256": readout_selection_sha,
            "official_calibration_trials": 3,
            "sessions": public_rows,
            "heldin_calibration_opened": 13,
            "heldout_calibration_opened": 14,
            "hidden_test_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        })
        package = {
            "schema": SCHEMA,
            "task": "h1",
            "state_dict": checkpoint["state_dict"],
            "model_kwargs": model_config()["model_kwargs"],
            "model_state_sha256": MODEL_STATE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
            "readout_selection_sha256": readout_selection_sha,
            "calibration_authority_sha256": calibration_sha,
            "window_size": 700,
            "prediction_divisor": 20.0,
            "calibration_trials": 3,
            "readout_family": READOUT_FAMILY,
            "readout_ridge": READOUT_RIDGE,
            "sessions": sessions,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_updates": 0,
            "evalai_submissions": 0,
        }
        buffer = io.BytesIO()
        torch.save(package, buffer)
        package_sha = _publish(package_target, buffer.getvalue())
        package_manifest_sha = _publish_json(result / "package.json", {
            "schema": f"{SCHEMA}_package",
            "status": "PASS_H1_M3RC_PACKAGE_BUILT",
            "package_path": str(package_target),
            "package_sha256": package_sha,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "model_state_sha256": MODEL_STATE_SHA256,
            "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
            "readout_selection_sha256": readout_selection_sha,
            "calibration_authority_sha256": calibration_sha,
            "session_payloads": 27,
            "heldout_exactly_m3_payloads": 14,
            "hidden_test_opened": False,
            "evalai_submissions": 0,
        })
        terminal = {
            "schema": SCHEMA,
            "artifact": "terminal",
            "status": STATUS,
            "finished_at_utc": _utc(),
            "attempt_sha256": attempt_sha,
            "readout_selection_sha256": readout_selection_sha,
            "calibration_authority_sha256": calibration_sha,
            "package_manifest_sha256": package_manifest_sha,
            "package_sha256": package_sha,
            "official_calibration_trials": 3,
            "selected_readout": {"family": READOUT_FAMILY, "ridge": READOUT_RIDGE},
            "hidden_test_opened": False,
            "evalai_submissions": 0,
        }
        terminal["terminal_sha256"] = _publish_json(result / "terminal.json", terminal)
        return terminal
    except Exception as error:
        _publish_json(result / "failure.json", {
            "schema": SCHEMA,
            "artifact": "failure",
            "status": "FAILED_H1_M3RC_PACKAGE",
            "attempt_sha256": attempt_sha,
            "exception_type": type(error).__name__,
            "exception_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "hidden_test_opened": False,
            "evalai_submissions": 0,
        })
        raise


__all__ = ("PackageError", "SCHEMA", "STATUS", "execute")
