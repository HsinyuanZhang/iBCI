"""Build the H1 EP-FILM cached-identity EvalAI deployment payload.

Per session tag the payload caches the public exactly-M3 support (activity,
carrier, profile) and one MAT7 readout fitted from that session's three
labelled calibration trials **with the EP-FILM arm** — the same build-time
computation the V3 local in-sample evaluation used for the identity itself.
The runtime decoder consumes only the cached payload; no hidden query label
is read, no optimizer step is taken, and no model update happens.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from h1_calibration_profile_film_v1.core import profile_from_support, require
from h1_m3_crossrecord_joint_v1.core import decode_with_identity, window_batch
from h1_support_resampled_postpool_v1.core import native_identity

from .evaluate import (
    _publish,
    _sha256_file,
    _utc_now,
    _verify_immutable,
    build_rows,
    load_all_source_model,
    rebuild_all_source_plan,
)
from .insample import RECEIPT_NAME as INSAMPLE_RECEIPT_NAME, _load_v3_checkpoint, _predict_ep
from .plan import (
    BATCH_SIZE,
    DATA_RELATIVE,
    M3RC_CALIBRATION_AUTHORITY_RELATIVE,
    M3RC_CALIBRATION_AUTHORITY_SHA256,
    M3RC_PACKAGE_RELATIVE,
    M3RC_PACKAGE_SHA256,
    PREDICTION_DIVISOR,
    RESULT_ROOT_RELATIVE,
    SCHEMA,
)

PACKAGE_SCHEMA = "h1_epfilm_evalai_package_v1"
PAYLOAD_RELATIVE = "SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/artifacts/h1_epfilm_evalai_v1"
M3RC_READOUT_SELECTION_RELATIVE = (
    "tfpd_exploration/h1_series_20260830/results/"
    "h1_m3_readout_calibration_evalai_package_v1/readout_selection.json"
)
M3RC_READOUT_SELECTION_SHA256 = "08169c6c8e3ead47550dd3426930498a45ba7c789c13aec83422f76e24177e18"
READOUT_FAMILY = "MAT7"
READOUT_RIDGE = 0.0


def _mapping_payload(mapping: Any) -> dict[str, Any]:
    return {
        "family": mapping.family,
        "ridge": mapping.ridge,
        "p_mean": mapping.p_mean,
        "p_scale": mapping.p_scale,
        "y_mean": mapping.y_mean,
        "y_scale": mapping.y_scale,
        "weight": mapping.weight,
        "intercept": mapping.intercept,
    }


def _calibration_mask(record: Any, values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(record.eval_mask, dtype=bool) & np.isin(
        np.asarray(record.trial_num), np.asarray(values, dtype=np.float64)
    )


def _film_decode_stream(net: Any, film: Any, neural: np.ndarray, endpoints: np.ndarray,
                         activity: Any, carrier: Any, profile: Any, *, device: str) -> np.ndarray:
    import torch
    from h1_calibration_profile_film_v1.core import film_identity

    with torch.inference_mode():
        identity = film_identity(net, activity, carrier, profile, film, late=False)
        output = np.empty((endpoints.size, 7), dtype=np.float32)
        for offset in range(0, endpoints.size, BATCH_SIZE):
            selected = endpoints[offset : offset + BATCH_SIZE]
            batch = torch.as_tensor(window_batch(neural, selected), dtype=torch.float32, device=device)
            prediction = decode_with_identity(net, batch, identity.expand(batch.shape[0], -1, -1))
            output[offset : offset + selected.size] = prediction[:, -1, :].cpu().numpy() / PREDICTION_DIVISOR
    return output


def build_payload(repo_root: Path, legacy_root: Path, *, device: str) -> dict[str, Any]:
    import torch
    from src.data.h1_m4_eb_pilot import (
        array_sha256,
        fit_deployment_carrier,
        index_heldin_calib,
        interpolate_trial_identity,
        load_record,
    )
    from src.data.h1_cal_aug_all_source_heldout_v1 import H1_HELDOUT_SESSIONS, index_heldout_calib
    from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
        HELDIN_SESSION_TO_FALCON_KEY,
        HELDOUT_SESSION_TO_FALCON_KEY,
    )
    from src.h1_hc_date_lodo_regen_v1 import model_config
    from src.h1_m4_cce_contract import NORMALIZER_FLOOR, state_hash
    from h1_m3_readout_calibration_v1.core import apply as readout_apply
    from h1_m3_readout_calibration_v1.core import fit as readout_fit
    from h1_m3_readout_calibration_v1.package import _load_heldout
    from h1_cross_record_postpool_v1.core import array_sha256 as cross_array_sha256
    from h1_cross_record_postpool_v1.evaluate import _r2
    from falcon_challenge.config import FalconConfig, FalconTask

    root = Path(repo_root).resolve()
    started_utc = _utc_now()
    wall_start = time.perf_counter()

    result_root = root / RESULT_ROOT_RELATIVE
    payload_path = root / PAYLOAD_RELATIVE
    artifact_root = root / ARTIFACT_ROOT_RELATIVE
    require(not payload_path.exists(), "EP-FILM deployment payload already exists; no rebuild")
    require(not (artifact_root / "calibration_authority.json").exists(),
            "EP-FILM calibration authority already exists; no rebuild")

    film, checkpoint_expected, checkpoint_authority = _load_v3_checkpoint(root)
    net, model_state, model_authority = load_all_source_model(root, device=device)
    plan, s_src, plan_authority = rebuild_all_source_plan(root, legacy_root)
    _verify_immutable(root / M3RC_PACKAGE_RELATIVE, M3RC_PACKAGE_SHA256)
    _verify_immutable(root / M3RC_CALIBRATION_AUTHORITY_RELATIVE, M3RC_CALIBRATION_AUTHORITY_SHA256)
    readout_selection_digest = _verify_immutable(root / M3RC_READOUT_SELECTION_RELATIVE,
                                                  M3RC_READOUT_SELECTION_SHA256)
    readout_selection = json.loads((root / M3RC_READOUT_SELECTION_RELATIVE).read_text(encoding="utf-8"))
    require(
        readout_selection.get("schema") == "h1_m3_readout_calibration_evalai_package_v1_selection"
        and readout_selection.get("selected", {}).get("family") == READOUT_FAMILY
        and float(readout_selection.get("selected", {}).get("ridge")) == READOUT_RIDGE,
        "M3RC readout selection authority drift",
    )

    insample_path = result_root / INSAMPLE_RECEIPT_NAME
    insample_digest = _sha256_file(insample_path)
    require((result_root / f"{INSAMPLE_RECEIPT_NAME}.sha256").read_text(encoding="ascii")
            == f"{insample_digest}  {INSAMPLE_RECEIPT_NAME}\n", "in-sample receipt sidecar drift")
    insample = json.loads(insample_path.read_text(encoding="utf-8"))
    require(insample.get("in_sample_evaluation") is True, "in-sample receipt flag drift")
    insample_predictions = {
        str(row["session"]): str(row["prediction_sha256"]["EP-FILM"]) for row in insample["sessions"]
    }

    film = film.to(torch.device(device))
    film.eval()
    for parameter in film.parameters():
        parameter.requires_grad_(False)
    film_state = state_hash(film.state_dict())
    require(film_state == checkpoint_expected["film_state_sha256"], "FiLM state drift after device move")

    data_root = root / DATA_RELATIVE
    heldin_paths = index_heldin_calib(data_root)
    heldout_paths = index_heldout_calib(data_root)
    rows = build_rows(root, plan, s_src)
    rows_by_session = {str(row["session"]): row for row in rows}

    config = FalconConfig(task=FalconTask.h1)
    public_rows: list[dict[str, Any]] = []
    sessions: dict[str, dict[str, Any]] = {}
    heldin_reference: dict[str, Any] = {}
    opened = {"heldin_calibration": 0, "heldout_calibration": 0}
    for session, key in list(HELDIN_SESSION_TO_FALCON_KEY) + list(HELDOUT_SESSION_TO_FALCON_KEY):
        session = str(session)
        heldout = session in H1_HELDOUT_SESSIONS
        if heldout:
            record = _load_heldout(heldout_paths[session])
            stem = f"sub-HumanPitt-held-out-calib_{session}"
            opened["heldout_calibration"] += 1
        else:
            record = load_record(heldin_paths[session])
            stem = f"sub-HumanPitt-held-in-minival_{session}"
            opened["heldin_calibration"] += 1
        require(config.hash_dataset(stem) == key, f"{session}: falcon key mapping drift for {stem}")
        values = tuple(float(value) for value in record.trial_values[:3])
        require(len(values) == 3 and len(set(values)) == 3, f"{session}: exactly-M3 support drift")
        activity = np.ascontiguousarray(
            np.stack([interpolate_trial_identity(record, value) for value in values]), dtype=np.float32
        )
        carrier = np.ascontiguousarray(
            fit_deployment_carrier(record, plan, values)["carrier"] / max(float(s_src), NORMALIZER_FLOOR),
            dtype=np.float32,
        )
        profile, profile_evidence = profile_from_support(record, values)
        profile = np.ascontiguousarray(profile, dtype=np.float32)
        require(activity.shape == (3, 1024, 176) and carrier.shape == (176, 4) and profile.shape == (176, 4),
                f"{session}: payload support geometry drift")

        if not heldout:
            bank_public = rows_by_session[session]["public"]["support_bank"][0]
            require(cross_array_sha256(activity) == bank_public["activity_sha256"],
                    f"{session}: payload activity drift vs training row")
            require(cross_array_sha256(carrier) == bank_public["carrier_sha256"],
                    f"{session}: payload carrier drift vs training row")
            require(cross_array_sha256(profile) == bank_public["task_profile_sha256"],
                    f"{session}: payload profile drift vs training row")

        activity_t = torch.as_tensor(activity, dtype=torch.float32, device=device).unsqueeze(0)
        carrier_t = torch.as_tensor(carrier, dtype=torch.float32, device=device).unsqueeze(0)
        profile_t = torch.as_tensor(profile, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.inference_mode():
            zero_identity = native_identity(net, activity_t, carrier_t)
            require(tuple(zero_identity.shape) == (1, 176, 700), f"{session}: native identity geometry drift")

        mask = _calibration_mask(record, values)
        endpoints = np.flatnonzero(mask).astype(np.int64)
        require(endpoints.size >= 8, f"{session}: M3 calibration fit surface is too small")
        calibration_prediction = _film_decode_stream(
            net, film, np.asarray(record.neural, dtype=np.float32), endpoints,
            activity_t, carrier_t, profile_t, device=device,
        )
        calibration_target = np.ascontiguousarray(
            np.asarray(record.velocity, dtype=np.float32)[endpoints], dtype=np.float32
        )
        readout = readout_fit(
            calibration_prediction.astype(np.float64), calibration_target.astype(np.float64),
            family=READOUT_FAMILY, ridge=READOUT_RIDGE,
        )
        entry = {
            "session": session,
            "identity": activity,
            "carrier": carrier,
            "profile": profile,
            "calibration_trials": list(values),
            "readout_calibration_trials": list(values),
            "readout": _mapping_payload(readout),
        }
        sessions[key] = entry
        row_public = {
            "session": session,
            "falcon_key": key,
            "scope": "held-out-calib" if heldout else "held-in-calib",
            "nwb_sha256": str(record.input_sha256),
            "calibration_trials": list(values),
            "calibration_bins": int(endpoints.size),
            "identity_sha256": array_sha256(activity),
            "carrier_sha256": array_sha256(carrier),
            "profile_sha256": array_sha256(profile),
            "task_profile": {
                "low_speed_quantile": profile_evidence["low_speed_quantile"],
                "high_speed_quantile": profile_evidence["high_speed_quantile"],
                "low_state_bins": profile_evidence["low_state_bins"],
                "high_state_bins": profile_evidence["high_state_bins"],
                "total_bins": profile_evidence["total_bins"],
            },
            "calibration_prediction_sha256": array_sha256(calibration_prediction),
            "calibration_target_sha256": array_sha256(calibration_target),
            "readout": {
                "family": readout.family,
                "ridge": readout.ridge,
                "array_sha256": {
                    name: array_sha256(np.asarray(getattr(readout, name)))
                    for name in ("p_mean", "p_scale", "y_mean", "y_scale", "weight", "intercept")
                },
            },
        }
        if not heldout:
            row = rows_by_session[session]
            raw_prediction, raw_r2 = _predict_ep(net, film, row, device=device)
            raw_sha = array_sha256(raw_prediction)
            prediction_matches_insample = raw_sha == insample_predictions[session]
            if str(device).startswith("cuda"):
                require(prediction_matches_insample,
                        f"{session}: EP-FILM minival prediction drift vs sealed in-sample receipt")
            target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
            corrected = readout_apply(readout, raw_prediction)
            row_public["deployment_reference"] = {
                "raw_ep_film_r2": raw_r2,
                "raw_prediction_sha256": raw_sha,
                "raw_prediction_sha256_matches_insample_receipt": prediction_matches_insample,
                "readout_ep_film_r2": _r2(corrected, target),
                "windows": int(np.asarray(row["endpoints"]).size),
            }
            heldin_reference[session] = row_public["deployment_reference"]
        public_rows.append(row_public)

    require(len(sessions) == 27 and opened["heldin_calibration"] == 13 and opened["heldout_calibration"] == 14,
            "27-session payload roster drift")

    training_receipt_sha = checkpoint_authority["training_receipt_sha256"]
    film_parameter_count = sum(parameter.numel() for parameter in film.parameters())
    require(film_parameter_count == 648, "packaged FiLM parameter count drift")
    package = {
        "schema": PACKAGE_SCHEMA,
        "task": "h1",
        "state_dict": {name: value.detach().cpu() for name, value in net.state_dict().items()},
        "model_kwargs": model_config()["model_kwargs"],
        "model_state_sha256": model_state,
        "checkpoint_sha256": model_authority["checkpoint_sha256"],
        "film": {
            "context_dim": 8,
            "rank": 8,
            "hidden_dim": 64,
            "parameter_count": film_parameter_count,
            "state_dict": {name: value.detach().cpu() for name, value in film.state_dict().items()},
        },
        "film_state_sha256": film_state,
        "film_checkpoint_sha256": checkpoint_expected["sha256"],
        "film_checkpoint_film_state_sha256": checkpoint_expected["film_state_sha256"],
        "v3_training_receipt_sha256": training_receipt_sha,
        "v3_insample_receipt_sha256": insample_digest,
        "source_authority_sha256": plan_authority["sealed_plan_json_sha256"],
        "readout_selection_sha256": readout_selection_digest,
        "window_size": 700,
        "prediction_divisor": PREDICTION_DIVISOR,
        "calibration_trials": 3,
        "readout_family": READOUT_FAMILY,
        "readout_ridge": READOUT_RIDGE,
        "sessions": sessions,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_name(f".{payload_path.name}.tmp")
    torch.save(package, temporary)
    temporary.replace(payload_path)
    import os

    os.chmod(payload_path, 0o444)
    payload_sha = _sha256_file(payload_path)
    side = payload_path.with_name(payload_path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{payload_sha}  {payload_path.name}\n")
    os.chmod(side, 0o444)

    artifact_root.mkdir(parents=True, exist_ok=True)
    authority_sha = _publish(artifact_root / "calibration_authority.json", {
        "schema": f"{SCHEMA}_calibration_authority",
        "status": "PASS_EXACTLY_M3_27_SESSION_EP_FILM_CALIBRATION",
        "payload_sha256": payload_sha,
        "payload_relative": PAYLOAD_RELATIVE,
        "readout_selection_sha256": readout_selection_digest,
        "official_calibration_trials": 3,
        "sessions": public_rows,
        "heldin_calibration_opened": opened["heldin_calibration"],
        "heldout_calibration_opened": opened["heldout_calibration"],
        "hidden_test_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    })
    return {
        "payload_relative": PAYLOAD_RELATIVE,
        "payload_sha256": payload_sha,
        "calibration_authority_relative": f"{ARTIFACT_ROOT_RELATIVE}/calibration_authority.json",
        "calibration_authority_sha256": authority_sha,
        "readout_selection_sha256": readout_selection_digest,
        "film_state_sha256": film_state,
        "film_checkpoint_sha256": checkpoint_expected["sha256"],
        "checkpoint_sha256": model_authority["checkpoint_sha256"],
        "model_state_sha256": model_state,
        "v3_training_receipt_sha256": training_receipt_sha,
        "v3_insample_receipt_sha256": insample_digest,
        "heldin_deployment_reference": heldin_reference,
        "wall_time_seconds": time.perf_counter() - wall_start,
        "started_at_utc": started_utc,
        "finished_at_utc": _utc_now(),
    }


__all__ = ("build_payload",)
