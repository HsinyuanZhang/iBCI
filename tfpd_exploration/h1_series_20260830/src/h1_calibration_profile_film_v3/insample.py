"""Local in-sample evaluation of the V3 all-source EP-FILM checkpoint.

Every scored window comes from the 13 held-in sessions whose minival streams
the FiLM was trained on (stride-4 subsets), so this surface is explicitly
IN-SAMPLE: it is a deployment-surface smoke readout, never a generalization
estimate.  The EP-ZERO control is the parameter-free native identity of the
same frozen all-source substrate on the same windows.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from h1_calibration_profile_film_v1.core import build_film, film_identity, require
from h1_calibration_profile_film_v1.evaluate import _device_support_bank
from h1_m3_crossrecord_joint_v1.core import decode_with_identity, window_batch
from h1_support_resampled_postpool_v1.core import native_identity

from .evaluate import (
    _publish,
    _sha256_file,
    _utc_now,
    _verify_immutable,
    build_rows,
    cross_authority,
    load_all_source_model,
    rebuild_all_source_plan,
    verify_v2_root,
)
from .plan import (
    BATCH_SIZE,
    PREDICTION_DIVISOR,
    RESULT_ROOT_RELATIVE,
    SCHEMA,
    V2_ROOT_RELATIVE,
)

RECEIPT_NAME = "local_insample_eval.json"
OVERFIT_RATIO_THRESHOLD = 2.0


def _predict_ep(net: Any, film: Any | None, row: Mapping[str, Any], *, device: str) -> tuple[np.ndarray, float]:
    import torch
    from h1_cross_record_postpool_v1.evaluate import _r2

    net.eval()
    if film is not None:
        film.eval()
    activity, carrier, profile = _device_support_bank(row, device=device)[0]
    with torch.inference_mode():
        if film is None:
            identity = native_identity(net, activity, carrier)
        else:
            identity = film_identity(net, activity, carrier, profile, film, late=False)
        endpoints = np.asarray(row["endpoints"], dtype=np.int64)
        output = np.empty((endpoints.size, 7), dtype=np.float32)
        for offset in range(0, endpoints.size, BATCH_SIZE):
            selected = endpoints[offset : offset + BATCH_SIZE]
            neural = torch.as_tensor(window_batch(row["neural"], selected), dtype=torch.float32, device=device)
            prediction = decode_with_identity(net, neural, identity)[:, -1, :] / PREDICTION_DIVISOR
            output[offset : offset + selected.size] = prediction.cpu().numpy()
    target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
    return output, _r2(output, target)


def _load_v3_checkpoint(repo_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve() / RESULT_ROOT_RELATIVE
    training_path = root / "training_all_source.json"
    side_text = (training_path.with_name(training_path.name + ".sha256")).read_text(encoding="ascii")
    require(side_text.endswith("  training_all_source.json\n") and len(side_text.split("  ", 1)[0]) == 64,
            "V3 training receipt sidecar format drift")
    digest = _verify_immutable(training_path, side_text.split("  ", 1)[0].strip())
    training = json.loads(training_path.read_text(encoding="utf-8"))
    expected = training["checkpoints"]["EP-FILM"]
    require(
        training.get("schema") == f"{SCHEMA}_training"
        and training.get("status") == "ALL_SOURCE_EP_FILM_FROZEN_DEPLOYMENT_CANDIDATE"
        and expected.get("path") == "checkpoint_all_source_ep-film.pt"
        and training.get("training", {}).get("final_film_state_sha256", {}).get("EP-FILM")
        == expected.get("film_state_sha256"),
        "V3 training receipt checkpoint linkage drift",
    )
    path = root / str(expected["path"])
    observed = _verify_immutable(path, str(expected["sha256"]))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        payload.get("schema") == f"{SCHEMA}_film_checkpoint"
        and payload.get("arm") == "EP-FILM"
        and payload.get("train_scope") == "all_source_heldin"
        and payload.get("base_state_sha256") == expected["base_state_sha256"]
        and payload.get("film_state_sha256") == expected["film_state_sha256"],
        "V3 EP-FILM checkpoint metadata drift",
    )
    module = build_film()
    incompatible = module.load_state_dict(payload["state_dict"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, "V3 FiLM strict load drift")
    require(state_hash(module.state_dict()) == payload["film_state_sha256"] == expected["film_state_sha256"],
            "V3 FiLM checkpoint state digest drift")
    return module, dict(expected), {"training_receipt_sha256": digest, "checkpoint_file_sha256": observed}


def run_insample(repo_root: Path, legacy_root: Path, *, device: str,
                 preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import torch
    from src.h1_m4_cce_contract import array_sha256, state_hash

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "in-sample evaluation requires isolated logical GPU0")
    root = Path(repo_root).resolve()
    receipt_root = root / RESULT_ROOT_RELATIVE
    require(not (receipt_root / RECEIPT_NAME).exists(), "in-sample evaluation receipt already exists")
    started_utc = _utc_now()
    wall_start = time.perf_counter()

    film, checkpoint_expected, checkpoint_authority = _load_v3_checkpoint(root)
    v2_authority = verify_v2_root(root)
    v2_source_rows = v2_authority.pop("source_rows_by_lodo_date")
    net, model_state, model_authority = load_all_source_model(root, device=device)
    plan, s_src, plan_authority = rebuild_all_source_plan(root, legacy_root)
    rows = build_rows(root, plan, s_src)
    cross = cross_authority(root, rows, {**v2_authority, "source_rows_by_lodo_date": v2_source_rows})

    film = film.to(torch.device(device))
    film.eval()
    for parameter in film.parameters():
        parameter.requires_grad_(False)
    state_before = {"substrate": state_hash(net.state_dict()), "film": state_hash(film.state_dict())}

    session_rows: list[dict[str, Any]] = []
    for row in rows:
        session = str(row["session"])
        zero_a, zero_r2 = _predict_ep(net, None, row, device=device)
        zero_b, zero_r2_repeat = _predict_ep(net, None, row, device=device)
        require(array_sha256(zero_a) == array_sha256(zero_b) and zero_r2 == zero_r2_repeat,
                f"{session}: EP-ZERO same-process repeat drift")
        film_pred, film_r2 = _predict_ep(net, film, row, device=device)
        target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
        session_rows.append({
            "session": session,
            "date": str(row["date"]),
            "windows": int(np.asarray(row["endpoints"]).size),
            "training_windows": int(row["public"]["training_windows"]),
            "target_sha256": array_sha256(target),
            "r2": {"EP-ZERO": zero_r2, "EP-FILM": film_r2},
            "delta_r2_ep_film_minus_ep_zero": film_r2 - zero_r2,
            "prediction_sha256": {"EP-ZERO": array_sha256(zero_a), "EP-FILM": array_sha256(film_pred)},
            "ep_zero_repeat_identical": True,
        })
    state_after = {"substrate": state_hash(net.state_dict()), "film": state_hash(film.state_dict())}
    require(state_before == state_after and state_before["substrate"] == model_state
            and state_before["film"] == checkpoint_expected["film_state_sha256"],
            "substrate/FiLM changed during in-sample evaluation")

    deltas = [row["delta_r2_ep_film_minus_ep_zero"] for row in session_rows]
    per_date: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        bucket = per_date.setdefault(row["date"], {"sessions": [], "delta": [], "ep_film": [], "ep_zero": []})
        bucket["sessions"].append(row["session"])
        bucket["delta"].append(row["delta_r2_ep_film_minus_ep_zero"])
        bucket["ep_film"].append(row["r2"]["EP-FILM"])
        bucket["ep_zero"].append(row["r2"]["EP-ZERO"])
    per_date_summary = {
        date: {
            "sessions": sorted(values["sessions"]),
            "session_count": len(values["sessions"]),
            "mean_delta": float(np.mean(values["delta"], dtype=np.float64)),
            "mean_ep_film_r2": float(np.mean(values["ep_film"], dtype=np.float64)),
            "mean_ep_zero_r2": float(np.mean(values["ep_zero"], dtype=np.float64)),
            "nonnegative_delta_sessions": int(sum(value >= 0.0 for value in values["delta"])),
        }
        for date, values in sorted(per_date.items())
    }
    session_mean_delta = float(np.mean(deltas, dtype=np.float64))
    date_mean_delta = float(np.mean([row["mean_delta"] for row in per_date_summary.values()], dtype=np.float64))

    v2_score = json.loads((root / V2_ROOT_RELATIVE / "score.json").read_text(encoding="utf-8"))
    v2_gains = {str(fold["outer_date"]): float(fold["early_film_gain"]) for fold in v2_score["folds"]}
    v2_mean_gain = float(sum(v2_gains.values()) / len(v2_gains))
    ratio = session_mean_delta / v2_mean_gain if v2_mean_gain != 0.0 else float("inf")
    overfit_signal = bool(ratio > OVERFIT_RATIO_THRESHOLD)
    receipt = {
        "schema": f"{SCHEMA}_local_insample_eval",
        "status": "COMPLETE_V3_ALL_SOURCE_EP_FILM_LOCAL_IN_SAMPLE_EVAL",
        "in_sample_evaluation": True,
        "in_sample_note": (
            "The scored windows are the full held-in minival eval surfaces of the 13 sessions whose "
            "stride-4 minival windows and support banks the FiLM was trained on; every session is a "
            "training session, so these deltas cannot be read as generalization."
        ),
        "checkpoint_authority": {**checkpoint_authority, **checkpoint_expected},
        "substrate_authority": model_authority,
        "plan_authority_summary": {
            "mean_scale_exact": all(plan_authority["array_sha256_exact"][name] for name in ("mean", "scale")),
            "pcs_exact": plan_authority["array_sha256_exact"]["pcs"],
            "U_exact": plan_authority["array_sha256_exact"]["U"],
            "mu_exact": plan_authority["array_sha256_exact"]["mu"],
            "s_src_exact": plan_authority["s_src_exact"],
            "carrier_relative_deviation_max": cross["carrier_relative_deviation_max"],
        },
        "cross_authority": cross,
        "sessions": session_rows,
        "aggregate": {
            "session_count": len(session_rows),
            "total_windows": int(sum(row["windows"] for row in session_rows)),
            "total_training_windows": int(sum(row["training_windows"] for row in session_rows)),
            "mean_ep_zero_r2": float(np.mean([row["r2"]["EP-ZERO"] for row in session_rows], dtype=np.float64)),
            "mean_ep_film_r2": float(np.mean([row["r2"]["EP-FILM"] for row in session_rows], dtype=np.float64)),
            "session_mean_delta": session_mean_delta,
            "date_mean_of_date_mean_delta": date_mean_delta,
            "nonnegative_delta_sessions": int(sum(delta >= 0.0 for delta in deltas)),
            "min_delta": float(min(deltas)),
            "max_delta": float(max(deltas)),
            "per_date": per_date_summary,
        },
        "v2_lodo_comparison": {
            "v2_oof_early_film_gain_by_lodo_date": v2_gains,
            "v2_oof_mean_gain": v2_mean_gain,
            "v2_oof_nonnegative_dates": int(sum(value >= 0.0 for value in v2_gains.values())),
            "v3_in_sample_session_mean_delta": session_mean_delta,
            "ratio_in_sample_over_v2_oof": ratio,
            "overfit_ratio_threshold": OVERFIT_RATIO_THRESHOLD,
            "overfitting_signal": overfit_signal,
            "criterion": "overfitting_signal := in_sample_session_mean_delta > 2 * v2_oof_mean_gain",
            "note": (
                "V2 deltas are leave-one-date-out (model never saw the scored date); V3 deltas are "
                "in-sample (model trained on every scored session). A materially larger in-sample "
                "delta is an overfitting signal, not a deployment estimate."
            ),
        },
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "wall_time_seconds": time.perf_counter() - wall_start,
        "started_at_utc": started_utc,
        "finished_at_utc": _utc_now(),
        "heldout_calibration_opened": False,
        "hidden_test_opened": False,
        "evalai_opened": False,
        "gpu1_touched": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    if preflight is not None:
        receipt["preflight"] = dict(preflight)
    receipt_sha = _publish(receipt_root / RECEIPT_NAME, receipt)
    return {
        "receipt_sha256": receipt_sha,
        "status": receipt["status"],
        "aggregate": receipt["aggregate"],
        "v2_lodo_comparison": receipt["v2_lodo_comparison"],
    }


__all__ = ("run_insample",)
