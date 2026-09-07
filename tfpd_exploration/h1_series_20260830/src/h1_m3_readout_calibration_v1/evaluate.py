"""Nested source-selected, exactly-M3 H1 readout screen."""
from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _load_minival, _r2, _support
from h1_m3_crossrecord_joint_v1.evaluate import _predict, _session_row
from h1_m3_crossrecord_joint_v1.core import require
from h1_m3_crossrecord_joint_v1.plan import BATCH_SIZE
from .core import ReadoutMap, apply, fit, template
from .plan import DATE_ORDER, FAMILIES, RIDGES, SCHEMA, decide


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def _calibration_row(record: Any, plan: Any, s_src: float) -> dict[str, Any]:
    activity, carrier, support = _support(record, plan, s_src)
    values = tuple(float(value) for value in record.trial_values[:3])
    mask = np.asarray(record.eval_mask, dtype=bool) & np.isin(np.asarray(record.trial_num), values)
    endpoints = np.flatnonzero(mask).astype(np.int64)
    require(endpoints.size >= 8, "M3 calibration fit surface is too small")
    return {
        "session": str(record.session_name),
        "date": str(record.date),
        "activity": activity,
        "carrier": carrier,
        "neural": np.ascontiguousarray(record.neural, dtype=np.float32),
        "target_stream": np.ascontiguousarray(record.velocity, dtype=np.float32),
        "endpoints": endpoints,
        "support": support,
    }


def _base_pair(net: Any, calibration: Mapping[str, Any], query: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    from h1_cross_record_postpool_v1.core import array_sha256

    calibration_prediction, _ = _predict(net, calibration, device=device, alpha=None)
    query_prediction, base_r2 = _predict(net, query, device=device, alpha=None)
    calibration_target = np.ascontiguousarray(
        calibration["target_stream"][calibration["endpoints"]], dtype=np.float32,
    )
    query_target = np.ascontiguousarray(query["target_stream"][query["endpoints"]], dtype=np.float32)
    return {
        "session": str(calibration["session"]),
        "date": str(calibration["date"]),
        "calibration_prediction": calibration_prediction,
        "calibration_target": calibration_target,
        "query_prediction": query_prediction,
        "query_target": query_target,
        "base_r2": base_r2,
        "public": {
            "session": str(calibration["session"]),
            "date": str(calibration["date"]),
            "support": calibration["support"],
            "calibration_windows": int(calibration_target.shape[0]),
            "query_windows": int(query_target.shape[0]),
            "calibration_prediction_sha256": array_sha256(calibration_prediction),
            "calibration_target_sha256": array_sha256(calibration_target),
            "query_prediction_sha256": array_sha256(query_prediction),
            "query_target_sha256": array_sha256(query_target),
            "base_r2": base_r2,
        },
    }


def _candidate(pair: Mapping[str, Any], family: str, ridge: float) -> tuple[float, ReadoutMap, np.ndarray]:
    mapping = fit(pair["calibration_prediction"], pair["calibration_target"], family=family, ridge=ridge)
    prediction = apply(mapping, pair["query_prediction"])
    return _r2(prediction, pair["query_target"]), mapping, prediction


def _equal_date_mean(rows: list[Mapping[str, Any]], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["date"])].append(float(row[key]))
    return float(np.mean([np.mean(values, dtype=np.float64) for values in grouped.values()], dtype=np.float64))


def _select_source(pairs: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for ridge in RIDGES:
            scores = []
            for pair in pairs:
                value, _, _ = _candidate(pair, family, ridge)
                scores.append({"date": pair["date"], "r2": value})
            candidate_rows.append({
                "family": family,
                "ridge": ridge,
                "equal_source_date_mean_r2": _equal_date_mean(scores, "r2"),
                "session_r2": [float(row["r2"]) for row in scores],
            })
    # Exact ties choose the lower-capacity diagonal family, then stronger ridge.
    selected = max(
        candidate_rows,
        key=lambda row: (
            float(row["equal_source_date_mean_r2"]),
            int(row["family"] == "DIA7"),
            float(row["ridge"]),
        ),
    )
    return dict(selected), candidate_rows


def run_fold(repo_root: Path, outer_date: str, *, device: str, receipt_root: Path) -> dict[str, Any]:
    from h1_causal_activity_completion_v1.stage1 import ARTIFACT_RELATIVE, C1_AUTHORITIES, _load_model, _load_plan
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve()
    require(outer_date in DATE_ORDER, "outer date drift")
    authority = C1_AUTHORITIES[outer_date]
    directory = root / ARTIFACT_RELATIVE / outer_date
    plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    model, initial_state, model_receipt = _load_model(directory, authority, outer_date, device)
    indexed = index_heldin_calib(root / DATA_RELATIVE)
    source_pairs = []
    for session in plan.source_sessions:
        record = load_record(indexed[session])
        calibration = _calibration_row(record, plan, s_src)
        query = _session_row(record=record, minival=_load_minival(root / DATA_RELATIVE, session), plan=plan, s_src=s_src)
        source_pairs.append(_base_pair(model, calibration, query, device=device))
    selected, candidates = _select_source(source_pairs)
    selection_sha = _publish(Path(receipt_root) / f"selection_{outer_date}.json", {
        "schema": f"{SCHEMA}_selection",
        "status": "SOURCE_SELECTED_BEFORE_OUTER_DATE_OPEN",
        "outer_date": outer_date,
        "source_sessions": list(plan.source_sessions),
        "source_pairs": [pair["public"] for pair in source_pairs],
        "candidate_rows": candidates,
        "selected": selected,
        "outer_date_calibration_opened": False,
        "outer_date_minival_opened": False,
    })

    target_records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=outer_date)
    target_rows = []
    for session, record in target_records.items():
        calibration = _calibration_row(record, plan, s_src)
        query = _session_row(record=record, minival=_load_minival(root / DATA_RELATIVE, session), plan=plan, s_src=s_src)
        pair = _base_pair(model, calibration, query, device=device)
        selected_r2, mapping, selected_prediction = _candidate(pair, str(selected["family"]), float(selected["ridge"]))
        template_prediction = template(pair["calibration_target"], pair["query_target"].shape[0])
        target_rows.append({
            **pair["public"],
            "selected_r2": selected_r2,
            "template_r2": _r2(template_prediction, pair["query_target"]),
            "selected_prediction_shape": list(selected_prediction.shape),
            "selected_family": mapping.family,
            "selected_ridge": mapping.ridge,
        })
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_OUTER_DATE_M3_READOUT_SCORE",
        "outer_date": outer_date,
        "selection_sha256": selection_sha,
        "selected": selected,
        "base_r2": float(np.mean([row["base_r2"] for row in target_rows], dtype=np.float64)),
        "selected_r2": float(np.mean([row["selected_r2"] for row in target_rows], dtype=np.float64)),
        "template_r2": float(np.mean([row["template_r2"] for row in target_rows], dtype=np.float64)),
        "target_rows": target_rows,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    require(initial_state == state_hash(model.state_dict()), "frozen C1 model changed")
    fold["fold_sha256"] = _publish(Path(receipt_root) / f"fold_{outer_date}.json", fold)
    return fold


def run(repo_root: Path, *, device: str, receipt_root: Path) -> dict[str, Any]:
    import gc
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "H1-M3RC requires isolated logical GPU0")
    rows = []
    for date in DATE_ORDER:
        rows.append(run_fold(repo_root, date, device=device, receipt_root=receipt_root))
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_M3_READOUT_CALIBRATION_SOURCE_OOF",
        "folds": rows,
        "decision": decide(rows),
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }


__all__ = ("run", "run_fold")

