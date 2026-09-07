"""Five-fold H1 source-fitted post-pool output-residual screen."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.core import array_sha256, identity_states, query_member, require
from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _decode, _load_minival, _r2, _support
from .plan import DATE_ORDER, SCHEMA, apply_beta, decide_oof, fit_beta


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    sidecar = path.with_name(path.name + ".sha256")
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(sidecar, 0o444)
    return digest


def _material(*, net: Any, record: Any, minival: Mapping[str, Any], plan: Any, s_src: float, device: str) -> dict[str, Any]:
    from h1_cross_record_postpool_v1.plan import CHUNK_LENGTH

    activity, carrier, support_receipt = _support(record, plan, s_src)
    member = query_member(minival["neural"])
    states = identity_states(net, activity, member, carrier, device=device)
    endpoints = np.asarray(minival["endpoints"], dtype=np.int64)
    target = np.ascontiguousarray(np.asarray(minival["target"], dtype=np.float32)[endpoints])
    after = endpoints >= CHUNK_LENGTH
    static = _decode(net, minival["neural"], endpoints, states["static"])
    post = static.copy()
    if bool(after.any()):
        post[after] = _decode(net, minival["neural"], endpoints[after], states["post"])
    return {
        "date": str(record.date),
        "session": str(record.session_name),
        "target": target,
        "static": static,
        "post": post,
        "public": {
            "date": str(record.date),
            "session": str(record.session_name),
            "support": support_receipt,
            "minival_input_sha256": minival["path_sha256"],
            "query_neural_sha256": array_sha256(minival["neural"]),
            "target_sha256": array_sha256(target),
            "endpoint_sha256": array_sha256(endpoints),
            "static_prediction_sha256": array_sha256(static),
            "post_prediction_sha256": array_sha256(post),
            "static_r2": _r2(static, target),
            "post_r2": _r2(post, target),
            "windows": int(len(endpoints)),
            "post_commit_windows": int(after.sum()),
            "target_updates": 0,
        },
    }


def _equal_recording_score(materials: list[Mapping[str, Any]], beta: float) -> tuple[float, list[dict[str, Any]]]:
    values: list[float] = []
    public: list[dict[str, Any]] = []
    for row in materials:
        prediction = apply_beta(row["static"], row["post"], beta)
        if float(beta) == 0.0:
            require(array_sha256(prediction) == array_sha256(row["static"]), "beta +0 static sentinel failed")
        r2 = _r2(prediction, row["target"])
        values.append(r2)
        public.append({
            **row["public"],
            "beta": float(beta),
            "prediction_sha256": array_sha256(prediction),
            "r2": r2,
        })
    return sum(values) / len(values), public


def run_fold(repo_root: Path, outer_date: str, *, device: str, receipt_root: Path) -> dict[str, Any]:
    from h1_causal_activity_completion_v1.stage1 import ARTIFACT_RELATIVE as C1_ARTIFACT, C1_AUTHORITIES, _load_model, _load_plan
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve()
    directory = root / C1_ARTIFACT / outer_date
    authority = C1_AUTHORITIES[outer_date]
    plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    model, state_before, model_receipt = _load_model(directory, authority, outer_date, device)
    net = getattr(model, "net", model)
    data_root = root / DATA_RELATIVE
    indexed = index_heldin_calib(data_root)
    source_names = tuple(plan.source_sessions)
    require(source_names and all(name in indexed and not name.startswith(f"ses-{outer_date}") for name in source_names),
            "fold source roster drift")
    source = [
        _material(net=net, record=load_record(indexed[session]), minival=_load_minival(data_root, session),
                  plan=plan, s_src=s_src, device=device)
        for session in source_names
    ]
    fit = fit_beta(source)
    source_static, _ = _equal_recording_score(source, 0.0)
    source_fused, _ = _equal_recording_score(source, fit["beta"])
    beta_sha = _publish(
        Path(receipt_root) / f"beta_{outer_date}.json",
        {
            "schema": f"{SCHEMA}_beta",
            "status": "BETA_FROZEN_BEFORE_OUTER_DATE_OPEN",
            "outer_date": outer_date,
            "source_sessions": list(source_names),
            "fit": fit,
            "source_static_equal_recording_r2": source_static,
            "source_fused_equal_recording_r2": source_fused,
            "source_gain": source_fused - source_static,
            "outer_date_calib_opened": False,
            "outer_date_minival_opened": False,
            "target_updates": 0,
        },
    )
    target_records = load_outer_date_target_records(data_root, outer_date=outer_date)
    target = [
        _material(net=net, record=record, minival=_load_minival(data_root, session),
                  plan=plan, s_src=s_src, device=device)
        for session, record in target_records.items()
    ]
    target_static, static_rows = _equal_recording_score(target, 0.0)
    target_fused, fused_rows = _equal_recording_score(target, fit["beta"])
    state_after = state_hash(model.state_dict())
    require(state_before == state_after, "frozen C1 model state changed")
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_SOURCE_FIT_OUTER_SCORE",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions": list(target_records),
        "fit": fit,
        "beta_sha256": beta_sha,
        "source_static_equal_recording_r2": source_static,
        "source_fused_equal_recording_r2": source_fused,
        "source_gain": source_fused - source_static,
        "target_static_equal_recording_r2": target_static,
        "target_fused_equal_recording_r2": target_fused,
        "delta_vs_static": target_fused - target_static,
        "target_static_rows": static_rows,
        "target_fused_rows": fused_rows,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    fold["fold_sha256"] = _publish(Path(receipt_root) / f"fold_{outer_date}.json", fold)
    return fold


def run(repo_root: Path, *, device: str, receipt_root: Path) -> dict[str, Any]:
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "screen requires isolated logical GPU0")
    folds = [run_fold(repo_root, date, device=device, receipt_root=receipt_root) for date in DATE_ORDER]
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_POSTPOOL_OUTPUT_RESIDUAL_SOURCE_OOF",
        "date_order": list(DATE_ORDER),
        "folds": folds,
        "decision": decide_oof(folds),
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "run_fold")
