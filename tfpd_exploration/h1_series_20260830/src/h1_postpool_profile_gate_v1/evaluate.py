"""Five-fold source-only training and outer-date scoring for the H1 profile gate."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.core import array_sha256, identity_states, query_member, require, stream_window
from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _decode, _load_minival, _r2, _support
from .core import profile_identity
from .plan import (
    BATCH_SIZE, DATE_ORDER, EPOCHS, LEARNING_RATE, PROFILE_LENGTH, SCHEMA, SEED,
    TRAIN_STRIDE, WEIGHT_DECAY, decide_oof,
)


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    sidecar = path.with_name(path.name + ".sha256")
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(sidecar, 0o444)
    return digest


def _session(*, net: Any, record: Any, minival: Mapping[str, Any], plan: Any, s_src: float, device: str) -> dict[str, Any]:
    from h1_cross_record_postpool_v1.plan import CHUNK_LENGTH

    activity, carrier, support_receipt = _support(record, plan, s_src)
    member = query_member(minival["neural"])
    states = identity_states(net, activity, member, carrier, device=device)
    endpoints = np.asarray(minival["endpoints"], dtype=np.int64)
    after = endpoints >= CHUNK_LENGTH
    train_endpoints = np.ascontiguousarray(endpoints[after][::TRAIN_STRIDE], dtype=np.int64)
    require(train_endpoints.size > 0, "session has no post-commit training rows")
    # Convert inference tensors into ordinary detached constants so autograd can
    # traverse the profile and frozen decoder without retaining identity graphs.
    static = states["static"].detach().cpu().numpy().copy()
    post = states["post"].detach().cpu().numpy().copy()
    return {
        "date": str(record.date),
        "session": str(record.session_name),
        "neural": np.ascontiguousarray(minival["neural"], dtype=np.float32),
        "target_stream": np.ascontiguousarray(minival["target"], dtype=np.float32),
        "endpoints": endpoints,
        "train_endpoints": train_endpoints,
        "static": static,
        "post": post,
        "public": {
            "date": str(record.date),
            "session": str(record.session_name),
            "support": support_receipt,
            "minival_input_sha256": minival["path_sha256"],
            "query_neural_sha256": array_sha256(minival["neural"]),
            "target_stream_sha256": array_sha256(minival["target"]),
            "endpoint_sha256": array_sha256(endpoints),
            "train_endpoint_sha256": array_sha256(train_endpoints),
            "static_identity_sha256": array_sha256(static),
            "post_identity_sha256": array_sha256(post),
            "windows": int(endpoints.size),
            "training_windows": int(train_endpoints.size),
        },
    }


def _ordinary_identity(row: Mapping[str, Any], device: str) -> tuple[Any, Any]:
    import torch

    static = torch.as_tensor(row["static"], dtype=torch.float32, device=device)
    post = torch.as_tensor(row["post"], dtype=torch.float32, device=device)
    return static, post


def _profile_sha(profile: Any) -> str:
    return array_sha256(profile.detach().cpu().numpy())


def train_profile(net: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from m1_h1_activity_headroom_v1.core import forward_with_cached_identity

    require(rows, "profile source rows empty")
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    profile = torch.nn.Parameter(torch.zeros(PROFILE_LENGTH, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam((profile,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    constants = {str(row["session"]): _ordinary_identity(row, device) for row in rows}
    zero_identity = profile_identity(*constants[str(rows[0]["session"])], profile)
    require(torch.equal(zero_identity, constants[str(rows[0]["session"])][0]),
            "trainable zero profile is not value-exact static")
    first_sha = _profile_sha(profile)
    epoch_rows: list[dict[str, Any]] = []
    total_steps = 0
    nonzero_gradient_steps = 0
    first_loss = None
    last_loss = None
    for epoch in range(EPOCHS):
        epoch_losses: list[float] = []
        ordered = list(rows)
        random.Random(f"{SCHEMA}|{SEED}|session-order|{epoch}").shuffle(ordered)
        for row in ordered:
            session = str(row["session"])
            endpoints = np.asarray(row["train_endpoints"], dtype=np.int64).copy()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
                f"{SCHEMA}|{SEED}|{epoch}|{session}".encode("utf-8")
            ).digest()[:8], "big"))
            rng.shuffle(endpoints)
            static, post = constants[session]
            for offset in range(0, len(endpoints), BATCH_SIZE):
                selected = endpoints[offset : offset + BATCH_SIZE]
                windows = np.stack([stream_window(row["neural"], int(endpoint)) for endpoint in selected])
                target = np.ascontiguousarray(row["target_stream"][selected], dtype=np.float32)
                optimizer.zero_grad(set_to_none=True)
                identity = profile_identity(static, post, profile)
                prediction = forward_with_cached_identity(net, windows, identity)[:, -1, :] / 20.0
                loss = torch.nn.functional.mse_loss(
                    prediction, torch.as_tensor(target, dtype=torch.float32, device=device),
                )
                require(bool(torch.isfinite(loss)), "profile loss nonfinite")
                loss.backward()
                require(profile.grad is not None and bool(torch.isfinite(profile.grad).all()), "profile gradient nonfinite")
                nonzero = bool(torch.count_nonzero(profile.grad).item())
                require(nonzero, "profile gradient is zero")
                nonzero_gradient_steps += int(nonzero)
                optimizer.step()
                require(bool(torch.isfinite(profile).all()), "profile became nonfinite")
                value = float(loss.detach().cpu())
                first_loss = value if first_loss is None else first_loss
                last_loss = value
                epoch_losses.append(value)
                total_steps += 1
        require(epoch_losses, "profile epoch has no steps")
        epoch_rows.append({
            "epoch_zero_based": epoch,
            "steps": len(epoch_losses),
            "mean_loss": float(np.mean(epoch_losses, dtype=np.float64)),
            "profile_sha256": _profile_sha(profile),
            "profile_tanh_abs_max": float(torch.tanh(profile).abs().max().detach().cpu()),
        })
    require(total_steps == nonzero_gradient_steps, "profile gradient coverage drift")
    return profile.detach(), {
        "epochs": EPOCHS,
        "steps": total_steps,
        "nonzero_gradient_steps": nonzero_gradient_steps,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "initial_profile_sha256": first_sha,
        "final_profile_sha256": _profile_sha(profile),
        "profile": [float(value) for value in profile.detach().cpu().tolist()],
        "profile_tanh_abs_max": float(torch.tanh(profile).abs().max().detach().cpu()),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "epoch_rows": epoch_rows,
        "optimizer_parameter_tensors": 1,
        "optimizer_parameter_count": PROFILE_LENGTH,
    }


def _score(net: Any, rows: list[Mapping[str, Any]], profile: Any, *, device: str) -> tuple[float, float, list[dict[str, Any]]]:
    import torch

    static_scores: list[float] = []
    profile_scores: list[float] = []
    public: list[dict[str, Any]] = []
    for row in rows:
        static, post = _ordinary_identity(row, device)
        identity = profile_identity(static, post, profile)
        endpoints = np.asarray(row["endpoints"], dtype=np.int64)
        target = np.ascontiguousarray(row["target_stream"][endpoints], dtype=np.float32)
        static_prediction = _decode(net, row["neural"], endpoints, static)
        profile_prediction = _decode(net, row["neural"], endpoints, identity)
        static_r2 = _r2(static_prediction, target)
        profile_r2 = _r2(profile_prediction, target)
        static_scores.append(static_r2)
        profile_scores.append(profile_r2)
        public.append({
            **row["public"],
            "target_sha256": array_sha256(target),
            "static_prediction_sha256": array_sha256(static_prediction),
            "profile_prediction_sha256": array_sha256(profile_prediction),
            "static_r2": static_r2,
            "profile_r2": profile_r2,
            "delta": profile_r2 - static_r2,
        })
    return sum(static_scores) / len(static_scores), sum(profile_scores) / len(profile_scores), public


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
        _session(net=net, record=load_record(indexed[session]), minival=_load_minival(data_root, session),
                 plan=plan, s_src=s_src, device=device)
        for session in source_names
    ]
    profile, training = train_profile(net, source, device=device)
    source_static, source_profile, _ = _score(net, source, profile, device=device)
    profile_sha = _publish(Path(receipt_root) / f"profile_{outer_date}.json", {
        "schema": f"{SCHEMA}_profile",
        "status": "PROFILE_FROZEN_BEFORE_OUTER_DATE_OPEN",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "training": training,
        "source_static_equal_recording_r2": source_static,
        "source_profile_equal_recording_r2": source_profile,
        "source_gain": source_profile - source_static,
        "outer_date_calib_opened": False,
        "outer_date_minival_opened": False,
        "base_model_state_sha256": state_before,
    })
    target_records = load_outer_date_target_records(data_root, outer_date=outer_date)
    target = [
        _session(net=net, record=record, minival=_load_minival(data_root, session),
                 plan=plan, s_src=s_src, device=device)
        for session, record in target_records.items()
    ]
    target_static, target_profile, target_rows = _score(net, target, profile, device=device)
    state_after = state_hash(model.state_dict())
    require(state_after == state_before, "base C1 state changed during profile training")
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_PROFILE_TRAIN_OUTER_SCORE",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions": list(target_records),
        "profile_sha256": profile_sha,
        "training": {key: value for key, value in training.items() if key != "profile"},
        "source_static_equal_recording_r2": source_static,
        "source_profile_equal_recording_r2": source_profile,
        "source_gain": source_profile - source_static,
        "target_static_equal_recording_r2": target_static,
        "target_profile_equal_recording_r2": target_profile,
        "delta_vs_static": target_profile - target_static,
        "target_rows": target_rows,
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
            "profile screen requires isolated logical GPU0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    folds = [run_fold(repo_root, date, device=device, receipt_root=receipt_root) for date in DATE_ORDER]
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_POSTPOOL_PROFILE_GATE_SOURCE_OOF",
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


__all__ = ("run", "run_fold", "train_profile")
