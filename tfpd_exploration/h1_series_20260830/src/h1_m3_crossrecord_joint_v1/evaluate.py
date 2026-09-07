"""Five-fold paired H1 exactly-M3 cross-record training and scoring."""
from __future__ import annotations

import copy
import gc
import hashlib
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.core import array_sha256
from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _load_minival, _r2, _support
from .core import (
    capture_rng,
    decode_with_identity,
    joint_identity,
    native_and_post_identity,
    require,
    restore_rng,
    rng_equal,
    window_batch,
)
from .plan import (
    ARMS,
    BATCH_SIZE,
    DATE_ORDER,
    EPOCHS,
    LEARNING_RATE,
    PREDICTION_DIVISOR,
    SCHEMA,
    SEED,
    TRAIN_STRIDE,
    WEIGHT_DECAY,
    decide_oof,
)


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def _session_row(*, record: Any, minival: Mapping[str, Any], plan: Any, s_src: float) -> dict[str, Any]:
    activity, carrier, support = _support(record, plan, s_src)
    endpoints = np.ascontiguousarray(minival["endpoints"], dtype=np.int64)
    train_endpoints = np.ascontiguousarray(endpoints[::TRAIN_STRIDE], dtype=np.int64)
    require(train_endpoints.size > 0, "source session has no stride-4 training endpoints")
    return {
        "session": str(record.session_name),
        "date": str(record.date),
        "activity": activity,
        "carrier": carrier,
        "neural": np.ascontiguousarray(minival["neural"], dtype=np.float32),
        "target_stream": np.ascontiguousarray(minival["target"], dtype=np.float32),
        "endpoints": endpoints,
        "train_endpoints": train_endpoints,
        "public": {
            "session": str(record.session_name),
            "date": str(record.date),
            "support": support,
            "minival_input_sha256": str(minival["path_sha256"]),
            "activity_sha256": array_sha256(activity),
            "carrier_sha256": array_sha256(carrier),
            "query_neural_sha256": array_sha256(minival["neural"]),
            "target_stream_sha256": array_sha256(minival["target"]),
            "endpoint_sha256": array_sha256(endpoints),
            "train_endpoint_sha256": array_sha256(train_endpoints),
            "windows": int(endpoints.size),
            "training_windows": int(train_endpoints.size),
        },
    }


def _support_tensors(row: Mapping[str, Any], *, device: str) -> tuple[Any, Any]:
    import torch

    activity = torch.as_tensor(row["activity"], dtype=torch.float32, device=device).unsqueeze(0)
    carrier = torch.as_tensor(row["carrier"], dtype=torch.float32, device=device).unsqueeze(0)
    return activity, carrier


def _identity(
    net: Any,
    row: Mapping[str, Any],
    *,
    device: str,
    alpha: Any | None,
    support_tensors: tuple[Any, Any] | None = None,
) -> Any:
    activity, carrier = support_tensors or _support_tensors(row, device=device)
    native, post = native_and_post_identity(net, activity, carrier)
    return native if alpha is None else joint_identity(native, post, alpha)


def _finite_gradients(net: Any, alpha: Any | None) -> tuple[int, int]:
    import torch

    materialized = 0
    nonzero = 0
    values = list(net.parameters()) + ([] if alpha is None else [alpha])
    for parameter in values:
        if parameter.grad is None:
            continue
        materialized += parameter.numel()
        require(bool(torch.isfinite(parameter.grad).all()), "training gradient nonfinite")
        nonzero += int(torch.count_nonzero(parameter.grad).item())
    require(materialized > 0 and nonzero > 0, "training graph has no nonzero finite gradient")
    return materialized, nonzero


def _step(
    net: Any,
    alpha: Any | None,
    optimizer: Any,
    neural: Any,
    target: Any,
    row: Mapping[str, Any],
    support_tensors: tuple[Any, Any],
) -> tuple[Any, Any, tuple[int, int]]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    identity = _identity(
        net,
        row,
        device=str(neural.device),
        alpha=alpha,
        support_tensors=support_tensors,
    )
    prediction = decode_with_identity(net, neural, identity)[:, -1, :] / PREDICTION_DIVISOR
    loss = torch.nn.functional.mse_loss(prediction, target)
    require(bool(torch.isfinite(loss)), "training loss nonfinite")
    loss.backward()
    gradient = _finite_gradients(net, alpha)
    optimizer.step()
    require(all(bool(torch.isfinite(parameter).all()) for parameter in net.parameters()), "trained model parameter nonfinite")
    if alpha is not None:
        require(bool(torch.isfinite(alpha).all()), "trained alpha nonfinite")
    return prediction.detach(), loss.detach(), gradient


def train_pair(base_net: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(rows, "source row roster empty")
    native = copy.deepcopy(base_net).to(device)
    joint = copy.deepcopy(base_net).to(device)
    for model in (native, joint):
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    initial_native = state_hash(native.state_dict())
    initial_joint = state_hash(joint.state_dict())
    require(initial_native == initial_joint, "paired model initial state mismatch")
    alpha = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    require(float(alpha.detach().cpu()) == 0.0 and not bool(torch.signbit(alpha.detach())), "alpha is not IEEE +0")
    native_optimizer = torch.optim.Adam(native.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    joint_optimizer = torch.optim.Adam([*joint.parameters(), alpha], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # M3 activity and carrier are immutable session constants.  Keep one shared
    # device copy per source session so paired arms do not repeatedly transfer
    # the same tensors at every optimizer step.
    device_support = {
        str(row["session"]): _support_tensors(row, device=device)
        for row in rows
    }

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    first_prediction_equal = False
    first_identity_equal = False
    rng_equal_steps = 0
    total_steps = 0
    native_nonzero_gradient_steps = 0
    joint_nonzero_gradient_steps = 0
    first_losses: dict[str, float] = {}
    last_losses: dict[str, float] = {}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(EPOCHS):
        ordered = list(rows)
        random.Random(f"{SCHEMA}|{SEED}|session-order|{epoch}").shuffle(ordered)
        loss_rows = {"N3-XR12": [], "J3-XR12": []}
        epoch_steps = 0
        for row in ordered:
            endpoints = np.asarray(row["train_endpoints"], dtype=np.int64).copy()
            token = hashlib.sha256(f"{SCHEMA}|{SEED}|{epoch}|{row['session']}".encode("utf-8")).digest()
            np.random.default_rng(int.from_bytes(token[:8], "big")).shuffle(endpoints)
            for offset in range(0, endpoints.size, BATCH_SIZE):
                selected = endpoints[offset : offset + BATCH_SIZE]
                neural_np = window_batch(row["neural"], selected)
                target_np = np.ascontiguousarray(row["target_stream"][selected], dtype=np.float32)
                neural = torch.as_tensor(neural_np, dtype=torch.float32, device=device)
                target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
                support = device_support[str(row["session"])]

                if total_steps == 0:
                    # This is evaluated before either arm has taken an update.
                    # At IEEE +0 the J3 residual must be the exact N3 identity,
                    # not merely shape-compatible with it.
                    with torch.no_grad():
                        native_identity = _identity(
                            native, row, device=device, alpha=None, support_tensors=support,
                        )
                        joint_identity_zero = _identity(
                            joint, row, device=device, alpha=alpha, support_tensors=support,
                        )
                    first_identity_equal = bool(torch.equal(native_identity, joint_identity_zero))
                    require(first_identity_equal, "N3/J3 first identity is not bitwise equal at alpha +0")

                before = capture_rng()
                native_prediction, native_loss, native_grad = _step(
                    native, None, native_optimizer, neural, target, row, support,
                )
                after_native = capture_rng()
                restore_rng(before)
                joint_prediction, joint_loss, joint_grad = _step(
                    joint, alpha, joint_optimizer, neural, target, row, support,
                )
                after_joint = capture_rng()
                require(rng_equal(after_native, after_joint), "paired RNG consumption diverged")
                restore_rng(after_native)
                rng_equal_steps += 1

                if total_steps == 0:
                    first_prediction_equal = bool(torch.equal(native_prediction, joint_prediction))
                    require(first_prediction_equal, "N3/J3 first prediction is not bitwise equal at alpha +0")
                    first_losses = {"N3-XR12": float(native_loss.cpu()), "J3-XR12": float(joint_loss.cpu())}
                native_nonzero_gradient_steps += int(native_grad[1] > 0)
                joint_nonzero_gradient_steps += int(joint_grad[1] > 0)
                loss_rows["N3-XR12"].append(float(native_loss.cpu()))
                loss_rows["J3-XR12"].append(float(joint_loss.cpu()))
                last_losses = {"N3-XR12": float(native_loss.cpu()), "J3-XR12": float(joint_loss.cpu())}
                total_steps += 1
                epoch_steps += 1
        require(epoch_steps > 0, "training epoch has no steps")
        state_sentinel = epoch in (0, EPOCHS // 2 - 1, EPOCHS - 1)
        epoch_rows.append({
            "epoch_zero_based": epoch,
            "steps": epoch_steps,
            "native_mean_loss": float(np.mean(loss_rows["N3-XR12"], dtype=np.float64)),
            "joint_mean_loss": float(np.mean(loss_rows["J3-XR12"], dtype=np.float64)),
            "alpha": float(alpha.detach().cpu()),
            "tanh_alpha": float(torch.tanh(alpha.detach()).cpu()),
            "state_hash_sentinel": state_sentinel,
            "native_state_sha256": state_hash(native.state_dict()) if state_sentinel else None,
            "joint_state_sha256": state_hash(joint.state_dict()) if state_sentinel else None,
        })
    require(total_steps == rng_equal_steps == native_nonzero_gradient_steps == joint_nonzero_gradient_steps,
            "paired training step/evidence count drift")
    torch.cuda.synchronize()
    return native, joint, alpha.detach(), {
        "epochs": EPOCHS,
        "steps": total_steps,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "initial_model_state_sha256": initial_native,
        "native_final_state_sha256": state_hash(native.state_dict()),
        "joint_final_state_sha256": state_hash(joint.state_dict()),
        "first_prediction_bitwise_equal": first_prediction_equal,
        "first_identity_bitwise_equal": first_identity_equal,
        "rng_equal_steps": rng_equal_steps,
        "native_nonzero_gradient_steps": native_nonzero_gradient_steps,
        "joint_nonzero_gradient_steps": joint_nonzero_gradient_steps,
        "first_losses": first_losses,
        "last_losses": last_losses,
        "alpha": float(alpha.detach().cpu()),
        "tanh_alpha": float(torch.tanh(alpha.detach()).cpu()),
        "epoch_rows": epoch_rows,
    }


def _predict(net: Any, row: Mapping[str, Any], *, device: str, alpha: Any | None) -> tuple[np.ndarray, float]:
    import torch

    net.eval()
    identity = _identity(net, row, device=device, alpha=alpha)
    endpoints = np.asarray(row["endpoints"], dtype=np.int64)
    output = np.empty((endpoints.size, 7), dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, endpoints.size, BATCH_SIZE):
            chosen = endpoints[offset : offset + BATCH_SIZE]
            neural = torch.as_tensor(window_batch(row["neural"], chosen), dtype=torch.float32, device=device)
            value = decode_with_identity(net, neural, identity)[:, -1, :] / PREDICTION_DIVISOR
            output[offset : offset + chosen.size] = value.detach().cpu().numpy()
    target = np.ascontiguousarray(row["target_stream"][endpoints], dtype=np.float32)
    return output, _r2(output, target)


def _score_three(base: Any, native: Any, joint: Any, alpha: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores = {arm: [] for arm in ARMS}
    public: list[dict[str, Any]] = []
    for row in rows:
        target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
        predictions: dict[str, np.ndarray] = {}
        for arm, model, gate in (
            ("FROZEN-C1", base, None),
            ("N3-XR12", native, None),
            ("J3-XR12", joint, alpha),
        ):
            prediction, value = _predict(model, row, device=device, alpha=gate)
            predictions[arm] = prediction
            scores[arm].append(value)
        public.append({
            **row["public"],
            "target_sha256": array_sha256(target),
            "r2": {arm: scores[arm][-1] for arm in ARMS},
            "prediction_sha256": {arm: array_sha256(predictions[arm]) for arm in ARMS},
        })
    means = {arm: sum(values) / len(values) for arm, values in scores.items()}
    return means, public


def run_fold(repo_root: Path, outer_date: str, *, device: str, receipt_root: Path) -> dict[str, Any]:
    from h1_causal_activity_completion_v1.stage1 import ARTIFACT_RELATIVE as C1_ARTIFACT, C1_AUTHORITIES, _load_model, _load_plan
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve()
    require(outer_date in DATE_ORDER, "unknown outer date")
    directory = root / C1_ARTIFACT / outer_date
    authority = C1_AUTHORITIES[outer_date]
    plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    base, base_state, model_receipt = _load_model(directory, authority, outer_date, device)
    indexed = index_heldin_calib(root / DATA_RELATIVE)
    source_names = tuple(plan.source_sessions)
    require(source_names and all(name in indexed and not name.startswith(f"ses-{outer_date}") for name in source_names),
            "fold source roster drift")
    source_rows = [
        _session_row(
            record=load_record(indexed[session]),
            minival=_load_minival(root / DATA_RELATIVE, session),
            plan=plan,
            s_src=s_src,
        )
        for session in source_names
    ]
    native, joint, alpha, training = train_pair(base, source_rows, device=device)
    train_sha = _publish(Path(receipt_root) / f"training_{outer_date}.json", {
        "schema": f"{SCHEMA}_training",
        "status": "TRAINING_FROZEN_BEFORE_OUTER_DATE_OPEN",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "source_rows": [row["public"] for row in source_rows],
        "training": training,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "outer_date_calib_opened": False,
        "outer_date_minival_opened": False,
    })

    target_records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=outer_date)
    target_rows = [
        _session_row(
            record=record,
            minival=_load_minival(root / DATA_RELATIVE, session),
            plan=plan,
            s_src=s_src,
        )
        for session, record in target_records.items()
    ]
    base_before = state_hash(base.state_dict())
    native_before = state_hash(native.state_dict())
    joint_before = state_hash(joint.state_dict())
    scores, rows = _score_three(base, native, joint, alpha, target_rows, device=device)
    require(base_before == base_state == state_hash(base.state_dict()), "frozen base state changed")
    require(native_before == state_hash(native.state_dict()) and joint_before == state_hash(joint.state_dict()),
            "trained model changed during target scoring")
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_PAIRED_TRAINING_OUTER_SCORE",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions": list(target_records),
        "training_sha256": train_sha,
        "scores": scores,
        "native_gain": scores["N3-XR12"] - scores["FROZEN-C1"],
        "joint_gain": scores["J3-XR12"] - scores["FROZEN-C1"],
        "joint_increment": scores["J3-XR12"] - scores["N3-XR12"],
        "alpha": float(alpha.cpu()),
        "tanh_alpha": float(__import__("torch").tanh(alpha).cpu()),
        "target_rows": rows,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    fold["fold_sha256"] = _publish(Path(receipt_root) / f"fold_{outer_date}.json", fold)
    return fold


def run(repo_root: Path, *, device: str, receipt_root: Path) -> dict[str, Any]:
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "H1 M3 matched training requires isolated logical GPU0")
    folds: list[dict[str, Any]] = []
    for date in DATE_ORDER:
        folds.append(run_fold(repo_root, date, device=device, receipt_root=receipt_root))
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_M3_CROSS_RECORD_JOINT_POSTPOOL_SOURCE_OOF",
        "date_order": list(DATE_ORDER),
        "arms": list(ARMS),
        "folds": folds,
        "decision": decide_oof(folds),
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "run_fold", "train_pair")
