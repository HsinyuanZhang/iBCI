"""One-GPU five-fold training and outer-date scoring for LP-F3/LP-R3/SRPD."""
from __future__ import annotations

from collections import OrderedDict
import copy
import gc
import hashlib
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.core import array_sha256
from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _load_minival, _r2
from h1_m3_crossrecord_joint_v1.core import decode_with_identity, window_batch
from .core import (
    freeze_decoder_train_identity,
    late_identity,
    native_identity,
    normalized_identity_distill,
    require,
    support_index,
)
from .plan import (
    ARMS,
    BATCH_SIZE,
    DATE_ORDER,
    DISTILL_WEIGHT,
    EPOCHS,
    LEARNING_RATE,
    PREDICTION_DIVISOR,
    SCHEMA,
    SEED,
    TRAIN_ARMS,
    TRAIN_STRIDE,
    WEIGHT_DECAY,
    decide_oof,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def _support_bank(record: Any, plan: Any, s_src: float, *, all_blocks: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from src.data.h1_m4_eb_pilot import fit_deployment_carrier, interpolate_trial_identity
    from src.h1_m4_cce_contract import NORMALIZER_FLOOR

    trial_values = tuple(float(value) for value in record.trial_values)
    require(len(trial_values) >= 3 and len(set(trial_values)) == len(trial_values), "session has no legal unique M3 support")
    starts = range(len(trial_values) - 2) if all_blocks else range(1)
    per_trial = {value: interpolate_trial_identity(record, value) for value in trial_values if all_blocks or value in trial_values[:3]}
    runtime: list[dict[str, Any]] = []
    public: list[dict[str, Any]] = []
    for start in starts:
        values = trial_values[start : start + 3]
        require(len(values) == 3, "support block truncated")
        activity = np.ascontiguousarray(np.stack([per_trial[value] for value in values]), dtype=np.float32)
        carrier = np.ascontiguousarray(
            fit_deployment_carrier(record, plan, values)["carrier"] / max(float(s_src), NORMALIZER_FLOOR),
            dtype=np.float32,
        )
        entry = {
            "block_index": int(start),
            "support_trials": list(values),
            "activity": activity,
            "carrier": carrier,
        }
        evidence = {
            "block_index": int(start),
            "support_trials": list(values),
            "activity_sha256": array_sha256(activity),
            "carrier_sha256": array_sha256(carrier),
        }
        runtime.append(entry)
        public.append(evidence)
    require(runtime and runtime[0]["support_trials"] == list(trial_values[:3]), "first-M3 support drift")
    if all_blocks:
        require(len(runtime) >= 2, "source session has no non-first contiguous M3 block")
    return runtime, public


def _session_row(*, record: Any, minival: Mapping[str, Any], plan: Any, s_src: float, all_blocks: bool) -> dict[str, Any]:
    bank, bank_public = _support_bank(record, plan, s_src, all_blocks=all_blocks)
    endpoints = np.ascontiguousarray(minival["endpoints"], dtype=np.int64)
    train_endpoints = np.ascontiguousarray(endpoints[::TRAIN_STRIDE], dtype=np.int64)
    require(endpoints.size > 0 and train_endpoints.size > 0, "session query surface is empty")
    return {
        "session": str(record.session_name),
        "date": str(record.date),
        "support_bank": bank,
        "neural": np.ascontiguousarray(minival["neural"], dtype=np.float32),
        "target_stream": np.ascontiguousarray(minival["target"], dtype=np.float32),
        "endpoints": endpoints,
        "train_endpoints": train_endpoints,
        "public": {
            "session": str(record.session_name),
            "date": str(record.date),
            "calibration_input_sha256": str(record.input_sha256),
            "support_bank": bank_public,
            "support_block_count": len(bank_public),
            "minival_input_sha256": str(minival["path_sha256"]),
            "query_neural_sha256": array_sha256(minival["neural"]),
            "target_stream_sha256": array_sha256(minival["target"]),
            "endpoint_sha256": array_sha256(endpoints),
            "train_endpoint_sha256": array_sha256(train_endpoints),
            "windows": int(endpoints.size),
            "training_windows": int(train_endpoints.size),
        },
    }


def _device_support_bank(row: Mapping[str, Any], *, device: str) -> tuple[tuple[Any, Any], ...]:
    import torch

    result = []
    for entry in row["support_bank"]:
        activity = torch.as_tensor(entry["activity"], dtype=torch.float32, device=device).unsqueeze(0)
        carrier = torch.as_tensor(entry["carrier"], dtype=torch.float32, device=device).unsqueeze(0)
        result.append((activity, carrier))
    return tuple(result)


def _frozen_state_dict(net: Any) -> OrderedDict[str, Any]:
    return OrderedDict(
        (name, value)
        for name, value in net.state_dict().items()
        if not name.startswith("carrier_pre_pool.") and not name.startswith("carrier_post_pool.")
    )


def _branch_module_states(net: Any) -> dict[str, OrderedDict[str, Any]]:
    return {
        "carrier_pre_pool": OrderedDict((name, value.detach().cpu()) for name, value in net.carrier_pre_pool.state_dict().items()),
        "carrier_post_pool": OrderedDict((name, value.detach().cpu()) for name, value in net.carrier_post_pool.state_dict().items()),
    }


def _branch_state_dict(net: Any) -> OrderedDict[str, Any]:
    modules = _branch_module_states(net)
    return OrderedDict(
        (f"{module_name}.{name}", value)
        for module_name, state in modules.items()
        for name, value in state.items()
    )


def _finite_branch_gradients(net: Any) -> tuple[int, int]:
    import torch

    materialized = 0
    nonzero = 0
    for name, parameter in net.named_parameters():
        branch = name.startswith("carrier_pre_pool.") or name.startswith("carrier_post_pool.")
        if branch:
            require(parameter.grad is not None, f"missing branch gradient: {name}")
            require(bool(torch.isfinite(parameter.grad).all()), f"nonfinite branch gradient: {name}")
            materialized += parameter.numel()
            nonzero += int(torch.count_nonzero(parameter.grad).item())
        else:
            require(parameter.grad is None, f"frozen decoder gradient materialized: {name}")
    require(materialized == 58_140 and nonzero > 0, "identity branch has no finite nonzero gradient")
    return materialized, nonzero


def _train_step(
    *,
    net: Any,
    optimizer: Any,
    neural: Any,
    target: Any,
    support: tuple[Any, Any],
    teacher_identity: Any | None,
) -> dict[str, float | int]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    identity = late_identity(net, *support)
    prediction = decode_with_identity(net, neural, identity)[:, -1, :] / PREDICTION_DIVISOR
    task_loss = torch.nn.functional.mse_loss(prediction, target)
    distill = torch.zeros((), dtype=task_loss.dtype, device=task_loss.device)
    if teacher_identity is not None:
        distill = normalized_identity_distill(identity, teacher_identity)
    loss = task_loss + DISTILL_WEIGHT * distill
    require(bool(torch.isfinite(loss)), "training loss is nonfinite")
    loss.backward()
    materialized, nonzero = _finite_branch_gradients(net)
    optimizer.step()
    for parameter in net.parameters():
        require(bool(torch.isfinite(parameter).all()), "model parameter became nonfinite")
    return {
        "loss": float(loss.detach().cpu()),
        "task_loss": float(task_loss.detach().cpu()),
        "distill_loss": float(distill.detach().cpu()),
        "gradient_materialized": materialized,
        "gradient_nonzero": nonzero,
    }


def train_three(base_net: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(rows, "source row roster is empty")
    base_net.eval()
    for parameter in base_net.parameters():
        parameter.requires_grad_(False)
    models = {arm: copy.deepcopy(base_net).to(device) for arm in TRAIN_ARMS}
    trainable = {arm: freeze_decoder_train_identity(model) for arm, model in models.items()}
    initial_hashes = {arm: state_hash(model.state_dict()) for arm, model in models.items()}
    require(len(set(initial_hashes.values())) == 1, "three training clones are not byte-identical")
    frozen_before = {arm: state_hash(_frozen_state_dict(model)) for arm, model in models.items()}
    require(len(set(frozen_before.values())) == 1, "frozen body initial state differs across arms")
    optimizers = {
        arm: torch.optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for arm, parameters in trainable.items()
    }
    device_support = {str(row["session"]): _device_support_bank(row, device=device) for row in rows}

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    total_steps = 0
    first_identity_equal = False
    first_prediction_equal = False
    arm_gradient_steps = {arm: 0 for arm in TRAIN_ARMS}
    first_losses: dict[str, dict[str, float | int]] = {}
    last_losses: dict[str, dict[str, float | int]] = {}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(EPOCHS):
        ordered = list(rows)
        random.Random(f"{SCHEMA}|{SEED}|session-order|{epoch}").shuffle(ordered)
        epoch_loss = {arm: [] for arm in TRAIN_ARMS}
        epoch_task = {arm: [] for arm in TRAIN_ARMS}
        epoch_distill = {arm: [] for arm in TRAIN_ARMS}
        support_counts = {arm: {"first_m3": 0, "nonfirst_uniform": 0} for arm in TRAIN_ARMS}
        unique_blocks: dict[str, dict[str, set[int]]] = {
            arm: {str(row["session"]): set() for row in rows} for arm in TRAIN_ARMS
        }
        epoch_steps = 0
        for row in ordered:
            session = str(row["session"])
            endpoints = np.asarray(row["train_endpoints"], dtype=np.int64).copy()
            token = hashlib.sha256(f"{SCHEMA}|{SEED}|endpoint-order|{epoch}|{session}".encode("utf-8")).digest()
            np.random.default_rng(int.from_bytes(token[:8], "big")).shuffle(endpoints)
            bank = device_support[session]
            for batch_ordinal, offset in enumerate(range(0, endpoints.size, BATCH_SIZE)):
                selected = endpoints[offset : offset + BATCH_SIZE]
                neural = torch.as_tensor(window_batch(row["neural"], selected), dtype=torch.float32, device=device)
                target = torch.as_tensor(row["target_stream"][selected], dtype=torch.float32, device=device)
                fixed_index, fixed_mode = support_index(
                    epoch=epoch, session=session, batch_ordinal=batch_ordinal,
                    block_count=len(bank), random_arm=False,
                )
                random_index, random_mode = support_index(
                    epoch=epoch, session=session, batch_ordinal=batch_ordinal,
                    block_count=len(bank), random_arm=True,
                )
                arm_support = {
                    "LP-F3": bank[fixed_index],
                    "LP-R3": bank[random_index],
                    "SRPD": bank[random_index],
                }
                arm_mode = {"LP-F3": fixed_mode, "LP-R3": random_mode, "SRPD": random_mode}
                arm_index = {"LP-F3": fixed_index, "LP-R3": random_index, "SRPD": random_index}

                if total_steps == 0:
                    require(fixed_index == random_index == 0, "first paired update is not first-M3 anchored")
                    with torch.no_grad():
                        identities = {arm: late_identity(models[arm], *arm_support[arm]) for arm in TRAIN_ARMS}
                        predictions = {
                            arm: decode_with_identity(models[arm], neural, identities[arm])[:, -1, :] / PREDICTION_DIVISOR
                            for arm in TRAIN_ARMS
                        }
                    first_identity_equal = all(torch.equal(identities["LP-F3"], identities[arm]) for arm in TRAIN_ARMS[1:])
                    first_prediction_equal = all(torch.equal(predictions["LP-F3"], predictions[arm]) for arm in TRAIN_ARMS[1:])
                    require(first_identity_equal and first_prediction_equal, "three-arm first-step parity failed")

                with torch.no_grad():
                    teacher = native_identity(base_net, *bank[random_index])
                for arm in TRAIN_ARMS:
                    evidence = _train_step(
                        net=models[arm], optimizer=optimizers[arm], neural=neural, target=target,
                        support=arm_support[arm], teacher_identity=teacher if arm == "SRPD" else None,
                    )
                    if total_steps == 0:
                        first_losses[arm] = evidence
                    last_losses[arm] = evidence
                    epoch_loss[arm].append(float(evidence["loss"]))
                    epoch_task[arm].append(float(evidence["task_loss"]))
                    epoch_distill[arm].append(float(evidence["distill_loss"]))
                    arm_gradient_steps[arm] += int(int(evidence["gradient_nonzero"]) > 0)
                    support_counts[arm][arm_mode[arm]] += 1
                    unique_blocks[arm][session].add(arm_index[arm])
                total_steps += 1
                epoch_steps += 1
        require(epoch_steps > 0, "epoch has no optimizer steps")
        require(support_counts["LP-R3"] == support_counts["SRPD"], "random support count parity failed")
        require(unique_blocks["LP-R3"] == unique_blocks["SRPD"], "random support identity parity failed")
        epoch_rows.append({
            "epoch_zero_based": epoch,
            "steps_per_arm": epoch_steps,
            "mean_loss": {arm: float(np.mean(values, dtype=np.float64)) for arm, values in epoch_loss.items()},
            "mean_task_loss": {arm: float(np.mean(values, dtype=np.float64)) for arm, values in epoch_task.items()},
            "mean_distill_loss": {arm: float(np.mean(values, dtype=np.float64)) for arm, values in epoch_distill.items()},
            "support_counts": support_counts,
            "unique_support_blocks": {
                arm: {session: sorted(values) for session, values in sessions.items()}
                for arm, sessions in unique_blocks.items()
            },
        })

    require(all(count == total_steps for count in arm_gradient_steps.values()), "branch gradient-step count drift")
    torch.cuda.synchronize()
    frozen_after = {arm: state_hash(_frozen_state_dict(model)) for arm, model in models.items()}
    require(frozen_before == frozen_after, "frozen C1 decoder/body changed during branch-only training")
    return models, {
        "epochs": EPOCHS,
        "steps_per_arm": total_steps,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "distill_weight": DISTILL_WEIGHT,
        "initial_state_sha256": initial_hashes,
        "final_state_sha256": {arm: state_hash(model.state_dict()) for arm, model in models.items()},
        "frozen_state_before_sha256": frozen_before,
        "frozen_state_after_sha256": frozen_after,
        "first_identity_bitwise_equal": first_identity_equal,
        "first_prediction_bitwise_equal": first_prediction_equal,
        "gradient_steps": arm_gradient_steps,
        "first_losses": first_losses,
        "last_losses": last_losses,
        "epoch_rows": epoch_rows,
    }


def _save_branch_checkpoint(path: Path, *, outer_date: str, arm: str, net: Any) -> dict[str, Any]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(not path.exists(), "checkpoint path already exists")
    modules = _branch_module_states(net)
    payload = {
        "schema": f"{SCHEMA}_branch_checkpoint",
        "outer_date": outer_date,
        "arm": arm,
        "carrier_pre_pool": modules["carrier_pre_pool"],
        "carrier_post_pool": modules["carrier_post_pool"],
        "branch_state_sha256": state_hash(_branch_state_dict(net)),
        "full_state_sha256": state_hash(net.state_dict()),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    os.chmod(path, 0o444)
    digest = _sha256_file(path)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return {"path": path.name, "sha256": digest, "branch_state_sha256": payload["branch_state_sha256"], "full_state_sha256": payload["full_state_sha256"]}


def _load_branch_checkpoint(base_net: Any, path: Path, *, expected: Mapping[str, Any], device: str) -> Any:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(_sha256_file(path) == expected["sha256"], "branch checkpoint byte digest drift")
    payload = torch.load(path, map_location=device, weights_only=True)
    require(payload["schema"] == f"{SCHEMA}_branch_checkpoint", "branch checkpoint schema drift")
    model = copy.deepcopy(base_net).to(device)
    freeze_decoder_train_identity(model)
    model.carrier_pre_pool.load_state_dict(payload["carrier_pre_pool"], strict=True)
    model.carrier_post_pool.load_state_dict(payload["carrier_post_pool"], strict=True)
    require(state_hash(_branch_state_dict(model)) == payload["branch_state_sha256"] == expected["branch_state_sha256"], "branch checkpoint state drift")
    require(state_hash(model.state_dict()) == payload["full_state_sha256"] == expected["full_state_sha256"], "strict-reloaded full state drift")
    return model


def _predict(net: Any, row: Mapping[str, Any], *, device: str, late: bool) -> tuple[np.ndarray, float]:
    import torch

    net.eval()
    support = _device_support_bank(row, device=device)[0]
    with torch.inference_mode():
        identity = late_identity(net, *support) if late else native_identity(net, *support)
        endpoints = np.asarray(row["endpoints"], dtype=np.int64)
        output = np.empty((endpoints.size, 7), dtype=np.float32)
        for offset in range(0, endpoints.size, BATCH_SIZE):
            selected = endpoints[offset : offset + BATCH_SIZE]
            neural = torch.as_tensor(window_batch(row["neural"], selected), dtype=torch.float32, device=device)
            prediction = decode_with_identity(net, neural, identity)[:, -1, :] / PREDICTION_DIVISOR
            output[offset : offset + selected.size] = prediction.cpu().numpy()
    target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
    return output, _r2(output, target)


def _score(base: Any, models: Mapping[str, Any], rows: list[Mapping[str, Any]], *, device: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values = {arm: [] for arm in ARMS}
    public: list[dict[str, Any]] = []
    for row in rows:
        predictions: dict[str, np.ndarray] = {}
        for arm in ARMS:
            prediction, score = _predict(base if arm == "FROZEN-C1" else models[arm], row, device=device, late=arm != "FROZEN-C1")
            predictions[arm] = prediction
            values[arm].append(score)
        target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
        public.append({
            **row["public"],
            "target_sha256": array_sha256(target),
            "r2": {arm: values[arm][-1] for arm in ARMS},
            "prediction_sha256": {arm: array_sha256(predictions[arm]) for arm in ARMS},
        })
    return {arm: sum(scores) / len(scores) for arm, scores in values.items()}, public


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
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    data_root = root / DATA_RELATIVE
    indexed = index_heldin_calib(data_root)
    source_names = tuple(plan.source_sessions)
    require(source_names and all(name in indexed and not name.startswith(f"ses-{outer_date}") for name in source_names), "fold source roster drift")
    source_rows = [
        _session_row(
            record=load_record(indexed[session]),
            minival=_load_minival(data_root, session),
            plan=plan,
            s_src=s_src,
            all_blocks=True,
        )
        for session in source_names
    ]
    models, training = train_three(base, source_rows, device=device)
    checkpoints = {
        arm: _save_branch_checkpoint(
            Path(receipt_root) / f"checkpoint_{outer_date}_{arm.lower()}.pt",
            outer_date=outer_date,
            arm=arm,
            net=models[arm],
        )
        for arm in TRAIN_ARMS
    }
    training_sha = _publish(Path(receipt_root) / f"training_{outer_date}.json", {
        "schema": f"{SCHEMA}_training",
        "status": "TRAINING_AND_CHECKPOINTS_FROZEN_BEFORE_OUTER_DATE_OPEN",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "source_rows": [row["public"] for row in source_rows],
        "training": training,
        "checkpoints": checkpoints,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "outer_date_calib_opened": False,
        "outer_date_minival_opened": False,
    })
    del models
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    models = {
        arm: _load_branch_checkpoint(
            base, Path(receipt_root) / checkpoints[arm]["path"], expected=checkpoints[arm], device=device,
        )
        for arm in TRAIN_ARMS
    }

    target_records = load_outer_date_target_records(data_root, outer_date=outer_date)
    target_rows = [
        _session_row(
            record=record,
            minival=_load_minival(data_root, session),
            plan=plan,
            s_src=s_src,
            all_blocks=False,
        )
        for session, record in target_records.items()
    ]
    before = {"FROZEN-C1": state_hash(base.state_dict()), **{arm: state_hash(model.state_dict()) for arm, model in models.items()}}
    scores, rows = _score(base, models, target_rows, device=device)
    after = {"FROZEN-C1": state_hash(base.state_dict()), **{arm: state_hash(model.state_dict()) for arm, model in models.items()}}
    require(before == after and before["FROZEN-C1"] == base_state, "model changed during outer-date scoring")
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_BRANCH_ONLY_TRAINING_AND_OUTER_SCORE",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions": list(target_records),
        "training_sha256": training_sha,
        "checkpoints": checkpoints,
        "scores": scores,
        "gains_vs_frozen": {arm: scores[arm] - scores["FROZEN-C1"] for arm in TRAIN_ARMS},
        "support_diversity": scores["LP-R3"] - scores["LP-F3"],
        "distillation_increment": scores["SRPD"] - scores["LP-R3"],
        "target_rows": rows,
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
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
            "H1 SRPD requires isolated logical GPU0")
    folds: list[dict[str, Any]] = []
    for outer_date in DATE_ORDER:
        folds.append(run_fold(repo_root, outer_date, device=device, receipt_root=receipt_root))
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_SUPPORT_RESAMPLED_POSTPOOL_SOURCE_OOF",
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


__all__ = ("run", "run_fold", "train_three")
