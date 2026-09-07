"""Train the two missing cells of the H1 activity/carrier 2x2 factorial."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.core import array_sha256
from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _load_minival
from h1_m3_crossrecord_joint_v1.core import window_batch
from h1_support_resampled_postpool_v1.core import freeze_decoder_train_identity, late_identity, require, support_index
from h1_support_resampled_postpool_v1.evaluate import (
    _device_support_bank,
    _frozen_state_dict,
    _load_branch_checkpoint,
    _predict,
    _publish,
    _save_branch_checkpoint,
    _session_row,
    _train_step,
)
from h1_support_resampled_postpool_v1.plan import (
    BATCH_SIZE,
    DATE_ORDER,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    TRAIN_STRIDE,
    WEIGHT_DECAY,
)
from .plan import NEW_ARMS, SCHEMA, decide_factorial


MAIN_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_support_resampled_postpool_v1"


def crossed_supports(fixed: tuple[Any, Any], random_support: tuple[Any, Any]) -> dict[str, tuple[Any, Any]]:
    """Activity-first/carrier-second factorial cells."""

    return {
        "LP-AR-CF": (random_support[0], fixed[1]),
        "LP-AF-CR": (fixed[0], random_support[1]),
    }


def train_two(base_net: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(rows, "factorial source row roster is empty")
    base_net.eval()
    for parameter in base_net.parameters():
        parameter.requires_grad_(False)
    models = {arm: copy.deepcopy(base_net).to(device) for arm in NEW_ARMS}
    trainable = {arm: freeze_decoder_train_identity(model) for arm, model in models.items()}
    initial = {arm: state_hash(model.state_dict()) for arm, model in models.items()}
    require(len(set(initial.values())) == 1, "factorial initial clones differ")
    frozen_before = {arm: state_hash(_frozen_state_dict(model)) for arm, model in models.items()}
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
    epoch_rows: list[dict[str, Any]] = []
    gradient_steps = {arm: 0 for arm in NEW_ARMS}

    for epoch in range(EPOCHS):
        ordered = list(rows)
        random.Random(f"h1_support_resampled_postpool_v1|{SEED}|session-order|{epoch}").shuffle(ordered)
        losses = {arm: [] for arm in NEW_ARMS}
        support_counts = {"first_m3": 0, "nonfirst_uniform": 0}
        unique_indices = {str(row["session"]): set() for row in rows}
        epoch_steps = 0
        for row in ordered:
            session = str(row["session"])
            endpoints = np.asarray(row["train_endpoints"], dtype=np.int64).copy()
            token = hashlib.sha256(
                f"h1_support_resampled_postpool_v1|{SEED}|endpoint-order|{epoch}|{session}".encode("utf-8")
            ).digest()
            np.random.default_rng(int.from_bytes(token[:8], "big")).shuffle(endpoints)
            bank = device_support[session]
            for batch_ordinal, offset in enumerate(range(0, endpoints.size, BATCH_SIZE)):
                selected = endpoints[offset : offset + BATCH_SIZE]
                neural = torch.as_tensor(window_batch(row["neural"], selected), dtype=torch.float32, device=device)
                target = torch.as_tensor(row["target_stream"][selected], dtype=torch.float32, device=device)
                index, mode = support_index(
                    epoch=epoch, session=session, batch_ordinal=batch_ordinal,
                    block_count=len(bank), random_arm=True,
                )
                supports = crossed_supports(bank[0], bank[index])
                support_counts[mode] += 1
                unique_indices[session].add(index)
                if total_steps == 0:
                    require(index == 0, "factorial first update is not first-M3 anchored")
                    with torch.no_grad():
                        identities = {arm: late_identity(models[arm], *supports[arm]) for arm in NEW_ARMS}
                    first_identity_equal = torch.equal(identities[NEW_ARMS[0]], identities[NEW_ARMS[1]])
                    with torch.no_grad():
                        from h1_m3_crossrecord_joint_v1.core import decode_with_identity
                        predictions = {
                            arm: decode_with_identity(models[arm], neural, identities[arm])[:, -1]
                            for arm in NEW_ARMS
                        }
                    first_prediction_equal = torch.equal(predictions[NEW_ARMS[0]], predictions[NEW_ARMS[1]])
                    require(first_identity_equal and first_prediction_equal, "factorial first-step parity failed")
                for arm in NEW_ARMS:
                    evidence = _train_step(
                        net=models[arm], optimizer=optimizers[arm], neural=neural, target=target,
                        support=supports[arm], teacher_identity=None,
                    )
                    losses[arm].append(float(evidence["task_loss"]))
                    gradient_steps[arm] += int(int(evidence["gradient_nonzero"]) > 0)
                total_steps += 1
                epoch_steps += 1
        require(epoch_steps > 0, "factorial epoch has no steps")
        epoch_rows.append({
            "epoch_zero_based": epoch,
            "steps_per_arm": epoch_steps,
            "mean_task_loss": {arm: float(np.mean(value, dtype=np.float64)) for arm, value in losses.items()},
            "support_counts": support_counts,
            "unique_random_block_indices": {session: sorted(values) for session, values in unique_indices.items()},
        })

    require(all(value == total_steps for value in gradient_steps.values()), "factorial gradient-step count drift")
    torch.cuda.synchronize()
    frozen_after = {arm: state_hash(_frozen_state_dict(model)) for arm, model in models.items()}
    require(frozen_before == frozen_after, "factorial frozen decoder/body changed")
    return models, {
        "epochs": EPOCHS,
        "steps_per_arm": total_steps,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "initial_state_sha256": initial,
        "final_state_sha256": {arm: state_hash(model.state_dict()) for arm, model in models.items()},
        "frozen_state_before_sha256": frozen_before,
        "frozen_state_after_sha256": frozen_after,
        "first_identity_bitwise_equal": bool(first_identity_equal),
        "first_prediction_bitwise_equal": bool(first_prediction_equal),
        "gradient_steps": gradient_steps,
        "epoch_rows": epoch_rows,
    }


def _read_main_fold_after_new_score(root: Path, outer_date: str, public_rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    path = root / MAIN_ROOT_RELATIVE / f"fold_{outer_date}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    side = path.with_name(path.name + ".sha256").read_text(encoding="ascii").strip().split()
    require(side == [digest, path.name], "main fold sidecar drift")
    main = json.loads(path.read_text(encoding="utf-8"))
    require(main["outer_date"] == outer_date, "main fold outer-date drift")
    old_rows = main["target_rows"]
    require([row["session"] for row in old_rows] == [row["session"] for row in public_rows], "target session order differs from main experiment")
    for old, new in zip(old_rows, public_rows):
        require(old["target_sha256"] == new["target_sha256"], "target digest differs from main experiment")
        require(old["endpoint_sha256"] == new["endpoint_sha256"], "endpoint digest differs from main experiment")
        require(old["support_bank"][0] == new["support_bank"][0], "first-M3 support differs from main experiment")
        require(old["prediction_sha256"]["FROZEN-C1"] == new["prediction_sha256"]["FROZEN-C1"], "frozen C1 prediction differs from main experiment")
    return main, digest


def _score_new(base: Any, models: Mapping[str, Any], rows: list[Mapping[str, Any]], *, device: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    scores = {"FROZEN-C1": [], **{arm: [] for arm in NEW_ARMS}}
    public: list[dict[str, Any]] = []
    for row in rows:
        predictions: dict[str, np.ndarray] = {}
        base_prediction, base_score = _predict(base, row, device=device, late=False)
        predictions["FROZEN-C1"] = base_prediction
        scores["FROZEN-C1"].append(base_score)
        for arm in NEW_ARMS:
            prediction, value = _predict(models[arm], row, device=device, late=True)
            predictions[arm] = prediction
            scores[arm].append(value)
        target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
        public.append({
            **row["public"],
            "target_sha256": array_sha256(target),
            "r2": {arm: scores[arm][-1] for arm in scores},
            "prediction_sha256": {arm: array_sha256(prediction) for arm, prediction in predictions.items()},
        })
    return {arm: sum(values) / len(values) for arm, values in scores.items()}, public


def run_fold(repo_root: Path, outer_date: str, *, device: str, receipt_root: Path) -> dict[str, Any]:
    from h1_causal_activity_completion_v1.stage1 import ARTIFACT_RELATIVE as C1_ARTIFACT, C1_AUTHORITIES, _load_model, _load_plan
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve()
    directory = root / C1_ARTIFACT / outer_date
    authority = C1_AUTHORITIES[outer_date]
    plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    base, base_state, model_receipt = _load_model(directory, authority, outer_date, device)
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    indexed = index_heldin_calib(root / DATA_RELATIVE)
    source_names = tuple(plan.source_sessions)
    source_rows = [
        _session_row(
            record=load_record(indexed[session]),
            minival=_load_minival(root / DATA_RELATIVE, session),
            plan=plan, s_src=s_src, all_blocks=True,
        )
        for session in source_names
    ]
    models, training = train_two(base, source_rows, device=device)
    checkpoints = {
        arm: _save_branch_checkpoint(
            Path(receipt_root) / f"checkpoint_{outer_date}_{arm.lower()}.pt",
            outer_date=outer_date, arm=arm, net=models[arm],
        )
        for arm in NEW_ARMS
    }
    training_sha = _publish(Path(receipt_root) / f"training_{outer_date}.json", {
        "schema": f"{SCHEMA}_training",
        "status": "TWO_CROSSED_CELLS_FROZEN_BEFORE_OUTER_DATE_OPEN",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "source_rows": [row["public"] for row in source_rows],
        "training": training,
        "checkpoints": checkpoints,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "outer_date_opened": False,
    })
    del models
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    models = {
        arm: _load_branch_checkpoint(base, Path(receipt_root) / checkpoints[arm]["path"], expected=checkpoints[arm], device=device)
        for arm in NEW_ARMS
    }

    target_records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=outer_date)
    target_rows = [
        _session_row(
            record=record,
            minival=_load_minival(root / DATA_RELATIVE, session),
            plan=plan, s_src=s_src, all_blocks=False,
        )
        for session, record in target_records.items()
    ]
    before = {"FROZEN-C1": state_hash(base.state_dict()), **{arm: state_hash(model.state_dict()) for arm, model in models.items()}}
    new_scores, public_rows = _score_new(base, models, target_rows, device=device)
    after = {"FROZEN-C1": state_hash(base.state_dict()), **{arm: state_hash(model.state_dict()) for arm, model in models.items()}}
    require(before == after and before["FROZEN-C1"] == base_state, "factorial target scoring changed model state")
    main, main_digest = _read_main_fold_after_new_score(root, outer_date, public_rows)
    require(abs(new_scores["FROZEN-C1"] - float(main["scores"]["FROZEN-C1"])) <= 1e-12, "frozen C1 R2 differs from main experiment")
    factorial = {
        "FF": float(main["scores"]["LP-F3"]),
        "RR": float(main["scores"]["LP-R3"]),
        "RF": new_scores["LP-AR-CF"],
        "FR": new_scores["LP-AF-CR"],
    }
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_TWO_CROSSED_CELLS_AND_BOUND_MAIN_FACTORIAL",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions": list(target_records),
        "training_sha256": training_sha,
        "checkpoints": checkpoints,
        "new_scores": new_scores,
        "factorial_scores": factorial,
        "main_fold_sha256": main_digest,
        "target_rows": public_rows,
        "model_state_before_sha256": before,
        "model_state_after_sha256": after,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    fold["fold_sha256"] = _publish(Path(receipt_root) / f"fold_{outer_date}.json", fold)
    return fold


def run(repo_root: Path, *, device: str, receipt_root: Path) -> dict[str, Any]:
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "factorial requires isolated logical GPU0")
    folds: list[dict[str, Any]] = []
    for outer_date in DATE_ORDER:
        folds.append(run_fold(repo_root, outer_date, device=device, receipt_root=receipt_root))
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_ACTIVITY_CARRIER_RESAMPLING_FACTORIAL",
        "date_order": list(DATE_ORDER),
        "new_arms": list(NEW_ARMS),
        "folds": folds,
        "decision": decide_factorial(folds),
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }


__all__ = ("crossed_supports", "run", "run_fold", "train_two")
