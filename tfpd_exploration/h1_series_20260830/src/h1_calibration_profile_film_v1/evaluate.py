"""Paired five-fold H1 CP-FiLM training and exact-anchor scoring."""
from __future__ import annotations

from collections import OrderedDict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import stat
from typing import Any, Mapping

import numpy as np

from h1_cross_record_postpool_v1.core import array_sha256
from h1_cross_record_postpool_v1.evaluate import DATA_RELATIVE, _load_minival, _r2
from h1_m3_crossrecord_joint_v1.core import decode_with_identity, window_batch
from h1_support_resampled_postpool_v1.core import late_identity, native_identity, support_index
from h1_support_resampled_postpool_v1.evaluate import (
    _load_branch_checkpoint,
    _predict as _predict_zero,
    _session_row as _base_session_row,
)

from .core import build_film, film_identity, profile_from_support, require
from .plan import (
    ARMS,
    BATCH_SIZE,
    DATE_ORDER,
    EPOCHS,
    FILM_ARMS,
    FILM_PARAMETERS,
    LEARNING_RATE,
    PREDICTION_DIVISOR,
    PREDECESSOR_ROOT_RELATIVE,
    PREDECESSOR_SCORE_SHA256,
    PREDECESSOR_TERMINAL_SHA256,
    SCHEMA,
    SEED,
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


def _verify_immutable(path: Path, expected: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing/symlinked authority: {path}")
    info = path.stat()
    require(stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
            f"authority must be immutable 0444/nlink1: {path}")
    observed = _sha256_file(path)
    require(observed == expected, f"authority SHA drift: {path}")
    side = path.with_name(path.name + ".sha256")
    require(side.is_file() and not side.is_symlink(), f"authority sidecar missing: {side}")
    side_info = side.stat()
    require(stat.S_IMODE(side_info.st_mode) == 0o444 and side_info.st_nlink == 1,
            f"authority sidecar must be immutable 0444/nlink1: {side}")
    require(side.read_text(encoding="ascii") == f"{observed}  {path.name}\n", f"authority sidecar drift: {side}")
    return observed


def _publish(path: Path, value: Mapping[str, Any]) -> str:
    from src.h1_m4_cce_contract import write_immutable_json

    _, digest = write_immutable_json(path, value)
    side = path.with_name(path.name + ".sha256")
    with side.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    os.chmod(side, 0o444)
    return digest


def load_predecessor(repo_root: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    root = Path(repo_root).resolve() / PREDECESSOR_ROOT_RELATIVE
    score_path, terminal_path = root / "score.json", root / "terminal.json"
    _verify_immutable(score_path, PREDECESSOR_SCORE_SHA256)
    _verify_immutable(terminal_path, PREDECESSOR_TERMINAL_SHA256)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    require(
        score.get("schema") == "h1_support_resampled_postpool_v1_score"
        and score.get("status") == "COMPLETE_H1_SUPPORT_RESAMPLED_POSTPOOL_SOURCE_OOF"
        and score.get("date_order") == list(DATE_ORDER)
        and score.get("decision", {}).get("selected_arm") == "LP-R3"
        and score.get("formal_heldout_opened") is False
        and score.get("evalai_opened") is False,
        "LP-R3 predecessor score semantics drift",
    )
    require(
        terminal.get("schema") == "h1_support_resampled_postpool_v1_terminal"
        and terminal.get("score_sha256") == PREDECESSOR_SCORE_SHA256
        and terminal.get("decision", {}).get("selected_arm") == "LP-R3"
        and terminal.get("formal_heldout_opened") is False
        and terminal.get("evalai_opened") is False,
        "LP-R3 predecessor terminal semantics drift",
    )
    folds = score.get("folds")
    require(isinstance(folds, list) and tuple(str(row.get("outer_date")) for row in folds) == DATE_ORDER,
            "LP-R3 predecessor fold order drift")
    by_date: dict[str, Mapping[str, Any]] = {}
    for row in folds:
        outer = str(row["outer_date"])
        checkpoint = row.get("checkpoints", {}).get("LP-R3")
        require(isinstance(checkpoint, Mapping), f"{outer}: LP-R3 checkpoint authority missing")
        _verify_immutable(root / str(checkpoint["path"]), str(checkpoint["sha256"]))
        require(row.get("target_optimizer_steps") == row.get("target_backward_steps") == row.get("target_model_updates") == 0,
                f"{outer}: predecessor target-update contract drift")
        by_date[outer] = row
    return by_date, {
        "root": PREDECESSOR_ROOT_RELATIVE,
        "score_sha256": PREDECESSOR_SCORE_SHA256,
        "terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
        "lp_r3_checkpoints": {date: dict(by_date[date]["checkpoints"]["LP-R3"]) for date in DATE_ORDER},
    }


def _profile_session_row(*, record: Any, minival: Mapping[str, Any], source_plan: Any,
                         s_src: float, all_blocks: bool) -> dict[str, Any]:
    row = _base_session_row(record=record, minival=minival, plan=source_plan, s_src=s_src, all_blocks=all_blocks)
    require(len(row["support_bank"]) == len(row["public"]["support_bank"]), "support/public bank length drift")
    for runtime, public in zip(row["support_bank"], row["public"]["support_bank"], strict=True):
        profile, evidence = profile_from_support(record, runtime["support_trials"])
        runtime["profile"] = profile
        raw_profile = np.ascontiguousarray(evidence.pop("raw_profile"), dtype=np.float32)
        public.update({
            "task_profile_sha256": array_sha256(profile),
            "task_profile_full_sha256": array_sha256(raw_profile),
            "task_profile": evidence,
        })
    return row


def _device_support_bank(row: Mapping[str, Any], *, device: str) -> tuple[tuple[Any, Any, Any], ...]:
    import torch

    result = []
    for entry in row["support_bank"]:
        result.append((
            torch.as_tensor(entry["activity"], dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(entry["carrier"], dtype=torch.float32, device=device).unsqueeze(0),
            torch.as_tensor(entry["profile"], dtype=torch.float32, device=device).unsqueeze(0),
        ))
    return tuple(result)


def _without_identity_branch(net: Any) -> OrderedDict[str, Any]:
    return OrderedDict(
        (name, value)
        for name, value in net.state_dict().items()
        if not name.startswith("carrier_pre_pool.") and not name.startswith("carrier_post_pool.")
    )


def _film_gradient_evidence(film: Any, *, name: str) -> dict[str, int]:
    import torch

    materialized = nonzero = 0
    for parameter_name, parameter in film.named_parameters():
        require(parameter.grad is not None, f"{name}: missing FiLM gradient {parameter_name}")
        require(bool(torch.isfinite(parameter.grad).all()), f"{name}: nonfinite FiLM gradient {parameter_name}")
        materialized += parameter.numel()
        nonzero += int(torch.count_nonzero(parameter.grad).item())
    require(materialized == FILM_PARAMETERS and nonzero > 0, f"{name}: FiLM gradient surface is empty")
    return {"materialized": materialized, "nonzero": nonzero}


def train_pair(early_net: Any, late_net: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(rows, "CP-FiLM source roster is empty")
    for net in (early_net, late_net):
        net.eval()
        for parameter in net.parameters():
            parameter.requires_grad_(False)
    early_body = state_hash(_without_identity_branch(early_net))
    late_body = state_hash(_without_identity_branch(late_net))
    require(early_body == late_body, "early/late frozen decoder bodies differ")
    frozen_before = {
        "early": state_hash(early_net.state_dict()),
        "late": state_hash(late_net.state_dict()),
        "decoder_body": early_body,
    }

    torch.manual_seed(SEED)
    template = build_film().to(device)
    films = {"EP-FILM": copy.deepcopy(template), "LP-FILM": copy.deepcopy(template)}
    initial = {name: state_hash(module.state_dict()) for name, module in films.items()}
    require(len(set(initial.values())) == 1, "paired FiLM initial states differ")
    optimizers = {
        name: torch.optim.Adam(module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for name, module in films.items()
    }
    supports = {str(row["session"]): _device_support_bank(row, device=device) for row in rows}

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    total_steps = 0
    gradient_steps = {name: 0 for name in FILM_ARMS}
    first_identity_equal = {name: False for name in FILM_ARMS}
    first_prediction_equal = {name: False for name in FILM_ARMS}
    first_losses: dict[str, float] = {}
    last_losses: dict[str, float] = {}
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(EPOCHS):
        ordered = list(rows)
        random.Random(f"{SCHEMA}|{SEED}|session-order|{epoch}").shuffle(ordered)
        losses = {name: [] for name in FILM_ARMS}
        support_counts = {"first_m3": 0, "nonfirst_uniform": 0}
        unique_blocks = {str(row["session"]): set() for row in rows}
        epoch_steps = 0
        for row in ordered:
            session = str(row["session"])
            endpoints = np.asarray(row["train_endpoints"], dtype=np.int64).copy()
            token = hashlib.sha256(f"{SCHEMA}|{SEED}|endpoint-order|{epoch}|{session}".encode("utf-8")).digest()
            np.random.default_rng(int.from_bytes(token[:8], "big")).shuffle(endpoints)
            bank = supports[session]
            for batch_ordinal, offset in enumerate(range(0, endpoints.size, BATCH_SIZE)):
                selected = endpoints[offset : offset + BATCH_SIZE]
                neural = torch.as_tensor(window_batch(row["neural"], selected), dtype=torch.float32, device=device)
                target = torch.as_tensor(row["target_stream"][selected], dtype=torch.float32, device=device)
                support_position, support_mode = support_index(
                    epoch=epoch, session=session, batch_ordinal=batch_ordinal,
                    block_count=len(bank), random_arm=True,
                )
                activity, carrier, profile = bank[support_position]

                if total_steps == 0:
                    require(support_position == 0 and support_mode == "first_m3", "first update is not first-M3 anchored")
                    with torch.no_grad():
                        early_zero = native_identity(early_net, activity, carrier)
                        late_zero = late_identity(late_net, activity, carrier)
                        early_film = film_identity(early_net, activity, carrier, profile, films["EP-FILM"], late=False)
                        late_film = film_identity(late_net, activity, carrier, profile, films["LP-FILM"], late=True)
                        first_identity_equal["EP-FILM"] = bool(torch.equal(early_zero, early_film))
                        first_identity_equal["LP-FILM"] = bool(torch.equal(late_zero, late_film))
                        ep0 = decode_with_identity(early_net, neural, early_zero)
                        epf = decode_with_identity(early_net, neural, early_film)
                        lp0 = decode_with_identity(late_net, neural, late_zero)
                        lpf = decode_with_identity(late_net, neural, late_film)
                        first_prediction_equal["EP-FILM"] = bool(torch.equal(ep0, epf))
                        first_prediction_equal["LP-FILM"] = bool(torch.equal(lp0, lpf))
                    require(all(first_identity_equal.values()) and all(first_prediction_equal.values()),
                            "zero-init FiLM does not reproduce its substrate")

                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
                early_identity = film_identity(early_net, activity, carrier, profile, films["EP-FILM"], late=False)
                late_identity_value = film_identity(late_net, activity, carrier, profile, films["LP-FILM"], late=True)
                count = neural.shape[0]
                packed_neural = torch.cat((neural, neural), dim=0)
                packed_identity = torch.cat((
                    early_identity.expand(count, -1, -1),
                    late_identity_value.expand(count, -1, -1),
                ), dim=0)
                prediction = decode_with_identity(early_net, packed_neural, packed_identity)[:, -1, :] / PREDICTION_DIVISOR
                early_loss = torch.nn.functional.mse_loss(prediction[:count], target)
                late_loss = torch.nn.functional.mse_loss(prediction[count:], target)
                total = early_loss + late_loss
                require(bool(torch.isfinite(total)), "CP-FiLM paired task loss is nonfinite")
                total.backward()
                for name, module in films.items():
                    evidence = _film_gradient_evidence(module, name=name)
                    gradient_steps[name] += int(evidence["nonzero"] > 0)
                for name, parameter in tuple(early_net.named_parameters()) + tuple(late_net.named_parameters()):
                    require(parameter.grad is None, f"frozen substrate gradient materialized: {name}")
                for optimizer in optimizers.values():
                    optimizer.step()
                for module in films.values():
                    require(all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters()),
                            "FiLM parameter became nonfinite")
                numeric = {"EP-FILM": float(early_loss.detach().cpu()), "LP-FILM": float(late_loss.detach().cpu())}
                for name, value in numeric.items():
                    if total_steps == 0:
                        first_losses[name] = value
                    last_losses[name] = value
                    losses[name].append(value)
                support_counts[support_mode] += 1
                unique_blocks[session].add(support_position)
                total_steps += 1
                epoch_steps += 1
        require(epoch_steps > 0, "CP-FiLM epoch has no updates")
        epoch_rows.append({
            "epoch_zero_based": epoch,
            "steps_per_arm": epoch_steps,
            "mean_task_loss": {name: float(np.mean(values, dtype=np.float64)) for name, values in losses.items()},
            "support_counts": support_counts,
            "unique_support_blocks": {name: sorted(values) for name, values in unique_blocks.items()},
            "film_state_sha256": {name: state_hash(module.state_dict()) for name, module in films.items()},
        })

    require(all(value == total_steps for value in gradient_steps.values()), "FiLM gradient-step coverage drift")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    frozen_after = {
        "early": state_hash(early_net.state_dict()),
        "late": state_hash(late_net.state_dict()),
        "decoder_body": state_hash(_without_identity_branch(early_net)),
    }
    require(frozen_before == frozen_after, "frozen H1 substrate changed during FiLM training")
    return films, {
        "epochs": EPOCHS,
        "steps_per_arm": total_steps,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "initial_film_state_sha256": initial,
        "final_film_state_sha256": {name: state_hash(module.state_dict()) for name, module in films.items()},
        "frozen_substrate_before_sha256": frozen_before,
        "frozen_substrate_after_sha256": frozen_after,
        "first_identity_bitwise_equal": first_identity_equal,
        "first_prediction_bitwise_equal": first_prediction_equal,
        "gradient_steps": gradient_steps,
        "first_losses": first_losses,
        "last_losses": last_losses,
        "epoch_rows": epoch_rows,
        "packed_decoder_forward": "2B_EP_FILM_THEN_LP_FILM",
    }


def _save_film(path: Path, *, outer_date: str, arm: str, module: Any, base_state_sha256: str) -> dict[str, Any]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(arm in FILM_ARMS and not path.exists(), "illegal/existing FiLM checkpoint path")
    payload = {
        "schema": f"{SCHEMA}_film_checkpoint",
        "outer_date": outer_date,
        "arm": arm,
        "base_state_sha256": base_state_sha256,
        "film_state_sha256": state_hash(module.state_dict()),
        "state_dict": OrderedDict((name, value.detach().cpu()) for name, value in module.state_dict().items()),
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
    return {
        "path": path.name,
        "sha256": digest,
        "film_state_sha256": payload["film_state_sha256"],
        "base_state_sha256": base_state_sha256,
    }


def _load_film(path: Path, *, expected: Mapping[str, Any], outer_date: str, arm: str, device: str) -> Any:
    import torch
    from src.h1_m4_cce_contract import state_hash

    _verify_immutable(path, str(expected["sha256"]))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        payload.get("schema") == f"{SCHEMA}_film_checkpoint"
        and payload.get("outer_date") == outer_date
        and payload.get("arm") == arm
        and payload.get("base_state_sha256") == expected["base_state_sha256"],
        "FiLM checkpoint metadata drift",
    )
    module = build_film()
    incompatible = module.load_state_dict(payload["state_dict"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, "FiLM strict load drift")
    require(state_hash(module.state_dict()) == payload["film_state_sha256"] == expected["film_state_sha256"],
            "FiLM checkpoint state digest drift")
    module.to(device).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _predict_film(net: Any, film: Any, row: Mapping[str, Any], *, device: str, late: bool) -> tuple[np.ndarray, float]:
    import torch

    net.eval()
    film.eval()
    support = _device_support_bank(row, device=device)[0]
    with torch.inference_mode():
        identity = film_identity(net, *support, film, late=late)
        endpoints = np.asarray(row["endpoints"], dtype=np.int64)
        output = np.empty((endpoints.size, 7), dtype=np.float32)
        for offset in range(0, endpoints.size, BATCH_SIZE):
            selected = endpoints[offset : offset + BATCH_SIZE]
            neural = torch.as_tensor(window_batch(row["neural"], selected), dtype=torch.float32, device=device)
            prediction = decode_with_identity(net, neural, identity)[:, -1, :] / PREDICTION_DIVISOR
            output[offset : offset + selected.size] = prediction.cpu().numpy()
    target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
    return output, _r2(output, target)


def _score(early_net: Any, late_net: Any, films: Mapping[str, Any], rows: list[Mapping[str, Any]],
           predecessor_rows: list[Mapping[str, Any]], *, device: str,
           require_historical_prediction_sha: bool = True,
           anchor_r2_tolerance: float = 1.0e-12) -> tuple[dict[str, float], list[dict[str, Any]]]:
    require(len(rows) == len(predecessor_rows), "target row count differs from LP-R3 predecessor")
    predecessor = {str(row["session"]): row for row in predecessor_rows}
    scores = {arm: [] for arm in ARMS}
    public: list[dict[str, Any]] = []
    for row in rows:
        session = str(row["session"])
        require(session in predecessor, f"missing predecessor target row: {session}")
        old = predecessor[session]
        ep_zero, ep_zero_r2 = _predict_zero(early_net, row, device=device, late=False)
        lp_zero, lp_zero_r2 = _predict_zero(late_net, row, device=device, late=True)
        ep_repeat, ep_repeat_r2 = _predict_zero(early_net, row, device=device, late=False)
        lp_repeat, lp_repeat_r2 = _predict_zero(late_net, row, device=device, late=True)
        require(array_sha256(ep_zero) == array_sha256(ep_repeat) and ep_zero_r2 == ep_repeat_r2,
                f"{session}: EP-ZERO same-process repeat drift")
        require(array_sha256(lp_zero) == array_sha256(lp_repeat) and lp_zero_r2 == lp_repeat_r2,
                f"{session}: LP-ZERO same-process repeat drift")
        ep_film, ep_film_r2 = _predict_film(early_net, films["EP-FILM"], row, device=device, late=False)
        lp_film, lp_film_r2 = _predict_film(late_net, films["LP-FILM"], row, device=device, late=True)
        predictions = {
            "EP-ZERO": ep_zero,
            "EP-FILM": ep_film,
            "LP-ZERO": lp_zero,
            "LP-FILM": lp_film,
        }
        values = {
            "EP-ZERO": ep_zero_r2,
            "EP-FILM": ep_film_r2,
            "LP-ZERO": lp_zero_r2,
            "LP-FILM": lp_film_r2,
        }
        ep_historical_sha_equal = array_sha256(ep_zero) == old["prediction_sha256"]["FROZEN-C1"]
        lp_historical_sha_equal = array_sha256(lp_zero) == old["prediction_sha256"]["LP-R3"]
        if require_historical_prediction_sha:
            require(ep_historical_sha_equal, f"{session}: EP-ZERO prediction anchor drift")
            require(lp_historical_sha_equal, f"{session}: LP-ZERO prediction anchor drift")
        require(abs(ep_zero_r2 - float(old["r2"]["FROZEN-C1"])) <= anchor_r2_tolerance,
                f"{session}: EP-ZERO R2 anchor drift")
        require(abs(lp_zero_r2 - float(old["r2"]["LP-R3"])) <= anchor_r2_tolerance,
                f"{session}: LP-ZERO R2 anchor drift")
        target = np.ascontiguousarray(row["target_stream"][row["endpoints"]], dtype=np.float32)
        require(array_sha256(target) == old["target_sha256"], f"{session}: target anchor drift")
        for arm in ARMS:
            scores[arm].append(values[arm])
        public.append({
            **row["public"],
            "target_sha256": array_sha256(target),
            "r2": values,
            "prediction_sha256": {arm: array_sha256(predictions[arm]) for arm in ARMS},
            "early_film_gain": ep_film_r2 - ep_zero_r2,
            "late_film_gain": lp_film_r2 - lp_zero_r2,
            "zero_anchor": {
                "same_process_repeat_prediction_sha_exact": True,
                "historical_prediction_sha_exact": {
                    "EP-ZERO": ep_historical_sha_equal,
                    "LP-ZERO": lp_historical_sha_equal,
                },
                "historical_r2_abs_error": {
                    "EP-ZERO": abs(ep_zero_r2 - float(old["r2"]["FROZEN-C1"])),
                    "LP-ZERO": abs(lp_zero_r2 - float(old["r2"]["LP-R3"])),
                },
                "required_historical_prediction_sha": bool(require_historical_prediction_sha),
                "r2_tolerance": float(anchor_r2_tolerance),
            },
        })
    return {arm: sum(values) / len(values) for arm, values in scores.items()}, public


def run_fold(repo_root: Path, outer_date: str, *, device: str, receipt_root: Path,
             predecessor_fold: Mapping[str, Any],
             require_historical_prediction_sha: bool = True,
             anchor_r2_tolerance: float = 1.0e-12) -> dict[str, Any]:
    from h1_causal_activity_completion_v1.stage1 import ARTIFACT_RELATIVE as C1_ARTIFACT, C1_AUTHORITIES, _load_model, _load_plan
    from src.data.h1_carrierid_date_lodo_target import load_outer_date_target_records
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from src.h1_m4_cce_contract import state_hash

    root = Path(repo_root).resolve()
    require(outer_date in DATE_ORDER and str(predecessor_fold.get("outer_date")) == outer_date, "outer date/predecessor drift")
    directory = root / C1_ARTIFACT / outer_date
    authority = C1_AUTHORITIES[outer_date]
    source_plan, s_src, plan_receipt = _load_plan(directory, authority, outer_date)
    early_net, early_state, model_receipt = _load_model(directory, authority, outer_date, device)
    predecessor_checkpoint = predecessor_fold["checkpoints"]["LP-R3"]
    late_net = _load_branch_checkpoint(
        early_net,
        root / PREDECESSOR_ROOT_RELATIVE / str(predecessor_checkpoint["path"]),
        expected=predecessor_checkpoint,
        device=device,
    )
    late_state = state_hash(late_net.state_dict())
    require(late_state == predecessor_checkpoint["full_state_sha256"], "LP-R3 strict state drift")

    data_root = root / DATA_RELATIVE
    indexed = index_heldin_calib(data_root)
    source_names = tuple(source_plan.source_sessions)
    require(source_names and all(name in indexed and not name.startswith(f"ses-{outer_date}") for name in source_names),
            "fold source roster drift")
    source_rows = [
        _profile_session_row(
            record=load_record(indexed[session]), minival=_load_minival(data_root, session),
            source_plan=source_plan, s_src=s_src, all_blocks=True,
        )
        for session in source_names
    ]
    films, training = train_pair(early_net, late_net, source_rows, device=device)
    checkpoints = {
        arm: _save_film(
            Path(receipt_root) / f"checkpoint_{outer_date}_{arm.lower()}.pt",
            outer_date=outer_date,
            arm=arm,
            module=films[arm],
            base_state_sha256=early_state if arm == "EP-FILM" else late_state,
        )
        for arm in FILM_ARMS
    }
    training_sha = _publish(Path(receipt_root) / f"training_{outer_date}.json", {
        "schema": f"{SCHEMA}_training",
        "status": "FILMS_AND_CHECKPOINTS_FROZEN_BEFORE_OUTER_DATE_OPEN",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "source_rows": [row["public"] for row in source_rows],
        "training": training,
        "checkpoints": checkpoints,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "lp_r3_predecessor_checkpoint": predecessor_checkpoint,
        "outer_date_calib_opened": False,
        "outer_date_minival_opened": False,
    })
    del films
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    films = {
        arm: _load_film(
            Path(receipt_root) / checkpoints[arm]["path"], expected=checkpoints[arm],
            outer_date=outer_date, arm=arm, device=device,
        )
        for arm in FILM_ARMS
    }

    target_records = load_outer_date_target_records(data_root, outer_date=outer_date)
    target_rows = [
        _profile_session_row(
            record=record, minival=_load_minival(data_root, session), source_plan=source_plan,
            s_src=s_src, all_blocks=False,
        )
        for session, record in target_records.items()
    ]
    before = {
        "early": state_hash(early_net.state_dict()),
        "late": state_hash(late_net.state_dict()),
        **{arm: state_hash(module.state_dict()) for arm, module in films.items()},
    }
    scores, rows = _score(
        early_net, late_net, films, target_rows, predecessor_fold["target_rows"], device=device,
        require_historical_prediction_sha=require_historical_prediction_sha,
        anchor_r2_tolerance=anchor_r2_tolerance,
    )
    after = {
        "early": state_hash(early_net.state_dict()),
        "late": state_hash(late_net.state_dict()),
        **{arm: state_hash(module.state_dict()) for arm, module in films.items()},
    }
    require(before == after and before["early"] == early_state and before["late"] == late_state,
            "model/FiLM changed during outer scoring")
    fold = {
        "schema": f"{SCHEMA}_fold",
        "status": "COMPLETE_PAIRED_FILM_TRAINING_AND_EXACT_ANCHOR_OUTER_SCORE",
        "outer_date": outer_date,
        "source_sessions": list(source_names),
        "target_sessions": list(target_records),
        "training_sha256": training_sha,
        "checkpoints": checkpoints,
        "scores": scores,
        "early_film_gain": scores["EP-FILM"] - scores["EP-ZERO"],
        "late_film_gain": scores["LP-FILM"] - scores["LP-ZERO"],
        "pooling_at_zero": scores["LP-ZERO"] - scores["EP-ZERO"],
        "pooling_with_film": scores["LP-FILM"] - scores["EP-FILM"],
        "interaction": (scores["LP-FILM"] - scores["LP-ZERO"]) - (scores["EP-FILM"] - scores["EP-ZERO"]),
        "target_rows": rows,
        "state_before_sha256": before,
        "state_after_sha256": after,
        "model_authority": model_receipt,
        "source_plan_authority": plan_receipt,
        "lp_r3_predecessor_checkpoint": predecessor_checkpoint,
        "zero_anchor_policy": {
            "require_historical_prediction_sha": bool(require_historical_prediction_sha),
            "historical_r2_tolerance": float(anchor_r2_tolerance),
            "same_process_repeat_prediction_sha_exact_required": True,
        },
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
    }
    fold["fold_sha256"] = _publish(Path(receipt_root) / f"fold_{outer_date}.json", fold)
    return fold


def run(repo_root: Path, *, device: str, receipt_root: Path,
        require_historical_prediction_sha: bool = True,
        anchor_r2_tolerance: float = 1.0e-12) -> dict[str, Any]:
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "H1 CP-FiLM requires isolated logical GPU0")
    predecessor, predecessor_authority = load_predecessor(repo_root)
    folds: list[dict[str, Any]] = []
    for outer_date in DATE_ORDER:
        folds.append(run_fold(
            repo_root, outer_date, device=device, receipt_root=receipt_root,
            predecessor_fold=predecessor[outer_date],
            require_historical_prediction_sha=require_historical_prediction_sha,
            anchor_r2_tolerance=anchor_r2_tolerance,
        ))
        gc.collect()
        torch.cuda.empty_cache()
    decision = decide_oof(folds)
    return {
        "schema": f"{SCHEMA}_score",
        "status": "COMPLETE_H1_CALIBRATION_PROFILE_FILM_SOURCE_OOF",
        "date_order": list(DATE_ORDER),
        "arms": list(ARMS),
        "folds": folds,
        "decision": decision,
        "predecessor_authority": predecessor_authority,
        "formal_heldout_opened": False,
        "evalai_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("load_predecessor", "run", "run_fold", "train_pair")
