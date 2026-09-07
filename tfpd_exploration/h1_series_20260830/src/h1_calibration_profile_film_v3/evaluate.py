"""H1 EP-FILM all-source training (V3): V2 internals with the LODO loop removed.

The training mathematics, schedule and hyperparameters are the sealed V2/V1
contract consumed by import; the only intended differences are:

1. there is no leave-one-date-out outer loop — one FiLM is trained on the
   episodes of **all** held-in dates/sessions of the all-source H-C plan;
2. only the EP-FILM arm is trained (LP arms are not part of the deployment
   artifact), and one EP-ZERO zero-initialization anchor pass is kept as the
   per-run bit-exact sanity;
3. the frozen substrate is the sealed all-source C1 checkpoint deployed by the
   M3RC/581792 chain instead of a date-LODO C1 cell.

The all-source H-C plan is rebuilt from the records by the pinned legacy
producer.  The sealed SVD output of that producer could not be reproduced
byte-exactly by any currently installed local numerical stack; the rebuild is
therefore gated on the pre-SVD statistics (mean/scale) and the selection, and
its roundoff-level carrier deviation from the sealed family-A plan is
quantified against the cached deployment carriers inside the M3RC package.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import time
from typing import Any, Mapping

import numpy as np

from h1_calibration_profile_film_v1.core import build_film, film_identity, require
from h1_calibration_profile_film_v1.evaluate import (
    _device_support_bank,
    _film_gradient_evidence,
    _profile_session_row,
)
from h1_calibration_profile_film_v1.plan import DATE_ORDER
from h1_m3_crossrecord_joint_v1.core import decode_with_identity, window_batch
from h1_support_resampled_postpool_v1.core import native_identity, support_index

from .plan import (
    ALL_SOURCE_CKPT_NAME,
    ALL_SOURCE_CKPT_SHA256,
    ALL_SOURCE_DOMAIN,
    ALL_SOURCE_IMPORT_RECEIPT_SHA256,
    ALL_SOURCE_MODEL_PARAMETERS,
    ALL_SOURCE_MODEL_STATE_SHA256,
    ALL_SOURCE_ROOT_RELATIVE,
    ALL_SOURCE_SELECTED,
    ANCHOR_ARM,
    ARM,
    BATCH_SIZE,
    CARRIER_RELATIVE_DEVIATION_BOUND,
    DATA_RELATIVE,
    EPOCHS,
    LEGACY_HEAD,
    LEGACY_PLAN_RELATIVE,
    LEGACY_PLAN_SHA256,
    LEARNING_RATE,
    M3RC_CALIBRATION_AUTHORITY_RELATIVE,
    M3RC_CALIBRATION_AUTHORITY_SHA256,
    M3RC_PACKAGE_RELATIVE,
    M3RC_PACKAGE_SHA256,
    PREDICTION_DIVISOR,
    SCHEMA,
    SEALED_SOURCE_NORMALIZER,
    SEALED_TRANSFORM_SHA256,
    SEED,
    SOURCE_NORMALIZER_RELATIVE_TOLERANCE,
    SOURCE_SELECTION_SHA256,
    TRAIN_STRIDE,
    V2_INITIAL_FILM_STATE_SHA256,
    V2_ROOT_RELATIVE,
    V2_SCORE_SHA256,
    V2_TERMINAL_SHA256,
    WEIGHT_DECAY,
)

PLAN_ARRAY_NAMES = ("mean", "scale", "pcs", "U", "mu")


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


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def verify_v2_root(repo_root: Path) -> dict[str, Any]:
    """Read-only verification of the sealed V2 root this run inherits from."""

    root = Path(repo_root).resolve() / V2_ROOT_RELATIVE
    expected: set[str] = {"attempt.json", "score.json", "terminal.json"}
    for date in DATE_ORDER:
        expected |= {f"training_{date}.json", f"fold_{date}.json"}
        for arm in ("ep-film", "lp-film"):
            expected.add(f"checkpoint_{date}_{arm}.pt")
    expected |= {f"{name}.sha256" for name in expected}
    require(root.is_dir() and not root.is_symlink(), "V2 sealed root missing/symlinked")
    require({path.name for path in root.iterdir()} == expected, "V2 sealed root topology drift")
    for name in expected:
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"V2 sealed leaf missing/symlinked: {name}")
        info = path.stat()
        require(stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1, f"V2 sealed leaf mode/link drift: {name}")
        digest = _sha256_file(path)
        if name.endswith(".sha256"):
            body = root / name[: -len(".sha256")]
            require(path.read_text(encoding="ascii") == f"{_sha256_file(body)}  {body.name}\n",
                    f"V2 sidecar drift: {name}")
        else:
            side = root / f"{name}.sha256"
            require(side.read_text(encoding="ascii") == f"{digest}  {name}\n", f"V2 sidecar drift: {name}")

    score = json.loads((root / "score.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(
        score.get("schema") == "h1_calibration_profile_film_v2_score"
        and score.get("status") == "COMPLETE_H1_CALIBRATION_PROFILE_FILM_V2_SOURCE_OOF"
        and score.get("decision", {}).get("classification") == "FILM_EARLY_REPLICATION_ONLY"
        and score.get("formal_heldout_opened") is False
        and score.get("evalai_opened") is False,
        "V2 score semantics drift",
    )
    require(
        terminal.get("schema") == "h1_calibration_profile_film_v2_terminal"
        and terminal.get("status") == score.get("status")
        and terminal.get("score_sha256") == V2_SCORE_SHA256,
        "V2 terminal semantics drift",
    )
    ep_final: dict[str, str] = {}
    v2_source_rows: dict[str, list[Mapping[str, Any]]] = {}
    for date in DATE_ORDER:
        training = json.loads((root / f"training_{date}.json").read_text(encoding="utf-8"))
        evidence = training.get("training", {})
        checkpoints = training.get("checkpoints", {})
        require(
            training.get("schema") == "h1_calibration_profile_film_v1_training"
            and training.get("status") == "FILMS_AND_CHECKPOINTS_FROZEN_BEFORE_OUTER_DATE_OPEN"
            and training.get("outer_date") == date
            and evidence.get("epochs") == EPOCHS
            and evidence.get("batch_size") == BATCH_SIZE
            and evidence.get("train_stride") == TRAIN_STRIDE
            and evidence.get("learning_rate") == LEARNING_RATE
            and evidence.get("weight_decay") == WEIGHT_DECAY
            and evidence.get("initial_film_state_sha256") == {"EP-FILM": V2_INITIAL_FILM_STATE_SHA256,
                                                               "LP-FILM": V2_INITIAL_FILM_STATE_SHA256}
            and checkpoints.get("EP-FILM", {}).get("path") == f"checkpoint_{date}_ep-film.pt"
            and checkpoints.get("EP-FILM", {}).get("film_state_sha256")
            == evidence.get("final_film_state_sha256", {}).get("EP-FILM"),
            f"V2 training contract drift: {date}",
        )
        require(_sha256_file(root / str(checkpoints["EP-FILM"]["path"])) == checkpoints["EP-FILM"]["sha256"],
                f"V2 EP-FILM checkpoint byte drift: {date}")
        ep_final[date] = str(checkpoints["EP-FILM"]["film_state_sha256"])
        v2_source_rows[date] = list(training.get("source_rows", []))
    return {
        "root": V2_ROOT_RELATIVE,
        "score_sha256": V2_SCORE_SHA256,
        "terminal_sha256": V2_TERMINAL_SHA256,
        "classification": "FILM_EARLY_REPLICATION_ONLY",
        "initial_film_state_sha256": V2_INITIAL_FILM_STATE_SHA256,
        "ep_film_final_state_sha256_by_lodo_date": ep_final,
        "source_rows_by_lodo_date": v2_source_rows,
    }


def load_all_source_model(repo_root: Path, *, device: str) -> tuple[Any, str, dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash
    from src.h1_hc_date_lodo_regen_v1 import _new_model

    root = Path(repo_root).resolve() / ALL_SOURCE_ROOT_RELATIVE
    ckpt_path = root / ALL_SOURCE_CKPT_NAME
    receipt_path = root / "import_receipt.json"
    require(ckpt_path.is_file() and not ckpt_path.is_symlink(), "all-source C1 checkpoint missing/symlinked")
    require(_sha256_file(ckpt_path) == ALL_SOURCE_CKPT_SHA256, "all-source C1 checkpoint SHA drift")
    receipt_digest = _verify_immutable(receipt_path, ALL_SOURCE_IMPORT_RECEIPT_SHA256)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(
        receipt.get("schema") == "h1_c1_all_source_epoch49_import_receipt_v1"
        and receipt.get("status") == "PASS_EXACT_HISTORICAL_C1_CHECKPOINT_CPU_STRICT_LOAD"
        and receipt.get("artifact_sha256") == ALL_SOURCE_CKPT_SHA256
        and receipt.get("checkpoint_schema") == "h1_cal_aug_all_source_m3_deployment_v1_checkpoint"
        and receipt.get("model_state_sha256") == ALL_SOURCE_MODEL_STATE_SHA256
        and receipt.get("model_parameter_count") == ALL_SOURCE_MODEL_PARAMETERS
        and receipt.get("epoch_zero_based") == 49
        and receipt.get("target_optimizer_steps") == 0,
        "all-source import receipt semantics drift",
    )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    require(payload.get("schema") == "h1_cal_aug_all_source_m3_deployment_v1_checkpoint",
            "all-source checkpoint schema drift")
    model = _new_model("cpu")
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            "all-source C1 strict load drift")
    state = state_hash(model.state_dict())
    require(state == ALL_SOURCE_MODEL_STATE_SHA256, "all-source C1 state SHA drift")
    require(sum(parameter.numel() for parameter in model.parameters()) == ALL_SOURCE_MODEL_PARAMETERS,
            "all-source C1 parameter-count drift")
    model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state, {
        "checkpoint_relative": f"{ALL_SOURCE_ROOT_RELATIVE}/{ALL_SOURCE_CKPT_NAME}",
        "checkpoint_sha256": ALL_SOURCE_CKPT_SHA256,
        "import_receipt_sha256": receipt_digest,
        "model_state_sha256": state,
        "model_parameter_count": ALL_SOURCE_MODEL_PARAMETERS,
        "source_commit": receipt.get("artifact_source_commit"),
        "strict_load_missing_keys": [],
        "strict_load_unexpected_keys": [],
    }


def rebuild_all_source_plan(repo_root: Path, legacy_root: Path) -> tuple[Any, float, dict[str, Any]]:
    from src.data.h1_m4_eb_pilot import array_sha256, fit_frozen_carrier
    from src.h1_cal_aug_all_source_m3_deployment_v1_exec import _load_all_source_records
    from src.h1_hc_date_lodo_regen_v1 import _legal_starts, _make_final_plan

    legacy = Path(legacy_root).resolve()
    sealed_path = legacy / LEGACY_PLAN_RELATIVE
    require(_sha256_file(sealed_path) == LEGACY_PLAN_SHA256, "sealed all-source plan.json drift in legacy checkout")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    require(
        sealed.get("schema") == "h1_cal_aug_all_source_m3_deployment_v1_plan"
        and set(sealed.get("array_sha256", {})) == set(PLAN_ARRAY_NAMES)
        and sealed.get("selection_sha256") == SOURCE_SELECTION_SHA256
        and sealed.get("transform_sha256") == SEALED_TRANSFORM_SHA256
        and int(sealed["q"]) == int(ALL_SOURCE_SELECTED["q"])
        and float(sealed["lambda"]) == float(ALL_SOURCE_SELECTED["lambda"]),
        "sealed all-source plan semantics drift",
    )
    data_root = Path(repo_root).resolve() / DATA_RELATIVE
    records = _load_all_source_records(data_root)
    plan = _make_final_plan(records, ALL_SOURCE_DOMAIN, ALL_SOURCE_SELECTED, SOURCE_SELECTION_SHA256)
    require(tuple(plan.source_sessions) == tuple(sealed["source_sessions"]),
            "rebuilt all-source roster drift against sealed plan")
    rebuilt = {name: array_sha256(getattr(plan, name)) for name in PLAN_ARRAY_NAMES}
    exact = {name: rebuilt[name] == sealed["array_sha256"][name] for name in PLAN_ARRAY_NAMES}
    require(exact["mean"] and exact["scale"],
            "all-source plan mean/scale rebuild drift (pre-SVD statistics are env-independent)")
    source_carriers = []
    for name, record in records.items():
        for start in _legal_starts(record):
            support = tuple(float(value) for value in record.trial_values[start : start + 4])
            source_carriers.append(np.asarray(fit_frozen_carrier(record, plan, support)["carrier"], np.float64))
    s_src = float(np.sqrt(np.mean(np.square(np.stack(source_carriers), dtype=np.float64), dtype=np.float64)))
    require(
        abs(s_src - SEALED_SOURCE_NORMALIZER) / SEALED_SOURCE_NORMALIZER <= SOURCE_NORMALIZER_RELATIVE_TOLERANCE,
        "all-source normalizer rebuild drift beyond tolerance",
    )
    authority = {
        "source": "pinned legacy checkout rebuild",
        "legacy_head": LEGACY_HEAD,
        "sealed_plan_json_sha256": LEGACY_PLAN_SHA256,
        "selection_sha256": SOURCE_SELECTION_SHA256,
        "q": int(plan.q),
        "ridge_lambda": float(plan.ridge_lambda),
        "source_sessions": list(plan.source_sessions),
        "sealed_array_sha256": dict(sealed["array_sha256"]),
        "sealed_tau2": float(sealed["tau2"]),
        "sealed_transform_sha256": str(sealed["transform_sha256"]),
        "sealed_arrays_file_sha256": str(sealed.get("arrays_file_sha256")),
        "rebuilt_array_sha256": rebuilt,
        "array_sha256_exact": exact,
        "rebuilt_tau2": float(plan.tau2),
        "rebuilt_transform_sha256": str(plan.transform_sha256),
        "transform_sha256_exact": str(plan.transform_sha256) == str(sealed["transform_sha256"]),
        "s_src": s_src,
        "s_src_exact": s_src == SEALED_SOURCE_NORMALIZER,
        "s_src_relative_error": abs(s_src - SEALED_SOURCE_NORMALIZER) / SEALED_SOURCE_NORMALIZER,
        "deviation_note": (
            "SVD outputs (pcs/U/mu/tau2) of the sealed producer are not byte-reproducible by the "
            "current local numerical stacks; mean/scale are exact and the carrier-level effect is "
            "quantified against the M3RC cached deployment carriers in cross_authority."
        ) if not all(exact.values()) else "byte-identical rebuild",
    }
    return plan, s_src, authority


def build_rows(repo_root: Path, plan: Any, s_src: float) -> list[dict[str, Any]]:
    from src.data.h1_m4_eb_pilot import index_heldin_calib, load_record
    from h1_cross_record_postpool_v1.evaluate import _load_minival

    data_root = Path(repo_root).resolve() / DATA_RELATIVE
    indexed = index_heldin_calib(data_root)
    rows = []
    for session in plan.source_sessions:
        require(str(session) in indexed, f"held-in calib record missing: {session}")
        rows.append(_profile_session_row(
            record=load_record(indexed[session]),
            minival=_load_minival(data_root, str(session)),
            source_plan=plan,
            s_src=s_src,
            all_blocks=True,
        ))
    require(rows, "all-source roster is empty")
    for row in rows:
        require(len(row["support_bank"]) >= 2, f"{row['session']}: no non-first contiguous M3 block")
    return rows


def cross_authority(repo_root: Path, rows: list[Mapping[str, Any]],
                    v2_authority: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    from src.h1_m4_cce_contract import array_sha256

    root = Path(repo_root).resolve()
    package_path = root / M3RC_PACKAGE_RELATIVE
    require(package_path.is_file() and not package_path.is_symlink(), "M3RC deployment package missing/symlinked")
    require(_sha256_file(package_path) == M3RC_PACKAGE_SHA256, "M3RC deployment package SHA drift")
    _verify_immutable(root / M3RC_CALIBRATION_AUTHORITY_RELATIVE, M3RC_CALIBRATION_AUTHORITY_SHA256)
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    require(
        package.get("schema") == "h1_m3_readout_calibration_evalai_package_v1"
        and package.get("checkpoint_sha256") == ALL_SOURCE_CKPT_SHA256
        and package.get("model_state_sha256") == ALL_SOURCE_MODEL_STATE_SHA256
        and len(package.get("sessions", {})) == 27,
        "M3RC deployment package semantics drift",
    )
    cached = {str(entry["session"]): entry for entry in package["sessions"].values()}

    v2_by_session: dict[str, Mapping[str, Any]] = {}
    for date, public_rows in v2_authority["source_rows_by_lodo_date"].items():
        for public in public_rows:
            v2_by_session[str(public["session"])] = public
    require(set(v2_by_session) >= {str(row["session"]) for row in rows},
            "V2 fold receipts do not cover the all-source roster")

    minival_matches: dict[str, bool] = {}
    activity_matches: dict[str, bool] = {}
    profile_matches: dict[str, bool] = {}
    carrier_deviation: dict[str, Any] = {}
    for row in rows:
        session = str(row["session"])
        public = row["public"]
        old = v2_by_session[session]
        minival_matches[session] = all([
            public["minival_input_sha256"] == old["minival_input_sha256"],
            public["query_neural_sha256"] == old["query_neural_sha256"],
            public["target_stream_sha256"] == old["target_stream_sha256"],
            public["endpoint_sha256"] == old["endpoint_sha256"],
            public["train_endpoint_sha256"] == old["train_endpoint_sha256"],
            public["training_windows"] == old["training_windows"],
        ])
        block_activity = [entry["activity_sha256"] for entry in public["support_bank"]]
        block_profile = [entry["task_profile_sha256"] for entry in public["support_bank"]]
        activity_matches[session] = (
            block_activity == [entry["activity_sha256"] for entry in old["support_bank"]]
        )
        profile_matches[session] = (
            block_profile == [entry["task_profile_sha256"] for entry in old["support_bank"]]
        )
        require(session in cached, f"{session}: M3RC cached deployment payload missing")
        entry = cached[session]
        identity_sha = array_sha256(row["support_bank"][0]["activity"])
        require(identity_sha == array_sha256(np.asarray(entry["identity"], dtype=np.float32)),
                f"{session}: M3 support activity/identity cache drift (record loading is not byte-stable)")
        mine = np.asarray(row["support_bank"][0]["carrier"], dtype=np.float64)
        sealed = np.asarray(entry["carrier"], dtype=np.float64)
        delta_abs = float(np.max(np.abs(mine - sealed)))
        delta_rel = delta_abs / max(float(np.max(np.abs(sealed))), 1.0e-30)
        require(delta_rel <= CARRIER_RELATIVE_DEVIATION_BOUND,
                f"{session}: first-M3 carrier deviation vs M3RC cache exceeds bound ({delta_rel:.3e})")
        carrier_deviation[session] = {
            "max_abs_delta": delta_abs,
            "max_relative_delta": delta_rel,
            "sealed_norm": float(np.linalg.norm(sealed)),
        }
    require(all(minival_matches.values()), "V2 minival surface drift (loader is not byte-stable)")
    require(all(activity_matches.values()), "V2 support activity drift (interpolation is not byte-stable)")
    require(all(profile_matches.values()), "V2 task-profile drift (profile computation is not byte-stable)")
    return {
        "m3rc_package_sha256": M3RC_PACKAGE_SHA256,
        "m3rc_calibration_authority_sha256": M3RC_CALIBRATION_AUTHORITY_SHA256,
        "v2_minival_surface_exact_by_session": minival_matches,
        "v2_support_activity_exact_by_session": activity_matches,
        "v2_task_profile_exact_by_session": profile_matches,
        "first_m3_identity_exact_vs_m3rc_cache": True,
        "first_m3_carrier_deviation_vs_m3rc_cache": carrier_deviation,
        "carrier_relative_deviation_bound": CARRIER_RELATIVE_DEVIATION_BOUND,
        "carrier_relative_deviation_max": max(
            row["max_relative_delta"] for row in carrier_deviation.values()
        ),
    }


def ep_zero_anchor(net: Any, rows: list[Mapping[str, Any]], *, device: str) -> dict[str, Any]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    torch.manual_seed(SEED)
    film = build_film().to(device)
    film.eval()
    state = state_hash(film.state_dict())
    require(state == V2_INITIAL_FILM_STATE_SHA256,
            "fresh zero-init FiLM state differs from the V2 zero-init anchor state")
    per_session: dict[str, Any] = {}
    for row in rows:
        activity, carrier, profile = _device_support_bank(row, device=device)[0]
        endpoints = np.asarray(row["endpoints"], dtype=np.int64)[:BATCH_SIZE]
        neural = torch.as_tensor(window_batch(row["neural"], endpoints), dtype=torch.float32, device=device)
        with torch.inference_mode():
            native = native_identity(net, activity, carrier)
            filmed = film_identity(net, activity, carrier, profile, film, late=False)
            identity_equal = bool(torch.equal(native, filmed))
            prediction_equal = bool(torch.equal(
                decode_with_identity(net, neural, native),
                decode_with_identity(net, neural, filmed),
            ))
        per_session[str(row["session"])] = {
            "support_block": 0,
            "identity_bitwise_equal": identity_equal,
            "prediction_bitwise_equal": prediction_equal,
        }
        require(identity_equal and prediction_equal,
                f"{row['session']}: EP-ZERO anchor is not bit-exact against native identity")
    return {
        "arm": ANCHOR_ARM,
        "policy": "zero-init FiLM identity must equal native identity bit-exactly on the first-M3 support",
        "zero_film_state_sha256": state,
        "matches_v2_initial_state": state == V2_INITIAL_FILM_STATE_SHA256,
        "sessions_verified": len(per_session),
        "per_session": per_session,
    }


def train_ep_film(net: Any, rows: list[Mapping[str, Any]], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(rows, "all-source CP-FiLM roster is empty")
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    frozen_before = state_hash(net.state_dict())

    torch.manual_seed(SEED)
    film = build_film().to(device)
    initial_state = state_hash(film.state_dict())
    require(initial_state == V2_INITIAL_FILM_STATE_SHA256,
            "training FiLM initial state differs from the V2 zero-init anchor state")
    optimizer = torch.optim.Adam(film.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    supports = {str(row["session"]): _device_support_bank(row, device=device) for row in rows}

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    total_steps = 0
    gradient_steps = 0
    first_identity_equal = False
    first_prediction_equal = False
    first_loss: float | None = None
    last_loss: float | None = None
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(EPOCHS):
        ordered = list(rows)
        random.Random(f"{SCHEMA}|{SEED}|session-order|{epoch}").shuffle(ordered)
        losses: list[float] = []
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
                    require(support_position == 0 and support_mode == "first_m3",
                            "first update is not first-M3 anchored")
                    with torch.no_grad():
                        zero = native_identity(net, activity, carrier)
                        filmed = film_identity(net, activity, carrier, profile, film, late=False)
                        first_identity_equal = bool(torch.equal(zero, filmed))
                        first_prediction_equal = bool(torch.equal(
                            decode_with_identity(net, neural, zero),
                            decode_with_identity(net, neural, filmed),
                        ))
                    require(first_identity_equal and first_prediction_equal,
                            "zero-init FiLM does not reproduce its substrate")

                optimizer.zero_grad(set_to_none=True)
                identity = film_identity(net, activity, carrier, profile, film, late=False)
                count = neural.shape[0]
                prediction = decode_with_identity(
                    net, neural, identity.expand(count, -1, -1)
                )[:, -1, :] / PREDICTION_DIVISOR
                loss = torch.nn.functional.mse_loss(prediction, target)
                require(bool(torch.isfinite(loss)), "EP-FILM task loss is nonfinite")
                loss.backward()
                _film_gradient_evidence(film, name=ARM)
                for name, parameter in tuple(net.named_parameters()):
                    require(parameter.grad is None, f"frozen substrate gradient materialized: {name}")
                optimizer.step()
                require(all(bool(torch.isfinite(parameter).all()) for parameter in film.parameters()),
                        "FiLM parameter became nonfinite")
                gradient_steps += 1
                value = float(loss.detach().cpu())
                if total_steps == 0:
                    first_loss = value
                last_loss = value
                losses.append(value)
                support_counts[support_mode] += 1
                unique_blocks[session].add(support_position)
                total_steps += 1
                epoch_steps += 1
        require(epoch_steps > 0, "EP-FILM epoch has no updates")
        epoch_rows.append({
            "epoch_zero_based": epoch,
            "steps": epoch_steps,
            "mean_task_loss": float(np.mean(losses, dtype=np.float64)),
            "support_counts": support_counts,
            "unique_support_blocks": {name: sorted(values) for name, values in unique_blocks.items()},
            "film_state_sha256": state_hash(film.state_dict()),
        })

    require(gradient_steps == total_steps, "EP-FILM gradient-step coverage drift")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    frozen_after = state_hash(net.state_dict())
    require(frozen_before == frozen_after, "frozen all-source substrate changed during FiLM training")
    return film, {
        "arm": ARM,
        "epochs": EPOCHS,
        "steps": total_steps,
        "steps_per_arm_v2_field_name": total_steps,
        "batch_size": BATCH_SIZE,
        "train_stride": TRAIN_STRIDE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "initial_film_state_sha256": {"EP-FILM": initial_state},
        "final_film_state_sha256": {"EP-FILM": state_hash(film.state_dict())},
        "frozen_substrate_before_sha256": frozen_before,
        "frozen_substrate_after_sha256": frozen_after,
        "first_identity_bitwise_equal": {"EP-FILM": first_identity_equal},
        "first_prediction_bitwise_equal": {"EP-FILM": first_prediction_equal},
        "gradient_steps": {"EP-FILM": gradient_steps},
        "first_loss": {"EP-FILM": first_loss},
        "last_loss": {"EP-FILM": last_loss},
        "epoch_rows": epoch_rows,
        "decoder_forward": "B_EP_FILM_SINGLE_ARM",
        "shuffle_token_schema": SCHEMA,
    }


def save_film(path: Path, *, module: Any, base_state_sha256: str) -> dict[str, Any]:
    import torch
    from src.h1_m4_cce_contract import state_hash

    require(not path.exists(), "FiLM checkpoint path already exists")
    payload = {
        "schema": f"{SCHEMA}_film_checkpoint",
        "train_scope": "all_source_heldin",
        "arm": ARM,
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
        "train_scope": "all_source_heldin",
        "arm": ARM,
    }


def run(repo_root: Path, legacy_root: Path, *, device: str, receipt_root: Path) -> dict[str, Any]:
    import torch

    require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "H1 EP-FILM all-source training requires isolated logical GPU0")
    root = Path(repo_root).resolve()
    started_utc = _utc_now()
    wall_start = time.perf_counter()

    v2_authority = verify_v2_root(root)
    v2_source_rows = v2_authority.pop("source_rows_by_lodo_date")
    net, model_state, model_authority = load_all_source_model(root, device=device)
    plan, s_src, plan_authority = rebuild_all_source_plan(root, legacy_root)
    rows = build_rows(root, plan, s_src)
    cross = cross_authority(root, rows, {**v2_authority, "source_rows_by_lodo_date": v2_source_rows})
    anchor = ep_zero_anchor(net, rows, device=device)
    anchor_seconds = time.perf_counter() - wall_start

    film, training = train_ep_film(net, rows, device=device)
    training_seconds = time.perf_counter() - wall_start - anchor_seconds
    checkpoint = save_film(
        Path(receipt_root) / "checkpoint_all_source_ep-film.pt",
        module=film,
        base_state_sha256=model_state,
    )
    del film
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    dates_covered: dict[str, list[str]] = {}
    for row in rows:
        dates_covered.setdefault(str(row["date"]), []).append(str(row["session"]))
    receipt = {
        "schema": f"{SCHEMA}_training",
        "status": "ALL_SOURCE_EP_FILM_FROZEN_DEPLOYMENT_CANDIDATE",
        "train_scope": {
            "lodo_outer_loop": "REMOVED",
            "dates_covered": {date: sorted(sessions) for date, sessions in sorted(dates_covered.items())},
            "source_sessions": [str(row["session"]) for row in rows],
            "per_date_film_state_sha256": None,
            "all_source_film_state_sha256": training["final_film_state_sha256"]["EP-FILM"],
            "heldout_calibration_opened": False,
            "hidden_test_opened": False,
        },
        "source_rows": [row["public"] for row in rows],
        "training": training,
        "ep_zero_anchor": anchor,
        "substrate_authority": model_authority,
        "plan_authority": plan_authority,
        "cross_authority": cross,
        "v2_authority": v2_authority,
        "checkpoints": {"EP-FILM": checkpoint},
        "wall_time_seconds": {
            "authority_and_anchor_seconds": anchor_seconds,
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - wall_start,
        },
        "started_at_utc": started_utc,
        "finished_at_utc": _utc_now(),
        "evalai_opened": False,
        "gpu1_touched": False,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    receipt_sha = _publish(Path(receipt_root) / "training_all_source.json", receipt)
    return {
        "receipt_sha256": receipt_sha,
        "status": receipt["status"],
        "checkpoints": {"EP-FILM": checkpoint},
        "training_summary": {
            "epochs": training["epochs"],
            "steps": training["steps"],
            "initial_film_state_sha256": training["initial_film_state_sha256"]["EP-FILM"],
            "final_film_state_sha256": training["final_film_state_sha256"]["EP-FILM"],
            "first_loss": training["first_loss"]["EP-FILM"],
            "last_loss": training["last_loss"]["EP-FILM"],
            "epoch_count": len(training["epoch_rows"]),
        },
        "ep_zero_anchor": {
            "sessions_verified": anchor["sessions_verified"],
            "all_identity_bitwise_equal": all(
                row["identity_bitwise_equal"] for row in anchor["per_session"].values()
            ),
            "all_prediction_bitwise_equal": all(
                row["prediction_bitwise_equal"] for row in anchor["per_session"].values()
            ),
            "zero_film_state_sha256": anchor["zero_film_state_sha256"],
        },
        "plan_authority_summary": {
            "mean_scale_exact": all(plan_authority["array_sha256_exact"][name] for name in ("mean", "scale")),
            "pcs_exact": plan_authority["array_sha256_exact"]["pcs"],
            "U_exact": plan_authority["array_sha256_exact"]["U"],
            "mu_exact": plan_authority["array_sha256_exact"]["mu"],
            "carrier_relative_deviation_max": cross["carrier_relative_deviation_max"],
        },
        "wall_time_seconds": receipt["wall_time_seconds"],
        "started_at_utc": receipt["started_at_utc"],
        "finished_at_utc": receipt["finished_at_utc"],
    }


__all__ = ("run",)
