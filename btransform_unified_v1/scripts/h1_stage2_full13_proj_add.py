"""H1 stage-2: 13-session full held-in retrain, L in {200, 250}, proj_add.

Stage-1 LODO locked L=250 over L=150 (Δ −0.133). User 2026-09-06 21:30 directed
a full-13 retrain of both L=200 and L=250, then pack the winner with exact-E
for EvalAI (register stays false until P32 clears and the user says push).

Protocol (ADDENDUM-SUBMISSION-PROTOCOL / M1 precedent):
  - train ALL 13 H1_ALL_SESSIONS (01-20 is a legal stage-2 member)
  - checkpoint = endpoint24 EMA (no SEL-2; 01-20 is polluted)
  - all-13 minival / 01-20 / sel2908 are DIAGNOSTIC only
  - L-winner = higher all-13 minival EMA equal_session_mean at e24
  - tie -> L=250 (stage-1 lock)
  - update caliber = measured sum_s ceil(n_s/32); formal12 reference is 731
  - GPU0 only; GPU1 is P32 / foreign — never touched

L=200 construction (same as F150): build_h1_bank(window=250) then
adapters._h1_windows to 200; assert X200 == X250[:, -200:, :]. Ends frozen.

Identity: B-transformer unified series, NOT SPINT.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import adapters, h1_config, plan, receipts  # noqa: E402
from btransform_unified_v1.bank import TaskBank, array_sha256  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.model import UNIT_DROPOUT_DOMAIN_META, unit_dropout_seed, whole_unit_dropout  # noqa: E402
from btransform_unified_v1.r2 import session_mean_report, variance_weighted_r2  # noqa: E402
from btransform_unified_v1.scale_bridge import assert_scale_bridge  # noqa: E402
from btransform_unified_v1.schedule import warmup_cosine_lr  # noqa: E402

ALLOWED_WINDOWS = (200, 250)
BANK_BUILD_WINDOW = 250
TRAIN_ENV_FLAG = "BTRANSFORM_H1_STAGE2_TRAIN"
SEED = plan.SEED
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
FOREIGN_MEM_MIB_LIMIT = 500
FIVEARM_PID = "1803132"
BUDGET_SECONDS = 4.0 * 3600.0
EPOCHS = plan.EPOCHS
SCALE = h1_config.TARGET_MULTIPLIER
EFFECTIVE_BATCH = plan.BATCH_SIZE
EVAL_BATCH = 32
SELECTION_FACE_COUNT = 2908
N_EXAM_FACES_EXPECTED = 2952
FORMAL12_UPDATES_PER_EPOCH = 731
CHECKPOINT_RULE = "endpoint24 EMA (stage-2; no surface pick; 01-20 polluted)"

WINDOW = 250  # overwritten in main before any data/train call

HEALTH_CHECKPOINT_EPOCH = 10
HEALTH_PLATEAU_TOL = 1e-4
HEALTH_PLATEAU_MIN_EPOCHS = 4
HEALTH_RAW_DECLINE_MIN_EPOCHS = 3
HEALTH_LOSS_BREAK_TARGET = 0.0065
HEALTH_RAW_FLOOR = -0.007
HEALTH_PIN_CENTER = 0.0069
HEALTH_PIN_TOL = 1e-4
HEALTH_RAW_DECLINE_BELOW = -0.01
HEALTH_PROTOCOL_TEXT = (
    "coordinator 2026-09-06 (same rule set as M-F700/M-F250), applied to "
    "stage-2 all-13 minival RAW: ruling at ep10 only; CONTINUE iff train_mse "
    "<= 0.0065 AND minivalRAW >= -0.007; EARLY FAIL iff train_mse pinned at "
    "0.0069+/-0.0001 OR minivalRAW declining 3 consecutive epochs below -0.01"
)


PROJ_DIM = 16
IDENTITY_MODE = "proj_add"
CUDA_PIN = "0"


def cell_name(window: int) -> str:
    mode = str(IDENTITY_MODE)
    if mode == "concat":
        return f"H1-STAGE2-L{window}-CONCAT"
    if int(PROJ_DIM) == 16:
        return f"H1-STAGE2-L{window}-PROJ-ADD"
    return f"H1-STAGE2-L{window}-P{int(PROJ_DIM)}-PROJ-ADD"


def _proj_kwargs() -> dict[str, int]:
    if str(IDENTITY_MODE) != "proj_add":
        return {}
    # Omit the kwarg on the P16 default so L=200 matches the already-running L=250 process.
    return {} if int(PROJ_DIM) == 16 else {"proj_dim": int(PROJ_DIM)}


def _expected_token_in() -> int:
    if str(IDENTITY_MODE) == "concat":
        return 16 + int(h1_config.FULL_E0_DIM) + 4
    return int(PROJ_DIM) + 4


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _seal(path: Path, payload: Any) -> str:
    return receipts.seal_json(path, payload)


def _make_geometry(window: int) -> dict[str, Any]:
    mode = str(IDENTITY_MODE)
    geometry = dict(plan.TASK_GEOMETRY["h1"])
    geometry["e0_dim"] = h1_config.matrix_e0_dim(mode)
    geometry["task"] = f"h1-stage2-L{window}-{mode}"
    return geometry


def _pinned_uuid() -> str:
    return GPU0_UUID if str(CUDA_PIN) == "0" else GPU1_UUID


def _nvidia_smi(args: list[str]) -> str:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def gpu_preflight(out_path: Path) -> dict[str, Any]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_rows = _nvidia_smi(
        ["--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    gpus = []
    for line in gpu_rows.splitlines():
        idx, uuid, util, mem_used, mem_total = [t.strip() for t in line.split(",")]
        gpus.append(
            {
                "index": int(idx),
                "uuid": uuid,
                "utilization_pct": float(util),
                "memory_used_mib": float(mem_used),
                "memory_total_mib": float(mem_total),
            }
        )
    app_rows = _nvidia_smi(
        ["--query-compute-apps=gpu_uuid,pid,used_memory,process_name", "--format=csv,noheader,nounits"]
    )
    apps = []
    for line in app_rows.splitlines():
        if not line.strip():
            continue
        parts = [t.strip() for t in line.split(",")]
        apps.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "used_mib": float(parts[2]),
                "process_name": ",".join(parts[3:]),
            }
        )
    own_pid = os.getpid()
    target_uuid = _pinned_uuid()
    foreign = [
        app
        for app in apps
        if app["gpu_uuid"] == target_uuid and app["pid"] != own_pid and app["used_mib"] > FOREIGN_MEM_MIB_LIMIT
    ]
    target = next(g for g in gpus if g["uuid"] == target_uuid)
    fivearm_proc = Path(f"/proc/{FIVEARM_PID}")
    report = {
        "schema": "btransform_unified_v1_h1_stage2_gpu_preflight",
        "cell": cell_name(WINDOW),
        "window": WINDOW,
        "unix": time.time(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "own_pid": own_pid,
        "cuda_visible_devices": raw,
        "cuda_pin": str(CUDA_PIN),
        "gpus": gpus,
        "compute_apps": apps,
        "gpu0": next(g for g in gpus if g["uuid"] == GPU0_UUID),
        "target_gpu": target,
        "foreign_pids_on_gpu0": foreign if str(CUDA_PIN) == "0" else [],
        "foreign_pids_on_target": foreign,
        "foreign_threshold_mib": FOREIGN_MEM_MIB_LIMIT,
        "fivearm_pid": FIVEARM_PID,
        "fivearm_proc_present": fivearm_proc.exists(),
        "gpu1_touched": str(CUDA_PIN) == "1",
        "gpu1_owner": "this cell" if str(CUDA_PIN) == "1" else "P32 / foreign (CUDA pinned to 0)",
        "gpu1_uuid": GPU1_UUID,
    }
    ok = (
        raw == str(CUDA_PIN)
        and not foreign
        and torch.cuda.device_count() == 1
        and target["memory_used_mib"] < FOREIGN_MEM_MIB_LIMIT
        and (str(CUDA_PIN) != "0" or not fivearm_proc.exists())
    )
    report["ok"] = bool(ok)
    _seal(out_path, report)
    return report


def _rebank(surface: str, session: str, window: int) -> tuple[TaskBank, dict[str, Any]]:
    bank = adapters.build_h1_bank(
        surface, session, budget=3, window=BANK_BUILD_WINDOW, identity_mode=str(IDENTITY_MODE)
    )
    row = adapters._h1_source_cache()[surface][session]
    neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
    if surface == "train":
        starts = np.asarray(row["query_starts"], dtype=np.int64)
        ends = starts + h1_config.FULL_WINDOW - 1
        eval_mask = np.asarray(row["eval_mask"], dtype=np.bool_)
        ends = ends[(ends >= 0) & (ends < eval_mask.shape[0])]
        ends = ends[eval_mask[ends]] if eval_mask.size else ends
    else:
        ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_)).astype(np.int64)
    plan.require(
        np.array_equal(np.asarray(bank.window_ids, dtype=np.int64), ends),
        f"recomputed ends != bank.window_ids for {surface}/{session}",
    )
    if window == BANK_BUILD_WINDOW:
        x_use = np.ascontiguousarray(bank.X_store, dtype=np.float32)
        tail_parity = True
    else:
        x_use = adapters._h1_windows(neural, ends, window)
        plan.require(
            np.array_equal(x_use, bank.X_store[:, -window:, :]),
            f"tail-{window} truncation parity failed for {surface}/{session}",
        )
        tail_parity = True
    meta = dict(bank.calibration_meta)
    meta["window"] = window
    meta["x_store_sha256"] = array_sha256(x_use)
    meta["rewindow_note"] = (
        f"build_h1_bank(window={BANK_BUILD_WINDOW}) then X_store to {window} "
        "(end-anchored; coordinates frozen)"
    )
    out = dataclasses.replace(bank, X_store=x_use, calibration_meta=meta)
    parity = {
        "n_windows": int(len(ends)),
        "ends_sha256": array_sha256(ends.astype(np.int64)),
        "x_sha256": array_sha256(x_use),
        "tail_parity": tail_parity,
        "E0_unchanged": True,
        "window": window,
    }
    return out, parity


def build_faces(window: int) -> dict[str, Any]:
    split = h1_config.lodo_split()
    train_sessions = list(h1_config.H1_ALL_SESSIONS)
    holdout_sessions = list(split["holdout_sessions"])
    plan.require(set(holdout_sessions).issubset(set(train_sessions)), "01-20 must be in the 13")

    train_banks, train_X, train_y, train_ids, parity_log = {}, {}, {}, {}, {}
    for s in train_sessions:
        bank, parity = _rebank("train", s, window)
        train_banks[s] = bank
        train_X[s] = bank.X_store
        train_y[s] = bank.target_store
        train_ids[s] = bank.window_ids
        parity_log[f"train/{s}"] = parity

    mini_banks, mini_X, mini_y, mini_ids = {}, {}, {}, {}
    for s in train_sessions:
        bank, parity = _rebank("minival", s, window)
        mini_banks[s] = bank
        mini_X[s] = bank.X_store
        mini_y[s] = bank.target_store
        mini_ids[s] = bank.window_ids
        parity_log[f"minival/{s}"] = parity

    exam_X = {s: mini_X[s] for s in holdout_sessions}
    exam_y = {s: mini_y[s] for s in holdout_sessions}
    exam_ids = {s: mini_ids[s] for s in holdout_sessions}
    exam_banks = {s: mini_banks[s] for s in holdout_sessions}
    n_exam = int(sum(len(v) for v in exam_ids.values()))
    plan.require(n_exam == N_EXAM_FACES_EXPECTED, f"01-20 minival faces {n_exam} != {N_EXAM_FACES_EXPECTED}")

    cache = adapters._h1_source_cache()
    sel_X, sel_y, sel_ids = {}, {}, {}
    for s in train_sessions:
        row = cache["minival"][s]
        ends = (np.asarray(row["query_starts"], dtype=np.int64) + h1_config.FULL_WINDOW - 1).astype(np.int64)
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        sel_X[s] = adapters._h1_windows(neural, ends, window)
        sel_y[s] = np.ascontiguousarray(np.asarray(row["velocity"], dtype=np.float32)[ends])
        sel_ids[s] = ends
    n_sel = int(sum(len(v) for v in sel_ids.values()))
    plan.require(n_sel == SELECTION_FACE_COUNT, f"sel2908 count {n_sel} != {SELECTION_FACE_COUNT}")

    return {
        "split": split,
        "window": window,
        "train": {"sessions": train_sessions, "banks": train_banks, "X": train_X, "y": train_y, "ids": train_ids},
        "minival": {"sessions": train_sessions, "banks": mini_banks, "X": mini_X, "y": mini_y, "ids": mini_ids},
        "exam": {"sessions": holdout_sessions, "banks": exam_banks, "X": exam_X, "y": exam_y, "ids": exam_ids},
        "sel": {"sessions": train_sessions, "banks": train_banks, "X": sel_X, "y": sel_y, "ids": sel_ids},
        "rewindow_parity": parity_log,
        "role": {
            "train": "all 13 held-in train surfaces (stage-2)",
            "minival": "DIAGNOSTIC all-13 source-minival (polluted; L-winner statistic at e24 EMA)",
            "exam": "DIAGNOSTIC 1925-01-20 minival only (polluted; never a pick)",
            "sel": "DIAGNOSTIC sel2908 (polluted; never a pick)",
        },
    }


@torch.no_grad()
def _score_faces(
    model: BTransformerUnifiedDecoderIdentity,
    faces: dict[str, Any],
    sessions: list[str],
    device: torch.device,
    *,
    bridge_check: bool = False,
) -> dict[str, Any]:
    model.eval()
    preds, targets, names = [], [], []
    raw_dump, native_dump = [], []
    for s in sessions:
        X, y = faces["X"][s], faces["y"][s]
        bank = faces["banks"][s]
        for off in range(0, len(X), EVAL_BATCH):
            xb = torch.from_numpy(X[off : off + EVAL_BATCH]).to(device)
            with torch.inference_mode():
                raw = model(xb, bank)
            raw_np = raw.detach().cpu().numpy()
            preds.append(raw_np / SCALE)
            targets.append(y[off : off + EVAL_BATCH])
            names.extend([s] * len(raw_np))
            if bridge_check:
                raw_dump.append(raw_np)
                native_dump.append(y[off : off + EVAL_BATCH])
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    name_arr = np.asarray(names)
    if bridge_check:
        assert_scale_bridge("h1", np.concatenate(raw_dump, axis=0), np.concatenate(native_dump, axis=0))
    report = session_mean_report(target, pred, np.repeat(name_arr, target.shape[1]))
    return {
        "pooled_r2": float(variance_weighted_r2(target, pred)),
        "equal_session_mean": float(report["session_mean_r2"]),
        "per_session_r2": {k: float(v) for k, v in sorted(report["per_session_r2"].items())},
        "n_faces": int(len(target)),
        "role": "DIAGNOSTIC (stage-2 polluted / same-source)",
        "r2_convention": (
            "skeleton r2.py: float64 flattened global-mean SStot (pooled) + equal mean of "
            "per-session flattened R2"
        ),
    }


def _score_view(
    model: BTransformerUnifiedDecoderIdentity,
    ema: DecoderEMA | None,
    view: str,
    faces: dict[str, Any],
    sessions: list[str],
    device: torch.device,
    *,
    bridge_check: bool = False,
) -> dict[str, Any]:
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    was_training = model.training
    try:
        if view == "EMA":
            plan.require(ema is not None and ema.n_updates > 0, "EMA view requested before any EMA update")
            with torch.no_grad():
                for name, param in named.items():
                    param.copy_(ema.shadow[name].to(device=param.device, dtype=param.dtype))
        report = _score_faces(model, faces, sessions, device, bridge_check=bridge_check)
        report["view"] = view
        return report
    finally:
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(backup[name].to(device=param.device, dtype=param.dtype))
        model.train(was_training)


def memory_probe(model, bank, X, y, device) -> dict[str, Any]:
    results = {}
    was_training = model.training
    model.train()
    named = model.trainable_parameters()
    backup = {name: param.detach().clone() for name, param in named.items()}
    for micro in (32, 16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        xb = torch.from_numpy(X[:micro]).to(device)
        yb = torch.from_numpy(y[:micro] * SCALE).to(device)
        keep = torch.ones(bank.unit_mask.shape[0], dtype=torch.bool)
        t0 = time.monotonic()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(xb, bank, dropout_keep=keep)
            loss = nn.functional.mse_loss(pred.float(), yb)
        loss.backward()
        torch.cuda.synchronize()
        results[f"micro{micro}"] = {
            "seconds": time.monotonic() - t0,
            "peak_alloc_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
        with torch.no_grad():
            for name, param in named.items():
                param.copy_(backup[name])
        model.zero_grad(set_to_none=True)
    model.train(was_training)
    return results


def _health_state(epoch: int, train_mse_series: dict[int, float], raw_series: dict[int, float]) -> dict[str, Any]:
    epochs = sorted(e for e in train_mse_series if e <= epoch)
    losses = [float(train_mse_series[e]) for e in epochs]
    raws = [float(raw_series[e]) for e in epochs]
    plateau = False
    if len(losses) >= HEALTH_PLATEAU_MIN_EPOCHS:
        tail = losses[-HEALTH_PLATEAU_MIN_EPOCHS:]
        plateau = (max(tail) - min(tail)) <= HEALTH_PLATEAU_TOL
    pinned_at_0069 = False
    if len(losses) >= HEALTH_PLATEAU_MIN_EPOCHS:
        tail = losses[-HEALTH_PLATEAU_MIN_EPOCHS:]
        pinned_at_0069 = all(abs(v - HEALTH_PIN_CENTER) <= HEALTH_PIN_TOL for v in tail)
    decline3 = False
    if len(raws) >= HEALTH_RAW_DECLINE_MIN_EPOCHS:
        tail = raws[-HEALTH_RAW_DECLINE_MIN_EPOCHS:]
        decline3 = all(tail[i + 1] < tail[i] for i in range(len(tail) - 1))
    decline3_below = bool(
        decline3 and all(v < HEALTH_RAW_DECLINE_BELOW for v in raws[-HEALTH_RAW_DECLINE_MIN_EPOCHS:])
    )
    return {
        "epoch": epoch,
        "train_mse": losses[-1] if losses else None,
        "minival_raw_pooled": raws[-1] if raws else None,
        "loss_plateau_4ep": plateau,
        "loss_pinned_0p0069": pinned_at_0069,
        "minival_raw_decline_3ep": decline3,
        "minival_raw_decline_3ep_below_minus0p01": decline3_below,
        "fail_triggers_present": bool(pinned_at_0069 or decline3_below),
    }


def health_adjudication(
    epoch: int,
    train_mse_series: dict[int, float],
    raw_series: dict[int, float],
) -> dict[str, Any]:
    state = _health_state(epoch, train_mse_series, raw_series)
    state.update(
        {
            "protocol": HEALTH_PROTOCOL_TEXT,
            "ruling": "NO_RULING_BEFORE_EP10" if epoch < HEALTH_CHECKPOINT_EPOCH else None,
        }
    )
    if epoch < HEALTH_CHECKPOINT_EPOCH:
        return state
    losses = [float(train_mse_series[e]) for e in sorted(train_mse_series) if e <= epoch]
    raws = [float(raw_series[e]) for e in sorted(raw_series) if e <= epoch]
    loss_ok = losses[-1] <= HEALTH_LOSS_BREAK_TARGET
    raw_ok = raws[-1] >= HEALTH_RAW_FLOOR
    if state["fail_triggers_present"]:
        state.update({"loss_break_le_0p0065": loss_ok, "minival_raw_ge_minus0p007": raw_ok, "ruling": "EARLY_FAIL"})
    else:
        state.update(
            {
                "loss_break_le_0p0065": loss_ok,
                "minival_raw_ge_minus0p007": raw_ok,
                "ruling": (
                    "CONTINUE (no fail trigger at ep10)"
                    if (loss_ok and raw_ok)
                    else "EARLY_FAIL (continue-conditions not met and no recovery evidence)"
                ),
            }
        )
    return state


def run_train(dest: Path, faces: dict[str, Any], micro_override: int | None, lr_peak: float) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    window = int(faces["window"])
    plan.require(window in ALLOWED_WINDOWS, f"window {window} not in {ALLOWED_WINDOWS}")
    plan.require(abs(float(lr_peak) - 1.0e-4) < 1e-15, "stage-2 recipe is peak 1e-4 only")

    geometry = _make_geometry(window)
    model = BTransformerUnifiedDecoderIdentity(
        geometry,
        seed=SEED,
        override_prefix=0,
        override_window=window,
        identity_mode=str(IDENTITY_MODE),
        **_proj_kwargs(),
    ).to(device)
    meta = model.init_meta
    plan.require(
        meta["identity_mode"] == str(IDENTITY_MODE) and int(meta["token_in"]) == _expected_token_in(),
        "identity geometry drift",
    )
    plan.require(model.window == window and model.l_in == window, "window override drift")

    exam0_s = faces["exam"]["sessions"][0]
    causal = model.causal_check(
        torch.from_numpy(faces["exam"]["X"][exam0_s][:2]).to(device), faces["exam"]["banks"][exam0_s]
    )
    plan.require(causal["passed"], "causal_check failed on real data")

    train_sessions = faces["train"]["sessions"]
    n_train_faces = int(sum(len(faces["train"]["X"][s]) for s in train_sessions))
    plan.require(len(train_sessions) == 13, "stage-2 must train all 13 sessions")

    probe_s = train_sessions[0]
    probe = memory_probe(
        model, faces["train"]["banks"][probe_s], faces["train"]["X"][probe_s], faces["train"]["y"][probe_s], device
    )
    if micro_override is not None:
        micro = int(micro_override)
        accum = EFFECTIVE_BATCH // micro
    elif probe.get("micro32", {}).get("peak_alloc_mib", 1e9) <= 18000:
        micro, accum = 32, 1
    else:
        micro, accum = 16, 2
    plan.require(micro * accum == EFFECTIVE_BATCH, "micro x accum != effective batch 32")

    updates_per_epoch = int(
        sum(math.ceil(len(faces["train"]["X"][s]) / EFFECTIVE_BATCH) for s in train_sessions)
    )
    warmup_updates = updates_per_epoch * plan.WARMUP_EPOCHS
    total_updates = updates_per_epoch * EPOCHS
    formal = plan.recipe_updates("h1", epochs=EPOCHS)
    plan.require(
        updates_per_epoch == FORMAL12_UPDATES_PER_EPOCH,
        f"update caliber drift: expected formal12 {FORMAL12_UPDATES_PER_EPOCH}/ep, got {updates_per_epoch}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr_peak),
        weight_decay=plan.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    rng = np.random.default_rng(SEED)

    _seal(
        dest / "run_meta.json",
        {
            "schema": "btransform_unified_v1_h1_stage2_run_meta",
            "cell": cell_name(window),
            "stage": 2,
            "checkpoint_rule": CHECKPOINT_RULE,
            "winner_statistic": "all-13 minival EMA equal_session_mean at endpoint24",
            "utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "lr_peak": float(lr_peak),
            "geometry": geometry,
            "window": model.window,
            "proj_dim": int(PROJ_DIM),
            "token_in": int(model.token_in),
            "identity_mode": model.identity_mode,
            "init_meta": meta,
            "param_count": meta["param_count"],
            "causal_check_startup": causal,
            "lodo_split_diagnostic_only": faces["split"],
            "train_sessions": train_sessions,
            "n_train_sessions": len(train_sessions),
            "train_face_counts": {s: int(len(faces["train"]["X"][s])) for s in train_sessions},
            "n_train_faces": n_train_faces,
            "minival_face_counts": {s: int(len(faces["minival"]["X"][s])) for s in faces["minival"]["sessions"]},
            "exam_0120_face_counts": {s: int(len(faces["exam"]["X"][s])) for s in faces["exam"]["sessions"]},
            "selection_face_counts": {s: int(len(faces["sel"]["X"][s])) for s in faces["sel"]["sessions"]},
            "surface_roles": faces["role"],
            "cal": {"mode": "CAL-2 fixed M3", "budget": h1_config.DEPLOY_BUDGET},
            "microbatch_probe": probe,
            "microbatch_decision": {"micro": micro, "accum": accum, "effective_batch": EFFECTIVE_BATCH},
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
            "formal_update_caliber": formal,
            "unit_dropout_p": plan.UNIT_DROPOUT,
            "unit_dropout_domain_meta": UNIT_DROPOUT_DOMAIN_META,
            "epochs": EPOCHS,
            "budget_seconds": BUDGET_SECONDS,
            "gpu_uuid": GPU0_UUID,
            "gpu1_owner": "P32 / foreign; never touched",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "source_cache_sha256": h1_config.SOURCE_CACHE_SHA256,
        },
    )
    (dest / "PROGRESS_10MIN.md").write_text(
        f"# H1 stage-2 L={window} (endpoint24 EMA; diagnostics polluted)\n\n"
        f"Cell: {cell_name(window)}, peak_lr={float(lr_peak):g}, "
        f"{updates_per_epoch} upd/ep, GPU0 {GPU0_UUID}\n\n"
        "| ep | train_mse | miniRAW pool | miniEMA eq | miniEMA pool | exam0120EMA eq | selEMA pool | ruling |\n"
        "|----|-----------|--------------|------------|--------------|----------------|-------------|--------|\n",
        encoding="utf-8",
    )

    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    started = time.monotonic()
    deadline = started + BUDGET_SECONDS
    global_step = 0
    train_mse_series: dict[int, float] = {}
    lr_series: dict[int, float] = {}
    mini_raw_series: dict[int, dict[str, Any]] = {}
    mini_ema_series: dict[int, dict[str, Any]] = {}
    exam_ema_series: dict[int, dict[str, Any]] = {}
    exam_raw_series: dict[int, dict[str, Any]] = {}
    sel_ema_series: dict[int, dict[str, Any]] = {}
    sel_raw_series: dict[int, dict[str, Any]] = {}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        order = list(train_sessions)
        rng.shuffle(order)
        losses: list[float] = []
        batch_id = -1
        for session in order:
            X, y = faces["train"]["X"][session], faces["train"]["y"][session]
            bank = faces["train"]["banks"][session]
            idx = rng.permutation(len(X))
            for offset in range(0, len(idx), EFFECTIVE_BATCH):
                if time.monotonic() >= deadline:
                    _seal(
                        dest / "budget_hit.json",
                        {
                            "schema": "btransform_unified_v1_h1_stage2_budget_hit",
                            "cell": cell_name(window),
                            "epoch": epoch,
                            "global_step": global_step,
                            "budget_seconds": BUDGET_SECONDS,
                            "elapsed_seconds": time.monotonic() - started,
                            "utc": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    raise RuntimeError(f"H1 stage-2 L={window} 4h GPU budget hit")
                take = idx[offset : offset + EFFECTIVE_BATCH]
                micro_losses = []
                for m_off in range(0, len(take), micro):
                    m_take = take[m_off : m_off + micro]
                    batch_id += 1
                    global_step += 1
                    lr = warmup_cosine_lr(
                        global_step,
                        total_steps=total_updates,
                        warmup_steps=warmup_updates,
                        peak=float(lr_peak),
                        min_factor=plan.LR_MIN_FACTOR,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
                    keep = whole_unit_dropout(
                        torch.from_numpy(bank.unit_mask.copy()), p=plan.UNIT_DROPOUT, generator=generator
                    )
                    xb = torch.from_numpy(X[m_take]).to(device)
                    yb = torch.from_numpy(y[m_take] * SCALE).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(xb, bank, dropout_keep=keep)
                        loss = nn.functional.mse_loss(pred.float(), yb) / accum
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    micro_losses.append(float(loss.detach().cpu()) * accum)
                nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.GRAD_CLIP)
                optimizer.step()
                ema.update_after_step(model)
                losses.extend(micro_losses)
                if global_step % 200 == 0:
                    _append_jsonl(
                        metrics_path,
                        {
                            "event": "step",
                            "cell": cell_name(window),
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": micro_losses[-1] if micro_losses else None,
                            "lr": float(lr),
                            "ema_updates": ema.n_updates,
                            "unix": time.time(),
                        },
                    )
                    _write_json(
                        heartbeat,
                        {
                            "cell": cell_name(window),
                            "epoch": epoch,
                            "global_step": global_step,
                            "lr": float(lr),
                            "ema_updates": ema.n_updates,
                            "unix": time.time(),
                            "gpu_uuid": GPU0_UUID,
                        },
                    )

        mini_raw = _score_view(model, ema, "RAW", faces["minival"], faces["minival"]["sessions"], device)
        mini_ema = _score_view(
            model, ema, "EMA", faces["minival"], faces["minival"]["sessions"], device, bridge_check=True
        )
        exam_raw = _score_view(model, ema, "RAW", faces["exam"], faces["exam"]["sessions"], device)
        exam_ema = _score_view(model, ema, "EMA", faces["exam"], faces["exam"]["sessions"], device)
        sel_raw = _score_view(model, ema, "RAW", faces["sel"], faces["sel"]["sessions"], device)
        sel_ema = _score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
        train_mse_series[epoch] = float(np.mean(losses)) if losses else float("nan")
        lr_series[epoch] = float(lr)
        mini_raw_series[epoch] = mini_raw
        mini_ema_series[epoch] = mini_ema
        exam_raw_series[epoch] = exam_raw
        exam_ema_series[epoch] = exam_ema
        sel_raw_series[epoch] = sel_raw
        sel_ema_series[epoch] = sel_ema
        row = {
            "event": "epoch",
            "epoch": epoch,
            "cell": cell_name(window),
            "window": window,
            "train_mse": train_mse_series[epoch],
            "minival13_raw": mini_raw,
            "minival13_ema": mini_ema,
            "exam_0120_raw": exam_raw,
            "exam_0120_ema": exam_ema,
            "sel2908_raw": sel_raw,
            "sel2908_ema": sel_ema,
            "lr": float(lr),
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
            "ema_updates": ema.n_updates,
            "unix": time.time(),
            "note": "all scored surfaces DIAGNOSTIC; checkpoint is endpoint24 EMA",
        }
        _append_jsonl(metrics_path, row)
        _write_json(heartbeat, row)
        health = health_adjudication(
            epoch, train_mse_series, {e: mini_raw_series[e]["pooled_r2"] for e in mini_raw_series}
        )
        _append_jsonl(dest / "health.jsonl", health)
        with (dest / "PROGRESS_10MIN.md").open("a", encoding="utf-8") as prog:
            prog.write(
                f"| {epoch} | {row['train_mse']:.6f} | {mini_raw['pooled_r2']:.4f} | "
                f"{mini_ema['equal_session_mean']:.4f} | {mini_ema['pooled_r2']:.4f} | "
                f"{exam_ema['equal_session_mean']:.4f} | {sel_ema['pooled_r2']:.4f} | {health['ruling']} |\n"
            )
        print(
            f"[stage2 L={window}] ep{epoch} loss={train_mse_series[epoch]:.5f} "
            f"miniEMA={mini_ema['equal_session_mean']:.4f}/{mini_ema['pooled_r2']:.4f} "
            f"exam0120EMA={exam_ema['equal_session_mean']:.4f} "
            f"health={health['ruling']} ({row['seconds']:.0f}s)",
            flush=True,
        )
        if health["ruling"] == "EARLY_FAIL":
            _seal(
                dest / "early_fail_receipt.json",
                {
                    "schema": "btransform_unified_v1_h1_stage2_early_fail",
                    "cell": cell_name(window),
                    "window": window,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "epoch_at_fail": epoch,
                    "health": health,
                    "train_mse_by_epoch": train_mse_series,
                    "minival13_ema_equal_mean": {e: mini_ema_series[e]["equal_session_mean"] for e in mini_ema_series},
                    "note": "early FAIL; GPU0 released; no self-retry",
                },
            )
            print(f"[stage2 L={window}] EARLY_FAIL — receipt sealed, GPU0 released", flush=True)
            return {
                "schema": "btransform_unified_v1_h1_stage2_train_receipt",
                "status": "EARLY_FAIL",
                "cell": cell_name(window),
                "window": window,
                "epochs_completed": list(range(1, epoch + 1)),
                "train_mse": train_mse_series,
            }
        ckpt = {
            "schema": "btransform_unified_v1_h1_stage2_ckpt",
            "cell": cell_name(window),
            "window": window,
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "checkpoint_rule": CHECKPOINT_RULE,
            "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "ema": ema.state_dict(),
            "lr": float(lr),
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")

    plan.require(global_step == total_updates, f"update accounting drift {global_step} != {total_updates}")
    summary = {
        "schema": "btransform_unified_v1_h1_stage2_train_receipt",
        "status": "COMPLETED",
        "cell": cell_name(window),
        "window": window,
        "checkpoint_rule": CHECKPOINT_RULE,
        "seed": SEED,
        "gpu_uuid": GPU0_UUID,
        "epochs_completed": list(range(1, EPOCHS + 1)),
        "global_updates": global_step,
        "ema_updates": ema.n_updates,
        "train_mse": train_mse_series,
        "lr_at_epoch_end": lr_series,
        "minival13_ema_equal_mean_by_epoch": {e: mini_ema_series[e]["equal_session_mean"] for e in mini_ema_series},
        "minival13_ema_pooled_by_epoch": {e: mini_ema_series[e]["pooled_r2"] for e in mini_ema_series},
        "exam_0120_ema_equal_mean_by_epoch": {e: exam_ema_series[e]["equal_session_mean"] for e in exam_ema_series},
        "sel2908_ema_pooled_by_epoch": {e: sel_ema_series[e]["pooled_r2"] for e in sel_ema_series},
        "endpoint24_minival13_ema_equal_mean": mini_ema_series[EPOCHS]["equal_session_mean"],
        "endpoint24_minival13_ema_pooled": mini_ema_series[EPOCHS]["pooled_r2"],
        "elapsed_s": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "note": "endpoint24 EMA is the only legal checkpoint; all series DIAGNOSTIC",
    }
    _seal(dest / "train_receipt.json", summary)
    return summary


def _read_epoch_metrics(dest: Path) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    path = dest / "metrics.jsonl"
    if not path.is_file():
        return metrics
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") == "epoch":
                metrics[int(row["epoch"])] = row
    return metrics


def run_cell_receipt(dest: Path, faces: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    window = int(faces["window"])
    metrics = _read_epoch_metrics(dest)
    plan.require(bool(metrics), "no epoch rows in metrics.jsonl")
    plan.require(EPOCHS in metrics, "missing endpoint24 metrics")
    ckpt_path = dest / f"epoch_{EPOCHS:03d}.pt"
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    geometry = _make_geometry(window)
    model = BTransformerUnifiedDecoderIdentity(
        geometry,
        seed=SEED,
        override_prefix=0,
        override_window=window,
        identity_mode=str(IDENTITY_MODE),
        **_proj_kwargs(),
    ).to(device)
    model.load_state_dict(ckpt["raw_state_dict"])
    ema = DecoderEMA.__new__(DecoderEMA)
    ema.decay = plan.EMA_DECAY
    ema.n_updates = int(ckpt["ema"]["n_updates"])
    ema.shadow = {k: v.detach().clone() for k, v in ckpt["ema"]["shadow"].items()}
    e24_mini = _score_view(model, ema, "EMA", faces["minival"], faces["minival"]["sessions"], device)
    e24_exam = _score_view(model, ema, "EMA", faces["exam"], faces["exam"]["sessions"], device)
    e24_sel = _score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
    _write_json(
        dest / "endpoint24_detail.json",
        {"minival13_ema": e24_mini, "exam_0120_ema": e24_exam, "sel2908_ema": e24_sel},
    )
    receipt = {
        "schema": "btransform_unified_v1_h1_stage2_cell_receipt",
        "cell": cell_name(window),
        "stage": 2,
        "window": window,
        "proj_dim": int(PROJ_DIM),
        "token_in": _expected_token_in(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "checkpoint_rule": CHECKPOINT_RULE,
        "legal_checkpoint": {
            "epoch": EPOCHS,
            "view": "EMA",
            "path": str(ckpt_path),
            "minival13_ema_equal_session_mean": e24_mini["equal_session_mean"],
            "minival13_ema_pooled": e24_mini["pooled_r2"],
            "minival13_per_session": e24_mini["per_session_r2"],
        },
        "diagnostic_only": {
            "exam_0120_ema_equal_session_mean": e24_exam["equal_session_mean"],
            "exam_0120_ema_pooled": e24_exam["pooled_r2"],
            "sel2908_ema_equal_session_mean": e24_sel["equal_session_mean"],
            "sel2908_ema_pooled": e24_sel["pooled_r2"],
            "note": "01-20 and sel2908 are in the stage-2 train set; not selectors",
        },
        "epoch_curves": {
            "minival13_ema_equal_mean": {
                e: float(metrics[e]["minival13_ema"]["equal_session_mean"]) for e in metrics
            },
            "minival13_ema_pooled": {e: float(metrics[e]["minival13_ema"]["pooled_r2"]) for e in metrics},
            "exam_0120_ema_equal_mean": {
                e: float(metrics[e]["exam_0120_ema"]["equal_session_mean"]) for e in metrics
            },
            "sel2908_ema_pooled": {e: float(metrics[e]["sel2908_ema"]["pooled_r2"]) for e in metrics},
            "train_mse": {e: float(metrics[e]["train_mse"]) for e in metrics},
        },
        "lodo_split_diagnostic_only": faces["split"],
        "surface_roles": faces["role"],
        "winner_use": (
            "compare legal_checkpoint.minival13_ema_equal_session_mean across L=200 and L=250; "
            "higher wins; tie -> L=250"
        ),
    }
    _write_json(dest / "cell_receipt_content.json", receipt)
    _seal(dest / "cell_receipt.json", receipt)
    return receipt


def main() -> int:
    global WINDOW, PROJ_DIM, IDENTITY_MODE, CUDA_PIN
    parser = argparse.ArgumentParser(description="H1 stage-2 full-13 retrain (L=200 or 250; proj_add or concat)")
    parser.add_argument("--window", type=int, required=True, choices=list(ALLOWED_WINDOWS))
    parser.add_argument("--proj-dim", type=int, default=16, choices=[16, 32])
    parser.add_argument("--identity-mode", choices=["proj_add", "concat"], default="proj_add")
    parser.add_argument("--cuda-visible", default="0", choices=["0", "1"])
    parser.add_argument("--stage", choices=["preflight", "build", "train", "receipt", "all"], default="all")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--micro", type=int, default=None)
    parser.add_argument("--peak-lr", type=float, default=1.0e-4)
    args = parser.parse_args()
    WINDOW = int(args.window)
    PROJ_DIM = int(args.proj_dim)
    IDENTITY_MODE = str(args.identity_mode)
    CUDA_PIN = str(args.cuda_visible)

    if args.stage in ("train", "all") and os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: training requires {TRAIN_ENV_FLAG}=1", file=sys.stderr)
        return 2
    if args.stage in ("train", "all", "receipt"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != CUDA_PIN:
            print(f"REFUSED: CUDA_VISIBLE_DEVICES must be pinned to {CUDA_PIN!r}", file=sys.stderr)
            return 2

    forbidden = {
        PACKAGE_ROOT / "results/h1_matrix/M_F250_lr1e4_20260906T094633Z",
        PACKAGE_ROOT / "results/h1_matrix/M_F150_lr1e4_20260906T125517Z",
    }
    dest = args.dest.resolve()
    plan.require(dest not in forbidden, f"refusing to overwrite historical root {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[stage2 L={WINDOW}] dest = {dest}", flush=True)

    if args.stage in ("preflight", "train", "receipt", "all"):
        pre = gpu_preflight(dest / "preflight_gpu.json")
        print(f"[stage2 L={WINDOW}] preflight ok={pre['ok']} foreign={pre.get('foreign_pids_on_target')}", flush=True)
        if not pre["ok"]:
            _seal(
                dest / "BLOCKED.json",
                {
                    "schema": "btransform_unified_v1_h1_stage2_blocked",
                    "reason": f"GPU{CUDA_PIN} not clean at preflight",
                    "preflight": pre,
                    "utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            return 2

    faces = None
    if args.stage in ("build", "train", "receipt", "all"):
        t0 = time.monotonic()
        faces = build_faces(WINDOW)
        _write_json(
            dest / "face_inventory.json",
            {
                "built_utc": datetime.now(timezone.utc).isoformat(),
                "seconds": time.monotonic() - t0,
                "window": WINDOW,
                "train": {s: int(len(v)) for s, v in faces["train"]["X"].items()},
                "minival": {s: int(len(v)) for s, v in faces["minival"]["X"].items()},
                "exam_0120": {s: int(len(v)) for s, v in faces["exam"]["X"].items()},
                "sel": {s: int(len(v)) for s, v in faces["sel"]["X"].items()},
                "train_total": int(sum(len(v) for v in faces["train"]["X"].values())),
                "minival_total": int(sum(len(v) for v in faces["minival"]["X"].values())),
                "exam_total": int(sum(len(v) for v in faces["exam"]["X"].values())),
                "sel_total": int(sum(len(v) for v in faces["sel"]["X"].values())),
                "rewindow_parity": faces["rewindow_parity"],
                "surface_roles": faces["role"],
            },
        )
        print(
            f"[stage2 L={WINDOW}] faces in {time.monotonic()-t0:.0f}s: "
            f"train={sum(len(v) for v in faces['train']['X'].values())} "
            f"minival={sum(len(v) for v in faces['minival']['X'].values())} "
            f"exam={sum(len(v) for v in faces['exam']['X'].values())} "
            f"sel={sum(len(v) for v in faces['sel']['X'].values())}",
            flush=True,
        )

    if args.stage in ("train", "all"):
        summary = run_train(dest, faces, args.micro, args.peak_lr)
        if summary.get("status") == "EARLY_FAIL":
            print("[stage2] early-fail; not writing cell receipt; no self-retry", flush=True)
            return 1
        print(f"[stage2 L={WINDOW}] train done in {summary['elapsed_s']:.0f}s", flush=True)

    if args.stage in ("receipt", "all"):
        if faces is None:
            faces = build_faces(WINDOW)
        receipt = run_cell_receipt(dest, faces)
        legal = receipt["legal_checkpoint"]
        print(
            f"[stage2 L={WINDOW}] endpoint24 EMA minival13 eq={legal['minival13_ema_equal_session_mean']:.4f} "
            f"pooled={legal['minival13_ema_pooled']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        dest = None
        for arg_idx, a in enumerate(sys.argv):
            if a == "--dest" and arg_idx + 1 < len(sys.argv):
                dest = Path(sys.argv[arg_idx + 1])
        trace = traceback.format_exc()
        print(trace, file=sys.stderr)
        try:
            from btransform_unified_v1 import plan as _plan, receipts as _receipts

            target = dest or (_plan.RESULT_ROOT / "h1_stage2_full13" / "error")
            target.mkdir(parents=True, exist_ok=True)
            _receipts.seal_json(
                target / "error_receipt.json",
                {
                    "schema": "btransform_unified_v1_h1_stage2_error",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "traceback": trace,
                },
            )
        except Exception:
            pass
        sys.exit(3)
