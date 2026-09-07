"""Local H1 source-dev R². Does not open hidden/eval held-out query files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m1_h1_activity_headroom_v1.core import variance_weighted_r2
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
)

from .config import HELDIN_SESSIONS, PREDICTION_DIVISOR, RESULT_ROOT, SEED, WINDOW
from .data import _index_split, load_session_arrays, materialize_banks

CHUNK = 16
VIEWS = (
    ("flat", 5, "EMA"),
    ("flat", 12, "EMA"),
    ("route", 5, "EMA"),
    ("route", 12, "EMA"),
)


def _load_ema(arm: str, epoch: int, device: torch.device):
    path = RESULT_ROOT / arm / f"epoch_{epoch:03d}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if arm == "flat":
        model = H1TemporalFlatDecoder(seed=SEED)
    else:
        template = H1TemporalFlatDecoder(seed=SEED)
        model = H1TemporalRouteDecoder(seed=SEED, flat_template=template)
    model.load_state_dict(ckpt["ema"]["shadow"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


def _windows(neural: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = int(indices.size)
    out = np.zeros((n, WINDOW, neural.shape[1]), dtype=np.float32)
    for i, last in enumerate(indices.tolist()):
        take = min(int(last) + 1, WINDOW)
        out[i, -take:] = neural[int(last) + 1 - take : int(last) + 1]
    return out


def _score_session(model, bank, arrays, device: torch.device) -> dict[str, Any]:
    mask = np.asarray(arrays.eval_mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        raise RuntimeError(f"{arrays.session}: not enough eval bins")
    preds = []
    e0 = bank.E0.to(device)
    hc = bank.T.to(device)
    keep = bank.unit_mask.to(device)
    gpu_bank = type(bank)(E0=e0, T=hc, unit_mask=keep)
    with torch.inference_mode():
        for offset in range(0, indices.size, CHUNK):
            sl = indices[offset : offset + CHUNK]
            x = torch.as_tensor(_windows(arrays.neural, sl), device=device)
            pred = model.forward_last(x, gpu_bank) / PREDICTION_DIVISOR
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
    pred = np.concatenate(preds, axis=0)
    target = np.ascontiguousarray(arrays.velocity[indices], dtype=np.float32)
    return {
        "session": arrays.session,
        "n_eval_bins": int(indices.size),
        "r2": variance_weighted_r2(pred, target),
        "pred": pred,
        "target": target,
    }


def _date_key(session: str) -> str:
    token = session.replace("ses-", "")
    return token.split("T", 1)[0]


def score_arm(arm: str, epoch: int, device: torch.device) -> dict[str, Any]:
    model, ckpt = _load_ema(arm, epoch, device)
    paths = _index_split("held-in-minival")
    sessions = {name: load_session_arrays(paths[name], name, skip_first3=False) for name in HELDIN_SESSIONS}
    banks = materialize_banks(sessions)
    per: dict[str, float] = {}
    by_date_pred: dict[str, list[np.ndarray]] = {}
    by_date_tgt: dict[str, list[np.ndarray]] = {}
    rows = []
    for name in HELDIN_SESSIONS:
        row = _score_session(model, banks[name], sessions[name], device)
        per[name] = float(row["r2"])
        date = _date_key(name)
        by_date_pred.setdefault(date, []).append(row["pred"])
        by_date_tgt.setdefault(date, []).append(row["target"])
        rows.append({"session": name, "n_eval_bins": row["n_eval_bins"], "r2": per[name]})
    date_r2 = {
        date: variance_weighted_r2(np.concatenate(by_date_pred[date]), np.concatenate(by_date_tgt[date]))
        for date in sorted(by_date_pred)
    }
    return {
        "arm": arm,
        "epoch": epoch,
        "view": "EMA",
        "ckpt": str(ckpt),
        "split": "held-in-minival",
        "disclosure": "source-development minival, not official hidden held-out",
        "equal_session_mean": float(np.mean(list(per.values()))),
        "worst_session": min(per, key=per.get),
        "worst_session_r2": float(min(per.values())),
        "equal_date_mean": float(np.mean(list(date_r2.values()))),
        "per_session_r2": per,
        "per_date_concat_r2": date_r2,
        "sessions": rows,
    }


def held_out_availability() -> dict[str, Any]:
    root = Path("/home/xinyuan/Work_host/SPINT/SPINT-main/data/000954")
    return {
        "held_out_eval_present": (root / "sub-HumanPitt-held-out-eval").is_dir()
        or bool(list(root.rglob("*held-out-eval*"))),
        "held_out_calib_nwb": len(list((root / "sub-HumanPitt-held-out-calib").rglob("*.nwb"))),
        "note": (
            "Official H1 held-out query labels are not on disk. "
            "held-out-calib is M3 support only and is not scored as independent R2."
        ),
    }


def main() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for local H1 scoring")
    device = torch.device("cuda:0")
    report = {
        "schema": "h1_temporal_local_minival_r2_v1",
        "updated": datetime.now(timezone.utc).isoformat(),
        "held_out": held_out_availability(),
        "cells": [],
    }
    for arm, epoch, _view in VIEWS:
        report["cells"].append(score_arm(arm, epoch, device))
        dest = RESULT_ROOT / "local_minival_r2.json"
        dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({cell["arm"] + f"_e{cell['epoch']}": cell["equal_session_mean"] for cell in report["cells"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
