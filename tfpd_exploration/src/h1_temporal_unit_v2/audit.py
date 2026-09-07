"""Read-only audit of the old H1 product. Does not retrain or submit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import RESULT_ROOT as OLD_ROOT
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.config import SEED, WINDOW
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import (
    _index_split,
    load_session_arrays,
    materialize_banks,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
)

from . import plan
from .metrics import report_pair, std_l2

VIEWS = (
    ("flat", 5, "EMA"),
    ("flat", 12, "EMA"),
    ("flat", 12, "RAW"),
    ("route", 5, "EMA"),
    ("route", 12, "EMA"),
    ("route", 12, "RAW"),
)


def _load_view(arm: str, epoch: int, view: str, device: torch.device):
    path = OLD_ROOT / arm / f"epoch_{epoch:03d}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if arm == "flat":
        model = H1TemporalFlatDecoder(seed=SEED)
    else:
        model = H1TemporalRouteDecoder(seed=SEED, flat_template=H1TemporalFlatDecoder(seed=SEED))
    key = "ema" if view == "EMA" else "raw_state_dict"
    state = ckpt["ema"]["shadow"] if view == "EMA" else ckpt["raw_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, path


def _full_history_windows(neural: np.ndarray, last_bins: np.ndarray) -> np.ndarray:
    out = np.zeros((int(last_bins.size), WINDOW, neural.shape[1]), dtype=np.float32)
    for i, last in enumerate(last_bins.tolist()):
        last = int(last)
        if last + 1 < WINDOW:
            raise RuntimeError("probe window is not full W700 history")
        out[i] = neural[last + 1 - WINDOW : last + 1]
    return out


def _predict(model, bank, neural: np.ndarray, last_bins: np.ndarray, device: torch.device) -> np.ndarray:
    gpu_bank = type(bank)(E0=bank.E0.to(device), T=bank.T.to(device), unit_mask=bank.unit_mask.to(device))
    preds = []
    with torch.inference_mode():
        for offset in range(0, last_bins.size, 8):
            sl = last_bins[offset : offset + 8]
            x = torch.as_tensor(_full_history_windows(neural, sl), device=device)
            preds.append(model.forward_last(x, gpu_bank).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def _pick_probe(starts: list[int], n: int, *, offset: int) -> np.ndarray:
    full = [start + WINDOW - 1 for start in starts if start >= 0]
    if len(full) < offset + n:
        raise RuntimeError(f"not enough full-history windows: {len(full)}")
    # Evenly spaced, skip the first `offset` to separate train/dev probes.
    chosen = full[offset : offset + n]
    return np.asarray(chosen, dtype=np.int64)


def _unit_reports(pred_raw: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    deploy = pred_raw / plan.DEPLOY_DIVISOR
    return {
        "train_units_pred_vs_y": report_pair(pred_raw, y),
        "deploy_units_pred_over_20_vs_y": report_pair(deploy, y),
        "train_units_pred_vs_20y": report_pair(pred_raw, y * plan.TRAIN_TARGET_SCALE),
    }


def audit_probe(device: torch.device) -> dict[str, Any]:
    calib = _index_split("held-in-calib")
    mini = _index_split("held-in-minival")
    name = plan.PROBE_SESSION
    train_arr = load_session_arrays(calib[name], name, skip_first3=True)
    dev_arr = load_session_arrays(mini[name], name, skip_first3=False)
    banks = materialize_banks({name: train_arr})
    bank = banks[name]
    train_last = _pick_probe(train_arr.query_starts, plan.PROBE_N, offset=0)
    dev_last = _pick_probe(dev_arr.query_starts, plan.PROBE_N, offset=16)
    cells = []
    for arm, epoch, view in VIEWS:
        model, ckpt = _load_view(arm, epoch, view, device)
        row: dict[str, Any] = {"arm": arm, "epoch": epoch, "view": view, "ckpt": str(ckpt)}
        for split, arrays, last in (("train", train_arr, train_last), ("dev", dev_arr, dev_last)):
            pred = _predict(model, bank, arrays.neural, last, device)
            y = arrays.velocity[last]
            stats = _unit_reports(pred, y)
            # Input sensitivity on the first window: noise should change output if not constant.
            x0 = torch.as_tensor(_full_history_windows(arrays.neural, last[:1]), device=device)
            gpu_bank = type(bank)(E0=bank.E0.to(device), T=bank.T.to(device), unit_mask=bank.unit_mask.to(device))
            with torch.inference_mode():
                p0 = model.forward_last(x0, gpu_bank)
                p1 = model.forward_last(x0 + 1.0, gpu_bank)
            stats["input_abs_delta_l2"] = float((p1 - p0).abs().square().sum().sqrt().item())
            row[split] = stats
        cells.append(row)
    return {
        "session": name,
        "train_last_bins": train_last.tolist(),
        "dev_last_bins": dev_last.tolist(),
        "full_w700_only": True,
        "cells": cells,
    }


def audit_baselines() -> dict[str, Any]:
    """Zero/mean on sealed stride-4 windows vs streaming eval_mask. No model."""
    mini = _index_split("held-in-minival")
    sealed_pred: list[np.ndarray] = []
    sealed_y: list[np.ndarray] = []
    stream_y: list[np.ndarray] = []
    for name in plan.HELDIN_SESSIONS:
        arrays = load_session_arrays(mini[name], name, skip_first3=False)
        last = np.asarray([s + WINDOW - 1 for s in arrays.query_starts], dtype=np.int64)
        sealed_y.append(arrays.velocity[last])
        sealed_pred.append(np.zeros_like(arrays.velocity[last]))
        idx = np.flatnonzero(np.asarray(arrays.eval_mask, dtype=bool))
        stream_y.append(arrays.velocity[idx])
    sy = np.concatenate(sealed_y, axis=0)
    ty = np.concatenate(stream_y, axis=0)
    return {
        "sealed_stride4_full_history": {
            "n_bins": int(sy.shape[0]),
            "zero": report_pair(np.zeros_like(sy), sy),
            "mean": report_pair(np.broadcast_to(sy.mean(axis=0, keepdims=True), sy.shape), sy),
            "target_std_l2": std_l2(sy),
        },
        "streaming_eval_mask": {
            "n_bins": int(ty.shape[0]),
            "zero": report_pair(np.zeros_like(ty), ty),
            "mean": report_pair(np.broadcast_to(ty.mean(axis=0, keepdims=True), ty.shape), ty),
            "target_std_l2": std_l2(ty),
            "note": "includes windows shorter than W700; not the sealed 2908-window selector",
        },
        "do_not_mix_caliber": True,
    }


def run_audit(device: torch.device) -> dict[str, Any]:
    plan.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": plan.SCHEMA + "_audit",
        "revision": plan.REVISION,
        "updated": datetime.now(timezone.utc).isoformat(),
        "no_submit": True,
        "old_product_not_performance_ready": True,
        "unit_contract": {
            "old_train_target": "raw y from NWB",
            "old_score_and_container": "pred / 20 vs raw y",
            "official_loader_scale": "none; returns the same velocity",
            "revision_train_target": "20y",
            "revision_deploy": "pred / 20 vs y",
        },
        "baselines": audit_baselines(),
        "probe": audit_probe(device),
    }
    dest = plan.RESULT_ROOT / "unit_learning_audit.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
