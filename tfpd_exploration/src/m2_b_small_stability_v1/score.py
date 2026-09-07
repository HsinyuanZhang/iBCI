"""Ext-4 scan after both 24-epoch trains finish. Does not read ext-4 during training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tfpd_exploration.src.m2_dual_track_v1 import champion as old_champion
from tfpd_exploration.src.m2_dual_track_v1 import contracts as old_contracts
from tfpd_exploration.src.m2_dual_track_v1 import data as old_data
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

from . import config as cfg
from .decoder import SmallTransformerDecoder
from .report import last_k_stats


VIEWS = ("RAW", "EMA")
CELLS = (cfg.CELL_S0, cfg.CELL_S1)
LR_CELLS = (cfg.CELL_N0, cfg.CELL_N1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selection_manifest_path(root: Path, seed: int) -> Path:
    if int(seed) == 42:
        return root / "source_selection_manifest.json"
    return root / f"source_selection_manifest_seed{int(seed)}.json"


def comparison_path(root: Path, seed: int) -> Path:
    if int(seed) == 42:
        return root / "comparison.json"
    return root / f"comparison_seed{int(seed)}.json"


def cell_dest(cell: str, *, seed: int = cfg.SEED, root: Path | None = None) -> Path:
    base = root or cfg.ACTIVE_RUN_ROOT
    return base / cell.replace("-", "_") / f"seed{seed}"


def load_summary(cell: str, *, seed: int = cfg.SEED, root: Path | None = None) -> dict[str, Any]:
    path = cell_dest(cell, seed=seed, root=root) / "summary.json"
    cfg.require(path.is_file(), f"missing finished summary {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_both_summaries(
    *,
    seed: int = cfg.SEED,
    root: Path | None = None,
    cells: tuple[str, ...] = CELLS,
) -> dict[str, dict[str, Any]]:
    return {cell: load_summary(cell, seed=seed, root=root) for cell in cells}


def select_epoch(scores: dict[int, float]) -> int:
    finite = {int(epoch): float(score) for epoch, score in scores.items() if np.isfinite(score)}
    cfg.require(bool(finite), "no finite scores for epoch-pick")
    best = max(finite.values())
    tied = [epoch for epoch, score in finite.items() if abs(best - score) <= 1.0e-10]
    return min(tied)


def ref_per_session() -> dict[str, float]:
    payload = json.loads((cfg.OLD_ROOT / "stage0" / "ref_clean.json").read_text(encoding="utf-8"))
    rows = payload["report"]["per_session"]
    return {session: float(rows[session]["r2"]) for session in old_plan.EXT4_SESSIONS}


def _lookup_minival(raw: dict[str, Any], epoch: int) -> float:
    if epoch in raw:
        return float(raw[epoch])
    return float(raw[str(epoch)])


def seal_source_picks(
    *,
    seed: int = cfg.SEED,
    root: Path | None = None,
    cells: tuple[str, ...] = CELLS,
    primary: str = cfg.PRIMARY_CANDIDATE,
) -> dict[str, Any]:
    summaries = require_both_summaries(seed=seed, root=root, cells=cells)
    dest = root or cfg.ACTIVE_RUN_ROOT
    payload = {
        "schema": "m2_b_small_stability_v1_source_selection",
        "primary_candidate": primary,
        "seed": seed,
        "picks": {
            cell: {
                "RAW": {
                    "epoch": int(summaries[cell]["source_pick_raw"]),
                    "source_minival": _lookup_minival(
                        summaries[cell]["epoch_minival_raw"], int(summaries[cell]["source_pick_raw"])
                    ),
                },
                "EMA": {
                    "epoch": int(summaries[cell]["source_pick_ema"]),
                    "source_minival": _lookup_minival(
                        summaries[cell]["epoch_minival_ema"], int(summaries[cell]["source_pick_ema"])
                    ),
                },
            }
            for cell in cells
        },
        "init_sha256": {cell: summaries[cell]["init_sha256"] for cell in cells},
        "sealed": datetime.now(timezone.utc).isoformat(),
        "note": "Source-minival picks only. Ext-4 is scored after this file exists.",
    }
    path = selection_manifest_path(dest, seed)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        cfg.require(
            existing["picks"] == payload["picks"] and existing["primary_candidate"] == payload["primary_candidate"],
            "refusing to rewrite a different source-selection manifest",
        )
        return existing
    _write_json(path, payload)
    return payload


def apply_view(model: nn.Module, ckpt: dict[str, Any], view: str) -> None:
    model.load_state_dict(ckpt["raw_state_dict"])
    if view == "RAW":
        return
    cfg.require(view == "EMA", f"unknown view {view}")
    shadow = ckpt["ema"]["shadow"]
    named = model.trainable_parameters()
    missing = set(named) - set(shadow)
    extra = set(shadow) - set(named)
    cfg.require(not missing and not extra, f"EMA/RAW key mismatch missing={missing} extra={extra}")
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(shadow[name].to(device=param.device, dtype=param.dtype))


def score_ext4_model(model: nn.Module, banks: dict[str, Any], device: torch.device) -> dict[str, Any]:
    per_session: dict[str, float] = {}
    rows: dict[str, Any] = {}
    model.eval()
    for session, bank in banks.items():
        targets: list[np.ndarray] = []
        preds: list[np.ndarray] = []
        for batch in old_data.iter_session_batches(
            bank,
            batch_size=old_plan.EFFECTIVE_BATCH,
            device=device,
            target_space=old_plan.SCORING_TARGET_SPACE,
        ):
            with torch.inference_mode():
                raw = model.forward_last(batch.X, batch.bank, batch.unit_mask)
            preds.append(
                np.ascontiguousarray(raw.detach().cpu().numpy() / old_plan.BEHAVIOR_SCALE, dtype=np.float32)
            )
            targets.append(batch.last_target.detach().cpu().numpy())
        target = np.concatenate(targets, axis=0)
        pred = np.concatenate(preds, axis=0)
        r2 = old_contracts.variance_weighted_r2(target, pred)
        per_session[session] = float(r2)
        rows[session] = {
            "r2": float(r2),
            "window_count": int(target.shape[0]),
            "prediction_digest": old_champion.array_sha256(pred),
        }
    summary = old_contracts.summarize_sessions(per_session)
    dates = old_contracts.date_equal_mean(per_session)
    r_session = float(summary["equal_session_mean"])
    return {
        "per_session": rows,
        "summary": summary,
        "date_sensitivity": dates,
        "R_session_equal_mean": r_session,
        "R_date_equal_mean": dates["equal_date_mean"],
        "delta_vs_ref": r_session - cfg.R_REF_SESSION,
        "development_evidence": True,
    }


def _epoch_series(scan: dict[str, Any], view: str) -> dict[int, float]:
    rows = scan["views"][view]["all_epochs"]
    return {int(epoch): float(row["R_session_equal_mean"]) for epoch, row in rows.items()}


def _view_report(scan: dict[str, Any], view: str, source_pick: int) -> dict[str, Any]:
    series = _epoch_series(scan, view)
    last4 = [series[epoch] for epoch in range(21, 25)]
    last8 = [series[epoch] for epoch in range(17, 25)]
    visible = select_epoch(series)
    return {
        "source_pick_epoch": int(source_pick),
        "source_pick_ext4": series[int(source_pick)],
        "source_pick_delta": series[int(source_pick)] - cfg.R_REF_SESSION,
        "endpoint12": series[12],
        "endpoint24": series[24],
        "last4": last_k_stats(last4),
        "last8": last_k_stats(last8),
        "visible_ext4_pick_epoch": visible,
        "visible_ext4_pick": series[visible],
        "visible_ext4_delta": series[visible] - cfg.R_REF_SESSION,
    }


def _route(primary: dict[str, Any], per_session: dict[str, float], ref_sessions: dict[str, float]) -> str:
    session_deltas = {session: per_session[session]["r2"] - ref_sessions[session] for session in ref_sessions}
    if primary["source_pick_delta"] >= 0.005 and all(delta >= -0.05 for delta in session_deltas.values()):
        last8_mean = primary["last8"]["mean"]
        if last8_mean >= cfg.R_REF_SESSION:
            return "PERFORMANCE_CANDIDATE"
        return "PICK_SENSITIVE_CANDIDATE"
    if -0.03 <= primary["source_pick_delta"] < 0.005:
        return "COMPLETE_NOT_OVER_GATE"
    return "MIGRATION_GAP"


def run_ext4_scan(
    *,
    seed: int = cfg.SEED,
    root: Path | None = None,
    cells: tuple[str, ...] = CELLS,
    primary: str = cfg.PRIMARY_CANDIDATE,
) -> dict[str, Any]:
    dest_root = root or cfg.ACTIVE_RUN_ROOT
    comparison_file = comparison_path(dest_root, seed)
    cfg.require(not comparison_file.exists(), f"refusing to overwrite {comparison_file}")
    summaries = require_both_summaries(seed=seed, root=dest_root, cells=cells)
    inits = {summaries[cell]["init_sha256"] for cell in cells}
    cfg.require(len(inits) == 1, "init SHA mismatch")
    picks = seal_source_picks(seed=seed, root=dest_root, cells=cells, primary=primary)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    banks = {
        session: old_data.load_session_bank("ext4", session, device=device)
        for session in old_plan.EXT4_SESSIONS
    }
    ref_sessions = ref_per_session()
    scans: dict[str, Any] = {}
    for cell in cells:
        dest = cell_dest(cell, seed=seed, root=dest_root)
        views: dict[str, Any] = {}
        model = SmallTransformerDecoder(seed=seed).to(device)
        for view in VIEWS:
            all_epochs: dict[str, Any] = {}
            for epoch in range(1, cfg.EPOCHS + 1):
                ckpt_path = dest / f"epoch_{epoch:03d}.pt"
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                apply_view(model, ckpt, view)
                report = score_ext4_model(model, banks, device)
                report["epoch"] = epoch
                report["view"] = view
                report["cell"] = cell
                report["ckpt"] = str(ckpt_path)
                all_epochs[str(epoch)] = report
            views[view] = {"all_epochs": all_epochs}
        scan = {
            "cell": cell,
            "seed": seed,
            "R_REF_clean": cfg.R_REF_SESSION,
            "R_REF_date": cfg.R_REF_DATE,
            "views": views,
        }
        _write_json(dest / "ext4_epoch_scan.json", scan)
        scans[cell] = scan

    reports = {
        cell: {
            view: _view_report(scans[cell], view, int(picks["picks"][cell][view]["epoch"]))
            for view in VIEWS
        }
        for cell in cells
    }
    primary_cell, primary_view = primary.split("/")
    primary_report = reports[primary_cell][primary_view]
    primary_row = scans[primary_cell]["views"][primary_view]["all_epochs"][str(primary_report["source_pick_epoch"])]
    routing = _route(primary_report, primary_row["per_session"], ref_sessions)
    comparison = {
        "schema": "m2_b_small_stability_v1_comparison",
        "primary_candidate": primary,
        "primary_source_pick": primary_report,
        "routing": routing,
        "R_REF_session": cfg.R_REF_SESSION,
        "R_REF_date": cfg.R_REF_DATE,
        "reports": reports,
        "seed": seed,
        "auto_seed43": False,
        "auto_48_epoch": False,
        "auto_mamba": False,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(comparison_file, comparison)
    return comparison
