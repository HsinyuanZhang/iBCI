"""Build a complete numeric pack for Astra from sealed small-stability roots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config as cfg
from .score import CELLS, LR_CELLS, VIEWS


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _epoch_events(metrics_path: Path) -> list[dict[str, Any]]:
    rows = []
    if not metrics_path.is_file():
        return rows
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("event") == "epoch":
            rows.append(payload)
    return rows


def collect_minival_rows(cell: str, seed: int, root: Path) -> list[dict[str, Any]]:
    dest = root / cell.replace("-", "_") / f"seed{seed}"
    out: list[dict[str, Any]] = []
    for event in _epoch_events(dest / "metrics.jsonl"):
        epoch = int(event["epoch"])
        for view, key in (("RAW", "minival_raw"), ("EMA", "minival_ema")):
            block = event.get(key) or {}
            per = block.get("per_session_r2") or {}
            values = [float(v) for v in per.values()] if per else []
            median = float(np.median(values)) if values else float("nan")
            for session, score in sorted(per.items()):
                out.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "view": view,
                        "epoch": epoch,
                        "session": session,
                        "r2": float(score),
                        "equal_session_mean": float(block.get("equal_session_mean", float("nan"))),
                        "session_median": median,
                        "train_mse": float(event.get("train_mse", float("nan"))),
                        "lr": float(event.get("lr", float("nan"))),
                    }
                )
    return out


def collect_ext4_rows(cell: str, seed: int, root: Path) -> list[dict[str, Any]]:
    scan_path = root / cell.replace("-", "_") / f"seed{seed}" / "ext4_epoch_scan.json"
    if not scan_path.is_file():
        return []
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for view in VIEWS:
        epochs = scan.get("views", {}).get(view, {}).get("all_epochs", {})
        for epoch_s, row in epochs.items():
            per = row.get("per_session") or {}
            for session, payload in sorted(per.items()):
                out.append(
                    {
                        "cell": cell,
                        "seed": seed,
                        "view": view,
                        "epoch": int(epoch_s),
                        "session": session,
                        "r2": float(payload["r2"]),
                        "window_count": int(payload["window_count"]),
                        "R_session_equal_mean": float(row["R_session_equal_mean"]),
                        "R_date_equal_mean": float(row["R_date_equal_mean"]),
                        "delta_vs_ref": float(row["delta_vs_ref"]),
                    }
                )
    return out


def matrix_rows(comparisons: list[tuple[str, Path, int]]) -> list[dict[str, Any]]:
    out = []
    for label, path, seed in comparisons:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for cell, views in payload.get("reports", {}).items():
            for view, row in views.items():
                last8 = row.get("last8") or {}
                last4 = row.get("last4") or {}
                out.append(
                    {
                        "pack": label,
                        "seed": seed,
                        "cell": cell,
                        "view": view,
                        "source_pick_epoch": row.get("source_pick_epoch"),
                        "source_pick_ext4": row.get("source_pick_ext4"),
                        "source_pick_delta": row.get("source_pick_delta"),
                        "endpoint12": row.get("endpoint12"),
                        "endpoint24": row.get("endpoint24"),
                        "last4_mean": last4.get("mean"),
                        "last4_std": last4.get("std"),
                        "last8_mean": last8.get("mean"),
                        "last8_std": last8.get("std"),
                        "visible_epoch": row.get("visible_ext4_pick_epoch"),
                        "visible_ext4": row.get("visible_ext4_pick"),
                    }
                )
    return out


def write_seed42_pack(dest: Path | None = None) -> Path:
    dest = dest or (cfg.RESULT_ROOT / "astra_pack_v1")
    dest.mkdir(parents=True, exist_ok=True)
    minival: list[dict[str, Any]] = []
    ext4: list[dict[str, Any]] = []
    for cell in CELLS:
        minival.extend(collect_minival_rows(cell, 42, cfg.INTERIM_RUN_ROOT))
        ext4.extend(collect_ext4_rows(cell, 42, cfg.INTERIM_RUN_ROOT))
    for cell in LR_CELLS:
        minival.extend(collect_minival_rows(cell, 42, cfg.LR_CONTRAST_RUN_ROOT))
        ext4.extend(collect_ext4_rows(cell, 42, cfg.LR_CONTRAST_RUN_ROOT))
    _write_csv(dest / "seed42_minival_per_session.csv", minival)
    _write_csv(dest / "seed42_ext4_per_session.csv", ext4)
    matrix = matrix_rows(
        [
            ("interim_3e4", cfg.INTERIM_RUN_ROOT / "comparison.json", 42),
            ("lr_1e4", cfg.LR_CONTRAST_RUN_ROOT / "comparison.json", 42),
        ]
    )
    _write_csv(dest / "seed42_matrix.csv", matrix)
    minival43: list[dict[str, Any]] = []
    ext443: list[dict[str, Any]] = []
    for cell in CELLS:
        minival43.extend(collect_minival_rows(cell, 43, cfg.INTERIM_RUN_ROOT))
        ext443.extend(collect_ext4_rows(cell, 43, cfg.INTERIM_RUN_ROOT))
    for cell in LR_CELLS:
        minival43.extend(collect_minival_rows(cell, 43, cfg.LR_CONTRAST_RUN_ROOT))
        ext443.extend(collect_ext4_rows(cell, 43, cfg.LR_CONTRAST_RUN_ROOT))
    if minival43:
        _write_csv(dest / "seed43_minival_per_session.csv", minival43)
    if ext443:
        _write_csv(dest / "seed43_ext4_per_session.csv", ext443)
    matrix43 = matrix_rows(
        [
            ("interim_3e4", cfg.INTERIM_RUN_ROOT / "comparison_seed43.json", 43),
            ("lr_1e4", cfg.LR_CONTRAST_RUN_ROOT / "comparison_seed43.json", 43),
        ]
    )
    if matrix43:
        _write_csv(dest / "seed43_matrix.csv", matrix43)
        matrix.extend(matrix43)
        _write_csv(dest / "matrix_2x2_both_seeds.csv", matrix)
    seed43_ready = bool(matrix43) and bool(ext443)
    (dest / "STATUS.json").write_text(
        json.dumps(
            {
                "seed42_complete": True,
                "seed43_trains": "DONE" if seed43_ready else "PENDING",
                "seed43_ext4": "DONE" if seed43_ready else "PENDING",
                "cells_planned": list(CELLS + LR_CELLS),
                "ready_for_astra": seed43_ready,
                "note": "Full 2x2 x two seeds plus P preflight. No 48ep/Mamba/P-12ep.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return dest
