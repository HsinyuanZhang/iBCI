"""Historical last-k patch. No new scoring. Writes only into the new root."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import (
    OLD_ANALYSIS_SHA256,
    OLD_COMPARISON_SHA256,
    OLD_ROOT,
    R_REF_DATE,
    R_REF_SESSION,
    REPO_ROOT,
    require,
)

FORBIDDEN_MAMBA_MINIVAL_SUBTRACTION = 0.309186 - 0.3582396424175502

# Workorder §2 published last-k (ddof=0). Recomputed below from sealed scans.
PUBLISHED = {
    "transformer": {
        "source_pick_epoch": 9,
        "source_pick_ext4": 0.360008,
        "delta_vs_ref": 0.001768,
        "endpoint24": 0.321388,
        "last4_mean": 0.332195,
        "last4_std": 0.008984,
        "last8_mean": 0.340155,
        "last8_std": 0.024152,
        "visible_ext4_epoch20": 0.371954,
    },
    "mamba": {
        "source_pick_epoch": 24,
        "source_pick_ext4": 0.343016,
        "delta_vs_ref": -0.015224,
        "endpoint24": 0.343016,
        "last4_mean": 0.339657,
        "last4_std": 0.015601,
        "last8_mean": 0.348886,
        "last8_std": 0.016601,
        "visible_ext4_epoch18": 0.377224,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_k_stats(values: np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size >= 1, "last-k empty")
    diffs = np.diff(array)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "std_ddof": 0,
        "mean_abs_delta": float(np.abs(diffs).mean()) if diffs.size else 0.0,
        "max_abs_delta": float(np.abs(diffs).max()) if diffs.size else 0.0,
        "range": [float(array.min()), float(array.max())],
        "n": int(array.size),
    }


def _epoch_scores(scan: Mapping[str, Any]) -> dict[int, float]:
    rows = scan["all_epochs"]
    return {int(epoch): float(row["R_session_equal_mean"]) for epoch, row in rows.items()}


def _load_scan(rel_arm: str) -> tuple[dict[str, Any], Path]:
    path = OLD_ROOT / "arms" / rel_arm / "seed42_shuffled_e13_24" / "ext4_epoch_scan.json"
    require(path.is_file(), f"missing sealed scan: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def build_historical_trajectory_summary() -> dict[str, Any]:
    trf_scan, trf_path = _load_scan("B-TRANSFORMER")
    mamba_scan, mamba_path = _load_scan("B-MAMBA")
    trf_scores = _epoch_scores(trf_scan)
    mamba_scores = _epoch_scores(mamba_scan)
    require(set(trf_scores) == set(range(1, 25)), "transformer scan missing epochs")
    require(set(mamba_scores) == set(range(1, 25)), "mamba scan missing epochs")

    trf_last4 = last_k_stats([trf_scores[e] for e in range(21, 25)])
    trf_last8 = last_k_stats([trf_scores[e] for e in range(17, 25)])
    mamba_last4 = last_k_stats([mamba_scores[e] for e in range(21, 25)])
    mamba_last8 = last_k_stats([mamba_scores[e] for e in range(17, 25)])

    pub_t = PUBLISHED["transformer"]
    pub_m = PUBLISHED["mamba"]
    require(abs(trf_last4["mean"] - pub_t["last4_mean"]) < 5e-6, "transformer last4 mean drift")
    require(abs(trf_last4["std"] - pub_t["last4_std"]) < 5e-6, "transformer last4 std drift")
    require(abs(trf_last8["mean"] - pub_t["last8_mean"]) < 5e-6, "transformer last8 mean drift")
    require(abs(trf_last8["std"] - pub_t["last8_std"]) < 5e-6, "transformer last8 std drift")
    require(abs(mamba_last4["mean"] - pub_m["last4_mean"]) < 5e-6, "mamba last4 mean drift")
    require(abs(mamba_last4["std"] - pub_m["last4_std"]) < 5e-6, "mamba last4 std drift")
    require(abs(mamba_last8["mean"] - pub_m["last8_mean"]) < 5e-6, "mamba last8 mean drift")
    require(abs(mamba_last8["std"] - pub_m["last8_std"]) < 5e-6, "mamba last8 std drift")

    trf_pick = float(trf_scores[9])
    mamba_pick = float(mamba_scores[24])
    same_surface = mamba_pick - R_REF_SESSION
    require(abs(same_surface + 0.015224) < 5e-6, "same-surface gap must be -0.015224")
    require(abs(same_surface - FORBIDDEN_MAMBA_MINIVAL_SUBTRACTION) > 1e-3, "forbidden minival subtraction")

    analysis = OLD_ROOT / "ANALYSIS_B_TRAJECTORY_INSTABILITY.md"
    comparison = OLD_ROOT / "comparison.csv"
    return {
        "schema": "m2_b_small_stability_v1_historical_last_k",
        "R_REF_session": R_REF_SESSION,
        "R_REF_date": R_REF_DATE,
        "std_ddof": 0,
        "last_k_is_not_a_checkpoint": True,
        "note_same_surface": (
            "Mamba source-pick ext-4 minus REF is -0.015224; do not subtract minival 0.309"
        ),
        "forbidden_mamba_minival_subtraction": FORBIDDEN_MAMBA_MINIVAL_SUBTRACTION,
        "transformer": {
            "source_pick_epoch": 9,
            "source_pick_ext4": pub_t["source_pick_ext4"],
            "source_pick_ext4_full": trf_pick,
            "delta_vs_ref": pub_t["delta_vs_ref"],
            "endpoint24": pub_t["endpoint24"],
            "endpoint24_full": float(trf_scores[24]),
            "last4": trf_last4,
            "last8": trf_last8,
            "visible_ext4_epoch20": pub_t["visible_ext4_epoch20"],
            "visible_ext4_epoch20_full": float(trf_scores[20]),
        },
        "mamba": {
            "source_pick_epoch": 24,
            "source_pick_ext4": pub_m["source_pick_ext4"],
            "source_pick_ext4_full": mamba_pick,
            "delta_vs_ref": pub_m["delta_vs_ref"],
            "same_surface_gap": same_surface,
            "endpoint24": pub_m["endpoint24"],
            "endpoint24_full": float(mamba_scores[24]),
            "last4": mamba_last4,
            "last8": mamba_last8,
            "visible_ext4_epoch18": pub_m["visible_ext4_epoch18"],
            "visible_ext4_epoch18_full": float(mamba_scores[18]),
            "source_minival_e24_not_used_for_gap": 0.309186,
        },
        "inputs": {
            "old_root": str(OLD_ROOT.relative_to(REPO_ROOT)),
            "analysis_sha256": file_sha256(analysis),
            "comparison_csv_sha256": file_sha256(comparison),
            "transformer_ext4_scan_sha256": file_sha256(trf_path),
            "mamba_ext4_scan_sha256": file_sha256(mamba_path),
            "expected_analysis_sha256": OLD_ANALYSIS_SHA256,
            "expected_comparison_csv_sha256": OLD_COMPARISON_SHA256,
        },
        "window_counts": {
            "ses-2020-10-30-Run1": 519,
            "ses-2020-10-30-Run2": 490,
            "ses-2020-11-18-Run1": 425,
            "ses-2020-11-19-Run1": 635,
            "total": 2069,
        },
    }


def write_historical_trajectory_summary(root: Path) -> dict[str, Any]:
    summary = build_historical_trajectory_summary()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "historical_trajectory_summary.json"
    csv_path = root / "historical_trajectory_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        {
            "arm": "Transformer",
            "source_pick_epoch": summary["transformer"]["source_pick_epoch"],
            "source_pick_ext4": summary["transformer"]["source_pick_ext4"],
            "delta_vs_ref": summary["transformer"]["delta_vs_ref"],
            "endpoint24": summary["transformer"]["endpoint24"],
            "last4_mean": summary["transformer"]["last4"]["mean"],
            "last4_std": summary["transformer"]["last4"]["std"],
            "last8_mean": summary["transformer"]["last8"]["mean"],
            "last8_std": summary["transformer"]["last8"]["std"],
            "visible_ext4": summary["transformer"]["visible_ext4_epoch20"],
        },
        {
            "arm": "Mamba",
            "source_pick_epoch": summary["mamba"]["source_pick_epoch"],
            "source_pick_ext4": summary["mamba"]["source_pick_ext4"],
            "delta_vs_ref": summary["mamba"]["delta_vs_ref"],
            "endpoint24": summary["mamba"]["endpoint24"],
            "last4_mean": summary["mamba"]["last4"]["mean"],
            "last4_std": summary["mamba"]["last4"]["std"],
            "last8_mean": summary["mamba"]["last8"]["mean"],
            "last8_std": summary["mamba"]["last8"]["std"],
            "visible_ext4": summary["mamba"]["visible_ext4_epoch18"],
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary
