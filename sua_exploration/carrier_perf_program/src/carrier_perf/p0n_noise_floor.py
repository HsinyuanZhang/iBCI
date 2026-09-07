"""P0N: M2 noise-floor audit from sealed native_mua_t4_v1 artifacts.

Native MUA v1 run directories often keep only ``best.ckpt`` + summary CSVs, but the
supervisor logs under ``results/native_mua_t4_v1/logs/`` contain Lightning
progress lines with ``val_heldin/r2_mean`` for epochs 0..11.

V4 protocol epoch ``E`` (1-indexed, window 5..12) maps to Lightning index
``E - 1``. Summary scores in ``aggregate_m2.json`` match the *maximum* of those
curves (best-checkpoint argmax) and must **not** be used to freeze thresholds.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from carrier_perf.protocol import (
    EXPECTED_EPOCH_WINDOW,
    mean,
    sample_std,
    sigma_delta_paired,
    two_sigma_paired,
)

SCHEMA = "carrier_perf_p0n_m2_sigma_v1"
REQUIRED_CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
MISSING_CELL = "fold2_seed43"
ARMS = ("f0", "t4", "ts4")
CONTRASTS = (("T4_minus_F0", "t4", "f0"), ("T4_minus_TS4", "t4", "ts4"))

# Lightning: `Epoch 11: 100%|...| val_heldin/r2_mean=0.638`
_EPOCH_R2_RE = re.compile(
    r"Epoch\s+(\d+):\s+100%.*?val_heldin/r2_mean=([0-9.eE+-]+)"
)


@dataclass
class RunCurveStatus:
    cell: str
    arm: str
    artifact_dir: str
    exists: bool
    has_epoch_curves: bool
    curve_source: str | None
    notes: list[str] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_lightning_epoch_r2(log_path: Path) -> dict[int, float]:
    """Parse last 100%-complete ``val_heldin/r2_mean`` per Lightning epoch index."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    by_lightning: dict[int, float] = {}
    for match in _EPOCH_R2_RE.finditer(text):
        by_lightning[int(match.group(1))] = float(match.group(2))
    return by_lightning


def lightning_to_protocol_epoch(lightning_index: int) -> int:
    """Lightning 0-indexed epoch -> protocol 1-indexed epoch."""
    return int(lightning_index) + 1


def protocol_window_from_lightning(by_lightning: Mapping[int, float]) -> dict[int, float]:
    """Return protocol-epoch -> r2 for EXPECTED_EPOCH_WINDOW; raise if incomplete."""
    out: dict[int, float] = {}
    for lightning_idx, value in by_lightning.items():
        protocol = lightning_to_protocol_epoch(lightning_idx)
        if protocol in EXPECTED_EPOCH_WINDOW:
            out[protocol] = float(value)
    missing = [e for e in EXPECTED_EPOCH_WINDOW if e not in out]
    if missing:
        raise ValueError(f"log missing protocol epochs {missing}; have {sorted(out)}")
    return out


def default_log_path(logs_dir: Path, arm: str, cell: str) -> Path:
    """Map fold1_seed42 -> m2_{arm}_f1_s42.log."""
    fold, seed = cell.split("_seed")
    fold_num = fold.replace("fold", "")
    return logs_dir / f"m2_{arm}_f{fold_num}_s{seed}.log"


def discover_epoch_curves(run_dir: Path) -> tuple[bool, str | None, list[str]]:
    """Search *inside a run artifact dir* only (legacy). Prefer log parser for v1."""
    notes: list[str] = []
    if not run_dir.is_dir():
        return False, None, [f"missing run dir: {run_dir}"]

    candidates = [
        run_dir / "metrics_per_epoch.csv",
        run_dir / "epoch_metrics.json",
        run_dir / "val_epoch_r2.json",
        run_dir / "metrics.json",
    ]
    for path in candidates:
        if path.is_file():
            if path.suffix == ".json":
                try:
                    payload = _load_json(path)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"unreadable {path.name}: {exc}")
                    continue
                if _json_has_epoch_window(payload):
                    return True, str(path), notes
                notes.append(f"{path.name} present but lacks epochs {EXPECTED_EPOCH_WINDOW}")
            else:
                text = path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
                header = text[0].lower() if text else ""
                if "epoch" in header:
                    notes.append(
                        f"{path.name} has an epoch column but contents are not validated here; "
                        "refusing to treat as V4 curves without a reviewed parser"
                    )
                else:
                    notes.append(f"{path.name} present but no epoch column in header")

    event_hits = list(run_dir.rglob("events.out.tfevents.*"))
    if event_hits:
        notes.append(
            f"found {len(event_hits)} tfevents file(s); scaffold refuses silent parse"
        )
    ckpt = run_dir / "checkpoints"
    if ckpt.is_dir():
        names = sorted(p.name for p in ckpt.iterdir())
        if names == ["best.ckpt"]:
            notes.append("only best.ckpt present; no per-epoch checkpoint curve")
        elif names:
            notes.append(f"checkpoints present: {names[:8]}")
    summary = run_dir / "metrics_summary.csv"
    if summary.is_file():
        notes.append("metrics_summary.csv present (final scores only; not a V4 epoch window)")
    if not notes:
        notes.append("no epoch-curve artifact found under run dir")
    return False, None, notes


def _json_has_epoch_window(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("per_epoch", "epoch_r2", "epochs"):
        block = payload.get(key)
        if isinstance(block, dict):
            keys = {int(k) for k in block.keys() if str(k).isdigit()}
            # Accept either protocol 5..12 or Lightning 4..11 for presence detection.
            if set(EXPECTED_EPOCH_WINDOW).issubset(keys):
                return True
            lightning_window = {e - 1 for e in EXPECTED_EPOCH_WINDOW}
            if lightning_window.issubset(keys):
                return True
        if isinstance(block, list) and (
            block == EXPECTED_EPOCH_WINDOW
            or block == [e - 1 for e in EXPECTED_EPOCH_WINDOW]
        ):
            return True
    return False


def audit_artifact_map(aggregate: Mapping[str, Any]) -> list[RunCurveStatus]:
    artifacts = aggregate["artifacts"]["m2"]
    statuses: list[RunCurveStatus] = []
    for arm in ARMS:
        for cell in REQUIRED_CELLS:
            path = Path(artifacts[arm][cell])
            exists = path.is_dir()
            has, source, notes = discover_epoch_curves(path) if exists else (False, None, ["missing"])
            statuses.append(
                RunCurveStatus(
                    cell=cell,
                    arm=arm,
                    artifact_dir=str(path),
                    exists=exists,
                    has_epoch_curves=has,
                    curve_source=source,
                    notes=notes,
                )
            )
    return statuses


def load_v4_curves_from_logs(
    logs_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """arm -> cell -> {protocol_epochs, window_mean, window_std, curve_max, source}."""
    out: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for cell in REQUIRED_CELLS:
            path = default_log_path(logs_dir, arm, cell)
            if not path.is_file():
                raise FileNotFoundError(f"missing epoch log: {path}")
            by_lightning = parse_lightning_epoch_r2(path)
            protocol = protocol_window_from_lightning(by_lightning)
            values = [protocol[e] for e in EXPECTED_EPOCH_WINDOW]
            out[arm][cell] = {
                "source": str(path),
                "protocol_epochs": {str(e): protocol[e] for e in EXPECTED_EPOCH_WINDOW},
                "window_mean": mean(values),
                "window_std": sample_std(values),
                "curve_max": max(values),
                "lightning_epochs_seen": sorted(by_lightning),
                "metric_name": "val_heldin/r2_mean",
                "metric_caveat": (
                    "Progress-bar value is rounded (~3 significant figures) and is not "
                    "literally metrics_summary.R2_variance_weighted; treat as the only "
                    "CPU-available epoch curve for these runs."
                ),
            }
    return out


def cell_paired_deltas_from_window(
    curves: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, treat, control in CONTRASTS:
        per_cell: dict[str, float] = {}
        for cell in REQUIRED_CELLS:
            per_cell[cell] = float(curves[treat][cell]["window_mean"]) - float(
                curves[control][cell]["window_mean"]
            )
        out[name] = per_cell
    return out


def cell_paired_deltas_from_summary(aggregate: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    scores = aggregate["scores"]["m2"]
    out: dict[str, dict[str, float]] = {}
    for name, treat, control in CONTRASTS:
        per_cell: dict[str, float] = {}
        for cell in REQUIRED_CELLS:
            per_cell[cell] = float(scores[treat][cell]) - float(scores[control][cell])
        out[name] = per_cell
    return out


def fold1_seed_paired_stats(per_cell: Mapping[str, float]) -> dict[str, Any]:
    seed_deltas = [per_cell["fold1_seed42"], per_cell["fold1_seed43"]]
    return {
        "n_seeds": 2,
        "seed_deltas": seed_deltas,
        "mean_delta": mean(seed_deltas),
        "sigma_delta_paired": sigma_delta_paired(seed_deltas),
        "two_sigma_delta_paired": two_sigma_paired(seed_deltas),
        "seed_std": sample_std(seed_deltas),
        "scope": "fold1_only_seeds_42_43",
    }


def cross_fold_dispersion(per_cell: Mapping[str, float]) -> dict[str, Any]:
    """Descriptive only — mixes fold difficulty with seed noise."""
    values = [per_cell[c] for c in REQUIRED_CELLS]
    return {
        "cells": list(REQUIRED_CELLS),
        "values": values,
        "mean": mean(values),
        "sample_std": sample_std(values),
        "warning": "cross-fold std mixes fold difficulty with seed noise; do not freeze thresholds from this alone",
    }


def build_p0n_report(
    aggregate: Mapping[str, Any],
    *,
    logs_dir: Path | None = None,
) -> dict[str, Any]:
    statuses = audit_artifact_map(aggregate)
    run_dir_curves = all(s.has_epoch_curves for s in statuses) if statuses else False

    summary_paired = cell_paired_deltas_from_summary(aggregate)
    summary_fold1 = {name: fold1_seed_paired_stats(cells) for name, cells in summary_paired.items()}

    v4_curves = None
    v4_paired = None
    v4_fold1 = None
    v4_cross = None
    log_error = None
    if logs_dir is not None:
        try:
            v4_curves = load_v4_curves_from_logs(logs_dir)
            v4_paired = cell_paired_deltas_from_window(v4_curves)
            v4_fold1 = {name: fold1_seed_paired_stats(cells) for name, cells in v4_paired.items()}
            v4_cross = {name: cross_fold_dispersion(cells) for name, cells in v4_paired.items()}
        except Exception as exc:  # noqa: BLE001
            log_error = str(exc)

    epoch_curves_found = v4_curves is not None
    missing_cell_note = (
        f"{MISSING_CELL} is absent from native_mua_t4_v1 aggregate; "
        "full cross-fold σ is unreliable until that cell is filled."
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "v4_log_curves_parsed"
            if epoch_curves_found
            else "partial_no_v4_curves"
        ),
        "no_gpu": True,
        "thresholds_frozen": False,
        "missing_cell": MISSING_CELL,
        "required_cells": list(REQUIRED_CELLS),
        "expected_epoch_window": list(EXPECTED_EPOCH_WINDOW),
        "epoch_curves_found": epoch_curves_found,
        "run_dir_epoch_curves_complete": run_dir_curves,
        "run_curve_status": [asdict(s) for s in statuses],
        "summary_score_deltas": {
            "provenance": "best_checkpoint_argmax",
            "not_v4_comparable": True,
            "discipline_note": (
                "aggregate_m2.json scores match max(epoch curve) for inspected cells; "
                "handoff forbids placing best_checkpoint_validation_r2 beside V4 numbers"
            ),
            "cell_paired_deltas": summary_paired,
            "fold1_seed_paired_stats": summary_fold1,
        },
        "limitations": [
            missing_cell_note,
            "Do not write PRACTICAL_EFFECT_FLOOR / GPU thresholds from this report until fold2_seed43 exists and a reviewed receipt freezes them.",
        ],
        "recommended_next": [
            "If V4 log curves parsed: use fold1 two_sigma_delta_paired as a *lower bound* on measurable effects, still do not freeze until missing cell is addressed.",
            "Fill fold2_seed43 before any MUA GPU gate.",
        ],
    }
    if log_error is not None:
        report["log_parse_error"] = log_error
        report["fail_closed_epoch_claim"] = True
    if epoch_curves_found:
        report["v4_from_logs"] = {
            "logs_dir": str(logs_dir),
            "curves": v4_curves,
            "cell_paired_deltas": v4_paired,
            "fold1_seed_paired_stats": v4_fold1,
            "cross_fold_dispersion": v4_cross,
        }
        report["fail_closed_epoch_claim"] = False
    else:
        report["fail_closed_epoch_claim"] = True
        if logs_dir is None:
            report["limitations"].append(
                "No --logs-dir provided; only summary-score (argmax) deltas are available."
            )
    return report


def write_report(report: Mapping[str, Any], output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output dir: {output_dir}")
    output_dir.mkdir(parents=True)
    path = output_dir / "m2_sigma.json"
    tmp = output_dir / "m2_sigma.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def dry_run_payload() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "dry_run",
        "no_gpu": True,
        "thresholds_frozen": False,
        "message": (
            "Pass --execute with CARRIER_PERF_REVIEWED_CPU=YES, --aggregate, "
            "--logs-dir, and --output-dir to audit real artifacts (CPU only)."
        ),
    }
