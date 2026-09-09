#!/usr/bin/env python3
"""Render the receipt-bound primary figure from a completed chronological summary.

This is a presentation-only endpoint.  It accepts the compact, already
audited 18-row summary and never opens model, target, prediction, or training
artifacts.  Validation is deliberately fail-closed so a changed aggregation,
roster, or endpoint cannot silently become a paper figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SCHEMA = "chronological_last2_summary_v1"
ARMS = ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT")
DATASETS = ("m1", "m2", "h1")
DATES = {
    "m1": ("2012-09-27", "2012-09-28"),
    "m2": ("2020-11-18", "2020-11-19"),
    "h1": ("1925-01-19", "1925-01-20"),
}
FINAL_EPOCH = {"m1": 24, "m2": 24, "h1": 32}
SESSION_COUNT = {"m1": 1, "m2": 1, "h1": 2}
ARM_LABELS = ("Z\nnone", "B\nactivity", "D\njoint")
COLORS = {"Z_NONE": "#777777", "B_ACTIVITY_ONLY": "#3776ab", "D_JOINT": "#c84f47"}
SHA256 = re.compile(r"[0-9a-f]{64}")
ROW_KEYS = {"dataset", "date", "arm", "selected_r2", "fixed_r2", "selected_epoch", "session_count"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    # bool is a JSON number in Python but never an R2 value.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def load_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        fail(f"summary is absent: {path}")
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"summary is not valid JSON: {path}: {exc}") from exc
    if not isinstance(summary, dict):
        fail("summary root must be an object")
    required = {"schema", "status", "protocol_sha256", "scope", "rows"}
    if set(summary) != required:
        fail(f"summary root keys changed: expected {sorted(required)}, got {sorted(summary)}")
    if summary["schema"] != SCHEMA or summary["status"] != "PASSED":
        fail("summary is not a PASSED chronological_last2_summary_v1")
    if not isinstance(summary["scope"], str) or not summary["scope"].strip():
        fail("summary scope must be a nonempty string")
    if not isinstance(summary["protocol_sha256"], str) or not SHA256.fullmatch(summary["protocol_sha256"]):
        fail("summary protocol_sha256 must be a lowercase SHA-256")
    raw = summary["rows"]
    if not isinstance(raw, list) or len(raw) != 18:
        fail("summary must contain exactly 18 rows (3 datasets × 2 dates × 3 arms)")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    epochs: dict[tuple[str, str], int] = {}
    for index, row in enumerate(raw):
        label = f"rows[{index}]"
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            fail(f"{label} keys changed; expected {sorted(ROW_KEYS)}")
        dataset, date, arm = row["dataset"], row["date"], row["arm"]
        if dataset not in DATASETS or date not in DATES.get(dataset, ()) or arm not in ARMS:
            fail(f"{label} has an unrecognised dataset/date/arm")
        key = (dataset, date, arm)
        if key in seen:
            fail(f"duplicate summary row: {key}")
        seen.add(key)
        epoch, sessions = row["selected_epoch"], row["session_count"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= FINAL_EPOCH[dataset]:
            fail(f"{label}.selected_epoch is invalid")
        if isinstance(sessions, bool) or sessions != SESSION_COUNT[dataset]:
            fail(f"{label}.session_count violates the fixed date aggregation")
        epoch_key = (dataset, arm)
        if epoch_key in epochs and epochs[epoch_key] != epoch:
            fail(f"{dataset}/{arm} selected_epoch differs between the two target dates")
        epochs[epoch_key] = epoch
        rows.append({"dataset": dataset, "date": date, "arm": arm,
                     "selected_r2": finite(row["selected_r2"], f"{label}.selected_r2"),
                     "fixed_r2": finite(row["fixed_r2"], f"{label}.fixed_r2"),
                     "selected_epoch": epoch, "session_count": sessions})

    expected = {(dataset, date, arm) for dataset in DATASETS for date in DATES[dataset] for arm in ARMS}
    if seen != expected:
        fail("summary does not have exactly one row for every fixed dataset/date/arm cell")
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["dataset"], row["date"], row["arm"]): row for row in rows}
    per_date: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for date in DATES[dataset]:
            arm_rows = {arm: lookup[(dataset, date, arm)] for arm in ARMS}
            date_row: dict[str, Any] = {"dataset": dataset, "date": date,
                                        "session_count": SESSION_COUNT[dataset]}
            for arm in ARMS:
                date_row[f"{arm}_selected_r2"] = arm_rows[arm]["selected_r2"]
                date_row[f"{arm}_fixed_r2"] = arm_rows[arm]["fixed_r2"]
                date_row[f"{arm}_selected_epoch"] = arm_rows[arm]["selected_epoch"]
            per_date.append(date_row)
            for endpoint in ("selected_r2", "fixed_r2"):
                z, b, d = (arm_rows[arm][endpoint] for arm in ARMS)
                contributions.append({"dataset": dataset, "date": date, "endpoint": endpoint,
                                      "activity_B_minus_Z": b - z, "carrier_D_minus_B": d - b})
    return per_date, contributions


def draw(rows: list[dict[str, Any]], destination: Path) -> None:
    lookup = {(row["dataset"], row["date"], row["arm"]): row for row in rows}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.26, top=0.81, wspace=0.22)
    for axis, dataset in zip(axes, DATASETS, strict=True):
        pairs = [[lookup[(dataset, date, arm)]["selected_r2"] for arm in ARMS] for date in DATES[dataset]]
        means = [sum(pair[index] for pair in pairs) / len(pairs) for index in range(3)]
        bars = axis.bar(range(3), means, color=[COLORS[arm] for arm in ARMS], width=0.67,
                        edgecolor="black", linewidth=0.4, zorder=2)
        for index, pair in enumerate(pairs):
            axis.plot(range(3), pair, color="#222222", alpha=0.48, linewidth=0.8, marker="o",
                      markersize=3.0, zorder=4, label="paired target dates" if index == 0 else None)
        for index, bar in enumerate(bars):
            axis.annotate(f"{means[index]:.3f}", (bar.get_x() + bar.get_width() / 2, means[index]),
                          xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
        activity, carrier = means[1] - means[0], means[2] - means[1]
        axis.set_title(dataset.upper(), fontsize=9, fontweight="bold", pad=2)
        axis.text(0.5, 0.98, f"B−Z {activity:+.3f}   D−B {carrier:+.3f}", transform=axis.transAxes,
                  ha="center", va="top", fontsize=6.4)
        axis.set_xticks(range(3), ARM_LABELS, fontsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.axhline(0, color="#4f4f4f", linewidth=0.5, zorder=1)
        if dataset == "h1":
            axis.legend(loc="lower right", fontsize=5.6, frameon=False, handlelength=1.1)
    axes[0].set_ylabel("Target $R^2$", fontsize=8)
    fig.text(0.5, 0.035,
             "Bars: equal mean of the two target-date means, selected EMA. Lines/points: paired dates. "
             "B−Z: activity contribution; D−B: carrier contribution.",
             ha="center", fontsize=6.5)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="PASSED chronological_last2_summary_v1 JSON")
    parser.add_argument("--dest", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    summary_path, destination = args.summary.resolve(), args.dest.resolve()
    if destination.exists():
        fail(f"destination must be new: {destination}")
    summary, raw = load_summary(summary_path)
    per_date, contribution = aggregates(raw)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "chronological_last2_raw.csv", raw,
              ["dataset", "date", "arm", "selected_r2", "fixed_r2", "selected_epoch", "session_count"])
    write_csv(destination / "chronological_last2_per_date.csv", per_date, list(per_date[0]))
    write_csv(destination / "chronological_last2_contribution.csv", contribution,
              ["dataset", "date", "endpoint", "activity_B_minus_Z", "carrier_D_minus_B"])
    draw(raw, destination / "chronological_last2_calibration_ablation")
    outputs = sorted(path for path in destination.iterdir() if path.is_file())
    receipt = {
        "schema": "chronological_last2_figure_receipt_v1",
        "status": "PASSED",
        "input_summary": {"path": str(summary_path), "sha256": sha256(summary_path),
                          "schema": summary["schema"], "protocol_sha256": summary["protocol_sha256"],
                          "scope": summary["scope"]},
        "rows": len(raw),
        "outputs": {path.name: sha256(path) for path in outputs},
        "figure": "chronological_last2_calibration_ablation.pdf/png",
        "primary_endpoint": "selected_r2; bars are equal two-date means",
        "sensitivity_endpoint": "fixed_r2 retained in CSV only",
    }
    (destination / "chronological_last2_figure_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
