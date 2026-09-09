#!/usr/bin/env python3
"""Render task-specific carrier-v4 figures from summarize.py summary JSON only.

The script uses supplied cell summaries and supplied paired bootstrap intervals.
It never reads runs, NPZ files, models, or source data, and it never refits or
resamples statistics.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Mapping


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    need(isinstance(value, dict), f"summary must be a JSON object: {path}")
    need(value.get("schema") == "carrier_v4_manifest_summary_v1", "unsupported summary schema")
    need(isinstance(value.get("cells"), list) and isinstance(value.get("comparisons"), list), "summary cells/comparisons missing")
    need(isinstance(value.get("noninferiority_margin"), (int, float)), "noninferiority_margin missing")
    return value


def arm_label(value: Any) -> str:
    labels = {"full": "FULL", "activity_only": "ACTIVITY", "carrier_only": "CARRIER", "none": "NONE"}
    return labels.get(str(value), str(value).replace("_", " ").upper())


def short_cell_label(cell: Mapping[str, Any]) -> str:
    if "signed" in str(cell.get("id", "")).lower() or "signed" in str(cell.get("carrier_family", "")).lower():
        return "signed-state baseline"
    arm = arm_label(cell.get("information_arm"))
    fusion, projection = cell.get("fusion"), cell.get("proj_dim")
    if fusion == "concat":
        return f"{arm} concat"
    if fusion == "proj_add" and projection in (16, 32):
        return f"{arm} P{projection}"
    return f"{arm} {fusion}"


def cells_for_task(summary: Mapping[str, Any], task: str) -> dict[str, Mapping[str, Any]]:
    return {str(cell["id"]): cell for cell in summary["cells"] if cell.get("task") == task and isinstance(cell.get("id"), str)}


def complete_cells(cells: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for cell in cells.values():
        fixed, selected = cell.get("fixed_epoch_row"), cell.get("selected_earliest_mean_max")
        if cell.get("complete") is True and isinstance(fixed, Mapping) and isinstance(selected, Mapping):
            if isinstance(fixed.get("mean"), (int, float)) and isinstance(selected.get("mean"), (int, float)):
                rows.append(cell)
    return rows


def completed_endpoints(summary: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for comparison in summary["comparisons"]:
        if comparison.get("status") == "pending" or str(comparison.get("cell")) not in cells:
            continue
        bootstrap = comparison.get("bootstrap")
        interval = bootstrap.get("two_sided_95") if isinstance(bootstrap, Mapping) else None
        endpoint = comparison.get("endpoint_label")
        if endpoint not in ("fixed_e3", "fixed_e16", "selected_each"):
            continue
        if not (isinstance(bootstrap, Mapping) and isinstance(interval, list) and len(interval) == 2):
            continue
        required = (bootstrap.get("mean_delta"), bootstrap.get("one_sided_lower_95"), interval[0], interval[1],
                    comparison.get("candidate_epoch"), comparison.get("reference_epoch"))
        if all(isinstance(value, (int, float)) for value in required):
            rows.append(comparison)
    return rows


def pending_text(summary: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]) -> str:
    pending_cells = [short_cell_label(cell) for cell in cells.values() if cell.get("complete") is not True]
    pending_pairs = []
    for comparison in summary["comparisons"]:
        if comparison.get("status") != "pending" or str(comparison.get("cell")) not in cells:
            continue
        candidate = cells.get(str(comparison.get("cell")), {})
        reference = cells.get(str(comparison.get("reference")), {})
        pending_pairs.append(f"{short_cell_label(candidate)} vs {short_cell_label(reference)}")
    fragments = []
    if pending_cells:
        fragments.append("Pending cells (not plotted): " + ", ".join(pending_cells))
    if pending_pairs:
        fragments.append("Pending comparisons (not plotted): " + ", ".join(pending_pairs))
    return "\n\n".join(fragments) if fragments else "No pending cells or comparisons."


def draw_scores(axis: Any, cells: list[Mapping[str, Any]], task: str) -> None:
    import numpy as np
    axis.set_title("Cell endpoints")
    if not cells:
        axis.text(.5, .5, "No completed cells", ha="center", va="center", transform=axis.transAxes)
        axis.set_xticks([])
        return
    x = np.arange(len(cells), dtype=float)
    fixed = [float(cell["fixed_epoch_row"]["mean"]) for cell in cells]
    selected = [float(cell["selected_earliest_mean_max"]["mean"]) for cell in cells]
    axis.scatter(x - .12, fixed, color="#1f77b4", marker="o", s=52, label="Fixed endpoint", zorder=3)
    axis.scatter(x + .12, selected, color="#d62728", marker="D", s=46, label="Selected mean", zorder=3)
    for index, cell in enumerate(cells):
        fixed_epoch = cell["fixed_epoch_row"].get("epoch", cell.get("fixed_epoch"))
        selected_epoch = cell["selected_earliest_mean_max"].get("epoch")
        axis.annotate(f"e{fixed_epoch}", (x[index] - .12, fixed[index]), xytext=(0, 7), textcoords="offset points",
                      ha="center", fontsize=8, color="#1f77b4")
        axis.annotate(f"e{selected_epoch}", (x[index] + .12, selected[index]), xytext=(0, -13), textcoords="offset points",
                      ha="center", fontsize=8, color="#d62728")
    axis.set_xticks(x, [short_cell_label(cell) for cell in cells], rotation=20, ha="right")
    axis.set_xlim(-.5, len(cells) - .5)
    axis.set_ylabel("Mean R2 (M1 equal-session)" if task == "m1" else "Mean R2 (H1 grouped)")
    axis.grid(axis="y", alpha=.28)
    axis.legend(frameon=False, loc="best")


def endpoint_label(comparison: Mapping[str, Any], cells: Mapping[str, Mapping[str, Any]]) -> str:
    candidate = short_cell_label(cells[str(comparison["cell"])])
    reference = short_cell_label(cells[str(comparison["reference"])])
    endpoint = "fixed" if str(comparison["endpoint_label"]).startswith("fixed_") else "selected each"
    return f"{candidate} vs {reference}\n{endpoint}: candidate e{comparison['candidate_epoch']}, reference e{comparison['reference_epoch']}"


def draw_forest(axis: Any, comparisons: list[Mapping[str, Any]], cells: Mapping[str, Mapping[str, Any]], margin: float) -> None:
    import numpy as np
    axis.set_title("Paired candidate - reference delta")
    axis.axvline(0, color="#555555", linewidth=1, zorder=1)
    axis.axvline(-margin, color="#9467bd", linewidth=1.4, linestyle="--", zorder=1)
    if not comparisons:
        axis.text(.5, .5, "No completed paired endpoints", ha="center", va="center", transform=axis.transAxes)
        axis.set_yticks([])
        axis.set_xlabel("Candidate - reference R2")
        return
    y = np.arange(len(comparisons))[::-1]
    labels = []
    for index, comparison in enumerate(comparisons):
        bootstrap = comparison["bootstrap"]
        mean = float(bootstrap["mean_delta"])
        low, high = (float(value) for value in bootstrap["two_sided_95"])
        noninferior = float(bootstrap["one_sided_lower_95"]) >= -margin
        color = "#1f77b4" if noninferior else "#d62728"
        axis.errorbar(mean, y[index], xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=3, zorder=3)
        labels.append(endpoint_label(comparison, cells))
    axis.set_yticks(y, labels, fontsize=8)
    axis.set_ylim(-.7, len(comparisons) - .3)
    axis.set_xlabel("Candidate - reference R2")
    axis.grid(axis="x", alpha=.28)


def render_task(summary: Mapping[str, Any], output_dir: Path, task: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = cells_for_task(summary, task)
    completed = complete_cells(cells)
    comparisons = completed_endpoints(summary, cells)
    height = max(9.2, 6.6 + .26 * max(len(cells), len(comparisons)))
    figure = plt.figure(figsize=(13.5, height), layout="constrained")
    grid = figure.add_gridspec(3, 1, height_ratios=(1.0, max(1.15, .16 * max(len(comparisons), 4)), .34))
    scores_axis = figure.add_subplot(grid[0]); forest_axis = figure.add_subplot(grid[1]); status_axis = figure.add_subplot(grid[2])
    draw_scores(scores_axis, completed, task)
    draw_forest(forest_axis, comparisons, cells, float(summary["noninferiority_margin"]))
    status_axis.axis("off")
    criterion = (f"Forest plot: bars are supplied two-sided 95% intervals; dashed line is -{float(summary['noninferiority_margin']):.2f}. "
                 "Blue: one-sided lower 95% >= -margin; Red: criterion not met. "
                 "This is the current public single-seed criterion.\n\n")
    status_axis.text(.01, .95, textwrap.fill(criterion + pending_text(summary, cells), width=135), va="top", ha="left", fontsize=9,
                     transform=status_axis.transAxes)
    figure.suptitle(f"Carrier v4 {task.upper()} public calibration summary (seed 42)\n"
                     + ("M1 paired sessions n=3" if task == "m1" else "H1 grouped sessions n=7"), fontsize=14)
    stem = f"{task}_results"
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    for path in outputs:
        figure.savefig(path, dpi=220 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(figure)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    need(not output_dir.exists(), "--output-dir must be fresh")
    summary = read_summary(args.summary_json.resolve())
    output_dir.mkdir(parents=True)
    outputs = [*render_task(summary, output_dir, "m1"), *render_task(summary, output_dir, "h1")]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
