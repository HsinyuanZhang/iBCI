#!/usr/bin/env python3
"""Fail closed when plotting the completed seed-42 M2 mechanism-control summary."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
HELPER_NAME = "summarize_m2_mechanism_controls.py"
ARMS = ("B", "C", "D", "shuffle", "mean", "nonattn")
CONTROLS = ("B", "C", "shuffle", "mean", "nonattn")
SESSIONS = ("ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1", "ses-2020-11-19-Run1")
SUMMARY_SCHEMA = "m2_mechanism_controls_summary_v1"
SURFACE = "fixed ext4 M33, four sessions / 2069 windows, visible development only"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"non-numeric {label}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}")
    return result


def helper_for(root: Path) -> tuple[Path, Any]:
    path = root / "scripts/diagnostics_v1" / HELPER_NAME
    spec = importlib.util.spec_from_file_location("frozen_m2_mechanism_controls", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


def require_completed(summary: Mapping[str, Any]) -> None:
    if not (
        summary.get("schema") == SUMMARY_SCHEMA
        and summary.get("status") == "COMPLETED"
        and summary.get("seed") == 42
        and summary.get("surface") == SURFACE
        and summary.get("official_test_used") is False
        and summary.get("no_ci_pvalue_or_multiseed_architecture_effect") is True
        and summary.get("calibration_information") == {
            "B": "activity-only calibration information",
            "C": "carrier-only calibration information; query activity remains used",
            "D": "joint activity-plus-carrier calibration information",
        }
    ):
        raise RuntimeError("requires completed frozen seed42 ext4 M33 mechanism-control summary")
    arms = summary.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise RuntimeError("summary must contain exactly B/C/D/shuffle/mean/nonattn")
    contrasts = summary.get("information_contrasts")
    effects = summary.get("effects")
    if not isinstance(contrasts, dict) or set(contrasts) != {
        "D_minus_B_independently_picked", "D_minus_C_independently_picked",
        "D_minus_B_fixed_epoch_24", "D_minus_C_fixed_epoch_24",
    }:
        raise RuntimeError("summary information contrasts incomplete")
    if not isinstance(effects, dict) or set(effects) != {"shuffle", "mean", "nonattn"}:
        raise RuntimeError("summary mechanism effects incomplete")


def verify_actual_summary(root: Path, summary: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    helper_path, helper = helper_for(root)
    actual = helper.summary(root)
    require_completed(actual)
    if actual != summary:
        raise RuntimeError("input summary differs from the frozen helper's freshly revalidated receipt-bound result")
    return helper_path, actual


def extract(summary: Mapping[str, Any]) -> dict[str, Any]:
    arms = summary["arms"]
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        item = arms[arm]
        selected_epoch = item.get("selected_epoch")
        if not isinstance(selected_epoch, int) or selected_epoch not in range(1, 25):
            raise RuntimeError(f"invalid selected epoch for {arm}")
        selected = item.get("selected")
        endpoint = item.get("endpoint24")
        if not isinstance(selected, dict) or not isinstance(endpoint, dict):
            raise RuntimeError(f"selected/endpoint rows absent for {arm}")
        rows.append({
            "arm": arm,
            "selected_epoch": selected_epoch,
            "selected_equal_session_mean": finite(selected.get("equal_session_mean"), f"{arm} selected"),
            "epoch_24_equal_session_mean": finite(endpoint.get("equal_session_mean"), f"{arm} epoch24"),
            "selected_checkpoint_sha256": selected.get("checkpoint_sha256"),
            "epoch_24_checkpoint_sha256": endpoint.get("checkpoint_sha256"),
        })
    deltas: list[dict[str, Any]] = []
    for control in CONTROLS:
        if control == "B":
            selected = summary["information_contrasts"]["D_minus_B_independently_picked"]
            endpoint = summary["information_contrasts"]["D_minus_B_fixed_epoch_24"]
        elif control == "C":
            selected = summary["information_contrasts"]["D_minus_C_independently_picked"]
            endpoint = summary["information_contrasts"]["D_minus_C_fixed_epoch_24"]
        else:
            selected = summary["effects"][control]["independently_picked"]
            endpoint = summary["effects"][control]["fixed_epoch_24"]
        if not isinstance(selected, dict) or not isinstance(endpoint, dict):
            raise RuntimeError(f"D-minus-{control} contrast absent")
        per = selected.get("per_session_D_minus_control")
        if not isinstance(per, dict) or set(per) != set(SESSIONS):
            raise RuntimeError(f"D-minus-{control} selected session surface incomplete")
        deltas.append({
            "control": control,
            "selected_D_minus_control": finite(selected.get("D_minus_control_equal_session_mean"), f"D-{control} selected"),
            "epoch_24_D_minus_control": finite(endpoint.get("D_minus_control_equal_session_mean"), f"D-{control} epoch24"),
            "selected_per_session_D_minus_control": {session: finite(per[session], f"D-{control}/{session}") for session in SESSIONS},
        })
    receipt_bindings = {
        arm: {
            "run": arms[arm].get("run"),
            "receipt_sha256": arms[arm].get("receipt_sha256"),
            "checkpoint_sha256_by_epoch": arms[arm].get("checkpoint_sha256_by_epoch"),
            "source_hashes": arms[arm].get("source_hashes"),
            "cache_hashes": arms[arm].get("cache_hashes"),
        }
        for arm in ARMS
    }
    return {
        "arms": rows,
        "D_minus_controls": deltas,
        "sessions": list(SESSIONS),
        "receipt_bindings": receipt_bindings,
        "frozen_manifest_digest": summary.get("frozen_manifest_digest"),
        "surface_note": "Seed42 visible ext4 M33 development only; sessions are repeated measurements within one seed.",
        "calibration_information": summary["calibration_information"],
        "no_ci_sd_or_significance": True,
    }


def plot(data: Mapping[str, Any], png: Path, pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = data["arms"]
    deltas = data["D_minus_controls"]
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 5.4), constrained_layout=True, gridspec_kw={"width_ratios": (1.2, 1.0, 1.2)})

    ax = axes[0]
    positions = np.arange(len(arms)); width = 0.36
    selected = [row["selected_equal_session_mean"] for row in arms]
    endpoint = [row["epoch_24_equal_session_mean"] for row in arms]
    ax.bar(positions - width / 2, selected, width, label="independently selected EMA", color="#4c78a8")
    ax.bar(positions + width / 2, endpoint, width, label="fixed EMA epoch 24", color="#bab0ab")
    for index, row in enumerate(arms):
        ax.text(index - width / 2, selected[index], f"e{row['selected_epoch']}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, [row["arm"] for row in arms])
    ax.set_ylabel("equal-session R²")
    ax.set_title("Six arms on ext4")
    ax.set_ylim(0.0, max(max(selected), max(endpoint)) * 1.30)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.20))

    ax = axes[1]
    positions = np.arange(len(deltas))
    selected = [row["selected_D_minus_control"] for row in deltas]
    endpoint = [row["epoch_24_D_minus_control"] for row in deltas]
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.scatter(positions - 0.13, selected, marker="o", color="#e45756", label="independently selected")
    ax.scatter(positions + 0.13, endpoint, marker="s", color="#4c78a8", label="fixed epoch 24")
    for index, value in enumerate(selected):
        ax.plot((index - 0.13, index + 0.13), (value, endpoint[index]), color="#9d9d9d", linewidth=0.8, zorder=0)
    ax.set_xticks(positions, [f"D − {row['control']}" for row in deltas], rotation=25, ha="right")
    ax.set_ylabel("equal-session R² difference")
    ax.set_title("Single-seed D-minus-control contrasts")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.02, 0.02, "seed42 descriptive values; no CI, SD, or significance", transform=ax.transAxes, fontsize=7, va="bottom")

    ax = axes[2]
    heat = np.asarray([[row["selected_per_session_D_minus_control"][session] for row in deltas] for session in SESSIONS], dtype=float)
    limit = max(float(np.abs(heat).max()), 1e-12)
    image = ax.imshow(heat, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(deltas)), [f"D − {row['control']}" for row in deltas], rotation=25, ha="right")
    ax.set_yticks(range(len(SESSIONS)), [session.replace("ses-2020-", "") for session in SESSIONS])
    ax.set_title("Selected per-session contrasts")
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            value = heat[row_index, column_index]
            color = "white" if abs(value) > limit * 0.55 else "black"
            ax.text(column_index, row_index, f"{value:+.3f}", ha="center", va="center", color=color, fontsize=7)
    colorbar = figure.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    colorbar.set_label("D − control R²")

    figure.text(0.5, -0.02, "Seed42 visible ext4 M33 development surface. Sessions are repeated measurements within one seed. C has carrier-only calibration information while query activity remains used.", ha="center", fontsize=8)
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(); summary_path = args.summary.resolve(); dest = args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"destination must be a new directory: {dest}")
    summary = load_json(summary_path)
    require_completed(summary)
    helper_path, actual = verify_actual_summary(root, summary)
    data = extract(actual)
    dest.mkdir(parents=True)
    png = dest / "m2_mechanism_controls.png"
    pdf = dest / "m2_mechanism_controls.pdf"
    figure_data = dest / "figure_data.json"
    payload = {
        "schema": "m2_mechanism_controls_figure_data_v1",
        "status": "COMPLETED",
        "input_summary": str(summary_path),
        "input_summary_sha256": sha(summary_path),
        "frozen_summary_helper": str(helper_path),
        "frozen_summary_helper_sha256": sha(helper_path),
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha(Path(__file__).resolve()),
        "actual_binding_rechecked": True,
        "summary_exactly_matches_fresh_helper_output": True,
        "surface": SURFACE,
        "seed": 42,
        "official_test_used": False,
        "no_ci_sd_or_significance": True,
        **data,
    }
    figure_data.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(payload, png, pdf)
    payload["artifacts"] = {
        "png": {"path": str(png), "sha256": sha(png), "dpi": 300},
        "pdf": {"path": str(pdf), "sha256": sha(pdf)},
    }
    figure_data.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "dest": str(dest), "figure_data_sha256": sha(figure_data)}, sort_keys=True))


if __name__ == "__main__":
    main()
