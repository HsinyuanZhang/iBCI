#!/usr/bin/env python3
"""Fail-closed paper figures from a completed M2 paired-three-seed ext4 summary."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
ARMS = {"B": "B_ACTIVITY_ONLY", "D": "D_JOINT"}
SESSIONS = ("ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1", "ses-2020-11-19-Run1")
SUMMARY_SCHEMA = "m2_joint_paired_seed_effects_v1"
HELPER = ROOT / "scripts/diagnostics_v1/summarize_m2_paired_seed_effects.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label}: non-finite")
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: JSON object required")
    return value


def load_helper():
    spec = importlib.util.spec_from_file_location("frozen_m2_pair_summary", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen paired-seed summary helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_completed_summary(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (summary.get("schema") != SUMMARY_SCHEMA or summary.get("status") != "COMPLETED" or
            summary.get("surface") != "ext4 mechanism control only; never ext6" or
            summary.get("arms") != ARMS or summary.get("seeds") != list(SEEDS) or
            summary.get("evalai_opened") is not False or summary.get("official_test_used") is not False or
            summary.get("no_significance_test") is not True or
            summary.get("independence_unit") != "seed-paired B versus D; sessions are within-seed repeated measurements"):
        raise RuntimeError("requires the completed, visible-ext4-only paired three-seed summary contract")
    pairs = summary.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 3 or {row.get("seed") for row in pairs} != set(SEEDS):
        raise RuntimeError("requires exactly three completed B/D pairs for seeds 42/43/44")
    return sorted(pairs, key=lambda row: int(row["seed"]))


def verify_actual_bindings(pairs: list[dict[str, Any]]) -> None:
    """Re-run the frozen helper's formal/hash/checkpoint validation, then compare.

    This does not score data.  Its load_completed contract validates the actual
    meta/train/score files, all 24 checkpoint hashes, source/cache hashes, and
    the selected EMA epoch before this plotting script accepts the summary.
    """
    helper = load_helper()
    for pair in pairs:
        seed = int(pair["seed"])
        for arm in ("B", "D"):
            actual, errors = helper.load_completed(ROOT, arm, seed)
            if errors or actual is None:
                raise RuntimeError(f"{arm}{seed}: frozen helper actual receipt validation failed: {errors}")
            reported = pair.get(arm)
            if reported != actual:
                raise RuntimeError(f"{arm}{seed}: summary no longer matches actual formal meta/train/score/checkpoint bindings")
            selected = reported.get("selected", {})
            if not isinstance(selected.get("checkpoint_sha256"), str) or not selected.get("per_session"):
                raise RuntimeError(f"{arm}{seed}: selected checkpoint/per-session binding missing")


def extract(pairs: list[dict[str, Any]], summary: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    heatmap: list[list[float]] = []
    selected_deltas: list[float] = []
    fixed_deltas: list[float] = []
    for pair in pairs:
        seed = int(pair["seed"]); b = pair["B"]; d = pair["D"]
        b_score = finite(b["selected"]["equal_session_mean"], f"B{seed} selected")
        d_score = finite(d["selected"]["equal_session_mean"], f"D{seed} selected")
        selected_delta = finite(pair["independent_epoch_pick_delta_D_minus_B"], f"D-B{seed} selected")
        fixed_delta = finite(pair["fixed_epoch_24_delta_D_minus_B"], f"D-B{seed} fixed24")
        if abs(selected_delta - (d_score - b_score)) > 1e-12:
            raise RuntimeError(f"seed {seed}: selected delta does not bind B/D scores")
        fixed_expected = finite(d["epoch_24"]["equal_session_mean"], f"D{seed} fixed24") - finite(b["epoch_24"]["equal_session_mean"], f"B{seed} fixed24")
        if abs(fixed_delta - fixed_expected) > 1e-12:
            raise RuntimeError(f"seed {seed}: fixed epoch-24 delta does not bind B/D scores")
        per_session = pair.get("independent_epoch_pick_per_session_delta_D_minus_B", {})
        if set(per_session) != set(SESSIONS):
            raise RuntimeError(f"seed {seed}: selected per-session delta contract drift")
        verified_session = []
        for session in SESSIONS:
            reported = finite(per_session[session], f"{seed}/{session}")
            expected = finite(d["selected"]["per_session"][session]["r2"], f"D{seed}/{session}") - finite(b["selected"]["per_session"][session]["r2"], f"B{seed}/{session}")
            if abs(reported - expected) > 1e-12:
                raise RuntimeError(f"seed {seed}/{session}: heatmap delta does not bind verified B/D selected scores")
            verified_session.append(expected)
        heatmap.append(verified_session)
        rows.append({"seed": seed, "B_selected_equal_session_mean": b_score, "D_selected_equal_session_mean": d_score,
                     "B_selected_epoch": int(b["selected_epoch"]), "D_selected_epoch": int(d["selected_epoch"]),
                     "D_minus_B_selected": selected_delta, "D_minus_B_fixed_epoch_24": fixed_delta,
                     "B_selected_checkpoint_sha256": b["selected"]["checkpoint_sha256"],
                     "D_selected_checkpoint_sha256": d["selected"]["checkpoint_sha256"]})
        selected_deltas.append(selected_delta); fixed_deltas.append(fixed_delta)
    summary_rows = summary.get("three_seed_summary", {})
    for key, values in (("independent_epoch_pick_delta_D_minus_B", selected_deltas), ("fixed_epoch_24_delta_D_minus_B", fixed_deltas)):
        stated = summary_rows.get(key, {})
        if (int(stated.get("n_independent_seed_pairs", -1)) != 3 or
                abs(finite(stated.get("mean"), key + " mean") - float(np.mean(values))) > 1e-12 or
                abs(finite(stated.get("sample_sd"), key + " sample_sd") - float(np.std(values, ddof=1))) > 1e-12):
            raise RuntimeError(f"three-seed descriptive summary drift: {key}")
    return {"seed_rows": rows, "sessions": list(SESSIONS), "selected_D_minus_B_by_seed_and_session": heatmap,
            "three_seed_descriptive": summary_rows,
            "interpretation": "Descriptive paired seed effects only; no CI, p-value, or session-level pseudo-replication.",
            "independence_unit": "Three model seeds are paired independent units; four sessions are repeated measurements within each seed."}


def plot(data: Mapping[str, Any], png: Path, pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["seed_rows"]; seeds = np.asarray([row["seed"] for row in rows]); selected = np.asarray([row["D_minus_B_selected"] for row in rows]); fixed = np.asarray([row["D_minus_B_fixed_epoch_24"] for row in rows]); heat = np.asarray(data["selected_D_minus_B_by_seed_and_session"], dtype=float).T
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.05, 1.2]})
    ax = axes[0]; x = np.arange(3); width = .34
    b = [row["B_selected_equal_session_mean"] for row in rows]; d = [row["D_selected_equal_session_mean"] for row in rows]
    ax.bar(x-width/2, b, width, label="B", color="#4c78a8"); ax.bar(x+width/2, d, width, label="D", color="#e45756")
    for i,row in enumerate(rows):
        ax.text(i-width/2, b[i], f"e{row['B_selected_epoch']}", ha="center", va="bottom", fontsize=8)
        ax.text(i+width/2, d[i], f"e{row['D_selected_epoch']}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"seed {seed}" for seed in seeds]); ax.set_ylabel("ext4 equal-session R²"); ax.set_title("Independent EMA selection"); ax.legend(frameon=False)
    ax = axes[1]; ax.axhline(0, color="black", linewidth=.8); ax.plot(seeds, selected, "o-", color="#e45756", label="selected epochs"); ax.plot(seeds, fixed, "s--", color="#4c78a8", label="fixed epoch 24")
    for label, values, color in (("selected", selected, "#e45756"), ("fixed e24", fixed, "#4c78a8")):
        ax.errorbar([44.45 if label == "selected" else 44.7], [np.mean(values)], yerr=[np.std(values, ddof=1)], fmt="o", color=color, capsize=3)
    ax.set_xlim(41.6, 45.15); ax.set_xticks(seeds); ax.set_xlabel("model seed"); ax.set_ylabel("D − B equal-session R²"); ax.set_title("Paired seed deltas")
    ax.legend(frameon=False, fontsize=8); ax.text(.02, .02, "dots at right: mean ± sample SD\n(n = 3 paired model seeds; descriptive)", transform=ax.transAxes, fontsize=7, va="bottom")
    ax = axes[2]; lim = max(abs(float(heat.min())), abs(float(heat.max())), 1e-12)
    image = ax.imshow(heat, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(3), [str(seed) for seed in seeds]); ax.set_xlabel("model seed"); ax.set_yticks(range(4), [s.replace("ses-2020-", "") for s in SESSIONS]); ax.set_title("Selected D − B by ext4 session")
    cb = fig.colorbar(image, ax=ax, fraction=.05, pad=.04); cb.set_label("D − B R²")
    fig.text(.5, -.02, "Visible ext4 mechanism-control surface only; model seeds vary while sampler_seed=42 is fixed. Sessions are within-seed repeated measurements, not 12 independent samples.", ha="center", fontsize=8)
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="plot completed M2 paired B/D seed effects")
    parser.add_argument("--summary", type=Path, required=True, help="COMPLETED JSON from summarize_m2_paired_seed_effects.py")
    parser.add_argument("--dest", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args(); summary_path = args.summary.resolve(); dest = args.dest.resolve()
    if dest.exists():
        raise FileExistsError(f"fresh destination required: {dest}")
    summary = load_json(summary_path); pairs = require_completed_summary(summary); verify_actual_bindings(pairs); data = extract(pairs, summary)
    dest.mkdir(parents=True)
    png, pdf, data_path = dest / "m2_three_seed_pairs.png", dest / "m2_three_seed_pairs.pdf", dest / "figure_data.json"
    payload = {"schema": "m2_three_seed_pairs_figure_data_v1", "status": "COMPLETED", "surface": "visible ext4 mechanism control only; never ext6 or official test", "model_seeds": list(SEEDS), "sampler_seed": 42, "input_summary": str(summary_path), "input_summary_sha256": sha(summary_path), "plot_script_sha256": sha(Path(__file__)), "frozen_summary_helper": str(HELPER), "frozen_summary_helper_sha256": sha(HELPER), "actual_binding_rechecked": True, "no_ci_p_values_or_significance_claims": True, **data}
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot(payload, png, pdf)
    payload["artifacts"] = {"png": {"path": str(png), "sha256": sha(png), "dpi": 300}, "pdf": {"path": str(pdf), "sha256": sha(pdf)}}
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "dest": str(dest), "figure_data_sha256": sha(data_path)}, sort_keys=True))

if __name__ == "__main__":
    main()
