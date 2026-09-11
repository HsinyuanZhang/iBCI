#!/usr/bin/env python3
"""Plot two completed original-carrier M1 24-EMA score curves descriptively.

The command reads only the three JSON receipts from each explicitly supplied
run. It does not read predictions, data, NPZ, NWB, checkpoints, or
official-test material.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from summarize import (
    EPOCHS,
    HO,
    METRIC,
    ORIGINAL_VARIANT,
    _atomic_json,
    _atomic_text,
    _difference,
    _need,
    _read_run,
    _same_contract,
    _same_targets_and_starts,
    _sha,
)

HERE = Path(__file__).resolve().parent


def _two_seed_identity(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the original pack, fit, and B3S source are identical for this pair."""
    reference = runs[0]
    pack = reference["carrier_pack_npz_sha256"]
    fit = reference["fit_sha256"]
    b3s = reference["b3s"]
    _need(isinstance(pack, str) and len(pack) == 64, "original-s42: carrier pack SHA missing")
    _need(isinstance(fit, str) and len(fit) == 64, "original-s42: fit SHA missing")
    for run in runs[1:]:
        _need(run["carrier_pack_npz_sha256"] == pack, f"original carrier pack SHA mismatch: {run['id']}")
        _need(run["fit_sha256"] == fit, f"original fit SHA mismatch: {run['id']}")
        _need(run["b3s"] == b3s, f"original B3S initialization source mismatch: {run['id']}")
    return {
        "original_carrier_pack_sha256": pack,
        "original_fit_sha256": fit,
        "original_b3s_initialization_source": b3s,
        "all_two_original_runs_identical": True,
    }


def _write_csv(path: Path, runs: list[Mapping[str, Any]]) -> None:
    fields = (
        "run_id", "seed", "epoch", METRIC,
        *[f"{session}_channel_variance_weighted_r2" for session in HO],
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for epoch in EPOCHS:
                point = run["curve"][str(epoch)]
                writer.writerow({
                    "run_id": run["id"], "seed": run["seed"], "epoch": epoch,
                    METRIC: point[METRIC],
                    **{
                        f"{session}_channel_variance_weighted_r2": point["per_session"][session]["channel_variance_weighted_r2"]
                        for session in HO
                    },
                })
    temporary.replace(path)


def _write_plot(path_png: Path, path_svg: Path, runs: list[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("HO3 mean", METRIC, None), *((session, "channel_variance_weighted_r2", session) for session in HO))
    colors = {42: "#1f77b4", 43: "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), constrained_layout=True)
    fig.suptitle(
        "Only two completed original seeds: descriptive 24-epoch curves\n"
        "No candidate, no 3-seed aggregate, no confidence interval, no official-test metric",
        fontsize=15,
        fontweight="bold",
    )
    for axis, (title, key, session) in zip(axes.flat, panels):
        for run in runs:
            values = [
                run["curve"][str(epoch)][key] if session is None
                else run["curve"][str(epoch)]["per_session"][session][key]
                for epoch in EPOCHS
            ]
            label = f"Seed {run['seed']}"
            axis.plot(EPOCHS, values, color=colors[run["seed"]], linewidth=2.2, label=label)
            selected_epoch = run["selected"]["epoch"]
            selected_value = values[selected_epoch - 1]
            axis.scatter(selected_epoch, selected_value, color=colors[run["seed"]], marker="o", s=56, zorder=3)
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("Epoch", fontsize=11)
        axis.set_ylabel("Channel variance-weighted R²", fontsize=11)
        axis.set_xticks((1, 3, 6, 12, 18, 24))
        axis.tick_params(labelsize=10)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=10, frameon=False, loc="best")
    fig.savefig(path_png, dpi=150)
    fig.savefig(path_svg)
    plt.close(fig)


def _endpoint_table(runs: list[Mapping[str, Any]], endpoint: str) -> str:
    return "\n".join(
        f"| Seed {run['seed']} | {run[endpoint]['epoch']} | {run[endpoint][METRIC]:.12g} |"
        for run in runs
    )


def _readme(runs: list[Mapping[str, Any]]) -> str:
    return f"""# Original-carrier seed-pair curves

This is a descriptive plot of two completed original-carrier M1 runs: seeds 42 and 43. Each run contains complete scoring for all 24 EMA epochs on visible HO3 calibration. `n_seeds = 2`; this output contains no three-seed aggregation, candidate comparison, confidence interval, or official-test result.

The command reads only `run_meta.json`, `train_receipt.json`, and `score_receipt.json` from each supplied run. It records receipt and code SHA-256 values, validates the shared source/sampler and HO contracts, validates all HO targets, starts, and window counts across both runs and all 24 epochs, and checks identical original carrier-pack, fit, and B3S provenance.

Selected epoch means the earliest maximum equal-session channel variance-weighted R² on visible HO3 calibration. `curves.csv` contains the complete 24-epoch curves; selected markers in the plots use that rule.

## Fixed epoch 3

| Run | Epoch | HO3 mean R² |
| --- | ---: | ---: |
{_endpoint_table(runs, 'fixed_epoch_3')}

## Independently selected epoch

| Run | Epoch | HO3 mean R² |
| --- | ---: | ---: |
{_endpoint_table(runs, 'selected')}
"""


def plot_pair(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    _need(not output.exists(), f"--output-dir must be new: {output}")
    original42 = _read_run(args.original_s42, run_id="original-s42", seed=42, variant=ORIGINAL_VARIANT, old_schema=True)
    original43 = _read_run(args.original_s43, run_id="original-s43", seed=43, variant=ORIGINAL_VARIANT, old_schema=False)
    runs = [original42, original43]
    source_contract = _same_contract(runs, "source_noncarrier_contract", "source/sampler noncarrier contract")
    ho_contract = _same_contract(runs, "ho_noncarrier_contract", "HO noncarrier contract")
    targets = _same_targets_and_starts(runs)
    identity = _two_seed_identity(runs)
    differences = {
        str(epoch): _difference(original43["curve"][str(epoch)], original42["curve"][str(epoch)])
        for epoch in EPOCHS
    }
    body = {
        "schema": "m1_muscle_r100_original_seed_pair_plot_v1",
        "status": "COMPLETED_TWO_SEED_DESCRIPTIVE_ONLY",
        "scope": "Exactly two completed original-carrier seeds; no candidate, no 3-seed aggregate, no confidence interval, and no official-test metric.",
        "input_policy": {
            "runs": [run["id"] for run in runs],
            "receipt_files_per_run": ["run_meta.json", "train_receipt.json", "score_receipt.json"],
            "read_scope": "JSON receipts only; no prediction artifacts, data, NPZ, NWB, checkpoints, GPU, or official-test material",
        },
        "input_receipts_sha256": {run["id"]: run["input_receipts_sha256"] for run in runs},
        "code_sha256": {
            "plot_original_seed_pair.py": _sha(Path(__file__).resolve()),
            "summarize.py": _sha(HERE / "summarize.py"),
        },
        "invariants": {
            "formal_epochs": 24,
            "source_and_sampler_noncarrier_contract": source_contract,
            "ho_noncarrier_contract": ho_contract,
            "ho_targets_starts_window_counts": targets,
            "two_original_carrier_fit_and_b3s_identity": identity,
        },
        "runs": {
            run["id"]: {key: value for key, value in run.items() if key not in {"source_noncarrier_contract", "ho_noncarrier_contract"}}
            for run in runs
        },
        "epoch_by_epoch_original_s43_minus_original_s42": differences,
        "selection_interpretation": "Selected epochs are independently selected per run as the earliest maximum equal-session mean channel variance-weighted R2 on visible HO3 calibration.",
    }
    output.mkdir(parents=True)
    _atomic_json(output / "pair_summary.json", body)
    _write_csv(output / "curves.csv", runs)
    _write_plot(output / "curves.png", output / "curves.svg", runs)
    _atomic_text(output / "README.md", _readme(runs))
    return {
        "status": body["status"],
        "summary": str(output / "pair_summary.json"),
        "csv": str(output / "curves.csv"),
        "plot_png": str(output / "curves.png"),
        "plot_svg": str(output / "curves.svg"),
        "readme": str(output / "README.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-s42", type=Path, required=True)
    parser.add_argument("--original-s43", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(plot_pair(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
