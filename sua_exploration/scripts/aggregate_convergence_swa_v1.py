"""E2 aggregator: combine the 3 per-seed convergence curves
(``eval_convergence_curve_dandi688.py`` outputs, ``curve_s{seed}.json``) into
``results/convergence_swa_v1/convergence.json``.

Computes, from the already-evaluated per-epoch protocol scores (never re-evaluates a
checkpoint -- this script only reads JSON):
  - the cross-seed mean/std curve at every epoch;
  - per-seed OLS slopes in successive 8-epoch windows (5-12, 13-20, 21-28, 29-36, plus the
    trailing partial window 37-40), each with the window's own within-window std and
    cumulative change (slope * n_epochs), mirroring the diagnostic style already used in
    sua_exploration/docs/CURRENT_RESULTS.md section I / the M3 window;
  - a dedicated final-8-epoch-window (33-40) slope, directly answering "was the model still
    improving when 40-epoch training stopped";
  - the overall 5-40 OLS slope per seed.

No GPU, no NWB data, no torch: pure JSON + arithmetic, same spirit as
aggregate_side_feature_ablation_v2.py / aggregate_attention_architecture_screen_v4.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence

SEEDS: tuple[int, ...] = (42, 43, 44)

# Successive 8-epoch windows over the 5..40 protocol-epoch range (36 epochs); the range does
# not divide evenly by 8 (36 = 4*8 + 4), so the last window is a labeled 4-epoch partial.
SUCCESSIVE_WINDOWS: tuple[tuple[int, int], ...] = ((5, 12), (13, 20), (21, 28), (29, 36), (37, 40))
FINAL_WINDOW: tuple[int, int] = (33, 40)  # dedicated, always-full-8-epoch "did it plateau" window
FULL_RANGE: tuple[int, int] = (5, 40)


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError(f"sample_std requires at least 2 values, got {len(values)}")
    return float(statistics.stdev(values))


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Ordinary-least-squares slope of ys against xs (closed form, no numpy dependency)."""
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have the same length")
    if n < 2:
        raise ValueError(f"ols_slope requires at least 2 points, got {n}")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0:
        raise ValueError("ols_slope requires xs to take at least two distinct values")
    return covariance / variance


def window_epochs(window: tuple[int, int]) -> list[int]:
    start, end = window
    if end < start:
        raise ValueError(f"invalid window {window}: end < start")
    return list(range(start, end + 1))


def window_label(window: tuple[int, int]) -> str:
    return f"{window[0]}-{window[1]}"


def summarize_window(per_epoch_mean_r2: dict[str, float], window: tuple[int, int]) -> dict:
    epochs = window_epochs(window)
    values = [per_epoch_mean_r2[str(epoch)] for epoch in epochs]
    slope = ols_slope([float(e) for e in epochs], values)
    return {
        "start_epoch": window[0],
        "end_epoch": window[1],
        "n_epochs": len(epochs),
        "partial": len(epochs) != 8,
        "mean_r2": mean(values),
        "within_window_std": sample_std(values) if len(values) >= 2 else None,
        "slope_per_epoch": slope,
        "cumulative_change": slope * (len(epochs) - 1),
    }


def load_curve_artifact(results_dir: Path, seed: int) -> dict:
    path = results_dir / f"curve_s{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing convergence curve artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_aggregation(results_dir: Path) -> dict:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    artifacts = {seed: load_curve_artifact(results_dir, seed) for seed in SEEDS}

    reference = artifacts[SEEDS[0]]
    reference_epoch_list = reference["epoch_list"]
    reference_splits = reference["session_splits"]
    reference_protocol = reference["protocol"]
    for seed, payload in artifacts.items():
        if payload["epoch_list"] != reference_epoch_list:
            raise ValueError(f"seed {seed}: epoch_list differs from seed {SEEDS[0]}")
        if payload["session_splits"] != reference_splits:
            raise ValueError(f"seed {seed}: session_splits differs from seed {SEEDS[0]}")
        if payload["protocol"] != reference_protocol:
            raise ValueError(f"seed {seed}: protocol differs from seed {SEEDS[0]}")
        if payload["variant"] != reference["variant"]:
            raise ValueError(f"seed {seed}: variant differs from seed {SEEDS[0]}")

    run_dirs = {seed: artifacts[seed]["run_dir"] for seed in SEEDS}
    if len(set(run_dirs.values())) != len(SEEDS):
        raise ValueError(f"Two or more seeds share a run directory (this was v3 bug H.4): {run_dirs}")

    per_seed_curve = {
        str(seed): {epoch: artifacts[seed]["per_epoch_mean_r2"][str(epoch)] for epoch in reference_epoch_list}
        for seed in SEEDS
    }

    cross_seed_mean_curve = {
        str(epoch): mean([per_seed_curve[str(seed)][epoch] for seed in SEEDS])
        for epoch in reference_epoch_list
    }
    cross_seed_std_curve = {
        str(epoch): sample_std([per_seed_curve[str(seed)][epoch] for seed in SEEDS])
        for epoch in reference_epoch_list
    }

    per_seed_window_slopes = {}
    for seed in SEEDS:
        per_epoch_mean_r2 = artifacts[seed]["per_epoch_mean_r2"]
        per_seed_window_slopes[str(seed)] = {
            window_label(window): summarize_window(per_epoch_mean_r2, window)
            for window in SUCCESSIVE_WINDOWS
        }

    cross_seed_window_slope_mean = {
        window_label(window): mean(
            [per_seed_window_slopes[str(seed)][window_label(window)]["slope_per_epoch"] for seed in SEEDS]
        )
        for window in SUCCESSIVE_WINDOWS
    }

    final_window_per_seed = {
        str(seed): summarize_window(artifacts[seed]["per_epoch_mean_r2"], FINAL_WINDOW) for seed in SEEDS
    }
    final_window_summary = {
        "window": window_label(FINAL_WINDOW),
        "per_seed": final_window_per_seed,
        "mean_slope_per_epoch": mean([final_window_per_seed[str(seed)]["slope_per_epoch"] for seed in SEEDS]),
        "mean_cumulative_change": mean(
            [final_window_per_seed[str(seed)]["cumulative_change"] for seed in SEEDS]
        ),
        "mean_within_window_std": mean(
            [final_window_per_seed[str(seed)]["within_window_std"] for seed in SEEDS]
        ),
    }

    overall_per_seed = {
        str(seed): summarize_window(artifacts[seed]["per_epoch_mean_r2"], FULL_RANGE) for seed in SEEDS
    }
    overall_summary = {
        "window": window_label(FULL_RANGE),
        "per_seed": overall_per_seed,
        "mean_slope_per_epoch": mean([overall_per_seed[str(seed)]["slope_per_epoch"] for seed in SEEDS]),
    }

    return {
        "schema_version": 1,
        "purpose": "convergence_swa_v1_e2_convergence_curve",
        "screen_id": "convergence_swa_v1",
        "protocol_docs": [
            "sua_exploration/ROADMAP.md",
            "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md",
        ],
        "variant": reference["variant"],
        "seeds": list(SEEDS),
        "epoch_list": reference_epoch_list,
        "protocol": reference_protocol,
        "session_splits": reference_splits,
        "per_seed_curve": per_seed_curve,
        "cross_seed_mean_curve": cross_seed_mean_curve,
        "cross_seed_std_curve": cross_seed_std_curve,
        "successive_8_epoch_windows": {
            window_label(window): {
                "per_seed": {str(seed): per_seed_window_slopes[str(seed)][window_label(window)] for seed in SEEDS},
                "mean_slope_per_epoch": cross_seed_window_slope_mean[window_label(window)],
            }
            for window in SUCCESSIVE_WINDOWS
        },
        "final_8_epoch_window_33_40": final_window_summary,
        "overall_5_to_40": overall_summary,
        "source_artifacts": {str(seed): str(results_dir / f"curve_s{seed}.json") for seed in SEEDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "convergence_swa_v1")

    payload = run_aggregation(results_dir)

    out_path = args.out_path or (results_dir / "convergence.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Final 8-epoch window (33-40) mean slope/epoch: {payload['final_8_epoch_window_33_40']['mean_slope_per_epoch']:.5f}")
    print(f"Final 8-epoch window (33-40) mean cumulative change: {payload['final_8_epoch_window_33_40']['mean_cumulative_change']:.5f}")


if __name__ == "__main__":
    main()
