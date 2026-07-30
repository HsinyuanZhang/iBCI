"""E1 aggregator: combine the 3 per-seed SWA evaluations (``eval_swa_dandi688.py`` outputs,
``swa_s{seed}.json``) with the 3 per-seed convergence curves (``eval_convergence_curve_
dandi688.py`` outputs, ``curve_s{seed}.json``) into ``results/convergence_swa_v1/swa.json``.

The question E1 asks is NOT "is the SWA model better on average" -- it is "does SWA reduce
sigma_seed" (sua_exploration/ROADMAP.md "当前实验计划"). So for each SWA window (last 5/10/20
epochs) this reports, side by side with the "plain epoch-window average" reference (the mean
of the INDIVIDUALLY-evaluated per-epoch protocol scores over that same trailing window --
i.e. what M3 already does, generalized to windows other than 8 epochs, computed here purely
by re-aggregating curve_s{seed}.json, no extra evaluation needed):
  - the 3 per-seed scores;
  - their across-seed std (the sigma_seed estimate that matters);
  - the mean.

A window is a "win" for SWA if it gives a comparable mean with a materially smaller
across-seed std than the plain-average reference at the same window size -- this script
reports the numbers; it does not itself declare a verdict (that judgment call belongs in
CONVERGENCE_AND_SWA.md, not baked into an aggregator that future re-runs will reuse
unmodified).

No GPU, no NWB data, no torch: pure JSON + arithmetic.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Sequence

SEEDS: tuple[int, ...] = (42, 43, 44)
WINDOWS: tuple[int, ...] = (5, 10, 20)
MAX_EPOCH = 40


def mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError(f"sample_std requires at least 2 values, got {len(values)}")
    return float(statistics.stdev(values))


def plain_window_average(per_epoch_mean_r2: dict[str, float], window: int, max_epoch: int = MAX_EPOCH) -> float:
    """Mean of the individually-evaluated per-epoch scores over the trailing `window` epochs
    [max_epoch-window+1, max_epoch] -- the M3-style "average measurements" reference, reused
    from the already-computed convergence curve (no extra evaluation)."""
    epochs = range(max_epoch - window + 1, max_epoch + 1)
    return mean([per_epoch_mean_r2[str(epoch)] for epoch in epochs])


def load_json(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_aggregation(results_dir: Path) -> dict:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    swa_artifacts = {
        seed: load_json(results_dir / f"swa_s{seed}.json", what=f"SWA artifact for seed {seed}")
        for seed in SEEDS
    }
    curve_artifacts = {
        seed: load_json(results_dir / f"curve_s{seed}.json", what=f"convergence curve artifact for seed {seed}")
        for seed in SEEDS
    }

    reference_splits = swa_artifacts[SEEDS[0]]["session_splits"]
    for seed in SEEDS:
        if swa_artifacts[seed]["session_splits"] != reference_splits:
            raise ValueError(f"seed {seed}: SWA session_splits differs from seed {SEEDS[0]}")
        if curve_artifacts[seed]["session_splits"] != reference_splits:
            raise ValueError(f"seed {seed}: curve session_splits differs from seed {SEEDS[0]}")
        swa_windows_present = {int(w) for w in swa_artifacts[seed]["per_window"]}
        if swa_windows_present != set(WINDOWS):
            raise ValueError(
                f"seed {seed}: SWA artifact windows {sorted(swa_windows_present)} != expected {list(WINDOWS)}"
            )

    swa_run_dirs = {seed: swa_artifacts[seed]["run_dir"] for seed in SEEDS}
    if len(set(swa_run_dirs.values())) != len(SEEDS):
        raise ValueError(f"Two or more seeds share a run directory (this was v3 bug H.4): {swa_run_dirs}")

    per_window: dict[str, dict] = {}
    for window in WINDOWS:
        swa_scores = {seed: swa_artifacts[seed]["per_window"][str(window)]["mean_r2"] for seed in SEEDS}
        plain_scores = {
            seed: plain_window_average(curve_artifacts[seed]["per_epoch_mean_r2"], window) for seed in SEEDS
        }
        swa_values = [swa_scores[seed] for seed in SEEDS]
        plain_values = [plain_scores[seed] for seed in SEEDS]
        swa_mean, swa_std = mean(swa_values), sample_std(swa_values)
        plain_mean, plain_std = mean(plain_values), sample_std(plain_values)
        per_window[str(window)] = {
            "swa": {
                "per_seed": {str(seed): swa_scores[seed] for seed in SEEDS},
                "mean": swa_mean,
                "across_seed_std": swa_std,
                "swa_checkpoint_sha256": {
                    str(seed): swa_artifacts[seed]["per_window"][str(window)]["swa_checkpoint_sha256"]
                    for seed in SEEDS
                },
            },
            "plain_epoch_window_average": {
                "per_seed": {str(seed): plain_scores[seed] for seed in SEEDS},
                "mean": plain_mean,
                "across_seed_std": plain_std,
            },
            "mean_delta_swa_minus_plain": swa_mean - plain_mean,
            "across_seed_std_ratio_swa_over_plain": (swa_std / plain_std) if plain_std > 0 else None,
        }

    return {
        "schema_version": 1,
        "purpose": "convergence_swa_v1_e1_swa_vs_plain_window_average",
        "screen_id": "convergence_swa_v1",
        "protocol_docs": ["sua_exploration/ROADMAP.md", "sua_exploration/docs/MEASUREMENT_PROTOCOL_V4.md"],
        "variant": swa_artifacts[SEEDS[0]]["variant"],
        "seeds": list(SEEDS),
        "windows": list(WINDOWS),
        "max_epoch": MAX_EPOCH,
        "session_splits": reference_splits,
        "key_question": (
            "Does averaging WEIGHTS (SWA) across a trailing epoch window reduce the "
            "across-seed std (sigma_seed) relative to averaging the individually-evaluated "
            "per-epoch SCORES over the same window (the plain_epoch_window_average "
            "reference, i.e. the M3 approach generalized to windows other than 8 epochs)? "
            "A window is a win for SWA if its mean is comparable to the plain-average mean "
            "AND its across_seed_std is materially smaller."
        ),
        "per_window": per_window,
        "source_artifacts": {
            "swa": {str(seed): str(results_dir / f"swa_s{seed}.json") for seed in SEEDS},
            "curve": {str(seed): str(results_dir / f"curve_s{seed}.json") for seed in SEEDS},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    results_dir = args.results_dir or (root / "sua_exploration" / "results" / "convergence_swa_v1")

    payload = run_aggregation(results_dir)

    out_path = args.out_path or (results_dir / "swa.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")
    for window in WINDOWS:
        w = payload["per_window"][str(window)]
        print(
            f"window last-{window}: swa std={w['swa']['across_seed_std']:.4f} mean={w['swa']['mean']:.4f}  "
            f"plain std={w['plain_epoch_window_average']['across_seed_std']:.4f} "
            f"mean={w['plain_epoch_window_average']['mean']:.4f}  "
            f"std_ratio(swa/plain)={w['across_seed_std_ratio_swa_over_plain']}"
        )


if __name__ == "__main__":
    main()
