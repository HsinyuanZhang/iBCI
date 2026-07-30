"""Strict aggregate for the native-FALCON M1/M2 F0/T4/TS4 internal-LOSO screen."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml

CELLS = ((1, 42), (1, 43), (2, 42))
GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "ts4": ("B3S", "ts4")}


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _read_score(metrics: Path) -> float:
    with metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test_heldin":
                return float(row["R2_variance_weighted"])
    raise ValueError(f"No test_heldin score in {metrics}")


def _find(root: Path, screen_id: str, group: str, task: str, fold: int, seed: int) -> Path:
    prefix = f"{screen_id}_{group}_{task}_f{fold}_s{seed}_"
    matches = []
    for metadata in root.glob("*/run_metadata.json"):
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if str(payload.get("run_id", "")).startswith(prefix):
            matches.append(metadata.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one artifact for {prefix}*, found {matches}")
    return matches[0]


def _validate(path: Path, *, group: str, task: str, fold: int, seed: int) -> dict:
    required = [path / name for name in ("resolved_config.yaml", "split_manifest.json", "run_metadata.json", "metrics_summary.csv")]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError(f"Incomplete native-MUA artifact {path}: {missing}")
    resolved = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    split = json.loads((path / "split_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    variant, side_group = GROUPS[group]
    data, model = resolved.get("data", {}), resolved.get("model", {})
    expected = {
        "seed": seed, "task": task, "fold": fold, "variant": variant,
        "side_feature_group": side_group, "random_calibration": False,
        "fit_heldout": False, "test_heldout": False, "protocol": "loso",
        "max_epochs": 12,
    }
    observed = {
        "seed": resolved.get("seed"), "task": data.get("task"), "fold": data.get("loso_fold"),
        "variant": model.get("variant"), "side_feature_group": data.get("side_feature_group"),
        "random_calibration": data.get("random_calibration"),
        "fit_heldout": data.get("include_heldout_in_fit"), "test_heldout": data.get("include_heldout_in_test"),
        "protocol": data.get("validation_protocol"), "max_epochs": resolved.get("trainer", {}).get("max_epochs"),
    }
    bad = {key: {"expected": expected[key], "observed": observed[key]} for key in expected if expected[key] != observed[key]}
    if bad:
        raise ValueError(f"Native-MUA contract mismatch for {path}: {bad}")
    if split.get("heldout_evaluated_in_fit") or split.get("heldout_evaluated_in_test"):
        raise ValueError(f"Held-out FALCON scope was evaluated by {path}")
    if metadata.get("fold_id") != fold or metadata.get("seed") != seed:
        raise ValueError(f"Run metadata fold/seed mismatch for {path}")
    return {"path": str(path), "score": _read_score(path / "metrics_summary.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--task", choices=("m1", "m2", "both"), default="both")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = root / "streaming_calibration_exp" / "outputs" / "streaming_calibration"
    scores, artifacts = {}, {}
    tasks = ("m1", "m2") if args.task == "both" else (args.task,)
    for task in tasks:
        scores[task], artifacts[task] = {}, {}
        for group in GROUPS:
            values, paths = {}, {}
            for fold, seed in CELLS:
                key = f"fold{fold}_seed{seed}"
                record = _validate(_find(output_root, args.screen_id, group, task, fold, seed), group=group, task=task, fold=fold, seed=seed)
                values[key], paths[key] = record["score"], record["path"]
            scores[task][group], artifacts[task][group] = values, paths
    deltas = {}
    for task in scores:
        t4_f0 = [scores[task]["t4"][key] - scores[task]["f0"][key] for key in scores[task]["f0"]]
        t4_ts4 = [scores[task]["t4"][key] - scores[task]["ts4"][key] for key in scores[task]["f0"]]
        deltas[task] = {
            "T4_minus_F0": {"per_cell": dict(zip(scores[task]["f0"], t4_f0)), "mean": _mean(t4_f0)},
            "T4_minus_TS4": {"per_cell": dict(zip(scores[task]["f0"], t4_ts4)), "mean": _mean(t4_ts4)},
        }
    payload = {
        "schema_version": 1,
        "purpose": "native_falcon_mua_t4_internal_loso_development_only",
        "screen_id": args.screen_id,
        "task_scope": list(tasks),
        "no_formal_test_sessions_evaluated": True,
        "cells": [f"fold{f}_seed{s}" for f, s in CELLS],
        "scores": scores,
        "artifacts": artifacts,
        "paired_deltas": deltas,
    }
    default_name = "aggregate.json" if args.task == "both" else f"aggregate_{args.task}.json"
    out = args.out_path or root / "sua_exploration" / "results" / args.screen_id / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({task: {name: value["mean"] for name, value in values.items()} for task, values in deltas.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
