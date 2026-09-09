"""Root-owned temporal holdout authority; one source fit scores both future dates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/chronological_last2_v1"
ARMS = ("Z_NONE", "B_ACTIVITY_ONLY", "D_JOINT")
SEED = 42
SCHEMA = "chronological_last2_calibration_ablation_v1"

DATASETS = {
    "m1": {
        "split_id": "m1_last2_20120927_20120928",
        "source_dates": ["2012-09-24", "2012-09-26"],
        "source_sessions": ["ses-20120924", "ses-20120926"],
        "target_dates": ["2012-09-27", "2012-09-28"],
        "target_sessions": ["ses-20120927", "ses-20120928"],
        "session_dates": {
            "ses-20120924": "2012-09-24", "ses-20120926": "2012-09-26",
            "ses-20120927": "2012-09-27", "ses-20120928": "2012-09-28",
        },
        "mode": "fresh_source_fit", "epochs": 24,
        "source_support": "native trial indices [0,10)",
        "source_train": "native trial indices [10,310)",
        "source_validation": "native trial indices [310,end)",
        "target_support": "native trial indices [0,10)",
        "target_query": "native trial indices [10,210)",
        "selection": "earliest maximum mean R2 across the two source dates",
        "carrier": "rSyn3, original primary computation; no residual-pilot substitution",
    },
    "m2": {
        "split_id": "m2_source7_last2_20201118_20201119",
        "source_dates": ["2020-10-19", "2020-10-20", "2020-10-27", "2020-10-28"],
        "source_sessions": [
            "ses-2020-10-19-Run1", "ses-2020-10-19-Run2",
            "ses-2020-10-20-Run1", "ses-2020-10-20-Run2",
            "ses-2020-10-27-Run1", "ses-2020-10-27-Run2", "ses-2020-10-28-Run1",
        ],
        "target_dates": ["2020-11-18", "2020-11-19"],
        "target_sessions": ["ses-2020-11-18-Run1", "ses-2020-11-19-Run1"],
        "session_dates": {
            "ses-2020-10-19-Run1": "2020-10-19", "ses-2020-10-19-Run2": "2020-10-19",
            "ses-2020-10-20-Run1": "2020-10-20", "ses-2020-10-20-Run2": "2020-10-20",
            "ses-2020-10-27-Run1": "2020-10-27", "ses-2020-10-27-Run2": "2020-10-27",
            "ses-2020-10-28-Run1": "2020-10-28", "ses-2020-11-18-Run1": "2020-11-18",
            "ses-2020-11-19-Run1": "2020-11-19",
        },
        "mode": "reuse_completed_source7_fit", "epochs": 24,
        "source_support": "first 33 native trials",
        "source_train": "first 80% of post-M33 whole source trials",
        "source_validation": "last 20% of post-M33 whole source trials, context isolated",
        "target_support": "first 33 native trials",
        "target_query": "available trials after M33",
        "selection": "retain original earliest maximum equal-session source7 validation R2",
        "carrier": "MOVE-T4",
        "unused_earlier_target_date": "2020-10-30 is a supplemental target, not a training date",
        "zero_query_date": "2020-11-24 has exactly 33 available trials in each run, hence no post-M33 query",
    },
    "h1": {
        "split_id": "h1_last2_19250119_19250120",
        "source_dates": ["1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15"],
        "source_sessions": [
            "ses-19250101T111740", "ses-19250101T112404",
            "ses-19250108T110520", "ses-19250108T111022", "ses-19250108T111455",
            "ses-19250113T120811", "ses-19250113T121303",
            "ses-19250115T110633", "ses-19250115T111328",
        ],
        "target_dates": ["1925-01-19", "1925-01-20"],
        "target_sessions": [
            "ses-19250119T113543", "ses-19250119T114045",
            "ses-19250120T115044", "ses-19250120T115537",
        ],
        "session_dates": {
            "ses-19250101T111740": "1925-01-01", "ses-19250101T112404": "1925-01-01",
            "ses-19250108T110520": "1925-01-08", "ses-19250108T111022": "1925-01-08", "ses-19250108T111455": "1925-01-08",
            "ses-19250113T120811": "1925-01-13", "ses-19250113T121303": "1925-01-13",
            "ses-19250115T110633": "1925-01-15", "ses-19250115T111328": "1925-01-15",
            "ses-19250119T113543": "1925-01-19", "ses-19250119T114045": "1925-01-19",
            "ses-19250120T115044": "1925-01-20", "ses-19250120T115537": "1925-01-20",
        },
        "mode": "fresh_source_fit", "epochs": 32,
        "source_support": "first three available eval-valid native trials",
        "source_train": "available native trial indices [3:-2], stride4",
        "source_validation": "last two available native trials, stride4",
        "target_support": "first three available eval-valid native trials",
        "target_query": "all remaining available native trials, stride1",
        "selection": "earliest maximum equal-date source-validation R2; sessions averaged within each date",
        "carrier": "H-C",
    },
}


def protocol() -> dict:
    for name, row in DATASETS.items():
        sources, targets = set(row["source_sessions"]), set(row["target_sessions"])
        if sources & targets or max(row["source_dates"]) >= min(row["target_dates"]):
            raise RuntimeError(f"nonchronological or overlapping split: {name}")
        if len(row["target_dates"]) != 2:
            raise RuntimeError(f"exactly two target dates required: {name}")
        for side in ("source", "target"):
            if {row["session_dates"][s] for s in row[f"{side}_sessions"]} != set(row[f"{side}_dates"]):
                raise RuntimeError(f"session/date roster mismatch: {name}/{side}")
    return {
        "schema": SCHEMA, "seed": SEED, "datasets": DATASETS,
        "arms": {"Z_NONE": "no calibration", "B_ACTIVITY_ONLY": "activity calibration", "D_JOINT": "activity calibration plus task carrier"},
        "primary_cells": 9, "fresh_training_cells": 6, "reused_training_cells": 3,
        "primary_endpoint": "source-validation-selected EMA, shared by both target dates",
        "sensitivity_endpoint": "fixed final EMA: e24 M1/M2, e32 H1",
        "target_aggregation": "mean per-session R2 within date, then equal mean of the two target dates",
        "contributions": {"activity": "B minus Z", "carrier": "D minus B"},
        "target_optimizer_steps": 0, "target_query_labels_used_for_selection": False,
        "scope": "public chronological transfer; single paired seed, descriptive evidence; not an independent hidden test",
        "prior_lodo": "superseded as primary, stopped; completed historical artifacts retained without relabelling",
        "m1_refinement_pilot": "historical exploratory three-fold result; fourth LOSO pilot cancelled with prior scan",
    }


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol():
        raise RuntimeError("refuse changing an existing protocol")
    atomic_json(path, protocol())
    print(json.dumps({"path": str(path), "sha256": sha(path)}))
