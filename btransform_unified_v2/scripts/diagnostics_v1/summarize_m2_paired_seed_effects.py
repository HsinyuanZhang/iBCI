#!/usr/bin/env python3
"""Read completed M2 joint B/D ext4 receipts as paired three-seed effects.

This never rescans data.  A full result needs B and D for seeds 42, 43, and
44.  Sessions are repeated measurements within a seed, not extra independent
replicates, so the only dispersion reported is across the three seed deltas.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (42, 43, 44)
ARMS = {"B": "B_ACTIVITY_ONLY", "D": "D_JOINT"}
SESSIONS = {"ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1", "ses-2020-11-19-Run1"}
WINDOWS = {"ses-2020-10-30-Run1": 519, "ses-2020-10-30-Run2": 490, "ses-2020-11-18-Run1": 425, "ses-2020-11-19-Run1": 635}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"results/rift_v1/m2_r50_joint_{arm.lower()}_s{seed}_formal_v1"


def load_completed(root: Path, arm: str, seed: int) -> tuple[dict[str, Any] | None, list[str]]:
    run = run_dir(root, arm, seed)
    meta_path, train_path, score_path = run / "run_meta.json", run / "train_receipt.json", run / "score_receipt.json"
    missing = [str(path) for path in (meta_path, train_path, score_path) if not path.is_file()]
    if missing:
        return None, missing
    meta, train, score = (json.loads(path.read_text()) for path in (meta_path, train_path, score_path))
    expected_epochs = {str(i) for i in range(1, 25)}
    if not (meta.get("schema") == "m2_rift_joint_train_v2" and meta.get("status") == "FORMAL" and meta.get("cell") == "M2-RIFT-R50-D4-JOINT-FILM-M33-V1" and meta.get("arm") == ARMS[arm] and meta.get("seed") == seed and meta.get("sampler_seed") == 42 and meta.get("epochs") == 24 and
            isinstance(meta.get("source_hashes"), dict) and bool(meta["source_hashes"]) and isinstance(meta.get("cache_hashes"), dict) and bool(meta["cache_hashes"]) and
            train.get("schema") == "m2_rift_joint_train_receipt_v2" and train.get("status") == "COMPLETED" and train.get("cell") == meta["cell"] and train.get("arm") == ARMS[arm] and train.get("seed") == seed and train.get("sampler_seed") == 42 and train.get("epochs") == 24 and train.get("global_step") == 75960 and train.get("source_hashes") == meta["source_hashes"] and train.get("cache_hashes") == meta["cache_hashes"] and
            score.get("schema") == "m2_rift_joint_ext4_epoch_scan_v1" and score.get("status") == "COMPLETED" and score.get("cell") == meta["cell"] and score.get("arm") == ARMS[arm] and score.get("seed") == seed and score.get("sampler_seed") == 42 and score.get("official_test_used") is False and
            score.get("source_hashes") == meta["source_hashes"] and score.get("cache_hashes") == meta["cache_hashes"] and set(score.get("ema_by_epoch", {})) == expected_epochs):
        return None, [f"invalid completed receipt contract: {run}"]
    if any(not Path(path).is_file() or sha(Path(path)) != digest for path, digest in meta["source_hashes"].items()):
        return None, [f"live source hash drift: {run}"]
    cache_root = root.parent / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"
    for surface, sessions in meta["cache_hashes"].items():
        for session, files in sessions.items():
            for name, digest in files.items():
                path = cache_root / surface / session / name
                if not path.is_file() or sha(path) != digest:
                    return None, [f"live cache hash drift: {path}"]
    rows = score["ema_by_epoch"]
    values: dict[int, float] = {}
    for epoch in range(1, 25):
        row = rows[str(epoch)]; checkpoint = run / f"epoch_{epoch:03d}.pt"
        per = row.get("per_session", {})
        if (not checkpoint.is_file() or row.get("checkpoint_sha256") != sha(checkpoint) or row.get("partial") is not False or row.get("n_windows") != sum(WINDOWS.values()) or
                set(per) != SESSIONS or any(int(per[s].get("window_count", -1)) != WINDOWS[s] or not math.isfinite(float(per[s].get("r2", float("nan")))) for s in SESSIONS)):
            return None, [f"invalid epoch row/checkpoint: {run}/epoch_{epoch:03d}.pt"]
        mean = float(np.mean([float(per[s]["r2"]) for s in sorted(SESSIONS)]))
        if not math.isfinite(float(row.get("equal_session_mean", float("nan")))) or abs(float(row["equal_session_mean"]) - mean) > 1e-12:
            return None, [f"forged/non-finite epoch mean: {run}/epoch_{epoch:03d}.pt"]
        values[epoch] = mean
    selected_epoch = int(score["selection"]["epoch"])
    selected = score["ema_by_epoch"].get(str(selected_epoch))
    endpoint = score["ema_by_epoch"].get("24")
    expected_best = max(range(1, 25), key=lambda epoch: (values[epoch], -epoch))
    if (not isinstance(selected, dict) or not isinstance(endpoint, dict) or set(selected.get("per_session", {})) != SESSIONS or set(endpoint.get("per_session", {})) != SESSIONS or
            selected_epoch != expected_best or score["selection"].get("rule") != "earliest best equal_session_mean" or not math.isfinite(float(score["selection"].get("equal_session_mean", float("nan")))) or abs(float(score["selection"].get("equal_session_mean", float("nan"))) - values[expected_best]) > 1e-12):
        return None, [f"incomplete ext4 rows: {run}"]
    return {"run": str(run), "run_meta_sha256": sha(meta_path), "train_receipt_sha256": sha(train_path), "score_receipt_sha256": sha(score_path),
            "selected_epoch": selected_epoch, "selected": selected, "epoch_24": endpoint}, []


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite completed receipt value")
    return result


def summarize(root: Path) -> dict[str, Any]:
    loaded: dict[tuple[str, int], dict[str, Any]] = {}; missing: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            row, absent = load_completed(root, arm, seed)
            missing.extend(absent)
            if row is not None:
                loaded[(arm, seed)] = row
    common = {"schema": "m2_joint_paired_seed_effects_v1", "surface": "ext4 mechanism control only; never ext6", "arms": ARMS, "seeds": list(SEEDS),
              "independence_unit": "seed-paired B versus D; sessions are within-seed repeated measurements", "no_significance_test": True,
              "evalai_opened": False, "official_test_used": False}
    if missing:
        return {**common, "status": "PENDING", "missing_or_invalid": sorted(missing), "completed_pairs": sorted(f"{arm}{seed}" for arm, seed in loaded),
                "reason": "all B/D seeds 42/43/44 are required before reporting a three-seed mean or SD"}
    pairs = []
    for seed in SEEDS:
        b, d = loaded[("B", seed)], loaded[("D", seed)]
        picked_sessions = {s: finite(d["selected"]["per_session"][s]["r2"]) - finite(b["selected"]["per_session"][s]["r2"]) for s in sorted(SESSIONS)}
        fixed_sessions = {s: finite(d["epoch_24"]["per_session"][s]["r2"]) - finite(b["epoch_24"]["per_session"][s]["r2"]) for s in sorted(SESSIONS)}
        pairs.append({"seed": seed, "B": b, "D": d,
                      "independent_epoch_pick_delta_D_minus_B": finite(d["selected"]["equal_session_mean"]) - finite(b["selected"]["equal_session_mean"]),
                      "fixed_epoch_24_delta_D_minus_B": finite(d["epoch_24"]["equal_session_mean"]) - finite(b["epoch_24"]["equal_session_mean"]),
                      "independent_epoch_pick_per_session_delta_D_minus_B": picked_sessions,
                      "fixed_epoch_24_per_session_delta_D_minus_B": fixed_sessions})
    def mean_sd(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in pairs], dtype=np.float64)
        return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)), "n_independent_seed_pairs": 3}
    return {**common, "status": "COMPLETED", "pairs": pairs,
            "three_seed_summary": {"independent_epoch_pick_delta_D_minus_B": mean_sd("independent_epoch_pick_delta_D_minus_B"),
                                   "fixed_epoch_24_delta_D_minus_B": mean_sd("fixed_epoch_24_delta_D_minus_B")},
            "interpretation": "Descriptive paired seed effects only; no CI, p-value, or session-level pseudo-replication."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = summarize(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
