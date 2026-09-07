#!/usr/bin/env python3
"""CPU-only 0a audit for the narrowly defined M2 strict T4-clean-SPINT endpoint.

This intentionally does not estimate an omnibus "M2 noise floor".  The strict
future-query data and the older incomplete fold-by-seed native artifacts are
kept in separate evidence sections; see the immutable v2 protocol receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sua_exploration" / "results" / "m2_uncertainty_identifiability_v1"
RECEIPT = RESULTS / "protocol_receipt_v2.json"
STRICT = ROOT / "sua_exploration" / "results" / "m2_t4_clean_spint_replication_v1" / "aggregate_3seed_final.json"
HISTORICAL = ROOT / "sua_exploration" / "results" / "native_mua_t4_v1" / "aggregate_m2.json"
EXPECTED_STRICT_SHA = "b3cace02693bd556f69282a2352c661c34773e828fd915f65a81aa784bff0e49"
EXPECTED_HISTORICAL_SHA = "c6eb1727456040b02f333260f67f47f134cade9e0f556ca912169b0da3fc8613"
SEEDS = (42, 43, 44)
HISTORICAL_CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
HISTORICAL_RUNS = {
    "f0_fold1_seed42": ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-29-14-05-18-148561_rid-native_mua_t4_v1_f0_m2_f1_s42",
    "t4_fold1_seed42": ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-29-14-20-47-856461_rid-native_mua_t4_v1_t4_m2_f1_s42",
    "f0_fold1_seed43": ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-29-14-51-54-956168_rid-native_mua_t4_v1_f0_m2_f1_s43",
    "t4_fold1_seed43": ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-29-15-23-34-030659_rid-native_mua_t4_v1_t4_m2_f1_s43",
    "f0_fold2_seed42": ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-29-15-07-21-571055_rid-native_mua_t4_v1_f0_m2_f2_s42",
    "t4_fold2_seed42": ROOT / "streaming_calibration_exp/logs/train/runs/2026-07-29-15-39-06-581407_rid-native_mua_t4_v1_t4_m2_f2_s42",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_sample_sd(values: list[float]) -> float | None:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) >= 2 else None


def epoch_availability() -> dict[str, Any]:
    """Extract only raw saved validation epochs; this is not a run variance."""
    records: list[dict[str, Any]] = []
    for name, run_dir in HISTORICAL_RUNS.items():
        record: dict[str, Any] = {
            "run": name,
            "path": str(run_dir),
            "available": False,
            "epochs_5_through_12": [],
            "within_run_epoch_window_sd": None,
            "interpretation": "missing",
        }
        if run_dir.is_dir():
            try:
                accumulator = EventAccumulator(str(run_dir))
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
                if "val_heldin/r2_mean" in tags:
                    vals = [(int(x.step) + 1, float(x.value)) for x in accumulator.Scalars("val_heldin/r2_mean")]
                    # Lightning saves 0-indexed steps, while the protocol calls them epochs 1--12.
                    trailing = [{"epoch": epoch, "r2": value} for epoch, value in vals if 5 <= epoch <= 12]
                    record.update({
                        "available": len(trailing) == 8,
                        "epochs_5_through_12": trailing,
                        "within_run_epoch_window_sd": finite_sample_sd([x["r2"] for x in trailing]),
                        "interpretation": "autocorrelated_within_run_fluctuation_not_independent_run_sigma" if len(trailing) == 8 else "incomplete_epoch_window",
                    })
            except Exception as exc:  # provenance audit must preserve missing, not guess.
                record["load_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return {
        "evidence_section": "B_historical_native_training_context_only",
        "design": {"cells": list(HISTORICAL_CELLS), "complete_fold_by_seed_factorial": False},
        "records": records,
        "prohibited_interpretation": [
            "not sigma_seed", "not sigma_fold", "not fold_by_seed_interaction", "not independent run sigma", "not a strict-future-query MDE input"
        ],
    }


def strict_rows(strict: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    by_seed = strict["per_seed_session_deltas_t4_minus_clean_spint"]
    expected = {str(seed) for seed in SEEDS}
    if set(by_seed) != expected:
        raise ValueError(f"strict evidence seed drift: expected {sorted(expected)}, found {sorted(by_seed)}")
    session_order = list(by_seed[str(SEEDS[0])])
    if len(session_order) != 6 or any(set(by_seed[str(seed)]) != set(session_order) for seed in SEEDS):
        raise ValueError("strict evidence must have the same six sessions for every seed")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for session in session_order:
            rows.append({"evidence": "A_strict_future_query", "fold": 1, "seed": seed, "session": session,
                         "t4_minus_clean_spint_r2": float(by_seed[str(seed)][session])})
    seed_means = {str(seed): float(np.mean([by_seed[str(seed)][s] for s in session_order])) for seed in SEEDS}
    session_means = {session: float(np.mean([by_seed[str(seed)][session] for seed in SEEDS])) for session in session_order}
    return rows, seed_means, session_means


def bootstrap_interval(rows: list[dict[str, Any]], n_resamples: int = 10_000, seed: int = 20260801) -> dict[str, Any]:
    sessions = sorted({str(row["session"]) for row in rows})
    data = np.asarray([[row["t4_minus_clean_spint_r2"] for row in rows if row["session"] == session] for session in sessions], dtype=float)
    if data.shape != (6, 3):
        raise ValueError(f"expected 6 session clusters by 3 within-fold seeds, got {data.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sessions), size=(n_resamples, len(sessions)))
    means = data[idx].mean(axis=(1, 2))
    return {
        "unit_resampled": "session cluster; all three within-fold seed observations are retained inside a selected session",
        "n_resamples": n_resamples,
        "rng_seed": seed,
        "equal_session_equal_seed_mean": float(data.mean()),
        "percentile_95_ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "local_development_caveat": "Conditions on the available fold-1 seeds and six visible local held-out sessions; not a population confidence interval over folds or hidden test sessions.",
    }


def mde_rows(observed_seed_sd: float | None, observed_session_sd: float | None) -> list[dict[str, Any]]:
    # Fixed normal-approximation sensitivity: MDE80 = (z_.975 + z_.80)*sigma/sqrt(n).
    # It is deliberately a sensitivity display, not a falsely precise power claim.
    sd_grid: list[tuple[float, str]] = [(x, "fixed_grid") for x in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)]
    for value, basis in ((observed_seed_sd, "observed_within_fold_seed_sd"), (observed_session_sd, "observed_session_cluster_sd")):
        if value is not None and all(abs(value - x) > 1e-12 for x, _ in sd_grid):
            sd_grid.append((value, basis))
    rows = []
    for n in (3, 4, 6, 8, 10, 12):
        for sd, sd_basis in sorted(sd_grid):
            rows.append({
                "effective_independent_session_clusters": n,
                "plausible_paired_delta_sd": sd,
                "sd_basis": sd_basis,
                "mde_80pct_power_two_sided_alpha_0_05_normal_approx": (1.959963984540054 + 0.8416212335729143) * sd / math.sqrt(n),
                "ci95_half_width_normal_approx": 1.959963984540054 * sd / math.sqrt(n),
                "notes": "Sensitivity only. n is an assumed number of independent session clusters; existing repeated seed fits do not increase n.",
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite result directory: {out}")
    receipt = read_json(RECEIPT)
    if receipt["status"] != "PENDING_ROOT_AUTHORIZATION":
        raise ValueError("unexpected receipt status")
    if sha256(STRICT) != EXPECTED_STRICT_SHA or sha256(HISTORICAL) != EXPECTED_HISTORICAL_SHA:
        raise ValueError("authoritative input SHA drift")
    strict = read_json(STRICT)
    raw_rows, seed_means, session_means = strict_rows(strict)
    observed_seed_sd = finite_sample_sd(list(seed_means.values()))
    observed_session_sd = finite_sample_sd(list(session_means.values()))
    bootstrap = bootstrap_interval(raw_rows)
    mde = mde_rows(observed_seed_sd, observed_session_sd)
    epochs = epoch_availability()
    out.mkdir(parents=True, exist_ok=False)
    raw = {
        "schema_version": 1,
        "receipt": {"path": str(RECEIPT.resolve()), "sha256": sha256(RECEIPT)},
        "strict_input": {"path": str(STRICT.resolve()), "sha256": sha256(STRICT)},
        "historical_context_input": {"path": str(HISTORICAL.resolve()), "sha256": sha256(HISTORICAL)},
        "rows": raw_rows,
        "per_seed_equal_session_means": seed_means,
        "per_session_seed_averaged_means": session_means,
    }
    (out / "evidence_A_raw_clustered_deltas.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "evidence_B_epoch_availability_audit.json").write_text(json.dumps(epochs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out / "evidence_A_mde_sensitivity.csv", mde)
    aggregate = {
        "schema_version": 1,
        "purpose": "0a_uncertainty_identifiability_audit",
        "receipt": {"path": str(RECEIPT.resolve()), "sha256": sha256(RECEIPT)},
        "evidence_A_strict_future_query": {
            "comparator": "T4_minus_clean_SPINT_only",
            "design": "fold1 seeds 42/43/44; six local held-out session clusters; M24 query_start=24",
            "per_seed_equal_session_means": seed_means,
            "per_session_seed_averaged_means": session_means,
            "descriptive_within_fold_seed_sd": observed_seed_sd,
            "descriptive_session_cluster_sd_of_seed_averages": observed_session_sd,
            "bootstrap": bootstrap,
            "SESOI": {"candidate_r2": 0.03, "role": "deployment relevance candidate, separate from noise"},
            "MDE_artifact": str((out / "evidence_A_mde_sensitivity.csv").resolve()),
        },
        "evidence_B_historical_context": {
            "design": "f1s42/f1s43/f2s42 incomplete factorial",
            "epoch_availability_artifact": str((out / "evidence_B_epoch_availability_audit.json").resolve()),
            "not_used_for": ["strict endpoint MDE", "sigma_seed", "sigma_fold", "general M2 noise floor"],
        },
        "non_identifiable": receipt["evidence_B_historical_native_training_context"]["non_identifiable"],
        "explicitly_not_claimed": ["unified M2 noise floor", "T4-F0 or K4 MDE", "18 iid observation inference", "M33 future-query uncertainty"],
    }
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_v2_correction(out: Path) -> None:
    """Write only new correction artifacts; preserve analysis_v1 provenance bytewise."""
    old = out / "aggregate.json"
    raw_path = out / "evidence_A_raw_clustered_deltas.json"
    if not old.is_file() or not raw_path.is_file():
        raise FileNotFoundError("correction requires the immutable analysis_v1 aggregate and raw rows")
    old_sha = sha256(old)
    if old_sha != "8083f59dc7f7ee2f25b1c329183b1eba0383fc4f4168f6fe1c11c6da38c4d623":
        raise ValueError("analysis_v1 aggregate SHA drift; refusing correction")
    mde_path = out / "evidence_A_mde_sensitivity_v2.csv"
    new_path = out / "aggregate_v2.json"
    if mde_path.exists() or new_path.exists():
        raise FileExistsError("refusing to overwrite an existing correction artifact")
    raw = read_json(raw_path)
    seed_means = {str(k): float(v) for k, v in raw["per_seed_equal_session_means"].items()}
    session_means = {str(k): float(v) for k, v in raw["per_session_seed_averaged_means"].items()}
    seed_sd = finite_sample_sd(list(seed_means.values()))
    session_sd = finite_sample_sd(list(session_means.values()))
    write_csv(mde_path, mde_rows(seed_sd, session_sd))
    old_record = read_json(old)
    new_record = dict(old_record)
    new_record.pop("non_identifiable", None)
    new_record["schema_version"] = 2
    new_record["supersedes"] = {"path": str(old.resolve()), "sha256": old_sha, "reason": "Evidence-A and Evidence-B non-identifiability claims were separated to avoid implying that a conditional Evidence-A MDE sensitivity is forbidden."}
    new_record["evidence_A_strict_future_query"]["MDE_artifact"] = str(mde_path.resolve())
    new_record["non_identifiable_by_evidence"] = {
        "A_strict_future_query_T4_minus_clean_SPINT": [
            "no unique population MDE", "no fold-to-fold variance", "no general M2 noise floor", "no T4-F0 or K4 MDE transfer"
        ],
        "B_historical_native_internal_f1s42_f1s43_f2s42": [
            "sigma_seed", "sigma_fold", "sigma_seed_by_fold_interaction", "independent run-to-run sigma inferred from epochs", "any MDE for the strict T4-minus-clean-SPINT future-query endpoint"
        ]
    }
    (new_path).write_text(json.dumps(new_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS / "analysis_v1")
    parser.add_argument("--correct-existing-analysis-v1", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.correct_existing_analysis_v1:
        write_v2_correction(out)
    else:
        run(out)


if __name__ == "__main__":
    main()
