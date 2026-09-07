#!/usr/bin/env python3
"""CPU-only, chronological K4/Gate-A split-half reliability curve.

This is deliberately a *reliability* audit, not a decoder experiment.  It
reuses Gate A's raw M2 block construction and OLS encoding primitive exactly:
100-ms non-overlapping raw neural blocks, +40-ms velocity lead, strict active
mask, no cross-trial blocks, and alpha=0.  It opens only M2 held-in-calib NWBs.

For every chronological prefix M, trials [0:floor(M/2)) and
[floor(M/2):M) are independently fit.  The existing Gate-A stability number
is flattened-W *Pearson correlation*.  This script reports that primary
metric, plus flattened-W cosine, rather than silently treating them as the
same quantity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
SCE_ROOT = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SCE_ROOT))

from falcon_challenge.config import FalconTask  # noqa: E402
from falcon_challenge.dataloaders import load_nwb  # noqa: E402
from mc_maze.general_carrier import fit_encoding  # noqa: E402
from src.data.falcon_t4_features import calibration_target_angles  # noqa: E402

TASK = "m2"
M_GRID = (8, 10, 12, 16, 20, 24, 28, 30, 32, 33)
N_PERMUTATIONS = 100
PERMUTATION_SEEDS = tuple(range(N_PERMUTATIONS))
BOOTSTRAP_SEED = 20260801
N_BOOTSTRAP = 10_000
RELIABILITY_THRESHOLD = 0.5
RAW_BIN_MS = 20
BLOCK_WIDTH_BINS = 5
BEHAVIOR_LEAD_BINS = 2
ACTIVE_EPSILON = 0.001
ENCODING_ALPHA = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def session_name(path: Path) -> str:
    return path.name.split("_", 1)[1].split(".nwb", 1)[0]


def heldin_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("**/*held-in-calib*.nwb"))
    if len(paths) != 7:
        raise ValueError(f"expected exactly seven M2 held-in-calib files, found {len(paths)}")
    if any("held-out" in str(path).lower() for path in paths):
        raise ValueError("held-out path reached a held-in reliability audit")
    return paths


def gate_a_trial_blocks(
    neural: np.ndarray, covariates: np.ndarray, trial_change: np.ndarray, *, max_trials: int
) -> dict[str, np.ndarray]:
    """Gate-A's fixed raw block definition, parameterized only by prefix size."""
    neural = np.asarray(neural, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    change = np.asarray(trial_change, dtype=bool)
    if neural.ndim != 2 or covariates.ndim != 2 or neural.shape[0] != covariates.shape[0]:
        raise ValueError("M2 neural/covariate arrays must be time-by-feature with shared time")
    if covariates.shape[1] != 2:
        raise ValueError(f"M2 carrier is frozen for 2-D finger velocity, got {covariates.shape}")
    starts = np.flatnonzero(change)
    if len(starts) < max_trials:
        raise ValueError(f"session has {len(starts)} trials, needs {max_trials}")
    ends = np.r_[starts[1:], len(change)]
    active = ~np.all(np.abs(covariates) < ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    behavior: list[np.ndarray] = []
    trial_ids: list[int] = []
    for trial_id, (start, end) in enumerate(zip(starts[:max_trials], ends[:max_trials])):
        for left in range(
            int(start), int(end) - BLOCK_WIDTH_BINS - BEHAVIOR_LEAD_BINS + 1, BLOCK_WIDTH_BINS
        ):
            right = left + BLOCK_WIDTH_BINS
            y_left = left + BEHAVIOR_LEAD_BINS
            y_right = y_left + BLOCK_WIDTH_BINS
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(neural[left:right].sum(axis=0) / (BLOCK_WIDTH_BINS * RAW_BIN_MS / 1000.0))
            behavior.append(covariates[y_left:y_right].mean(axis=0))
            trial_ids.append(trial_id)
    if len(rates) < 10:
        raise ValueError("fewer than ten active movement-aligned raw blocks")
    return {
        "rate": np.asarray(rates, dtype=np.float64),
        "behavior": np.asarray(behavior, dtype=np.float64),
        "trial_id": np.asarray(trial_ids, dtype=np.int64),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.ravel(), right.ravel()) / denom) if denom > 1e-12 else float("nan")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left, right = left.ravel(), right.ravel()
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def reliability_metrics(weights_a: np.ndarray, weights_b: np.ndarray) -> dict[str, float]:
    """Metrics on Gate-A W=[channel, 2], including explicitly distinct correlations."""
    if weights_a.shape != weights_b.shape or weights_a.ndim != 2 or weights_a.shape[1] != 2:
        raise ValueError("expected matched [channel, 2] carrier weights")
    row_denom = np.linalg.norm(weights_a, axis=1) * np.linalg.norm(weights_b, axis=1)
    row_cos = np.sum(weights_a * weights_b, axis=1) / np.maximum(row_denom, 1e-12)
    # A direction is intrinsically unidentified for a nearly zero modulation
    # vector.  These are sensitivity metrics only: Gate-A's threshold remains
    # the session-level flattened W Pearson correlation.
    min_magnitude = np.minimum(np.linalg.norm(weights_a, axis=1), np.linalg.norm(weights_b, axis=1))
    q25 = float(np.quantile(min_magnitude, 0.25))
    retained = min_magnitude >= q25
    return {
        "flattened_w_pearson": _pearson(weights_a, weights_b),
        "flattened_w_cosine": _cosine(weights_a, weights_b),
        "per_electrode_2d_cosine_median": float(np.median(row_cos)),
        "per_electrode_2d_cosine_mean": float(np.mean(row_cos)),
        "magnitude_weighted_2d_cosine": float(np.sum(weights_a * weights_b) / np.maximum(np.sum(row_denom), 1e-12)),
        "min_half_modulation_norm_q25": q25,
        "per_electrode_2d_cosine_median_excluding_lowest_magnitude_quartile": float(np.median(row_cos[retained])),
        "n_electrodes_excluding_lowest_magnitude_quartile": int(retained.sum()),
        "n_electrodes": int(weights_a.shape[0]),
    }


def _fit_half(blocks: dict[str, np.ndarray], lo_trial: int, hi_trial: int):
    mask = (blocks["trial_id"] >= lo_trial) & (blocks["trial_id"] < hi_trial)
    if int(mask.sum()) < 10:
        raise ValueError(f"half [{lo_trial}:{hi_trial}) has fewer than 10 active blocks")
    behavior = blocks["behavior"][mask]
    design = np.column_stack([np.ones(len(behavior), dtype=float), behavior])
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else float("inf")
    eig = np.linalg.eigvalsh(np.cov(behavior, rowvar=False))[::-1]
    audit = {
        "active_blocks": int(mask.sum()), "design_rank": rank, "design_condition": condition,
        "velocity_covariance_eigenvalues": [float(value) for value in eig],
        "velocity_balance_min_over_max": float(eig[-1] / max(eig[0], 1e-12)),
    }
    fit = fit_encoding(
        blocks["rate"], blocks["behavior"], mask, blocks["trial_id"], lag_bins=0, alpha=ENCODING_ALPHA
    )
    return fit, audit


def evaluate_session_m(blocks: dict[str, np.ndarray], m: int) -> dict[str, Any]:
    split = m // 2
    fit_a, audit_a = _fit_half(blocks, 0, split)
    fit_b, audit_b = _fit_half(blocks, split, m)
    observed = reliability_metrics(fit_a.weights, fit_b.weights)
    null_pearson, null_cosine = [], []
    for seed in PERMUTATION_SEEDS:
        order = np.random.RandomState(seed).permutation(fit_b.weights.shape[0])
        null = reliability_metrics(fit_a.weights, fit_b.weights[order])
        null_pearson.append(null["flattened_w_pearson"])
        null_cosine.append(null["flattened_w_cosine"])
    observed.update(
        {
            "m": int(m),
            "first_half_trials": int(split),
            "second_half_trials": int(m - split),
            "first_half_active_blocks": audit_a["active_blocks"],
            "second_half_active_blocks": audit_b["active_blocks"],
            "first_half_design_rank": audit_a["design_rank"],
            "second_half_design_rank": audit_b["design_rank"],
            "first_half_design_condition": audit_a["design_condition"],
            "second_half_design_condition": audit_b["design_condition"],
            "first_half_velocity_covariance_eigenvalues": audit_a["velocity_covariance_eigenvalues"],
            "second_half_velocity_covariance_eigenvalues": audit_b["velocity_covariance_eigenvalues"],
            "first_half_velocity_balance_min_over_max": audit_a["velocity_balance_min_over_max"],
            "second_half_velocity_balance_min_over_max": audit_b["velocity_balance_min_over_max"],
            "channel_permutation_null_n": N_PERMUTATIONS,
            "channel_permutation_null_pearson_median": float(np.median(null_pearson)),
            "channel_permutation_null_cosine_median": float(np.median(null_cosine)),
            "channel_permutation_observed_pearson_beats": int(
                np.sum(observed["flattened_w_pearson"] > np.asarray(null_pearson))
            ),
            "channel_permutation_observed_cosine_beats": int(
                np.sum(observed["flattened_w_cosine"] > np.asarray(null_cosine))
            ),
        }
    )
    return observed


def bootstrap_summary(values: np.ndarray, *, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    sample = values[rng.randint(0, len(values), size=(N_BOOTSTRAP, len(values)))]
    means = sample.mean(axis=1)
    medians = np.median(sample, axis=1)
    return {
        "n_sessions": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "bootstrap_mean_ci95_low": float(np.quantile(means, 0.025)),
        "bootstrap_mean_ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_median_ci95_low": float(np.quantile(medians, 0.025)),
        "bootstrap_median_ci95_high": float(np.quantile(medians, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": N_BOOTSTRAP,
        "interpretation": "descriptive resampling across the seven fixed development sessions, not an independent-population confidence interval",
    }


def write_protocol(out_dir: Path, data_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=False)
    paths = heldin_paths(data_dir)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_curve_results",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "0-GPU M2 held-in chronological K4/Gate-A W split-half reliability curve",
        "forbidden": ["held-out calibration/query files", "EvalAI", "decoder fitting", "decoding R2", "GPU", "hyperparameter selection"],
        "inputs": [
            {"session": session_name(path), "path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
        "source_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "gate_a_reference": {
            "path": str((SUA_ROOT / "results/general_carrier_proxy_v1/audit_m2_heldin_v2.json").resolve()),
            "sha256": sha256_file(SUA_ROOT / "results/general_carrier_proxy_v1/audit_m2_heldin_v2.json"),
            "existing_stability_metric": "flattened-W Pearson correlation (not cosine)",
            "reference_split": "first 17 vs next 16 chronological trials of first 33",
        },
        "frozen_protocol": {
            "task": TASK, "m_grid_requested": list(M_GRID), "chronological_prefix": True,
            "split": "trials[0:floor(M/2)] vs trials[floor(M/2):M]",
            "raw_bin_ms": RAW_BIN_MS, "nonoverlap_block_bins": BLOCK_WIDTH_BINS,
            "behavior_lead_raw_bins": BEHAVIOR_LEAD_BINS,
            "active_rule": "every neural and shifted-velocity sample satisfies ~all(abs(v)<0.001)",
            "fit": "Gate-A fit_encoding, alpha=0, lag_bins=0 after fixed +40-ms raw alignment",
            "primary_metric": "flattened-W Pearson correlation", "also_report": ["flattened-W cosine", "per-electrode 2D cosine median", "magnitude-weighted 2D cosine"],
            "per_electrode_direction_sensitivity": "also report median after excluding lowest quartile of min(||W_A||,||W_B||); never use it for the Gate-A threshold",
            "half_direction_coverage_audit": "record 2-D design rank, condition, covariance eigenvalues, and min/max eigenvalue balance for each chronological half",
            "channel_permutation_control": {"n": N_PERMUTATIONS, "seeds": list(PERMUTATION_SEEDS)},
            "session_bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED},
            "descriptive_operating_point": RELIABILITY_THRESHOLD,
            "m_min_all": "first M where every session has Pearson AND cosine >=0.5",
            "m_min_6of7": "first M where at least six sessions have Pearson AND cosine >=0.5",
            "deployment_feasibility_boundary": "M<=24",
        },
        "git_status_before_run": git_status(),
    }
    receipt = out_dir / "protocol_receipt.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def threshold_minimum(by_m: dict[int, list[dict[str, Any]]], required: int) -> int | None:
    for m, rows in sorted(by_m.items()):
        n = sum(
            row["flattened_w_pearson"] >= RELIABILITY_THRESHOLD
            and row["flattened_w_cosine"] >= RELIABILITY_THRESHOLD
            for row in rows
        )
        if n >= required:
            return m
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["session", *sorted({key for row in rows for key in row if key != "session"})]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = sorted({row["session"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for metric, ax, label in (
        ("flattened_w_pearson", axes[0], "Flattened-W Pearson correlation (Gate-A metric)"),
        ("flattened_w_cosine", axes[1], "Flattened-W cosine"),
    ):
        for session in sessions:
            selected = sorted((row for row in rows if row["session"] == session), key=lambda item: item["m"])
            ax.plot([row["m"] for row in selected], [row[metric] for row in selected], marker="o", linewidth=1.4, label=session.replace("sub-MonkeyN-held-in-calib_", ""))
        ax.axhline(RELIABILITY_THRESHOLD, color="black", linestyle="--", linewidth=1, label="operating point = 0.5")
        ax.axvline(24, color="tab:red", linestyle=":", linewidth=1.5, label="few-shot boundary M=24")
        ax.set_xlabel("Chronological calibration trials M")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("M2 held-in K4/Gate-A W split-half reliability (0 GPU; no query decoding)")
    fig.tight_layout()
    fig.savefig(out_dir / "split_half_reliability_curve.png", dpi=220)
    fig.savefig(out_dir / "split_half_reliability_curve.pdf")
    plt.close(fig)


def _handoff(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    all_min, six_min = summary["m_min_all"], summary["m_min_6of7"]
    status = (
        "passes the descriptive reliability operating point within the M<=24 deployment budget"
        if all_min is not None and all_min <= 24
        else "does not establish all-session reliability within the M<=24 deployment budget"
    )
    return f"""# M2 K4 split-half reliability curve — handoff

Scope: seven M2 **held-in calibration** sessions only; CPU-only; no held-out,
EvalAI, query labels, decoder, or R² read.  The curve uses Gate A's same raw
block construction and OLS fit.  Gate A's original stability metric is
flattened-W **Pearson correlation**; cosine is reported separately.

- `M_min_all` (all 7 sessions, both Pearson and cosine ≥0.5): `{all_min}`
- `M_min_6of7` (at least 6 sessions, both metrics ≥0.5): `{six_min}`
- Deployment boundary: `M<=24`; result: **{status}**.

This is an estimation-reliability diagnostic, not a causal explanation of the
previous held-out decoding failure.  In particular, if a stable operating point
exists at or below 24 while held-out decoding remains negative, W stability is
not a sufficient explanation; if it appears only above 24, the carrier conflicts
with this dataset's few-shot/query budget rather than proving the carrier absent.

Artifacts: `raw_curve.json`, `raw_curve.csv`, `split_half_reliability_curve.png`,
`split_half_reliability_curve.pdf`, and the hash-bound `protocol_receipt.json`.
"""


def run(out_dir: Path, data_dir: Path, expected_protocol_sha: str) -> Path:
    receipt = out_dir / "protocol_receipt.json"
    if not receipt.exists():
        raise FileNotFoundError("write the frozen protocol receipt before running the curve")
    actual = sha256_file(receipt)
    if actual != expected_protocol_sha:
        raise ValueError(f"protocol SHA mismatch: expected {expected_protocol_sha}, got {actual}")
    protocol = json.loads(receipt.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_curve_results":
        raise ValueError("protocol receipt is not a pre-result freeze")
    paths = heldin_paths(data_dir)
    expected_inputs = {item["path"]: item["sha256"] for item in protocol["inputs"]}
    for path in paths:
        if expected_inputs.get(str(path.resolve())) != sha256_file(path):
            raise ValueError(f"input SHA drift: {path}")

    records: list[dict[str, Any]] = []
    for path in paths:
        neural, velocity, trial_change, _ = load_nwb(path, FalconTask.m2)
        # This checks the original M2 calibration trial alignment without using
        # angles in a fit or result; it is a Gate-A input validation invariant.
        angles = calibration_target_angles(path, TASK)
        n_trials = int(np.sum(np.asarray(trial_change, dtype=bool)))
        if len(angles) != n_trials:
            raise ValueError(f"trial-label alignment mismatch in {path.name}")
        blocks = gate_a_trial_blocks(neural, velocity, trial_change, max_trials=max(M_GRID))
        for m in M_GRID:
            row = evaluate_session_m(blocks, m)
            row["session"] = session_name(path)
            row["source_path"] = str(path.resolve())
            row["source_sha256"] = sha256_file(path)
            row["available_trials"] = n_trials
            records.append(row)

    by_m: dict[int, list[dict[str, Any]]] = {m: [row for row in records if row["m"] == m] for m in M_GRID}
    summaries = {}
    for m, rows in by_m.items():
        pearson = np.asarray([row["flattened_w_pearson"] for row in rows])
        cosine = np.asarray([row["flattened_w_cosine"] for row in rows])
        summaries[str(m)] = {
            "flattened_w_pearson": bootstrap_summary(pearson, seed=BOOTSTRAP_SEED + m),
            "flattened_w_cosine": bootstrap_summary(cosine, seed=BOOTSTRAP_SEED + 1000 + m),
            "n_sessions_passing_both_0_5": int(sum((pearson >= RELIABILITY_THRESHOLD) & (cosine >= RELIABILITY_THRESHOLD))),
            "all_sessions_passing_both_0_5": bool(np.all((pearson >= RELIABILITY_THRESHOLD) & (cosine >= RELIABILITY_THRESHOLD))),
        }
    m_all = threshold_minimum(by_m, required=7)
    m_6of7 = threshold_minimum(by_m, required=6)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "M2 held-in calibration only; 0-GPU reliability audit; no decoder/query/R2",
        "protocol_receipt": {"path": str(receipt.resolve()), "sha256": actual},
        "records": records,
        "summary": {
            "m_grid": list(M_GRID), "threshold": RELIABILITY_THRESHOLD,
            "m_min_all": m_all, "m_min_6of7": m_6of7,
            "m_le_24_all_session_reliability": bool(m_all is not None and m_all <= 24),
            "m_le_24_six_of_seven_reliability": bool(m_6of7 is not None and m_6of7 <= 24),
            "per_m": summaries,
            "interpretation_guardrail": "Reliability curve does not establish that W instability caused held-out decoding failure.",
        },
    }
    json_path = out_dir / "raw_curve.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_dir / "raw_curve.csv", records)
    _plot(out_dir, records)
    (out_dir / "HANDOFF_K4_SPLIT_HALF_RELIABILITY.md").write_text(_handoff(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000953")
    parser.add_argument("--out-dir", type=Path, default=SUA_ROOT / "results/m2_k4_split_half_curve_v1")
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--protocol-sha", type=str)
    args = parser.parse_args()
    if args.write_protocol == args.run:
        raise ValueError("select exactly one of --write-protocol or --run")
    out_dir, data_dir = args.out_dir.resolve(), args.data_dir.resolve()
    if args.write_protocol:
        receipt = write_protocol(out_dir, data_dir)
        print(f"{receipt}\nsha256={sha256_file(receipt)}")
        return
    if not args.protocol_sha:
        raise ValueError("--run requires --protocol-sha")
    run(out_dir, data_dir, args.protocol_sha)


if __name__ == "__main__":
    main()
