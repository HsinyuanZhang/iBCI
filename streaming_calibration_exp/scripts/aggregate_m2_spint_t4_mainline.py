"""Fail-closed aggregate for the matched M2 SPINT/T4 FP32 mainline."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import wilcoxon


SEEDS = (42, 43, 44)
ARMS = ("t4", "ts4")
EXPECTED_SESSIONS = (
    "ses-2020-10-30-Run1",
    "ses-2020-10-30-Run2",
    "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1",
    "ses-2020-11-24-Run1",
    "ses-2020-11-24-Run2",
)


def load_baseline(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {
        row["session"]: float(row["R2_variance_weighted"])
        for row in rows
        if row["split"] == "test_heldout"
    }
    if tuple(sorted(observed)) != tuple(sorted(EXPECTED_SESSIONS)):
        raise ValueError(f"SPINT baseline session set mismatch: {sorted(observed)}")
    if any(int(row["M"]) != 33 for row in rows if row["split"] == "test_heldout"):
        raise ValueError("SPINT baseline must use exactly 33 calibration trials")
    return observed


def resolve_one_run(root: Path, screen_id: str, arm: str, seed: int) -> Path:
    hits = sorted(root.glob(f"{screen_id}_{arm}_m2_s{seed}_*"))
    if len(hits) != 1:
        raise ValueError(f"expected one {arm}/seed{seed} artifact, found {hits}")
    return hits[0]


def load_arm_run(run_dir: Path, arm: str, seed: int) -> dict[str, float]:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    data = cfg["data"]
    expected_side = {"t4": "t4", "ts4": "ts4"}[arm]
    checks = {
        "seed": cfg["seed"] == seed,
        "variant": cfg["model"]["variant"] == "B3S",
        "task": data["task"] == "m2",
        "validation_protocol": data["validation_protocol"] == "minival",
        "loso_fold": data["loso_fold"] is None,
        "calibration_n_trials": data["calibration_n_trials"] == 33,
        "random_calibration": data["random_calibration"] is False,
        "include_heldout_in_fit": data["include_heldout_in_fit"] is False,
        "include_heldout_in_test": data["include_heldout_in_test"] is True,
        "side_feature_group": data["side_feature_group"] == expected_side,
        "fixed_epoch_budget": cfg["no_early_stopping"] is True,
        "max_epochs": cfg["trainer"]["max_epochs"] == 12,
        "freeze_decoder": cfg["model"]["freeze_decoder"] is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{run_dir}: protocol mismatch in {failed}")

    with (run_dir / "metrics_per_session.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["split"] == "test_heldout"
        ]
    observed = {
        row["session"]: float(row["R2_variance_weighted"]) for row in rows
    }
    if tuple(sorted(observed)) != tuple(sorted(EXPECTED_SESSIONS)):
        raise ValueError(f"{run_dir}: held-out session set mismatch")
    if any(int(row["M"]) != 33 for row in rows):
        raise ValueError(f"{run_dir}: metrics do not record M=33")
    return observed


def hierarchical_ci(delta: np.ndarray, draws: int = 50_000) -> list[float]:
    rng = np.random.default_rng(20260730)
    n_seed, n_session = delta.shape
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, n_seed, n_seed)
        sessions = rng.integers(0, n_session, n_session)
        sampled[index] = delta[np.ix_(seeds, sessions)].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def summarize_contrast(
    treatment: np.ndarray, control: np.ndarray, name: str
) -> dict:
    delta = treatment - control
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    wilcoxon_result = wilcoxon(
        session_means,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
    )
    ci = hierarchical_ci(delta)
    gates = {
        "mean_delta_at_least_0p03": float(delta.mean()) >= 0.03,
        "all_three_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "all_six_session_means_positive": bool(np.all(session_means > 0.0)),
        "hierarchical_bootstrap_95ci_lower_positive": ci[0] > 0.0,
        "session_paired_exact_wilcoxon_two_sided_le_0p05": (
            float(wilcoxon_result.pvalue) <= 0.05
        ),
    }
    return {
        "name": name,
        "mean_paired_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(SEEDS, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value)
            for session, value in zip(EXPECTED_SESSIONS, session_means)
        },
        "positive_seed_count": int((seed_means > 0.0).sum()),
        "positive_session_count": int((session_means > 0.0).sum()),
        "hierarchical_bootstrap_95ci": ci,
        "session_paired_exact_wilcoxon_two_sided_p": float(
            wilcoxon_result.pvalue
        ),
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen_id", default="m2_spint_t4_mainline_fp32_v1")
    parser.add_argument(
        "--artifact_root",
        type=Path,
        default=Path("outputs/streaming_calibration"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/streaming_calibration/b0_baseline/metrics_per_session.csv"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    baseline_matrix = np.asarray(
        [[baseline[session] for session in EXPECTED_SESSIONS]] * len(SEEDS),
        dtype=np.float64,
    )
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {}
    values: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        rows = []
        artifacts[arm] = {}
        values[arm] = {}
        for seed in SEEDS:
            run_dir = resolve_one_run(
                args.artifact_root, args.screen_id, arm, seed
            )
            session_r2 = load_arm_run(run_dir, arm, seed)
            artifacts[arm][str(seed)] = str(run_dir.resolve())
            values[arm][str(seed)] = session_r2
            rows.append([session_r2[session] for session in EXPECTED_SESSIONS])
        matrices[arm] = np.asarray(rows, dtype=np.float64)

    t4_vs_spint = summarize_contrast(
        matrices["t4"], baseline_matrix, "T4_minus_local_SPINT"
    )
    t4_vs_ts4 = summarize_contrast(
        matrices["t4"], matrices["ts4"], "T4_minus_TS4"
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "screen_id": args.screen_id,
        "claim_scope": (
            "local FALCON M2 held-out-calibration replay; not hidden EvalAI query/test"
        ),
        "protocol": {
            "training_sessions": "all seven M2 held-in sessions",
            "calibration_selection": "chronological first 33 trials",
            "same_trial_count_and_prefix_for_all_arms": True,
            "spint_uses_calibration_spikes": True,
            "t4_uses_target_labels_from_same_calibration_trials": True,
            "heldout_calibration_backward_gradients": False,
            "training_epochs": 12,
            "seeds": list(SEEDS),
            "sessions": list(EXPECTED_SESSIONS),
        },
        "baseline_path": str(args.baseline.resolve()),
        "baseline_per_session_r2": baseline,
        "artifacts": artifacts,
        "per_seed_per_session_r2": values,
        "arm_mean_r2": {
            "spint": float(baseline_matrix.mean()),
            "t4": float(matrices["t4"].mean()),
            "ts4": float(matrices["ts4"].mean()),
        },
        "contrasts": {
            "t4_vs_spint": t4_vs_spint,
            "t4_vs_ts4": t4_vs_ts4,
        },
        "main_claim_pass": t4_vs_spint["passes_all_gates"],
        "label_information_control_pass": t4_vs_ts4["passes_all_gates"],
        "inference_warning": (
            "The held-out sessions were inspected in earlier internal-LOSO work, so this "
            "is strong local replication evidence, not a prospectively untouched test."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["arm_mean_r2"], indent=2))
    print(json.dumps(payload["contrasts"], indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
