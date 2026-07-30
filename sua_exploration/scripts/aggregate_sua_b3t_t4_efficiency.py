"""Fail-closed aggregate for fresh T4/B3T+T4/B3T+TS4 SUA runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ARMS = ("t4", "b3t_t4", "b3t_ts4")
EXPECTED = {
    "t4": ("B3S", "t4"),
    "b3t_t4": ("B3TS", "t4"),
    "b3t_ts4": ("B3TS", "ts4"),
}
EXPECTED_COST = {
    "t4": {
        "parameter_count": 18_290,
        "mac_per_session": 13_033_472,
        "support_state_bytes": 16_384,
        "trial_buffer_bytes": 25_600,
        "peak_live_state_bytes": 41_984,
        "supports_bin_streaming": False,
    },
    "b3t_t4": {
        "parameter_count": 12_658,
        "mac_per_session": 4_524_032,
        "support_state_bytes": 16_384,
        "trial_buffer_bytes": 3_072,
        "peak_live_state_bytes": 19_456,
        "supports_bin_streaming": True,
    },
    "b3t_ts4": {
        "parameter_count": 12_658,
        "mac_per_session": 4_524_032,
        "support_state_bytes": 16_384,
        "trial_buffer_bytes": 3_072,
        "peak_live_state_bytes": 19_456,
        "supports_bin_streaming": True,
    },
}
EPOCHS = list(range(5, 13))
EXPECTED_SESSIONS = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    return seeds


def require(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def load_cell(
    path: Path, arm: str, seed: int
) -> tuple[list[str], np.ndarray, dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    variant, side = EXPECTED[arm]
    require(f"{path}: artifact variant", artifact.get("variant"), variant)
    require(f"{path}: artifact seed", artifact.get("seed"), seed)
    require(f"{path}: split", artifact.get("split_counts"), [27, 6, 6])
    require(f"{path}: max units", artifact.get("max_units_exclusive"), 100)
    require(f"{path}: no formal test", artifact.get("no_test_files_evaluated"), True)
    require(f"{path}: eval backward", artifact.get("uses_backward_gradients"), False)
    require(
        f"{path}: eval behavior updates",
        artifact.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    require(
        f"{path}: label-derived calibration feature",
        artifact.get("calibration_features_use_behavior_labels"),
        True,
    )
    require(
        f"{path}: label scope",
        artifact.get("calibration_feature_label_scope"),
        "chronological_rewarded_trials[0:30]",
    )
    protocol = artifact.get("protocol") or {}
    require(f"{path}: calibration n", protocol.get("calibration_n"), 30)
    require(f"{path}: activity calibration n", protocol.get("train_activity_calibration_n"), 30)
    require(f"{path}: labelled calibration n", protocol.get("label_feature_calibration_n"), 30)
    require(f"{path}: common eval start", protocol.get("pool_size"), 30)
    require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)

    metadata_path = Path(artifact.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: metadata path is missing: {metadata_path}")
    require(
        f"{path}: metadata SHA",
        sha256_file(metadata_path),
        artifact.get("run_metadata_sha256"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(f"{path}: metadata variant", metadata.get("variant"), variant)
    require(f"{path}: metadata seed", metadata.get("seed"), seed)
    require(f"{path}: side group", (metadata.get("side_features") or {}).get("group"), side)
    require(f"{path}: side pool", (metadata.get("side_features") or {}).get("pool_size"), 30)
    require(
        f"{path}: activity support",
        (metadata.get("training") or {}).get("calibration_n_trials"),
        30,
    )
    require(f"{path}: fixed epochs", (metadata.get("training") or {}).get("max_epochs"), 12)
    require(
        f"{path}: no early stopping",
        (metadata.get("training") or {}).get("no_early_stopping"),
        True,
    )
    require(f"{path}: completed", metadata.get("status"), "completed")
    require(f"{path}: training formal seal", metadata.get("held_out_test_evaluated"), False)
    require(f"{path}: no warm-start", metadata.get("encoder_warmstart_path"), None)

    cost = metadata.get("encoder_cost_profile_reference") or {}
    require(
        f"{path}: cost reference shape",
        cost.get("reference_shape"),
        {"num_neurons": 64, "trial_length": 100, "num_trials": 30},
    )
    for key, expected in EXPECTED_COST[arm].items():
        require(f"{path}: cost {key}", cost.get(key), expected)

    rows: list[list[float]] = []
    sessions: list[str] | None = None
    for epoch in EPOCHS:
        values = ((artifact.get("per_epoch") or {}).get(str(epoch)) or {}).get(
            "per_session_r2"
        )
        if not isinstance(values, dict) or len(values) != EXPECTED_SESSIONS:
            raise ValueError(f"{path}: epoch {epoch} must have six validation sessions")
        observed = sorted(values)
        if sessions is None:
            sessions = observed
        elif observed != sessions:
            raise ValueError(f"{path}: session names drift at epoch {epoch}")
        rows.append([float(values[name]) for name in observed])
    assert sessions is not None
    return sessions, np.asarray(rows, dtype=np.float64).mean(axis=0), cost


def hierarchical_ci(delta: np.ndarray, draws: int = 50_000) -> list[float]:
    rng = np.random.default_rng(20260731)
    n_seed, n_session = delta.shape
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, n_seed, n_seed)
        sessions = rng.integers(0, n_session, n_session)
        sampled[index] = delta[np.ix_(seeds, sessions)].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def summarize(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    seeds: tuple[int, ...],
    sessions: list[str],
) -> dict:
    delta = treatment - control
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    try:
        p_value = float(
            wilcoxon(
                session_means,
                alternative="two-sided",
                zero_method="wilcox",
                method="exact",
            ).pvalue
        )
    except ValueError:
        p_value = 1.0
    ci = hierarchical_ci(delta)
    stage0_gates = {
        "mean_delta_at_least_0p03": float(delta.mean()) >= 0.03,
        "all_observed_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "all_six_session_means_positive": bool(np.all(session_means > 0.0)),
        "session_paired_exact_wilcoxon_two_sided_le_0p05": p_value <= 0.05,
    }
    gates = {
        **stage0_gates,
        "hierarchical_bootstrap_95ci_lower_positive": ci[0] > 0.0,
        "at_least_three_predeclared_seeds": len(seeds) >= 3,
    }
    return {
        "mean_paired_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(seeds, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value) for session, value in zip(sessions, session_means)
        },
        "positive_seed_count": int((seed_means > 0.0).sum()),
        "positive_session_count": int((session_means > 0.0).sum()),
        "hierarchical_bootstrap_95ci": ci,
        "session_paired_exact_wilcoxon_two_sided_p": p_value,
        "stage0_descriptive_gates": stage0_gates,
        "passes_stage0_descriptive_gates": all(stage0_gates.values()),
        "strict_superiority_gates": gates,
        "passes_strict_superiority": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    result_dir = args.result_dir.expanduser().resolve()

    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    costs: dict[str, dict] = {}
    session_names: list[str] | None = None
    for arm in ARMS:
        rows: list[np.ndarray] = []
        for seed in seeds:
            path = result_dir / f"{arm}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, cost = load_cell(path, arm, seed)
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: paired validation-session set differs")
            if arm in costs and cost != costs[arm]:
                raise ValueError(f"{path}: encoder cost profile drifted across seeds")
            costs[arm] = cost
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
        matrices[arm] = np.asarray(rows, dtype=np.float64)
    assert session_names is not None

    accuracy = summarize(
        matrices["b3t_t4"],
        matrices["t4"],
        seeds=seeds,
        sessions=session_names,
    )
    content = summarize(
        matrices["b3t_t4"],
        matrices["b3t_ts4"],
        seeds=seeds,
        sessions=session_names,
    )
    seed_delta = (matrices["b3t_t4"] - matrices["t4"]).mean(axis=1)
    if len(seeds) >= 2:
        seed_se = float(seed_delta.std(ddof=1) / np.sqrt(len(seed_delta)))
        lower_2se = float(seed_delta.mean() - 2.0 * seed_se)
    else:
        seed_se = None
        lower_2se = None
    parameter_reduction = 1.0 - (
        costs["b3t_t4"]["parameter_count"] / costs["t4"]["parameter_count"]
    )
    mac_reduction = 1.0 - (
        costs["b3t_t4"]["mac_per_session"] / costs["t4"]["mac_per_session"]
    )
    state_change = (
        costs["b3t_t4"]["support_state_bytes"]
        - costs["t4"]["support_state_bytes"]
    )
    efficiency_gates = {
        "at_least_three_predeclared_seeds": len(seeds) >= 3,
        "paired_seed_lower_2se_at_least_minus_0p03": (
            lower_2se is not None and lower_2se >= -0.03
        ),
        "parameter_reduction_at_least_25pct": parameter_reduction >= 0.25,
        "session_mac_reduction_at_least_25pct": mac_reduction >= 0.25,
        "no_support_state_increase": state_change <= 0,
        "t4_content_strict_gate": content["passes_strict_superiority"],
    }
    efficiency_pass = all(efficiency_gates.values())
    stage0_efficiency_gates = {
        "observed_mean_delta_at_least_minus_0p03": float(seed_delta.mean()) >= -0.03,
        "parameter_reduction_at_least_25pct": parameter_reduction >= 0.25,
        "session_mac_reduction_at_least_25pct": mac_reduction >= 0.25,
        "no_support_state_increase": state_change <= 0,
        "t4_content_stage0_gate": content["passes_stage0_descriptive_gates"],
    }
    stage0_efficiency_pass = all(stage0_efficiency_gates.values())
    stage0_candidate = (
        accuracy["passes_stage0_descriptive_gates"] or stage0_efficiency_pass
    )
    effective = accuracy["passes_strict_superiority"] or efficiency_pass

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "protocol": {
            "fresh_training_for_every_arm": True,
            "activity_calibration_n": 30,
            "t4_label_calibration_n": 30,
            "common_evaluation_start": 30,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
        },
        "artifacts": artifacts,
        "arm_mean_r2": {
            arm: float(matrix.mean()) for arm, matrix in matrices.items()
        },
        "cost_profiles": costs,
        "contrasts": {
            "b3t_t4_vs_fresh_t4": accuracy,
            "b3t_t4_vs_b3t_ts4": content,
        },
        "efficiency_noninferiority": {
            "mean_seed_paired_delta_r2": float(seed_delta.mean()),
            "seed_level_standard_error": seed_se,
            "paired_lower_2se": lower_2se,
            "margin": -0.03,
            "parameter_reduction_fraction": float(parameter_reduction),
            "session_mac_reduction_fraction": float(mac_reduction),
            "support_state_change_bytes": int(state_change),
            "gates": efficiency_gates,
            "passes_all_gates": efficiency_pass,
        },
        "stage0_efficiency_screen": {
            "gates": stage0_efficiency_gates,
            "passes_all_gates": stage0_efficiency_pass,
        },
        "stage0_candidate_for_multiseed_expansion": stage0_candidate,
        "effective_by_accuracy_superiority": accuracy["passes_strict_superiority"],
        "effective_by_deployment_efficiency": efficiency_pass,
        "overall_effective": effective,
        "formal_test_files_opened": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "arm_mean_r2": payload["arm_mean_r2"],
                "accuracy_delta": accuracy["mean_paired_delta_r2"],
                "content_delta": content["mean_paired_delta_r2"],
                "efficiency_noninferiority": payload["efficiency_noninferiority"],
                "overall_effective": effective,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
