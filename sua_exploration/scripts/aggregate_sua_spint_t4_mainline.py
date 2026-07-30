"""Fail-closed aggregate for matched SUA B0/T4/TS4 FP32 validation runs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


SEEDS = (42, 43, 44)
ARMS = ("b0", "t4", "ts4")
EXPECTED = {
    "b0": ("B0", "none"),
    "t4": ("B3S", "t4"),
    "ts4": ("B3S", "ts4"),
}


def load_artifact(path: Path, arm: str, seed: int) -> tuple[list[str], np.ndarray]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    metadata_path = Path(artifact["run_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_variant, expected_side = EXPECTED[arm]
    protocol = artifact["protocol"]
    checks = {
        "artifact_seed": artifact["seed"] == seed,
        "artifact_variant": artifact["variant"] == expected_variant,
        "metadata_seed": metadata["seed"] == seed,
        "metadata_variant": metadata["variant"] == expected_variant,
        "side_group": metadata["side_features"]["group"] == expected_side,
        "side_pool_size": metadata["side_features"]["pool_size"] == 30,
        "training_calibration_n": metadata["training"]["calibration_n_trials"] == 30,
        "fixed_epochs": metadata["training"]["max_epochs"] == 12,
        "no_early_stopping": metadata["training"]["no_early_stopping"] is True,
        "checkpoint_every_epoch": metadata["training"]["checkpoint_every_epoch"] is True,
        "run_completed": metadata["status"] == "completed",
        "no_formal_test": artifact["no_test_files_evaluated"] is True,
        "held_out_test_evaluated": metadata["held_out_test_evaluated"] is False,
        "selection_mode": protocol["selection_mode"] == "first",
        "evaluation_calibration_n": protocol["calibration_n"] == 30,
        "evaluation_pool_size": protocol["pool_size"] == 30,
        "train_eval_calibration_match": (
            protocol["train_activity_calibration_n"]
            == protocol["evaluation_forward_calibration_n"]
            == 30
        ),
        "no_eval_backward": artifact["uses_backward_gradients"] is False,
        "no_eval_weight_update": artifact["uses_behavior_labels_for_weight_updates"] is False,
        "label_provenance": artifact["calibration_features_use_behavior_labels"]
        is (arm != "b0"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path}: protocol mismatch in {failed}")

    epochs = [str(epoch) for epoch in artifact["epoch_list"]]
    session_sets = [
        set(artifact["per_epoch"][epoch]["per_session_r2"]) for epoch in epochs
    ]
    if not session_sets or any(observed != session_sets[0] for observed in session_sets):
        raise ValueError(f"{path}: per-epoch validation session drift")
    sessions = sorted(session_sets[0])
    if len(sessions) != 6:
        raise ValueError(f"{path}: expected 6 validation sessions, got {sessions}")
    values = np.asarray(
        [
            [
                artifact["per_epoch"][epoch]["per_session_r2"][session]
                for session in sessions
            ]
            for epoch in epochs
        ],
        dtype=np.float64,
    ).mean(axis=0)
    return sessions, values


def hierarchical_ci(delta: np.ndarray, draws: int = 50_000) -> list[float]:
    rng = np.random.default_rng(20260730)
    n_seed, n_session = delta.shape
    sampled = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, n_seed, n_seed)
        sessions = rng.integers(0, n_session, n_session)
        sampled[index] = delta[np.ix_(seeds, sessions)].mean()
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def summarize(treatment: np.ndarray, control: np.ndarray, sessions: list[str]) -> dict:
    delta = treatment - control
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    test = wilcoxon(
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
        "session_paired_exact_wilcoxon_two_sided_le_0p05": float(test.pvalue) <= 0.05,
    }
    return {
        "mean_paired_delta_r2": float(delta.mean()),
        "per_seed_mean_delta_r2": {
            str(seed): float(value) for seed, value in zip(SEEDS, seed_means)
        },
        "per_session_mean_delta_r2": {
            session: float(value) for session, value in zip(sessions, session_means)
        },
        "positive_seed_count": int((seed_means > 0.0).sum()),
        "positive_session_count": int((session_means > 0.0).sum()),
        "hierarchical_bootstrap_95ci": ci,
        "session_paired_exact_wilcoxon_two_sided_p": float(test.pvalue),
        "gates": gates,
        "passes_all_gates": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=Path("sua_exploration/results/sua_spint_t4_mainline_fp32_v1"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.result_dir / "aggregate.json"

    matrices: dict[str, np.ndarray] = {}
    artifact_paths: dict[str, dict[str, str]] = {}
    session_names: list[str] | None = None
    for arm in ARMS:
        rows = []
        artifact_paths[arm] = {}
        for seed in SEEDS:
            path = args.result_dir / f"{arm}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values = load_artifact(path, arm, seed)
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: session split differs from paired matrix")
            rows.append(values)
            artifact_paths[arm][str(seed)] = str(path.resolve())
        matrices[arm] = np.asarray(rows, dtype=np.float64)
    assert session_names is not None

    t4_vs_b0 = summarize(matrices["t4"], matrices["b0"], session_names)
    t4_vs_ts4 = summarize(matrices["t4"], matrices["ts4"], session_names)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "claim_scope": "DANDI 000688 sub-C CO six-session validation; formal test unopened",
        "protocol": {
            "split": "strict chronological 27 train / 6 validation / 6 sealed test receipts",
            "training_calibration": "chronological first 30 trials",
            "evaluation_calibration": "same chronological first 30 trials",
            "evaluation_windows": "trials[30:] only",
            "same_trial_count_and_prefix_for_all_arms": True,
            "t4_uses_behavior_labels_from_same_30_trials": True,
            "evaluation_backward_gradients": False,
            "epochs": 12,
            "scored_epoch_window": list(range(5, 13)),
            "seeds": list(SEEDS),
            "sessions": session_names,
        },
        "artifacts": artifact_paths,
        "arm_mean_r2": {
            arm: float(matrix.mean()) for arm, matrix in matrices.items()
        },
        "contrasts": {
            "t4_vs_original_spint_b0": t4_vs_b0,
            "t4_vs_shuffled_label_ts4": t4_vs_ts4,
        },
        "main_claim_pass": t4_vs_b0["passes_all_gates"],
        "label_information_control_pass": t4_vs_ts4["passes_all_gates"],
        "formal_test_files_opened": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["arm_mean_r2"], indent=2))
    print(json.dumps(payload["contrasts"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
