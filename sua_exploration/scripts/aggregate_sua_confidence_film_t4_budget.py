"""Fail-closed aggregate for the selected-T4 confidence-FiLM budget screen."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc_maze.confidence_film_protocol import T4_BUDGETS, make_protocol


ARMS = (
    "t4_continuation",
    "film",
    "confidence_shuffle",
    "nofilm_match",
    "film_ts4",
)
EXPECTED = {
    "t4_continuation": ("B3S", "t4"),
    "film": ("B3SCF", "t4cf"),
    "confidence_shuffle": ("B3SCFS", "t4cf_confidence_shuffled"),
    "nofilm_match": ("B3SCFA", "t4cf"),
    "film_ts4": ("B3SCF", "t4cf_ts4"),
    "residual_film": ("B3SCFR", "t4cf_residual"),
    "residual_shuffle": ("B3SCFRS", "t4cf_residual_shuffled"),
    "residual_nofilm": ("B3SCFRA", "t4cf_residual"),
}
EXPECTED_FEATURE_VERSION = {
    "t4_continuation": 1,
    "film": 2,
    "confidence_shuffle": 2,
    "nofilm_match": 2,
    "film_ts4": 2,
    "residual_film": 2,
    "residual_shuffle": 2,
    "residual_nofilm": 2,
}
CONTRASTS = {
    "film_vs_t4_continuation": "t4_continuation",
    "film_vs_confidence_shuffle": "confidence_shuffle",
    "film_vs_nofilm_match": "nofilm_match",
    "film_vs_ts4": "film_ts4",
}
REQUIRED_MECHANISM_CONTRASTS = (
    "film_vs_t4_continuation",
    "film_vs_confidence_shuffle",
    "film_vs_nofilm_match",
)
EPOCHS = list(range(5, 13))
EXPECTED_SESSION_COUNT = 6


def _parse_int_csv(value: str, *, allowed: set[int] | None = None) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers: {value}") from exc
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"Expected nonempty unique integers: {value}")
    if allowed is not None and not set(parsed).issubset(allowed):
        raise argparse.ArgumentTypeError(
            f"Values {parsed} must be a subset of {sorted(allowed)}"
        )
    return parsed


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def _per_session_epoch_mean(payload: dict, path: Path) -> tuple[list[str], np.ndarray]:
    per_epoch = payload.get("per_epoch") or {}
    names: list[str] | None = None
    epoch_rows: list[list[float]] = []
    for epoch in EPOCHS:
        scores = (per_epoch.get(str(epoch)) or {}).get("per_session_r2")
        if not isinstance(scores, dict) or len(scores) != EXPECTED_SESSION_COUNT:
            raise ValueError(
                f"{path}: epoch {epoch} must have {EXPECTED_SESSION_COUNT} validation scores"
            )
        observed_names = sorted(scores)
        if names is None:
            names = observed_names
        elif observed_names != names:
            raise ValueError(f"{path}: validation sessions drifted at epoch {epoch}")
        epoch_rows.append([float(scores[name]) for name in observed_names])
    assert names is not None
    return names, np.asarray(epoch_rows, dtype=np.float64).mean(axis=0)


def validate_arm(
    path: Path, arm: str, budget: int, seed: int
) -> tuple[list[str], np.ndarray, str, str]:
    protocol = make_protocol(budget)
    payload = _load(path)
    variant, group = EXPECTED[arm]
    _require(f"{path}: variant", payload.get("variant"), variant)
    _require(f"{path}: seed", payload.get("seed"), seed)
    _require(f"{path}: split", payload.get("split_counts"), list(protocol.split_counts))
    _require(f"{path}: units", payload.get("max_units_exclusive"), protocol.max_units_exclusive)
    _require(f"{path}: no test", payload.get("no_test_files_evaluated"), True)
    _require(f"{path}: eval backward", payload.get("uses_backward_gradients"), False)
    _require(
        f"{path}: eval label weight updates",
        payload.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    _require(
        f"{path}: calibration labels",
        payload.get("calibration_features_use_behavior_labels"),
        True,
    )

    eval_protocol = payload.get("protocol") or {}
    _require(
        f"{path}: evaluation calibration",
        eval_protocol.get("calibration_n"),
        protocol.evaluation_calibration_n,
    )
    _require(
        f"{path}: evaluation pool/start",
        eval_protocol.get("pool_size"),
        protocol.common_evaluation_start,
    )
    _require(
        f"{path}: labelled feature calibration",
        eval_protocol.get("label_feature_calibration_n"),
        budget,
    )
    _require(f"{path}: epoch window", eval_protocol.get("epoch_window"), EPOCHS)
    _require(
        f"{path}: calibration label scope",
        payload.get("calibration_feature_label_scope"),
        f"chronological_rewarded_trials[0:{budget}]",
    )

    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: run metadata is missing: {metadata_path}")
    metadata = _load(metadata_path)
    _require(f"{path}: metadata variant", metadata.get("variant"), variant)
    _require(f"{path}: metadata seed", metadata.get("seed"), seed)
    _require(
        f"{path}: metadata group",
        (metadata.get("side_features") or {}).get("group"),
        group,
    )
    _require(
        f"{path}: side-feature semantic version",
        (metadata.get("side_features") or {}).get("feature_version"),
        EXPECTED_FEATURE_VERSION[arm],
    )
    _require(
        f"{path}: activity calibration",
        (metadata.get("training") or {}).get("calibration_n_trials"),
        protocol.activity_calibration_n,
    )
    _require(
        f"{path}: T4 budget",
        (metadata.get("side_features") or {}).get("pool_size"),
        budget,
    )
    _require(
        f"{path}: fixed epochs",
        (metadata.get("training") or {}).get("max_epochs"),
        12,
    )
    _require(
        f"{path}: no early stopping",
        (metadata.get("training") or {}).get("no_early_stopping"),
        True,
    )
    _require(f"{path}: completed", metadata.get("status"), "completed")
    _require(f"{path}: training no-test", metadata.get("held_out_test_evaluated"), False)

    warmstart = metadata.get("encoder_warmstart_sha256")
    warmstart_path = metadata.get("encoder_warmstart_path")
    if not isinstance(warmstart, str) or len(warmstart) != 64:
        raise ValueError(f"{path}: selected-T4 warm-start SHA-256 is missing")
    if not isinstance(warmstart_path, str) or Path(warmstart_path).name != "epoch_011.ckpt":
        raise ValueError(f"{path}: warm-start must use the predeclared epoch_011.ckpt")

    sessions, values = _per_session_epoch_mean(payload, path)
    return sessions, values, warmstart, str(metadata_path.resolve())


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
        test = wilcoxon(
            session_means,
            alternative="two-sided",
            zero_method="wilcox",
            method="exact",
        )
        p_value = float(test.pvalue)
    except ValueError:
        p_value = 1.0
    ci = hierarchical_ci(delta)
    descriptive_gates = {
        "mean_delta_at_least_0p03": float(delta.mean()) >= 0.03,
        "all_observed_seed_means_positive": bool(np.all(seed_means > 0.0)),
        "all_six_session_means_positive": bool(np.all(session_means > 0.0)),
        "session_paired_exact_wilcoxon_two_sided_le_0p05": p_value <= 0.05,
    }
    formal_gates = {
        **descriptive_gates,
        "at_least_three_predeclared_seeds": len(seeds) >= 3,
        "hierarchical_bootstrap_95ci_lower_positive": ci[0] > 0.0,
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
        "descriptive_stage0_gates": descriptive_gates,
        "passes_stage0_descriptive_gates": all(descriptive_gates.values()),
        "formal_effectiveness_gates": formal_gates,
        "passes_formal_effectiveness_gates": all(formal_gates.values()),
    }


def aggregate_budget(
    result_dir: Path, budget: int, seeds: tuple[int, ...]
) -> dict:
    matrices: dict[str, np.ndarray] = {}
    artifacts: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    metadata_paths: dict[str, dict[str, str]] = {arm: {} for arm in ARMS}
    anchor_hashes: dict[str, str] = {}
    session_names: list[str] | None = None

    for arm in ARMS:
        rows: list[np.ndarray] = []
        for seed in seeds:
            path = result_dir / f"{arm}_m{budget}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            sessions, values, warmstart, metadata_path = validate_arm(
                path, arm, budget, seed
            )
            if session_names is None:
                session_names = sessions
            elif sessions != session_names:
                raise ValueError(f"{path}: validation session split differs from paired matrix")
            previous_hash = anchor_hashes.setdefault(str(seed), warmstart)
            if warmstart != previous_hash:
                raise ValueError(
                    f"M_T4={budget}, seed={seed}: all five arms must share one anchor SHA-256"
                )
            rows.append(values)
            artifacts[arm][str(seed)] = str(path.resolve())
            metadata_paths[arm][str(seed)] = metadata_path
        matrices[arm] = np.asarray(rows, dtype=np.float64)

    assert session_names is not None
    contrasts = {
        name: summarize(
            matrices["film"],
            matrices[control],
            seeds=seeds,
            sessions=session_names,
        )
        for name, control in CONTRASTS.items()
    }
    required_stage0 = all(
        contrasts[name]["passes_stage0_descriptive_gates"]
        for name in REQUIRED_MECHANISM_CONTRASTS
    )
    required_formal = all(
        contrasts[name]["passes_formal_effectiveness_gates"]
        for name in REQUIRED_MECHANISM_CONTRASTS
    )
    return {
        "protocol": {
            "M_activity": 30,
            "M_T4": budget,
            "common_evaluation_start": 50,
            "epochs": 12,
            "scored_epoch_window": EPOCHS,
            "seeds": list(seeds),
            "sessions": session_names,
            "formal_test_evaluated": False,
        },
        "selected_t4_anchor_sha256_by_seed": anchor_hashes,
        "artifacts": artifacts,
        "run_metadata": metadata_paths,
        "arm_mean_r2": {
            arm: float(matrix.mean()) for arm, matrix in matrices.items()
        },
        "contrasts": contrasts,
        "required_mechanism_contrasts": list(REQUIRED_MECHANISM_CONTRASTS),
        "stage0_descriptive_mechanism_pass": required_stage0,
        "formal_effectiveness_pass": required_formal,
        "formal_effectiveness_eligible": len(seeds) >= 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in T4_BUDGETS),
        help="Comma-separated subset of 10,15,20,30,50.",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="Comma-separated paired seeds; formal effectiveness requires at least three.",
    )
    args = parser.parse_args()
    budgets = _parse_int_csv(args.budgets, allowed=set(T4_BUDGETS))
    seeds = _parse_int_csv(args.seeds)
    result_dir = args.result_dir.expanduser().resolve()

    payload = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": "selected_t4_confidence_film_budget_screen",
        "claim_scope": "DANDI 000688 sub-C CO validation; formal test unopened",
        "formal_test_evaluated": False,
        "budgets": {
            str(budget): aggregate_budget(result_dir, budget, seeds)
            for budget in budgets
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                budget: {
                    "arm_mean_r2": record["arm_mean_r2"],
                    "stage0_descriptive_mechanism_pass": record[
                        "stage0_descriptive_mechanism_pass"
                    ],
                    "formal_effectiveness_pass": record["formal_effectiveness_pass"],
                }
                for budget, record in payload["budgets"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
