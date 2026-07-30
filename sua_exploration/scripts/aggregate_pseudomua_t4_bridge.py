#!/usr/bin/env python3
"""Aggregate the validation-only SUA -> pseudo-MUA T4 bridge.

The aggregator is deliberately strict.  It refuses artifacts that do not match
the frozen 27/6/6, 12-epoch, epoch-5..12, first/n=30/pool=50 protocol, and it
verifies each artifact against its sha256-pinned ``run_metadata.json``.

No NWB file is opened here.  The six formal test sessions are represented only
by their session names in the already-written split metadata.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_side_feature_ablation_v2 import (  # noqa: E402
    classify_group_verdict,
    classify_pair_verdict,
    implied_seed_correlation,
    sigma_delta_paired,
    sigma_delta_standard_error,
)

GROUPS: tuple[str, ...] = ("F0", "T4", "TS4")
SEEDS: tuple[int, ...] = (42, 43, 44)
VIEWS: tuple[str, ...] = ("sua", "pseudo_mua")
GROUP_CONTRACT: dict[str, dict[str, str]] = {
    "F0": {"variant": "B3", "side_features_group": "none"},
    "T4": {"variant": "B3S", "side_features_group": "t4"},
    "TS4": {"variant": "B3S", "side_features_group": "ts4"},
}
PAIR_CONTRACT: tuple[tuple[str, str], ...] = (
    ("T4", "F0"),
    ("T4", "TS4"),
)

EXPECTED_SPLIT_COUNTS = [27, 6, 6]
EXPECTED_EPOCHS = list(range(5, 13))
EXPECTED_TOTAL_EPOCHS = 12
EXPECTED_BURN_IN = 4
EXPECTED_VAL_SESSIONS = 6
EXPECTED_PROTOCOL_FIELDS = {
    "selection_mode": "first",
    "calibration_n": 30,
    "pool_size": 50,
}
EFFECTIVE_MEAN_DELTA = 0.03
EFFECTIVE_MIN_POSITIVE_SESSIONS = 5
INTERACTION_TOLERANCE = 0.03


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(statistics.fmean(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(results_dir: Path, group: str, seed: int) -> Path:
    return results_dir / f"{group.lower()}_s{seed}.json"


def _require_equal(path: Path, field: str, observed, expected) -> None:
    if observed != expected:
        raise ValueError(
            f"{path}: {field} mismatch: expected {expected!r}, observed {observed!r}"
        )


def _load_json_object(path: Path, *, kind: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {kind} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: {kind} must be a JSON object")
    return payload


def _validate_session_splits(path: Path, splits: object) -> dict[str, list[str]]:
    if not isinstance(splits, dict):
        raise ValueError(f"{path}: session_splits must be an object")
    expected_lengths = {"train": 27, "val": 6, "test": 6}
    normalized: dict[str, list[str]] = {}
    for split, expected_length in expected_lengths.items():
        names = splits.get(split)
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path}: session_splits.{split} must be a list of names")
        if len(names) != expected_length or len(set(names)) != expected_length:
            raise ValueError(
                f"{path}: session_splits.{split} must contain {expected_length} unique "
                f"sessions, observed {len(names)} rows/{len(set(names))} unique"
            )
        normalized[split] = names
    all_names = normalized["train"] + normalized["val"] + normalized["test"]
    if len(set(all_names)) != sum(expected_lengths.values()):
        raise ValueError(f"{path}: train/val/test session names are not disjoint")
    return normalized


def _validate_epoch_payload(
    payload: dict,
    *,
    path: Path,
    val_sessions: Sequence[str],
) -> None:
    per_epoch = payload.get("per_epoch")
    if not isinstance(per_epoch, dict):
        raise ValueError(f"{path}: per_epoch must be an object")
    expected_keys = {str(epoch) for epoch in EXPECTED_EPOCHS}
    if set(per_epoch) != expected_keys:
        raise ValueError(
            f"{path}: per_epoch keys must be exactly {sorted(expected_keys)}, "
            f"observed {sorted(per_epoch)}"
        )
    expected_sessions = set(val_sessions)
    for epoch in EXPECTED_EPOCHS:
        epoch_payload = per_epoch[str(epoch)]
        if not isinstance(epoch_payload, dict):
            raise ValueError(f"{path}: per_epoch[{epoch}] must be an object")
        per_session = epoch_payload.get("per_session_r2")
        if not isinstance(per_session, dict) or set(per_session) != expected_sessions:
            raise ValueError(
                f"{path}: epoch {epoch} per_session_r2 does not exactly match the six "
                "validation sessions"
            )
        values = list(per_session.values())
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError(f"{path}: epoch {epoch} contains non-finite/non-numeric R2")
        observed_mean = epoch_payload.get("mean_r2")
        if not isinstance(observed_mean, (int, float)) or not math.isfinite(observed_mean):
            raise ValueError(f"{path}: epoch {epoch} mean_r2 must be finite")
        if not math.isclose(float(observed_mean), mean(values), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{path}: epoch {epoch} mean_r2 is inconsistent with per_session_r2"
            )


def load_artifact(path: Path, *, group: str, seed: int, view: str) -> dict:
    """Load and validate one epoch-window artifact plus its pinned run metadata."""
    payload = _load_json_object(path, kind="epoch-window artifact")
    expected = GROUP_CONTRACT[group]

    _require_equal(path, "variant", payload.get("variant"), expected["variant"])
    _require_equal(path, "seed", payload.get("seed"), seed)
    _require_equal(path, "signal_view", payload.get("signal_view"), view)
    _require_equal(path, "task", payload.get("task"), "CO")
    _require_equal(path, "split_counts", payload.get("split_counts"), EXPECTED_SPLIT_COUNTS)
    _require_equal(path, "max_units_exclusive", payload.get("max_units_exclusive"), 100)
    _require_equal(
        path, "no_test_files_evaluated", payload.get("no_test_files_evaluated"), True
    )
    _require_equal(
        path,
        "calibration_trial_selection_uses_behavior_labels",
        payload.get("calibration_trial_selection_uses_behavior_labels"),
        False,
    )
    _require_equal(
        path,
        "uses_behavior_labels_for_weight_updates",
        payload.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    _require_equal(
        path, "uses_backward_gradients", payload.get("uses_backward_gradients"), False
    )

    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: protocol must be an object")
    _require_equal(path, "protocol.total_epochs", protocol.get("total_epochs"), 12)
    _require_equal(path, "protocol.burn_in_epochs", protocol.get("burn_in_epochs"), 4)
    _require_equal(path, "protocol.epoch_window", protocol.get("epoch_window"), EXPECTED_EPOCHS)
    _require_equal(path, "epoch_list", payload.get("epoch_list"), EXPECTED_EPOCHS)
    for field, value in EXPECTED_PROTOCOL_FIELDS.items():
        _require_equal(path, f"protocol.{field}", protocol.get(field), value)

    splits = _validate_session_splits(path, payload.get("session_splits"))
    _validate_epoch_payload(payload, path=path, val_sessions=splits["val"])

    run_dir_value = payload.get("run_dir")
    metadata_path_value = payload.get("run_metadata_path")
    if not isinstance(run_dir_value, str) or not isinstance(metadata_path_value, str):
        raise ValueError(f"{path}: run_dir/run_metadata_path must be strings")
    run_dir = Path(run_dir_value).expanduser().resolve()
    metadata_path = Path(metadata_path_value).expanduser().resolve()
    if metadata_path.parent != run_dir:
        raise ValueError(
            f"{path}: run_metadata_path parent {metadata_path.parent} != run_dir {run_dir}"
        )
    metadata = _load_json_object(metadata_path, kind="run metadata")
    _require_equal(
        path,
        "run_metadata_sha256",
        sha256_file(metadata_path),
        payload.get("run_metadata_sha256"),
    )

    metadata_contract = {
        "status": "completed",
        "variant": expected["variant"],
        "seed": seed,
        "signal_view": view,
        "task": "CO",
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "max_units_exclusive": 100,
        "held_out_test_evaluated": False,
        "output_dir": str(run_dir),
        "session_splits": splits,
    }
    for field, expected_value in metadata_contract.items():
        _require_equal(path, f"run_metadata.{field}", metadata.get(field), expected_value)

    side_metadata = metadata.get("side_features")
    if not isinstance(side_metadata, dict):
        raise ValueError(f"{path}: run_metadata.side_features must be an object")
    _require_equal(
        path,
        "run_metadata.side_features.group",
        side_metadata.get("group"),
        expected["side_features_group"],
    )
    if group != "F0":
        _require_equal(
            path, "run_metadata.side_features.pool_size", side_metadata.get("pool_size"), 50
        )

    training = metadata.get("training")
    if not isinstance(training, dict):
        raise ValueError(f"{path}: run_metadata.training must be an object")
    training_contract = {
        "max_epochs": 12,
        "no_early_stopping": True,
        "checkpoint_every_epoch": True,
    }
    for field, expected_value in training_contract.items():
        _require_equal(
            path, f"run_metadata.training.{field}", training.get(field), expected_value
        )

    # Preserve resolved metadata for cross-artifact validation without trusting
    # paths/hashes a second time.
    payload["_validated_run_dir"] = str(run_dir)
    payload["_validated_metadata_path"] = str(metadata_path)
    payload["_artifact_sha256"] = sha256_file(path)
    return payload


def session_scores(payload: Mapping) -> dict[str, float]:
    """Epoch-5..12 score for each validation session."""
    return {
        session: mean(
            [
                float(payload["per_epoch"][str(epoch)]["per_session_r2"][session])
                for epoch in EXPECTED_EPOCHS
            ]
        )
        for session in payload["session_splits"]["val"]
    }


def summarize_arm(
    table: Mapping[tuple[str, int], Mapping[str, float]],
    *,
    group: str,
    sessions: Sequence[str],
) -> dict:
    per_seed_mean = {
        str(seed): mean([table[group, seed][session] for session in sessions])
        for seed in SEEDS
    }
    per_session_seed_mean = {
        session: mean([table[group, seed][session] for seed in SEEDS])
        for session in sessions
    }
    return {
        "per_seed_mean": per_seed_mean,
        "per_session_seed_mean": per_session_seed_mean,
        "mean_score": mean(list(per_seed_mean.values())),
        "across_seed_sample_std": float(statistics.stdev(per_seed_mean.values())),
    }


def compute_pair(
    table: Mapping[tuple[str, int], Mapping[str, float]],
    treatment: str,
    control: str,
    sessions: Sequence[str],
    *,
    effective_mean_delta: float = EFFECTIVE_MEAN_DELTA,
) -> dict:
    """Compute a same-seed, same-session paired comparison under V4."""
    per_seed_mean = {
        str(seed): mean(
            [
                table[treatment, seed][session] - table[control, seed][session]
                for session in sessions
            ]
        )
        for seed in SEEDS
    }
    per_session_seed_mean = {
        session: mean(
            [
                table[treatment, seed][session] - table[control, seed][session]
                for seed in SEEDS
            ]
        )
        for session in sessions
    }
    seed_deltas = list(per_seed_mean.values())
    mean_delta = mean(seed_deltas)
    paired_se = sigma_delta_paired(seed_deltas)

    treatment_seed_scores = [
        mean([table[treatment, seed][session] for session in sessions]) for seed in SEEDS
    ]
    control_seed_scores = [
        mean([table[control, seed][session] for session in sessions]) for seed in SEEDS
    ]
    sigma_treatment = float(statistics.stdev(treatment_seed_scores))
    sigma_control = float(statistics.stdev(control_seed_scores))
    unpaired_se = sigma_delta_standard_error(
        sigma_treatment, sigma_control, len(SEEDS)
    )
    implied_rho = (
        implied_seed_correlation(
            sigma_a=sigma_treatment,
            sigma_b=sigma_control,
            per_seed_deltas=seed_deltas,
        )
        if sigma_treatment > 0.0 and sigma_control > 0.0
        else None
    )
    n_sessions_positive = sum(
        delta > 0.0 for delta in per_session_seed_mean.values()
    )
    verdict, decided_by = classify_pair_verdict(
        mean_delta=mean_delta,
        n_sessions_positive=n_sessions_positive,
        n_sessions_total=len(sessions),
        per_seed_means=seed_deltas,
        sigma_delta_paired=paired_se,
        effective_mean_delta_threshold=effective_mean_delta,
        effective_min_positive_sessions=EFFECTIVE_MIN_POSITIVE_SESSIONS,
    )
    return {
        "treatment": treatment,
        "control": control,
        "per_seed_mean": per_seed_mean,
        "per_session_seed_mean": per_session_seed_mean,
        "mean_delta": mean_delta,
        "paired_se": paired_se,
        "two_se_interval": [
            mean_delta - 2.0 * paired_se,
            mean_delta + 2.0 * paired_se,
        ],
        "unpaired_quadrature_se": unpaired_se,
        "implied_seed_correlation": implied_rho,
        "treatment_across_seed_sample_std": sigma_treatment,
        "control_across_seed_sample_std": sigma_control,
        "n_sessions_positive": n_sessions_positive,
        "n_sessions_total": len(sessions),
        "n_seeds_positive": sum(delta > 0.0 for delta in seed_deltas),
        "n_seeds_total": len(SEEDS),
        "effective_mean_delta": effective_mean_delta,
        "verdict": verdict,
        "decided_by": decided_by,
    }


# Backward-compatible name for focused unit tests and interactive analysis.
pair = compute_pair


def compute_gamma(
    scores: Mapping[tuple[str, str, int], Mapping[str, float]],
    *,
    sessions: Sequence[str],
    interaction_tolerance: float = INTERACTION_TOLERANCE,
) -> dict:
    """Difference-in-differences with seed, not session×seed, as the replicate."""

    def cell(seed: int, session: str) -> float:
        sua_delta = (
            scores["sua", "T4", seed][session]
            - scores["sua", "F0", seed][session]
        )
        pseudo_delta = (
            scores["pseudo_mua", "T4", seed][session]
            - scores["pseudo_mua", "F0", seed][session]
        )
        return float(sua_delta - pseudo_delta)

    per_seed_mean = {
        str(seed): mean([cell(seed, session) for session in sessions])
        for seed in SEEDS
    }
    per_session_seed_mean = {
        session: mean([cell(seed, session) for seed in SEEDS])
        for session in sessions
    }
    per_seed_per_session = {
        str(seed): {session: cell(seed, session) for session in sessions}
        for seed in SEEDS
    }
    gamma = mean(list(per_seed_mean.values()))
    paired_se = sigma_delta_paired(list(per_seed_mean.values()))
    lower = gamma - 2.0 * paired_se
    upper = gamma + 2.0 * paired_se

    if lower > interaction_tolerance:
        verdict = "sua_specific_amplification"
        decided_by = (
            f"lower 2SE bound {lower:.6f} > +{interaction_tolerance:.6f}"
        )
    elif upper < -interaction_tolerance:
        verdict = "pseudo_mua_amplification"
        decided_by = (
            f"upper 2SE bound {upper:.6f} < -{interaction_tolerance:.6f}"
        )
    elif lower >= -interaction_tolerance and upper <= interaction_tolerance:
        verdict = "view_invariant_within_tolerance"
        decided_by = (
            f"2SE interval [{lower:.6f}, {upper:.6f}] is contained in "
            f"[-{interaction_tolerance:.6f}, +{interaction_tolerance:.6f}]"
        )
    else:
        verdict = "indeterminate"
        decided_by = (
            f"2SE interval [{lower:.6f}, {upper:.6f}] neither clears nor fits within "
            f"±{interaction_tolerance:.6f}"
        )

    return {
        "definition": "(T4-F0)_SUA-(T4-F0)_pseudo_MUA",
        "replicate_definition": (
            "For each seed, mean the same-session difference-in-differences over the six "
            "validation sessions; paired_se is the SE across the three seed means."
        ),
        "per_seed_mean": per_seed_mean,
        "per_session_seed_mean": per_session_seed_mean,
        "per_seed_per_session": per_seed_per_session,
        "mean": gamma,
        "paired_se": paired_se,
        "two_se_interval": [lower, upper],
        "interaction_tolerance": interaction_tolerance,
        "n_sessions_positive": sum(
            value > 0.0 for value in per_session_seed_mean.values()
        ),
        "n_seeds_positive": sum(value > 0.0 for value in per_seed_mean.values()),
        "verdict": verdict,
        "decided_by": decided_by,
    }


def aggregate(
    pseudomua_results_dir: Path,
    sua_results_dir: Path,
    *,
    effective_mean_delta: float = EFFECTIVE_MEAN_DELTA,
    interaction_tolerance: float = INTERACTION_TOLERANCE,
) -> dict:
    result_dirs = {
        "sua": sua_results_dir.expanduser().resolve(),
        "pseudo_mua": pseudomua_results_dir.expanduser().resolve(),
    }
    artifacts: dict[tuple[str, str, int], dict] = {}
    scores: dict[tuple[str, str, int], dict[str, float]] = {}

    for view in VIEWS:
        for group in GROUPS:
            for seed in SEEDS:
                path = artifact_path(result_dirs[view], group, seed)
                payload = load_artifact(path, group=group, seed=seed, view=view)
                artifacts[view, group, seed] = payload
                scores[view, group, seed] = session_scores(payload)

    reference = artifacts["sua", "F0", 42]
    reference_splits = reference["session_splits"]
    reference_protocol = reference["protocol"]
    mismatched_splits = [
        key
        for key, payload in artifacts.items()
        if payload["session_splits"] != reference_splits
    ]
    if mismatched_splits:
        raise ValueError(
            f"session_splits disagree across SUA/pseudo-MUA artifacts: {mismatched_splits}"
        )
    mismatched_protocols = [
        key for key, payload in artifacts.items() if payload["protocol"] != reference_protocol
    ]
    if mismatched_protocols:
        raise ValueError(
            f"protocol disagrees across SUA/pseudo-MUA artifacts: {mismatched_protocols}"
        )
    run_dirs = [
        payload["_validated_run_dir"] for payload in artifacts.values()
    ]
    if len(set(run_dirs)) != len(run_dirs):
        raise ValueError("All 18 SUA/pseudo-MUA group×seed artifacts need exclusive run dirs")

    sessions = list(reference_splits["val"])
    output = {
        "schema_version": 2,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "purpose": "validation-only SUA to pseudo-MUA T4 bridge",
        "no_test_files_evaluated": True,
        "protocol": {
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "total_epochs": EXPECTED_TOTAL_EPOCHS,
            "burn_in_epochs": EXPECTED_BURN_IN,
            "epoch_window": EXPECTED_EPOCHS,
            **EXPECTED_PROTOCOL_FIELDS,
            "effective_mean_delta": effective_mean_delta,
            "effective_min_positive_sessions": EFFECTIVE_MIN_POSITIVE_SESSIONS,
            "interaction_tolerance": interaction_tolerance,
            "seeds": list(SEEDS),
            "validation_sessions": sessions,
        },
        "input_directories": {
            view: str(result_dirs[view]) for view in VIEWS
        },
        "input_artifacts": {
            view: {
                group: {
                    str(seed): {
                        "path": str(artifact_path(result_dirs[view], group, seed)),
                        "sha256": artifacts[view, group, seed]["_artifact_sha256"],
                        "run_metadata_path": artifacts[view, group, seed][
                            "_validated_metadata_path"
                        ],
                        "run_dir": artifacts[view, group, seed]["_validated_run_dir"],
                    }
                    for seed in SEEDS
                }
                for group in GROUPS
            }
            for view in VIEWS
        },
        "views": {},
    }

    for view in VIEWS:
        view_table = {
            (group, seed): scores[view, group, seed]
            for group in GROUPS
            for seed in SEEDS
        }
        comparisons = {
            f"{treatment}_minus_{control}": compute_pair(
                view_table,
                treatment,
                control,
                sessions,
                effective_mean_delta=effective_mean_delta,
            )
            for treatment, control in PAIR_CONTRACT
        }
        group_verdict, group_decided_by = classify_group_verdict(
            pair_results={
                pair_name: (pair_payload["verdict"], pair_payload["decided_by"])
                for pair_name, pair_payload in comparisons.items()
            }
        )
        output["views"][view] = {
            "arm_scores": {
                group: summarize_arm(view_table, group=group, sessions=sessions)
                for group in GROUPS
            },
            "comparisons": comparisons,
            "T4_group_verdict": group_verdict,
            "T4_group_decided_by": group_decided_by,
        }

    output["Gamma"] = compute_gamma(
        scores,
        sessions=sessions,
        interaction_tolerance=interaction_tolerance,
    )
    return output


def write_json_atomically(path: Path, payload: Mapping) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".json.tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def print_summary(payload: Mapping) -> None:
    print("pseudo-MUA T4 bridge — validation-only")
    print("view         F0        T4       TS4      T4-F0    T4-TS4   verdict")
    for view in VIEWS:
        view_payload = payload["views"][view]
        arms = view_payload["arm_scores"]
        pairs = view_payload["comparisons"]
        print(
            f"{view:10s} "
            f"{arms['F0']['mean_score']:+.6f} "
            f"{arms['T4']['mean_score']:+.6f} "
            f"{arms['TS4']['mean_score']:+.6f} "
            f"{pairs['T4_minus_F0']['mean_delta']:+.6f} "
            f"{pairs['T4_minus_TS4']['mean_delta']:+.6f} "
            f"{view_payload['T4_group_verdict']}"
        )
    gamma = payload["Gamma"]
    print(
        "Gamma=(T4-F0)_SUA-(T4-F0)_pseudo-MUA: "
        f"{gamma['mean']:+.6f}, paired SE={gamma['paired_se']:.6f}, "
        f"2SE=[{gamma['two_se_interval'][0]:+.6f}, "
        f"{gamma['two_se_interval'][1]:+.6f}], verdict={gamma['verdict']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudomua_results_dir", type=Path, required=True)
    parser.add_argument("--sua_results_dir", type=Path, required=True)
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument(
        "--effective_mean_delta", type=float, default=EFFECTIVE_MEAN_DELTA
    )
    parser.add_argument(
        "--interaction_tolerance", type=float, default=INTERACTION_TOLERANCE
    )
    args = parser.parse_args()
    if args.effective_mean_delta <= 0.0:
        parser.error("--effective_mean_delta must be positive")
    if args.interaction_tolerance <= 0.0:
        parser.error("--interaction_tolerance must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = aggregate(
        args.pseudomua_results_dir,
        args.sua_results_dir,
        effective_mean_delta=args.effective_mean_delta,
        interaction_tolerance=args.interaction_tolerance,
    )
    write_json_atomically(args.out_path, payload)
    print_summary(payload)
    print(f"Saved {args.out_path.expanduser().resolve()}")


if __name__ == "__main__":
    main()
