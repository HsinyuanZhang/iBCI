"""Aggregate the preregistered B15 architecture-screen development results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

VARIANTS = ("B3", "B15P", "B15D", "B15")
CONTROLS = ("B3", "B15P", "B15D")
MUA_CELLS = ((1, 42), (1, 43), (2, 42))
SUA_TASK = "CO"
SUA_SPLIT_COUNTS = [27, 6, 6]
SUA_MAX_UNITS_EXCLUSIVE = 100
SUA_PROTOCOL = {"selection_mode": "first", "calibration_n": 30, "pool_size": 50}


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def summarize_deltas(deltas: list[float]) -> dict[str, float | int]:
    return {
        "mean": mean(deltas),
        "median": float(statistics.median(deltas)),
        "minimum": float(min(deltas)),
        "maximum": float(max(deltas)),
        "positive_count": sum(value > 0.0 for value in deltas),
        "n": len(deltas),
    }


def selected_scores(payload: dict) -> dict[str, float]:
    selected = payload["selected_protocol"]
    name = f"gradient_free_calibrated_{selected['selection_mode']}_n{selected['calibration_n']}"
    return {
        session: float(configs[name])
        for session, configs in payload["per_session_r2"].items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sua_artifact(path: Path, *, variant: str, seed: int) -> dict:
    if not path.is_file():
        raise ValueError(f"SUA artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_keys = {
        "ckpt",
        "ckpt_sha256",
        "fixed_protocol",
        "max_units_exclusive",
        "no_test_files_evaluated",
        "per_session_r2",
        "protocol_selection_uses_validation_behavior_labels",
        "selected_protocol",
        "session_splits",
        "signal_view",
        "split_counts",
        "task",
        "training_run_metadata",
        "training_run_metadata_sha256",
        "uses_backward_gradients",
        "uses_behavior_labels_for_weight_updates",
        "validation_complete",
        "variant",
        "seed",
    }
    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise ValueError(f"SUA artifact is incomplete: {path}: missing {missing_keys}")

    expected = {
        "variant": variant,
        "seed": seed,
        "task": SUA_TASK,
        "signal_view": "sua",
        "split_counts": SUA_SPLIT_COUNTS,
        "max_units_exclusive": SUA_MAX_UNITS_EXCLUSIVE,
        "fixed_protocol": True,
        "validation_complete": True,
        "no_test_files_evaluated": True,
        "protocol_selection_uses_validation_behavior_labels": False,
        "uses_behavior_labels_for_weight_updates": False,
        "uses_backward_gradients": False,
        "selected_protocol": SUA_PROTOCOL,
    }
    observed = {key: payload.get(key) for key in expected}
    observed["selected_protocol"] = {
        key: payload["selected_protocol"].get(key) for key in SUA_PROTOCOL
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"SUA artifact contract mismatch for {path}: {mismatches}")

    checkpoint_path = Path(payload["ckpt"])
    metadata_path = Path(payload["training_run_metadata"])
    for artifact_path, expected_hash, label in (
        (checkpoint_path, payload["ckpt_sha256"], "checkpoint"),
        (metadata_path, payload["training_run_metadata_sha256"], "training metadata"),
    ):
        if not artifact_path.is_file():
            raise ValueError(f"SUA {label} is missing for {path}: {artifact_path}")
        observed_hash = sha256_file(artifact_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"SUA {label} hash mismatch for {path}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_expected = {
        "metadata.status": "completed",
        "metadata.variant": variant,
        "metadata.seed": seed,
        "metadata.task": SUA_TASK,
        "metadata.signal_view": "sua",
        "metadata.split_counts": SUA_SPLIT_COUNTS,
        "metadata.max_units_exclusive": SUA_MAX_UNITS_EXCLUSIVE,
        "metadata.held_out_test_evaluated": False,
    }
    metadata_observed = {
        "metadata.status": metadata.get("status"),
        "metadata.variant": metadata.get("variant"),
        "metadata.seed": metadata.get("seed"),
        "metadata.task": metadata.get("task"),
        "metadata.signal_view": metadata.get("signal_view", "sua"),
        "metadata.split_counts": metadata.get("split_counts"),
        "metadata.max_units_exclusive": metadata.get("max_units_exclusive"),
        "metadata.held_out_test_evaluated": metadata.get("held_out_evaluation_protocol", {}).get(
            "held_out_test_evaluated"
        ),
    }
    metadata_mismatches = {
        key: {"expected": metadata_expected[key], "observed": metadata_observed[key]}
        for key in metadata_expected
        if metadata_observed[key] != metadata_expected[key]
    }
    if metadata_mismatches:
        raise ValueError(f"SUA training metadata mismatch for {path}: {metadata_mismatches}")

    splits = payload["session_splits"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError(f"SUA artifact has invalid session split keys: {path}: {splits.keys()}")
    split_sets = {name: set(sessions) for name, sessions in splits.items()}
    if any(len(split_sets[left] & split_sets[right]) for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"SUA artifact session splits overlap: {path}")
    if [len(splits[name]) for name in ("train", "val", "test")] != SUA_SPLIT_COUNTS:
        raise ValueError(f"SUA artifact session split sizes are invalid: {path}: {splits}")

    per_session = payload["per_session_r2"]
    if set(per_session) != split_sets["val"]:
        raise ValueError(
            f"SUA artifact per-session scores do not exactly match validation sessions: {path}"
        )
    score_name = "gradient_free_calibrated_first_n30"
    missing_scores = [session for session, configs in per_session.items() if score_name not in configs]
    if missing_scores:
        raise ValueError(f"SUA artifact lacks selected scores for {path}: {missing_scores}")
    return {
        "variant": variant,
        "seed": seed,
        "task": SUA_TASK,
        "signal_view": "sua",
        "validation_sessions": sorted(split_sets["val"]),
        "test_sessions_evaluated": False,
        "fixed_protocol": SUA_PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "training_metadata": str(metadata_path),
    }


def read_mua_score(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test_heldin":
                return float(row["R2_variance_weighted"])
    raise ValueError(f"No test_heldin R2 in {path}")


def validate_mua_artifact(
    artifact_dir: Path,
    *,
    variant: str,
    fold: int,
    seed: int,
    task: str,
) -> dict:
    resolved_path = artifact_dir / "resolved_config.yaml"
    split_path = artifact_dir / "split_manifest.json"
    metadata_path = artifact_dir / "run_metadata.json"
    required_paths = (resolved_path, split_path, metadata_path, artifact_dir / "metrics_summary.csv")
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise ValueError(f"MUA artifact is incomplete: {artifact_dir}: {missing_paths}")

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = resolved.get("data", {})
    model = resolved.get("model", {})
    expected = {
        "resolved.seed": seed,
        "resolved.data.task": task,
        "resolved.data.validation_protocol": "loso",
        "resolved.data.loso_fold": fold,
        "resolved.data.include_heldout_in_fit": False,
        "resolved.data.include_heldout_in_test": False,
        "resolved.model.variant": variant,
        "resolved.model.task": task,
        "resolved.model.freeze_decoder": True,
        "resolved.model.loss_mode": "task_plus_y_plus_E",
        "resolved.model.lambda_y": 1.0,
        "resolved.model.lambda_E": 0.1,
        "split.validation_protocol": "loso",
        "split.fold_id": fold,
        "split.heldout_evaluated_in_fit": False,
        "split.heldout_evaluated_in_test": False,
        "metadata.variant": variant,
        "metadata.seed": seed,
        "metadata.fold_id": fold,
        "metadata.validation_protocol": "loso",
        "metadata.heldout_evaluated_in_fit": False,
        "metadata.heldout_evaluated_in_test": False,
    }
    observed = {
        "resolved.seed": resolved.get("seed"),
        "resolved.data.task": data.get("task"),
        "resolved.data.validation_protocol": data.get("validation_protocol"),
        "resolved.data.loso_fold": data.get("loso_fold"),
        "resolved.data.include_heldout_in_fit": data.get("include_heldout_in_fit"),
        "resolved.data.include_heldout_in_test": data.get("include_heldout_in_test"),
        "resolved.model.variant": model.get("variant"),
        "resolved.model.task": model.get("task"),
        "resolved.model.freeze_decoder": model.get("freeze_decoder"),
        "resolved.model.loss_mode": model.get("loss_mode"),
        "resolved.model.lambda_y": model.get("lambda_y"),
        "resolved.model.lambda_E": model.get("lambda_E"),
        "split.validation_protocol": split.get("validation_protocol"),
        "split.fold_id": split.get("fold_id"),
        "split.heldout_evaluated_in_fit": split.get("heldout_evaluated_in_fit"),
        "split.heldout_evaluated_in_test": split.get("heldout_evaluated_in_test"),
        "metadata.variant": metadata.get("variant"),
        "metadata.seed": metadata.get("seed"),
        "metadata.fold_id": metadata.get("fold_id"),
        "metadata.validation_protocol": metadata.get("validation_protocol"),
        "metadata.heldout_evaluated_in_fit": metadata.get("heldout_evaluated_in_fit"),
        "metadata.heldout_evaluated_in_test": metadata.get("heldout_evaluated_in_test"),
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"MUA artifact contract mismatch for {artifact_dir}: {mismatches}")
    return {
        "run_id": metadata["run_id"],
        "variant": variant,
        "task": task,
        "seed": seed,
        "fold": fold,
        "validation_protocol": "loso",
        "heldout_evaluated_in_fit": False,
        "heldout_evaluated_in_test": False,
        "freeze_decoder": True,
        "loss_mode": "task_plus_y_plus_E",
    }


def find_mua_artifact(
    mua_root: Path,
    screen_id: str,
    variant: str,
    fold: int,
    seed: int,
    *,
    task: str,
) -> Path:
    expected_run_prefix = f"{screen_id}_{variant.lower()}_{task}_f{fold}_s{seed}_"
    matches: list[Path] = []
    for metadata_path in mua_root.glob("*/run_metadata.json"):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(payload.get("run_id", "")).startswith(expected_run_prefix):
            matches.append(metadata_path.parent)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one MUA artifact for {expected_run_prefix}*, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    screen_dir = root / "sua_exploration" / "results" / args.screen_id
    if not screen_dir.is_dir():
        raise FileNotFoundError(f"Screen directory does not exist: {screen_dir}")

    sua: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    sua_contracts: dict[str, dict[str, dict]] = {variant: {} for variant in VARIANTS}
    reference_sua_splits: dict[str, list[str]] | None = None
    for variant in VARIANTS:
        for seed in (42, 43):
            path = screen_dir / f"sua_{variant.lower()}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing SUA result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            contract = validate_sua_artifact(path, variant=variant, seed=seed)
            splits = payload["session_splits"]
            if reference_sua_splits is None:
                reference_sua_splits = splits
            elif splits != reference_sua_splits:
                raise ValueError(f"SUA session split mismatch against first artifact: {path}")
            sua_contracts[variant][f"seed{seed}"] = contract
            sua[variant][seed] = selected_scores(payload)

    sua_variant_means = {
        variant: mean([mean(list(seed_scores.values())) for seed_scores in sua[variant].values()])
        for variant in VARIANTS
    }
    sua_deltas: dict[str, dict] = {}
    for control in CONTROLS:
        per_session = {}
        for session in sorted(sua["B15"][42]):
            values = [sua["B15"][seed][session] - sua[control][seed][session] for seed in (42, 43)]
            per_session[session] = mean(values)
        sua_deltas[f"B15_minus_{control}"] = {
            "per_session_seed_mean": per_session,
            "summary": summarize_deltas(list(per_session.values())),
        }

    mua_root = root / "streaming_calibration_exp" / "outputs" / "streaming_calibration"
    cells = MUA_CELLS
    mua_scores: dict[str, dict[str, float]] = {variant: {} for variant in VARIANTS}
    artifact_paths: dict[str, dict[str, str]] = {variant: {} for variant in VARIANTS}
    artifact_contracts: dict[str, dict[str, dict]] = {variant: {} for variant in VARIANTS}
    for fold, seed in cells:
        key = f"fold{fold}_seed{seed}"
        for variant in VARIANTS:
            path = find_mua_artifact(
                mua_root, args.screen_id, variant, fold, seed, task="m2"
            )
            artifact_contracts[variant][key] = validate_mua_artifact(
                path, variant=variant, fold=fold, seed=seed, task="m2"
            )
            mua_scores[variant][key] = read_mua_score(path / "metrics_summary.csv")
            artifact_paths[variant][key] = str(path)

    mua_deltas = {
        f"B15_minus_{control}": {
            "per_cell": {
                cell: mua_scores["B15"][cell] - mua_scores[control][cell]
                for cell in sorted(mua_scores["B15"])
            }
        }
        for control in CONTROLS
    }
    for payload in mua_deltas.values():
        payload["summary"] = summarize_deltas(list(payload["per_cell"].values()))

    def passes_domain(summary: dict, *, required_positive: int, n: int) -> bool:
        return (
            summary["mean"] >= 0.005
            and summary["minimum"] >= -0.03
            and summary["positive_count"] >= required_positive
            and summary["n"] == n
        )

    gates = {
        "sua_architecture_usable": (
            sua_variant_means["B15"] > 0.0
            and sua_deltas["B15_minus_B3"]["summary"]["mean"] >= 0.0
            and sua_deltas["B15_minus_B3"]["summary"]["minimum"] >= -0.03
        ),
        "mua_architecture_usable": (
            mean(list(mua_scores["B15"].values())) > 0.0
            and mua_deltas["B15_minus_B3"]["summary"]["mean"] >= 0.0
            and mua_deltas["B15_minus_B3"]["summary"]["minimum"] >= -0.03
        ),
        "sua_attention_screen": all(
            passes_domain(sua_deltas[f"B15_minus_{control}"]["summary"], required_positive=4, n=6)
            for control in ("B15P", "B15D")
        ),
        "mua_attention_screen": all(
            passes_domain(mua_deltas[f"B15_minus_{control}"]["summary"], required_positive=2, n=3)
            for control in ("B15P", "B15D")
        ),
    }
    gates["advance_to_paired_pilot"] = all(gates.values())
    payload = {
        "schema_version": 1,
        "purpose": "attention_architecture_screen_development_only",
        "screen_id": args.screen_id,
        "no_formal_test_sessions_evaluated": True,
        "sua": {
            "fixed_protocol": SUA_PROTOCOL,
            "session_splits": reference_sua_splits,
            "artifact_contracts": sua_contracts,
            "variant_mean_r2": sua_variant_means,
            "paired_deltas": sua_deltas,
        },
        "mua": {
            "internal_loso_cells": [f"fold{fold}_seed{seed}" for fold, seed in cells],
            "b3_baseline_policy": "current_source_matched_artifact_required",
            "scores": mua_scores,
            "artifacts": artifact_paths,
            "artifact_contracts": artifact_contracts,
            "paired_deltas": mua_deltas,
        },
        "gates": gates,
    }
    out_path = args.out_path or screen_dir / "aggregate.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gates"], indent=2, sort_keys=True))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
