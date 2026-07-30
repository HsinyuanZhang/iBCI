"""Aggregate preregistered per-unit side-feature ablation development results."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

SUA_TASK = "CO"
SUA_SPLIT_COUNTS = [27, 6, 6]
SUA_MAX_UNITS_EXCLUSIVE = 100
SUA_PROTOCOL = {"selection_mode": "first", "calibration_n": 30, "pool_size": 50}
FEATURE_GROUPS = ("F0", "F1", "F2", "FS")
SEEDS = (42, 43)
CONTENT_MEAN_DELTA = 0.005
CONTENT_MIN_DELTA = -0.03
CONTENT_POSITIVE_SESSIONS = 4


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_scores(payload: dict) -> dict[str, float]:
    selected = payload["selected_protocol"]
    name = f"gradient_free_calibrated_{selected['selection_mode']}_n{selected['calibration_n']}"
    return {
        session: float(configs[name])
        for session, configs in payload["per_session_r2"].items()
    }


def validate_artifact(path: Path, *, feature_group: str, seed: int) -> dict:
    if not path.is_file():
        raise ValueError(f"Side-feature artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_variant = "B3" if feature_group == "F0" else "B3S"
    required_keys = {
        "ckpt",
        "ckpt_sha256",
        "fixed_protocol",
        "max_units_exclusive",
        "no_test_files_evaluated",
        "per_session_r2",
        "selected_protocol",
        "session_splits",
        "signal_view",
        "split_counts",
        "task",
        "training_run_metadata",
        "training_run_metadata_sha256",
        "variant",
        "seed",
        "validation_complete",
    }
    missing = sorted(required_keys - payload.keys())
    if missing:
        raise ValueError(f"Artifact incomplete for {path}: missing {missing}")

    expected = {
        "variant": expected_variant,
        "seed": seed,
        "task": SUA_TASK,
        "signal_view": "sua",
        "split_counts": SUA_SPLIT_COUNTS,
        "max_units_exclusive": SUA_MAX_UNITS_EXCLUSIVE,
        "fixed_protocol": True,
        "validation_complete": True,
        "no_test_files_evaluated": True,
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
        raise ValueError(f"Artifact contract mismatch for {path}: {mismatches}")

    metadata_path = Path(payload["training_run_metadata"])
    checkpoint_path = Path(payload["ckpt"])
    for artifact_path, expected_hash, label in (
        (checkpoint_path, payload["ckpt_sha256"], "checkpoint"),
        (metadata_path, payload["training_run_metadata_sha256"], "training metadata"),
    ):
        if not artifact_path.is_file():
            raise ValueError(f"{label} missing for {path}: {artifact_path}")
        observed_hash = sha256_file(artifact_path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"{label} hash mismatch for {path}: expected {expected_hash}, observed {observed_hash}"
            )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    side_meta = metadata.get("side_features", {})
    expected_side_group = {
        "F0": "none",
        "F1": "f1",
        "F2": "f2",
        "FS": "fs",
    }[feature_group]
    if side_meta.get("group", "none") != expected_side_group:
        raise ValueError(
            f"Training metadata side feature group mismatch for {path}: "
            f"expected {expected_side_group}, found {side_meta.get('group')!r}"
        )

    splits = payload["session_splits"]
    if [len(splits[name]) for name in ("train", "val", "test")] != SUA_SPLIT_COUNTS:
        raise ValueError(f"Invalid session split sizes in {path}: {splits}")
    val_sessions = set(splits["val"])
    if set(selected_scores(payload)) != val_sessions:
        raise ValueError(f"Per-session scores do not match validation sessions for {path}")

    return {
        "feature_group": feature_group,
        "seed": seed,
        "variant": expected_variant,
        "checkpoint": str(checkpoint_path),
        "training_metadata": str(metadata_path),
        "validation_sessions": sorted(val_sessions),
        "scores": selected_scores(payload),
    }


def passes_content_gate(summary: dict[str, float | int]) -> bool:
    return (
        summary["mean"] >= CONTENT_MEAN_DELTA
        and summary["minimum"] >= CONTENT_MIN_DELTA
        and summary["positive_count"] >= CONTENT_POSITIVE_SESSIONS
        and summary["n"] == 6
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_dir",
        default="sua_exploration/results/side_feature_ablation_v1",
    )
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir).expanduser().resolve()
    contracts: dict[str, dict] = {}
    group_scores: dict[str, dict[int, dict[str, float]]] = {group: {} for group in FEATURE_GROUPS}
    reference_splits = None

    for feature_group in FEATURE_GROUPS:
        for seed in SEEDS:
            artifact_path = ablation_dir / f"sua_{feature_group.lower()}_s{seed}.json"
            contract = validate_artifact(
                artifact_path, feature_group=feature_group, seed=seed
            )
            key = f"{feature_group}_s{seed}"
            contracts[key] = contract
            group_scores[feature_group][seed] = contract["scores"]
            splits_path = json.loads(artifact_path.read_text(encoding="utf-8"))["session_splits"]
            if reference_splits is None:
                reference_splits = splits_path
            elif splits_path != reference_splits:
                raise ValueError(
                    f"Session splits mismatch for {artifact_path}: expected {reference_splits}"
                )

    sessions = sorted(reference_splits["val"])
    group_means: dict[str, float] = {}
    for feature_group in FEATURE_GROUPS:
        seed_means = [
            mean(list(group_scores[feature_group][seed].values()))
            for seed in SEEDS
        ]
        group_means[feature_group] = mean(seed_means)

    paired_deltas: dict[str, dict] = {}
    for feature_group in ("F1", "F2", "FS"):
        per_session_seed_mean_vs_f0: dict[str, float] = {}
        per_session_seed_mean_vs_fs: dict[str, float] = {}
        for session in sessions:
            deltas_vs_f0 = [
                group_scores[feature_group][seed][session]
                - group_scores["F0"][seed][session]
                for seed in SEEDS
            ]
            per_session_seed_mean_vs_f0[session] = mean(deltas_vs_f0)
            if feature_group != "FS":
                deltas_vs_fs = [
                    group_scores[feature_group][seed][session]
                    - group_scores["FS"][seed][session]
                    for seed in SEEDS
                ]
                per_session_seed_mean_vs_fs[session] = mean(deltas_vs_fs)
        paired_deltas[f"{feature_group}_minus_F0"] = {
            "per_session_seed_mean": per_session_seed_mean_vs_f0,
            "summary": summarize_deltas(list(per_session_seed_mean_vs_f0.values())),
        }
        if feature_group != "FS":
            paired_deltas[f"{feature_group}_minus_FS"] = {
                "per_session_seed_mean": per_session_seed_mean_vs_fs,
                "summary": summarize_deltas(list(per_session_seed_mean_vs_fs.values())),
            }

    def content_effective(group: str) -> bool:
        if group == "F0":
            return False
        vs_f0 = paired_deltas[f"{group}_minus_F0"]["summary"]
        vs_fs = paired_deltas.get(f"{group}_minus_FS", {}).get("summary")
        if vs_fs is None:
            return passes_content_gate(vs_f0)
        return passes_content_gate(vs_f0) and passes_content_gate(vs_fs)

    gates = {
        "f0_usable": group_means["F0"] > 0.0,
        "f1_usable": group_means["F1"] > 0.0,
        "f2_usable": group_means["F2"] > 0.0,
        "fs_usable": group_means["FS"] > 0.0,
        "f1_content_effective": content_effective("F1"),
        "f2_content_effective": content_effective("F2"),
        "advance_to_next_stage": any(content_effective(group) for group in ("F1", "F2")),
    }

    payload = {
        "schema_version": 1,
        "purpose": "side_feature_ablation_development_only",
        "no_formal_test_sessions_evaluated": True,
        "fixed_protocol": SUA_PROTOCOL,
        "session_splits": reference_splits,
        "feature_groups": list(FEATURE_GROUPS),
        "seeds": list(SEEDS),
        "artifact_contracts": contracts,
        "group_mean_r2": group_means,
        "paired_deltas": paired_deltas,
        "gates": gates,
        "thresholds": {
            "variant_usable": "mean R2 > 0",
            "content_effective": {
                "mean_delta": CONTENT_MEAN_DELTA,
                "minimum_delta": CONTENT_MIN_DELTA,
                "positive_sessions": CONTENT_POSITIVE_SESSIONS,
            },
        },
    }
    out_path = args.out_path or ablation_dir / "aggregate.json"
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2, sort_keys=True))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
