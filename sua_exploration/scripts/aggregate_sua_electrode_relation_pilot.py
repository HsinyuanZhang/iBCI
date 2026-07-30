"""Fail-closed seed-42 aggregator for the strict SUA electrode-relation pilot.

This is deliberately a *pilot* aggregator, not a multi-seed significance test.  It
accepts exactly the four fixed epoch-window JSON artifacts produced by
``run_sua_electrode_relation_pilot.sh`` and rejects any drift in the frozen data,
calibration, or selection contracts before reporting REL−T4, REL−REL-MS, and
REL−REL-NG six-session paired deltas.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean


EXPECTED = {
    "t4": ("B3S", "t4"),
    "relation": ("B3SER", "t4rel"),
    "membership_shuffle": ("B3SER", "t4rel_membership_shuffled"),
    "no_group": ("B3SERN", "t4rel_nogroup"),
}
EXPECTED_EPOCHS = list(range(5, 13))
EXPECTED_SPLIT = [27, 6, 6]
EXPECTED_MANIFEST = "subc_co_27_6_strict_train_val_manifest.json"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON artifact {path}: {exc}") from exc


def require_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def validate_arm(
    path: Path, arm: str, *, expected_seed: int
) -> tuple[dict, dict[str, float]]:
    payload = read_json(path)
    expected_variant, expected_group = EXPECTED[arm]
    require_equal(f"{path}: variant", payload.get("variant"), expected_variant)
    require_equal(f"{path}: seed", payload.get("seed"), expected_seed)
    require_equal(f"{path}: split_counts", payload.get("split_counts"), EXPECTED_SPLIT)
    require_equal(f"{path}: max_units_exclusive", payload.get("max_units_exclusive"), 100)
    require_equal(f"{path}: signal_view", payload.get("signal_view"), "sua")
    if payload.get("no_test_files_evaluated") is not True:
        raise ValueError(f"{path}: no_test_files_evaluated must be true")
    manifest_path = payload.get("train_val_manifest")
    if not isinstance(manifest_path, str) or not manifest_path.endswith(EXPECTED_MANIFEST):
        raise ValueError(f"{path}: strict frozen manifest path is missing or unexpected")
    manifest_hash = payload.get("train_val_manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError(f"{path}: strict frozen manifest SHA-256 is missing")

    protocol = payload.get("protocol") or {}
    require_equal(f"{path}: protocol.total_epochs", protocol.get("total_epochs"), 12)
    require_equal(f"{path}: protocol.epoch_window", protocol.get("epoch_window"), EXPECTED_EPOCHS)
    require_equal(f"{path}: protocol.selection_mode", protocol.get("selection_mode"), "first")
    require_equal(f"{path}: protocol.pool_size", protocol.get("pool_size"), 50)
    require_equal(f"{path}: protocol.train_activity_calibration_n", protocol.get("train_activity_calibration_n"), 10)
    require_equal(f"{path}: protocol.evaluation_forward_calibration_n", protocol.get("evaluation_forward_calibration_n"), 30)

    metadata_path = Path(payload.get("run_metadata_path", ""))
    metadata = read_json(metadata_path)
    require_equal(f"{path}: metadata.variant", metadata.get("variant"), expected_variant)
    require_equal(f"{path}: metadata.side_features.group", (metadata.get("side_features") or {}).get("group"), expected_group)
    require_equal(f"{path}: metadata.training.calibration_n_trials", (metadata.get("training") or {}).get("calibration_n_trials"), 10)
    if metadata.get("held_out_test_evaluated") is not False:
        raise ValueError(f"{path}: metadata.held_out_test_evaluated must be false")
    require_equal(f"{path}: metadata manifest path", metadata.get("train_val_manifest"), manifest_path)
    require_equal(f"{path}: metadata manifest SHA-256", metadata.get("train_val_manifest_sha256"), manifest_hash)

    per_epoch = payload.get("per_epoch") or {}
    session_names: set[str] | None = None
    per_session_values: dict[str, list[float]] = {}
    for epoch in EXPECTED_EPOCHS:
        epoch_payload = per_epoch.get(str(epoch))
        if not isinstance(epoch_payload, dict):
            raise ValueError(f"{path}: missing protocol epoch {epoch}")
        scores = epoch_payload.get("per_session_r2")
        if not isinstance(scores, dict) or len(scores) != 6:
            raise ValueError(f"{path}: epoch {epoch} must contain exactly six validation R2 scores")
        names = set(scores)
        if session_names is None:
            session_names = names
        elif names != session_names:
            raise ValueError(f"{path}: validation session set drifted at epoch {epoch}")
        for session, value in scores.items():
            per_session_values.setdefault(session, []).append(float(value))
    assert session_names is not None
    return payload, {session: mean(values) for session, values in sorted(per_session_values.items())}


def compute_relation_control_deltas(
    records: dict[str, dict[str, float]]
) -> dict[str, dict]:
    """Compute the pre-registered REL treatment contrasts, never control-vs-T4 proxies."""
    sessions = set(records["t4"])
    if any(set(scores) != sessions for scores in records.values()):
        raise ValueError("all four arms must score exactly the same six validation sessions")
    paired = {}
    for treatment, control, name in (
        ("relation", "t4", "REL_minus_T4"),
        ("relation", "membership_shuffle", "REL_minus_REL_MS"),
        ("relation", "no_group", "REL_minus_REL_NG"),
    ):
        deltas = {
            session: records[treatment][session] - records[control][session]
            for session in sorted(sessions)
        }
        paired[name] = {
            "treatment": treatment,
            "control": control,
            "per_session_epoch_window_mean_delta_r2": deltas,
            "mean_paired_delta_r2": mean(deltas.values()),
        }
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in EXPECTED:
        parser.add_argument(f"--{arm.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    artifacts = {
        arm: getattr(args, arm).expanduser().resolve()
        for arm in EXPECTED
    }
    records: dict[str, dict[str, float]] = {}
    provenance: dict[str, dict] = {}
    for arm, path in artifacts.items():
        payload, scores = validate_arm(path, arm, expected_seed=args.seed)
        records[arm] = scores
        provenance[arm] = {
            "artifact": str(path),
            "artifact_manifest_sha256": payload["train_val_manifest_sha256"],
            "run_metadata_path": payload["run_metadata_path"],
        }
    hashes = {entry["artifact_manifest_sha256"] for entry in provenance.values()}
    if len(hashes) != 1:
        raise ValueError("all four arms must use exactly the same frozen manifest SHA-256")
    paired = compute_relation_control_deltas(records)

    output = {
        "schema_version": 1,
        "purpose": "strict_single_seed_sua_electrode_relation_pilot",
        "created_at": datetime.now().astimezone().isoformat(),
        "aggregate_contract": {
            "seed": args.seed,
            "arms": list(EXPECTED),
            "split_counts": EXPECTED_SPLIT,
            "signal_view": "sua",
            "max_units_exclusive": 100,
            "training_activity_calibration_n": 10,
            "evaluation_forward_calibration_n": 30,
            "pool_size": 50,
            "epoch_window": EXPECTED_EPOCHS,
            "formal_test_evaluated": False,
            "multi_seed_inference": (
                f"not performed; this is a seed-{args.seed} aggregate only"
            ),
        },
        "provenance": provenance,
        "per_arm_per_session_epoch_window_mean_r2": records,
        "paired_deltas": paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote strict seed-{args.seed} relation pilot aggregate: {args.out}")


if __name__ == "__main__":
    main()
