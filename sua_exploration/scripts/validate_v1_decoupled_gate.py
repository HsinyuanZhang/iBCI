"""Validate the complete v1 seed-42 evidence gate before any v2 launch."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_sua_decoupled_kv import (  # noqa: E402
    ARMS,
    validate_arm,
)


EPOCHS = list(range(5, 13))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _require(label: str, observed, expected) -> None:
    if observed != expected:
        raise ValueError(
            f"{label}: expected {expected!r}, found {observed!r}"
        )


def validate_result_payload(path: Path, payload: dict) -> None:
    protocol = payload.get("protocol") or {}
    _require(f"{path}: schema", payload.get("schema_version"), 1)
    _require(
        f"{path}: purpose",
        payload.get("purpose"),
        "epoch_window_deterministic_checkpoint_selection",
    )
    _require(f"{path}: variant", payload.get("variant"), "B3S")
    _require(f"{path}: seed", payload.get("seed"), 42)
    _require(f"{path}: signal", payload.get("signal_view"), "sua")
    _require(f"{path}: split", payload.get("split_counts"), [27, 6, 6])
    _require(f"{path}: units", payload.get("max_units_exclusive"), 100)
    _require(f"{path}: epoch list", payload.get("epoch_list"), EPOCHS)
    _require(f"{path}: total epochs", protocol.get("total_epochs"), 12)
    _require(f"{path}: burn-in", protocol.get("burn_in_epochs"), 4)
    _require(f"{path}: epoch window", protocol.get("epoch_window"), EPOCHS)
    _require(f"{path}: selection", protocol.get("selection_mode"), "first")
    _require(f"{path}: calibration", protocol.get("calibration_n"), 30)
    _require(
        f"{path}: forward calibration",
        protocol.get("evaluation_forward_calibration_n"),
        30,
    )
    _require(
        f"{path}: activity calibration",
        protocol.get("train_activity_calibration_n"),
        30,
    )
    _require(
        f"{path}: T4 label calibration",
        protocol.get("label_feature_calibration_n"),
        50,
    )
    _require(f"{path}: pool", protocol.get("pool_size"), 50)
    _require(
        f"{path}: no test files",
        payload.get("no_test_files_evaluated"),
        True,
    )
    _require(
        f"{path}: no behavior-label weight update",
        payload.get("uses_behavior_labels_for_weight_updates"),
        False,
    )
    _require(
        f"{path}: no backward gradients",
        payload.get("uses_backward_gradients"),
        False,
    )
    metadata_path = Path(payload.get("run_metadata_path", ""))
    if not metadata_path.is_file():
        raise ValueError(f"{path}: run metadata is missing")
    if sha256_file(metadata_path) != payload.get("run_metadata_sha256"):
        raise ValueError(f"{path}: run metadata SHA256 drifted")


def validate_gate(
    *,
    aggregate_path: Path,
    result_dir: Path,
    arm_validator: Callable = validate_arm,
) -> dict:
    aggregate = _load_json(aggregate_path)
    protocol = aggregate.get("protocol") or {}
    _require("aggregate schema", aggregate.get("schema_version"), 1)
    _require(
        "aggregate purpose",
        aggregate.get("purpose"),
        "fresh_t4_decoupled_key_value_screen",
    )
    _require("aggregate seeds", protocol.get("seeds"), [42])
    _require("aggregate M_activity", protocol.get("M_activity"), 30)
    _require("aggregate M_T4", protocol.get("M_T4"), 50)
    _require(
        "aggregate evaluation start",
        protocol.get("common_evaluation_start"),
        50,
    )
    _require("aggregate epochs", protocol.get("epochs"), 12)
    _require(
        "aggregate scored window",
        protocol.get("scored_epoch_window"),
        EPOCHS,
    )
    _require(
        "aggregate no formal",
        protocol.get("formal_test_evaluated"),
        False,
    )
    artifacts = aggregate.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARMS):
        raise ValueError("aggregate must contain exactly the five v1 arms")
    if set((aggregate.get("arm_mean_r2") or {})) != set(ARMS):
        raise ValueError("aggregate arm_mean_r2 must contain all five arms")
    if not isinstance(
        aggregate.get("stage0_descriptive_mechanism_pass"), bool
    ):
        raise ValueError("aggregate Stage-0 gate must be boolean")

    result_hashes: dict[str, str] = {}
    for arm in ARMS:
        path = (result_dir / f"{arm}_m50_s42.json").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mapped = (artifacts.get(arm) or {}).get("42")
        _require(f"aggregate artifact {arm}", mapped, str(path))
        payload = _load_json(path)
        validate_result_payload(path, payload)
        arm_validator(path, arm, 42)
        result_hashes[arm] = sha256_file(path)
    return {
        "schema_version": 1,
        "gate": "complete_validated_v1_decoupled_seed42",
        "aggregate_path": str(aggregate_path.resolve()),
        "aggregate_sha256": sha256_file(aggregate_path),
        "result_sha256_by_arm": result_hashes,
        "formal_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = validate_gate(
        aggregate_path=args.aggregate.expanduser().resolve(),
        result_dir=args.result_dir.expanduser().resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
