#!/usr/bin/env python3
"""Audit the sealed historical H1 R300 recency/flat pair using JSON only.

This helper deliberately reads no checkpoints, model code, NWB/NPZ inputs, query
data, or APIs.  It pins the seven source JSON files below by SHA-256, validates
the paired training contract, and recomputes each arm's own earliest maximum
over a complete 32-epoch HO-M3 curve.  The result is historical supplementary
evidence only: it does not evaluate the current signed_state14 line or any M1
mean-rate flat arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PAIR_ROOT = REPO_ROOT / "results/rift_v1/h1_r300_pair_20260907T075312Z"
DEFAULT_OUTPUT = PAIR_ROOT / "recency_flat_existing_h1_pair_audit_v1.json"

# These hashes pin the current JSON receipts, rather than merely recording them
# after an analysis has already run.
SOURCES = {
    "ready": (
        PAIR_ROOT / "ready.json",
        "7bb94423e43693eb0706bb64be35f8a3cbf096a459d8004446c546e1ae993f6a",
    ),
    "launch_receipt": (
        PAIR_ROOT / "launch_receipt.json",
        "dd9e6b3130d8d7c9c08ce68a2a868e605f353268a28349b2aecbf202fb4a3fe3",
    ),
    "launch_state": (
        PAIR_ROOT / "launch_state.json",
        "8cbe3e90f1ee6afd3f57091cc16b825a814553a32cc477bf3497e24711a5af91",
    ),
    "recency_train_receipt": (
        PAIR_ROOT / "recency_20260907_155347/formal/train_receipt.json",
        "64f9ff37ef3ee6b6341bb7ab538915d50ff7a31bdb1cc005661d74415e9d29bb",
    ),
    "recency_selection": (
        PAIR_ROOT / "recency_20260907_155347/formal/ho_m3_selection.json",
        "985879a7a58d63e2361de9b5df1c77bac8c976f0683ab9f30d1a88d0fd2d3741",
    ),
    "flat_train_receipt": (
        PAIR_ROOT / "flat_20260907_155347/formal/train_receipt.json",
        "03b95920d2ea1f5ad465b6244c3ac95e8a070318571da4f38ba4ce8702d62a07",
    ),
    "flat_selection": (
        PAIR_ROOT / "flat_20260907_155347/formal/ho_m3_selection.json",
        "43b8c4bdd79c10321ef4f0c71c056deff83851fcbd2542e188f4d3ccc21ff85a",
    ),
}


def fail(message: str) -> None:
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_pinned_sources() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    values: dict[str, Any] = {}
    manifest: dict[str, dict[str, str]] = {}
    for name, (path, expected_hash) in SOURCES.items():
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual_hash == expected_hash, f"pinned source SHA mismatch: {path}")
        try:
            values[name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            fail(f"invalid JSON in {path}: {error}")
        manifest[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": actual_hash,
        }
    return values, manifest


def argv_value(argv: list[Any], option: str) -> Any:
    require(argv.count(option) == 1, f"expected one {option} in argv")
    index = argv.index(option)
    require(index + 1 < len(argv), f"missing value after {option}")
    return argv[index + 1]


def validate_ready(ready: dict[str, Any]) -> None:
    require(ready.get("schema") == "rift_h1_r300_pair_ready_v1", "unexpected ready schema")
    variants = ready.get("variants")
    require(isinstance(variants, dict) and set(variants) == {"recency", "flat"}, "ready variants")
    for arm in ("recency", "flat"):
        train_argv = variants[arm].get("train_argv")
        require(isinstance(train_argv, list), f"{arm} ready train argv")
        require(argv_value(train_argv, "--variant") == arm, f"{arm} ready identity")
        require(argv_value(train_argv, "--epochs") == "32", f"{arm} ready epoch contract")
        require(argv_value(train_argv, "--microbatch") == "8", f"{arm} ready microbatch contract")


def validate_launch(launch: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    require(launch.get("schema") == "rift_h1_r300_launch_receipt_v1", "unexpected launch schema")
    pairing = launch.get("pairing")
    require(isinstance(pairing, dict), "missing pairing record")
    for field in ("initialization_equal", "inventory_equal", "smoke_sequence_equal", "formal_from_same_fresh_init"):
        require(pairing.get(field) is True, f"pairing check failed: {field}")
    runs = launch.get("runs")
    require(isinstance(runs, dict) and set(runs) == {"recency", "flat"}, "launch arms")
    metas: dict[str, dict[str, Any]] = {}
    for arm in ("recency", "flat"):
        meta = runs[arm].get("meta")
        require(isinstance(meta, dict), f"missing {arm} meta")
        require(meta.get("schema") == "rift_h1_r300_v1", f"{arm} task/schema is not H1 R300")
        require(meta.get("variant") == arm, f"{arm} mode identity")
        require(meta.get("context_bins") == 300, f"{arm} context is not R300")
        require(meta.get("seed") == 42, f"{arm} seed is not 42")
        require(meta.get("epochs") == 32, f"{arm} epochs are not 32")
        require(meta.get("updates_per_epoch") == 731, f"{arm} updates/epoch")
        sessions = meta.get("train_sessions")
        require(isinstance(sessions, list) and len(sessions) == 13, f"{arm} source-session count")
        require(meta.get("official_test_used") is False, f"{arm} unexpectedly used official test")
        metas[arm] = meta
    for field in ("initialization_sha256", "pairing_digests"):
        require(metas["recency"].get(field) == metas["flat"].get(field), f"unpaired {field}")
    init_hash = metas["recency"].get("initialization_sha256")
    require(isinstance(init_hash, str) and len(init_hash) == 64, "missing paired initialization hash")
    digests = metas["recency"].get("pairing_digests")
    require(isinstance(digests, dict), "missing pairing digests")
    for field in ("endpoint_inventory_sha256", "bank_roster_sha256", "valid_mask_inventory_sha256"):
        value = digests.get(field)
        require(isinstance(value, str) and len(value) == 64, f"missing {field}")
    state_runs = state.get("runs")
    require(isinstance(state_runs, dict), "missing launch-state arms")
    for arm in ("recency", "flat"):
        require(state_runs.get(arm, {}).get("status") == "completed", f"{arm} launch state incomplete")
        require(state_runs[arm].get("exit_code") == 0, f"{arm} launch exit code")
    return {"initialization_sha256": init_hash, "pairing_digests": digests, "metas": metas}


def number(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"non-numeric {field}")
    result = float(value)
    require(math.isfinite(result), f"non-finite {field}")
    return result


def recompute_arm(arm: str, receipt: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    require(receipt.get("status") == "COMPLETED", f"{arm} train status")
    require(receipt.get("epochs") == 32, f"{arm} receipt epochs")
    require(receipt.get("updates") == 23392, f"{arm} receipt updates")
    require(receipt.get("official_test_used") is False, f"{arm} receipt official-test flag")
    require(selection.get("status") == "HO_M3_DEVELOPMENT_SELECTION", f"{arm} selection status")
    curve = selection.get("curve")
    require(isinstance(curve, list) and len(curve) == 32, f"{arm} lacks complete 32-epoch curve")
    best: dict[str, Any] | None = None
    for position, row in enumerate(curve, start=1):
        require(isinstance(row, dict), f"{arm} curve row {position}")
        require(row.get("epoch") == position and row.get("epoch_zero_based") == position - 1, f"{arm} curve epoch map")
        mean = number(row.get("val_ho_m3_grouped/r2_mean"), f"{arm} curve mean e{position}")
        worst = number(row.get("worst_session_r2"), f"{arm} curve worst e{position}")
        candidate = {"epoch": position, "mean_r2": mean, "worst_session_r2": worst}
        # Strict > preserves the earliest epoch when maxima tie.
        if best is None or mean > best["mean_r2"]:
            best = candidate
    require(best is not None, f"{arm} empty curve")
    selected = selection.get("selected")
    require(isinstance(selected, dict), f"{arm} missing selected row")
    require(selected.get("epoch") == best["epoch"], f"{arm} selected epoch disagrees with recomputed earliest maximum")
    require(receipt.get("selected_epoch") == best["epoch"], f"{arm} receipt selected epoch disagrees")
    selected_mean = number(selected.get("val_ho_m3_grouped/r2_mean"), f"{arm} selected mean")
    selected_worst = number(selected.get("worst_session_r2"), f"{arm} selected worst")
    require(math.isclose(selected_mean, best["mean_r2"], abs_tol=1e-12), f"{arm} selected mean mismatch")
    require(math.isclose(selected_worst, best["worst_session_r2"], abs_tol=1e-12), f"{arm} selected worst mismatch")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new JSON path; refuses to overwrite")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    values, sources = load_pinned_sources()
    validate_ready(values["ready"])
    contract = validate_launch(values["launch_receipt"], values["launch_state"])
    recency = recompute_arm("recency", values["recency_train_receipt"], values["recency_selection"])
    flat = recompute_arm("flat", values["flat_train_receipt"], values["flat_selection"])
    launch_runs = values["launch_receipt"]["runs"]
    early = {
        arm: number(launch_runs[arm]["heartbeat"]["elapsed_seconds"], f"{arm} 300-update elapsed seconds")
        for arm in ("recency", "flat")
    }
    report = {
        "schema": "recency_flat_existing_h1_pair_audit_v1",
        "status": "PASSED_HISTORICAL_H1_R300_PAIRED_DEVELOPMENT_EVIDENCE",
        "scope": {
            "task": "H1 historical RIFT R300 pair",
            "not_current_signed_state14": True,
            "not_m1_meanrate_flat": True,
            "official_test_used": False,
            "selection_surface": "HO-M3 development; each arm owns a complete 32-epoch earliest-maximum selection",
        },
        "pinned_sources": sources,
        "validated_contract": {
            "context_bins": 300,
            "seed": 42,
            "epochs": 32,
            "updates_per_epoch": 731,
            "total_updates_per_arm": 23392,
            "source_session_count": 13,
            "paired_initialization_sha256": contract["initialization_sha256"],
            "paired_inventory_sha256": contract["pairing_digests"],
            "paired_mode_identity": ["recency", "flat"],
        },
        "effect_development_only": {
            "recency_each_own_earliest_max": recency,
            "flat_each_own_earliest_max": flat,
            "recency_minus_flat_mean_r2": recency["mean_r2"] - flat["mean_r2"],
            "recency_minus_flat_worst_session_r2": recency["worst_session_r2"] - flat["worst_session_r2"],
        },
        "early_training_throughput_record_only": {
            "updates_per_arm": 300,
            "recency_elapsed_seconds": early["recency"],
            "flat_elapsed_seconds": early["flat"],
            "not_full_training_cost": True,
            "not_cpu_latency": True,
        },
        "verification_gap": {
            "checkpoint_or_bias_slope_inspection": "Not performed: JSON receipts cannot establish checkpoint contents or recency-bias slopes.",
            "required_if_needed": "ROOT must inspect the selected checkpoints and relevant metadata separately; this helper intentionally reads no weights.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
