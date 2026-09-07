#!/usr/bin/env python3
"""Independently verify an immutable H1 sparse-event semantic V4 receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "h1_event_carrier_semantic_v4_source_screen_v1"
PROTOCOL = "h1_event_carrier_endpoint_velocity_semantics_20260812_v1"
CANDIDATES = (
    "pca_delta_q4",
    "pca_mean_velocity_q4",
    "ser_mean_velocity_q4",
    "ser_direction_speed_q4",
)
DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: Any, right: Any, tolerance: float = 1.0e-12) -> None:
    if left is None or right is None:
        need(left is right, f"None mismatch: {left!r} vs {right!r}")
        return
    need(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance),
         f"numeric mismatch: {left!r} vs {right!r}")


def summary(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    values = np.asarray([float(value) for _name, value in rows], dtype=np.float64)
    need(values.size == 13 and np.isfinite(values).all(), "V4 aggregate must contain 13 finite sessions")
    remove = int(np.argmax(np.abs(values)))
    kept = np.delete(values, remove)
    return {
        "defined_sessions": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive": int(np.sum(values > 0)),
        "zero": int(np.sum(values == 0)),
        "negative": int(np.sum(values < 0)),
        "leave_largest_absolute_out_mean": float(kept.mean()),
        "removed_session": rows[remove][0],
    }


def verify_summary(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        need(observed.get(key) == expected.get(key), f"V4 summary drift at {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        close(observed.get(key), expected.get(key))


def positive(value: Mapping[str, Any]) -> bool:
    return bool(
        value["defined_sessions"] == 13
        and value["mean"] > 0
        and value["median"] > 0
        and value["positive"] >= 10
        and value["leave_largest_absolute_out_mean"] > 0
    )


def verify(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    need(resolved.is_file(), f"missing V4 receipt: {resolved}")
    need(stat.S_IMODE(resolved.stat().st_mode) == 0o444, "V4 receipt is not immutable mode 0444")
    receipt_sha = file_sha(resolved)
    body = json.loads(resolved.read_text(encoding="utf-8"))
    need(body.get("schema") == SCHEMA and body.get("protocol") == PROTOCOL, "V4 schema/protocol mismatch")
    constants = body["frozen_constants"]
    need(constants["rank"] == 4 and constants["carrier_dim"] == 5, "V4 width drift")
    need(tuple(constants["support_budgets"]) == (3, 4), "V4 budget drift")
    need(tuple(constants["candidates_in_fixed_order"]) == CANDIDATES, "V4 candidate matrix drift")

    scope = body["scope"]
    expected_scope = {
        "native_position_endpoints_per_event": 2,
        "native_event_timestamps_per_event": 2,
        "dense_velocity_opened": False,
        "within_event_position_trajectory_opened": False,
        "target_session_optimizer_steps": 0,
        "target_session_backward_steps": 0,
        "public_held_in_calibration_nwbs_opened": 13,
        "minival_nwbs_opened": 0,
        "held_out_nwbs_opened": 0,
        "formal_test_labels_opened": 0,
        "decoder_constructed": False,
        "trainer_constructed": False,
        "cuda_used": False,
    }
    for key, value in expected_scope.items():
        need(scope.get(key) == value, f"V4 scope drift at {key}")

    binding = body["implementation_binding"]
    for prefix in ("module", "runner", "shared_design", "event_parser"):
        bound_path = Path(binding[f"{prefix}_path"]).resolve()
        need(file_sha(bound_path) == binding[f"{prefix}_sha256"], f"V4 {prefix} SHA drift")
    reproduction = body["baseline_reproduction"]
    need(reproduction["passed"] is True and reproduction["comparisons"] == 78, "V4 baseline reproduction missing")
    need(float(reproduction["maximum_absolute_difference"]) <= float(reproduction["absolute_tolerance"]),
         "V4 baseline reproduction exceeds tolerance")
    need(file_sha(Path(reproduction["reference_path"])) == reproduction["reference_sha256"],
         "V4 H-SE5 reference SHA drift")

    sources = body["source_binding"]
    session_names = tuple(sources["sessions"])
    need(len(session_names) == 13 and len(set(session_names)) == 13, "V4 source session allowlist drift")
    need(tuple(sources["dates"]) == DATES, "V4 source dates drift")
    for row in sources["files"]:
        need(file_sha(Path(row["path"])) == row["sha256"], f"V4 source input SHA drift: {row['session']}")

    passing: list[str] = []
    for budget in (3, 4):
        candidates = body["budgets"][f"M{budget}"]["candidates"]
        # The immutable receipt is serialized with sort_keys=True.  Scientific
        # order is bound above by candidates_in_fixed_order; the map itself is
        # therefore checked as a set.
        need(set(candidates) == set(CANDIDATES), f"V4 M{budget} candidate set drift")
        baseline_rows = candidates["pca_delta_q4"]["sessions"]
        need(tuple(baseline_rows) == session_names, f"V4 M{budget} baseline session order drift")
        for candidate_name in CANDIDATES:
            candidate = candidates[candidate_name]
            spec = candidate["spec"]
            need(spec["rank"] == 4 and spec["carrier_dim"] == 5, "V4 candidate width drift")
            need(spec["reads_only_endpoint_positions_and_event_timestamps"] is True,
                 "V4 sparse endpoint declaration missing")
            rows = candidate["sessions"]
            need(tuple(rows) == session_names, f"V4 M{budget}/{candidate_name} sessions drift")
            fields = {
                "correct_r2": [(name, float(rows[name]["median_r2_correct"])) for name in session_names],
                "correct_minus_hse5": [
                    (name, float(rows[name]["median_r2_correct"] - baseline_rows[name]["median_r2_correct"]))
                    for name in session_names
                ],
                "correct_minus_label_shuffle": [
                    (name, float(rows[name]["median_delta_label_shuffle"])) for name in session_names
                ],
                "correct_minus_intercept": [
                    (name, float(rows[name]["median_delta_intercept"])) for name in session_names
                ],
            }
            for key, values in fields.items():
                verify_summary(candidate["aggregate"][key], summary(values))
            aggregate = candidate["aggregate"]
            delta = aggregate["correct_minus_hse5"]
            material = bool(positive(delta) and delta["mean"] >= 0.02 and delta["median"] >= 0.01)
            expected_gate = bool(
                candidate_name != "pca_delta_q4"
                and material
                and positive(aggregate["correct_minus_label_shuffle"])
                and positive(aggregate["correct_minus_intercept"])
            )
            gate = candidate["gate"]
            need(gate["material_gain_vs_hse5"] is material and gate["passed"] is expected_gate,
                 f"V4 M{budget}/{candidate_name} gate arithmetic drift")
        for candidate_name in CANDIDATES[1:]:
            if candidates[candidate_name]["gate"]["passed"]:
                passing.append(f"M{budget}:{candidate_name}")

    both = [
        candidate_name for candidate_name in CANDIDATES[1:]
        if all(body["budgets"][f"M{budget}"]["candidates"][candidate_name]["gate"]["passed"] for budget in (3, 4))
    ]
    need(body["passing_candidates"] == both, "V4 cross-budget passing set drift")
    need((body["selected_candidate"] is None) == (not both), "V4 selection/null drift")
    expected_status = "PASS_CPU_SEMANTIC_CANDIDATE_SELECTED" if both else "STOP_CPU_SEMANTIC_CANDIDATE_NOT_MATERIAL"
    need(body["status"] == expected_status and body["gpu_authorized_by_this_screen"] is False,
         "V4 terminal status drift")
    return {
        "status": "PASS",
        "receipt": str(resolved),
        "receipt_sha256": receipt_sha,
        "terminal_status": body["status"],
        "cross_budget_passing_candidates": both,
        "single_budget_passes": passing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
