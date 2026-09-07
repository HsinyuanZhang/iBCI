#!/usr/bin/env python3
"""Independent arithmetic/scope verifier for the H-SE5 V2 CPU receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SESSIONS = (
    "ses-19250101T111740", "ses-19250101T112404", "ses-19250108T110520",
    "ses-19250108T111022", "ses-19250108T111455", "ses-19250113T120811",
    "ses-19250113T121303", "ses-19250115T110633", "ses-19250115T111328",
    "ses-19250119T113543", "ses-19250119T114045", "ses-19250120T115044",
    "ses-19250120T115537",
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


def summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    pairs = [(name, float(rows[name]["forward"][field])) for name in SESSIONS]
    values = [value for _, value in pairs]
    ordered = sorted(values)
    remove = max(range(13), key=lambda index: abs(values[index]))
    kept = values[:remove] + values[remove + 1 :]
    return {
        "defined_sessions": 13,
        "mean": sum(values) / 13,
        "median": ordered[6],
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "leave_largest_absolute_out_mean": sum(kept) / 12,
        "removed_session": pairs[remove][0],
    }


def same_summary(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        if first[key] != second[key]: return False
    return all(math.isclose(float(first[key]), float(second[key]), abs_tol=1e-12, rel_tol=0.0)
               for key in ("mean", "median", "leave_largest_absolute_out_mean"))


def verify_budget(name: str, body: Mapping[str, Any]) -> bool:
    budget = int(name[1:])
    need(budget in (3, 4) and body["budget_trials"] == budget, f"{name}: budget mismatch")
    bases, rows = body["basis_by_outer_date"], body["sessions"]
    need(tuple(bases) == DATES and tuple(rows) == SESSIONS, f"{name}: roster mismatch")
    for session, row in rows.items():
        need(row["session"] == session and row["budget"] == budget, f"{name}/{session}: binding mismatch")
        need(row["support_events"] >= 8 and row["design_rank"] == 5, f"{name}/{session}: undefined fit")
        need(row["forward"]["status"] == "defined", f"{name}/{session}: forward undefined")
        need(row["forward"]["shuffled_fit"]["shuffle"]["fixed_points"] == 0,
             f"{name}/{session}: label shuffle fixed point")
        need(row["row_shuffle"]["fixed_points"] == 0, f"{name}/{session}: row shuffle fixed point")
        labels = row["label_accounting"]
        need(labels["acquisition_endpoint_position_scalars"] == 14 * row["support_events"],
             f"{name}/{session}: acquisition accounting mismatch")
        need(row["position_source"]["dense_velocity_series_opened"] is False,
             f"{name}/{session}: dense velocity access")
    shuffle = summary(rows, "median_delta_shuffle")
    intercept = summary(rows, "median_delta_intercept")
    need(same_summary(shuffle, body["aggregate"]["correct_minus_shuffle"]), f"{name}: shuffle summary drift")
    need(same_summary(intercept, body["aggregate"]["correct_minus_intercept"]), f"{name}: intercept summary drift")
    retained = sorted(float(bases[date]["retained_variance"]) for date in DATES)
    retained_median = (retained[2] + retained[3]) / 2
    basis_gate = retained[0] >= 0.65 and retained_median >= 0.70
    def functional(item: Mapping[str, Any]) -> bool:
        return (item["mean"] > 0 and item["median"] > 0 and item["positive"] >= 10
                and item["leave_largest_absolute_out_mean"] > 0)
    passed = basis_gate and functional(shuffle) and functional(intercept)
    gate = body["gpu_entrance_gate"]
    need(gate["defined_carriers_all_13"] is True, f"{name}: defined gate false")
    need(gate["basis_retained_variance_gate"] == basis_gate, f"{name}: basis gate drift")
    need(gate["correct_minus_shuffle_gate"] == functional(shuffle), f"{name}: shuffle gate drift")
    need(gate["correct_minus_intercept_gate"] == functional(intercept), f"{name}: intercept gate drift")
    need(gate["passed"] == passed, f"{name}: pass drift")
    return passed


def verify(path: Path) -> dict[str, Any]:
    path = path.resolve()
    body = json.loads(path.read_text(encoding="utf-8"))
    need(body["schema"] == "h1_sparse_event_endpoint_v2_source_audit_v1", "schema mismatch")
    need(body["protocol"] == "h1_sparse_event_endpoint_q4_ridge3_20260811_v2", "protocol mismatch")
    need(tuple(body["source_binding"]["sessions"]) == SESSIONS, "session roster mismatch")
    need(tuple(body["source_binding"]["dates"]) == DATES, "date roster mismatch")
    for row in body["source_binding"]["files"]:
        source = Path(row["path"])
        need(source.is_file() and file_sha(source) == row["sha256"], f"source SHA mismatch: {source}")
    binding = body["implementation_binding"]
    for prefix in ("v1_protocol", "v2_protocol", "v1_parser", "v2_implementation", "runner"):
        bound = Path(binding[f"{prefix}_path"])
        need(bound.is_file() and file_sha(bound) == binding[f"{prefix}_sha256"], f"binding drift: {prefix}")
    need(body["scope"] == {
        "public_held_in_calibration_nwbs_opened": 13,
        "minival_nwbs_opened": 0,
        "held_out_nwbs_opened": 0,
        "formal_test_labels_opened": 0,
        "dense_velocity_series_opened": False,
        "decoder_constructed": False,
        "trainer_constructed": False,
        "cuda_used": False,
    }, "scope mismatch")
    passed = {name: verify_budget(name, body["budgets"][name]) for name in ("M3", "M4")}
    passing = [budget for budget in (3, 4) if passed[f"M{budget}"]]
    selected = 3 if 3 in passing else (4 if 4 in passing else None)
    need(body["passing_budgets"] == passing and body["selected_gpu_budget"] == selected, "selection mismatch")
    expected = "PASS_CPU_HSE5_M3_GPU_READY" if selected == 3 else (
        "PASS_CPU_HSE5_M4_DEVELOPMENT_GPU_READY" if selected == 4 else "STOP_CPU_HSE5_GPU_GATE_FAILED"
    )
    need(body["status"] == expected, "status mismatch")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = file_sha(path)
    need(sidecar.is_file() and sidecar.read_text(encoding="ascii").split()[0] == digest, "sidecar mismatch")
    return {"status": "PASS", "receipt": str(path), "sha256": digest, "selected_gpu_budget": selected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
