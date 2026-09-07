#!/usr/bin/env python3
"""CPU-only H1 sample-complexity accounting audit for carrier vs dense ridge readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_sample_complexity_audit_v1"
PROTOCOL = "h1_sample_complexity_audit_20260812_v1"
SUPPORT_BUDGETS: tuple[int, ...] = (3, 4)
POSITION_DIM = v1.POSITION_DIM
LATENT_DIM = v2.LATENT_DIM
CARRIER_DIM = v2.CARRIER_DIM
RIDGE_HISTORY_BINS = 50
RIDGE_CHANNELS = v1.EXPECTED_NEURONS
RIDGE_OUTPUTS = POSITION_DIM
FOLD0_SESSIONS: tuple[str, ...] = (
    "ses-19250101T111740",
    "ses-19250101T112404",
)
V2R2_RECEIPT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json"
DOCUMENTED_FOLD0_M4 = {
    "support_events": 41,
    "acquisition_endpoint_coordinates": 574,
    "dense_velocity_rows": 6088,
    "dense_velocity_coordinates": 42616,
}
V2R2_LABEL_FIELDS: tuple[str, ...] = (
    "support_events",
    "acquisition_endpoint_position_scalars",
    "derived_displacement_scalars",
    "projected_model_input_scalars",
    "dense_eval_bins_reference",
    "dense_per_bin_velocity_scalars_reference",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquisition_endpoint_coordinates(support_events: int) -> int:
    return int(support_events * 2 * POSITION_DIM)


def derived_displacement_coordinates(support_events: int) -> int:
    return int(support_events * POSITION_DIM)


def projected_estimator_inputs(support_events: int) -> int:
    return int(support_events * LATENT_DIM)


def dense_velocity_coordinates(eval_bins: int) -> int:
    return int(eval_bins * POSITION_DIM)


def coordinate_ratio(*, dense_coords: int, endpoint_coords: int) -> float:
    need(endpoint_coords > 0, "endpoint coordinate count must be positive for ratio")
    return float(dense_coords) / float(endpoint_coords)


def conservative_ratio_100ms(*, eval_bins: int, endpoint_coords: int) -> float:
    need(endpoint_coords > 0, "endpoint coordinate count must be positive for ratio")
    return float(eval_bins / 5.0 * POSITION_DIM) / float(endpoint_coords)


def carrier_observations_per_parameter(support_events: int) -> float:
    return float(support_events) / float(CARRIER_DIM)


def ridge_observations_per_parameter(eval_bins: int) -> float:
    return float(eval_bins) / float(RIDGE_HISTORY_BINS * RIDGE_CHANNELS)


def posedness_ratio(*, carrier_opp: float, ridge_opp: float) -> float:
    need(ridge_opp > 0.0, "ridge observations per parameter must be positive")
    return carrier_opp / ridge_opp


def _support_events(session: v1.EventSession, budget: int) -> int:
    return sum(1 for event in session.events if event.trial_index < budget)


def _eval_bins(session: v1.EventSession, budget: int) -> int:
    return int(sum(session.eval_bins_per_trial[:budget]))


def session_metrics(session: v1.EventSession, budget: int) -> dict[str, Any]:
    support = _support_events(session, budget)
    eval_bins = _eval_bins(session, budget)
    endpoint_coords = acquisition_endpoint_coordinates(support)
    dense_coords = dense_velocity_coordinates(eval_bins)
    carrier_opp = carrier_observations_per_parameter(support)
    ridge_opp = ridge_observations_per_parameter(eval_bins)
    return {
        "session": session.session_name,
        "date": session.date,
        "budget": budget,
        "support_events": support,
        "eval_bins": eval_bins,
        "acquisition_endpoint_coordinates": endpoint_coords,
        "derived_displacement_coordinates": derived_displacement_coordinates(support),
        "projected_estimator_inputs": projected_estimator_inputs(support),
        "dense_velocity_coordinates": dense_coords,
        "coordinate_ratio": coordinate_ratio(dense_coords=dense_coords, endpoint_coords=endpoint_coords),
        "conservative_ratio_100ms": conservative_ratio_100ms(eval_bins=eval_bins, endpoint_coords=endpoint_coords),
        "carrier_observations_per_parameter": carrier_opp,
        "ridge_observations_per_parameter": ridge_opp,
        "posedness_ratio": posedness_ratio(carrier_opp=carrier_opp, ridge_opp=ridge_opp),
        "carrier_underdetermined": support < CARRIER_DIM,
        "ridge_underdetermined": eval_bins < RIDGE_HISTORY_BINS * RIDGE_CHANNELS,
    }


def pooled_from_sums(*, support_events: int, eval_bins: int, endpoint_coords: int, dense_coords: int) -> dict[str, Any]:
    carrier_opp = carrier_observations_per_parameter(support_events)
    ridge_opp = ridge_observations_per_parameter(eval_bins)
    return {
        "support_events": support_events,
        "eval_bins": eval_bins,
        "acquisition_endpoint_coordinates": endpoint_coords,
        "dense_velocity_coordinates": dense_coords,
        "coordinate_ratio_from_sums": coordinate_ratio(dense_coords=dense_coords, endpoint_coords=endpoint_coords),
        "conservative_ratio_100ms_from_sums": conservative_ratio_100ms(
            eval_bins=eval_bins,
            endpoint_coords=endpoint_coords,
        ),
        "carrier_observations_per_parameter_from_sums": carrier_opp,
        "ridge_observations_per_parameter_from_sums": ridge_opp,
        "posedness_ratio_from_sums": posedness_ratio(carrier_opp=carrier_opp, ridge_opp=ridge_opp),
    }


def mean_of_per_session_ratios(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    need(rows, "cannot average ratios over an empty session set")
    return {
        "coordinate_ratio_mean_of_sessions": float(np.mean([float(row["coordinate_ratio"]) for row in rows])),
        "conservative_ratio_100ms_mean_of_sessions": float(
            np.mean([float(row["conservative_ratio_100ms"]) for row in rows])
        ),
        "posedness_ratio_mean_of_sessions": float(np.mean([float(row["posedness_ratio"]) for row in rows])),
    }


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def compare_documented(actual: Mapping[str, int | float], documented: Mapping[str, int | float]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, expected in documented.items():
        observed = actual[key]
        absolute_difference = float(observed) - float(expected)
        relative_difference = (
            absolute_difference / float(expected)
            if float(expected) != 0.0
            else (0.0 if absolute_difference == 0.0 else float("inf"))
        )
        rows[key] = {
            "computed": observed,
            "documented": expected,
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
        }
    return rows


def v2r2_label_totals(receipt: Mapping[str, Any], *, budget: int) -> dict[str, int]:
    sessions = receipt["budgets"][f"M{budget}"]["sessions"]
    totals = {field: 0 for field in V2R2_LABEL_FIELDS}
    for row in sessions.values():
        totals["support_events"] += int(row["support_events"])
        accounting = row["label_accounting"]
        totals["acquisition_endpoint_position_scalars"] += int(accounting["acquisition_endpoint_position_scalars"])
        totals["derived_displacement_scalars"] += int(accounting["derived_displacement_scalars"])
        totals["projected_model_input_scalars"] += int(accounting["projected_model_input_scalars"])
        totals["dense_eval_bins_reference"] += int(accounting["dense_eval_bins_reference"])
        totals["dense_per_bin_velocity_scalars_reference"] += int(
            accounting["dense_per_bin_velocity_scalars_reference"]
        )
    return totals


def v2r2_cross_check(computed_m3: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    if not receipt_path.is_file():
        return {"v2r2_receipt_present": False}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded = v2r2_label_totals(receipt, budget=3)
    computed = {
        "support_events": int(computed_m3["pooled_from_sums"]["support_events"]),
        "acquisition_endpoint_position_scalars": int(
            computed_m3["pooled_from_sums"]["acquisition_endpoint_coordinates"]
        ),
        "derived_displacement_scalars": int(
            sum(row["derived_displacement_coordinates"] for row in computed_m3["sessions"].values())
        ),
        "projected_model_input_scalars": int(
            sum(row["projected_estimator_inputs"] for row in computed_m3["sessions"].values())
        ),
        "dense_eval_bins_reference": int(computed_m3["pooled_from_sums"]["eval_bins"]),
        "dense_per_bin_velocity_scalars_reference": int(
            computed_m3["pooled_from_sums"]["dense_velocity_coordinates"]
        ),
    }
    differences: dict[str, Any] = {}
    for field in V2R2_LABEL_FIELDS:
        observed = computed[field]
        expected = recorded[field]
        delta = observed - expected
        differences[field] = {
            "computed": observed,
            "recorded": expected,
            "absolute_difference": delta,
            "relative_difference": (delta / expected) if expected else (0.0 if delta == 0 else float("inf")),
        }
    return {
        "v2r2_receipt_present": True,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": file_sha256(receipt_path),
        "recorded_m3_totals": recorded,
        "computed_m3_totals": computed,
        "differences": differences,
    }


def budget_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    support_events = int(sum(int(row["support_events"]) for row in rows))
    eval_bins = int(sum(int(row["eval_bins"]) for row in rows))
    endpoint_coords = int(sum(int(row["acquisition_endpoint_coordinates"]) for row in rows))
    dense_coords = int(sum(int(row["dense_velocity_coordinates"]) for row in rows))
    return {
        "sessions": {str(row["session"]): dict(row) for row in rows},
        "pooled_from_sums": pooled_from_sums(
            support_events=support_events,
            eval_bins=eval_bins,
            endpoint_coords=endpoint_coords,
            dense_coords=dense_coords,
        ),
        "mean_of_per_session_ratios": mean_of_per_session_ratios(rows),
        "carrier_observations_per_parameter_distribution": distribution(
            [float(row["carrier_observations_per_parameter"]) for row in rows]
        ),
        "ridge_observations_per_parameter_distribution": distribution(
            [float(row["ridge_observations_per_parameter"]) for row in rows]
        ),
    }


def fold0_m4_block(all_sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    rows = [session_metrics(all_sessions[name], 4) for name in FOLD0_SESSIONS]
    block = budget_block(rows)
    pooled = block["pooled_from_sums"]
    documented_comparison = compare_documented(
        {
            "support_events": pooled["support_events"],
            "acquisition_endpoint_coordinates": pooled["acquisition_endpoint_coordinates"],
            "dense_velocity_rows": pooled["eval_bins"],
            "dense_velocity_coordinates": pooled["dense_velocity_coordinates"],
        },
        DOCUMENTED_FOLD0_M4,
    )
    return {
        "sessions": list(FOLD0_SESSIONS),
        "budget": 4,
        **block,
        "documented_cross_check": documented_comparison,
    }


def write_immutable(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite sample-complexity audit receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "sample-complexity audit receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def print_summary(body: Mapping[str, Any]) -> None:
    header = (
        f"{'budget':>6}  {'coord_ratio':>12}  {'cons_100ms':>12}  "
        f"{'med_carrier_opp':>16}  {'med_ridge_opp':>14}  {'med_posedness':>13}"
    )
    print(header)
    print("-" * len(header))
    for budget in SUPPORT_BUDGETS:
        block = body["budgets"][f"M{budget}"]
        pooled = block["pooled_from_sums"]
        carrier_dist = block["carrier_observations_per_parameter_distribution"]
        ridge_dist = block["ridge_observations_per_parameter_distribution"]
        posedness_median = float(
            np.median([float(row["posedness_ratio"]) for row in block["sessions"].values()])
        )
        print(
            f"M{budget:>5}  "
            f"{pooled['coordinate_ratio_from_sums']:12.6f}  "
            f"{pooled['conservative_ratio_100ms_from_sums']:12.6f}  "
            f"{carrier_dist['median']:16.6f}  "
            f"{ridge_dist['median']:14.6f}  "
            f"{posedness_median:13.6f}"
        )
    print()
    print("fold-0 M4 cross-check (computed vs documented):")
    for key, row in body["fold0_m4"]["documented_cross_check"].items():
        print(
            f"  {key}: computed={row['computed']} documented={row['documented']} "
            f"abs_diff={row['absolute_difference']} rel_diff={row['relative_difference']:.6g}"
        )


def run(data_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    indexed = v1.index_heldin_calib(data_root.resolve())
    sessions = {name: v1.load_event_session(indexed[name]) for name in v1.H1_HELDIN_SESSIONS}
    budgets: dict[str, Any] = {}
    for budget in SUPPORT_BUDGETS:
        rows = [session_metrics(sessions[name], budget) for name in v1.H1_HELDIN_SESSIONS]
        budgets[f"M{budget}"] = budget_block(rows)
    fold0 = fold0_m4_block(sessions)
    runner_path = Path(__file__).resolve()
    parser_path = Path(v1.__file__).resolve()
    v2_path = Path(v2.__file__).resolve()
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "status": "PASS",
        "budgets": budgets,
        "fold0_m4": fold0,
        "v2r2_cross_check": v2r2_cross_check(budgets["M3"], V2R2_RECEIPT),
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {
                    "session": name,
                    "path": str(sessions[name].path),
                    "sha256": sessions[name].input_sha256,
                }
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "script_path": str(runner_path),
            "script_sha256": file_sha256(runner_path),
            "event_parser_path": str(parser_path),
            "event_parser_sha256": file_sha256(parser_path),
            "carrier_v2_path": str(v2_path),
            "carrier_v2_sha256": file_sha256(v2_path),
        },
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "dense_velocity_series_opened": False,
            "public_held_in_calibration_nwbs_opened": len(v1.H1_HELDIN_SESSIONS),
        },
        "constants": {
            "position_dim": POSITION_DIM,
            "latent_dim": LATENT_DIM,
            "carrier_dim": CARRIER_DIM,
            "ridge_history_bins": RIDGE_HISTORY_BINS,
            "ridge_channels": RIDGE_CHANNELS,
            "ridge_outputs": RIDGE_OUTPUTS,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sua_exploration/results/h1_sample_complexity_audit/audit.json",
    )
    args = parser.parse_args()
    body = run(args.data_root)
    output, digest = write_immutable(args.output, body)
    print_summary(body)
    print()
    print(json.dumps({"receipt": str(output), "sha256": digest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
