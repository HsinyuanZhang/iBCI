#!/usr/bin/env python3
"""Run the frozen public-source H1 sparse-event endpoint carrier audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as hse  # noqa: E402


DEFAULT_DATA = ROOT / "SPINT-main/data/000954"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v1/source_audit.json"
PROTOCOL_PATH = ROOT / "sua_exploration/docs/H1_SPARSE_EVENT_ENDPOINT_CARRIER_PROTOCOL_20260811.md"
IMPLEMENTATION_PATH = ROOT / "sua_exploration/mc_maze/h1_sparse_event_endpoint.py"


def _publish_once(path: Path, payload: bytes) -> str:
    output = path.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite source-audit receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = hse.sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    side_descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(side_descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _session_row(
    session: hse.EventSession,
    basis: hse.EndpointBasis,
    *,
    budget: int,
) -> dict[str, Any]:
    manifest = hse.session_manifest(session, budget=budget)
    support = session.events_before(budget)
    try:
        carrier, fit = hse.fit_session_carrier(session, basis, budget=budget)
        row_carrier, row_manifest = hse.channel_row_shuffle(
            carrier, session=session.session_name, budget=budget,
        )
        row_carrier_sha = hse.array_sha256(row_carrier)
        forward = hse.forward_transfer(session, basis, budget=budget)
        stability = hse.coefficient_split_stability(session, basis, budget=budget)
        return {
            **manifest,
            "basis_sha256": basis.basis_sha256,
            "design_rank": fit["design_rank"],
            "carrier_sha256": fit["carrier_sha256"],
            "row_shuffle": {**row_manifest, "carrier_sha256": row_carrier_sha},
            "forward": forward,
            "coefficient_trial_split_stability": stability,
        }
    except hse.SparseEventEndpointError as error:
        return {
            **manifest,
            "basis_sha256": basis.basis_sha256,
            "design_rank": None,
            "carrier_sha256": None,
            "row_shuffle": None,
            "forward": {
                "status": "undefined_fit",
                "reason": str(error),
                "support_events": len(support),
                "later_events": len(session.events_after(budget)),
                "median_delta_shuffle": None,
                "median_delta_intercept": None,
            },
            "coefficient_trial_split_stability": {"status": "undefined_parent_fit"},
        }


def run(*, data_root: Path) -> dict[str, Any]:
    started = time.time()
    paths = hse.index_heldin_calib(data_root)
    sessions = {name: hse.load_event_session(paths[name]) for name in hse.H1_HELDIN_SESSIONS}
    budget_receipts: dict[str, Any] = {}
    for budget in hse.SUPPORT_BUDGETS:
        bases = {
            date: hse.fit_endpoint_basis(sessions, outer_date=date, budget=budget)
            for date in hse.H1_DATES
        }
        session_rows = {
            name: _session_row(sessions[name], bases[sessions[name].date], budget=budget)
            for name in hse.H1_HELDIN_SESSIONS
        }
        basis_rows = [bases[date].manifest() for date in hse.H1_DATES]
        gate = hse.evaluate_gpu_gate(session_rows=session_rows, basis_rows=basis_rows)
        budget_receipts[f"M{budget}"] = {
            "budget_trials": budget,
            "basis_by_outer_date": {date: bases[date].manifest() for date in hse.H1_DATES},
            "sessions": session_rows,
            "aggregate": {
                "correct_minus_shuffle": hse.paired_summary([
                    (name, session_rows[name]["forward"].get("median_delta_shuffle"))
                    for name in hse.H1_HELDIN_SESSIONS
                ]),
                "correct_minus_intercept": hse.paired_summary([
                    (name, session_rows[name]["forward"].get("median_delta_intercept"))
                    for name in hse.H1_HELDIN_SESSIONS
                ]),
                "support_event_count": {
                    "minimum": min(int(row["support_events"]) for row in session_rows.values()),
                    "median": float(__import__("numpy").median([row["support_events"] for row in session_rows.values()])),
                    "maximum": max(int(row["support_events"]) for row in session_rows.values()),
                },
                "label_accounting_total": {
                    key: int(sum(row["label_accounting"][key] for row in session_rows.values()))
                    for key in next(iter(session_rows.values()))["label_accounting"]
                },
            },
            "gpu_entrance_gate": gate,
        }
    passing = [budget for budget in hse.SUPPORT_BUDGETS if budget_receipts[f"M{budget}"]["gpu_entrance_gate"]["passed"]]
    selected = 3 if 3 in passing else (4 if 4 in passing else None)
    status = (
        "PASS_CPU_HSE4_M3_GPU_ENTRANCE_READY"
        if selected == 3
        else "PASS_CPU_HSE4_M4_DEVELOPMENT_GPU_ENTRANCE_READY"
        if selected == 4
        else "STOP_CPU_HSE4_GPU_ENTRANCE_GATE_FAILED"
    )
    return {
        "schema": hse.SCHEMA,
        "protocol": hse.PROTOCOL,
        "status": status,
        "selected_gpu_budget": selected,
        "passing_budgets": passing,
        "frozen_constants": {
            "movement_tags": list(hse.MOVEMENT_TAGS),
            "support_budgets": list(hse.SUPPORT_BUDGETS),
            "position_dim": hse.POSITION_DIM,
            "latent_dim": hse.LATENT_DIM,
            "carrier_dim": hse.CARRIER_DIM,
            "bin_seconds": hse.BIN_SECONDS,
            "min_eval_bins": hse.MIN_EVAL_BINS,
            "ridge_lambda": hse.RIDGE_LAMBDA,
            "scale_floor": hse.SCALE_FLOOR,
            "shuffle_namespace": hse.SHUFFLE_NAMESPACE,
        },
        "source_binding": {
            "sessions": list(hse.H1_HELDIN_SESSIONS),
            "dates": list(hse.H1_DATES),
            "files": [
                {"session": name, "path": str(paths[name].resolve()), "sha256": sessions[name].input_sha256}
                for name in hse.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "protocol_path": str(PROTOCOL_PATH.resolve()),
            "protocol_sha256": hse.sha256_file(PROTOCOL_PATH),
            "implementation_path": str(IMPLEMENTATION_PATH.resolve()),
            "implementation_sha256": hse.sha256_file(IMPLEMENTATION_PATH),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": hse.sha256_file(Path(__file__).resolve()),
        },
        "budgets": budget_receipts,
        "scope": {
            "public_held_in_calibration_nwbs_opened": len(paths),
            "minival_nwbs_opened": 0,
            "held_out_nwbs_opened": 0,
            "formal_test_labels_opened": 0,
            "dense_velocity_series_opened": False,
            "decoder_constructed": False,
            "trainer_constructed": False,
            "cuda_used": False,
        },
        "runtime_seconds": float(time.time() - started),
        "interpretation_boundary": {
            "positive": "a passing budget permits one matched development GPU pilot of the new H-SE4 arm",
            "negative": "a failed gate rejects this frozen endpoint estimator; it does not prove all sparse H1 event carriers impossible",
            "not_yet_claimed": [
                "decoder R2 improvement",
                "organizer-held improvement",
                "non-inferiority to dense CarrierID",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = run(data_root=args.data_root)
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    digest = _publish_once(args.output, payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": digest,
        "status": receipt["status"],
        "selected_gpu_budget": receipt["selected_gpu_budget"],
        "runtime_seconds": receipt["runtime_seconds"],
    }, indent=2, sort_keys=True))
    return 0 if receipt["selected_gpu_budget"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
