#!/usr/bin/env python3
"""Run the frozen q4/ridge3 H1 sparse-endpoint V2 source audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2  # noqa: E402


DEFAULT_DATA = ROOT / "SPINT-main/data/000954"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json"
V1_PROTOCOL = ROOT / "sua_exploration/docs/H1_SPARSE_EVENT_ENDPOINT_CARRIER_PROTOCOL_20260811.md"
V2_PROTOCOL = ROOT / "sua_exploration/docs/H1_SPARSE_EVENT_ENDPOINT_V2_ADDENDUM_20260811.md"
V1_CODE = ROOT / "sua_exploration/mc_maze/h1_sparse_event_endpoint.py"
V2_CODE = ROOT / "sua_exploration/mc_maze/h1_sparse_event_endpoint_v2.py"


def _publish_once(path: Path, body: dict[str, Any]) -> str:
    output = path.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite V2 receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()
    digest = v1.sha256_file(output)
    fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n"); handle.flush(); os.fsync(handle.fileno())
    return digest


def run(data_root: Path) -> dict[str, Any]:
    started = time.time()
    paths = v1.index_heldin_calib(data_root)
    sessions = {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}
    bases = {date: v2.fit_source_all_event_basis(sessions, outer_date=date) for date in v1.H1_DATES}
    budgets: dict[str, Any] = {}
    for budget in v1.SUPPORT_BUDGETS:
        rows: dict[str, Any] = {}
        for name in v1.H1_HELDIN_SESSIONS:
            session = sessions[name]
            basis = bases[session.date]
            manifest = v1.session_manifest(session, budget=budget)
            try:
                carrier, fit = v2.fit_session_range(session, basis, start_index=0, budget=budget)
                shuffled_rows, row_manifest = v2.row_shuffle(carrier, session=name, budget=budget)
                forward = v2.forward_transfer(session, basis, budget=budget)
                rows[name] = {
                    **manifest,
                    "basis_sha256": basis.basis_sha256,
                    "design_rank": fit["design_rank"],
                    "carrier_sha256": fit["carrier_sha256"],
                    "row_shuffle": {**row_manifest, "carrier_sha256": v1.array_sha256(shuffled_rows)},
                    "forward": forward,
                }
            except v1.SparseEventEndpointError as error:
                rows[name] = {
                    **manifest,
                    "basis_sha256": basis.basis_sha256,
                    "design_rank": None,
                    "carrier_sha256": None,
                    "row_shuffle": None,
                    "forward": {"status": "undefined_fit", "reason": str(error)},
                }
        basis_rows = [bases[date].manifest() for date in v1.H1_DATES]
        gate = v2.evaluate_gate(session_rows=rows, basis_rows=basis_rows)
        budgets[f"M{budget}"] = {
            "budget_trials": budget,
            "basis_by_outer_date": {date: bases[date].manifest() for date in v1.H1_DATES},
            "sessions": rows,
            "aggregate": {
                "support_event_count": {
                    "minimum": min(row["support_events"] for row in rows.values()),
                    "median": float(np.median([row["support_events"] for row in rows.values()])),
                    "maximum": max(row["support_events"] for row in rows.values()),
                },
                "correct_minus_shuffle": v1.paired_summary([
                    (name, row["forward"].get("median_delta_shuffle")) for name, row in rows.items()
                ]),
                "correct_minus_intercept": v1.paired_summary([
                    (name, row["forward"].get("median_delta_intercept")) for name, row in rows.items()
                ]),
            },
            "gpu_entrance_gate": gate,
        }
    passing = [budget for budget in v1.SUPPORT_BUDGETS if budgets[f"M{budget}"]["gpu_entrance_gate"]["passed"]]
    selected = 3 if 3 in passing else (4 if 4 in passing else None)
    status = "PASS_CPU_HSE5_M3_GPU_READY" if selected == 3 else (
        "PASS_CPU_HSE5_M4_DEVELOPMENT_GPU_READY" if selected == 4 else "STOP_CPU_HSE5_GPU_GATE_FAILED"
    )
    return {
        "schema": v2.SCHEMA,
        "protocol": v2.PROTOCOL,
        "status": status,
        "passing_budgets": passing,
        "selected_gpu_budget": selected,
        "frozen_constants": {
            "latent_dim": v2.LATENT_DIM,
            "carrier_dim": v2.CARRIER_DIM,
            "ridge_lambda": v2.RIDGE_LAMBDA,
            "movement_tags": list(v1.MOVEMENT_TAGS),
            "support_budgets": list(v1.SUPPORT_BUDGETS),
            "basis_source_scope": "all valid movement events in non-outer-date public source recordings",
        },
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {"session": name, "path": str(paths[name]), "sha256": sessions[name].input_sha256}
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "v1_protocol_path": str(V1_PROTOCOL), "v1_protocol_sha256": v1.sha256_file(V1_PROTOCOL),
            "v2_protocol_path": str(V2_PROTOCOL), "v2_protocol_sha256": v1.sha256_file(V2_PROTOCOL),
            "v1_parser_path": str(V1_CODE), "v1_parser_sha256": v1.sha256_file(V1_CODE),
            "v2_implementation_path": str(V2_CODE), "v2_implementation_sha256": v1.sha256_file(V2_CODE),
            "runner_path": str(Path(__file__).resolve()), "runner_sha256": v1.sha256_file(Path(__file__).resolve()),
        },
        "budgets": budgets,
        "scope": {
            "public_held_in_calibration_nwbs_opened": 13,
            "minival_nwbs_opened": 0,
            "held_out_nwbs_opened": 0,
            "formal_test_labels_opened": 0,
            "dense_velocity_series_opened": False,
            "decoder_constructed": False,
            "trainer_constructed": False,
            "cuda_used": False,
        },
        "runtime_seconds": float(time.time() - started),
        "interpretation": {
            "development_status": "V2 was motivated by the immutable V1 stop and is not pristine confirmation",
            "gpu_permission": "one matched fold0 seed42 H-SE5/Zero5 development pair if M3 passes",
            "not_yet_claimed": ["decoder R2 improvement", "external-date confirmation", "organizer-held improvement"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = run(args.data_root)
    digest = _publish_once(args.output, body)
    print(json.dumps({
        "status": body["status"], "selected_gpu_budget": body["selected_gpu_budget"],
        "output": str(args.output.resolve()), "sha256": digest,
    }, indent=2, sort_keys=True))
    return 0 if body["selected_gpu_budget"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
