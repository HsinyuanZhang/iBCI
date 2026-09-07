#!/usr/bin/env python3
"""Run the frozen CPU-only H1 tag-free position-context carrier screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_tagfree_position_context as screen


DEFAULT_SEALED_RECEIPT = ROOT / "sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json"
DEFAULT_SEALED_SHA = "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite tag-free position-context receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "tag-free position-context receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def compact_summary(body: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rows: dict[str, Any] = {}
    for candidate in screen.CANDIDATES:
        candidate_name = candidate.name
        per_budget: dict[str, Any] = {}
        for budget_key in ("M3", "M4"):
            block = body["budgets"][budget_key]["candidates"][candidate_name]
            vs = block["aggregate"]["correct_minus_pca_delta_q4"]
            per_budget[budget_key] = {
                "mean_vs_pca_delta_q4": vs["mean"],
                "median_vs_pca_delta_q4": vs["median"],
                "positive_vs_pca_delta_q4": vs["positive"],
                "retained_energy_at_rank_mean": block["mean_retained_energy_at_rank"],
            }
        candidate_rows[candidate_name] = per_budget

    gate_table = {
        name: body["candidate_gates"][name]
        for name in screen.TAG_FREE_CANDIDATES
    }
    return {
        "schema": body["schema"],
        "status": body["status"],
        "selected_candidate": body["selected_candidate"],
        "passing_tag_free_candidates": body["passing_tag_free_candidates"],
        "integrity_check": body["integrity_check"],
        "runtime_seconds": body["runtime_seconds"],
        "candidates": candidate_rows,
        "gate_table": gate_table,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    sealed_path = args.sealed_receipt.resolve()
    sealed_sha = file_sha(sealed_path)
    need(sealed_sha == DEFAULT_SEALED_SHA, f"sealed receipt SHA mismatch: {sealed_sha}")
    sealed_receipt = json.loads(sealed_path.read_text(encoding="utf-8"))
    sessions = screen.load_context_sessions(args.data_root.resolve())
    body = screen.run_screen(sessions)
    integrity = screen.verify_sealed_integrity(body, sealed_receipt)
    module_path = Path(screen.__file__).resolve()
    runner_path = Path(__file__).resolve()
    body.update({
        "runtime_seconds": time.monotonic() - started,
        "integrity_check": {
            **integrity,
            "sealed_receipt_path": str(sealed_path),
            "sealed_receipt_sha256": sealed_sha,
        },
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {
                    "session": name,
                    "path": str(sessions[name].base.path),
                    "sha256": sessions[name].base.input_sha256,
                }
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "module_path": str(module_path),
            "module_sha256": file_sha(module_path),
            "runner_path": str(runner_path),
            "runner_sha256": file_sha(runner_path),
            "sealed_design_screen_module_path": str(Path(__import__(
                "sua_exploration.mc_maze.h1_event_carrier_design_screen", fromlist=["x"]
            ).__file__).resolve()),
            "sealed_design_screen_module_sha256": file_sha(Path(__import__(
                "sua_exploration.mc_maze.h1_event_carrier_design_screen", fromlist=["x"]
            ).__file__).resolve()),
        },
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument("--sealed-receipt", type=Path, default=DEFAULT_SEALED_RECEIPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sua_exploration/results/h1_tagfree_position_context/source_screen.json",
    )
    args = parser.parse_args()
    result = run(args)
    output, digest = write_immutable(args.output, result)
    summary = compact_summary(result)
    print(json.dumps({**summary, "receipt": str(output), "sha256": digest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
