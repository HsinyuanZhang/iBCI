#!/usr/bin/env python3
"""Run the immutable, CPU-only H1 LRT5 source-screen receipt.

The command is intentionally a one-shot publisher.  It does not overwrite an
existing receipt, opens only the 13 public held-in-calibration NWBs through the
bound parser, and invokes no CUDA/training code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_event_carrier_lrt5 as lrt5  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5  # noqa: E402


DEFAULT_DATA = ROOT / "SPINT-main/data/000954"
DEFAULT_HSE5_RECEIPT = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json"
PREVIOUS_RECEIPT = ROOT / "sua_exploration/results/h1_event_carrier_lrt5/source_screen_v1.json"
PREVIOUS_RECEIPT_SHA256 = "f19e333dc1a57830593d380778f5bef10cf41a5bb1ecd7717c21f386ae25bc3a"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/h1_event_carrier_lrt5/source_screen_v2.json"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float, *, tolerance: float = 1.0e-10) -> float:
    difference = abs(float(left) - float(right))
    _need(difference <= tolerance, f"LRT5 failed exact H-SE5 reproduction: {difference}")
    return difference


def _hse5_reproduction(body: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    _need(reference.get("schema") == hse5.SCHEMA, "LRT5 H-SE5 reference schema mismatch")
    fields = {
        "median_r2_correct": "median_r2_correct",
        "median_delta_label_shuffle": "median_delta_shuffle",
        "median_delta_intercept": "median_delta_intercept",
    }
    rows: list[dict[str, Any]] = []
    for budget in lrt5.SUPPORT_BUDGETS:
        observed = body["budgets"][f"M{budget}"]["hse5_baseline_sessions"]
        expected = reference["budgets"][f"M{budget}"]["sessions"]
        for session in v1.H1_HELDIN_SESSIONS:
            for observed_field, expected_field in fields.items():
                difference = _close(observed[session][observed_field], expected[session]["forward"][expected_field])
                rows.append({
                    "budget": budget, "session": session, "field": observed_field,
                    "absolute_difference": difference,
                })
    return {
        "passed": True, "comparisons": len(rows), "absolute_tolerance": 1.0e-10,
        "maximum_absolute_difference": max(row["absolute_difference"] for row in rows),
        "reference_comparison_rows": rows,
    }


def _publish_immutable(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    _need(not output.exists() and not sidecar.exists(), f"refusing to overwrite LRT5 receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "LRT5 receipt chmod drift")
    digest = _sha(output)
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n"); handle.flush(); os.fsync(handle.fileno())
    return output, digest


def run(data_root: Path, hse5_receipt: Path) -> dict[str, Any]:
    started = time.monotonic()
    # r1 is deliberately retained unchanged as the historical receipt.  V2
    # cannot silently replace it because the old source-prior arm omitted its
    # nonzero-slope mean alignment.
    _need(PREVIOUS_RECEIPT.is_file() and stat.S_IMODE(PREVIOUS_RECEIPT.stat().st_mode) == 0o444,
          "LRT5 r1 receipt is not preserved immutable")
    _need(_sha(PREVIOUS_RECEIPT) == PREVIOUS_RECEIPT_SHA256, "LRT5 r1 receipt SHA drift")
    reference_path = hse5_receipt.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    paths = v1.index_heldin_calib(data_root.resolve())
    sessions = {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}
    body = lrt5.run_screen(sessions)
    body.update({
        "runtime_seconds": float(time.monotonic() - started),
        "baseline_reproduction": {
            **_hse5_reproduction(body, reference),
            "reference_path": str(reference_path), "reference_sha256": _sha(reference_path),
        },
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS), "dates": list(v1.H1_DATES),
            "files": [{"session": name, "path": str(paths[name]), "sha256": sessions[name].input_sha256}
                      for name in v1.H1_HELDIN_SESSIONS],
        },
        "implementation_binding": {
            "module_path": str(Path(lrt5.__file__).resolve()), "module_sha256": _sha(Path(lrt5.__file__).resolve()),
            "runner_path": str(Path(__file__).resolve()), "runner_sha256": _sha(Path(__file__).resolve()),
            "event_parser_path": str(Path(v1.__file__).resolve()), "event_parser_sha256": _sha(Path(v1.__file__).resolve()),
            "hse5_path": str(Path(hse5.__file__).resolve()), "hse5_sha256": _sha(Path(hse5.__file__).resolve()),
        },
        "scope": {
            **body["scope"], "native_position_endpoints_per_event": 2,
            "native_event_timestamps_per_event": 2, "within_event_position_trajectory_opened": False,
        },
        "receipt_integrity": {
            "publication_mode": "write-once-immutable-0444", "candidate_grid_literal_in_module": True,
            "execution_priority": "nice -n 19 / one CPU thread requested by run command",
        },
        "supersedes": {
            "previous_receipt_path": str(PREVIOUS_RECEIPT.resolve()),
            "previous_receipt_sha256": PREVIOUS_RECEIPT_SHA256,
            "previous_receipt_preserved_immutable": True,
            "previous_result_use": "invalidated for source-prior-control comparison only",
            "correction": (
                "r1 source-prior-only carrier used b=mean(Y) despite nonzero W; r2 uses "
                "b=mean(Y)-mean(Z)@W. Correct LRT5 and H-SE5 baseline/grid are unchanged."
            ),
            "candidate_grid_unchanged": True,
            "correct_lrt5_arm_unchanged": True,
        },
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--hse5-receipt", type=Path, default=DEFAULT_HSE5_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = run(args.data_root, args.hse5_receipt)
    output, digest = _publish_immutable(args.output, body)
    print(json.dumps({
        "status": body["status"], "output": str(output), "sha256": digest,
        "M3_gate": body["budgets"]["M3"]["gate"], "M4_gate": body["budgets"]["M4"]["gate"],
        "runtime_seconds": body["runtime_seconds"],
    }, indent=2, sort_keys=True))
    return 0 if body["status"] == "PASS_CPU_LRT5_MATERIAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
