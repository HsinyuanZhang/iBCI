#!/usr/bin/env python3
"""Publish the one-shot CPU-only H1 PNO5 source-screen receipt.

Run under ``nice -n 19``; this process also fixes BLAS/OpenMP thread counts to
one before importing NumPy-backed H1 code.  It opens only the public held-in
calibration NWBs and refuses to overwrite a result.
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

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_event_carrier_pno5 as pno5  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1  # noqa: E402
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5  # noqa: E402


DEFAULT_DATA = ROOT / "SPINT-main/data/000954"
DEFAULT_HSE5 = ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json"
DEFAULT_PREDECLARATION = ROOT / "sua_exploration/results/h1_event_carrier_pno5/predeclaration.json"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/h1_event_carrier_pno5/source_screen_v2.json"


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reproduce(body: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    _need(reference.get("schema") == hse5.SCHEMA, "PNO5 H-SE5 reference schema mismatch")
    pairs = {"median_r2_correct": "median_r2_correct", "median_delta_label_shuffle": "median_delta_shuffle",
             "median_delta_intercept": "median_delta_intercept"}
    deltas: list[float] = []
    for budget in pno5.SUPPORT_BUDGETS:
        observed, expected = body["budgets"][f"M{budget}"]["hse5_baseline_sessions"], reference["budgets"][f"M{budget}"]["sessions"]
        for session in v1.H1_HELDIN_SESSIONS:
            for left, right in pairs.items():
                deltas.append(abs(float(observed[session][left]) - float(expected[session]["forward"][right])))
    maximum = max(deltas); _need(maximum <= 1.0e-10, f"PNO5 H-SE5 exact reproduction drift {maximum}")
    return {"passed": True, "comparisons": len(deltas), "absolute_tolerance": 1.0e-10, "maximum_absolute_difference": maximum}


def _publish(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve(); sidecar = output.with_suffix(output.suffix + ".sha256")
    _need(not output.exists() and not sidecar.exists(), f"refusing to overwrite PNO5 receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444); os.link(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "PNO5 receipt chmod drift")
    digest = _sha(output)
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(f"{digest}  {output.name}\n"); handle.flush(); os.fsync(handle.fileno())
    _need(stat.S_IMODE(sidecar.stat().st_mode) == 0o444, "PNO5 sidecar chmod drift")
    return output, digest


def run(data_root: Path, hse5_receipt: Path) -> dict[str, Any]:
    started = time.monotonic(); reference_path = hse5_receipt.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    paths = v1.index_heldin_calib(data_root.resolve())
    sessions = {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}
    body = pno5.run_screen(sessions)
    body.update({"runtime_seconds": float(time.monotonic() - started), "baseline_reproduction": {**_reproduce(body, reference),
                 "reference_path": str(reference_path), "reference_sha256": _sha(reference_path)},
                 "source_binding": {"sessions": list(v1.H1_HELDIN_SESSIONS), "dates": list(v1.H1_DATES),
                    "files": [{"session": name, "path": str(paths[name]), "sha256": sessions[name].input_sha256} for name in v1.H1_HELDIN_SESSIONS]},
                 "implementation_binding": {"module_path": str(Path(pno5.__file__).resolve()), "module_sha256": _sha(Path(pno5.__file__).resolve()),
                    "runner_path": str(Path(__file__).resolve()), "runner_sha256": _sha(Path(__file__).resolve()),
                    "event_parser_path": str(Path(v1.__file__).resolve()), "event_parser_sha256": _sha(Path(v1.__file__).resolve()),
                    "hse5_path": str(Path(hse5.__file__).resolve()), "hse5_sha256": _sha(Path(hse5.__file__).resolve())},
                 "receipt_integrity": {"publication_mode": "write-once-immutable-0444", "requested_threads": {key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                    "execution_priority": "nice -n 19"}})
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA); parser.add_argument("--hse5-receipt", type=Path, default=DEFAULT_HSE5)
    parser.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args()
    # This no-data declaration is published before the runner indexes even one
    # NWB.  Its write-once SHA becomes a binding of the subsequent receipt.
    predeclaration, predeclaration_sha = _publish(args.predeclaration, {
        "schema": "h1_event_carrier_pno5_predeclaration_v1", "protocol": pno5.PROTOCOL,
        "candidate_matrix_predeclared_before_data_run": True, "predeclaration": dict(pno5.PREDECLARATION),
        "frozen_constants": {"rank": pno5.RANK, "carrier_dim": pno5.CARRIER_DIM,
                             "support_budgets": list(pno5.SUPPORT_BUDGETS), "model_grid": []},
        "data_opened_before_declaration": False,
    })
    body = run(args.data_root, args.hse5_receipt)
    body["predeclaration_binding"] = {"path": str(predeclaration), "sha256": predeclaration_sha,
                                       "sidecar_path": str(predeclaration.with_suffix(predeclaration.suffix + ".sha256")),
                                       "published_before_nwb_open": True}
    output, digest = _publish(args.output, body)
    print(json.dumps({"status": body["status"], "receipt": str(output), "sha256": digest,
                      "M3_gate": body["budgets"]["M3"]["gate"], "M4_gate": body["budgets"]["M4"]["gate"], "runtime_seconds": body["runtime_seconds"]}, indent=2, sort_keys=True))
    return 0 if body["status"] == "PASS_CPU_PNO5_MATERIAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
