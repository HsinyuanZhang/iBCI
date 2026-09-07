#!/usr/bin/env python3
"""Final program-level verifier for the H1 sparse movement-event endpoint work.

This script is intentionally terminal-gated.  It combines the independently
verified source estimator, four CPU design screens, matched H-SE5/Zero5 GPU
receipt, structural checkpoint audit, and independent prediction-level R2
recomputation.  A scientifically negative first cell may still complete the
design/implementation objective, but missing or inconsistent evidence may not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT / "SPINT-main"
SCRIPTS = ROOT / "sua_exploration/scripts"
RESULTS = ROOT / "sua_exploration/results"
PILOT = SPINT / "pilot_artifacts/h1_sparse_event_endpoint"

EXPECTED = {
    "v1": "de50d18d6905affa73267c557c76b8fcbd52e7667eb5b1ad4a2546ad7a9f148c",
    "v2": "e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f",
    "v2r2": "de4c23ac3fd21f96c68e54b5190538189543be6f7b19e21cb5533665872a28a4",
    "screen_v1": "74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3",
    "screen_v2": "2c273cd78b20b4dc1a43d197588ac5a2b58823d093514196c67069e23584ae2c",
    "screen_v3": "e52a7cdd6c18c9d39af3d3cfddddd9f340d13bb5e4184feaa8a7f3a92fd80933",
    "screen_v4": "5a96b20493b45f448446e44005549f4d1611a11de292299be0c6bb872475f0f3",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, label: str) -> Mapping[str, Any]:
    resolved = path.resolve()
    need(resolved.is_file(), f"missing {label}: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def run_verifier(script: Path, arguments: Sequence[Path]) -> Mapping[str, Any]:
    command = [sys.executable, str(script.resolve()), *(str(path.resolve()) for path in arguments)]
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    body = json.loads(result.stdout)
    need(body.get("status") == "PASS", f"verifier did not pass: {script.name}")
    return body


def close(first: Any, second: Any, label: str, tolerance: float = 2.0e-6) -> None:
    need(math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=tolerance),
         f"{label}: {first!r} != {second!r}")


def verify_static() -> dict[str, Any]:
    paths = {
        "v1": RESULTS / "h1_sparse_event_endpoint_v1/source_audit.json",
        "v2": RESULTS / "h1_sparse_event_endpoint_v2/source_audit.json",
        "v2r2": RESULTS / "h1_sparse_event_endpoint_v2/source_audit_v2r2.json",
        "screen_v1": RESULTS / "h1_event_carrier_design_screen_v1/source_screen.json",
        "screen_v2": RESULTS / "h1_event_carrier_nested_context_v2/source_screen.json",
        "screen_v3": RESULTS / "h1_event_carrier_meta_basis_v3/source_screen.json",
        "screen_v4": RESULTS / "h1_event_carrier_semantic_v4/source_screen.json",
    }
    for name, path in paths.items():
        need(path.is_file() and file_sha(path) == EXPECTED[name], f"{name} receipt SHA drift")

    source_verifiers = {
        "v1": run_verifier(
            SCRIPTS / "verify_h1_sparse_event_endpoint_receipt.py", [paths["v1"]],
        ),
        "v2": run_verifier(
            SCRIPTS / "verify_h1_sparse_event_endpoint_v2_receipt.py", [paths["v2"]],
        ),
        "v2r2": run_verifier(
            SCRIPTS / "verify_h1_sparse_event_endpoint_v2r2_accounting.py", [paths["v2r2"]],
        ),
        "screens_v1_v3": run_verifier(
            SCRIPTS / "verify_h1_event_carrier_cpu_screens.py",
            [paths["screen_v1"], paths["screen_v2"], paths["screen_v3"]],
        ),
        "screen_v4": run_verifier(
            SCRIPTS / "verify_h1_event_carrier_semantic_v4.py", [paths["screen_v4"]],
        ),
    }
    return {
        "status": "PASS_PREFLIGHT_TERMINAL_PENDING",
        "source_and_cpu_verifiers": source_verifiers,
        "bound_receipts": {
            name: {"path": str(path.resolve()), "sha256": EXPECTED[name]}
            for name, path in paths.items()
        },
        "pending_terminal_artifacts": [
            "matched H-SE5/Zero5 terminal receipt",
            "independent structural checkpoint audit",
            "independent batch-29 prediction R2 recomputation",
        ],
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    static = verify_static()

    terminal_path = args.terminal.resolve()
    structural_path = args.structural.resolve()
    recomputation_path = args.recomputation.resolve()
    terminal = load(terminal_path, "GPU terminal receipt")
    structural = load(structural_path, "GPU structural verification")
    recomputation = load(recomputation_path, "independent R2 recomputation")
    need(stat.S_IMODE(terminal_path.stat().st_mode) == 0o444, "terminal receipt is not immutable")
    need(stat.S_IMODE(structural_path.stat().st_mode) == 0o444, "structural audit is not immutable")
    need(stat.S_IMODE(recomputation_path.stat().st_mode) == 0o444, "R2 recomputation is not immutable")

    live_structural = run_verifier(
        SCRIPTS / "verify_h1_sparse_event_endpoint_gpu_receipt.py", [terminal_path],
    )
    terminal_sha = file_sha(terminal_path)
    need(structural.get("status") == "PASS" and structural.get("sha256") == terminal_sha,
         "stored structural audit does not bind terminal receipt")
    need(live_structural.get("sha256") == terminal_sha, "live structural verifier SHA mismatch")
    for key in ("scientific_status", "hse5_r2", "gate_pass", "margins"):
        need(live_structural.get(key) == structural.get(key), f"stored/live structural drift at {key}")

    need(recomputation.get("status") == "PASS", "independent R2 recomputation did not pass")
    need(Path(recomputation["receipt"]).resolve() == terminal_path, "R2 recomputation receipt path drift")
    need(recomputation.get("independent_batch_size") == 29, "independent batch size drift")
    need(0 < float(recomputation.get("absolute_r2_tolerance", 0)) <= 2.0e-6,
         "independent R2 tolerance drift")
    expected_metrics = terminal["metrics"]
    full_expected = expected_metrics["hse5_same_checkpoint_interventions"]["full"]
    zero_expected = expected_metrics["separately_trained_zero5"]
    for observed, expected, label in (
        (recomputation["full"], full_expected, "Full"),
        (recomputation["zero"], zero_expected, "Zero5"),
    ):
        need(observed["state_immutable"] is True, f"{label}: state changed in independent scoring")
        need(observed["query_window_indices_sha256"] == expected["query_window_indices_sha256"],
             f"{label}: query SHA drift")
        need(observed["samples"] == expected["samples"] == 8965, f"{label}: sample count drift")
        close(observed["pooled_r2"], expected["pooled_r2"], f"{label} pooled R2")
        for session in expected["per_session"]:
            need(observed["per_session"][session]["samples"] == expected["per_session"][session]["samples"],
                 f"{label}/{session}: sample drift")
            close(observed["per_session"][session]["r2"], expected["per_session"][session]["r2"],
                  f"{label}/{session} R2")
    close(
        recomputation["full_minus_zero"],
        float(full_expected["pooled_r2"]) - float(zero_expected["pooled_r2"]),
        "Full-Zero5",
    )

    need(terminal["scope"]["target_optimizer_steps"] == 0, "target optimizer steps are nonzero")
    need(terminal["scope"]["target_backward_steps"] == 0, "target backward steps are nonzero")
    need(terminal["scope"]["formal_opened"] is False and terminal["scope"]["heldout_opened"] is False,
         "forbidden target scope opened")
    need(terminal["source_manifest"]["deployment_carrier_dense_velocity_opened"] is False,
         "deployment carrier opened dense velocity")
    need(terminal["source_manifest"]["carrier_dim"] == 5, "terminal carrier width drift")

    scientific_pass = bool(terminal["gate"]["pass"])
    expected_scientific = "PASS_HSE5_FIRST_CELL" if scientific_pass else "STOP_HSE5_FIRST_CELL"
    need(terminal["status"] == expected_scientific, "terminal scientific status drift")
    return {
        "status": "PASS",
        "program_status": (
            "IMPLEMENTATION_COMPLETE_SCIENTIFIC_FIRST_CELL_PASS"
            if scientific_pass else "IMPLEMENTATION_COMPLETE_SCIENTIFIC_FIRST_CELL_STOP"
        ),
        "scientific_status": terminal["status"],
        "terminal_receipt": str(terminal_path),
        "terminal_receipt_sha256": terminal_sha,
        "hse5_r2": float(full_expected["pooled_r2"]),
        "zero5_r2": float(zero_expected["pooled_r2"]),
        "hse5_minus_zero5": float(full_expected["pooled_r2"]) - float(zero_expected["pooled_r2"]),
        "source_and_cpu_verifiers": static["source_and_cpu_verifiers"],
        "bound_source_and_cpu_receipts": static["bound_receipts"],
        "program_verifier_sha256": file_sha(Path(__file__).resolve()),
        "structural_verifier_sha256": file_sha(structural_path),
        "r2_recomputation_sha256": file_sha(recomputation_path),
        "requirements": {
            "one_observation_per_valid_movement_event": True,
            "endpoint_displacement_low_rank_q4": True,
            "no_eight_phase_profile_requirement": True,
            "five_value_per_channel_carrier": True,
            "target_session_backward_steps": 0,
            "dense_velocity_not_used_by_deployment_carrier": True,
            "matched_query_and_independent_zero_control": True,
            "independent_prediction_recomputation": True,
        },
    }


def write_immutable(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite program audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    return output, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--terminal", type=Path,
        default=PILOT / "H1_SE5_M4_FOLD0_TERMINAL_v1.json",
    )
    parser.add_argument(
        "--structural", type=Path,
        default=PILOT / "H1_SE5_M4_FOLD0_TERMINAL_v1_verification.json",
    )
    parser.add_argument(
        "--recomputation", type=Path,
        default=PILOT / "H1_SE5_M4_FOLD0_TERMINAL_v1_recomputed_r2.json",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="verify all source/CPU evidence and report the three expected terminal artifacts as pending",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    body = verify_static() if args.preflight_only else verify(args)
    body = {**body, "program_verifier_sha256": file_sha(Path(__file__).resolve())}
    if args.output is not None:
        output, digest = write_immutable(args.output, body)
        body = {**body, "program_audit": str(output), "program_audit_sha256": digest}
    print(json.dumps(body, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
