#!/usr/bin/env python3
"""Read-only supplemental CPU closure for audited H1 sparse-event routes.

This does not modify or replace the historic GPU-program preflight.  It binds
that preflight by SHA and closes only the subsequently audited CPU candidate
routes.  It never constructs a decoder/trainer or invokes a GPU path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sua_exploration/results"
SCRIPTS = ROOT / "sua_exploration/scripts"
PREFLIGHT = RESULTS / "h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_PROGRAM_PREFLIGHT_v1.json"
PREFLIGHT_SHA = "18f02403141c2196874e84a02c0267a8bdf2a5b4127857f485fd1b51141456ae"
OUTPUT = RESULTS / "h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v1.json"

EXPECTED = {
    "hse5_v5r1_historical": (RESULTS / "h1_sparse_event_endpoint_v2/source_audit.json", "e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f", "PASS_CPU_HSE5_M3_GPU_READY"),
    "hse5_v5r2_authoritative": (RESULTS / "h1_sparse_event_endpoint_v2/source_audit_v2r2.json", "de4c23ac3fd21f96c68e54b5190538189543be6f7b19e21cb5533665872a28a4", "PASS_CPU_HSE5_M3_GPU_READY"),
    "c2f5_predeclaration": (RESULTS / "h1_calibration_future_correction_c2f5/C2F5_PREDECLARATION_v1.json", "c05ae8d1489846a893b1e17d12164b99a2f2fdbc3bd5f2b91c446146c703e9a2", None),
    "c2f5_receipt": (RESULTS / "h1_calibration_future_correction_c2f5/source_screen.json", "4096221bbfaf0c7c3c39e50a59e47073f48d60d6d8fcc03ea289eea413d829e8", "STOP_CPU_C2F5_NOT_MATERIAL"),
    "lrt5r1_historical": (RESULTS / "h1_event_carrier_lrt5/source_screen_v1.json", "f19e333dc1a57830593d380778f5bef10cf41a5bb1ecd7717c21f386ae25bc3a", "STOP_CPU_LRT5_NOT_MATERIAL"),
    "lrt5r2_authoritative": (RESULTS / "h1_event_carrier_lrt5/source_screen_v2.json", "cf4a8a6256a9743a8cc0ee26eef62debae28a4a51db48e8ea485063a32ed4729", "STOP_CPU_LRT5_NOT_MATERIAL"),
    "nle5r1_predeclaration_historical": (RESULTS / "h1_event_carrier_nle5/predeclaration_v1.json", "51cb49abdb0f706af9ccdcb86252d9f0e71238f0f093a8e8566592a42f96abe8", None),
    "nle5r1_historical_structural_stop": (RESULTS / "h1_event_carrier_nle5/source_screen_v1.json", "b2e40498f202f10ef3536b5b11d5e83ce06b4afaeb6e0cf041185bca6284fa41", "STOP_CPU_NLE5_UNIDENTIFIED_SOURCE_TAG_SUPPORT"),
    "nle5r2_predeclaration": (RESULTS / "h1_event_carrier_nle5/predeclaration_v2.json", "15824e841e7283c1a02734e75bb38360177b869df8650c165a06fa6c6603824c", None),
    "nle5r2_authoritative": (RESULTS / "h1_event_carrier_nle5/source_screen_v2.json", "45f346decbda51742235efd63cf2feecbdb2650c2a2a141aaa1769117abef3d9", "STOP_CPU_NLE5_V2_NOT_MATERIAL"),
    "pno5_predeclaration": (RESULTS / "h1_event_carrier_pno5/predeclaration.json", "8d5238b3ee546718cc980308ed170271103ec3555498cea51796eaca802439d7", None),
    "pno5_receipt": (RESULTS / "h1_event_carrier_pno5/source_screen_v2.json", "0ad94ca988b92a23b69e5be2d0f3ce42aca41c4d2e4536d3f0ef123cf7fa4f52", "STOP_CPU_PNO5_NOT_MATERIAL"),
    "qc2f5_predeclaration": (RESULTS / "h1_calibration_future_quadratic_c2f5/QC2F5_PREDECLARATION_v1.json", "292a25992e3775357840da961381b50874f09fae78734ae666a87c1ecf641d90", None),
    "qc2f5_receipt": (RESULTS / "h1_calibration_future_quadratic_c2f5/source_screen.json", "ef8b216c90046ca465c4bd687ea0737eb261c18caa1d787a657feb84a24fa51f", "STOP_CPU_QC2F5_NOT_MATERIAL"),
    "qc2f5_metadata_correction": (RESULTS / "h1_calibration_future_quadratic_c2f5/QC2F5_METADATA_CORRECTION_v1.json", "5d66a23179bd16adba96a2f224dd4b8d9e80ba389426ebbfed1f9d53cd22e180", None),
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable(path: Path, expected_sha: str) -> Mapping[str, Any]:
    need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"immutable mode/path drift: {path}")
    need(sha(path) == expected_sha, f"SHA drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def call(script: str, *arguments: str) -> Mapping[str, Any]:
    """Invoke existing independent receipt verifiers; no model/data execution."""

    result = subprocess.run([sys.executable, str(SCRIPTS / script), *arguments], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    body = json.loads(result.stdout)
    need(body.get("status") in {"PASS", "PASS_FAIL_CLOSED"}, f"independent verifier did not pass: {script}")
    return body


def _hse_reproduction(body: Mapping[str, Any], label: str) -> None:
    reproduction = body["baseline_reproduction"]
    if isinstance(reproduction, Mapping) and "M3" in reproduction:
        for budget in ("M3", "M4"):
            item = reproduction[budget]
            need(item["passed"] is True and float(item["maximum_absolute_difference"]) <= 1e-10,
                 f"{label} H-SE exact reproduction drift {budget}")
    else:
        need(reproduction["passed"] is True and float(reproduction["maximum_absolute_difference"]) <= 1e-10,
             f"{label} H-SE exact reproduction drift")


def audit() -> dict[str, Any]:
    """Validate all bound immutable artifacts and invoke all route verifiers."""

    preflight = immutable(PREFLIGHT, PREFLIGHT_SHA)
    need(preflight["status"] == "PASS_PREFLIGHT_TERMINAL_PENDING", "historic program preflight status drift")
    bodies = {name: immutable(path, digest) for name, (path, digest, _status) in EXPECTED.items()}
    for name, (_path, _digest, status) in EXPECTED.items():
        if status is not None:
            need(bodies[name]["status"] == status, f"terminal status drift: {name}")
    # Sidecar-bearing legacy routes retain their immutable companion hashes.
    for name in ("hse5_v5r2_authoritative", "lrt5r2_authoritative", "nle5r2_authoritative", "pno5_receipt"):
        path, digest, _ = EXPECTED[name]
        sidecar = path.with_suffix(path.suffix + ".sha256")
        need(sidecar.is_file() and stat.S_IMODE(sidecar.stat().st_mode) == 0o444 and sidecar.read_text(encoding="ascii").split()[0] == digest,
             f"sidecar drift: {name}")

    verifier_results = {
        "hse5_v5r1_historical": call("verify_h1_sparse_event_endpoint_v2_receipt.py", str(EXPECTED["hse5_v5r1_historical"][0])),
        "hse5_v5r2_accounting": call("verify_h1_sparse_event_endpoint_v2r2_accounting.py", str(EXPECTED["hse5_v5r2_authoritative"][0])),
        "c2f5": call("verify_h1_calibration_future_correction_c2f5.py", str(EXPECTED["c2f5_receipt"][0])),
        "lrt5r2": call("verify_h1_event_carrier_lrt5.py", str(EXPECTED["lrt5r2_authoritative"][0])),
        "nle5r1_historical": call("verify_h1_event_carrier_nle5.py", str(EXPECTED["nle5r1_historical_structural_stop"][0])),
        "nle5r2": call("verify_h1_event_carrier_nle5_v2.py", str(EXPECTED["nle5r2_authoritative"][0])),
        "pno5": call("verify_h1_event_carrier_pno5.py", str(EXPECTED["pno5_receipt"][0]), "--no-recompute"),
        "qc2f5": call("verify_h1_calibration_future_quadratic_c2f5.py", str(EXPECTED["qc2f5_receipt"][0])),
        "qc2f5_metadata_correction": call("verify_h1_calibration_future_quadratic_c2f5_metadata_correction.py", str(EXPECTED["qc2f5_metadata_correction"][0])),
    }
    for name in ("c2f5_receipt", "lrt5r2_authoritative", "nle5r2_authoritative", "pno5_receipt", "qc2f5_receipt"):
        _hse_reproduction(bodies[name], name)
    need(bodies["qc2f5_metadata_correction"]["no_data_gpu_retraining_or_reevaluation"]["gpu_used"] is False,
         "QC2F5 correction GPU scope drift")
    return {
        "schema": "h1_sparse_event_endpoint_extended_cpu_closure_v1",
        "artifact_role": "supplemental_cpu_closure_not_a_rewrite_of_historic_gpu_program_preflight",
        "historic_gpu_program_preflight": {"path": str(PREFLIGHT), "sha256": PREFLIGHT_SHA, "status": preflight["status"], "modified": False},
        "audited_artifacts": {name: {"path": str(path), "sha256": digest, "immutable_mode": "0444", "terminal_status": status}
                              for name, (path, digest, status) in EXPECTED.items()},
        "independent_verifier_results": verifier_results,
        "route_final_statuses": {
            "HSE5_V5r2": bodies["hse5_v5r2_authoritative"]["status"],
            "C2F5": bodies["c2f5_receipt"]["status"], "LRT5r2": bodies["lrt5r2_authoritative"]["status"],
            "NLE5r2": bodies["nle5r2_authoritative"]["status"], "PNO5": bodies["pno5_receipt"]["status"],
            "QC2F5": bodies["qc2f5_receipt"]["status"],
        },
        "historical_only_non_authoritative_performance_receipts": {
            "HSE5_V5r1": "preserved predecessor; V5r2 accounting receipt is authoritative",
            "LRT5r1": "preserved prior-control receipt; LRT5r2 supersession is authoritative",
            "NLE5r1": "preserved structural source-tag-support STOP; NLE5r2 is authoritative performance screen",
        },
        "new_gpu_arms_authorized": 0,
        "scope": {"source_nwb_opened": 0, "gpu_used": False, "decoder_constructed": False, "trainer_constructed": False,
                  "retraining": False, "outer_score_reevaluation": False},
        "status": "PASS_EXTENDED_CPU_CLOSURE_NO_NEW_GPU_ARMS",
    }


def immutable_write_once(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    path = path.resolve()
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        need(stat.S_IMODE(path.stat().st_mode) == 0o444 and path.read_bytes() == encoded,
             f"refusing to alter immutable closure: {path}")
        return path, sha(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o444)
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, "closure immutable mode drift")
    return path, hashlib.sha256(encoded).hexdigest()


def verify_closure(path: Path) -> dict[str, Any]:
    path = path.resolve()
    need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444, "closure artifact must be immutable")
    expected = audit()
    need(json.loads(path.read_text(encoding="utf-8")) == expected, "closure artifact content drift")
    return {"status": "PASS", "closure": str(path), "closure_sha256": sha(path), "new_gpu_arms_authorized": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_closure(args.output)
    else:
        output, digest = immutable_write_once(args.output, audit())
        result = {"status": "PASS", "closure": str(output), "closure_sha256": digest, "new_gpu_arms_authorized": 0}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
