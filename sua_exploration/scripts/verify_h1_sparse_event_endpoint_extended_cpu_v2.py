#!/usr/bin/env python3
"""Correctly named supplemental CPU closure, v2 (read-only, no GPU/recompute)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.scripts import verify_h1_sparse_event_endpoint_extended_cpu as v1


ROOT = v1.ROOT
RESULTS = v1.RESULTS
PREFLIGHT = v1.PREFLIGHT
PREFLIGHT_SHA = v1.PREFLIGHT_SHA
CLOSURE_V1 = RESULTS / "h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v1.json"
CLOSURE_V1_SHA = "39c2b02d6ba963ea1e4ba67c6f5d56476a26575c8e3ab3e043962939f67786a3"
OUTPUT = RESULTS / "h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v2.json"

# These are intentionally named after their actual schemas: sparse endpoint
# V2/V2r2, rather than the distinct event-carrier estimator V5 receipts.
ARTIFACTS = {
    "hse5_v2_source_audit": (RESULTS / "h1_sparse_event_endpoint_v2/source_audit.json", "e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f", "PASS_CPU_HSE5_M3_GPU_READY"),
    "hse5_v2r2_label_accounting": (RESULTS / "h1_sparse_event_endpoint_v2/source_audit_v2r2.json", "de4c23ac3fd21f96c68e54b5190538189543be6f7b19e21cb5533665872a28a4", "PASS_CPU_HSE5_M3_GPU_READY"),
    "estimator_v5r1_historical_invalidated": (RESULTS / "h1_event_carrier_estimator_v5/source_screen.json", "a50ff3706c3b2bbc922d83bae09fc6b8cb4dee6d8eae74989a4a183af29eeb86", "STOP_CPU_ESTIMATOR_CANDIDATES_NOT_MATERIAL"),
    "estimator_v5r2_authoritative": (RESULTS / "h1_event_carrier_estimator_v5/source_screen_v2.json", "6fa4405e29abc478a64f1c03619f89f1bcf31e1a77db314be014b241764d2645", "STOP_CPU_ESTIMATOR_CANDIDATES_NOT_MATERIAL"),
    **{key: value for key, value in v1.EXPECTED.items() if key not in {"hse5_v5r1_historical", "hse5_v5r2_authoritative"}},
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_sidecar(name: str) -> None:
    path, digest, _status = ARTIFACTS[name]
    sidecar = path.with_suffix(path.suffix + ".sha256")
    need(sidecar.is_file() and stat.S_IMODE(sidecar.stat().st_mode) == 0o444 and sidecar.read_text(encoding="ascii").split()[0] == digest,
         f"sidecar drift: {name}")


def audit() -> dict[str, Any]:
    """Invoke existing independent verifiers against correctly named receipts."""

    preflight = v1.immutable(PREFLIGHT, PREFLIGHT_SHA)
    prior = v1.immutable(CLOSURE_V1, CLOSURE_V1_SHA)
    need(preflight["status"] == "PASS_PREFLIGHT_TERMINAL_PENDING", "historic GPU preflight drift")
    need(prior["status"] == "PASS_EXTENDED_CPU_CLOSURE_NO_NEW_GPU_ARMS", "closure v1 drift")
    bodies = {name: v1.immutable(path, digest) for name, (path, digest, _status) in ARTIFACTS.items()}
    for name, (_path, _digest, expected_status) in ARTIFACTS.items():
        if expected_status is not None:
            need(bodies[name]["status"] == expected_status, f"terminal status drift: {name}")
    for name in ("hse5_v2_source_audit", "hse5_v2r2_label_accounting", "lrt5r2_authoritative", "nle5r2_authoritative", "pno5_receipt"):
        _verify_sidecar(name)

    verifier_results = {
        "hse5_v2_source_audit": v1.call("verify_h1_sparse_event_endpoint_v2_receipt.py", str(ARTIFACTS["hse5_v2_source_audit"][0])),
        "hse5_v2r2_label_accounting": v1.call("verify_h1_sparse_event_endpoint_v2r2_accounting.py", str(ARTIFACTS["hse5_v2r2_label_accounting"][0])),
        "estimator_v5r2": v1.call("verify_h1_event_carrier_estimator_v5.py", str(ARTIFACTS["estimator_v5r2_authoritative"][0])),
        "c2f5": v1.call("verify_h1_calibration_future_correction_c2f5.py", str(ARTIFACTS["c2f5_receipt"][0])),
        "lrt5r2": v1.call("verify_h1_event_carrier_lrt5.py", str(ARTIFACTS["lrt5r2_authoritative"][0])),
        "nle5r1_historical": v1.call("verify_h1_event_carrier_nle5.py", str(ARTIFACTS["nle5r1_historical_structural_stop"][0])),
        "nle5r2": v1.call("verify_h1_event_carrier_nle5_v2.py", str(ARTIFACTS["nle5r2_authoritative"][0])),
        "pno5": v1.call("verify_h1_event_carrier_pno5.py", str(ARTIFACTS["pno5_receipt"][0]), "--no-recompute"),
        "qc2f5": v1.call("verify_h1_calibration_future_quadratic_c2f5.py", str(ARTIFACTS["qc2f5_receipt"][0])),
        "qc2f5_metadata_correction": v1.call("verify_h1_calibration_future_quadratic_c2f5_metadata_correction.py", str(ARTIFACTS["qc2f5_metadata_correction"][0])),
    }
    for name in ("estimator_v5r1_historical_invalidated", "estimator_v5r2_authoritative", "c2f5_receipt", "lrt5r2_authoritative", "nle5r2_authoritative", "pno5_receipt", "qc2f5_receipt"):
        v1._hse_reproduction(bodies[name], name)
    supersedes = bodies["estimator_v5r2_authoritative"]["supersedes"]
    need(supersedes["invalidated_receipt_sha256"] == ARTIFACTS["estimator_v5r1_historical_invalidated"][1], "estimator V5r1 invalidation binding drift")
    need("double-applying the exposure offset" in supersedes["invalidated_reason"], "estimator V5r1 offset invalidation drift")
    return {
        "schema": "h1_sparse_event_endpoint_extended_cpu_closure_v2",
        "artifact_role": "supplemental_cpu_closure_corrected_nomenclature_and_estimator_v5_coverage",
        "supersedes": {"closure_v1_path": str(CLOSURE_V1), "closure_v1_sha256": CLOSURE_V1_SHA,
                       "closure_v1_preserved_unmodified": True,
                       "reason": "closure v1 mislabeled sparse-endpoint V2/V2r2 as V5 and omitted true estimator V5 receipts"},
        "historic_gpu_program_preflight": {"path": str(PREFLIGHT), "sha256": PREFLIGHT_SHA, "status": preflight["status"], "modified": False},
        "audited_artifacts": {name: {"path": str(path), "sha256": digest, "immutable_mode": "0444", "terminal_status": status}
                              for name, (path, digest, status) in ARTIFACTS.items()},
        "hse5_v2_authority": {
            "source_audit": "hse5_v2_source_audit is the authoritative metric/provenance receipt",
            "label_accounting": "hse5_v2r2_label_accounting is the complementary authoritative projected-label accounting receipt",
            "not_estimator_v5": True,
            "not_historical_invalid": True,
        },
        "independent_verifier_results": verifier_results,
        "route_final_statuses": {
            "HSE5_V2_SOURCE_GATE": bodies["hse5_v2_source_audit"]["status"],
            "HSE5_V2R2_LABEL_ACCOUNTING": bodies["hse5_v2r2_label_accounting"]["status"],
            "ESTIMATOR_V5R2": bodies["estimator_v5r2_authoritative"]["status"],
            "C2F5": bodies["c2f5_receipt"]["status"], "LRT5r2": bodies["lrt5r2_authoritative"]["status"],
            "NLE5r2": bodies["nle5r2_authoritative"]["status"], "PNO5": bodies["pno5_receipt"]["status"], "QC2F5": bodies["qc2f5_receipt"]["status"],
        },
        "historical_only_non_authoritative_performance_receipts": {
            "ESTIMATOR_V5R1": "preserved but invalidated: double exposure-offset subtraction and wrong covariance; its numbers are non-authoritative",
            "LRT5r1": "preserved prior-control receipt; LRT5r2 supersession is authoritative",
            "NLE5r1": "preserved structural source-tag-support STOP; NLE5r2 is authoritative performance screen",
        },
        "new_gpu_arms_authorized": 0,
        "scope": {"source_nwb_opened": 0, "gpu_used": False, "decoder_constructed": False, "trainer_constructed": False,
                  "retraining": False, "outer_score_reevaluation": False},
        "status": "PASS_EXTENDED_CPU_CLOSURE_V2_NO_NEW_GPU_ARMS",
    }


def immutable_write_once(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    path = path.resolve()
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists():
        need(stat.S_IMODE(path.stat().st_mode) == 0o444 and path.read_bytes() == encoded, f"refusing immutable v2 closure change: {path}")
        return path, v1.sha(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o444)
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, "v2 closure mode drift")
    return path, hashlib.sha256(encoded).hexdigest()


def verify_closure(path: Path) -> dict[str, Any]:
    path = path.resolve()
    need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444, "v2 closure must be immutable")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = audit()
    need(observed == expected, "v2 closure content drift")
    need("estimator_v5r2_authoritative" in observed["audited_artifacts"], "missing estimator V5r2")
    need("hse5_v2_source_audit" in observed["audited_artifacts"] and "hse5_v5r" not in json.dumps(observed), "HSE V2 mislabeled")
    return {"status": "PASS", "closure": str(path), "closure_sha256": v1.sha(path), "new_gpu_arms_authorized": 0}


def main() -> None:
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
