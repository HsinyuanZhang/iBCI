#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the minimal A1 H/T4 pilot."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
STREAMING = REPO / "streaming_calibration_exp"
for root in (SUA, STREAMING):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from a1_hidden_carrier.a2_anchors import verify_sealed_a2_reuse  # noqa: E402
from a1_hidden_carrier.artifacts import write_immutable_json  # noqa: E402
from a1_hidden_carrier.contract import (  # noqa: E402
    ATTACHMENT_CONTROL,
    FRESH_TRAINING_FAMILIES,
    LOGICAL_CELLS,
    PILOT_SEED,
    SCREEN_ID,
)
from a1_hidden_carrier.cpu_proofs import run_cpu_proofs  # noqa: E402
from a1_hidden_carrier.evidence import (  # noqa: E402
    current_implementation_bindings,
    implementation_bindings_sha256,
)

DEFAULT_RESULT_ROOT = SUA / "results" / SCREEN_ID
DEFAULT_RECEIPT = DEFAULT_RESULT_ROOT / "official_cpu_preflight.json"


def blocker(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def build_receipt(
    *, result_root: Path, verify_checkpoint_bytes: bool, run_model_proofs: bool
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    bindings: dict[str, dict[str, str]] | None = None
    anchor: dict[str, object] | None = None
    proofs: dict[str, object] | None = None
    try:
        bindings = current_implementation_bindings()
    except Exception as exc:
        blockers.append(blocker("IMPLEMENTATION_BINDING_FAILURE", f"{type(exc).__name__}: {exc}"))
    try:
        anchor = verify_sealed_a2_reuse(
            seed=PILOT_SEED, verify_checkpoint_bytes=verify_checkpoint_bytes
        )
    except Exception as exc:
        blockers.append(blocker("SEALED_A2_REUSE_FAILURE", f"{type(exc).__name__}: {exc}"))
    if run_model_proofs:
        try:
            proofs = run_cpu_proofs()
            if proofs.get("all_passed") is not True:
                blockers.append(blocker("PRODUCTION_CPU_PROOFS_FAILED", json.dumps(proofs.get("proofs"), sort_keys=True)))
        except Exception as exc:
            blockers.append(blocker("PRODUCTION_CPU_PROOFS_ERROR", f"{type(exc).__name__}: {exc}"))
    else:
        blockers.append(blocker("MODEL_PROOFS_SKIPPED", "official preflight requires production CPU proofs"))

    result_root = result_root.expanduser().resolve()
    if result_root.exists():
        # The official preflight must be minted into a clean A1-v2 namespace.
        entries = [entry for entry in result_root.iterdir() if entry.name not in {"official_cpu_preflight.json", "official_cpu_preflight.json.sha256"}]
        if entries:
            blockers.append(blocker("NONEMPTY_A1_RESULT_ROOT", str(sorted(map(str, entries)))))
    run_dir = SUA / "checkpoints" / f"{SCREEN_ID}_h_t4_dandi688_co_s{PILOT_SEED}"
    if run_dir.exists():
        blockers.append(blocker("EXISTING_H_T4_RUN_DIRECTORY", str(run_dir)))

    return {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_official_preflight",
        "screen_id": SCREEN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CPU_PREFLIGHT_PASSED_AWAITING_ROOT_GO" if not blockers else "STOP_A1_PREFLIGHT_BLOCKERS",
        "non_authorizing_status": True,
        "authorizes_gpu": False,
        "cpu_only": True,
        "gpu_used": False,
        "training_started": False,
        "checkpoint_loaded_by_preflight": verify_checkpoint_bytes,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
        "logical_matrix": list(LOGICAL_CELLS),
        "fresh_training_families": list(FRESH_TRAINING_FAMILIES),
        "fresh_gpu_training_cells_in_pilot": 1,
        "pilot_seed": PILOT_SEED,
        "h_z4_structural_alias_of": "W/Z4",
        "attachment_control": ATTACHMENT_CONTROL,
        "attachment_control_training_cells": 0,
        "a2_reuse_evidence": anchor,
        "production_cpu_proofs": proofs,
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": implementation_bindings_sha256(bindings) if bindings else None,
        "implementation_blockers": blockers,
        "result_root": str(result_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--skip-checkpoint-bytes", action="store_true", help="test-only; causes blocker")
    parser.add_argument("--skip-model-proofs", action="store_true", help="test-only; causes blocker")
    args = parser.parse_args()
    receipt = build_receipt(
        result_root=args.result_root,
        verify_checkpoint_bytes=not args.skip_checkpoint_bytes,
        run_model_proofs=not args.skip_model_proofs,
    )
    if args.skip_checkpoint_bytes:
        receipt["implementation_blockers"].append(
            blocker("CHECKPOINT_BYTE_AUDIT_SKIPPED", "official preflight requires all 16 A2 checkpoint payload audits")
        )
        receipt["status"] = "STOP_A1_PREFLIGHT_BLOCKERS"
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.receipt:
        body, sidecar, digest = write_immutable_json(args.receipt, receipt)
        print(json.dumps({"receipt": str(body), "sidecar": str(sidecar), "sha256": digest, "status": receipt["status"]}, sort_keys=True))
    else:
        print(text, end="")
    if receipt["implementation_blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
