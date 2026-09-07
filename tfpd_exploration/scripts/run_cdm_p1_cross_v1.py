#!/usr/bin/env python
"""CDM x P1 cross-dataset factorial, Part A runner: attempt -> replay -> terminal.

Work order: ``docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md`` section 2
(Part A only; Part B is a separate work order and is NOT started here).

Stages
------
``attempt``     CPU only, before any data/model/CUDA access: reserve the fresh
                result root, pin the owned module bytes, the work order, every
                immutable predecessor (the full frozen Stage-P chain and
                receipts, the sealed cal_aug_v1 C1 SWA artifact and its sealed
                deployment/training terminals) by SHA-256, re-verify the sealed
                Stage-P promotion anchor (P1@m4) and the sealed C1 bindings,
                and pre-register the F00/F10/F01/F11 cells, the sealed P1
                hyperparameters, the weight-swap law and the section 2 gates.
``replay``      one process on GPU 1 ONLY: materialize the frozen activity-only
                inputs once (identical trial order for every cell), run F00 and
                F01 under the sealed weights (both must reproduce the sealed
                Stage-P rows bit-exactly), swap in the C1 SWA by the strict-load
                + state-digest proof, run F10 and F11, restore and re-verify the
                sealed model.  M30 stays the exact no-op per weight arm.
                Refuses to rerun.
``terminal``    receipt-only (no data/model/CUDA): evaluate the work-order
                section 2 gates (per-cell promotion on external M4, the F11
                additivity clause, the within/M30 safety floors), the paired
                contrasts vs F00 and the stop/disclosure rows; write the atomic
                terminal receipt.
``dry-plan``    print the frozen contract without touching data.

Examples
--------
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_cdm_p1_cross_v1.py --stage attempt
    SUBC_DATA_ROOT=... SUBM_DATA_ROOT=... CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
        python scripts/run_cdm_p1_cross_v1.py --stage replay
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_cdm_p1_cross_v1.py --stage terminal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.tfpd_lane import receipt as lane_receipt

from src.cdm_p1_cross_v1 import plan, replay


def _base() -> Path:
    return Path(__file__).resolve().parents[2]


def _verify_receipt(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        raise SystemExit(f"missing sidecar: {sidecar}")
    if sidecar.read_text(encoding="ascii").strip() != f"{digest}  {path.name}":
        raise SystemExit(f"sidecar drift for {path.name}")
    return digest


def _require_gpu_binding() -> str:
    """Part A is bound to GPU 1 ONLY (GPU 0 belongs to another live route)."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw != str(plan.GPU_INDEX):
        raise SystemExit(
            "the Part-A GPU stages require CUDA_VISIBLE_DEVICES bound to exactly "
            f"GPU {plan.GPU_INDEX} (never GPU 0); found {raw!r}"
        )
    return raw


def _stage_attempt(base: Path, output: Path) -> dict[str, object]:
    canonical = base / plan.RESULT_ROOT_RELATIVE
    if canonical.exists() and canonical != output:
        raise SystemExit(f"the canonical Part-A result root is not fresh: {canonical}")
    if output.exists():
        raise SystemExit(f"the Part-A result root already exists: {output}")
    environment = lane_receipt.enforce_environment()
    owned = plan.owned_sha256s(base)
    predecessors = plan.predecessor_sha256s(base)
    sealed_digest = predecessors[plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE]
    if sealed_digest != plan.SEALED_ACTIVITY_ONLY_RESULT_SHA256:
        raise SystemExit(
            "the sealed activity-only quick_v2 result no longer matches the "
            "immutable evidence anchor inherited from Stage O/P"
        )
    go_anchor = replay.verify_stage_p_go_anchor(base)
    selection_binding = replay.verify_sealed_selection_from_terminal(base)
    c1_binding = replay.verify_c1_bindings(base)
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "scope": "part_a_dandi_inference_only_factorial",
        "work_order": {
            "path": plan.WORK_ORDER_RELATIVE,
            "sha256": predecessors[plan.WORK_ORDER_RELATIVE],
            "section": "2 (Part A)",
        },
        "stage_p_go_anchor": go_anchor,
        "sealed_selection_binding": selection_binding,
        "c1_weight_binding": c1_binding,
        "owned_sha256s": owned,
        "predecessor_sha256s": predecessors,
        "predecessor_binding": {
            "sealed_activity_only_result_sha256": sealed_digest,
            "stage_p": "the sealed Stage-P modules, chain and attempt/replay/terminal receipts",
            "cal_aug_v1": "the sealed C1 SWA artifact and its training/deployment terminals",
        },
        "environment": environment,
        "gpu_binding_plan": {
            "gpu_index": plan.GPU_INDEX,
            "cuda_visible_devices": str(plan.GPU_INDEX),
            "gpu_0_policy": "never touched (another live route owns GPU 0)",
            "one_launch_one_process": True,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "runtime_estimate_note": (
                "expect roughly 2.5-3.5 GPU-hours (four cells over 3 budgets x 21 "
                "sessions at the sealed Stage-P per-trial cost); the hard timeout "
                "is the binding 6 h work-order ceiling"
            ),
        },
        "pre_registration": plan.pre_registration_payload(),
        "inference_only": True,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "target_optimizer_backward_update": 0,
    }
    plan.validate_pre_registration(payload["pre_registration"])
    output.mkdir(mode=0o755, parents=True, exist_ok=False)
    lane_receipt.write_receipt_transactionally(output / "attempt.json", payload)
    return {
        "stage": "attempt",
        "result_root": str(output),
        "owned_files": len(payload["owned_sha256s"]),
        "predecessor_files": len(payload["predecessor_sha256s"]),
        "stage_p_go_anchor_verified": True,
        "stage_p_decision": go_anchor["decision"],
        "stage_p_driving_cell": go_anchor["driving_cell"],
        "sealed_selection_verified": True,
        "c1_weight_binding_verified": True,
    }


def _stage_replay(base: Path, output: Path) -> dict[str, object]:
    _require_gpu_binding()
    if (output / "terminal.json").exists():
        raise SystemExit("the Part-A terminal receipt already exists; refusing to rerun")
    if (output / "replay.json").exists():
        raise SystemExit(
            "the Part-A replay receipt already exists; run --stage terminal, never a second replay"
        )
    payload = replay.run_cross_replay(base, gpu_index=plan.GPU_INDEX, output_root=output)
    lane_receipt.write_receipt_transactionally(output / "replay.json", payload)
    gate_view = {
        budget: {
            surface: {cell: entry["mean_r2"] for cell, entry in cells.items()}
            for surface, cells in surfaces.items()
        }
        for budget, surfaces in payload["matrix"].items()
    }
    return {
        "stage": "replay",
        "result_root": str(output),
        "gpu_index": payload["gpu_index"],
        "f00_bit_anchors_all_exact": bool(
            payload["anchors"]["f00_vs_sealed_activity_only_all_exact"]
            and payload["anchors"]["f00_vs_sealed_stage_p_all_exact"]
        ),
        "f01_bit_anchor_all_exact": bool(payload["anchors"]["f01_vs_sealed_stage_p_all_exact"]),
        "initial_carrier_invariance_all_exact": bool(
            payload["anchors"]["initial_carrier_invariance_all_exact"]
        ),
        "m30_noop_all_cells": bool(payload["anchors"]["m30_noop_all_cells"]),
        "causality_state_chains_all_rows": bool(payload["causality_state_chains_all_rows"]),
        "sealed_model_restored_and_verified": bool(
            payload["model_state_digests"]["sealed_model_restored_and_verified"]
        ),
        "matrix_mean_r2": gate_view,
        "wall_seconds": payload["wall_seconds"],
    }


def _stage_terminal(base: Path, output: Path) -> dict[str, object]:
    if (output / "terminal.json").exists():
        raise SystemExit("the Part-A terminal receipt already exists; refusing to overwrite")
    attempt_sha = _verify_receipt(output / "attempt.json")
    replay_sha = _verify_receipt(output / "replay.json")
    replay_payload = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    if str(replay_payload.get("attempt_sha256")) != attempt_sha:
        raise SystemExit("the replay receipt was produced under a different attempt")
    started = time.monotonic()
    body = replay.build_terminal(replay_payload=replay_payload)
    gate = body["gate"]
    stops = body["stop_conditions"]
    payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "part_a_dandi_inference_only_factorial",
        "attempt_sha256": attempt_sha,
        "replay_sha256": replay_sha,
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "cells": list(plan.CELL_ORDER),
        "gate_summary": {
            "per_cell_external_m4_delta": {
                cell: float(entry["external_m4_delta"])
                for cell, entry in gate["per_cell"].items()
            },
            "per_cell_positive_external_m4_sessions": {
                cell: int(entry["positive_external_m4_sessions"])
                for cell, entry in gate["per_cell"].items()
            },
            "per_cell_promoted": {
                cell: bool(entry["promoted"]) for cell, entry in gate["per_cell"].items()
            },
            "additivity_held": bool(gate["additivity"]["held"]),
            "safety_all_pass": bool(gate["safety_all_pass"]),
            "stop_conditions_fired": stops["fired"],
        },
        "model_or_checkpoint_updated": False,
        "decoder_training": False,
        "external_roster_opened_by_terminal": False,
        "target_optimizer_backward_update": 0,
        "wall_seconds": float(time.monotonic() - started),
        **body,
    }
    lane_receipt.write_receipt_transactionally(output / "terminal.json", payload)
    os.chmod(output, 0o555)
    return {
        "stage": "terminal",
        "result_root": str(output),
        "per_cell_promoted": payload["gate_summary"]["per_cell_promoted"],
        "additivity_held": payload["gate_summary"]["additivity_held"],
        "safety_all_pass": payload["gate_summary"]["safety_all_pass"],
        "stop_conditions_fired": stops["fired"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CDM x P1 cross-dataset factorial, Part A (inference-only)",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["attempt", "replay", "terminal", "dry-plan"],
    )
    parser.add_argument("--gpu-index", type=int, default=plan.GPU_INDEX, choices=(plan.GPU_INDEX,))
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()
    base = _base()
    output = (
        Path(args.output_root).absolute()
        if args.output_root else base / plan.RESULT_ROOT_RELATIVE
    )
    if args.stage == "dry-plan":
        print(json.dumps(plan.dry_plan(), indent=2, sort_keys=True, default=str))
        return 0
    if args.stage == "attempt":
        result = _stage_attempt(base, output)
    elif args.stage == "replay":
        result = _stage_replay(base, output)
    else:
        result = _stage_terminal(base, output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
