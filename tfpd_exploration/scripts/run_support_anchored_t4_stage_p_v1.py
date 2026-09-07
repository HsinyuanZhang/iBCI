#!/usr/bin/env python
"""Support-Anchored T4 Stage-P runner: attempt -> replay -> terminal.

Work order: ``docs/WORKORDER_SUPPORT_ANCHORED_T4_STAGE_P_V1_20260830.md``
(the sealed Stage-O GO anchor is embedded there and re-verified here).
Design: ``docs/DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md``
§5/§6/§7.2/§8/§9.

Stages
------
``attempt``     CPU only, before any data/model/CUDA access: reserve the fresh
                result root, pin the owned module bytes, the design, the work
                order and every immutable predecessor (the sealed activity-only
                receipts, the Stage-O receipts carrying the GO decision and the
                Stage-O modules this route imports) by SHA-256, re-verify the
                sealed Stage-O GO anchor, and pre-register the P0--P5 cells,
                the measurement laws, the three-factor commit law, the
                hyperparameter candidate grid, the within-6-only selection law,
                the c_M calibration law and the §8 gate boundaries.
``replay``      one process on one bound GPU: run the pre-registered factored
                hyperparameter selection on the within-6 folds ONLY, calibrate
                c_M by the Stage-O median law, then run P0--P5 over within-6
                and external-15 at M4/M10/M30 under the full three-factor gate
                (M30 = exact no-op, alpha_M = 0).  P0 must reproduce the sealed
                activity-only rows bit-exactly and every P row at M30 must
                reproduce P0@M30 bit-exactly.  Refuses to rerun.
``terminal``    receipt-only (no data/model/CUDA): evaluate the design §8.1
                promotion gate, the §8.2 continuity and §8.3 confidence
                attributions, the §8.4 stop conditions and the §6.4 movement
                ordering; write the atomic terminal receipt.
``dry-plan``    print the frozen contract without touching data.

Examples
--------
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_support_anchored_t4_stage_p_v1.py --stage attempt
    SUBC_DATA_ROOT=... SUBM_DATA_ROOT=... CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
        python scripts/run_support_anchored_t4_stage_p_v1.py --stage replay --gpu-index 1
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_support_anchored_t4_stage_p_v1.py --stage terminal
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

from src.support_anchored_t4_stage_p_v1 import plan, replay


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
    """One process, one GPU: refuse anything but a single-device binding."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [item.strip() for item in raw.split(",") if item.strip() != ""]
    if len(tokens) != 1 or tokens[0] not in ("0", "1"):
        raise SystemExit(
            "the Stage-P GPU stages require CUDA_VISIBLE_DEVICES bound to exactly one "
            f"GPU (0 or 1); found {raw!r}"
        )
    return tokens[0]


def _stage_attempt(base: Path, output: Path) -> dict[str, object]:
    canonical = base / plan.RESULT_ROOT_RELATIVE
    if canonical.exists() and canonical != output:
        raise SystemExit(f"the canonical Stage-P result root is not fresh: {canonical}")
    if output.exists():
        raise SystemExit(f"the Stage-P result root already exists: {output}")
    environment = lane_receipt.enforce_environment()
    owned = plan.owned_sha256s(base)
    predecessors = plan.predecessor_sha256s(base)
    sealed_digest = predecessors[plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE]
    if sealed_digest != plan.SEALED_ACTIVITY_ONLY_RESULT_SHA256:
        raise SystemExit(
            "the sealed activity-only quick_v2 result.json no longer matches the "
            "design's immutable evidence anchor"
        )
    # Re-verify the sealed Stage-O GO anchor BEFORE reserving anything.
    go_anchor = replay.verify_stage_o_go_anchor(base)
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "scope": "non_governing_score_only_deployable_pseudo_direction_stage_p",
        "design": {
            "path": plan.DESIGN_RELATIVE,
            "sha256": owned[plan.DESIGN_RELATIVE],
            "sections": ["5 (deployable direction measurement)", "6 (three-factor gate)",
                         "7.2 (stage P matrix)", "8 (promotion/attribution/stopping)",
                         "9 (receipts)"],
        },
        "work_order": {
            "path": plan.WORK_ORDER_RELATIVE,
            "sha256": owned[plan.WORK_ORDER_RELATIVE],
        },
        "stage_o_go_anchor": go_anchor,
        "owned_sha256s": owned,
        "predecessor_sha256s": predecessors,
        "predecessor_binding": {
            "sealed_activity_only_result_sha256": sealed_digest,
            "stage_o_receipts": "the sealed Stage-O attempt/directions/replay/terminal receipts",
            "stage_o_modules": "the Stage-O package bytes this route imports (anchor/trust_region/replay)",
        },
        "environment": environment,
        "gpu_binding_plan": {
            "gpu_index": plan.GPU_INDEX,
            "cuda_visible_devices": str(plan.GPU_INDEX),
            "one_launch_one_process": True,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "runtime_estimate_note": (
                "expect roughly 10-11 GPU-hours (selection ~24 within-6 P2 rollouts "
                "per low budget plus the six-row governing grid with four held-group "
                "forwards per trial); launch only after code review, on an idle card"
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
        "sealed_activity_only_anchor_verified": True,
        "stage_o_go_anchor_verified": True,
        "stage_o_decision": go_anchor["decision"],
    }


def _stage_replay(base: Path, output: Path, gpu_index: int) -> dict[str, object]:
    _require_gpu_binding()
    if (output / "terminal.json").exists():
        raise SystemExit("the Stage-P terminal receipt already exists; refusing to rerun")
    if (output / "replay.json").exists():
        raise SystemExit(
            "the Stage-P replay receipt already exists; run --stage terminal, never a second replay"
        )
    payload = replay.run_stage_p_replay(base, gpu_index=gpu_index, output_root=output)
    lane_receipt.write_receipt_transactionally(output / "replay.json", payload)
    gate_view = {
        budget: {
            surface: {row: entry["mean_r2"] for row, entry in rows.items()}
            for surface, rows in surfaces.items()
        }
        for budget, surfaces in payload["matrix"].items()
    }
    selected = {
        key: value.get("selected")
        for key, value in payload["hyperparameter_selection"].items()
        if isinstance(value, dict) and "selected" in value
    }
    return {
        "stage": "replay",
        "result_root": str(output),
        "gpu_index": payload["gpu_index"],
        "p0_anchor_all_sessions_bit_exact": bool(payload["anchors"]["sealed_activity_only_all_exact"]),
        "m30_noop_all_rows": bool(payload["anchors"]["m30_noop_all_rows"]),
        "causality_state_chains_all_rows": bool(payload["causality_state_chains_all_rows"]),
        "model_state_digest_unchanged": bool(payload["model_state_digest_unchanged"]),
        "trust_region_no_committed_drift": bool(payload["trust_region_no_committed_drift"]),
        "selected_hyperparameters": selected,
        "matrix_mean_r2": gate_view,
        "wall_seconds": payload["wall_seconds"],
    }


def _stage_terminal(base: Path, output: Path) -> dict[str, object]:
    if (output / "terminal.json").exists():
        raise SystemExit("the Stage-P terminal receipt already exists; refusing to overwrite")
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
        "scope": "non_governing_score_only_deployable_pseudo_direction_stage_p",
        "attempt_sha256": attempt_sha,
        "replay_sha256": replay_sha,
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "decision": gate["decision"],
        "driving_cell": gate["driving_cell"],
        "safety": gate["safety"],
        "gate_summary": {
            "m4_external_p2_minus_p0": gate["per_budget"]["m4"]["P2"]["external_delta"],
            "m4_external_p2_positive_sessions": gate["per_budget"]["m4"]["P2"]["positive_external_sessions"],
            "m4_external_p1_minus_p0": gate["per_budget"]["m4"]["P1"]["external_delta"],
            "m10_external_p2_minus_p0": gate["per_budget"]["m10"]["P2"]["external_delta"],
            "m10_external_p2_positive_sessions": gate["per_budget"]["m10"]["P2"]["positive_external_sessions"],
            "stop_conditions_fired": stops["fired"],
        },
        "movement_ordering": body["movement_ordering"],
        "stage_o_go_anchor": body["stage_o_go_anchor"],
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
        "decision": gate["decision"],
        "driving_cell": gate["driving_cell"],
        "stop_conditions_fired": stops["fired"],
        "movement_ordering_holds": body["movement_ordering"]["ordering_holds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Support-Anchored Causal T4 Memory, Stage P (deployable pseudo-direction)",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["attempt", "replay", "terminal", "dry-plan"],
    )
    parser.add_argument("--gpu-index", type=int, default=plan.GPU_INDEX, choices=(0, 1))
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
        result = _stage_replay(base, output, args.gpu_index)
    else:
        result = _stage_terminal(base, output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
