#!/usr/bin/env python
"""Support-Anchored T4 Stage-O runner: attempt -> directions -> replay -> terminal.

Work order: ``docs/WORKORDER_SUPPORT_ANCHORED_T4_STAGE_O_V1_20260830.md``
(binding, including operator amendment 1).
Design: ``docs/DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md`` §3/§7.1/§13.

Stages
------
``attempt``     CPU only, before any data/model/CUDA access: reserve the fresh
                result root, pin the owned module bytes, the design, the work
                order and every immutable predecessor by SHA-256, and
                pre-register the cells, the O2 always-commit law, the
                hyperparameters (rho_M=1.0, block=1, alpha_M 0.5 primary), the
                c_M within-6 calibration law and the §3.5 gate boundaries.
``directions``  one process on one bound GPU: parse the full grid through the
                frozen runtime and publish the digest-pinned TRUE
                completed-trial direction table (no model forward).
``replay``      one process on one bound GPU: calibrate c_M on within-6 ONLY,
                then run O0/O1/O2 (+ non-governing alpha sensitivities) over
                within-6 and external-15 at M4/M10/M30; O0 must reproduce the
                sealed activity-only rows bit-exactly.  Refuses to rerun.
``terminal``    receipt-only (no data/model/CUDA): evaluate the §3.5
                GO/HOLD/STOP gate, the interpretation rows and the §6.4
                movement ordering; write the atomic terminal receipt.
``dry-plan``    print the frozen contract without touching data.

Examples
--------
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_support_anchored_t4_stage_o_v1.py --stage attempt
    SUBC_DATA_ROOT=... SUBM_DATA_ROOT=... CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
        python scripts/run_support_anchored_t4_stage_o_v1.py --stage directions --gpu-index 1
    SUBC_DATA_ROOT=... SUBM_DATA_ROOT=... CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
        python scripts/run_support_anchored_t4_stage_o_v1.py --stage replay --gpu-index 1
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_support_anchored_t4_stage_o_v1.py --stage terminal
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

from src.support_anchored_t4_stage_o_v1 import directions, plan, replay


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
            "the Stage-O GPU stages require CUDA_VISIBLE_DEVICES bound to exactly one "
            f"GPU (0 or 1); found {raw!r}"
        )
    return tokens[0]


def _stage_attempt(base: Path, output: Path) -> dict[str, object]:
    canonical = base / plan.RESULT_ROOT_RELATIVE
    if canonical.exists() and canonical != output:
        raise SystemExit(f"the canonical Stage-O result root is not fresh: {canonical}")
    if output.exists():
        raise SystemExit(f"the Stage-O result root already exists: {output}")
    environment = lane_receipt.enforce_environment()
    owned = plan.owned_sha256s(base)
    predecessors = plan.predecessor_sha256s(base)
    sealed_digest = predecessors[plan.ACTIVITY_ONLY_V2_RESULT_RELATIVE]
    if sealed_digest != plan.SEALED_ACTIVITY_ONLY_RESULT_SHA256:
        raise SystemExit(
            "the sealed activity-only quick_v2 result.json no longer matches the "
            "design's immutable evidence anchor"
        )
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "scope": "non_governing_score_only_oracle_headroom_stage_o",
        "design": {
            "path": plan.DESIGN_RELATIVE,
            "sha256": owned[plan.DESIGN_RELATIVE],
            "sections": ["3 (cells)", "4 (block refit / trust region)", "7.1 (stage O)",
                          "9 (receipts)", "10 (boundaries)", "13 (order)"],
        },
        "work_order": {
            "path": plan.WORK_ORDER_RELATIVE,
            "sha256": owned[plan.WORK_ORDER_RELATIVE],
            "binding_amendment_1": dict(plan.O2_COMMIT_LAW),
        },
        "owned_sha256s": owned,
        "predecessor_sha256s": predecessors,
        "predecessor_binding": {
            "sealed_activity_only_result_sha256": sealed_digest,
            "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "p2prime_stage_cop": "read-only cross-reference (the coherent-oracle upper bound)",
        },
        "environment": environment,
        "gpu_binding_plan": {
            "gpu_index": plan.GPU_INDEX,
            "cuda_visible_devices": str(plan.GPU_INDEX),
            "one_launch_one_process": True,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "note": "launch only after code review, on an idle card; C2/C3 mainline has priority",
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
    }


def _stage_directions(base: Path, output: Path, gpu_index: int) -> dict[str, object]:
    _require_gpu_binding()
    if (output / "replay.json").exists() or (output / "terminal.json").exists():
        raise SystemExit("the Stage-O directions stage must precede replay/terminal")
    attempt_sha = _verify_receipt(output / "attempt.json")
    owned = plan.owned_sha256s(base)
    attempt = json.loads((output / "attempt.json").read_text(encoding="utf-8"))
    if owned != attempt["owned_sha256s"]:
        raise SystemExit("an owned Stage-O module drifted after the attempt was reserved")
    payload = directions.build_direction_table(base, gpu_index=gpu_index, attempt_sha256=attempt_sha)
    lane_receipt.write_receipt_transactionally(output / "directions.json", payload)
    cross = payload["substudy_cross_reference"]
    return {
        "stage": "directions",
        "result_root": str(output),
        "gpu_index": payload["gpu_index"],
        "n_cells": len(payload["cells"]),
        "accepted_trials_total": sum(item["accepted_trials"] for item in payload["cells"].values()),
        "rejected_trials_total": sum(item["rejected_trials"] for item in payload["cells"].values()),
        "substudy_cross_reference_all_equal": bool(cross.get("all_equal")),
        "padded_true_rows_total": payload["padded_true_rows_total"],
        "wall_seconds": payload["wall_seconds"],
    }


def _stage_replay(base: Path, output: Path, gpu_index: int) -> dict[str, object]:
    _require_gpu_binding()
    if (output / "terminal.json").exists():
        raise SystemExit("the Stage-O terminal receipt already exists; refusing to rerun")
    if (output / "replay.json").exists():
        raise SystemExit(
            "the Stage-O replay receipt already exists; run --stage terminal, never a second replay"
        )
    payload = replay.run_stage_o_replay(base, gpu_index=gpu_index, output_root=output)
    lane_receipt.write_receipt_transactionally(output / "replay.json", payload)
    gate_view = {
        budget: {
            surface: {row: entry["mean_r2"] for row, entry in rows.items()}
            for surface, rows in surfaces.items()
        }
        for budget, surfaces in payload["matrix"].items()
    }
    return {
        "stage": "replay",
        "result_root": str(output),
        "gpu_index": payload["gpu_index"],
        "o0_anchor_all_sessions_bit_exact": bool(payload["anchors"]["sealed_activity_only_all_exact"]),
        "filter_line_cache_all_bitwise": bool(payload["anchors"]["filter_line_cache_all_bitwise"]),
        "causality_state_chains_all_rows": bool(payload["causality_state_chains_all_rows"]),
        "model_state_digest_unchanged": bool(payload["model_state_digest_unchanged"]),
        "c_M": {
            key: value["c_M"] for key, value in payload["c_M_calibration"].items()
            if isinstance(value, dict) and "c_M" in value
        },
        "matrix_mean_r2": gate_view,
        "wall_seconds": payload["wall_seconds"],
    }


def _stage_terminal(base: Path, output: Path) -> dict[str, object]:
    if (output / "terminal.json").exists():
        raise SystemExit("the Stage-O terminal receipt already exists; refusing to overwrite")
    attempt_sha = _verify_receipt(output / "attempt.json")
    directions_sha = _verify_receipt(output / "directions.json")
    replay_sha = _verify_receipt(output / "replay.json")
    replay_payload = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    if str(replay_payload.get("attempt_sha256")) != attempt_sha:
        raise SystemExit("the replay receipt was produced under a different attempt")
    started = time.monotonic()
    body = replay.build_terminal(replay_payload=replay_payload)
    gate = body["gate"]
    payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "non_governing_score_only_oracle_headroom_stage_o",
        "attempt_sha256": attempt_sha,
        "directions_sha256": directions_sha,
        "replay_sha256": replay_sha,
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "decision": gate["decision"],
        "driving_budget": gate["driving_budget"],
        "safety": gate["safety"],
        "gate_summary": {
            "m4_external_o2_minus_o0": gate["per_budget"]["m4"]["external"]["equal_session_mean_delta"],
            "m4_positive_external_sessions": gate["per_budget"]["m4"]["external"]["positive_sessions"],
            "m10_external_o2_minus_o0": gate["per_budget"]["m10"]["external"]["equal_session_mean_delta"],
            "m10_positive_external_sessions": gate["per_budget"]["m10"]["external"]["positive_sessions"],
            "within_bounds": gate["within_bounds"],
        },
        "movement_ordering": body["movement_ordering"],
        "p2prime_cross_reference": body["p2prime_cross_reference"],
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
        "movement_ordering_holds": body["movement_ordering"]["ordering_holds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Support-Anchored Causal T4 Memory, Stage O (oracle headroom)",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["attempt", "directions", "replay", "terminal", "dry-plan"],
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
    elif args.stage == "directions":
        result = _stage_directions(base, output, args.gpu_index)
    elif args.stage == "replay":
        result = _stage_replay(base, output, args.gpu_index)
    else:
        result = _stage_terminal(base, output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
