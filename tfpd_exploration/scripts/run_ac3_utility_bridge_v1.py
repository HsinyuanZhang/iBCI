#!/usr/bin/env python
"""AC3-U runner: attempt -> directions (CPU) -> replay (GPU) -> terminal.

Work order: ``docs/WORKORDER_AC3_UTILITY_BRIDGE_V1_20260829.md`` (binding).
Guidance: ``docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md``
§4 (queue item 1) and §13 (launch boundary).

Stages
------
``attempt``     CPU only, before any data or model access: reserve the fresh
                result root, pin the owned module bytes, the work order, the
                guidance and every immutable predecessor by SHA-256, and
                pre-register the rows, the rotation law, the seam, the scoring
                domains and the §8 gates.
``directions``  CPU only: rebuild R0/R0.5/R2 with the frozen screen's own
                loaders and verify every digest anchor against the sealed
                screen.json; publish the per-trial theta vectors the replay
                consumes plus the session/trial binding.
``replay``      one process on one bound GPU: the frozen P2' coherent oracle
                replay for U0/UGE/U2 with the process-local rotation seam;
                U0 must reproduce sealed stage-cop within-M4 O0 bit-exactly.
``terminal``    evaluate the §8 gates on the governing raw matrix R2 and write
                the atomic terminal-or-failure receipt.
``dry-plan``    print the frozen contract without touching data.

Examples
--------
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_ac3_utility_bridge_v1.py --stage attempt
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_ac3_utility_bridge_v1.py --stage directions
    SUBC_DATA_ROOT=... SUBM_DATA_ROOT=... CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \\
        python scripts/run_ac3_utility_bridge_v1.py --stage replay --gpu-index 1
    CUDA_VISIBLE_DEVICES=1 python scripts/run_ac3_utility_bridge_v1.py --stage terminal --gpu-index 1
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

from src.ac3_utility_bridge_v1 import directions, plan, replay


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
            "the AC3-U GPU stages require CUDA_VISIBLE_DEVICES bound to exactly one "
            f"GPU (0 or 1); found {raw!r}"
        )
    return tokens[0]


def _build_attempt(base: Path) -> dict[str, object]:
    environment = lane_receipt.enforce_environment()
    owned = plan.owned_sha256s(base)
    return {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "scope": "non_governing_m4_only_zero_learning_utility_bridge",
        "work_order": {
            "path": plan.WORK_ORDER_RELATIVE,
            "sha256": owned[plan.WORK_ORDER_RELATIVE],
        },
        "guidance": {
            "path": plan.GUIDANCE_RELATIVE,
            "sha256": hashlib.sha256((base / plan.GUIDANCE_RELATIVE).read_bytes()).hexdigest(),
            "sections": ["4 (queue item 1)", "13 (launch boundary)"],
        },
        "owned_sha256s": owned,
        "predecessor_sha256s": {
            key: value for key, value in directions.verify_predecessors(base)["files"].items()
        },
        "predecessor_binding": {
            "trajectories_npz": (
                "no .sha256 sidecar exists; bound bit-for-bit through the "
                "sidecar-pinned materialize.json array_digests block via the "
                "frozen screen's verify_input_lock at the directions stage"
            ),
            "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
        },
        "environment": environment,
        "gpu_binding_plan": {
            "gpu_index": plan.GPU_INDEX,
            "device": "NVIDIA GeForce RTX 3090",
            "cuda_visible_devices": str(plan.GPU_INDEX),
            "one_launch_one_process": True,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "note": "GPU 0 stays reserved for the parallel SLOT-AUDIT line",
        },
        "pre_registration": plan.gate_spec_payload(),
        "inference_only": True,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "target_optimizer_backward_update": 0,
    }


def _stage_attempt(base: Path, output: Path) -> dict[str, object]:
    canonical = base / plan.RESULT_ROOT_RELATIVE
    if canonical.exists() and canonical != output:
        raise SystemExit(f"the canonical AC3-U result root is not fresh: {canonical}")
    if output.exists():
        raise SystemExit(f"the AC3-U result root already exists: {output}")
    payload = _build_attempt(base)
    output.mkdir(mode=0o755, parents=True, exist_ok=False)
    lane_receipt.write_receipt_transactionally(output / "attempt.json", payload)
    return {
        "stage": "attempt",
        "result_root": str(output),
        "owned_files": len(payload["owned_sha256s"]),
        "predecessor_files": len(payload["predecessor_sha256s"]),
        "gpu_binding_plan": payload["gpu_binding_plan"],
    }


def _stage_directions(base: Path, output: Path) -> dict[str, object]:
    lane_receipt.enforce_environment()
    attempt_sha = _verify_receipt(output / "attempt.json")
    payload = directions.rebuild_directions(base, attempt_sha256=attempt_sha)
    lane_receipt.write_receipt_transactionally(output / "directions.json", payload)
    return {
        "stage": "directions",
        "result_root": str(output),
        "n_trials": payload["n_trials"],
        "rows": {
            row: {
                "theta_digest_matches_sealed_screen": bool(
                    item["theta_digest_matches_sealed_screen"]
                ),
                "n_defined": int(item["n_defined"]),
                "n_undefined_theta_fallback": int(item["n_undefined_theta_fallback"]),
            }
            for row, item in payload["rows"].items()
        },
        "input_lock_sha256": payload["input_lock_sha256"],
        "wall_seconds": payload["wall_seconds"],
    }


def _stage_replay(base: Path, output: Path, gpu_index: int) -> dict[str, object]:
    _require_gpu_binding()
    if (output / "terminal.json").exists():
        raise SystemExit("the AC3-U terminal receipt already exists; refusing to rerun")
    if (output / "replay.json").exists():
        raise SystemExit(
            "the AC3-U replay receipt already exists; run --stage terminal, never a second replay"
        )
    payload = replay.run_replay(base, gpu_index=gpu_index, output_root=output)
    lane_receipt.write_receipt_transactionally(output / "replay.json", payload)
    return {
        "stage": "replay",
        "result_root": str(output),
        "gpu_index": payload["gpu_index"],
        "u0_anchor_all_sessions_bit_exact": bool(payload["u0_anchor_all_sessions_bit_exact"]),
        "model_state_digest_unchanged": bool(payload["model_state_digest_unchanged"]),
        "governing_raw_matrix_r2": {
            row: payload["rows"][row]["governing_raw_matrix_r2"]
            for row in plan.ROW_ORDER
        },
        "wall_seconds": payload["wall_seconds"],
    }


def _stage_terminal(base: Path, output: Path, gpu_index: int) -> dict[str, object]:
    _require_gpu_binding()
    if (output / "terminal.json").exists():
        raise SystemExit("the AC3-U terminal receipt already exists; refusing to overwrite")
    attempt_sha = _verify_receipt(output / "attempt.json")
    directions_sha = _verify_receipt(output / "directions.json")
    replay_sha = _verify_receipt(output / "replay.json")
    replay_payload = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    if str(replay_payload.get("attempt_sha256")) != attempt_sha:
        raise SystemExit("the replay receipt was produced under a different attempt")
    started = time.monotonic()
    body = replay.build_terminal(replay_payload=replay_payload)
    verdict = body["gates"]["disposition"]
    payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "non_governing_m4_only_zero_learning_utility_bridge",
        "attempt_sha256": attempt_sha,
        "directions_sha256": directions_sha,
        "replay_sha256": replay_sha,
        "budget": plan.BUDGET,
        "surface": plan.SURFACE,
        "sessions": list(plan.SESSIONS),
        "high_error_session": plan.HIGH_ERROR_SESSION,
        "high_error_session_in_governing_mean": True,
        "disposition": verdict,
        "disposition_strings": body["gates"]["disposition_strings"],
        "gate_summary": {
            "UGE_minus_U0_equal_session_mean": body["gates"]["UGE_minus_U0"]["equal_session_mean_delta"],
            "UGE_positive_sessions": body["gates"]["UGE_minus_U0"]["positive_sessions"],
            "UGE_gate_passed": body["gates"]["UGE_gate"]["passed"],
            "U2_minus_UGE_equal_session_mean": body["gates"]["U2_minus_UGE"]["equal_session_mean_delta"],
            "U2_rescue_passed": body["gates"]["U2_rescue"]["passed"],
        },
        "u0_anchor_all_sessions_bit_exact": bool(
            replay_payload["u0_anchor_all_sessions_bit_exact"]
        ),
        "model_state_digest_before_sha256": replay_payload["model_state_digest_before_sha256"],
        "model_state_digest_after_sha256": replay_payload["model_state_digest_after_sha256"],
        "model_state_digest_unchanged": bool(replay_payload["model_state_digest_unchanged"]),
        "model_or_checkpoint_updated": False,
        "external_roster_opened": False,
        "decoder_training": False,
        "target_optimizer_backward_update": 0,
        "wall_seconds": float(time.monotonic() - started),
        **body,
    }
    lane_receipt.write_receipt_transactionally(output / "terminal.json", payload)
    os.chmod(output, 0o555)
    return {
        "stage": "terminal",
        "result_root": str(output),
        "disposition": verdict,
        "gate_summary": payload["gate_summary"],
        "equal_session_mean_r2": body["gates"]["equal_session_mean_r2"],
        "five_session_sensitivity_non_governing": (
            body["gates"]["five_session_sensitivity_non_governing"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AC3-U M4-only zero-learning utility bridge (work order 2026-08-29)",
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
        print(json.dumps(plan.gate_spec_payload(), indent=2, sort_keys=True, default=str))
        return 0
    if args.stage == "attempt":
        result = _stage_attempt(base, output)
    elif args.stage == "directions":
        result = _stage_directions(base, output)
    elif args.stage == "replay":
        result = _stage_replay(base, output, args.gpu_index)
    else:
        result = _stage_terminal(base, output, args.gpu_index)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
