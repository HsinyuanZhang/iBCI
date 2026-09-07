#!/usr/bin/env python3
"""Fail-closed readiness/runner for A2 fixed-K temporal-prototype controls.

Without ``--execute-source-audit`` this script only writes an immutable
prelaunch receipt and does not resolve or open any NWB.  Execution is guarded
by a root review bound to that receipt, opens only the existing four-entry
held-in-calib manifest through the already-audited v3 exact-source loader, and
remains CPU/source-only.  It never builds a decoder or accesses minival,
held-out, formal, EvalAI, or CUDA paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
if str(SUA) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SUA))

from mc_maze.fixed_k_temporal_prototypes import streaming_cost_receipt  # noqa: E402
from mc_maze.m1_fixed_k_prototype_gate_a import future_neural_oracle_target  # noqa: E402
from mc_maze.m1_fixed_k_prototype_gate_a2 import (  # noqa: E402
    A2_SEMANTICS_VERSION,
    RIDGE,
    SLOT_NULL_REPLICATES,
    TIME_ORDER_NULL_REPLICATES,
    a2_cpu_gate,
    a2_control_contract_receipt,
    build_a2_base_carriers,
    canonical_slot_identity_rank_diagnostic,
    exact_group_upper_rank,
    exact_slot_null_plans,
    exact_slot_null_statistics,
    exact_slot_chunk_ranges,
    generic_outer_loso_proxy,
    mean_arm_metric,
    paired_b20_content_summary,
    scheduled_time_upper_rank,
    slot_null_carriers,
    time_null_receipt,
    time_order_null_carriers,
    time_order_null_score_matrix,
)


V3_MANIFEST = SUA / "manifests" / "m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json"
PROTOCOL = SUA / "docs" / "M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A2_NULL_PROTOCOL.md"
PRELAUNCH_DEFAULT = SUA / "results" / "m1_fixed_k_temporal_prototype_gate_a2_prelaunch_v1"
AUDIT_DEFAULT = SUA / "results" / "m1_fixed_k_temporal_prototype_gate_a2_v1"
SCHEMA_PRELAUNCH = "m1_fixed_k_temporal_prototype_gate_a2_prelaunch_v1"
SCHEMA_AUDIT = "m1_fixed_k_temporal_prototype_gate_a2_v1"
REVIEW_AUTHORIZATION = "root_approved_source_only_cpu_gate_a2_v1"
SOURCE_ONLY_CONFIRMATION = "I_CONFIRM_EXACT_FOUR_M1_SOURCE_NWBS_CPU_ONLY_A2_V1"
EXPECTED_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Stable repository-relative display, with a test-friendly absolute fallback."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if hasattr(value, "__dict__"):
        return strict_json(vars(value))
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def load_exact_source_manifest(path: Path = V3_MANIFEST) -> dict[str, Any]:
    """Read only metadata; this never resolves or opens source NWBs."""
    data = json.loads(path.read_text())
    if data.get("schema_version") != "m1_fixed_k_temporal_prototype_gate_a_source_manifest_v2":
        raise ValueError("A2 requires the audited v3 exact-four-source manifest schema")
    rows = data.get("sessions")
    if not isinstance(rows, list) or tuple(row.get("session") for row in rows) != EXPECTED_SESSIONS:
        raise ValueError("A2 manifest must be the exact ordered four M1 held-in-calib sources")
    for row in rows:
        relative = str(row.get("relative_path", ""))
        if not relative.startswith("SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"):
            raise ValueError("A2 source leaves held-in-calib scope")
        if any(token in relative.lower() for token in ("held-out", "minival", "test", "evalai", "formal")):
            raise ValueError("A2 manifest contains a prohibited path scope")
        if len(str(row.get("sha256", ""))) != 64:
            raise ValueError("A2 manifest has invalid source checksum")
    return data


def conservative_no_nwb_runtime_benchmark() -> dict[str, Any]:
    """Benchmark one complete synthetic schedule at frozen worst-case raw size.

    This is deliberately a *no-NWB* conservative benchmark, not a data proxy:
    four sessions x ten trials x 1,024 valid bins x 64 channels, all generated
    deterministically in RAM.  It times the actual NumPy carrier/anchor/ridge
    implementation once for Stage 0, one exact-slot configuration, and one
    time-order schedule, then reports a transparent serial projection.
    The real source trial lengths are neither read nor inferred here.
    """
    names = EXPECTED_SESSIONS
    trials = {
        name: tuple(np.full((1024, 64), (session_index + trial_index) % 3, dtype=np.int64) for trial_index in range(10))
        for session_index, name in enumerate(names)
    }
    target = {
        name: np.add.outer(np.arange(64, dtype=np.float64) * 0.001 + index, np.arange(4, dtype=np.float64) * 0.01)
        for index, name in enumerate(names)
    }
    start = time.perf_counter()
    base = build_a2_base_carriers(trials)
    generic_outer_loso_proxy(base.public, target, ridge=RIDGE)
    stage0_seconds = time.perf_counter() - start
    plan = exact_slot_null_plans(names)[1]
    start = time.perf_counter()
    generic_outer_loso_proxy(slot_null_carriers(base, plan_for_replicate=plan), target, ridge=RIDGE)
    slot_config_seconds = time.perf_counter() - start
    start = time.perf_counter()
    generic_outer_loso_proxy(time_order_null_carriers(trials, replicate=1), target, ridge=RIDGE)
    time_schedule_seconds = time.perf_counter() - start
    slot_projection = slot_config_seconds * SLOT_NULL_REPLICATES
    time_projection = time_schedule_seconds * (TIME_ORDER_NULL_REPLICATES - 1)
    total_projection = stage0_seconds + slot_projection + time_projection
    return {
        "kind": "deterministic_synthetic_no_NWB_conservative_raw_shape_benchmark",
        "shape": {"sessions": 4, "trials_per_session": 10, "valid_bins_per_trial": 1024, "channels": 64},
        "stage0_seconds": stage0_seconds,
        "one_slot_configuration_seconds": slot_config_seconds,
        "one_time_schedule_seconds": time_schedule_seconds,
        "worker_caps": {"slot_max": 8, "time_max": 16, "runner_current": 1},
        "serial_projection_seconds": {
            "stage0": stage0_seconds,
            "slot_13824": slot_projection,
            "time_4095": time_projection,
            "total": total_projection,
        },
        "projection_assumptions": "current runner is serial (one CPU worker); excludes process startup/I/O/contention; actual reviewed execution logs elapsed time and fails closed if 24h ceiling is exceeded",
        "within_24h_serial_projection": bool(total_projection <= 24.0 * 3600.0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def build_prelaunch_receipt() -> dict[str, Any]:
    """Build a no-NWB receipt and fail closed until the separate A2 protocol exists."""
    manifest = load_exact_source_manifest()
    if not PROTOCOL.is_file():
        raise FileNotFoundError("A2 protocol is required before a prelaunch receipt may be created")
    benchmark = conservative_no_nwb_runtime_benchmark()
    if not benchmark["within_24h_serial_projection"]:
        raise RuntimeError("A2 conservative no-NWB projection exceeds immutable 24-hour CPU ceiling")
    return {
        "schema_version": SCHEMA_PRELAUNCH,
        "semantics_version": A2_SEMANTICS_VERSION,
        "generated_by": "audit_m1_fixed_k_prototypes_gate_a2.py --write-prelaunch",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "root-review-only A2 source-only CPU control audit; not a result and not decoder authorization",
        "manifest": {
            "path": str(V3_MANIFEST.relative_to(ROOT)),
            "sha256": sha256(V3_MANIFEST),
            "sessions": manifest["sessions"],
            "loader": "v3 audited exact-four-source loader; invoked only after execution guards",
        },
        "protocol": {"path": display_path(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "code": {
            "a2_runner": {"path": display_path(Path(__file__)), "sha256": sha256(Path(__file__))},
            "a2_controls": {
                "path": "sua_exploration/mc_maze/m1_fixed_k_prototype_gate_a2.py",
                "sha256": sha256(SUA / "mc_maze" / "m1_fixed_k_prototype_gate_a2.py"),
            },
            "fixed_k_canonical_representation": {
                "path": "sua_exploration/mc_maze/fixed_k_temporal_prototypes.py",
                "sha256": sha256(SUA / "mc_maze" / "fixed_k_temporal_prototypes.py"),
            },
            "v3_exact_source_loader": {
                "path": "sua_exploration/scripts/audit_m1_fixed_k_prototypes_gate_a.py",
                "sha256": sha256(SUA / "scripts" / "audit_m1_fixed_k_prototypes_gate_a.py"),
            },
        },
        "a2_controls": a2_control_contract_receipt(),
        "execution_plan": {
            "slot_exact_workers_max": 8,
            "slot_lexicographic_chunks": [list(chunk) for chunk in exact_slot_chunk_ranges(workers=8)],
            "time_workers_max": 16,
            "time_schedule_count_including_identity": TIME_ORDER_NULL_REPLICATES,
            "wall_time_limit_hours": 24,
            "runtime_benchmark": benchmark,
        },
        "endpoint": {
            "support_range": [0, 10],
            "target_range": [210, "end"],
            "target": "later-neural log1p(mean per-trial Hz), scorer-only",
            "outer_loso": "four exact source sessions; fold anchors use only other three first-ten supports",
            "readout": "source-only ridge=1, fixed before any A2 execution",
            "primary_slot_statistic": "mean four-session bounded U=TSS/(TSS+RSS)",
            "raw_R2": "secondary descriptive only; never rank-tested",
        },
        "hard_exclusions": {
            "cuda": False,
            "minival": False,
            "held_out": False,
            "formal": False,
            "decoder": False,
            "evalai": False,
            "nwb_opened_for_this_receipt": False,
        },
        "execution_guard": {
            "review_authorization": REVIEW_AUTHORIZATION,
            "source_only_confirmation": SOURCE_ONLY_CONFIRMATION,
            "fresh_output_required": True,
            "no_automatic_gpu_or_decoder_authorization": True,
        },
        "status": "pending_root_review_no_nwb_opened",
    }


def write_prelaunch(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite A2 prelaunch output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "prelaunch_receipt.json"
    target.write_text(json.dumps(strict_json(build_prelaunch_receipt()), indent=2, sort_keys=True) + "\n")
    (output_dir / "prelaunch_receipt.sha256").write_text(f"{sha256(target)}  {target.name}\n")
    return target


def _load_sessions_after_review(manifest: Mapping[str, Any]):
    """Deferred call into the previously audited exact-four-source v3 loader."""
    # Importing this runner does not construct a datamodule or read a file.
    # The function below is called only after both CLI execution guards succeed.
    from scripts.audit_m1_fixed_k_prototypes_gate_a import load_raw_source_sessions

    sessions = load_raw_source_sessions(manifest)
    if tuple(sorted(sessions)) != EXPECTED_SESSIONS:
        raise RuntimeError("v3 loader did not return exactly A2's immutable four sources")
    return sessions


def _write_audit_result(output_dir: Path, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "source_gate_a2.json"
    path.write_text(json.dumps(strict_json(result), indent=2, sort_keys=True) + "\n")
    (output_dir / "source_gate_a2.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return path


def run_source_audit(output_dir: Path) -> Path:
    """Execute the guarded A2 CPU audit; callers must satisfy CLI review checks."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite A2 source-audit output: {output_dir}")
    run_start = time.perf_counter()
    deadline_seconds = 24.0 * 3600.0
    manifest = load_exact_source_manifest()
    sessions = _load_sessions_after_review(manifest)
    support = {name: row.support_trials for name, row in sessions.items()}
    targets = {name: future_neural_oracle_target(row) for name, row in sessions.items()}
    stage0_start = time.perf_counter()
    base = build_a2_base_carriers(support)
    observed_proxy = generic_outer_loso_proxy(base.public, targets, ridge=RIDGE)
    observed_u = mean_arm_metric(observed_proxy, "prototype", metric="bounded_u")
    content = paired_b20_content_summary(observed_proxy)
    # Reuse the v3 thinning code only after the guarded source data access. It
    # is a prerequisite check, not an input to carriers or scores.
    from scripts.audit_m1_fixed_k_prototypes_gate_a import repeatability_receipt
    v3_fold_receipts = {left: {"anchor_receipt": base.anchor_receipts[left]} for left in base.anchor_receipts}
    repeatability, repeatability_pass = repeatability_receipt(sessions, v3_fold_receipts)
    base_validity = bool(
        all(np.isfinite(value).all() for fold in base.public.values() for arm in fold.values() for value in arm.values())
        and all(np.isfinite(value).all() for value in targets.values())
        and all(left not in base.anchor_receipts[left]["source_sessions"] for left in base.anchor_receipts)
        and all(value.shape[1] == 20 for fold in base.public.values() for arm in fold.values() for value in arm.values())
        and streaming_cost_receipt(64).raw_support_matrix_elements == 0
    )
    content_pass = bool(
        content["all_four_positive"] and float(content["paired_ci95"][0]) >= 0.030 and float(content["mde80"]) <= 0.030
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_AUDIT,
        "semantics_version": A2_SEMANTICS_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "source-only CPU A2 control audit; later-neural proxy only, not behavior decoder/EvalAI R2",
        "manifest": {"path": str(V3_MANIFEST.relative_to(ROOT)), "sha256": sha256(V3_MANIFEST)},
        "input_sessions": {name: {"path": row.source_path, "sha256": row.source_sha256} for name, row in sessions.items()},
        "observed_proxy": observed_proxy,
        "B20_content": content,
        "repeatability": repeatability,
        "a2_controls": a2_control_contract_receipt(),
        "contracts": {
            "raw": "v3 exact source loader validated raw noninterpolated integer prefixes, -1 tails, and spike-sum agreement",
            "oracle": "future object labels appear only in future_neural_oracle_target scorer dictionary",
            "state": {
                "all_fold_anchors_exclude_left_out": all(left not in base.anchor_receipts[left]["source_sessions"] for left in base.anchor_receipts),
                "all_A2_carriers_width_20": all(value.shape[1] == 20 for fold in base.public.values() for arm in fold.values() for value in arm.values()),
                "streaming_cost_N64": strict_json(streaming_cost_receipt(64)),
            },
            "mac": strict_json(streaming_cost_receipt(64)),
            "no_labelled_D4_arm": "D4 omitted from A2 carrier/ridge path; rate_only is descriptive only and B20 is the sole content comparator",
        },
        "runtime": {
            "wall_time_ceiling_seconds": deadline_seconds,
            "runner_workers": {"slot": 1, "time": 1, "protocol_max_slot": 8, "protocol_max_time": 16},
            "stages_seconds": {"stage0": time.perf_counter() - stage0_start, "stage1": None, "stage2": None},
            "elapsed_seconds": time.perf_counter() - run_start,
        },
        "hard_exclusions": {"cuda": False, "minival": False, "held_out": False, "formal": False, "decoder": False, "evalai": False},
        "stage_status": {
            "stage0_observed_P20_vs_B20_and_contracts": "pass" if content_pass and base_validity and repeatability_pass else "fail",
            "stage1_exact_relative_slot": "not_run_due_to_predeclared_early_stop",
            "stage2_time_order": "not_run_due_to_predeclared_early_stop",
        },
    }
    if not (content_pass and base_validity and repeatability_pass):
        result["decision"] = {
            "decision": (
                "invalid_execution_stop" if not (base_validity and repeatability_pass)
                else ("marginal_baseline_not_beaten_stop" if not content["all_four_positive"] else "precision_insufficient_stop")
            ),
            "gpu_authorized": False,
            "decoder_authorized": False,
            "slot_null_exact": "not_run_due_to_predeclared_early_stop",
            "time_order_null": "not_run_due_to_predeclared_early_stop",
        }
        result["runtime"]["elapsed_seconds"] = time.perf_counter() - run_start
        if result["runtime"]["elapsed_seconds"] > deadline_seconds:
            result["decision"]["decision"] = "fixed_null_budget_infeasible_stop"
        return _write_audit_result(output_dir, result)

    # Stage 1 runs only after the content/validity gate passes.
    stage1_start = time.perf_counter()
    slot_values, slot_u, slot_plan = exact_slot_null_statistics(base, targets)
    slot_rank = exact_group_upper_rank(observed_u, slot_values)
    slot_conditional = canonical_slot_identity_rank_diagnostic(
        exact_slot_null_plans(tuple(sorted(sessions))), slot_values
    )
    observed_rows = {str(row["left_out_session"]): float(row["bounded_u"]) for row in observed_proxy["arms"]["prototype"]}
    slot_sessions = tuple(sorted(observed_rows))
    slot_by_session = {session: slot_u[:, index] for index, session in enumerate(slot_sessions)}
    slot_session_passes = {
        session: observed_rows[session] > float(np.median(slot_by_session[session])) for session in slot_sessions
    }
    slot_pass = bool(
        float(slot_rank["exact_upper_tail_p_value"]) < 0.025
        and observed_u > float(np.median(slot_values))
        and all(slot_session_passes.values())
    )
    result["slot_null_exact"] = {
        "plan": slot_plan, "rank": slot_rank,
        "session_order": list(slot_sessions),
        "canonical_U_above_full_13824_null_median": slot_session_passes,
        "canonical_identity_rank_diagnostic_descriptive_only": slot_conditional,
        "all_per_schedule_per_fold_bounded_U": slot_u,
    }
    result["stage_status"]["stage1_exact_relative_slot"] = "pass" if slot_pass else "fail"
    result["runtime"]["stages_seconds"]["stage1"] = time.perf_counter() - stage1_start
    result["runtime"]["elapsed_seconds"] = time.perf_counter() - run_start
    if not slot_pass:
        result["decision"] = {
            "decision": "slot_coordinate_not_distinguishable_stop", "gpu_authorized": False, "decoder_authorized": False,
            "time_order_null": "not_run_due_to_predeclared_early_stop",
        }
        if result["runtime"]["elapsed_seconds"] > deadline_seconds:
            result["decision"]["decision"] = "fixed_null_budget_infeasible_stop"
        return _write_audit_result(output_dir, result)

    # Stage 2 is intentionally unreachable unless both preceding stages pass.
    stage2_start = time.perf_counter()
    time_sessions, time_u = time_order_null_score_matrix(support, targets, replicates=TIME_ORDER_NULL_REPLICATES)
    time_values = time_u.mean(axis=1)
    time_tail = scheduled_time_upper_rank(observed_u, time_values)
    time_by_session = {session: time_u[:, index] for index, session in enumerate(time_sessions)}
    gate = a2_cpu_gate(
        content=content, slot_rank=slot_rank,
        time_rank=time_tail, observed_h=observed_u, slot_h=slot_values, time_h=time_values,
        observed_u_by_session=observed_rows, slot_u_by_session=slot_by_session, time_u_by_session=time_by_session,
        validity_contract_pass=base_validity and np.isfinite(slot_u).all() and np.isfinite(time_u).all(),
        repeatability_contract_pass=repeatability_pass,
    )
    result["time_order_null_scheduled_reference"] = {
        "plan": time_null_receipt(support, replicates=TIME_ORDER_NULL_REPLICATES), "tail": time_tail,
        "session_order": list(time_sessions), "all_per_schedule_per_fold_bounded_U": time_u,
    }
    result["stage_status"]["stage2_time_order"] = "pass" if gate["time_order_pass"] else "fail"
    result["decision"] = gate
    result["runtime"]["stages_seconds"]["stage2"] = time.perf_counter() - stage2_start
    result["runtime"]["elapsed_seconds"] = time.perf_counter() - run_start
    if result["runtime"]["elapsed_seconds"] > deadline_seconds:
        result["decision"] = {
            "decision": "fixed_null_budget_infeasible_stop", "gpu_authorized": False, "decoder_authorized": False,
            "reason": "complete stage exceeded immutable 24-hour wall-time ceiling",
        }
    return _write_audit_result(output_dir, result)


def _check_execution_review(review_path: Path, prelaunch_path: Path) -> None:
    review = json.loads(review_path.read_text())
    if review.get("authorization") != REVIEW_AUTHORIZATION:
        raise PermissionError("review authorization does not permit A2 source-only CPU audit")
    if review.get("prelaunch_receipt_sha256") != sha256(prelaunch_path):
        raise PermissionError("review is not bound to the exact immutable A2 prelaunch receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-prelaunch", action="store_true")
    parser.add_argument("--prelaunch-output", type=Path, default=PRELAUNCH_DEFAULT)
    parser.add_argument("--execute-source-audit", action="store_true")
    parser.add_argument("--reviewed-prelaunch", type=Path)
    parser.add_argument("--source-only-confirmation")
    parser.add_argument("--output", type=Path, default=AUDIT_DEFAULT)
    args = parser.parse_args()
    if args.write_prelaunch and args.execute_source_audit:
        raise ValueError("prelaunch and A2 execution must be separate invocations")
    if args.write_prelaunch:
        print(write_prelaunch(args.prelaunch_output)); return
    if not args.execute_source_audit:
        raise RuntimeError("refusing implicit data access; write a receipt or satisfy both A2 execution guards")
    prelaunch = args.prelaunch_output / "prelaunch_receipt.json"
    if args.reviewed_prelaunch is None or not prelaunch.is_file():
        raise PermissionError("A2 source audit requires a prelaunch receipt and separate reviewed-prelaunch authorization")
    if args.source_only_confirmation != SOURCE_ONLY_CONFIRMATION:
        raise PermissionError("A2 exact source-only confirmation missing")
    _check_execution_review(args.reviewed_prelaunch, prelaunch)
    print(run_source_audit(args.output))


if __name__ == "__main__":
    main()
