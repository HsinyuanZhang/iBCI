#!/usr/bin/env python3
"""Fail-closed source-only runner for the reviewed M1 B20 characterization.

Prelaunch mode is no-NWB.  Actual source execution is deliberately impossible
without a separate root authorization bound to the immutable prelaunch receipt,
the reviewed protocol, and the exact four-source manifest.  The runner remains
CPU/source-only and never authorizes a decoder, GPU, held-out, or EvalAI path.
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

from mc_maze.m1_b20_source_characterization import (  # noqa: E402
    A2_B20_R2_BY_SESSION,
    B20_CHARACTERIZATION_SEMANTICS_VERSION,
    RELIABILITY_REPEATS,
    RIDGE,
    ROW_ATTACHMENT_RANDOM_SCHEDULES,
    ROW_ATTACHMENT_TOTAL_SCHEDULES,
    a2_b20_reproduction_receipt,
    attachment_shuffled_outer_carriers,
    b20_split_reliability,
    build_outer_loso_carriers,
    characterization_contract_receipt,
    family_distinguishable,
    monte_carlo_upper_rank,
    outer_loso_proxy,
    row_attachment_schedule_receipt,
    terminal_characterization_decision,
)


MANIFEST = SUA / "manifests" / "m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json"
PROTOCOL = SUA / "docs" / "M1_B20_ROW_ATTACHMENT_CHARACTERIZATION_PROTOCOL.md"
PRELAUNCH_DEFAULT = SUA / "results" / "m1_b20_source_characterization_prelaunch_v5"
AUDIT_DEFAULT = SUA / "results" / "m1_b20_source_characterization_v1"
SCHEMA_PRELAUNCH = "m1_b20_source_characterization_prelaunch_v5"
SCHEMA_AUDIT = "m1_b20_source_characterization_v1"
REVIEW_AUTHORIZATION = "root_approved_m1_b20_source_characterization_v1"
SOURCE_ONLY_CONFIRMATION = "I_CONFIRM_EXACT_FOUR_M1_SOURCE_NWBS_CPU_ONLY_B20_V1"
EXPECTED_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
CPU_WORKERS = 1
CPU_WALL_TIME_SECONDS = 24 * 3600
BENCHMARK_RELIABILITY_REPEATS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, np.integer)):
        return int(value) if isinstance(value, np.integer) else value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json(item) for item in value]
    if hasattr(value, "__dict__"):
        return strict_json(vars(value))
    raise TypeError(f"unserializable value: {type(value)!r}")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_exact_source_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    """Read only strict manifest metadata; no source path is resolved here."""
    data = json.loads(path.read_text())
    if data.get("schema_version") != "m1_fixed_k_temporal_prototype_gate_a_source_manifest_v2":
        raise ValueError("B20 requires the exact audited v3 source manifest schema")
    rows = data.get("sessions")
    if not isinstance(rows, list) or tuple(row.get("session") for row in rows) != EXPECTED_SESSIONS:
        raise ValueError("B20 requires exactly the ordered four A2 source sessions")
    for row in rows:
        relative = str(row.get("relative_path", ""))
        if not relative.startswith("SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"):
            raise ValueError("B20 source leaves held-in-calib scope")
        if any(token in relative.lower() for token in ("held-out", "minival", "test", "evalai", "formal")):
            raise ValueError("B20 source contains prohibited path scope")
        if len(str(row.get("sha256", ""))) != 64:
            raise ValueError("B20 manifest source checksum is malformed")
    return data


def _synthetic_support() -> dict[str, tuple[np.ndarray, ...]]:
    """Fixed worst-case no-NWB support fixture: 4x10x1024x64 raw count bins."""
    return {
        name: tuple(np.broadcast_to(((session + trial + np.arange(64, dtype=np.int64)) % 5)[None, :], (1024, 64)).copy() for trial in range(10))
        for session, name in enumerate(EXPECTED_SESSIONS)
    }


def _synthetic_targets() -> dict[str, np.ndarray]:
    return {
        name: np.add.outer(np.arange(64, dtype=np.float64) * 0.001 + session, np.arange(4, dtype=np.float64) * 0.01)
        for session, name in enumerate(EXPECTED_SESSIONS)
    }


def conservative_no_nwb_runtime_benchmark() -> dict[str, Any]:
    """Project all 12,285 random scores plus 1,024 reliability repeats on synthetic worst case."""
    support = _synthetic_support()
    target = _synthetic_targets()
    start = time.perf_counter()
    base = build_outer_loso_carriers(support)
    outer_loso_proxy(base.public, target, ridge=RIDGE)
    canonical_seconds = time.perf_counter() - start
    start = time.perf_counter()
    outer_loso_proxy(attachment_shuffled_outer_carriers(base, replicate=1, family="A-all"), target, ridge=RIDGE)
    one_schedule_seconds = time.perf_counter() - start
    start = time.perf_counter()
    b20_split_reliability(
        support[EXPECTED_SESSIONS[0]], session_name=EXPECTED_SESSIONS[0], repeats=BENCHMARK_RELIABILITY_REPEATS
    )
    one_session_reliability_seconds = time.perf_counter() - start
    random_scores = 3 * ROW_ATTACHMENT_RANDOM_SCHEDULES
    reliability_projection = 4 * one_session_reliability_seconds * (RELIABILITY_REPEATS / BENCHMARK_RELIABILITY_REPEATS)
    projected = canonical_seconds + random_scores * one_schedule_seconds + reliability_projection
    return {
        "kind": "deterministic_synthetic_no_NWB_worst_case_projection",
        "shape": {"sessions": 4, "support_trials": 10, "valid_bins_per_trial": 1024, "channels": 64},
        "fixed_cpu_workers": CPU_WORKERS,
        "canonical_score_seconds": canonical_seconds,
        "one_random_schedule_score_seconds": one_schedule_seconds,
        "one_session_benchmark_reliability_seconds": one_session_reliability_seconds,
        "benchmark_reliability_repeats": BENCHMARK_RELIABILITY_REPEATS,
        "random_schedule_scores": random_scores,
        "projection_seconds": {
            "all_12285_random_schedule_scores": random_scores * one_schedule_seconds,
            "all_four_sessions_1024_repeat_reliability": reliability_projection,
            "canonical_and_total": projected,
        },
        "within_24h_fixed_worker_projection": bool(projected <= CPU_WALL_TIME_SECONDS),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def build_prelaunch_receipt() -> dict[str, Any]:
    """Write a hash-bound no-NWB readiness receipt; fail if worst-case projection is too slow."""
    manifest = load_exact_source_manifest()
    if not PROTOCOL.is_file():
        raise FileNotFoundError("reviewed B20 protocol is required before prelaunch")
    benchmark = conservative_no_nwb_runtime_benchmark()
    if not benchmark["within_24h_fixed_worker_projection"]:
        raise RuntimeError("B20 worst-case no-NWB projection exceeds frozen 24-hour CPU ceiling")
    return {
        "schema_version": SCHEMA_PRELAUNCH,
        "semantics_version": B20_CHARACTERIZATION_SEMANTICS_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "audit_m1_b20_source_characterization.py --write-prelaunch",
        "purpose": "root-review-only M1 B20 source characterization prelaunch; not a result",
        "manifest": {"path": display_path(MANIFEST), "sha256": sha256(MANIFEST), "sessions": manifest["sessions"]},
        "protocol": {"path": display_path(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "code": {
            "runner": {"path": display_path(Path(__file__)), "sha256": sha256(Path(__file__))},
            "pure_contracts": {"path": "sua_exploration/mc_maze/m1_b20_source_characterization.py", "sha256": sha256(SUA / "mc_maze" / "m1_b20_source_characterization.py")},
            "a2_exact_raw_loader": {"path": "sua_exploration/scripts/audit_m1_fixed_k_prototypes_gate_a.py", "sha256": sha256(SUA / "scripts" / "audit_m1_fixed_k_prototypes_gate_a.py")},
        },
        "contracts": characterization_contract_receipt(),
        "a2_reproduction": {"expected_r2_by_session": A2_B20_R2_BY_SESSION, "absolute_tolerance": 1.0e-10},
        "execution_plan": {
            "random_schedules_each_family": ROW_ATTACHMENT_RANDOM_SCHEDULES,
            "families": ["A-all", "A-R", "A-Q"],
            "total_random_schedule_scores": 3 * ROW_ATTACHMENT_RANDOM_SCHEDULES,
            "reliability_repeats_per_session": RELIABILITY_REPEATS,
            "fixed_cpu_workers": CPU_WORKERS,
            "wall_time_ceiling_seconds": CPU_WALL_TIME_SECONDS,
            "runtime_benchmark": benchmark,
        },
        "hard_exclusions": {"nwb_opened_for_this_receipt": False, "cuda": False, "decoder": False, "minival": False, "held_out": False, "formal": False, "evalai": False},
        "execution_guard": {"review_authorization": REVIEW_AUTHORIZATION, "source_only_confirmation": SOURCE_ONLY_CONFIRMATION, "fresh_output_required": True},
        "status": "pending_root_review_no_nwb_opened",
    }


def write_prelaunch(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite B20 prelaunch output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "prelaunch_receipt.json"
    path.write_text(json.dumps(strict_json(build_prelaunch_receipt()), indent=2, sort_keys=True) + "\n")
    (output_dir / "prelaunch_receipt.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return path


def _load_sessions_after_review(manifest: Mapping[str, Any]):
    """Reuse the audited A2/v3 exact raw loader only after both CLI guards pass."""
    from scripts.audit_m1_fixed_k_prototypes_gate_a import load_raw_source_sessions

    sessions = load_raw_source_sessions(manifest)
    if tuple(sessions) != EXPECTED_SESSIONS:
        raise RuntimeError("exact A2 raw loader did not return the immutable B20 source order")
    if any(int(row.support_trials[0].shape[1]) != 64 for row in sessions.values()):
        raise RuntimeError("B20 source requires exactly 64 channel rows in every session")
    return sessions


def _family_score_matrix(base, targets: Mapping[str, np.ndarray], *, family: str) -> tuple[dict[str, list[dict[str, object]]], np.ndarray]:
    """Score all 4,095 frozen random schedules for one family; no adaptive stopping inside a family."""
    sessions = tuple(sorted(targets))
    rows: dict[str, list[dict[str, object]]] = {session: [] for session in sessions}
    h_values = np.empty(ROW_ATTACHMENT_RANDOM_SCHEDULES, dtype=np.float64)
    for index, replicate in enumerate(range(1, ROW_ATTACHMENT_RANDOM_SCHEDULES + 1)):
        proxy = outer_loso_proxy(attachment_shuffled_outer_carriers(base, replicate=replicate, family=family), targets, ridge=RIDGE)
        family_rows = proxy["arms"][family]
        per_session = {str(row["left_out_session"]): row for row in family_rows}
        if set(per_session) != set(sessions):
            raise RuntimeError("B20 random family source fold mismatch")
        h_values[index] = float(np.mean([float(per_session[session]["bounded_u"]) for session in sessions]))
        for session in sessions:
            rows[session].append(dict(per_session[session]))
    return rows, h_values


def _observed_by_session(proxy: Mapping[str, object]) -> tuple[dict[str, float], dict[str, float], float]:
    rows = proxy["arms"]["B20"]
    utility = {str(row["left_out_session"]): float(row["bounded_u"]) for row in rows}
    r2 = {str(row["left_out_session"]): float(row["r2"]) for row in rows}
    if set(utility) != set(EXPECTED_SESSIONS) or set(r2) != set(EXPECTED_SESSIONS):
        raise RuntimeError("canonical B20 folds mismatch")
    return utility, r2, float(np.mean(list(utility.values())))


def _family_result(base, targets: Mapping[str, np.ndarray], *, family: str, observed_u: Mapping[str, float], observed_h: float) -> dict[str, object]:
    rows, h_random = _family_score_matrix(base, targets, family=family)
    rank = monte_carlo_upper_rank(observed_h, np.concatenate(([observed_h], h_random)))
    per_session_u = {session: [float(row["bounded_u"]) for row in values] for session, values in rows.items()}
    score_payload = json.dumps(strict_json(rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    h_payload = np.asarray(h_random, dtype="<f8").tobytes(order="C")
    return {
        "family": family, "random_scores_by_left_out_session": rows, "random_H": h_random,
        "score_matrix_checksum": {
            "random_per_fold_rows_sha256": hashlib.sha256(score_payload).hexdigest(),
            "random_H_little_endian_float64_sha256": hashlib.sha256(h_payload).hexdigest(),
            "convention": "rows JSON sort_keys compact separators; H is C-order little-endian float64 length 4095",
        },
        "rank": rank, "criterion": family_distinguishable(observed_u_by_session=observed_u, null_u_by_session=per_session_u, rank=rank),
    }


def _write_result(output_dir: Path, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "source_characterization.json"
    path.write_text(json.dumps(strict_json(result), indent=2, sort_keys=True) + "\n")
    (output_dir / "source_characterization.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return path


def run_source_characterization(output_dir: Path, *, prelaunch_path: Path, review_path: Path) -> Path:
    """Run only after CLI authorization; source-only CPU characterization with immutable stages."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite B20 source output: {output_dir}")
    started = time.perf_counter()
    manifest = load_exact_source_manifest()
    sessions = _load_sessions_after_review(manifest)
    from mc_maze.m1_fixed_k_prototype_gate_a import future_neural_oracle_target

    support = {name: row.support_trials for name, row in sessions.items()}
    targets = {name: future_neural_oracle_target(row) for name, row in sessions.items()}
    base = build_outer_loso_carriers(support)
    observed_proxy = outer_loso_proxy(base.public, targets, ridge=RIDGE)
    reproduction = a2_b20_reproduction_receipt(observed_proxy)
    observed_u, observed_r2, observed_h = _observed_by_session(observed_proxy)
    validity = bool(all(np.isfinite(value).all() for fold in base.public.values() for arm in fold.values() for value in arm.values()) and all(np.isfinite(value).all() for value in targets.values()))
    schedule = row_attachment_schedule_receipt(EXPECTED_SESSIONS, units=64)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_AUDIT, "semantics_version": B20_CHARACTERIZATION_SEMANTICS_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "M1 already-seen source-only B20 characterization; later-neural proxy only, no behavioural or deployment claim",
        "manifest": {"path": display_path(MANIFEST), "sha256": sha256(MANIFEST)},
        "protocol": {"path": display_path(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "execution_binding": {
            "prelaunch_receipt": {"path": display_path(prelaunch_path), "sha256": sha256(prelaunch_path)},
            "root_review": {"path": display_path(review_path), "sha256": sha256(review_path)},
            "runner_sha256": sha256(Path(__file__)),
            "pure_contracts_sha256": sha256(SUA / "mc_maze" / "m1_b20_source_characterization.py"),
        },
        "input_sessions": {
            name: {"path": str(row.source_path), "sha256": row.source_sha256}
            for name, row in sessions.items()
        },
        "row_attachment_schedule": schedule,
        "contracts": {"source_only": True, "M1_source_already_seen": True, "formal_held_out": False, "decoder_authorized": False, "gpu_authorized": False, "EvalAI_authorized": False, "validity": validity},
        "canonical_B20": {"proxy": observed_proxy, "observed_bounded_U_by_session": observed_u, "observed_r2_by_session": observed_r2, "observed_H": observed_h},
        "a2_reproduction": reproduction,
        "stage_status": {"canonical_and_reproduction": "pass" if validity and reproduction["pass"] else "fail", "A-all": "not_run_due_to_predeclared_early_stop", "A-R": "not_run_due_to_predeclared_early_stop", "A-Q": "not_run_due_to_predeclared_early_stop", "reliability": "not_run_due_to_predeclared_early_stop"},
        "runtime": {"fixed_cpu_workers": CPU_WORKERS, "wall_time_ceiling_seconds": CPU_WALL_TIME_SECONDS},
    }
    if not validity or not reproduction["pass"]:
        result["decision"] = terminal_characterization_decision(valid=validity, a2_reproduction_pass=bool(reproduction["pass"]), all_family=None, r_family=None, q_family=None, d_r_by_session=None, d_q_by_session=None)
        result["runtime"]["elapsed_seconds"] = time.perf_counter() - started
        return _write_result(output_dir, result)
    all_result = _family_result(base, targets, family="A-all", observed_u=observed_u, observed_h=observed_h)
    result["A-all"] = all_result; result["stage_status"]["A-all"] = "pass" if all_result["criterion"]["distinguishable"] else "fail"
    result["runtime"]["after_A_all_seconds"] = time.perf_counter() - started
    if result["runtime"]["after_A_all_seconds"] > CPU_WALL_TIME_SECONDS:
        result["decision"] = {"decision": "fixed_null_budget_infeasible_stop", "gpu_authorized": False, "decoder_authorized": False, "reason": "deadline exceeded after complete A-all before component families"}
        result["stage_status"]["A-R"] = "not_run_due_to_predeclared_deadline_stop"; result["stage_status"]["A-Q"] = "not_run_due_to_predeclared_deadline_stop"; result["stage_status"]["reliability"] = "not_run_due_to_predeclared_deadline_stop"
        result["runtime"]["elapsed_seconds"] = time.perf_counter() - started
        return _write_result(output_dir, result)
    if not all_result["criterion"]["distinguishable"]:
        result["decision"] = terminal_characterization_decision(valid=True, a2_reproduction_pass=True, all_family=all_result["criterion"], r_family=None, q_family=None, d_r_by_session=None, d_q_by_session=None)
        result["runtime"]["elapsed_seconds"] = time.perf_counter() - started
        return _write_result(output_dir, result)
    r_result = _family_result(base, targets, family="A-R", observed_u=observed_u, observed_h=observed_h)
    q_result = _family_result(base, targets, family="A-Q", observed_u=observed_u, observed_h=observed_h)
    result["A-R"] = r_result; result["A-Q"] = q_result
    result["stage_status"]["A-R"] = "pass" if r_result["criterion"]["distinguishable"] else "fail"
    result["stage_status"]["A-Q"] = "pass" if q_result["criterion"]["distinguishable"] else "fail"
    result["runtime"]["after_component_families_seconds"] = time.perf_counter() - started
    if result["runtime"]["after_component_families_seconds"] > CPU_WALL_TIME_SECONDS:
        result["decision"] = {"decision": "fixed_null_budget_infeasible_stop", "gpu_authorized": False, "decoder_authorized": False, "reason": "deadline exceeded after both complete component families before reliability"}
        result["stage_status"]["reliability"] = "not_run_due_to_predeclared_deadline_stop"
        result["runtime"]["elapsed_seconds"] = time.perf_counter() - started
        return _write_result(output_dir, result)
    d_r = [observed_r2[session] - float(np.median([row["r2"] for row in r_result["random_scores_by_left_out_session"][session]])) for session in EXPECTED_SESSIONS]
    d_q = [observed_r2[session] - float(np.median([row["r2"] for row in q_result["random_scores_by_left_out_session"][session]])) for session in EXPECTED_SESSIONS]
    result["component_loss_r2_by_session"] = {"D_R": dict(zip(EXPECTED_SESSIONS, d_r)), "D_Q": dict(zip(EXPECTED_SESSIONS, d_q))}
    result["reliability"] = {name: b20_split_reliability(support[name], session_name=name, repeats=RELIABILITY_REPEATS) for name in EXPECTED_SESSIONS}
    result["stage_status"]["reliability"] = "completed_diagnostic_only"
    result["decision"] = terminal_characterization_decision(valid=True, a2_reproduction_pass=True, all_family=all_result["criterion"], r_family=r_result["criterion"], q_family=q_result["criterion"], d_r_by_session=d_r, d_q_by_session=d_q)
    result["runtime"]["elapsed_seconds"] = time.perf_counter() - started
    if result["runtime"]["elapsed_seconds"] > CPU_WALL_TIME_SECONDS:
        result["decision"] = {"decision": "fixed_null_budget_infeasible_stop", "gpu_authorized": False, "decoder_authorized": False}
    return _write_result(output_dir, result)


def _check_execution_review(review_path: Path, prelaunch_path: Path) -> None:
    prelaunch = json.loads(prelaunch_path.read_text())
    if prelaunch.get("schema_version") != SCHEMA_PRELAUNCH:
        raise PermissionError("prelaunch schema does not match B20 execution contract")
    expected_bindings = {
        ("manifest", "sha256"): sha256(MANIFEST),
        ("protocol", "sha256"): sha256(PROTOCOL),
        ("code", "runner", "sha256"): sha256(Path(__file__)),
        ("code", "pure_contracts", "sha256"): sha256(SUA / "mc_maze" / "m1_b20_source_characterization.py"),
        ("code", "a2_exact_raw_loader", "sha256"): sha256(SUA / "scripts" / "audit_m1_fixed_k_prototypes_gate_a.py"),
    }
    for path, expected in expected_bindings.items():
        current: Any = prelaunch
        for key in path:
            if not isinstance(current, Mapping):
                raise PermissionError("prelaunch missing a required immutable binding")
            current = current.get(key)
        if current != expected:
            raise PermissionError(f"prelaunch binding is stale or mismatched at {'.'.join(path)}")
    review = json.loads(review_path.read_text())
    if review.get("authorization") != REVIEW_AUTHORIZATION:
        raise PermissionError("root review does not authorize B20 source-only characterization")
    if review.get("prelaunch_receipt_sha256") != sha256(prelaunch_path):
        raise PermissionError("root review is not bound to this immutable B20 prelaunch receipt")
    if review.get("protocol_sha256") != sha256(PROTOCOL) or review.get("manifest_sha256") != sha256(MANIFEST):
        raise PermissionError("root review protocol/manifest binding mismatch")


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
        raise ValueError("prelaunch and source execution must be separate invocations")
    if args.write_prelaunch:
        print(write_prelaunch(args.prelaunch_output)); return
    if not args.execute_source_audit:
        raise RuntimeError("refusing implicit source access; write prelaunch or satisfy both B20 execution guards")
    prelaunch = args.prelaunch_output / "prelaunch_receipt.json"
    if args.reviewed_prelaunch is None or not prelaunch.is_file():
        raise PermissionError("B20 execution requires immutable prelaunch plus a separate root review")
    if args.source_only_confirmation != SOURCE_ONLY_CONFIRMATION:
        raise PermissionError("B20 exact source-only confirmation missing")
    _check_execution_review(args.reviewed_prelaunch, prelaunch)
    print(run_source_characterization(args.output, prelaunch_path=prelaunch, review_path=args.reviewed_prelaunch))


if __name__ == "__main__":
    main()
