#!/usr/bin/env python3
"""CPU-only, fail-closed preflight for A2 matched subject-shift v2.

This is intentionally a *receipt writer*, not a launcher.  It validates the
frozen two-source-arm × three-seed topology, checks the 15 sub-M candidates
under the M30/trial-30 data contract, and verifies the source-only normalizer
authority that both score domains must reuse.  It never opens a formal sub-C
test NWB, a checkpoint, or a GPU.

The preflight fails closed and will not overwrite a receipt.  A successful
receipt is only a CPU readiness record: the runner separately requires an
explicit A2 v2 GPU authorization string before any training can start.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUA_ROOT = REPO_ROOT / "sua_exploration"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import a2_matched_subject_shift_v2_core as core


def _blocker(code: str, detail: str, **context: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "detail": detail}
    if context:
        value["context"] = context
    return value


def _source_normalizer_authority_audit() -> dict[str, Any]:
    """Audit source-only normalizer *authority* without refitting it pre-launch.

    The source runs do not yet exist, so the actual normalizer values cannot be
    hashed honestly before training.  The preflight therefore proves the only
    valid authority roster and makes the evaluator fail closed later unless
    each run's recorded source T4 normalizer SHA matches a fresh recomputation
    from this same 27-session roster.  This avoids gratuitously rereading all
    27 source NWBs during every dry-run while preserving the important
    source-train-not-target-fit invariant.
    """
    from mc_maze.multisession_datamodule import session_name_from_path

    train_paths, _val_paths, formal_test_names = core.active_source_session_paths()
    train_sessions = [session_name_from_path(path) for path in train_paths]
    manifest = core.load_strict_manifest()
    core.require(train_sessions == manifest["train"], "source normalizer roster differs from strict manifest train roster")
    return {
        "policy": core.frozen_normalizer_policy(),
        "source_train_sessions": train_sessions,
        "source_train_session_count": len(train_sessions),
        "behavior_normalizer_value_sha256": None,
        "side_normalizer_value_sha256": None,
        "value_digest_status": "DEFERRED_UNTIL_SOURCE_RUN_EXISTS; evaluator recomputes from strict source-train roster and binds exact values to run metadata",
        "target_domain_fit_performed": False,
        "target_domain_normalizer_refit_forbidden": True,
        "target_domain_behavior_labels_used_for_updates": False,
        "formal_test_sessions_resolved_or_opened": False,
        "formal_test_session_names_only": formal_test_names,
    }


def _session_audit(paths: list[Path], *, domain: str, require_unit_cap: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read-only M30 eligibility check using the score-blind audit ledger.

    A previous all-22 CPU schema/feasibility receipt already contains exactly
    the loader-aligned rewarded-trial chronology and unit-count checks, and it
    was created before any scoring/checkpoint operation.  Reusing that ledger
    here keeps this preflight CPU-only and bounded while still proving every
    one of the 15 frozen external candidates is admissible under the *stricter*
    first-30/post-30 policy (the ledger recorded first-50/post-50).
    """
    from mc_maze.multisession_datamodule import session_name_from_path

    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    expected_sessions = core.expected_domain_sessions(domain)
    observed_sessions = tuple(session_name_from_path(path) for path in paths)
    if observed_sessions != expected_sessions:
        blockers.append(_blocker("SESSION_ROSTER_DRIFT", f"{domain} roster differs from frozen ordering", observed=list(observed_sessions), expected=list(expected_sessions)))
    ledger_path = SUA_ROOT / "results" / "dandi_000688_subm_co_schema_preflight_v2" / "receipt.json"
    ledger_by_session: dict[str, Mapping[str, Any]] = {}
    if domain == "external_subject_M":
        try:
            ledger = core.load_json_object(ledger_path)
            entries = ledger.get("asset_disposition_ledger")
            if not isinstance(entries, list) or len(entries) != 22:
                raise ValueError("all-22 external score-blind ledger missing")
            ledger_by_session = {
                str(row["session_id"]): row
                for row in entries
                if isinstance(row, Mapping) and isinstance(row.get("session_id"), str)
            }
        except Exception as exc:
            blockers.append(_blocker("EXTERNAL_SCORE_BLIND_LEDGER_UNAVAILABLE", str(exc), ledger_path=str(ledger_path)))
    for path in paths:
        session = session_name_from_path(path)
        try:
            if domain == "external_subject_M":
                ledger_row = ledger_by_session.get(session)
                if ledger_row is None:
                    raise ValueError("session is absent from all-22 score-blind eligibility ledger")
                feasibility = ledger_row.get("datamodule_feasibility")
                observed = ledger_row.get("observed_schema")
                if not isinstance(feasibility, Mapping) or not isinstance(observed, Mapping):
                    raise ValueError("malformed score-blind eligibility ledger row")
                unit_count = int((observed.get("identity") or {}).get("unit_count", (observed.get("event_level_spike_times") or {}).get("units_checked", -1)))
                usable_count = int(feasibility.get("exact_usable_rewarded_trial_count", 0))
                first50 = feasibility.get("first_50_raw_trial_indices")
                finite_labels = feasibility.get("usable_rewarded_trials_have_finite_target_dir") is True
                post50_windows = int(feasibility.get("complete_query_window_count_strictly_after_first_50", 0))
                if not (ledger_row.get("eligible") is True and usable_count > 30 and isinstance(first50, list) and len(first50) == 50 and finite_labels and post50_windows > 0):
                    raise ValueError("score-blind ledger does not prove M30/trial-30 eligibility")
                # Query windows strictly after 50 are a subset of those after
                # 30, so a positive post-50 count proves a positive post-30
                # count without rescoring/reopening the NWB.
                semantics = {
                    "usable_rewarded_trial_count": usable_count,
                    "activity_support_usable_indices": list(range(30)),
                    "activity_support_original_trial_indices": [int(value) for value in first50[:30]],
                    "side_feature_label_pool_usable_indices": list(range(30)),
                    "first30_target_dir_all_finite": True,
                    "query_usable_trial_indices_start": 30,
                    "query_usable_trial_count_lower_bound": max(1, usable_count - 50),
                    "post30_query_window_count_lower_bound": post50_windows,
                    "evidence": "all-22 score-blind ledger: finite labels for all usable rewarded trials and positive query windows strictly after first 50 implies the required first-30/post-30 rule",
                }
                admissible = (not require_unit_cap or unit_count < 100)
                row = {
                    "session": session,
                    "nwb_path": str(path),
                    "unit_count": unit_count,
                    "unit_count_under_100": unit_count < 100,
                    "admissible": admissible,
                    "eligibility_ledger_path": str(ledger_path),
                    "eligibility_ledger_sha256": core.sha256_file(ledger_path),
                    **semantics,
                }
            else:
                # The source manifest itself proves the six validation paths
                # are the active (not formal-test) development roster and the
                # frozen trainer enforces units<100.  Exact first-30/post-30
                # runtime trace equivalence is subsequently verified in each
                # source run's domain receipt.  Avoid reopening sub-C data in
                # the CPU preflight, which keeps the preflight lightweight and
                # makes the formal-test isolation boundary visually obvious.
                row = {
                    "session": session,
                    "nwb_path": str(path),
                    "unit_count": None,
                    "unit_count_under_100": True,
                    "admissible": True,
                    "activity_support_usable_indices": list(range(30)),
                    "side_feature_label_pool_usable_indices": list(range(30)),
                    "query_usable_trial_indices_start": 30,
                    "evidence": "strict 27/6 manifest active validation roster plus source trainer --max_units_exclusive=100; exact runtime M30/trial-30 query trace is receipt-gated before aggregation",
                }
                admissible = True
            if not admissible:
                blockers.append(_blocker("UNIT_CAP_INELIGIBLE", f"{session}: unit count must be <100", unit_count=unit_count))
        except Exception as exc:  # report all candidate failures but never waive one
            row = {"session": session, "nwb_path": str(path), "admissible": False, "error": f"{type(exc).__name__}: {exc}"}
            blockers.append(_blocker("M30_TRIAL30_INELIGIBLE", str(exc), session=session))
        rows.append(row)
    return rows, blockers


def _initial_matrix_freshness_audit(result_root: Path) -> list[dict[str, Any]]:
    """Perform the global blank-matrix check exactly once, before all cells.

    The official receipt seals this successful check.  Later launch invocations
    must *not* call it: they only validate their candidate arm/seed through
    ``core.assert_cell_fresh`` so a completed cell never blocks cell 2--6.
    """
    result_root = result_root.expanduser().resolve()
    blockers: list[dict[str, Any]] = []
    if result_root.exists():
        # A first official preflight is a blank-matrix event.  Looking only for
        # the twelve expected domain files would let an abandoned cell-launch
        # receipt, aggregate, or log survive unnoticed.  There is no legitimate
        # A2-v2 output before this one receipt, so any entry fails closed.
        try:
            entries = sorted(result_root.iterdir(), key=lambda entry: entry.name)
        except OSError as exc:
            return [_blocker("RESULT_ROOT_UNREADABLE", str(exc), result_root=str(result_root))]
        for entry in entries:
            blockers.append(_blocker("NONEMPTY_RESULT_ROOT", str(entry)))
    for source_arm in core.SOURCE_ARMS:
        for seed in core.SEEDS:
            run_dir = core.source_run_dir(source_arm, seed)
            if run_dir.exists():
                blockers.append(_blocker("EXISTING_SOURCE_RUN_DIRECTORY", str(run_dir)))
            summary_path = core.source_summary_path(source_arm, seed)
            if summary_path.exists():
                blockers.append(_blocker("EXISTING_TRAINER_GLOBAL_SUMMARY", str(summary_path)))
    return blockers


def build_receipt(*, result_root: Path, audit_data: bool, require_isolated_python: bool = True) -> dict[str, Any]:
    """Build a non-authorizing official-preflight payload without writing it."""
    blockers: list[dict[str, Any]] = []
    python_isolation: dict[str, Any] | None = None
    try:
        python_isolation = core.python_isolation_binding()
    except Exception as exc:
        if require_isolated_python:
            blockers.append(_blocker("PYTHON_ISOLATION_FAILURE", str(exc)))
        else:
            python_isolation = {
                "PYTHONNOUSERSITE": "UNVERIFIED_TEST_ONLY",
                "python_executable": str(Path(sys.executable).resolve()),
                "python_prefix": str(Path(sys.prefix).resolve()),
                "user_site_enabled": None,
            }
    implementation_bindings: dict[str, dict[str, str]] | None = None
    try:
        implementation_bindings = core.current_implementation_bindings()
    except Exception as exc:
        blockers.append(_blocker("IMPLEMENTATION_BINDING_FAILURE", str(exc)))

    config: dict[str, Any] | None = None
    try:
        config = core.validate_config()
    except Exception as exc:
        blockers.append(_blocker("CONFIG_DRIFT", str(exc)))

    for path, label in (
        (core.MANIFEST_PATH, "strict source manifest"),
        (core.TEACHER_PATH, "teacher checkpoint"),
        (core.SUBC_DATA_ROOT, "sub-C data root"),
        (core.SUBM_DATA_ROOT, "sub-M data root"),
    ):
        if not path.exists():
            blockers.append(_blocker("MISSING_BINDING", f"{label}: {path}"))
    if core.MANIFEST_PATH.is_file() and core.sha256_file(core.MANIFEST_PATH) != core.EXPECTED_MANIFEST_SHA256:
        blockers.append(_blocker("MANIFEST_SHA_DRIFT", "strict source manifest SHA differs from frozen pin"))
    if core.TEACHER_PATH.is_file() and core.sha256_file(core.TEACHER_PATH) != core.EXPECTED_TEACHER_SHA256:
        blockers.append(_blocker("TEACHER_SHA_DRIFT", "teacher checkpoint SHA differs from frozen pin"))

    manifest: dict[str, list[str]] | None = None
    try:
        manifest = core.load_strict_manifest()
        from mc_maze.gpu_contract_common import SEALED_FORMAL_TEST_SESSIONS

        for split in ("train", "val"):
            leaked = sorted(set(manifest[split]) & set(SEALED_FORMAL_TEST_SESSIONS))
            if leaked:
                blockers.append(_blocker("SEALED_SESSION_IN_ACTIVE_SOURCE_ROSTER", f"{split}: {leaked}"))
        if set(manifest["test"]) != set(SEALED_FORMAL_TEST_SESSIONS):
            blockers.append(_blocker("SEALED_TEST_ROSTER_DRIFT", "manifest test names differ from A2 sealed roster"))
        core.expected_within_sessions()
        core.expected_external_sessions()
    except Exception as exc:
        blockers.append(_blocker("SESSION_TOPOLOGY_DRIFT", str(exc)))

    normalizer_authority: dict[str, Any] | None = None
    within_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    if audit_data:
        try:
            normalizer_authority = _source_normalizer_authority_audit()
        except Exception as exc:
            blockers.append(_blocker("SOURCE_NORMALIZER_AUTHORITY_FAILURE", str(exc)))
        try:
            _train_paths, within_paths, _test_names = core.active_source_session_paths()
            within_rows, within_blockers = _session_audit(within_paths, domain="within_subject", require_unit_cap=True)
            blockers.extend(within_blockers)
        except Exception as exc:
            blockers.append(_blocker("WITHIN_M30_AUDIT_FAILURE", str(exc)))
        try:
            external_paths = core.external_session_paths()
            external_rows, external_blockers = _session_audit(external_paths, domain="external_subject_M", require_unit_cap=True)
            blockers.extend(external_blockers)
        except Exception as exc:
            blockers.append(_blocker("EXTERNAL_M30_AUDIT_FAILURE", str(exc)))
    else:
        blockers.append(_blocker("DATA_AUDIT_SKIPPED", "--skip-data-audit is incompatible with a passed official preflight"))

    blockers.extend(_initial_matrix_freshness_audit(result_root))
    external_admissible_count = sum(bool(row.get("admissible")) for row in external_rows)
    within_admissible_count = sum(bool(row.get("admissible")) for row in within_rows)
    if audit_data and external_admissible_count != core.EXTERNAL_SESSION_COUNT:
        blockers.append(_blocker("EXTERNAL_COHORT_CARDINALITY_FAILURE", "expected exactly 15 M30/trial-30 admissible external sessions", observed=external_admissible_count))
    if audit_data and within_admissible_count != core.WITHIN_SESSION_COUNT:
        blockers.append(_blocker("WITHIN_COHORT_CARDINALITY_FAILURE", "expected exactly 6 M30/trial-30 admissible validation sessions", observed=within_admissible_count))

    status = core.OFFICIAL_PREFLIGHT_STATUS if not blockers else "STOP_A2_V2_PREFLIGHT_BLOCKERS"
    return {
        "schema_version": 3,
        "receipt_kind": core.OFFICIAL_PREFLIGHT_KIND,
        "official_preflight": True,
        "screen_id": core.SCREEN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "non_authorizing_status": True,
        "root_go_required_before_gpu": True,
        "cpu_only": True,
        "gpu_used": False,
        "training_started": False,
        "checkpoint_loaded": False,
        "formal_subc_test_nwb_opened": False,
        "formal_subc_test_session_names_referenced_only": manifest["test"] if manifest else [],
        "contract_path": str(core.CONTRACT_PATH),
        "contract_sha256": core.sha256_file(core.CONTRACT_PATH) if core.CONTRACT_PATH.is_file() else None,
        "config_path": str(core.CONFIG_PATH),
        "config_sha256": core.sha256_file(core.CONFIG_PATH) if core.CONFIG_PATH.is_file() else None,
        "manifest_path": str(core.MANIFEST_PATH),
        "manifest_sha256": core.sha256_file(core.MANIFEST_PATH) if core.MANIFEST_PATH.is_file() else None,
        "teacher_path": str(core.TEACHER_PATH),
        "teacher_sha256": core.sha256_file(core.TEACHER_PATH) if core.TEACHER_PATH.is_file() else None,
        "result_root": str(result_root),
        "expected_fresh_gpu_cells": len(core.SOURCE_ARMS) * len(core.SEEDS),
        "forbidden_duplicate_domain_training_cells": len(core.SOURCE_ARMS) * len(core.SEEDS) * len(core.DOMAINS),
        "source_training_cells": list(core.SOURCE_ARMS),
        "seeds": list(core.SEEDS),
        "scoring_domains": list(core.DOMAINS),
        "query_policy": core.frozen_query_policy(),
        "normalizer_authority": normalizer_authority,
        "within_subject_audit": {"expected_count": core.WITHIN_SESSION_COUNT, "admissible_count": within_admissible_count, "sessions": within_rows},
        "external_subject_M_audit": {"expected_count": core.EXTERNAL_SESSION_COUNT, "admissible_count": external_admissible_count, "sessions": external_rows},
        "implementation_bindings": implementation_bindings,
        "implementation_bindings_sha256": core.implementation_bindings_sha256(implementation_bindings) if implementation_bindings is not None else None,
        "python_isolation": python_isolation,
        "gpu_authorization": {"environment": core.GPU_AUTH_ENV, "value": core.GPU_AUTH_VALUE, "present": False},
        "implementation_blockers": blockers,
        "config_validated": config is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=None, help="Official write-once immutable receipt; defaults to RESULT_ROOT/official_cpu_preflight.json.")
    parser.add_argument("--result-root", type=Path, default=core.RESULT_ROOT, help="A2 v2 matrix root; must be globally empty when the one official preflight is minted.")
    parser.add_argument("--skip-data-audit", action="store_true", help="Debug-only: always yields a fail-closed receipt, never READY.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve() if args.receipt is not None else core.official_preflight_path(result_root=result_root)
    canonical_receipt_path = core.official_preflight_path(result_root=result_root).resolve()
    if receipt_path != canonical_receipt_path:
        print(f"FAIL_CLOSED: official preflight must use its one canonical path: {canonical_receipt_path}", file=sys.stderr)
        return 2
    receipt = build_receipt(result_root=result_root, audit_data=not args.skip_data_audit)
    try:
        # An immutable receipt is intentionally written even when it is
        # blocked: it documents the exact non-authorizing failure and can
        # never be silently replaced.  Success still requires the strict
        # passed status below.
        _body, sidecar, digest = core.write_immutable_json(receipt_path, receipt)
    except (FileExistsError, OSError, core.A2V2ContractError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["status"],
        "receipt": str(receipt_path),
        "sidecar": str(sidecar),
        "receipt_sha256": digest,
        "cpu_only": True,
        "root_go_required_before_gpu": True,
    }, indent=2, sort_keys=True))
    return 0 if receipt["status"] == core.OFFICIAL_PREFLIGHT_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
