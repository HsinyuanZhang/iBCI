#!/usr/bin/env python3
"""Read-only readiness verifier for the deferred RT AFC4 XLS-v2 program.

This deliberately lives outside the active RT train/data/model/evaluator import
graph.  It reads only source text and immutable JSON receipts: the frozen XLS
v2 support audit, the reviewed integration draft, and (when it exists) the
MB4 15-fold receipt-only aggregate.  It never enumerates or opens an NWB,
imports a DataModule/model, starts a Trainer/CUDA process, or launches tmux.

The output is a handoff checklist, not launch authorization.  In particular,
an absent MB4 aggregate yields a useful ``WAITING`` state rather than a
workaround or a partial XLSv2 run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import stat
import sys
from typing import Any, Mapping


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
FOLDS = tuple(range(15))
SEED = 42
M = 24
QUERY_START = 24
WINDOW_SIZE = 50
GPU1_ASC_FOLDS = tuple(range(0, 8))
GPU0_DESC_FOLDS = tuple(range(14, 7, -1))
AUDIT_SCHEMA = "rt_afc4_ls_null_strength_support_audit_v2"
AUDIT_STATUS = "PASS_CPU_SUPPORT_ONLY_RT_AFC4_LS_NULL_STRENGTH_AUDIT_V2"
MB4_SCHEMA = "rt_mb4_matched_full_minus_mb4_aggregate_v1"
MB4_STATUS = "PASS_RT_FULL_MINUS_MB4_ALL_15_PAIRED"
DRAFT_SCHEMA = "rt_afc4_xls_v2_matched_plan_draft_v1"
DRAFT_STATUS = "DRAFT_NOT_LAUNCHED_NO_ACTIVE_INTEGRATION"
DEFAULT_DRAFT = WORKSPACE / "sua_exploration/results/rt_afc4_xls_v2_integration_draft_v1/RT_AFC4_XLS_V2_MATCHED_PLAN_DRAFT_v1.json"
DEFAULT_MB4_AGGREGATE = WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_FULL15_AGGREGATE_v1.json"
DEFAULT_BLUEPRINT = WORKSPACE / "sua_exploration/docs/RT_AFC4_XLS_V2_INTEGRATION_BLUEPRINT_20260808.md"
DEFAULT_PRIMITIVE = PROJECT / "src/data/afc4_xls_v2.py"
DEFAULT_FUTURE_ROOTS = {
    "raw_root": PROJECT / "outputs/rt_xls_v2_matched_clean_nested_v1",
    "import_root": WORKSPACE / "sua_exploration/results/rt_xls_v2_matched_clean_nested_v1/imported_cells",
    "aggregate_output": WORKSPACE / "sua_exploration/results/rt_xls_v2_matched_clean_nested_v1/RT_XLS_V2_RC_MINUS_XLSV2_ALL15_v1.json",
}


class XlsV2ReadinessError(ValueError):
    """A non-compute XLSv2 prerequisite or isolation boundary drifted."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise XlsV2ReadinessError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_json(path: Path, *, schema: str, status: str, label: str,
                    require_immutable: bool = True) -> tuple[Path, dict[str, Any], str]:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"{label} must be a regular JSON file: {candidate}")
    if require_immutable:
        _need(stat.S_IMODE(candidate.stat().st_mode) == 0o444,
              f"{label} must be an immutable mode-0444 JSON file: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise XlsV2ReadinessError(f"{label} is not readable JSON: {candidate}") from error
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"{label} schema/status drift: {candidate}")
    return candidate, body, _sha256(candidate)


def static_partitions() -> dict[str, dict[str, Any]]:
    """The sole 15-fold split; it is independent of any score or runtime state."""

    left, right = set(GPU1_ASC_FOLDS), set(GPU0_DESC_FOLDS)
    _need(not (left & right) and left | right == set(FOLDS), "XLSv2 static partitions do not cover folds 0..14 exactly once")
    return {
        "gpu1_ascending": {"physical_gpu": 1, "folds": list(GPU1_ASC_FOLDS), "order": "ascending"},
        "gpu0_descending": {"physical_gpu": 0, "folds": list(GPU0_DESC_FOLDS), "order": "descending"},
    }


def finalizer_command_template(*, raw_root: Path, import_root: Path, aggregate_output: Path,
                               mb4_aggregate: Path, audit_path: Path) -> str:
    """Return a template only; this function never creates the tmux session."""

    workers = ("rt_xls_v2_gpu1_asc", "rt_xls_v2_gpu0_desc")
    words = " ".join(shlex.quote(item) for item in workers)
    inner = " ".join((
        "set -euo pipefail;", "cd", shlex.quote(str(PROJECT)), ";",
        "for", "session", "in", words, ";", "do", "tmux", "has-session", "-t", '"$session"', "2>/dev/null", "||",
        "{", "echo", '"required XLSv2 worker session was never observed: $session"', ">&2;", "exit", "2;", "};", "done;",
        "while", "tmux", "has-session", "-t", shlex.quote(workers[0]), "2>/dev/null", "||", "tmux", "has-session", "-t", shlex.quote(workers[1]), "2>/dev/null;",
        "do", "sleep", "15;", "done;",
        shlex.quote(sys.executable), "scripts/rt_xls_v2_matched_continuation_v1.py", "--copy-import-all-and-finalize",
        "--work-root", shlex.quote(str(raw_root)), "--import-root", shlex.quote(str(import_root)),
        "--aggregate-output", shlex.quote(str(aggregate_output)), "--mb4-aggregate", shlex.quote(str(mb4_aggregate)),
        "--xls-v2-audit", shlex.quote(str(audit_path)),
    ))
    return "tmux new-session -d -s rt_xls_v2_import_finalize_after_workers " + shlex.quote(inner)


def _validate_draft(draft_path: Path, *, blueprint_path: Path, primitive_path: Path) -> tuple[Path, dict[str, Any], str, Path, str]:
    # The input is explicitly a mutable draft (not evidence); its byte hash is
    # recorded below, while only the support audit and future MB4 aggregate are
    # required to be immutable.  Changing its mode here would be an unrelated
    # workspace mutation and would not make it a reviewed implementation.
    path, draft, digest = _immutable_json(
        draft_path, schema=DRAFT_SCHEMA, status=DRAFT_STATUS, label="XLSv2 integration draft", require_immutable=False,
    )
    arm, protocol, roots = draft.get("arm"), draft.get("frozen_protocol"), draft.get("new_roots_only")
    _need(isinstance(arm, Mapping) and arm.get("canonical_arm") == "afc4_xls_v2" and arm.get("side_dim") == 4 and
          arm.get("descriptor_coordinates") == ["W_x", "W_y", "W_norm", "b"], "XLSv2 draft arm/width contract drift")
    forbidden = arm.get("forbidden_compensation")
    _need(isinstance(forbidden, list) and "checkpoint reuse or warm start from R-C" in forbidden and
          "source-or-target shared 2x2 inverse" in forbidden, "XLSv2 inverse/warm-start prohibition drift")
    _need(isinstance(protocol, Mapping) and protocol.get("folds") == list(FOLDS) and protocol.get("seed") == SEED and
          protocol.get("support_trial_index_range") == [0, M] and protocol.get("query_start_trial") == QUERY_START and
          protocol.get("window_size_bins") == WINDOW_SIZE and
          protocol.get("checkpoint_selection") == "inner validation only: val_heldin/r2_mean", "XLSv2 frozen protocol drift")
    _need(isinstance(roots, Mapping) and set(roots) == set(DEFAULT_FUTURE_ROOTS), "XLSv2 draft root set drift")
    _need(len(set(roots.values())) == 3, "XLSv2 raw/import/aggregate roots collide")
    audit = draft.get("v2_audit_binding")
    _need(isinstance(audit, Mapping), "XLSv2 draft audit binding missing")
    audit_path = (WORKSPACE / str(audit.get("path", ""))).resolve()
    audit_loaded, body, audit_sha = _immutable_json(audit_path, schema=AUDIT_SCHEMA, status=AUDIT_STATUS, label="XLSv2 support audit")
    _need(audit_sha == audit.get("sha256") and audit.get("seed") == SEED, "XLSv2 audit file/seed SHA binding drift")
    rows = body.get("fold_rows")
    expected = audit.get("per_session_expected_permutation_sha256")
    _need(isinstance(rows, list) and len(rows) == len(FOLDS) and isinstance(expected, Mapping), "XLSv2 support audit rows absent")
    actual = {
        row.get("session_name"): row.get("v2_random_cross_reach_null", {}).get("permutation_sha256")
        for row in rows if isinstance(row, Mapping)
    }
    _need(len(actual) == len(FOLDS) and actual == expected and all(isinstance(value, str) and len(value) == 64 for value in actual.values()),
          "XLSv2 per-session permutation SHA parity drift")
    _need(blueprint_path.is_file() and primitive_path.is_file(), "XLSv2 blueprint or isolated primitive is missing")
    blueprint = blueprint_path.read_text(encoding="utf-8")
    primitive = primitive_path.read_text(encoding="utf-8")
    for token in ("afc4_xls_v2", "side_dim=4", "one-shot", "共同 `2×2` inverse", "不启动"):
        _need(token in blueprint, f"XLSv2 blueprint lost required term {token!r}")
    for forbidden_token in ("pynwb", "torch", "falcon_k4_features", "DataModule", "lstsq"):
        _need(forbidden_token not in primitive, f"XLSv2 primitive ceased to be isolated: {forbidden_token!r}")
    return path, draft, digest, audit_loaded, audit_sha


def _mb4_gate(path: Path) -> dict[str, Any]:
    """Return WAITING for an absent receipt; malformed/premature evidence fails closed."""

    candidate = path.resolve()
    if not candidate.exists():
        return {"state": "WAITING_FOR_MB4_FULL15_IMMUTABLE_AGGREGATE", "path": str(candidate), "exists": False}
    aggregate_path, aggregate, digest = _immutable_json(candidate, schema=MB4_SCHEMA, status=MB4_STATUS,
                                                         label="MB4 Full15 aggregate")
    rows = aggregate.get("rows")
    _need(isinstance(rows, list) and len(rows) == len(FOLDS), "MB4 aggregate lacks exactly 15 paired rows")
    folds = [row.get("fold") for row in rows if isinstance(row, Mapping)]
    _need(tuple(sorted(folds)) == FOLDS, "MB4 aggregate fold matrix drift")
    _need(aggregate.get("arm") == "afc4_mb4" and aggregate.get("full_comparator_arm") == "afc4_vel" and
          aggregate.get("seed") == SEED, "MB4 aggregate comparator/seed drift")
    return {"state": "PASS_MB4_FULL15_IMMUTABLE_AGGREGATE", "path": str(aggregate_path), "sha256": digest,
            "folds": list(FOLDS)}


def audit_readiness(*, draft_path: Path = DEFAULT_DRAFT, mb4_aggregate: Path = DEFAULT_MB4_AGGREGATE,
                    blueprint_path: Path = DEFAULT_BLUEPRINT, primitive_path: Path = DEFAULT_PRIMITIVE) -> dict[str, Any]:
    """Read-only static review that can be rerun before any future XLSv2 apply."""

    draft, body, draft_sha, audit_path, audit_sha = _validate_draft(
        draft_path.resolve(), blueprint_path=blueprint_path.resolve(), primitive_path=primitive_path.resolve(),
    )
    # Draft roots are workspace-relative.  Resolving them against the process
    # CWD would silently produce ``streaming_calibration_exp/streaming_...``
    # for the raw root when this verifier is invoked from the project folder.
    roots = {name: (WORKSPACE / str(value)).resolve() for name, value in body["new_roots_only"].items()}
    for name, path in roots.items():
        _need(not path.exists(), f"future XLSv2 {name} must remain absent before reviewed apply: {path}")
    mb4 = _mb4_gate(mb4_aggregate)
    status = ("READY_FOR_REVIEWED_XLSV2_APPLY_NOT_LAUNCHED" if mb4["state"].startswith("PASS_")
              else "WAITING_FOR_MB4_FULL15_IMMUTABLE_AGGREGATE")
    partitions = static_partitions()
    return {
        "schema": "rt_xls_v2_post_mb4_readiness_audit_v1", "status": status,
        "mode": "read_only_source_text_and_immutable_receipts_no_nwb_no_trainer_no_cuda_no_tmux",
        "draft": {"path": str(draft), "sha256": draft_sha},
        "support_audit": {"path": str(audit_path), "sha256": audit_sha, "seed": SEED,
                          "per_target_session_permutation_sha_parity": "validated_exactly"},
        "mb4_gate": mb4,
        "frozen_protocol": {"folds": list(FOLDS), "seed": SEED, "support_trial_index_range": [0, M],
                            "query_start_trial": QUERY_START, "window_size_bins": WINDOW_SIZE,
                            "fresh_source_training_per_fold": True, "reuse_r_c_checkpoint": False,
                            "common_inverse_or_alignment_map": "FORBIDDEN", "outer_target_backpropagation": False,
                            "outer_target_optimizer": False, "outer_evaluation": "one_shot",
                            "query_labels": "scoring_only", "all_folds_required": True},
        "static_partitions": partitions,
        "new_roots": {name: str(path) for name, path in roots.items()},
        "finalizer_template": finalizer_command_template(raw_root=roots["raw_root"], import_root=roots["import_root"],
                                                           aggregate_output=roots["aggregate_output"],
                                                           mb4_aggregate=mb4_aggregate.resolve(), audit_path=audit_path),
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0, "trainer_constructed_or_launched": False,
                  "cuda_constructed_or_launched": False, "tmux_session_created": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--mb4-aggregate", type=Path, default=DEFAULT_MB4_AGGREGATE)
    parser.add_argument("--blueprint", type=Path, default=DEFAULT_BLUEPRINT)
    parser.add_argument("--primitive", type=Path, default=DEFAULT_PRIMITIVE)
    args = parser.parse_args()
    print(json.dumps(audit_readiness(draft_path=args.draft, mb4_aggregate=args.mb4_aggregate,
                                     blueprint_path=args.blueprint, primitive_path=args.primitive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
