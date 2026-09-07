"""Isolated RT 15-fold source-only authority materialization runner.

This additive route materializes *only* a requested outer fold's 14 source
sessions in a fresh child process.  It never loads all 15 RT sessions and then
slices them, never accepts a target path or target session argument, and never
imports CEBRA, opens a checkpoint, uses CUDA, or scores a query.

The public smoke launcher runs exactly one fixed fold.  The full 15-fold
launcher is deliberately Python-only and requires an explicit root flag; it is
provided for a future separately authorised continuation and is not called by
this module, its CLI, or its tests.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_live_contract as live
import track_b_v2_metric_pointer_authority as metric_pointer
import track_b_v2_source_adapter as source


RT_SOURCE_FOLD_WORKER_SCHEMA = "track_b_v2_rt_source_only_fold_worker_receipt_v1"
RT_SOURCE_SMOKE_MANIFEST_SCHEMA = "track_b_v2_rt_source_only_smoke_manifest_v1"
RT_SOURCE_SMOKE_AGGREGATE_SCHEMA = "track_b_v2_rt_source_only_smoke_aggregate_v1"
RT_SOURCE_FULL_MANIFEST_SCHEMA = "track_b_v2_rt_15fold_source_authority_manifest_v1"
RT_SOURCE_FULL_AGGREGATE_SCHEMA = "track_b_v2_rt_15fold_source_authority_aggregate_v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = REPO_ROOT / "cebra_exploration" / "results"
_FULL_ROOT = _RESULTS_ROOT / "track_b_v2_rt_15fold_source_authority_20260814_dev"
_SMOKE_ROOT_PREFIX = "track_b_v2_rt_15fold_source_authority_20260814_smoke_"
_RUNNER_SCRIPT = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_rt_source_authority.py"
_BUNDLE_MEMBER_NAMES = (
    "source_roster",
    "source_coverage",
    "source_neural_input_authority",
    "source_behavior_auxiliary_scaler_authority",
    "source_readout_embedding_identity_authority",
    "source_only_dual_geometry_selection_plan",
)
_BUNDLE_FILENAMES = {
    "source_roster": "source_roster.json",
    "source_coverage": "source_coverage.json",
    "source_neural_input_authority": "source_neural_input_authority.json",
    "source_behavior_auxiliary_scaler_authority": "source_behavior_auxiliary_scaler_authority.json",
    "source_readout_embedding_identity_authority": "source_readout_embedding_identity_authority.json",
    "source_only_dual_geometry_selection_plan": "source_only_dual_geometry_selection_plan.json",
}
_SMOKE_FOLD_ID = "rt_outer_fold_00"


class TrackBV2RTSourceAuthorityRunnerError(live.TrackBV2LiveContractError):
    """Raised when an RT source-only child could violate its isolation contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2RTSourceAuthorityRunnerError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_root(*, run_kind: str, outer_fold_id: str | None = None) -> Path:
    require(run_kind in {"smoke", "full"}, "RT source authority run kind invalid")
    if run_kind == "full":
        require(outer_fold_id is None, "full RT source authority root may not select one fold")
        return _FULL_ROOT
    require(isinstance(outer_fold_id, str) and outer_fold_id, "smoke RT source authority needs an outer fold ID")
    return _RESULTS_ROOT / f"{_SMOKE_ROOT_PREFIX}{outer_fold_id}_dev"


def _assert_fresh_real_directory(path: Path) -> None:
    path = Path(path)
    require(path.is_absolute(), "RT source authority output root must be an absolute canonical path")
    require(not os.path.lexists(path), "RT source authority output root already exists or is symlinked")
    parent = path.parent
    require(parent.is_dir() and not parent.is_symlink(), "RT source authority results parent is missing or symlinked")
    require(parent.resolve() == parent, "RT source authority results parent resolves through an alias")


def _reserve_root(path: Path) -> None:
    _assert_fresh_real_directory(path)
    Path(path).mkdir(mode=0o755)
    info = Path(path).lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "RT source authority output root creation topology drift")


def _rt_pointer_pair() -> live.ExplicitSealedReceiptPair:
    body = metric_pointer.canonical_metric_pointer_body_path("rt", None)
    return live.ExplicitSealedReceiptPair(
        role="canonical_reference_body_pointer",
        body_path=body,
        sidecar_path=body.with_name(f"{body.name}.sha256"),
    )


def _validate_rt_metric_pointer() -> dict[str, Any]:
    """Require root's sealed RT pointer before any source path is derived."""
    validation = metric_pointer.validate_root_audited_metric_pointer_pair(
        dataset="rt", view=None, pointer_pair=_rt_pointer_pair(),
    )
    require(validation.get("pointer_body_sha256") == "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6",
            "RT source authority requires the sealed canonical RT metric pointer SHA d2e53165…")
    return validation


def _validated_plan_and_fold(outer_fold_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = source.build_rt_15fold_source_authority_plan()
    source.validate_rt_15fold_source_authority_plan(plan)
    require(isinstance(outer_fold_id, str) and outer_fold_id, "RT outer fold ID invalid")
    matches = [item for item in plan["outer_folds"] if item["outer_fold_id"] == outer_fold_id]
    require(len(matches) == 1, "RT outer fold missing or ambiguous in 15-fold source plan")
    fold = dict(matches[0])
    ids = tuple(fold["source_session_ids"])
    held = fold["opaque_held_out_target_session_id"]
    require(len(ids) == 14 and held not in ids and len(set(ids)) == 14,
            "RT outer fold must contain exactly 14 unique non-held source sessions")
    return plan, fold


def _same_fd_pair_payload(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    return live._strict_readonly_pair(live.ExplicitSealedReceiptPair(
        role=role, body_path=path, sidecar_path=path.with_name(f"{path.name}.sha256"),
    ))


def _cpu_runtime() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "worker_pid": os.getpid(),
        "worker_parent_pid": os.getppid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_required_empty": True,
        "cebra_imported": "cebra" in sys.modules,
        "gpu_used": False,
        "runner_source_sha256": _sha_file(Path(__file__)),
        "source_adapter_source_sha256": _sha_file(REPO_ROOT / "cebra_exploration" / "src" / "track_b_v2_source_adapter.py"),
        "loader_contract_source_sha256": _sha_file(REPO_ROOT / "cebra_exploration" / "src" / "track_b_v2_live_contract.py"),
    }


def _materialize_exact_fold_sources(
    *, fold: Mapping[str, Any]
) -> tuple[dict[str, Any], Sequence[source.SourceSessionMaterialization], dict[str, Any]]:
    """Open exactly the 14 planned source paths and record every derivation call."""
    planned_ids = tuple(fold["source_session_ids"])
    held = fold["opaque_held_out_target_session_id"]
    seen: list[str] = []
    original = source._canonical_source_path

    def guarded_path(dataset: str, source_session_id: str) -> Path:
        require(dataset == "rt", "RT source worker attempted a non-RT source path")
        require(source_session_id in planned_ids, "RT source worker attempted a held or unplanned session path")
        require(source_session_id != held, "RT source worker attempted to derive its held target path")
        seen.append(source_session_id)
        return original(dataset, source_session_id)

    source._canonical_source_path = guarded_path
    try:
        request, sessions = source.materialize_canonical_source_sessions(
            dataset="rt", view=None, source_session_ids=planned_ids,
        )
    finally:
        source._canonical_source_path = original
    require(tuple(seen) == planned_ids, "RT source worker did not derive exactly its ordered 14 source paths")
    require(tuple(row.session_id for row in sessions) == planned_ids,
            "RT source worker materialized a session outside its planned source roster")
    require(held not in {row.session_id for row in sessions}, "RT source worker materialized its held target")
    return request, sessions, {
        "source_path_derivation_call_count": len(seen),
        "source_path_derivation_session_ids": list(seen),
        "source_path_derivation_exact_planned_14": True,
        "held_out_target_path_derivation_count": 0,
        "held_out_target_data_opened": False,
        "target_data_opened": False,
        "target_query_opened": False,
    }


def _source_cost(sessions: Sequence[source.SourceSessionMaterialization], *, elapsed_seconds: float) -> dict[str, Any]:
    total_rows = sum(int(row.neural.shape[0]) for row in sessions)
    total_neural_scalars = sum(int(row.neural.size) for row in sessions)
    total_behavior_scalars = sum(int(row.dense_behavior.size) for row in sessions)
    return {
        "source_session_count": len(sessions),
        "source_neural_row_count": total_rows,
        "source_neural_scalar_count": total_neural_scalars,
        "source_dense_behavior_scalar_count": total_behavior_scalars,
        "cebra_parameter_count": 0,
        "cebra_training_mac_count": 0,
        "cebra_state_bytes": 0,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "target_score_emitted": False,
        "elapsed_seconds": elapsed_seconds,
        "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _fold_dir(root: Path, fold: Mapping[str, Any]) -> Path:
    return root / "folds" / str(fold["outer_fold_id"])


def run_rt_source_authority_worker(*, run_kind: str, output_root: Path, outer_fold_id: str) -> dict[str, Any]:
    """Child-only one-fold executor; it has no target argument or discovery path."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "RT source worker requires CUDA_VISIBLE_DEVICES='' ")
    root = _canonical_root(run_kind=run_kind, outer_fold_id=outer_fold_id if run_kind == "smoke" else None)
    require(Path(output_root) == root, "RT source worker output root is not canonical for its run kind/fold")
    require(root.is_dir() and not root.is_symlink(), "RT source worker root was not exclusively reserved by parent")
    pointer_validation = _validate_rt_metric_pointer()
    plan, fold = _validated_plan_and_fold(outer_fold_id)
    directory = _fold_dir(root, fold)
    require(not os.path.lexists(directory), "RT source worker fold output directory already exists")
    folds_root = directory.parent
    if not os.path.lexists(folds_root):
        folds_root.mkdir(mode=0o755)
    info = folds_root.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "RT source worker folds parent is missing or symlinked")

    started = time.monotonic()
    request, sessions, isolation = _materialize_exact_fold_sources(fold=fold)
    materialization_elapsed = time.monotonic() - started
    bundle = source.build_rt_outer_fold_source_authority_bundle(
        rt_15fold_plan=plan, outer_fold_id=outer_fold_id, request=request, sessions=sessions,
    )
    receipt_map = source.write_development_source_only_authority_bundle(output_dir=directory, bundle=bundle)
    require(set(receipt_map) == set(_BUNDLE_MEMBER_NAMES), "RT source worker bundle member set drift")
    for name in _BUNDLE_MEMBER_NAMES:
        path = directory / _BUNDLE_FILENAMES[name]
        payload, body_sha = _same_fd_pair_payload(path, role=f"rt_source_authority_{name}")
        require(body_sha == receipt_map[name]["body_sha256"], "RT source worker receipt body SHA drift")
        require(payload.get("dataset") == "rt" and payload.get("view") is None,
                "RT source worker receipt scope drift")
        require(payload.get("target_data_opened") is False and payload.get("target_query_opened") is False,
                "RT source worker receipt claims target access")
    roster = bundle["source_roster"]
    lineage = roster.get("outer_fold_lineage")
    require(isinstance(lineage, Mapping), "RT source worker roster lacks opaque held-out lineage")
    require(lineage.get("opaque_held_out_target_session_id") == fold["opaque_held_out_target_session_id"],
            "RT source worker roster held target lineage drift")

    runtime = _cpu_runtime()
    require(runtime["cebra_imported"] is False, "RT source worker unexpectedly imported CEBRA")
    costs = _source_cost(sessions, elapsed_seconds=materialization_elapsed)
    receipt_payload = {
        "schema": RT_SOURCE_FOLD_WORKER_SCHEMA,
        "status": "RT_SOURCE_ONLY_FOLD_MATERIALIZED__NO_TARGET_NO_CEBRA_NO_SCORE",
        "run_kind": run_kind,
        "dataset": "rt",
        "view": None,
        "outer_fold_id": fold["outer_fold_id"],
        "outer_fold_index": fold["outer_fold_index"],
        "opaque_held_out_target_session_id": fold["opaque_held_out_target_session_id"],
        "rt_metric_pointer": {
            "body_path": str(metric_pointer.canonical_metric_pointer_body_path("rt", None)),
            "body_sha256": pointer_validation["pointer_body_sha256"],
            "validation_schema": pointer_validation["schema"],
        },
        "rt_15fold_source_authority_plan_sha256": plan["rt_15fold_source_authority_plan_sha256"],
        "loader_semantics_sha256": request["loader_semantics_sha256"],
        "source_session_ids": list(fold["source_session_ids"]),
        "source_nwb_sha256_by_session": {row.session_id: row.source_nwb_sha256 for row in sessions},
        "source_authority_receipt_sha256_by_member": {
            name: receipt_map[name]["body_sha256"] for name in _BUNDLE_MEMBER_NAMES
        },
        "source_authority_receipt_sidecar_sha256_by_member": {
            name: receipt_map[name]["sidecar_sha256"] for name in _BUNDLE_MEMBER_NAMES
        },
        "process_isolation": {
            "independent_child_process_required": True,
            "worker_pid": runtime["worker_pid"],
            **isolation,
        },
        "runtime_code_and_loader": runtime,
        "cost": costs,
        "target_data_discovered": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
    fold_receipt = base.write_immutable_receipt(directory / "rt_source_only_fold_execution_receipt.json", receipt_payload)
    return receipt_payload | {
        "fold_execution_receipt_body_path": fold_receipt["body_path"],
        "fold_execution_receipt_body_sha256": fold_receipt["body_sha256"],
        "fold_execution_receipt_sidecar_path": fold_receipt["sidecar_path"],
    }


def _worker_command(*, run_kind: str, root: Path, outer_fold_id: str) -> list[str]:
    return [
        sys.executable,
        str(_RUNNER_SCRIPT),
        "--_worker-run-kind", run_kind,
        "--_worker-output-root", str(root),
        "--_worker-fold-id", outer_fold_id,
    ]


def _run_one_child(*, run_kind: str, root: Path, outer_fold_id: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    src = str(REPO_ROOT / "cebra_exploration" / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}{os.pathsep}{env['PYTHONPATH']}"
    result = subprocess.run(
        _worker_command(run_kind=run_kind, root=root, outer_fold_id=outer_fold_id),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0,
            f"RT source worker failed for {outer_fold_id}: {result.stderr[-4000:]}")
    # Child stdout is intentionally non-authoritative: imported canonical
    # loaders may emit informational output.  The parent consumes only the
    # child-created immutable pair at its deterministic fold path.
    receipt_path = root / "folds" / outer_fold_id / "rt_source_only_fold_execution_receipt.json"
    require(receipt_path.is_file(), "RT source worker lacks its deterministic fold execution receipt")
    payload, sha = _same_fd_pair_payload(receipt_path, role="rt_source_only_fold_execution")
    require(payload.get("outer_fold_id") == outer_fold_id, "RT source worker receipt fold drift")
    return dict(payload) | {
        "fold_execution_receipt_body_path": str(receipt_path),
        "fold_execution_receipt_body_sha256": sha,
        "fold_execution_receipt_sidecar_path": str(receipt_path.with_name(f"{receipt_path.name}.sha256")),
    }


def _write_root_summary(*, run_kind: str, root: Path, plan: Mapping[str, Any], children: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    require(children, "RT source root summary needs at least one child")
    expected_count = 1 if run_kind == "smoke" else 15
    require(len(children) == expected_count, "RT source root child count/run-kind drift")
    ordered = sorted(children, key=lambda item: int(item["outer_fold_index"]))
    manifest_payload = {
        "schema": RT_SOURCE_SMOKE_MANIFEST_SCHEMA if run_kind == "smoke" else RT_SOURCE_FULL_MANIFEST_SCHEMA,
        "status": "RT_SOURCE_ONLY_MANIFEST__NO_TARGET_NO_CEBRA_NO_SCORE",
        "run_kind": run_kind,
        "dataset": "rt",
        "view": None,
        "rt_metric_pointer_body_sha256": "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6",
        "rt_15fold_source_authority_plan_sha256": plan["rt_15fold_source_authority_plan_sha256"],
        "outer_fold_count": len(ordered),
        "fold_execution_receipts": [
            {
                "outer_fold_id": item["outer_fold_id"],
                "opaque_held_out_target_session_id": item["opaque_held_out_target_session_id"],
                "body_path": item["fold_execution_receipt_body_path"],
                "body_sha256": item["fold_execution_receipt_body_sha256"],
                "source_authority_receipt_sha256_by_member": item["source_authority_receipt_sha256_by_member"],
                "target_data_opened": item["target_data_opened"],
                "cebra_imported": item["cebra_imported"],
                "gpu_used": item["gpu_used"],
                "score_emitted": item["score_emitted"],
            }
            for item in ordered
        ],
        "target_data_discovered": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
    manifest_name = "rt_source_only_smoke_manifest.json" if run_kind == "smoke" else "rt_15fold_source_authority_manifest.json"
    manifest = base.write_immutable_receipt(root / manifest_name, manifest_payload)
    costs = [item["cost"] for item in ordered]
    aggregate_payload = {
        "schema": RT_SOURCE_SMOKE_AGGREGATE_SCHEMA if run_kind == "smoke" else RT_SOURCE_FULL_AGGREGATE_SCHEMA,
        "status": "RT_SOURCE_ONLY_COST_AGGREGATE__NO_TARGET_NO_CEBRA_NO_SCORE",
        "run_kind": run_kind,
        "dataset": "rt",
        "view": None,
        "manifest_body_sha256": manifest["body_sha256"],
        "outer_fold_count": len(costs),
        "sum_source_neural_row_count": sum(item["source_neural_row_count"] for item in costs),
        "sum_source_neural_scalar_count": sum(item["source_neural_scalar_count"] for item in costs),
        "sum_source_dense_behavior_scalar_count": sum(item["source_dense_behavior_scalar_count"] for item in costs),
        "sum_elapsed_seconds": sum(item["elapsed_seconds"] for item in costs),
        "max_process_peak_rss_kib": max(item["process_peak_rss_kib"] for item in costs),
        "cebra_parameter_count": 0,
        "cebra_training_mac_count": 0,
        "cebra_state_bytes": 0,
        "target_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
    aggregate_name = "rt_source_only_smoke_aggregate.json" if run_kind == "smoke" else "rt_15fold_source_authority_aggregate.json"
    aggregate = base.write_immutable_receipt(root / aggregate_name, aggregate_payload)
    return {
        "manifest_body_path": manifest["body_path"],
        "manifest_body_sha256": manifest["body_sha256"],
        "aggregate_body_path": aggregate["body_path"],
        "aggregate_body_sha256": aggregate["body_sha256"],
    }


def run_rt_source_authority_smoke(*, outer_fold_id: str = _SMOKE_FOLD_ID) -> dict[str, Any]:
    """Launch exactly one fresh RT-source worker; no target or CEBRA work."""
    require(outer_fold_id == _SMOKE_FOLD_ID,
            "the first real RT source-only smoke is predeclared as rt_outer_fold_00 only")
    pointer_validation = _validate_rt_metric_pointer()
    plan, fold = _validated_plan_and_fold(outer_fold_id)
    root = _canonical_root(run_kind="smoke", outer_fold_id=outer_fold_id)
    _reserve_root(root)
    child = _run_one_child(run_kind="smoke", root=root, outer_fold_id=outer_fold_id)
    summary = _write_root_summary(run_kind="smoke", root=root, plan=plan, children=(child,))
    return {
        "status": "RT_SOURCE_ONLY_SMOKE_COMPLETE__NO_TARGET_NO_CEBRA_NO_SCORE",
        "output_root": str(root),
        "outer_fold_id": fold["outer_fold_id"],
        "rt_metric_pointer_body_sha256": pointer_validation["pointer_body_sha256"],
        "rt_15fold_source_authority_plan_sha256": plan["rt_15fold_source_authority_plan_sha256"],
        "fold_execution_receipt_body_sha256": child["fold_execution_receipt_body_sha256"],
        **summary,
        "target_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }


def run_full_rt_15fold_source_authority(*, root_authorized_full_run: bool = False) -> dict[str, Any]:
    """Reserved root-only continuation; never called automatically by this route."""
    require(root_authorized_full_run is True,
            "refusing a 15-fold RT source authority launch without explicit root_authorized_full_run=True")
    pointer_validation = _validate_rt_metric_pointer()
    plan = source.build_rt_15fold_source_authority_plan()
    source.validate_rt_15fold_source_authority_plan(plan)
    root = _canonical_root(run_kind="full")
    _reserve_root(root)
    children = tuple(
        _run_one_child(run_kind="full", root=root, outer_fold_id=item["outer_fold_id"])
        for item in plan["outer_folds"]
    )
    summary = _write_root_summary(run_kind="full", root=root, plan=plan, children=children)
    return {
        "status": "RT_15FOLD_SOURCE_ONLY_AUTHORITY_COMPLETE__NO_TARGET_NO_CEBRA_NO_SCORE",
        "output_root": str(root),
        "rt_metric_pointer_body_sha256": pointer_validation["pointer_body_sha256"],
        "rt_15fold_source_authority_plan_sha256": plan["rt_15fold_source_authority_plan_sha256"],
        **summary,
        "target_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
