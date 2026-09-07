"""No-data authority bridge for the canonical 15 RT local assets.

The completed RT source-only graph already proves a stable local path and NWB
SHA for every one of the 15 sessions: each asset occurs as a source in exactly
14 sealed outer-fold coverage receipts.  It does *not* record file byte sizes,
so it is not by itself a complete local-asset authority.

This module never discovers files.  A future root publisher may explicitly
provide exactly 15 ``(session_id, canonical_path, byte_size, sha256)`` rows.
Paths and SHAs must match the sealed 14-fold consensus; byte sizes remain an
explicit root attestation until a later consumer opens, hashes, and parses the
same file descriptor.  Importing, auditing, planning, and publishing do not
open, stat, glob, or rehash an NWB/NPZ.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_live_contract as live
import track_b_v2_metric_pointer_authority as metric_pointer
import track_b_v2_source_adapter as source


SCHEMA = "track_b_v2_rt_15session_local_asset_authority_v1"
STATUS = "ROOT_PUBLISHED_RT_LOCAL_ASSET_IDENTITY__BEFORE_TARGET_ACCESS"
AUDIT_SCHEMA = "track_b_v2_rt_existing_asset_identity_graph_audit_v1"
AUDIT_STATUS = "PATH_AND_SHA_COMPLETE__BYTE_SIZE_MISSING__ROOT_INPUT_REQUIRED"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_rt_15fold_source_authority_20260814_dev"
)
MANIFEST = SOURCE_ROOT / "rt_15fold_source_authority_manifest.json"
AGGREGATE = SOURCE_ROOT / "rt_15fold_source_authority_aggregate.json"
POINTER = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_root_metric_pointer_authority_v2/rt_metric_pointer.json"
)
CANONICAL_OUTPUT = (
    REPO_ROOT / "cebra_exploration/results"
    / "track_b_v2_rt_local_asset_authority_v1/rt_15session_local_asset_authority.json"
)
CANONICAL_PUBLISHER = (
    REPO_ROOT / "cebra_exploration/scripts/publish_track_b_v2_rt_local_asset_authority.py"
)
FOCUSED_TEST = REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_rt_local_asset_authority.py"
POINTER_SHA256 = "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6"
MANIFEST_SHA256 = "775212fbd800129eb32ca03a68e53089a0ac97746be748491a840f8a694d0e02"
AGGREGATE_SHA256 = "66dd4a42567de87d0113f1f3c312208308e150f1643e1f5ee21776033f508b35"


class TrackBV2RTLocalAssetAuthorityError(live.TrackBV2LiveContractError):
    """Fail-closed error raised before an RT target asset may be resolved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2RTLocalAssetAuthorityError(message)


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without resolving a possible symlink."""
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path(os.path.abspath(str(expanded)))


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev, info.st_ino, info.st_size, info.st_mode,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _assert_real_directory_chain(directory: Path, *, create: bool) -> Path:
    """Reject symlinked parents, creating only individually verified directories."""
    lexical = _absolute_lexical(directory)
    require(lexical.is_absolute(), "RT authority output parent must be absolute")
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            require(create, f"RT authority output parent is absent: {current}")
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            info = current.lstat()
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"RT authority output parent must be a real directory: {current}")
    return lexical


def _fsync_directory(directory: Path) -> None:
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required by RT authority")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_identity_matches(path: Path, expected: tuple[int, int, int, int, int, int], label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise TrackBV2RTLocalAssetAuthorityError(f"{label} pathname disappeared") from exc
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"{label} pathname is not a regular non-symlink")
    require(_identity(info) == expected, f"{label} pathname identity changed")


def _write_all_fd(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        require(written > 0, "short zero-byte write while publishing RT authority")
        offset += written


def _write_exclusive_regular(
    path: Path, raw: bytes, *, mode: int,
) -> tuple[int, int, int, int, int, int]:
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW required by RT authority")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_all_fd(descriptor, raw)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == mode,
            "new RT authority output lost its requested mode")
    return _identity(info)


def _rollback_owned_output(
    path: Path, expected: tuple[int, int, int, int, int, int], label: str,
) -> None:
    """Remove only the exact inode created by this transaction."""
    try:
        _path_identity_matches(path, expected, label)
    except TrackBV2RTLocalAssetAuthorityError as exc:
        raise TrackBV2RTLocalAssetAuthorityError(
            f"{label} rollback unsafe; refusing to unlink a replacement"
        ) from exc
    os.unlink(path)


def _write_immutable_pair(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    """Publish a durable O_EXCL/0444 pair with owned-inode rollback."""
    body = _absolute_lexical(path)
    _assert_real_directory_chain(body.parent, create=True)
    sidecar = body.with_name(f"{body.name}.sha256")
    require(not os.path.lexists(body) and not os.path.lexists(sidecar),
            "canonical RT local asset authority output pair must be fresh")
    raw = base.canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    sidecar_raw = f"{digest}  {body.name}\n".encode("ascii")
    created: list[tuple[Path, tuple[int, int, int, int, int, int], str]] = []
    try:
        created.append((body, _write_exclusive_regular(body, raw, mode=0o444), "RT authority body"))
        _fsync_directory(body.parent)
        created.append((sidecar, _write_exclusive_regular(
            sidecar, sidecar_raw, mode=0o444), "RT authority sidecar"))
        _fsync_directory(body.parent)
        verified, verified_sha = _read_pair(body, "new_canonical_rt_local_asset_authority")
        require(verified_sha == digest and dict(verified) == dict(payload),
                "new RT authority pair verification drift")
    except BaseException as exc:
        rollback_errors: list[str] = []
        for created_path, created_identity, label in reversed(created):
            try:
                _rollback_owned_output(created_path, created_identity, label)
            except (OSError, TrackBV2RTLocalAssetAuthorityError) as rollback_exc:
                rollback_errors.append(str(rollback_exc))
        try:
            _fsync_directory(body.parent)
        except OSError as rollback_exc:
            rollback_errors.append(f"directory fsync after rollback: {rollback_exc}")
        if rollback_errors:
            raise TrackBV2RTLocalAssetAuthorityError(
                "RT authority transaction failed and rollback was incomplete: " + "; ".join(rollback_errors)
            ) from exc
        if isinstance(exc, TrackBV2RTLocalAssetAuthorityError):
            raise
        raise TrackBV2RTLocalAssetAuthorityError(
            "RT authority transaction failed; owned outputs were rolled back"
        ) from exc
    return {
        "body_path": str(body),
        "body_sha256": digest,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": hashlib.sha256(sidecar_raw).hexdigest(),
    }


def _pair(path: Path, role: str) -> live.ExplicitSealedReceiptPair:
    body = Path(path)
    return live.ExplicitSealedReceiptPair(
        role=role,
        body_path=body,
        sidecar_path=body.with_name(f"{body.name}.sha256"),
    )


def _read_pair(path: Path, role: str) -> tuple[dict[str, Any], str]:
    try:
        return live._strict_readonly_pair(_pair(path, role))
    except live.TrackBV2LiveContractError as exc:
        raise TrackBV2RTLocalAssetAuthorityError(str(exc)) from exc


def _no_execution(payload: Mapping[str, Any], label: str) -> None:
    for key in (
        "target_data_discovered", "target_data_opened", "target_query_opened",
        "formal_data_opened", "cebra_imported", "cebra_solver_called",
        "gpu_used", "score_emitted", "official_execution_receipt_minted",
    ):
        if key in payload:
            require(payload[key] is False, f"{label}: prohibited flag {key} is not false")


def _validate_pointer() -> dict[str, Any]:
    try:
        validation = metric_pointer.validate_root_audited_metric_pointer_pair(
            dataset="rt", view=None, pointer_pair=_pair(POINTER, "canonical_reference_body_pointer"),
        )
    except metric_pointer.TrackBV2MetricPointerAuthorityError as exc:
        raise TrackBV2RTLocalAssetAuthorityError(str(exc)) from exc
    require(validation.get("pointer_body_sha256") == POINTER_SHA256,
            "canonical RT metric pointer SHA drift")
    return validation


def audit_existing_asset_identity_graph() -> dict[str, Any]:
    """Read only named sealed receipts and prove the 15 x 14 identity graph.

    No directory enumeration or data-file operation occurs.  Fold receipt and
    coverage paths are derived solely from the sealed full15 topology.
    """
    pointer_validation = _validate_pointer()
    manifest, manifest_sha = _read_pair(MANIFEST, "rt_full15_source_manifest")
    aggregate, aggregate_sha = _read_pair(AGGREGATE, "rt_full15_source_aggregate")
    require(manifest_sha == MANIFEST_SHA256, "RT full15 manifest SHA drift")
    require(aggregate_sha == AGGREGATE_SHA256, "RT full15 aggregate SHA drift")
    require(manifest.get("schema") == "track_b_v2_rt_15fold_source_authority_manifest_v1",
            "RT full15 manifest schema drift")
    require(manifest.get("status") == "RT_SOURCE_ONLY_MANIFEST__NO_TARGET_NO_CEBRA_NO_SCORE",
            "RT full15 manifest status drift")
    require((manifest.get("run_kind"), manifest.get("dataset"), manifest.get("view"),
             manifest.get("outer_fold_count")) == ("full", "rt", None, 15),
            "RT full15 manifest scope/count drift")
    require(manifest.get("rt_metric_pointer_body_sha256") == POINTER_SHA256,
            "RT full15 manifest pointer binding drift")
    require(aggregate.get("schema") == "track_b_v2_rt_15fold_source_authority_aggregate_v1",
            "RT full15 aggregate schema drift")
    require(aggregate.get("status") == "RT_SOURCE_ONLY_COST_AGGREGATE__NO_TARGET_NO_CEBRA_NO_SCORE",
            "RT full15 aggregate status drift")
    require(aggregate.get("manifest_body_sha256") == MANIFEST_SHA256,
            "RT aggregate/full15 manifest binding drift")
    require((aggregate.get("run_kind"), aggregate.get("dataset"), aggregate.get("view"),
             aggregate.get("outer_fold_count")) == ("full", "rt", None, 15),
            "RT full15 aggregate scope/count drift")
    _no_execution(manifest, "RT full15 manifest")
    _no_execution(aggregate, "RT full15 aggregate")

    plan = source.build_rt_15fold_source_authority_plan()
    source.validate_rt_15fold_source_authority_plan(plan)
    plan_sha = plan["rt_15fold_source_authority_plan_sha256"]
    require(manifest.get("rt_15fold_source_authority_plan_sha256") == plan_sha,
            "RT full15 topology plan binding drift")
    expected_folds = plan.get("outer_folds")
    manifest_folds = manifest.get("fold_execution_receipts")
    require(isinstance(expected_folds, list) and len(expected_folds) == 15,
            "RT full15 plan fold count drift")
    require(isinstance(manifest_folds, list) and len(manifest_folds) == 15,
            "RT full15 manifest fold count drift")
    require([row.get("outer_fold_id") for row in manifest_folds] ==
            [row.get("outer_fold_id") for row in expected_folds],
            "RT full15 manifest fold order drift")

    observations: dict[str, list[dict[str, Any]]] = {}
    fold_bindings: list[dict[str, Any]] = []
    held_order: list[str] = []
    for expected_fold, manifest_entry in zip(expected_folds, manifest_folds, strict=True):
        require(isinstance(expected_fold, Mapping) and isinstance(manifest_entry, Mapping),
                "RT fold topology row malformed")
        fold_id = expected_fold.get("outer_fold_id")
        fold_index = expected_fold.get("outer_fold_index")
        held = expected_fold.get("opaque_held_out_target_session_id")
        source_ids = expected_fold.get("source_session_ids")
        require(isinstance(fold_id, str) and isinstance(fold_index, int) and isinstance(held, str),
                "RT expected fold identity malformed")
        require(isinstance(source_ids, list) and len(source_ids) == 14 and held not in source_ids,
                "RT expected fold source/held topology drift")
        expected_receipt = SOURCE_ROOT / "folds" / fold_id / "rt_source_only_fold_execution_receipt.json"
        require(manifest_entry.get("body_path") == str(expected_receipt),
                "RT fold execution receipt path is not canonical")
        fold_payload, fold_sha = _read_pair(expected_receipt, f"RT fold execution {fold_id}")
        require(manifest_entry.get("body_sha256") == fold_sha,
                "RT manifest/fold execution body SHA drift")
        require(fold_payload.get("schema") == "track_b_v2_rt_source_only_fold_worker_receipt_v1",
                "RT fold execution schema drift")
        require(fold_payload.get("status") == "RT_SOURCE_ONLY_FOLD_MATERIALIZED__NO_TARGET_NO_CEBRA_NO_SCORE",
                "RT fold execution status drift")
        require((fold_payload.get("outer_fold_id"), fold_payload.get("outer_fold_index"),
                 fold_payload.get("opaque_held_out_target_session_id")) ==
                (fold_id, fold_index, held), "RT fold held-target identity drift")
        require(fold_payload.get("source_session_ids") == source_ids,
                "RT fold ordered 14-source roster drift")
        require(fold_payload.get("rt_15fold_source_authority_plan_sha256") == plan_sha,
                "RT fold topology plan binding drift")
        fold_pointer = fold_payload.get("rt_metric_pointer")
        require(isinstance(fold_pointer, Mapping) and
                fold_pointer.get("body_sha256") == POINTER_SHA256,
                "RT fold pointer binding drift")
        _no_execution(fold_payload, f"RT fold execution {fold_id}")

        coverage_path = expected_receipt.parent / "source_coverage.json"
        coverage, coverage_sha = _read_pair(coverage_path, f"RT fold coverage {fold_id}")
        claimed_members = fold_payload.get("source_authority_receipt_sha256_by_member")
        require(isinstance(claimed_members, Mapping) and
                claimed_members.get("source_coverage") == coverage_sha,
                "RT fold execution/coverage SHA binding drift")
        require(coverage.get("schema") == "track_b_v2_source_coverage_receipt_v1" and
                coverage.get("status") ==
                "DEVELOPMENT_SOURCE_ONLY__FULL_SOURCE_RAW_COVERAGE__NOT_CEBRA_EXECUTION",
                "RT fold coverage schema/status drift")
        require((coverage.get("dataset"), coverage.get("view")) == ("rt", None),
                "RT fold coverage scope drift")
        _no_execution(coverage, f"RT fold coverage {fold_id}")
        rows = coverage.get("source_sessions")
        require(isinstance(rows, list) and len(rows) == 14,
                "RT fold coverage needs exactly 14 source rows")
        require([row.get("session_id") for row in rows if isinstance(row, Mapping)] == source_ids,
                "RT fold coverage ordered source roster drift")
        for row in rows:
            require(isinstance(row, Mapping), "RT fold coverage asset row malformed")
            session_id = row.get("session_id")
            path = row.get("source_path")
            sha = row.get("source_nwb_sha256")
            require(isinstance(session_id, str) and isinstance(path, str) and path.startswith("/"),
                    "RT fold coverage session/path malformed")
            require(_valid_sha(sha), "RT fold coverage NWB SHA malformed")
            byte_fields = tuple(sorted(
                key for key in row if "byte" in str(key).lower() or "size" in str(key).lower()
            ))
            observations.setdefault(session_id, []).append({
                "outer_fold_id": fold_id,
                "canonical_path": path,
                "sha256": sha,
                "byte_size_fields_present": list(byte_fields),
            })
        held_order.append(held)
        fold_bindings.append({
            "outer_fold_id": fold_id,
            "outer_fold_index": fold_index,
            "held_target_session_id": held,
            "ordered_source_session_ids": list(source_ids),
            "fold_execution_receipt_path": str(expected_receipt),
            "fold_execution_receipt_sha256": fold_sha,
            "source_coverage_receipt_path": str(coverage_path),
            "source_coverage_receipt_sha256": coverage_sha,
        })

    require(len(observations) == 15 and len(set(held_order)) == 15,
            "RT full15 graph does not cover 15 unique assets/held targets")
    require(set(observations) == set(held_order),
            "RT asset identity set differs from held-target fold set")
    asset_identities: list[dict[str, Any]] = []
    for session_id in held_order:
        seen = observations[session_id]
        require(len(seen) == 14, "each RT asset must occur in exactly 14 source coverage receipts")
        variants = {(row["canonical_path"], row["sha256"]) for row in seen}
        require(len(variants) == 1, "RT path/SHA identity differs across source folds")
        require(all(not row["byte_size_fields_present"] for row in seen),
                "unexpected byte-size field appeared; successor audit required")
        path, sha = next(iter(variants))
        held_binding = next(row for row in fold_bindings if row["held_target_session_id"] == session_id)
        asset_identities.append({
            "session_id": session_id,
            "canonical_path": path,
            "sha256": sha,
            "byte_size": None,
            "identity_source_fold_occurrence_count": 14,
            "identity_variant_count": 1,
            "held_target_outer_fold_id": held_binding["outer_fold_id"],
            "held_target_outer_fold_index": held_binding["outer_fold_index"],
        })

    graph = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "dataset": "rt",
        "view": None,
        "metric_pointer": {
            "body_path": str(POINTER), "body_sha256": POINTER_SHA256,
            "validation_schema": pointer_validation.get("schema"),
        },
        "full15_source_manifest": {
            "body_path": str(MANIFEST), "body_sha256": manifest_sha,
            "schema": manifest["schema"],
        },
        "full15_source_aggregate": {
            "body_path": str(AGGREGATE), "body_sha256": aggregate_sha,
            "schema": aggregate["schema"],
        },
        "rt_15fold_source_authority_plan_sha256": plan_sha,
        "fold_bindings": fold_bindings,
        "asset_identities": asset_identities,
        "asset_count": 15,
        "fold_count": 15,
        "source_identity_occurrences_per_asset": 14,
        "path_and_sha_authority_complete": True,
        "byte_size_authority_complete": False,
        "complete_path_byte_size_sha_authority_exists": False,
        "missing_authority_field": "byte_size",
        "directory_enumeration_used": False,
        "data_file_opened": False,
        "data_file_statted": False,
        "data_file_rehashed": False,
        "target_or_formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
    }
    return graph | {"identity_graph_sha256": _sha_json(graph)}


def _validate_graph(graph: Mapping[str, Any]) -> None:
    require(graph.get("schema") == AUDIT_SCHEMA and graph.get("status") == AUDIT_STATUS,
            "RT existing identity graph schema/status drift")
    require(graph.get("asset_count") == graph.get("fold_count") == 15,
            "RT existing identity graph count drift")
    require(graph.get("source_identity_occurrences_per_asset") == 14,
            "RT existing identity occurrence count drift")
    require(graph.get("path_and_sha_authority_complete") is True and
            graph.get("byte_size_authority_complete") is False and
            graph.get("complete_path_byte_size_sha_authority_exists") is False,
            "RT existing identity completeness finding drift")
    require(graph.get("missing_authority_field") == "byte_size",
            "RT existing identity missing-field finding drift")
    require(graph.get("metric_pointer", {}).get("body_sha256") == POINTER_SHA256 and
            graph.get("full15_source_manifest", {}).get("body_sha256") == MANIFEST_SHA256 and
            graph.get("full15_source_aggregate", {}).get("body_sha256") == AGGREGATE_SHA256,
            "RT existing identity parent binding drift")
    bare = dict(graph)
    declared = bare.pop("identity_graph_sha256", None)
    require(declared == _sha_json(bare), "RT existing identity graph internal SHA drift")
    assets = graph.get("asset_identities")
    folds = graph.get("fold_bindings")
    require(isinstance(assets, list) and isinstance(folds, list) and len(assets) == len(folds) == 15,
            "RT existing identity rows/folds drift")
    require([row.get("session_id") for row in assets] ==
            [row.get("held_target_session_id") for row in folds],
            "RT asset order differs from exact held-target fold order")
    for index, (asset, fold) in enumerate(zip(assets, folds, strict=True)):
        require(asset.get("held_target_outer_fold_id") == fold.get("outer_fold_id") and
                asset.get("held_target_outer_fold_index") == index == fold.get("outer_fold_index"),
                "RT asset held-target fold mapping drift")
        require(asset.get("byte_size") is None and
                asset.get("identity_source_fold_occurrence_count") == 14 and
                asset.get("identity_variant_count") == 1,
                "RT asset source-consensus evidence drift")


def _implementation_closure(entrypoint: Path) -> dict[str, dict[str, Any]]:
    publisher = _absolute_lexical(entrypoint)
    require(publisher == CANONICAL_PUBLISHER,
            "RT local asset authority must be built by the canonical publisher")
    paths = {
        "rt_local_asset_authority_core": Path(__file__).absolute(),
        "rt_local_asset_authority_publisher": publisher,
        "rt_local_asset_authority_focused_test": FOCUSED_TEST,
        "track_b_v2_contract": Path(base.__file__).absolute(),
        "track_b_v2_live_contract": Path(live.__file__).absolute(),
        "track_b_v2_metric_pointer_authority": Path(metric_pointer.__file__).absolute(),
        "track_b_v2_source_adapter": Path(source.__file__).absolute(),
    }
    closure: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        raw = source._read_regular_file(path, label=f"live implementation {label}")
        closure[label] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return closure


def build_unminted_authority(
    *, root_asset_rows: Sequence[Mapping[str, Any]], identity_graph: Mapping[str, Any],
    entrypoint: Path,
) -> dict[str, Any]:
    """Validate explicit root rows without resolving, statting, or opening them."""
    _validate_graph(identity_graph)
    require(dict(identity_graph) == audit_existing_asset_identity_graph(),
            "caller-supplied RT identity graph differs from live sealed receipt graph")
    expected = identity_graph["asset_identities"]
    rows = [dict(row) for row in root_asset_rows]
    require(len(rows) == 15, "root publisher must provide exactly 15 explicit RT asset rows")
    require([row.get("session_id") for row in rows] == [row["session_id"] for row in expected],
            "root asset rows must follow exact sealed held-target fold order")
    canonical_rows: list[dict[str, Any]] = []
    for root_row, source_row in zip(rows, expected, strict=True):
        require(set(root_row) == {"session_id", "canonical_path", "byte_size", "sha256"},
                "root asset row field set drift")
        require(root_row["session_id"] == source_row["session_id"], "root asset session drift")
        require(root_row["canonical_path"] == source_row["canonical_path"],
                "root asset path differs from sealed 14-fold source consensus")
        require(root_row["sha256"] == source_row["sha256"] and _valid_sha(root_row["sha256"]),
                "root asset SHA differs from sealed 14-fold source consensus")
        require(type(root_row["byte_size"]) is int and root_row["byte_size"] > 0,
                "root asset byte_size must be a positive exact integer")
        canonical_rows.append({
            **root_row,
            "held_target_outer_fold_id": source_row["held_target_outer_fold_id"],
            "held_target_outer_fold_index": source_row["held_target_outer_fold_index"],
            "path_sha_source_fold_occurrence_count": 14,
            "path_sha_identity_variant_count": 1,
            "byte_size_authority": "EXPLICIT_ROOT_INPUT__NOT_INFERRED_OR_REHASHED_BY_PUBLISHER",
        })
    closure = _implementation_closure(entrypoint)
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "dataset": "rt",
        "view": None,
        "asset_count": 15,
        "asset_rows": canonical_rows,
        "asset_row_order": "EXACT_SEALED_OUTER_FOLD_INDEX_0_TO_14_HELD_TARGET_ORDER",
        "parent_bindings": {
            "rt_metric_pointer_body_sha256": POINTER_SHA256,
            "rt_full15_source_manifest_body_sha256": MANIFEST_SHA256,
            "rt_full15_source_aggregate_body_sha256": AGGREGATE_SHA256,
            "rt_15fold_source_authority_plan_sha256":
                identity_graph["rt_15fold_source_authority_plan_sha256"],
            "existing_identity_graph_sha256": identity_graph["identity_graph_sha256"],
            "fold_bindings": identity_graph["fold_bindings"],
        },
        "root_input_contract": {
            "all_15_rows_explicit": True,
            "directory_glob_scan_or_available_tree_inference_used": False,
            "path_and_sha_exactly_reconciled_to_14_fold_source_consensus": True,
            "byte_size_is_explicit_root_attestation": True,
            "publisher_independently_verified_data_bytes": False,
        },
        "same_fd_consumer_required": True,
        "same_fd_consumer_semantics": (
            "load this immutable authority before path resolution; open O_NOFOLLOW once; fstat size; "
            "hash the opened bytes; rewind; parse from the same unchanged descriptor"
        ),
        "implementation_closure": closure,
        "implementation_closure_sha256": _sha_json(closure),
        "publisher_data_directory_enumeration_used": False,
        "publisher_data_path_resolved": False,
        "publisher_data_file_statted": False,
        "publisher_data_file_opened": False,
        "publisher_data_file_rehashed": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_scientific_receipt_minted": False,
    }
    return payload | {"authority_payload_sha256": _sha_json(payload)}


def _validate_published_payload(payload: Mapping[str, Any], identity_graph: Mapping[str, Any]) -> None:
    _validate_graph(identity_graph)
    require(payload.get("schema") == SCHEMA and payload.get("status") == STATUS,
            "published RT local asset authority schema/status drift")
    require((payload.get("dataset"), payload.get("view"), payload.get("asset_count")) == ("rt", None, 15),
            "published RT local asset authority scope/count drift")
    bare = dict(payload)
    declared = bare.pop("authority_payload_sha256", None)
    require(declared == _sha_json(bare), "published RT local asset internal SHA drift")
    require(set(payload) == {
        "schema", "status", "dataset", "view", "asset_count", "asset_rows", "asset_row_order",
        "parent_bindings", "root_input_contract", "same_fd_consumer_required",
        "same_fd_consumer_semantics", "implementation_closure", "implementation_closure_sha256",
        "publisher_data_directory_enumeration_used", "publisher_data_path_resolved",
        "publisher_data_file_statted", "publisher_data_file_opened", "publisher_data_file_rehashed",
        "target_data_opened", "target_query_opened", "formal_data_opened", "cebra_imported",
        "gpu_used", "score_emitted", "official_scientific_receipt_minted",
        "authority_payload_sha256",
    }, "published RT local asset top-level field set drift")
    parents = payload.get("parent_bindings")
    require(isinstance(parents, Mapping), "published RT local asset parent bindings missing")
    require(parents.get("rt_metric_pointer_body_sha256") == POINTER_SHA256 and
            parents.get("rt_full15_source_manifest_body_sha256") == MANIFEST_SHA256 and
            parents.get("rt_full15_source_aggregate_body_sha256") == AGGREGATE_SHA256 and
            parents.get("existing_identity_graph_sha256") == identity_graph["identity_graph_sha256"],
            "published RT local asset parent binding drift")
    require(parents.get("rt_15fold_source_authority_plan_sha256") ==
            identity_graph["rt_15fold_source_authority_plan_sha256"] and
            parents.get("fold_bindings") == identity_graph["fold_bindings"],
            "published RT local asset fold-graph binding drift")
    require(payload.get("asset_row_order") ==
            "EXACT_SEALED_OUTER_FOLD_INDEX_0_TO_14_HELD_TARGET_ORDER",
            "published RT local asset row-order contract drift")
    rows = payload.get("asset_rows")
    require(isinstance(rows, list) and len(rows) == 15,
            "published RT local asset rows missing")
    expected = identity_graph["asset_identities"]
    for row, source_row in zip(rows, expected, strict=True):
        require(set(row) == {
            "session_id", "canonical_path", "byte_size", "sha256",
            "held_target_outer_fold_id", "held_target_outer_fold_index",
            "path_sha_source_fold_occurrence_count", "path_sha_identity_variant_count",
            "byte_size_authority",
        }, "published RT local asset row field set drift")
        require(row.get("session_id") == source_row["session_id"] and
                row.get("canonical_path") == source_row["canonical_path"] and
                row.get("sha256") == source_row["sha256"] and
                type(row.get("byte_size")) is int and row["byte_size"] > 0,
                "published RT local asset row drift")
        require((row.get("held_target_outer_fold_id"), row.get("held_target_outer_fold_index")) ==
                (source_row["held_target_outer_fold_id"], source_row["held_target_outer_fold_index"]),
                "published RT local asset held-fold mapping drift")
        require(row.get("path_sha_source_fold_occurrence_count") == 14 and
                row.get("path_sha_identity_variant_count") == 1 and
                row.get("byte_size_authority") ==
                "EXPLICIT_ROOT_INPUT__NOT_INFERRED_OR_REHASHED_BY_PUBLISHER",
                "published RT local asset row provenance drift")
    require(payload.get("root_input_contract") == {
        "all_15_rows_explicit": True,
        "directory_glob_scan_or_available_tree_inference_used": False,
        "path_and_sha_exactly_reconciled_to_14_fold_source_consensus": True,
        "byte_size_is_explicit_root_attestation": True,
        "publisher_independently_verified_data_bytes": False,
    }, "published RT local asset root-input contract drift")
    require(payload.get("same_fd_consumer_required") is True and
            payload.get("same_fd_consumer_semantics") == (
                "load this immutable authority before path resolution; open O_NOFOLLOW once; fstat size; "
                "hash the opened bytes; rewind; parse from the same unchanged descriptor"
            ), "published RT local asset same-FD contract drift")
    for key in (
        "publisher_data_directory_enumeration_used", "publisher_data_path_resolved",
        "publisher_data_file_statted", "publisher_data_file_opened", "publisher_data_file_rehashed",
        "target_data_opened", "target_query_opened", "formal_data_opened", "cebra_imported",
        "gpu_used", "score_emitted", "official_scientific_receipt_minted",
    ):
        require(payload.get(key) is False, f"published RT local asset prohibited flag drift: {key}")
    closure = payload.get("implementation_closure")
    require(isinstance(closure, Mapping), "published RT local asset implementation closure missing")
    require(payload.get("implementation_closure_sha256") == _sha_json(closure),
            "published RT local asset implementation closure SHA drift")
    publisher = closure.get("rt_local_asset_authority_publisher")
    require(isinstance(publisher, Mapping) and isinstance(publisher.get("path"), str),
            "published RT local asset publisher binding missing")
    require(dict(closure) == _implementation_closure(Path(publisher["path"])),
            "published RT local asset implementation closure drift")


def load_published_authority_before_target_access() -> tuple[dict[str, Any], str]:
    """Load the sole canonical immutable pair through the same-FD verifier."""
    identity_graph = audit_existing_asset_identity_graph()
    payload, body_sha = _read_pair(CANONICAL_OUTPUT, "canonical_rt_local_asset_authority")
    _validate_published_payload(payload, identity_graph)
    return payload, body_sha


@contextmanager
def _verified_same_fd_asset_handle(
    *, authority: Mapping[str, Any], session_id: str,
) -> Iterator[BinaryIO]:
    """Open/hash/rewind/yield one asset on the same non-following descriptor.

    This is deliberately the only data-opening function in the module.  The
    publisher and graph audit never call it.  A future target consumer must
    first obtain ``authority`` from :func:`load_published_authority_before_target_access`.
    """
    rows = authority.get("asset_rows")
    require(isinstance(rows, list), "same-FD consumer lacks asset authority rows")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("session_id") == session_id]
    require(len(matches) == 1, "same-FD consumer session missing or ambiguous")
    row = matches[0]
    path = Path(str(row.get("canonical_path")))
    require(path.is_absolute(), "same-FD consumer asset path is not absolute")
    require(hasattr(os, "O_NOFOLLOW"), "same-FD consumer platform lacks O_NOFOLLOW")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise TrackBV2RTLocalAssetAuthorityError("cannot open authoritative RT asset without following links") from exc
    handle: BinaryIO | None = None
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "authoritative RT asset is not a regular file")
        require(before.st_size == row.get("byte_size"), "authoritative RT asset byte size drift")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
        require(digest.hexdigest() == row.get("sha256"), "authoritative RT asset SHA drift")
        after_hash = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_mode, before.st_size) ==
                (after_hash.st_dev, after_hash.st_ino, after_hash.st_mode, after_hash.st_size),
                "authoritative RT asset changed during verification")
        os.lseek(descriptor, 0, os.SEEK_SET)
        named = path.lstat()
        require(not stat.S_ISLNK(named.st_mode) and
                (named.st_dev, named.st_ino, named.st_mode, named.st_size) ==
                (before.st_dev, before.st_ino, before.st_mode, before.st_size),
                "authoritative RT asset pathname changed before parsing")
        handle = os.fdopen(descriptor, "rb", closefd=False)
        yield handle
        after_parse = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_mode, before.st_size) ==
                (after_parse.st_dev, after_parse.st_ino, after_parse.st_mode, after_parse.st_size),
                "authoritative RT asset changed during same-FD parse")
        named_after = path.lstat()
        require(not stat.S_ISLNK(named_after.st_mode) and
                _identity(named_after) == _identity(before),
                "authoritative RT asset pathname changed during same-FD parse")
    finally:
        if handle is not None:
            handle.close()
        os.close(descriptor)


@contextmanager
def open_canonical_verified_asset_after_authority(session_id: str) -> Iterator[BinaryIO]:
    """Load the canonical authority pair, then open/hash/parse one asset FD."""
    authority, _authority_body_sha = load_published_authority_before_target_access()
    with _verified_same_fd_asset_handle(authority=authority, session_id=session_id) as handle:
        yield handle


def publish_authority(
    *, root_asset_rows: Sequence[Mapping[str, Any]], entrypoint: Path,
) -> dict[str, str]:
    """Future root-only publisher; caller must enforce dual CLI authorization."""
    require(not os.path.lexists(CANONICAL_OUTPUT) and
            not os.path.lexists(CANONICAL_OUTPUT.with_name(f"{CANONICAL_OUTPUT.name}.sha256")),
            "canonical RT local asset authority output pair must be fresh")
    graph = audit_existing_asset_identity_graph()
    payload = build_unminted_authority(
        root_asset_rows=root_asset_rows, identity_graph=graph, entrypoint=entrypoint,
    )
    return _write_immutable_pair(CANONICAL_OUTPUT, payload)
