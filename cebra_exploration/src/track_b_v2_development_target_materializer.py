"""Fail-closed target/query materialization boundary for Track-B v2.

This additive module deliberately contains no NWB/NPZ loader, no CEBRA import,
no model construction, no readout fitting, no scoring path, and no writer.  It
has two narrow purposes:

* rebuild and consume the canonical development target authority before an
  asset ID or path is considered; and
* provide a synthetic/in-memory verifier for the exact objects a later,
  separately authorised target loader must materialize.

The future loader must first prove the local target bytes against a sealed
asset ledger, then may hand its already-materialized arrays to the verifier.
Subject-M has a pre-existing A2/ledger/manifest byte authority.  RT consumes a
separately root-published immutable 15-session path+size+SHA authority.  Both
routes remain no-data plans here: a later reviewed loader must preserve the
same verified inode (or a private snapshot made from its held descriptor)
through parsing.  Directory discovery/globbing is intentionally absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Mapping, Sequence

import track_b_v2_contract as base
import track_b_v2_development_target_authority as development_authority
import track_b_v2_rt_local_asset_authority as rt_asset_authority
import track_b_v2_target_query_scaffold as target_scaffold


DEVELOPMENT_TARGET_MATERIALIZER_DRY_PLAN_SCHEMA = "track_b_v2_development_target_materializer_dry_plan_v1"
SUBJECT_M_TARGET_ASSET_LEDGER_SCHEMA = "track_b_v2_subject_m_a2_target_asset_ledger_authority_v1"
RT_TARGET_ASSET_LEDGER_SCHEMA = "track_b_v2_rt_target_asset_ledger_authority_binding_v1"
MATERIALIZED_TARGET_QUERY_PROOF_SCHEMA = "track_b_v2_materialized_target_query_proof_v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUA_ROOT = _REPO_ROOT / "sua_exploration"
_A2_PREFLIGHT = _SUA_ROOT / "results" / "a2_matched_subject_shift_v2" / "official_cpu_preflight.json"
_SUBM_SCHEMA_LEDGER = _SUA_ROOT / "results" / "dandi_000688_subm_co_schema_preflight_v2" / "receipt.json"
_SUBM_SCOPE_MANIFEST = _SUA_ROOT / "manifests" / "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json"

_A2_PREFLIGHT_SHA256 = "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd"
_SUBM_SCHEMA_LEDGER_SHA256 = "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283"
_SUBM_SCOPE_MANIFEST_SHA256 = "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55"
_SUBM_EXTERNAL_TARGET_COUNT = 15
_RT_LOCAL_ASSET_AUTHORITY_SHA256 = "771ab920531a322fbccc9a745d80cc1cb6b3325bcc469283161e2561f7da9352"


class TrackBV2DevelopmentTargetMaterializerError(development_authority.TrackBV2DevelopmentTargetAuthorityError):
    """Raised before an unverified target can become a materialization input."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2DevelopmentTargetMaterializerError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _absolute_lexical(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path(os.path.abspath(str(expanded)))


def _assert_real_directory_chain(directory: Path, *, label: str) -> Path:
    """Reject every symlink in an existing authority/output parent chain."""
    directory = _absolute_lexical(directory)
    current = Path(directory.anchor)
    for part in directory.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            raise TrackBV2DevelopmentTargetMaterializerError(
                f"{label} parent is absent: {current}"
            ) from exc
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
                f"{label} parent must be a real directory: {current}")
    return directory


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mode, info.st_mtime_ns, info.st_ctime_ns)


def _assert_path_identity(path: Path, identity: tuple[int, int, int, int, int, int], *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TrackBV2DevelopmentTargetMaterializerError(
            f"{label} pathname disappeared or was replaced after open"
        ) from exc
    require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            f"{label} pathname is no longer a regular non-symlink")
    require(_file_identity(info) == identity, f"{label} pathname identity changed after open")


@dataclass(frozen=True)
class _VerifiedRegularFile:
    path: Path
    raw: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int]
    mode: int


def _read_mode_0444_same_fd(path: Path, *, label: str, expected_sha256: str) -> _VerifiedRegularFile:
    """Read one immutable JSON authority with O_NOFOLLOW and post-open proof."""
    require(_valid_sha(expected_sha256), f"{label} expected SHA invalid")
    lexical = _absolute_lexical(path)
    _assert_real_directory_chain(lexical.parent, label=label)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW for immutable authority reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise TrackBV2DevelopmentTargetMaterializerError(
            f"cannot open {label} without following symlinks"
        ) from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        require(stat.S_IMODE(before.st_mode) == 0o444, f"{label} must be mode 0444")
        raw_blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            raw_blocks.append(block)
        raw = b"".join(raw_blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _file_identity(before)
    require(_file_identity(after) == identity and len(raw) == before.st_size,
            f"{label} changed while read")
    _assert_path_identity(lexical, identity, label=label)
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == expected_sha256, f"{label} SHA drift")
    return _VerifiedRegularFile(
        path=lexical, raw=raw, sha256=digest, identity=identity,
        mode=stat.S_IMODE(before.st_mode),
    )


def _parse_json(verified: _VerifiedRegularFile, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(verified.raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2DevelopmentTargetMaterializerError(f"{label} is not JSON") from exc
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _read_strict_pair(path: Path, *, label: str, expected_sha256: str) -> tuple[dict[str, Any], _VerifiedRegularFile]:
    body = _read_mode_0444_same_fd(path, label=label, expected_sha256=expected_sha256)
    sidecar_path = body.path.with_name(f"{body.path.name}.sha256")
    sidecar = _read_mode_0444_same_fd(
        sidecar_path, label=f"{label} sidecar",
        expected_sha256=hashlib.sha256(
            f"{body.sha256}  {body.path.name}\n".encode("ascii")
        ).hexdigest(),
    )
    # The sidecar's expected *content* above does not establish that its body
    # had the required text unless it is compared directly.  Recheck the body
    # pathname after consuming the sidecar to reject a pair split by rename.
    _assert_path_identity(body.path, body.identity, label=label)
    require(sidecar.raw == f"{body.sha256}  {body.path.name}\n".encode("ascii"),
            f"{label} body/sidecar mismatch")
    return _parse_json(body, label=label), body


def _read_legacy_sidecarless(path: Path, *, label: str, expected_sha256: str) -> tuple[dict[str, Any], _VerifiedRegularFile]:
    body = _read_mode_0444_same_fd(path, label=label, expected_sha256=expected_sha256)
    require(not os.path.lexists(body.path.with_name(f"{body.path.name}.sha256")),
            f"{label} is legacy sidecarless authority; an adjacent sidecar is forbidden")
    return _parse_json(body, label=label), body


def _canonical_development_authority(
    *, dataset: str, view: str | None, outer_fold_id: str, target_session_id: object,
) -> dict[str, Any]:
    """Rebuild the sole permitted authority before inspecting asset identity."""
    try:
        return development_authority.build_development_target_query_authority(
            dataset=dataset, view=view, outer_fold_id=outer_fold_id,
            target_session_id=target_session_id,
        )
    except development_authority.TrackBV2DevelopmentTargetAuthorityError as exc:
        raise TrackBV2DevelopmentTargetMaterializerError(str(exc)) from exc


def _validate_subject_m_ledger_authorities(
    *, canonical_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Join the A2 preflight, v2 ledger, and frozen manifest without NWB I/O."""
    require(canonical_authority.get("dataset") == "subject_m", "subject-M asset ledger requires subject_m authority")
    view = canonical_authority.get("view")
    require(view in ("sua", "pseudo_mua"), "subject-M asset ledger view invalid")
    target_session_id = canonical_authority.get("target_session_id")
    require(isinstance(target_session_id, str), "canonical authority target ID missing")
    membership = canonical_authority.get("target_reference_lineage_membership")
    require(isinstance(membership, Mapping), "canonical authority target lineage missing")
    target_asset_id = membership.get("target_asset_id")
    require(isinstance(target_asset_id, str) and target_asset_id,
            "canonical authority must bind the target asset before ledger coercion")

    official, official_body = _read_strict_pair(
        _A2_PREFLIGHT, label="A2 official preflight", expected_sha256=_A2_PREFLIGHT_SHA256,
    )
    ledger, ledger_body = _read_legacy_sidecarless(
        _SUBM_SCHEMA_LEDGER, label="sub-M schema ledger", expected_sha256=_SUBM_SCHEMA_LEDGER_SHA256,
    )
    manifest, manifest_body = _read_legacy_sidecarless(
        _SUBM_SCOPE_MANIFEST, label="sub-M scope manifest", expected_sha256=_SUBM_SCOPE_MANIFEST_SHA256,
    )
    require(official.get("screen_id") == "a2_matched_subject_shift_v2" and
            official.get("official_preflight") is True and
            official.get("receipt_kind") == "a2_matched_subject_shift_v2_official_preflight",
            "A2 official preflight identity drift")
    external = official.get("external_subject_M_audit")
    require(isinstance(external, Mapping) and external.get("expected_count") == _SUBM_EXTERNAL_TARGET_COUNT and
            external.get("admissible_count") == _SUBM_EXTERNAL_TARGET_COUNT,
            "A2 official external subject-M audit drift")
    a2_rows = external.get("sessions")
    require(isinstance(a2_rows, list) and len(a2_rows) == _SUBM_EXTERNAL_TARGET_COUNT,
            "A2 official external target rows drift")
    a2_by_session = {str(row.get("session")): row for row in a2_rows if isinstance(row, Mapping)}
    require(len(a2_by_session) == _SUBM_EXTERNAL_TARGET_COUNT and target_session_id in a2_by_session,
            "canonical target is absent from A2 external target audit")
    a2_row = a2_by_session[target_session_id]
    require(a2_row.get("admissible") is True and
            a2_row.get("eligibility_ledger_sha256") == _SUBM_SCHEMA_LEDGER_SHA256 and
            isinstance(a2_row.get("nwb_path"), str),
            "A2 external target eligibility/ledger binding drift")

    require(ledger.get("schema_version") == 2 and
            ledger.get("receipt_kind") == "dandi_000688_subm_co_score_blind_schema_preflight_v2" and
            ledger.get("scope_id") == "dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2" and
            ledger.get("status") == "COMPLETE_SCORE_BLIND_SCHEMA_PREFLIGHT" and
            ledger.get("eligible_session_count") == _SUBM_EXTERNAL_TARGET_COUNT,
            "sub-M schema ledger identity/status drift")
    manifest_binding = ledger.get("immutable_v2_manifest")
    require(isinstance(manifest_binding, Mapping) and
            manifest_binding.get("path") == "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json" and
            manifest_binding.get("sha256") == _SUBM_SCOPE_MANIFEST_SHA256 and
            manifest_binding.get("selected_asset_count") == 22,
            "sub-M ledger/manifest binding drift")
    require(manifest.get("scope_id") == ledger.get("scope_id") and
            manifest.get("status") == "candidate_frozen_metadata_only_external_runner_blocked",
            "sub-M scope manifest identity drift")
    selected = manifest.get("selected_assets")
    downloads = ledger.get("verified_downloads")
    dispositions = ledger.get("asset_disposition_ledger")
    eligible_ids = ledger.get("eligible_session_ids")
    require(isinstance(selected, list) and len(selected) == 22 and
            isinstance(downloads, list) and len(downloads) == 22 and
            isinstance(dispositions, list) and len(dispositions) == 22 and
            isinstance(eligible_ids, list) and len(eligible_ids) == _SUBM_EXTERNAL_TARGET_COUNT,
            "sub-M manifest/ledger row counts drift")
    selected_by_session = {str(row.get("session_id")): row for row in selected if isinstance(row, Mapping)}
    download_by_asset = {str(row.get("asset_id")): row for row in downloads if isinstance(row, Mapping)}
    disposition_by_session = {str(row.get("session_id")): row for row in dispositions if isinstance(row, Mapping)}
    require(len(selected_by_session) == len(download_by_asset) == len(disposition_by_session) == 22,
            "sub-M manifest/download/disposition uniqueness drift")
    asset = selected_by_session.get(target_session_id)
    disposition = disposition_by_session.get(target_session_id)
    require(isinstance(asset, Mapping) and isinstance(disposition, Mapping),
            "canonical target missing from sealed manifest/disposition ledger")
    asset_id = asset.get("asset_id")
    frozen_path = asset.get("path")
    expected_sha = asset.get("sha256")
    expected_bytes = asset.get("size")
    require(asset_id == target_asset_id, "canonical pointer-bound target asset differs from frozen manifest")
    require(isinstance(asset_id, str) and isinstance(frozen_path, str) and _valid_sha(expected_sha) and
            isinstance(expected_bytes, int) and expected_bytes > 0,
            "sub-M sealed asset row malformed")
    download = download_by_asset.get(asset_id)
    require(isinstance(download, Mapping) and download.get("asset_id") == asset_id and
            download.get("sha256") == expected_sha and download.get("bytes") == expected_bytes and
            download.get("size_and_sha256_verified_before_nwb_open") is True,
            "sub-M verified-download path/size/SHA binding drift")
    require(disposition.get("asset_id") == asset_id and disposition.get("frozen_path") == frozen_path and
            disposition.get("eligible") is True and disposition.get("disposition") == "ELIGIBLE" and
            disposition.get("score_blind") is True and asset_id in eligible_ids,
            "sub-M target eligibility disposition drift")
    local_path = _absolute_lexical(Path(str(a2_row["nwb_path"])))
    frozen = Path(frozen_path)
    require(local_path.name == frozen.name and local_path.parts[-2:] == tuple(frozen.parts),
            "A2 official target local path differs from frozen manifest")
    payload = {
        "schema": SUBJECT_M_TARGET_ASSET_LEDGER_SCHEMA,
        "status": "A2_PRECHECK_AND_V2_LEDGER_MANIFEST_VERIFIED__LOCAL_NWB_NOT_OPENED",
        "dataset": "subject_m",
        "view": view,
        "canonical_development_target_authority_sha256": canonical_authority[
            "development_target_query_authority_sha256"
        ],
        "authorities": {
            "a2_official_preflight": {
                "body_path": str(official_body.path), "body_sha256": official_body.sha256,
                "sidecar_path": str(official_body.path.with_name(f"{official_body.path.name}.sha256")),
                "mode": "0444", "legacy_sidecarless": False,
            },
            "subm_schema_ledger": {
                "body_path": str(ledger_body.path), "body_sha256": ledger_body.sha256,
                "mode": "0444", "legacy_sidecarless": True,
            },
            "subm_scope_manifest": {
                "body_path": str(manifest_body.path), "body_sha256": manifest_body.sha256,
                "mode": "0444", "legacy_sidecarless": True,
            },
        },
        "target_asset": {
            "session_id": target_session_id,
            "asset_id": asset_id,
            "frozen_relative_path": frozen_path,
            "a2_official_local_nwb_path": str(local_path),
            "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
        },
        "verification_required_before_nwb_parser": "same_fd_O_NOFOLLOW_size_sha256_and_post_open_path_identity",
        "verification_required_during_nwb_parser": (
            "parser_must_consume_the_continuously_held_verified_inode_or_a_private_snapshot_copied_from_that_fd"
        ),
        "pathname_pre_and_post_hash_alone_is_sufficient": False,
        "verification_required_after_nwb_parser": (
            "held_verified_inode_or_private_snapshot_identity_and_bytes_remain_unchanged"
        ),
        "target_data_discovered": False,
        "target_data_opened": False,
        "target_nwb_opened_while_building_authority": False,
    }
    # This is a projection of the three sealed bodies plus the already-canonical
    # development authority, not a new competing asset authority or receipt.
    return payload


def _validate_rt_ledger_authority(*, canonical_authority: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one canonical RT target to the root-published 15-asset authority.

    The authority itself is consumed and live-closure-validated before this
    function accepts a target row.  This function neither resolves nor opens
    the target pathname.
    """
    require(canonical_authority.get("dataset") == "rt", "RT asset-ledger gate requires RT authority")
    target_session_id = canonical_authority.get("target_session_id")
    outer_fold_id = canonical_authority.get("outer_fold_id")
    require(isinstance(target_session_id, str) and target_session_id.startswith("ses-RT-"),
            "RT canonical target session malformed")
    require(isinstance(outer_fold_id, str) and outer_fold_id.startswith("rt_outer_fold_"),
            "RT canonical outer fold malformed")
    source_binding = canonical_authority.get("source_authority_binding")
    membership = canonical_authority.get("target_reference_lineage_membership")
    require(isinstance(source_binding, Mapping) and isinstance(membership, Mapping),
            "RT canonical source/membership authority missing")
    require(source_binding.get("selected_outer_fold_id") == outer_fold_id and
            source_binding.get("opaque_held_out_target_session_id") == target_session_id and
            membership.get("target_session_id") == target_session_id,
            "RT canonical target/fold membership drift")

    try:
        authority, body_sha = rt_asset_authority.load_published_authority_before_target_access()
    except rt_asset_authority.TrackBV2RTLocalAssetAuthorityError as exc:
        raise TrackBV2DevelopmentTargetMaterializerError(str(exc)) from exc
    require(body_sha == _RT_LOCAL_ASSET_AUTHORITY_SHA256,
            "RT local asset authority differs from root-reviewed external SHA anchor")
    require(authority.get("schema") == rt_asset_authority.SCHEMA and
            authority.get("status") == rt_asset_authority.STATUS and
            (authority.get("dataset"), authority.get("view"), authority.get("asset_count")) ==
            ("rt", None, 15), "RT local asset authority scope/schema/status drift")
    rows = authority.get("asset_rows")
    require(isinstance(rows, list) and len(rows) == 15,
            "RT local asset authority must contain exactly 15 rows")
    matches = [row for row in rows if isinstance(row, Mapping) and
               row.get("session_id") == target_session_id]
    require(len(matches) == 1, "RT canonical target missing or ambiguous in local asset authority")
    row = matches[0]
    path = row.get("canonical_path")
    expected_sha = row.get("sha256")
    expected_bytes = row.get("byte_size")
    fold_index = row.get("held_target_outer_fold_index")
    require(row.get("held_target_outer_fold_id") == outer_fold_id and
            isinstance(fold_index, int) and not isinstance(fold_index, bool),
            "RT local asset held-target fold binding drift")
    require(isinstance(path, str) and Path(path).is_absolute() and _valid_sha(expected_sha) and
            isinstance(expected_bytes, int) and not isinstance(expected_bytes, bool) and expected_bytes > 0,
            "RT local asset target row malformed")
    return {
        "schema": RT_TARGET_ASSET_LEDGER_SCHEMA,
        "status": "ROOT_PUBLISHED_RT_LOCAL_ASSET_AUTHORITY_VALIDATED__NO_TARGET_OPEN",
        "canonical_development_target_authority_sha256": canonical_authority[
            "development_target_query_authority_sha256"
        ],
        "authority": {
            "body_path": str(rt_asset_authority.CANONICAL_OUTPUT),
            "body_sha256": body_sha,
            "sidecar_path": str(rt_asset_authority.CANONICAL_OUTPUT.with_name(
                f"{rt_asset_authority.CANONICAL_OUTPUT.name}.sha256")),
            "implementation_closure_sha256": authority["implementation_closure_sha256"],
            "existing_identity_graph_sha256": authority["parent_bindings"][
                "existing_identity_graph_sha256"
            ],
            "asset_count": authority["asset_count"],
        },
        "target_asset": {
            "session_id": target_session_id,
            "held_target_outer_fold_id": outer_fold_id,
            "held_target_outer_fold_index": fold_index,
            "canonical_local_nwb_path": path,
            "expected_sha256": expected_sha,
            "expected_bytes": expected_bytes,
        },
        "directory_discovery_or_glob_permitted": False,
        "synthetic_or_derived_directory_roster_permitted": False,
        "verification_required_before_nwb_parser": "same_fd_O_NOFOLLOW_size_sha256_and_post_open_path_identity",
        "verification_required_during_nwb_parser": (
            "parser_must_consume_the_continuously_held_verified_inode_or_a_private_snapshot_copied_from_that_fd"
        ),
        "required_live_parser_entrypoint": (
            "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority"
        ),
        "caller_supplied_asset_ledger_mapping_permitted": False,
        "pathname_pre_and_post_hash_alone_is_sufficient": False,
        "target_data_discovered": False,
        "target_data_opened": False,
    }


def build_development_target_materializer_dry_plan(
    *, dataset: str, view: str | None, outer_fold_id: str, target_session_id: object,
    proposed_target_path: object | None = None, proposed_target_discovery: object | None = None,
    execution_requested: bool = False, device: str = "cpu",
) -> dict[str, Any]:
    """Build a no-data plan from the canonical authority, never caller SHA strings.

    Scope is resolved first.  The canonical development authority then rebuilds
    the sealed pointer/source graph *before* the target ID can flow into an
    asset-ledger lookup.  The CLI for this function exposes no target path or
    execution option.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(proposed_target_path is None, "development target materializer accepts no target path")
    require(proposed_target_discovery is None, "development target materializer accepts no discovery callable")
    require(execution_requested is False, "development target materializer has no execution mode")
    require(device == "cpu", "development target materializer is no-data/CPU only")
    canonical = _canonical_development_authority(
        dataset=dataset, view=view, outer_fold_id=outer_fold_id, target_session_id=target_session_id,
    )
    canonical_sha = canonical.get("development_target_query_authority_sha256")
    require(_valid_sha(canonical_sha), "canonical development authority SHA missing")
    if dataset == "subject_m":
        asset_gate = _validate_subject_m_ledger_authorities(canonical_authority=canonical)
        status = "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS"
    else:
        asset_gate = _validate_rt_ledger_authority(canonical_authority=canonical)
        status = "CANONICAL_DEVELOPMENT_AUTHORITY_AND_RT_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS"
    payload = {
        "schema": DEVELOPMENT_TARGET_MATERIALIZER_DRY_PLAN_SCHEMA,
        "status": status,
        "dataset": dataset,
        "view": view,
        "outer_fold_id": canonical["outer_fold_id"],
        "target_session_id": canonical["target_session_id"],
        "canonical_development_authority": {
            "schema": canonical["schema"],
            "status": canonical["status"],
            "development_target_query_authority_sha256": canonical_sha,
            "metric_pointer_body_sha256": canonical["metric_pointer_validation"]["pointer_body_sha256"],
            "fixed_canonical_geometry_contract_sha256": _sha_json(canonical["fixed_canonical_geometry_contract"]),
        },
        "target_asset_ledger_gate": asset_gate,
        "future_target_array_materialization_contract": {
            "continuous_chronological_support_prefix": "M50_FOR_SUBJECT_M__M24_FOR_RT",
            "query": "STRICT_POST_M_ONLY",
            "prediction_endpoints": "exactly_valid_window_start_plus_49",
            "standard_offset10_actual_receptive_field": "every_scored_RF_wholly_inside_query_and_disjoint_from_support",
            "standard_offset10_previous_raw_bins": 5,
            "standard_offset10_right_extent_including_target_raw_bins": 5,
            "standard_offset10_strictly_future_raw_bins": 4,
            "standard_offset10_receptive_field_width_raw_bins": 10,
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "ordered_target_float32_bytes_must_equal_T4_reference": True,
            "subject_m_sua_and_pmua_share_target_behavior_order": True,
            "pseudo_mua_requires_actual_target_pooling_provenance": True,
            "formal_subc_discovery_permitted": False,
        },
        "readout_roles": _fixed_readout_roles(canonical),
        "source_geometry_selection_performed": False,
        "target_geometry_selection_performed": False,
        "posthoc_geometry_seed_or_readout_selection_permitted": False,
        "target_data_discovery_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
    # The freshly rebuilt development-authority SHA above is the only authority
    # binding consumed by this dry route; this plan intentionally mints no
    # competing materializer/asset authority digest.
    return payload


def _fixed_readout_roles(canonical_authority: Mapping[str, Any]) -> dict[str, Any]:
    """Render fixed roles without carrying legacy selector fields forward."""
    fixed = canonical_authority.get("fixed_canonical_geometry_contract")
    require(isinstance(fixed, Mapping), "canonical fixed geometry contract missing")
    require(fixed.get("source_geometry_selection_performed") is False and
            fixed.get("target_geometry_selection_performed") is False and
            fixed.get("source_selector_fit_count") == 0 and fixed.get("target_selector_fit_count") == 0,
            "canonical authority no longer freezes zero-fit geometry")
    geometry_sha = _sha_json(fixed)
    return {
        "fixed_geometry_contract_sha256": geometry_sha,
        "fixed_geometry": {
            "output_dimension": 8,
            "iterations": 10_000,
            "linear_ridge_normalized_lambda": 0.01,
            "cosine_knn_k": 3,
            "selection_performed": False,
        },
        "mandatory_decoders": ["linear_ridge", "knn_cosine_k3"],
        "all_routes_mandatory_once_later_score_is_authorized": True,
        "routes": {
            "source_only_consumer_mechanism_alignment": {
                "readout_fit_scope": "source_fit_only",
                "target_support_dense_labels_in_readout_fit": False,
                "scientific_role": "mechanism_alignment_not_accuracy_headline",
            },
            "target_support_only_standard_cebra_accuracy": {
                "readout_fit_scope": "target_support_only",
                "target_support_dense_labels_in_readout_fit": True,
                "scientific_role": "accuracy_table_headline_with_extra_target_readout_supervision_declared",
            },
            "source_plus_target_support_hybrid_sensitivity": {
                "readout_fit_scope": "source_fit_plus_target_support",
                "target_support_dense_labels_in_readout_fit": True,
                "scientific_role": "mandatory_sensitivity_not_upper_bound_not_headline",
            },
        },
        "runtime_route_or_decoder_selection_permitted": False,
    }


@dataclass(frozen=True)
class VerifiedTargetAssetBeforeOpen:
    """Same-FD byte and identity proof that a future parser must preserve."""

    session_id: str
    path: Path
    sha256: str
    byte_count: int
    identity: tuple[int, int, int, int, int, int]

    def as_dict(self, *, phase: str) -> dict[str, Any]:
        return {
            "phase": phase,
            "session_id": self.session_id,
            "path": str(self.path),
            "actual_sha256": self.sha256,
            "actual_bytes": self.byte_count,
            "identity": {
                "st_dev": self.identity[0], "st_ino": self.identity[1],
                "st_size": self.identity[2], "st_mode": self.identity[3],
                "st_mtime_ns": self.identity[4], "st_ctime_ns": self.identity[5],
            },
            "identity_sha256": _sha_json(self.identity),
        }


def _stream_hash_same_fd(path: Path, *, label: str, expected_sha256: str, expected_bytes: int) -> VerifiedTargetAssetBeforeOpen:
    """Future-only local-byte verifier; it never parses NWB content."""
    lexical = _absolute_lexical(path)
    _assert_real_directory_chain(lexical.parent, label=label)
    require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW for target byte verification")
    try:
        descriptor = os.open(lexical, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise TrackBV2DevelopmentTargetMaterializerError(
            f"cannot open {label} without following symlinks"
        ) from exc
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
        digest = hashlib.sha256()
        count = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
            count += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _file_identity(before)
    require(_file_identity(after) == identity and count == before.st_size,
            f"{label} changed while hashed")
    _assert_path_identity(lexical, identity, label=label)
    require(count == expected_bytes and identity[2] == expected_bytes, f"{label} byte-size drift")
    observed = digest.hexdigest()
    require(observed == expected_sha256, f"{label} SHA drift")
    return VerifiedTargetAssetBeforeOpen(
        session_id="", path=lexical, sha256=observed, byte_count=count, identity=identity,
    )


def verify_subject_m_target_asset_before_loader(
    *, asset_ledger_authority: Mapping[str, Any], target_path: Path,
) -> VerifiedTargetAssetBeforeOpen:
    """Verify a current Subject-M NWB *before* a future NWB loader sees it."""
    require(asset_ledger_authority.get("schema") == SUBJECT_M_TARGET_ASSET_LEDGER_SCHEMA,
            "subject-M target asset ledger schema drift")
    asset = asset_ledger_authority.get("target_asset")
    require(isinstance(asset, Mapping), "subject-M target asset row missing")
    session = asset.get("session_id")
    expected_path = asset.get("a2_official_local_nwb_path")
    expected_sha = asset.get("expected_sha256")
    expected_bytes = asset.get("expected_bytes")
    require(isinstance(session, str) and isinstance(expected_path, str) and _valid_sha(expected_sha) and
            isinstance(expected_bytes, int) and expected_bytes > 0,
            "subject-M target asset row malformed")
    lexical = _absolute_lexical(target_path)
    require(lexical == _absolute_lexical(Path(expected_path)),
            "target path differs from A2 official preflight ledger path")
    frozen = Path(str(asset.get("frozen_relative_path", "")))
    require(lexical.name == frozen.name and lexical.parts[-2:] == tuple(frozen.parts),
            "target path differs from frozen manifest path")
    proof = _stream_hash_same_fd(
        lexical, label=f"subject-M target {session} before loader",
        expected_sha256=str(expected_sha), expected_bytes=expected_bytes,
    )
    return VerifiedTargetAssetBeforeOpen(
        session_id=session, path=proof.path, sha256=proof.sha256,
        byte_count=proof.byte_count, identity=proof.identity,
    )


def verify_subject_m_target_asset_after_loader(*, before: VerifiedTargetAssetBeforeOpen) -> dict[str, Any]:
    """Post-loader hardening check; never sufficient as parser-consumption proof.

    A reviewed live loader must additionally keep the verified descriptor open
    through parsing (or parse a private snapshot copied from it).  Reopening the
    pathname before and after parsing cannot exclude an intermediate ABA swap.
    """
    require(before.session_id.startswith("sub-M_ses-CO-"), "after-loader proof has invalid subject-M session")
    after = _stream_hash_same_fd(
        before.path, label=f"subject-M target {before.session_id} after loader",
        expected_sha256=before.sha256, expected_bytes=before.byte_count,
    )
    require(after.identity == before.identity,
            "target pathname inode/metadata changed between pre- and post-loader verification")
    return after.as_dict(phase="after_nwb_loader") | {
        "pre_loader_identity_sha256": _sha_json(before.identity),
        "same_inode_as_pre_loader": True,
        "pathname_pre_and_post_hash_alone_is_sufficient": False,
        "same_held_fd_or_private_snapshot_consumption_proof_still_required": True,
    }


def _canonical_nonnegative_indices(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    result = tuple(values)
    require(result and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in result),
            f"{label} must contain nonnegative integer indices")
    require(len(set(result)) == len(result) and tuple(sorted(result)) == result,
            f"{label} must be unique and chronologically sorted")
    return result


def _validate_trial_ordinals(values: Sequence[int], *, budget: int) -> tuple[int, ...]:
    ordinals = tuple(values)
    require(ordinals and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in ordinals),
            "raw-bin trial ordinals must be nonnegative integers")
    require(tuple(sorted(ordinals)) == ordinals, "raw-bin trial ordinals must be chronological")
    observed_support = set(ordinals[:]) & set(range(budget))
    require(observed_support == set(range(budget)), "all M50/M24 support trials must occur in chronological record")
    require(any(item >= budget for item in ordinals), "strict post-M query rows are required")
    return ordinals


def _validate_float32_matrix_bytes(raw: bytes, *, rows: int, label: str) -> str:
    require(isinstance(raw, bytes), f"{label} must be raw bytes")
    require(rows > 0 and len(raw) == rows * 2 * 4,
            f"{label} must encode contiguous float32 [rows,2] bytes")
    values = tuple(struct.iter_unpack("<ff", raw))
    require(len(values) == rows and all(math.isfinite(value) for pair in values for value in pair),
            f"{label} must contain finite little-endian float32 [rows,2] values")
    return hashlib.sha256(raw).hexdigest()


def _unique_float32_rows(raw: bytes, *, rows: int) -> int:
    """Count numerical [x,y] pairs from the exact already-validated bytes."""
    values = tuple(struct.iter_unpack("<ff", raw))
    require(len(values) == rows, "dense support float32 row parser drift")
    return len(set(values))


def _target_lineage_requirements(
    *, dataset: str, target_reference_lineage: Mapping[str, Any], query_rows: int,
) -> dict[str, str]:
    required = {
        "t4_reference_query_identity_sha256",
        "metric_implementation_authority_sha256",
        "t4_runtime_receipt_sha256",
    }
    if dataset == "subject_m":
        required |= {
            "v9_commit_receipt_sha256", "v9_runtime_base_input_trace_sha256",
            "v9_runtime_query_behavior_trace_sha256", "t4_predictions_targets_npz_sha256",
        }
    require(set(target_reference_lineage) >= required,
            "target reference lineage lacks required exact query/metric authorities")
    result = {name: target_reference_lineage[name] for name in sorted(required)}
    require(all(_valid_sha(value) for value in result.values()), "target reference lineage SHA invalid")
    declared_rows = target_reference_lineage.get("t4_reference_query_row_count")
    if declared_rows is not None:
        require(declared_rows == query_rows, "T4 reference query row count differs from ordered endpoint rows")
    return result


def _validate_target_pooling(
    *, view: str | None, target_feature_sha256: str, pooling_provenance: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    require(_valid_sha(target_feature_sha256), "target neural feature SHA invalid")
    if view == "sua":
        require(pooling_provenance is None, "SUA target materialization may not claim pseudo-MUA pooling")
        return None
    require(isinstance(pooling_provenance, Mapping), "pMUA target requires actual pooling provenance")
    require(pooling_provenance.get("method") == "electrode_ids_from_units_then_pool_spikes_by_electrode" and
            pooling_provenance.get("output_pseudo_mua_feature_sha256") == target_feature_sha256 and
            pooling_provenance.get("channel_order") == "np_unique_ascending_electrode_id" and
            pooling_provenance.get("output_dtype") == "float32",
            "pMUA target pooling provenance drift")
    return dict(pooling_provenance)


def materialize_synthetic_target_query_proof(
    *, dataset: str, view: str | None, outer_fold_id: str, target_session_id: object,
    raw_trial_ordinal_by_bin: Sequence[int], valid_window_start_indices: Sequence[int],
    full_receptive_field_raw_indices: Sequence[Sequence[int]],
    t4_target_float32_bytes: bytes, cebra_target_float32_bytes: bytes,
    dense_support_velocity_float32_bytes: bytes, dense_support_unique_row_count: int,
    target_feature_sha256: str, target_reference_lineage: Mapping[str, Any],
    vendored_model_alignment_authority_sha256: str, pooling_provenance: Mapping[str, Any] | None = None,
    synthetic_fixture: bool = False, target_asset_preopen_proof: object | None = None,
    execution_requested: bool = False,
) -> dict[str, Any]:
    """Verify synthetic/in-memory arrays against the canonical no-data authority.

    This is intentionally not a target loader.  ``synthetic_fixture=True`` is
    mandatory so no caller can accidentally pass arrays obtained from target
    data during the current no-target stage.  A later reviewed live executor
    must first call the same-FD asset verifier and then expose an explicitly
    authorised non-synthetic bridge to this array-level logic.
    """
    dataset, view = base.validate_scope(dataset, view)
    require(execution_requested is False, "synthetic target proof has no execution mode")
    require(synthetic_fixture is True, "target materialization is not authorized outside synthetic fixtures")
    require(target_asset_preopen_proof is None,
            "synthetic target proof may not accept a real target asset open proof")
    canonical = _canonical_development_authority(
        dataset=dataset, view=view, outer_fold_id=outer_fold_id, target_session_id=target_session_id,
    )
    target_id = canonical["target_session_id"]
    contract = canonical["target_support_query_trial_window_contract"]["support_query_contract"]
    budget = base.SUPPORT_BUDGET_TRIALS[dataset]
    ordinals = _validate_trial_ordinals(raw_trial_ordinal_by_bin, budget=budget)
    support_stop = max(index for index, ordinal in enumerate(ordinals) if ordinal < budget)
    require(all(ordinal >= budget for ordinal in ordinals[support_stop + 1:]),
            "continuous support prefix must end at M50/M24 before strict post-M query")
    support_indices = tuple(range(0, support_stop + 1))
    query_indices = tuple(range(support_stop + 1, len(ordinals)))
    starts = _canonical_nonnegative_indices(valid_window_start_indices, label="ordered valid window starts")
    require(set(starts).issubset(set(query_indices)), "valid window starts must be strictly post-M query rows")
    endpoints = tuple(start + 49 for start in starts)
    require(set(endpoints).issubset(set(query_indices)),
            "valid-start-plus-49 endpoints must be strictly post-M query rows")
    t4_sha = _validate_float32_matrix_bytes(t4_target_float32_bytes, rows=len(endpoints), label="T4 target")
    cebra_sha = _validate_float32_matrix_bytes(cebra_target_float32_bytes, rows=len(endpoints), label="CEBRA target")
    require(t4_target_float32_bytes == cebra_target_float32_bytes,
            "CEBRA ordered target float32 bytes differ from T4 reference target bytes")
    dense_sha = _validate_float32_matrix_bytes(
        dense_support_velocity_float32_bytes, rows=len(support_indices), label="dense support velocity",
    )
    observed_unique_rows = _unique_float32_rows(
        dense_support_velocity_float32_bytes, rows=len(support_indices),
    )
    require(dense_support_unique_row_count == observed_unique_rows,
            "dense support unique-row count does not equal exact float32 support labels")
    lineage = _target_lineage_requirements(
        dataset=dataset, target_reference_lineage=target_reference_lineage, query_rows=len(endpoints),
    )
    require(_valid_sha(vendored_model_alignment_authority_sha256), "vendored model alignment SHA invalid")
    pooling = _validate_target_pooling(
        view=view, target_feature_sha256=target_feature_sha256, pooling_provenance=pooling_provenance,
    )
    receptive_fields = tuple(tuple(item) for item in full_receptive_field_raw_indices)
    expected_receptive_fields = tuple(
        tuple(range(endpoint - 5, endpoint + 5)) for endpoint in endpoints
    )
    require(receptive_fields == expected_receptive_fields,
            "offset10 receptive fields must be exact half-open [endpoint-5, endpoint+5) raw-bin ranges")
    proof = target_scaffold.QueryReceptiveFieldAndTargetByteProof(
        dataset=dataset, view=view, target_session_id=target_id,
        support_raw_indices=support_indices, query_raw_indices=query_indices,
        valid_window_start_indices=starts,
        t4_reference_ordered_prediction_target_raw_bin_indices=endpoints,
        cebra_scored_ordered_prediction_target_raw_bin_indices=endpoints,
        full_receptive_field_raw_indices=receptive_fields,
        vendored_model_alignment_authority_sha256=vendored_model_alignment_authority_sha256,
        t4_target_float32_raw_bytes_sha256=t4_sha,
        cebra_target_float32_raw_bytes_sha256=cebra_sha,
        t4_reference_query_identity_sha256=lineage["t4_reference_query_identity_sha256"],
        metric_implementation_authority_sha256=lineage["metric_implementation_authority_sha256"],
    ).as_dict()
    temporal = proof["vendored_model_receptive_field_alignment"][
        "future_raw_bins_relative_to_prediction_target"
    ]
    require(proof["vendored_model_receptive_field_alignment"]["receptive_field_width_min"] == 10 and
            proof["vendored_model_receptive_field_alignment"]["receptive_field_width_max"] == 10 and
            temporal["maximum_future_raw_bins"] == 4,
            "offset10 receptive-field width/future exposure differs from vendored Offset(5,5) semantics")
    shared_subject_m_behavior_order: dict[str, Any] | None = None
    if dataset == "subject_m":
        shared_subject_m_behavior_order = {
            "ordered_prediction_target_raw_bin_indices_sha256": proof[
                "ordered_prediction_target_raw_bin_indices"]["ordered_index_sha256"],
            "ordered_target_float32_raw_bytes_sha256": t4_sha,
            "t4_reference_query_identity_sha256": lineage["t4_reference_query_identity_sha256"],
        }
        shared_subject_m_behavior_order["cross_view_behavior_order_authority_sha256"] = _sha_json(
            shared_subject_m_behavior_order
        )
    payload = {
        "schema": MATERIALIZED_TARGET_QUERY_PROOF_SCHEMA,
        "status": "SYNTHETIC_IN_MEMORY_CONTRACT_PROOF_ONLY__NOT_TARGET_EXECUTION",
        "dataset": dataset,
        "view": view,
        "outer_fold_id": canonical["outer_fold_id"],
        "target_session_id": target_id,
        "canonical_development_target_authority_sha256": canonical[
            "development_target_query_authority_sha256"
        ],
        "canonical_metric_pointer_body_sha256": canonical["metric_pointer_validation"]["pointer_body_sha256"],
        "fixed_geometry_contract_sha256": _sha_json(canonical["fixed_canonical_geometry_contract"]),
        "target_support": {
            "sequence_semantics": "one_continuous_chronological_prefix",
            "support_trial_budget": budget,
            "support_raw_coverage": target_scaffold._compact_sorted_index_authority(
                support_indices, role="continuous_target_support_prefix_through_M50_or_M24",
            ),
            "support_stop_raw_bin": support_stop,
            "all_intervening_raw_rows_included": True,
            "rewarded_segments_concatenated": False,
            "dense_velocity_float32_raw_bytes_sha256": dense_sha,
            "dense_label_row_count": len(support_indices),
            "dense_label_scalar_count": len(support_indices) * 2,
            "dense_label_unique_row_count": observed_unique_rows,
            "information_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
        },
        "target_query": {
            "strict_post_M_only": True,
            "raw_coverage": target_scaffold._compact_sorted_index_authority(
                query_indices, role="strict_post_M_target_query_raw_coverage",
            ),
            "ordered_prediction_target_semantics": "valid_window_start_plus_49",
            "ordered_target_float32_raw_bytes_sha256": t4_sha,
            "t4_and_cebra_target_bytes_exactly_equal": True,
            "t4_reference_lineage": lineage,
            "receptive_field_and_target_byte_proof": proof,
            "subject_m_cross_view_behavior_order": shared_subject_m_behavior_order,
            "subject_m_sua_and_pseudo_mua_must_match_this_authority": dataset == "subject_m",
        },
        "target_neural_feature": {
            "target_feature_sha256": target_feature_sha256,
            "view": view,
            "pseudo_mua_pooling_provenance": pooling,
        },
        "readout_roles": _fixed_readout_roles(canonical),
        "standard_offset10_noncausal_disclosure": {
            "actual_offset_authority_sha256": vendored_model_alignment_authority_sha256,
            "previous_raw_bins_relative_to_prediction_target": 5,
            "right_extent_including_prediction_target_raw_bins": 5,
            "strictly_future_raw_bins_relative_to_prediction_target": 4,
            "receptive_field_width_raw_bins": 10,
            "half_open_offsets_relative_to_prediction_target": [-5, 5],
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "online_or_latency_equivalent_language_permitted": False,
        },
        "synthetic_fixture": True,
        "target_asset_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_trained": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
    return payload | {"materialized_target_query_proof_sha256": _sha_json(payload)}
