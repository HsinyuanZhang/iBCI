"""Root-audited metric-only pointer publisher and validator for Track-B v2.

This module is deliberately about *legacy receipt authority*, not target data.
It opens only the three explicitly named sidecarless baseline JSON bodies after
their immutable mode/SHA checks.  It does not discover NWB/NPZ data, import or
run CEBRA, construct an adapter, produce a score, or mint a pointer unless an
explicit root-authorised publisher is called by a later owner.

The canonical subject-M/RT baselines are multi-fold summaries.  The new pointer
may therefore authority the exact aggregate metric only.  Per-session×seed and
per-fold bodies are bound as lineage companions with
``metric_authority=false``; they may never be promoted to the headline metric
by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import track_b_v2_contract as base
import track_b_v2_live_contract as live


METRIC_POINTER_DRY_PLAN_SCHEMA = "track_b_v2_metric_pointer_dry_plan_v2"
METRIC_POINTER_VALIDATION_SCHEMA = "track_b_v2_metric_pointer_validation_v2"
RT_OUTER_FOLD_LINEAGE_PLAN_SCHEMA = "track_b_v2_rt_outer_fold_lineage_plan_v1"
ROOT_AUDIT_ATTESTATION_SCHEMA = "track_b_v2_root_metric_pointer_audit_attestation_v2"
ROOT_AUDIT_ATTESTATION_DRY_PLAN_SCHEMA = "track_b_v2_root_metric_pointer_audit_attestation_dry_plan_v2"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_POINTER_ROOT = _REPO_ROOT / "cebra_exploration" / "results" / "track_b_v2_root_metric_pointer_authority_v2"
_CANONICAL_POINTER_BODY_PATHS: dict[tuple[str, str | None], Path] = {
    ("subject_m", "sua"): _CANONICAL_POINTER_ROOT / "subject_m_sua_metric_pointer.json",
    ("subject_m", "pseudo_mua"): _CANONICAL_POINTER_ROOT / "subject_m_pseudo_mua_metric_pointer.json",
    ("rt", None): _CANONICAL_POINTER_ROOT / "rt_metric_pointer.json",
}
_CANONICAL_AUDIT_ATTESTATION_BODY_PATHS: dict[tuple[str, str | None], Path] = {
    ("subject_m", "sua"): _CANONICAL_POINTER_ROOT / "subject_m_sua_root_audit_attestation.json",
    ("subject_m", "pseudo_mua"): _CANONICAL_POINTER_ROOT / "subject_m_pseudo_mua_root_audit_attestation.json",
    ("rt", None): _CANONICAL_POINTER_ROOT / "rt_root_audit_attestation.json",
}


class TrackBV2MetricPointerAuthorityError(live.TrackBV2LiveContractError):
    """Raised before a root pointer may be treated as metric-only authority."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2MetricPointerAuthorityError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_metric_pointer_body_path(dataset: str, view: str | None = None) -> Path:
    """Return the single allowed official pointer body location for one scope."""
    dataset, view = base.validate_scope(dataset, view)
    return _CANONICAL_POINTER_BODY_PATHS[(dataset, view)]


def canonical_root_metric_pointer_audit_attestation_body_path(dataset: str, view: str | None = None) -> Path:
    """Return the single allowed immutable root-audit attestation location."""
    dataset, view = base.validate_scope(dataset, view)
    return _CANONICAL_AUDIT_ATTESTATION_BODY_PATHS[(dataset, view)]


def _require_exact_canonical_pair(
    *, pair: live.ExplicitSealedReceiptPair, dataset: str, view: str | None, role: str
) -> Path:
    """Reject relative aliases, copies, alternate filenames, and sidecar aliases."""
    if role == "canonical_reference_body_pointer":
        expected_body = canonical_metric_pointer_body_path(dataset, view)
    elif role == "root_metric_pointer_audit_attestation":
        expected_body = canonical_root_metric_pointer_audit_attestation_body_path(dataset, view)
    else:  # pragma: no cover - local callers use the two explicit roles above.
        raise TrackBV2MetricPointerAuthorityError(f"unsupported canonical pair role: {role}")
    expected_sidecar = expected_body.with_name(f"{expected_body.name}.sha256")
    require(pair.role == role, f"{role}: pair role drift")
    require(Path(pair.body_path) == expected_body,
            f"{role}: body path is not the one canonical output path for this scope")
    require(Path(pair.sidecar_path) == expected_sidecar,
            f"{role}: sidecar path is not the canonical body basename sidecar")
    return expected_body


def _ensure_canonical_output_parent(body: Path) -> None:
    """Create only a non-symlink canonical parent after checking its ancestry."""
    parent = Path(body).parent
    ancestor = parent
    while not os.path.lexists(ancestor):
        require(ancestor != ancestor.parent, "canonical output parent has no existing safe ancestor")
        ancestor = ancestor.parent
    info = ancestor.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "canonical output parent ancestry is not a real directory")
    parent.mkdir(parents=True, exist_ok=True)
    info = parent.lstat()
    require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode),
            "canonical output parent is symlinked or not a directory")
    require(parent.resolve() == parent, "canonical output parent resolves through an alias")


# These are the only legacy bodies this additive route may ever read.  They
# are sidecarless mode-0444 sealed history; the new pointer is the durable pair
# that binds their exact bytes without mutating them.
_SCOPE_SPECS: dict[tuple[str, str | None], dict[str, Any]] = {
    ("subject_m", "sua"): {
        "canonical": {
            "path": _REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json",
            "sha256": "12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2",
            "schema": "dandi_000688_subm_v9_t4_label_budget_torchmetrics151_v1",
            "metrics": {"t4_m50_sua_mean_r2": "/summary/sua/50/mean_r2"},
        },
        "lineage": {
            "per_session_seed_body": {
                "path": _REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805/aggregate/endpoint_aggregate_torchmetrics151.json",
                "sha256": "8a5ba373169cc28237666917f21fc893003afcc9ca486aa8f4b78d3f65aca7f4",
                "schema": "dandi_000688_subm_co_v9_torchmetrics151_authoritative_v1",
                "lineage_array_json_pointer": "/cells",
                "required_filter": {"arm": "shared_t4", "view": "sua"},
                "required_matching_cell_count": 45,
                "authority_scope": "M50_15_sessions_x_3_seeds_paired_distribution_lineage_only",
            },
        },
    },
    ("subject_m", "pseudo_mua"): {
        "canonical": {
            "path": _REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json",
            "sha256": "12c4aead244631ed55e5a3eae7f99c87c4cfc4131ac37aa1f84aa55e5e0d4cc2",
            "schema": "dandi_000688_subm_v9_t4_label_budget_torchmetrics151_v1",
            "metrics": {"t4_m50_pseudo_mua_mean_r2": "/summary/pseudo_mua/50/mean_r2"},
        },
        "lineage": {
            "per_session_seed_body": {
                "path": _REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805/aggregate/endpoint_aggregate_torchmetrics151.json",
                "sha256": "8a5ba373169cc28237666917f21fc893003afcc9ca486aa8f4b78d3f65aca7f4",
                "schema": "dandi_000688_subm_co_v9_torchmetrics151_authoritative_v1",
                "lineage_array_json_pointer": "/cells",
                "required_filter": {"arm": "shared_t4", "view": "pseudo_mua"},
                "required_matching_cell_count": 45,
                "authority_scope": "M50_15_sessions_x_3_seeds_paired_distribution_lineage_only",
            },
        },
    },
    ("rt", None): {
        "canonical": {
            "path": _REPO_ROOT / "sua_exploration/comparators/receipts/rt_classical_comparators/rt_classical_comparators_receipt.json",
            "sha256": "c51cb0ff7dadd3c40ca7861dea92d80f3457c709801ea8ec7ba3e91b9e52b042",
            "schema": "rt_classical_comparators_v1",
            "metrics": {"t4d_mean_r2": "/results/arms/t4d_reference/mean"},
        },
        "lineage": {
            "per_fold_t4d_body": {
                "path": _REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/rt_sparse_t4d_b2_forward_reeval_v2_20260811/RT_T4D_VS_B2_D1024_FORWARD_ONLY_15FOLD_FINAL_v1.json",
                "sha256": "c36ec0e31ed913ed4e8077f9a4d9d634d53529ce037ad06af1f48d279b16820e",
                "schema": "rt_t4d_vs_b2_d1024_forward_only_15fold_final_v1",
                "lineage_scalar_json_pointer": "/rows/*/t4d_r2",
                "required_scalar_count": 15,
                "authority_scope": "15_fold_T4d_per_session_lineage_only",
            },
            "stage2_delta_companion": {
                "path": _REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/STAGE2_MATRIX_AGGREGATE_v1.json",
                "sha256": "bb2806953e979180c408fb55744534be6fa470d4144f210cc50917a9b1006b7d",
                "schema": "rt_sparse_endpoint_stage2_matrix_aggregate_v1",
                "authority_scope": "T4d_minus_full_and_T4d_minus_zero4_diagnostics_only__not_absolute_metric",
            },
        },
    },
}


def _pointer_values(decoded: Any, pointer: str, *, label: str) -> tuple[float, ...]:
    require(isinstance(pointer, str) and pointer.startswith("/"), f"{label}: invalid JSON pointer")
    nodes: list[Any] = [decoded]
    for encoded in pointer.split("/")[1:]:
        token = encoded.replace("~1", "/").replace("~0", "~")
        next_nodes: list[Any] = []
        for node in nodes:
            if token == "*":
                require(isinstance(node, list), f"{label}: wildcard requires an array")
                next_nodes.extend(node)
            elif isinstance(node, Mapping):
                require(token in node, f"{label}: JSON pointer component missing: {token}")
                next_nodes.append(node[token])
            elif isinstance(node, list) and token.isdecimal():
                index = int(token)
                require(index < len(node), f"{label}: JSON pointer list index out of range")
                next_nodes.append(node[index])
            else:
                raise TrackBV2MetricPointerAuthorityError(f"{label}: JSON pointer traverses scalar")
        nodes = next_nodes
    require(nodes, f"{label}: JSON pointer selected no value")
    result: list[float] = []
    for value in nodes:
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                f"{label}: JSON pointer value is not finite numeric")
        result.append(float(value))
    return tuple(result)


def _read_legacy_spec(spec: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], bytes]:
    path = Path(spec["path"])
    raw = live._read_immutable_regular_same_fd(path, label=label)
    require(hashlib.sha256(raw).hexdigest() == spec["sha256"], f"{label}: sealed body SHA drift")
    # The decisive mode/type/path identity check was already performed by
    # _read_immutable_regular_same_fd on this exact opened descriptor.  Do not
    # follow it with an unrelated lstat whose inode could race after the read.
    require(not os.path.lexists(path.with_name(f"{path.name}.sha256")),
            f"{label}: legacy body unexpectedly acquired an adjacent sidecar")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrackBV2MetricPointerAuthorityError(f"{label}: legacy body is not UTF-8 JSON") from exc
    require(isinstance(decoded, dict), f"{label}: legacy body root must be an object")
    require(decoded.get("schema") == spec["schema"], f"{label}: legacy body schema drift")
    return decoded, raw


def _validate_subject_m_lineage_rows(rows: Any, *, view: str) -> dict[str, Any]:
    """Verify the complete 15-session × 3-seed subject-M M50 lineage."""
    require(isinstance(rows, list), "subject-M lineage cells must be an array")
    matching = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("arm") == "shared_t4" and row.get("view") == view
    ]
    require(len(matching) == 45, "subject-M lineage must contain exactly 45 filtered M50 cells")
    records: list[dict[str, Any]] = []
    for row in matching:
        session_id = row.get("session_id")
        seed = row.get("seed")
        r2 = row.get("r2")
        query_window_count = row.get("query_window_count")
        asset_id = row.get("asset_id")
        require(isinstance(session_id, str) and session_id, "subject-M lineage session ID invalid")
        require(isinstance(seed, int) and not isinstance(seed, bool), "subject-M lineage seed invalid")
        require(isinstance(r2, (int, float)) and not isinstance(r2, bool) and math.isfinite(float(r2)),
                "subject-M lineage r2 invalid")
        require(isinstance(query_window_count, int) and query_window_count > 0,
                "subject-M lineage query-window count invalid")
        require(isinstance(asset_id, str) and asset_id, "subject-M lineage asset ID invalid")
        records.append({
            "session_id": session_id,
            "seed": seed,
            "r2": float(r2),
            "query_window_count": query_window_count,
            "asset_id": asset_id,
        })
    records.sort(key=lambda item: (item["session_id"], item["seed"]))
    pairs = {(item["session_id"], item["seed"]) for item in records}
    sessions = {item["session_id"] for item in records}
    require(len(pairs) == 45, "subject-M lineage has duplicate session×seed cells")
    require(len(sessions) == 15, "subject-M lineage must contain 15 unique sessions")
    require({item["seed"] for item in records} == {42, 43, 44},
            "subject-M lineage seeds must be exactly {42,43,44}")
    require(all((session, seed) in pairs for session in sessions for seed in (42, 43, 44)),
            "subject-M lineage has missing session×seed cells")
    return {
        "matching_cell_count_from_same_fd_bytes": len(records),
        "unique_session_count": len(sessions),
        "seed_set": [42, 43, 44],
        "ordered_session_seed_r2_query_window_asset_sha256": _sha_json({
            "role": "subject_m_shared_t4_M50_session_seed_lineage",
            "records": records,
        }),
        "arithmetic_mean_r2_from_same_fd_bytes": sum(item["r2"] for item in records) / len(records),
    }


def _validate_rt_per_fold_rows(rows: Any) -> dict[str, Any]:
    """Verify RT's 15 unique folds/sessions without promoting them to a metric."""
    require(isinstance(rows, list) and len(rows) == 15, "RT lineage must contain exactly 15 fold rows")
    records: list[dict[str, Any]] = []
    for row in rows:
        require(isinstance(row, Mapping), "RT lineage row invalid")
        fold, session_id, r2 = row.get("fold"), row.get("session"), row.get("t4d_r2")
        require(isinstance(fold, int) and not isinstance(fold, bool), "RT lineage fold invalid")
        require(isinstance(session_id, str) and session_id, "RT lineage session ID invalid")
        require(isinstance(r2, (int, float)) and not isinstance(r2, bool) and math.isfinite(float(r2)),
                "RT lineage T4d R2 invalid")
        records.append({"fold": fold, "session_id": session_id, "t4d_r2": float(r2)})
    records.sort(key=lambda item: item["fold"])
    require([item["fold"] for item in records] == list(range(15)), "RT lineage folds must be exactly 0..14")
    require(len({item["session_id"] for item in records}) == 15, "RT lineage session IDs must be unique")
    return {
        "matching_scalar_count_from_same_fd_bytes": len(records),
        "folds": list(range(15)),
        "unique_session_count": 15,
        "ordered_fold_session_t4d_r2_sha256": _sha_json({
            "role": "rt_T4d_per_fold_lineage", "records": records,
        }),
        "arithmetic_mean_t4d_r2_from_same_fd_bytes": sum(item["t4d_r2"] for item in records) / len(records),
    }


def _lineage_binding(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    decoded, _raw = _read_legacy_spec(spec, label=f"legacy lineage {name}")
    result = {
        "sealed_body_path": str(Path(spec["path"]).resolve()),
        "sealed_body_sha256": spec["sha256"],
        "sealed_body_mode": "0444",
        "sealed_body_has_adjacent_sidecar": False,
        "sealed_body_schema": {"field": "schema", "value": spec["schema"]},
        "metric_authority": False,
        "authority_scope": spec["authority_scope"],
        "verified_from_same_fd_bytes": True,
    }
    if "lineage_array_json_pointer" in spec:
        pointer = spec["lineage_array_json_pointer"]
        # Array traversal is intentionally separate from metric extraction.
        current: Any = decoded
        for token in pointer.removeprefix("/").split("/"):
            require(isinstance(current, Mapping) and token in current,
                    f"legacy lineage {name}: array pointer drift")
            current = current[token]
        require(isinstance(current, list), f"legacy lineage {name}: expected array")
        audited = _validate_subject_m_lineage_rows(current, view=spec["required_filter"]["view"])
        require(audited["matching_cell_count_from_same_fd_bytes"] == spec["required_matching_cell_count"],
                f"legacy lineage {name}: required cell filter/count drift")
        result |= {
            "lineage_array_json_pointer": pointer,
            "required_filter": dict(spec["required_filter"]),
            "required_matching_cell_count": spec["required_matching_cell_count"],
            **audited,
        }
    if "lineage_scalar_json_pointer" in spec:
        values = _pointer_values(decoded, spec["lineage_scalar_json_pointer"], label=f"legacy lineage {name}")
        require(len(values) == spec["required_scalar_count"], f"legacy lineage {name}: scalar count drift")
        rows = decoded.get("rows")
        audited = _validate_rt_per_fold_rows(rows)
        require(tuple(item for item in values) == tuple(
            item["t4d_r2"] for item in sorted(rows, key=lambda row: row["fold"])
        ), "legacy lineage RT scalar pointer/order drift")
        result |= {
            "lineage_scalar_json_pointer": spec["lineage_scalar_json_pointer"],
            "required_scalar_count": spec["required_scalar_count"],
            "resolved_scalars_from_same_fd_bytes": list(values),
            **audited,
        }
    return result


def build_metric_pointer_dry_plan(dataset: str, view: str | None = None) -> dict[str, Any]:
    """Read only known legacy baseline bytes and render a non-writing plan."""
    dataset, view = base.validate_scope(dataset, view)
    spec = _SCOPE_SPECS[(dataset, view)]
    canonical, _raw = _read_legacy_spec(spec["canonical"], label="legacy canonical metric body")
    metrics = {
        name: _pointer_values(canonical, pointer, label=f"legacy canonical metric {name}")[0]
        for name, pointer in spec["canonical"]["metrics"].items()
    }
    lineage = {
        name: _lineage_binding(name, lineage_spec)
        for name, lineage_spec in spec["lineage"].items()
    }
    if dataset == "subject_m":
        lineage_mean = lineage["per_session_seed_body"]["arithmetic_mean_r2_from_same_fd_bytes"]
        require(next(iter(metrics.values())) == lineage_mean,
                "subject-M summary metric does not exactly equal filtered 15x3 lineage mean")
    else:
        lineage_mean = lineage["per_fold_t4d_body"]["arithmetic_mean_t4d_r2_from_same_fd_bytes"]
        require(next(iter(metrics.values())) == lineage_mean,
                "RT aggregate metric does not exactly equal ordered per-fold lineage mean")
    pointer_template = {
        "schema": live.SEALED_BODY_POINTER_SCHEMA,
        "status": "ROOT_AUDITED_POINTER__NOT_A_RESULT",
        "dataset": dataset,
        "view": view,
        "pointer_role": "canonical_reference_terminal",
        "authority_scope": "aggregate_metric_reference_only__not_live_fold_lineage",
        "sealed_body_path": str(Path(spec["canonical"]["path"]).resolve()),
        "sealed_body_sha256": spec["canonical"]["sha256"],
        "sealed_body_mode": "0444",
        "sealed_body_has_adjacent_sidecar": False,
        "sealed_body_schema": {"field": "schema", "value": spec["canonical"]["schema"]},
        "reference_metric_json_pointers": dict(spec["canonical"]["metrics"]),
        "metric_values_resolved_from_same_fd_bytes": metrics,
        "lineage_companions": lineage,
        "rounded_literals_accepted": False,
        "legacy_bodies_modified": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
    return {
        "schema": METRIC_POINTER_DRY_PLAN_SCHEMA,
        "status": "DRY_PLAN_ONLY__ROOT_AUDIT_AND_EXPLICIT_MINT_REQUIRED",
        "dataset": dataset,
        "view": view,
        "pointer_payload_template": pointer_template,
        "pointer_payload_template_sha256": _sha_json(pointer_template),
        "root_audit_attestation_pair_required": True,
        "canonical_pointer_body_path": str(canonical_metric_pointer_body_path(dataset, view)),
        "canonical_root_audit_attestation_body_path": str(
            canonical_root_metric_pointer_audit_attestation_body_path(dataset, view)
        ),
        "official_pointer_pair_present": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_receipt_minted": False,
    }


def build_rt_outer_fold_lineage_plan() -> dict[str, Any]:
    """Render the exact 15-fold RT source/held-out topology without data access.

    RT's reference aggregate is an average over 15 outer folds.  A later
    source-only authority therefore cannot use one convenient two-session
    smoke roster for every target.  This plan derives all outer folds from the
    same-FD verified per-fold lineage body, but deliberately does *not* expose
    any per-fold R² values: those remain a metric-lineage companion, not an
    input to source construction or geometry selection.

    The held-out session name is an opaque identifier only.  This function
    never discovers its path, opens its data, or passes it to a loader.
    """
    dry_plan = build_metric_pointer_dry_plan("rt", None)
    lineage_spec = _SCOPE_SPECS[("rt", None)]["lineage"]["per_fold_t4d_body"]
    decoded, _raw = _read_legacy_spec(lineage_spec, label="RT outer-fold lineage body")
    _validate_rt_per_fold_rows(decoded.get("rows"))
    rows = decoded.get("rows")
    require(isinstance(rows, list), "RT outer-fold lineage rows missing")
    # Reconstruct only the ordered fold/session topology from verified bytes.
    records = sorted(
        (
            {"fold": row["fold"], "session_id": row["session"]}
            for row in rows
            if isinstance(row, Mapping)
        ),
        key=lambda item: item["fold"],
    )
    require([item["fold"] for item in records] == list(range(15)),
            "RT outer-fold topology must cover folds 0..14")
    sessions = tuple(item["session_id"] for item in records)
    require(len(set(sessions)) == 15, "RT outer-fold topology requires 15 unique sessions")
    lineage = dry_plan["pointer_payload_template"]["lineage_companions"]["per_fold_t4d_body"]
    folds: list[dict[str, Any]] = []
    for item in records:
        held_out = item["session_id"]
        source_ids = tuple(sorted(session for session in sessions if session != held_out))
        require(len(source_ids) == 14 and held_out not in source_ids,
                "RT outer-fold source roster must contain exactly the other 14 sessions")
        folds.append({
            "outer_fold_id": f"rt_outer_fold_{item['fold']:02d}",
            "outer_fold_index": item["fold"],
            "opaque_held_out_target_session_id": held_out,
            "held_out_target_identifier_origin": "same_fd_verified_RT_T4d_per_fold_lineage_rows_fold_session",
            "source_session_ids": list(source_ids),
            "source_session_count": 14,
            "held_out_target_data_discovered": False,
            "held_out_target_data_opened": False,
            "held_out_target_passed_to_source_loader": False,
            "target_support_budget_trials": 24,
            "target_support_semantics": "chronological_first_24_trials__M24",
            "target_query_semantics": "sealed_RT_outer_q24_eligible_full_windows_only",
            "standard_cebra_support_sequence": {
                "sequence_semantics": "one_continuous_chronological_prefix",
                "start": "CANONICAL_TARGET_RECORD_START_RAW_BIN",
                "stop": "STOP_OF_CHRONOLOGICAL_TRIAL_24",
                "all_intervening_raw_rows_included": True,
                "rewarded_segments_concatenated": False,
                "boundary_matched": True,
                "trial_bin_or_neural_exposure_matched": False,
                "bias_direction": "favors_CEBRA_accuracy",
            },
        })
    payload = {
        "schema": RT_OUTER_FOLD_LINEAGE_PLAN_SCHEMA,
        "status": "RT_15_OUTER_FOLD_TOPOLOGY_ONLY__NO_TARGET_DISCOVERY__NO_CEBRA_NO_SCORE",
        "dataset": "rt",
        "view": None,
        "reference_per_fold_lineage_body_sha256": lineage["sealed_body_sha256"],
        "reference_per_fold_lineage_ordered_digest": lineage["ordered_fold_session_t4d_r2_sha256"],
        "reference_per_fold_lineage_metric_authority": False,
        "reference_per_fold_lineage_scalar_values_exposed": False,
        "outer_fold_count": 15,
        "outer_folds": folds,
        "all_rt_session_ids_sha256": _sha_json({"role": "rt_15_outer_fold_session_roster", "ids": list(sessions)}),
        "target_data_discovered": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_receipt_minted": False,
    }
    return payload | {"rt_outer_fold_lineage_plan_sha256": _sha_json(payload)}


def validate_rt_outer_fold_lineage_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and compare the RT plan, rejecting caller-supplied rosters."""
    require(isinstance(payload, Mapping), "RT outer-fold lineage plan must be a mapping")
    expected = build_rt_outer_fold_lineage_plan()
    require(dict(payload) == expected, "RT outer-fold lineage plan differs from same-FD verified topology")
    return expected


def _require_metric_pointer_dry_plan(dry_plan: Mapping[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    """Validate the common non-writing plan shape before an audit/mint step."""
    require(isinstance(dry_plan, Mapping), "metric pointer dry plan must be a mapping")
    require(dry_plan.get("schema") == METRIC_POINTER_DRY_PLAN_SCHEMA, "metric pointer dry-plan schema drift")
    require(dry_plan.get("status") == "DRY_PLAN_ONLY__ROOT_AUDIT_AND_EXPLICIT_MINT_REQUIRED",
            "metric pointer dry-plan status drift")
    dataset, view = base.validate_scope(dry_plan.get("dataset"), dry_plan.get("view"))
    template = dry_plan.get("pointer_payload_template")
    require(isinstance(template, Mapping), "metric pointer dry-plan template missing")
    expected = build_metric_pointer_dry_plan(dataset, view)
    require(dict(dry_plan) == expected, "metric pointer dry plan differs from fresh same-FD verified plan")
    return dataset, view, dict(template)


def _current_code_and_test_evidence() -> dict[str, Any]:
    """Bind the exact code and focused tests that root reviewed before minting."""
    paths = {
        "metric_pointer_authority": _REPO_ROOT / "cebra_exploration/src/track_b_v2_metric_pointer_authority.py",
        "live_contract": _REPO_ROOT / "cebra_exploration/src/track_b_v2_live_contract.py",
        "target_query_scaffold": _REPO_ROOT / "cebra_exploration/src/track_b_v2_target_query_scaffold.py",
        "metric_pointer_tests": _REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_metric_pointer_authority.py",
        "target_query_tests": _REPO_ROOT / "cebra_exploration/tests/test_track_b_v2_target_query_scaffold.py",
    }
    require(all(path.is_file() and not path.is_symlink() for path in paths.values()),
            "root-audit code/test evidence file is missing or symlinked")
    return {
        "source_sha256_by_route_file": {
            str(path.relative_to(_REPO_ROOT)): _sha_file(path) for path in paths.values()
        },
        "focused_test_command": (
            "PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            "PYTHONPATH=cebra_exploration/src /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest -q "
            "cebra_exploration/tests/test_track_b_v2_contract.py "
            "cebra_exploration/tests/test_track_b_v2_live_contract.py "
            "cebra_exploration/tests/test_track_b_v2_source_adapter.py "
            "cebra_exploration/tests/test_track_b_v2_metric_pointer_authority.py "
            "cebra_exploration/tests/test_track_b_v2_target_query_scaffold.py"
        ),
        "focused_test_expected_result": "PASS__ROOT_MUST_RUN_AND_REVIEW_BEFORE_PUBLISH",
        "dry_plan_command": "python cebra_exploration/scripts/run_track_b_v2_metric_pointer_dry_plan.py --dataset SCOPE [--view VIEW]",
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }


def render_root_metric_pointer_audit_attestation_payload(*, dry_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Render the exact immutable root-audit attestation body; never write it.

    The body itself—not a caller-supplied hexadecimal literal—is the authority
    that root reviewed the exact fresh plan, current code, and focused tests.
    Its canonical path is scoped and later verified as a strict 0444 pair.
    """
    dataset, view, template = _require_metric_pointer_dry_plan(dry_plan)
    return {
        "schema": ROOT_AUDIT_ATTESTATION_SCHEMA,
        "status": "ROOT_AUDIT_COMPLETE__CANONICAL_METRIC_POINTER_PUBLISH_AUTHORIZED",
        "attestation_role": "root_metric_pointer_audit_attestation",
        "dataset": dataset,
        "view": view,
        "canonical_metric_pointer_body_path": str(canonical_metric_pointer_body_path(dataset, view)),
        "root_authorizes_canonical_pointer_publish": True,
        "dry_plan_evidence": {
            "dry_plan_schema": METRIC_POINTER_DRY_PLAN_SCHEMA,
            "dry_plan_sha256": _sha_json(dry_plan),
            "pointer_payload_template_sha256": dry_plan["pointer_payload_template_sha256"],
            "canonical_reference_body_sha256": template["sealed_body_sha256"],
            "metric_values_sha256": _sha_json(template["metric_values_resolved_from_same_fd_bytes"]),
            "lineage_companions_sha256": _sha_json(template["lineage_companions"]),
        },
        "code_and_test_evidence": _current_code_and_test_evidence(),
        "legacy_bodies_modified": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_metric_pointer_published_by_this_attestation": False,
    }


def build_root_metric_pointer_audit_attestation_dry_plan(*, dry_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a no-write root attestation plan with a canonical output path."""
    dataset, view, _template = _require_metric_pointer_dry_plan(dry_plan)
    payload = render_root_metric_pointer_audit_attestation_payload(dry_plan=dry_plan)
    return {
        "schema": ROOT_AUDIT_ATTESTATION_DRY_PLAN_SCHEMA,
        "status": "DRY_PLAN_ONLY__ROOT_AUDIT_ATTESTATION_MINT_REQUIRED",
        "dataset": dataset,
        "view": view,
        "canonical_output_body_path": str(canonical_root_metric_pointer_audit_attestation_body_path(dataset, view)),
        "attestation_payload_template": payload,
        "attestation_payload_template_sha256": _sha_json(payload),
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_receipt_minted": False,
    }


def publish_root_metric_pointer_audit_attestation_pair(
    *,
    dry_plan: Mapping[str, Any],
    output_body_path: Path | None = None,
    root_authorized_publish: bool = False,
) -> live.ExplicitSealedReceiptPair:
    """Root-only O_EXCL/0444 mint for the canonical audit-attestation pair."""
    require(root_authorized_publish is True,
            "refusing to mint root audit attestation without explicit root_authorized_publish=True")
    dataset, view, _template = _require_metric_pointer_dry_plan(dry_plan)
    body = canonical_root_metric_pointer_audit_attestation_body_path(dataset, view)
    if output_body_path is not None:
        require(Path(output_body_path) == body,
                "root audit attestation output path is not the canonical path for this scope")
    pair = live.ExplicitSealedReceiptPair(
        role="root_metric_pointer_audit_attestation",
        body_path=body,
        sidecar_path=body.with_name(f"{body.name}.sha256"),
    )
    _require_exact_canonical_pair(pair=pair, dataset=dataset, view=view, role=pair.role)
    require(not os.path.lexists(body) and not os.path.lexists(pair.sidecar_path),
            "root audit attestation canonical output pair already exists or is symlinked")
    _ensure_canonical_output_parent(body)
    base.write_immutable_receipt(body, render_root_metric_pointer_audit_attestation_payload(dry_plan=dry_plan))
    return pair


def validate_root_metric_pointer_audit_attestation_pair(
    *, dry_plan: Mapping[str, Any], attestation_pair: live.ExplicitSealedReceiptPair
) -> dict[str, Any]:
    """Strictly validate root's immutable audit body+sidecar before pointer use."""
    dataset, view, _template = _require_metric_pointer_dry_plan(dry_plan)
    body = _require_exact_canonical_pair(
        pair=attestation_pair, dataset=dataset, view=view, role="root_metric_pointer_audit_attestation",
    )
    payload, body_sha = live._strict_readonly_pair(attestation_pair)
    expected = render_root_metric_pointer_audit_attestation_payload(dry_plan=dry_plan)
    require(payload == expected,
            "root audit attestation payload differs from fresh scope/template/code/test/dry-plan evidence")
    return {
        "attestation_role": attestation_pair.role,
        "body_path": str(body),
        "body_sha256": body_sha,
        "sidecar_path": str(attestation_pair.sidecar_path),
        "schema": ROOT_AUDIT_ATTESTATION_SCHEMA,
        "root_authorizes_canonical_pointer_publish": True,
    }


def render_root_audited_metric_pointer_payload(
    *, dry_plan: Mapping[str, Any], root_audit_attestation_pair: live.ExplicitSealedReceiptPair
) -> dict[str, Any]:
    """Bind a verified immutable audit pair into the canonical pointer body."""
    dataset, view, template = _require_metric_pointer_dry_plan(dry_plan)
    attestation = validate_root_metric_pointer_audit_attestation_pair(
        dry_plan=dry_plan, attestation_pair=root_audit_attestation_pair,
    )
    payload = dict(template)
    payload["canonical_pointer_body_path"] = str(canonical_metric_pointer_body_path(dataset, view))
    payload["root_metric_pointer_audit_attestation"] = attestation
    payload["root_audit_template_sha256"] = dry_plan["pointer_payload_template_sha256"]
    return payload


def publish_root_audited_metric_pointer_pair(
    *,
    dry_plan: Mapping[str, Any],
    root_audit_attestation_pair: live.ExplicitSealedReceiptPair,
    output_body_path: Path | None = None,
    root_authorized_publish: bool = False,
) -> live.ExplicitSealedReceiptPair:
    """Root-only O_EXCL/0444 mint at the one canonical pointer path per scope."""
    require(root_authorized_publish is True,
            "refusing to mint a root metric pointer without explicit root_authorized_publish=True")
    dataset, view, _template = _require_metric_pointer_dry_plan(dry_plan)
    body = canonical_metric_pointer_body_path(dataset, view)
    if output_body_path is not None:
        require(Path(output_body_path) == body,
                "root metric pointer output path is not the canonical path for this scope")
    pair = live.ExplicitSealedReceiptPair(
        role="canonical_reference_body_pointer",
        body_path=body,
        sidecar_path=body.with_name(f"{body.name}.sha256"),
    )
    _require_exact_canonical_pair(pair=pair, dataset=dataset, view=view, role=pair.role)
    require(not os.path.lexists(body) and not os.path.lexists(pair.sidecar_path),
            "root metric pointer canonical output pair already exists or is symlinked")
    payload = render_root_audited_metric_pointer_payload(
        dry_plan=dry_plan, root_audit_attestation_pair=root_audit_attestation_pair,
    )
    _ensure_canonical_output_parent(body)
    base.write_immutable_receipt(body, payload)
    return pair


def validate_root_audited_metric_pointer_pair(
    *, dataset: str, view: str | None, pointer_pair: live.ExplicitSealedReceiptPair
) -> dict[str, Any]:
    """Validate the one canonical pointer against attestation and fresh legacy bytes."""
    dataset, view = base.validate_scope(dataset, view)
    body = _require_exact_canonical_pair(
        pair=pointer_pair, dataset=dataset, view=view, role="canonical_reference_body_pointer",
    )
    payload, pointer_sha = live._strict_readonly_pair(pointer_pair)
    dry_plan = build_metric_pointer_dry_plan(dataset, view)
    attestation_pair = live.ExplicitSealedReceiptPair(
        role="root_metric_pointer_audit_attestation",
        body_path=canonical_root_metric_pointer_audit_attestation_body_path(dataset, view),
        sidecar_path=canonical_root_metric_pointer_audit_attestation_body_path(dataset, view).with_name(
            f"{canonical_root_metric_pointer_audit_attestation_body_path(dataset, view).name}.sha256"
        ),
    )
    expected = render_root_audited_metric_pointer_payload(
        dry_plan=dry_plan, root_audit_attestation_pair=attestation_pair,
    )
    require(payload == expected, "root metric pointer payload differs from same-FD verified dry plan/attestation")
    # Reuse the existing pointer implementation too: this proves future
    # consumers resolve the same scalar metrics from the pointed sealed bytes.
    adapter = live.canonical_adapter_spec(dataset, view)
    authority = live.build_canonical_metric_reference_authority_from_sealed_pointer(
        adapter=adapter, canonical_reference_body_pointer=pointer_pair,
    )
    expected_metrics = expected["metric_values_resolved_from_same_fd_bytes"]
    require(authority["reference_metrics"] == expected_metrics,
            "existing pointer consumer metric resolution differs from dry plan")
    lineage = expected["lineage_companions"]
    require(all(item.get("metric_authority") is False for item in lineage.values()),
            "lineage companions must never claim metric authority")
    return {
        "schema": METRIC_POINTER_VALIDATION_SCHEMA,
        "status": "ROOT_AUDITED_IMMUTABLE_PAIR_VALIDATED__METRIC_ONLY__NOT_A_RESULT",
        "dataset": dataset,
        "view": view,
        "pointer_body_sha256": pointer_sha,
        "pointer_path": str(body),
        "root_metric_pointer_audit_attestation": expected["root_metric_pointer_audit_attestation"],
        "canonical_reference_body_sha256": authority["canonical_reference_receipt_sha256"],
        "reference_metrics": authority["reference_metrics"],
        "lineage_companion_body_sha256s": {
            name: item["sealed_body_sha256"] for name, item in lineage.items()
        },
        "metric_authority_scope": "aggregate_metric_reference_only__not_live_fold_lineage",
        "lineage_companions_are_not_metric_authority": True,
        "target_data_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "gpu_used": False,
        "score_emitted": False,
    }
