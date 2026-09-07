"""Publish consumer-specific runtime controls from audited synthetic evidence.

Default use is a no-write dry plan.  This module imports neither Torch nor
CEBRA and contains no source/target/NWB/NPZ loader.  A future root-reviewed
publication writes the incompatible Subject-M and RT contracts as one
transactional four-file set (two JSON bodies and two standard sidecars).
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Mapping, Sequence

import track_b_v2_actual_cpu_route as route
import track_b_v2_post_cost_synthetic_controls_v2 as synthetic
import track_b_v2_rt_one_cell_live_executor as rt_consumer
import track_b_v2_source_adapter as source
import track_b_v2_subject_m_stagep_runtime as subject_consumer


SCHEMA_DRY_PLAN = "track_b_v2_post_synthetic_runtime_control_authority_dry_plan_v1"
STATUS_DRY_PLAN = "DRY_PLAN_PASS__TWO_CONSUMER_PAYLOADS_READY__NOT_MINTED__NO_TARGET"
EVIDENCE_SCHEMA = "track_b_v2_post_synthetic_raw_bound_control_evidence_v1"
POSITIVE_THRESHOLD_R2 = 0.70
DERANGED_HARD_NULL_THRESHOLD_R2 = 0.60
POSITIVE_THRESHOLD_ORIGIN = "PREDECLARED_BEFORE_SYNTHETIC_V2_TERMINAL"
HARD_NULL_THRESHOLD_ORIGIN = (
    "ROOT_FROZEN_AFTER_IMMUTABLE_SYNTHETIC_V2_TERMINAL_BEFORE_REAL_TARGET"
)
EXPECTED_COST_SHA256 = "ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2"
EXPECTED_TERMINAL_SHA256 = "99269afd2770ac333b5a529768920a1adbd4be560dcd942839697edab2c14de8"
EXPECTED_TERMINAL_CLOSURE_SHA256 = "025a34913925973cab6faf255ae44c5f008bb4646da5237f4bb4a0571316affe"
EXPECTED_PERMUTATION_SHA256 = "b101d5fb8d0d7b4703a0df87377253c055f653e970e799de52b733f7250a9444"
SETTLED_FILES = {
    "v2_core": "56226348b110988f227311703cbcba8495a2e703da1058ad77d98fa009997fd7",
    "v2_cli": "4efc13b8836138e8599ac0958e27f098183e03f90e7ec93c3cd1c984ec038d88",
    "v2_focused_tests": "2b5800b9eb82e6060776b81ae3277f974961a98454a4c15d68cdccae39c9f0fe",
}
POSITIVE_ARMS = ("cebra_joint_behavior", "cebra_frozen_source_adapt")
UNALIGNED_ARM = "cebra_adapt_unaligned"
DERANGED_ARM = "cebra_joint_behavior__target_support_auxiliary_rows_deranged"
DECODERS = ("linear_ridge", "knn_cosine_k3")
HEADLINE_ROUTE = "target_support_only_standard_cebra_accuracy"

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "cebra_exploration/scripts/publish_track_b_v2_post_synthetic_runtime_controls.py"
TEST = ROOT / "cebra_exploration/tests/test_track_b_v2_post_synthetic_runtime_control_authority.py"
DOC = ROOT / "cebra_exploration/docs/TRACK_B_V2_POST_SYNTHETIC_RUNTIME_CONTROL_AUTHORITY.md"
TERMINAL_BODY = synthetic.CANONICAL_OUTPUT
SUBJECT_BODY = subject_consumer.RUNTIME_CONTROL_PATH
RT_BODY = rt_consumer.RUNTIME_CONTROL_BODY


class RuntimeControlAuthorityError(RuntimeError):
    """Fail-closed authority publisher error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeControlAuthorityError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(route.canonical_json_bytes(value)).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def _assert_canonical_outputs_fresh() -> None:
    require(SUBJECT_BODY == subject_consumer.RUNTIME_CONTROL_PATH and
            RT_BODY == rt_consumer.RUNTIME_CONTROL_BODY,
            "consumer canonical runtime-control path alias drift")
    for label, body in (("Subject-M", SUBJECT_BODY), ("RT", RT_BODY)):
        require(body.is_absolute(), f"{label} output is not absolute")
        require(not os.path.lexists(body) and not os.path.lexists(_sidecar(body)),
                f"{label} canonical body/sidecar must both be fresh")


def implementation_closure() -> dict[str, Any]:
    paths = {
        "publisher_core": Path(__file__).absolute(),
        "publisher_cli": CLI,
        "publisher_tests": TEST,
        "publisher_protocol_note": DOC,
        "subject_m_consumer": Path(subject_consumer.__file__).absolute(),
        "subject_m_consumer_tests": ROOT / "cebra_exploration/tests/test_track_b_v2_subject_m_stagep_runtime.py",
        "rt_consumer": Path(rt_consumer.__file__).absolute(),
        "rt_consumer_tests": ROOT / "cebra_exploration/tests/test_track_b_v2_rt_one_cell_live_executor.py",
        "synthetic_v2_core": Path(synthetic.__file__).absolute(),
        "synthetic_v2_cli": synthetic.CLI,
        "synthetic_v2_tests": synthetic.TEST,
        "immutable_pair_implementation": Path(route.__file__).absolute(),
        "same_fd_reader": Path(source.__file__).absolute(),
    }
    rows = {}
    for role, path in paths.items():
        raw = source._read_regular_file(path, label=f"runtime-control authority {role}")
        rows[role] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    require(rows["synthetic_v2_core"]["sha256"] == SETTLED_FILES["v2_core"] and
            rows["synthetic_v2_cli"]["sha256"] == SETTLED_FILES["v2_cli"] and
            rows["synthetic_v2_tests"]["sha256"] == SETTLED_FILES["v2_focused_tests"],
            "settled synthetic-v2 implementation SHA drift")
    return {"files": rows, "closure_sha256": _sha_json(rows)}


def _group(rows: Sequence[Mapping[str, Any]], *, arm: str, decoder: str,
           role: str) -> dict[str, Any]:
    selected = sorted((row for row in rows if row["arm"] == arm and row["decoder"] == decoder),
                      key=lambda row: int(row["seed"]))
    values = [float(row["target_query_r2"]) for row in selected]
    require([row["seed"] for row in selected] == list(range(8)) and len(values) == 8,
            f"raw group coverage drift: {arm}/{decoder}")
    threshold = (POSITIVE_THRESHOLD_R2 if role == "positive" else
                 DERANGED_HARD_NULL_THRESHOLD_R2 if role == "deranged_hard_null" else None)
    strict_count = sum(value > threshold for value in values) if role == "positive" else (
        sum(value < threshold for value in values) if role == "deranged_hard_null" else 0)
    return {
        "arm": arm,
        "decoder": decoder,
        "role": role,
        "seed_order": list(range(8)),
        "values_by_seed": values,
        "count": 8,
        "min": min(values),
        "max": max(values),
        "decision_threshold_r2": threshold,
        "strict_relation": ("greater_than" if role == "positive" else
                            "less_than" if role == "deranged_hard_null" else "not_gated"),
        "strict_pass_count": strict_count if role != "diagnostic_non_gating" else None,
        "all_eight_strictly_pass": strict_count == 8 if role != "diagnostic_non_gating" else None,
    }


def load_raw_bound_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read canonical inputs once and independently recompute all threshold predicates."""
    _assert_canonical_outputs_fresh()
    cost = synthetic.validate_cost_pair()
    require(cost.get("canonical_body_sha256") == EXPECTED_COST_SHA256,
            "fixed-cost canonical SHA drift")
    terminal, terminal_binding = route.load_immutable_json(
        TERMINAL_BODY, expected_schema=synthetic.SCHEMA_RECEIPT)
    require(terminal_binding.get("sha256") == EXPECTED_TERMINAL_SHA256 and
            terminal_binding.get("mode") == "0444" and
            terminal_binding.get("read_once_from_verified_fd") is True,
            "synthetic-v2 terminal immutable binding drift")
    synthetic.validate_measurement_receipt(terminal)
    launch_terminal_closure = terminal.get("implementation_closure_at_launch")
    final_terminal_closure = terminal.get("implementation_closure_at_final")
    require(isinstance(launch_terminal_closure, Mapping) and
            launch_terminal_closure == final_terminal_closure and
            launch_terminal_closure.get("complete_closure_sha256") == EXPECTED_TERMINAL_CLOSURE_SHA256 and
            terminal.get("launch_final_closure_exact_equal") is True,
            "synthetic-v2 terminal launch/final closure drift")
    # The shared H1-excluded protocol legitimately gained a later additive
    # section after this terminal run.  Do not rewrite history by treating that
    # successor-doc change as terminal execution drift.  Instead, verify the
    # immutable launch/final closure recorded in the terminal and independently
    # revalidate the settled v2 core/CLI/test bytes below.
    fixed = launch_terminal_closure["fixed_files"]
    require({role: fixed[role]["sha256"] for role in SETTLED_FILES} == SETTLED_FILES,
            "synthetic-v2 settled file map drift")
    require(terminal.get("arm_run_count") == 32 and terminal.get("cebra_fit_call_count") == 56 and
            terminal.get("decoder_measurement_count") == 192,
            "synthetic-v2 execution topology drift")
    require(terminal.get("derangement_authority", {}).get("permutation_int64_sha256") ==
            EXPECTED_PERMUTATION_SHA256,
            "synthetic-v2 permutation authority drift")
    all_rows = terminal.get("measurements")
    require(isinstance(all_rows, list) and len(all_rows) == 192 and
            all(row.get("query_neural_or_auxiliary_in_fit") is False for row in all_rows),
            "synthetic-v2 measurement/query-fit coverage drift")
    headline = [row for row in all_rows if row.get("readout_route") == HEADLINE_ROUTE]
    require(len(headline) == 64, "target-support-only raw evidence count drift")
    raw_rows = [{
        "seed": int(row["seed"]),
        "arm": row["arm"],
        "decoder": row["decoder"],
        "readout_route": row["readout_route"],
        "target_query_r2": float(row["target_query_r2"]),
        "query_neural_or_auxiliary_in_fit": row["query_neural_or_auxiliary_in_fit"],
        "prediction_float32_sha256": row["metric_runtime"]["prediction_float32_sha256"],
        "target_float32_sha256": row["metric_runtime"]["target_float32_sha256"],
    } for row in sorted(headline, key=lambda row: (int(row["seed"]), row["arm"], row["decoder"]))]
    groups = {}
    for arm in POSITIVE_ARMS:
        for decoder in DECODERS:
            key = f"{arm}__{decoder}"
            groups[key] = _group(raw_rows, arm=arm, decoder=decoder, role="positive")
            require(groups[key]["strict_pass_count"] == 8,
                    f"positive group is not 8/8 strictly above predeclared 0.70: {key}")
    for decoder in DECODERS:
        key = f"{DERANGED_ARM}__{decoder}"
        groups[key] = _group(raw_rows, arm=DERANGED_ARM, decoder=decoder, role="deranged_hard_null")
        require(groups[key]["strict_pass_count"] == 8,
                f"deranged group is not 8/8 strictly below predeclared 0.70: {key}")
        key = f"{UNALIGNED_ARM}__{decoder}"
        groups[key] = _group(raw_rows, arm=UNALIGNED_ARM, decoder=decoder, role="diagnostic_non_gating")
    complete_grid_fingerprint = [{
        "seed": row["seed"], "arm": row["arm"], "readout_route": row["readout_route"],
        "decoder": row["decoder"], "target_query_r2": row["target_query_r2"],
        "prediction_float32_sha256": row["metric_runtime"]["prediction_float32_sha256"],
        "target_float32_sha256": row["metric_runtime"]["target_float32_sha256"],
    } for row in sorted(all_rows, key=lambda row: (
        int(row["seed"]), row["arm"], row["readout_route"], row["decoder"]))]
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "fixed_cost_body_sha256": EXPECTED_COST_SHA256,
        "synthetic_v2_terminal_path": str(TERMINAL_BODY),
        "synthetic_v2_terminal_body_sha256": EXPECTED_TERMINAL_SHA256,
        "synthetic_v2_terminal_sidecar_sha256": terminal_binding["sidecar_sha256"],
        "synthetic_v2_terminal_closure_sha256": EXPECTED_TERMINAL_CLOSURE_SHA256,
        "settled_synthetic_v2_file_sha256": dict(SETTLED_FILES),
        "fixed_geometry": {"output_dimension": 8, "iterations": 10_000,
                           "linear_ridge_normalized_lambda": 0.01, "cosine_knn_k": 3},
        "positive_control_threshold_r2": POSITIVE_THRESHOLD_R2,
        "positive_threshold_origin": POSITIVE_THRESHOLD_ORIGIN,
        "deranged_hard_null_threshold_r2": DERANGED_HARD_NULL_THRESHOLD_R2,
        "hard_null_threshold_origin": HARD_NULL_THRESHOLD_ORIGIN,
        "hard_null_threshold_selection": (
            "ROOT_REVIEWED_ROUND_0.60_ABOVE_OBSERVED_MAX_0.5055904984474182"
            "__BELOW_PREDECLARED_POSITIVE_0.70__NOT_TERMINAL_MIDPOINT"
        ),
        "hard_null_threshold_may_be_relaxed_or_backfilled": False,
        "comparison": "strict_greater_positive_threshold__strict_less_hard_null_threshold",
        "permutation_authority_sha256": EXPECTED_PERMUTATION_SHA256,
        "arm_run_count": 32,
        "cebra_fit_call_count": 56,
        "decoder_measurement_count": 192,
        "target_support_only_measurement_count": 64,
        "target_support_only_raw_measurements": raw_rows,
        "target_support_only_groups": groups,
        "complete_192_measurement_grid_sha256": _sha_json(complete_grid_fingerprint),
        "ordinary_unaligned_distribution_reported": True,
        "ordinary_unaligned_role": "diagnostic_distribution_only__not_a_hard_null_gate",
        "no_query_neural_or_auxiliary_in_any_fit": True,
        "target_data_discovered": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
    }
    evidence["raw_bound_evidence_sha256"] = _sha_json(evidence)
    return evidence, {"cost": cost, "terminal": terminal_binding}


def _pair_set(*, evidence_sha256: str) -> dict[str, Any]:
    descriptor = {
        "schema": "track_b_v2_post_synthetic_paired_runtime_control_set_v1",
        "shared_raw_bound_evidence_sha256": evidence_sha256,
        "positive_control_threshold_r2": POSITIVE_THRESHOLD_R2,
        "deranged_hard_null_threshold_r2": DERANGED_HARD_NULL_THRESHOLD_R2,
        "subject_m": {"path": str(SUBJECT_BODY),
                      "schema": "track_b_v2_subject_m_fixed_runtime_control_hard_null_v1",
                      "status": "ROOT_REVIEWED_FIXED_GEOMETRY_RUNTIME_CONTROLS_PASS"},
        "rt": {"path": str(RT_BODY), "schema": rt_consumer.RUNTIME_CONTROL_SCHEMA,
               "status": rt_consumer.RUNTIME_CONTROL_STATUS},
        "both_pairs_required_for_pair_set_completion": True,
        "one_consumer_pair_does_not_alias_or_replace_the_other": True,
    }
    return descriptor | {"paired_authority_set_id_sha256": _sha_json(descriptor)}


def build_consumer_payloads(*, evidence: Mapping[str, Any], closure: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(evidence.get("raw_bound_evidence_sha256") == _sha_json({
        key: value for key, value in evidence.items() if key != "raw_bound_evidence_sha256"}),
        "raw-bound evidence self SHA drift")
    groups = evidence["target_support_only_groups"]
    positive_min = min(groups[f"{arm}__{decoder}"]["min"]
                       for arm in POSITIVE_ARMS for decoder in DECODERS)
    ordinary_max = max(groups[f"{UNALIGNED_ARM}__{decoder}"]["max"] for decoder in DECODERS)
    deranged_max = max(groups[f"{DERANGED_ARM}__{decoder}"]["max"] for decoder in DECODERS)
    pair_set = _pair_set(evidence_sha256=evidence["raw_bound_evidence_sha256"])
    common = {
        "raw_bound_control_evidence": dict(evidence),
        "paired_runtime_control_set": pair_set,
        "publisher_implementation_closure_at_launch": dict(closure),
        "publisher_implementation_closure_at_final": dict(closure),
        "publisher_launch_final_live_closure_equal": True,
        "target_data_discovered": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
        "target_execution_authorized_by_this_pair": False,
    }
    subject = {
        "schema": "track_b_v2_subject_m_fixed_runtime_control_hard_null_v1",
        "status": "ROOT_REVIEWED_FIXED_GEOMETRY_RUNTIME_CONTROLS_PASS",
        "fixed_geometry": {"output_dimension": 8, "iterations": 10_000,
                           "linear_ridge_normalized_lambda": 0.01, "cosine_knn_k": 3},
        "bound_d8it250_cost_body_sha256": EXPECTED_COST_SHA256,
        "bound_synthetic_v2_terminal_body_sha256": EXPECTED_TERMINAL_SHA256,
        "control_evidence_closure_sha256": EXPECTED_TERMINAL_CLOSURE_SHA256,
        "settled_synthetic_v2_file_sha256": dict(SETTLED_FILES),
        "decision_threshold_r2": POSITIVE_THRESHOLD_R2,
        "positive_threshold_origin": POSITIVE_THRESHOLD_ORIGIN,
        "deranged_hard_null_threshold_r2": DERANGED_HARD_NULL_THRESHOLD_R2,
        "hard_null_threshold_origin": HARD_NULL_THRESHOLD_ORIGIN,
        "control_execution_scale": {"frozen_by_root_after_cost_review": True,
                                    "root_pending": False, "actual_runtime_cell_count": 32},
        "positive_control": {"both_decoders_reported": True,
                             "positive_arms": list(POSITIVE_ARMS),
                             "threshold": POSITIVE_THRESHOLD_R2,
                             "threshold_origin": POSITIVE_THRESHOLD_ORIGIN,
                             "threshold_frozen_before_target": True,
                             "joint_ridge_strict_pass_count": 8,
                             "joint_knn_strict_pass_count": 8,
                             "frozen_ridge_strict_pass_count": 8,
                             "frozen_knn_strict_pass_count": 8},
        "deranged_support_hard_null": {
            "joint_multisession": True,
            "target_support_neural_unchanged": True,
            "target_support_auxiliary_label_multiset_unchanged": True,
            "permutation_is_seed_independent": True,
            "permutation_authority_sha256_required": True,
            "true_target_query_labels_used_for_scoring_only": True,
            "threshold_frozen_before_target": True,
            "permutation_authority_sha256": EXPECTED_PERMUTATION_SHA256,
            "threshold": DERANGED_HARD_NULL_THRESHOLD_R2,
            "threshold_origin": HARD_NULL_THRESHOLD_ORIGIN,
            "threshold_may_be_relaxed_or_backfilled": False,
            "ridge_strict_pass_count": 8,
            "knn_strict_pass_count": 8,
        },
        "diagnostic_unaligned_distribution_reported_not_hard_gate": True,
        "old_selector_or_synthetic_layout_authorizes_target_execution": False,
        **common,
    }
    rt = {
        "schema": rt_consumer.RUNTIME_CONTROL_SCHEMA,
        "status": rt_consumer.RUNTIME_CONTROL_STATUS,
        "fixed_geometry": {"output_dimension": 8, "iterations": 10_000},
        "vendored_cebra_version": "0.6.1",
        "minted_after_fixed_gpu_cost_review": True,
        "fixed_gpu_cost_receipt_body_sha256": EXPECTED_COST_SHA256,
        "bound_synthetic_v2_terminal_body_sha256": EXPECTED_TERMINAL_SHA256,
        "decision_threshold_r2": POSITIVE_THRESHOLD_R2,
        "positive_threshold_origin": POSITIVE_THRESHOLD_ORIGIN,
        "deranged_hard_null_threshold_r2": DERANGED_HARD_NULL_THRESHOLD_R2,
        "hard_null_threshold_origin": HARD_NULL_THRESHOLD_ORIGIN,
        "controls": {
            "positive_control_pass": True,
            "ordinary_unaligned_distribution_reported": True,
            "ordinary_unaligned_role": "diagnostic_distribution_only__not_a_hard_null_gate",
            "deranged_support_hard_null_pass": True,
            "deranged_permutation_fixed_seed_independent_nonidentity": True,
            "thresholds_frozen_before_target": True,
        },
        "control_scale_and_threshold": {
            "frozen_after_cost_review_before_target": True,
            "synthetic_signal_scale": 1.0,
            "positive_control_threshold_r2": POSITIVE_THRESHOLD_R2,
            "deranged_hard_null_threshold_r2": DERANGED_HARD_NULL_THRESHOLD_R2,
            "positive_threshold_origin": POSITIVE_THRESHOLD_ORIGIN,
            "hard_null_threshold_origin": HARD_NULL_THRESHOLD_ORIGIN,
            "positive_min": float(positive_min),
            "ordinary_negative_max": float(ordinary_max),
            "deranged_hard_null_max": float(deranged_max),
        },
        "ordinary_unaligned_is_threshold_or_pass_gate": False,
        "score_emitted": False,
        **common,
    }
    return {"subject_m": subject, "rt": rt}


def build_dry_plan() -> dict[str, Any]:
    _assert_canonical_outputs_fresh()
    evidence, input_bindings = load_raw_bound_evidence()
    closure = implementation_closure()
    payloads = build_consumer_payloads(evidence=evidence, closure=closure)
    body = {
        "schema": SCHEMA_DRY_PLAN,
        "status": STATUS_DRY_PLAN,
        "canonical_outputs": {"subject_m": str(SUBJECT_BODY), "rt": str(RT_BODY)},
        "hypothetical_payload_sha256": {role: _sha_json(payload) for role, payload in payloads.items()},
        "paired_authority_set": payloads["subject_m"]["paired_runtime_control_set"],
        "raw_bound_evidence_sha256": evidence["raw_bound_evidence_sha256"],
        "input_bindings": input_bindings,
        "implementation_closure": closure,
        "implementation_closure_sha256": closure["closure_sha256"],
        "positive_control_threshold_r2": POSITIVE_THRESHOLD_R2,
        "positive_threshold_origin": POSITIVE_THRESHOLD_ORIGIN,
        "deranged_hard_null_threshold_r2": DERANGED_HARD_NULL_THRESHOLD_R2,
        "hard_null_threshold_origin": HARD_NULL_THRESHOLD_ORIGIN,
        "hard_null_threshold_may_be_relaxed_or_backfilled": False,
        "subject_m_payload": payloads["subject_m"],
        "rt_payload": payloads["rt"],
        "outputs_fresh": True,
        "receipt_minted": False,
        "torch_imported": False,
        "cebra_imported": False,
        "gpu_used": False,
        "target_data_discovered": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "NWB_or_NPZ_opened": False,
    }
    return body | {"dry_plan_sha256": _sha_json(body)}


def _safe_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.resolve() == path.parent and not path.parent.is_symlink(),
            "canonical output parent is symlinked or aliased")


def _bind_parent_directory(path: Path) -> tuple[int, tuple[int, int]]:
    """Open and bind a canonical parent inode before any temporary is created."""
    _safe_parent(path)
    parent = path.parent
    before = parent.lstat()
    require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
            "canonical output parent must be a real directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent, flags)
    try:
        opened = os.fstat(descriptor)
        require(stat.S_ISDIR(opened.st_mode) and
                (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                "canonical output parent inode changed while binding")
        return descriptor, (opened.st_dev, opened.st_ino)
    except BaseException:
        os.close(descriptor)
        raise


def _recheck_bound_parent(path: Path, identity: tuple[int, int]) -> None:
    """Require the canonical path to still name the directory bound at launch."""
    parent = path.parent
    require(parent.resolve() == parent and not parent.is_symlink(),
            "canonical output parent path changed after publication")
    current = parent.lstat()
    require(stat.S_ISDIR(current.st_mode) and
            (current.st_dev, current.st_ino) == identity,
            "canonical output parent identity changed after publication")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent, flags)
    try:
        opened = os.fstat(descriptor)
        require((opened.st_dev, opened.st_ino) == identity,
                "canonical output parent reopened as a different inode")
    finally:
        os.close(descriptor)


def _write_fsync_at(parent_fd: int, name: str, data: bytes) -> None:
    """Create an immutable temporary relative to an already verified directory FD."""
    require(Path(name).name == name, "temporary filename must be one path component")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_at(parent_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def publish_two_pairs_transactionally(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Publish both incompatible consumer pairs as one rollback-safe set."""
    require(set(payloads) == {"subject_m", "rt"}, "two exact consumer payloads required")
    _assert_canonical_outputs_fresh()
    targets = {"subject_m": SUBJECT_BODY, "rt": RT_BODY}
    launch_closures = {
        _sha_json(payloads[role].get("publisher_implementation_closure_at_launch")):
        payloads[role].get("publisher_implementation_closure_at_launch")
        for role in targets
    }
    require(len(launch_closures) == 1, "consumer payload publisher launch closures disagree")
    expected_closure = next(iter(launch_closures.values()))
    require(isinstance(expected_closure, Mapping), "publisher launch closure is missing")
    require(all(payloads[role].get("publisher_implementation_closure_at_final") == expected_closure
                for role in targets),
            "consumer payload publisher launch/final closure claims disagree")
    parent_bindings: dict[Path, tuple[int, tuple[int, int]]] = {}
    try:
        for path in targets.values():
            if path.parent not in parent_bindings:
                parent_bindings[path.parent] = _bind_parent_directory(path)
    except BaseException:
        for descriptor, _identity in parent_bindings.values():
            os.close(descriptor)
        raise
    encoded = {role: route.canonical_json_bytes(dict(payloads[role])) for role in targets}
    digests = {role: hashlib.sha256(data).hexdigest() for role, data in encoded.items()}
    nonce = f"{os.getpid()}.{time.time_ns()}"
    temporaries: list[tuple[int, str]] = []
    created: list[tuple[int, str]] = []
    bindings: dict[str, Any] | None = None
    try:
        for role, body in targets.items():
            parent_fd = parent_bindings[body.parent][0]
            temp_body = f".{body.name}.{nonce}.{role}.body.tmp"
            temp_sidecar = f".{body.name}.{nonce}.{role}.sidecar.tmp"
            _write_fsync_at(parent_fd, temp_body, encoded[role])
            _write_fsync_at(parent_fd, temp_sidecar,
                            f"{digests[role]}  {body.name}\n".encode("ascii"))
            temporaries.extend(((parent_fd, temp_body), (parent_fd, temp_sidecar)))
        for index, (role, body) in enumerate(targets.items()):
            parent_fd = parent_bindings[body.parent][0]
            temp_body = temporaries[index * 2][1]
            temp_sidecar = temporaries[index * 2 + 1][1]
            os.link(temp_body, body.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    follow_symlinks=False)
            created.append((parent_fd, body.name))
            sidecar_name = _sidecar(body).name
            os.link(temp_sidecar, sidecar_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                    follow_symlinks=False)
            created.append((parent_fd, sidecar_name))
        for descriptor, _identity in parent_bindings.values():
            os.fsync(descriptor)
        for body in targets.values():
            _recheck_bound_parent(body, parent_bindings[body.parent][1])

        # Keep this check inside the rollback boundary: the launch/final/live
        # claim is true only if the complete closure stays byte-identical after
        # all four links are durable.
        final_closure = implementation_closure()
        require(final_closure == expected_closure,
                "publisher implementation closure drift after publication")
        bindings = {
            role: route.load_immutable_json(body, expected_schema=payloads[role]["schema"])[1]
            for role, body in targets.items()
        }
    except BaseException:
        for descriptor, name in reversed(created):
            _unlink_at(descriptor, name)
        for descriptor, _identity in parent_bindings.values():
            os.fsync(descriptor)
        raise
    finally:
        for descriptor, name in temporaries:
            _unlink_at(descriptor, name)
        for descriptor, _identity in parent_bindings.values():
            os.fsync(descriptor)
            os.close(descriptor)
    require(bindings is not None, "transaction completed without verified output bindings")
    return bindings
