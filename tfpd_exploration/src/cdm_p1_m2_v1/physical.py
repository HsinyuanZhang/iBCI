"""Physical executor of the M2 prerequisite audit (read-only, CPU-only).

The audit opens the frozen M2 model and data exactly as the sealed m2_t4
screen does (``load_frozen_model_and_data``), re-reads the official FALCON
evaluation client source that a submitted decoder actually runs under, and
measures the three pre-registered facts.  It never touches a GPU, never
writes a checkpoint, and publishes one immutable ``audit.json`` pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import audit, plan


class AuditExecutionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditExecutionError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_official_contract() -> dict[str, Any]:
    """Bind and re-read the installed FALCON evaluation client source."""
    import inspect

    module = __import__(plan.EVALUATOR_MODULE, fromlist=["FalconEvaluator"])
    source = inspect.getsource(module)
    markers = {
        name: marker in source for name, marker in plan.EVALUATOR_SOURCE_MARKERS.items()
    }
    assertions = {
        name: marker in source for name, marker in plan.EVALUATOR_ASSERTIONS.items()
    }
    _require(
        markers["continual_tasks"] and markers["predict_receives_neural_only"],
        "the installed falcon_challenge evaluator no longer matches the audited contract",
    )
    gated = markers["on_done_gated_on_not_continual"] and assertions["gated_on_done_call"]
    _require(gated and assertions["continual_branch_covers_m2"],
             "the evaluator on_done gate could not be bound")
    evaluator_path = Path(module.__file__)
    deployed = Path(plan.DEPLOYED_DECODER_RELATIVE)
    return {
        "evaluator_module": plan.EVALUATOR_MODULE,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": _sha256_file(evaluator_path),
        "markers": {**markers, "on_done_gated_on_not_continual": bool(gated)},
        "assertions": assertions,
        "loop_reading": (
            "for a continual task (h1/m1/m2) the evaluator calls only "
            "reset(dataset_tags) and predict(neural_observations) per step; "
            "on_done is called ONLY for non-continual tasks and observe is "
            "never called on this path"
        ),
        "deployed_decoder_relative": plan.DEPLOYED_DECODER_RELATIVE,
        "deployed_decoder_on_done_is_noop": True,
        "deployed_decoder_note": (
            "the submitted cached-identity T4 image implements on_done as an "
            "explicit no-op, so boundaries would not reach any state machine "
            "even if the server delivered them"
        ),
        "calibration_metadata_available": (
            "the labelled calibration NWBs (trial_target_angles) are part of "
            "the official calibration phase; the QUERY stream is not"
        ),
    }


def _session_boundary_rows(raw_sessions: Mapping[str, Any], surface: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for session in sorted(raw_sessions):
        payload = raw_sessions[session]
        neural = np.asarray(payload["neural"], dtype=np.float64)
        trial_change = np.asarray(payload["trial_change"], dtype=bool)
        _require(neural.ndim == 2 and trial_change.shape == (neural.shape[0],),
                 f"{surface}/{session} raw topology drift")
        _require(int(trial_change.sum()) >= 1, f"{surface}/{session} has no boundaries")
        statistics = audit.boundary_statistics(neural)
        rows[session] = {
            name: audit.rank_auc(values, trial_change)
            for name, values in statistics.items()
        }
    return rows


def _direction_rows(dataset: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    canonical = np.asarray([-3.0 * np.pi / 4.0 + index * (np.pi / 4.0) for index in range(8)])
    for session in sorted(dataset.calib_trial_target_angles):
        angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
        _require(angles.size >= plan.ACTIVITY_HORIZON, f"{session} lacks a first-30 angle pool")
        first30 = angles[: plan.ACTIVITY_HORIZON]
        finite = np.isfinite(first30)
        histogram = [0] * canonical.size
        for value in first30[finite]:
            distance = np.abs(np.arctan2(np.sin(value - canonical), np.cos(value - canonical)))
            histogram[int(np.argmin(distance))] += 1
        rows[session] = {
            "first30_pool": int(first30.size),
            "finite_direction_trials": int(finite.sum()),
            "fraction_with_defined_direction": float(finite.sum() / first30.size),
            "canonical_histogram": histogram,
            "distinct_canonical_directions": int(np.count_nonzero(np.asarray(histogram))),
        }
    return rows


def _anchor_rows(dataset: Any) -> tuple[dict[str, Any], bool]:
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    rows: dict[str, Any] = {}
    all_bitwise = True
    for session in sorted(dataset.calib_trialized_neural_features):
        sums = np.asarray(dataset.calib_trial_spike_sums[session], dtype=np.float64)
        lengths = np.asarray(dataset.calib_trial_lengths[session], dtype=np.float64)
        angles = np.asarray(dataset.calib_trial_target_angles[session], dtype=np.float64)
        _require(sums.shape[0] == lengths.shape[0] == angles.shape[0],
                 f"{session} calibration axis drift")
        usable = np.isfinite(angles)
        _require(int(usable.sum()) >= 3, f"{session} has fewer than three directional calibration trials")
        rates = sums[usable] / lengths[usable, None]
        sealed, evidence = fit_ridge_t4(
            rates, angles[usable], normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
        )
        parity = audit.support_anchor_parity(
            rates, angles[usable],
            normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA, sealed_raw_t4=sealed,
        )
        all_bitwise = all_bitwise and bool(parity["rebuilt_bitwise_equal"])
        rows[session] = {
            **parity,
            "sealed_fit_evidence": {
                key: evidence[key] for key in ("normalized_lambda", "design_rank", "design_condition")
            },
            "usable_directional_trials": int(usable.sum()),
        }
    return rows, all_bitwise


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    body = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def execute(repo_root: Path) -> dict[str, Any]:
    """Run the audit once and publish ``audit.json`` under a fresh root."""
    repo_root = Path(repo_root).absolute()
    root = plan.result_root(repo_root)
    _require(not (root / "audit.json").exists(), "the audit receipt already exists")
    _require((root / "attempt.json").exists(), "reserve the attempt receipt first")
    attempt_path = root / "attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt_digest = _sha256_file(attempt_path)
    sidecar = attempt_path.with_name(attempt_path.name + ".sha256")
    _require(sidecar.exists(), "missing attempt sidecar")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{attempt_digest}  attempt.json",
        "attempt sidecar drift",
    )
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
             "the M2 audit is CPU-only by law; unset CUDA_VISIBLE_DEVICES")

    for entry in ("streaming_calibration_exp", "sua_exploration"):
        candidate = repo_root.parent / entry
        if str(candidate.parent) not in sys.path:
            sys.path.insert(0, str(candidate.parent))

    contract = read_official_contract()
    started = time.monotonic()
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == plan.CHECKPOINT_SHA256, "M2 checkpoint drift")
    _require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "M2 normalizer drift")
    train_raw = data_module.train_calib_heldin_sessions
    val_raw = data_module.val_calib_heldout_sessions
    _require(len(train_raw) == plan.EXPECTED_WITHIN_SESSIONS, "within raw session count drift")
    _require(len(val_raw) == plan.EXPECTED_EXTERNAL_SESSIONS, "external raw session count drift")

    within_auc = _session_boundary_rows(train_raw, "within")
    external_auc = _session_boundary_rows(val_raw, "external")
    direction_within = _direction_rows(data_module.train_dataset)
    direction_external = _direction_rows(data_module.val_heldout_dataset)
    anchor_within, within_parity = _anchor_rows(data_module.train_dataset)
    anchor_external, external_parity = _anchor_rows(data_module.val_heldout_dataset)

    verdict = audit.boundary_verdict(
        contract_markers_present=all(bool(value) for value in contract["markers"].values()),
        on_done_served_to_decoder=False,
        source_auc_by_session=within_auc,
        auc_floor=plan.BOUNDARY_AUC_PASS_MIN,
    )
    source_best = {
        session: max(values.values()) for session, values in within_auc.items()
    }
    payload = {
        "schema": f"{plan.SCHEMA}_audit_v1",
        "status": "AUDIT_TERMINAL",
        "attempt_sha256": attempt_digest,
        "work_order_section": "3 (Part B2 prerequisite audit)",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "prerequisites": {
            "a_trial_boundaries": {
                "question": (
                    "are completed-trial boundaries available or causally "
                    "reconstructible at inference under the M2 evaluation contract?"
                ),
                "official_contract": contract,
                "reconstruction_law": {
                    "family": "causal neural-only threshold statistics",
                    "members": list(plan.BOUNDARY_STATISTICS),
                    "coverage_statistic": "per-session rank AUC vs the true trial_change indicator",
                    "selection_surface": "M2 OWN source (within) sessions only; external AUCs are disclosed diagnostics",
                    "auc_floor": plan.BOUNDARY_AUC_PASS_MIN,
                    "decoder_output_laws_rejected": (
                        "a boundary law read from the decoder's own predictions is "
                        "circular with the P1 carrier update under test and is not "
                        "pre-registrable here"
                    ),
                },
                "within_source_auc": within_auc,
                "external_diagnostic_auc": external_auc,
                "best_auc_by_source_session": source_best,
                "verdict": verdict,
            },
            "b_t4_equivalent_carrier": {
                "question": (
                    "does a support-anchor (A0 = Xs^T Xs + lambda*I, b0) fit exist or "
                    "is it cheaply fittable from the calibration block with the same "
                    "law as src/support_anchored_t4_stage_o_v1/anchor.py?"
                ),
                "law": dict(plan.ANCHOR_PARITY_LAW),
                "within_anchor_rows": anchor_within,
                "external_anchor_rows": anchor_external,
                "all_sessions_bitwise_parity": bool(within_parity and external_parity),
            },
            "c_direction_parametrization": {
                "question": "what is the direction parametrization for the M2 task?",
                "law": dict(plan.DIRECTION_LAW),
                "within_rows": direction_within,
                "external_rows": direction_external,
                "fraction_with_defined_direction_all_sessions": float(np.mean([
                    row["fraction_with_defined_direction"]
                    for row in {**direction_within, **direction_external}.values()
                ])),
            },
        },
        "m2_arm_disposition": (
            "P1-M2 NOT_EVALUABLE_OFFICIAL_CONTRACT"
            if verdict["verdict"] == "NOT_EVALUABLE_OFFICIAL_CONTRACT"
            else "see prerequisites.a_trial_boundaries.verdict"
        ),
        "part_a_context_binding": {
            "part_a_terminal_relative": "tfpd_exploration/results/cdm_p1_cross_v1/terminal.json",
            "part_a_champion": "F01 external M4 0.2663675067260204 (delta +0.020463084852528825)",
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    _require(
        payload["prerequisites"]["b_t4_equivalent_carrier"]["all_sessions_bitwise_parity"],
        "the anchor parity law failed; the audit must fail closed",
    )
    digest = _publish(root / "audit.json", payload)
    return {"audit_sha256": digest, "verdict": verdict["verdict"],
            "m2_arm_disposition": payload["m2_arm_disposition"]}
