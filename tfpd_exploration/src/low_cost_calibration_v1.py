"""Low-cost carrier-support diagnostics on the sealed Cell-D scorer.

This module is additive and non-authorizing.  The public CLI remains dry; the
reviewed execution function is called directly by the root operator after an
exact environment/fresh-root check.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_LOW_COST_CALIBRATION_DIAGNOSTICS_20260823.md"
)
FAILED_V1_ROOT_RELATIVE = "tfpd_exploration/results/low_cost_calibration_diagnostics_v1"
FAILED_V2_ROOT_RELATIVE = "tfpd_exploration/results/low_cost_calibration_diagnostics_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/low_cost_calibration_diagnostics_v3"
SCHEMA = "low_cost_calibration_diagnostics_v3"
FAILED_V1_SHA256 = {
    "attempt.json": "8969cf3071d4b8a8a567d941cf45a2b22b00537058446a210cd0c8a913b1b732",
    "failure.json": "cec97326146fe920499a3646ada0b8074b81886670ed636d3f99d960eaf0520d",
}
FAILED_V2_SHA256 = {
    "attempt.json": "53515263e8766b688fd88c3ed2021b34b7517c2e7343ad984248faa7e2453f8b",
    "failure.json": "dd699fb4f196506b20126e957ed582ef4166a096a22a644e9602c5d6dc16e952",
}
SUPPORTS = (
    "C0_contiguous",
    "C1_cue_balanced_early",
    "C2_cue_matched_scattered",
    "C3_full_session_oracle",
)
BUDGETS = (4, 30)
CPU_BUDGETS = (4, 10, 15, 20, 30, 50)
V4_SCORE_SHA256 = "dea4bc361d083fe27ebd1f5d17045122c2c0e178a8c88506a46b9c8b1c8da04f"
EXPECTED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
}


class DiagnosticError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": "), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _compact_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _array_sha(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    payload = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes_sha256": _sha(value.tobytes()),
    }
    return _sha(_compact_json(payload))


def cue_matched_scattered_indices(
    direction_indices: Any, early_selected_indices: Any,
) -> Any:
    """Match C1's cue multiset while spreading its rows over full time.

    Target locations are evenly spaced over the complete session.  For each
    C1 cue, choose the nearest still-unused occurrence of that cue.  Stable
    distance/index tie breaking makes the result deterministic.
    """
    import numpy as np

    directions = np.asarray(direction_indices, dtype=np.int64).reshape(-1)
    early = np.asarray(early_selected_indices, dtype=np.int64).reshape(-1)
    _require(directions.size >= early.size > 0, "invalid scattered-selection inputs")
    _require(int(early.min()) >= 0 and int(early.max()) < directions.size, "early index outside rows")
    desired = directions[early]
    targets = np.linspace(0.0, float(directions.size - 1), early.size, dtype=np.float64)
    used: set[int] = set()
    chosen: list[int] = []
    for target, cue in zip(targets.tolist(), desired.tolist()):
        candidates = [int(index) for index in np.flatnonzero(directions == int(cue)) if int(index) not in used]
        _require(bool(candidates), f"insufficient occurrences for cue {cue}")
        selected = min(candidates, key=lambda index: (abs(float(index) - target), index))
        used.add(selected)
        chosen.append(selected)
    result = np.sort(np.asarray(chosen, dtype=np.int64))
    _require(
        sorted(directions[result].tolist()) == sorted(desired.tolist()),
        "scattered selection changed the cue multiset",
    )
    return result


def support_indices(thetas: Any, direction_indices: Any, *, budget: int) -> dict[str, Any]:
    import numpy as np
    from mc_maze.d_optimal_calibration_design import greedy_forward_d_optimal_indices

    theta = np.asarray(thetas, dtype=np.float64).reshape(-1)
    directions = np.asarray(direction_indices, dtype=np.int64).reshape(-1)
    _require(theta.shape == directions.shape and theta.size >= 50, "support rows are incomplete")
    _require(budget in CPU_BUDGETS and budget <= 50, "unsupported calibration budget")
    c0 = np.arange(budget, dtype=np.int64)
    c1 = np.sort(greedy_forward_d_optimal_indices(theta[:50], budget))
    c2 = cue_matched_scattered_indices(directions, c1)
    c3 = np.arange(theta.size, dtype=np.int64)
    return {
        "C0_contiguous": c0,
        "C1_cue_balanced_early": c1,
        "C2_cue_matched_scattered": c2,
        "C3_full_session_oracle": c3,
    }


def carrier_rows(
    *, rates: Any, thetas: Any, directions: Any, trial_times: Any, budget: int,
) -> dict[str, dict[str, object]]:
    import numpy as np
    from mc_maze.d_optimal_calibration_design import (
        ac_cosine_per_unit,
        design_matrix_from_thetas,
        design_metrics,
        fit_carriers_from_selected_trials,
        per_unit_rmse,
        predict_rates_from_carrier,
    )

    rate = np.asarray(rates, dtype=np.float64)
    theta = np.asarray(thetas, dtype=np.float64).reshape(-1)
    direction = np.asarray(directions, dtype=np.int64).reshape(-1)
    times = np.asarray(trial_times, dtype=np.float64).reshape(-1)
    _require(rate.ndim == 2 and rate.shape[0] == theta.size == direction.size == times.size, "carrier row alignment drift")
    selections = support_indices(theta, direction, budget=budget)
    full = fit_carriers_from_selected_trials(rate, direction, selections["C3_full_session_oracle"])
    result: dict[str, dict[str, object]] = {}
    for support in SUPPORTS:
        selected = selections[support]
        fitted = fit_carriers_from_selected_trials(rate, direction, selected)
        cosine = ac_cosine_per_unit(fitted[:, :2], full[:, :2])
        cosine = cosine[np.isfinite(cosine)]
        predicted = predict_rates_from_carrier(fitted, theta)
        rmse = per_unit_rmse(rate, predicted)
        counts = np.bincount(direction[selected], minlength=8).astype(np.int64)
        metrics = design_metrics(design_matrix_from_thetas(theta[selected]))
        result[support] = {
            "budget": budget,
            "selected_indices": selected.tolist(),
            "selected_indices_sha256": _array_sha(selected),
            "selected_direction_counts": counts.tolist(),
            "selected_direction_counts_sha256": _array_sha(counts),
            "selected_trial_count": int(selected.size),
            "index_span": int(selected.max() - selected.min()),
            "time_span_seconds": float(times[selected].max() - times[selected].min()),
            "design_rank": int(metrics["design_rank"]),
            "design_condition": (
                float(metrics["design_condition"])
                if math.isfinite(float(metrics["design_condition"])) else None
            ),
            "median_ac_cosine_vs_full": float(np.median(cosine)),
            "defined_ac_cosine_units": int(cosine.size),
            "median_full_session_rate_rmse_hz": float(np.median(rmse)),
            "raw_t4_sha256": _array_sha(np.ascontiguousarray(fitted, dtype=np.float32)),
            "raw_t4": np.ascontiguousarray(fitted, dtype=np.float32),
        }
    _require(
        result["C1_cue_balanced_early"]["selected_direction_counts"]
        == result["C2_cue_matched_scattered"]["selected_direction_counts"],
        "C1/C2 cue count matching failed",
    )
    return result


def score_decomposition(prediction: Any, target: Any) -> dict[str, object]:
    import numpy as np

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _require(pred.shape == truth.shape and pred.ndim == 2 and pred.shape[1] == 2, "score decomposition shape drift")
    _require(np.isfinite(pred).all() and np.isfinite(truth).all(), "score decomposition nonfinite")

    def r2_columns(left: Any, right: Any) -> Any:
        residual = np.sum((left - right) ** 2, axis=0)
        centered = right - right.mean(axis=0, keepdims=True)
        total = np.sum(centered ** 2, axis=0)
        _require(bool(np.all(total > 0.0)), "zero target variance in score decomposition")
        return 1.0 - residual / total, residual, total

    coordinate, residual, total = r2_columns(pred, truth)
    variance_weighted = float(1.0 - float(residual.sum()) / float(total.sum()))
    covariance = np.cov(truth, rowvar=False, ddof=0)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    rotated_pred = pred @ eigenvectors
    rotated_truth = truth @ eigenvectors
    eigen_r2, _, _ = r2_columns(rotated_pred, rotated_truth)
    participation_ratio = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))
    return {
        "variance_weighted_r2": variance_weighted,
        "coordinate_r2": [float(value) for value in coordinate],
        "target_covariance": covariance.tolist(),
        "target_eigenvalues_descending": [float(value) for value in eigenvalues],
        "target_eigenvectors_columns": eigenvectors.tolist(),
        "eigenbasis_r2_descending": [float(value) for value in eigen_r2],
        "target_participation_ratio": participation_ratio,
    }


def aggregate_session_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    import numpy as np

    _require(bool(rows), "cannot aggregate empty session rows")
    keys = ("variance_weighted_r2", "coordinate_r2", "eigenbasis_r2_descending", "target_participation_ratio")
    for row in rows:
        _require(all(key in row for key in keys), "aggregate session metric missing")
    scalar = np.asarray([float(row["variance_weighted_r2"]) for row in rows], dtype=np.float64)
    coordinate = np.asarray([row["coordinate_r2"] for row in rows], dtype=np.float64)
    eigen = np.asarray([row["eigenbasis_r2_descending"] for row in rows], dtype=np.float64)
    pr = np.asarray([float(row["target_participation_ratio"]) for row in rows], dtype=np.float64)
    return {
        "session_count": len(rows),
        "equal_session_mean_r2": float(scalar.mean()),
        "equal_session_median_r2": float(np.median(scalar)),
        "equal_session_mean_coordinate_r2": coordinate.mean(axis=0).tolist(),
        "equal_session_mean_eigenbasis_r2": eigen.mean(axis=0).tolist(),
        "target_participation_ratio_mean": float(pr.mean()),
        "target_participation_ratio_median": float(np.median(pr)),
    }


def _read_exact_json_pair(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, object], str]:
    before = os.lstat(path)
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"unsafe receipt path {path}")
    body = path.read_bytes()
    after = os.lstat(path)
    _require((before.st_dev, before.st_ino, before.st_size) == (after.st_dev, after.st_ino, after.st_size), "receipt changed while read")
    digest = _sha(body)
    _require(expected_sha256 is None or digest == expected_sha256, f"receipt SHA drift: {path}")
    sidecar = path.with_name(path.name + ".sha256").read_bytes()
    _require(sidecar == f"{digest}  {path.name}\n".encode("ascii"), f"receipt sidecar drift: {path}")
    payload = json.loads(body)
    _require(isinstance(payload, dict), "receipt JSON root is not an object")
    return payload, digest


def validate_failed_v1(root: Path) -> dict[str, object]:
    directory = Path(root).absolute() / FAILED_V1_ROOT_RELATIVE
    names = sorted(item.name for item in directory.iterdir())
    expected_names = sorted(
        name for body in FAILED_V1_SHA256 for name in (body, body + ".sha256")
    )
    _require(names == expected_names, "failed V1 topology drift")
    attempt, attempt_sha = _read_exact_json_pair(directory / "attempt.json", FAILED_V1_SHA256["attempt.json"])
    failure, failure_sha = _read_exact_json_pair(directory / "failure.json", FAILED_V1_SHA256["failure.json"])
    _require(attempt.get("schema") == "low_cost_calibration_diagnostics_v1_attempt", "failed V1 attempt schema drift")
    _require(attempt.get("status") == "ATTEMPT_RESERVED", "failed V1 attempt status drift")
    _require(failure.get("schema") == "low_cost_calibration_diagnostics_v1_failure", "failed V1 failure schema drift")
    _require(failure.get("status") == "DIAGNOSTIC_FAILED", "failed V1 failure status drift")
    _require(failure.get("attempt_sha256") == attempt_sha, "failed V1 attempt/failure binding drift")
    _require(failure.get("error_class") == "AttributeError", "failed V1 error class drift")
    _require(all(failure.get(key) == 0 for key in (
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
    )), "failed V1 target-update boundary drift")
    _require(failure.get("formal_opened") is False, "failed V1 formal boundary drift")
    return {
        "root_relative": FAILED_V1_ROOT_RELATIVE,
        "attempt_sha256": attempt_sha,
        "failure_sha256": failure_sha,
        "stage": "post_input_materialization_pre_experimental_forward",
        "reason": "wrong_opaque_session_module_alias",
        "no_experimental_forward": True,
        "no_target_updates": True,
        "formal_opened": False,
    }


def validate_failed_v2(root: Path) -> dict[str, object]:
    directory = Path(root).absolute() / FAILED_V2_ROOT_RELATIVE
    names = sorted(item.name for item in directory.iterdir())
    expected_names = sorted(
        name for body in FAILED_V2_SHA256 for name in (body, body + ".sha256")
    )
    _require(names == expected_names, "failed V2 topology drift")
    attempt, attempt_sha = _read_exact_json_pair(directory / "attempt.json", FAILED_V2_SHA256["attempt.json"])
    failure, failure_sha = _read_exact_json_pair(directory / "failure.json", FAILED_V2_SHA256["failure.json"])
    _require(attempt.get("schema") == "low_cost_calibration_diagnostics_v2_attempt", "failed V2 attempt schema drift")
    _require(attempt.get("status") == "ATTEMPT_RESERVED", "failed V2 attempt status drift")
    _require(attempt.get("failed_v1") == validate_failed_v1(root), "failed V2 predecessor binding drift")
    _require(failure.get("schema") == "low_cost_calibration_diagnostics_v2_failure", "failed V2 failure schema drift")
    _require(failure.get("status") == "DIAGNOSTIC_FAILED", "failed V2 failure status drift")
    _require(failure.get("attempt_sha256") == attempt_sha, "failed V2 attempt/failure binding drift")
    _require(failure.get("error_class") == "DiagnosticError", "failed V2 error class drift")
    _require(all(failure.get(key) == 0 for key in (
        "target_optimizer_steps", "target_backward_calls", "target_update_calls",
    )), "failed V2 target-update boundary drift")
    _require(failure.get("formal_opened") is False, "failed V2 formal boundary drift")
    return {
        "root_relative": FAILED_V2_ROOT_RELATIVE,
        "attempt_sha256": attempt_sha,
        "failure_sha256": failure_sha,
        "stage": "experimental_inference_C0_metric_parity",
        "reason": "float64_audit_r2_compared_to_governing_torchmetrics_float32",
        "prediction_sha_parity_passed_before_metric_check": True,
        "no_target_updates": True,
        "formal_opened": False,
    }


def _publish_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = _json_bytes(payload)
    digest = _sha(body)
    for leaf, content in ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode("ascii"))):
        descriptor = os.open(directory / leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                _require(written > 0, f"short immutable write: {leaf}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(directory / leaf, 0o444)
    dirfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    return digest


def implementation_closure(root: Path) -> dict[str, object]:
    paths = (
        WORKORDER_RELATIVE,
        "tfpd_exploration/src/low_cost_calibration_v1.py",
        "tfpd_exploration/scripts/run_low_cost_calibration_v1.py",
        "tfpd_exploration/tests/test_low_cost_calibration_v1.py",
        "sua_exploration/mc_maze/d_optimal_calibration_design.py",
        "sua_exploration/mc_maze/d_optimal_calibration_replay.py",
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "sua_exploration/mc_maze/unit_side_features.py",
        "tfpd_exploration/src/posterior_marginalized_cell_d_v4/matched_score.py",
        "tfpd_exploration/src/posterior_marginalized_cell_d_v4/matched_score_physical.py",
    )
    hashes = {relative: _sha((Path(root) / relative).read_bytes()) for relative in paths}
    payload: dict[str, object] = {"schema": "low_cost_calibration_closure_v1", "sha256_by_path": hashes}
    payload["closure_sha256"] = _sha(_compact_json(payload))
    return payload


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "supports": list(SUPPORTS),
        "budgets": list(BUDGETS),
        "cpu_budgets": list(CPU_BUDGETS),
        "result_root": RESULT_ROOT_RELATIVE,
        "failed_v1_root": FAILED_V1_ROOT_RELATIVE,
        "failed_v2_root": FAILED_V2_ROOT_RELATIVE,
        "formal_opened": False,
        "training_authorized": False,
    }


class CalibrationDiagnosticBackend:
    """Composition wrapper around the immutable V4 backend/runtime."""

    def __init__(self, base: Any) -> None:
        self.base = base

    def sealed_model(self) -> Any:
        from src.posterior_marginalized_cell_d_v4 import matched_score as v4score

        model = self.base._models.get(v4score.SYSTEM_SEALED)
        _require(model is not None, "sealed Cell-D model unavailable")
        return model

    def sessions(self, surface: str) -> Mapping[str, Any]:
        value = self.base._sessions.get(surface)
        _require(isinstance(value, Mapping) and value, f"missing materialized surface {surface}")
        return value

    def close(self) -> None:
        self.base.close()


def _build_reviewed_backend(root: Path, profile: Mapping[str, object]) -> tuple[Any, Any, Any]:
    from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
    from src.posterior_marginalized_cell_d_v3 import matched_score as v3score
    from src.posterior_marginalized_cell_d_v3 import matched_score_physical as v3p
    from src.posterior_marginalized_cell_d_v4 import matched_score as v4score
    from src.posterior_marginalized_cell_d_v4 import matched_score_physical as v4p

    graph = v4p.validate_failed_v3_graph(root)
    failed_v2 = v3p.validate_failed_v2_graph(root)
    v3_closure = v3score.implementation_closure(root, failed_v2_graph=failed_v2.payload())
    provenance = v1p.load_verified_provenance(root)
    base_identity = v3p.score_identity_from_provenance_v3(
        provenance=provenance, closure=v3_closure, failed_v2_graph=failed_v2,
    )
    closure = v4score.implementation_closure(
        root, v3_base_closure=v3_closure.payload(), failed_v3_graph=graph.payload(),
    )
    identity = v4score.V4ScoreIdentity(
        base_identity=base_identity, closure=closure,
        v3_input_authority_sha256=graph.body_sha256["input_authority.json"],
    )
    runtime = v4p.V4PhysicalRuntime(root=root, device_profile=profile)
    base = v4p.V4PhysicalMatchedScoreBackend(
        root=root, runtime=runtime, device_profile=profile,
        provenance_loader=v1p.load_verified_provenance, v3_graph=graph,
    )
    base.prepare(identity=identity)
    reader = v1p._load_exact_module(
        "_low_cost_calibration_asset_reader",
        root / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    )
    rows = reader.derive_fixed_input_assets(
        root, within_roster=base_identity.within_roster, external_roster=base_identity.external_roster,
    )
    assets = v1p._assets_from_rows(rows)
    authority = base.materialize_inputs(identity=identity, assets=assets)
    return CalibrationDiagnosticBackend(base), identity, authority


def _v4_baseline_rows(root: Path) -> tuple[dict[tuple[str, int, str], Mapping[str, object]], str]:
    payload, digest = _read_exact_json_pair(
        root / "tfpd_exploration/results/posterior_marginalized_cell_d_score_v4/score.json",
        V4_SCORE_SHA256,
    )
    rows: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for cell in payload.get("cells", []):
        spec = cell.get("cell", {})
        if spec.get("system") != "sealed_cell_d_swa":
            continue
        for row in cell.get("sessions", []):
            rows[(str(spec["surface"]), int(spec["budget"]), str(row["session"]))] = row
    _require(len(rows) == 42, "V4 sealed baseline row topology drift")
    return rows, digest


def execute_reviewed(root: Path) -> Mapping[str, object]:
    """Run the one-shot diagnostic and publish a fresh immutable receipt graph."""
    import numpy as np
    import torch
    from mc_maze.d_optimal_calibration_design import direction_indices_from_thetas
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
    from src.posterior_marginalized_cell_d_v1 import plan as v1plan
    from src.posterior_marginalized_cell_d_v4 import matched_score as v4score

    root = Path(root).absolute()
    _require(all(os.environ.get(key) == value for key, value in EXPECTED_ENVIRONMENT.items()), "launch environment drift")
    _require(os.environ.get("SUBC_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-C"), "SUBC root drift")
    _require(os.environ.get("SUBM_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-M"), "SUBM root drift")
    failed_predecessors = {
        "v1": validate_failed_v1(root),
        "v2": validate_failed_v2(root),
    }
    result_root = root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "diagnostic result root is not fresh")
    result_root.mkdir(parents=False, mode=0o700)
    closure = implementation_closure(root)
    baseline, baseline_sha = _v4_baseline_rows(root)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_RESERVED",
        "closure": closure,
        "v4_score_sha256": baseline_sha,
        "failed_predecessors": failed_predecessors,
        "supports": list(SUPPORTS),
        "budgets": list(BUDGETS),
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "formal_opened": False,
    }
    attempt_sha = _publish_pair(result_root, "attempt.json", attempt)
    backend: CalibrationDiagnosticBackend | None = None
    try:
        profile = v1plan.validate_compatible_device_profile(v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
        backend, identity, authority = _build_reviewed_backend(root, profile)
        authority_payload = authority.payload(identity=identity.base_identity)
        input_sha = _sha(_compact_json(authority_payload))
        _require(
            input_sha == identity.v3_input_authority_sha256,
            "rematerialized input authority differs from held V3/V4 authority",
        )
        model = backend.sealed_model()
        state_before = backend.base.runtime.state_digest(model)
        mean = torch.tensor(v1plan.SEALED_OLS_T4_MEAN_FLOAT32, device="cuda:0", dtype=torch.float32)
        std = torch.tensor(v1plan.SEALED_OLS_T4_STD_FLOAT32, device="cuda:0", dtype=torch.float32)
        all_rows: list[dict[str, object]] = []
        cpu_rows: list[dict[str, object]] = []
        d_optimal_payloads: list[dict[str, object]] = []
        for surface in v4score.SURFACES:
            for session_name, prepared in backend.sessions(surface).items():
                private = prepared.opaque
                _require(isinstance(private, v1p._TorchPreparedSession), "prepared opaque session drift")
                snapshot = private.held_asset.private_snapshot()
                try:
                    trials = list_datamodule_rewarded_trials(
                        snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
                    )
                    theta = np.asarray([trial["target_dir"] for trial in trials], dtype=np.float64)
                    _require(np.isfinite(theta).all(), f"{session_name}: missing target cue")
                    directions = direction_indices_from_thetas(theta)
                    rates_by_unit, units = _pool_trial_rate_matrix(snapshot.path, trials)
                    rates = np.ascontiguousarray(rates_by_unit.T, dtype=np.float64)
                    _require(units == private.neural.shape[1], f"{session_name}: unit-axis drift")
                    times = np.asarray([trial["start_time"] for trial in trials], dtype=np.float64)
                    support_by_budget = {
                        budget: carrier_rows(
                            rates=rates, thetas=theta, directions=directions,
                            trial_times=times, budget=budget,
                        )
                        for budget in CPU_BUDGETS
                    }
                    snapshot.reverify()
                finally:
                    snapshot.close()
                cpu_rows.append({
                    "surface": surface,
                    "session": session_name,
                    "trial_count": len(trials),
                    "unit_count": units,
                    "budgets": {
                        str(budget): {
                            support: {key: value for key, value in row.items() if key != "raw_t4"}
                            for support, row in support_by_budget[budget].items()
                        }
                        for budget in CPU_BUDGETS
                    },
                })
                if surface == v4score.WITHIN:
                    d_optimal_payloads.append({
                        "session_name": session_name,
                        "candidate_pool_k": 50,
                        "post_pool_trials": len(trials) - 50,
                        "direction_indices": directions.tolist(),
                        "thetas_rad": theta.tolist(),
                        "trial_times": times.tolist(),
                        "trial_rates": rates.tolist(),
                        "units": units,
                    })
                for budget in BUDGETS:
                    for support in SUPPORTS:
                        carrier = support_by_budget[budget][support]
                        if support == "C0_contiguous":
                            side = private.side_by_budget[budget]
                        else:
                            raw = torch.as_tensor(carrier["raw_t4"], dtype=torch.float32, device="cuda:0")
                            side = ((raw - mean) / std).detach().unsqueeze(0)
                        side_hash = _array_sha(side.detach().cpu().contiguous().numpy())
                        replacement_map = dict(private.side_by_budget)
                        replacement_map[budget] = side
                        replaced_private = dataclasses.replace(private, side_by_budget=replacement_map)
                        token = _sha(_compact_json({
                            "base_input": prepared.input_token_sha256,
                            "surface": surface,
                            "session": session_name,
                            "budget": budget,
                            "support": support,
                            "normalized_side_sha256": side_hash,
                        }))
                        replaced = dataclasses.replace(prepared, opaque=replaced_private, input_token_sha256=token)
                        result = backend.base.runtime.forward(
                            system=v4score.SYSTEM_SEALED, budget=budget, model=model, session=replaced,
                        )
                        metric = score_decomposition(
                            result.last_bin_predictions.numpy(), result.last_bin_targets.numpy(),
                        )
                        manual_float64_r2 = float(metric["variance_weighted_r2"])
                        governing_r2 = float(
                            backend.base.runtime.score_result(result, session=replaced)
                        )
                        metric["manual_float64_variance_weighted_r2"] = manual_float64_r2
                        metric["manual_float64_minus_governing_r2"] = manual_float64_r2 - governing_r2
                        metric["variance_weighted_r2"] = governing_r2
                        baseline_row = baseline[(surface, budget, session_name)]
                        if support == "C0_contiguous":
                            _require(result.prediction_sha256 == baseline_row["prediction_sha256"], "C0 V4 prediction parity failure")
                            _require(abs(float(metric["variance_weighted_r2"]) - float(baseline_row["r2"])) <= 1.0e-7, "C0 V4 R2 parity failure")
                        all_rows.append({
                            "surface": surface,
                            "session": session_name,
                            "budget": budget,
                            "support": support,
                            "n_windows": prepared.n_windows,
                            "prediction_sha256": result.prediction_sha256,
                            "target_sha256": prepared.target_sha256,
                            "normalized_side_sha256": side_hash,
                            "selected_indices_sha256": carrier["selected_indices_sha256"],
                            "selected_direction_counts": carrier["selected_direction_counts"],
                            "selected_trial_count": carrier["selected_trial_count"],
                            "index_span": carrier["index_span"],
                            "time_span_seconds": carrier["time_span_seconds"],
                            "design_condition": carrier["design_condition"],
                            **metric,
                        })
        state_after = backend.base.runtime.state_digest(model)
        _require(state_before == state_after, "sealed Cell-D model changed during diagnostic")
        from mc_maze.d_optimal_calibration_replay import build_receipt as build_d_optimal_receipt
        from mc_maze.d_optimal_calibration_replay import validate_receipt as validate_d_optimal_receipt

        d_optimal_receipt = build_d_optimal_receipt(d_optimal_payloads, seed=42, device="cpu")
        d_optimal_validation = validate_d_optimal_receipt(d_optimal_receipt)
        d_optimal_gate = {
            "receipt_sha256": d_optimal_receipt["receipt_sha256"],
            "validation": d_optimal_validation,
            "primary_gate_summary": d_optimal_receipt["primary_gate_summary"],
            "per_session": [
                {
                    "session": row["session_name"],
                    "M10_delta": row["per_budget"]["M10"]["primary_endpoint"]["paired_delta_d_optimal_minus_chronological"],
                    "M15_delta": row["per_budget"]["M15"]["primary_endpoint"]["paired_delta_d_optimal_minus_chronological"],
                }
                for row in d_optimal_receipt["session_results"]
            ],
        }
        cells: list[dict[str, object]] = []
        for surface in v4score.SURFACES:
            for budget in BUDGETS:
                for support in SUPPORTS:
                    rows = [row for row in all_rows if row["surface"] == surface and row["budget"] == budget and row["support"] == support]
                    cells.append({
                        "surface": surface,
                        "budget": budget,
                        "support": support,
                        "sessions": rows,
                        "summary": aggregate_session_rows(rows),
                    })
        paired: list[dict[str, object]] = []
        for surface in v4score.SURFACES:
            for budget in BUDGETS:
                by_support = {
                    support: [row for row in all_rows if row["surface"] == surface and row["budget"] == budget and row["support"] == support]
                    for support in SUPPORTS
                }
                roster = [row["session"] for row in by_support["C0_contiguous"]]
                for support in SUPPORTS[1:]:
                    other = {row["session"]: row for row in by_support[support]}
                    deltas = [float(other[name]["variance_weighted_r2"]) - float(by_support["C0_contiguous"][i]["variance_weighted_r2"]) for i, name in enumerate(roster)]
                    paired.append({
                        "surface": surface,
                        "budget": budget,
                        "contrast": support + "_minus_C0_contiguous",
                        "per_session": [{"session": name, "delta_r2": delta} for name, delta in zip(roster, deltas)],
                        "mean_delta_r2": float(np.mean(deltas)),
                        "median_delta_r2": float(np.median(deltas)),
                        "positive_sessions": int(np.sum(np.asarray(deltas) > 0.0)),
                        "session_count": len(deltas),
                    })
        receipt: dict[str, object] = {
            "schema": SCHEMA + "_receipt",
            "status": "DIAGNOSTIC_COMPLETE_NON_GOVERNING",
            "attempt_sha256": attempt_sha,
            "closure": closure,
            "v4_score_sha256": baseline_sha,
            "failed_predecessors": failed_predecessors,
            "v4_input_authority_sha256": input_sha,
            "sealed_cell_d_state_sha256": state_before,
            "supports": list(SUPPORTS),
            "budgets": list(BUDGETS),
            "cpu_budgets": list(CPU_BUDGETS),
            "cpu_carrier_rows": cpu_rows,
            "d_optimal_carrier_gate": d_optimal_gate,
            "cells": cells,
            "paired_contrasts": paired,
            "boundaries": {
                "same_model": True,
                "same_query_inputs": True,
                "same_m30_b3s_calibration": True,
                "sealed_ordinary_ols_normalizer": True,
                "target_optimizer_steps": 0,
                "target_backward_calls": 0,
                "target_update_calls": 0,
                "formal_opened": False,
                "training_authorized": False,
                "C2_and_C3_are_labeled_leakage_oracles": True,
            },
        }
        receipt_sha = _publish_pair(result_root, "receipt.json", receipt)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "TERMINAL",
            "attempt_sha256": attempt_sha,
            "receipt_sha256": receipt_sha,
            "closure": closure,
            "failed_predecessors": failed_predecessors,
            "formal_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "training_authorized": False,
        }
        terminal_sha = _publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {"receipt_sha256": receipt_sha, "terminal_sha256": terminal_sha, "receipt": receipt}
    except BaseException as error:
        failure = {
            "schema": SCHEMA + "_failure",
            "status": "DIAGNOSTIC_FAILED",
            "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_sha256": _sha(f"{type(error).__name__}: {error}".encode("utf-8")),
            "formal_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }
        _publish_pair(result_root, "failure.json", failure)
        raise
    finally:
        if backend is not None:
            backend.close()
