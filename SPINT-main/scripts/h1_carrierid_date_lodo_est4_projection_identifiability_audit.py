#!/usr/bin/env python3
"""Source-only, gauge-aware identifiability audit for H1-EST4.

EST4 source-trains a 16-by-176 neural projection (``pcs``), a 7-by-4
kinematic projection (``U``), an output prior (``mu``), and two positive
scalars.  Some numerical changes to those tensors are exact reparameteriza-
tions of the same estimator/consumer function.  This CPU-only audit therefore
records invariant quantities (rank, condition number, normalized spectra,
subspace principal angles and Procrustes residuals), rather than falsely
treating a raw basis rotation or common scale change as a performance result.

The audit reads exactly three immutable files from each of the five existing
Phase-1 source bundles: ``shared_source_manifest.json``,
``frozen_m4_plan.manifest.json``, and ``frozen_m4_plan.npz``.  It never opens
an NWB, outer-date recording, minival, formal test, checkpoint, trainer, CUDA
context, or GPU.  It is intentionally separate from the active EST4
preflight/data/model/terminal chain.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
BUNDLE_SCHEMA = "h1_carrierid_date_lodo_shared_source_manifest_v1"
BUNDLE_STATUS = "PASS_SOURCE_ONLY_SHARED_BUNDLE_NOT_LAUNCHED"
PLAN_SCHEMA = "h1_carrierid_date_lodo_frozen_m4_plan_v1"
AUDIT_SCHEMA = "h1_carrierid_date_lodo_est4_source_projection_identifiability_audit_v1"
AUDIT_STATUS = "PASS_SOURCE_ONLY_EST4_PROJECTION_AUDIT__CONSUMER_ATTRIBUTION_BLOCKED"
DEFAULT_BUNDLE_ROOT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/source_bundles_v1"
DEFAULT_OUTPUT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_est4_projection_identifiability/H1_CARRIERID_DATE_LODO_EST4_SOURCE_PROJECTION_IDENTIFIABILITY_AUDIT_v1.json"
DEFAULT_COMPONENT = ROOT / "src/models/components/h1_carrierid_est4_spint.py"
DEFAULT_CONSUMER = ROOT / "src/models/components/h1_carrierid_spint.py"
Q, N, K, D = 16, 176, 7, 4


class Est4ProjectionAuditError(RuntimeError):
    """The source-only audit cannot establish one of its frozen invariants."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4ProjectionAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} is missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be immutable mode 0444: {path}")


def _read_immutable_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _immutable(path, label)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Est4ProjectionAuditError(f"{label} is invalid JSON: {path}") from error
    _need(isinstance(body, dict), f"{label} must be a JSON object: {path}")
    return body, _sha256_file(path)


def _finite(value: float, label: str) -> float:
    _need(math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _float_list(values: np.ndarray) -> list[float]:
    return [_finite(float(item), "numeric audit output") for item in np.asarray(values).ravel()]


def _matrix_summary(matrix: np.ndarray, *, name: str, expected_rank: int) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    _need(values.ndim == 2 and np.isfinite(values).all(), f"{name} matrix is malformed")
    singular = np.linalg.svd(values, compute_uv=False)
    _need(singular.size > 0 and singular[0] > 0.0, f"{name} has zero spectral norm")
    tolerance = max(values.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    _need(rank == expected_rank, f"{name} rank={rank}, expected {expected_rank}")
    retained = singular[:rank]
    condition = float(retained[0] / retained[-1])
    return {
        "shape": list(values.shape), "rank": rank, "rank_tolerance": float(tolerance),
        "singular_values": _float_list(retained),
        "normalized_singular_values": _float_list(retained / retained[0]),
        "condition_number": condition,
        "frobenius_norm": float(np.linalg.norm(values, ord="fro")),
        "parameter_rms": float(math.sqrt(float(np.mean(np.square(values))))),
    }


def _row_space_basis(matrix: np.ndarray, *, expected_rank: int, label: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    q, _ = np.linalg.qr(values.T, mode="reduced")
    _need(q.shape[1] >= expected_rank, f"{label} QR rank drift")
    basis = q[:, :expected_rank]
    _need(np.allclose(basis.T @ basis, np.eye(expected_rank), atol=1e-10, rtol=1e-10),
          f"{label} basis is not orthonormal")
    return basis


def _principal_angle_summary(first: np.ndarray, second: np.ndarray, *, rank: int, label: str) -> dict[str, Any]:
    left = _row_space_basis(first, expected_rank=rank, label=f"{label} left")
    right = _row_space_basis(second, expected_rank=rank, label=f"{label} right")
    cosines = np.clip(np.linalg.svd(left.T @ right, compute_uv=False), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    projector_distance = float(np.linalg.norm(left @ left.T - right @ right.T, ord="fro"))
    return {
        "principal_cosines": _float_list(cosines), "principal_angles_degrees": _float_list(angles),
        "max_principal_angle_degrees": float(np.max(angles)),
        "rms_principal_angle_degrees": float(math.sqrt(float(np.mean(np.square(angles))))),
        "projector_frobenius_distance": projector_distance,
    }


def _scale_rotation_procrustes(first: np.ndarray, second: np.ndarray, *, label: str) -> dict[str, Any]:
    """Residual after the exact positive-scale/orthogonal-basis gauge."""

    left, right = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    _need(left.shape == right.shape and left.ndim == 2, f"{label} Procrustes shape drift")
    left_norm, right_norm = np.linalg.norm(left, ord="fro"), np.linalg.norm(right, ord="fro")
    _need(left_norm > 0.0 and right_norm > 0.0, f"{label} Procrustes requires nonzero matrices")
    a, b = left / left_norm, right / right_norm
    # max_Q trace(A^T Q B) is the nuclear norm of A B^T.  The corresponding
    # residual is invariant to a common positive scale and an orthogonal basis
    # change in the leading (row) dimension.
    aligned_trace = float(np.sum(np.linalg.svd(a @ b.T, compute_uv=False)))
    squared = max(0.0, 2.0 - 2.0 * aligned_trace)
    return {
        "best_positive_scale_second_to_first": float(left_norm / right_norm),
        "orthogonal_scale_aligned_relative_residual": float(math.sqrt(squared)),
        "raw_frobenius_norm_ratio_first_over_second": float(left_norm / right_norm),
    }


def _pairwise_subspaces(rows: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {"pcs_row_space": [], "effective_neural_row_space": [], "U_column_space": []}
    for first_date, second_date in itertools.combinations(DATES, 2):
        first, second = rows[first_date], rows[second_date]
        pair = {"dates": [first_date, second_date]}
        output["pcs_row_space"].append({
            **pair,
            **_principal_angle_summary(first["pcs"], second["pcs"], rank=Q, label="pcs"),
            "orthogonal_scale_alignment": _scale_rotation_procrustes(first["pcs"], second["pcs"], label="pcs"),
        })
        output["effective_neural_row_space"].append({
            **pair,
            **_principal_angle_summary(first["effective_pcs"], second["effective_pcs"], rank=Q, label="effective pcs"),
            "orthogonal_scale_alignment": _scale_rotation_procrustes(first["effective_pcs"], second["effective_pcs"], label="effective pcs"),
        })
        # U maps seven kinematic coordinates into four carrier coordinates;
        # its meaningful object under output-coordinate rotations is its column
        # space, represented as the row space of U.T.
        output["U_column_space"].append({
            **pair,
            **_principal_angle_summary(first["U"].T, second["U"].T, rank=D, label="U"),
            "orthogonal_scale_alignment": _scale_rotation_procrustes(first["U"].T, second["U"].T, label="U"),
        })
    return output


def _constraint_audit(component_path: Path, consumer_path: Path) -> dict[str, Any]:
    """Static source audit: do not import Torch or construct a model."""

    _need(component_path.is_file() and consumer_path.is_file(), "EST4 component/consumer source is missing")
    component = component_path.read_text(encoding="utf-8")
    consumer = consumer_path.read_text(encoding="utf-8")
    for snippet in (
        "self.pcs = nn.Parameter", "self.U = nn.Parameter", "self.mu = nn.Parameter",
        "self.log_lambda = nn.Parameter", "self.log_tau2 = nn.Parameter",
        "torch.exp(self.log_lambda).clamp_min(EST4_MIN_LAMBDA)",
        "torch.exp(self.log_tau2).clamp_min(EST4_MIN_TAU2)",
    ):
        _need(snippet in component, f"EST4 source constraint contract drift: missing {snippet!r}")
    _need("nn.Linear(self.carrier_hidden_dim + self.carrier_dim, self.carrier_hidden_dim)" in consumer,
          "CarrierID trainable four-dimensional consumer input contract drift")
    return {
        "component_path": str(component_path.resolve()), "component_sha256": _sha256_file(component_path),
        "consumer_path": str(consumer_path.resolve()), "consumer_sha256": _sha256_file(consumer_path),
        "pcs": "trainable nn.Parameter; no Stiefel/orthogonal/fixed-norm constraint",
        "U": "trainable nn.Parameter; no orthogonal/fixed-norm constraint",
        "mu": "trainable nn.Parameter; no norm/basis constraint",
        "lambda": "trainable log_lambda; exp then lower-clamped positive, no upper/fixed-scale constraint",
        "tau2": "trainable log_tau2; exp then lower-clamped positive, no upper/fixed-scale constraint",
        "consumer": "CarrierID carrier_post_pool begins with a trainable affine map from four carrier coordinates",
    }


def _load_bundle(bundle_root: Path, date: str) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    directory = (bundle_root / date).resolve()
    shared_path = directory / "shared_source_manifest.json"
    plan_manifest_path = directory / "frozen_m4_plan.manifest.json"
    plan_array_path = directory / "frozen_m4_plan.npz"
    shared, shared_sha = _read_immutable_json(shared_path, f"{date} source bundle manifest")
    manifest, manifest_sha = _read_immutable_json(plan_manifest_path, f"{date} frozen-plan manifest")
    _immutable(plan_array_path, f"{date} frozen-plan array")
    _need(shared.get("schema") == BUNDLE_SCHEMA and shared.get("status") == BUNDLE_STATUS and shared.get("outer_date") == date,
          f"{date} source bundle schema/status/date drift")
    scope = shared.get("source_only_scope")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0 and scope.get("target_bytes_read") == 0 and
          scope.get("trainer_constructed_or_launched") is False and scope.get("cuda_constructed_or_launched") is False,
          f"{date} source bundle scope is not target-free/CPU-only")
    frozen = shared.get("frozen_plan")
    _need(isinstance(frozen, Mapping) and Path(str(frozen.get("manifest_path", ""))).resolve() == plan_manifest_path and
          frozen.get("manifest_sha256") == manifest_sha, f"{date} frozen-plan binding drift")
    # ``raw_plan_sha256`` is the upstream analytic-plan provenance hash, not
    # the byte hash of this persisted ``.npz`` container.  The persisted plan
    # is instead bound by the shared-manifest hash plus the per-array hashes
    # checked below; conflating these two domains would falsely reject all
    # real Phase-1 bundles.
    _need(manifest.get("schema") == PLAN_SCHEMA and manifest.get("outer_date") == date and
          isinstance(manifest.get("raw_plan_sha256"), str) and len(manifest["raw_plan_sha256"]) == 64,
          f"{date} plan manifest provenance drift")
    try:
        with np.load(plan_array_path, allow_pickle=False) as archive:
            expected = {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}
            _need(set(archive.files) == expected, f"{date} plan array members drift")
            arrays = {name: np.asarray(archive[name], dtype=np.float64) for name in expected}
    except (OSError, ValueError) as error:
        raise Est4ProjectionAuditError(f"{date} cannot read frozen source plan") from error
    _need(arrays["mean"].shape == (N,) and arrays["scale"].shape == (N,) and arrays["pcs"].shape == (Q, N) and
          arrays["U"].shape == (K, D) and arrays["mu"].shape == (D,) and int(arrays["q"].item()) == Q,
          f"{date} EST4 frozen plan shape/q drift")
    _need(np.isfinite(arrays["mean"]).all() and np.isfinite(arrays["scale"]).all() and np.all(arrays["scale"] > 0.0),
          f"{date} source standardizer is invalid")
    _need(all(np.isfinite(arrays[name]).all() for name in ("pcs", "U", "mu", "lambda", "tau2")) and
          float(arrays["lambda"].item()) > 0.0 and float(arrays["tau2"].item()) > 0.0,
          f"{date} EST4 projection/scalar is invalid")
    claimed_arrays = manifest.get("array_sha256")
    _need(isinstance(claimed_arrays, Mapping), f"{date} plan array hashes absent")
    for name in ("mean", "scale", "pcs", "U", "mu"):
        _need(claimed_arrays.get(name) == _array_sha256(arrays[name]), f"{date} plan {name} hash drift")
    normalizer = shared.get("normalizer")
    _need(isinstance(normalizer, Mapping) and _finite(normalizer.get("denominator"), f"{date} source RMS denominator") > 0.0,
          f"{date} source RMS normalizer invalid")
    source_files = shared.get("source_files")
    _need(isinstance(source_files, list) and len(source_files) == shared.get("source_session_count") and
          all(isinstance(row, Mapping) and row.get("role") == "source_heldin_calib" for row in source_files),
          f"{date} bundle source-file scope drift")
    provenance = {
        "bundle_directory": str(directory),
        "shared_source_manifest": {"path": str(shared_path), "sha256": shared_sha},
        "frozen_plan_manifest": {"path": str(plan_manifest_path), "sha256": manifest_sha},
        "frozen_plan_array": {"path": str(plan_array_path), "sha256": _sha256_file(plan_array_path)},
        "source_session_count": int(shared["source_session_count"]),
        "source_session_names": list(shared["source_sessions"]),
        "source_rms_denominator": float(normalizer["denominator"]),
    }
    return provenance, arrays, shared


def _per_date_metrics(arrays: Mapping[str, np.ndarray], shared: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pcs, scale, U, mu = arrays["pcs"], arrays["scale"], arrays["U"], arrays["mu"]
    effective_pcs = pcs / scale.reshape(1, -1)
    gram = pcs @ pcs.T
    result = {
        "pcs": {
            **_matrix_summary(pcs, name="pcs", expected_rank=Q),
            "row_gram_max_abs_deviation_from_identity": float(np.max(np.abs(gram - np.eye(Q)))),
            "row_gram_frobenius_deviation_from_identity": float(np.linalg.norm(gram - np.eye(Q), ord="fro")),
        },
        "effective_neural_projection_pcs_over_scale": _matrix_summary(effective_pcs, name="pcs/scale", expected_rank=Q),
        "U": _matrix_summary(U, name="U", expected_rank=D),
        "mu": {"shape": list(mu.shape), "parameter_rms": float(math.sqrt(float(np.mean(np.square(mu))))),
               "l2_norm": float(np.linalg.norm(mu))},
        "positive_scalars": {"lambda": float(arrays["lambda"].item()), "tau2": float(arrays["tau2"].item()),
                             "lambda_over_pcs_frobenius_squared": float(arrays["lambda"].item() / np.square(np.linalg.norm(pcs, ord="fro"))),
                             "tau2_over_U_frobenius_squared": float(arrays["tau2"].item() / np.square(np.linalg.norm(U, ord="fro")))},
        "source_rms_scale": {
            "frozen_source_normalizer": float(shared["normalizer"]["denominator"]),
            "interpretation": "This is the ordinary frozen source-carrier normalization denominator, not a measured trained EST4 carrier-output RMS.",
        },
    }
    return result, {"pcs": pcs, "effective_pcs": effective_pcs, "U": U}


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    _need(not path.exists(), f"refusing to overwrite immutable EST4 audit receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o444)
    _immutable(path, "new EST4 audit receipt")
    return _sha256_file(path)


def run(*, bundle_root: Path, component_path: Path, consumer_path: Path, output: Path) -> dict[str, Any]:
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
          "EST4 projection audit requires CUDA_VISIBLE_DEVICES unset/empty")
    _need(not output.exists(), f"EST4 projection audit output already exists: {output}")
    constraints = _constraint_audit(component_path.resolve(), consumer_path.resolve())
    provenance: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}
    for date in DATES:
        item, arrays, shared = _load_bundle(bundle_root.resolve(), date)
        date_metrics, date_matrices = _per_date_metrics(arrays, shared)
        provenance[date], metrics[date], matrices[date] = item, date_metrics, date_matrices
    body = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "mode": "explicit_cpu_only_source_bundle_audit_no_nwb_no_trainer_no_gpu_no_target",
        "source_scope": {
            "required_outer_dates": list(DATES), "bundle_root": str(bundle_root.resolve()),
            "files_read_per_date": ["shared_source_manifest.json", "frozen_m4_plan.manifest.json", "frozen_m4_plan.npz"],
            "outer_target_recordings_opened": 0, "outer_target_bytes_read": 0,
            "formal_test_opened": 0, "formal_test_bytes_read": 0,
            "nwb_opened": 0, "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
        },
        "source_bundles": provenance,
        "learned_projection_constraints": constraints,
        "per_date_initialization_metrics": metrics,
        "cross_date_gauge_aware_subspace_metrics": _pairwise_subspaces(matrices),
        "identifiability": {
            "exact_pcs_gauge": "For alpha>0 and orthogonal Q16, pcs -> alpha*Q16*pcs and lambda -> alpha^2*lambda leave the ridge prediction, raw_rows, covariance factor and carrier unchanged while lambda is above its lower clamp.",
            "exact_output_consumer_gauge": "For beta>0 and orthogonal R4, U -> beta*U*R4, mu -> beta*mu*R4, tau2 -> beta^2*tau2, and the four carrier columns of the trainable first consumer affine map -> W_carrier*R4/beta preserve the full pre-ReLU consumer input while tau2 is above its lower clamp.",
            "identifiable_or_gauge_invariant_quantities": [
                "pcs/effective-pcs rank, condition number, normalized singular spectrum, and neural row-space principal angles",
                "U rank, condition number, normalized singular spectrum, and kinematic column-space principal angles",
                "Procrustes residual after positive-scale/orthogonal alignment",
                "lambda/||pcs||_F^2 and tau2/||U||_F^2 under the stated coupled scale gauges",
            ],
            "not_identifiable_from_raw_parameter_coordinates": [
                "pcs row orientation/sign and absolute pcs RMS independently of lambda",
                "U/carrier-coordinate orientation and absolute U or mu RMS independently of tau2 and consumer carrier weights",
                "a trained EST4 carrier-output RMS from a frozen initialization plan alone",
            ],
        },
        "fail_closed_judgment": {
            "existing_source_receipts_sufficient_for": [
                "source-only/no-target provenance of the five frozen plans",
                "fresh matched whole-pipeline arm execution when each later checkpoint/evaluator is independently bound",
            ],
            "existing_source_receipts_not_sufficient_for": [
                "interpreting raw pcs/U/mu/lambda/tau2 coordinates as a stable learned functional mechanism",
                "attributing any EST4-vs-baseline predictive difference to the estimator rather than joint estimator-consumer reparameterization",
                "claiming trained EST4 carrier output scale/RMS without a deterministic source-checkpoint forward audit",
            ],
            "est4_consumer_attribution": "BLOCKED_FAIL_CLOSED",
            "whole_pipeline_predictive_comparison": "NOT_YET_MEASURED; gauge nonidentifiability does not invalidate a future fresh matched B-C/L-C prediction comparison, but it prevents a parameter-mechanism claim.",
            "required_before_unblocking_consumer_attribution": [
                "freeze either a canonical gauge constraint (for example, row-orthonormal pcs plus fixed scale and a canonical U/output convention) before source training, or explicitly limit claims to whole-pipeline prediction",
                "add a source-only post-training checkpoint audit of gauge-invariant condition/rank/spectrum/subspace quantities and deterministic source carrier-output RMS; bind it before any target evaluator",
                "keep a fresh matched ordinary H-C/CI32 consumer reference under the same source date, schedule, normalizer, seed and initial state",
            ],
        },
        "code_sha256": {"audit": _sha256_file(Path(__file__).resolve()), "component": _sha256_file(component_path),
                          "consumer": _sha256_file(consumer_path)},
    }
    digest = _write_immutable(output.resolve(), body)
    return {"status": AUDIT_STATUS, "receipt_path": str(output.resolve()), "receipt_sha256": digest,
            "est4_consumer_attribution": "BLOCKED_FAIL_CLOSED"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source-only-audit", action="store_true")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--component", type=Path, default=DEFAULT_COMPONENT)
    parser.add_argument("--consumer", type=Path, default=DEFAULT_CONSUMER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.run_source_only_audit:
        raise SystemExit("refusing implicit source-bundle access; pass --run-source-only-audit explicitly")
    print(json.dumps(run(bundle_root=args.bundle_root, component_path=args.component, consumer_path=args.consumer,
                         output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
