#!/usr/bin/env python3
"""Source-only strength audit for H1's existing label-rotation carrier null.

``label_rotation_carrier`` has a deliberately narrow intervention: for every
100-ms support block sequence in a trial it applies a deterministic, nonzero
cyclic roll to the *dense velocity labels*, then refits the ordinary frozen
four-dimensional carrier.  It is consequently a temporal-misalignment
control.  It is not, by construction, an exchangeable ``labels contain no
information`` null.

This script quantifies how strong that misalignment actually is on the five
immutable date-LODO Phase-1 source bundles.  It reads only the sessions named
as a bundle's source sessions; the same date's outer recordings are neither
indexed nor opened.  No model, Trainer, checkpoint, target evaluator, or CUDA
object is constructed.

The audit answers three limited, source-only questions:

1. How similar are the true and cyclically rolled velocity labels at the
   carrier-fit inputs?
2. How similar are the resulting raw and source-RMS-normalized carriers?
3. Could one fixed 4x4 linear or orthogonal map, learned on other *source*
   sessions, undo the label-rotation carrier change on a held-out source
   session?

It cannot establish predictive efficacy, nor can it convert a temporal roll
into a pure label-deletion null.  In particular, this receipt must not be used
to open an outer date, choose a model, or authorize a GPU job.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_m4_eb_pilot import (  # noqa: E402
    EXPECTED_NEURONS,
    ROTATION_SEED,
    SUPPORT_TRIALS,
    VELOCITY_DIM,
    H1PilotRecord,
    array_sha256,
    carrier_sha256,
    fit_frozen_carrier,
    label_rotation_carrier,
    load_record,
    session_date,
    sha256_file,
)
from src.h1_m4_cce_contract import (  # noqa: E402
    CONFIRMATORY_DATES,
    NORMALIZER_FLOOR,
    canonical_sha256,
)


AUDIT_SCHEMA = "h1_carrierid_label_rotation_null_strength_source_audit_v1"
AUDIT_STATUS = "PASS_H1_CARRIERID_LABEL_ROTATION_SOURCE_NULL_STRENGTH_AUDIT__TEMPORAL_MISALIGNMENT_ONLY"
BUNDLE_ROOT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/source_bundles_v1"
DEFAULT_DATA_ROOT = ROOT / "data/000954"
DEFAULT_OUTPUT = (
    ROOT / "pilot_artifacts/h1_carrierid_label_rotation_null_strength"
    / "H1_CARRIERID_LABEL_ROTATION_SOURCE_NULL_STRENGTH_AUDIT_v1.json"
)


class LabelRotationAuditError(RuntimeError):
    """A source-only audit invariant failed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise LabelRotationAuditError(message)


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file(), f"{label} missing: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} must be mode 0444: {path}")


def _read_immutable_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _immutable(path, label)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LabelRotationAuditError(f"{label} is not valid JSON: {path}") from error
    _need(isinstance(body, dict), f"{label} must be a JSON object: {path}")
    return body, sha256_file(path)


def _read_immutable_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    _immutable(path, label)
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise LabelRotationAuditError(f"{label} is invalid: {path}") from error


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    _need(x.shape == y.shape and x.size > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "invalid cosine inputs")
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    _need(denominator > 0.0 and math.isfinite(denominator), "undefined cosine due to zero norm")
    return float(np.dot(x, y) / denominator)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    _need(x.shape == y.shape and x.size >= 2 and np.isfinite(x).all() and np.isfinite(y).all(), "invalid Pearson inputs")
    x = x - float(x.mean())
    y = y - float(y.mean())
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    _need(denominator > 0.0 and math.isfinite(denominator), "undefined Pearson due to zero centered norm")
    return float(np.dot(x, y) / denominator)


def _relative_error(prediction: np.ndarray, target: np.ndarray) -> float:
    value = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _need(value.shape == truth.shape and value.size > 0, "invalid relative-error inputs")
    denominator = float(np.linalg.norm(truth))
    _need(denominator > 0.0 and math.isfinite(denominator), "undefined relative error due to zero target norm")
    return float(np.linalg.norm(value - truth) / denominator)


def _quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    _need(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(), "empty/nonfinite summary values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _row_cosines(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    _need(x.shape == y.shape and x.ndim == 2 and x.shape[1] == 4, "carrier row cosine requires [N,4]")
    values: list[float] = []
    for row_x, row_y in zip(x, y):
        denominator = float(np.linalg.norm(row_x) * np.linalg.norm(row_y))
        if denominator > 1.0e-18:
            values.append(float(np.dot(row_x, row_y) / denominator))
    _need(values, "all carrier rows had zero norm")
    return _quantiles(values)


def rotation_shift(*, session_name: str, trial_number: float, block_count: int) -> int:
    """Return the literal existing H1 label-rotation shift, fail-closed."""

    _need(block_count >= 2, "label rotation needs at least two support blocks")
    # This byte string deliberately matches the *existing* legacy
    # ``label_rotation_carrier`` implementation, including its replicate-0
    # suffix.  Do not replace it with a more descriptive token: the point of
    # this audit is to quantify the exact intervention that was trained.
    token = hashlib.sha256(
        f"{ROTATION_SEED}|{session_name}|{float(trial_number)}|0".encode("utf-8")
    ).digest()
    shift = 1 + int.from_bytes(token[:8], "big") % (int(block_count) - 1)
    _need(1 <= shift < int(block_count), "label rotation shift was identity/out of range")
    return int(shift)


def rotation_overrides(record: H1PilotRecord, values: Sequence[float]) -> tuple[dict[float, np.ndarray], list[dict[str, Any]]]:
    """Build the exact existing LS labels plus auditable shift metadata."""

    _need(len(values) == SUPPORT_TRIALS, "LS support must contain exactly M=4 trials")
    override: dict[float, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for value in values:
        trial = record.blocks_for(float(value))
        count = int(trial.velocity.shape[0])
        shift = rotation_shift(session_name=record.session_name, trial_number=trial.trial_number, block_count=count)
        override[float(value)] = np.roll(np.asarray(trial.velocity, dtype=np.float64), shift, axis=0)
        metadata.append({
            "session_name": record.session_name,
            "trial_number": float(trial.trial_number),
            "block_count": count,
            "shift_blocks": shift,
            "shift_fraction_of_trial": float(shift / count),
        })
    return override, metadata


@dataclass(frozen=True)
class SourcePlan:
    """Minimal immutable plan interface accepted by the legacy analytic fitter."""

    outer_date: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    q: int
    ridge_lambda: float
    U: np.ndarray
    mu: np.ndarray
    tau2: float
    raw_plan_sha256: str
    raw_receipt_sha256: str
    eb_receipt_sha256: str
    transform_sha256: str


@dataclass(frozen=True)
class Bundle:
    outer_date: str
    directory: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    plan: SourcePlan
    raw_entries: tuple[Mapping[str, Any], ...]
    raw_carriers: np.ndarray
    denominator: float
    normalizer_sha256: str


def _source_nwb_path(data_root: Path, session_name: str) -> Path:
    _need(session_name.startswith("ses-"), f"invalid H1 session name: {session_name!r}")
    path = data_root / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_{session_name}.nwb"
    _need(path.is_file(), f"declared source NWB is absent: {path}")
    return path.resolve()


def _load_bundle(bundle_root: Path, outer_date: str) -> Bundle:
    directory = (bundle_root / outer_date).resolve()
    manifest_path = directory / "shared_source_manifest.json"
    manifest, manifest_sha = _read_immutable_json(manifest_path, f"{outer_date} shared source manifest")
    _need(manifest.get("schema") == "h1_carrierid_date_lodo_shared_source_manifest_v1", "source manifest schema drift")
    _need(manifest.get("status") == "PASS_SOURCE_ONLY_SHARED_BUNDLE_NOT_LAUNCHED", "source manifest status drift")
    _need(manifest.get("outer_date") == outer_date, "source manifest outer-date drift")
    source_sessions = tuple(str(name) for name in manifest.get("source_sessions", ()))
    _need(source_sessions and all(session_date(name) != outer_date for name in source_sessions),
          "outer-date recording appears in a declared source partition")
    scope = manifest.get("source_only_scope")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0 and scope.get("target_bytes_read") == 0,
          "Phase-1 bundle does not establish source-only provenance")
    _need(scope.get("cuda_constructed_or_launched") is False and scope.get("trainer_constructed_or_launched") is False,
          "Phase-1 bundle source scope drifted into CUDA/Trainer")

    plan_manifest_path = directory / "frozen_m4_plan.manifest.json"
    plan_manifest, plan_manifest_sha = _read_immutable_json(plan_manifest_path, f"{outer_date} frozen plan manifest")
    bound_plan = manifest.get("frozen_plan")
    _need(isinstance(bound_plan, Mapping) and bound_plan.get("manifest_path") == str(plan_manifest_path) and
          bound_plan.get("manifest_sha256") == plan_manifest_sha, "shared manifest/frozen plan binding drift")
    _need(plan_manifest.get("outer_date") == outer_date and tuple(plan_manifest.get("source_sessions", ())) == source_sessions,
          "frozen plan source partition drift")
    plan_arrays = _read_immutable_npz(directory / "frozen_m4_plan.npz", f"{outer_date} frozen plan arrays")
    _need(set(plan_arrays) == {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}, "frozen plan array names drift")
    expected_shapes = {"mean": (EXPECTED_NEURONS,), "scale": (EXPECTED_NEURONS,), "pcs": (16, EXPECTED_NEURONS),
                       "U": (VELOCITY_DIM, 4), "mu": (4,)}
    for name, shape in expected_shapes.items():
        array = np.asarray(plan_arrays[name], dtype=np.float64)
        _need(array.shape == shape and np.isfinite(array).all(), f"frozen plan {name} shape/finite drift")
        _need(array_sha256(array) == plan_manifest["array_sha256"][name], f"frozen plan {name} hash drift")
    q = int(np.asarray(plan_arrays["q"]).item())
    ridge_lambda = float(np.asarray(plan_arrays["lambda"], dtype=np.float64).item())
    tau2 = float(np.asarray(plan_arrays["tau2"], dtype=np.float64).item())
    _need(q == int(plan_manifest["q"]) == 16 and ridge_lambda == float(plan_manifest["lambda"]) == 100.0 and
          tau2 == float(plan_manifest["tau2"]) and tau2 > 0.0, "frozen plan scalar drift")
    plan = SourcePlan(
        outer_date=outer_date,
        source_sessions=source_sessions,
        source_input_sha256=tuple(str(item) for item in plan_manifest["source_input_sha256"]),
        mean=np.asarray(plan_arrays["mean"], dtype=np.float64), scale=np.asarray(plan_arrays["scale"], dtype=np.float64),
        pcs=np.asarray(plan_arrays["pcs"], dtype=np.float64), q=q, ridge_lambda=ridge_lambda,
        U=np.asarray(plan_arrays["U"], dtype=np.float64), mu=np.asarray(plan_arrays["mu"], dtype=np.float64), tau2=tau2,
        raw_plan_sha256=str(plan_manifest["raw_plan_sha256"]), raw_receipt_sha256=str(plan_manifest["raw_receipt_sha256"]),
        eb_receipt_sha256=str(plan_manifest["eb_receipt_sha256"]), transform_sha256=str(plan_manifest["transform_sha256"]),
    )

    cache_manifest_path = directory / "source_m4_carriers.manifest.json"
    cache_manifest, cache_manifest_sha = _read_immutable_json(cache_manifest_path, f"{outer_date} source carrier manifest")
    bound_cache = manifest.get("carrier_cache")
    _need(isinstance(bound_cache, Mapping) and bound_cache.get("manifest_path") == str(cache_manifest_path) and
          bound_cache.get("manifest_sha256") == cache_manifest_sha, "shared manifest/source carrier binding drift")
    _need(cache_manifest.get("outer_date") == outer_date and tuple(cache_manifest.get("source_sessions", ())) == source_sessions and
          cache_manifest.get("transform_sha256") == plan.transform_sha256, "source cache partition/transform drift")
    raw = _read_immutable_npz(directory / "source_m4_carriers.npz", f"{outer_date} source carrier arrays")
    _need(tuple(raw) == ("carriers",), "source carrier array names drift")
    raw_carriers = np.asarray(raw["carriers"], dtype=np.float64)
    rows = cache_manifest.get("entries")
    _need(isinstance(rows, list) and len(rows) == raw_carriers.shape[0] and raw_carriers.shape[1:] == (EXPECTED_NEURONS, 4) and
          np.isfinite(raw_carriers).all(), "source carrier cache shape/entry drift")
    _need(str(cache_manifest.get("carrier_shape")) == str(list(raw_carriers.shape)), "source carrier shape manifest drift")
    core = dict(cache_manifest)
    claimed_cache_sha = str(core.pop("cache_sha256", ""))
    _need(canonical_sha256(core) == claimed_cache_sha == str(bound_cache.get("cache_sha256")), "source cache canonical hash drift")
    for index, row in enumerate(rows):
        _need(isinstance(row, Mapping) and str(row.get("session")) in source_sessions and
              len(tuple(row.get("trial_values", ()))) == SUPPORT_TRIALS, "malformed source carrier entry")
        _need(carrier_sha256(raw_carriers[index]) == str(row.get("carrier_sha256")), "raw carrier entry hash drift")

    normalizer_row = manifest.get("normalizer")
    _need(isinstance(normalizer_row, Mapping), "source manifest lacks normalizer")
    normalizer_path = directory / "source_rms_normalizer.manifest.json"
    normalizer, normalizer_file_sha = _read_immutable_json(normalizer_path, f"{outer_date} source normalizer")
    _need(normalizer_row.get("manifest_path") == str(normalizer_path) and normalizer_row.get("manifest_file_sha256") == normalizer_file_sha,
          "normalizer manifest binding drift")
    denominator = float(normalizer["denominator"])
    _need(math.isfinite(denominator) and denominator == max(float(normalizer["s_src"]), NORMALIZER_FLOOR) and denominator > 0.0,
          "normalizer denominator drift")
    _need(str(normalizer["source_cache_sha256"]) == claimed_cache_sha and int(normalizer["entries"]) == len(rows) and
          int(normalizer["rows"]) == EXPECTED_NEURONS and int(normalizer["dims"]) == 4,
          "normalizer/cache linkage drift")

    normalized_manifest_path = directory / "normalized_source_m4_carriers.manifest.json"
    normalized_manifest, normalized_manifest_sha = _read_immutable_json(normalized_manifest_path, f"{outer_date} normalized cache manifest")
    normalized_row = manifest.get("normalized_cache")
    _need(isinstance(normalized_row, Mapping) and normalized_row.get("manifest_path") == str(normalized_manifest_path) and
          normalized_row.get("manifest_sha256") == normalized_manifest_sha, "normalized cache manifest binding drift")
    normalized = _read_immutable_npz(directory / "normalized_source_m4_carriers.npz", f"{outer_date} normalized carrier arrays")
    _need(tuple(normalized) == ("carriers",), "normalized carrier array names drift")
    normalized_carriers = np.asarray(normalized["carriers"], dtype=np.float64)
    _need(np.array_equal(normalized_carriers, raw_carriers / denominator), "normalized cache does not equal bound source scalar transform")
    _need(array_sha256(normalized_carriers) == normalized_manifest["normalized_carriers_sha256"] == normalized_row["normalized_carriers_sha256"],
          "normalized cache tensor hash drift")
    return Bundle(
        outer_date=outer_date, directory=directory, manifest=manifest, manifest_sha256=manifest_sha, plan=plan,
        raw_entries=tuple(rows), raw_carriers=raw_carriers, denominator=denominator,
        normalizer_sha256=str(normalizer["normalizer_sha256"]),
    )


def _load_bundle_source_records(bundle: Bundle, data_root: Path) -> tuple[dict[str, H1PilotRecord], list[dict[str, Any]]]:
    """Open only the immutable bundle's named non-outer source recordings."""

    source_files = bundle.manifest.get("source_files")
    _need(isinstance(source_files, list) and len(source_files) == len(bundle.plan.source_sessions), "source file receipt drift")
    by_name = {str(row.get("session")): row for row in source_files if isinstance(row, Mapping)}
    _need(tuple(by_name) == bundle.plan.source_sessions, "source file receipt ordering/partition drift")
    records: dict[str, H1PilotRecord] = {}
    audit: list[dict[str, Any]] = []
    for index, name in enumerate(bundle.plan.source_sessions):
        _need(session_date(name) != bundle.outer_date, "attempt to open the bundle outer-date recording")
        path = _source_nwb_path(data_root, name)
        expected = by_name[name]
        _need(str(expected.get("role")) == "source_heldin_calib" and str(expected.get("date")) == session_date(name),
              "source receipt role/date drift")
        before = sha256_file(path)
        _need(before == str(expected.get("sha256")) == bundle.plan.source_input_sha256[index], "source NWB hash drift")
        record = load_record(path)
        _need(record.session_name == name and record.date != bundle.outer_date and record.input_sha256 == before,
              "source record identity/hash/outer-date drift")
        records[name] = record
        audit.append({"session_name": name, "sha256": before, "path": str(path), "role": "source_heldin_calib"})
    _need(tuple(records) == bundle.plan.source_sessions, "source record load order drift")
    return records, audit


def _fit_orthogonal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    _need(x.ndim == y.ndim == 2 and x.shape == y.shape and x.shape[1] == 4 and x.shape[0] >= 4,
          "orthogonal fit requires paired [samples>=4,4]")
    left, _singular, right_t = np.linalg.svd(x.T @ y, full_matrices=False)
    result = left @ right_t
    _need(result.shape == (4, 4) and np.isfinite(result).all() and np.allclose(result.T @ result, np.eye(4), atol=1e-10),
          "orthogonal Procrustes fit failed")
    return result


def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    _need(x.ndim == y.ndim == 2 and x.shape == y.shape and x.shape[1] == 4 and x.shape[0] >= 4,
          "linear fit requires paired [samples>=4,4]")
    result, _residuals, rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    _need(rank == 4 and result.shape == (4, 4) and np.isfinite(result).all(), "common linear map is rank deficient")
    return result


def _alignment_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"frobenius_cosine": _cosine(x, y), "flattened_pearson": _pearson(x, y), "relative_frobenius_error": _relative_error(x, y)}


def _common_inverse_audit(per_session: Mapping[str, Mapping[str, list[np.ndarray]]]) -> dict[str, Any]:
    """Source-session LO(S)O recovery of full from cyclically rolled carrier.

    The map has no intercept, is fit only on other source sessions, and is
    evaluated on a whole held-out source session.  This measures a possible
    shared representation equivalence; it is not a decoder experiment.
    """

    sessions = tuple(per_session)
    _need(len(sessions) >= 2, "common-map audit needs at least two source sessions")
    held_rows: list[dict[str, Any]] = []
    for held in sessions:
        x_train = np.concatenate([np.concatenate(per_session[name]["label"], axis=0) for name in sessions if name != held], axis=0)
        y_train = np.concatenate([np.concatenate(per_session[name]["full"], axis=0) for name in sessions if name != held], axis=0)
        x_test = np.concatenate(per_session[held]["label"], axis=0)
        y_test = np.concatenate(per_session[held]["full"], axis=0)
        linear = _fit_linear(x_train, y_train)
        orthogonal = _fit_orthogonal(x_train, y_train)
        own_linear = _fit_linear(x_test, y_test)
        own_orthogonal = _fit_orthogonal(x_test, y_test)
        held_rows.append({
            "held_out_source_session": held,
            "carrier_rows": int(x_test.shape[0]),
            "unaligned": _alignment_metrics(x_test, y_test),
            "common_linear": {**_alignment_metrics(x_test @ linear, y_test), "condition_number": float(np.linalg.cond(linear))},
            "common_orthogonal": _alignment_metrics(x_test @ orthogonal, y_test),
            "within_session_linear_upper_bound": _alignment_metrics(x_test @ own_linear, y_test),
            "within_session_orthogonal_upper_bound": _alignment_metrics(x_test @ own_orthogonal, y_test),
        })

    def _summary(method: str) -> dict[str, Any]:
        return {
            field: _quantiles([float(row[method][field]) for row in held_rows])
            for field in ("frobenius_cosine", "flattened_pearson", "relative_frobenius_error")
        }

    common_linear = _summary("common_linear")
    common_orthogonal = _summary("common_orthogonal")
    unaligned = _summary("unaligned")
    # This deliberately stringent criterion is descriptive only.  It prevents
    # a weak average correlation from being called an invertible common change.
    linear_recoverable = (
        common_linear["frobenius_cosine"]["median"] >= 0.95
        and common_linear["relative_frobenius_error"]["median"] <= 0.25
    )
    orthogonal_recoverable = (
        common_orthogonal["frobenius_cosine"]["median"] >= 0.95
        and common_orthogonal["relative_frobenius_error"]["median"] <= 0.25
    )
    return {
        "method": "leave-one-source-session-out; fit C_label @ A ~= C_full on all other source sessions; no intercept; no target data",
        "per_held_out_source_session": held_rows,
        "summary": {"unaligned": unaligned, "common_linear": common_linear, "common_orthogonal": common_orthogonal},
        "strict_recoverability_rule": "median held-out cosine >= 0.95 and median held-out relative Frobenius error <= 0.25",
        "strict_common_linear_recoverable": bool(linear_recoverable),
        "strict_common_orthogonal_recoverable": bool(orthogonal_recoverable),
    }


def _entry_metrics(*, full_raw: np.ndarray, label_raw: np.ndarray, denominator: float) -> dict[str, Any]:
    normalized_full = np.asarray(full_raw / denominator, dtype=np.float64)
    normalized_label = np.asarray(label_raw / denominator, dtype=np.float64)
    raw = {
        **_alignment_metrics(label_raw, full_raw),
        "rowwise_cosine": _row_cosines(label_raw, full_raw),
        "full_rms": float(np.sqrt(np.mean(np.square(full_raw), dtype=np.float64))),
        "label_rms": float(np.sqrt(np.mean(np.square(label_raw), dtype=np.float64))),
    }
    normalized = {
        **_alignment_metrics(normalized_label, normalized_full),
        "rowwise_cosine": _row_cosines(normalized_label, normalized_full),
        "full_rms": float(np.sqrt(np.mean(np.square(normalized_full), dtype=np.float64))),
        "label_rms": float(np.sqrt(np.mean(np.square(normalized_label), dtype=np.float64))),
    }
    _need(np.isclose(raw["frobenius_cosine"], normalized["frobenius_cosine"], rtol=0.0, atol=1e-14),
          "one shared scalar unexpectedly changed carrier cosine")
    _need(np.isclose(raw["flattened_pearson"], normalized["flattened_pearson"], rtol=0.0, atol=1e-14),
          "one shared scalar unexpectedly changed carrier Pearson")
    return {"raw": raw, "source_rms_normalized": normalized}


def _summarize_entry_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def grab(*keys: str) -> list[float]:
        current: Any
        values: list[float] = []
        for row in rows:
            current = row
            for key in keys:
                current = current[key]
            values.append(float(current))
        return values

    return {
        "raw": {
            "frobenius_cosine": _quantiles(grab("carrier", "raw", "frobenius_cosine")),
            "flattened_pearson": _quantiles(grab("carrier", "raw", "flattened_pearson")),
            "relative_frobenius_error": _quantiles(grab("carrier", "raw", "relative_frobenius_error")),
            "full_rms": _quantiles(grab("carrier", "raw", "full_rms")),
            "label_rms": _quantiles(grab("carrier", "raw", "label_rms")),
            "rowwise_cosine_median": _quantiles(grab("carrier", "raw", "rowwise_cosine", "median")),
        },
        "source_rms_normalized": {
            "frobenius_cosine": _quantiles(grab("carrier", "source_rms_normalized", "frobenius_cosine")),
            "flattened_pearson": _quantiles(grab("carrier", "source_rms_normalized", "flattened_pearson")),
            "relative_frobenius_error": _quantiles(grab("carrier", "source_rms_normalized", "relative_frobenius_error")),
            "full_rms": _quantiles(grab("carrier", "source_rms_normalized", "full_rms")),
            "label_rms": _quantiles(grab("carrier", "source_rms_normalized", "label_rms")),
            "rowwise_cosine_median": _quantiles(grab("carrier", "source_rms_normalized", "rowwise_cosine", "median")),
        },
        "fit_input_velocity": {
            "frobenius_cosine": _quantiles(grab("velocity", "frobenius_cosine")),
            "flattened_pearson": _quantiles(grab("velocity", "flattened_pearson")),
            "relative_frobenius_error": _quantiles(grab("velocity", "relative_frobenius_error")),
        },
    }


def _audit_bundle(bundle: Bundle, records: Mapping[str, H1PilotRecord]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    per_session: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"full": [], "label": []} for name in bundle.plan.source_sessions
    }
    support_trial_metadata: list[dict[str, Any]] = []
    for index, row in enumerate(bundle.raw_entries):
        session = str(row["session"])
        record = records[session]
        values = tuple(float(item) for item in row["trial_values"])
        _need(len(values) == SUPPORT_TRIALS, "cache entry M=4 support drift")
        override, shift_rows = rotation_overrides(record, values)
        full_fit = fit_frozen_carrier(record, bundle.plan, values)
        full = np.asarray(full_fit["carrier"], dtype=np.float64)
        cached = np.asarray(bundle.raw_carriers[index], dtype=np.float64)
        _need(np.array_equal(full, cached) and carrier_sha256(full) == str(row["carrier_sha256"]),
              "frozen plan/source record no longer reconstructs the immutable full cache")
        label_fit = fit_frozen_carrier(record, bundle.plan, values, labels_override=override)
        label = np.asarray(label_fit["carrier"], dtype=np.float64)
        legacy = label_rotation_carrier(record, bundle.plan, values)
        _need(np.array_equal(label, legacy), "audit rotation differs from the existing label_rotation_carrier implementation")
        original_labels = np.concatenate([record.blocks_for(value).velocity for value in values], axis=0)
        rotated_labels = np.concatenate([override[value] for value in values], axis=0)
        _need(original_labels.shape == rotated_labels.shape and original_labels.shape[1] == VELOCITY_DIM and
              not np.array_equal(original_labels, rotated_labels), "label rotation became identity")
        metric = {
            "entry_index": index,
            "session_name": session,
            "start_index": int(row["start_index"]),
            "trial_values": list(values),
            "fit_input_blocks": int(original_labels.shape[0]),
            "velocity": _alignment_metrics(rotated_labels, original_labels),
            "carrier": _entry_metrics(full_raw=full, label_raw=label, denominator=bundle.denominator),
            "full_carrier_sha256": carrier_sha256(full),
            "label_rotation_carrier_sha256": carrier_sha256(label),
            "label_rotation_nonidentity": not np.array_equal(label, full),
        }
        _need(metric["label_rotation_nonidentity"], "label rotation carrier became identity")
        entries.append(metric)
        per_session[session]["full"].append(full)
        per_session[session]["label"].append(label)
        support_trial_metadata.extend(shift_rows)

    _need(len(entries) == bundle.raw_carriers.shape[0], "entry audit count drift")
    unique_trials: dict[tuple[str, float], dict[str, Any]] = {}
    for row in support_trial_metadata:
        key = (str(row["session_name"]), float(row["trial_number"]))
        prior = unique_trials.get(key)
        _need(prior is None or prior == row, "same source trial has incompatible deterministic roll metadata")
        unique_trials[key] = row
    return {
        "outer_date": bundle.outer_date,
        "source_sessions": list(bundle.plan.source_sessions),
        "source_session_count": len(bundle.plan.source_sessions),
        "source_carrier_entries": len(entries),
        "carrier_shape": list(bundle.raw_carriers.shape),
        "source_normalizer": {"denominator": bundle.denominator, "normalizer_sha256": bundle.normalizer_sha256},
        "label_rotation_definition": {
            "seed": ROTATION_SEED,
            "algorithm": "per source support trial: shift=1+SHA256(seed|session|trial|0) mod (block_count-1); np.roll(velocity, shift, axis=0); refit fixed analytic carrier",
            "preserves": ["every within-trial velocity sample", "per-trial velocity marginal distribution", "within-trial temporal autocorrelation/trajectory structure up to cyclic wrap"],
            "does_not_establish": ["exchangeable label deletion", "independence of neural rates and labels", "outer-date predictive effect"],
        },
        "all_support_trial_instances": {
            "count": len(support_trial_metadata),
            "block_count": _quantiles([float(row["block_count"]) for row in support_trial_metadata]),
            "shift_blocks": _quantiles([float(row["shift_blocks"]) for row in support_trial_metadata]),
            "shift_fraction_of_trial": _quantiles([float(row["shift_fraction_of_trial"]) for row in support_trial_metadata]),
        },
        "unique_source_trials": {
            "count": len(unique_trials),
            "block_count": _quantiles([float(row["block_count"]) for row in unique_trials.values()]),
            "shift_blocks": _quantiles([float(row["shift_blocks"]) for row in unique_trials.values()]),
            "shift_fraction_of_trial": _quantiles([float(row["shift_fraction_of_trial"]) for row in unique_trials.values()]),
        },
        "entry_level_summary": _summarize_entry_rows(entries),
        "cross_source_common_inverse": _common_inverse_audit(per_session),
        "entry_digests": [
            {key: row[key] for key in ("entry_index", "session_name", "start_index", "trial_values", "fit_input_blocks",
                                         "full_carrier_sha256", "label_rotation_carrier_sha256", "label_rotation_nonidentity")}
            for row in entries
        ],
    }


def _write_immutable_json(path: Path, body: Mapping[str, Any]) -> str:
    output = path.resolve()
    _need(not output.exists(), f"refusing to overwrite existing audit receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o444)
    _immutable(output, "published source-only audit receipt")
    return sha256_file(output)


def run_source_only_audit(*, data_root: Path, bundle_root: Path, output: Path) -> dict[str, Any]:
    """Run the five-bundle source-only audit; this function has no GPU route."""

    data_root = data_root.resolve()
    bundle_root = bundle_root.resolve()
    _need(data_root.name == "000954" and (data_root / "sub-HumanPitt-held-in-calib").is_dir(),
          f"expected public held-in H1 data root, got {data_root}")
    _need(bundle_root.is_dir(), f"missing Phase-1 bundle root: {bundle_root}")
    bundles = [_load_bundle(bundle_root, date) for date in CONFIRMATORY_DATES]
    source_open_audit: dict[str, list[dict[str, Any]]] = {}
    date_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        records, opened = _load_bundle_source_records(bundle, data_root)
        source_open_audit[bundle.outer_date] = opened
        date_rows.append(_audit_bundle(bundle, records))

    # No numerical observation can turn a within-trial roll into an
    # exchangeable label-deletion null.  This is intentionally fail-closed;
    # the practical cosine/recovery measurements above describe *how weak or
    # strong* the temporal misalignment is, not a license to overclaim it.
    body: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "mode": "source_only_cpu_read_only_no_model_no_trainer_no_checkpoint_no_cuda",
        "scope": {
            "phase1_outer_dates": list(CONFIRMATORY_DATES),
            "outer_date_recordings_opened": 0,
            "outer_date_bytes_read": 0,
            "formal_or_evalai_opened_or_enumerated": False,
            "target_loader_imported_or_called": False,
            "target_evaluator_imported_or_called": False,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "cuda_constructed_or_launched": False,
            "source_records_opened_by_bundle": source_open_audit,
            "policy": "For each date bundle, only manifest-listed sessions whose calendar date differs from that bundle's outer date are opened directly by exact filename; outer files are neither globbed nor read.",
        },
        "input_binding": {
            "data_root": str(data_root),
            "bundle_root": str(bundle_root),
            "bundle_manifests": {
                bundle.outer_date: {"path": str(bundle.directory / "shared_source_manifest.json"), "sha256": bundle.manifest_sha256}
                for bundle in bundles
            },
        },
        "interpretation": {
            "conclusion": "FAIL_CLOSED__EXISTING_LABEL_ROTATION_IS_CERTIFIED_ONLY_AS_A_TEMPORAL_MISALIGNMENT_CONTROL_NOT_AS_A_PURE_LABEL_DELETION_NULL",
            "why": [
                "The intervention deterministically retains the complete within-trial label sequence and merely changes its phase relative to neural rates.",
                "Its retained label/carrier similarities and source-session inverse-map diagnostics are reported quantitatively below.",
                "A non-significant LS-vs-Full model result therefore cannot by itself refute the value of correct behavioral labels unless a stronger exchangeable/null control is separately specified and evaluated.",
            ],
            "permitted_use": "Describe existing LS as a temporal velocity-label misalignment intervention and cite the measured strength/recoverability statistics.",
            "forbidden_use": [
                "call LS an independent or exchangeable label-deletion null",
                "attribute a future EST4 or CarrierID predictive comparison to estimator content rather than consumer co-adaptation",
                "select a model or launch GPU work from this source-only receipt",
            ],
        },
        "date_bundles": date_rows,
        "code_sha256": {"audit": sha256_file(Path(__file__).resolve())},
    }
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source-only-audit", action="store_true", help="explicitly run the read-only five-bundle audit")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.run_source_only_audit:
        parser.error("refusing to perform any work without --run-source-only-audit")
    body = run_source_only_audit(data_root=args.data_root, bundle_root=args.bundle_root, output=args.output)
    receipt_sha = _write_immutable_json(args.output, body)
    print(json.dumps({"status": body["status"], "receipt_path": str(args.output.resolve()), "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
