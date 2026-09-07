"""Physical driver of the M2 T4-reliance diagnostic (CPU-only, inference-only).

One launch, zero training, zero target gradients.  The receipt law mirrors the
sealed CPU diagnostics: ``attempt.json`` (reserved before any data or model
access), ``replay.json`` (the perturbation grid), ``terminal.json`` (the
composed verdict).  Nothing under any frozen result root or frozen package is
created or modified; the sealed static-family recipe and the same-query/M2
machinery are reused BY IMPORT only:

* ``cdm_p1_m2_local_v1.replay.build_session_material`` -- the sealed surface
  law (post-30 common windows / the official query) and per-session targets;
* ``cdm_p1_m2_local_v1.replay.sealed_carrier``      -- the sealed
  ``ridge_static_m{budget}`` recipe (D-opt-4/chronological support, ridge T4
  fit, train-only normalization, static activity);
* ``cdm_p1_m2_local_v1.replay.M2Decoder``           -- the frozen cached-
  identity decode path over the frozen student;
* ``cdm_p1_m2_local_v1.anchor.M2SupportAnchor``     -- the bitwise carrier
  parity anchor;
* ``m2_t4_activity_budget_screen_v1.core``          -- the parent static
  family's digest/R2 laws.

The ONLY intervention is the ``[N, 4]`` normalized T4 side input to
``compute_identity``; the activity support is held verbatim.
"""

from __future__ import annotations

import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.cdm_p1_m2_local_v1 import physical as governing_physical
from src.cdm_p1_m2_local_v1 import replay as governing_replay

from . import plan


class M2T4RelianceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2T4RelianceError(message)


_sha256_file = governing_physical._sha256_file
_verify_sidecar = governing_physical._verify_sidecar
_publish = governing_physical._publish
_bind_namespaces = governing_physical._bind_namespaces
_load_sealed_static_rows = governing_physical._load_sealed_same_query_rows


def _validate_environment() -> None:
    """The CPU isolation law: no GPU may even be visible to this process."""
    _require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "",
             "this diagnostic is CPU-only: CUDA_VISIBLE_DEVICES must be empty "
             "(both GPUs are owned by other live routes)")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1",
             "PYTHONNOUSERSITE=1 is required (the sealed environment law)")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _require(os.environ.get(variable) == str(int(plan.Torch_THREADS)),
                 f"{variable}={plan.Torch_THREADS} is required (the CPU "
                 f"thread cap that keeps the neighbouring GPU training "
                 f"routes unaffected)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# The perturbation law (pure; unit-tested on synthetic arrays).
# ---------------------------------------------------------------------------


def perturbed_side(side: np.ndarray, cell: str) -> np.ndarray:
    """Apply the pre-registered cell law to the normalized ``[96, 4]`` side."""
    values = np.ascontiguousarray(np.asarray(side), dtype=np.float32)
    _require(values.shape == (plan.CHANNELS, plan.SIDE_DIM),
             f"side shape drift: {values.shape}")
    _require(np.isfinite(values).all(), "normalized side became nonfinite")
    _require(cell in plan.CELLS, f"unknown cell {cell}")
    if cell == "BASELINE":
        return values
    if cell == "T4_ZERO":
        return np.zeros_like(values, dtype=np.float32)
    if cell == "T4_SHUFFLE_UNITS":
        permutation = np.asarray(plan.UNIT_PERMUTATION, dtype=np.int64)
        _require(permutation.size == plan.CHANNELS
                 and np.array_equal(np.sort(permutation),
                                    np.arange(plan.CHANNELS, dtype=np.int64)),
                 "the frozen unit permutation is not a permutation")
        return np.ascontiguousarray(values[permutation], dtype=np.float32)
    permutation = np.asarray(plan.COL_PERMUTATION, dtype=np.int64)
    _require(permutation.size == plan.SIDE_DIM
             and np.array_equal(np.sort(permutation),
                                np.arange(plan.SIDE_DIM, dtype=np.int64)),
             "the frozen column permutation is not a permutation")
    return np.ascontiguousarray(values[:, permutation], dtype=np.float32)


def side_structure_evidence(side: np.ndarray, cell: str) -> dict[str, Any]:
    """The within-run structural proof that the perturbation is the briefed law."""
    values = np.ascontiguousarray(np.asarray(side), dtype=np.float32)
    perturbed = perturbed_side(values, cell)
    evidence: dict[str, Any] = {"cell": cell,
                                "perturbed_side_sha256": None,
                                "restores_under_inverse": None}
    if cell == "BASELINE":
        evidence["bitwise_equal_sealed_side"] = bool(np.array_equal(perturbed, values))
        evidence["restores_under_inverse"] = True
    elif cell == "T4_ZERO":
        evidence["all_zero"] = bool(np.count_nonzero(perturbed) == 0)
        evidence["restores_under_inverse"] = True
    elif cell == "T4_SHUFFLE_UNITS":
        permutation = np.asarray(plan.UNIT_PERMUTATION, dtype=np.int64)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size, dtype=np.int64)
        evidence["unit_permutation_fixed_points"] = int(
            np.count_nonzero(permutation == np.arange(permutation.size)))
        evidence["restores_under_inverse"] = bool(
            np.array_equal(perturbed[inverse], values))
    else:
        permutation = np.asarray(plan.COL_PERMUTATION, dtype=np.int64)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(permutation.size, dtype=np.int64)
        evidence["column_source_of_column_j"] = permutation.tolist()
        evidence["restores_under_inverse"] = bool(
            np.array_equal(perturbed[:, inverse], values))
    return evidence


# ---------------------------------------------------------------------------
# Summaries, paired deltas and the pre-registered verdict (pure).
# ---------------------------------------------------------------------------


def summarize_sessions(values: Mapping[str, float]) -> dict[str, Any]:
    _require(bool(values), "empty session map")
    ordered = {key: float(values[key]) for key in sorted(values)}
    array = np.asarray(list(ordered.values()), dtype=np.float64)
    _require(np.isfinite(array).all(), "session R2 nonfinite")
    return {
        "session_count": int(array.size),
        "equal_session_mean": float(array.mean()),
        "equal_session_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "equal_session_sd_ddof": 1,
        "equal_session_median": float(np.median(array)),
        "per_session_r2": ordered,
    }


def paired_delta(candidate: Mapping[str, float],
                 baseline: Mapping[str, float]) -> dict[str, Any]:
    _require(set(candidate) == set(baseline) and bool(candidate),
             "paired session set mismatch")
    deltas = {key: float(candidate[key]) - float(baseline[key])
              for key in sorted(candidate)}
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    return {
        "per_session_delta": deltas,
        "equal_session_mean_delta": float(values.mean()),
        "equal_session_sd_delta": (
            float(values.std(ddof=1)) if values.size > 1 else 0.0),
        "equal_session_median_delta": float(np.median(values)),
        "negative_sessions": int(np.count_nonzero(values < 0.0)),
        "positive_sessions": int(np.count_nonzero(values > 0.0)),
        "min_delta": float(values.min()),
        "max_delta": float(values.max()),
        "session_count": int(values.size),
    }


def aggregate_cell(deltas_by_grid_cell: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Pool one perturbation cell's paired deltas over the 6 grid cells."""
    _require(bool(deltas_by_grid_cell), "empty grid for aggregation")
    means = np.asarray(
        [float(deltas_by_grid_cell[key]["equal_session_mean_delta"])
         for key in sorted(deltas_by_grid_cell)], dtype=np.float64)
    negative_sessions = int(sum(
        int(deltas_by_grid_cell[key]["negative_sessions"])
        for key in deltas_by_grid_cell))
    total_sessions = int(sum(
        int(deltas_by_grid_cell[key]["session_count"])
        for key in deltas_by_grid_cell))
    return {
        "grid_cell_count": int(means.size),
        "mean_of_mean_deltas": float(means.mean()),
        "min_cell_mean_delta": float(means.min()),
        "max_cell_mean_delta": float(means.max()),
        "negative_grid_cells": int(np.count_nonzero(means < 0.0)),
        "pooled_negative_sessions": negative_sessions,
        "pooled_session_count": total_sessions,
        "pooled_negative_share": float(negative_sessions / total_sessions),
    }


def evaluate_verdict(aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered three-string verdict on the T4_ZERO primary."""
    expected = set(plan.CELLS) - {"BASELINE"}
    _require(set(aggregates) == expected, "the verdict needs every perturbation cell")
    primary = float(aggregates[plan.VERDICT_LAW["primary_cell"]]["mean_of_mean_deltas"])
    if primary <= -0.05:
        verdict = plan.VERDICT_STRONG
    elif primary <= -0.005:
        verdict = plan.VERDICT_MODERATE
    else:
        verdict = plan.VERDICT_NEGLIGIBLE
    return {
        "law": dict(plan.VERDICT_LAW),
        "primary_cell": plan.VERDICT_LAW["primary_cell"],
        "primary_statistic": primary,
        "secondary_statistics": {
            cell: float(aggregates[cell]["mean_of_mean_deltas"])
            for cell in plan.VERDICT_LAW["secondary_cells"]
        },
        "verdict": verdict,
        "verdict_string_pre_registered": True,
    }


def _leakage_labels() -> dict[str, bool]:
    return dict(plan.LEAKAGE_LAW["all_rows"])


# ---------------------------------------------------------------------------
# The frozen decode path (chunked over the sealed query windows).
# ---------------------------------------------------------------------------


def _decode_starts(decoder: Any, material: Any, identity: Any) -> np.ndarray:
    """Decode every sealed query window in bounded-memory chunks."""
    outputs: list[np.ndarray] = []
    starts = np.asarray(material.starts, dtype=np.int64)
    for offset in range(0, starts.size, plan.DECODE_CHUNK):
        chunk = starts[offset : offset + plan.DECODE_CHUNK]
        windows = decoder.windows(material.neural, chunk)
        outputs.append(decoder.decode(windows, identity))
    return np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)


def _score_cell(*, decoder: Any, material: Any, carrier: Mapping[str, Any],
                cell: str) -> dict[str, Any]:
    """One (surface, budget, session, cell) row: perturb ONLY the T4 side."""
    side = perturbed_side(carrier["side"], cell)
    structure = side_structure_evidence(carrier["side"], cell)
    _require(structure["restores_under_inverse"] is True,
             f"{cell}: the perturbation failed its structural law")
    identity = decoder.identity(carrier["activity"], side)
    prediction = _decode_starts(decoder, material, identity)
    return {
        "cell": cell,
        "side_law": plan.CELL_LAWS[cell],
        "side_structure": structure,
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# The sealed anchor row loaders.
# ---------------------------------------------------------------------------


def _load_sealed_t4_mean_rows(repo_root: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    path = repo_root / plan.SAME_QUERY_SCORE_RELATIVE
    _require(_sha256_file(path) == governing_replay.plan.SAME_QUERY_SCORE_SHA256,
             "sealed same-query score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_same_query_comparator_v1"
             and payload.get("status") == "TERMINAL",
             "sealed comparator schema drift")
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if str(row.get("cell", "")).startswith("t4_mean_side_m"):
            rows[(str(row["surface"]), str(row["session"]), int(row["budget"]))] = row
    _require(len(rows) == (plan.EXPECTED_WITHIN_SESSIONS
                           + plan.EXPECTED_EXTERNAL_SESSIONS) * len(plan.BUDGETS),
             "sealed t4_mean anchor row topology drift")
    return rows


def _load_parent_static_rows(repo_root: Path) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """The sealed static-family parent rows (r2 + query digests) themselves."""
    path = repo_root / (plan.PARENT_ROOT_RELATIVE + "/score.json")
    from src.m2_same_query_comparator_v1.plan import PARENT_SCORE_SHA256

    _require(_sha256_file(path) == PARENT_SCORE_SHA256,
             "sealed static-family parent score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_t4_activity_budget_screen_v1"
             and payload.get("status") == "TERMINAL"
             and payload.get("parameter_updates") == 0,
             "the static-family parent is not a sealed zero-update terminal")
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if str(row.get("cell", "")).startswith("ridge_static_m"):
            key = (str(row["surface"]), str(row["session"]), str(row["cell"]))
            _require(key not in rows, "duplicate parent static row")
            rows[key] = row
    _require(len(rows) == (plan.EXPECTED_WITHIN_SESSIONS
                           + plan.EXPECTED_EXTERNAL_SESSIONS) * len(plan.BUDGETS),
             "sealed parent static row topology drift")
    return rows


# ---------------------------------------------------------------------------
# Execute.
# ---------------------------------------------------------------------------


def execute(repo_root: Path, *, batch_size: int = plan.BATCH_SIZE) -> dict[str, Any]:
    _validate_environment()
    repo_root = Path(repo_root).absolute()
    root = plan.result_root(repo_root)
    _require(not (root / "replay.json").exists(), "the replay receipt already exists")
    _require(not (root / "terminal.json").exists(), "the terminal receipt already exists")
    _require((root / "attempt.json").exists(), "reserve the attempt receipt first")
    attempt_digest = _verify_sidecar(root / "attempt.json")
    attempt = json.loads((root / "attempt.json").read_text(encoding="utf-8"))
    for relative, digest in attempt["predecessor_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an immutable predecessor drifted: {relative}")
    for relative, digest in attempt["owned_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an owned diagnostic module drifted: {relative}")

    _bind_namespaces(repo_root)
    from src.m2_t4_activity_budget_screen_v1.core import (
        array_sha256 as sealed_digest,
    )
    from src.m2_t4_activity_budget_screen_v1.core import variance_weighted_r2
    from src.m2_t4_activity_budget_screen_v1 import plan as screen_plan

    from src.cdm_p1_m2_local_v1.anchor import M2SupportAnchor, build_groups

    sealed_static_rows = _load_sealed_static_rows(repo_root)
    sealed_mean_rows = _load_sealed_t4_mean_rows(repo_root)
    parent_static_rows = _load_parent_static_rows(repo_root)

    import torch

    _require(not torch.cuda.is_available(),
             "CUDA must not be visible to this CPU-only diagnostic")
    torch.set_num_threads(int(plan.Torch_THREADS))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    started = time.monotonic()

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == screen_plan.CHECKPOINT_SHA256,
             "T4 checkpoint drift")
    _require(metadata["teacher_checkpoint_sha256"]
             == governing_replay.plan.SPINT_CHECKPOINT_SHA256,
             "SPINT teacher checkpoint drift")
    _require(metadata["normalization_sha256"] == screen_plan.NORMALIZATION_SHA256,
             "T4 normalization drift")
    student = model.student.to(device).eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    decoder = governing_replay.M2Decoder(torch, student, device, batch_size=batch_size)

    surfaces = {
        "within_post30": data_module.train_dataset,
        "external_official_query": data_module.val_heldout_dataset,
    }
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
    }

    materials: dict[str, dict[str, Any]] = {}
    carriers: dict[str, dict[str, dict[int, Mapping[str, Any]]]] = {}
    binding_evidence: dict[str, Any] = {}
    for surface, dataset in surfaces.items():
        _require(dataset is not None, f"{surface}: dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface],
                 f"{surface} session count drift")
        materials[surface] = {}
        carriers[surface] = {}
        for session in sessions:
            materials[surface][session] = governing_replay.build_session_material(
                dataset=dataset, session=session, surface=surface)
            carriers[surface][session] = {}
            for budget in plan.BUDGETS:
                carrier = governing_replay.sealed_carrier(dataset, session, budget)
                groups = build_groups(carrier["raw_t4"],
                                      materials[surface][session].channel_ids)
                anchor = M2SupportAnchor.from_labeled_support(
                    groups=groups,
                    support_trial_rates=carrier["rates"],
                    support_angles_rad=carrier["theta"].tolist(),
                    support_t4=carrier["raw_t4"],
                )
                _require(anchor.support_coefficient_parity["rebuilt_rows_bitwise_equal"],
                         f"anchor zero-evidence fallback drifted ({surface}/{session} m{budget})")
                sealed_row = sealed_static_rows[(surface, session, budget)]
                _require(carrier["selected"].tolist()
                         == sealed_row["side_evidence"]["selected_indices"],
                         f"selected support drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["raw_t4"])
                         == sealed_row["side_evidence"]["raw_t4_sha256"],
                         f"raw carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["side"])
                         == sealed_row["side_evidence"]["normalized_t4_sha256"],
                         f"normalized carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["activity"])
                         == sealed_row["activity_sha256"],
                         f"activity digest drift {surface}/{session} m{budget}")
                parent_row = parent_static_rows[(surface, session, f"ridge_static_m{budget}")]
                _require(float(parent_row["r2"]) == float(sealed_row["r2"]),
                         f"parent/comparator static row drift {surface}/{session} m{budget}")
                carriers[surface][session][budget] = carrier
                binding_evidence[f"{surface}|{session}|m{budget}"] = {
                    "selected": carrier["selected"].tolist(),
                    "raw_t4_sha256": carrier["raw_t4_sha256"],
                    "side_sha256": carrier["side_sha256"],
                    "activity_sha256": carrier["activity_sha256"],
                    "anchor_payload_digest": anchor.digest,
                    "anchor_parity": anchor.support_coefficient_parity,
                }
        print(f"[{time.monotonic() - started:8.1f}s] bound {surface}: "
              f"{len(materials[surface])} sessions", flush=True)

    # -- the grid: 4 cells x 3 budgets x 13 sessions ------------------------
    rows: list[dict[str, Any]] = []
    r2: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        surface: {budget: {cell: {} for cell in plan.CELLS} for budget in plan.BUDGETS}
        for surface in plan.SURFACES
    }
    baseline_anchor: dict[str, Any] = {}
    zero_anchor: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for budget in plan.BUDGETS:
            for session in sorted(materials[surface]):
                material = materials[surface][session]
                carrier = carriers[surface][session][budget]
                starts_digest = sealed_digest(material.starts)
                targets_digest = sealed_digest(material.targets)
                for cell in plan.CELLS:
                    scored = _score_cell(decoder=decoder, material=material,
                                         carrier=carrier, cell=cell)
                    prediction = scored["prediction"]
                    side_digest = sealed_digest(perturbed_side(carrier["side"], cell))
                    value = float(variance_weighted_r2(material.targets, prediction))
                    row: dict[str, Any] = {
                        "surface": surface, "session": session, "cell": cell,
                        "budget": int(budget),
                        "window_count": int(material.starts.size),
                        "ordered_window_starts_sha256": starts_digest,
                        "target_sha256": targets_digest,
                        "activity_sha256": carrier["activity_sha256"],
                        "side_sha256": side_digest,
                        "side_law": scored["side_law"],
                        "side_structure": scored["side_structure"],
                        "prediction_sha256": sealed_digest(prediction),
                        "r2": value,
                        "target_gradients": 0,
                        "parameter_updates": 0,
                        **_leakage_labels(),
                    }
                    if cell == "BASELINE":
                        sealed_row = sealed_static_rows[(surface, session, budget)]
                        checks = {
                            "window_count_equal":
                                row["window_count"] == int(sealed_row["window_count"]),
                            "starts_digest_equal":
                                starts_digest == sealed_row["ordered_window_starts_sha256"],
                            "target_digest_equal":
                                targets_digest == sealed_row["target_sha256"],
                            "side_digest_equal_sealed_carrier":
                                side_digest
                                == sealed_row["side_evidence"]["normalized_t4_sha256"],
                            "abs_delta_r2_vs_sealed":
                                abs(value - float(sealed_row["r2"])),
                            "tolerance": float(plan.ANCHOR_TOLERANCE_R2),
                        }
                        checks["pass"] = bool(
                            checks["window_count_equal"]
                            and checks["starts_digest_equal"]
                            and checks["target_digest_equal"]
                            and checks["side_digest_equal_sealed_carrier"]
                            and checks["abs_delta_r2_vs_sealed"] <= float(plan.ANCHOR_TOLERANCE_R2))
                        baseline_anchor[f"{surface}|{session}|m{budget}"] = checks
                        _require(checks["pass"],
                                 f"the BASELINE anchor failed at {surface}/{session}/m{budget}")
                    if cell == "T4_ZERO":
                        sealed_row = sealed_mean_rows[(surface, session, budget)]
                        checks = {
                            "window_count_equal":
                                row["window_count"] == int(sealed_row["window_count"]),
                            "starts_digest_equal":
                                starts_digest == sealed_row["ordered_window_starts_sha256"],
                            "target_digest_equal":
                                targets_digest == sealed_row["target_sha256"],
                            "activity_digest_equal":
                                sealed_digest(carrier["activity"])
                                == sealed_row["activity_sha256"],
                            "side_digest_equal_sealed_zero_side":
                                side_digest
                                == sealed_row["side_evidence"]["normalized_t4_sha256"],
                            "abs_delta_r2_vs_sealed":
                                abs(value - float(sealed_row["r2"])),
                            "tolerance": float(plan.ANCHOR_TOLERANCE_R2),
                        }
                        checks["pass"] = bool(
                            checks["window_count_equal"]
                            and checks["starts_digest_equal"]
                            and checks["target_digest_equal"]
                            and checks["activity_digest_equal"]
                            and checks["side_digest_equal_sealed_zero_side"]
                            and checks["abs_delta_r2_vs_sealed"] <= float(plan.ANCHOR_TOLERANCE_R2))
                        zero_anchor[f"{surface}|{session}|m{budget}"] = checks
                        _require(checks["pass"],
                                 f"the T4_ZERO anchor failed at {surface}/{session}/m{budget}")
                    rows.append(row)
                    r2[surface][budget][cell][session] = value
                    _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                             "the CPU hard timeout fired during the grid")
                decoder.clear_cache()
            print(f"[{time.monotonic() - started:8.1f}s] grid {surface} m{budget} done",
                  flush=True)

    _require(len(rows) == len(plan.CELLS) * len(plan.BUDGETS)
             * (plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS),
             "final row cardinality drift")

    # -- summaries, paired deltas, aggregates, verdict ----------------------
    summaries = {
        surface: {
            budget: {
                cell: summarize_sessions(r2[surface][budget][cell])
                for cell in plan.CELLS
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    deltas = {
        surface: {
            budget: {
                cell: paired_delta(r2[surface][budget][cell],
                                   r2[surface][budget]["BASELINE"])
                for cell in plan.CELLS if cell != "BASELINE"
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    aggregates = {
        cell: aggregate_cell({
            f"{surface}|m{budget}": deltas[surface][budget][cell]
            for surface in plan.SURFACES for budget in plan.BUDGETS
        })
        for cell in plan.CELLS if cell != "BASELINE"
    }
    verdict = evaluate_verdict(aggregates)

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "inference_time_t4_side_perturbation_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": plan.AUTHORITY,
        "scientific_role": plan.SCIENTIFIC_ROLE,
        "eval_time_vs_train_time_disclosure": plan.EVAL_TIME_VS_TRAIN_TIME_DISCLOSURE,
        "foundation": {
            "static_family_root": plan.PARENT_ROOT_RELATIVE,
            "static_family_schema": "m2_t4_activity_budget_screen_v1",
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
            "normalization_sha256": metadata["normalization_sha256"],
            "machinery": "src/cdm_p1_m2_local_v1 (build_session_material, "
                         "sealed_carrier, M2Decoder, M2SupportAnchor) + "
                         "src/m2_t4_activity_budget_screen_v1 core laws, by import",
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "batch_size": int(batch_size),
            "decode_chunk": int(plan.DECODE_CHUNK),
            "cpu_count_threads": int(plan.Torch_THREADS),
            "torch_num_threads_actual": int(torch.get_num_threads()),
        },
        "cells": dict(plan.CELL_LAWS),
        "permutation_law": dict(plan.PERMUTATION_LAW),
        "surfaces": list(plan.SURFACES),
        "budgets": list(plan.BUDGETS),
        "cell_order": list(plan.CELLS),
        "row_count": len(rows),
        "carrier_support_binding": binding_evidence,
        "rows": rows,
        "summaries": summaries,
        "paired_deltas_vs_baseline": deltas,
        "aggregates": aggregates,
        "anchors": {
            "law": dict(plan.ANCHORS),
            "baseline_vs_sealed_static_rows": baseline_anchor,
            "baseline_all_pass": all(item["pass"] for item in baseline_anchor.values()),
            "t4_zero_vs_sealed_t4_mean_rows": zero_anchor,
            "t4_zero_all_pass": all(item["pass"] for item in zero_anchor.values()),
            "tolerance_law": {
                "per_session_abs_delta_r2_max_allowed": float(plan.ANCHOR_TOLERANCE_R2),
                "reason": (
                    "CPU float-reduction order cannot bitwise-match GPU "
                    "receipts; window counts, query starts, targets and side "
                    "inputs remain bit-exact via their sha256 digests"
                ),
            },
            "max_abs_delta_baseline": max(
                item["abs_delta_r2_vs_sealed"] for item in baseline_anchor.values()),
            "max_abs_delta_zero": max(
                item["abs_delta_r2_vs_sealed"] for item in zero_anchor.values()),
        },
        "leakage_law": dict(plan.LEAKAGE_LAW),
        "verdict": verdict,
        "deviations": list(plan.DEVIATIONS),
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "started_utc": attempt.get("started_utc"),
        "finished_utc": _utc_now(),
    }
    _require(replay_payload["anchors"]["baseline_all_pass"],
             "the BASELINE reproduction anchor failed at least one row")
    _require(replay_payload["anchors"]["t4_zero_all_pass"],
             "the T4_ZERO reproduction anchor failed at least one row")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "inference_time_m2_t4_reliance_diagnostic_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "authority": plan.AUTHORITY,
        "scientific_role": plan.SCIENTIFIC_ROLE,
        "eval_time_vs_train_time_disclosure": plan.EVAL_TIME_VS_TRAIN_TIME_DISCLOSURE,
        "leakage_statement": plan.LEAKAGE_LAW["receipt_statement"],
        "can_this_run_select_or_promote": False,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "foundation": replay_payload["foundation"],
        "equal_session_means": {
            surface: {
                f"m{budget}": {
                    cell: {
                        "equal_session_mean": summaries[surface][budget][cell]["equal_session_mean"],
                        "equal_session_sd": summaries[surface][budget][cell]["equal_session_sd"],
                    }
                    for cell in plan.CELLS
                }
                for budget in plan.BUDGETS
            }
            for surface in plan.SURFACES
        },
        "paired_deltas_vs_baseline": {
            surface: {
                f"m{budget}": {
                    cell: {
                        "equal_session_mean_delta":
                            deltas[surface][budget][cell]["equal_session_mean_delta"],
                        "equal_session_sd_delta":
                            deltas[surface][budget][cell]["equal_session_sd_delta"],
                        "negative_sessions":
                            deltas[surface][budget][cell]["negative_sessions"],
                        "session_count":
                            deltas[surface][budget][cell]["session_count"],
                    }
                    for cell in plan.CELLS if cell != "BASELINE"
                }
                for budget in plan.BUDGETS
            }
            for surface in plan.SURFACES
        },
        "aggregates": aggregates,
        "anchors": {
            "baseline_all_pass": replay_payload["anchors"]["baseline_all_pass"],
            "t4_zero_all_pass": replay_payload["anchors"]["t4_zero_all_pass"],
            "carrier_support_binding_all_pass": True,
        },
        "verdict": verdict,
        "deviations": list(plan.DEVIATIONS),
        "environment": replay_payload["environment"],
        "wall_seconds": replay_payload["wall_seconds"],
        "finished_utc": replay_payload["finished_utc"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "verdict": verdict["verdict"],
        "primary_statistic": verdict["primary_statistic"],
        "secondary_statistics": verdict["secondary_statistics"],
        "aggregates": aggregates,
        "anchors_all_pass": (
            replay_payload["anchors"]["baseline_all_pass"]
            and replay_payload["anchors"]["t4_zero_all_pass"]
        ),
        "wall_seconds": replay_payload["wall_seconds"],
    }
