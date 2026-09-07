"""Physical driver of the M2 P1-carrier alpha_M post-hoc diagnostic.

CPU-only, inference-only, zero training, one launch.  The receipt law mirrors
the governing run: ``attempt.json`` (reserved before any data or model
access), ``replay.json`` (the diagnostic grid), ``terminal.json`` (the
composed verdict).  Nothing under any frozen result root or frozen package is
created or modified; the governing machinery is reused BY IMPORT only.

The verdict is pre-registered in :mod:`.plan` and evaluated exactly as
written; no number produced here can select a hyperparameter or promote a
cell.
"""

from __future__ import annotations

import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.cdm_p1_m2_local_v1 import gates as governing_gates
from src.cdm_p1_m2_local_v1 import physical as governing_physical
from src.cdm_p1_m2_local_v1 import replay as governing_replay
from src.cdm_p1_m2_local_v1.anchor import M2SupportAnchor, build_groups
from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from . import plan


class M2CarrierAlphaDiagnosticError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2CarrierAlphaDiagnosticError(message)


_sha256_file = governing_physical._sha256_file
_verify_sidecar = governing_physical._verify_sidecar
_publish = governing_physical._publish
_bind_namespaces = governing_physical._bind_namespaces
_load_sealed_same_query_rows = governing_physical._load_sealed_same_query_rows


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
# Sealed-foundation readers.
# ---------------------------------------------------------------------------


def _load_sealed_governing(repo_root: Path) -> dict[str, Any]:
    root = repo_root / plan.GOVERNING_ROOT_RELATIVE
    replay_digest = _verify_sidecar(root / "replay.json")
    terminal_digest = _verify_sidecar(root / "terminal.json")
    _verify_sidecar(root / "attempt.json")
    _require(replay_digest == plan.GOVERNING_REPLAY_SHA256,
             "the sealed governing replay receipt drifted")
    _require(terminal_digest == plan.GOVERNING_TERMINAL_SHA256,
             "the sealed governing terminal receipt drifted")
    _require(_sha256_file(repo_root / plan.GOVERNING_ROOT_RELATIVE / "attempt.json")
             == plan.GOVERNING_ATTEMPT_SHA256,
             "the sealed governing attempt receipt drifted")
    replay_payload = json.loads(
        (root / "replay.json").read_text(encoding="utf-8"))
    terminal_payload = json.loads(
        (root / "terminal.json").read_text(encoding="utf-8"))
    _require(replay_payload.get("status") == "REPLAY_COMPLETE"
             and terminal_payload.get("status") == "TERMINAL",
             "the governing run is not sealed terminal")
    return {"replay": replay_payload, "terminal": terminal_payload}


def _governing_hp(sealed: Mapping[str, Any], budget: int) -> dict[str, Any]:
    key = f"m{int(budget)}"
    vector = dict(sealed["replay"]["hyperparameters"][key])
    expected = plan.GOVERNING_BY_BUDGET[int(budget)]
    _require(vector == expected,
             f"the sealed governing {key} vector drifted from the frozen plan")
    return vector


def _thresholds(vector: Mapping[str, Any]) -> stage_p_gate.GateThresholds:
    return stage_p_gate.GateThresholds(
        tau_d=float(vector["thresholds"]["tau_d_rad"]),
        r_max=int(vector["thresholds"]["r_max_repetition_per_direction"]),
        d_min=int(vector["thresholds"]["d_min_distinct_directions"]),
        max_mass_relative=float(
            vector["thresholds"]["max_pseudo_mass_relative_to_support_rows"]),
    )


def _hp_with_alpha(vector: Mapping[str, Any], alpha: float,
                   *, c_M: Optional[float]) -> stage_p_replay.CellHyperparameters:
    return stage_p_replay.CellHyperparameters(
        rho_M=float(vector["rho_M"]), alpha_M=float(alpha),
        c_M=c_M, thresholds=_thresholds(vector),
    )


def _stage2_scored(sealed: Mapping[str, Any], budget: int,
                   rho_M: float, alpha_M: float) -> Mapping[str, Any]:
    for row in sealed["replay"]["hyperparameter_selection"][f"m{int(budget)}"]["stage2_scored"]:
        if (float(row["vector"]["rho_M"]) == float(rho_M)
                and float(row["vector"]["alpha_M"]) == float(alpha_M)):
            return row
    raise M2CarrierAlphaDiagnosticError(
        f"the sealed stage-2 grid lacks (rho={rho_M}, alpha={alpha_M}) at m{budget}")


# ---------------------------------------------------------------------------
# Verdict and monotonicity (pre-registered; unit-tested on synthetic input).
# ---------------------------------------------------------------------------


def evaluate_verdict(deltas: Mapping[str, float], *, epsilon: float) -> dict[str, Any]:
    """The pre-registered held-out verdict.

    ``deltas`` maps ``"m{budget}|alpha{alpha}"`` held-out equal-session mean
    deltas (D@alpha - D@alpha=0).  Every delta <= 0 within the epsilon band
    => WITHIN_SELECTION_MATCHED_HELDOUT; any delta > 0 beyond the band =>
    SELECTION_SURFACE_MISMATCH.
    """
    _require(bool(deltas), "the verdict needs at least one held-out delta")
    rows: dict[str, dict[str, Any]] = {}
    any_positive = False
    for key in sorted(deltas):
        value = float(deltas[key])
        within_band = abs(value) <= float(epsilon)
        positive = bool(value > 0.0 and not within_band)
        any_positive = any_positive or positive
        rows[key] = {
            "delta": value,
            "sign": ("zero_band" if within_band else ("positive" if value > 0.0 else "negative")),
            "within_epsilon_band_of_zero": bool(within_band),
            "counts_as_positive": positive,
        }
    verdict = plan.VERDICT_MISMATCH if any_positive else plan.VERDICT_MATCHED
    return {
        "law": dict(plan.VERDICT_LAW),
        "rows": rows,
        "any_alpha_positive_on_heldout": bool(any_positive),
        "verdict": verdict,
        "verdict_string_pre_registered": True,
    }


def evaluate_monotonicity(means: Mapping[float, float]) -> dict[str, Any]:
    """Per (surface, budget): is the alpha-scan non-increasing, and non-positive?"""
    _require(set(map(float, means)) == set(map(float, plan.ALPHA_ALL)),
             "monotonicity needs the full pre-registered alpha axis")
    anchor = float(means[float(plan.ALPHA_ANCHOR)])
    scan = {float(alpha): float(means[float(alpha)]) for alpha in plan.ALPHA_SCAN}
    ordered = [anchor] + [scan[float(alpha)] for alpha in plan.ALPHA_SCAN]
    return {
        "means_by_alpha": {str(float(alpha)): float(means[float(alpha)])
                           for alpha in plan.ALPHA_ALL},
        "nonincreasing_in_alpha": bool(all(
            ordered[index] >= ordered[index + 1] for index in range(len(ordered) - 1))),
        "anchor_is_maximum": bool(all(anchor >= value for value in scan.values())),
        "anchor_mean": anchor,
    }


def _leakage_labels(alpha: float) -> dict[str, bool]:
    if float(alpha) == float(plan.ALPHA_ANCHOR):
        return dict(plan.LEAKAGE_LAW["alpha_zero_rows"])
    return dict(plan.LEAKAGE_LAW["alpha_gt_zero_rows"])


# ---------------------------------------------------------------------------
# The rollout wrapper (one cell of one family over one session).
# ---------------------------------------------------------------------------


def _rollout(decoder: governing_replay.M2Decoder, material: governing_replay.SessionMaterial,
             *, budget: int, anchor: M2SupportAnchor, carrier: Mapping[str, Any],
             hp: stage_p_replay.CellHyperparameters,
             row_spec: stage_p_replay.RowSpec) -> dict[str, Any]:
    return governing_replay.rollout_session_f(
        decoder, material, budget=int(budget),
        spec=governing_replay.CellSpec("F01m", "p1_online", True),
        anchor=anchor, carrier=carrier, hp=hp,
        config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
        row_spec=row_spec,
    )


def _causality_chain(receipts: Sequence[Mapping[str, Any]]) -> bool:
    previous_ca: Optional[str] = None
    for receipt in receipts:
        if previous_ca is not None and receipt["cb"] != previous_ca:
            return False
        previous_ca = receipt["ca"]
    return True


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

    sealed = _load_sealed_governing(repo_root)
    governing_vectors = {budget: _governing_hp(sealed, budget) for budget in plan.BUDGETS}
    sealed_f01m = {
        (surface, budget, session): row
        for surface, by_budget in sealed["replay"]["matrix"].items()
        for budget, by_cell in by_budget.items()
        for session, row in by_cell["F01m"].items()
    }
    sealed_f00m = {
        (surface, budget, session): row
        for surface, by_budget in sealed["replay"]["matrix"].items()
        for budget, by_cell in by_budget.items()
        for session, row in by_cell["F00m"].items()
    }

    _bind_namespaces(repo_root)
    from src.m2_same_query_comparator_v1.core import (
        array_sha256 as sealed_digest,
    )
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2

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
    _require(metadata["checkpoint_sha256"] == governing_replay.plan.T4_CHECKPOINT_SHA256,
             "T4 checkpoint drift")
    _require(metadata["teacher_checkpoint_sha256"] == governing_replay.plan.SPINT_CHECKPOINT_SHA256,
             "SPINT teacher checkpoint drift")
    _require(metadata["normalization_sha256"] == governing_replay.plan.NORMALIZATION_SHA256,
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
    sealed_rows = _load_sealed_same_query_rows(repo_root)

    materials: dict[str, dict[str, governing_replay.SessionMaterial]] = {}
    carriers: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    anchors: dict[str, dict[str, dict[int, M2SupportAnchor]]] = {}
    binding_evidence: dict[str, Any] = {}
    for surface, dataset in surfaces.items():
        _require(dataset is not None, f"{surface}: dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        materials[surface] = {}
        carriers[surface] = {}
        anchors[surface] = {}
        for session in sessions:
            materials[surface][session] = governing_replay.build_session_material(
                dataset=dataset, session=session, surface=surface,
            )
            carriers[surface][session] = {}
            anchors[surface][session] = {}
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
                sealed_carrier_row = sealed_rows[(surface, session, budget)]
                _require(carrier["selected"].tolist()
                         == sealed_carrier_row["side_evidence"]["selected_indices"],
                         f"selected support drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["raw_t4"])
                         == sealed_carrier_row["side_evidence"]["raw_t4_sha256"],
                         f"raw carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["side"])
                         == sealed_carrier_row["side_evidence"]["normalized_t4_sha256"],
                         f"normalized carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["activity"])
                         == sealed_carrier_row["activity_sha256"],
                         f"activity digest drift {surface}/{session} m{budget}")
                carriers[surface][session][budget] = carrier
                anchors[surface][session][budget] = anchor
                binding_evidence[f"{surface}|{session}|m{budget}"] = {
                    "selected": carrier["selected"].tolist(),
                    "raw_t4_sha256": carrier["raw_t4_sha256"],
                    "side_sha256": carrier["side_sha256"],
                    "activity_sha256": carrier["activity_sha256"],
                    "sealed_row_raw_t4_sha256": sealed_digest(carrier["raw_t4"]),
                    "sealed_row_side_sha256": sealed_digest(carrier["side"]),
                    "sealed_row_activity_sha256": sealed_digest(carrier["activity"]),
                    "anchor_payload_digest": anchor.digest,
                    "anchor_parity": anchor.support_coefficient_parity,
                }
        print(f"[{time.monotonic() - started:8.1f}s] bound {surface}: "
              f"{len(materials[surface])} sessions", flush=True)

    # -- the D family: the governing F01m law with only alpha moved ----------
    row_spec_p1 = stage_p_replay.ROW_SPECS["P1"]
    d_rows: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = {
        surface: {budget: {} for budget in plan.BUDGETS} for surface in plan.SURFACES
    }
    d_r2: dict[str, dict[int, dict[float, dict[str, float]]]] = {
        surface: {budget: {} for budget in plan.BUDGETS} for surface in plan.SURFACES
    }
    for surface in plan.SURFACES:
        for budget in plan.BUDGETS:
            vector = governing_vectors[budget]
            for alpha in plan.ALPHA_ALL:
                hp = _hp_with_alpha(vector, alpha, c_M=(
                    None if vector["c_M"] is None else float(vector["c_M"])))
                for session in sorted(materials[surface]):
                    material = materials[surface][session]
                    rollout = _rollout(
                        decoder, material, budget=budget,
                        anchor=anchors[surface][session][budget],
                        carrier=carriers[surface][session][budget],
                        hp=hp, row_spec=row_spec_p1,
                    )
                    r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
                    receipts = rollout["receipts"]
                    zero_movement = all(receipt["cb"] == receipt["ca"] for receipt in receipts)
                    d_rows[surface][budget].setdefault(alpha, {})[session] = {
                        "surface": surface, "session": session,
                        "family": "D", "cell": "F01m_governing_law_alpha_replay",
                        "budget": int(budget), "alpha_M": float(alpha),
                        "hyperparameters": hp.payload(),
                        "row_spec": row_spec_p1.payload(),
                        "window_count": int(rollout["prediction"].shape[0]),
                        "query_starts_sha256": sealed_digest(material.starts),
                        "target_sha256": sealed_digest(rollout["targets"]),
                        "prediction_sha256": sealed_digest(rollout["prediction"]),
                        "initial_carrier_sha256": rollout["initial_carrier_sha256"],
                        "final_carrier_sha256": rollout["final_carrier_sha256"],
                        "committed_rows": int(rollout["committed_rows"]),
                        "committed_label_counts": list(rollout["committed_label_counts"]),
                        "measurement_reasons": dict(rollout["measurement_reasons"]),
                        "gate_reasons": dict(rollout["gate_reasons"]),
                        "movements_count": len(rollout["movements"]),
                        "max_movement": (float(np.max(rollout["movements"]))
                                         if rollout["movements"] else 0.0),
                        "bank_digest_chain_length": len(rollout["bank_digest_chain"]),
                        "bank_final_sha256": rollout["bank_digest_chain"][-1],
                        "r2": r2,
                        "zero_carrier_movement_every_trial": bool(zero_movement),
                        "causality_chain_verified": bool(_causality_chain(receipts)),
                        "receipts": receipts,
                        **_leakage_labels(alpha),
                    }
                    d_r2[surface][budget].setdefault(alpha, {})[session] = r2
                    _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                             "the CPU hard timeout fired during the D family")
                decoder.clear_cache()
            print(f"[{time.monotonic() - started:8.1f}s] D {surface} m{budget} done "
                  f"(alphas {list(plan.ALPHA_ALL)})", flush=True)

    # -- the R family: the sealed stage-2 selection-grid binding -------------
    row_spec_p2 = stage_p_replay.ROW_SPECS["P2"]
    r_rows: dict[int, dict[float, dict[str, dict[str, Any]]]] = {
        budget: {} for budget in plan.BUDGETS
    }
    reproduction_checks: dict[str, Any] = {}
    for budget in plan.BUDGETS:
        vector = governing_vectors[budget]
        for alpha in plan.ALPHA_SCAN:
            hp = _hp_with_alpha(vector, alpha, c_M=None)
            recorded = _stage2_scored(sealed, budget, float(vector["rho_M"]), float(alpha))
            for session in sorted(materials[plan.SELECTION_SURFACE]):
                material = materials[plan.SELECTION_SURFACE][session]
                rollout = _rollout(
                    decoder, material, budget=budget,
                    anchor=anchors[plan.SELECTION_SURFACE][session][budget],
                    carrier=carriers[plan.SELECTION_SURFACE][session][budget],
                    hp=hp, row_spec=row_spec_p2,
                )
                r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
                sealed_value = float(recorded["per_session"][session])
                r_rows[budget].setdefault(alpha, {})[session] = {
                    "surface": plan.SELECTION_SURFACE, "session": session,
                    "family": "R", "cell": "F01m_selection_grid_binding",
                    "budget": int(budget), "alpha_M": float(alpha),
                    "hyperparameters": hp.payload(),
                    "row_spec": row_spec_p2.payload(),
                    "r2": r2,
                    "sealed_stage2_r2": sealed_value,
                    "abs_delta_vs_sealed": abs(r2 - sealed_value),
                    "committed_rows": int(rollout["committed_rows"]),
                    "bank_final_sha256": rollout["bank_digest_chain"][-1],
                    "causality_chain_verified": bool(
                        _causality_chain(rollout["receipts"])),
                    **_leakage_labels(alpha),
                }
                _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                         "the CPU hard timeout fired during the R family")
            decoder.clear_cache()
            per_session = r_rows[budget][alpha]
            max_abs = max(row["abs_delta_vs_sealed"] for row in per_session.values())
            reproduction_checks[f"m{budget}|alpha{float(alpha)}"] = {
                "recorded_mean_r2": float(recorded["mean_r2"]),
                "replay_mean_r2": governing_gates.equal_session_mean(
                    {name: row["r2"] for name, row in per_session.items()}),
                "per_session_abs_delta_max": float(max_abs),
                "tolerance": float(plan.ANCHOR_TOLERANCE_R2),
                "pass": bool(max_abs <= float(plan.ANCHOR_TOLERANCE_R2)),
            }
            _require(reproduction_checks[f"m{budget}|alpha{float(alpha)}"]["pass"],
                     f"the selection-grid reproduction check failed at m{budget} "
                     f"alpha={alpha} (max |delta| = {max_abs})")
            print(f"[{time.monotonic() - started:8.1f}s] R m{budget} alpha={alpha} "
                  f"reproduced (max |delta| = {max_abs:.3e})", flush=True)

    # -- anchors --------------------------------------------------------------
    d0_anchor_checks: dict[str, Any] = {}
    for surface in plan.SURFACES:
        for budget in plan.BUDGETS:
            for session in sorted(materials[surface]):
                mine = d_rows[surface][budget][float(plan.ALPHA_ANCHOR)][session]
                sealed_f01 = sealed_f01m[(surface, str(budget), session)]
                sealed_f00 = sealed_f00m[(surface, str(budget), session)]
                checks = {
                    "window_count_equal": mine["window_count"] == int(sealed_f01["window_count"]),
                    "query_starts_sha256_equal": (
                        mine["query_starts_sha256"] == sealed_f01["query_starts_sha256"]),
                    "target_sha256_equal": (
                        mine["target_sha256"] == sealed_f01["target_sha256"]),
                    "abs_delta_r2_vs_sealed_f01m": abs(
                        mine["r2"] - float(sealed_f01["r2"])),
                    "abs_delta_r2_vs_sealed_f00m": abs(
                        mine["r2"] - float(sealed_f00["r2"])),
                    "zero_movement_every_trial": mine["zero_carrier_movement_every_trial"],
                    "max_movement_zero": mine["max_movement"] == 0.0,
                    "final_carrier_equals_initial": (
                        mine["final_carrier_sha256"] == mine["initial_carrier_sha256"]),
                }
                checks["pass"] = bool(
                    checks["window_count_equal"]
                    and checks["query_starts_sha256_equal"]
                    and checks["target_sha256_equal"]
                    and checks["abs_delta_r2_vs_sealed_f01m"] <= float(plan.ANCHOR_TOLERANCE_R2)
                    and checks["abs_delta_r2_vs_sealed_f00m"] <= float(plan.ANCHOR_TOLERANCE_R2)
                    and checks["zero_movement_every_trial"]
                    and checks["max_movement_zero"]
                    and checks["final_carrier_equals_initial"]
                )
                d0_anchor_checks[f"{surface}|{session}|m{budget}"] = checks
                _require(checks["pass"],
                         f"the alpha=0 anchor failed at {surface}/{session}/m{budget}")
    noop_all = all(item["zero_movement_every_trial"]
                   and item["max_movement_zero"]
                   for item in d0_anchor_checks.values())

    # -- summaries, deltas, monotonicity, verdict ----------------------------
    summaries = {
        surface: {
            budget: {
                alpha: {
                    "equal_session_mean": governing_gates.equal_session_mean(
                        d_r2[surface][budget][alpha]),
                    "per_session_r2": dict(sorted(d_r2[surface][budget][alpha].items())),
                }
                for alpha in plan.ALPHA_ALL
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    deltas = {
        surface: {
            budget: {
                alpha: governing_gates.paired_delta(
                    d_r2[surface][budget][alpha], d_r2[surface][budget][float(plan.ALPHA_ANCHOR)])
                for alpha in plan.ALPHA_SCAN
            }
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    monotonicity = {
        surface: {
            budget: evaluate_monotonicity(
                {alpha: summaries[surface][budget][alpha]["equal_session_mean"]
                 for alpha in plan.ALPHA_ALL})
            for budget in plan.BUDGETS
        }
        for surface in plan.SURFACES
    }
    held_out_deltas = {
        f"m{budget}|alpha{float(alpha)}":
            deltas[plan.HELD_OUT_SURFACE][budget][alpha]["equal_session_mean_delta"]
        for budget in plan.BUDGETS for alpha in plan.ALPHA_SCAN
    }
    verdict = evaluate_verdict(held_out_deltas, epsilon=float(
        plan.VERDICT_LAW["boundary_epsilon"]))
    within_monotone_all = all(
        monotonicity[plan.SELECTION_SURFACE][budget]["nonincreasing_in_alpha"]
        and monotonicity[plan.SELECTION_SURFACE][budget]["anchor_is_maximum"]
        for budget in plan.BUDGETS
    )
    heldout_monotone_all = all(
        monotonicity[plan.HELD_OUT_SURFACE][budget]["nonincreasing_in_alpha"]
        and monotonicity[plan.HELD_OUT_SURFACE][budget]["anchor_is_maximum"]
        for budget in plan.BUDGETS
    )
    monotonicity_answer = {
        "question": plan.MONOTONICITY_LAW["question"],
        "within_scan_monotone_nonpositive": bool(within_monotone_all),
        "heldout_scan_monotone_nonpositive": bool(heldout_monotone_all),
        "answer": (
            "the held-out alpha-scan is ALSO monotonically non-positive in alpha "
            if heldout_monotone_all else
            "the held-out alpha-scan is NOT monotonically non-positive in alpha"
        ),
    }

    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "post_hoc_alpha_replay_inference_only_cpu",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": plan.AUTHORITY,
        "foundation": {
            "governing_root": plan.GOVERNING_ROOT_RELATIVE,
            "governing_attempt_sha256": plan.GOVERNING_ATTEMPT_SHA256,
            "governing_replay_sha256": plan.GOVERNING_REPLAY_SHA256,
            "governing_terminal_sha256": plan.GOVERNING_TERMINAL_SHA256,
            "audit_relative": plan.AUDIT_RELATIVE,
            "audit_sha256": plan.AUDIT_SHA256,
            "audit_verdict": governing_replay.plan.AUDIT_VERDICT,
            "machinery": dict(governing_replay.plan.MACHINERY),
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "batch_size": int(batch_size),
            "seeds": [42],
            "cpu_count_threads": int(plan.Torch_THREADS),
        },
        "cells": dict(plan.CELL_FAMILIES),
        "selection_grid_binding_disclosure": dict(plan.SELECTION_GRID_BINDING_DISCLOSURE),
        "hyperparameters_governing_verbatim": {
            f"m{budget}": dict(governing_vectors[budget]) for budget in plan.BUDGETS
        },
        "carrier_support_binding": binding_evidence,
        "d_family_rows": d_rows,
        "r_family_rows": r_rows,
        "summaries": summaries,
        "deltas_vs_alpha0_anchor": deltas,
        "monotonicity": monotonicity,
        "anchors": {
            "d0_vs_sealed_f01m_rows": d0_anchor_checks,
            "d0_all_pass": all(item["pass"] for item in d0_anchor_checks.values()),
            "d0_noop_all_pass": bool(noop_all),
            "r_vs_sealed_stage2_scores": reproduction_checks,
            "r_all_pass": all(item["pass"] for item in reproduction_checks.values()),
            "tolerance_law": {
                "per_session_abs_delta_r2_max_allowed": float(plan.ANCHOR_TOLERANCE_R2),
                "reason": (
                    "CPU float-reduction order cannot bitwise-match GPU "
                    "receipts; window counts, query starts and targets remain "
                    "bit-exact and the alpha=0 no-op law is exact by construction"
                ),
            },
        },
        "leakage_law": dict(plan.LEAKAGE_LAW),
        "verdict": verdict,
        "monotonicity_answer": monotonicity_answer,
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
    _require(replay_payload["anchors"]["d0_all_pass"],
             "the alpha=0 reproduction anchor failed at least one row")
    _require(replay_payload["anchors"]["d0_noop_all_pass"],
             "the alpha=0 exact no-op law failed")
    _require(replay_payload["anchors"]["r_all_pass"],
             "the selection-grid reproduction anchor failed")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "post_hoc_m2_carrier_alpha_diagnostic_inference_only_cpu",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "authority": plan.AUTHORITY,
        "leakage_statement": plan.LEAKAGE_LAW["receipt_statement"],
        "can_this_run_select_or_promote": False,
        "checkpoint_selection_eligible": False,
        "deployment_eligible": False,
        "post_hoc_diagnostic": True,
        "foundation": replay_payload["foundation"],
        "hyperparameters_governing_verbatim": replay_payload["hyperparameters_governing_verbatim"],
        "equal_session_means": summaries,
        "held_out_deltas": deltas[plan.HELD_OUT_SURFACE],
        "within_deltas": deltas[plan.SELECTION_SURFACE],
        "monotonicity": monotonicity,
        "monotonicity_answer": monotonicity_answer,
        "selection_grid_reproduction": reproduction_checks,
        "anchors": {
            "d0_all_pass": replay_payload["anchors"]["d0_all_pass"],
            "d0_noop_all_pass": replay_payload["anchors"]["d0_noop_all_pass"],
            "r_all_pass": replay_payload["anchors"]["r_all_pass"],
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
        "monotonicity_answer": monotonicity_answer["answer"],
        "held_out_deltas": held_out_deltas,
        "anchors_all_pass": (
            replay_payload["anchors"]["d0_all_pass"]
            and replay_payload["anchors"]["d0_noop_all_pass"]
            and replay_payload["anchors"]["r_all_pass"]
        ),
        "wall_seconds": replay_payload["wall_seconds"],
    }
