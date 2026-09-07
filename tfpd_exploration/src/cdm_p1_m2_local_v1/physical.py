"""Physical driver of the LOCAL-M2 transfer: selection, cells, anchors, receipts.

GPU 1 only, inference only, one launch.  The staged receipt law mirrors Part A
and the SUA transfer: ``attempt.json`` (reserved before any data),
``replay.json`` (the governing grid), ``terminal.json`` (receipt-only
composition).  Nothing under any frozen result root is created or modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from . import gates, plan, replay
from .anchor import M2SupportAnchor, build_groups


class M2LocalPhysicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M2LocalPhysicalError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    body = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _validate_environment() -> None:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == plan.ENVIRONMENT_LAW["cuda_visible_devices"],
             "the LOCAL-M2 route is bound to GPU 1 (the work-order device law)")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == plan.ENVIRONMENT_LAW["cuda_device_order"],
             "CUDA_DEVICE_ORDER mismatch")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == plan.ENVIRONMENT_LAW["cublas_workspace_config"],
             "CUBLAS_WORKSPACE_CONFIG mismatch")


def _bind_namespaces(repo_root: Path) -> None:
    """Make the tfpd ``src`` package and the streaming ``src`` tree coexist.

    The frozen stage-O/P modules and this package import ``src.*`` from
    tfpd_exploration; the frozen M2 runtime (via ``export_t4_payload``) imports
    ``src.data``/``src.models`` from streaming_calibration_exp.  Extending the
    tfpd ``src`` package path with the streaming source root (the sealed
    cdm_p1_sua_v1 pattern) resolves both under one namespace.
    """
    for entry in (str(repo_root), str(repo_root / "tfpd_exploration"),
                  str(repo_root / "sua_exploration")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import src as tfpd_src_package

    streaming_src = repo_root / "streaming_calibration_exp" / "src"
    _require(streaming_src.is_dir(), "the streaming source root is absent")
    if str(streaming_src) not in tfpd_src_package.__path__:
        tfpd_src_package.__path__.append(str(streaming_src))


def _verify_sidecar(path: Path) -> str:
    digest = _sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    _require(sidecar.read_text(encoding="ascii").strip() == f"{digest}  {path.name}",
             f"sidecar drift for {path.name}")
    return digest


def _load_sealed_same_query_rows(repo_root: Path) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    path = repo_root / plan.SAME_QUERY_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.SAME_QUERY_SCORE_SHA256, "sealed same-query score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_same_query_comparator_v1"
             and payload.get("status") == "TERMINAL", "sealed comparator schema drift")
    rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if str(row.get("cell", "")).startswith("t4_ridge_static_m"):
            budget = int(row["budget"])
            rows[(str(row["surface"]), str(row["session"]), budget)] = row
    _require(len(rows) == (plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS)
             * len(plan.BUDGETS_F),
             "sealed anchor row topology drift")
    return rows


def _load_sealed_cdm_rows(repo_root: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    path = repo_root / plan.CDM_SCREEN_SCORE_RELATIVE
    _require(_sha256_file(path) == plan.CDM_SCREEN_SCORE_SHA256, "sealed CDM screen score drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "m2_precision_cdm_v2_screen_v1"
             and payload.get("status") == "TERMINAL", "sealed CDM screen schema drift")
    rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row.get("cell") == "m4_activity_only":
            rows[(str(row["surface"]), str(row["session_id"]))] = row
    _require(len(rows) == 13, "sealed m4_activity_only row topology drift")
    return rows


def execute(repo_root: Path, *, gpu_index: int = plan.GPU_INDEX,
            batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    _validate_environment()
    _require(int(gpu_index) == plan.GPU_INDEX, "the LOCAL-M2 route is bound to GPU 1")
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
                 f"an owned LOCAL-M2 module drifted: {relative}")

    _bind_namespaces(repo_root)
    from src.m2_same_query_comparator_v1.core import (
        array_sha256 as sealed_digest,
    )
    from src.m2_same_query_comparator_v1.core import variance_weighted_r2

    import torch

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    _require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    started = time.monotonic()

    model, data_module, _task_config, metadata = load_frozen_model_and_data()
    _require(metadata["checkpoint_sha256"] == plan.T4_CHECKPOINT_SHA256, "T4 checkpoint drift")
    _require(metadata["teacher_checkpoint_sha256"] == plan.SPINT_CHECKPOINT_SHA256,
             "SPINT teacher checkpoint drift")
    _require(metadata["normalization_sha256"] == plan.NORMALIZATION_SHA256, "T4 normalization drift")
    student = model.student.to(device).eval()
    # The sealed screens' freeze law (the comparator's own loop): inference-only
    # evaluation with every parameter pinned non-trainable before any forward.
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    side_mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    side_std = np.asarray(metadata["normalization_std"], dtype=np.float32)
    decoder = replay.M2Decoder(torch, student, device, batch_size=batch_size)

    surfaces = {
        "within_post30": (data_module.train_dataset, data_module.train_calib_heldin_sessions),
        "external_official_query": (data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions),
    }
    expected_counts = {
        "within_post30": plan.EXPECTED_WITHIN_SESSIONS,
        "external_official_query": plan.EXPECTED_EXTERNAL_SESSIONS,
    }
    sealed_rows = _load_sealed_same_query_rows(repo_root)
    sealed_cdm_rows = _load_sealed_cdm_rows(repo_root)

    materials: dict[str, dict[str, replay.SessionMaterial]] = {}
    carriers: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    anchors: dict[str, dict[str, dict[int, M2SupportAnchor]]] = {}
    anchor_evidence: dict[str, Any] = {}
    for surface, (dataset, _raw) in surfaces.items():
        _require(dataset is not None, f"{surface}: dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        _require(len(sessions) == expected_counts[surface], f"{surface}: session count drift")
        materials[surface] = {}
        carriers[surface] = {}
        anchors[surface] = {}
        for session in sessions:
            materials[surface][session] = replay.build_session_material(
                dataset=dataset, session=session, surface=surface,
            )
            carriers[surface][session] = {}
            anchors[surface][session] = {}
            for budget in plan.BUDGETS_F:
                carrier = replay.sealed_carrier(dataset, session, budget)
                groups = build_groups(carrier["raw_t4"], materials[surface][session].channel_ids)
                anchor = M2SupportAnchor.from_labeled_support(
                    groups=groups,
                    support_trial_rates=carrier["rates"],
                    support_angles_rad=carrier["theta"].tolist(),
                    support_t4=carrier["raw_t4"],
                )
                _require(anchor.support_coefficient_parity["rebuilt_rows_bitwise_equal"],
                         f"anchor zero-evidence fallback drifted from the sealed carrier "
                         f"({surface}/{session} m{budget})")
                sealed = sealed_rows[(surface, session, budget)]
                _require(carrier["selected"].tolist() == sealed["side_evidence"]["selected_indices"],
                         f"selected support drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["raw_t4"]) == sealed["side_evidence"]["raw_t4_sha256"],
                         f"raw carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["side"]) == sealed["side_evidence"]["normalized_t4_sha256"],
                         f"normalized carrier digest drift {surface}/{session} m{budget}")
                _require(sealed_digest(carrier["activity"]) == sealed["activity_sha256"],
                         f"activity digest drift {surface}/{session} m{budget}")
                carriers[surface][session][budget] = carrier
                anchors[surface][session][budget] = anchor
                anchor_evidence[f"{surface}|{session}|m{budget}"] = {
                    "selected": carrier["selected"].tolist(),
                    "selected_sha256": carrier["selected_sha256"],
                    "theta_sha256": carrier["theta_sha256"],
                    "rates_sha256": carrier["rates_sha256"],
                    "raw_t4_sha256": carrier["raw_t4_sha256"],
                    "side_sha256": carrier["side_sha256"],
                    "activity_sha256": carrier["activity_sha256"],
                    "fit_evidence": carrier["fit_evidence"],
                    "anchor_payload_digest": anchor.digest,
                    "anchor_parity": anchor.support_coefficient_parity,
                    "groups_digest": anchor.groups.digest,
                    "group_sizes": [
                        int(anchor.groups.unit_indices(group).size)
                        for group in range(cdm_core.GROUP_COUNT)
                    ],
                    "invalid_units": int((~np.asarray(anchor.groups.valid_mask)).sum()),
                }

    selection = {
        f"m{budget}": replay.selection_pass(
            decoder,
            [materials["within_post30"][session] for session in sorted(materials["within_post30"])],
            {session: anchors["within_post30"][session][budget]
             for session in anchors["within_post30"]},
            {session: carriers["within_post30"][session][budget]
             for session in carriers["within_post30"]},
            budget=budget, started=started,
        )
        for budget in (plan.M4, plan.M10)
    }
    selected_hp: dict[int, stage_p_replay.CellHyperparameters] = {}
    for budget in (plan.M4, plan.M10):
        selected = selection[f"m{budget}"]["selected"]
        thresholds = stage_p_gate.GateThresholds(
            tau_d=float(selected["thresholds"]["tau_d_rad"]),
            r_max=int(selected["thresholds"]["r_max_repetition_per_direction"]),
            d_min=int(selected["thresholds"]["d_min_distinct_directions"]),
            max_mass_relative=float(
                selected["thresholds"]["max_pseudo_mass_relative_to_support_rows"]
            ),
        )
        selected_hp[budget] = stage_p_replay.CellHyperparameters(
            rho_M=float(selected["rho_M"]), alpha_M=float(selected["alpha_M"]),
            c_M=(None if selected["c_M"] is None else float(selected["c_M"])),
            thresholds=thresholds,
        )
    hp_m30 = stage_p_replay.CellHyperparameters(
        rho_M=float(plan.M30_NOOP_LAW["rho_M"]),
        alpha_M=float(plan.M30_NOOP_LAW["alpha_M"]),
        c_M=None, thresholds=selected_hp[plan.M4].thresholds,
    )

    matrix: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]] = {
        surface: {budget: {} for budget in plan.BUDGETS_F} for surface in surfaces
    }
    anchors_sealed: dict[str, Any] = {}
    noop_proofs: dict[str, Any] = {}
    causality_all = True
    row_spec = stage_p_replay.ROW_SPECS["P1"]
    for surface in plan.SURFACES_F:
        for budget in plan.BUDGETS_F:
            for spec in replay.CELL_SPECS_F:
                cell_rows: dict[str, dict[str, Any]] = {}
                for session in sorted(materials[surface]):
                    material = materials[surface][session]
                    carrier = carriers[surface][session][budget]
                    anchor = anchors[surface][session][budget]
                    hp = selected_hp[budget] if budget != plan.M30 else hp_m30
                    rollout = replay.rollout_session_f(
                        decoder, material, budget=budget, spec=spec,
                        anchor=anchor, carrier=carrier,
                        hp=hp if spec.online else None,
                        config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
                        row_spec=row_spec if spec.online else None,
                    )
                    r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
                    receipts = rollout["receipts"]
                    chain_ok = True
                    previous_ca: Optional[str] = None
                    for receipt in receipts:
                        if previous_ca is not None and receipt["cb"] != previous_ca:
                            chain_ok = False
                        previous_ca = receipt["ca"]
                    row = {
                        "surface": surface, "session": session, "cell": spec.name,
                        "budget": int(budget), "carrier_law": spec.carrier_law,
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
                        "max_movement": (float(np.max(rollout["movements"])) if rollout["movements"] else 0.0),
                        "identity_sha256": rollout["identity_sha256"],
                        "r2": r2,
                        "receipts": receipts,
                        "causality_chain_verified": bool(chain_ok),
                    }
                    causality_all = causality_all and row["causality_chain_verified"]
                    if spec.name == "F00m_anchor":
                        sealed = sealed_rows[(surface, session, budget)]
                        matches = {
                            "prediction_sha256": row["prediction_sha256"] == sealed["prediction_sha256"],
                            "target_sha256": row["target_sha256"] == sealed["target_sha256"],
                            "r2": row["r2"] == float(sealed["r2"]),
                            "window_count": row["window_count"] == int(sealed["window_count"]),
                            "ordered_window_starts_sha256": (
                                row["query_starts_sha256"] == sealed["ordered_window_starts_sha256"]
                            ),
                            "activity_sha256": (
                                sealed_digest(carrier["activity"]) == sealed["activity_sha256"]
                            ),
                        }
                        anchors_sealed[f"{surface}|{session}|m{budget}"] = {
                            "field_matches": matches,
                            "exact_match": all(bool(value) for value in matches.values()),
                        }
                        _require(anchors_sealed[f"{surface}|{session}|m{budget}"]["exact_match"],
                                 f"the F00m anchor failed against the sealed row "
                                 f"{surface}/{session}/m{budget}")
                    cell_rows[session] = row
                    _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                             "the M2 hard timeout fired during the governing grid")
                matrix[surface][budget][spec.name] = cell_rows
            decoder.clear_cache()

    for surface in plan.SURFACES_F:
        for session in sorted(materials[surface]):
            f00 = matrix[surface][plan.M30]["F00m"][session]
            f01 = matrix[surface][plan.M30]["F01m"][session]
            equal = f00["prediction_sha256"] == f01["prediction_sha256"] and f00["r2"] == f01["r2"]
            zero_movement = all(receipt["cb"] == receipt["ca"] for receipt in f01["receipts"])
            noop_proofs[f"{surface}|{session}"] = {
                "bitwise_equal_predictions": bool(equal),
                "carrier_unchanged_every_trial": bool(zero_movement),
                "committed_rows": int(f01["committed_rows"]),
            }
            _require(equal and zero_movement,
                     f"the M30 exact no-op law broke at {surface}/{session}")

    r2_view: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        surface: {
            budget: {
                cell: {
                    session: float(matrix[surface][budget][cell][session]["r2"])
                    for session in matrix[surface][budget][cell]
                }
                for cell in matrix[surface][budget]
            }
            for budget in plan.BUDGETS_F
        }
        for surface in matrix
    }
    summaries = {
        surface: {
            budget: {
                cell: {
                    "equal_session_mean": gates.equal_session_mean(r2_view[surface][budget][cell]),
                    "per_session_r2": dict(sorted(r2_view[surface][budget][cell].items())),
                }
                for cell in r2_view[surface][budget]
            }
            for budget in plan.BUDGETS_F
        }
        for surface in r2_view
    }
    within_by_budget = {
        budget: r2_view["within_post30"][budget] for budget in plan.BUDGETS_F
    }
    safety = {
        "f00m_bit_anchor_vs_sealed_same_query_rows": all(
            item["exact_match"] for item in anchors_sealed.values()
        ),
        "anchor_zero_evidence_parity_all_selected_supports": True,
        "m30_exact_noop": all(
            item["bitwise_equal_predictions"] and item["carrier_unchanged_every_trial"]
            for item in noop_proofs.values()
        ),
        "causality_receipt_chains": bool(causality_all),
        "hyperparameters_reselected_on_m2_source_folds": True,
        "dandi_hyperparameters_ported_as_claims": False,
        "zero_target_updates": True,
    }
    gate = gates.evaluate_m2_local_gate(
        external_m4=r2_view["external_official_query"][plan.M4],
        within_by_budget=within_by_budget,
        external_m30=r2_view["external_official_query"][plan.M30],
        safety=safety, gates_law=plan.GATES,
    )

    # -- the G family (full champion stack) ---------------------------------
    from src.pseudo_mua_precision_cdm_v2_screen_v1.core import (
        variance_weighted_r2 as g_r2,
    )

    g_surfaces = {
        "within_post30": (data_module.train_dataset, data_module.train_calib_heldin_sessions),
        "external_post30_local": (data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions),
    }
    g_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        surface: {"G00m": {}, "G01m": {}} for surface in g_surfaces
    }
    g_anchor_evidence: dict[str, Any] = {}
    g_anchors_sealed: dict[str, Any] = {}
    for surface, (dataset, raw_sessions) in g_surfaces.items():
        for session in sorted(dataset.calib_trialized_neural_features):
            views = replay._g_session_views(raw_sessions, session)
            support = replay.g_support_material(session=session, views=views)
            query_rows = replay.g_query_rows(ds=dataset, session=session, views=views)
            anchor_g, carrier0 = replay.build_g_anchor(support)
            parity = anchor_g.support_coefficient_parity
            g_anchor_evidence[f"{surface}|{session}"] = {
                "selected": support["selected"].tolist(),
                "anchor_digest": anchor_g.digest,
                "anchor_parity": parity,
                "groups_digest": anchor_g.groups.digest,
            }
            _require(parity.get("rebuilt_bitwise_equal", False) or parity.get(
                "rebuilt_rows_bitwise_equal", False),
                f"the G anchor parity failed at {surface}/{session}")
            g00 = replay.rollout_g00m(
                torch=torch, model=model, ds=dataset, session=session, views=views,
                support=support, query_rows=query_rows, side_mean=side_mean,
                side_std=side_std, device=device, batch_size=batch_size,
            )
            sealed = sealed_cdm_rows[(surface, session)]
            matches = {
                "prediction_sha256": g00["prediction_sha256"] == sealed["prediction_sha256"],
                "target_sha256": g00["target_sha256"] == sealed["target_sha256"],
                "r2": float(g00["r2"]) == float(sealed["r2"]),
                "window_count": int(g00["window_count"]) == int(sealed["window_count"]),
            }
            g_anchors_sealed[f"{surface}|{session}"] = {
                "field_matches": matches,
                "exact_match": all(bool(value) for value in matches.values()),
            }
            _require(g_anchors_sealed[f"{surface}|{session}"]["exact_match"],
                     f"the G00m anchor failed against the sealed row {surface}/{session}")
            g01 = replay.rollout_g01m(
                torch=torch, model=model, ds=dataset, session=session, views=views,
                support=support, query_rows=query_rows, side_mean=side_mean,
                side_std=side_std, device=device, batch_size=batch_size,
                anchor=anchor_g, hp=selected_hp[plan.M4], row_spec=row_spec,
            )
            g01_r2 = float(g_r2(g01["targets"], g01["prediction"]))
            g_rows[surface]["G00m"][session] = {
                "surface": surface, "session": session, "cell": "G00m",
                "budget": plan.BUDGET_G, "r2": float(g00["r2"]),
                "window_count": int(g00["window_count"]),
                "prediction_sha256": g00["prediction_sha256"],
                "target_sha256": g00["target_sha256"],
                "query_starts_sha256": g00["query_starts_sha256"],
                "source": "sealed_m4_activity_only_law_verbatim",
                "parameter_updates": 0,
            }
            g_rows[surface]["G01m"][session] = {
                "surface": surface, "session": session, "cell": "G01m",
                "budget": plan.BUDGET_G, "r2": g01_r2,
                "window_count": int(g01["window_count"]),
                "prediction_sha256": g01["prediction_sha256"],
                "target_sha256": g01["target_sha256"],
                "query_starts_sha256": g01["query_starts_sha256"],
                "initial_carrier_sha256": g01["initial_carrier_sha256"],
                "final_carrier_sha256": g01["final_carrier_sha256"],
                "committed_rows": int(g01["committed_rows"]),
                "committed_label_counts": list(g01["committed_label_counts"]),
                "measurement_reasons": dict(g01["measurement_reasons"]),
                "gate_reasons": dict(g01["gate_reasons"]),
                "movements_count": len(g01["movements"]),
                "max_movement": (float(np.max(g01["movements"])) if g01["movements"] else 0.0),
                "receipts": g01["receipts"],
                "source": "new_full_stack_p1_online",
                "parameter_updates": 0,
            }
            _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                     "the M2 hard timeout fired during the G family")
    g_r2_view = {
        surface: {
            cell: {session: float(g_rows[surface][cell][session]["r2"])
                   for session in g_rows[surface][cell]}
            for cell in g_rows[surface]
        }
        for surface in g_rows
    }
    g_summaries = {
        surface: {
            cell: {
                "equal_session_mean": gates.equal_session_mean(g_r2_view[surface][cell]),
                "per_session_r2": dict(sorted(g_r2_view[surface][cell].items())),
            }
            for cell in g_r2_view[surface]
        }
        for surface in g_r2_view
    }
    secondary = gates.evaluate_secondary_gate(
        external_m4=g_r2_view["external_post30_local"], gates_law=plan.GATES,
    )
    stops = gates.stop_conditions(gate, secondary)
    f00m_vs_anchor_batching = {
        f"{surface}|{session}|m{budget}": {
            "per_trial_r2": float(matrix[surface][budget]["F00m"][session]["r2"]),
            "sealed_batching_r2": float(matrix[surface][budget]["F00m_anchor"][session]["r2"]),
            "abs_delta": abs(
                float(matrix[surface][budget]["F00m"][session]["r2"])
                - float(matrix[surface][budget]["F00m_anchor"][session]["r2"])
            ),
        }
        for surface in plan.SURFACES_F for budget in plan.BUDGETS_F
        for session in sorted(materials[surface])
    }
    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "f00m_f01m_g00m_g01m_inference_only_local_m2",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "audit_foundation": {
            "audit_relative": plan.AUDIT_RELATIVE,
            "audit_sha256": plan.AUDIT_SHA256,
            "verdict": plan.AUDIT_VERDICT,
            "reused": dict(plan.AUDIT_REUSED),
            "m2_arm_disposition": (
                "P1-M2 NOT_EVALUABLE_OFFICIAL_CONTRACT (sealed); this route is "
                "LOCAL-protocol evidence only"
            ),
        },
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "gpu_name": torch.cuda.get_device_name(0),
            "batch_size": int(batch_size),
            "seeds": [42],
        },
        "surface": {
            "f_family": dict(plan.SURFACE_QUERY_LAW),
            "g_family": "the sealed Native-M2 CDM screen surfaces",
            "within_sessions": sorted(materials["within_post30"]),
            "external_sessions": sorted(materials["external_official_query"]),
            "evidence_stream": (
                f"completed trials with position >= {plan.EVIDENCE_START_POSITION} "
                "(the first-30 calibration pool never enters the bank)"
            ),
            "velocity_unit_law": dict(plan.VELOCITY_UNIT_LAW),
            "short_trial_policy": plan.SHORT_TRIAL_POLICY,
        },
        "cells": {
            "F00m_anchor": dict(plan.CARRIER_AXIS_LAW)["F00m_anchor"],
            "F00m": dict(plan.CARRIER_AXIS_LAW)["F00m"],
            "F01m": dict(plan.CARRIER_AXIS_LAW)["F01m"],
            "G00m": dict(plan.FULL_STACK_LAW)["G00m"],
            "G01m": dict(plan.FULL_STACK_LAW)["G01m"],
        },
        "machinery": dict(plan.MACHINERY),
        "carrier_evidence": anchor_evidence,
        "hyperparameter_selection": selection,
        "hyperparameters": {
            "m4": selected_hp[plan.M4].payload(),
            "m10": selected_hp[plan.M10].payload(),
            "m30": {
                "rho_M": float(plan.M30_NOOP_LAW["rho_M"]),
                "alpha_M": float(plan.M30_NOOP_LAW["alpha_M"]),
                "c_M": None,
                "thresholds": selected_hp[plan.M4].thresholds.payload(),
                "inherits_thresholds_from": "the selected M4 vector",
            },
            "g_family_uses": "the selected M4 vector (the DANDI full-stack inheritance shape)",
            "law": dict(plan.SELECTION_LAW),
        },
        "summaries": summaries,
        "matrix": matrix,
        "anchors": {
            "f00m_vs_sealed_same_query_rows": anchors_sealed,
            "all_exact": all(item["exact_match"] for item in anchors_sealed.values()),
            "g00m_vs_sealed_cdm_rows": g_anchors_sealed,
            "g00m_all_exact": all(item["exact_match"] for item in g_anchors_sealed.values()),
            "g_anchor_evidence": g_anchor_evidence,
            "m30_noop_proofs": noop_proofs,
            "m30_noop_all_pass": all(
                item["bitwise_equal_predictions"] and item["carrier_unchanged_every_trial"]
                for item in noop_proofs.values()
            ),
            "f00m_vs_anchor_batching_disclosure": f00m_vs_anchor_batching,
            "f00m_batching_max_abs_delta": max(
                item["abs_delta"] for item in f00m_vs_anchor_batching.values()
            ),
        },
        "g_family": {
            "summaries": g_summaries,
            "rows": g_rows,
            "secondary_gate": secondary,
            "law": dict(plan.FULL_STACK_LAW),
        },
        "gate": gate,
        "stop_conditions": stops,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    _require(replay_payload["anchors"]["all_exact"],
             "the F00m anchor failed at least one sealed same-query row")
    _require(replay_payload["anchors"]["g00m_all_exact"],
             "the G00m anchor failed at least one sealed CDM-screen row")
    _require(replay_payload["anchors"]["m30_noop_all_pass"], "the M30 exact no-op law failed")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "part_b2_local_m2_inference_only_transfer",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "audit_foundation": replay_payload["audit_foundation"],
        "gate": gate,
        "secondary_full_stack_gate": secondary,
        "stop_conditions": stops,
        "equal_session_means": summaries,
        "g_family_equal_session_means": g_summaries,
        "hyperparameters": replay_payload["hyperparameters"],
        "anchors": {
            "f00m_all_exact": replay_payload["anchors"]["all_exact"],
            "g00m_all_exact": replay_payload["anchors"]["g00m_all_exact"],
            "m30_noop_all_pass": replay_payload["anchors"]["m30_noop_all_pass"],
            "f00m_batching_max_abs_delta": replay_payload["anchors"]["f00m_batching_max_abs_delta"],
        },
        "external_roster_opened_by_terminal": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "gate_summary": {
            "promoted": gate["promoted"],
            "external_m4_delta": gate["promotion"]["delta"]["equal_session_mean_delta"],
            "positive_external_m4_sessions": gate["promotion"]["delta"]["positive_sessions"],
            "safety_all_pass": gate["safety_all_pass"],
            "stop_conditions_fired": stops["fired"],
            "secondary_g_delta": secondary["delta"]["equal_session_mean_delta"],
            "secondary_g_promoted": secondary["promoted"],
        },
    }
