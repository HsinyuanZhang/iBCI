"""Physical driver of the SUA transfer: selection, cells, anchors, receipts.

GPU 1 only, inference only, one launch.  The staged receipt law mirrors Part A:
``attempt.json`` (reserved before any data), ``replay.json`` (the governing
grid), ``terminal.json`` (receipt-only composition).
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from . import gates, plan, replay
from .anchor import SUASupportAnchor, build_groups


class SuaPhysicalError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuaPhysicalError(message)


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
             "Part B is bound to GPU 1 (the work-order device law)")
    _require(os.environ.get("CUDA_DEVICE_ORDER") == plan.ENVIRONMENT_LAW["cuda_device_order"],
             "CUDA_DEVICE_ORDER mismatch")
    _require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == plan.ENVIRONMENT_LAW["cublas_workspace_config"],
             "CUBLAS_WORKSPACE_CONFIG mismatch")


def _load_sealed_anchor_rows(repo_root: Path) -> dict[str, Mapping[str, Any]]:
    path = repo_root / plan.SEALED_PAIRED_SCORE_RELATIVE
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[str, Mapping[str, Any]] = {}
    for row in payload["rows"]:
        if row["view"] == plan.VIEW and row["cell"] == plan.SEALED_ANCHOR_CELL:
            rows[f"{row['session_id']}|seed{row['seed']}"] = row
    _require(len(rows) == plan.EXPECTED_EXTERNAL_SESSIONS * len(plan.SEEDS),
             "sealed anchor row topology drift")
    return rows


def execute(repo_root: Path, *, gpu_index: int = plan.GPU_INDEX,
            batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    _validate_environment()
    _require(int(gpu_index) == plan.GPU_INDEX, "Part B is bound to GPU 1")
    repo_root = Path(repo_root).absolute()
    root = plan.result_root(repo_root)
    _require(not (root / "replay.json").exists(), "the replay receipt already exists")
    _require(not (root / "terminal.json").exists(), "the terminal receipt already exists")
    _require((root / "attempt.json").exists(), "reserve the attempt receipt first")
    attempt_path = root / "attempt.json"
    attempt_digest = _sha256_file(attempt_path)
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    sidecar = attempt_path.with_name(attempt_path.name + ".sha256")
    _require(sidecar.exists() and sidecar.read_text(encoding="ascii").strip()
             == f"{attempt_digest}  attempt.json", "attempt sidecar drift")
    for relative, digest in attempt["predecessor_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an immutable predecessor drifted: {relative}")
    for relative, digest in attempt["owned_sha256s"].items():
        _require(_sha256_file(repo_root / relative) == digest,
                 f"an owned Part-B module drifted: {relative}")

    import sys

    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))
    # The SUA runtime's owners import ``eval_adaptation_dandi688``, which
    # imports ``src.models...`` from streaming_calibration_exp, while this
    # package (and the frozen stage-O/P modules) import ``src.*`` from
    # tfpd_exploration.  Extend the tfpd ``src`` package path with the
    # streaming source root so BOTH namespaces resolve under one ``src``.
    import src as tfpd_src_package

    streaming_src = repo_root / "streaming_calibration_exp" / "src"
    _require(streaming_src.is_dir(), "the streaming source root is absent")
    for entry in (str(streaming_src),):
        if entry not in tfpd_src_package.__path__:
            tfpd_src_package.__path__.append(entry)

    from src.sua_paired_activity_budget_screen_v1 import physical as paired_physical
    from src.sua_paired_activity_budget_screen_v1.core import (
        array_sha256 as paired_digest,
    )
    from src.sua_paired_activity_budget_screen_v1.core import variance_weighted_r2
    from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate

    contract, authority = paired_physical._relocated_contract(repo_root)
    paired_physical._verify_selected_inputs(contract, nwb_root=(repo_root / plan.NWB_ROOT_RELATIVE).resolve())
    sealed_rows = _load_sealed_anchor_rows(repo_root)

    for entry in (repo_root / "sua_exploration", repo_root / "sua_exploration" / "scripts",
                  repo_root / "streaming_calibration_exp"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime

    owners = runtime._runtime_owners(repo_root)
    torch = owners["torch"]
    _require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    models = {
        seed: runtime._load_model(contract.checkpoint("shared_t4", seed), contract, str(device), owners)
        for seed in plan.SEEDS
    }
    for model in models.values():
        model.eval()
        for parameter in model.parameters():
            _require(parameter.requires_grad is False, "frozen evaluator exposes trainable parameters")
    behavior_stats = runtime._load_mean_std(
        contract.behavior_normalizer_paths[plan.VIEW], label=f"{plan.VIEW} behavior")
    side_stats = runtime._load_mean_std(
        contract.side_normalizer_paths[plan.VIEW], label=f"{plan.VIEW} side")
    decoders = {
        seed: replay.SuaDecoder(torch, models[seed], device, batch_size=batch_size)
        for seed in plan.SEEDS
    }

    started = time.monotonic()
    nwb_root = (repo_root / plan.NWB_ROOT_RELATIVE).resolve()
    external_ids = [row.session_id for row in contract.cohort]
    _require(len(external_ids) == plan.EXPECTED_EXTERNAL_SESSIONS, "external roster drift")
    materials: dict[str, replay.SessionMaterial] = {}
    for session_id in list(plan.WITHIN_SESSIONS) + external_ids:
        path = next((nwb_root / row.frozen_path for row in contract.cohort if row.session_id == session_id),
                    plan.nwb_path(repo_root, session_id))
        materials[session_id] = replay.build_session_material(
            repo_root=repo_root, nwb_path=path, owners=owners,
            behavior_mean=behavior_stats[0], behavior_std=behavior_stats[1],
            side_mean=side_stats[0], side_std=side_stats[1],
        )
        _require(materials[session_id].session_id == session_id, "session identity drift")
    total_windows = sum(
        materials[session].query_starts.size for session in external_ids
    )
    _require(total_windows == plan.EXPECTED_QUERY_WINDOWS, "external query window count drift")

    carriers: dict[str, dict[int, dict[str, Any]]] = {}
    anchors: dict[str, dict[int, SUASupportAnchor]] = {}
    for session_id, material in materials.items():
        carriers[session_id] = {}
        anchors[session_id] = {}
        for budget in plan.BUDGETS:
            carrier = replay.sealed_carrier(material, budget)
            groups = build_groups(carrier["raw_t4"], material.channel_ids)
            anchor = SUASupportAnchor.from_labeled_support(
                groups=groups,
                support_trial_rates_hz=carrier["rates"],
                support_angles_rad=carrier["theta"].tolist(),
                support_t4=carrier["raw_t4"],
            )
            _require(anchor.support_coefficient_parity["rebuilt_rows_bitwise_equal"],
                     f"anchor zero-evidence fallback drifted from the sealed carrier "
                     f"({session_id} m{budget})")
            carriers[session_id][budget] = carrier
            anchors[session_id][budget] = anchor

    selection = replay.selection_pass(
        decoders[plan.SELECTION_SEED],
        [materials[session] for session in plan.WITHIN_SESSIONS],
        {session: anchors[session][plan.M4] for session in plan.WITHIN_SESSIONS},
        {session: carriers[session][plan.M4] for session in plan.WITHIN_SESSIONS},
        budget=plan.M4, started=started,
    )
    selected = selection["selected"]
    selected_thresholds = stage_p_gate.GateThresholds(
        tau_d=float(selected["thresholds"]["tau_d_rad"]),
        r_max=int(selected["thresholds"]["r_max_repetition_per_direction"]),
        d_min=int(selected["thresholds"]["d_min_distinct_directions"]),
        max_mass_relative=float(
            selected["thresholds"]["max_pseudo_mass_relative_to_support_rows"]
        ),
    )
    hp_m4 = stage_p_replay.CellHyperparameters(
        rho_M=float(selected["rho_M"]), alpha_M=float(selected["alpha_M"]),
        c_M=(None if selected["c_M"] is None else float(selected["c_M"])),
        thresholds=selected_thresholds,
    )
    hp_m30 = stage_p_replay.CellHyperparameters(
        rho_M=float(plan.M30_NOOP_LAW["rho_M"]),
        alpha_M=float(plan.M30_NOOP_LAW["alpha_M"]),
        c_M=None, thresholds=selected_thresholds,
    )

    matrix: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        roster: {budget: {} for budget in plan.BUDGETS}
        for roster in (plan.WITHIN_ROSTER_KEY, plan.EXTERNAL_ROSTER_KEY)
    }
    anchors_sealed: dict[str, Any] = {}
    noop_proofs: dict[str, Any] = {}
    causality_all = True
    for roster, session_ids in (
        (plan.WITHIN_ROSTER_KEY, list(plan.WITHIN_SESSIONS)),
        (plan.EXTERNAL_ROSTER_KEY, external_ids),
    ):
        for budget in plan.BUDGETS:
            for spec in replay.CELL_SPECS:
                if spec.name == "F00s_anchor" and (roster != plan.EXTERNAL_ROSTER_KEY or budget != plan.M4):
                    continue
                cell_rows: dict[str, dict[str, Any]] = {}
                for session_id in session_ids:
                    material = materials[session_id]
                    carrier = carriers[session_id][budget]
                    anchor = anchors[session_id][budget]
                    row_spec = stage_p_replay.ROW_SPECS["P1"]
                    for seed in plan.SEEDS:
                        decoder = decoders[seed]
                        hp = hp_m4 if budget == plan.M4 else hp_m30
                        rollout = replay.rollout_session(
                            decoder, material, budget=budget, spec=spec,
                            anchor=anchor, carrier=carrier,
                            hp=hp if spec.online else None,
                            config=cdm_core.CDMDConfig(support_budget_m=int(budget)),
                            row_spec=row_spec if spec.online else None,
                        )
                        decoder.clear_cache()
                        r2 = float(variance_weighted_r2(rollout["targets"], rollout["prediction"]))
                        receipts = rollout["receipts"]
                        chain_ok = True
                        previous_ca: Optional[str] = None
                        for receipt in receipts:
                            if previous_ca is not None and receipt["cb"] != previous_ca:
                                chain_ok = False
                            previous_ca = receipt["ca"]
                        row = {
                            "session": session_id, "seed": seed, "cell": spec.name,
                            "budget": int(budget), "carrier_law": spec.carrier_law,
                            "window_count": int(rollout["prediction"].shape[0]),
                            "query_starts_sha256": paired_digest(material.query_starts),
                            "target_sha256": paired_digest(rollout["targets"]),
                            "prediction_sha256": paired_digest(rollout["prediction"]),
                            "side_sha256": paired_digest(carrier["side"]),
                            "initial_carrier_sha256": rollout["initial_carrier_sha256"],
                            "final_carrier_sha256": rollout["final_carrier_sha256"],
                            "identity_sha256": rollout["identity_sha256"],
                            "committed_rows": int(rollout["committed_rows"]),
                            "committed_label_counts": list(rollout["committed_label_counts"]),
                            "measurement_reasons": dict(rollout["measurement_reasons"]),
                            "gate_reasons": dict(rollout["gate_reasons"]),
                            "movements_count": len(rollout["movements"]),
                            "max_movement": (float(np.max(rollout["movements"])) if rollout["movements"] else 0.0),
                            "r2": r2,
                            "receipts": receipts,
                            "causality_chain_verified": bool(chain_ok),
                        }
                        causality_all = causality_all and row["causality_chain_verified"]
                        if spec.name == "F00s_anchor":
                            sealed = sealed_rows[f"{session_id}|seed{seed}"]
                            matches = {
                                "prediction_sha256": row["prediction_sha256"] == sealed["prediction_sha256"],
                                "target_sha256": row["target_sha256"] == sealed["target_sha256"],
                                "r2": row["r2"] == float(sealed["r2"]),
                                "window_count": row["window_count"] == int(sealed["query_window_count"]),
                                "side_sha256": (
                                    row["side_sha256"]
                                    == sealed["side_evidence"]["normalized_t4_sha256"]
                                ),
                                "selected_support": (
                                    carrier["selected"].tolist()
                                    == sealed["side_evidence"]["selected_first30_indices"]
                                ),
                            }
                            anchors_sealed[f"{session_id}|seed{seed}"] = {
                                "field_matches": matches,
                                "exact_match": all(bool(value) for value in matches.values()),
                            }
                            _require(
                                anchors_sealed[f"{session_id}|seed{seed}"]["exact_match"],
                                f"the F00s anchor failed against the sealed row {session_id}/s{seed}",
                            )
                        cell_rows.setdefault(session_id, {})[seed] = row
                        _require(time.monotonic() - started <= plan.HARD_TIMEOUT_SECONDS,
                                 "the SUA hard timeout fired during the governing grid")
                matrix[roster][budget][spec.name] = cell_rows

    for session_id in external_ids + list(plan.WITHIN_SESSIONS):
        for seed in plan.SEEDS:
            frozen = matrix[plan.EXTERNAL_ROSTER_KEY] if session_id in external_ids \
                else matrix[plan.WITHIN_ROSTER_KEY]
            f00 = frozen[plan.M30]["F00s"][session_id][seed]
            f01 = frozen[plan.M30]["F01s"][session_id][seed]
            equal = f00["prediction_sha256"] == f01["prediction_sha256"] and f00["r2"] == f01["r2"]
            zero_movement = all(receipt["cb"] == receipt["ca"] for receipt in f01["receipts"])
            noop_proofs[f"{session_id}|seed{seed}"] = {
                "bitwise_equal_predictions": bool(equal),
                "carrier_unchanged_every_trial": bool(zero_movement),
                "committed_rows": int(f01["committed_rows"]),
            }
            _require(equal and zero_movement, f"the M30 exact no-op law broke at {session_id}/s{seed}")

    values: dict[str, dict[int, dict[str, dict[str, Mapping[int, Any]]]]] = {
        roster: {
            budget: {
                cell: {
                    session: matrix[roster][budget][cell][session]
                    for session in matrix[roster][budget][cell]
                }
                for cell in matrix[roster][budget]
            }
            for budget in plan.BUDGETS
        }
        for roster in matrix
    }
    r2_view: dict[str, dict[int, dict[str, dict[str, dict[int, float]]]]] = {
        roster: {
            budget: {
                cell: {
                    session: {int(seed): float(row["r2"]) for seed, row in per_session.items()}
                    for session, per_session in values[roster][budget][cell].items()
                }
                for cell in values[roster][budget]
            }
            for budget in plan.BUDGETS
        }
        for roster in values
    }
    summaries = {
        roster: {
            budget: {
                cell: {
                    "equal_session_mean_after_seed_average": gates.equal_session_mean(
                        gates.seed_averaged_session_means(r2_view[roster][budget][cell])
                    ),
                    "per_session_r2": {
                        session: {str(seed): value for seed, value in per_seed.items()}
                        for session, per_seed in r2_view[roster][budget][cell].items()
                    },
                }
                for cell in r2_view[roster][budget]
            }
            for budget in plan.BUDGETS
        }
        for roster in r2_view
    }
    external_m4 = r2_view[plan.EXTERNAL_ROSTER_KEY][plan.M4]
    within_m4 = r2_view[plan.WITHIN_ROSTER_KEY][plan.M4]
    external_m30 = r2_view[plan.EXTERNAL_ROSTER_KEY][plan.M30]
    within_m30 = r2_view[plan.WITHIN_ROSTER_KEY][plan.M30]
    safety = {
        "f00s_bit_anchor_vs_sealed_paired_row": all(
            item["exact_match"] for item in anchors_sealed.values()
        ),
        "anchor_zero_evidence_parity_all_sessions": True,
        "m30_exact_noop": all(
            item["bitwise_equal_predictions"] and item["carrier_unchanged_every_trial"]
            for item in noop_proofs.values()
        ),
        "causality_receipt_chains": bool(causality_all),
        "hyperparameters_reselected_on_sua_source_folds": True,
        "dandi_hyperparameters_ported_as_claims": False,
        "zero_target_updates": True,
    }
    gate = gates.evaluate_sua_gate(
        external_m4=external_m4, within_m4=within_m4,
        external_m30=external_m30, within_m30=within_m30,
        safety=safety, gates_law=plan.GATES,
    )
    stops = gates.stop_conditions(gate)
    contrast = {
        "external_m4_f01s_minus_f00s": {
            session: {
                str(seed): float(
                    external_m4["F01s"][session][seed] - external_m4["F00s"][session][seed]
                ) for seed in plan.SEEDS
            } for session in external_m4["F00s"]
        },
    }
    replay_payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "f00s_f01s_inference_only_sua_transfer",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_digest,
        "authority": authority,
        "environment": {
            **plan.ENVIRONMENT_LAW,
            "gpu_name": torch.cuda.get_device_name(0),
            "batch_size": int(batch_size),
            "seeds": list(plan.SEEDS),
        },
        "surface": {
            "view": plan.VIEW,
            "runtime": "the sua_exploration V9 matched-scorer shared_t4 family",
            "external_roster": external_ids,
            "within_roster": list(plan.WITHIN_SESSIONS),
            "query_rule": "strictly_after_rewarded_trial_50__whole_windows_per_rewarded_trial",
            "external_query_windows": int(total_windows),
        },
        "cells": {
            "F00s_anchor": {
                "law": "the sealed paired-screen decode (whole-session batching); must reproduce the sealed ridge_m4_activity30 rows bit-exactly",
                "scope": "external-15 x M4 x 3 seeds",
            },
            "F00s": {"law": "frozen sealed M4/M30 ridge carrier, per-trial decode (the gate baseline)"},
            "F01s": {"law": "P1 online anchored block refit under the three-factor gate, per-trial decode"},
            "M10_not_run": plan.M10_DISCLOSURE,
        },
        "carrier_evidence": {
            f"{session}|m{budget}": {
                "selected": carriers[session][budget]["selected"].tolist(),
                "selected_sha256": carriers[session][budget]["selected_sha256"],
                "theta_sha256": carriers[session][budget]["theta_sha256"],
                "rates_sha256": carriers[session][budget]["rates_sha256"],
                "raw_t4_sha256": carriers[session][budget]["raw_t4_sha256"],
                "side_sha256": carriers[session][budget]["side_sha256"],
                "fit_evidence": carriers[session][budget]["fit_evidence"],
                "anchor_payload_digest": anchors[session][budget].digest,
                "anchor_parity": anchors[session][budget].support_coefficient_parity,
                "groups_digest": anchors[session][budget].groups.digest,
                "group_sizes": [
                    int(anchors[session][budget].groups.unit_indices(group).size)
                    for group in range(cdm_core.GROUP_COUNT)
                ],
                "invalid_units": int((~np.asarray(anchors[session][budget].groups.valid_mask)).sum()),
            }
            for session in materials for budget in plan.BUDGETS
        },
        "hyperparameter_selection": selection,
        "hyperparameters": {
            "m4": selected,
            "m30": {
                "rho_M": float(plan.M30_NOOP_LAW["rho_M"]),
                "alpha_M": float(plan.M30_NOOP_LAW["alpha_M"]),
                "c_M": None,
                "thresholds": selected["thresholds"],
                "inherits_thresholds_from": "the selected M4 vector",
            },
            "law": dict(plan.SELECTION_LAW),
        },
        "summaries": summaries,
        "matrix": matrix,
        "anchors": {
            "f00s_vs_sealed_paired_rows": anchors_sealed,
            "all_exact": all(item["exact_match"] for item in anchors_sealed.values()),
            "m30_noop_proofs": noop_proofs,
            "m30_noop_all_pass": all(
                item["bitwise_equal_predictions"] and item["carrier_unchanged_every_trial"]
                for item in noop_proofs.values()
            ),
        },
        "gate": gate,
        "stop_conditions": stops,
        "external_m4_contrast": contrast,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    _require(replay_payload["anchors"]["all_exact"],
             "the F00s anchor failed at least one sealed paired-screen row")
    _require(replay_payload["anchors"]["m30_noop_all_pass"], "the M30 exact no-op law failed")
    replay_digest = _publish(root / "replay.json", replay_payload)
    terminal_payload = {
        "schema": f"{plan.SCHEMA}_terminal_v1",
        "status": "TERMINAL",
        "scope": "part_b1_sua_inference_only_transfer",
        "replay_sha256": replay_digest,
        "attempt_sha256": attempt_digest,
        "gate": gate,
        "stop_conditions": stops,
        "equal_session_means": summaries,
        "hyperparameters": replay_payload["hyperparameters"],
        "anchors": replay_payload["anchors"],
        "external_m4_contrast": contrast,
        "m10_disclosure": plan.M10_DISCLOSURE,
        "external_roster_opened_by_terminal": False,
        "wall_seconds": replay_payload["wall_seconds"],
    }
    _publish(root / "terminal.json", terminal_payload)
    return {
        "status": terminal_payload["status"],
        "replay_sha256": replay_digest,
        "gate_summary": {
            "promoted": gate["promoted"],
            "external_m4_delta": gate["external_m4_delta"],
            "positive_external_m4_sessions": gate["positive_external_m4_sessions"],
            "safety_all_pass": gate["safety_all_pass"],
            "stop_conditions_fired": stops["fired"],
        },
    }
