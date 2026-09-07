"""CPU Stage-0 driver for EMG-rSyn3. Does not rewrite the parent FAIL root."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3

from . import plan
from . import rectify
from . import syn3


class Stage0Error(RuntimeError):
    """Fail closed for Stage-0 constructibility."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage0Error(message)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _mask_budget(record: parent_data.SessionBins, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = record.rate_trial_ids < int(budget)
    _require(np.any(mask), f"no bins in budget M={budget}")
    return record.emg[record.emg_trial_ids < int(budget)], record.rates[mask], record.rate_trial_ids[mask]


def verify_parent_fail_sealed(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root) / plan.PARENT_STAGE0_ROOT_RELATIVE
    decision = root / "decision.json"
    signal = root / "signal_view.json"
    terminal = root / "terminal.json"
    _require(decision.is_file() and signal.is_file() and terminal.is_file(), "parent FAIL missing")
    decision_sha = plan.sha256_bytes(decision.read_bytes())
    signal_sha = plan.sha256_bytes(signal.read_bytes())
    terminal_sha = plan.sha256_bytes(terminal.read_bytes())
    _require(decision_sha == plan.PARENT_DECISION_SHA256, "parent decision sha drift")
    _require(signal_sha == plan.PARENT_SIGNAL_VIEW_SHA256, "parent signal_view sha drift")
    _require(terminal_sha == plan.PARENT_TERMINAL_SHA256, "parent terminal sha drift")
    payload = json.loads(decision.read_text(encoding="utf-8"))
    _require(payload.get("decision") == "FAIL", "parent FAIL decision drift")
    failures = payload.get("failures") or []
    _require(any("frozen rectifier" in str(item) for item in failures), "parent FAIL text drift")
    return {
        "root": plan.PARENT_STAGE0_ROOT_RELATIVE,
        "decision": "FAIL",
        "decision_sha256": decision_sha,
        "signal_view_sha256": signal_sha,
        "terminal_sha256": terminal_sha,
        "rewritten": False,
    }


def _t0c1_has_no_carrier_interface(root: Path) -> dict[str, object]:
    phase3 = root / "tfpd_exploration/src/m1_t0c1_prefix_v1/phase3.py"
    trainer = root / "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/phase3.py"
    texts = []
    for path in (phase3, trainer):
        _require(path.is_file(), f"sealed T0/C1 source missing for inspection: {path}")
        texts.append(path.read_text(encoding="utf-8"))
    return {
        "inspected": [str(phase3.relative_to(root)), str(trainer.relative_to(root))],
        "side_features_token_present": any("side_features" in text for text in texts),
        "result_roots_opened": False,
        "usable_as_matched_zero_carrier_control": False,
    }


def execute_stage0(repo_root: Path) -> dict[str, object]:
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    rectify.assert_frozen_law()
    parent_fail = verify_parent_fail_sealed(repo_root)
    teacher = repo_root / plan.TEACHER_CKPT_RELATIVE
    _require(teacher.is_file(), "teacher checkpoint missing")
    teacher_sha = parent_data.file_sha256(teacher)
    _require(teacher_sha == plan.TEACHER_CKPT_SHA256, "teacher sha drift")

    paths = parent_data.allowlisted_paths()
    sessions: dict[str, parent_data.SessionBins] = {}
    for session, path in paths.items():
        sessions[session] = parent_data.load_support_bins(
            path, emg_trial_stop=None, neural_trial_stop=plan.SUPPORT_TRIALS,
        )
        _require(sessions[session].path_sha256 == plan.SOURCE_FILE_SHA256[session], "session sha")

    stored_views = {name: record.signal_view for name, record in sessions.items()}
    rectifier_mass = {name: rectify.mass_report(record.emg) for name, record in sessions.items()}
    rectified_emg = {
        name: rectify.relu_nonnegative_projection(record.emg) for name, record in sessions.items()
    }
    rectified_views = {
        name: rectify.rectified_signal_view(stored_views[name], rectified_emg[name], rectifier_mass[name])
        for name in sessions
    }
    alignment_ok = all(bool(view["recorded_time_alignment_ok"]) for view in stored_views.values())
    rectified_signed = any(float(view["negative_fraction"]) > 0.0 for view in rectified_views.values())

    reliability_rows: list[dict[str, object]] = []
    fold_receipts: dict[str, object] = {}
    failures: list[str] = []
    if not alignment_ok:
        failures.append("recorded-time 20-ms alignment contract failed")
    if rectified_signed:
        failures.append("rectified EMG remains signed after relu_nonnegative_projection")

    if not failures:
        for fold in sorted(plan.FOLD_TARGETS):
            target_name = plan.FOLD_TARGETS[fold]
            source_names = plan.source_sessions_for_fold(fold)
            source_emg = np.concatenate([rectified_emg[name] for name in source_names], axis=0)
            try:
                nmf = syn3.fit_source_nmf(source_emg)
                pca = syn3.fit_source_pca(source_emg)
            except parent_syn3.Syn3Error as error:
                failures.append(f"fold{fold} basis: {error}")
                continue
            target_probe = rectified_emg[source_names[0]][:8]
            before = nmf.dictionary.copy()
            syn3.nnls_activations(syn3.apply_scale(target_probe, nmf.scale), nmf.dictionary)
            _require(np.array_equal(before, nmf.dictionary), "target mutated dictionary")

            source_raw: list[np.ndarray] = []
            for name in source_names:
                emg_m, rates_m, ids_m = _mask_budget(sessions[name], plan.SUPPORT_TRIALS)
                z_m = syn3.project_basis(emg_m, nmf)
                weights, intercepts = syn3.fit_all_units(z_m, rates_m)
                source_raw.append(syn3.carrier_from_encoding(weights, intercepts))
            norm_mean, norm_scale = syn3.source_normalizer(source_raw)

            fold_rows: list[dict[str, object]] = []
            for budget in plan.CARRIER_BUDGETS:
                emg_b, rates_b, ids_b = _mask_budget(sessions[target_name], budget)
                syn3.require_support_bins(ids_b, budget=budget)
                z_b = syn3.project_basis(emg_b, nmf)
                z_pca = syn3.project_basis(emg_b, pca)
                weights, intercepts = syn3.fit_all_units(z_b, rates_b)
                carrier = syn3.carrier_from_encoding(weights, intercepts)
                pca_w, pca_b = syn3.fit_all_units(z_pca, rates_b)
                pca_carrier = syn3.carrier_from_encoding(pca_w, pca_b)
                coverage = syn3.coverage_report(z_b, rates_b, trial_ids=ids_b, budget=budget)
                split = syn3.trial_stratified_split_half(z_b, rates_b, trial_ids=ids_b)
                trial_ids_unique = np.unique(ids_b)
                emg_means = np.stack([emg_b[ids_b == trial].mean(axis=0) for trial in trial_ids_unique])
                rate_means = np.stack([rates_b[ids_b == trial].mean(axis=0) for trial in trial_ids_unique])
                historical = syn3.trial_mean_pca_reliability(emg_means, rate_means)
                ls_scores, ls_rates = syn3.derange_trial_association(
                    z_b, rates_b, ids_b, target_name, plan.SEED,
                )
                ls_w, ls_i = syn3.fit_all_units(ls_scores, ls_rates)
                controls = syn3.build_controls(
                    raw_syn3=carrier,
                    normalizer_mean=norm_mean,
                    normalizer_scale=norm_scale,
                    session_name=target_name,
                    seed=plan.SEED,
                    n_units=carrier.shape[0],
                    target_fit_invoked=budget == plan.SUPPORT_TRIALS,
                    raw_ls4=syn3.carrier_from_encoding(ls_w, ls_i),
                    raw_pca3=pca_carrier,
                )
                row = {
                    "fold": fold,
                    "target": target_name,
                    "budget": budget,
                    "coverage": coverage,
                    "split_half_rsyn3": split,
                    "split_half_trial_mean_pca": historical,
                    "carrier_digest": syn3.array_digest(carrier),
                    "coefficient_norm": float(np.linalg.norm(carrier)),
                    "finite_carrier": bool(np.isfinite(carrier).all()),
                    "zero4_is_zero": bool(np.array_equal(controls["Zero4"], np.zeros_like(controls["Zero4"]))),
                    "rs4_differs": bool(not np.array_equal(controls["RS4"], controls["rSyn3"])),
                    "nmf_reconstruction_digest": nmf.reconstruction_digest,
                    "pca_reconstruction_digest": pca.reconstruction_digest,
                    "rectifier": plan.RECTIFIER_LAW["name"],
                }
                fold_rows.append(row)
                reliability_rows.append(row)
            fold_receipts[str(fold)] = {
                "source_sessions": list(source_names),
                "target": target_name,
                "rectifier": dict(plan.RECTIFIER_LAW),
                "nmf": {
                    "kind": nmf.kind,
                    "scale_digest": syn3.array_digest(nmf.scale),
                    "dictionary_digest": syn3.array_digest(nmf.dictionary),
                    "order": list(nmf.order),
                    "library": dict(nmf.library),
                    "n_iter": nmf.extra["n_iter"],
                    "reconstruction_digest": nmf.reconstruction_digest,
                },
                "pca": {
                    "kind": pca.kind,
                    "dictionary_digest": syn3.array_digest(pca.dictionary),
                    "reconstruction_digest": pca.reconstruction_digest,
                },
                "normalizer_mean": norm_mean.tolist(),
                "normalizer_scale": norm_scale.tolist(),
                "rows": fold_rows,
            }

    constructible = (
        not failures
        and len(reliability_rows) == len(plan.FOLD_TARGETS) * len(plan.CARRIER_BUDGETS)
        and all(bool(row["finite_carrier"]) and bool(row["coverage"]["finite_rates"])
                for row in reliability_rows)
        and all(row["coverage"]["rejected_for_trial_count"] is False for row in reliability_rows)
    )
    decision = "PASS" if constructible else "FAIL"
    return _jsonable({
        "schema": "m1_emg_rsyn3_stage0_body_v1",
        "decision": decision,
        "failures": failures,
        "parent_fail": parent_fail,
        "rectifier": dict(plan.RECTIFIER_LAW),
        "stored_signal_views": stored_views,
        "rectified_signal_views": rectified_views,
        "rectifier_mass": rectifier_mass,
        "reliability_table": reliability_rows,
        "fold_receipts": fold_receipts,
        "checkpoint_authority": {
            "relative": plan.TEACHER_CKPT_RELATIVE,
            "sha256": teacher_sha,
            "strict_load_deferred_to_gpu_capability": True,
            "torch_loaded": False,
        },
        "t0c1_carrier_interface": _t0c1_has_no_carrier_interface(repo_root),
        "query_isolation": {
            "support_trials": [0, plan.SUPPORT_TRIALS],
            "report_query": [plan.QUERY_START, plan.QUERY_STOP_EXCLUSIVE],
            "query_values_read": False,
        },
        "lag_bins": plan.LAG_BINS,
        "target_optimizer_backward_update": 0,
        "cuda_initialized": False,
        "decoder_r2_computed": False,
    })


def publish_stage0(repo_root: Path) -> tuple[dict[str, str], str | None, str | None]:
    from . import receipts as rsyn3_receipts

    repo_root = Path(repo_root)
    parent = repo_root / "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1"
    rsyn3_receipts.refuse_parent_root(parent)
    parent.mkdir(parents=True, exist_ok=True)

    body_holder: dict[str, object] = {}

    def _launch() -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_stage0_launch_v1",
            "device": "cpu",
            "cuda_visible_devices": "",
            "workorder_sha256": plan.WORKORDER_SHA256,
            "design_sha256": plan.DESIGN_SHA256,
            "parent_decision_sha256": plan.PARENT_DECISION_SHA256,
            "rectifier": dict(plan.RECTIFIER_LAW),
        }

    def _body(artifact) -> dict[str, str]:
        body_holder.update(execute_stage0(repo_root))
        return {
            "signal_view.json": artifact.publish_json(
                "signal_view.json",
                {
                    "stored": body_holder["stored_signal_views"],
                    "rectified": body_holder["rectified_signal_views"],
                    "rectifier_mass": body_holder["rectifier_mass"],
                    "rectifier": body_holder["rectifier"],
                },
            ),
            "reliability_table.json": artifact.publish_json(
                "reliability_table.json", {"rows": body_holder["reliability_table"]},
            ),
            "fold_receipts.json": artifact.publish_json("fold_receipts.json", body_holder["fold_receipts"]),
            "checkpoint_authority.json": artifact.publish_json(
                "checkpoint_authority.json", body_holder["checkpoint_authority"],
            ),
            "decision.json": artifact.publish_json(
                "decision.json",
                {
                    "decision": body_holder["decision"],
                    "failures": body_holder["failures"],
                    "parent_fail": body_holder["parent_fail"],
                    "rectifier": body_holder["rectifier"],
                    "t0c1_carrier_interface": body_holder["t0c1_carrier_interface"],
                    "query_isolation": body_holder["query_isolation"],
                    "decoder_r2_computed": False,
                },
            ),
        }

    def _terminal(shas: dict[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_stage0_terminal_v1",
            "status": "COMPLETE_STAGE0",
            "decision": body_holder.get("decision"),
            "body_sha256": shas,
            "gpu_capability_issued": False,
            "parent_fail_rewritten": False,
        }

    return rsyn3_receipts.run_stage0(
        parent,
        attempt_payload={
            "schema": "m1_emg_rsyn3_stage0_attempt_v1",
            "status": "ATTEMPT_RESERVED",
            "phase": plan.PHASE,
            "data_or_model_accessed": False,
            "workorder_sha256": plan.WORKORDER_SHA256,
            "parent_decision_sha256": plan.PARENT_DECISION_SHA256,
        },
        launch_builder=_launch,
        body_publisher=_body,
        terminal_builder=_terminal,
        relative="stage0",
    )
