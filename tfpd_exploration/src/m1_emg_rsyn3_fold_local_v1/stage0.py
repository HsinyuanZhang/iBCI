"""CPU Stage-0 driver for fold-local EMG-rSyn3. Sealed V1 roots stay read-only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3 as parent_syn3

from . import compare as fold_compare
from . import data as fold_data
from . import plan
from . import syn3 as fold_syn3


class Stage0Error(RuntimeError):
    """Fail closed for fold-local Stage-0 constructibility."""


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


def verify_sealed_roots(repo_root: Path) -> dict[str, object]:
    fail_root = Path(repo_root) / plan.PARENT_FAIL_ROOT_RELATIVE
    pass_root = Path(repo_root) / plan.RSYN3_PASS_ROOT_RELATIVE
    fail_decision = fail_root / "decision.json"
    fail_terminal = fail_root / "terminal.json"
    pass_decision = pass_root / "decision.json"
    pass_terminal = pass_root / "terminal.json"
    pass_folds = pass_root / "fold_receipts.json"
    pass_table = pass_root / "reliability_table.json"
    _require(fail_decision.is_file() and fail_terminal.is_file(), "parent FAIL missing")
    _require(pass_decision.is_file() and pass_terminal.is_file(), "rSyn3 PASS missing")
    fail_decision_sha = plan.sha256_bytes(fail_decision.read_bytes())
    fail_terminal_sha = plan.sha256_bytes(fail_terminal.read_bytes())
    pass_decision_sha = plan.sha256_bytes(pass_decision.read_bytes())
    pass_terminal_sha = plan.sha256_bytes(pass_terminal.read_bytes())
    pass_folds_sha = plan.sha256_bytes(pass_folds.read_bytes())
    pass_table_sha = plan.sha256_bytes(pass_table.read_bytes())
    _require(fail_decision_sha == plan.PARENT_FAIL_DECISION_SHA256, "parent FAIL decision sha drift")
    _require(fail_terminal_sha == plan.PARENT_FAIL_TERMINAL_SHA256, "parent FAIL terminal sha drift")
    _require(pass_decision_sha == plan.RSYN3_PASS_DECISION_SHA256, "rSyn3 PASS decision sha drift")
    _require(pass_terminal_sha == plan.RSYN3_PASS_TERMINAL_SHA256, "rSyn3 PASS terminal sha drift")
    _require(pass_folds_sha == plan.RSYN3_PASS_FOLD_RECEIPTS_SHA256, "rSyn3 PASS fold_receipts sha drift")
    _require(pass_table_sha == plan.RSYN3_PASS_RELIABILITY_SHA256, "rSyn3 PASS reliability sha drift")
    fail_payload = json.loads(fail_decision.read_text(encoding="utf-8"))
    pass_payload = json.loads(pass_decision.read_text(encoding="utf-8"))
    _require(fail_payload.get("decision") == "FAIL", "parent FAIL decision drift")
    _require(pass_payload.get("decision") == "PASS", "rSyn3 PASS decision drift")
    return {
        "parent_fail": {
            "root": plan.PARENT_FAIL_ROOT_RELATIVE,
            "decision": "FAIL",
            "decision_sha256": fail_decision_sha,
            "terminal_sha256": fail_terminal_sha,
            "rewritten": False,
        },
        "rsyn3_pass": {
            "root": plan.RSYN3_PASS_ROOT_RELATIVE,
            "decision": "PASS",
            "decision_sha256": pass_decision_sha,
            "terminal_sha256": pass_terminal_sha,
            "fold_receipts_sha256": pass_folds_sha,
            "reliability_table_sha256": pass_table_sha,
            "rewritten": False,
        },
        "sealed_fold_receipts": json.loads(pass_folds.read_text(encoding="utf-8")),
    }


def execute_stage0(repo_root: Path) -> dict[str, object]:
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    rectify.assert_frozen_law()
    sealed = verify_sealed_roots(repo_root)
    teacher = repo_root / plan.TEACHER_CKPT_RELATIVE
    _require(teacher.is_file(), "teacher checkpoint missing")
    teacher_sha = parent_data.file_sha256(teacher)
    _require(teacher_sha == plan.TEACHER_CKPT_SHA256, "teacher sha drift")

    stored_views: dict[str, object] = {}
    rectifier_mass: dict[str, object] = {}
    rectified_views: dict[str, object] = {}
    reliability_rows: list[dict[str, object]] = []
    fold_receipts: dict[str, object] = {}
    isolations: dict[str, object] = {}
    failures: list[str] = []

    for fold in sorted(plan.FOLD_TARGETS):
        loaded = fold_data.load_fold_scope(fold=fold)
        isolations[str(fold)] = loaded["isolation"]
        if bool(loaded["isolation"]["target_query_values_read"]):
            failures.append(f"fold{fold} target query values were read")
        sources = loaded["sources"]
        target = loaded["target"]
        _require(
            target.path_sha256 == plan.SOURCE_FILE_SHA256[target.session],
            f"target sha {target.session}",
        )
        for name, record in sources.items():
            _require(record.path_sha256 == plan.SOURCE_FILE_SHA256[name], f"source sha {name}")
            stored_views.setdefault(f"source:{fold}:{name}", record.signal_view)
            mass = rectify.mass_report(record.emg)
            rectifier_mass[f"source:{fold}:{name}"] = mass
            rectified = rectify.relu_nonnegative_projection(record.emg)
            rectified_views[f"source:{fold}:{name}"] = rectify.rectified_signal_view(
                record.signal_view, rectified, mass,
            )
        stored_views[f"target:{fold}:{target.session}"] = target.signal_view
        target_mass = rectify.mass_report(target.emg)
        rectifier_mass[f"target:{fold}:{target.session}"] = target_mass
        target_rect = rectify.relu_nonnegative_projection(target.emg)
        rectified_views[f"target:{fold}:{target.session}"] = rectify.rectified_signal_view(
            target.signal_view, target_rect, target_mass,
        )
        if float(np.mean(target_rect < 0.0)) > 0.0:
            failures.append(f"fold{fold} rectified target EMG remains signed")
            continue
        alignment_ok = bool(target.signal_view["recorded_time_alignment_ok"]) and all(
            bool(record.signal_view["recorded_time_alignment_ok"]) for record in sources.values()
        )
        if not alignment_ok:
            failures.append(f"fold{fold} recorded-time 20-ms alignment contract failed")
            continue

        source_emg = np.concatenate(
            [rectify.relu_nonnegative_projection(record.emg) for record in sources.values()],
            axis=0,
        )
        try:
            nmf = rsyn3.fit_source_nmf(source_emg)
            pca = rsyn3.fit_source_pca(source_emg)
        except parent_syn3.Syn3Error as error:
            failures.append(f"fold{fold} basis: {error}")
            continue
        probe = fold_syn3.dictionary_immutability_probe(nmf, target_rect)
        if probe["mutated"] or probe["source"] != "target_support_emg":
            failures.append(f"fold{fold} dictionary immutability failed")
            continue

        source_raw: list[np.ndarray] = []
        for record in sources.values():
            emg_m, rates_m, ids_m = _mask_budget(record, plan.SUPPORT_TRIALS)
            z_m = rsyn3.project_basis(emg_m, nmf)
            weights, intercepts = rsyn3.fit_all_units(z_m, rates_m)
            source_raw.append(rsyn3.carrier_from_encoding(weights, intercepts))
        norm_mean, norm_scale = rsyn3.source_normalizer(source_raw)

        fold_rows: list[dict[str, object]] = []
        for budget in plan.CARRIER_BUDGETS:
            emg_b, rates_b, ids_b = _mask_budget(target, budget)
            rsyn3.require_support_bins(ids_b, budget=budget)
            _require(int(np.max(ids_b)) < plan.SUPPORT_TRIALS, "target carrier used query trials")
            z_b = rsyn3.project_basis(emg_b, nmf)
            z_pca = rsyn3.project_basis(emg_b, pca)
            weights, intercepts = rsyn3.fit_all_units(z_b, rates_b)
            carrier = rsyn3.carrier_from_encoding(weights, intercepts)
            pca_w, pca_b = rsyn3.fit_all_units(z_pca, rates_b)
            pca_carrier = rsyn3.carrier_from_encoding(pca_w, pca_b)
            coverage = rsyn3.coverage_report(z_b, rates_b, trial_ids=ids_b, budget=budget)
            split = rsyn3.trial_stratified_split_half(z_b, rates_b, trial_ids=ids_b)
            pca3 = fold_syn3.rectified_trial_mean_pca3_reliability(emg_b, rates_b, ids_b)
            ls_scores, ls_rates = rsyn3.derange_trial_association(
                z_b, rates_b, ids_b, target.session, plan.SEED,
            )
            ls_w, ls_i = rsyn3.fit_all_units(ls_scores, ls_rates)
            controls = fold_syn3.zero4_never_fits_target(rsyn3.build_controls(
                raw_syn3=carrier,
                normalizer_mean=norm_mean,
                normalizer_scale=norm_scale,
                session_name=target.session,
                seed=plan.SEED,
                n_units=carrier.shape[0],
                target_fit_invoked=budget == plan.SUPPORT_TRIALS,
                raw_ls4=rsyn3.carrier_from_encoding(ls_w, ls_i),
                raw_pca3=pca_carrier,
            ))
            row = {
                "fold": fold,
                "target": target.session,
                "budget": budget,
                "coverage": coverage,
                "split_half_rsyn3": split,
                "split_half_rectified_trial_mean_pca3": pca3,
                "carrier_digest": rsyn3.array_digest(carrier),
                "coefficient_norm": float(np.linalg.norm(carrier)),
                "finite_carrier": bool(np.isfinite(carrier).all()),
                "zero4_is_zero": bool(np.array_equal(controls["Zero4"], np.zeros_like(controls["Zero4"]))),
                "rs4_differs": bool(not np.array_equal(controls["RS4"], controls["rSyn3"])),
                "ls4_computed_for_disclosure_only": True,
                "ls4_enabled_in_stage1_pilot": False,
                "nmf_reconstruction_digest": nmf.reconstruction_digest,
                "pca_reconstruction_digest": pca.reconstruction_digest,
                "rectifier": plan.RECTIFIER_LAW["name"],
            }
            fold_rows.append(row)
            reliability_rows.append(row)
        fold_receipts[str(fold)] = {
            "source_sessions": list(sources),
            "target": target.session,
            "rectifier": dict(plan.RECTIFIER_LAW),
            "ls4": dict(plan.LS4_LAW),
            "isolation": loaded["isolation"],
            "dictionary_immutability": probe,
            "nmf": {
                "kind": nmf.kind,
                "scale_digest": rsyn3.array_digest(nmf.scale),
                "dictionary_digest": rsyn3.array_digest(nmf.dictionary),
                "order": list(nmf.order),
                "library": dict(nmf.library),
                "n_iter": nmf.extra["n_iter"],
                "reconstruction_digest": nmf.reconstruction_digest,
            },
            "pca": {
                "kind": pca.kind,
                "dictionary_digest": rsyn3.array_digest(pca.dictionary),
                "reconstruction_digest": pca.reconstruction_digest,
                "diagnostic_label": fold_syn3.RECTIFIED_TRIAL_MEAN_PCA3_LABEL,
            },
            "normalizer_mean": norm_mean.tolist(),
            "normalizer_scale": norm_scale.tolist(),
            "rows": fold_rows,
        }

    jsonable_folds = _jsonable(fold_receipts)
    comparison = fold_compare.compare_all(jsonable_folds, sealed["sealed_fold_receipts"])
    constructible = (
        not failures
        and len(reliability_rows) == len(plan.FOLD_TARGETS) * len(plan.CARRIER_BUDGETS)
        and all(bool(row["finite_carrier"]) and bool(row["coverage"]["finite_rates"])
                for row in reliability_rows)
        and all(row["coverage"]["rejected_for_trial_count"] is False for row in reliability_rows)
        and all(not bool(item["target_query_values_read"]) for item in isolations.values())
    )
    decision = "PASS" if constructible else "FAIL"
    return _jsonable({
        "schema": "m1_emg_rsyn3_fold_local_stage0_body_v1",
        "decision": decision,
        "failures": failures,
        "parent_fail": sealed["parent_fail"],
        "rsyn3_pass": sealed["rsyn3_pass"],
        "rectifier": dict(plan.RECTIFIER_LAW),
        "ls4": dict(plan.LS4_LAW),
        "stored_signal_views": stored_views,
        "rectified_signal_views": rectified_views,
        "rectifier_mass": rectifier_mass,
        "reliability_table": reliability_rows,
        "fold_receipts": fold_receipts,
        "fold_isolations": isolations,
        "sealed_pass_comparison": comparison,
        "checkpoint_authority": {
            "relative": plan.TEACHER_CKPT_RELATIVE,
            "sha256": teacher_sha,
            "strict_load_deferred_to_gpu_capability": True,
            "torch_loaded": False,
        },
        "query_isolation": {
            "support_trials": [0, plan.SUPPORT_TRIALS],
            "report_query": [plan.QUERY_START, plan.QUERY_STOP_EXCLUSIVE],
            "target_query_values_read": False,
            "fold_local_source_emg": "full_session",
            "fold_local_target_emg_and_neural": [0, plan.SUPPORT_TRIALS],
        },
        "decoder_r2_computed": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
    })


def publish_stage0(repo_root: Path) -> tuple[dict[str, str], str | None, str | None]:
    from . import receipts as fold_receipts

    repo_root = Path(repo_root)
    verify_sealed_roots(repo_root)
    parent = repo_root / plan.RESULT_ROOT_RELATIVE
    fold_receipts.refuse_sealed_roots(parent)
    parent.mkdir(parents=True, exist_ok=True)

    body_holder: dict[str, object] = {}

    def _launch() -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_fold_local_stage0_launch_v1",
            "device": "cpu",
            "cuda_visible_devices": "",
            "workorder_sha256": plan.WORKORDER_SHA256,
            "design_sha256": plan.DESIGN_SHA256,
            "ls4": dict(plan.LS4_LAW),
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
                    "fold_isolations": body_holder["fold_isolations"],
                },
            ),
            "reliability_table.json": artifact.publish_json(
                "reliability_table.json", {"rows": body_holder["reliability_table"]},
            ),
            "fold_receipts.json": artifact.publish_json("fold_receipts.json", body_holder["fold_receipts"]),
            "sealed_pass_comparison.json": artifact.publish_json(
                "sealed_pass_comparison.json", body_holder["sealed_pass_comparison"],
            ),
            "checkpoint_authority.json": artifact.publish_json(
                "checkpoint_authority.json", body_holder["checkpoint_authority"],
            ),
            "decision.json": artifact.publish_json(
                "decision.json",
                {
                    "decision": body_holder["decision"],
                    "failures": body_holder["failures"],
                    "parent_fail": body_holder["parent_fail"],
                    "rsyn3_pass": body_holder["rsyn3_pass"],
                    "rectifier": body_holder["rectifier"],
                    "ls4": body_holder["ls4"],
                    "query_isolation": body_holder["query_isolation"],
                    "sealed_pass_comparison_all_equal": body_holder["sealed_pass_comparison"]["all_equal"],
                    "decoder_r2_computed": False,
                },
            ),
        }

    def _terminal(shas: dict[str, str]) -> dict[str, object]:
        return {
            "schema": "m1_emg_rsyn3_fold_local_stage0_terminal_v1",
            "status": "COMPLETE_STAGE0",
            "decision": body_holder.get("decision"),
            "body_sha256": shas,
            "gpu_capability_issued": False,
            "parent_fail_rewritten": False,
            "rsyn3_pass_rewritten": False,
            "ls4_enabled_in_stage1_pilot": False,
        }

    return fold_receipts.run_stage0(
        parent,
        attempt_payload={
            "schema": "m1_emg_rsyn3_fold_local_stage0_attempt_v1",
            "status": "ATTEMPT_RESERVED",
            "phase": plan.PHASE,
            "data_or_model_accessed": False,
            "workorder_sha256": plan.WORKORDER_SHA256,
            "parent_fail_decision_sha256": plan.PARENT_FAIL_DECISION_SHA256,
            "rsyn3_pass_decision_sha256": plan.RSYN3_PASS_DECISION_SHA256,
        },
        launch_builder=_launch,
        body_publisher=_body,
        terminal_builder=_terminal,
        relative="stage0",
    )
