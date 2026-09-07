"""E0 inventory: actual paths, hashes, shapes, lineage, and selection scope."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import plan
from . import receipts


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(repo_root: Path, relative: str, sha256: str, role: str, *, epochs_present: list[str]) -> dict[str, object]:
    record = receipts.file_record(repo_root, relative, role=role, verify=sha256)
    record["available_per_epoch_checkpoints"] = epochs_present
    record["all_12_epochs_saved"] = len(epochs_present) >= 12
    return record


def build(repo_root: Path) -> dict[str, object]:
    repo_root = Path(repo_root)
    documents = []
    for item in plan.BOUND_DOCUMENTS:
        documents.append(receipts.file_record(repo_root, str(item["relative"]), role=str(item["role"]), verify=str(item["sha256"])))
    operators = [
        receipts.file_record(repo_root, plan.H1_OPERATOR_RELATIVE, role="h1_estimator", verify=plan.H1_OPERATOR_SHA256),
        receipts.file_record(repo_root, plan.M1_SYN3_OPERATOR_RELATIVE, role="m1_ridge_estimator", verify=plan.M1_SYN3_OPERATOR_SHA256),
        receipts.file_record(repo_root, plan.M1_CARRIER_BANK_RELATIVE, role="m1_source_frozen_bank", verify=plan.M1_CARRIER_BANK_SHA256),
    ]
    receipts_bound = [
        receipts.file_record(repo_root, plan.H1_RAW_RECEIPT_RELATIVE, role="h1_raw_m4_lodo", verify=plan.H1_RAW_RECEIPT_SHA256),
        receipts.file_record(repo_root, plan.H1_EB_RECEIPT_RELATIVE, role="h1_eb_lodo", verify=plan.H1_EB_RECEIPT_SHA256),
        receipts.file_record(
            repo_root,
            plan.H1_FROZEN_PLAN_MANIFEST_RELATIVE,
            role="h1_fold0_frozen_transform_manifest",
            verify=plan.H1_FROZEN_PLAN_MANIFEST_SHA256,
        ),
        receipts.file_record(
            repo_root, plan.TOKEN_PROBE_SUMMARY_RELATIVE, role="m1_token_probe_v1", verify=plan.TOKEN_PROBE_SUMMARY_SHA256
        ),
        receipts.file_record(
            repo_root, plan.BUDGET_PROBE_SUMMARY_RELATIVE, role="m1_budget_probe_v1", verify=plan.BUDGET_PROBE_SUMMARY_SHA256
        ),
        receipts.file_record(repo_root, plan.FOLD_LOCAL_RELIABILITY_RELATIVE, role="m1_fold_local_stage0_reliability"),
    ]

    h1_plan = _json(repo_root / plan.H1_FROZEN_PLAN_MANIFEST_RELATIVE)
    token = _json(repo_root / plan.TOKEN_PROBE_SUMMARY_RELATIVE)
    budget = _json(repo_root / plan.BUDGET_PROBE_SUMMARY_RELATIVE)

    m1_nwb = []
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as syn3_plan

    for session, relative in syn3_plan.SOURCE_RELATIVE.items():
        record = receipts.file_record(repo_root, relative, role="m1_heldin_nwb", verify=syn3_plan.SOURCE_FILE_SHA256[session])
        record["session"] = session
        record["cheap_shape_from_token_probe"] = token.get("support_receipt", {}).get(session)
        m1_nwb.append(record)

    h1_nwb = []
    data_dir = repo_root / plan.H1_DATA_DIR_RELATIVE
    h1_index_error = None
    try:
        import sys

        plan.reject_forbidden_path(data_dir)
        project = str(repo_root / "SPINT-main")
        if project not in sys.path:
            sys.path.insert(0, project)
        from src.data.h1_m4_eb_pilot import index_heldin_calib

        for name, path in index_heldin_calib(data_dir).items():
            relative = str(path.relative_to(repo_root))
            record = receipts.file_record(repo_root, relative, role="h1_heldin_nwb")
            record["session"] = name
            record["cheap_expected_shape"] = {"neural": ["T", plan.H1_CHANNELS], "velocity": ["T", plan.H1_VELOCITY_DIM]}
            h1_nwb.append(record)
    except Exception as error:  # noqa: BLE001
        h1_index_error = f"{type(error).__name__}: {error}"

    z_fix_epochs = ["epoch_011.pt"]
    s_fix_epochs = ["epoch_011.pt"]
    s_acyc_epochs = ["epoch_011.pt"]
    checkpoints = {
        "m1_fold0_z_fix_epoch011": _checkpoint(
            repo_root, plan.Z_FIX_EPOCH011_RELATIVE, plan.Z_FIX_EPOCH011_SHA256,
            "activity_only_fold0_consumer_last_epoch", epochs_present=z_fix_epochs,
        ),
        "m1_fold0_s_fix_epoch011": _checkpoint(
            repo_root, plan.S_FIX_EPOCH011_RELATIVE, plan.S_FIX_EPOCH011_SHA256,
            "carrier_aware_fold0_consumer_last_epoch", epochs_present=s_fix_epochs,
        ),
        "m1_fold0_s_acyc_epoch011": _checkpoint(
            repo_root, plan.S_ACYC_EPOCH011_RELATIVE, plan.S_ACYC_EPOCH011_SHA256,
            "carrier_aware_activitycycle_fold0_consumer_last_epoch", epochs_present=s_acyc_epochs,
        ),
        "m1_fold0_source_teacher_epoch018": _checkpoint(
            repo_root, plan.FOLD0_SOURCE_TEACHER_RELATIVE, plan.FOLD0_SOURCE_TEACHER_SHA256,
            "fold0_named_source_teacher", epochs_present=["epoch_018.ckpt"],
        ),
        "m1_allsource_teacher_epoch019": _checkpoint(
            repo_root, plan.ALLSOURCE_TEACHER_RELATIVE, plan.ALLSOURCE_TEACHER_SHA256,
            "allsource_teacher_cannot_support_clean_loso", epochs_present=["epoch_019.ckpt"],
        ),
        "h1_hc_h32_fold0_epoch049": _checkpoint(
            repo_root, plan.H1_HC_CHECKPOINT_RELATIVE, plan.H1_HC_CHECKPOINT_SHA256,
            "h1_carrierid_compact_consumer_context_only", epochs_present=["epoch_049.ckpt"],
        ),
    }

    return {
        "schema": plan.SCHEMA_INVENTORY,
        "phase": plan.PHASE,
        "contract_version": plan.CONTRACT_VERSION,
        "workorder_sha256": plan.WORKORDER_SHA256,
        "result_root": plan.STAGE0_ROOT_RELATIVE,
        "gpu_work_started": False,
        "hidden_or_evalai_opened": False,
        "documents_used": documents,
        "operators_used": operators,
        "receipts_reused": receipts_bound,
        "datasets": {
            "M1": {
                "sessions": list(plan.M1_SESSIONS),
                "fold0_target": plan.M1_FOLD0_TARGET,
                "fold0_sources": list(plan.M1_FOLD0_SOURCES),
                "estimator_direction": plan.M1_ESTIMATOR_DIRECTION,
                "label_type": plan.M1_LABEL_TYPE,
                "output_dim": plan.M1_EMG_DIM,
                "native_scoring_space": "signed_emg_16",
                "source_only_scale": True,
                "rectifier": plan.M1_RECTIFIER,
                "nnmf_provenance": "sklearn.decomposition.NMF nndsvda seed42 on source-rectified EMG",
                "rectifier_does_not_change_decoder_target": True,
                "calibration": {
                    "law": "chronological_first_10_legal_trials",
                    "support_trials": plan.M1_SUPPORT_TRIALS,
                    "query_must_be_entirely_post_support": True,
                    "window": plan.M1_WINDOW,
                    "aligned_target_view": "last_bin_of_window_100_unless_later_spec",
                    "do_not_use_m4_last6_proxy": True,
                },
                "normalization_authority": "source_pooled_mean_std_of_raw_[w1,w2,w3,b] on fold sources",
                "ridge_objective": plan.inherited_ridge_objective(),
                "nwb": m1_nwb,
                "token_probe_v1": {
                    "relative": plan.TOKEN_PROBE_SUMMARY_RELATIVE,
                    "sha256": plan.TOKEN_PROBE_SUMMARY_SHA256,
                    "verdict": token.get("verdict"),
                    "zfix_weights_r2": token.get("pooled_table", {}).get("Z-Fix", {}).get("weights_r2"),
                    "reused_not_repeated": True,
                },
                "budget_probe_v1": {
                    "relative": plan.BUDGET_PROBE_SUMMARY_RELATIVE,
                    "sha256": plan.BUDGET_PROBE_SUMMARY_SHA256,
                    "verdict": budget.get("verdict"),
                    "reused_not_repeated": True,
                },
                "training_lineage": {
                    "fold0_pilot_consumer": "m1_emg_rsyn3_fold_local_v1/pilot_r3",
                    "source_selection_scope": "fold0 sources 20120926/27/28; target 20120924 support-only for carrier",
                    "per_epoch_checkpoints": "only epoch_011.pt present per arm; not all 12 epochs",
                    "teacher_fold0_source": plan.FOLD0_SOURCE_TEACHER_RELATIVE,
                    "teacher_allsource": plan.ALLSOURCE_TEACHER_RELATIVE,
                    "allsource_cannot_support_clean_loso": True,
                    "teacher_pretraining_config_inspected": False,
                    "teacher_pretraining_scope": "UNRESOLVED: named fold0 source decoder vs all-source teacher; config yaml not uniquely bound in this inventory",
                },
                "source_selection_scope": {
                    "clean_fold_local_decoder_r2": "fold0 target 20120924 only; other folds have reliability/coverage, not a trained fold-local consumer score",
                    "do_not_fill_missing_fold_cells_with_allsource": True,
                    "visible_product_pick": "not declared for this workstream",
                },
            },
            "H1": {
                "estimator_direction": plan.H1_ESTIMATOR_DIRECTION,
                "label_type": plan.H1_LABEL_TYPE,
                "m3_deployment": "first 3 chronological eval-valid TrialNum; no fourth-trial padding",
                "m4_historical": "first 4 TrialNum; audit separately from M3",
                "block_law": "100-ms sum/BLOCK_SECONDS rates, mean velocity",
                "normalization_authority": "source-date-LODO mean/scale/pcs/U/mu/tau2 frozen on fold0 sources",
                "ridge_objective": plan.h1_ridge_objective(),
                "frozen_plan_shapes": h1_plan.get("array_shape"),
                "frozen_plan_hashes": h1_plan.get("array_sha256"),
                "q": h1_plan.get("q"),
                "lambda": h1_plan.get("lambda"),
                "source_sessions": h1_plan.get("source_sessions"),
                "nwb": h1_nwb,
                "nwb_index_error": h1_index_error,
                "training_lineage": {
                    "h_c_compact_consumer": plan.H1_HC_CHECKPOINT_RELATIVE,
                    "not_a_p_parent": True,
                },
                "source_selection_scope": "public held-in calibration records; no hidden/calibration export side effects",
            },
            "M2": {
                "estimator_direction": "forward_reach_conditioned_affine",
                "descriptor": "[a,c,sqrt(a^2+c^2),b] with intercept",
                "caveat": "reach-conditioned intercept is not an independent hold response; bind MOVE-window estimator, do not replace with historical whole-trial T4",
                "assay_this_turn": "inventory_only",
            },
            "DANDI688": {
                "estimator_direction": "forward_canonical_direction_fit",
                "descriptor": "[a,c,m,b]",
                "caveat": "same family as M2 does not imply the same estimator",
                "assay_this_turn": "inventory_only",
                "do_not_open_external15": True,
            },
        },
        "checkpoints": checkpoints,
        "p_unresolved": {
            "parent_bytes": "UNRESOLVED",
            "f_eta_shape": "UNRESOLVED",
            "differentiable_ridge": "UNRESOLVED",
        },
    }
