"""E3: coverage and argument audit. Reuse receipts; do not invent n_eff from bins."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import contracts
from . import plan


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(repo_root: Path, *, prior: Mapping[str, Any]) -> contracts.AssayStatus:
    repo_root = Path(repo_root)
    reliability_path = repo_root / plan.FOLD_LOCAL_RELIABILITY_RELATIVE
    budget_path = repo_root / plan.BUDGET_PROBE_SUMMARY_RELATIVE
    token_path = repo_root / plan.TOKEN_PROBE_SUMMARY_RELATIVE
    h1_raw_path = repo_root / plan.H1_RAW_RECEIPT_RELATIVE
    missing = [str(path) for path in (reliability_path, budget_path, token_path, h1_raw_path) if not path.is_file()]
    if missing:
        return contracts.AssayStatus(
            name="E3",
            status="BLOCKED",
            blocker="missing coverage receipts: " + ", ".join(missing),
        )

    reliability = _load_json(reliability_path)
    budget = _load_json(budget_path)
    token = _load_json(token_path)
    h1_raw = _load_json(h1_raw_path)

    m1_m10 = []
    for row in reliability.get("rows", []):
        if int(row.get("budget", -1)) != 10:
            continue
        coverage = dict(row["coverage"])
        m1_m10.append(
            {
                "fold": row["fold"],
                "target": row["target"],
                "n_trials": coverage["n_trials"],
                "valid_bins": coverage["valid_bins"],
                "valid_seconds": float(coverage["valid_seconds"]),
                "design_rank": coverage["design_rank"],
                "design_condition": float(coverage["design_condition"]),
                "normalized_gram_eigenvalues": [float(value) for value in coverage["normalized_gram_eigenvalues"]],
                "smallest_eigenvalue": float(coverage["smallest_eigenvalue"]),
                "per_synergy_dispersion": [float(value) for value in coverage["per_synergy_dispersion"]],
                "trial_stratified_occupancy": coverage["trial_stratified_occupancy"],
                "rsyn3_split_half_w": float(row["split_half_rsyn3"]["weight_flattened_pearson"]),
                "rsyn3_split_half_intercept": float(row["split_half_rsyn3"]["intercept_pearson"]),
            }
        )

    h1_support = []
    for date_row in h1_raw.get("date_lodo", []):
        for record in date_row.get("records", []):
            support = list(record.get("support_trial_numbers", []))
            h1_support.append(
                {
                    "date": date_row["date"],
                    "session_name": record["session_name"],
                    "m4_support_trial_ids": support,
                    "first3_trial_ids": support[:3],
                    "support_100ms_blocks": record.get("support_100ms_blocks"),
                    "support_exposure_seconds": record.get("support_exposure_seconds"),
                    "query_100ms_blocks": record.get("query_100ms_blocks"),
                    "query_exposure_seconds": record.get("query_exposure_seconds"),
                    "note": "M4 historical support; E1 uses first3 of chronological eval-valid TrialNum",
                }
            )

    e1 = prior.get("E1") if isinstance(prior.get("E1"), Mapping) else {}
    e2 = prior.get("E2") if isinstance(prior.get("E2"), Mapping) else {}
    payload = {
        "schema": plan.SCHEMA_E3,
        "reused_receipts": {
            "token_probe_v1": {
                "relative": plan.TOKEN_PROBE_SUMMARY_RELATIVE,
                "sha256": plan.TOKEN_PROBE_SUMMARY_SHA256,
                "verdict": token.get("verdict"),
                "zfix_weights_r2": token.get("pooled_table", {}).get("Z-Fix", {}).get("weights_r2"),
                "note": "linear token redundancy; not a proof that nonlinear useful information is absent",
            },
            "budget_probe_v1": {
                "relative": plan.BUDGET_PROBE_SUMMARY_RELATIVE,
                "sha256": plan.BUDGET_PROBE_SUMMARY_SHA256,
                "verdict": budget.get("verdict"),
                "reliability_floor": 0.30,
            },
            "fold_local_reliability": {"relative": plan.FOLD_LOCAL_RELIABILITY_RELATIVE},
            "h1_raw_m4_receipt": {
                "relative": plan.H1_RAW_RECEIPT_RELATIVE,
                "sha256": plan.H1_RAW_RECEIPT_SHA256,
            },
        },
        "m1_m10_coverage": m1_m10,
        "h1_m4_support_from_receipt": h1_support,
        "e1_live_first3": [
            {
                "session_name": row.get("session_name"),
                "first3_trial_ids": row.get("first3_trial_ids"),
                "n_support_blocks": row.get("n_support_blocks"),
                "support_seconds": row.get("support_seconds"),
            }
            for row in list(e1.get("sessions") or [])
        ],
        "e2_live_exposure": [
            {
                "session_name": row.get("session_name"),
                "legal_rows": row.get("legal_rows"),
                "n_fit_intersection": row.get("n_fit_intersection"),
                "n_eval_intersection": row.get("n_eval_intersection"),
                "fit_seconds": row.get("fit_seconds"),
                "eval_seconds": row.get("eval_seconds"),
                "static_smallest_eigenvalue": (
                    (row.get("static") or {}).get("spectrum", {}).get("smallest_eigenvalue")
                    if row.get("legal_rows")
                    else None
                ),
                "dynamic_smallest_eigenvalue": (
                    (row.get("dynamic") or {}).get("spectrum", {}).get("smallest_eigenvalue")
                    if row.get("legal_rows")
                    else None
                ),
            }
            for row in list(e2.get("sessions") or [])
        ],
        "do_not_infer_neff_from_bin_count": True,
        "argument_audit": {
            "feature_family_misspecification": (
                "H1 carrier is a backward decoder-weight descriptor after source PCA/"
                "ridge/U/EB. M1 carrier is a forward rSyn3 encoding. Shared [N,4] "
                "interface is not a shared physiological axis. FiLM V5 empty +0.0254 "
                "vs full +0.0239 shows profile-content misspecification relative to "
                "budget adaptation, not a global identity ceiling."
            ),
            "estimation_reliability": (
                "M1 rSyn3 M10 split-half W is 0.82-0.93 (fold0 0.911). Reliability "
                "alone is not a content gate. Budget probe PARTIAL_DECAY: weights_r2 "
                "falls as support shrinks while remaining above a weak RAW_STATS "
                "baseline. H1 three long trials are many correlated 100-ms blocks, "
                "not three scalar labels and not thousands of independent trials."
            ),
            "train_deploy_mismatch": (
                "Fold-local Stage1 trained the consumer on source sessions with a "
                "source-frozen dictionary; target uses M10 closed-form only. An "
                "all-source parent cannot be relabeled as clean LOSO. Visible-epoch "
                "picks are a product view, not unseen-session evidence."
            ),
            "consumer_sensitivity": (
                "Calibration-aware bottleneck gained 0.065-0.072 on DirectRidge and "
                "lost 0.021957 as a hard projection of full SPINT. FABLE TKD M1 "
                "closed-form consumer failed. A +0.005 later P delta would be a "
                "resource-routing value, not a measured noise floor or saturation."
            ),
        },
    }
    return contracts.AssayStatus(name="E3", status="READY", payload=payload)
