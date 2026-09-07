"""Compare fold-local receipts against the sealed rSyn3 V1 PASS table."""
from __future__ import annotations

from typing import Any, Mapping

from . import plan


def _coverage_payload(coverage: Mapping[str, object]) -> dict[str, object]:
    return {
        "budget": coverage.get("budget"),
        "n_trials": coverage.get("n_trials"),
        "valid_bins": coverage.get("valid_bins"),
        "valid_seconds": coverage.get("valid_seconds"),
        "design_rank": coverage.get("design_rank"),
        "design_condition": coverage.get("design_condition"),
        "smallest_eigenvalue": coverage.get("smallest_eigenvalue"),
        "rejected_for_trial_count": coverage.get("rejected_for_trial_count"),
        "finite_rates": coverage.get("finite_rates"),
        "trial_stratified_occupancy": coverage.get("trial_stratified_occupancy"),
        "normalized_gram_eigenvalues": coverage.get("normalized_gram_eigenvalues"),
        "per_synergy_dispersion": coverage.get("per_synergy_dispersion"),
    }


def _split_half(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in row:
            value = dict(row[name]) if isinstance(row[name], Mapping) else row[name]
            if isinstance(value, dict):
                value.pop("label", None)
            return value
    return None


def comparison_keys(fold_receipt: Mapping[str, object]) -> dict[str, object]:
    nmf = plan.require_mapping(fold_receipt.get("nmf") or {}, "nmf")
    rows = []
    for row in fold_receipt.get("rows") or []:
        mapping = plan.require_mapping(row, "row")
        rows.append({
            "budget": mapping.get("budget"),
            "carrier_digest": mapping.get("carrier_digest"),
            "coverage": _coverage_payload(plan.require_mapping(mapping.get("coverage") or {}, "coverage")),
            "split_half_rsyn3": mapping.get("split_half_rsyn3"),
            "split_half_pca3": _split_half(
                mapping,
                "split_half_rectified_trial_mean_pca3",
                "split_half_trial_mean_pca",
            ),
        })
    return {
        "dictionary_digest": nmf.get("dictionary_digest"),
        "scale_digest": nmf.get("scale_digest"),
        "reconstruction_digest": nmf.get("reconstruction_digest"),
        "order": nmf.get("order"),
        "rows": rows,
    }


def compare_fold(new_fold: Mapping[str, object], sealed_fold: Mapping[str, object]) -> dict[str, object]:
    left = comparison_keys(new_fold)
    right = comparison_keys(sealed_fold)
    mismatches: list[str] = []
    for key in ("dictionary_digest", "scale_digest", "reconstruction_digest", "order"):
        if left.get(key) != right.get(key):
            mismatches.append(str(key))
    left_rows = {int(row["budget"]): row for row in left["rows"]}
    right_rows = {int(row["budget"]): row for row in right["rows"]}
    if set(left_rows) != set(right_rows):
        mismatches.append("budgets")
    for budget in sorted(set(left_rows) & set(right_rows)):
        for field in ("carrier_digest", "coverage", "split_half_rsyn3", "split_half_pca3"):
            if left_rows[budget].get(field) != right_rows[budget].get(field):
                mismatches.append(f"M{budget}.{field}")
    return {
        "equal": not mismatches,
        "mismatches": mismatches,
        "n_rows": len(left["rows"]),
    }


def compare_all(new_receipts: Mapping[str, Any], sealed_receipts: Mapping[str, Any]) -> dict[str, object]:
    folds = {}
    all_equal = True
    for key in sorted(sealed_receipts, key=lambda item: int(item)):
        report = compare_fold(new_receipts[key], sealed_receipts[key])
        folds[str(key)] = report
        all_equal = all_equal and bool(report["equal"])
    return {
        "schema": "m1_emg_rsyn3_fold_local_sealed_pass_compare_v1",
        "sealed_root": plan.RSYN3_PASS_ROOT_RELATIVE,
        "sealed_fold_receipts_sha256": plan.RSYN3_PASS_FOLD_RECEIPTS_SHA256,
        "all_equal": all_equal,
        "folds": folds,
    }
