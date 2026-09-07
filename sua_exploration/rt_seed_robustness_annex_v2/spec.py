"""RT seed-43/44 robustness-annex v2 (CPU-only draft).

This version deliberately does not rank folds from hashes of result files.
The fold subset is the fixed, performance-independent rule ``fold % 3 == 0``
on the 15-fold development split.  Seed-42 results were known when the draft
was frozen; that disclosure is retained, but seed-42 scores cannot affect the
subset.  The target-evaluation accounting fields are explicitly named so they
cannot be confused with the optimizer used by source training.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sua_exploration.rt_seed_robustness_annex import spec as v1


ANNEX_ID = "rt_seed_robustness_annex_v2"
ROOT = v1.ROOT
FULL_SOURCE_REL = v1.FULL_SOURCE_REL
EVAL_SOURCE_REL = v1.EVAL_SOURCE_REL
EXPERIMENT_CONFIG_REL = v1.EXPERIMENT_CONFIG_REL
DATA_MODULE_REL = v1.DATA_MODULE_REL
SELECTION_CALLBACK_REL = v1.SELECTION_CALLBACK_REL
MODEL_REL = v1.MODEL_REL
DATA_CONFIG_REL = v1.DATA_CONFIG_REL
MODEL_CONFIG_REL = v1.MODEL_CONFIG_REL
TRAINER_CONFIG_REL = v1.TRAINER_CONFIG_REL

EXPECTED_FOLDS = tuple(range(15))
EXPECTED_SEEDS = (43, 44)
FULL_ARM = v1.FULL_ARM
MB4_ARM = v1.MB4_ARM
EXPECTED_STATUS = v1.EXPECTED_STATUS
SELECTED_FOLDS: tuple[int, ...] = tuple(fold for fold in EXPECTED_FOLDS if fold % 3 == 0)

# The two raw hashes below are intentionally retained per arm in every cell.
# They include arm-specific metadata (`arm`, `requested_side_feature_group`,
# and the normalizer's `feature_group`) and therefore must not be compared
# across Full/MB4.  Pairing uses the explicit invariant projections instead.
SOURCE_SPLIT_SCOPE_FIELDS: tuple[str, ...] = (
    "protocol",
    "validation_protocol",
    "outer_loso_fold",
    "loso_fold",
    "nested_selection",
    "calibration",
    "query",
    "source_sessions",
    "outer_source_sessions",
    "inner_train_sessions",
    "inner_validation_session",
    "target_session",
    "session_names",
    "all_sessions",
    "target_session_loaded_during_fit",
    "session_count",
    "loaded_fit_sessions",
    "source_sampler",
    "formal_heldout_opened",
    "development_only",
    "task",
)
SOURCE_NORMALIZER_NUMERIC_FIELDS: tuple[str, ...] = (
    "fit_scope",
    "fit_sessions",
    "excluded_inner_validation_session",
    "excluded_outer_target_session",
    "mean",
    "std",
)

# This metadata-only table is the split binding.  It is intentionally not
# read from a score-bearing aggregate at selection time.
FOLD_BINDINGS: tuple[dict[str, str | int], ...] = (
    {"fold": 0, "target_session": "ses-RT-20131009", "inner_validation_session": "ses-RT-20131010"},
    {"fold": 1, "target_session": "ses-RT-20131010", "inner_validation_session": "ses-RT-20131011"},
    {"fold": 2, "target_session": "ses-RT-20131011", "inner_validation_session": "ses-RT-20131028"},
    {"fold": 3, "target_session": "ses-RT-20131028", "inner_validation_session": "ses-RT-20131029"},
    {"fold": 4, "target_session": "ses-RT-20131029", "inner_validation_session": "ses-RT-20131209"},
    {"fold": 5, "target_session": "ses-RT-20131209", "inner_validation_session": "ses-RT-20131210"},
    {"fold": 6, "target_session": "ses-RT-20131210", "inner_validation_session": "ses-RT-20131212"},
    {"fold": 7, "target_session": "ses-RT-20131212", "inner_validation_session": "ses-RT-20131213"},
    {"fold": 8, "target_session": "ses-RT-20131213", "inner_validation_session": "ses-RT-20131217"},
    {"fold": 9, "target_session": "ses-RT-20131217", "inner_validation_session": "ses-RT-20131218"},
    {"fold": 10, "target_session": "ses-RT-20131218", "inner_validation_session": "ses-RT-20150316"},
    {"fold": 11, "target_session": "ses-RT-20150316", "inner_validation_session": "ses-RT-20150317"},
    {"fold": 12, "target_session": "ses-RT-20150317", "inner_validation_session": "ses-RT-20150318"},
    {"fold": 13, "target_session": "ses-RT-20150318", "inner_validation_session": "ses-RT-20150320"},
    {"fold": 14, "target_session": "ses-RT-20150320", "inner_validation_session": "ses-RT-20131009"},
)


class SpecError(ValueError):
    """Raised when a draft cell violates the v2 contract."""


def _projection_hash(value: Mapping[str, Any]) -> str:
    return v1._sha256_bytes(v1.canonical_json_bytes(value))


def source_split_scope_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen arm-invariant nested-LOSO split projection.

    This deliberately rejects a missing field instead of silently dropping a
    future split change.  Arm-specific carrier/audit fields are not part of
    the projection; they remain covered by each arm's raw manifest hash.
    """

    missing = [field for field in SOURCE_SPLIT_SCOPE_FIELDS if field not in manifest]
    if missing:
        raise SpecError(f"source split scope projection missing fields: {missing}")
    return {
        "schema": "rt_seed_robustness_annex_v2_source_split_scope_projection_v1",
        "fields": {field: manifest[field] for field in SOURCE_SPLIT_SCOPE_FIELDS},
    }


def source_split_scope_sha256(manifest: Mapping[str, Any]) -> str:
    return _projection_hash(source_split_scope_projection(manifest))


def source_normalizer_numeric_projection(normalizer: Mapping[str, Any]) -> dict[str, Any]:
    """Return the numeric/source-scope normalizer projection.

    `feature_group` is intentionally excluded.  Fit scope, source sessions,
    exclusions, mean and standard deviation remain frozen and are therefore
    able to detect source/session/numeric drift.
    """

    missing = [field for field in SOURCE_NORMALIZER_NUMERIC_FIELDS if field not in normalizer]
    if missing:
        raise SpecError(f"source normalizer projection missing fields: {missing}")
    return {
        "schema": "rt_seed_robustness_annex_v2_source_normalizer_numeric_projection_v1",
        "fields": {field: normalizer[field] for field in SOURCE_NORMALIZER_NUMERIC_FIELDS},
    }


def source_normalizer_numeric_sha256(normalizer: Mapping[str, Any]) -> str:
    return _projection_hash(source_normalizer_numeric_projection(normalizer))


def _bindings() -> list[dict[str, Any]]:
    return [dict(item) for item in FOLD_BINDINGS]


def compute_selection() -> dict[str, Any]:
    """Return the fixed fold-ID selection without reading result bytes."""

    preimage = {
        "protocol": ANNEX_ID,
        "candidate_fold_ids": list(EXPECTED_FOLDS),
        "candidate_fold_bindings": _bindings(),
        "rule": "select fold IDs satisfying fold % 3 == 0",
        "selected_fold_ids": list(SELECTED_FOLDS),
    }
    encoded = v1.canonical_json_bytes(preimage)
    return {
        "method": "performance_independent_fold_id_modulo_v2",
        "selection_used_scores": False,
        "score_bearing_anchor_hashes_used": False,
        "selection_preimage": preimage,
        "selection_preimage_sha256": v1._sha256_bytes(encoded),
        "canonical_json_sha256_input_bytes": len(encoded),
        "selected_fold_ids": list(SELECTED_FOLDS),
        "selected_rule_order": list(SELECTED_FOLDS),
        "ranked_folds": [],
    }


def _implementation_snapshot() -> dict[str, Any]:
    return v1._implementation_snapshot()


def _runner_blockers() -> list[dict[str, str]]:
    return v1._runner_blockers()


def build_draft_receipt(*, freeze_date: str = "2026-08-10") -> dict[str, Any]:
    """Build the v2 no-GPU draft receipt."""

    selection = compute_selection()
    bindings = {int(item["fold"]): item for item in selection["selection_preimage"]["candidate_fold_bindings"]}
    cells = [
        {
            "fold": fold,
            "target_session": bindings[fold]["target_session"],
            "inner_validation_session": bindings[fold]["inner_validation_session"],
            "seed": seed,
            "arms": [FULL_ARM, MB4_ARM],
            "fresh_fit": True,
            "paired_within_seed_and_fold": True,
        }
        for seed in EXPECTED_SEEDS
        for fold in SELECTED_FOLDS
    ]
    anchor_meta = {
        "full_seed42": {
            "path": v1.FULL_AGGREGATE_REL.as_posix(),
            "schema": "rt_seed42_clean_nested_loso_aggregate_v1",
            "status": "PASS_RT_SEALED",
            "seed": 42,
            "folds": list(EXPECTED_FOLDS),
            "score_bytes_used_for_selection": False,
        },
        "mb4_seed42": {
            "path": v1.MB4_AGGREGATE_REL.as_posix(),
            "schema": "rt_mb4_matched_full_minus_mb4_aggregate_v1",
            "status": "PASS_RT_FULL_MINUS_MB4_ALL_15_PAIRED",
            "seed": 42,
            "folds": list(EXPECTED_FOLDS),
            "score_bytes_used_for_selection": False,
        },
        "seed42_seal_marker": {
            "path": v1.SEAL_MARKER_REL.as_posix(),
            "schema": "rt_seed42_clean_nested_loso_seal_marker_v1",
            "status": "PASS_RT_SEALED",
            "seed": 42,
            "folds": list(EXPECTED_FOLDS),
            "score_bytes_used_for_selection": False,
        },
    }
    return {
        "schema": ANNEX_ID,
        "status": "DRAFT_PREREGISTRATION_NOT_AUTHORIZED_NO_GPU",
        "supersedes": "rt_seed_robustness_annex_v1",
        "development_only": True,
        "formal_heldout_opened": False,
        "freeze": {
            "date": freeze_date,
            "seed42_primary_known_at_freeze": True,
            "seed42_blinded": False,
            "seed43_44_outputs_known_at_freeze": False,
            "selection_used_scores": False,
            "selection_rule": "fold IDs [0,3,6,9,12] (fold % 3 == 0), fixed before seed-43/44 outputs",
            "selection_disclosure": "Seed-42 results were known when this draft was frozen. The fixed fold-ID rule and metadata-only binding table do not read or hash seed-42 performance fields; this is procedural score-independence, not a blinded seed-42 selection.",
        },
        "immutable_anchor_metadata": anchor_meta,
        "implementation_snapshot": _implementation_snapshot(),
        "selection": selection,
        "protocol": {
            "task": "rt",
            "split": "development_clean_nested_outer_LOSO",
            "outer_folds": list(SELECTED_FOLDS),
            "outer_fold_sampling_unit": "fold_within_seed; seed is a nested training repeat; never pool seed×fold as independent samples",
            "arms": {
                FULL_ARM: {"side_dim": 4, "descriptor": "[w_x,w_y,||W||,b]", "implementation_group": "k4"},
                MB4_ARM: {"side_dim": 4, "descriptor": "[0,0,||W||,b]", "implementation_group": "k4__normalized_component_mask"},
            },
            "seeds": list(EXPECTED_SEEDS),
            "cells": len(EXPECTED_SEEDS) * len(SELECTED_FOLDS) * 2,
            "paired_fold_seed_pairs": len(EXPECTED_SEEDS) * len(SELECTED_FOLDS),
            "fresh_fit_from_scratch": True,
            "no_seed42_checkpoint_reuse": True,
            "support_budget_trials": 24,
            "query_start_trial": 24,
            "bin_size_ms": 20,
            "window_size_bins": 50,
            "max_epochs": 35,
            "no_early_stopping": True,
            "checkpoint_selection": "val_heldin/r2_mean on the cyclic inner-validation session only",
            "target_evaluation": "one-shot outer target; no target optimizer, backward, Trainer.test, or target query labels during fit/selection",
            "target_calibration": "support velocity labels only; target query labels are scoring-only",
            "paired_initialization": "same fold/seed must record identical initial_state_hash for Full and MB4; missing hash is fail-closed",
            "source_training_optimizer": "present and allowed; this field is distinct from target_optimizer_present",
        },
        "per_cell_required_fields": [
            "seed", "fold", "arm", "target_session", "inner_validation_session",
            "r2_variance_weighted", "query_windows_evaluated", "selected_epoch", "selected_global_step", "status",
            "target_model_state_unchanged", "target_model_state_before_sha256", "target_model_state_after_sha256",
            "target_backpropagation", "target_optimizer_present", "source_split_manifest_sha256", "source_normalizer_sha256",
            "source_split_scope_sha256", "source_normalizer_numeric_sha256",
            "checkpoint_sha256", "parameter_count", "macs_per_decode_call", "cached_state_bytes", "initial_state_hash",
            "implementation_snapshot_sha256", "paired_initial_state_equal_before_target_eval",
            "paired_initial_state_receipt", "target_forward_only_accounting",
        ],
        "target_evaluation_fields": {
            "target_model_state_unchanged": "must be true; hash before/after the outer forward-only evaluation",
            "target_optimizer_present": "must be false; source-training optimizers are not described by this field",
            "target_backpropagation": "must be false",
            "target_model_state_hashes": ["target_model_state_before_sha256", "target_model_state_after_sha256"],
        },
        "pair_invariants": {
            "exact_equal": [
                "parameter_count", "macs_per_decode_call", "cached_state_bytes",
                "initial_state_hash", "source_split_scope_sha256",
                "source_normalizer_numeric_sha256",
            ],
            "raw_hashes_retained_per_arm_only": [
                "source_split_manifest_sha256", "source_normalizer_sha256",
            ],
            "required": {"target_model_state_unchanged": True, "target_backpropagation": False, "target_optimizer_present": False},
        },
        "readout": {
            "pair_delta": "Full R2 minus MB4 R2, by fold and seed",
            "per_seed": ["mean", "median", "positive_negative_tie_counts", "sample_sd", "sample_se"],
            "cross_seed": ["two_seed_means", "two_seed_medians", "mean_of_seed_means", "both_seed_mean_positive", "each_seed_at_least_4_of_5_positive", "all_10_sign_count"],
            "labels": {
                "ROBUST_DIRECTIONAL": "both seed means and medians > 0 and each seed has at least 4/5 positive folds",
                "MIXED_DIRECTION": "complete cells but the ROBUST_DIRECTIONAL rule is not met",
                "INCOMPLETE_NO_AGGREGATE": "any missing, failed, invalid, or accounting-incomplete cell",
            },
            "inferential_claim": "none; descriptive robustness only because folds are nested within seed",
        },
        "failure_policy": {
            "missing_or_failed_cell": "no aggregate, no imputation, no substitute fold/seed, no score-based rerun; preserve seed-42 primary unchanged",
            "accounting_or_pair_mismatch": "invalid and no aggregate",
        },
        "resource_plan": {
            "gpu_fit_cells": len(EXPECTED_SEEDS) * len(SELECTED_FOLDS) * 2,
            "outer_score_only_evals": len(EXPECTED_SEEDS) * len(SELECTED_FOLDS) * 2,
            "paired_waves_if_two_gpus": len(EXPECTED_SEEDS) * len(SELECTED_FOLDS),
            "epoch_equivalent_fit_units": len(EXPECTED_SEEDS) * len(SELECTED_FOLDS) * 2 * 35,
            "wall_clock_estimate_status": "UNAVAILABLE_NO_SEALED_TIMING",
            "storage_estimate_status": "UNAVAILABLE_NO_SEALED_CHECKPOINT_SIZE",
            "no_gpu_launched_by_this_receipt": True,
        },
        "launch_blockers": _runner_blockers(),
        "cells": cells,
    }


def _require_keys(row: Mapping[str, Any], required: Iterable[str], *, label: str) -> None:
    missing = sorted(key for key in required if key not in row)
    if missing:
        raise SpecError(f"{label} missing required fields: {missing}")


def _sample_sd(values: Sequence[float]) -> float:
    return float("nan") if len(values) < 2 else statistics.stdev(values)


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {"positive": sum(value > 0 for value in values), "negative": sum(value < 0 for value in values), "tie": sum(value == 0 for value in values)}


def _summarize_seed(seed: int, deltas: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in deltas]
    sd = _sample_sd(values)
    return {"seed": int(seed), "n_folds": len(values), "mean_delta": statistics.fmean(values), "median_delta": statistics.median(values), "sign_counts": _sign_counts(values), "sample_sd": sd, "sample_se": sd / math.sqrt(len(values)) if math.isfinite(sd) else float("nan"), "deltas": values}


def _validate_metric_row(row: Mapping[str, Any]) -> None:
    required = tuple(build_draft_receipt()["per_cell_required_fields"])
    _require_keys(row, required, label=f"cell seed={row.get('seed')} fold={row.get('fold')} arm={row.get('arm')}")
    if int(row["seed"]) not in EXPECTED_SEEDS or int(row["fold"]) not in SELECTED_FOLDS:
        raise SpecError("unexpected annex seed or fold")
    if row["arm"] not in (FULL_ARM, MB4_ARM):
        raise SpecError(f"unexpected annex arm: {row['arm']!r}")
    if row["status"] != EXPECTED_STATUS:
        raise SpecError(f"cell is not a clean outer-evaluation pass: {row['status']!r}")
    if row["target_model_state_unchanged"] is not True:
        raise SpecError("target model state changed")
    if row["target_backpropagation"] is not False:
        raise SpecError("target backpropagation is forbidden")
    if row["target_optimizer_present"] is not False:
        raise SpecError("target optimizer is present")
    if row["paired_initial_state_equal_before_target_eval"] is not True:
        raise SpecError("paired initial-state equality was not established before target evaluation")
    for field in (
        "target_model_state_before_sha256",
        "target_model_state_after_sha256",
        "source_split_manifest_sha256",
        "source_normalizer_sha256",
        "source_split_scope_sha256",
        "source_normalizer_numeric_sha256",
        "checkpoint_sha256",
        "initial_state_hash",
        "implementation_snapshot_sha256",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise SpecError(f"{field} must be a non-empty hash string")
    if row["target_model_state_before_sha256"] != row["target_model_state_after_sha256"]:
        raise SpecError("target model state before/after hash mismatch")
    try:
        r2 = float(row["r2_variance_weighted"])
    except (TypeError, ValueError) as exc:
        raise SpecError("R2 is not numeric") from exc
    if not math.isfinite(r2) or int(row["query_windows_evaluated"]) <= 0:
        raise SpecError("R2 must be finite and query window count positive")
    for field in ("parameter_count", "macs_per_decode_call", "cached_state_bytes"):
        if int(row[field]) < 0:
            raise SpecError(f"{field} must be non-negative")
    accounting = row["target_forward_only_accounting"]
    if not isinstance(accounting, Mapping):
        raise SpecError("target_forward_only_accounting must be a mapping")
    for field in ("parameter_count", "macs_per_decode_call", "cached_state_bytes"):
        if int(accounting.get(field, -1)) != int(row[field]):
            raise SpecError(f"target forward-only accounting mismatch: {field}")
    if accounting.get("paired_full_mb4_equal_before_target_eval") is not True:
        raise SpecError("target accounting was not pair-validated before target evaluation")


def aggregate_cells(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and summarize the complete v2 20-cell grid."""

    expected = {(seed, fold, arm) for seed in EXPECTED_SEEDS for fold in SELECTED_FOLDS for arm in (FULL_ARM, MB4_ARM)}
    bindings = {int(item["fold"]): item for item in FOLD_BINDINGS}
    seen: set[tuple[int, int, str]] = set()
    by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        _validate_metric_row(row)
        key = (int(row["seed"]), int(row["fold"]), str(row["arm"]))
        binding = bindings[key[1]]
        if row["target_session"] != binding["target_session"] or row["inner_validation_session"] != binding["inner_validation_session"]:
            raise SpecError(f"fold/session binding mismatch at {key}")
        if key in seen:
            raise SpecError(f"duplicate cell: {key}")
        seen.add(key)
        by_key[key] = row
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise SpecError(f"incomplete annex grid; missing={missing!r}, extra={extra!r}")

    deltas_by_seed: dict[int, list[float]] = {seed: [] for seed in EXPECTED_SEEDS}
    pair_rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for fold in SELECTED_FOLDS:
            full = by_key[(seed, fold, FULL_ARM)]
            mb4 = by_key[(seed, fold, MB4_ARM)]
            for field in (
                "target_session",
                "inner_validation_session",
                "query_windows_evaluated",
                "parameter_count",
                "macs_per_decode_call",
                "cached_state_bytes",
                "initial_state_hash",
                "source_split_scope_sha256",
                "source_normalizer_numeric_sha256",
            ):
                if full[field] != mb4[field]:
                    raise SpecError(f"paired {field} mismatch at seed={seed} fold={fold}")
            delta = float(full["r2_variance_weighted"]) - float(mb4["r2_variance_weighted"])
            deltas_by_seed[seed].append(delta)
            pair_rows.append({"seed": seed, "fold": fold, "target_session": full["target_session"], "inner_validation_session": full["inner_validation_session"], "full_r2": float(full["r2_variance_weighted"]), "mb4_r2": float(mb4["r2_variance_weighted"]), "delta_full_minus_mb4": delta})
    per_seed = [_summarize_seed(seed, deltas_by_seed[seed]) for seed in EXPECTED_SEEDS]
    all_deltas = [delta for values in deltas_by_seed.values() for delta in values]
    both_means_positive = all(item["mean_delta"] > 0 for item in per_seed)
    both_medians_positive = all(item["median_delta"] > 0 for item in per_seed)
    each_seed_four_positive = all(item["sign_counts"]["positive"] >= 4 for item in per_seed)
    label = "ROBUST_DIRECTIONAL" if both_means_positive and both_medians_positive and each_seed_four_positive else "MIXED_DIRECTION"
    first = by_key[(EXPECTED_SEEDS[0], SELECTED_FOLDS[0], FULL_ARM)]
    return {
        "status": "COMPLETE_DESCRIPTIVE_ONLY",
        "label": label,
        "sampling_unit": "outer_fold_within_seed; seeds are nested training repeats",
        "do_not_pool_seed_fold_as_independent": True,
        "pair_count": len(pair_rows),
        "per_seed": per_seed,
        "cross_seed": {"seed_means": [item["mean_delta"] for item in per_seed], "seed_medians": [item["median_delta"] for item in per_seed], "mean_of_seed_means": statistics.fmean(item["mean_delta"] for item in per_seed), "both_seed_mean_positive": both_means_positive, "both_seed_median_positive": both_medians_positive, "each_seed_at_least_4_of_5_positive": each_seed_four_positive, "all_10_sign_counts": _sign_counts(all_deltas)},
        "accounting": {"parameter_count": first["parameter_count"], "macs_per_decode_call": first["macs_per_decode_call"], "cached_state_bytes": first["cached_state_bytes"], "paired_equal_all_cells": True, "paired_scope_projection_equal_all_cells": True, "paired_numeric_normalizer_equal_all_cells": True, "target_model_state_unchanged_all_cells": True, "target_backpropagation_false_all_cells": True, "target_optimizer_absent_all_cells": True},
        "pairs": pair_rows,
        "inferential_significance_claim": "none",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-draft", type=Path, help="write a no-GPU v2 draft receipt")
    args = parser.parse_args(argv)
    payload = build_draft_receipt()
    if args.write_draft is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _write_json(args.write_draft, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
