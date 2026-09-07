"""RT seed-43/44 robustness-annex specification and CPU aggregation.

The annex is intentionally a preregistration, not a launcher.  All selection
and validation here are pure-Python operations over immutable receipts; no
torch, CUDA, NWB, or target-session data are imported.  The five outer folds
are selected by a SHA-256 rank anchored to the already sealed seed-42 assets,
not by any R2 value.  The disclosure is explicit that this is not a blind
seed-42 selection: seed 42 was known when the draft was frozen, while seeds
43/44 were not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ANNEX_ID = "rt_seed_robustness_annex_v1"
ROOT = Path(__file__).resolve().parents[2]
FULL_AGGREGATE_REL = Path(
    "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_aggregate.json"
)
MB4_AGGREGATE_REL = Path(
    "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_FULL15_AGGREGATE_v1.json"
)
SEAL_MARKER_REL = Path(
    "sua_exploration/results/k4_rt_loso_v1/rt_seed42_clean_nested_seal.marker"
)

FULL_SOURCE_REL = Path("streaming_calibration_exp/scripts/run_rt_clean_nested_loso.py")
EVAL_SOURCE_REL = Path("streaming_calibration_exp/src/rt_clean_nested_loso_eval.py")
EXPERIMENT_CONFIG_REL = Path(
    "streaming_calibration_exp/configs/experiment/rt_clean_nested_loso_m24.yaml"
)
DATA_MODULE_REL = Path("streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py")
SELECTION_CALLBACK_REL = Path(
    "streaming_calibration_exp/src/callbacks/rt_nested_selection_receipt.py"
)
MODEL_REL = Path("streaming_calibration_exp/src/models/components/streaming_spint.py")
DATA_CONFIG_REL = Path("streaming_calibration_exp/configs/data/rt_nested_loso_m24.yaml")
MODEL_CONFIG_REL = Path("streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml")
TRAINER_CONFIG_REL = Path("streaming_calibration_exp/configs/trainer/default.yaml")

EXPECTED_FOLDS = tuple(range(15))
EXPECTED_SEEDS = (43, 44)
FULL_ARM = "afc4_vel"
MB4_ARM = "afc4_mb4"
EXPECTED_STATUS = "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP"
SELECTED_FOLDS: tuple[int, ...] = (0, 3, 4, 11, 12)


class SpecError(ValueError):
    """Raised when an immutable anchor or a future cell violates the contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read JSON anchor {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecError(f"JSON anchor is not an object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical bytes used for all selection hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_path(rel: Path) -> str:
    return rel.as_posix()


def _anchor(rel: Path, *, schema: str, status: str, seed: int, folds: Sequence[int]) -> dict[str, Any]:
    path = ROOT / rel
    if not path.is_file():
        raise SpecError(f"missing immutable anchor: {path}")
    payload = _read_json(path)
    actual_hash = sha256_file(path)
    if payload.get("schema") != schema:
        raise SpecError(f"anchor schema drift for {rel}: {payload.get('schema')!r}")
    if payload.get("status") != status:
        raise SpecError(f"anchor status drift for {rel}: {payload.get('status')!r}")
    if int(payload.get("seed", -1)) != int(seed):
        raise SpecError(f"anchor seed drift for {rel}: {payload.get('seed')!r}")
    listed = tuple(int(x) for x in payload.get("folds", ()))
    if listed != tuple(folds):
        raise SpecError(f"anchor fold list drift for {rel}: {listed!r}")
    return {
        "path": _canonical_path(rel),
        "sha256": actual_hash,
        "schema": schema,
        "status": status,
        "seed": int(seed),
        "folds": list(folds),
        "mode": f"{path.stat().st_mode & 0o777:03o}",
    }


def _full_fold_bindings() -> list[dict[str, Any]]:
    path = ROOT / FULL_AGGREGATE_REL
    payload = _read_json(path)
    rows = [row for row in payload.get("cells", []) if row.get("arm") == FULL_ARM]
    by_fold: dict[int, dict[str, Any]] = {}
    for row in rows:
        fold = int(row["fold"])
        if fold in by_fold:
            raise SpecError(f"duplicate Full fold in seed-42 aggregate: {fold}")
        by_fold[fold] = {
            "fold": fold,
            "target_session": str(row["target_session"]),
            "inner_validation_session": str(row["inner_validation_session"]),
        }
    if tuple(sorted(by_fold)) != EXPECTED_FOLDS:
        raise SpecError(f"Full anchor does not expose exactly folds 0..14: {sorted(by_fold)}")
    return [by_fold[fold] for fold in EXPECTED_FOLDS]


def selection_preimage(
    *, full_sha256: str, mb4_sha256: str, seal_sha256: str, bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the score-independent object committed by the subset hash.

    Do not add R2, score, metric, checkpoint-selection values, or any result
    field to this object.  The source anchor hashes are deliberately included
    so a changed seed-42 protocol cannot silently select another subset.
    """

    return {
        "protocol": ANNEX_ID,
        "candidate_fold_ids": list(EXPECTED_FOLDS),
        "candidate_fold_bindings": [dict(item) for item in bindings],
        "seed42_full_aggregate_sha256": full_sha256,
        "seed42_mb4_aggregate_sha256": mb4_sha256,
        "seed42_seal_sha256": seal_sha256,
    }


def compute_selection(
    *, full_sha256: str | None = None, mb4_sha256: str | None = None, seal_sha256: str | None = None
) -> dict[str, Any]:
    """Return the deterministic five-fold selection and its audit trail."""

    if full_sha256 is None:
        full_sha256 = sha256_file(ROOT / FULL_AGGREGATE_REL)
    if mb4_sha256 is None:
        mb4_sha256 = sha256_file(ROOT / MB4_AGGREGATE_REL)
    if seal_sha256 is None:
        seal_sha256 = sha256_file(ROOT / SEAL_MARKER_REL)
    bindings = _full_fold_bindings()
    preimage = selection_preimage(
        full_sha256=full_sha256,
        mb4_sha256=mb4_sha256,
        seal_sha256=seal_sha256,
        bindings=bindings,
    )
    preimage_bytes = canonical_json_bytes(preimage)
    preimage_sha256 = _sha256_bytes(preimage_bytes)
    ranked = sorted(
        (
            _sha256_bytes(f"{preimage_sha256}|fold={fold}".encode("ascii")),
            fold,
        )
        for fold in EXPECTED_FOLDS
    )
    selected_ranked = ranked[:5]
    selected_sorted = sorted(fold for _, fold in selected_ranked)
    return {
        "method": "sha256_anchor_rank_v1",
        "selection_used_scores": False,
        "selection_preimage": preimage,
        "selection_preimage_sha256": preimage_sha256,
        "canonical_json_sha256_input_bytes": len(preimage_bytes),
        "ranked_folds": [
            {"rank": rank, "fold": fold, "rank_sha256": digest}
            for rank, (digest, fold) in enumerate(ranked, start=1)
        ],
        "selected_rank_order": [fold for _, fold in selected_ranked],
        "selected_fold_ids": selected_sorted,
    }


def _implementation_snapshot() -> dict[str, Any]:
    paths = (
        FULL_SOURCE_REL,
        EVAL_SOURCE_REL,
        EXPERIMENT_CONFIG_REL,
        DATA_MODULE_REL,
        SELECTION_CALLBACK_REL,
        MODEL_REL,
        DATA_CONFIG_REL,
        MODEL_CONFIG_REL,
        TRAINER_CONFIG_REL,
    )
    snapshot: dict[str, Any] = {}
    for rel in paths:
        path = ROOT / rel
        if not path.is_file():
            raise SpecError(f"missing implementation anchor: {path}")
        snapshot[_canonical_path(rel)] = {
            "sha256": sha256_file(path),
            "mode": f"{path.stat().st_mode & 0o777:03o}",
        }
    return snapshot


def _runner_blockers() -> list[dict[str, str]]:
    runner = (ROOT / FULL_SOURCE_REL).read_text(encoding="utf-8")
    blockers: list[dict[str, str]] = []
    # The data module/evaluator know afc4_mb4, but the only checked-in CLI
    # launcher currently rejects it.  Refuse to call an unreviewed workaround.
    if "afc4_mb4" not in runner or "choices=(\"afc4_vel\", \"zero4\", \"afc4_rs\", \"afc4_ls\", \"afc4_b4\", \"afc4_w4\")" in runner:
        blockers.append(
            {
                "code": "RUNNER_MISSING_AFC4_MB4_CLI",
                "detail": "run_rt_clean_nested_loso.py does not accept afc4_mb4; add and hash-review the launcher before GPU launch.",
            }
        )
    blockers.append(
        {
            "code": "INITIAL_STATE_HASH_NOT_EXPOSED",
            "detail": "future paired cells must record an initial_state_hash; no sealed receipt currently exposes this field.",
        }
    )
    blockers.append(
        {
            "code": "ACCOUNTING_REQUIRED_AT_RUN_TIME",
            "detail": "seed-42 MB4 receipt explicitly lacks parameter/MAC/state accounting; every new cell must emit all three fields.",
        }
    )
    return blockers


def build_draft_receipt(*, freeze_date: str = "2026-08-10") -> dict[str, Any]:
    """Build a machine-readable, no-GPU draft receipt."""

    full = _anchor(
        FULL_AGGREGATE_REL,
        schema="rt_seed42_clean_nested_loso_aggregate_v1",
        status="PASS_RT_SEALED",
        seed=42,
        folds=EXPECTED_FOLDS,
    )
    mb4 = _anchor(
        MB4_AGGREGATE_REL,
        schema="rt_mb4_matched_full_minus_mb4_aggregate_v1",
        status="PASS_RT_FULL_MINUS_MB4_ALL_15_PAIRED",
        seed=42,
        folds=EXPECTED_FOLDS,
    )
    seal = _anchor(
        SEAL_MARKER_REL,
        schema="rt_seed42_clean_nested_loso_seal_marker_v1",
        status="PASS_RT_SEALED",
        seed=42,
        folds=EXPECTED_FOLDS,
    )
    full_payload = _read_json(ROOT / FULL_AGGREGATE_REL)
    mb4_payload = _read_json(ROOT / MB4_AGGREGATE_REL)
    if mb4_payload.get("full_comparator_arm") != FULL_ARM:
        raise SpecError("MB4 anchor does not identify afc4_vel as its Full comparator")
    if mb4_payload.get("r_c_reference", {}).get("aggregate_sha256") != full["sha256"]:
        raise SpecError("MB4 anchor does not reference the sealed Full aggregate")
    if mb4_payload.get("r_c_reference", {}).get("seal_sha256") != seal["sha256"]:
        raise SpecError("MB4 anchor does not reference the sealed seal marker")
    selection = compute_selection(
        full_sha256=full["sha256"], mb4_sha256=mb4["sha256"], seal_sha256=seal["sha256"]
    )
    bindings = selection["selection_preimage"]["candidate_fold_bindings"]
    by_fold = {int(row["fold"]): row for row in bindings}

    cells = [
        {
            "fold": fold,
            "target_session": by_fold[fold]["target_session"],
            "inner_validation_session": by_fold[fold]["inner_validation_session"],
            "seed": seed,
            "arms": [FULL_ARM, MB4_ARM],
            "fresh_fit": True,
            "paired_within_seed_and_fold": True,
        }
        for seed in EXPECTED_SEEDS
        for fold in selection["selected_fold_ids"]
    ]
    return {
        "schema": ANNEX_ID,
        "status": "DRAFT_PREREGISTRATION_NOT_AUTHORIZED_NO_GPU",
        "development_only": True,
        "formal_heldout_opened": False,
        "freeze": {
            "date": freeze_date,
            "seed42_primary_known_at_freeze": True,
            "seed42_blinded": False,
            "seed43_44_outputs_known_at_freeze": False,
            "selection_used_scores": False,
            "selection_disclosure": "Subset was chosen after sealed seed-42 results were known and before seed-43/44 outputs were known; this is procedural score-independence, not a blinded seed-42 selection.",
        },
        "immutable_anchors": {"full_seed42": full, "mb4_seed42": mb4, "seed42_seal_marker": seal},
        "implementation_snapshot": _implementation_snapshot(),
        "selection": selection,
        "protocol": {
            "task": "rt",
            "split": "development_clean_nested_outer_LOSO",
            "outer_folds": selection["selected_fold_ids"],
            "outer_fold_sampling_unit": "fold_within_seed; seed is a nested training repeat; never pool seed×fold as independent samples",
            "arms": {
                FULL_ARM: {
                    "side_dim": 4,
                    "descriptor": "[w_x,w_y,||W||,b]",
                    "implementation_group": "k4",
                },
                MB4_ARM: {
                    "side_dim": 4,
                    "descriptor": "[0,0,||W||,b]",
                    "implementation_group": "k4__normalized_component_mask",
                },
            },
            "seeds": list(EXPECTED_SEEDS),
            "cells": 20,
            "paired_fold_seed_pairs": 10,
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
        },
        "per_cell_required_fields": [
            "seed",
            "fold",
            "arm",
            "target_session",
            "inner_validation_session",
            "r2_variance_weighted",
            "query_windows_evaluated",
            "selected_epoch",
            "selected_global_step",
            "status",
            "model_state_unchanged",
            "model_state_before_sha256",
            "model_state_after_sha256",
            "target_backpropagation",
            "optimizer_present",
            "source_split_manifest_sha256",
            "source_normalizer_sha256",
            "checkpoint_sha256",
            "parameter_count",
            "macs_per_decode_call",
            "cached_state_bytes",
            "initial_state_hash",
        ],
        "pair_invariants": {
            "exact_equal": ["parameter_count", "macs_per_decode_call", "cached_state_bytes", "initial_state_hash"],
            "required": {"model_state_unchanged": True, "target_backpropagation": False, "optimizer_present": False},
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
            "gpu_fit_cells": 20,
            "outer_score_only_evals": 20,
            "paired_waves_if_two_gpus": 10,
            "epoch_equivalent_fit_units": 700,
            "wall_clock_estimate_status": "UNAVAILABLE_NO_SEALED_TIMING",
            "storage_estimate_status": "UNAVAILABLE_NO_SEALED_CHECKPOINT_SIZE",
            "no_gpu_launched_by_this_receipt": True,
        },
        "launch_blockers": _runner_blockers(),
    }


def _require_keys(row: Mapping[str, Any], required: Iterable[str], *, label: str) -> None:
    missing = sorted(key for key in required if key not in row)
    if missing:
        raise SpecError(f"{label} missing required fields: {missing}")


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values)


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "positive": sum(value > 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "tie": sum(value == 0 for value in values),
    }


def _summarize_seed(seed: int, deltas: Sequence[float]) -> dict[str, Any]:
    values = list(float(x) for x in deltas)
    sd = _sample_sd(values)
    return {
        "seed": int(seed),
        "n_folds": len(values),
        "mean_delta": statistics.fmean(values),
        "median_delta": statistics.median(values),
        "sign_counts": _sign_counts(values),
        "sample_sd": sd,
        "sample_se": (sd / math.sqrt(len(values))) if math.isfinite(sd) else float("nan"),
        "deltas": values,
    }


def _validate_metric_row(row: Mapping[str, Any]) -> None:
    required = (
        "seed",
        "fold",
        "arm",
        "target_session",
        "inner_validation_session",
        "r2_variance_weighted",
        "query_windows_evaluated",
        "selected_epoch",
        "selected_global_step",
        "status",
        "model_state_unchanged",
        "model_state_before_sha256",
        "model_state_after_sha256",
        "target_backpropagation",
        "optimizer_present",
        "source_split_manifest_sha256",
        "source_normalizer_sha256",
        "checkpoint_sha256",
        "parameter_count",
        "macs_per_decode_call",
        "cached_state_bytes",
        "initial_state_hash",
    )
    _require_keys(row, required, label=f"cell seed={row.get('seed')} fold={row.get('fold')} arm={row.get('arm')}")
    if int(row["seed"]) not in EXPECTED_SEEDS:
        raise SpecError(f"unexpected annex seed: {row['seed']!r}")
    if int(row["fold"]) not in SELECTED_FOLDS:
        raise SpecError(f"unexpected annex fold: {row['fold']!r}")
    if row["arm"] not in (FULL_ARM, MB4_ARM):
        raise SpecError(f"unexpected annex arm: {row['arm']!r}")
    if row["status"] != EXPECTED_STATUS:
        raise SpecError(f"cell is not a clean outer-evaluation pass: {row['status']!r}")
    if row["model_state_unchanged"] is not True:
        raise SpecError("outer model state changed")
    if row["target_backpropagation"] is not False:
        raise SpecError("target backpropagation is forbidden")
    if row["optimizer_present"] is not False:
        raise SpecError("target optimizer is present")
    for field in (
        "model_state_before_sha256",
        "model_state_after_sha256",
        "source_split_manifest_sha256",
        "source_normalizer_sha256",
        "checkpoint_sha256",
        "initial_state_hash",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise SpecError(f"{field} must be a non-empty hash string")
    if row["model_state_before_sha256"] != row["model_state_after_sha256"]:
        raise SpecError("outer model state before/after hash mismatch")
    try:
        r2 = float(row["r2_variance_weighted"])
    except (TypeError, ValueError) as exc:
        raise SpecError("R2 is not numeric") from exc
    if not math.isfinite(r2):
        raise SpecError("R2 is not finite")
    if int(row["query_windows_evaluated"]) <= 0:
        raise SpecError("query window count must be positive")
    for field in ("parameter_count", "macs_per_decode_call", "cached_state_bytes"):
        if int(row[field]) < 0:
            raise SpecError(f"{field} must be non-negative")
    if not str(row["initial_state_hash"]):
        raise SpecError("initial_state_hash is empty")


def aggregate_cells(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and summarize the complete 20-cell annex result.

    Raises :class:`SpecError` for any missing, duplicate, failed, invalid, or
    accounting-mismatched cell.  No score-based rerun or imputation is ever
    performed here.
    """

    expected = {(seed, fold, arm) for seed in EXPECTED_SEEDS for fold in SELECTED_FOLDS for arm in (FULL_ARM, MB4_ARM)}
    expected_bindings = {int(item["fold"]): item for item in _full_fold_bindings()}
    seen: set[tuple[int, int, str]] = set()
    by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        _validate_metric_row(row)
        key = (int(row["seed"]), int(row["fold"]), str(row["arm"]))
        binding = expected_bindings[int(row["fold"])]
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
            for field in ("target_session", "inner_validation_session", "query_windows_evaluated"):
                if full[field] != mb4[field]:
                    raise SpecError(f"paired {field} mismatch at seed={seed} fold={fold}")
            for field in (
                "parameter_count",
                "macs_per_decode_call",
                "cached_state_bytes",
                "initial_state_hash",
                "source_split_manifest_sha256",
                "source_normalizer_sha256",
            ):
                if full[field] != mb4[field]:
                    raise SpecError(f"paired {field} mismatch at seed={seed} fold={fold}")
            delta = float(full["r2_variance_weighted"]) - float(mb4["r2_variance_weighted"])
            deltas_by_seed[seed].append(delta)
            pair_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "target_session": full["target_session"],
                    "inner_validation_session": full["inner_validation_session"],
                    "full_r2": float(full["r2_variance_weighted"]),
                    "mb4_r2": float(mb4["r2_variance_weighted"]),
                    "delta_full_minus_mb4": delta,
                }
            )

    per_seed = [_summarize_seed(seed, deltas_by_seed[seed]) for seed in EXPECTED_SEEDS]
    all_deltas = [delta for values in deltas_by_seed.values() for delta in values]
    both_means_positive = all(item["mean_delta"] > 0 for item in per_seed)
    both_medians_positive = all(item["median_delta"] > 0 for item in per_seed)
    each_seed_four_positive = all(item["sign_counts"]["positive"] >= 4 for item in per_seed)
    label = "ROBUST_DIRECTIONAL" if (both_means_positive and both_medians_positive and each_seed_four_positive) else "MIXED_DIRECTION"
    accounting = {
        "parameter_count": by_key[(EXPECTED_SEEDS[0], SELECTED_FOLDS[0], FULL_ARM)]["parameter_count"],
        "macs_per_decode_call": by_key[(EXPECTED_SEEDS[0], SELECTED_FOLDS[0], FULL_ARM)]["macs_per_decode_call"],
        "cached_state_bytes": by_key[(EXPECTED_SEEDS[0], SELECTED_FOLDS[0], FULL_ARM)]["cached_state_bytes"],
        "paired_equal_all_cells": True,
        "model_state_unchanged_all_cells": True,
        "target_backpropagation_false_all_cells": True,
        "optimizer_absent_all_cells": True,
    }
    return {
        "status": "COMPLETE_DESCRIPTIVE_ONLY",
        "label": label,
        "sampling_unit": "outer_fold_within_seed; seeds are nested training repeats",
        "do_not_pool_seed_fold_as_independent": True,
        "pair_count": len(pair_rows),
        "per_seed": per_seed,
        "cross_seed": {
            "seed_means": [item["mean_delta"] for item in per_seed],
            "seed_medians": [item["median_delta"] for item in per_seed],
            "mean_of_seed_means": statistics.fmean(item["mean_delta"] for item in per_seed),
            "both_seed_mean_positive": both_means_positive,
            "both_seed_median_positive": both_medians_positive,
            "each_seed_at_least_4_of_5_positive": each_seed_four_positive,
            "all_10_sign_counts": _sign_counts(all_deltas),
        },
        "accounting": accounting,
        "pairs": pair_rows,
        "inferential_significance_claim": "none",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-draft", type=Path, help="write a no-GPU draft receipt")
    args = parser.parse_args(argv)
    if args.write_draft is None:
        print(json.dumps(build_draft_receipt(), indent=2, sort_keys=True))
    else:
        _write_json(args.write_draft, build_draft_receipt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
