#!/usr/bin/env python3
"""Guarded source-only CPU audit for causal T4@M cross-budget readiness.

The default modes are no-data readiness/manifest validation.  The real audit additionally needs
both an explicit command flag and an exact reviewed environment guard.  It resolves only the 27
frozen training and 6 frozen development-validation NWBs by explicit manifest names; formal-test
names stay sealed receipt strings and are never resolved, opened, or discovered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from mc_maze.t4_cross_budget_features import (  # noqa: E402
    T4_CROSS_BUDGET_CACHE_NAMESPACE,
    T4_CROSS_BUDGET_FEATURE_VERSION,
)
from mc_maze.t4_cross_budget_protocol import (  # noqa: E402
    DEFAULT_T4_BUDGETS,
    RELIABILITY_FEATURE_NAMES,
    T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION,
)
from mc_maze.t4_cross_budget_audit import load_frozen_source_development_manifest  # noqa: E402
from mc_maze.t4_cross_budget_audit import (  # noqa: E402
    CrossBudgetSessionAudit,
    descriptor_error_to_t50,
    deterministic_nonidentity_row_shuffle,
    fit_source_q_error_models,
    paired_mde,
    pool_trial_count_matrix_from_receipt,
    score_target_free_q_error_models,
    split_half_t4_repeatability,
    nested_source_q_error_audit,
)
from mc_maze.t4_cross_budget_features import load_or_compute_cross_budget_features  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dry_run_receipt() -> dict:
    """Static evidence that this readiness entrypoint cannot open data or use a GPU."""
    protocol = SUA / "mc_maze" / "t4_cross_budget_protocol.py"
    features = SUA / "mc_maze" / "t4_cross_budget_features.py"
    return {
        "schema_version": "sua_t4_cross_budget_step2a_dry_run_v1",
        "mode": "dry_run_only",
        "no_dataset_opened": True,
        "no_formal_test_opened": True,
        "no_gpu_launch": True,
        "budgets": list(DEFAULT_T4_BUDGETS),
        "evaluation_start_trial": 50,
        "cache_namespace": T4_CROSS_BUDGET_CACHE_NAMESPACE,
        "feature_version": T4_CROSS_BUDGET_FEATURE_VERSION,
        "fit_semantics_version": T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION,
        "reliability_feature_names": list(RELIABILITY_FEATURE_NAMES),
        "repeatability_protocol": {
            "raw_counts": "source-only RAM extraction from compact first-50 receipt; never cached",
            "replicates": 8, "seeds": [1201, 1207, 1213, 1217, 1223, 1229, 1231, 1237],
            "split": "disjoint two-way multinomial per unit/trial, rates rescaled by two",
            "undefined_rule": "rank-deficient partition is reported undefined and never zero-imputed",
        },
        "contracts": {
            "actual_chronological_prefix_refits": True,
            "rank_deficient_prefix_is_undefined": True,
            "within_trial_thinning_preserves_trial_labels": True,
            "formal_test_paths_accepted": False,
            "shared_training_or_active_residual_files_modified": False,
        },
        "touchpoints": {
            "protocol": {"path": str(protocol), "sha256": sha256(protocol)},
            "features": {"path": str(features), "sha256": sha256(features)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
    }


def strict_json(value: Any) -> Any:
    """JSON conversion that prevents NaN/Inf and raw ndarrays escaping a receipt."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def resolve_development_paths(manifest_receipt: dict, data_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Resolve only explicitly frozen development names; never glob/discover formal names."""
    base = Path(data_dir).resolve() / "sub-C"
    if not base.is_dir():
        raise FileNotFoundError(f"expected SUA subject directory: {base}")
    def resolve(names: tuple[str, ...]) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for name in names:
            path = (base / f"{name}_behavior+ecephys.nwb").resolve()
            if path.parent != base or not path.is_file():
                raise FileNotFoundError(f"frozen development session is missing: {path}")
            result[name] = path
        return result
    return (
        resolve(tuple(manifest_receipt["nested_source_session_names"])),
        resolve(tuple(manifest_receipt["development_validation_names"])),
    )


def _equal_session_summary(values_by_session: dict[str, float | None]) -> dict:
    finite = {name: float(value) for name, value in values_by_session.items() if value is not None and math.isfinite(float(value))}
    values = list(finite.values())
    return {
        "aggregation": "equal_session_mean", "defined_session_count": len(values),
        "undefined_session_count": len(values_by_session) - len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "median": (float(np.median(values)) if values else None),
        "paired_mde_80pct_two_sided_5pct_vs_t450_zero": paired_mde(values),
    }


def _session_result(name: str, path: Path, *, cache_dir: Path) -> tuple[CrossBudgetSessionAudit, dict]:
    """Compute one source/dev session, retaining counts only while repeatability is calculated."""
    bundle, cache_path, cache_hit = load_or_compute_cross_budget_features(
        path, cache_dir=cache_dir, budgets=DEFAULT_T4_BUDGETS, signal_view="sua",
    )
    durations = bundle.trial_stop_times - bundle.trial_start_times
    counts = pool_trial_count_matrix_from_receipt(
        path, trial_start_times=bundle.trial_start_times, trial_stop_times=bundle.trial_stop_times,
        signal_view=bundle.signal_view,
    )
    if counts.shape[0] != bundle.fits[50].t4.shape[0] or counts.shape[1] != 50:
        raise ValueError(f"{name}: count/descriptor receipt shape mismatch")
    repeatability = {
        str(budget): split_half_t4_repeatability(
            counts, durations, bundle.trial_direction_indices, budget=budget
        ) for budget in DEFAULT_T4_BUDGETS
    }
    reference = bundle.fits[50]
    descriptor_error: dict[str, dict] = {}
    for budget in DEFAULT_T4_BUDGETS:
        fit = bundle.fits[budget]
        if not (fit.fit_defined and reference.fit_defined):
            descriptor_error[str(budget)] = {"status": "undefined_rank"}
            continue
        error = descriptor_error_to_t50(fit, reference)
        descriptor_error[str(budget)] = {
            "status": "defined", "equal_unit_mean_squared_error": float(error.mean()),
            "median_unit_squared_error": float(np.median(error)),
        }
    attachment = {}
    for budget in (10, 15, 20):
        fit = bundle.fits[budget]
        if not fit.fit_defined or fit.t4.shape[0] < 2:
            attachment[str(budget)] = {"status": "undefined_or_singleton"}
            continue
        shuffled, permutation = deterministic_nonidentity_row_shuffle(fit.t4, session_name=name, seed=42)
        attachment[str(budget)] = {
            "status": "diagnostic_only", "permutation_nonidentity": bool((permutation != np.arange(permutation.size)).any()),
            "column_marginals_preserved": bool(np.array_equal(np.sort(shuffled, axis=0), np.sort(fit.t4, axis=0))),
            "permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
        }
    # `counts` goes out of scope on return and is never placed in the result/cache.
    session_result = {
        "name": name, "source_path": str(path.resolve()), "source_fingerprint": bundle.cache_payload["source"],
        "source_sha256": sha256(path),
        "cache_path": str(cache_path.resolve()), "cache_sha256": sha256(cache_path), "cache_hit": cache_hit,
        "cache_key": bundle.cache_key, "chronology_receipt": {
            "trial_count": int(bundle.trial_ordinals.size), "ordinals": bundle.trial_ordinals,
            "start_times": bundle.trial_start_times, "stop_times": bundle.trial_stop_times,
            "raw_target_dirs_rad": bundle.trial_target_dirs_rad, "snapped_direction_indices": bundle.trial_direction_indices,
        },
        "fits": {str(budget): {
            "fit_defined": fit.fit_defined, "design_rank": fit.design_rank, "design_condition": fit.design_condition,
            "direction_counts": fit.direction_counts, "direction_balance": fit.direction_balance,
            "unit_count": int(fit.t4.shape[0]), "finite_descriptor_count": int(np.isfinite(fit.t4).all(axis=1).sum()),
        } for budget, fit in bundle.fits.items()},
        "descriptor_error_to_t450": descriptor_error, "repeatability": repeatability,
        "attachment_row_shuffle_diagnostic_only": attachment,
    }
    return CrossBudgetSessionAudit(name, bundle.fits), strict_json(session_result)


def execute_source_audit(*, manifest_path: Path, data_dir: Path, cache_dir: Path, output_dir: Path) -> dict:
    """Run the reviewed CPU-only audit.  Caller must enforce explicit authorization first."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory: {output_dir}")
    manifest = load_frozen_source_development_manifest(manifest_path)
    train_paths, validation_paths = resolve_development_paths(manifest, data_dir)
    cache_dir = Path(cache_dir).resolve()
    train_sessions: list[CrossBudgetSessionAudit] = []; validation_sessions: list[CrossBudgetSessionAudit] = []
    train_receipts: dict[str, dict] = {}; validation_receipts: dict[str, dict] = {}
    for name, path in train_paths.items():
        session, receipt = _session_result(name, path, cache_dir=cache_dir); train_sessions.append(session); train_receipts[name] = receipt
    for name, path in validation_paths.items():
        session, receipt = _session_result(name, path, cache_dir=cache_dir); validation_sessions.append(session); validation_receipts[name] = receipt
    nested = nested_source_q_error_audit(train_sessions, budgets=DEFAULT_T4_BUDGETS)
    models = fit_source_q_error_models(train_sessions, budgets=DEFAULT_T4_BUDGETS)
    validation_q = {}
    for session in validation_sessions:
        try:
            validation_q[session.name] = score_target_free_q_error_models(models, session, budgets=DEFAULT_T4_BUDGETS)
        except ValueError:
            validation_q[session.name] = {"status": "undefined_no_valid_t4"}
    def per_budget_error(receipts: dict[str, dict], budget: int) -> dict:
        return _equal_session_summary({name: (row["descriptor_error_to_t450"][str(budget)].get("equal_unit_mean_squared_error") if row["descriptor_error_to_t450"][str(budget)]["status"] == "defined" else None) for name, row in receipts.items()})
    output = {
        "schema_version": "sua_t4_cross_budget_source_audit_v1", "mode": "executed_source_only_cpu_audit",
        "no_gpu_launch": True, "no_formal_test_opened": True, "formal_test_paths_resolved": False,
        "manifest": manifest, "budgets": list(DEFAULT_T4_BUDGETS), "evaluation_start_trial": 50,
        "activity_calibration_trials": 30, "reliability_feature_names": list(RELIABILITY_FEATURE_NAMES),
        "source_train_sessions": train_receipts, "development_validation_sessions": validation_receipts,
        "nested_source_loso_q_error": nested, "target_free_validation_q_error": validation_q,
        "equal_session_descriptor_error_to_t450": {str(budget): per_budget_error(train_receipts, budget) for budget in DEFAULT_T4_BUDGETS},
        "input_hashes": {"manifest_sha256": manifest["manifest_sha256"], "script_sha256": sha256(Path(__file__).resolve())},
        "contains_raw_counts": False, "decoder_r2_claim": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "source_audit.json"
    result_path.write_text(json.dumps(strict_json(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="emit a no-data readiness receipt")
    parser.add_argument("--validate-source-manifest", type=Path, help="validate frozen manifest only; resolve no NWB paths")
    parser.add_argument("--execute-source-audit", action="store_true", help="run guarded real CPU audit on frozen train/validation names only")
    parser.add_argument("--manifest", type=Path, help="frozen manifest required by --execute-source-audit")
    parser.add_argument("--data-dir", type=Path, help="DANDI 000688 data root required by --execute-source-audit")
    parser.add_argument("--cache-dir", type=Path, help="isolated cache root required by --execute-source-audit")
    parser.add_argument("--output-dir", type=Path, help="fresh result directory required by --execute-source-audit")
    parser.add_argument("--output", type=Path, help="optional fresh JSON receipt path")
    args = parser.parse_args()
    mode_count = int(args.dry_run) + int(args.validate_source_manifest is not None) + int(args.execute_source_audit)
    if mode_count != 1:
        parser.error("choose exactly one mode: --dry-run, --validate-source-manifest, or --execute-source-audit")
    if args.execute_source_audit:
        if os.environ.get("STEP2A_REVIEWED_SOURCE_AUDIT") != "YES":
            parser.error("--execute-source-audit requires STEP2A_REVIEWED_SOURCE_AUDIT=YES")
        required = {"--manifest": args.manifest, "--data-dir": args.data_dir, "--cache-dir": args.cache_dir, "--output-dir": args.output_dir}
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            parser.error(f"--execute-source-audit requires {' '.join(missing)}")
        result = execute_source_audit(
            manifest_path=args.manifest, data_dir=args.data_dir, cache_dir=args.cache_dir, output_dir=args.output_dir,
        )
        print(json.dumps(strict_json(result), indent=2, sort_keys=True))
        return
    receipt = build_dry_run_receipt()
    if args.validate_source_manifest is not None:
        receipt["source_development_manifest"] = load_frozen_source_development_manifest(
            args.validate_source_manifest
        )
        receipt["mode"] = "manifest_validation_only"
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
