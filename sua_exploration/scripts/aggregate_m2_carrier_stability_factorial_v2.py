#!/usr/bin/env python3
"""Read-only aggregate-v2 for the M2 held-in carrier factorial raw audit.

It never opens NWBs and never recomputes a fit.  A hash-bound v3 addendum
authorizes one new aggregate artifact while preserving raw_factorial.json and
aggregate.json unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sua_exploration/results/m2_carrier_stability_factorial_v1"
RECEIPT_SHA = "b094e3f7674dd75571947c262d63879639c8b41ffe99f6725b4e1764e1e0e573"
V1_SHA = "f7c2c0a6546113f88a0ce6081c618edf783527ad3bc363d21ba764216af4628f"
V2_SHA = "ca662938a50188f6f48f9dd7b371a9749839908ca82248e0195b53c76740dbe1"
RAW_SHA = "bf82a57659eda92afe627d3dbe575a806366c043a9c96a23879dd7a331da9098"
OLD_AGG_SHA = "ad6676f53ce9240c428a3498cbe86e068b5dfea68790aa0f1bec3098618ce12c"
V3_ADDENDUM_SHA = "0dba67a7d61eccb710d4f20bae4f7cd32661706a35076a7aa9150531e8856ff4"
AGG_V2_SHA = "3d42f8cbefb68d374ea6a8580830ab40c5d358c27531a66c63c9a58d0ed50db8"
METRICS = ("w_flattened_pearson", "w_per_channel_cosine_median", "w_modulation_weighted_cosine")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def strict(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [strict(v) for v in value]
    if isinstance(value, np.ndarray): return strict(value.tolist())
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def median(values: list[float | None]) -> float | None:
    good = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.median(good)) if good else None


def base_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for m in sorted({row["m"] for row in rows}):
        for split in sorted({row["split"] for row in rows}):
            group = [row for row in rows if row["m"] == m and row["split"] == split]
            defined = [row for row in group if row.get("defined")]
            keys = sorted({key for row in defined for key, value in row.items() if isinstance(value, (int, float)) and key not in {"m", "chosen_global_lead_bins"}})
            grouped[f"M={m}|{split}"] = {
                "n_sessions": len(group), "n_defined": len(defined),
                "defined_fraction": len(defined) / len(group) if group else None,
                "session_median": {key: median([row.get(key) for row in defined]) for key in keys},
            }
    return strict(grouped)


def random_accounting(arms: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, int]:
    """Separate coverage-eligible trials from actually matched balanced pairs."""
    coverage_eligible = [(row, arm) for row, arm in arms
                         if arm["balanced_coverage"]["first"].get("defined") and arm["balanced_coverage"]["second"].get("defined")]
    paired = [(row, arm) for row, arm in coverage_eligible
              if arm["balanced_ols"].get("defined") and arm["random_equal_size_ols"].get("defined")]
    fit_failures = [(row, arm) for row, arm in coverage_eligible
                    if not (arm["balanced_ols"].get("defined") and arm["random_equal_size_ols"].get("defined"))]
    return {
        "coverage_eligible_requested": sum(int(arm["random_equal_size_ols"].get("n_requested", 0)) for _row, arm in coverage_eligible),
        "fit_failure_arms": len(fit_failures),
        "matched_pair_arms": len(paired),
        "matched_pair_requested": sum(int(arm["random_equal_size_ols"].get("n_requested", 0)) for _row, arm in paired),
        "matched_pair_valid": sum(int(arm["random_equal_size_ols"].get("n_valid", 0) or 0) for _row, arm in paired),
    }


def factorial_summary(raw: dict[str, Any]) -> dict[str, Any]:
    rows, details = raw["k4_rows"], raw["details"]
    summaries: dict[str, Any] = {}
    for m in sorted({row["m"] for row in rows}):
        for split in sorted({row["split"] for row in rows}):
            selected = [row for row in rows if row["m"] == m and row["split"] == split and row.get("defined")]
            arms = []
            for row in selected:
                detail = details[f"{row['session']}|M={m}"]["splits"][split]["balanced"]
                arms.append((row, detail))
            balanced = [(row, arm["balanced_ols"]) for row, arm in arms if arm["balanced_ols"].get("defined")]
            random = [(row, arm["random_equal_size_ols"]) for row, arm in arms if arm["random_equal_size_ols"].get("defined")]
            paired = [(row, arm["balanced_ols"], arm["random_equal_size_ols"]) for row, arm in arms
                      if arm["balanced_ols"].get("defined") and arm["random_equal_size_ols"].get("defined")]
            all_pairs = [(row, arm["all_block_ols"], arm["all_block_ridge"]) for row, arm in arms
                         if arm["all_block_ols"].get("defined") and arm["all_block_ridge"].get("defined")]
            coverage_undefined = [arm["balanced_coverage"] for _row, arm in arms
                                  if not arm["balanced_coverage"]["first"].get("defined") or not arm["balanced_coverage"]["second"].get("defined")]
            random_fit_failures = [arm["random_equal_size_ols"] for _row, arm in arms
                                  if arm["balanced_coverage"]["first"].get("defined") and arm["balanced_coverage"]["second"].get("defined")
                                  and not arm["random_equal_size_ols"].get("defined")]
            accounting = random_accounting(arms)
            summaries[f"M={m}|{split}"] = {
                "n_k4_defined_sessions": len(selected),
                "balanced": {"n_defined": len(balanced), "defined_fraction_among_k4_defined": len(balanced) / len(selected) if selected else None,
                             "coverage_insufficient_sessions": len(coverage_undefined),
                             "coverage_reasons": coverage_undefined},
                "matched_random_ols": {**accounting, "fit_failure_note": "coverage was adequate but balanced and/or matched-random OLS was not fit-valid, usually because balanced total block count was <10 or fit design was unavailable"},
                "balanced_minus_random_median": {metric: median([
                    left["metrics"][metric] - right["median_metrics"][metric] for _row, left, right in paired
                ]) for metric in METRICS},
                "balanced_minus_random_n_pairs": len(paired),
                "all_block_ridge_minus_ols_median": {metric: median([
                    ridge["metrics"][metric] - ols["metrics"][metric] for _row, ols, ridge in all_pairs
                ]) for metric in METRICS},
                "all_block_ridge_minus_ols_n_pairs": len(all_pairs),
            }
    return strict(summaries)


def write_addendum(out: Path) -> Path:
    destination = out / "protocol_aggregate_addendum_v3.json"
    if destination.exists(): raise FileExistsError("refusing to overwrite v3 aggregate addendum")
    required = {"protocol_receipt.json": RECEIPT_SHA, "protocol_feasibility_addendum_v1.json": V1_SHA,
                "protocol_implementation_fix_addendum_v2.json": V2_SHA, "raw_factorial.json": RAW_SHA, "aggregate.json": OLD_AGG_SHA}
    for filename, expected in required.items():
        if sha256(out / filename) != expected: raise ValueError(f"input SHA drift: {filename}")
    payload = {"schema_version": 1, "status": "frozen_before_aggregate_v2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "authority": "aggregate-only v3: no NWB access and no refitting",
               "bound_inputs_sha256": required, "aggregator_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
               "output": "aggregate_v2.json; aggregate.json and raw_factorial.json remain immutable",
               "new_fields": ["balanced defined fraction / coverage insufficiency", "matched random requested/valid repetitions",
                              "balanced-minus-random paired medians", "all-block ridge-minus-OLS paired medians"]}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); return destination


def write_v4_addendum(out: Path) -> Path:
    """Freeze a wording-only aggregate-v3 correction; raw inputs stay immutable."""
    destination = out / "protocol_aggregate_semantics_addendum_v4.json"
    if destination.exists(): raise FileExistsError("refusing to overwrite v4 semantic addendum")
    required = {"protocol_receipt.json": RECEIPT_SHA, "protocol_feasibility_addendum_v1.json": V1_SHA,
                "protocol_implementation_fix_addendum_v2.json": V2_SHA, "raw_factorial.json": RAW_SHA,
                "aggregate.json": OLD_AGG_SHA, "protocol_aggregate_addendum_v3.json": V3_ADDENDUM_SHA,
                "aggregate_v2.json": AGG_V2_SHA}
    for filename, expected in required.items():
        if sha256(out / filename) != expected: raise ValueError(f"input SHA drift: {filename}")
    payload = {"schema_version": 1, "status": "frozen_before_aggregate_v3", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "authority": "aggregate-only semantic correction; no NWB access and no fit recomputation",
               "bound_inputs_sha256": required, "aggregator_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
               "output": "aggregate_v3.json; raw_factorial.json, aggregate.json, and aggregate_v2.json remain immutable",
               "semantic_correction": "split coverage_eligible_requested from matched_pair_{arms,requested,valid}; balanced-minus-random uses matched pairs only"}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); return destination


def run(out: Path, addendum_sha: str) -> Path:
    addendum = out / "protocol_aggregate_addendum_v3.json"
    if sha256(addendum) != addendum_sha: raise ValueError("v3 addendum SHA mismatch")
    meta = json.loads(addendum.read_text())
    if meta["aggregator_script"]["sha256"] != sha256(Path(__file__)): raise ValueError("aggregator source drift")
    for filename, expected in meta["bound_inputs_sha256"].items():
        if sha256(out / filename) != expected: raise ValueError(f"bound input SHA drift: {filename}")
    destination = out / "aggregate_v2.json"
    if destination.exists(): raise FileExistsError("refusing to overwrite aggregate_v2")
    raw = json.loads((out / "raw_factorial.json").read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    payload = {"schema_version": 2, "scope": "read-only aggregate over held-in-only raw factorial audit; no held-out/query/decoder result",
               "addendum": {"path": str(addendum.resolve()), "sha256": addendum_sha}, "k4_base": base_summary(raw["k4_rows"]),
               "t4_base": base_summary(raw["t4_rows"]), "k4_balance_random_ridge": factorial_summary(raw), "no_posthoc_gate": True}
    destination.write_text(json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return destination


def run_v3(out: Path, addendum_sha: str) -> Path:
    addendum = out / "protocol_aggregate_semantics_addendum_v4.json"
    if sha256(addendum) != addendum_sha: raise ValueError("v4 addendum SHA mismatch")
    meta = json.loads(addendum.read_text())
    if meta["aggregator_script"]["sha256"] != sha256(Path(__file__)): raise ValueError("aggregator source drift")
    for filename, expected in meta["bound_inputs_sha256"].items():
        if sha256(out / filename) != expected: raise ValueError(f"bound input SHA drift: {filename}")
    destination = out / "aggregate_v3.json"
    if destination.exists(): raise FileExistsError("refusing to overwrite aggregate_v3")
    raw = json.loads((out / "raw_factorial.json").read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    payload = {"schema_version": 3, "scope": "read-only aggregate over held-in-only raw factorial audit; no held-out/query/decoder result",
               "addendum": {"path": str(addendum.resolve()), "sha256": addendum_sha}, "k4_base": base_summary(raw["k4_rows"]),
               "t4_base": base_summary(raw["t4_rows"]), "k4_balance_random_ridge": factorial_summary(raw), "no_posthoc_gate": True}
    destination.write_text(json.dumps(strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--write-addendum", action="store_true")
    parser.add_argument("--write-v4-addendum", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-v3", action="store_true")
    parser.add_argument("--addendum-sha")
    args = parser.parse_args(); out = args.out_dir.resolve()
    if sum((args.write_addendum, args.write_v4_addendum, args.run, args.run_v3)) != 1: raise ValueError("choose exactly one operation")
    if args.write_addendum:
        path = write_addendum(out); print(f"{path}\nsha256={sha256(path)}"); return
    if args.write_v4_addendum:
        path = write_v4_addendum(out); print(f"{path}\nsha256={sha256(path)}"); return
    if not args.addendum_sha: raise ValueError("--run requires --addendum-sha")
    print(run_v3(out, args.addendum_sha) if args.run_v3 else run(out, args.addendum_sha))


if __name__ == "__main__": main()
