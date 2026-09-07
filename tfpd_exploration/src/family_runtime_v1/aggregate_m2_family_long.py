"""Strict archive-only aggregation of the six M2 family long-run receipts.

This deliberately does not instantiate a decoder or touch a source bank.  It
only accepts the fixed 3 x T1 and 3 x T2 receipt set and refuses to summarize
it when an authority, timing, or oracle invariant has drifted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RECEIPT_DIR = ROOT / "tfpd_exploration/results/family_runtime_v1/container_comparison"
OUTPUT = ROOT / "tfpd_exploration/results/family_runtime_v1/m2_actual_family_long_summary_v2.json"
ARMS = ("FLAT", "ROUTE", "ORIGINALdefault", "ORIGINALdeclared")
MODEL_ARMS = ("FLAT", "ROUTE")
BASELINES = ("ORIGINALdefault", "ORIGINALdeclared")
STATS = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms")
EXPECTED_SCHEMA = "m2_family_actual_spint_vs_selected_public_b7_source_train_v1"
EXPECTED_STATUS = "PASS_PUBLIC_STREAM_TIMING_ONLY"
EXPECTED_IMAGE = "sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f"


def default_receipts() -> list[Path]:
    return [
        RECEIPT_DIR / f"m2_actual_family_spint_b7_t{threads}_2048{repeat}_v1.json"
        for threads in (1, 2)
        for repeat in ("", "_r2", "_r3")
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _triplet(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def _require(receipt: dict[str, Any], path: Path, expected_threads: int, reference: dict[str, Any] | None) -> None:
    label = path.name
    if receipt.get("schema") != EXPECTED_SCHEMA or receipt.get("status") != EXPECTED_STATUS:
        raise ValueError(f"{label}: unexpected schema/status")
    if receipt.get("threads") != expected_threads or receipt.get("batch") != 7:
        raise ValueError(f"{label}: canonical path/thread or batch contract failed")
    if receipt.get("warmup") != 128 or receipt.get("calls") != 2048:
        raise ValueError(f"{label}: expected warmup=128 and calls=2048")
    if receipt.get("interop_threads") != 1 or receipt.get("first_call_excluded_from_warmup_and_steady") is not True:
        raise ValueError(f"{label}: invalid steady-state timing contract")
    for key in ("authority_pre", "authority_post", "immutable_hashes_pre", "immutable_hashes_post",
                "source_train_bank_sha256", "frozen_spint_wrapper_and_payload_sha256"):
        if not isinstance(receipt.get(key), dict) or not receipt[key]:
            raise ValueError(f"{label}: required {key} is missing or empty")
    if receipt.get("authority_pre") != receipt.get("authority_post"):
        raise ValueError(f"{label}: authority changed during measurement")
    if receipt.get("immutable_hashes_pre") != receipt.get("immutable_hashes_post"):
        raise ValueError(f"{label}: immutable hashes changed during measurement")
    if receipt.get("source_surface") != "source_train_continuous_raw_bins_only":
        raise ValueError(f"{label}: wrong source surface")
    if receipt.get("source_train_target_store_hash_bytes_only") is not True or receipt.get("nwb_opened"):
        raise ValueError(f"{label}: timing source-scope contract failed")
    if receipt.get("targets_used_for_timing_or_quality"):
        raise ValueError(f"{label}: targets were used")
    if receipt.get("not_official_latency") is not True or receipt.get("host_not_exclusive") is not True:
        raise ValueError(f"{label}: reporting scope contract failed")
    if receipt.get("historical_spint_image") != EXPECTED_IMAGE:
        raise ValueError(f"{label}: historical image does not match the frozen authority")

    timings = receipt.get("steady_public_call_ms")
    if not isinstance(timings, dict) or set(timings) != set(ARMS):
        raise ValueError(f"{label}: expected exactly the four timing arms")
    for arm in ARMS:
        values = timings[arm]
        if not isinstance(values, dict) or set(values) != set(STATS):
            raise ValueError(f"{label}: malformed stats for {arm}")
        for stat in STATS:
            if _finite_number(values[stat], f"{label}.{arm}.{stat}") <= 0:
                raise ValueError(f"{label}: non-positive {arm}.{stat}")
        if not (values["p50_ms"] <= values["p95_ms"] <= values["p99_ms"] <= values["max_ms"]):
            raise ValueError(f"{label}: invalid percentile order for {arm}")
    oracle = receipt.get("max_public_vs_full_model_oracle_abs_error")
    if not isinstance(oracle, dict) or set(oracle) != set(ARMS):
        raise ValueError(f"{label}: missing oracle checks")
    for arm, value in oracle.items():
        oracle_error = _finite_number(value, f"{label}.oracle.{arm}")
        if oracle_error < 0 or oracle_error > 1e-5:
            raise ValueError(f"{label}: oracle tolerance exceeded for {arm}")

    if reference is not None:
        for key in (
            "immutable_hashes_pre", "authority_pre", "source_train_bank_sha256",
            "frozen_spint_wrapper_and_payload_sha256", "historical_spint_image", "oracle_indices",
        ):
            if receipt.get(key) != reference.get(key):
                raise ValueError(f"{label}: {key} differs from the frozen receipt set")


def aggregate(paths: list[Path]) -> dict[str, Any]:
    expected = default_receipts()
    if paths != expected:
        raise ValueError("only the canonical six archived long-run receipts may be aggregated")
    if len({str(path) for path in paths}) != 6 or any(not path.is_file() for path in paths):
        raise ValueError("all six canonical receipts must exist exactly once")
    self_pre_sha256 = sha256(Path(__file__))
    receipt_pre_sha256 = {path: sha256(path) for path in paths}
    loaded = [(path, json.loads(path.read_text())) for path in paths]
    reference = loaded[0][1]
    for index, (path, receipt) in enumerate(loaded):
        _require(receipt, path, 1 if index < 3 else 2, reference)

    by_threads: dict[str, list[tuple[Path, dict[str, Any]]]] = {"T1": [], "T2": []}
    for pair in loaded:
        by_threads[f"T{pair[1]['threads']}"] += [pair]
    if {key: len(value) for key, value in by_threads.items()} != {"T1": 3, "T2": 3}:
        raise ValueError("expected exactly three repetitions at each thread count")

    thread_summary: dict[str, Any] = {}
    paired_summary: dict[str, Any] = {}
    per_repeat: dict[str, list[dict[str, Any]]] = {}
    for thread, repetitions in by_threads.items():
        timing_summary = {
            arm: {stat: _triplet([float(r["steady_public_call_ms"][arm][stat]) for _, r in repetitions]) for stat in STATS}
            for arm in ARMS
        }
        pair_summary: dict[str, Any] = {}
        for arm in MODEL_ARMS:
            pair_summary[arm] = {}
            for baseline in BASELINES:
                ratios = [
                    float(r["steady_public_call_ms"][arm]["p95_ms"]) /
                    float(r["steady_public_call_ms"][baseline]["p95_ms"])
                    for _, r in repetitions
                ]
                pair_summary[arm][f"p95_ratio_to_{baseline}"] = _triplet(ratios)
        thread_summary[thread] = {"repetitions": len(repetitions), "steady_public_call_ms": timing_summary}
        paired_summary[thread] = pair_summary
        per_repeat[thread] = []
        for repetition, (path, receipt) in enumerate(repetitions, start=1):
            p95 = {arm: float(receipt["steady_public_call_ms"][arm]["p95_ms"]) for arm in ARMS}
            per_repeat[thread].append({
                "repeat": repetition,
                "receipt_path": str(path.relative_to(ROOT)),
                "p95_ms_by_arm": p95,
                "paired_p95_ratios": {
                    arm: {f"to_{baseline}": p95[arm] / p95[baseline] for baseline in BASELINES}
                    for arm in MODEL_ARMS
                },
            })

    receipt_post_sha256 = {path: sha256(path) for path in paths}
    self_post_sha256 = sha256(Path(__file__))
    if receipt_pre_sha256 != receipt_post_sha256:
        raise ValueError("a sealed input receipt changed during aggregation")
    if self_pre_sha256 != self_post_sha256:
        raise ValueError("the aggregator source changed during aggregation")
    oracle_max = {
        arm: max(float(receipt["max_public_vs_full_model_oracle_abs_error"][arm]) for _, receipt in loaded)
        for arm in ARMS
    }

    return {
        "schema": "m2_actual_family_long_summary_v2",
        "status": "PASS_ARCHIVE_ONLY_SEALED_AGGREGATION",
        "scope": {
            "shared_host_public_predict_only": True,
            "not_official_latency": True,
            "not_cold_start_measurement": True,
            "host_not_exclusive": True,
            "not_quality_generalization_or_selection_claim": True,
            "source_surface": "source_train_continuous_raw_bins_only",
            "targets_used_for_timing_or_quality": False,
            "raw_timing_samples_available_in_receipts": False,
            "limitation": "This aggregates receipt-reported timing statistics and declared calls; raw samples are absent, so their count and quantiles cannot be independently recomputed.",
        },
        "protocol": {"batch": 7, "warmup": 128, "steady_calls_per_receipt": 2048, "threads": {"T1": 1, "T2": 2}},
        "receipt_set": [
            {"path": str(path.relative_to(ROOT)), "pre_sha256": receipt_pre_sha256[path], "post_sha256": receipt_post_sha256[path], "threads": receipt["threads"], "declared_steady_calls_per_arm": receipt["calls"]}
            for path, receipt in loaded
        ],
        "aggregation_seal": {"aggregator_pre_sha256": self_pre_sha256, "aggregator_post_sha256": self_post_sha256},
        "frozen_authority": {
            "immutable_hashes": reference["immutable_hashes_pre"],
            "authority": reference["authority_pre"],
            "source_train_bank_sha256": reference["source_train_bank_sha256"],
            "frozen_spint_wrapper_and_payload_sha256": reference["frozen_spint_wrapper_and_payload_sha256"],
            "historical_spint_image": reference["historical_spint_image"],
            "oracle_indices": reference["oracle_indices"],
            "oracle_max_abs_error_by_arm_across_all_six_receipts": oracle_max,
        },
        "steady_public_call_ms_by_threads": thread_summary,
        "per_repeat_p95_by_threads": per_repeat,
        "paired_p95_ratio_by_threads": paired_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite an existing result: {args.output}")
    result = aggregate(default_receipts())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
