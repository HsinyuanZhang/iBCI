#!/usr/bin/env python3
"""Aggregate the frozen carrier value-mask forward diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for value in (SUA_ROOT, SUA_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mc_maze import a2_matched_subject_shift_v2_core as a2  # noqa: E402
from mc_maze import carrier_value_mask_core as core  # noqa: E402


class CarrierValueMaskAggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierValueMaskAggregateError(message)


def _mean(row: Mapping[str, Any]) -> float:
    values = [float(value) for value in row["per_session_mean_r2"].values()]
    require(bool(values) and all(math.isfinite(value) for value in values), "invalid mask score values")
    return sum(values) / len(values)


def summarize_domain(scores: Mapping[str, Mapping[str, Any]], parents: Mapping[str, Mapping[str, Any]]):
    require(set(scores) == set(core.INTERVENTIONS), "incomplete carrier-mask lattice")
    require(set(parents) == {"source_t4", "source_z4"}, "incomplete carrier-mask parents")
    means = {name: _mean(row) for name, row in scores.items()}
    t4 = _mean(parents["source_t4"])
    z4 = _mean(parents["source_z4"])
    deltas = {
        "low_t4": means["low_t4"] - t4,
        "random_t4": means["random_t4"] - t4,
        "high_t4": means["high_t4"] - t4,
        "low_z4": means["low_z4"] - z4,
        "random_z4": means["random_z4"] - z4,
    }
    return {
        "cell_mean_r2": {**means, "a2_t4": t4, "a2_z4": z4},
        "deltas_from_matched_parent": deltas,
        "low_mask_carrier_interaction": deltas["low_t4"] - deltas["low_z4"],
        "low_t4_minus_random_t4": means["low_t4"] - means["random_t4"],
        "low_t4_minus_high_t4": means["low_t4"] - means["high_t4"],
        "random_mask_carrier_interaction": deltas["random_t4"] - deltas["random_z4"],
    }


def _require_canonical_output(path: Path) -> None:
    expected = (core.RESULT_ROOT / "terminal_mask_aggregate.json").resolve()
    require(path.resolve() == expected, f"mask aggregate output must use canonical path: {expected}")


def _require_fresh_output(path: Path) -> None:
    require(not os.path.lexists(path) and not os.path.lexists(Path(str(path) + ".sha256")),
            "mask aggregate output exists")


def _validate_engineering_receipt(value: Mapping[str, Any]) -> None:
    shape = value.get("mask_runtime_shape_by_session")
    state = value.get("persistent_mask_state_bytes_by_session")
    require(isinstance(shape, Mapping) and isinstance(state, Mapping), "mask engineering receipt missing")
    require(set(shape) == set(value["per_session_mean_r2"]) == set(state), "mask engineering roster drift")
    for session, row in shape.items():
        require(isinstance(row, Mapping), f"mask engineering row missing: {session}")
        unit_count = int(row.get("unit_count", 0))
        masked = int(row.get("masked_unit_count", 0))
        remaining = int(row.get("remaining_unit_count", 0))
        bytes_per_element = int(row.get("persistent_boolean_mask_element_bytes", 0))
        expected_state = int(row.get("persistent_boolean_mask_state_bytes", -1))
        require(unit_count > 1 and masked > 0 and remaining > 0 and masked + remaining == unit_count,
                f"mask cardinality drift: {session}")
        require(bytes_per_element == 1 and expected_state == unit_count * bytes_per_element,
                f"mask state-byte accounting drift: {session}")
        require(int(state[session]) == expected_state, f"mask state receipt drift: {session}")
    require(value.get("persistent_mask_state_definition") ==
            "one cached torch.bool [N] mask per target session", "mask state-definition drift")
    require(value.get("analytic_dense_mha_mac_delta_current_key_padding_mask_path") == 0,
            "mask MAC accounting drift")
    require(value.get("masked_tokens_are_not_physically_compacted_in_current_pytorch_path") is True,
            "mask compaction accounting drift")
    require(int(value.get("model_parameter_count", 0)) > 0 and value.get("parameter_delta") == 0,
            "mask parameter accounting drift")
    total_seconds = float(value.get("measured_forward_wall_seconds", float("nan")))
    setup_seconds = float(value.get("measured_mask_setup_wall_seconds", float("nan")))
    samples = int(value.get("measured_forward_samples_across_epochs", 0))
    per_sample = float(value.get("measured_forward_seconds_per_sample", float("nan")))
    require(math.isfinite(total_seconds) and total_seconds >= 0 and math.isfinite(setup_seconds) and setup_seconds >= 0,
            "mask latency receipt non-finite")
    require(samples > 0 and math.isfinite(per_sample) and per_sample >= 0,
            "mask latency/sample receipt drift")


def execute(output: Path) -> dict[str, Any]:
    _require_canonical_output(output)
    _require_fresh_output(output)
    gate, gate_sha = core.load_source_gate()
    require(gate.get("gate_passed") is True, "source mask gate did not permit target aggregate")
    scores: dict[str, dict[str, Any]] = {domain: {} for domain in core.DOMAINS}
    parents: dict[str, dict[str, Any]] = {domain: {} for domain in core.DOMAINS}
    receipt_sha: dict[str, str] = {}
    for domain in core.DOMAINS:
        for name, spec in core.INTERVENTIONS.items():
            value, digest = a2.load_verified_immutable_json(
                core.score_path(name, domain), label="carrier value-mask domain score"
            )
            require(value.get("receipt_kind") == "carrier_value_mask_domain_score", "mask score kind drift")
            require(value.get("intervention") == name and value.get("domain") == domain, "mask score cell drift")
            require(value.get("seed") == core.SEED, "mask score seed drift")
            require(value.get("source_gate_sha256") == gate_sha, "mask score source-gate drift")
            require(value.get("implementation_bindings") == gate["implementation_bindings"],
                    "mask score implementation drift")
            _validate_engineering_receipt(value)
            require(value.get("target_optimizer_steps") == 0 and value.get("target_backward_steps") == 0,
                    "mask target update occurred")
            require(value.get("formal_subc_test_nwb_opened") is False, "formal mask data opened")
            source_arm = str(spec["source_arm"])
            parent, parent_sha = a2.load_verified_immutable_json(
                core.a2_receipt_path(source_arm, domain), label="sealed A2 mask parent"
            )
            require(value.get("a2_baseline_receipt_path") ==
                    str(core.a2_receipt_path(source_arm, domain).resolve()), "mask parent path drift")
            require(value.get("a2_baseline_receipt_sha256") == parent_sha, "mask parent SHA drift")
            require(value.get("a2_official_preflight_sha256") == core.EXPECTED_A2_PREFLIGHT_SHA256,
                    "mask A2 preflight anchor drift")
            require(value.get("source_run") == parent.get("source_run"), "mask source-run provenance drift")
            require(value.get("source_checkpoint_sha256_bundle_sha256") ==
                    parent.get("source_checkpoint_sha256_bundle_sha256"), "mask checkpoint bundle drift")
            require(value.get("normalizer_authority") == parent.get("normalizer_authority"),
                    "mask normalizer provenance drift")
            require(value.get("query_policy") == parent.get("query_policy"), "mask query-policy provenance drift")
            require(value.get("session_query_receipts") == parent.get("session_query_receipts"),
                    "mask query receipts differ from sealed A2 parent")
            require(tuple(value["per_session_mean_r2"]) == tuple(parent["per_session_mean_r2"]),
                    "mask session roster/order drift")
            scores[domain][name] = value
            parents[domain][source_arm] = parent
            receipt_sha[f"{domain}_{name}"] = digest
            receipt_sha[f"{domain}_{source_arm}_parent"] = parent_sha
    domains = {domain: summarize_domain(scores[domain], parents[domain]) for domain in core.DOMAINS}
    within = domains["within_subject"]
    external = domains["external_subject_M"]
    checks = {
        "external_low_t4_absolute_delta_positive":
            external["deltas_from_matched_parent"]["low_t4"] > 0,
        "external_low_mask_carrier_interaction_positive":
            external["low_mask_carrier_interaction"] > 0,
        "external_low_t4_beats_random_same_count":
            external["low_t4_minus_random_t4"] > 0,
        "external_low_t4_beats_high_gain_mask":
            external["low_t4_minus_high_t4"] > 0,
        "within_low_t4_delta_at_least_minus_0p03":
            within["deltas_from_matched_parent"]["low_t4"] >= -0.03,
    }
    positive = all(checks.values())
    bindings = core.current_bindings()
    payload = {
        "schema_version": 1,
        "receipt_kind": "carrier_value_mask_terminal_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "seed": core.SEED,
        "source_gate_sha256": gate_sha,
        "implementation_bindings": bindings,
        "implementation_bindings_sha256": core.canonical_sha256(bindings),
        "receipt_sha256": receipt_sha,
        "domains": domains,
        "routing_checks": checks,
        "positive_forward_mask_signal": positive,
        "verdict": (
            "VALUE_WEIGHTED_LOW_GAIN_MASK_POSITIVE__DESIGN_SEPARATE_TRAINED_ROUTE"
            if positive else "VALUE_WEIGHTED_LOW_GAIN_MASK_FLAT_OR_NONSPECIFIC__ADVANCE_QUEUE"
        ),
        "interpretation_boundary": (
            "This is a frozen-forward diagnostic. A flat result closes only the fixed bottom-quartile mask; "
            "a positive result does not by itself establish a trained sparse router."
        ),
        "parameter_delta": 0,
        "persistent_state_is_one_boolean_per_unit_and_reported_by_cell": True,
        "current_dense_mha_mac_delta": 0,
        "formal_subc_test_nwb_opened": False,
    }
    core.write_immutable(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=core.RESULT_ROOT / "terminal_mask_aggregate.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA",
            "required_scores": [
                str(core.score_path(name, domain)) for domain in core.DOMAINS for name in core.INTERVENTIONS
            ],
            "output": str(args.output),
        }, indent=2, sort_keys=True))
        return 0
    payload = execute(args.output)
    print(json.dumps({"status": "COMPLETE", "verdict": payload["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
