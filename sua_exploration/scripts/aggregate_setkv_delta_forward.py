#!/usr/bin/env python3
"""Aggregate the frozen SetKV-delta forward-only diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for value in (SUA_ROOT, SUA_ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mc_maze import setkv_delta_forward_core as core  # noqa: E402


class SetKVAggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SetKVAggregateError(message)


def _mean(values: Mapping[str, float]) -> float:
    require(bool(values), "empty session mapping")
    rows = [float(value) for value in values.values()]
    require(all(math.isfinite(value) for value in rows), "non-finite score")
    return sum(rows) / len(rows)


def summarize_domain(
    score_values: Mapping[str, Mapping[str, Any]],
    baseline_values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the frozen carrier, attachment, and set-size contrasts."""
    require(set(score_values) == set(core.INTERVENTIONS), "incomplete SetKV intervention lattice")
    require(set(baseline_values) == {"source_t4", "source_z4"}, "incomplete A2 parent lattice")
    means = {name: _mean(value["per_session_mean_r2"]) for name, value in score_values.items()}
    base_t4 = _mean(baseline_values["source_t4"]["per_session_mean_r2"])
    base_z4 = _mean(baseline_values["source_z4"]["per_session_mean_r2"])
    delta_t4 = means["setkv_t4"] - base_t4
    delta_z4 = means["setkv_z4"] - base_z4
    delta_rs4 = means["setkv_rs4"] - base_t4
    delta_duplicate = means["duplicate_activity_t4"] - base_t4
    return {
        "cell_mean_r2": {**means, "a2_t4": base_t4, "a2_z4": base_z4},
        "deltas_from_matched_a2_parent": {
            "setkv_t4": delta_t4,
            "setkv_z4": delta_z4,
            "setkv_rs4": delta_rs4,
            "duplicate_activity_t4": delta_duplicate,
        },
        "carrier_specific_interaction": delta_t4 - delta_z4,
        "attachment_contrast_setkv_t4_minus_rs4": means["setkv_t4"] - means["setkv_rs4"],
        "specificity_vs_duplicate_delta": delta_t4 - delta_duplicate,
    }


def _require_canonical_output(path: Path) -> None:
    expected = (core.RESULT_ROOT / "terminal_forward_aggregate.json").resolve()
    require(path.resolve() == expected, f"aggregate output must use canonical path: {expected}")


def execute(output: Path) -> dict[str, Any]:
    _require_canonical_output(output)
    preflight, preflight_sha = core.load_official_preflight()
    scores: dict[str, dict[str, Mapping[str, Any]]] = {domain: {} for domain in core.DOMAINS}
    receipt_sha: dict[str, str] = {}
    baseline: dict[str, dict[str, Mapping[str, Any]]] = {domain: {} for domain in core.DOMAINS}
    for domain in core.DOMAINS:
        for name, spec in core.INTERVENTIONS.items():
            value, digest = core.load_immutable(core.score_path(name, domain), "SetKV domain score")
            require(value.get("receipt_kind") == "setkv_delta_forward_domain_score", "score kind drift")
            require(value.get("intervention") == name and value.get("domain") == domain, "score cell drift")
            require(value.get("seed") == core.SEED, "score seed drift")
            require(value.get("official_preflight_sha256") == preflight_sha, "score preflight drift")
            require(value.get("implementation_bindings") == preflight["implementation_bindings"], "score implementation drift")
            require(value.get("parameter_delta") == 0, "SetKV score added parameters")
            require(value.get("target_optimizer_steps") == 0 and value.get("target_backward_steps") == 0,
                    "target update occurred")
            require(value.get("formal_subc_test_nwb_opened") is False, "formal data opened")
            source_arm = str(spec["source_arm"])
            base, base_sha = core.load_immutable(
                core.a2_baseline_receipt_path(source_arm, domain), "sealed A2 baseline"
            )
            require(value.get("a2_baseline_receipt_path") ==
                    str(core.a2_baseline_receipt_path(source_arm, domain).resolve()),
                    "score A2 baseline path drift")
            require(value.get("a2_baseline_receipt_sha256") == base_sha,
                    "score A2 baseline receipt SHA drift")
            require(value.get("source_run") == base.get("source_run"), "source-run provenance drift")
            require(value.get("query_policy") == base.get("query_policy"), "query-policy provenance drift")
            require(value.get("normalizer_authority") == base.get("normalizer_authority"),
                    "normalizer provenance drift")
            require(value.get("domain_sessions") == base.get("domain_sessions"), "domain-session provenance drift")
            require(value.get("source_checkpoint_sha256_bundle_sha256") ==
                    base.get("source_checkpoint_sha256_bundle_sha256"), "source checkpoint bundle drift")
            require(value.get("session_query_receipts") == base.get("session_query_receipts"),
                    "query receipt differs from A2 baseline")
            require(tuple(value["per_session_mean_r2"]) == tuple(base["per_session_mean_r2"]),
                    "session roster/order drift")
            scores[domain][name] = value
            baseline[domain][source_arm] = base
            receipt_sha[f"{domain}_{name}"] = digest
            receipt_sha[f"{domain}_{source_arm}_baseline"] = base_sha

    domains: dict[str, Any] = {}
    for domain in core.DOMAINS:
        domains[domain] = summarize_domain(scores[domain], baseline[domain])

    external = domains["external_subject_M"]
    within = domains["within_subject"]
    routing_checks = {
        "external_setkv_t4_absolute_delta_positive":
            external["deltas_from_matched_a2_parent"]["setkv_t4"] > 0,
        "external_carrier_specific_interaction_positive":
            external["carrier_specific_interaction"] > 0,
        "external_attachment_contrast_positive":
            external["attachment_contrast_setkv_t4_minus_rs4"] > 0,
        "external_specificity_vs_duplicate_positive":
            external["specificity_vs_duplicate_delta"] > 0,
        "within_setkv_t4_delta_at_least_minus_0p03":
            within["deltas_from_matched_a2_parent"]["setkv_t4"] >= -0.03,
    }
    promoted = all(routing_checks.values())
    payload = {
        "schema_version": 1,
        "receipt_kind": "setkv_delta_forward_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "seed": core.SEED,
        "official_preflight_sha256": preflight_sha,
        "implementation_bindings": preflight["implementation_bindings"],
        "receipt_sha256": receipt_sha,
        "domains": domains,
        "routing_checks": routing_checks,
        "promote_joint_training_design": promoted,
        "verdict": (
            "SETKV_FORWARD_SIGNAL_POSITIVE__DESIGN_JOINT_TRAINING"
            if promoted else
            "SETKV_FORWARD_SIGNAL_INCOMPLETE_OR_FLAT__NO_JOINT_TRAINING_FROM_THIS_DIAGNOSTIC"
        ),
        "interpretation_boundary": (
            "A frozen-consumer forward signal is diagnostic only. A flat result does not prove a jointly "
            "trained SetKV consumer impossible; it advances the queue without tuning this intervention."
        ),
        "parameter_delta": 0,
        "formal_subc_test_nwb_opened": False,
    }
    core.write_immutable(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=core.RESULT_ROOT / "terminal_forward_aggregate.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA",
            "required_scores": [str(core.score_path(name, domain)) for domain in core.DOMAINS for name in core.INTERVENTIONS],
            "output": str(args.output),
        }, indent=2, sort_keys=True))
        return 0
    payload = execute(args.output)
    print(json.dumps({"status": "COMPLETE", "verdict": payload["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
