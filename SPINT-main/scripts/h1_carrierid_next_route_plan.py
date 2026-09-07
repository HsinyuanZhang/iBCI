#!/usr/bin/env python3
"""Static/no-data plan for the H1 CarrierID post-distribution routes.

This intentionally has no training, dataset, checkpoint, target, CUDA, or
launch path.  It makes the next H1 route names and parameter/control contracts
machine-checkable while the source-only LODO implementation is still absent.
"""
from __future__ import annotations

import json
from typing import Any


SCHEMA = "h1_carrierid_next_route_static_plan_v1"
SPINT_ID_PARAMETERS = 5_965_500
BASE_NON_ID_WHOLE_MODEL_PARAMETERS = 10_889_696
TRIAL_LENGTH = 1_024
CARRIER_DIM = 4
TOKEN_WIDTH = 700


def _nonnegative_int(value: int, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def topology_parameters(*, hidden_dim: int, interface_dim: int) -> dict[str, int]:
    """Return exact affine-layer parameters for the declared CarrierID topology.

    ``hidden_dim`` is the neural pre-pool and final carrier-token width;
    ``interface_dim`` is the width of the one joint pooled-neural/carrier
    layer.  Biases are included in all values.
    """

    h = _nonnegative_int(hidden_dim, "hidden_dim")
    interface = _nonnegative_int(interface_dim, "interface_dim")
    pre_pool = (TRIAL_LENGTH + 1) * h
    first_post = (h + CARRIER_DIM + 1) * interface
    second_post = (interface + 1) * h
    output_post = (h + 1) * TOKEN_WIDTH
    post_pool = first_post + second_post + output_post
    return {
        "hidden_dim": h,
        "interface_dim": interface,
        "pre_pool": pre_pool,
        "first_post": first_post,
        "second_post": second_post,
        "output_post": output_post,
        "post_pool": post_pool,
        "identity_encoder": pre_pool + post_pool,
        "whole_model": BASE_NON_ID_WHOLE_MODEL_PARAMETERS + pre_pool + post_pool,
        "carrier_entry_weights": CARRIER_DIM * interface,
    }


def _topology(name: str, *, hidden_dim: int, interface_dim: int, role: str) -> dict[str, Any]:
    counts = topology_parameters(hidden_dim=hidden_dim, interface_dim=interface_dim)
    return {
        "name": name,
        "role": role,
        "topology": (
            f"pooled[{counts['hidden_dim']}] + carrier[{CARRIER_DIM}] -> "
            f"{counts['hidden_dim'] + CARRIER_DIM} -> {counts['interface_dim']} -> "
            f"{counts['hidden_dim']} -> {TOKEN_WIDTH}"
        ),
        "parameters": {
            **counts,
            "spint_identity_parameter_ratio": SPINT_ID_PARAMETERS / counts["identity_encoder"],
        },
        "requires_fresh_source_training": True,
        "checkpoint_warm_start_forbidden": True,
        "carrier_columns_zero_at_initialization": True,
    }


def build_static_plan() -> dict[str, Any]:
    """Build the immutable-in-spirit plan without touching any experiment state."""

    ci32 = _topology("H1-CI32", hidden_dim=32, interface_dim=32, role="current carrier reference")
    ci64 = _topology("H1-CI64", hidden_dim=32, interface_dim=64, role="interface-only consumer probe")
    h64 = _topology("H1-H64", hidden_dim=64, interface_dim=64, role="conditional full-width escalation")
    return {
        "schema": SCHEMA,
        "mode": "static_plan_only_no_data_no_launch",
        "scope": {
            "source_nwb_opened": 0,
            "target_nwb_opened": 0,
            "minival_or_formal_or_evalai_opened": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed": False,
            "checkpoint_created_or_selected": False,
        },
        "naming": {
            "retired_ambiguous_label": "C7",
            "forbidden_h1_names": ["C7"],
            "canonical": {
                "estimator": "H1-EST4-SLODO",
                "consumer_interface": "H1-CI64-SLODO",
                "full_width_escalation": "H1-H64-SLODO",
                "unresolved_replication": "H1-DIST-XDATE-REPL",
            },
        },
        "d_s4e_d_q4e_hypothesis_routes": {
            "estimator-limited evidence": {
                "next_preparation": "H1-EST4-SLODO",
                "selection_boundary": "independent source-only estimator screen; D-Q4e does not select width/epoch/checkpoint",
            },
            "consumer-limited evidence": {
                "next_preparation": "H1-CI64-SLODO",
                "selection_boundary": "independent source-only CI64 mechanism screen; H64 remains prohibited until its gate passes",
            },
            "unresolved": {
                "next_preparation": "H1-DIST-XDATE-REPL",
                "selection_boundary": "fresh predeclared cross-date replication; no estimator/consumer architecture selection",
            },
            "invalid_exposure_repair__q4e_minus_s4e_not_interpretable": {
                "next_preparation": "H1-DIST-REPAIR-ONLY",
                "selection_boundary": "no architecture experiment is eligible",
            },
        },
        "consumer_topologies": {"H1-CI32": ci32, "H1-CI64": ci64, "H1-H64": h64},
        "controls_required_per_new_topology": ["full", "C0", "LS", "RS"],
        "source_only_selection_prerequisite": {
            "status": "MISSING_IMPLEMENTATION_FAIL_CLOSED",
            "must_open_only": "the 11 existing fold-0 source recordings, never date 19250101",
            "source_dates": ["19250108", "19250113", "19250115", "19250119", "19250120"],
            "required_evaluation": "five source-date LODO folds; strict post-support windows of the held source date",
            "required_provenance": [
                "source_file_hashes", "carrier_schedule_hash", "normalizer_hash", "initial_state_hash",
                "fixed_epoch_checkpoint_hash", "source_date_partition",
            ],
            "forbidden": ["target score selection", "minival", "formal heldout", "EvalAI", "checkpoint warm start"],
        },
        "h64_escalation_gate": {
            "run_first": ["H1-CI32", "H1-CI64"],
            "all_required": [
                "mean_date_r2(CI64-full - CI32-full) > 0 and at_least_4_of_5_dates_positive",
                "mean_date_r2(CI64-full - CI64-C0) > 0 and at_least_4_of_5_dates_positive",
                "mean_date_r2(CI64-full - CI64-LS) > 0 and at_least_4_of_5_dates_positive",
            ],
            "if_pass": "H1-H64-SLODO is eligible as one fixed full-width escalation",
            "if_fail": "STOP_CONSUMER_WIDTH_ROUTE_NO_H64",
        },
    }


def main() -> None:
    print(json.dumps(build_static_plan(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
