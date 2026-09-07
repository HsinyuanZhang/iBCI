"""Receipt builders and fail-closed aggregation for B8 pseudo-label gate."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mc_maze.decoder_attention_diagnostic import assert_no_sealed_sessions
from mc_maze.pseudo_label_carrier_gate import (
    DEFAULT_BUDGET_M,
    FROZEN_GATE_PARAMETERS,
    evaluate_pseudo_label_gate,
    synthetic_tuning_rates,
    thetas_from_direction_indices,
)

SCHEMA_VERSION = "pseudo_label_carrier_gate_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_synthetic_session_payload(
    session_name: str,
    *,
    seed: int = 42,
    units: int = 12,
    num_trials: int = 50,
    pseudo_label_mode: str = "correct",
) -> dict[str, Any]:
    """Synthetic session with optional shuffled pseudo-labels."""
    assert_no_sealed_sessions([session_name])
    rng = np.random.Generator(np.random.PCG64(seed))
    direction_indices = rng.integers(0, 8, size=num_trials, dtype=np.int64)
    thetas = thetas_from_direction_indices(direction_indices)
    trial_rates = synthetic_tuning_rates(thetas, units, seed=seed + 1)
    if pseudo_label_mode == "correct":
        pseudo_direction_indices = direction_indices.copy()
    elif pseudo_label_mode == "shuffled":
        pseudo_direction_indices = rng.permutation(direction_indices)
    else:
        raise ValueError(f"unknown pseudo_label_mode {pseudo_label_mode!r}")
    return {
        "session_name": session_name,
        "direction_indices": direction_indices.tolist(),
        "pseudo_direction_indices": pseudo_direction_indices.tolist(),
        "trial_rates": trial_rates.tolist(),
        "units": units,
        "num_trials": num_trials,
    }


def replay_session(
    session_payload: Mapping[str, Any],
    *,
    budget_m: int = DEFAULT_BUDGET_M,
    shuffle_seed: int = 42,
) -> dict[str, Any]:
    session_name = str(session_payload["session_name"])
    assert_no_sealed_sessions([session_name])
    rates = np.asarray(session_payload["trial_rates"], dtype=np.float64)
    true_dirs = np.asarray(session_payload["direction_indices"], dtype=np.int64)
    pseudo_dirs = np.asarray(session_payload["pseudo_direction_indices"], dtype=np.int64)
    result = evaluate_pseudo_label_gate(
        rates,
        true_dirs,
        pseudo_dirs,
        budget_m=budget_m,
        session_name=session_name,
        shuffle_seed=shuffle_seed,
    )
    return result.as_dict()


def build_receipt(
    session_payloads: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    budget_m: int = DEFAULT_BUDGET_M,
    device: str = "cpu",
) -> dict[str, Any]:
    session_names = [str(item["session_name"]) for item in session_payloads]
    assert_no_sealed_sessions(session_names)
    session_results = [
        replay_session(item, budget_m=budget_m, shuffle_seed=seed) for item in session_payloads
    ]
    pooled_correct = [row["median_cosine_correct"] for row in session_results if row["median_cosine_correct"] is not None]
    pooled_shuffle = [row["median_cosine_shuffled"] for row in session_results if row["median_cosine_shuffled"] is not None]
    pooled_delta = [row["correct_minus_shuffle"] for row in session_results if row["correct_minus_shuffle"] is not None]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "device": device,
        "cuda_visible_devices": "",
        "sealed_test_sessions_opened": False,
        "sessions": session_names,
        "resolved_seed": int(seed),
        "budget_m": int(budget_m),
        "frozen_gates": dict(FROZEN_GATE_PARAMETERS),
        "session_results": session_results,
        "pooled_summary": {
            "median_of_session_medians_correct": float(np.median(pooled_correct)) if pooled_correct else None,
            "median_of_session_medians_shuffled": float(np.median(pooled_shuffle)) if pooled_shuffle else None,
            "median_of_session_deltas": float(np.median(pooled_delta)) if pooled_delta else None,
            "sessions_passing_all_gates": int(
                sum(1 for row in session_results if row.get("gates", {}).get("all_predeclared_gates"))
            ),
        },
        "endpoint_scope": "pseudo_label_carrier_constructibility_not_decoder_r2",
        "rls_wired_to_training": False,
        "gpu_authorized": False,
        "training_authorized": False,
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    receipt["generated_at"] = datetime.now(timezone.utc).isoformat()
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    _require(receipt.get("schema_version") == SCHEMA_VERSION, "schema_version drift")
    _require(receipt.get("sealed_test_sessions_opened") is False, "sealed sessions opened")
    _require(receipt.get("device") == "cpu", "non-CPU receipt")
    _require(receipt.get("rls_wired_to_training") is False, "RLS training wiring claimed")
    _require(receipt.get("gpu_authorized") is False, "GPU authorized in receipt")
    gates = receipt.get("frozen_gates") or {}
    for key, value in FROZEN_GATE_PARAMETERS.items():
        _require(gates.get(key) == value, f"frozen gate parameter drift: {key}")
    sessions = receipt.get("sessions") or []
    assert_no_sealed_sessions(sessions)
    session_results = receipt.get("session_results") or []
    _require(len(session_results) == len(sessions), "session_results incomplete")
    for row in session_results:
        _require(row.get("gates") is not None, "missing per-session gates")
    return {
        "status": "valid",
        "sessions": len(sessions),
        "budget_m": receipt.get("budget_m"),
    }


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
