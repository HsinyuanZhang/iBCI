#!/usr/bin/env python3
"""Terminal three-seed aggregate for the frozen SUA pseudo-session experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from preflight_cebra_pseudosession_sua import write_immutable_pair  # noqa: E402
from score_cebra_pseudosession_sua import RESULT_ROOT, SCREEN_ID, result_path  # noqa: E402


A2_ROOT = SUA_ROOT / "results" / "a2_matched_subject_shift_v2"
STAGE_P = RESULT_ROOT / "stage_p_seed42_aggregate.json"
SEEDS = (42, 43, 44)
DOMAINS = ("within_subject", "external_subject_M")
ARMS = ("t4", "z4")
BOOTSTRAP_SEED = 20260814
BOOTSTRAP_DRAWS = 20_000


class TerminalAggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalAggregateError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_matches(path: Path, digest: str, text: str) -> bool:
    """Accept the two immutable sidecar formats used by this experiment lineage."""
    return text in {
        digest,
        f"{digest}\n",
        f"{digest}  {path.name}",
        f"{digest}  {path.name}\n",
    }


def load_pair(path: Path, expected_kind: str | None = None) -> tuple[dict[str, Any], str]:
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"missing immutable pair: {path}")
    digest = sha256_file(path)
    require(sidecar_matches(path, digest, sidecar.read_text(encoding="ascii")),
            f"sidecar drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"receipt is not an object: {path}")
    if expected_kind is not None:
        require(payload.get("receipt_kind") == expected_kind, f"receipt kind drift: {path}")
    return payload, digest


def parent_path(domain: str, arm: str, seed: int) -> Path:
    return A2_ROOT / f"{domain}_source_{arm}_s{seed}.json"


def schedule_authority_path(seed: int) -> Path:
    return RESULT_ROOT / f"stage_f_seed{seed}_source_schedule_preflight.json"


def _mean(rows: Mapping[str, float]) -> float:
    require(bool(rows), "empty session mapping")
    values = [float(value) for value in rows.values()]
    require(all(math.isfinite(value) for value in values), "non-finite session value")
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    require(bool(values), "empty median")
    ordered = sorted(float(value) for value in values)
    size = len(ordered)
    return ordered[size // 2] if size % 2 else 0.5 * (
        ordered[size // 2 - 1] + ordered[size // 2]
    )


def _percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def crossed_seed_session_bootstrap(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    require(matrix.ndim == 2 and matrix.shape[0] == 3 and matrix.shape[1] > 0,
            "bootstrap matrix must be [3, sessions]")
    require(np.isfinite(matrix).all(), "bootstrap matrix is non-finite")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(BOOTSTRAP_DRAWS):
        seed_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        session_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = matrix[np.ix_(seed_indices, session_indices)].mean()
    return _percentile_interval(draws)


def compute_terminal(
    parent: Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]],
    mixed: Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> dict[str, Any]:
    seed_deltas: dict[str, dict[str, float]] = {}
    per_session: dict[int, dict[str, dict[str, float]]] = {}
    external_sessions: tuple[str, ...] | None = None
    within_sessions: tuple[str, ...] | None = None
    external_t4_matrix: list[list[float]] = []
    external_interaction_matrix: list[list[float]] = []

    for seed in SEEDS:
        require(seed in parent and seed in mixed, f"missing seed {seed}")
        per_session[seed] = {}
        deltas: dict[str, float] = {}
        external_arm_delta: dict[str, dict[str, float]] = {}
        for domain in DOMAINS:
            for arm in ARMS:
                old_rows = parent[seed][domain][arm]
                new_rows = mixed[seed][domain][arm]
                require(tuple(old_rows) == tuple(new_rows), f"seed{seed}/{domain}/{arm}: roster drift")
                roster = tuple(old_rows)
                if domain == "external_subject_M":
                    if external_sessions is None:
                        external_sessions = roster
                    require(roster == external_sessions, "external roster drift across seeds/arms")
                else:
                    if within_sessions is None:
                        within_sessions = roster
                    require(roster == within_sessions, "within roster drift across seeds/arms")
                rows = {
                    session: float(new_rows[session]) - float(old_rows[session])
                    for session in roster
                }
                require(all(math.isfinite(value) for value in rows.values()), "non-finite delta")
                key = f"{domain}_{arm}"
                per_session[seed][key] = rows
                deltas[key] = _mean(rows)
                if domain == "external_subject_M":
                    external_arm_delta[arm] = rows
        deltas["external_carrier_interaction"] = (
            deltas["external_subject_M_t4"] - deltas["external_subject_M_z4"]
        )
        deltas["within_carrier_interaction"] = (
            deltas["within_subject_t4"] - deltas["within_subject_z4"]
        )
        seed_deltas[str(seed)] = deltas
        external_t4_matrix.append([
            per_session[seed]["external_subject_M_t4"][session]
            for session in external_sessions or ()
        ])
        external_interaction_matrix.append([
            external_arm_delta["t4"][session] - external_arm_delta["z4"][session]
            for session in external_sessions or ()
        ])

    ext_t4 = [seed_deltas[str(seed)]["external_subject_M_t4"] for seed in SEEDS]
    within_t4 = [seed_deltas[str(seed)]["within_subject_t4"] for seed in SEEDS]
    ext_interaction = [seed_deltas[str(seed)]["external_carrier_interaction"] for seed in SEEDS]
    ext_t4_matrix_np = np.asarray(external_t4_matrix, dtype=np.float64)
    ext_interaction_matrix_np = np.asarray(external_interaction_matrix, dtype=np.float64)
    session_mean_t4 = {
        session: float(ext_t4_matrix_np[:, index].mean())
        for index, session in enumerate(external_sessions or ())
    }
    session_mean_interaction = {
        session: float(ext_interaction_matrix_np[:, index].mean())
        for index, session in enumerate(external_sessions or ())
    }
    mean_ext_t4 = float(np.mean(ext_t4))
    mean_within_t4 = float(np.mean(within_t4))
    mean_ext_interaction = float(np.mean(ext_interaction))
    accuracy_gates = {
        "mean_external_t4_delta_at_least_0p03": mean_ext_t4 >= 0.03,
        "all_three_external_t4_seed_deltas_positive": all(value > 0 for value in ext_t4),
        "external_session_median_positive": _median(list(session_mean_t4.values())) > 0,
        "at_least_10_of_15_external_sessions_positive":
            sum(value > 0 for value in session_mean_t4.values()) >= 10,
        "mean_within_t4_delta_at_least_minus_0p03": mean_within_t4 >= -0.03,
    }
    accuracy_effective = all(accuracy_gates.values())
    carrier_gates = {
        "accuracy_effective": accuracy_effective,
        "mean_external_interaction_at_least_0p03": mean_ext_interaction >= 0.03,
        "all_three_external_seed_interactions_positive": all(value > 0 for value in ext_interaction),
    }
    carrier_specific = all(carrier_gates.values())
    if carrier_specific:
        verdict = "TERMINAL_CARRIER_SPECIFIC_PSEUDOSESSION_EFFECTIVE"
    elif accuracy_effective:
        verdict = "TERMINAL_GENERIC_PSEUDOSESSION_AUGMENTATION_EFFECTIVE"
    else:
        verdict = "TERMINAL_PSEUDOSESSION_NEGATIVE__STOP_ROUTE"
    return {
        "per_seed_deltas": seed_deltas,
        "external_session_seed_averaged_t4_delta": session_mean_t4,
        "external_session_seed_averaged_interaction": session_mean_interaction,
        "summary": {
            "mean_external_t4_delta": mean_ext_t4,
            "mean_within_t4_delta": mean_within_t4,
            "mean_external_carrier_interaction": mean_ext_interaction,
            "external_t4_session_median": _median(list(session_mean_t4.values())),
            "external_t4_positive_session_count": sum(value > 0 for value in session_mean_t4.values()),
            "external_session_count": len(session_mean_t4),
            "external_t4_crossed_seed_session_bootstrap_95ci":
                crossed_seed_session_bootstrap(ext_t4_matrix_np),
            "external_interaction_crossed_seed_session_bootstrap_95ci":
                crossed_seed_session_bootstrap(ext_interaction_matrix_np),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "accuracy_gates": accuracy_gates,
        "carrier_specific_gates": carrier_gates,
        "accuracy_effective": accuracy_effective,
        "carrier_specific": carrier_specific,
        "verdict": verdict,
        "interpretation": (
            "absolute_external_T4_lift_with_carrier_specific_interaction"
            if carrier_specific else
            "absolute_external_T4_lift_shared_with_Z4__generic_augmentation"
            if accuracy_effective else
            "frozen_accuracy_gate_failed__no_more_pseudo_session_tuning"
        ),
    }


def execute(output: Path) -> dict[str, Any]:
    stage_p, stage_p_sha = load_pair(
        STAGE_P, "cebra_pseudosession_sua_stage_p_aggregate"
    )
    require(stage_p.get("passes_stage_p") is True, "Stage-P did not authorize expansion")
    require(stage_p.get("verdict") == "STAGE_P_PASS__EXPAND_SEEDS_43_44",
            "Stage-P verdict drift")
    authorities: dict[int, tuple[dict[str, Any], str]] = {}
    for seed in (43, 44):
        authority, digest = load_pair(
            schedule_authority_path(seed),
            "cebra_pseudosession_sua_stage_f_seed_schedule_preflight",
        )
        require(authority.get("seed") == seed, "schedule authority seed drift")
        require(authority.get("stage_p_aggregate_sha256") == stage_p_sha,
                "schedule authority Stage-P binding drift")
        require(authority.get("status") == "STAGE_F_SOURCE_SCHEDULE_PASSED__GPU_NOT_LAUNCHED",
                "schedule authority status drift")
        require(authority.get("formal_subc_test_nwb_opened") is False,
                "schedule authority opened formal data")
        authorities[seed] = (authority, digest)

    parent: dict[int, dict[str, dict[str, Mapping[str, float]]]] = {
        seed: {domain: {} for domain in DOMAINS} for seed in SEEDS
    }
    mixed: dict[int, dict[str, dict[str, Mapping[str, float]]]] = {
        seed: {domain: {} for domain in DOMAINS} for seed in SEEDS
    }
    receipt_sha: dict[str, str] = {"stage_p": stage_p_sha}
    new_bindings = None
    for seed in SEEDS:
        source_by_arm: dict[str, Mapping[str, Any]] = {}
        for domain in DOMAINS:
            for arm in ARMS:
                old, old_sha = load_pair(parent_path(domain, arm, seed))
                new, new_sha = load_pair(
                    result_path(arm, seed, domain),
                    "cebra_pseudosession_sua_domain_score",
                )
                require(old.get("seed") == new.get("seed") == seed, "seed drift")
                require(old.get("domain") == new.get("domain") == domain, "domain drift")
                require(new.get("arm") == f"mix_{arm}", "mixed arm drift")
                require(old.get("formal_subc_test_nwb_opened") is False and
                        new.get("formal_subc_test_nwb_opened") is False,
                        "formal data opened")
                require(old.get("session_query_receipts") == new.get("session_query_receipts"),
                        f"seed{seed}/{domain}/{arm}: query receipt differs from A2")
                require(tuple(old["per_session_mean_r2"]) == tuple(new["per_session_mean_r2"]),
                        "session order drift")
                if new_bindings is None:
                    new_bindings = new["implementation_bindings"]
                else:
                    require(new_bindings == new["implementation_bindings"],
                            "score implementation drift across terminal lattice")
                if arm not in source_by_arm:
                    source_by_arm[arm] = new["source_run"]
                else:
                    require(source_by_arm[arm] == new["source_run"],
                            f"seed{seed}/{arm}: domains did not reuse one source bundle")
                if seed in authorities:
                    require(new["source_run"].get("pseudo_session_schedule_sha256") ==
                            authorities[seed][0].get("source_schedule_sha256"),
                            f"seed{seed}: runtime/source-audit schedule drift")
                parent[seed][domain][arm] = old["per_session_mean_r2"]
                mixed[seed][domain][arm] = new["per_session_mean_r2"]
                receipt_sha[f"parent_s{seed}_{domain}_{arm}"] = old_sha
                receipt_sha[f"mix_s{seed}_{domain}_{arm}"] = new_sha
    result = compute_terminal(parent, mixed)
    payload = {
        "schema_version": 1,
        "receipt_kind": "cebra_pseudosession_sua_terminal_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": SCREEN_ID,
        "seeds": list(SEEDS),
        "receipt_sha256": receipt_sha,
        "stage_f_schedule_authority_sha256": {
            str(seed): digest for seed, (_, digest) in authorities.items()
        },
        "same_source_bundle_reused_across_domains": True,
        "query_receipts_exactly_match_sealed_a2_parent": True,
        "formal_subc_test_nwb_opened": False,
        **result,
    }
    write_immutable_pair(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=RESULT_ROOT / "terminal_three_seed_aggregate.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA",
            "required_new_receipts": [
                str(result_path(arm, seed, domain))
                for seed in SEEDS for domain in DOMAINS for arm in ARMS
            ],
            "required_schedule_authorities": [
                str(schedule_authority_path(seed)) for seed in (43, 44)
            ],
            "output": str(args.output),
        }, indent=2, sort_keys=True))
        return 0
    payload = execute(args.output)
    print(json.dumps({"status": "COMPLETE", "verdict": payload["verdict"],
                      "summary": payload["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
