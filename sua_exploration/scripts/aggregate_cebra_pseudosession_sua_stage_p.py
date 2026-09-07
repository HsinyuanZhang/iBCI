#!/usr/bin/env python3
"""Aggregate the frozen seed-42 pseudo-session T4/Z4 routing lattice."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))

from preflight_cebra_pseudosession_sua import write_immutable_pair  # noqa: E402
from score_cebra_pseudosession_sua import RESULT_ROOT, SCREEN_ID, result_path  # noqa: E402


A2_ROOT = SUA_ROOT / "results/a2_matched_subject_shift_v2"
DOMAINS = ("within_subject", "external_subject_M")
ARMS = ("t4", "z4")


class AggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


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


def load_pair(path: Path) -> tuple[dict[str, Any], str]:
    sidecar = Path(str(path) + ".sha256")
    require(path.is_file() and sidecar.is_file(), f"missing immutable pair: {path}")
    digest = sha256_file(path)
    require(sidecar_matches(path, digest, sidecar.read_text(encoding="ascii")),
            f"sidecar drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"receipt is not an object: {path}")
    return value, digest


def parent_path(domain: str, arm: str) -> Path:
    return A2_ROOT / f"{domain}_source_{arm}_s42.json"


def _mean(values: Mapping[str, float]) -> float:
    require(values, "empty session mapping")
    return sum(float(value) for value in values.values()) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    size = len(ordered)
    return ordered[size // 2] if size % 2 else 0.5 * (ordered[size // 2 - 1] + ordered[size // 2])


def compute_stage_p(
    parent: Mapping[str, Mapping[str, Mapping[str, float]]],
    mixed: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    cell_means: dict[str, float] = {}
    session_delta: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        for arm in ARMS:
            parent_rows = parent[domain][arm]
            mixed_rows = mixed[domain][arm]
            require(tuple(parent_rows) == tuple(mixed_rows), f"{domain}/{arm}: session roster drift")
            require(all(math.isfinite(float(value)) for value in parent_rows.values()), "non-finite parent")
            require(all(math.isfinite(float(value)) for value in mixed_rows.values()), "non-finite mixed")
            cell_means[f"parent_{domain}_{arm}"] = _mean(parent_rows)
            cell_means[f"mix_{domain}_{arm}"] = _mean(mixed_rows)
            session_delta[f"{domain}_{arm}"] = {
                session: float(mixed_rows[session]) - float(parent_rows[session])
                for session in parent_rows
            }
    external_t4 = cell_means["mix_external_subject_M_t4"] - cell_means["parent_external_subject_M_t4"]
    within_t4 = cell_means["mix_within_subject_t4"] - cell_means["parent_within_subject_t4"]
    external_z4 = cell_means["mix_external_subject_M_z4"] - cell_means["parent_external_subject_M_z4"]
    within_z4 = cell_means["mix_within_subject_z4"] - cell_means["parent_within_subject_z4"]
    external_interaction = external_t4 - external_z4
    within_interaction = within_t4 - within_z4
    gates = {
        "external_absolute_t4_delta_at_least_0p03": external_t4 >= 0.03,
        "within_t4_noninferior_at_minus_0p03": within_t4 >= -0.03,
    }
    passed = all(gates.values())
    external_session_values = list(session_delta["external_subject_M_t4"].values())
    return {
        "cell_mean_r2": cell_means,
        "deltas": {
            "external_t4": external_t4,
            "within_t4": within_t4,
            "external_z4": external_z4,
            "within_z4": within_z4,
            "external_carrier_specific_interaction": external_interaction,
            "within_carrier_specific_interaction": within_interaction,
        },
        "per_session_deltas": session_delta,
        "external_t4_session_summary": {
            "mean": _mean(session_delta["external_subject_M_t4"]),
            "median": _median(external_session_values),
            "positive_count": sum(value > 0 for value in external_session_values),
            "session_count": len(external_session_values),
        },
        "frozen_gates": gates,
        "passes_stage_p": passed,
        "verdict": "STAGE_P_PASS__EXPAND_SEEDS_43_44" if passed
                   else "STAGE_P_STOP__NO_PARAMETER_OR_ARCHITECTURE_TUNING",
        "interpretation": (
            "carrier_specific_candidate" if passed and external_interaction >= 0.03
            else "generic_or_unresolved_training_distribution_effect" if passed
            else "negative_routing_result"
        ),
    }


def execute(output: Path) -> dict[str, Any]:
    parent: dict[str, dict[str, Mapping[str, float]]] = {domain: {} for domain in DOMAINS}
    mixed: dict[str, dict[str, Mapping[str, float]]] = {domain: {} for domain in DOMAINS}
    receipt_sha: dict[str, str] = {}
    new_bindings = None
    source_by_arm: dict[str, Mapping[str, Any]] = {}
    for domain in DOMAINS:
        for arm in ARMS:
            old, old_sha = load_pair(parent_path(domain, arm))
            new, new_sha = load_pair(result_path(arm, 42, domain))
            require(old.get("seed") == new.get("seed") == 42, "seed drift")
            require(old.get("domain") == new.get("domain") == domain, "domain drift")
            require(new.get("arm") == f"mix_{arm}", "mixed arm drift")
            require(old.get("formal_subc_test_nwb_opened") is False and
                    new.get("formal_subc_test_nwb_opened") is False, "formal data opened")
            require(old.get("session_query_receipts") == new.get("session_query_receipts"),
                    f"{domain}/{arm}: query receipt differs from sealed A2 parent")
            require(tuple(old["per_session_mean_r2"]) == tuple(new["per_session_mean_r2"]),
                    f"{domain}/{arm}: session order drift")
            if new_bindings is None:
                new_bindings = new["implementation_bindings"]
            else:
                require(new_bindings == new["implementation_bindings"],
                        "new score implementation drift across lattice")
            if arm not in source_by_arm:
                source_by_arm[arm] = new["source_run"]
            else:
                require(source_by_arm[arm] == new["source_run"],
                        f"{arm}: within/external did not reuse one source checkpoint bundle")
            parent[domain][arm] = old["per_session_mean_r2"]
            mixed[domain][arm] = new["per_session_mean_r2"]
            receipt_sha[f"parent_{domain}_{arm}"] = old_sha
            receipt_sha[f"mix_{domain}_{arm}"] = new_sha
    result = compute_stage_p(parent, mixed)
    payload = {
        "schema_version": 1,
        "receipt_kind": "cebra_pseudosession_sua_stage_p_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": SCREEN_ID,
        "seed": 42,
        "receipt_sha256": receipt_sha,
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
                        default=RESULT_ROOT / "stage_p_seed42_aggregate.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "DRY_RUN", "required_new_receipts": [
            str(result_path(arm, 42, domain)) for domain in DOMAINS for arm in ARMS
        ], "output": str(args.output)}, indent=2, sort_keys=True))
        return 0
    payload = execute(args.output)
    print(json.dumps({"status": "COMPLETE", "verdict": payload["verdict"],
                      "deltas": payload["deltas"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
