#!/usr/bin/env python3
"""Fail-closed aggregator for the C1 teacher-domain 2x2.

Primary quantities are absolute external T4 lift and the matched
teacher-by-carrier interaction.  A collapsing Z4 cannot mint a win.  A generic
lift shared by T4 and Z4 cannot be reported as greater use of carrier content.

This CLI never opens NWB or a GPU.  Missing cells, sealed-session names in a
receipt, add-site drift, or H-add contamination fail closed.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for path in (REPO_ROOT, SUA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mc_maze import c1_teacher_domain_ablation as core
from mc_maze.gpu_contract_common import SEALED_FORMAL_TEST_SESSIONS


def _load_cell_receipt(
    path: Path,
    *,
    teacher: str,
    carrier: str,
    seed: int,
    domain: str,
    contract_sha256: str,
) -> dict[str, Any]:
    payload = core.load_json_object(path)
    core.require(payload.get("schema_version") == 1, f"{path}: schema drift")
    core.require(payload.get("screen_id") == core.SCREEN_ID, f"{path}: screen id drift")
    core.require(payload.get("contract_sha256") == contract_sha256, f"{path}: contract SHA drift")
    core.require(payload.get("teacher_domain") == teacher, f"{path}: teacher domain drift")
    core.require(payload.get("carrier") == carrier, f"{path}: carrier drift")
    core.require(payload.get("seed") == seed, f"{path}: seed drift")
    core.require(payload.get("domain") == domain, f"{path}: domain drift")
    core.require(payload.get("add_site") == core.ADD_SITE, f"{path}: add-site drift; W-add must stay fixed")
    core.require(payload.get("sampling") == core.SAMPLING, f"{path}: sampling drift")
    core.require(payload.get("hidden_space_adapter") is False, f"{path}: A1 H-add contamination")
    core.require(payload.get("loss_mode") == core.LOSS_MODE, f"{path}: loss-mode drift")
    core.require(payload.get("variant") == core.VARIANT, f"{path}: variant drift")
    core.require(payload.get("encoder_warmstart_path") in (None, ""), f"{path}: selected-T4 encoder warm-start is forbidden")
    core.require(payload.get("sealed_formal_test_sessions_opened") is False, f"{path}: sealed test sessions opened")
    core.require(payload.get("formal_test_sessions_opened") is False, f"{path}: formal-test flag drift")
    sessions = payload.get("per_session_r2")
    core.require(isinstance(sessions, Mapping), f"{path}: per_session_r2 missing")
    expected = core.expected_domain_sessions(domain)
    core.require(tuple(sessions) == expected, f"{path}: session roster/order drift")
    core.refuse_sealed_sessions(list(sessions), label=str(path))
    hit = [name for name in sessions if name in SEALED_FORMAL_TEST_SESSIONS]
    core.require(not hit, f"{path}: sealed session keys present: {hit}")
    values = [float(sessions[name]) for name in expected]
    core.require(all(v == v for v in values), f"{path}: non-finite R2")  # NaN check
    declared = float(payload.get("mean_r2"))
    expected_mean = sum(values) / len(values)
    core.require(declared == expected_mean, f"{path}: declared mean_r2 is not exact")
    payload["_path"] = str(path)
    return payload


def aggregate(result_dir: Path, *, contract: Path = core.CONTRACT_PATH) -> dict[str, Any]:
    result_dir = Path(result_dir).expanduser().resolve()
    contract_sha = core.sha256_file(contract)
    matrix = core.empty_score_matrix()
    loaded: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    missing: list[str] = []
    for teacher in core.TEACHER_DOMAINS:
        for carrier in core.CARRIERS:
            for seed in core.SEEDS:
                for domain in core.DOMAINS:
                    name = core.domain_receipt_name(teacher, carrier, seed, domain)
                    path = result_dir / name
                    if not path.is_file():
                        missing.append(name)
                        continue
                    receipt = _load_cell_receipt(
                        path,
                        teacher=teacher,
                        carrier=carrier,
                        seed=seed,
                        domain=domain,
                        contract_sha256=contract_sha,
                    )
                    matrix[teacher][carrier][seed][domain] = float(receipt["mean_r2"])
                    loaded[(teacher, carrier, seed, domain)] = receipt
    core.require(not missing, f"fail-closed: missing domain receipts: {missing}")
    core.require(len(loaded) == core.DOMAIN_RECEIPT_COUNT, "fail-closed: incomplete C1 matrix")
    gate = core.evaluate_c1_gates(matrix)
    return {
        "schema_version": 1,
        "screen_id": core.SCREEN_ID,
        "kind": "c1_teacher_domain_ablation_aggregate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(Path(contract).resolve()),
        "contract_sha256": contract_sha,
        "result_dir": str(result_dir),
        "add_site": core.ADD_SITE,
        "sampling": core.SAMPLING,
        "hidden_space_adapter": False,
        "nwb_opened": False,
        "gpu_used": False,
        "sealed_formal_test_sessions_opened": False,
        "cell_count": len(loaded),
        "matrix_mean_r2": {
            teacher: {
                carrier: {
                    str(seed): {domain: matrix[teacher][carrier][seed][domain] for domain in core.DOMAINS}
                    for seed in core.SEEDS
                }
                for carrier in core.CARRIERS
            }
            for teacher in core.TEACHER_DOMAINS
        },
        "gate": gate,
        "hypothesis_status": "plausible_contributor_not_isolated_cause",
        "isolated_cause_claimed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=core.CONTRACT_PATH)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = aggregate(args.result_dir, contract=args.contract)
        body, sidecar, digest = core.write_immutable_json(args.out, payload)
    except (core.C1ContractError, FileExistsError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        core.pretty_json_bytes(
            {
                "passes_all_gates": payload["gate"]["passes_all_gates"],
                "verdict": payload["gate"]["verdict"],
                "out": str(body),
                "sidecar": str(sidecar),
                "sha256": digest,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
