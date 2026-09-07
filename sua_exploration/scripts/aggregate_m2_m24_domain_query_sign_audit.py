#!/usr/bin/env python3
"""Fail-closed aggregate for the frozen M2/M24 domain--query sign audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "sua_exploration/results/m2_m24_domain_query_sign_audit_v1"
HELDOUT = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
)
EXPECTED: dict[tuple[str, str, int, str], str] = {
    ("f0", "heldin", 0, "best"): "profile_f0_heldin_q0_best.json",
    ("f0", "heldin", 0, "epoch9"): "profile_f0_heldin_q0_epoch9.json",
    ("t4", "heldin", 0, "best"): "profile_t4_heldin_q0_best.json",
    ("t4", "heldin", 0, "epoch9"): "profile_t4_heldin_q0_epoch9.json",
    ("f0", "heldout", 0, "best"): "profile_f0_heldout_q0_best_metric_route.json",
    ("f0", "heldout", 0, "epoch9"): "profile_f0_heldout_q0_epoch9_metric_route.json",
    ("t4", "heldout", 0, "best"): "profile_t4_heldout_q0_best_metric_route.json",
    ("t4", "heldout", 0, "epoch9"): "profile_t4_heldout_q0_epoch9_metric_route.json",
    ("f0", "heldout", 24, "best"): "profile_f0_heldout_q24_best_metric_route.json",
    ("f0", "heldout", 24, "epoch9"): "profile_f0_heldout_q24_epoch9_metric_route.json",
    ("t4", "heldout", 24, "best"): "profile_t4_heldout_q24_best_metric_route.json",
    ("t4", "heldout", 24, "epoch9"): "profile_t4_heldout_q24_epoch9_metric_route.json",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def expected_cell(key: tuple[str, str, int, str]) -> dict[str, Any]:
    arm, domain, query, policy = key
    return {"arm": arm, "domain": domain, "query_start_trial": query, "checkpoint_policy": policy,
            "task": "m2", "fold": 1, "seed": 42, "calibration_trials": 24, "window_size": 50}


def build_seal_list(root: Path) -> dict[str, Any]:
    cells = []
    for key, name in sorted(EXPECTED.items()):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"cannot seal missing profile: {path}")
        cells.append({"cell": expected_cell(key), "path": str(path.resolve()), "sha256": digest(path)})
    return {
        "schema_version": 1,
        "purpose": "immutable_12_score_cell_seal_for_m2_m24_domain_query_sign_audit",
        "profiles": cells,
        "structural_ineligible_not_score_cell": {
            "cell": {"domain": "heldin", "query_start_trial": 24, "task": "m2", "fold": 1, "seed": 42,
                     "calibration_trials": 24, "window_size": 50},
            "reason": "source held-in validation session has only two query trial boundaries; M=24 leaves zero future query",
            "evidence": str((root / "heldin_q24_trial_boundary_audit.json").resolve()),
        },
        "label_information_disclosure": "Both arms use 24 chronological support trials. T4 consumes one target-direction label per trial; any K4 comparison is outside this seal because K4 consumes dense per-bin velocity and is not label-information matched to T4.",
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_sealed(seal: dict[str, Any]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    entries = seal.get("profiles")
    require(isinstance(entries, list) and len(entries) == len(EXPECTED), "seal must name exactly twelve profiles")
    observed: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for entry in entries:
        cell = entry.get("cell", {})
        key = (cell.get("arm"), cell.get("domain"), cell.get("query_start_trial"), cell.get("checkpoint_policy"))
        require(key in EXPECTED and key not in observed, f"unexpected or duplicate sealed key: {key}")
        path = Path(entry.get("path", ""))
        require(path.is_file(), f"sealed profile missing: {path}")
        require(digest(path) == entry.get("sha256"), f"sealed profile SHA mismatch: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        require(row.get("cell") == expected_cell(key), f"profile cell contract drift: {path}")
        require(row.get("scope") == "CPU-only, frozen-checkpoint, test-only; no backward/optimizer/checkpoint selection", f"scope drift: {path}")
        require(row.get("cuda_available_at_preflight") is False, f"CUDA policy drift: {path}")
        scores = row.get("session_r2")
        require(isinstance(scores, dict) and scores, f"missing scores: {path}")
        require(all(math.isfinite(float(v)) for v in scores.values()), f"non-finite score: {path}")
        if key[1] == "heldout":
            require(set(scores) == set(HELDOUT), f"heldout sessions drift: {path}")
            route = row.get("metric_route", {})
            require(route.get("target_loader_index") == 1 and route.get("target_metric_prefix") == "test_heldout_", f"heldout metric route drift: {path}")
            ignored = row.get("anchor_metrics_ignored", {})
            require(ignored.get("role") == "ignored_provenance_only_not_used_for_selection_or_aggregation", f"anchor role drift: {path}")
            if key[2] == 24:
                require(row.get("query_start_role") == "chronological_future_query", f"q24 role drift: {path}")
                audits = row.get("query_window_audit", {})
                require(set(audits) == set(HELDOUT), f"q24 audit sessions drift: {path}")
                for session, audit in audits.items():
                    require(audit.get("full_window_disjoint") is True and audit.get("minimum_window_start_padded_bin") == audit.get("raw_query_start_bin", -50) + 49, f"q24 is not full-window disjoint: {path}/{session}")
            else:
                require(row.get("query_start_role") == "diagnostic_resubstitution_not_future_query", f"q0 role drift: {path}")
        else:
            require(key[2] == 0 and len(scores) == 1, f"heldin score-cell scope drift: {path}")
        observed[key] = row
    require(set(observed) == set(EXPECTED), "seal does not cover exact expected cell grid")
    return observed


def mean(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)


def paired(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    values = {s: float(left[s]) - float(right[s]) for s in sorted(left)}
    return {"per_session_r2": values, "equal_session_mean_r2": mean(values), "positive_sessions": sum(v > 0 for v in values.values()), "n_sessions": len(values)}


def aggregate(rows: dict[tuple[str, str, int, str], dict[str, Any]], seal_path: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for policy in ("best", "epoch9"):
        policy_report: dict[str, Any] = {}
        for domain, query, label in (("heldin", 0, "heldin_q0"), ("heldout", 0, "heldout_q0"), ("heldout", 24, "heldout_q24")):
            f0 = rows[("f0", domain, query, policy)]["session_r2"]
            t4 = rows[("t4", domain, query, policy)]["session_r2"]
            policy_report[label] = {"F0": {"mean_r2": mean(f0), "session_r2": f0}, "T4": {"mean_r2": mean(t4), "session_r2": t4}, "T4_minus_F0": paired(t4, f0)}
        reports[policy] = policy_report
    horizon: dict[str, Any] = {}
    checkpoint_shift: dict[str, Any] = {}
    for arm in ("f0", "t4"):
        horizon[arm] = {}
        for policy in ("best", "epoch9"):
            horizon[arm][policy] = paired(rows[(arm, "heldout", 24, policy)]["session_r2"], rows[(arm, "heldout", 0, policy)]["session_r2"])
        checkpoint_shift[arm] = {}
        for domain, query, label in (("heldin", 0, "heldin_q0"), ("heldout", 0, "heldout_q0"), ("heldout", 24, "heldout_q24")):
            checkpoint_shift[arm][label] = paired(rows[(arm, domain, query, "epoch9")]["session_r2"], rows[(arm, domain, query, "best")]["session_r2"])
    return {
        "schema_version": 1,
        "purpose": "M2_M24_internal_vs_local_heldout_domain_query_sign_audit",
        "seal_list": {"path": str(seal_path.resolve()), "sha256": digest(seal_path)},
        "reports_by_checkpoint_policy": reports,
        "heldout_q24_minus_q0_by_arm": horizon,
        "epoch9_minus_best_by_arm": checkpoint_shift,
        "interpretation_contract": {
            "heldin_q0": "internal held-in diagnostic/resubstitution score; it is not a future-query held-in endpoint",
            "heldout_q0": "local held-out session diagnostic/resubstitution score; it is not a chronological future-query endpoint",
            "heldout_q24": "local held-out chronological future-query score with all scored 50-bin histories disjoint from 24-trial support",
            "endpoint_composition_limit": "The observed heldin-vs-heldout q0 sign contrast is not a pure session-domain effect: heldin q0 is one two-trial minival file/129 windows, while heldout q0 scores full 33--43-trial session files and overlaps support.",
            "domain_by_query_identifiability": "not identifiable: held-in q24 is structurally ineligible, so unique domain, horizon, and domain×query-interaction attribution cannot be separated. Report only that sign follows the observed heldin-vs-heldout endpoints in both checkpoint policies.",
            "label_information": "T4 uses one direction label per support trial. K4 is deliberately excluded from this aggregate because its dense velocity labels are not information-matched to T4.",
        },
        "scope": "frozen-checkpoint CPU re-evaluation only; local data replay, not hidden EvalAI/challenge evaluation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--seal-list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-seal-list", action="store_true")
    args = parser.parse_args()
    if args.write_seal_list:
        require(not args.seal_list.exists(), f"refusing to overwrite seal list {args.seal_list}")
        args.seal_list.parent.mkdir(parents=True, exist_ok=True)
        args.seal_list.write_text(json.dumps(build_seal_list(args.root.resolve()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require(args.seal_list.is_file(), f"seal list does not exist: {args.seal_list}")
    require(not args.out.exists(), f"refusing to overwrite aggregate {args.out}")
    rows = read_sealed(json.loads(args.seal_list.read_text(encoding="utf-8")))
    payload = aggregate(rows, args.seal_list)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"heldout_q24_best_delta": payload["reports_by_checkpoint_policy"]["best"]["heldout_q24"]["T4_minus_F0"]["equal_session_mean_r2"]}, indent=2))


if __name__ == "__main__":
    main()
