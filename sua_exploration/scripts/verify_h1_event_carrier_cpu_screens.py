#!/usr/bin/env python3
"""Independent arithmetic/provenance verifier for H1 event-carrier CPU screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


SESSIONS = (
    "ses-19250101T111740", "ses-19250101T112404", "ses-19250108T110520",
    "ses-19250108T111022", "ses-19250108T111455", "ses-19250113T120811",
    "ses-19250113T121303", "ses-19250115T110633", "ses-19250115T111328",
    "ses-19250119T113543", "ses-19250119T114045", "ses-19250120T115044",
    "ses-19250120T115537",
)
DATES = ("19250101", "19250108", "19250113", "19250115", "19250119", "19250120")


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(first: Any, second: Any, label: str) -> None:
    need(math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1.0e-12),
         f"{label}: {first!r} != {second!r}")


def summary(pairs: Sequence[tuple[str, float]]) -> dict[str, Any]:
    need(bool(pairs), "empty paired summary")
    values = [float(value) for _name, value in pairs]
    remove = max(range(len(values)), key=lambda index: abs(values[index]))
    kept = values[:remove] + values[remove + 1 :]
    ordered = sorted(values)
    middle = len(values) // 2
    median = ordered[middle] if len(values) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "defined_sessions": len(values),
        "mean": sum(values) / len(values),
        "median": median,
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "leave_largest_absolute_out_mean": sum(kept) / len(kept) if kept else None,
        "removed_session": pairs[remove][0],
    }


def same_summary(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key in ("defined_sessions", "positive", "zero", "negative", "removed_session"):
        need(observed[key] == expected[key], f"{label}: summary drift at {key}")
    for key in ("mean", "median", "leave_largest_absolute_out_mean"):
        close(observed[key], expected[key], f"{label}/{key}")


def robust(row: Mapping[str, Any], minimum_positive: int = 10) -> bool:
    return bool(
        row["defined_sessions"] == 13 and row["mean"] > 0 and row["median"] > 0
        and row["positive"] >= minimum_positive and row["leave_largest_absolute_out_mean"] > 0
    )


def verify_bindings(body: Mapping[str, Any], label: str) -> None:
    need(tuple(body["source_binding"]["sessions"]) == SESSIONS, f"{label}: source session roster")
    need(tuple(body["source_binding"]["dates"]) == DATES, f"{label}: source date roster")
    for row in body["source_binding"]["files"]:
        path = Path(row["path"])
        need(path.is_file() and file_sha(path) == row["sha256"], f"{label}: source SHA mismatch {path}")
    for key, value in body["implementation_binding"].items():
        if not key.endswith("_path"):
            continue
        sha_key = key[:-5] + "_sha256"
        if sha_key not in body["implementation_binding"]:
            continue
        path = Path(value)
        need(path.is_file() and file_sha(path) == body["implementation_binding"][sha_key],
             f"{label}: implementation binding drift {key}")
    scope = body["scope"]
    need(scope["public_held_in_calibration_nwbs_opened"] == 13, f"{label}: public scope")
    for key in ("minival_nwbs_opened", "held_out_nwbs_opened", "formal_test_labels_opened"):
        need(scope[key] == 0, f"{label}: forbidden scope {key}")
    need(scope["dense_velocity_series_opened_by_carrier_screen"] is False, f"{label}: dense velocity scope")
    need(scope["decoder_constructed"] is False and scope["trainer_constructed"] is False,
         f"{label}: decoder/trainer constructed")
    need(scope["cuda_used"] is False and scope["target_session_optimizer_steps"] == 0
         and scope["target_session_backward_steps"] == 0, f"{label}: target/CUDA scope")


def aggregate_rows(candidate: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_minus_baseline": summary([
            (name, candidate[name]["median_r2_correct"] - baseline[name]["median_r2_correct"])
            for name in candidate
        ]),
        "candidate_minus_intercept": summary([(name, candidate[name]["median_delta_intercept"]) for name in candidate]),
        "candidate_minus_label_shuffle": summary([(name, candidate[name]["median_delta_label_shuffle"]) for name in candidate]),
        "candidate_minus_tag_shuffle": summary([(name, candidate[name]["median_delta_tag_shuffle"]) for name in candidate]),
    }


def verify_v1(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text())
    need(body["schema"] == "h1_event_carrier_cpu_design_screen_v1", "V1 schema")
    verify_bindings(body, "V1")
    need(body["baseline_reproduction"]["passed"] is True
         and body["baseline_reproduction"]["maximum_absolute_difference"] == 0.0, "V1 baseline reproduction")
    passing: list[str] = []
    for budget in ("M3", "M4"):
        candidates = body["budgets"][budget]["candidates"]
        baseline = candidates["pca_delta_q4"]["sessions"]
        need(tuple(baseline) == SESSIONS, f"V1/{budget}: baseline roster")
        for name, candidate in candidates.items():
            rows = candidate["sessions"]
            need(tuple(rows) == SESSIONS, f"V1/{budget}/{name}: roster")
            augmented = {
                session: {**row, "delta": row["median_r2_correct"] - baseline[session]["median_r2_correct"]}
                for session, row in rows.items()
            }
            expected = {
                "correct_r2": summary([(session, row["median_r2_correct"]) for session, row in rows.items()]),
                "correct_minus_pca_delta_q4": summary([(session, row["delta"]) for session, row in augmented.items()]),
                "correct_minus_label_shuffle": summary([(session, row["median_delta_label_shuffle"]) for session, row in rows.items()]),
                "correct_minus_intercept": summary([(session, row["median_delta_intercept"]) for session, row in rows.items()]),
            }
            if all("median_delta_tag_shuffle" in row for row in rows.values()):
                expected["correct_minus_tag_shuffle"] = summary([
                    (session, row["median_delta_tag_shuffle"]) for session, row in rows.items()
                ])
            for key, value in expected.items():
                same_summary(candidate["aggregate"][key], value, f"V1/{budget}/{name}/{key}")
            spec = candidate["spec"]
            vs = expected["correct_minus_pca_delta_q4"]
            material = robust(vs) and vs["mean"] >= 0.02 and vs["median"] >= 0.01
            label_gate = robust(expected["correct_minus_label_shuffle"])
            intercept_gate = robust(expected["correct_minus_intercept"])
            tag_gate = robust(expected["correct_minus_tag_shuffle"]) if "correct_minus_tag_shuffle" in expected else True
            passed = not spec["diagnostic_only"] and spec["rank"] == 4 and material and label_gate and intercept_gate and tag_gate
            need(candidate["gate"]["passed"] == passed, f"V1/{budget}/{name}: gate drift")
    for name, candidate in body["budgets"]["M3"]["candidates"].items():
        if name != "pca_delta_q4" and not candidate["spec"]["diagnostic_only"]:
            if candidate["gate"]["passed"] and body["budgets"]["M4"]["candidates"][name]["gate"]["passed"]:
                passing.append(name)
    need(body["passing_candidates"] == passing and body["selected_candidate"] is None and not passing,
         "V1 terminal selection drift")
    need(body["status"] == "STOP_CPU_NO_MATERIAL_CARRIER_CANDIDATE", "V1 terminal status")
    return {"status": "PASS", "sha256": file_sha(path)}


def inner_eligible(summary_by_budget: Mapping[str, Mapping[str, Any]]) -> bool:
    for budget in ("M3", "M4"):
        total = summary_by_budget[budget]["candidate_minus_baseline"]["defined_sessions"]
        minimum = math.ceil(0.6 * total)
        for key in (
            "candidate_minus_baseline", "candidate_minus_intercept",
            "candidate_minus_label_shuffle", "candidate_minus_tag_shuffle",
        ):
            row = summary_by_budget[budget][key]
            if not (row["mean"] > 0 and row["median"] > 0 and row["positive"] >= minimum
                    and row["leave_largest_absolute_out_mean"] > 0):
                return False
    return True


def verify_v2(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text())
    need(body["schema"] == "h1_event_carrier_nested_context_cpu_v2", "V2 schema")
    verify_bindings(body, "V2")
    need(all(row["passed"] for row in body["reproduction"].values()), "V2 reproduction")
    for date in DATES:
        outer = body["outer_dates"][date]
        eligible = []
        for name, row in outer["inner_summaries"].items():
            expected = inner_eligible(row["summaries"])
            need(row["eligible"] == expected, f"V2/{date}/{name}: inner eligibility")
            if expected: eligible.append(name)
        need(set(outer["eligible_configurations"]) == set(eligible), f"V2/{date}: eligible roster")
        def key(name: str) -> tuple[float, float, str]:
            summaries = outer["inner_summaries"][name]["summaries"]
            medians = [summaries[b]["candidate_minus_baseline"]["median"] for b in ("M3", "M4")]
            means = [summaries[b]["candidate_minus_baseline"]["mean"] for b in ("M3", "M4")]
            return min(medians), min(means), name
        need(outer["selected_configuration"] == max(eligible, key=key), f"V2/{date}: selection drift")
    for budget in ("M3", "M4"):
        expected = aggregate_rows(body["outer_rows"][budget], body["outer_baseline_rows"][budget])
        for key, value in expected.items():
            same_summary(body["aggregate"][budget][key], value, f"V2/{budget}/{key}")
        row = expected["candidate_minus_baseline"]
        clauses = {
            "mean_delta_at_least_0p02": row["mean"] >= 0.02,
            "median_delta_at_least_0p01": row["median"] >= 0.01,
            "positive_sessions_at_least_10": row["positive"] >= 10,
            "leave_largest_delta_positive": row["leave_largest_absolute_out_mean"] > 0,
            "beats_intercept_robustly": robust(expected["candidate_minus_intercept"]),
            "beats_label_shuffle_robustly": robust(expected["candidate_minus_label_shuffle"]),
            "beats_tag_shuffle_robustly": robust(expected["candidate_minus_tag_shuffle"]),
        }
        need(body["gate"]["budgets"][budget]["clauses"] == clauses
             and body["gate"]["budgets"][budget]["passed"] == all(clauses.values()), f"V2/{budget}: gate")
    need(body["gate"]["passed"] is False and body["status"] == "STOP_CPU_NESTED_CONTEXT_NOT_MATERIAL",
         "V2 terminal status")
    return {"status": "PASS", "sha256": file_sha(path)}


def verify_v3(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text())
    need(body["schema"] == "h1_event_carrier_meta_basis_cpu_v3", "V3 schema")
    verify_bindings(body, "V3")
    need(body["baseline_reproduction"]["passed"] is True
         and body["baseline_reproduction"]["maximum_absolute_difference"] == 0.0, "V3 reproduction")
    passing = []
    for name, arm in body["arms"].items():
        for mapping in arm["maps"].values():
            optimizer = mapping["optimizer"]
            need(math.isfinite(optimizer["initial_loss"]) and math.isfinite(optimizer["final_loss"])
                 and optimizer["final_loss"] < optimizer["initial_loss"], f"V3/{name}: source optimization")
        for budget in ("M3", "M4"):
            expected = aggregate_rows(arm["rows"][budget], body["baseline_rows"][budget])
            for key, value in expected.items():
                same_summary(arm["aggregate"][budget][key], value, f"V3/{name}/{budget}/{key}")
            material = expected["candidate_minus_baseline"]
            clauses = {
                "mean_delta_at_least_0p02": material["mean"] >= 0.02,
                "median_delta_at_least_0p01": material["median"] >= 0.01,
                "positive_sessions_at_least_10": material["positive"] >= 10,
                "leave_largest_delta_positive": material["leave_largest_absolute_out_mean"] > 0,
                "beats_intercept_robustly": robust(expected["candidate_minus_intercept"]),
                "beats_label_shuffle_robustly": robust(expected["candidate_minus_label_shuffle"]),
                "beats_tag_shuffle_robustly": robust(expected["candidate_minus_tag_shuffle"]),
            }
            need(arm["gate"]["budgets"][budget]["clauses"] == clauses
                 and arm["gate"]["budgets"][budget]["passed"] == all(clauses.values()),
                 f"V3/{name}/{budget}: gate")
        if arm["gate"]["passed"]: passing.append(name)
    need(body["passing_arms"] == passing and not passing and body["selected_arm"] is None,
         "V3 passing arm drift")
    need(body["status"] == "STOP_CPU_META_BASIS_NOT_MATERIAL", "V3 terminal status")
    return {"status": "PASS", "sha256": file_sha(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v1", type=Path)
    parser.add_argument("v2", type=Path)
    parser.add_argument("v3", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [args.v1.resolve(), args.v2.resolve(), args.v3.resolve()]
    for path in paths:
        need(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"receipt is not immutable: {path}")
    result = {
        "status": "PASS",
        "v1": {"path": str(paths[0]), **verify_v1(paths[0])},
        "v2": {"path": str(paths[1]), **verify_v2(paths[1])},
        "v3": {"path": str(paths[2]), **verify_v3(paths[2])},
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        need(not output.exists(), f"refusing to overwrite verifier artifact: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        output.chmod(0o444)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
