#!/usr/bin/env python3
"""Render a bound, read-only v2-versus-adaptive carrier comparison.

This tool reads only final/source audit JSON documents.  It never opens scoring
artifacts, training outputs, NWB files, or NPZ files, and it does not score or
train.  Every reported number is copied from a PASSED final audit after the
v2 B/legacy-D reference rows and their target bindings have been matched.
H1 equal-session aggregates and deltas are arithmetic derived from those
already-audited scalar report values; they are not a replay or rescoring step.
`training_horizon_epochs` records the frozen training horizon (24 for M1, 32
for H1).  A `selected` endpoint is each arm's source-validation-selected
checkpoint; this report does not infer its actual epoch from that horizon.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_ROOT = (ROOT / "results" / "carrier_adaptive_v3").resolve()
DATASETS = ("m1", "h1")
PROFILES = {"m1": "muscle", "h1": "state"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing required JSON: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"finite numeric value required: {context}")
    result = float(value)
    if not math.isfinite(result):
        fail(f"nonfinite value: {context}")
    return result


def canonical(value: Any) -> str:
    """Stable representation used only to compare audit-provided bindings."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_gate_binding(path: Path, audit: dict[str, Any], dataset: str, stage: str, profile: str) -> dict[str, str] | None:
    """Validate the sibling source gate when the final audit declares its SHA."""
    expected = audit.get("source_gate_sha256")
    if expected is None:
        return None
    require(isinstance(expected, str) and len(expected) == 64, f"bad source_gate_sha256: {path}")
    gate_path = path.parent / "source_gate.json"
    gate = read_object(gate_path)
    actual = sha256(gate_path)
    require(actual == expected, f"final/source gate SHA drift: {path}")
    identity = {
        "schema": "carrier_last1_v2_paired_audit",
        "status": "PASSED",
        "phase": "source",
        "stage": stage,
        "dataset": dataset,
        "profiles": ["old", profile],
    }
    for key, value in identity.items():
        require(gate.get(key) == value, f"source gate identity drift ({key}): {gate_path}")
    return {"path": str(gate_path.resolve()), "sha256": actual}


def validate_audit(path: Path, dataset: str, stage: str, profile: str) -> tuple[dict[str, Any], dict[str, str] | None]:
    audit = read_object(path)
    identity = {
        "schema": "carrier_last1_v2_paired_audit",
        "status": "PASSED",
        "phase": "final",
        "stage": stage,
        "dataset": dataset,
        "profiles": ["old", profile],
    }
    for key, value in identity.items():
        require(audit.get(key) == value, f"final audit identity drift ({key}): {path}")
    require(isinstance(audit.get("target_replay"), dict), f"target_replay object required: {path}")
    return audit, source_gate_binding(path, audit, dataset, stage, profile)


def endpoint_names(dataset: str) -> tuple[tuple[str, str], tuple[str, str]]:
    return (("fixed", "fixed_e24" if dataset == "m1" else "fixed_e32"), ("selected", "selected"))


def simple_rows(audit: dict[str, Any], dataset: str, arms: dict[str, str], origin: str) -> list[dict[str, Any]]:
    """Read inner_reference.py's compact final-audit schema."""
    replay = audit["target_replay"]
    channel = audit.get("channelwise_replay")
    require(isinstance(channel, dict), f"channelwise_replay required: {origin}")
    rows: list[dict[str, Any]] = []
    binding_by_arm = audit.get("target_typed_bindings", {})
    require(isinstance(binding_by_arm, dict), f"target_typed_bindings required: {origin}")
    for audit_arm, arm in arms.items():
        values, diagnostics = replay.get(audit_arm), channel.get(audit_arm)
        require(isinstance(values, dict) and isinstance(diagnostics, dict), f"compact arm missing: {origin}:{audit_arm}")
        # M1 inner_reference.final stores one shared fixed/selected binding,
        # while H1 stores a separate session-signature list for each arm.
        binding = binding_by_arm if dataset == "m1" else binding_by_arm.get(audit_arm)
        require(binding is not None, f"compact target binding missing: {origin}:{audit_arm}")
        if dataset == "h1":
            require(isinstance(binding, list) and binding, f"H1 compact target binding malformed: {origin}:{audit_arm}")
            h1_bindings: dict[str, dict[str, str]] = {}
            for item in binding:
                require(isinstance(item, list) and len(item) == 3 and all(isinstance(part, str) for part in item), f"H1 compact target binding item malformed: {origin}:{audit_arm}")
                session, target_sha, endpoint_sha = item
                require(session not in h1_bindings, f"duplicate H1 compact binding: {origin}:{audit_arm}:{session}")
                h1_bindings[session] = {"target": target_sha, "endpoints": endpoint_sha}
        for endpoint, key in endpoint_names(dataset):
            legacy = finite(values.get(key), f"{origin}:{audit_arm}:{key}:legacy")
            diagnostic = diagnostics.get(key)
            if dataset == "m1":
                require(isinstance(binding, dict) and binding.get(key) is not None, f"M1 compact target binding missing endpoint: {origin}:{audit_arm}:{key}")
                endpoint_binding = binding[key]
                channel_value = finite(diagnostic, f"{origin}:{audit_arm}:{key}:channel")
                rows.extend((
                    {"endpoint": endpoint, "session": "__aggregate__", "metric": "legacy_flattened_r2", "arm": arm, "value": legacy, "target_binding": endpoint_binding},
                    {"endpoint": endpoint, "session": "__aggregate__", "metric": "channel_variance_weighted_r2", "arm": arm, "value": channel_value, "target_binding": endpoint_binding},
                ))
            else:
                require(isinstance(diagnostic, dict), f"H1 channel aggregate/per-session required: {origin}:{audit_arm}:{key}")
                per = diagnostic.get("per_session")
                aggregate = finite(diagnostic.get("equal_session_mean"), f"{origin}:{audit_arm}:{key}:channel aggregate")
                require(isinstance(per, dict) and per, f"H1 per_session channel values required: {origin}:{audit_arm}:{key}")
                rows.extend((
                    {"endpoint": endpoint, "session": "__equal_session_mean__", "metric": "legacy_flattened_r2", "arm": arm, "value": legacy, "target_binding": h1_bindings},
                    {"endpoint": endpoint, "session": "__equal_session_mean__", "metric": "channel_variance_weighted_r2", "arm": arm, "value": aggregate, "target_binding": h1_bindings},
                ))
                for session, value in sorted(per.items()):
                    require(str(session) in h1_bindings, f"H1 per-session binding missing: {origin}:{audit_arm}:{key}:{session}")
                    rows.append({"endpoint": endpoint, "session": str(session), "metric": "channel_variance_weighted_r2", "arm": arm, "value": finite(value, f"{origin}:{audit_arm}:{key}:{session}"), "target_binding": h1_bindings[str(session)]})
    return rows


def paired_rows(audit: dict[str, Any], dataset: str, arms: dict[str, str], origin: str) -> list[dict[str, Any]]:
    """Read paired_audit.py's final-audit schema without reopening artifacts."""
    replay = audit["target_replay"]
    rows: list[dict[str, Any]] = []
    names = {"m1": {"fixed": "target_epoch24_ema", "selected": "target_selected_ema"}, "h1": {"fixed": "fixed_e32_ema", "selected": "selected_ema"}}[dataset]
    for audit_arm, arm in arms.items():
        value = replay.get(audit_arm)
        if dataset == "m1":
            require(isinstance(value, list) and value, f"M1 paired replay missing: {origin}:{audit_arm}")
            for target_row in value:
                require(isinstance(target_row, dict) and isinstance(target_row.get("target"), str), f"M1 target row malformed: {origin}:{audit_arm}")
                metrics = {item.get("kind"): item for item in target_row.get("replay", []) if isinstance(item, dict)}
                require(set(metrics) == set(names.values()), f"M1 endpoint roster drift: {origin}:{audit_arm}")
                for endpoint, kind in names.items():
                    item = metrics[kind]
                    binding = item.get("target_binding")
                    require(isinstance(binding, dict) and binding, f"M1 target binding missing: {origin}:{audit_arm}:{kind}")
                    for metric, field in (("legacy_flattened_r2", "r2"), ("channel_variance_weighted_r2", "channel_variance_weighted_r2")):
                        rows.append({"endpoint": endpoint, "session": target_row["target"], "metric": metric, "arm": arm, "value": finite(item.get(field), f"{origin}:{audit_arm}:{kind}:{metric}"), "target_binding": binding})
        else:
            require(isinstance(value, dict), f"H1 paired replay missing: {origin}:{audit_arm}")
            metrics = {item.get("kind"): item for item in value.get("replay", []) if isinstance(item, dict)}
            require(set(metrics) == set(names.values()), f"H1 endpoint roster drift: {origin}:{audit_arm}")
            for endpoint, kind in names.items():
                item = metrics[kind]
                legacy, channel = item.get("per_session_r2"), item.get("per_session_channel_variance_weighted_r2")
                bindings = item.get("target_bindings")
                require(isinstance(legacy, dict) and legacy and set(legacy) == set(channel or {}) and isinstance(bindings, dict), f"H1 per-session metrics/bindings missing: {origin}:{audit_arm}:{kind}")
                for session in sorted(legacy):
                    binding = bindings.get(session)
                    require(binding is not None, f"H1 target binding missing: {origin}:{audit_arm}:{kind}:{session}")
                    rows.extend((
                        {"endpoint": endpoint, "session": session, "metric": "legacy_flattened_r2", "arm": arm, "value": finite(legacy[session], f"{origin}:{audit_arm}:{kind}:{session}:legacy"), "target_binding": binding},
                        {"endpoint": endpoint, "session": session, "metric": "channel_variance_weighted_r2", "arm": arm, "value": finite(channel[session], f"{origin}:{audit_arm}:{kind}:{session}:channel"), "target_binding": binding},
                    ))
                legacy_mean = sum(finite(x, f"{origin}:{audit_arm}:{kind}:legacy") for x in legacy.values()) / len(legacy)
                rows.append({"endpoint": endpoint, "session": "__equal_session_mean__", "metric": "legacy_flattened_r2", "arm": arm, "value": legacy_mean, "target_binding": bindings})
                channel_mean = sum(finite(x, f"{origin}:{audit_arm}:{kind}:channel") for x in channel.values()) / len(channel)
                stated_channel = finite(item.get("sessionmean_channel_variance_weighted_r2"), f"{origin}:{audit_arm}:{kind}:channel:aggregate")
                require(math.isclose(stated_channel, channel_mean, rel_tol=0.0, abs_tol=1e-12), f"H1 channel aggregate drift: {origin}:{audit_arm}:{kind}")
                rows.append({"endpoint": endpoint, "session": "__equal_session_mean__", "metric": "channel_variance_weighted_r2", "arm": arm, "value": stated_channel, "target_binding": bindings})
    return rows


def audit_rows(audit: dict[str, Any], dataset: str, arms: dict[str, str], origin: str) -> list[dict[str, Any]]:
    replay = audit["target_replay"]
    # The compact historical-inner format has direct endpoint maps for B.
    compact = isinstance(replay.get("B"), dict) and any(key in replay["B"] for _, key in endpoint_names(dataset))
    return simple_rows(audit, dataset, arms, origin) if compact else paired_rows(audit, dataset, arms, origin)


def index_rows(rows: list[dict[str, Any]], origin: str) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["endpoint"], row["session"], row["metric"], row["arm"])
        require(key not in indexed, f"duplicate report row: {origin}:{key}")
        indexed[key] = row
    return indexed


def compare_reference(v2: dict[tuple[str, str, str, str], dict[str, Any]], v3: dict[tuple[str, str, str, str], dict[str, Any]], label: str) -> None:
    for arm in ("B_ACTIVITY_ONLY", "legacy_D"):
        left = {key[:3]: row for key, row in v2.items() if key[3] == arm}
        right = {key[:3]: row for key, row in v3.items() if key[3] == arm}
        require(set(left) == set(right) and left, f"v2/v3 {arm} endpoint roster drift: {label}")
        for key in left:
            require(math.isclose(left[key]["value"], right[key]["value"], rel_tol=0.0, abs_tol=1e-12), f"v2/v3 {arm} score drift: {label}:{key}")
            require(canonical(left[key]["target_binding"]) == canonical(right[key]["target_binding"]), f"v2/v3 {arm} target binding drift: {label}:{key}")


def make_block(dataset: str, stage: str, v2_path: Path, v3_path: Path) -> dict[str, Any]:
    profile = PROFILES[dataset]
    v2_audit, v2_gate = validate_audit(v2_path, dataset, stage, profile)
    v3_audit, v3_gate = validate_audit(v3_path, dataset, stage, profile)
    v2_candidate_arm = "candidate_D" if "candidate_D" in v2_audit["target_replay"] else f"{profile}_D"
    require(v2_candidate_arm in v2_audit["target_replay"], f"v2 candidate arm missing: {v2_path}")
    v2 = index_rows(audit_rows(v2_audit, dataset, {"B": "B_ACTIVITY_ONLY", "old_D": "legacy_D", v2_candidate_arm: "v2_profile_D"}, str(v2_path)), str(v2_path))
    v3_candidate_arm = "candidate_D" if "candidate_D" in v3_audit["target_replay"] else f"{profile}_D"
    require(v3_candidate_arm in v3_audit["target_replay"], f"adaptive candidate arm missing: {v3_path}")
    v3 = index_rows(audit_rows(v3_audit, dataset, {"B": "B_ACTIVITY_ONLY", "old_D": "legacy_D", v3_candidate_arm: "adaptive_D"}, str(v3_path)), str(v3_path))
    compare_reference(v2, v3, f"{dataset}/{stage}")
    v2_candidate = {key[:3]: row for key, row in v2.items() if key[3] == "v2_profile_D"}
    v3_candidate = {key[:3]: row for key, row in v3.items() if key[3] == "adaptive_D"}
    bases = {key[:3]: row for key, row in v3.items() if key[3] == "B_ACTIVITY_ONLY"}
    require(set(v2_candidate) == set(v3_candidate) == set(bases) and v2_candidate, f"candidate/B endpoint roster drift: {dataset}/{stage}")
    rows = []
    for key in sorted(v2_candidate):
        endpoint, session, metric = key
        b, old = bases[key], v3[(endpoint, session, metric, "legacy_D")]
        old_v2 = v2[(endpoint, session, metric, "legacy_D")]
        require(math.isclose(old["value"], old_v2["value"], rel_tol=0.0, abs_tol=1e-12), f"legacy_D internal binding drift: {dataset}/{stage}:{key}")
        require(canonical(v2_candidate[key]["target_binding"]) == canonical(v2[(endpoint, session, metric, "B_ACTIVITY_ONLY")]["target_binding"]), f"v2 candidate/B target binding drift: {dataset}/{stage}:{key}")
        require(canonical(v3_candidate[key]["target_binding"]) == canonical(b["target_binding"]), f"adaptive candidate/B target binding drift: {dataset}/{stage}:{key}")
        v2_value, adaptive = v2_candidate[key]["value"], v3_candidate[key]["value"]
        rows.append({"dataset": dataset, "stage": stage, "endpoint": endpoint, "training_horizon_epochs": 24 if dataset == "m1" else 32, "session": session, "metric": metric, "B_ACTIVITY_ONLY": b["value"], "legacy_D": old["value"], "v2_profile_D": v2_value, "adaptive_D": adaptive, "adaptive_minus_v2_profile_D": adaptive - v2_value, "adaptive_minus_B_ACTIVITY_ONLY": adaptive - b["value"]})
    inputs = {
        "v2_final_audit": {"path": str(v2_path.resolve()), "sha256": sha256(v2_path)},
        "adaptive_final_audit": {"path": str(v3_path.resolve()), "sha256": sha256(v3_path)},
    }
    if v2_gate is not None: inputs["v2_source_gate"] = v2_gate
    if v3_gate is not None: inputs["adaptive_source_gate"] = v3_gate
    return {"dataset": dataset, "stage": stage, "loader_profile": profile, "adaptive_carrier_label": "adaptive carrier", "inputs": inputs, "rows": rows}


def require_destination(path: Path) -> Path:
    resolved = path.resolve()
    require(RESULTS_ROOT in (resolved, *resolved.parents), f"--dest must be below {RESULTS_ROOT}: {resolved}")
    require(resolved != RESULTS_ROOT, "--dest must name a new report directory below results/carrier_adaptive_v3")
    require(not resolved.exists(), f"--dest must not already exist: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--m1-inner-audit", type=Path, required=True)
    parser.add_argument("--h1-inner-audit", type=Path, required=True)
    parser.add_argument("--m1-outer-audit", type=Path, required=True)
    parser.add_argument("--h1-outer-audit", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    dest = require_destination(args.dest)
    v2_root = args.v2_root.resolve()
    require(v2_root.is_dir(), f"--v2-root must be an existing directory: {v2_root}")
    paths = {
        ("m1", "inner"): (v2_root / "inner" / "m1" / "final_audit.json", args.m1_inner_audit.resolve()),
        ("h1", "inner"): (v2_root / "inner" / "h1" / "final_audit.json", args.h1_inner_audit.resolve()),
        ("m1", "outer"): (v2_root / "outer" / "m1" / "final_audit.json", args.m1_outer_audit.resolve()),
        ("h1", "outer"): (v2_root / "outer" / "h1" / "final_audit.json", args.h1_outer_audit.resolve()),
    }
    blocks = [make_block(dataset, stage, v2_path, v3_path) for (dataset, stage), (v2_path, v3_path) in paths.items()]
    flat = [row for block in blocks for row in block["rows"]]
    require(flat, "no report rows")
    payload = {
        "schema": "carrier_adaptive_v3_comparison_report_v1",
        "status": "PASSED",
        "scope": "audit-reported endpoint comparison only; no scoring, training, runtime calculation, significance inference, or generalization claim",
        "metric_definitions": {
            "legacy_flattened_r2": "historical scorer-compatible flattened/global-mean R2 as reported by the sealed audit",
            "channel_variance_weighted_r2": "per-channel-centered variance-weighted R2 as reported by the sealed audit",
        },
        "blocks": blocks,
    }
    # Validate every nested report field before creating the new output directory.
    json.dumps(payload, sort_keys=True, allow_nan=False)
    dest.mkdir(parents=True)
    json_path, csv_path = dest / "comparison.json", dest / "comparison.csv"
    with json_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    fields = ("dataset", "stage", "endpoint", "training_horizon_epochs", "session", "metric", "B_ACTIVITY_ONLY", "legacy_D", "v2_profile_D", "adaptive_D", "adaptive_minus_v2_profile_D", "adaptive_minus_B_ACTIVITY_ONLY")
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
