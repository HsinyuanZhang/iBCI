#!/usr/bin/env python3
"""Write a fail-closed receipt for one M2 M24 test-only held-out replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


GROUPS = {"f0": ("B3", "none"), "t4": ("B3S", "t4"), "k4": ("B3S", "k4"), "ks4": ("B3S", "ks4")}
EXPECTED_HELDOUT = {
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
}
M, WINDOW = 24, 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_equal(observed: object, expected: object, what: str) -> None:
    if observed != expected:
        raise ValueError(f"{what}: expected {expected!r}, found {observed!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--internal-aggregate", type=Path, required=True)
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    aggregate_path = args.internal_aggregate.resolve()
    required = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json", "metrics_per_session.csv"]
    if missing := [name for name in required if not (artifact / name).is_file()]:
        raise ValueError(f"{artifact}: missing {missing}")
    internal = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if internal.get("formal_heldout_evaluated") is not False or not internal.get("gate", {}).get("all_three_pass"):
        raise ValueError("a passed, held-in-only internal gate is required")
    cell = internal.get("cell", {})
    require_equal((cell.get("task"), cell.get("fold"), cell.get("seed"), cell.get("M")), ("m2", 1, 42, M), "internal cell")
    source = internal.get("arms", {}).get(args.group, {}).get("checkpoint", {})
    source_path = Path(source.get("path", "")).resolve()
    source_sha = source.get("sha256")
    if not source_path.is_file() or sha256(source_path) != source_sha:
        raise ValueError("frozen source checkpoint receipt does not verify")

    cfg = yaml.safe_load((artifact / "resolved_config.yaml").read_text(encoding="utf-8"))
    data, model = cfg["data"], cfg["model"]
    variant, side = GROUPS[args.group]
    checks = {
        "data.task": (data.get("task"), "m2"), "data.loso_fold": (data.get("loso_fold"), 1),
        "seed": (cfg.get("seed"), 42), "model.variant": (model.get("variant"), variant),
        "data.side_feature_group": (data.get("side_feature_group"), side),
        "data.calibration_n_trials": (data.get("calibration_n_trials"), M),
        "data.random_calibration": (data.get("random_calibration"), False),
        "data.include_heldout_in_fit": (data.get("include_heldout_in_fit"), False),
        "data.include_heldout_in_test": (data.get("include_heldout_in_test"), True),
        "data.query_start_trial": (data.get("query_start_trial"), M),
        "train": (cfg.get("train"), False), "test": (cfg.get("test"), True),
        "ckpt_path": (str(Path(str(cfg.get("ckpt_path"))).resolve()), str(source_path)),
    }
    for name, (observed, expected) in checks.items():
        require_equal(observed, expected, name)

    split = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
    require_equal(split.get("heldout_evaluated_in_fit"), False, "heldout_evaluated_in_fit")
    require_equal(split.get("heldout_evaluated_in_test"), True, "heldout_evaluated_in_test")
    require_equal(split.get("query_start_trial"), M, "split query_start_trial")
    audit = split.get("heldout_query_window_audit")
    if not isinstance(audit, dict) or set(audit) != EXPECTED_HELDOUT:
        raise ValueError(f"expected exactly six M2 held-out window audits, found {sorted(audit or {})}")
    for session, row in audit.items():
        if not isinstance(row, dict):
            raise ValueError(f"{session}: malformed query audit")
        for key, value in {"support_trials": M, "query_start_trial": M, "window_size": WINDOW, "full_window_disjoint": True}.items():
            require_equal(row.get(key), value, f"{session}.{key}")
        if int(row.get("query_trials", 0)) <= 0 or int(row.get("eligible_windows", 0)) <= 0:
            raise ValueError(f"{session}: held-out query is empty")
        # With 49 left-padding samples for a 50-bin window, first legal padded
        # window start must be exactly the raw support boundary + 49.
        if row.get("minimum_window_start_padded_bin") != row.get("raw_query_start_bin", -WINDOW) + WINDOW - 1:
            raise ValueError(f"{session}: temporal-history boundary is not exact")

    checkpoint = json.loads((artifact / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if checkpoint.get("source_checkpoint_sha256") != source_sha or checkpoint.get("artifact_checkpoint_sha256") != source_sha:
        raise ValueError("test-only checkpoint content does not match the frozen source checkpoint")
    if args.group in {"k4", "ks4"}:
        est, norm = split.get("k4_estimator", {}), split.get("native_k4_normalization", {})
        require_equal(est.get("calibration_trials"), M, "K4 calibration trials")
        require_equal(norm.get("feature_group"), args.group, "K4 normalization group")
        heldout = split.get("heldout_k4_calibration_audit")
        if not isinstance(heldout, dict) or set(heldout) != EXPECTED_HELDOUT:
            raise ValueError("K4 held-out calibration receipt must cover the exact six sessions")
        for session, row in heldout.items():
            if row.get("calibration_trials") != M or row.get("design_rank") != 3 or row.get("active_blocks", 0) <= 2:
                raise ValueError(f"{session}: invalid K4 M24 fit audit")
            if not math.isfinite(float(row.get("design_condition", float("inf")))):
                raise ValueError(f"{session}: invalid K4 design conditioning audit")
            for key, value in {"raw_bin_ms": 20, "block_width_bins": 5, "behavior_lead_bins": 2, "max_trial_length_used": False}.items():
                require_equal(row.get(key), value, f"{session}.{key}")
    if args.group == "t4":
        require_equal(split.get("native_t4_normalization", {}).get("feature_group"), "t4", "T4 normalization group")

    out = artifact / "heldout_m24_provenance.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    payload = {
        "schema_version": 1,
        "purpose": "M2_M24_local_heldout_test_only_chronological_disjoint_replay",
        "formal_heldout_evaluated": True,
        "hidden_evalai_evaluated": False,
        "group": args.group,
        "cell": {"task": "m2", "fold": 1, "seed": 42, "calibration_trials": M, "window_size": WINDOW},
        "test_only": True,
        "no_backward_or_checkpoint_selection_on_heldout": True,
        "internal_gate": {"path": str(aggregate_path), "sha256": sha256(aggregate_path)},
        "frozen_source_checkpoint": {"path": str(source_path), "sha256": source_sha},
        "heldout_sessions": sorted(audit),
        "query_window_audit": audit,
        "label_information": {
            "F0": "no calibration target labels used by the identity feature",
            "T4": "uses first-24 trial target-direction labels",
            "K4_KS4": "uses first-24 trial continuous finger-velocity samples; this is not label-information matched to T4",
        },
        "evaluation_disclosure": "This is a local held-out-calibration-file chronological support/query replay. These sessions were visible to the development process; it is not a hidden EvalAI/challenge test.",
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
