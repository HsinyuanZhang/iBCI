#!/usr/bin/env python3
"""Finalize the immutable receipt for an M2 held-in-only clean teacher.

The receipt is written only after the selected checkpoint exists.  It is not a
training helper: it validates the embedded runtime provenance, the one
held-in-only checkpoint selector, the frozen external manifest, and binds the
resolved Hydra config plus source-file hashes for downstream student runs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = ROOT / "streaming_calibration_exp"
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from src.utils.clean_teacher_validation import (  # noqa: E402
    PROTOCOL,
    sha256_file,
    validate_checkpoint_provenance,
    validate_clean_teacher_receipt,
)


def source_manifest() -> dict[str, object]:
    files = [
        ROOT / "SPINT-main/src/data/falcon_datamodule.py",
        ROOT / "SPINT-main/src/models/falcon_module.py",
        ROOT / "SPINT-main/src/train.py",
        ROOT / "SPINT-main/src/callbacks/clean_teacher_provenance.py",
        ROOT / "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        ROOT / "streaming_calibration_exp/src/utils/clean_teacher_validation.py",
    ]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("clean teacher source manifest has a missing required file")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    return {
        "git_revision": revision,
        "files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in files],
    }


def parse_strace_inputs(trace_path: Path, runtime_manifest: dict[str, object]) -> dict[str, object]:
    """Independently certify the trace's exact NWB path set, not caller claims."""
    text = trace_path.read_text(errors="replace")
    traced = {str(Path(value).resolve()) for value in re.findall(r'"([^"\n]+\.nwb)"', text)}
    rows = runtime_manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("runtime manifest has no file rows for strace comparison")
    expected = {str(Path(str(row.get("path", ""))).resolve()) for row in rows if isinstance(row, dict)}
    if len(expected) != 14 or traced != expected:
        raise ValueError("strace NWB paths do not exactly equal the fourteen runtime-manifest inputs")
    forbidden = runtime_manifest.get("forbidden_heldout_sessions")
    if not isinstance(forbidden, list):
        raise ValueError("runtime manifest has no forbidden held-out session list")
    forbidden_hits = sum(text.count(token) for token in ["held-out", *map(str, forbidden)])
    if forbidden_hits:
        raise ValueError("strace contains a held-out path/session token")
    roles = {"heldin_calib": 0, "heldin_minival": 0}
    for row in rows:
        if not isinstance(row, dict) or row.get("role") not in roles:
            raise ValueError("runtime manifest has an unexpected input role")
        roles[row["role"]] += 1
    if roles != {"heldin_calib": 7, "heldin_minival": 7}:
        raise ValueError("runtime manifest roles are not the expected 7+7 split")
    return {"unique_nwb_count": len(traced), "roles": roles, "heldout_hits": forbidden_hits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strace-path", type=Path, required=True)
    parser.add_argument("--strace-sha256", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite clean teacher receipt: {output}")
    if not args.resolved_config.is_file():
        raise FileNotFoundError(args.resolved_config)
    if not args.strace_path.is_file() or sha256_file(args.strace_path) != args.strace_sha256:
        raise ValueError("clean teacher strace evidence is missing or SHA-mismatched")
    facts = validate_checkpoint_provenance(args.checkpoint)
    provenance = facts["provenance"]
    strace_summary = parse_strace_inputs(args.strace_path, provenance["runtime_manifest"])
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "selected_checkpoint": {
            "path": facts["checkpoint_path"],
            "sha256": facts["checkpoint_sha256"],
            "epoch": facts["checkpoint_epoch"],
            "selection_score": facts["selection_score"],
        },
        "checkpoint_provenance_sha256": facts["provenance_sha256"],
        "callback_contract": facts["callback_contract"],
        "input_manifest": {
            "path": provenance["input_manifest_path"],
            "sha256": provenance["input_manifest_sha256"],
            "runtime_manifest_sha256": provenance["runtime_manifest_sha256"],
        },
        "resolved_config": {
            "path": str(args.resolved_config.resolve()),
            "sha256": sha256_file(args.resolved_config),
        },
        "source_manifest": source_manifest(),
        "strace_preflight": {
            "path": str(args.strace_path.resolve()),
            "sha256": args.strace_sha256,
            **strace_summary,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    # Validate the exact external artifact that downstream runs will consume.
    validate_clean_teacher_receipt(args.checkpoint, output)
    print(output)


if __name__ == "__main__":
    main()
