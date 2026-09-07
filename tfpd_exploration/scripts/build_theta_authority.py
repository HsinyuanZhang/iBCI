#!/usr/bin/env python3
"""Build the immutable source-only RAW-T4 theta authority (workorder §6.1).

For every strict-27 train session, computes theta = atan2(raw_c, raw_a) from
the UN-NORMALIZED closed-form T4 ([m*cos(phi), m*sin(phi), m, b]) in canonical
unit order, plus the direction-validity mask raw_m > MODULATION_EPS.  The
authority is sealed (0444 + sidecar) and bound by an alignment proof: per
session, n_units equals the training datamodule's neural channel count and
the standardized side-feature row count (canonical unit order), and theta is
shown to be invariant to the normalizer (recomputing from inverted
standardization would be the forbidden path; we never take it).

Theta is a MASKING AUTHORITY ONLY — it is never concatenated into any
model-visible token.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(REPO / "sua_exploration"))

BOUND_PATTERNS = (
    "src/tfpd_lane/sparsification.py",
    "scripts/build_theta_authority.py",
    "src/tfpd_lane/receipt.py",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/sparsification_theta_authority_v1")
    args = parser.parse_args()
    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    out_dir = Path(args.output_root)
    if out_dir.exists():
        print(f"fresh output root required: {out_dir}", file=sys.stderr)
        return 2

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    sparsification = _load_module(
        "tfpd_lane_sparsification", ROOT / "src/tfpd_lane/sparsification.py"
    )
    arm_runner = _load_module(
        "tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py"
    )
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True)
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    train_paths, _val, _test = a2.active_source_session_paths()
    authority = sparsification.build_theta_authority(train_paths)

    # ---- alignment proof against the training datamodule (train-only) ------
    ns = type("NS", (), {"train_batch_size": 32, "num_workers": 0, "seed": 42})()
    dm, _a2 = arm_runner.build_datamodule(ns)
    alignment = {}
    for name, entry in sorted(authority.items()):
        record = dm.train_dataset.sessions[name]
        alignment[name] = {
            "theta_units": entry["n_units"],
            "datamodule_channels": int(record.neural.shape[1]),
            "side_rows": int(record.side_features.shape[0]),
            "aligned": bool(
                entry["n_units"] == record.neural.shape[1]
                == record.side_features.shape[0]
            ),
            "n_valid_directions": int(entry["valid"].sum()),
            "n_undefined": int((~entry["valid"]).sum()),
        }
    if not all(row["aligned"] for row in alignment.values()):
        raise SystemExit("theta authority failed canonical-unit-order alignment")
    if any(row["n_valid_directions"] < 2 for row in alignment.values()):
        raise SystemExit("session with fewer than two valid directions")

    artifact = out_dir / "theta_authority.pt"
    payload = {
        "kind": "tfpd_sparsification_theta_authority_v1",
        "authority": {
            name: {
                "theta": torch.from_numpy(entry["theta"]),
                "valid": torch.from_numpy(entry["valid"]),
                "n_units": entry["n_units"],
                "raw_t4_sha256": entry["raw_t4_sha256"],
            }
            for name, entry in authority.items()
        },
        "authority_sha256": sparsification.authority_sha256(authority),
        "rule": "theta = atan2(raw_c, raw_a) from un-normalized T4; valid = raw_m > MODULATION_EPS; never z-scored; masking authority only",
    }
    torch.save(payload, artifact)
    os.chmod(artifact, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    artifact_sha = receipt_mod.sha256_file(artifact)
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    sidecar.write_text(artifact_sha + "  " + artifact.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    receipt = {
        "schema": "tfpd_sparsification_theta_authority_v1",
        "status": "THETA_AUTHORITY_SEALED",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": {"path": str(artifact), "sha256": artifact_sha},
        "authority_sha256": payload["authority_sha256"],
        "rule": payload["rule"],
        "n_sessions": len(authority),
        "alignment_proof": alignment,
        "disclosures": {
            "theta_model_visible": False,
            "z_scored_values_used": False,
            "within_dev_or_external_opened": False,
            "formal_or_organizer_held_data_opened": False,
        },
        "source_closure": closure,
        "environment": {"no_user_site": bool(sys.flags.no_user_site)},
    }
    receipt_mod.write_receipt_transactionally(out_dir / "theta_authority_receipt.json", receipt)
    print(json.dumps({
        "status": receipt["status"],
        "n_sessions": len(authority),
        "authority_sha256": payload["authority_sha256"][:16],
        "valid_directions_total": sum(r["n_valid_directions"] for r in alignment.values()),
        "undefined_total": sum(r["n_undefined"] for r in alignment.values()),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
