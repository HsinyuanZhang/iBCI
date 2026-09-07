#!/usr/bin/env python3
"""Freeze train-only T4/TS4 normalization provenance for M33 correction seals.

The 20260729 source *training* split manifests predate normalization export.
The historical M33 replay split manifests contain that train-only receipt.  This
script reads only their split manifests (never metrics CSVs) and binds them to
the frozen source checkpoint and immutable correction protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCREEN = "m2_m33_disjoint_replay_correction_v1"
PROTOCOL_SHA = "da4eb05db43401f2e314351e8882c6606cc4d020aed3ed45e860d52db9e59df1"
GROUPS = ("t4", "ts4")
CELLS = ("fold1_seed42", "fold1_seed43", "fold2_seed42")
PENDING = {
    "t4/fold1_seed42": "m2_m33_disjoint_replay_correction_v1_t4_m2_final1_f1_s42_20260801_120203",
    "t4/fold1_seed43": "m2_m33_disjoint_replay_correction_v1_t4_m2_final1_f1_s43_20260801_120203",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eq(observed: object, expected: object, what: str) -> None:
    if observed != expected:
        raise ValueError(f"{what}: expected {expected!r}, found {observed!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / f"sua_exploration/results/{SCREEN}/protocol_receipt.json"
    eq(sha256(protocol_path), PROTOCOL_SHA, "protocol receipt SHA")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    norms = {}
    for group in GROUPS:
        norms[group] = {}
        for cell in CELLS:
            frozen = protocol["frozen_source_arms"][group][cell]
            replay = Path(frozen["legacy_replay_artifact"]).resolve()
            split_path = replay / "split_manifest.json"
            if not split_path.is_file():
                raise ValueError(f"{group}/{cell}: historical replay split manifest missing")
            eq(sha256(split_path), frozen["legacy_replay_artifact_sha256"]["split_manifest.json"], f"{group}/{cell} replay split SHA")
            split = json.loads(split_path.read_text(encoding="utf-8"))
            norm = split.get("native_t4_normalization")
            if not isinstance(norm, dict):
                raise ValueError(f"{group}/{cell}: historical replay normalization receipt missing")
            eq(norm.get("feature_group"), group, f"{group}/{cell} feature group")
            norms[group][cell] = {
                "legacy_replay_split_manifest": str(split_path),
                "legacy_replay_split_manifest_sha256": sha256(split_path),
                "source_checkpoint": frozen["checkpoint"],
                "native_t4_normalization": norm,
            }
    pending = {}
    for key, name in PENDING.items():
        artifact = root / "streaming_calibration_exp/outputs/streaming_calibration" / name
        required = ["resolved_config.yaml", "split_manifest.json", "checkpoint_manifest.json", "metrics_per_session.csv"]
        if missing := [x for x in required if not (artifact / x).is_file()]:
            raise ValueError(f"{key}: pending artifact missing {missing}")
        pending[key] = {
            "path": str(artifact),
            "hashes": {item: sha256(artifact / item) for item in required if item != "metrics_per_session.csv"},
            "score_policy": "forbidden_until_a_future_validator_seal_after_this_addendum_is_verified",
        }
    payload = {
        "schema_version": 1,
        "purpose": "M2_M33_correction_T4_TS4_train_normalization_parity_addendum",
        "protocol": {"path": str(protocol_path), "sha256": PROTOCOL_SHA},
        "source_training_split_limitation": "The frozen 20260729 source training split manifests do not export native_t4_normalization. This addendum binds the same frozen source checkpoints to their historical replay split receipts, which do export the train-only moments.",
        "norms": norms,
        "pending_unsealed_artifacts": pending,
        "data_access_disclosure": "This script reads only protocol JSON, replay split_manifest.json, and pending artifact config/split/checkpoint manifests. It does not open metrics CSVs, aggregates, scores, predictions, or any historical M33 result values.",
        "seal_rule": "Current T4/TS4 split native_t4_normalization must equal the corresponding addendum receipt exactly; T4 and TS4 must additionally match mean/std/train_sessions within each cell. Pending artifacts remain score-forbidden until independently sealed.",
    }
    out = args.out or root / f"sua_exploration/results/{SCREEN}/normalization_parity_addendum_v1.json"
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable addendum {out}")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
    print(sha256(out))


if __name__ == "__main__":
    main()
