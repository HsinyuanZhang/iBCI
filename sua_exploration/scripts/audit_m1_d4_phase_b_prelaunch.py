#!/usr/bin/env python3
"""Run the real-data D4/DS4 attachment audit before the Phase-B GPU launch."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
OUT = ROOT / "sua_exploration/results/m1_d4_pilot_v1/prelaunch/runtime_attachment_audit.json"
if str(SCE) not in sys.path:
    sys.path.insert(0, str(SCE))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(group: str) -> Path:
    root = Path(f"/tmp/m1_d4_phase_b_cpu_dryrun_{group}/artifacts")
    matches = sorted(root.glob(f"m1_d4_phase_b_cpu_dryrun_{group}_f1_s42_*"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one CPU dry-run artifact for {group}, found {len(matches)}")
    return matches[0]


def row_multiset(values: np.ndarray) -> Counter[tuple[float, ...]]:
    return Counter(tuple(float(x) for x in row) for row in np.asarray(values))


def label_counts(dataset) -> dict[str, dict[str, int]]:
    answer = {}
    for session, labels in sorted(dataset.calib_trial_obj_ids.items()):
        first_ten = np.asarray(labels[:10], dtype=np.int64)
        if first_ten.shape != (10,):
            raise ValueError(f"{session}: first-ten D4 labels have shape {first_ten.shape}")
        counts = {str(level): int((first_ten == level).sum()) for level in (1, 2, 3, 4)}
        if any(value == 0 for value in counts.values()):
            raise ValueError(f"{session}: D4 first-ten support lacks a required level: {counts}")
        answer[session] = counts
    return answer


def load(group: str):
    dry_artifact = artifact(group)
    cfg = OmegaConf.load(dry_artifact / "resolved_config.yaml")
    # The archived resolved config retains the original relative data path;
    # resolve it against the streaming project rather than this audit script.
    cfg.data.data_dir = str(ROOT / "SPINT-main/data/000941/")
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    manifest = datamodule.get_split_manifest()
    side = np.asarray(datamodule.val_heldin_dataset[0][-1], dtype=np.float32)
    return dry_artifact, datamodule, manifest, side


def main() -> None:
    d4_artifact, d4_dm, d4_manifest, d4_side = load("d4")
    ds4_artifact, ds4_dm, ds4_manifest, ds4_side = load("ds4")
    if d4_side.shape != ds4_side.shape or d4_side.ndim != 2 or d4_side.shape[1] != 4:
        raise ValueError(f"D4/DS4 side shape mismatch: {d4_side.shape} vs {ds4_side.shape}")
    if np.array_equal(d4_side, ds4_side):
        raise ValueError("DS4 attachment is identity on the real validation session")
    if row_multiset(d4_side) != row_multiset(ds4_side):
        raise ValueError("DS4 changed the real D4 row multiset instead of only attachment")

    d4_norm = d4_manifest["native_d4_normalization"]
    ds4_norm = ds4_manifest["native_d4_normalization"]
    if d4_norm["train_sessions"] != ds4_norm["train_sessions"]:
        raise ValueError("D4/DS4 train session sets differ")
    if not np.array_equal(np.asarray(d4_norm["mean"]), np.asarray(ds4_norm["mean"])):
        raise ValueError("D4/DS4 normalization mean differs")
    if not np.array_equal(np.asarray(d4_norm["std"]), np.asarray(ds4_norm["std"])):
        raise ValueError("D4/DS4 normalization std differs")
    if d4_manifest["heldin_query_window_audit"] != ds4_manifest["heldin_query_window_audit"]:
        raise ValueError("D4/DS4 query window audit differs")
    if d4_manifest["d4_estimator"]["shuffle"] != "none":
        raise ValueError("D4 estimator unexpectedly declares a shuffle")
    if "nonidentity" not in ds4_manifest["d4_estimator"]["shuffle"]:
        raise ValueError("DS4 estimator does not declare a nonidentity row shuffle")

    record = {
        "schema_version": "m1_d4_phase_b_real_data_attachment_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cell": {"fold": 1, "seed": 42, "left_out_session": "ses-20120926"},
        "data_source": "real held-in calibration NWBs loaded by FalconDataModule during CPU preflight",
        "train_sessions": d4_manifest["train_sessions"],
        "validation_sessions": d4_manifest["validation_sessions"],
        "first10_obj_id_level_counts": {
            "d4_train_dataset": label_counts(d4_dm.train_dataset),
            "d4_validation_calibration": label_counts(d4_dm.val_heldin_dataset),
        },
        "query_window_audit": d4_manifest["heldin_query_window_audit"],
        "normalization": {
            "train_only_sessions": d4_norm["train_sessions"],
            "mean": d4_norm["mean"],
            "std": d4_norm["std"],
            "numeric_d4_ds4_equal": True,
            "d4_manifest_sha256": d4_norm["sha256"],
            "ds4_manifest_sha256": ds4_norm["sha256"],
            "manifest_sha_diff_reason": "feature_group is included in the manifest hash (d4 versus ds4), while mean/std are exactly equal",
        },
        "attachment_control": {
            "feature_shape": list(d4_side.shape),
            "d4_ds4_matrix_identical": False,
            "complete_row_multiset_equal": True,
            "d4_shuffle": d4_manifest["d4_estimator"]["shuffle"],
            "ds4_shuffle": ds4_manifest["d4_estimator"]["shuffle"],
        },
        "cpu_preflight_inputs": {
            "d4_resolved_config_sha256": sha(d4_artifact / "resolved_config.yaml"),
            "ds4_resolved_config_sha256": sha(ds4_artifact / "resolved_config.yaml"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing runtime audit: {OUT}")
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
