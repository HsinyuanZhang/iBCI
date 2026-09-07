#!/usr/bin/env python3
"""Write the fail-closed provenance receipt before the M1 D4 pilot launches."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
RESULT = ROOT / "sua_exploration/results/m1_d4_pilot_v1/phase_b_receipt.json"
PRELAUNCH = ROOT / "sua_exploration/results/m1_d4_pilot_v1/prelaunch"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"required file is missing: {path}")
    try:
        displayed_path = str(path.relative_to(ROOT))
    except ValueError:
        displayed_path = str(path)
    return {"path": displayed_path, "sha256": sha(path)}


def unique_artifact(pattern: str) -> Path:
    paths = sorted((SCE / "outputs/streaming_calibration").glob(pattern))
    if len(paths) != 1:
        raise SystemExit(f"expected exactly one artifact for {pattern}, found {len(paths)}")
    return paths[0]


def main() -> None:
    replace_prelaunch = sys.argv[1:] == ["--replace-prelaunch"]
    if sys.argv[1:] and not replace_prelaunch:
        raise SystemExit("usage: write_m1_d4_phase_b_receipt.py [--replace-prelaunch]")
    source = SCE / "src/data/falcon_datamodule.py"
    source_text = source.read_text()
    required_markers = [
        "{'none', 't4', 'ts4', 'd4', 'ds4', 'k4', 'ks4'}",
        "calibration_obj_id_labels",
        "fit_train_d4_stats",
        "deterministic_d4_row_permutation",
        "D4 only supports chronological calibration trials[0:10]",
        "query_labels_used",
    ]
    absent = [marker for marker in required_markers if marker not in source_text]
    if absent:
        raise SystemExit(f"D4 integration is incomplete; missing source markers: {absent}")

    baseline = {}
    for group in ("f0", "t4"):
        artifact = unique_artifact(f"m1_clean_selection_v1_{group}_m1_f1_s42_*")
        ckpt = artifact / "checkpoints/best.ckpt"
        baseline[group] = {
            "artifact": str(artifact.relative_to(ROOT)),
            "clean_best_checkpoint": file_record(ckpt),
        }

    cpu_dry_runs = {}
    for group in ("d4", "ds4"):
        dry_root = PRELAUNCH / group
        cpu_dry_runs[group] = {
            "command_contract": "CPU, max_epochs=0, two Lightning sanity-validation batches; no optimizer step",
            "run_log": file_record(dry_root / "run.log"),
            "resolved_config": file_record(dry_root / "resolved_config.yaml"),
            "split_manifest": file_record(dry_root / "split_manifest.json"),
            "source_manifest": file_record(dry_root / "source_manifest.json"),
        }

    protocol = ROOT / "sua_exploration/docs/M1_D4_MINIMAL_GPU_PILOT_PROTOCOL.md"
    phase_a = ROOT / "sua_exploration/results/m1_d4_pilot_v1/protocol_receipt.json"
    clean_protocol = ROOT / "sua_exploration/results/m1_clean_selection_v1/protocol_receipt.json"
    clean_report = ROOT / "sua_exploration/results/m1_clean_selection_v1/report.md"
    records = [
        source,
        SCE / "src/data/falcon_d4_features.py",
        SCE / "tests/test_falcon_d4_features.py",
        SCE / "tests/test_falcon_d4_datamodule_integration.py",
        SCE / "configs/data/falcon_m1.yaml",
        SCE / "configs/experiment/b3s_d4_m1_loso_internal.yaml",
        SCE / "configs/experiment/b3s_ds4_m1_loso_internal.yaml",
        SCE / "configs/experiment/m1_d4_pilot_d4.yaml",
        SCE / "configs/experiment/m1_d4_pilot_ds4.yaml",
        ROOT / "sua_exploration/scripts/run_m1_d4_pilot_one_arm.sh",
        ROOT / "sua_exploration/scripts/write_m1_d4_phase_b_receipt.py",
        ROOT / "sua_exploration/scripts/audit_m1_d4_phase_b_prelaunch.py",
        protocol,
        phase_a,
        clean_protocol,
        clean_report,
    ]
    receipt = {
        "schema_version": "m1_d4_phase_b_receipt_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prelaunch_checks_passed_gpu_launch_requires_root_review",
        "isolation": {
            "sealed_clean_selection_artifacts_modified": False,
            "sealed_report_window_evaluated": False,
            "hidden_or_evalai_accessed": False,
            "authorized_cell": {"fold": 1, "seed": 42, "left_out_session": "ses-20120926"},
            "authorized_arms": ["d4", "ds4"],
        },
        "frozen_windows": {
            "support": [0, 10],
            "selection": [10, 210],
            "report": [210, None],
            "report_access": "forbidden until D4 and DS4 training/checkpoint selection complete",
        },
        "d4_contract": {
            "task": "m1",
            "calibration_labels": "calibration NWB trials.obj_id only",
            "query_labels": "not loaded or used",
            "required_levels": [1, 2, 3, 4],
            "feature": "per-channel exposure-corrected category mean rates [mu_1,mu_2,mu_3,mu_4]",
            "normalization": "fit on train-session D4 rows only",
            "ds4": "post-normalization deterministic complete-row nonidentity permutation keyed by session and seed",
        },
        "current_datamodule_pre_additive_baseline_sha256": "4f71cb7e47d3993073bd151caa05f7a881cc3e2dccfd06aef87bb3ff2068961e",
        "baseline_anchors": baseline,
        "cpu_dry_runs": cpu_dry_runs,
        "real_data_attachment_audit": file_record(PRELAUNCH / "runtime_attachment_audit.json"),
        "files": [file_record(path) for path in records],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        old = json.loads(RESULT.read_text())
        old_files = old.get("files")
        if (
            old_files == receipt["files"]
            and old.get("baseline_anchors") == baseline
            and old.get("cpu_dry_runs") == cpu_dry_runs
        ):
            print(f"unchanged receipt: {RESULT}")
            return
        if not replace_prelaunch or old.get("status") != "prelaunch_checks_passed_gpu_launch_requires_root_review":
            raise SystemExit(f"refusing to overwrite a changed Phase-B receipt: {RESULT}")
    RESULT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(RESULT)


if __name__ == "__main__":
    main()
