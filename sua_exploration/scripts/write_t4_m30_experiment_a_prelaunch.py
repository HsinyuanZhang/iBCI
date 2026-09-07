#!/usr/bin/env python3
"""CPU-only, write-once prelaunch receipt for Experiment A's 18 M30 component runs.

This program intentionally contains no training, checkpoint loading, scoring, or formal-SUA
access.  It fits only PH4's source-only normalizer on the 27 strict source sessions, binds the
already-qualified M30 v2 receipt and ordinary-T4 normalizer, and rejects any existing planned
run/checkpoint/result path before a future runner can be authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
MANIFEST = SUA / "configs/subc_co_27_6_strict_train_val_manifest.json"
TEACHER = SUA / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
M30_V2 = SUA / "results/t4_m30_experiment_a_cpu_preflight_v2_20260802/receipt.json"
ORDINARY_T4_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
SEEDS = (42, 43, 44)
ARMS = ("z4", "ph4", "ac4", "mb4", "b4", "ls4")
SCREEN_ID = "sua_t4_m30_component_attribution_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def planned_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            name = f"{SCREEN_ID}_{arm}_dandi688_co_s{seed}"
            checkpoint_dir = SUA / "checkpoints" / name
            result_path = SUA / "results" / SCREEN_ID / f"{arm}_s{seed}.json"
            collision = [str(path.relative_to(REPO)) for path in (checkpoint_dir, result_path) if path.exists()]
            require(not collision, f"exclusive run-path collision for {arm}@{seed}: {collision}")
            rows.append({
                "arm": arm.upper(), "seed": seed, "variant": "B3S", "side_dim": 4,
                "checkpoint_dir": str(checkpoint_dir.relative_to(REPO)),
                "result_path": str(result_path.relative_to(REPO)),
                "collision_free": True,
            })
    return rows


def fit_phase_normalizer(manifest: dict[str, Any]) -> dict[str, Any]:
    import sys
    sys.path.insert(0, str(SUA))
    from mc_maze.unit_side_features import fit_side_feature_stats, side_feature_stats_sha256

    train_files = [SUA / "data/dandi_000688/sub-C" / f"{name}_behavior+ecephys.nwb" for name in manifest["session_splits"]["train"]]
    require(len(train_files) == 27 and all(path.is_file() for path in train_files), "strict source files are incomplete")
    mean, std = fit_side_feature_stats(train_files, feature_group="ph4", pool_size=30, cache_dir=None)
    require(mean.shape == (4,) and std.shape == (4,), f"PH4 normalizer shape is not [4]: {mean.shape}/{std.shape}")
    require(float(mean[2]) == 0.0 and float(mean[3]) == 0.0, "PH4 zero-padded raw columns must retain zero source mean")
    require(float(std[2]) == 1.0 and float(std[3]) == 1.0, "PH4 zero-padded raw columns must retain unit source std")
    return {"feature_group": "ph4", "pool_size": 30, "fit_scope": "27 strict source sessions only", "sha256": side_feature_stats_sha256(mean, std), "mean": mean.tolist(), "std": std.tolist()}


def build_receipt() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    v2 = json.loads(M30_V2.read_text())
    require(v2.get("eligibility", {}).get("status") == "pass", "M30 v2 receipt is not a pass")
    require(v2.get("execution_scope", {}).get("formal_sua_nwb_opened") is False, "M30 v2 receipt formal scope drift")
    require(manifest.get("split_counts") == [27, 6, 6], "strict manifest drift")
    phase = fit_phase_normalizer(manifest)
    source_paths = [
        SUA / "mc_maze/unit_side_features.py", SUA / "mc_maze/multisession_datamodule.py",
        SUA / "scripts/train_variant_dandi688.py", SUA / "scripts/eval_epoch_window_generic_dandi688.py",
        SUA / "scripts/select_gradient_free_protocol_dandi688.py", Path(__file__),
    ]
    return {
        "schema_version": 1,
        "audit_name": "t4_m30_experiment_a_descriptor_prelaunch",
        "execution_scope": {"cpu_only": True, "training_started": False, "gpu_used": False, "formal_sua_nwb_opened": False, "checkpoint_loaded": False, "only_source_nwbs_read_for_phase_normalizer": True},
        "m30_v2_receipt": {"path": str(M30_V2.relative_to(REPO)), "sha256": sha(M30_V2)},
        "teacher": {"path": str(TEACHER.relative_to(REPO)), "sha256": sha(TEACHER)},
        "strict_manifest": {"path": str(MANIFEST.relative_to(REPO)), "sha256": sha(MANIFEST), "session_splits": manifest["session_splits"]},
        "normalizers": {"ordinary_t4_reused_for_z4_ac4_mb4_b4_ls4": ORDINARY_T4_NORMALIZER_SHA256, "ph4_source_only": phase},
        "descriptor_contract": {
            "ordinary_t4_masks": {"Z4": "[0,0,0,0]", "AC4": "[a~,c~,0,0]", "MB4": "[0,0,m~,b~]", "B4": "[0,0,0,b~]"},
            "ph4": "raw [a/m,c/m] only when m!=0; exact m==0 -> [0,0]; source-normalize phase columns; zero-pad to width 4",
            "ls4": "deterministic session+seed support target_dir permutation; refit a/c/m; copy aligned b bitwise; ordinary T4 normalizer",
            "fixed": {"architecture": "B3S", "side_dim": 4, "activity_support": 30, "label_rate_pool": 30, "score_start": 30, "epochs": list(range(5, 13)), "seeds": list(SEEDS)},
        },
        "cost_accounting": {"learned_parameter_delta_vs_ordinary_T4": 0, "fp32_descriptor_bytes_per_session": "4 * N_units * 4", "persistent_descriptor_bytes_per_unit": 16, "temporary_estimator_state_not_counted_as_persistent": True, "training_gpu_memory": "not measured; no run launched"},
        "matrix": planned_matrix(),
        "current_source_sha256": {str(path.relative_to(REPO)): sha(path) for path in source_paths},
        "implementation_status": {"descriptor_library_implemented": True, "runner_generated": False, "model_run_launched": False},
        "eligibility": {"status": "pass", "gpu_launch_authorized": False, "next_authority_required": "explicit root authorization to generate and launch the sealed 18-run matrix"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), f"write-once receipt directory already exists: {args.output_dir}")
    receipt = build_receipt()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "RECEIPT.md").write_text("# Experiment A descriptor prelaunch\n\nStatus: PASS for CPU descriptor contract; GPU/model launch is not authorized by this receipt.\n")
    print(args.output_dir / "receipt.json")


if __name__ == "__main__":
    main()
