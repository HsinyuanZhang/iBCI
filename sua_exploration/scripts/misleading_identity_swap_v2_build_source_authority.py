#!/usr/bin/env python3
"""Build the immutable strict-27/M30 hidden-descriptor authority (CPU/source only)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
for entry in (SUA_ROOT, REPO_ROOT / "streaming_calibration_exp"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mc_maze import a2_matched_subject_shift_v2_core as a2
from mc_maze import misleading_identity_swap_v2_core as core
from mc_maze.multisession_datamodule import session_name_from_path
from mc_maze.unit_side_features import (
    feature_semantics_version,
    fit_side_feature_stats,
    load_unit_side_features,
    side_feature_stats_sha256,
)

DEFAULT_ROOT = SUA_ROOT / "results" / "misleading_identity_swap_v2_source_authority_dev"
DEFAULT_AUTHORITY = DEFAULT_ROOT / "strict27_m30_matching_authority_v3.json"
DEFAULT_LINEAGE = DEFAULT_ROOT / "strict27_m30_source_lineage_v3.json"
A2_REFERENCE_METADATA = (
    SUA_ROOT / "checkpoints" /
    "a2_matched_subject_shift_v2_source_t4_dandi688_co_s42" /
    "run_metadata.json"
)


def strict_train_paths() -> tuple[list[str], list[Path], list[str], list[str]]:
    """Resolve only the 27 source-train files; val/test remain inert names."""
    manifest = a2.load_strict_manifest()
    root = a2.SUBC_DATA_ROOT.resolve()
    paths: list[Path] = []
    for session in manifest["train"]:
        path = (root / f"{session}_behavior+ecephys.nwb").resolve()
        core.require(path.parent == root and path.is_file(), f"missing strict source NWB: {path}")
        paths.append(path)
    return list(manifest["train"]), paths, list(manifest["val"]), list(manifest["test"])


def _a2_reference_normalizer_sha() -> tuple[str, str]:
    payload = core.load_json_object(A2_REFERENCE_METADATA)
    side = payload.get("side_features")
    core.require(isinstance(side, dict), "A2 T4 reference metadata lacks side_features")
    core.require(side.get("group") == "t4" and side.get("pool_size") == 30,
                 "A2 T4 reference substrate drift")
    digest = side.get("normalization_sha256")
    core.require(isinstance(digest, str) and len(digest) == 64,
                 "A2 T4 reference normalizer SHA malformed")
    return digest, core.sha256_file(A2_REFERENCE_METADATA)


def build_real_source_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    train_names, train_paths, val_names, formal_names = strict_train_paths()
    mean, std = fit_side_feature_stats(
        train_paths,
        feature_group="t4",
        pool_size=30,
        cache_dir=a2.SOURCE_CACHE_ROOT,
        bin_size_ms=20,
        window_size=50,
        trial_result_filter="R",
        signal_view="sua",
    )
    normalizer_sha = side_feature_stats_sha256(mean, std)
    reference_sha, reference_metadata_sha = _a2_reference_normalizer_sha()
    core.require(normalizer_sha == reference_sha,
                 "strict-27 M30 normalizer differs from frozen matched A2 T4 authority")
    descriptors: dict[str, torch.Tensor] = {}
    source_files: list[dict[str, Any]] = []
    for expected, path in zip(train_names, train_paths, strict=True):
        core.require(session_name_from_path(path) == expected, "strict source session/path drift")
        values, metadata = load_unit_side_features(
            path,
            feature_group="t4",
            pool_size=30,
            mean=mean,
            std=std,
            cache_dir=a2.SOURCE_CACHE_ROOT,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
            signal_view="sua",
        )
        core.require(values.ndim == 2 and values.shape[1] == 4,
                     f"{expected}: normalized M30 descriptor shape drift")
        core.require(np.isfinite(values).all(), f"{expected}: non-finite M30 descriptor")
        descriptors[expected] = torch.as_tensor(values, dtype=torch.float64)
        source_files.append({
            "session": expected,
            "path": str(path),
            "sha256": core.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "feature_version": int(metadata.feature_version),
            "unit_count": int(values.shape[0]),
        })
    core.require(list(descriptors) == train_names, "strict source authority ordering drift")
    authority = core.build_matching_authority(descriptors)
    lineage = {
        "schema_version": 1,
        "receipt_kind": "misleading_identity_swap_v2_strict27_m30_source_lineage",
        "status": "DEVELOPMENT_SOURCE_ONLY_AUTHORITY_BUILT__NOT_OFFICIAL",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_id": core.SCREEN_ID,
        "source_only": True,
        "strict_manifest_path": str(a2.MANIFEST_PATH.resolve()),
        "strict_manifest_sha256": core.sha256_file(a2.MANIFEST_PATH),
        "source_train_sessions": train_names,
        "source_train_session_count": 27,
        "validation_session_names_only": val_names,
        "formal_test_session_names_only": formal_names,
        "validation_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "source_files": source_files,
        "descriptor": {
            "columns": list(core.DESCRIPTOR_COLUMNS),
            "feature_group": "t4",
            "feature_semantics_version": feature_semantics_version("t4"),
            "pool_size": 30,
            "support": "chronological_first_30_rewarded_trials",
            "bin_size_ms": 20,
            "window_size_bins": 50,
            "signal_view": "sua",
            "normalizer_mean_float32": np.asarray(mean, dtype=np.float32).tolist(),
            "normalizer_std_float32": np.asarray(std, dtype=np.float32).tolist(),
            "normalizer_value_sha256": normalizer_sha,
            "frozen_a2_reference_normalizer_value_sha256": reference_sha,
            "frozen_a2_reference_metadata_path": str(A2_REFERENCE_METADATA.resolve()),
            "frozen_a2_reference_metadata_sha256": reference_metadata_sha,
            "cache_root": str(a2.SOURCE_CACHE_ROOT.resolve()),
        },
        "matching_authority_canonical_sha256": core.canonical_json_sha256(authority),
        "t4_z4_shared_authority_required": True,
        "target_nwb_opened": False,
        "gpu_used": False,
        "implementation_bindings": {
            name: {"path": str(path.resolve()), "sha256": core.sha256_file(path)}
            for name, path in {
                "config": core.CONFIG_PATH,
                "contract": core.CONTRACT_PATH,
                "core": SUA_ROOT / "mc_maze" / "misleading_identity_swap_v2_core.py",
                "source_builder": Path(__file__).resolve(),
                "shared_t4_features": SUA_ROOT / "mc_maze" / "unit_side_features.py",
                "shared_a2_core": SUA_ROOT / "mc_maze" / "a2_matched_subject_shift_v2_core.py",
                "strict_manifest": a2.MANIFEST_PATH,
            }.items()
        },
    }
    return authority, lineage


def execute(authority_path: Path, lineage_path: Path) -> dict[str, Any]:
    for path in (authority_path, core.sidecar_path(authority_path), lineage_path, core.sidecar_path(lineage_path)):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"source authority output is not fresh: {path}")
    authority, lineage = build_real_source_payloads()
    authority_body, _authority_side, authority_sha = core.write_immutable_json_pair(authority_path, authority)
    lineage["matching_authority_path"] = str(authority_body.resolve())
    lineage["matching_authority_consumed_bytes_sha256"] = authority_sha
    lineage_body, _lineage_side, lineage_sha = core.write_immutable_json_pair(lineage_path, lineage)
    return {
        "status": "SOURCE_ONLY_CPU_SMOKE_PASS__DEVELOPMENT_NOT_OFFICIAL",
        "authority": str(authority_body),
        "authority_sha256": authority_sha,
        "lineage": str(lineage_body),
        "lineage_sha256": lineage_sha,
        "source_sessions_opened": 27,
        "validation_nwb_opened": False,
        "target_nwb_opened": False,
        "formal_subc_test_nwb_opened": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--execute-source-only", action="store_true")
    args = parser.parse_args()
    if not args.execute_source_only:
        print(json.dumps({
            "status": "DRY_RUN__NO_DATA_NO_GPU",
            "source_session_count": 27,
            "authority": str(args.authority.resolve()),
            "lineage": str(args.lineage.resolve()),
            "target_nwb_opened": False,
            "formal_subc_test_nwb_opened": False,
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(execute(args.authority, args.lineage), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
