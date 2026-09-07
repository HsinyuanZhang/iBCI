#!/usr/bin/env python3
"""Write a static, non-authorizing sub-C scorer-adapter parity prelaunch bundle.

This program is deliberately metadata/source-only.  It does not import torch,
PynWB, the scorer modules, or a model; it never opens an NWB, checkpoint, or
normalizer NPZ and it never executes a forward or metric calculation.  It
instead seals the exact future parity fixture, source hashes, CPU policy, and
the reasons that external sub-M scoring remains unauthorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/dandi_000688_subm_co_scorer_adapter_parity_prelaunch_v1"

DOCUMENT = "sua_exploration/docs/DANDI_000688_SUBM_CO_SCORER_ADAPTER_PARITY_PROTOCOL_V1.md"
MANIFEST = "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/c1_train_val_33_manifest.json"
RUN_METADATA = "sua_exploration/results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure/shared_t4_s44/run_metadata.json"

SOURCE_PATHS = (
    "sua_exploration/mc_maze/subm_co_score_only.py",
    "sua_exploration/mc_maze/subm_co_score_only_v2.py",
    "sua_exploration/scripts/run_dandi688_subm_co_score_only_v2.py",
    "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v2.py",
    "sua_exploration/scripts/eval_paired_view_c1_epoch_window.py",
    "sua_exploration/scripts/eval_adaptation_dandi688.py",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
)

TERMINAL_CHECKPOINT = {
    "path": "sua_exploration/checkpoints/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s44/epoch_ckpts/epoch_011.ckpt",
    "sha256": "a3786023772d5099d709dbd6013812ec70108d0f8fb439ae3a901cd35da271f6",
    "bytes": 64769167,
    "arm": "shared_t4",
    "seed": 44,
    "terminal_epoch": "epoch_011.ckpt",
    "opened_by_this_prelaunch": False,
}
DEVELOPMENT_FIXTURE = {
    "split": "val",
    "selection": "first_manifest_val_entry_only",
    "session": "sub-C_ses-CO-20151103",
    "filename": "sub-C_ses-CO-20151103_behavior+ecephys.nwb",
    "sha256": "7770b3c4ae13fde65276af4d67bcd157e878e7fa4b75d858ba92d90309e14de7",
    "bytes": 62145872,
    "already_consumed_c1_development_only": True,
    "nwb_opened_by_this_prelaunch": False,
}
SOURCE_NORMALIZERS = {
    "view": "sua",
    "scope": "source_train_27_only",
    "behavior": {
        "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/behavior_stats/be50f588491c004f721e.npz",
        "sha256": "821e98bc0b884d1db1347fbcb5eb654a3e01c23405e84dadcb3ccd86944235cd",
        "bytes": 397,
    },
    "t4_side_feature": {
        "path": "sua_exploration/cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804/sua/side_feature_stats/dd3da1f59700c1b96ab8.npz",
        "sha256": "32d32a7fe1b80a139571aae0ce3c3a1d802aec21c99b23a4cff72b8a60261701",
        "bytes": 614,
        "semantic_sha256": "ac5156097864110685e0b2fbfe314edcb747e69dc821c10451984a089be8a7a7",
    },
    "opened_by_this_prelaunch": False,
    "fitting_calls_by_this_prelaunch": 0,
}
PARITY_SCHEMA = "dandi_000688_subc_scorer_adapter_parity_receipt_v1"


class StaticParityError(RuntimeError):
    """Raised on source or static-authority drift before any execution is possible."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticParityError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticParityError(f"cannot parse {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} JSON root is not an object")
    return value


def static_authority(root: Path = ROOT) -> dict[str, Any]:
    """Verify only source text and JSON metadata; do not touch model/data artifacts."""
    root = root.resolve()
    source_hashes: dict[str, str] = {}
    source_text: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = (root / relative).resolve()
        require(path.is_file() and not path.is_symlink(), f"source missing or unsafe: {relative}")
        source_hashes[relative] = sha256_file(path)
        source_text[relative] = path.read_text(encoding="utf-8")

    document_path = (root / DOCUMENT).resolve()
    require(document_path.is_file() and not document_path.is_symlink(), "parity protocol document missing or unsafe")
    document_sha = sha256_file(document_path)
    document_text = document_path.read_text(encoding="utf-8")
    require(PARITY_SCHEMA in document_text, "parity protocol schema missing from document")
    require("STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED" in document_text, "parity document status drift")

    manifest_path = (root / MANIFEST).resolve()
    metadata_path = (root / RUN_METADATA).resolve()
    manifest = read_json(manifest_path, "C1 27/6 manifest")
    metadata = read_json(metadata_path, "C1 shared-T4 seed-44 run metadata")
    require(sha256_file(manifest_path) == "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb", "C1 27/6 manifest SHA drift")
    require(sha256_file(metadata_path) == "d3e84a1c248a0d2fc97dd2d009f4e31ec7800de4f0358161fab23d7c689807d7", "C1 shared-T4 seed-44 metadata SHA drift")
    require(metadata.get("status") == "completed", "C1 seed-44 run is not completed")
    require(metadata.get("training_kind") == "shared_paired_view" and metadata.get("seed") == 44, "C1 shared-T4 seed-44 identity drift")
    require(metadata.get("held_out_test_evaluated") is False and metadata.get("formal_sua_files_opened") is False, "C1 metadata formal-access drift")
    require(metadata.get("data_manifest_sha256") == "1ab97fd67bea26cb2c16ef970bdc6f08e4ba02b7a287a63706cb002e3405ddeb", "C1 data-manifest binding drift")
    sua_config = metadata.get("view_configs", {}).get("sua", {})
    require(sua_config.get("signal_view") == "sua", "C1 SUA view identity drift")
    side = sua_config.get("side_features", {})
    require(
        (side.get("group"), side.get("pool_size"), side.get("side_dim"), side.get("normalization_sha256"))
        == ("t4", 50, 4, SOURCE_NORMALIZERS["t4_side_feature"]["semantic_sha256"]),
        "C1 SUA T4 normalizer semantic binding drift",
    )
    require(manifest.get("file_count") == 33, "C1 manifest file-count drift")
    require(manifest.get("formal_test_paths_resolved") is False, "C1 manifest resolves formal paths")
    require(manifest.get("formal_test_file_paths") == [] and manifest.get("formal_test_file_hashes") == [], "C1 manifest exposes formal artifacts")
    val = manifest.get("file_inventory", {}).get("val")
    require(isinstance(val, list) and val and val[0] == {key: DEVELOPMENT_FIXTURE[key] for key in ("bytes", "filename", "session", "sha256")}, "fixed consumed development fixture drift")

    v1_source = source_text["sua_exploration/mc_maze/subm_co_score_only.py"]
    for value in (
        TERMINAL_CHECKPOINT["sha256"],
        SOURCE_NORMALIZERS["behavior"]["sha256"],
        SOURCE_NORMALIZERS["t4_side_feature"]["sha256"],
        SOURCE_NORMALIZERS["t4_side_feature"]["semantic_sha256"],
    ):
        require(value in v1_source, f"v1 score-only source is missing pinned parity value {value}")
    v2_source = source_text["sua_exploration/mc_maze/subm_co_score_only_v2.py"]
    require("SCORER_ADAPTER_PARITY_RECEIPT_PIN: FilePin | None = None" in v2_source, "v2 parity pin state changed; static prelaunch cannot assert the current blocker")
    require(PARITY_SCHEMA in v2_source, "v2 required parity-receipt schema drift")
    require("recompute_torchmetrics_r2_cpu" in v2_source and "R2_RECOMPUTE_ATOL = 1.0e-6" in v2_source, "v2 sealed R2 recomputation contract drift")
    required_reference_symbols = {
        "sua_exploration/scripts/eval_adaptation_dandi688.py": (
            "def load_session_with_trials(",
            "def build_calib_trials_for_indices(",
            "def make_subset_dataset(",
            "def attach_side_features(",
            "def eval_r2(",
        ),
        "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py": ("def load_frozen_model(",),
        "sua_exploration/scripts/eval_paired_view_c1_epoch_window.py": ("def _view_feature_config(", "forward Q=30, pool=50", "score_start"),
        "sua_exploration/mc_maze/multisession_datamodule.py": ("def load_frozen_train_val_manifest(", "def pool_spikes_by_electrode("),
    }
    for relative, symbols in required_reference_symbols.items():
        for symbol in symbols:
            require(symbol in source_text[relative], f"required C1 reference symbol missing: {relative}:{symbol}")

    return {
        "source_hashes": source_hashes,
        "protocol_document": {"path": DOCUMENT, "sha256": document_sha},
        "c1_static_metadata": {
            "manifest": {"path": MANIFEST, "sha256": sha256_file(manifest_path)},
            "run_metadata": {"path": RUN_METADATA, "sha256": sha256_file(metadata_path)},
        },
    }


def build_draft(root: Path = ROOT) -> dict[str, Any]:
    audit = static_authority(root)
    return {
        "schema_version": 1,
        "kind": "dandi_000688_subc_scorer_adapter_parity_prelaunch_draft_v1",
        "status": "NOT_AUTHORIZED_FOR_PARITY_EXECUTION",
        "append_only": True,
        "parity_execution_receipt_schema": PARITY_SCHEMA,
        "static_authority": audit,
        "fixed_fixture": {
            "terminal_checkpoint": TERMINAL_CHECKPOINT,
            "source_normalizers": SOURCE_NORMALIZERS,
            "consumed_development_session": DEVELOPMENT_FIXTURE,
            "chronology": {
                "rewarded_trial_support": "trials[0:50]",
                "identity_selection": "C1 first_n30 only within trials[0:50]",
                "query": "trials[50:] only; valid 50-bin windows",
                "spike_bin_ms": 20,
                "history_bins": 50,
                "trial_length_bins": 100,
                "pad_value": -1.0,
                "finite_batch": "first 16 rows of the first C1 post50 DataLoader batch; DataLoader(batch_size=128, shuffle=False, num_workers=0)",
            },
        },
        "required_shared_reference_calls": [
            "eval_adaptation_dandi688.load_session_with_trials",
            "eval_adaptation_dandi688.build_calib_trials_for_indices",
            "eval_adaptation_dandi688.make_subset_dataset",
            "eval_adaptation_dandi688.attach_side_features",
            "select_gradient_free_protocol_dandi688.load_frozen_model",
            "eval_adaptation_dandi688.eval_r2",
            "subm_co_score_only_v2.recompute_torchmetrics_r2_cpu",
        ],
        "comparison_thresholds": {
            "chronology_spike_history_targets_unit_order_t4_rows_normalizers_model_inputs": "exact dtype/shape/order/SHA-256 equality; max_abs=0",
            "prediction_values": "exact dtype/shape/order/SHA-256 equality; max_abs=0; max_rel=0",
            "adapter_vs_reference_r2_atol": 1.0e-6,
            "sealed_array_cpu_torchmetrics_r2_atol": 1.0e-6,
            "nonfinite_value": "FAIL_CLOSED",
        },
        "device_policy": {
            "parity_execution_device": "cpu",
            "cuda_visible_devices": "",
            "cuda_initialization": "FORBIDDEN",
            "torch_deterministic_algorithms": True,
            "tf32": False,
            "autocast": False,
            "intraop_threads": 1,
            "interop_threads": 1,
        },
        "future_sealed_artifacts": {
            "single_use_root": True,
            "exclusive_create_fsync_sha256_then_mode": "0444",
            "files": [
                "input_manifest.json",
                "c1_reference_trace.json",
                "adapter_trace.json",
                "reference_prediction_target.npz",
                "adapter_prediction_target.npz",
                "receipt.json",
                "seal.json",
            ],
        },
        "blocked_until": [
            "a shared observation-only C1 batch-trace helper exists; independent reimplementation is forbidden",
            "a root-reviewed single-use consumed-sub-C parity execution receipt is produced",
            "a later append-only runner revision pins that receipt SHA-256",
            "a root Ed25519 public key file and SHA-256 are pinned",
        ],
        "operations_by_this_draft": {
            "checkpoint_files_opened": 0,
            "normalizer_files_opened": 0,
            "nwb_files_opened": 0,
            "subm_nwb_paths_constructed": False,
            "subm_nwb_files_accessed": False,
            "model_forward_calls": 0,
            "prediction_calls": 0,
            "torchmetrics_calls": 0,
            "gpu_used": False,
            "normalizer_fitting_calls": 0,
            "optimizer_or_backward_calls": 0,
        },
    }


def build_receipt(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    draft = build_draft(root)
    receipt = {
        "schema_version": 1,
        "receipt_kind": "dandi_000688_subc_scorer_adapter_parity_prelaunch_receipt_v1",
        "status": "STATIC_PROTOCOL_AUDITED_PARITY_EXECUTION_NOT_AUTHORIZED",
        "append_only": True,
        "prelaunch_draft": {"filename": "parity_protocol_draft.json", "sha256": canonical_sha256(draft)},
        "operations": dict(draft["operations_by_this_draft"]),
        "blockers": list(draft["blocked_until"]),
    }
    return draft, receipt


def write_once_immutable(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444, f"immutable mode failed: {path}")
    return hashlib.sha256(payload).hexdigest()


def write_prelaunch(output_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"append-only parity prelaunch output already exists: {output_dir}")
    draft, receipt = build_receipt(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    draft_path = output_dir / "parity_protocol_draft.json"
    receipt_path = output_dir / "receipt.json"
    seal_path = output_dir / "seal.json"
    draft_sha = write_once_immutable(draft_path, draft)
    require(draft_sha == receipt["prelaunch_draft"]["sha256"], "internal draft receipt binding drift")
    receipt_sha = write_once_immutable(receipt_path, receipt)
    seal = {
        "schema_version": 1,
        "kind": "dandi_000688_subc_scorer_adapter_parity_prelaunch_seal_v1",
        "status": receipt["status"],
        "append_only": True,
        "artifacts": [
            {"path": draft_path.name, "sha256": draft_sha, "bytes": draft_path.stat().st_size, "mode": "0444"},
            {"path": receipt_path.name, "sha256": receipt_sha, "bytes": receipt_path.stat().st_size, "mode": "0444"},
        ],
        "parity_execution_still_not_authorized": True,
    }
    seal_sha = write_once_immutable(seal_path, seal)
    return {
        "output_dir": str(output_dir),
        "draft": {"path": str(draft_path), "sha256": draft_sha},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "seal": {"path": str(seal_path), "sha256": seal_sha},
        "status": receipt["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true", help="run static source/metadata audit only; write nothing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            print(json.dumps(static_authority(args.repo_root), indent=2, sort_keys=True))
        else:
            print(json.dumps(write_prelaunch(args.output_dir, args.repo_root), indent=2, sort_keys=True))
        return 0
    except StaticParityError as exc:
        print(f"FAIL_CLOSED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
