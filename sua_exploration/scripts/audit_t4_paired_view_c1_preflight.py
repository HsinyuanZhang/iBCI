#!/usr/bin/env python3
"""Fail-closed CPU provenance audit for paired SUA/pseudo-MUA C1.

This is deliberately an artifact-and-source audit.  It never discovers, opens,
or statistics-scans NWB files, never imports CUDA/Torch, and never trains a
model.  The target experiment is the independently budgeted
``bridge-Q30/T4-50-source10`` C1 proposal in
``docs/T4_NEXT_EXPERIMENT_PROTOCOL_V2.md``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


AUDIT_ID = "t4_paired_view_c1_preflight_v1"
SEEDS = (42, 43, 44)
GROUPS = {"F0": ("B3", "none"), "T4": ("B3S", "t4"), "TS4": ("B3S", "ts4")}
VIEWS = {"sua": "e3_tuning_ablation", "pseudo_mua": "pseudomua_t4_bridge_v1"}
EXPECTED_EPOCHS = list(range(5, 13))
EXPECTED_TEACHER_SHA = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
EXPECTED_MANIFEST_SHA = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"


class AuditFailure(RuntimeError):
    """A contract ambiguity that must block C1 implementation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditFailure(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditFailure(f"JSON object required: {path}")
    return value


def require(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AuditFailure(f"{label}: expected {expected!r}, observed {observed!r}")


def git_state(root: Path, path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(root))
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).returncode == 0
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "--", relative], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    ).returncode != 0
    return {"path": relative, "sha256": sha256_file(path), "git_tracked": tracked, "git_dirty": dirty}


def audit_scorer_semantics(evaluator_text: str, protocol_text: str) -> dict[str, Any]:
    """Verify the two literal source-level facts needed to resolve Q30 vs pool50.

    This intentionally does not execute the evaluator.  It proves the semantics
    of the current source tree, but cannot retroactively prove the exact source
    bytes used by the July 28 historical artifacts.
    """
    required_evaluator = (
        "pool_size=args.pool_size",
        "selection_mode=FIXED_SELECTION_MODE",
        "calibration_n=args.calibration_n",
        '"label_feature_calibration_n": label_feature_pool_size',
    )
    required_protocol = (
        'eval_trials = rec["trials"][pool_size:]',
        "indices = select_calibration_trial_indices(rec[\"trials\"], calibration_n, pool_size, mode)",
    )
    for token in required_evaluator:
        if token not in evaluator_text:
            raise AuditFailure(f"current epoch evaluator lacks required scorer token: {token!r}")
    for token in required_protocol:
        if token not in protocol_text:
            raise AuditFailure(f"current forward scorer lacks required token: {token!r}")
    return {
        "forward_activity_support": {"selection_mode": "first", "trial_list_indices": [0, 29]},
        "t4_label_rate_pool": {"trial_list_indices": [0, 49]},
        "score_trials": {"start_trial_list_index": 50, "definition": "rec['trials'][pool_size:]"},
        "overlap": {"activity_vs_score": False, "t4_pool_vs_score": False},
        "resolution": (
            "training.calibration_n_trials=10 belongs to source model training; "
            "the epoch-window validation scorer uses forward Q=30, label pool=50, and scores [50:]."
        ),
    }


def audit_view_artifacts(root: Path, manifest_splits: Mapping[str, Any]) -> dict[str, Any]:
    results_root = root / "sua_exploration" / "results"
    all_rows: dict[str, Any] = {}
    canonical_protocol: dict[str, Any] | None = None
    canonical_teacher: str | None = None
    for view, directory_name in VIEWS.items():
        directory = results_root / directory_name
        rows: dict[str, Any] = {}
        for group, (variant, side_group) in GROUPS.items():
            group_rows: dict[str, Any] = {}
            for seed in SEEDS:
                artifact_path = directory / f"{group.lower()}_s{seed}.json"
                artifact = load_json(artifact_path)
                metadata_path = Path(str(artifact.get("run_metadata_path", ""))).resolve()
                metadata = load_json(metadata_path)
                require(sha256_file(metadata_path), artifact.get("run_metadata_sha256"), f"{artifact_path}.run_metadata_sha256")
                require(artifact.get("seed"), seed, f"{artifact_path}.seed")
                require(artifact.get("signal_view"), view, f"{artifact_path}.signal_view")
                require(artifact.get("variant"), variant, f"{artifact_path}.variant")
                require(artifact.get("no_test_files_evaluated"), True, f"{artifact_path}.no_test_files_evaluated")
                require(artifact.get("session_splits"), manifest_splits, f"{artifact_path}.session_splits")
                protocol = artifact.get("protocol")
                if not isinstance(protocol, dict):
                    raise AuditFailure(f"{artifact_path}.protocol must be an object")
                for field, expected in {
                    "total_epochs": 12, "burn_in_epochs": 4, "epoch_window": EXPECTED_EPOCHS,
                    "selection_mode": "first", "calibration_n": 30, "pool_size": 50,
                    "protocol_metric_source": "select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions",
                }.items():
                    require(protocol.get(field), expected, f"{artifact_path}.protocol.{field}")
                require(artifact.get("epoch_list"), EXPECTED_EPOCHS, f"{artifact_path}.epoch_list")
                require(metadata.get("held_out_test_evaluated"), False, f"{metadata_path}.held_out_test_evaluated")
                require(metadata.get("signal_view"), view, f"{metadata_path}.signal_view")
                require(metadata.get("variant"), variant, f"{metadata_path}.variant")
                require(metadata.get("session_splits"), manifest_splits, f"{metadata_path}.session_splits")
                training = metadata.get("training") or {}
                require(training.get("calibration_n_trials"), 10, f"{metadata_path}.training.calibration_n_trials")
                require(training.get("max_epochs"), 12, f"{metadata_path}.training.max_epochs")
                require(training.get("no_early_stopping"), True, f"{metadata_path}.training.no_early_stopping")
                side = metadata.get("side_features") or {}
                require(side.get("group"), side_group, f"{metadata_path}.side_features.group")
                if group != "F0":
                    require(side.get("pool_size"), 50, f"{metadata_path}.side_features.pool_size")
                    require(side.get("side_dim"), 4, f"{metadata_path}.side_features.side_dim")
                teacher_sha = artifact.get("teacher_ckpt_sha256")
                require(teacher_sha, EXPECTED_TEACHER_SHA, f"{artifact_path}.teacher_ckpt_sha256")
                require(metadata.get("teacher_sha256"), teacher_sha, f"{metadata_path}.teacher_sha256")
                if canonical_protocol is None:
                    canonical_protocol = protocol
                    canonical_teacher = teacher_sha
                else:
                    require(protocol, canonical_protocol, f"{artifact_path}.protocol cross-view")
                    require(teacher_sha, canonical_teacher, f"{artifact_path}.teacher cross-view")
                group_rows[str(seed)] = {
                    "artifact_path": str(artifact_path.relative_to(root)),
                    "artifact_sha256": sha256_file(artifact_path),
                    "run_metadata_path": str(metadata_path.relative_to(root)),
                    "run_metadata_sha256": sha256_file(metadata_path),
                    "normalization_sha256": side.get("normalization_sha256"),
                    "source_training_calibration_n_trials": training["calibration_n_trials"],
                }
            rows[group] = group_rows
        all_rows[view] = rows
    normalizers = {
        view: all_rows[view]["T4"]["42"]["normalization_sha256"] for view in VIEWS
    }
    if normalizers["sua"] == normalizers["pseudo_mua"]:
        raise AuditFailure("SUA and pseudo-MUA T4 normalizer hashes unexpectedly collide")
    return {"references": all_rows, "normalizer_isolation": {"hashes": normalizers, "distinct": True}}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
    manifest = load_json(manifest_path)
    require(sha256_file(manifest_path), EXPECTED_MANIFEST_SHA, "strict manifest SHA-256")
    splits = manifest.get("session_splits")
    if not isinstance(splits, dict):
        raise AuditFailure("strict manifest has no session_splits object")
    require(manifest.get("split_counts"), [27, 6, 6], "strict manifest split_counts")
    source_paths = {
        "epoch_evaluator": root / "sua_exploration/scripts/eval_epoch_window_generic_dandi688.py",
        "forward_scorer": root / "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
        "selection_helper": root / "sua_exploration/scripts/dandi688_gradient_free_protocol.py",
        "t4_features": root / "sua_exploration/mc_maze/unit_side_features.py",
        "datamodule": root / "sua_exploration/mc_maze/multisession_datamodule.py",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise AuditFailure(f"required source file missing: {path}")
    scorer = audit_scorer_semantics(
        source_paths["epoch_evaluator"].read_text(encoding="utf-8"),
        source_paths["forward_scorer"].read_text(encoding="utf-8"),
    )
    t4_text = source_paths["t4_features"].read_text(encoding="utf-8")
    for token in (
        'if signal_view == "pseudo_mua":',
        "rates, _ = pool_trial_rates_by_electrode(rates, electrode_ids)",
        "Never average sorted-unit T4 values",
        'cache_payload["signal_view"] = signal_view',
        'cache_payload["electrode_mapping"] = _electrode_mapping_fingerprint(nwb_path)',
    ):
        if token not in t4_text:
            raise AuditFailure(f"pseudo-MUA T4 implementation lacks required semantic token: {token!r}")
    references = audit_view_artifacts(root, splits)
    source_state = {name: git_state(root, path) for name, path in source_paths.items()}
    historical_source_sha_provenance = False
    blockers = [
        {
            "id": "historical_scorer_source_sha_absent",
            "severity": "hard",
            "detail": (
                "The 18 historical epoch-window artifacts pin run_metadata but do not pin the "
                "evaluator/forward-scorer source bytes. Current source inspection resolves the "
                "semantics but cannot prove the exact July-28 scorer revision."
            ),
        },
        {
            "id": "historical_run_metadata_manifest_absent",
            "severity": "hard",
            "detail": (
                "Historical run_metadata has null train_val_manifest fields. The audit binds the "
                "current strict manifest and verifies its session names, but cannot prove those "
                "exact manifest bytes were recorded during historical training."
            ),
        },
    ]
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "scope": "CPU-only provenance and semantic audit; no NWB, CUDA, training, or formal SUA file access",
        "formal_sua_files_opened": False,
        "formal_sua_paths_resolved": False,
        "data_files_opened": [],
        "bridge_identity": "paired-view bridge-Q30/T4-50-source10",
        "strict_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha256_file(manifest_path), "session_splits": splits},
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "semantics": scorer,
        "pseudo_mua_semantics": {
            "spikes": "sum sorted-unit binned spikes by electrode",
            "t4": "pool calibration trial rates by electrode before cosine fit; no unit-T4 averaging",
            "normalizers": "separate train-only SUA and pseudo-MUA normalizers required",
            "cache_isolation": "pseudo-MUA cache payload includes signal_view and electrode_mapping",
        },
        "artifact_audit": references,
        "current_source_state": source_state,
        "historical_source_sha_provenance_present": historical_source_sha_provenance,
        "hard_blockers": blockers,
        "c1_implementation_may_begin": False,
        "verdict": "blocked_fail_closed",
        "required_unblock": [
            "Create a historical-source/provenance bridge that pins the exact evaluator, forward scorer, selection helper, T4 feature, and datamodule source bytes used by all 18 references, or explicitly supersede/rebuild all separate-model references under a single frozen current-source receipt.",
            "Create a source-training manifest bridge for all 18 references, or rebuild the reference substrate with the strict manifest path/SHA recorded at training time.",
            "Do not change C1 architecture or begin GPU work until both provenance gaps are closed.",
        ],
    }


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable receipt already exists: {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_markdown_once(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable receipt already exists: {path}")
    semantics = receipt["semantics"]
    lines = [
        "# Paired-view C1 CPU preflight receipt",
        "",
        f"- **Audit:** `{receipt['audit_id']}`",
        f"- **Verdict:** `{receipt['verdict']}`",
        f"- **C1 implementation may begin:** `{receipt['c1_implementation_may_begin']}`",
        "- **Data access:** no NWB/data file was opened; no formal SUA path was resolved.",
        "",
        "## Resolved semantics",
        "",
        f"- Source training activity calibration: first `{receipt['artifact_audit']['references']['sua']['T4']['42']['source_training_calibration_n_trials']}` trials.",
        f"- Forward activity support: chronological first `{semantics['forward_activity_support']['trial_list_indices'][1] + 1}` trials.",
        f"- T4 label/rate pool: chronological first `{semantics['t4_label_rate_pool']['trial_list_indices'][1] + 1}` trials.",
        f"- Score start: trial-list index `{semantics['score_trials']['start_trial_list_index']}`.",
        "",
        "## Why this is blocked",
        "",
    ]
    for blocker in receipt["hard_blockers"]:
        lines.append(f"- `{blocker['id']}` — {blocker['detail']}")
    lines += ["", "## Required unblock", ""]
    lines += [f"- {item}" for item in receipt["required_unblock"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = (args.out_dir or root / "sua_exploration/results" / AUDIT_ID).resolve()
    try:
        receipt = build_receipt(root)
    except AuditFailure as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    write_json_once(out_dir / "receipt.json", receipt)
    write_markdown_once(out_dir / "RECEIPT.md", receipt)
    print(f"Wrote {out_dir / 'receipt.json'}")
    print(f"C1 implementation may begin: {receipt['c1_implementation_may_begin']}")
    return 0 if receipt["c1_implementation_may_begin"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
