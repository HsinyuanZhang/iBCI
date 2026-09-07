#!/usr/bin/env python3
"""Write-once CPU preflight for Experiment A's qualified M30 B0/T4/TS4 references.

This v2 audit replaces neither historical artifacts nor the v1 receipt.  It reads exactly
the six reused-development NWBs through the datamodule's rewarded/usable-trial helper to
reconstruct original trial-table indices.  It never opens a formal-session NWB, loads a
checkpoint, creates a model, trains, or scores R2.

The reconstructed current-helper semantics are explicitly distinct from historical source
byte provenance: the artifacts, launch scripts, manifest.env, and source interfaces bind
the intended M30 invocation, but no historical source-SHA receipt was persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# This assignment precedes the datamodule import and prevents the CPU audit from exposing a
# CUDA device through an optional scientific dependency.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
RESULT_ROOT = SUA_ROOT / "results" / "sua_spint_t4_mainline_fp32_v1"
CHECKPOINT_ROOT = SUA_ROOT / "checkpoints"
MANIFEST_PATH = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
DATA_DIR = SUA_ROOT / "data" / "dandi_000688" / "sub-C"
TEACHER_PATH = CHECKPOINT_ROOT / "teacher_mc_maze" / "best-epoch=083-val_heldin" / "r2_mean=0.9061.ckpt"
GENERIC_EPOCH_SCORER = SUA_ROOT / "scripts" / "eval_epoch_window_generic_dandi688.py"
SESSION_SCORER = SUA_ROOT / "scripts" / "select_gradient_free_protocol_dandi688.py"
MAINLINE_LAUNCHER = SUA_ROOT / "scripts" / "run_sua_spint_t4_mainline.sh"
SINGLE_SEED_LAUNCHER = SUA_ROOT / "scripts" / "run_sua_spint_t4_single_seed.sh"
FEATURE_HELPER = SUA_ROOT / "mc_maze" / "unit_side_features.py"
MANIFEST_ENV = RESULT_ROOT / "manifest.env"
SEEDS = (42, 43, 44)
ARMS = ("B0", "T4", "TS4")
M30 = 30
EPOCH_WINDOW = list(range(5, 13))
EXPECTED_ARTIFACT_SHA256 = {
    ("B0", 42): "24477be67d6855ccaadaf64f2651e949e16a2ff972afd821d881a2f6d3f325fd",
    ("B0", 43): "8cd4872c6419145a8b5c349207d42d7868ae80b4c72e607d33a00e9fd98031ee",
    ("B0", 44): "415dde1d40106ccb0d5481fa413dda123bc1e99f9a3a2464c48475befc0a5f90",
    ("T4", 42): "b8f659a46ad55eea766cbad1be70e1cc99df4c3c4c6c38863f2a5a5ee3104148",
    ("T4", 43): "e18a52a750b44f426ba1e39f3a8f791806f53228e5a8a69b73a491178b8a09b4",
    ("T4", 44): "704f5a40bb07e53dc3267d4ed70e434255c84cb9e9b60dcfcb8dd7dcecac3e64",
    ("TS4", 42): "0178e384eb8976b931b0fcc47ce53354e84500a65672ee51fd4d9c19ab41c841",
    ("TS4", 43): "ca146f795aa342560a1220c7a6e3218f4370f10493bbbb76f7e808a1591936ba",
    ("TS4", 44): "e697e870d40efe5357b9c3ea410eb286fdd2d537e0134007b2b33f19d6641f7d",
}
EXPECTED_TEACHER_SHA256 = "9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d"
EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
EXPECTED_NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def artifact_path(arm: str, seed: int) -> Path:
    return RESULT_ROOT / f"{arm.lower()}_s{seed}.json"


def metadata_path(arm: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / f"sua_spint_t4_mainline_fp32_v1_{arm.lower()}_dandi688_co_s{seed}" / "run_metadata.json"


def m30_partition(usable_original_trial_indices: Iterable[int]) -> dict[str, Any]:
    """Resolve M30 scorer positions into original NWB trial-table indices."""
    usable = [int(value) for value in usable_original_trial_indices]
    require(len(usable) > M30, f"need more than {M30} usable trials for post-pool scoring; found {len(usable)}")
    support = usable[:M30]
    scored = usable[M30:]
    require(not set(support).intersection(scored), "M30 support overlaps scored trial indices")
    return {
        "usable_trial_list_length": len(usable),
        "activity_forward_support": {
            "usable_trial_list_slice": "[0:30]",
            "original_trial_indices": support,
        },
        "t4_label_rate_fit": {
            "usable_trial_list_slice": "[0:30]",
            "original_trial_indices": support,
        },
        "scored": {
            "usable_trial_list_slice": "[30:end]",
            "original_trial_indices": scored,
        },
        "support_feature_fit_equal": True,
        "t4_fit_scored_overlap_original_trial_indices": [],
    }


def source_and_launch_evidence() -> dict[str, Any]:
    """Bind M30 CLI capability and recorded launch intent without claiming old bytes."""
    paths = (GENERIC_EPOCH_SCORER, SESSION_SCORER, MAINLINE_LAUNCHER, SINGLE_SEED_LAUNCHER, FEATURE_HELPER, MANIFEST_ENV)
    for path in paths:
        require(path.is_file(), f"required source/launch evidence is absent: {path}")
    generic = GENERIC_EPOCH_SCORER.read_text(encoding="utf-8")
    session = SESSION_SCORER.read_text(encoding="utf-8")
    mainline = MAINLINE_LAUNCHER.read_text(encoding="utf-8")
    single_seed = SINGLE_SEED_LAUNCHER.read_text(encoding="utf-8")
    feature = FEATURE_HELPER.read_text(encoding="utf-8")
    manifest_env = MANIFEST_ENV.read_text(encoding="utf-8")
    checks = {
        "generic scorer exposes --pool_size": 'parser.add_argument(\n        "--pool_size"' in generic,
        "generic scorer passes args.pool_size into evaluator": "pool_size=args.pool_size" in generic,
        "generic scorer records args.pool_size": '"pool_size": args.pool_size' in generic,
        "session scorer scores only after pool": 'eval_trials = rec["trials"][pool_size:]' in session,
        "T4 helper uses first pool_size usable/rewarded trials": "pool_trials = pool_trials[:pool_size]" in feature,
        "mainline launcher passes --pool_size 30": "--pool_size 30" in mainline,
        "single-seed launcher passes --pool_size 30": "--pool_size 30" in single_seed,
        "manifest.env records evaluation_pool_size=30": "evaluation_pool_size=30" in manifest_env,
        "manifest.env records forward n=30": "evaluation_forward_calibration_n=30" in manifest_env,
        "manifest.env records activity n=30": "training_activity_calibration_n=30" in manifest_env,
    }
    for name, passed in checks.items():
        require(passed, f"M30 launch/scorer contract drift: {name}")
    return {
        "current_files": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths},
        "checked_contracts": checks,
        "reconstructed_semantics": {
            "usable_trial_list": "chronological rewarded trials that pass the datamodule duration/window filter",
            "activity_forward_support": "usable trials[0:30]",
            "t4_label_rate_fit": "usable trials[0:30]",
            "scored": "usable trials[30:end]",
        },
        "historical_source_byte_provenance": {
            "status": "not_persisted_in_historical_m30_artifacts",
            "limitation": "Current source hashes and launch-script semantics are hash-bound evidence, not a contemporaneous source-byte receipt. This is the same reconstructed-source limitation acknowledged by the accepted M50 boundary audit and is not elevated to a unique M30 blocker.",
        },
    }


def bind_reused_artifacts() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    require(sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256, "strict manifest SHA-256 mismatch")
    require(sha256_file(TEACHER_PATH) == EXPECTED_TEACHER_SHA256, "teacher SHA-256 mismatch")
    require(manifest.get("split_counts") == [27, 6, 6], "strict manifest split counts drift")
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            leaf_path = artifact_path(arm, seed)
            run_metadata_path = metadata_path(arm, seed)
            leaf = read_json(leaf_path)
            metadata = read_json(run_metadata_path)
            protocol = leaf.get("protocol") or {}
            side = metadata.get("side_features") or {}
            label = f"{arm}@{seed}"
            leaf_sha = sha256_file(leaf_path)
            metadata_sha = sha256_file(run_metadata_path)
            require(leaf_sha == EXPECTED_ARTIFACT_SHA256[(arm, seed)], f"{label}: artifact SHA-256 mismatch")
            require(leaf.get("run_metadata_sha256") == metadata_sha, f"{label}: artifact does not bind run_metadata bytes")
            require(leaf.get("teacher_ckpt_sha256") == EXPECTED_TEACHER_SHA256, f"{label}: leaf teacher SHA mismatch")
            require(leaf.get("train_val_manifest_sha256") == EXPECTED_MANIFEST_SHA256, f"{label}: leaf manifest SHA mismatch")
            require(metadata.get("teacher_sha256") == EXPECTED_TEACHER_SHA256, f"{label}: metadata teacher SHA mismatch")
            require(metadata.get("train_val_manifest_sha256") == EXPECTED_MANIFEST_SHA256, f"{label}: metadata manifest SHA mismatch")
            require(metadata.get("session_splits") == manifest["session_splits"], f"{label}: metadata split/order drift")
            require(leaf.get("session_splits") == manifest["session_splits"], f"{label}: leaf split/order drift")
            require(leaf.get("signal_view") == "sua" and metadata.get("signal_view") == "sua", f"{label}: non-SUA signal view")
            require(leaf.get("no_test_files_evaluated") is True, f"{label}: leaf test-use flag is not false")
            require(metadata.get("held_out_test_evaluated") is False, f"{label}: metadata formal-test flag is not false")
            require((metadata.get("session_files") or {}).get("test") == [], f"{label}: training metadata lists test files")
            training = metadata.get("training") or {}
            require(training.get("calibration_n_trials") == M30, f"{label}: training activity n is not 30")
            require(training.get("max_epochs") == 12 and training.get("no_early_stopping") is True and training.get("checkpoint_every_epoch") is True, f"{label}: 12-epoch fixed-window training contract drift")
            for key, expected in (("calibration_n", M30), ("train_activity_calibration_n", M30), ("evaluation_forward_calibration_n", M30), ("pool_size", M30), ("epoch_window", EPOCH_WINDOW), ("total_epochs", 12), ("selection_mode", "first")):
                require(protocol.get(key) == expected, f"{label}: protocol.{key} expected {expected!r}, found {protocol.get(key)!r}")
            require(leaf.get("epoch_list") == EPOCH_WINDOW, f"{label}: artifact epoch window drift")
            expected_group = {"B0": "none", "T4": "t4", "TS4": "ts4"}[arm]
            expected_side_dim = 0 if arm == "B0" else 4
            require(side.get("group") == expected_group and side.get("side_dim") == expected_side_dim, f"{label}: side group/dim drift")
            if arm != "B0":
                require(side.get("pool_size") == M30, f"{label}: side feature pool is not M30")
                require(side.get("normalization_sha256") == EXPECTED_NORMALIZER_SHA256, f"{label}: normalizer SHA mismatch")
                # Older leaves said chronological_trials while generic scorer output now names
                # the underlying datamodule filter precisely as chronological_rewarded_trials.
                scope = leaf.get("calibration_feature_label_scope")
                require(scope in {"chronological_trials[0:30]", "chronological_rewarded_trials[0:30]"}, f"{label}: unsupported label feature scope {scope!r}")
            else:
                scope = None
            if arm == "TS4":
                require(side.get("permutation_seed") == seed, f"{label}: row-shuffle seed drift")
            rows.append({
                "arm": arm,
                "seed": seed,
                "artifact": {"path": str(leaf_path.relative_to(REPO_ROOT)), "sha256": leaf_sha},
                "run_metadata": {"path": str(run_metadata_path.relative_to(REPO_ROOT)), "sha256": metadata_sha},
                "feature_group": side.get("group"),
                "side_dim": side.get("side_dim"),
                "normalization_sha256": side.get("normalization_sha256"),
                "permutation_seed": side.get("permutation_seed"),
                "historical_label_scope_wording": scope,
                "declared_m_activity": training.get("calibration_n_trials"),
                "declared_m_t4": side.get("pool_size") if arm != "B0" else None,
                "declared_score_start": protocol.get("pool_size"),
                "epoch_window": leaf.get("epoch_list"),
            })
    return {
        "teacher": {"path": str(TEACHER_PATH.relative_to(REPO_ROOT)), "sha256": sha256_file(TEACHER_PATH)},
        "strict_manifest": {"path": str(MANIFEST_PATH.relative_to(REPO_ROOT)), "sha256": sha256_file(MANIFEST_PATH), "session_splits": manifest["session_splits"]},
        "shared_t4_ts4_normalizer_sha256": EXPECTED_NORMALIZER_SHA256,
        "artifacts": rows,
    }


def reconstruct_validation_indices() -> dict[str, dict[str, Any]]:
    """Use the single datamodule helper on six development NWBs, never formal NWBs."""
    import sys

    sys.path.insert(0, str(SUA_ROOT))
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials

    manifest = read_json(MANIFEST_PATH)
    validation_names = manifest["session_splits"]["val"]
    formal_names = set(manifest["session_splits"]["test"])
    output: dict[str, dict[str, Any]] = {}
    for session_name in validation_names:
        require(session_name not in formal_names, f"validation/formal manifest overlap: {session_name}")
        candidates = list(DATA_DIR.glob(f"{session_name}_behavior+ecephys.nwb"))
        require(len(candidates) == 1, f"expected one validation NWB for {session_name}, found {len(candidates)}")
        trials = list_datamodule_rewarded_trials(candidates[0], bin_size_ms=20, window_size=50, trial_result_filter="R")
        indices = [int(trial["trial_index"]) for trial in trials]
        require(len(indices) == len(set(indices)), f"{session_name}: helper returned duplicate original trial indices")
        output[session_name] = m30_partition(indices)
    return output


def build_report() -> dict[str, Any]:
    artifacts = bind_reused_artifacts()
    source = source_and_launch_evidence()
    partitions = reconstruct_validation_indices()
    return {
        "schema_version": 2,
        "audit_name": "t4_m30_experiment_a_cpu_preflight_v2",
        "supersedes": "none; this is a new receipt that leaves v1 immutable",
        "execution_scope": {
            "cpu_only": True,
            "models_or_checkpoints_loaded": False,
            "training_or_scoring_performed": False,
            "formal_sua_nwb_opened": False,
            "only_reused_development_nwbs_read": True,
            "reused_development_nwb_sessions_read": list(artifacts["strict_manifest"]["session_splits"]["val"]),
        },
        "artifact_identity": artifacts,
        "launch_and_current_source_evidence": source,
        "resolved_trial_semantics": {
            "activity_forward_support": "chronological usable/rewarded trials[0:30]",
            "t4_label_rate_fit": "chronological usable/rewarded trials[0:30]",
            "scored": "chronological usable/rewarded trials[30:end]",
            "t4_fit_scored_overlap": False,
            "conclusion": "M_activity=M_T4=30; every scored usable trial is post-support and post-feature-fit.",
        },
        "per_seed_validation_original_trial_indices": {str(seed): partitions for seed in SEEDS},
        "historical_claim_status": {
            "m30_references": "qualified_for_Experiment_A_descriptor_implementation",
            "limitation": "Historical M30 leaves omit a contemporaneous persisted per-trial scorer trace and source-byte SHA. This receipt reconstructs exact original indices via the current datamodule helper and binds current source/launch evidence; the limitation is reported, not treated differently from the accepted M50 reconstruction audit.",
            "formal_scope": "DANDI 000688 sub-C/CO reused-development evidence; formal SUA sessions unopened.",
        },
        "implementation_scope": {
            "descriptor_implementations_added": [],
            "ph4": {
                "implemented": False,
                "authorized_by_this_receipt": False,
                "next_required_step": "separate descriptor-contract audit with a source-only phase normalizer",
            },
        },
        "eligibility": {
            "status": "pass",
            "experiment_a_m30_reference_qualified": True,
            "may_proceed_to_descriptor_implementation": True,
            "residual_blockers": [],
        },
    }


def markdown_receipt(report: dict[str, Any]) -> str:
    identity = report["artifact_identity"]
    lines = [
        "# Experiment A M30 CPU preflight v2 receipt",
        "",
        "**Status:** `PASS` — the M30 reference substrate may proceed to descriptor implementation.",
        "",
        "This is a CPU-only reused-development audit. It loaded only the six validation NWBs through the datamodule helper; no checkpoint/model, formal SUA NWB, GPU, training, or scoring path was used.",
        "",
        "## Bound M30 semantics",
        "",
        "- Activity/forward support: chronological usable/rewarded trials `[0:30]`.",
        "- T4 label/rate fit: the same chronological usable/rewarded trials `[0:30]`.",
        "- Scoring: chronological usable/rewarded trials `[30:end]`; feature-fit/scored overlap is empty.",
        "- The generic epoch scorer's default pool of 50 is not the launch value: both historical mainline launchers explicitly supply `--pool_size 30`, `manifest.env` records `evaluation_pool_size=30`, and each leaf records pool 30.",
        "",
        "## Immutable artifact identities",
        "",
        f"- Teacher SHA-256: `{identity['teacher']['sha256']}`",
        f"- Strict manifest SHA-256: `{identity['strict_manifest']['sha256']}`",
        f"- Shared T4/TS4 train-only normalizer SHA-256: `{identity['shared_t4_ts4_normalizer_sha256']}`",
        "",
        "| Arm | Seed | Artifact SHA-256 | run_metadata SHA-256 | Feature group / side dim |",
        "|---|---:|---|---|---|",
    ]
    for row in identity["artifacts"]:
        lines.append(f"| {row['arm']} | {row['seed']} | `{row['artifact']['sha256']}` | `{row['run_metadata']['sha256']}` | `{row['feature_group']}` / `{row['side_dim']}` |")
    lines += [
        "",
        "## Historical-source limitation",
        "",
        report["historical_claim_status"]["limitation"],
        "",
        "PH4 is not implemented or authorized by this receipt. It remains a later descriptor-contract step requiring a source-only phase normalizer.",
    ]
    return "\n".join(lines) + "\n"


def write_receipt(report: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable v2 receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {**report, "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    (output_dir / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "RECEIPT.md").write_text(markdown_receipt(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build_report()
    write_receipt(report, args.output_dir)
    print(args.output_dir / "receipt.json")


if __name__ == "__main__":
    main()
