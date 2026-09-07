#!/usr/bin/env python3
"""Read-only, CPU-only scorer-boundary audit for the selected SUA T4@50 anchors.

This is deliberately a provenance/audit tool, not an evaluator: it never loads a
checkpoint, instantiates a model, trains, scores R2, or resolves a formal-test NWB
file.  It binds the three selected leaf results and their run metadata to the
historical launch logs, inspects the scorer's fixed pool boundary, and reads only
the six reused-development NWBs to emit the *original* trial-table indices that
the scorer's ordered usable-trial list denotes.

The output is write-once.  A report path that already exists is an error, so a
later run cannot silently replace this receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

# This must precede optional scientific imports.  The audit has no model path and
# must not make a CUDA device available accidentally through an imported package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SEEDS = (42, 43, 44)
SUPPORT_N = 30
POOL_N = 50
RESULT_ROOT = SUA_ROOT / "results" / "sua_t4_confidence_film_v1"
RUN_ROOT = SUA_ROOT / "checkpoints"
DEFAULT_OUT = (
    SUA_ROOT / "results" / "sua_t4_m50_scorer_boundary_audit_v1_20260802"
    / "audit.json"
)


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


def ordered_trial_partition(usable_original_indices: Iterable[int]) -> dict[str, Any]:
    """Partition one scorer-ordered usable-trial sequence without scoring it.

    The ranges are positions in the evaluator's filtered chronological list, while
    the emitted values are original NWB trial-table row indices.  This distinction
    matters because rewarded/usable trial indices need not be consecutive.
    """
    usable = [int(value) for value in usable_original_indices]
    require(len(usable) > POOL_N, f"need > {POOL_N} usable trials; found {len(usable)}")
    support = usable[:SUPPORT_N]
    feature_fit = usable[:POOL_N]
    scorer_excluded_feature_tail = usable[SUPPORT_N:POOL_N]
    scored = usable[POOL_N:]
    require(not set(scored).intersection(feature_fit), "scored trials overlap the T4 fit pool")
    require(set(support).issubset(feature_fit), "activity support must be within T4 fit pool")
    return {
        "usable_trial_list_length": len(usable),
        "activity_forward_support": {
            "usable_trial_list_slice": "[0:30]",
            "original_trial_indices": support,
        },
        "t4_label_rate_fit": {
            "usable_trial_list_slice": "[0:50]",
            "original_trial_indices": feature_fit,
        },
        "trials_30_49": {
            "usable_trial_list_slice": "[30:50]",
            "original_trial_indices": scorer_excluded_feature_tail,
            "scored": False,
            "used_to_fit_t4_label_rate_features": True,
        },
        "scored": {
            "usable_trial_list_slice": "[50:end]",
            "original_trial_indices": scored,
        },
        "t4_fit_scored_overlap_original_trial_indices": [],
    }


def scorer_source_evidence() -> dict[str, Any]:
    """Fail closed unless the checked scorer retains the required pool-boundary code."""
    scorer = SUA_ROOT / "scripts" / "select_gradient_free_protocol_dandi688.py"
    epoch_runner = SUA_ROOT / "scripts" / "eval_epoch_window_generic_dandi688.py"
    feature_code = SUA_ROOT / "mc_maze" / "unit_side_features.py"
    scorer_text = scorer.read_text(encoding="utf-8")
    epoch_text = epoch_runner.read_text(encoding="utf-8")
    feature_text = feature_code.read_text(encoding="utf-8")
    required = {
        "scorer uses pool suffix": "eval_trials = rec[\"trials\"][pool_size:]" in scorer_text,
        "scorer selects first/n configuration": "configs = [(selection_mode, calibration_n)]" in scorer_text,
        "M50 epoch runner fixes pool": "FIXED_POOL_SIZE = 50" in epoch_text,
        "M50 epoch runner fixes forward n": "FIXED_CALIBRATION_N = 30" in epoch_text,
        "T4 fitting slices pool": "pool_trials = pool_trials[:pool_size]" in feature_text,
    }
    for label, passed in required.items():
        require(passed, f"scorer-source contract drift: {label}")
    return {
        "paths": {
            str(scorer.relative_to(REPO_ROOT)): sha256_file(scorer),
            str(epoch_runner.relative_to(REPO_ROOT)): sha256_file(epoch_runner),
            str(feature_code.relative_to(REPO_ROOT)): sha256_file(feature_code),
        },
        "checked_contracts": required,
        "semantics": {
            "forward_support": "first selected calibration_n=30 trials inside the 50-trial pool",
            "t4_label_rate_fit": "pool_trials[:pool_size], with pool_size=50",
            "scorer": "eval_trials = rec['trials'][pool_size:]",
        },
    }


def _run_paths(seed: int) -> tuple[Path, Path, Path]:
    leaf = RESULT_ROOT / f"t4m50_s{seed}.json"
    run_dir = RUN_ROOT / f"sua_t4_confidence_film_v1_t4m50_dandi688_co_s{seed}"
    return leaf, run_dir / "run_metadata.json", RESULT_ROOT / "logs" / f"t4m50_s{seed}.log"


def bind_selected_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = SUA_ROOT / "manifests" / "sua_t4_final_architecture_selection_v1.json"
    selection = read_json(selection_path)
    require(selection.get("evaluation_start_trial") == 50, "selection receipt lacks start=50")
    require(selection.get("activity_calibration_n") == 30, "selection receipt lacks activity n=30")
    require(selection.get("t4_label_feature_pool_n") == 50, "selection receipt lacks T4 pool=50")
    evidence: dict[str, Any] = {
        "selection_receipt": {
            "path": str(selection_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(selection_path),
        },
        "seeds": {},
    }
    for seed in SEEDS:
        leaf_path, metadata_path, log_path = _run_paths(seed)
        leaf = read_json(leaf_path)
        metadata = read_json(metadata_path)
        log_text = log_path.read_text(encoding="utf-8")
        protocol = leaf.get("protocol", {})
        require(leaf.get("seed") == seed, f"seed {seed}: leaf seed mismatch")
        require(protocol.get("selection_mode") == "first", f"seed {seed}: not first selection")
        require(protocol.get("calibration_n") == SUPPORT_N, f"seed {seed}: leaf n != 30")
        require(protocol.get("pool_size") == POOL_N, f"seed {seed}: leaf pool != 50")
        require(protocol.get("evaluation_forward_calibration_n") == SUPPORT_N, f"seed {seed}: forward n != 30")
        require(protocol.get("label_feature_calibration_n") == POOL_N, f"seed {seed}: label fit != 50")
        require(leaf.get("calibration_feature_label_scope") == "chronological_rewarded_trials[0:50]", f"seed {seed}: leaf label scope drift")
        require(metadata.get("training", {}).get("calibration_n_trials") == SUPPORT_N, f"seed {seed}: training n != 30")
        require(metadata.get("side_features", {}).get("group") == "t4", f"seed {seed}: not T4")
        require(metadata.get("side_features", {}).get("pool_size") == POOL_N, f"seed {seed}: side pool != 50")
        require(metadata.get("validation_protocol", {}).get("evaluation_windows") == "trials[calibration_n_trials:] only", f"seed {seed}: generic training metadata drift")
        require("protocol=M_activity=30; M_T4=50; evaluation=trials[50:]" in log_text, f"seed {seed}: historical launch log does not record M50 boundary")
        require(leaf.get("run_metadata_sha256") == sha256_file(metadata_path), f"seed {seed}: leaf does not bind metadata bytes")
        selected = selection["selected_seed_artifacts"][str(seed)]
        require(selected["validation_result"]["sha256"] == sha256_file(leaf_path), f"seed {seed}: selection leaf SHA mismatch")
        require(selected["run_metadata"]["sha256"] == sha256_file(metadata_path), f"seed {seed}: selection metadata SHA mismatch")
        evidence["seeds"][str(seed)] = {
            "leaf": {"path": str(leaf_path.relative_to(REPO_ROOT)), "sha256": sha256_file(leaf_path)},
            "run_metadata": {"path": str(metadata_path.relative_to(REPO_ROOT)), "sha256": sha256_file(metadata_path)},
            "launch_log": {"path": str(log_path.relative_to(REPO_ROOT)), "sha256": sha256_file(log_path)},
            "variant_score": leaf["variant_score"],
        }
    return selection, evidence


def read_validation_trial_indices(manifest: Path, data_dir: Path) -> dict[str, list[int]]:
    """Read only the six development NWBs and mirror the scorer's usable-trial filter.

    This is the `load_session_with_trials` filter expressed without building neural
    tensors or behavior arrays: rewarded (`result == 'R'`), chronological trial-table
    order, and a >=50-bin duration after clipping to the session's spike-derived bin
    range.  It never accesses the manifest's formal-test paths.
    """
    import numpy as np
    from pynwb import NWBHDF5IO

    manifest_data = read_json(manifest)
    session_splits = manifest_data["session_splits"]
    validation_names = session_splits["val"]
    formal_names = set(session_splits["test"])
    output: dict[str, list[int]] = {}
    for session_name in validation_names:
        require(session_name not in formal_names, f"manifest makes a validation/formal overlap: {session_name}")
        matches = list(data_dir.glob(f"{session_name}_behavior+ecephys.nwb"))
        require(len(matches) == 1, f"expected exactly one validation NWB for {session_name}")
        nwb_path = matches[0]
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            units_df = nwb.units.to_dataframe()
            all_spikes = np.concatenate(units_df["spike_times"].values)
            bin_edges = np.arange(float(all_spikes.min()), float(all_spikes.max()) + 0.020, 0.020)
            num_bins = len(bin_edges) - 1
            original_indices: list[int] = []
            for original_index, trial in nwb.intervals["trials"].to_dataframe().iterrows():
                if trial["result"] != "R":
                    continue
                start = max(0, int(np.searchsorted(bin_edges, trial["start_time"])))
                stop = min(num_bins, int(np.searchsorted(bin_edges, trial["stop_time"])))
                if stop - start >= 50:
                    original_indices.append(int(original_index))
        output[session_name] = original_indices
    return output


def build_report() -> dict[str, Any]:
    selection, artifact_evidence = bind_selected_artifacts()
    source_evidence = scorer_source_evidence()
    manifest = SUA_ROOT / "configs" / "subc_co_27_6_strict_train_val_manifest.json"
    data_dir = SUA_ROOT / "data" / "dandi_000688" / "sub-C"
    trial_indices = read_validation_trial_indices(manifest, data_dir)
    partitions = {name: ordered_trial_partition(indices) for name, indices in trial_indices.items()}
    seed_partitions = {str(seed): partitions for seed in SEEDS}
    return {
        "schema_version": 1,
        "audit_name": "selected_sua_t4_m50_scorer_boundary",
        "execution_scope": {
            "cpu_only": True,
            "models_or_checkpoints_loaded": False,
            "training_or_scoring_performed": False,
            "formal_sua_nwb_opened": False,
            "only_reused_development_nwbs_read": True,
        },
        "evidence": {
            **artifact_evidence,
            "strict_manifest": {"path": str(manifest.relative_to(REPO_ROOT)), "sha256": sha256_file(manifest)},
            "current_scorer_source": source_evidence,
        },
        "resolved_trial_semantics": {
            "activity_forward_support": "usable chronological trials[0:30]",
            "t4_label_rate_fit": "usable chronological trials[0:50]",
            "scored": "usable chronological trials[50:end]",
            "trials_30_49_scored": False,
            "trials_30_49_overlap_t4_fit": True,
            "t4_fit_scored_overlap": False,
            "conclusion": "No trials in positions 30:50 were scored; they are T4-fit-only and are excluded from scoring.",
        },
        "per_seed_validation_original_trial_indices": seed_partitions,
        "historical_claim_status": {
            "m50_anchor": "qualified_for_disjoint_post-T4-fit reused-development scoring",
            "m50_anchor_limitation": "not a formal SUA result; historical leaf receipts omit a persisted per-trial scorer trace, so this is a reconstructed, hash-bound audit rather than a contemporaneous trace receipt",
            "int8_trigger": "selection receipt's M50 score-boundary conflict is resolved as non-overlapping; any separate INT8 gate must still satisfy its own prelaunch and paired-quantization requirements",
        },
        "selection_receipt_summary": {
            "evaluation_start_trial": selection["evaluation_start_trial"],
            "activity_calibration_n": selection["activity_calibration_n"],
            "t4_label_feature_pool_n": selection["t4_label_feature_pool_n"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable audit output already exists: {out}")
    report = build_report()
    out.parent.mkdir(parents=True, exist_ok=False)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
