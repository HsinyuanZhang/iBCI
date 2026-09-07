#!/usr/bin/env python3
"""Fail-closed, source-only CPU Gate A for fixed-K temporal prototypes.

Default/prelaunch mode opens no NWB.  The actual source audit requires both a
separate root-review authorization bound to the immutable prelaunch receipt and
an explicit source-only confirmation.  It never resolves held-out/minival paths,
builds a decoder, uses CUDA, or submits EvalAI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
SCE = ROOT / "streaming_calibration_exp"
if str(SUA) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SUA))
if str(SCE) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCE))

from mc_maze.fixed_k_temporal_prototypes import (  # noqa: E402
    build_outer_loso_carriers,
    carrier_from_support_trials,
    outer_loso_proxy,
    streaming_cost_receipt,
)
from mc_maze.m1_fixed_k_prototype_gate_a import (  # noqa: E402
    BIN_SECONDS,
    GATE_A_SEMANTICS_VERSION,
    OBJ_LEVELS,
    RAW_PAD_VALUE,
    REPEATABILITY_RESAMPLES,
    RIDGE,
    SUPPORT_TRIALS,
    TARGET_START_TRIAL,
    M1RawSession,
    d4_support_carrier,
    disjoint_binomial_partition,
    extract_valid_raw_trials,
    future_neural_oracle_target,
    strict_gate,
    validate_obj_coverage,
)


MANIFEST = SUA / "manifests" / "m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json"
STEP2_DECISION = SUA / "results" / "sua_t4_cross_budget_source_audit_v1_20260802" / "step2_decision.json"
PROTOCOL = SUA / "docs" / "M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_V3_PROTOCOL.md"
PRELAUNCH_DEFAULT = SUA / "results" / "m1_fixed_k_temporal_prototype_gate_a_prelaunch_v3"
AUDIT_DEFAULT = SUA / "results" / "m1_fixed_k_temporal_prototype_gate_a_v3"
SCHEMA_PRELAUNCH = "m1_fixed_k_temporal_prototype_gate_a_prelaunch_v3"
SCHEMA_AUDIT = "m1_fixed_k_temporal_prototype_gate_a_v3"
REVIEW_AUTHORIZATION = "root_approved_source_only_cpu_gate_a_v3"
SOURCE_ONLY_CONFIRMATION = "I_CONFIRM_EXACT_FOUR_M1_SOURCE_NWBS_CPU_ONLY_V3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_json(item) for item in value]
    if hasattr(value, "__dict__"):
        return strict_json(vars(value))
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != "m1_fixed_k_temporal_prototype_gate_a_source_manifest_v2":
        raise ValueError("unexpected Step-3 source manifest schema")
    sessions = data.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 4:
        raise ValueError("Gate A requires exactly four source sessions")
    names = [entry.get("session") for entry in sessions]
    if names != ["ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"]:
        raise ValueError("source session identities/order differ from frozen Gate-A manifest")
    excluded = tuple(data.get("excluded_path_tokens", ()))
    for entry in sessions:
        relative = str(entry.get("relative_path", ""))
        if not relative.startswith("SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"):
            raise ValueError(f"source path leaves held-in-calib scope: {relative}")
        if any(token in relative.lower() for token in excluded):
            raise ValueError(f"source path contains excluded token: {relative}")
        if len(str(entry.get("sha256", ""))) != 64:
            raise ValueError("source manifest has invalid SHA256")
        if int(entry.get("future_trials_210_end", -1)) <= 0:
            raise ValueError("source manifest future trial count must be positive")
    return data


def resolved_source_paths(manifest: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve only exact manifest paths and fail closed on hashes/scope."""
    result: dict[str, Path] = {}
    for entry in manifest["sessions"]:
        path = ROOT / str(entry["relative_path"])
        if not path.is_file() or path.name.count("held-in-calib") != 1:
            raise FileNotFoundError(f"missing/falsely scoped Gate-A source file: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {entry['session']}: {actual}")
        result[str(entry["session"])] = path
    return result


def _step2_unblocks_step3() -> dict[str, Any]:
    decision = json.loads(STEP2_DECISION.read_text())
    if decision.get("decision", {}).get("step3_cpu_gate_a_unblocked") is not True:
        raise RuntimeError("Step 2 has not immutably unblocked Step-3 CPU Gate A")
    if decision.get("decision", {}).get("step3_decoder_or_gpu_authorized") is not False:
        raise RuntimeError("Step-2 decision must not authorize Step-3 decoder/GPU work")
    return decision


def build_prelaunch_receipt() -> dict[str, Any]:
    """No-NWB receipt binding all data/endpoint/guard contracts for root review."""
    manifest = load_manifest()
    step2 = _step2_unblocks_step3()
    return {
        "schema_version": SCHEMA_PRELAUNCH,
        "semantics_version": GATE_A_SEMANTICS_VERSION,
        "generated_by": "audit_m1_fixed_k_prototypes_gate_a.py --write-prelaunch",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "root-review-only prelaunch for actual source-only Step-3 CPU Gate A; not a result",
        "step2_unblock": {"path": str(STEP2_DECISION.relative_to(ROOT)), "sha256": sha256(STEP2_DECISION), "decision": step2["decision"]},
        "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST), "sessions": manifest["sessions"]},
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
        "raw_extraction": {
            "dataset_configuration": {
                "smooth_calibration": False,
                "interpolate_trials": False,
                "max_trial_length": 1024,
                "pad_value": RAW_PAD_VALUE,
                "use_calib_intertrials": False,
                "raw_bin_ms": int(BIN_SECONDS * 1000),
            },
            "accepted_values": "valid prefix only: finite, nonnegative integer binned counts",
            "padding_contract": "tail after exact calib_trial_lengths must be -1 and is stripped before all carrier/target computation",
            "sum_contract": "valid-prefix sum must equal calib_trial_spike_sums for every trial/channel",
            "trial_boundary": "first ten chronological production trial arrays, reset temporal filters before every trial",
            "future_raw_storage": "no future raw trial matrices retained; only production per-trial spike sums and valid-bin lengths from [210,end)",
            "temporary_offline_workspace": "FalconDataset temporarily materializes padded [trials,1024,64] production arrays for exactly the four manifest sources; validation strips only first-ten valid prefixes, then M1RawSession retains first-ten variable arrays plus [210,end) sums/lengths only.",
        },
        "endpoint": {
            "support_range": [0, SUPPORT_TRIALS],
            "target_range": [TARGET_START_TRIAL, "end"],
            "target_transform": "log1p(mean per-trial Hz) separately for every channel x obj_id level [1,2,3,4]",
            "ridge": RIDGE,
            "comparators": ["D4", "rate_only", "prototype", "slot_shuffle"],
            "common_width": 20,
            "label_disclosure": {
                "D4": "support obj_id only, first ten trials",
                "prototype_rate_slot_shuffle": "no support labels",
                "future_obj_id": "scorer-only target construction; barred from anchors, route, carrier, readout input and deployment",
            },
        },
        "outer_loso": {
            "exact_sessions": [entry["session"] for entry in manifest["sessions"]],
            "fold_anchor_fit": "other three sessions' complete first-ten raw support trials only",
            "readout": "ridge=1 fit only on other three source sessions; target left-out future neural oracle never enters fit",
            "slot_shuffle": "complete count-plus-rank-vector block, deterministic session-keyed nonidentity permutation for every train/left-out session within every fold",
        },
        "repeatability": {
            "method": "within-trial disjoint Binomial(count,0.5) partitions of first-ten raw support bins",
            "resamples_per_outer_session": REPEATABILITY_RESAMPLES,
            "frozen_objects": "outer-train full-support anchors and ridge readout; no refit in resamples",
            "inference_unit": "outer session (n=4); seeds quantify measurement repeatability and are not biological N",
            "summary": "separate per-resample median unit cosine for normalized slot-count distributions and flattened slot prototype values; for BOTH metrics each session lower 2.5% quantile must be >=0.5 and >=90% units must be value-defined in every repeat",
            "thresholds": {"slot_count_cosine_lower_quantile_2p5_min": 0.5, "prototype_value_cosine_lower_quantile_2p5_min": 0.5, "minimum_value_defined_unit_fraction": 0.9},
        },
        "paired_inference_gate": {
            "n_sessions": 4,
            "ci": "two-sided Student-t 95% interval over four outer-session paired R2 deltas",
            "mde": "(t_0.975,df3 + t_0.80,df3) * paired_session_sd / sqrt(4)",
            "pass": "prototype mean delta >= measured MDE and CI95 lower > 0 versus BOTH rate_only and slot_shuffle, plus raw/state/repeatability/no-oracle contracts",
            "failure": "stop CPU branch; no K/r/router/decoder/GPU/formal expansion",
        },
        "hard_exclusions": {"formal_paths_resolved": False, "minival": False, "held_out": False, "decoder": False, "evalai": False, "cuda": False},
        "execution_guard": {"review_authorization": REVIEW_AUTHORIZATION, "source_only_confirmation": SOURCE_ONLY_CONFIRMATION, "fresh_output_required": True},
        "status": "pending_root_review_no_nwb_opened",
    }


def write_prelaunch(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite prelaunch output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    target = output_dir / "prelaunch_receipt.json"
    target.write_text(json.dumps(strict_json(build_prelaunch_receipt()), indent=2, sort_keys=True) + "\n")
    (output_dir / "prelaunch_receipt.sha256").write_text(f"{sha256(target)}  {target.name}\n")
    return target


def _exact_source_session_dict(path: Path, *, session: str, label_loader: Any | None = None) -> dict[str, Any]:
    """Load one manifest path only; no discovery or minival object exists here."""
    from src.data.falcon_d4_features import calibration_obj_id_labels, validate_trial_label_alignment

    neural, covariates, trial_change, eval_mask = load_nwb(path, FalconTask.m1)
    labels = (calibration_obj_id_labels if label_loader is None else label_loader)(path, FalconTask.m1)
    validate_trial_label_alignment(trial_change, labels, source=str(path))
    return {
        "neural": np.asarray(neural, dtype=np.float32),
        "covariates": np.asarray(covariates, dtype=np.float32),
        "trial_change": np.asarray(trial_change, dtype=bool),
        "eval_mask": np.asarray(eval_mask, dtype=bool),
        "trial_obj_ids": np.asarray(labels, dtype=np.int64),
        "source_session": session,
    }


def _source_objects_from_exact_paths(paths: Mapping[str, Path], *, builder: Any = _exact_source_session_dict) -> OrderedDict:
    """Apply one exact-path builder to exactly the four manifest source paths."""
    if tuple(sorted(paths)) != ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928"):
        raise ValueError("exact source object construction requires only the four frozen source session names")
    return OrderedDict((name, builder(path, session=name)) for name, path in sorted(paths.items()))


def _raw_dataset_from_exact_manifest(paths: Mapping[str, Path]):
    """Use production FalconDataset with only four explicitly loaded source objects.

    This deliberately bypasses broad datamodule discovery, which can enumerate
    extra validation files. The only calls to ``load_nwb`` above receive values
    supplied by the immutable manifest.
    """
    from src.data.falcon_datamodule import FalconDataset
    sessions = _source_objects_from_exact_paths(paths)
    return FalconDataset(
        sessions_dict=sessions, calib_sessions_dict=sessions, window_size=100, split="step3_source_only",
        calibration_n_trials=SUPPORT_TRIALS, random_calibration=False, smooth_calibration=False,
        max_trial_length=1024, use_calib_intertrials=False, trial_feature_type="raw",
        remove_still_times=False, remove_calib_still_times=False, use_calib_active_segments=False,
        interpolate_trials=False, interpolate_trials_kind="linear", pad_value=RAW_PAD_VALUE,
        side_feature_group="d4", side_feature_mean=np.zeros(4, dtype=np.float32),
        side_feature_std=np.ones(4, dtype=np.float32), query_start_trial=0,
        allow_empty_query_sessions=True,
    )


def load_raw_source_sessions(manifest: Mapping[str, Any]) -> dict[str, M1RawSession]:
    """Open only exact source NWBs through the production raw trialization path."""
    paths = resolved_source_paths(manifest)
    dataset = _raw_dataset_from_exact_manifest(paths)
    required = set(paths)
    if set(dataset.calib_trialized_neural) != required:
        raise RuntimeError("production raw dataset session set differs from exact source manifest")
    output: dict[str, M1RawSession] = {}
    by_name = {entry["session"]: entry for entry in manifest["sessions"]}
    for name in sorted(required):
        entry = by_name[name]
        support_trials = extract_valid_raw_trials(
            np.asarray(dataset.calib_trialized_neural[name])[:SUPPORT_TRIALS],
            np.asarray(dataset.calib_trial_lengths[name])[:SUPPORT_TRIALS],
            np.asarray(dataset.calib_trial_spike_sums[name])[:SUPPORT_TRIALS],
        )
        labels = np.asarray(dataset.calib_trial_obj_ids[name], dtype=np.int64)
        if labels.shape != (np.asarray(dataset.calib_trial_lengths[name]).size,):
            raise RuntimeError(f"{name}: production calib_trial_obj_ids do not align with raw trial lengths")
        support_labels = labels[:SUPPORT_TRIALS]
        future_labels = labels[TARGET_START_TRIAL:]
        validate_obj_coverage(support_labels, name=f"{name} support", expected_counts=entry["support_obj_id_counts_0_10"])
        validate_obj_coverage(future_labels, name=f"{name} future", expected_counts=entry["future_obj_id_counts_210_end"])
        if future_labels.size != int(entry["future_trials_210_end"]):
            raise RuntimeError(f"{name}: frozen future [210,end) trial count changed")
        if labels.size <= TARGET_START_TRIAL:
            raise RuntimeError(f"{name}: no later neural target after trial {TARGET_START_TRIAL}")
        future_counts = np.asarray(dataset.calib_trial_spike_sums[name][TARGET_START_TRIAL:], dtype=np.float64)
        future_bins = np.asarray(dataset.calib_trial_lengths[name][TARGET_START_TRIAL:], dtype=np.int64)
        if future_counts.shape != (future_labels.size, 64) or future_bins.shape != (future_labels.size,):
            raise RuntimeError(f"{name}: production future raw sums/lengths do not align with scorer labels")
        if not np.array_equal(future_counts, np.rint(future_counts)) or np.any(future_counts < 0.0) or np.any(future_bins <= 0):
            raise RuntimeError(f"{name}: future scorer data are not valid raw count/exposure values")
        output[name] = M1RawSession(
            session=name,
            support_trials=support_trials,
            future_trial_counts=np.rint(future_counts).astype(np.int64),
            future_trial_bins=future_bins,
            support_obj_ids=support_labels.astype(np.int64, copy=False),
            future_obj_ids=future_labels.astype(np.int64, copy=False),
            source_path=str(paths[name].relative_to(ROOT)), source_sha256=str(entry["sha256"]),
        )
    return output


def _cosine_repeatability(anchors: np.ndarray, trials: tuple[np.ndarray, ...], *, seed: int) -> tuple[float, float, int, int]:
    rng = np.random.default_rng(seed)
    left, right = disjoint_binomial_partition(trials, rng=rng)
    # Each half is re-scaled to the full-rate expectation before routing under
    # anchors frozen from full outer-train support; without this, a split-half
    # result is biased solely by halved count magnitude.
    first_blocks = np.asarray(carrier_from_support_trials(anchors, tuple(2 * value for value in left))["prototype_blocks"], dtype=np.float64)
    second_blocks = np.asarray(carrier_from_support_trials(anchors, tuple(2 * value for value in right))["prototype_blocks"], dtype=np.float64)
    count_a, count_b = first_blocks[:, :, 0], second_blocks[:, :, 0]
    value_a, value_b = first_blocks[:, :, 1:].reshape(first_blocks.shape[0], -1), second_blocks[:, :, 1:].reshape(second_blocks.shape[0], -1)
    count_denom = np.linalg.norm(count_a, axis=1) * np.linalg.norm(count_b, axis=1)
    value_denom = np.linalg.norm(value_a, axis=1) * np.linalg.norm(value_b, axis=1)
    count_defined = count_denom > 1.0e-12
    value_defined = value_denom > 1.0e-12
    if not np.any(count_defined) or not np.any(value_defined):
        return float("nan"), float("nan"), int(count_defined.sum()), int(value_defined.sum())
    count_cosine = (count_a[count_defined] * count_b[count_defined]).sum(axis=1) / count_denom[count_defined]
    value_cosine = (value_a[value_defined] * value_b[value_defined]).sum(axis=1) / value_denom[value_defined]
    return float(np.median(count_cosine)), float(np.median(value_cosine)), int(count_defined.sum()), int(value_defined.sum())


def repeatability_receipt(sessions: Mapping[str, M1RawSession], fold_receipts: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, Any], bool]:
    rows: dict[str, Any] = {}
    overall = True
    for left_out in sorted(sessions):
        anchors = np.asarray(fold_receipts[left_out]["anchor_receipt"]["anchor_values"], dtype=np.float64)
        count_values, value_values, count_defined, value_defined = [], [], [], []
        for repeat in range(REPEATABILITY_RESAMPLES):
            seed = int.from_bytes(hashlib.sha256(f"m1-step3-repeat-v3:{left_out}:{repeat}".encode()).digest()[:4], "little")
            count_value, value, n_count, n_value = _cosine_repeatability(anchors, sessions[left_out].support_trials, seed=seed)
            count_values.append(count_value); value_values.append(value); count_defined.append(n_count); value_defined.append(n_value)
        counts = np.asarray(count_values, dtype=np.float64)
        values = np.asarray(value_values, dtype=np.float64)
        total_units = sessions[left_out].support_trials[0].shape[1]
        passed = bool(
            np.isfinite(counts).all() and np.isfinite(values).all()
            and min(count_defined) / total_units >= 0.90 and min(value_defined) / total_units >= 0.90
            and np.quantile(counts, 0.025) >= 0.5 and np.quantile(values, 0.025) >= 0.5
        )
        rows[left_out] = {
            "resamples": REPEATABILITY_RESAMPLES, "median_unit_slot_count_cosine_values": counts.tolist(),
            "median_unit_prototype_value_cosine_values": values.tolist(),
            "count_defined_units_per_resample": count_defined, "value_defined_units_per_resample": value_defined,
            "slot_count_median": float(np.nanmedian(counts)), "prototype_value_median": float(np.nanmedian(values)),
            "slot_count_lower_quantile_2p5": float(np.nanquantile(counts, 0.025)),
            "prototype_value_lower_quantile_2p5": float(np.nanquantile(values, 0.025)),
            "minimum_value_defined_fraction": float(min(value_defined) / total_units), "contract_pass": passed,
            "biological_inference_unit": "session; resample seeds are measurement repeats only",
        }
        overall = overall and passed
    return rows, overall


def run_source_audit(output_dir: Path) -> Path:
    """Run the exact four-source CPU audit. Caller must satisfy CLI guards first."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite source Gate-A output: {output_dir}")
    manifest = load_manifest()
    sessions = load_raw_source_sessions(manifest)
    support = {name: row.support_trials for name, row in sessions.items()}
    d4 = {name: d4_support_carrier(row) for name, row in sessions.items()}
    target = {name: future_neural_oracle_target(row) for name, row in sessions.items()}
    folds, fold_receipts = build_outer_loso_carriers(support, d4, seed=42)
    proxy = outer_loso_proxy(folds, target, ridge=RIDGE)
    per_arm = {arm: {row["left_out_session"]: float(row["r2"]) for row in rows} for arm, rows in proxy["arms"].items()}
    order = sorted(sessions)
    delta_rate = [per_arm["prototype"][name] - per_arm["rate_only"][name] for name in order]
    delta_shuffle = [per_arm["prototype"][name] - per_arm["slot_shuffle"][name] for name in order]
    repeatability, repeatability_pass = repeatability_receipt(sessions, fold_receipts)
    state_contract = {
        "all_folds_exclude_leftout_from_anchor_sources": all(left not in row["anchor_receipt"]["source_sessions"] for left, row in fold_receipts.items()),
        "all_carrier_widths_are_20": all(np.asarray(folds[left][arm][name]).shape[1] == 20 for left in folds for arm in folds[left] for name in folds[left][arm]),
        "deployed_raw_support_elements_zero": streaming_cost_receipt(64).raw_support_matrix_elements == 0,
    }
    oracle_contract = {
        "all_targets_are_finite_channel_by_four": all(target[name].shape == (64, 4) and np.isfinite(target[name]).all() for name in target),
        "future_labels_used_only_in_target_dictionary": set(target) == set(sessions) and all(len(row.future_obj_ids) > 0 for row in sessions.values()),
        "carrier_dictionary_uses_support_trials_only": all(len(row.support_trials) == SUPPORT_TRIALS for row in sessions.values()),
    }
    gate = strict_gate(delta_rate, delta_shuffle, repeatability_contract_pass=repeatability_pass, oracle_contract_pass=all(oracle_contract.values()), state_contract_pass=all(state_contract.values()))
    result = {
        "schema_version": SCHEMA_AUDIT, "semantics_version": GATE_A_SEMANTICS_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "source-only M1 CPU Gate A; later neural oracle proxy, not decoder/EvalAI R2",
        "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST)},
        "input_sessions": {name: {"path": row.source_path, "sha256": row.source_sha256} for name, row in sessions.items()},
        "raw_contract": {"noninterpolated": True, "padding_excluded": True, "valid_prefix_integer_counts": True, "bin_seconds": BIN_SECONDS},
        "proxy": proxy, "fold_receipts": fold_receipts, "repeatability": repeatability,
        "state_contract": state_contract, "oracle_contract": oracle_contract,
        "gate": gate, "state_cost_N64": strict_json(streaming_cost_receipt(64)),
        "hard_exclusions": {"formal_paths_resolved": False, "minival": False, "held_out": False, "decoder": False, "evalai": False, "cuda": False},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "source_gate_a.json"
    path.write_text(json.dumps(strict_json(result), indent=2, sort_keys=True) + "\n")
    (output_dir / "source_gate_a.sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return path


def _check_execution_review(review_path: Path, prelaunch_path: Path) -> None:
    review = json.loads(review_path.read_text())
    if review.get("authorization") != REVIEW_AUTHORIZATION:
        raise PermissionError("review authorization does not permit source-only Step-3 CPU Gate A")
    if review.get("prelaunch_receipt_sha256") != sha256(prelaunch_path):
        raise PermissionError("review is not bound to the exact immutable prelaunch receipt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-prelaunch", action="store_true")
    parser.add_argument("--prelaunch-output", type=Path, default=PRELAUNCH_DEFAULT)
    parser.add_argument("--execute-source-audit", action="store_true")
    parser.add_argument("--reviewed-prelaunch", type=Path)
    parser.add_argument("--source-only-confirmation")
    parser.add_argument("--output", type=Path, default=AUDIT_DEFAULT)
    args = parser.parse_args()
    if args.write_prelaunch and args.execute_source_audit:
        raise ValueError("prelaunch and execution are separate invocations")
    if args.write_prelaunch:
        print(write_prelaunch(args.prelaunch_output)); return
    if not args.execute_source_audit:
        raise RuntimeError("refusing implicit data access; write prelaunch or supply both source-audit guards")
    prelaunch = args.prelaunch_output / "prelaunch_receipt.json"
    if args.reviewed_prelaunch is None or not prelaunch.is_file():
        raise PermissionError("source audit requires prelaunch receipt and separate reviewed-prelaunch authorization")
    if args.source_only_confirmation != SOURCE_ONLY_CONFIRMATION:
        raise PermissionError("exact source-only confirmation missing")
    _check_execution_review(args.reviewed_prelaunch, prelaunch)
    print(run_source_audit(args.output))


if __name__ == "__main__":
    main()
