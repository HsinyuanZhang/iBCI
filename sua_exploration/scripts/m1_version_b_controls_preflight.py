#!/usr/bin/env python3
"""Fail-closed local dry-run for M1 Version-B B4/RS4/LS4 controls.

This is not an execution authorization and has no launcher, Trainer, tmux,
SSH, CUDA, or remote synchronization path.  It binds the immutable expansion
precommit and pilot specification, composes the Full comparator and the three
separately trained controls, and optionally reconstructs their source-only
PCA/normalizer and target-M10 descriptors on CPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STREAMING_ROOT = REPO_ROOT / "streaming_calibration_exp"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

from src.data.falcon_emg_afc4_features import (  # noqa: E402
    AFC4_SUPPORT_TRIALS,
    SourceFrozenEMGAFC4Plan,
    _fit_encoding,
    deterministic_afc4_label_derangement,
    deterministic_afc4_row_permutation,
)
from sua_exploration.scripts.m1_version_b_preflight import canonical_sessions  # noqa: E402


PRECOMMIT = REPO_ROOT / "sua_exploration/results/m1_version_b_preflight/control_first_expansion_precommit_v1.json"
SPEC = REPO_ROOT / "sua_exploration/docs/M1_VERSION_B_CARRIERID_SOURCE_ONLY_PILOT_SPEC_20260809.md"
DATA_ROOT = REPO_ROOT / "SPINT-main/data/000941"
COMPARATOR = ("m1_version_b_c", "full")
CONTROLS: tuple[tuple[str, str], ...] = (
    ("m1_version_b_b4", "b4"),
    ("m1_version_b_rs4", "rs4"),
    ("m1_version_b_ls4", "ls4"),
)
EXPECTED_SOURCE = ("ses-20120926", "ses-20120927", "ses-20120928")
EXPECTED_TARGET = "ses-20120924"
EXPECTED_TEACHER_SHA256 = "f2921cabea819fed58b15e169f9cb899472416d30ee5a9b12c4c2087e96cb6be"


class ControlPreflightError(RuntimeError):
    """Frozen M1 Version-B control contract violation."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ControlPreflightError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _immutable(path: Path, label: str) -> None:
    _need(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} is not immutable mode 0444: {path}")


def bind_frozen_precommit() -> dict[str, Any]:
    _immutable(PRECOMMIT, "control-first precommit")
    _need(SPEC.is_file() and not SPEC.is_symlink(), f"Version-B spec missing: {SPEC}")
    body = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    _need(body.get("schema") == "m1_version_b_control_first_expansion_precommit_v1", "precommit schema drift")
    _need(
        body.get("status") == "FROZEN_CONTROL_FIRST_EXPANSION_BEFORE_BC_OR_HS_RESULT",
        "precommit status drift",
    )
    _need(body.get("mechanism_controls_in_fixed_order") == ["B-B4", "B-RS4", "B-LS4"],
          "precommitted control order drift")
    gates = body.get("mechanism_gates", {})
    _need(set(gates) == {
        "B-C_minus_B-B4_minimum_r2", "B-C_minus_B-LS4_minimum_r2", "B-C_minus_B-RS4_minimum_r2",
    } and all(float(value) == 0.03 for value in gates.values()), "mechanism gates drift")
    _need(body.get("next_folds_if_all_gates_pass") == [1, 2], "post-control fold order drift")
    _need(body.get("specification", {}).get("path") == str(SPEC.relative_to(REPO_ROOT)), "spec path drift")
    _need(body.get("specification", {}).get("sha256") == sha256_file(SPEC), "spec SHA drift")
    _need(
        body.get("training_time_control_requirement")
        == "All mechanism controls are separately trained with the same compact topology, source split, seed, fixed epoch and query windows; same-checkpoint interventions do not substitute for them.",
        "separate-training requirement drift",
    )
    return body


def _compose(experiment: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str((STREAMING_ROOT / "configs").resolve()), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                f"paths.root_dir={REPO_ROOT}",
                "hydra.job.chdir=false",
            ],
        )


def _plain_config(cfg: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    # Keep Hydra runtime interpolations (notably ``hydra:runtime.output_dir``)
    # symbolic: compose-only preflight deliberately does not initialize a
    # Hydra job or create an output directory. All experiment-dependent fields
    # are checked explicitly before this matched-structure hash is computed.
    value = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(value, dict), "resolved Hydra config is not a mapping")
    return value


def _matched_training_hash(cfg: Any) -> str:
    body = copy.deepcopy(_plain_config(cfg))
    for key in ("task_name", "run_id", "tags"):
        body.pop(key, None)
    body["data"]["afc4_arm"] = "<CONTROL_ARM>"
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_dry_run() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    matched_hashes: dict[str, str] = {}
    teacher_paths: set[str] = set()
    for experiment, arm in (COMPARATOR, *CONTROLS):
        cfg = _compose(experiment)
        _need(str(cfg.data.afc4_arm) == arm, f"{experiment}: afc4_arm drift")
        _need(str(cfg.model.variant) == "B3S" and int(cfg.model.hidden_dim) == 64,
              f"{experiment}: compact topology drift")
        _need(int(cfg.model.side_dim) == 4 and int(cfg.model.electrode_embed_dim) == 0,
              f"{experiment}: side/electrode width drift")
        _need(bool(cfg.model.freeze_decoder) is False, f"{experiment}: decoder must be trainable")
        _need(str(cfg.model.loss_mode) == "task_only", f"{experiment}: source loss drift")
        _need(float(cfg.model.lambda_y) == 0.0 and float(cfg.model.lambda_E) == 0.0,
              f"{experiment}: auxiliary loss drift")
        _need(cfg.ckpt_path is None and bool(cfg.train) and bool(cfg.test),
              f"{experiment}: control must train from fresh student initialization and terminal-test")
        _need(int(cfg.seed) == 42 and int(cfg.data.loso_fold) == 0,
              f"{experiment}: seed/fold drift")
        _need(tuple(cfg.data.source_session_names) == EXPECTED_SOURCE,
              f"{experiment}: source order drift")
        _need(int(cfg.data.calibration_n_trials) == AFC4_SUPPORT_TRIALS,
              f"{experiment}: calibration budget drift")
        _need(int(cfg.data.heldin_query_start_trial) == 10 and int(cfg.data.heldin_query_end_trial) == 210,
              f"{experiment}: query boundary drift")
        _need(not bool(cfg.data.include_heldout_in_fit) and not bool(cfg.data.include_heldout_in_test),
              f"{experiment}: held-out scope enabled")
        _need(int(cfg.trainer.min_epochs) == 12 and int(cfg.trainer.max_epochs) == 12,
              f"{experiment}: epoch budget drift")
        _need(int(cfg.trainer.limit_val_batches) == 0 and int(cfg.trainer.num_sanity_val_steps) == 0,
              f"{experiment}: target validation selector enabled")
        checkpoint = cfg.callbacks.fixed_last_checkpoint
        _need(checkpoint.monitor is None and not bool(checkpoint.save_last),
              f"{experiment}: checkpoint selection drift")
        _need(int(checkpoint.save_top_k) == -1 and int(checkpoint.every_n_epochs) == 12,
              f"{experiment}: fixed epoch-11 checkpoint drift")
        _need(cfg.callbacks.early_stopping is None, f"{experiment}: early stopping enabled")
        teacher = str(Path(str(cfg.model.teacher_ckpt_path)).resolve())
        teacher_paths.add(teacher)
        matched_hashes[experiment] = _matched_training_hash(cfg)
        rows[experiment] = {
            "arm": arm,
            "seed": int(cfg.seed),
            "fold": int(cfg.data.loso_fold),
            "source_sessions": list(cfg.data.source_session_names),
            "support_trials": [0, int(cfg.data.calibration_n_trials)],
            "query_trials": [int(cfg.data.heldin_query_start_trial), int(cfg.data.heldin_query_end_trial)],
            "fixed_epoch": 11,
            "fresh_training_ckpt_path": None,
            "target_backward": False,
            "matched_training_hash": matched_hashes[experiment],
        }
    _need(len(set(matched_hashes.values())) == 1, f"Full/control resolved configs are not matched: {matched_hashes}")
    _need(len(teacher_paths) == 1, f"Full/control teacher paths differ: {teacher_paths}")
    teacher_path = Path(next(iter(teacher_paths)))
    _need(teacher_path.is_file(), f"Version-B teacher checkpoint missing: {teacher_path}")
    _need(sha256_file(teacher_path) == EXPECTED_TEACHER_SHA256, "Version-B teacher checkpoint SHA drift")
    return {
        "arms": rows,
        "common_matched_training_hash": next(iter(matched_hashes.values())),
        "teacher_checkpoint": {"path": str(teacher_path), "sha256": sha256_file(teacher_path)},
    }


def _basis_and_normalizer_hash(plan: SourceFrozenEMGAFC4Plan) -> dict[str, str]:
    return {
        "pca_mean": array_sha256(plan.basis.mean),
        "pca_scale": array_sha256(plan.basis.scale),
        "pca_components": array_sha256(plan.basis.components),
        "normalizer_mean": array_sha256(plan.mean),
        "normalizer_std": array_sha256(plan.std),
    }


def live_source_semantics(data_root: Path = DATA_ROOT) -> dict[str, Any]:
    sessions = canonical_sessions(data_root)
    _need(tuple(sorted(sessions)) == tuple(sorted((EXPECTED_TARGET, *EXPECTED_SOURCE))),
          "live dry-run source session set drift")
    source_paths = {name: sessions[name] for name in EXPECTED_SOURCE}
    plans: dict[str, SourceFrozenEMGAFC4Plan] = {}
    carriers: dict[str, dict[str, np.ndarray]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    all_sessions = (*EXPECTED_SOURCE, EXPECTED_TARGET)
    for arm in ("full", "b4", "rs4", "ls4"):
        plan = SourceFrozenEMGAFC4Plan(source_paths, shuffle_seed=42)
        _need(plan.add_target(sessions[EXPECTED_TARGET]) == EXPECTED_TARGET,
              f"{arm}: target session binding drift")
        carriers[arm] = {name: plan.normalized(name, arm=arm) for name in all_sessions}
        plans[arm] = plan
        receipts[arm] = plan.receipt(arm=arm)

    hashes = {arm: _basis_and_normalizer_hash(plan) for arm, plan in plans.items()}
    _need(len({json.dumps(value, sort_keys=True) for value in hashes.values()}) == 1,
          f"Full/B4/RS4/LS4 PCA or normalizer differs: {hashes}")
    query_hashes = {arm: receipts[arm]["query_checksums"] for arm in receipts}
    _need(len({json.dumps(value, sort_keys=True) for value in query_hashes.values()}) == 1,
          "Full/control query-boundary checksums differ")

    session_rows: dict[str, Any] = {}
    for name in all_sessions:
        full, b4, rs4, ls4 = (carriers[arm][name] for arm in ("full", "b4", "rs4", "ls4"))
        _need(np.array_equal(b4[:, :3], np.zeros_like(b4[:, :3])), f"{name}: B4 kept non-rate coordinates")
        _need(np.array_equal(b4[:, 3], full[:, 3]), f"{name}: B4 changed normalized baseline rate")
        row_order = deterministic_afc4_row_permutation(full.shape[0], session_name=name, seed=42)
        _need(np.all(row_order != np.arange(full.shape[0])), f"{name}: RS4 has fixed rows")
        _need(np.array_equal(rs4, full[row_order]), f"{name}: RS4 is not the exact normalized-row permutation")

        ls_plan = plans["ls4"]
        record = ls_plan.support[name]
        score_order = deterministic_afc4_label_derangement(session_name=name, seed=42)
        _need(np.all(score_order != np.arange(AFC4_SUPPORT_TRIALS)), f"{name}: LS4 has fixed trial labels")
        scores = ls_plan.basis.project(record.support_emg)
        direct_raw, _ = _fit_encoding(scores[score_order], record.support_rates)
        direct = ((direct_raw - ls_plan.mean) / ls_plan.std).astype(np.float32)
        _need(np.array_equal(ls4, direct), f"{name}: LS4 differs from score-only pairing derangement")
        # The PCA inputs, firing rates and normalizer must remain exactly those
        # of the independently reconstructed Full plan.
        _need(np.array_equal(record.support_rates, plans["full"].support[name].support_rates),
              f"{name}: LS4 changed firing-rate rows")
        _need(np.array_equal(record.support_emg, plans["full"].support[name].support_emg),
              f"{name}: LS4 changed EMG values instead of only their pairing")
        session_rows[name] = {
            "full_sha256": array_sha256(full),
            "b4_sha256": array_sha256(b4),
            "rs4_sha256": array_sha256(rs4),
            "ls4_sha256": array_sha256(ls4),
            "row_permutation": row_order.tolist(),
            "label_derangement": score_order.tolist(),
            "support_rates_sha256": array_sha256(record.support_rates),
            "support_emg_sha256": array_sha256(record.support_emg),
            "query_checksum": str(receipts["full"]["query_checksums"][name]),
        }
    return {
        "source_files": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in sessions.items()
        },
        "shared_source_pca_normalizer": next(iter(hashes.values())),
        "sessions": session_rows,
        "target_materialization": {
            "session": EXPECTED_TARGET,
            "emg_trial_range": [0, AFC4_SUPPORT_TRIALS],
            "raw_spike_trial_range": [0, AFC4_SUPPORT_TRIALS],
            "query_emg_or_neural_values_used_by_carrier": False,
        },
    }


def live_datamodule_smoke() -> dict[str, Any]:
    """Build each control's real CPU datamodule without model or Trainer."""

    import gc
    from hydra.utils import instantiate

    rows: dict[str, Any] = {}
    source_sampler_hashes: set[str] = set()
    query_sampler_hashes: set[str] = set()
    query_window_hashes: set[str] = set()
    for experiment, arm in CONTROLS:
        cfg = _compose(experiment)
        datamodule = instantiate(cfg.data)
        datamodule.setup("fit")
        manifest = datamodule.get_split_manifest()
        _need(manifest.get("carrier_arm") == arm, f"{experiment}: live manifest arm drift")
        _need(manifest.get("outer_fold") == 0 and manifest.get("outer_left_out") == EXPECTED_TARGET,
              f"{experiment}: live fold/target drift")
        _need(tuple(manifest.get("train_sessions", ())) == EXPECTED_SOURCE,
              f"{experiment}: live source order drift")
        _need(manifest.get("support_trials") == [0, 10] and manifest.get("query_trials") == [10, 210],
              f"{experiment}: live support/query boundary drift")
        _need(manifest.get("minival_files_opened") is False and manifest.get("heldout_files_opened") is False,
              f"{experiment}: live scope widened")
        _need(manifest.get("target_backpropagation") is False, f"{experiment}: target backward enabled")
        _need(manifest.get("target_query_values_used_for_optimizer_or_checkpoint_selection") is False,
              f"{experiment}: target query selector enabled")
        _need(manifest.get("checkpoint_selection") == "fixed_last_epoch_11_train_source_only",
              f"{experiment}: checkpoint policy drift")
        _need(len(datamodule.train_dataset) == 158487, f"{experiment}: source window count drift")
        target_audit = manifest["query_window_audit"][EXPECTED_TARGET]
        _need(target_audit.get("eligible_windows") == 26517, f"{experiment}: target eligible-window count drift")
        _need(manifest.get("query_scored_windows") == 26496, f"{experiment}: scored query window count drift")
        _need(manifest.get("query_batch_count") == 828, f"{experiment}: query batch count drift")
        receipt = manifest.get("carrier_plan_receipt", {})
        _need(receipt.get("arm") == arm, f"{experiment}: arm-specific plan receipt lost")
        target_scope = receipt.get("materialization_scope", {}).get(EXPECTED_TARGET, {})
        _need(target_scope.get("emg_trial_range") == [0, 10], f"{experiment}: target EMG scope drift")
        _need(target_scope.get("raw_spike_trial_range") == [0, 10], f"{experiment}: target spike scope drift")
        _need(target_scope.get("target_query_emg_or_neural_values_read") is False,
              f"{experiment}: carrier loader read target query values")
        sample = datamodule.val_heldin_dataset[0]
        _need(len(sample) == 5 and str(sample[3]) == EXPECTED_TARGET,
              f"{experiment}: live dataset side tuple drift")
        expected_side = datamodule.carrier_plan.normalized(EXPECTED_TARGET, arm=arm)
        _need(np.array_equal(np.asarray(sample[4]), expected_side),
              f"{experiment}: live dataset did not deliver exact {arm} carrier")
        source_sampler_hashes.add(str(manifest["train_sampler_sha256"]))
        query_sampler_hashes.add(str(manifest["query_sampler_sha256"]))
        query_window_hashes.add(
            hashlib.sha256(
                json.dumps(manifest["query_window_audit"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        rows[experiment] = {
            "arm": arm,
            "source_windows": len(datamodule.train_dataset),
            "source_scored_windows_per_epoch": int(manifest["train_scored_windows_per_epoch"]),
            "source_sampler_sha256": str(manifest["train_sampler_sha256"]),
            "target_eligible_windows": int(target_audit["eligible_windows"]),
            "target_scored_windows": int(manifest["query_scored_windows"]),
            "target_query_batches": int(manifest["query_batch_count"]),
            "target_sampler_sha256": str(manifest["query_sampler_sha256"]),
            "carrier_plan_arm": str(receipt["arm"]),
        }
        del datamodule
        gc.collect()
    _need(len(source_sampler_hashes) == 1, "control source sampler hashes differ")
    _need(len(query_sampler_hashes) == 1, "control target sampler hashes differ")
    _need(len(query_window_hashes) == 1, "control query-window audits differ")
    return {
        "controls": rows,
        "common_source_sampler_sha256": next(iter(source_sampler_hashes)),
        "common_target_sampler_sha256": next(iter(query_sampler_hashes)),
        "common_query_window_audit_sha256": next(iter(query_window_hashes)),
    }


def build_receipt(*, live_data: bool = True, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    precommit = bind_frozen_precommit()
    body: dict[str, Any] = {
        "schema": "m1_version_b_controls_local_preflight_v1",
        "status": "PASS_LOCAL_CONTROL_IMPLEMENTATION_DRY_RUN_NOT_EXECUTION_AUTHORIZATION",
        "execution_authorized": False,
        "gpu_started": False,
        "remote_connected_or_synchronized": False,
        "main_gate_result_read": False,
        "scope": {
            "task": "M1 Version-B fold0 development",
            "seed": 42,
            "source_sessions": list(EXPECTED_SOURCE),
            "target_session": EXPECTED_TARGET,
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "heldout_or_formal_opened": False,
            "minival_opened": False,
            "target_backward": False,
        },
        "precommit": {
            "path": str(PRECOMMIT),
            "sha256": sha256_file(PRECOMMIT),
            "schema": precommit["schema"],
            "status": precommit["status"],
        },
        "specification": {"path": str(SPEC), "sha256": sha256_file(SPEC)},
        "config_dry_run": config_dry_run(),
        "live_source_semantics": live_source_semantics(data_root) if live_data else "SKIPPED_BY_EXPLICIT_TEST_FLAG",
        "live_datamodule_smoke": live_datamodule_smoke() if live_data else "SKIPPED_BY_EXPLICIT_TEST_FLAG",
    }
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--skip-live-data", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional local audit JSON; never interpreted as execution authorization",
    )
    args = parser.parse_args()
    body = build_receipt(live_data=not args.skip_live_data, data_root=args.data_root)
    encoded = json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"control preflight refuses to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
