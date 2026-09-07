"""CPU source/local sanity audit for the isolated released-SPINT H1 baseline.

This command opens only the 13 held-in calibration and matching minival files,
checks the actual random-calibration semantics, runs one CPU model forward,
and writes an immutable receipt.  It never opens ``held-out-calib`` and does
not start a trainer or occupy a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.falcon_datamodule import random as falcon_random
from src.data.h1_baseline_datamodule import (
    H1_BASELINE_HELDIN_SESSIONS,
    H1_BASELINE_HELDOUT_SESSIONS,
    H1BaselineDataModule,
    h1_session_date,
)
from src.models.components.spint import SpintModel


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_reference() -> dict[str, Any]:
    """Bind the receipt to the installed official FALCON metric implementation."""

    expected_version = "1.0.2"
    expected_sha256 = "2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418"
    try:
        version = importlib.metadata.version("falcon-challenge")
        evaluator = importlib.import_module("falcon_challenge.evaluator")
    except Exception as exc:  # pragma: no cover - exercised only on an invalid host
        raise RuntimeError("official falcon-challenge evaluator is unavailable") from exc
    evaluator_path = Path(evaluator.__file__).resolve()
    observed_sha256 = _sha256_file(evaluator_path)
    if version != expected_version or observed_sha256 != expected_sha256:
        raise RuntimeError(
            "installed FALCON evaluator drifted from the frozen reference "
            f"version/hash ({version}, {observed_sha256})"
        )
    return {
        "package": "falcon-challenge",
        "version": version,
        "evaluator_module": "falcon_challenge.evaluator",
        "evaluator_sha256": observed_sha256,
        "evaluator_path": str(evaluator_path),
    }


def _configuration_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    relative = (
        "src/data/h1_baseline_datamodule.py",
        "src/models/falcon_module.py",
        "src/models/components/spint.py",
        "scripts/h1_baseline_preflight.py",
        "scripts/h1_baseline_eval.py",
        "scripts/run_h1_baseline_staged.sh",
        "configs/data/falcon_h1_baseline.yaml",
        "configs/model/falcon_h1_baseline.yaml",
        "configs/callbacks/h1_baseline_fixed_epoch.yaml",
        "configs/experiment/h1_baseline_paper_lr.yaml",
        "configs/experiment/h1_baseline_released_code_lr.yaml",
    )
    return {path: _sha256_file(root / path) for path in relative}


def _recursive_config_diff(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_recursive_config_diff(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        return [] if left == right else [prefix]
    return [] if left == right else [prefix]


def _validate_resolved_variant_diff(paper: dict[str, Any], released: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless composed variants differ only in LR/declared metadata."""

    all_differences = _recursive_config_diff(paper, released)
    scientific_whitelist = {"model.optimizer.lr"}
    metadata_whitelist = {"protocol_variant", "comparison_role", "task_name", "tags"}
    unexpected = [path for path in all_differences if path not in scientific_whitelist | metadata_whitelist]
    if unexpected:
        raise ValueError(f"H1 baseline paper/released resolved config drift is not LR-only: {unexpected}")
    return {
        "resolved_config_diff_paths": all_differences,
        "scientific_difference_whitelist": sorted(scientific_whitelist),
        "metadata_difference_whitelist": sorted(metadata_whitelist),
        "unexpected_differences": unexpected,
        "paper_optimizer_lr": paper["model"]["optimizer"]["lr"],
        "released_code_optimizer_lr": released["model"]["optimizer"]["lr"],
        "paper_resolved_constants": {
            "data": paper["data"],
            "model_net": paper["model"]["net"],
            "trainer": paper["trainer"],
            "callbacks": paper["callbacks"],
            "seed": paper["seed"],
        },
        "released_resolved_constants": {
            "data": released["data"],
            "model_net": released["model"]["net"],
            "trainer": released["trainer"],
            "callbacks": released["callbacks"],
            "seed": released["seed"],
        },
        "roles_frozen_before_results": True,
        "primary_role": "released_code_lr_5e-5",
        "sensitivity_role": "paper_lr_1e-5",
    }


def _resolved_variant_diff_evidence() -> dict[str, Any]:
    """Compose both Hydra variants and fail closed on any non-whitelisted diff."""

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    overrides_common = [
        "paths.root_dir=/tmp/h1_baseline_resolved_diff",
        "paths.output_dir=/tmp/h1_baseline_resolved_diff",
        "paths.work_dir=/tmp/h1_baseline_resolved_diff",
        "hydra.run.dir=/tmp/h1_baseline_resolved_diff",
        "extras.enforce_tags=false",
        "extras.print_config=false",
    ]

    def compose_variant(name: str) -> dict[str, Any]:
        with initialize_config_dir(
            version_base="1.3", config_dir=str((root / "configs").resolve()), job_name=f"h1_diff_{name}"
        ):
            config = compose(
                config_name="train.yaml",
                overrides=[f"experiment={name}", *overrides_common],
            )
            return OmegaConf.to_container(config, resolve=True)

    paper = compose_variant("h1_baseline_paper_lr")
    released = compose_variant("h1_baseline_released_code_lr")

    return _validate_resolved_variant_diff(paper, released)


def _assert_random_calibration_semantics(module: H1BaselineDataModule) -> dict[str, Any]:
    dataset = module.train_dataset
    session = H1_BASELINE_HELDIN_SESSIONS[0]
    sample_index = next(
        index
        for index, (session_name, _start) in enumerate(dataset.window_indices)
        if session_name == session
    )
    original_randint = falcon_random.randint
    try:
        falcon_random.randint = lambda low, high: low
        low_features = dataset[sample_index][2]
        falcon_random.randint = lambda low, high: high
        high_features = dataset[sample_index][2]
    finally:
        falcon_random.randint = original_randint
    source_features = dataset.calib_trialized_neural_features[session]
    np.testing.assert_allclose(low_features, source_features[:2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(high_features, source_features[-2:], rtol=0.0, atol=0.0)

    val_index = next(
        index
        for index, (session_name, _start) in enumerate(module.val_heldin_dataset.window_indices)
        if session_name == session
    )
    val_a = module.val_heldin_dataset[val_index][2]
    val_b = module.val_heldin_dataset[val_index][2]
    np.testing.assert_allclose(val_a, val_b, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(val_a, source_features[:2], rtol=0.0, atol=0.0)
    return {
        "train_selector": "uniform contiguous start randint(0, total_trials-2) per sample",
        "train_low_start_shape": list(low_features.shape),
        "train_high_start_shape": list(high_features.shape),
        "heldin_validation_selector": "deterministic first two trials",
        "heldin_validation_shape": list(val_a.shape),
    }


def _model_forward(module: H1BaselineDataModule) -> dict[str, Any]:
    sample_index = 0
    neural, _behavior, calibration, _session = module.val_heldin_dataset[sample_index]
    net = SpintModel(
        model_dim=1024,
        num_covariates=7,
        window_size=700,
        num_heads=64,
        num_layers=1,
        num_id_layers=3,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=True,
        dynamic_dropout_low=0.0,
        dynamic_dropout_high=1.0,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    ).eval()
    with torch.no_grad():
        output = net(
            torch.as_tensor(neural, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(calibration, dtype=torch.float32).unsqueeze(0),
        )
    parameter_count = sum(parameter.numel() for parameter in net.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in net.parameters())
    return {
        "output_shape": list(output.shape),
        "parameter_count": int(parameter_count),
        "parameter_bytes_fp32": int(parameter_bytes),
        "model_dim": 1024,
        "num_heads": 64,
        "dynamic_dropout": True,
    }


def run_preflight(data_dir: str | Path) -> dict[str, Any]:
    module = H1BaselineDataModule(
        task="h1",
        data_dir=str(data_dir),
        heldin_session_names=H1_BASELINE_HELDIN_SESSIONS,
        batch_size=32,
        window_size=700,
        calibration_n_trials=2,
        random_calibration=True,
        smooth_calibration=False,
        max_trial_length=1024,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        num_workers=0,
        pin_memory=False,
    )
    module.setup("fit")
    manifest = module.input_manifest()
    assert manifest["formal_heldout_opened"] is False
    assert len(manifest["files"]) == 26
    assert all("held-out" not in row["path"] for row in manifest["files"])
    assert set(H1_BASELINE_HELDOUT_SESSIONS).isdisjoint(H1_BASELINE_HELDIN_SESSIONS)
    random_semantics = _assert_random_calibration_semantics(module)
    model = _model_forward(module)

    date_groups: dict[str, list[str]] = defaultdict(list)
    for session in H1_BASELINE_HELDIN_SESSIONS:
        date_groups[h1_session_date(session)].append(session)
    train_batches = len(module.train_batch_sampler)
    val_batches = len(module.val_heldin_batch_sampler)
    parameter_bytes = model["parameter_bytes_fp32"]
    input_manifest_sha = _canonical_sha256(manifest)
    return {
        "schema": "spint_h1_baseline_cpu_preflight_v3",
        "status": "PASS_H1_BASELINE_PREFLIGHT_HELDIN_ONLY",
        "protocol": "spint_h1_baseline_reproduction_v1",
        "receipt_scope": "source_only_cpu_preflight_and_heldin_local_sanity",
        "data_scope": {
            "task": "h1",
            "heldin_recordings": 13,
            "heldin_date_groups": {key: sorted(value) for key, value in sorted(date_groups.items())},
            "discrete_direction_labels_available": False,
            "dense_7d_velocity_covariates_available": True,
            "spint_id_input": "neural_only_trialized_calibration_features",
            "formal_heldout_opened": False,
            "formal_heldout_recordings_forbidden": list(H1_BASELINE_HELDOUT_SESSIONS),
        },
        "protocol_constants": {
            "calibration_n_trials": 2,
            "random_calibration": True,
            "window_size": 700,
            "max_trial_length": 1024,
            "model_dim": 1024,
            "num_heads": 64,
            "dynamic_dropout": True,
            "fixed_epoch_budget": 50,
            "validation_selects_epoch": False,
            "paper_lr": 1.0e-5,
            "released_code_lr": 5.0e-5,
            "paper_behavior_scale": 0.05,
            "released_code_behavior_scaling_factor": 20.0,
            "behavior_scale_equivalence": "paper target scale 0.05 equals released-code forward divide by 20",
            "trainer_deterministic": False,
            "seed": 42,
        },
        "variant_roles": {
            "released_code_lr_5e-5": "implementation_primary_public_repository_reproduction",
            "paper_lr_1e-5": "appendix_paper_reference_sensitivity",
            "roles_frozen_before_results": True,
            "do_not_swap_names_by_result": True,
        },
        "execution_only_deviations": {
            "num_workers": {
                "upstream_released_value": None,
                "reproduction_value": 0,
                "reason": "The upstream FalconDataModule normalizes null to os.cpu_count()-1; the isolated module explicitly uses zero workers to avoid host-dependent worker fan-out while preserving ordering and calibration semantics.",
                "changes_support_or_labels": False,
            },
            "trainer_deterministic": {
                "released_default": False,
                "reproduction_value": False,
                "reason": "Keep the public trainer default; seed=42 is retained but deterministic kernels are not claimed.",
            },
        },
        "random_calibration_semantics": random_semantics,
        "dataset_volume": {
            "train_windows": len(module.train_dataset),
            "heldin_validation_windows": len(module.val_heldin_dataset),
            "train_batches_per_epoch": train_batches,
            "heldin_validation_batches_per_epoch": val_batches,
            "fixed_epoch_train_optimizer_steps": train_batches * 50,
            "fixed_epoch_heldin_validation_batches": val_batches * 50,
        },
        "model_sanity": model,
        "memory_and_run_estimate": {
            "adam_fp32_parameter_gradient_and_two_state_lower_bound_bytes": int(parameter_bytes * 4),
            "adam_fp32_parameter_gradient_and_two_state_lower_bound_gib": float(parameter_bytes * 4 / 2**30),
            "input_batch_fp32_bytes": int((32 * 700 * 176 + 32 * 2 * 1024 * 176 + 32 * 700 * 7) * 4),
            "paper_reported_peak_gpu_memory": "<2 GiB on A40 for batch32 H1",
            "paper_reported_training_time_hours": "approximately 8 hours for 50 epochs",
            "local_peak_gpu_memory_measured": False,
            "local_runtime_measured": False,
            "scheduling_note": "The paper's A40/time figures are reported references, not a local RTX3090 measurement; obtain a separately authorized local measurement after RT is sealed.",
        },
        "input_manifest": manifest,
        "input_manifest_sha256": input_manifest_sha,
        "source_and_configuration_sha256": _configuration_hashes(),
        "metric_convention": {
            "evaluator": "falcon_challenge.evaluator.compute_metrics_regression",
            "day_reduce": "same-date recordings concatenated before variance-weighted R2; six date-group R2 values equally averaged",
            "std": "np.std ddof=0 population (official primary); ddof=1 sample std is secondary only",
            "date_groups": 6,
        },
        "metric_reference": _metric_reference(),
        "resolved_variant_diff": _resolved_variant_diff_evidence(),
        "fit_and_validation_scope": {
            "fit_scope": "all_13_heldin_calib_recordings",
            "minival_scope": "all_13_matching_heldin_minival_recordings_validation_only",
            "formal_heldout_scope": "forbidden_and_not_opened",
        },
        "comparison_boundary": {
            "q3_pilot_is_not_spint_baseline": True,
            "q3_pilot_protocol": "source-only nested LOSO, q3 analytic carrier, target one-shot; not all-13 SPINT ID reproduction",
            "baseline_generalization": "all 13 held-in recordings together",
            "future_generalization_grouping": "prefer six held-in calendar-date groups before session-level splits",
            "terminal_official_metric": "concatenate recordings sharing h1_session_date, then equal-mean six date-group variance-weighted R2; 13-recording diagnostics remain secondary",
            "carrier_input_difference": "SPINT ID consumes neural-only calibration features; afc4_kin would consume dense 7-D velocity covariates",
            "fixed_epoch_checkpoint_note": "With Lightning's zero-based epoch placeholder, the terminal 50-epoch checkpoint is normally named epoch_049.ckpt; the receipt must report epochs_completed=50 and global_step rather than infer completion from the filename.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable preflight receipt: {output}")
    receipt = run_preflight(args.data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "output": str(output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
