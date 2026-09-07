#!/usr/bin/env python3
"""Real-data CPU engineering preflight for the isolated H1 M=4 EB pilot."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader

from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_SOURCE,
    H1_M4_FOLD0_TARGET,
    H1M4EBStrictTargetDataset,
    TrialBlocks,
    array_sha256,
    load_target_records,
    query_mutation_invariance,
    validate_target_receipt_binding,
)
from src.h1_m4_eb_pilot_contract import sha256_file, state_hash, write_immutable_json


PREFLIGHT_STATUS = "PASS_H1_M4_EB_FOLD0_REAL_DATA_CPU_PREFLIGHT_NONLAUNCH"


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rng_state() -> tuple[Any, Any, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def _set_rng_state(state: tuple[Any, Any, torch.Tensor]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _compose(root: Path, arm: str, cache_dir: Path):
    overrides = [
        f"experiment=h1_m4_eb_fold0_exploratory_pilot_{arm}",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        f"pilot.shared_cache_dir={cache_dir}",
        f"paths.root_dir={root}",
        f"paths.work_dir={root}",
        f"paths.data_dir={root / 'data'}",
        f"paths.output_dir={cache_dir / ('compose_' + arm)}",
        "hydra.run.dir=/tmp/h1_m4_eb_preflight_hydra",
        "extras.enforce_tags=false",
        "extras.print_config=false",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs"), job_name=f"h1_m4_{arm}"):
        config = compose(config_name="train.yaml", overrides=overrides)
    if config.trainer.accelerator != "cpu" or config.pilot.arm != arm:
        raise RuntimeError("preflight Hydra CPU/arm override failed")
    return config


def _materialize(model, identity: torch.Tensor) -> None:
    lazy = model.net.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(identity.permute(0, 1, 3, 2))


def _backward_smoke(model, batch, rng_state) -> dict[str, Any]:
    _set_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    before = state_hash(model.state_dict())
    loss, prediction, target, sessions = model.model_step(batch)
    loss.backward()
    after = state_hash(model.state_dict())
    if before != after:
        raise RuntimeError("forward/backward mutated parameter/buffer values before optimizer step")
    shared_gradient_hashes = {
        name: array_sha256(parameter.grad.detach().cpu().numpy())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and name != "net.eb_residual"
    }
    residual_gradient = model.net.eb_residual.grad
    return {
        "loss": float(loss.detach()),
        "prediction_sha256": array_sha256(prediction.detach().cpu().numpy()),
        "target_sha256": array_sha256(target.detach().cpu().numpy()),
        "sessions": list(sessions),
        "shared_gradient_sha256": shared_gradient_hashes,
        "residual_gradient_sha256": None if residual_gradient is None else array_sha256(residual_gradient.detach().cpu().numpy()),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_immutable_without_optimizer_step": before == after,
    }


def _target_forward_smoke(model, datasets: dict[str, H1M4EBStrictTargetDataset]) -> dict[str, Any]:
    model.eval()
    results: dict[str, Any] = {}
    for intervention, dataset in datasets.items():
        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
        neural, target, identity, sessions, carrier = batch
        before = state_hash(model.state_dict())
        random.seed(991)
        np.random.seed(991)
        torch.manual_seed(991)
        with torch.no_grad():
            output = model(neural.float(), calib_trialized_neural_features=identity.float(), carrier=carrier.float())
        after = state_hash(model.state_dict())
        if before != after or tuple(output.shape) != (2, 700, 7):
            raise RuntimeError(f"target {model.pilot_arm}/{intervention} smoke failed")
        results[intervention] = {
            "output_shape": list(output.shape),
            "output_sha256": array_sha256(output.cpu().numpy()),
            "target_sha256": array_sha256(target.numpy()),
            "carrier_sha256": array_sha256(carrier.numpy()),
            "sessions": list(sessions),
            "state_sha256_before": before,
            "state_sha256_after": after,
            "state_immutable": True,
        }
    return results


def _mutated_query_records(records, reference: H1M4EBStrictTargetDataset):
    mutated = dict(records)
    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        boundary = reference.support[name].query_first_bin
        neural = record.neural.copy()
        velocity = record.velocity.copy()
        neural[boundary:] += np.float32(3.0)
        velocity[boundary:] *= np.float32(-2.0)
        trials = []
        support_values = set(reference.support[name].trial_values)
        for trial in record.trials:
            if trial.trial_number in support_values:
                trials.append(trial)
            else:
                trials.append(
                    TrialBlocks(
                        trial.trial_number,
                        trial.rates + 7.0,
                        trial.velocity * -3.0,
                        trial.block_indices,
                    )
                )
        mutated[name] = replace(record, neural=neural, velocity=velocity, trials=tuple(trials))
    return mutated


def run_preflight(
    *,
    project_root: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES unset or empty")
    root = Path(project_root).resolve()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    configs = {arm: _compose(root, arm, cache) for arm in ("base", "joint")}
    config_hashes = {
        arm: hashlib.sha256(OmegaConf.to_yaml(configs[arm], resolve=True, sort_keys=True).encode()).hexdigest()
        for arm in configs
    }

    # Both independently instantiated DataModules must reconstruct the exact
    # same source windows, batches and all-epoch M=4 schedule.
    datamodules = {}
    for arm in ("base", "joint"):
        datamodule = hydra.utils.instantiate(configs[arm].data)
        datamodule.setup("fit")
        datamodules[arm] = datamodule
    base_dm, joint_dm = datamodules["base"], datamodules["joint"]
    parity_fields = (
        "source_window_indices_sha256",
        "batch_order_sha256",
        "calibration_schedule_sha256",
        "carrier_cache_sha256",
        "transform_sha256",
    )
    for field in parity_fields:
        if base_dm.pilot_manifest()[field] != joint_dm.pilot_manifest()[field]:
            raise RuntimeError(f"base/joint source manifest mismatch at {field}")
    if base_dm.pilot_manifest_sha256 != joint_dm.pilot_manifest_sha256:
        raise RuntimeError("base/joint complete source manifest SHA mismatch")
    base_dm.train_batch_sampler.reset_epoch()
    joint_dm.train_batch_sampler.reset_epoch()
    base_batch = next(iter(base_dm.train_dataloader()))
    joint_batch = next(iter(joint_dm.train_dataloader()))
    for index in (0, 1, 2, 4):
        if not torch.equal(base_batch[index], joint_batch[index]):
            raise RuntimeError(f"base/joint first actual batch tensor mismatch at field {index}")
    if list(base_batch[3]) != list(joint_batch[3]):
        raise RuntimeError("base/joint first actual batch session order mismatch")

    # Independent same-seed construction plus same-seed lazy materialization.
    _seed(42)
    base_model = hydra.utils.instantiate(configs["base"].model)
    post_constructor_rng = _rng_state()
    _seed(42)
    joint_model = hydra.utils.instantiate(configs["joint"].model)
    _set_rng_state(post_constructor_rng)
    _materialize(base_model, base_batch[2])
    post_lazy_rng = _rng_state()
    _set_rng_state(post_constructor_rng)
    _materialize(joint_model, joint_batch[2])
    if state_hash(base_model.state_dict()) != state_hash(joint_model.state_dict()):
        raise RuntimeError("base/joint state differs after synchronized LazyLinear materialization")
    shared_initial_state_sha = state_hash(base_model.state_dict())

    source_forward_rng = post_lazy_rng
    base_backward = _backward_smoke(base_model, base_batch, source_forward_rng)
    joint_backward = _backward_smoke(joint_model, joint_batch, source_forward_rng)
    if (
        base_backward["loss"] != joint_backward["loss"]
        or base_backward["prediction_sha256"] != joint_backward["prediction_sha256"]
        or base_backward["shared_gradient_sha256"] != joint_backward["shared_gradient_sha256"]
    ):
        raise RuntimeError("synchronized zero-residual source forward/backward parity failed")
    if base_backward["residual_gradient_sha256"] is not None or joint_backward["residual_gradient_sha256"] is None:
        raise RuntimeError("base/joint residual gradient trainability contract failed")

    # Only after source construction/backward checks, open the two target NWBs.
    target_records = load_target_records(root / "data/000954")
    validate_target_receipt_binding(
        target_records,
        base_dm.plan,
        base_dm.hparams.raw_receipt_path,
        base_dm.hparams.eb_receipt_path,
    )
    full = H1M4EBStrictTargetDataset(target_records, base_dm.plan, "full")
    variants = {name: full.with_intervention(name) for name in full.INTERVENTIONS}
    query_invariance = query_mutation_invariance(full)
    mutated_full = H1M4EBStrictTargetDataset(_mutated_query_records(target_records, full), base_dm.plan, "full")
    if mutated_full.support_and_carrier_hashes() != full.support_and_carrier_hashes():
        raise RuntimeError("mutating trial5+ changed support/carrier hashes")
    if mutated_full[0][0].tobytes() == full[0][0].tobytes():
        raise RuntimeError("query mutation smoke did not actually mutate a strict query window")
    target_smoke = {
        "base": _target_forward_smoke(base_model, variants),
        "joint": _target_forward_smoke(joint_model, variants),
    }
    original_sample = next(iter(DataLoader(full, batch_size=2, shuffle=False, num_workers=0)))
    mutated_sample = next(iter(DataLoader(mutated_full, batch_size=2, shuffle=False, num_workers=0)))
    before_prediction_state = state_hash(joint_model.state_dict())
    joint_model.eval()
    with torch.no_grad():
        original_prediction = joint_model(
            original_sample[0].float(),
            calib_trialized_neural_features=original_sample[2].float(),
            carrier=original_sample[4].float(),
        )
        mutated_prediction = joint_model(
            mutated_sample[0].float(),
            calib_trialized_neural_features=mutated_sample[2].float(),
            carrier=mutated_sample[4].float(),
        )
    after_prediction_state = state_hash(joint_model.state_dict())
    if torch.equal(original_prediction, mutated_prediction):
        raise RuntimeError("real query mutation did not change initialized-model predictions")
    if before_prediction_state != after_prediction_state:
        raise RuntimeError("query-mutation prediction comparison changed model state")
    query_mutation_prediction = {
        "original_prediction_sha256": array_sha256(original_prediction.numpy()),
        "mutated_prediction_sha256": array_sha256(mutated_prediction.numpy()),
        "predictions_changed": True,
        "model_state_immutable": True,
    }
    del base_model, joint_model

    source_hashes = {}
    for relative in (
        "src/data/h1_m4_eb_pilot.py",
        "src/data/h1_m4_eb_fold0_datamodule.py",
        "src/models/components/h1_m4_eb_residual_spint.py",
        "src/models/h1_m4_eb_pilot_module.py",
        "src/h1_m4_eb_pilot_contract.py",
        "scripts/h1_m4_eb_fold0_pilot_preflight.py",
        "scripts/h1_m4_eb_fold0_pilot_evaluate.py",
        "scripts/h1_m4_eb_fold0_paired_launcher.py",
        "configs/data/falcon_h1_m4_eb_fold0_pilot.yaml",
        "configs/model/falcon_h1_m4_eb_pilot.yaml",
        "configs/callbacks/h1_m4_eb_pilot_terminal.yaml",
        "configs/experiment/h1_m4_eb_fold0_exploratory_pilot_base.yaml",
        "configs/experiment/h1_m4_eb_fold0_exploratory_pilot_joint.yaml",
    ):
        source_hashes[relative] = sha256_file(root / relative)
    receipt = {
        "schema": "h1_m4_eb_fold0_real_data_cpu_preflight_v1",
        "status": PREFLIGHT_STATUS,
        "scope": {
            "opened": "exactly 13 public held-in-calib NWBs",
            "source_recordings": 11,
            "target_recordings": 2,
            "minival_opened_or_enumerated": False,
            "heldout_opened_or_enumerated": False,
            "gpu_constructed_or_launched": False,
            "fake_checkpoint_created": False,
        },
        "hydra": {
            "base_composed_and_instantiated": True,
            "joint_composed_and_instantiated": True,
            "resolved_config_sha256": config_hashes,
            "accelerator": "cpu",
        },
        "source_manifest": base_dm.pilot_manifest(),
        "source_manifest_sha256": base_dm.pilot_manifest_sha256,
        "all_source_input_sha256": {name: base_dm.records[name].input_sha256 for name in H1_M4_FOLD0_SOURCE},
        "all_target_input_sha256": {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
        "frozen_transform": base_dm.plan_manifest,
        "shared_initial_state_sha256_after_lazy_materialization": shared_initial_state_sha,
        "parity": {
            "fields": list(parity_fields),
            "same_ordered_batch_indices": True,
            "same_calibration_start_schedule": True,
            "same_first_actual_batch": True,
            "source_forward_backward": {"base": base_backward, "joint": joint_backward},
        },
        "target": {
            "strict_query_windows": len(full),
            "strict_query_window_indices_sha256": full.window_indices_sha256,
            "support_and_carrier_hashes": full.support_and_carrier_hashes(),
            "query_mutation_invariance": query_invariance,
            "actual_trial5plus_mutation_rebuild_invariant": True,
            "actual_query_mutation_prediction": query_mutation_prediction,
            "engineering_smoke_no_r2_selection": target_smoke,
            "all_four_interventions_carrier_only_and_nonidentity": True,
            "all_model_states_immutable": True,
        },
        "source_sha256": source_hashes,
        "launch": {
            "launch_authorized": False,
            "reason": "preflight is engineering evidence only; original H1 baseline terminal+seal and root review remain required",
            "requires_original_h1_baseline_terminal_marker": True,
            "requires_original_h1_baseline_seal_marker": True,
            "requires_explicit_root_review_marker": True,
        },
    }
    output, digest = write_immutable_json(output_path, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(project_root=args.project_root, cache_dir=args.cache_dir, output_path=args.output)
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
