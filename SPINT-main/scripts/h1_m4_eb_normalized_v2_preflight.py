#!/usr/bin/env python3
"""Real-data CPU preflight for the H1 M=4 EB normalized-V2 repair.

The source-only section never opens a target recording and never enumerates
minival, heldout, formal, or EvalAI paths.  Numerical-life checks
operate on the first real source batch and a disposable clone; no checkpoint
or root-review marker is produced.
"""
from __future__ import annotations

import argparse
import copy
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

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader

from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2DataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    V2_RECEIPT_SCHEMA,
    NormalizedV2ContractError,
    array_sha256,
    assert_state_immutable,
    canonical_sha256,
    fit_source_scalar_normalizer,
    sha256_file,
    state_hash,
    validate_intervention_normalization_algebra,
    validate_raw_normalized_roundtrip,
    write_immutable_json,
)
from src.models.components.h1_m4_eb_normalized_v2_residual_spint import H1M4EBNormalizedV2ResidualSpint
from src.models.h1_m4_eb_normalized_v2_module import H1M4EBNormalizedV2PilotLitModule


PREFLIGHT_STATUS = "PASS_H1_M4_EB_NORMALIZED_V2_REAL_DATA_CPU_PREFLIGHT_NONLAUNCH"
DEFAULT_RAW = PROJECT_ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
DEFAULT_EB = PROJECT_ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"


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


def _new_module(arm: str) -> H1M4EBNormalizedV2PilotLitModule:
    net = H1M4EBNormalizedV2ResidualSpint(
        train_residual=arm == "joint",
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
    )
    return H1M4EBNormalizedV2PilotLitModule(
        task="h1",
        net=net,
        pilot_arm=arm,
        fold_date="19250101",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=20.0,
        optimizer=torch.optim.Adam,
        scheduler=None,
        compile=False,
        clean_teacher=True,
        scheduler_monitor="val_heldin/r2_mean",
    )


def _materialize(model: H1M4EBNormalizedV2PilotLitModule, identity: torch.Tensor) -> None:
    lazy = model.net.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(identity.permute(0, 1, 3, 2))


def _batch(dm: H1M4EBNormalizedV2DataModule):
    dm.train_batch_sampler.reset_epoch()
    return next(iter(dm.train_dataloader()))


def _backward(model: H1M4EBNormalizedV2PilotLitModule, batch, rng_state) -> dict[str, Any]:
    _set_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    before = state_hash(model.state_dict())
    loss, prediction, target, sessions = model.model_step(batch)
    loss.backward()
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, f"{model.pilot_arm} source forward/backward")
    grad = model.net.eb_residual.grad
    shared_hashes = {
        name: array_sha256(parameter.grad.detach().cpu().numpy())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and name != "net.eb_residual"
    }
    return {
        "loss": float(loss.detach()),
        "prediction_sha256": array_sha256(prediction.detach().cpu().numpy()),
        "target_sha256": array_sha256(target.detach().cpu().numpy()),
        "sessions": list(sessions),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_immutable_without_optimizer_step": before == after,
        "shared_gradient_sha256": shared_hashes,
        "residual_gradient_sha256": None if grad is None else array_sha256(grad.detach().cpu().numpy()),
        "residual_grad_rms": None if grad is None else float(torch.sqrt(torch.mean(torch.square(grad))).detach()),
        "residual_grad_max": None if grad is None else float(torch.max(torch.abs(grad)).detach()),
    }


def _clone_step(
    model: H1M4EBNormalizedV2PilotLitModule,
    batch,
    carrier: torch.Tensor,
    rng_state,
    *,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    """One disposable Adam step and post-step carrier/identity measurements."""

    clone = copy.deepcopy(model.net)
    clone.train()
    clone.zero_grad(set_to_none=True)
    neural, target, identity, _sessions, _carrier = batch
    _set_rng_state(rng_state)
    prediction = clone(neural.float(), calib_trialized_neural_features=identity.float(), carrier=carrier.float())
    prediction = prediction[:, -1:, :] / 20.0
    target_last = target[:, -1:, :].float()
    loss = torch.nn.functional.mse_loss(prediction, target_last)
    loss.backward()
    grad = clone.eb_residual.grad.detach().clone()
    if not torch.isfinite(grad).all() or torch.count_nonzero(grad).item() == 0:
        raise NormalizedV2ContractError("normalized V2 residual gradient is zero/nonfinite")
    grad_rms = float(torch.sqrt(torch.mean(torch.square(grad))))
    grad_max = float(torch.max(torch.abs(grad)))
    if grad_rms / eps <= 1.0:
        raise NormalizedV2ContractError("normalized V2 gradient/Adam_eps is not >1")
    before_w = clone.eb_residual.detach().clone()
    optimizer = torch.optim.Adam(clone.parameters(), lr=5.0e-5, weight_decay=0.0, eps=eps)
    optimizer.step()
    after_w = clone.eb_residual.detach().clone()
    delta = after_w - before_w
    delta_rms = float(torch.sqrt(torch.mean(torch.square(delta))))
    before_rms = float(torch.sqrt(torch.mean(torch.square(before_w))))
    after_rms = float(torch.sqrt(torch.mean(torch.square(after_w))))
    if not torch.isfinite(after_w).all() or delta_rms == 0.0:
        raise NormalizedV2ContractError("normalized V2 Adam clone step was zero/nonfinite")
    # The original identity is measured before adding C@W.
    with torch.no_grad():
        original_identity = clone.identity_projection(identity.float())
        carrier_effect = carrier.float() @ after_w
        _set_rng_state(rng_state)
        clone.eval()
        full_output = clone(neural.float(), calib_trialized_neural_features=identity.float(), carrier=carrier.float())
        zero_output = clone(
            neural.float(),
            calib_trialized_neural_features=identity.float(),
            carrier=torch.zeros_like(carrier),
        )
    difference = full_output - zero_output
    result = {
        "grad_rms": grad_rms,
        "grad_max": grad_max,
        "grad_rms_over_adam_eps": grad_rms / eps,
        "adam_eps": eps,
        "W_rms_before": before_rms,
        "W_rms_after": after_rms,
        "deltaW_rms": delta_rms,
        "deltaW_over_W_rms": delta_rms / max(before_rms, 1.0e-12),
        "step_normalized_CatW_rms": float(torch.sqrt(torch.mean(torch.square(carrier_effect)))),
        "original_identity_rms": float(torch.sqrt(torch.mean(torch.square(original_identity)))),
        "step_normalized_CatW_over_identity_rms": float(
            torch.sqrt(torch.mean(torch.square(carrier_effect)))
            / max(float(torch.sqrt(torch.mean(torch.square(original_identity)))), 1.0e-12)
        ),
        "full_vs_zero_source_prediction_max_abs": float(torch.max(torch.abs(difference))),
        "full_vs_zero_source_prediction_rms": float(torch.sqrt(torch.mean(torch.square(difference)))),
        "full_vs_zero_source_prediction_non_bit_identical": not torch.equal(full_output, zero_output),
        "post_step_W_sha256": array_sha256(after_w.cpu().numpy()),
    }
    if not result["full_vs_zero_source_prediction_non_bit_identical"]:
        raise NormalizedV2ContractError("normalized V2 Full-vs-Zero source prediction is bit-identical")
    return result


def _raw_coordinate_diagnostic(model, batch, raw_carrier: torch.Tensor, rng_state) -> dict[str, Any]:
    """Record V1-coordinate clone behavior for diagnosis only; never gate V2."""

    clone = copy.deepcopy(model.net)
    clone.train()
    clone.zero_grad(set_to_none=True)
    neural, target, identity, _sessions, _carrier = batch
    _set_rng_state(rng_state)
    prediction = clone(neural.float(), calib_trialized_neural_features=identity.float(), carrier=raw_carrier.float())
    loss = torch.nn.functional.mse_loss(prediction[:, -1:, :] / 20.0, target[:, -1:, :].float())
    loss.backward()
    grad = clone.eb_residual.grad.detach().clone()
    optimizer = torch.optim.Adam(clone.parameters(), lr=5.0e-5, weight_decay=0.0)
    optimizer.step()
    delta = clone.eb_residual.detach()
    return {
        "coordinate": "V1_raw_diagnostic_only",
        "grad_rms": float(torch.sqrt(torch.mean(torch.square(grad)))),
        "grad_max": float(torch.max(torch.abs(grad))),
        "deltaW_rms": float(torch.sqrt(torch.mean(torch.square(delta)))),
        "used_for_v2_selection": False,
    }


def run_preflight(
    *,
    project_root: str | Path = PROJECT_ROOT,
    cache_dir: str | Path,
    output_path: str | Path,
    raw_receipt_path: str | Path = DEFAULT_RAW,
    eb_receipt_path: str | Path = DEFAULT_EB,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise NormalizedV2ContractError("normalized V2 CPU preflight requires CUDA_VISIBLE_DEVICES unset or empty")
    root = Path(project_root).resolve()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if not (root / "data/000954/sub-HumanPitt-held-in-calib").is_dir():
        raise FileNotFoundError("V2 preflight requires the public held-in-calib directory")

    # Separate caches demonstrate independent base/joint reconstruction while
    # all V1 estimator inputs remain source-only.
    datamodules: dict[str, H1M4EBNormalizedV2DataModule] = {}
    for arm in ("base", "joint"):
        dm = H1M4EBNormalizedV2DataModule(
            task="h1",
            data_dir=str(root / "data/000954"),
            raw_receipt_path=str(Path(raw_receipt_path).resolve()),
            eb_receipt_path=str(Path(eb_receipt_path).resolve()),
            cache_dir=str(cache / arm),
        )
        dm.setup("fit")
        datamodules[arm] = dm
    base_dm, joint_dm = datamodules["base"], datamodules["joint"]
    parity_fields = (
        "source_window_indices_sha256",
        "batch_order_sha256",
        "calibration_schedule_sha256",
        "carrier_cache_sha256",
        "normalized_cache_sha256",
        "normalizer_sha256",
        "transform_sha256",
    )
    for field in parity_fields:
        if base_dm.pilot_manifest()[field] != joint_dm.pilot_manifest()[field]:
            raise NormalizedV2ContractError(f"base/joint V2 source manifest mismatch at {field}")
    if base_dm.pilot_manifest_sha256 != joint_dm.pilot_manifest_sha256:
        raise NormalizedV2ContractError("base/joint V2 complete source manifest SHA mismatch")
    base_batch = _batch(base_dm)
    joint_batch = _batch(joint_dm)
    for index in (0, 1, 2, 4):
        if not torch.equal(base_batch[index], joint_batch[index]):
            raise NormalizedV2ContractError(f"base/joint V2 first source batch mismatch at field {index}")
    if list(base_batch[3]) != list(joint_batch[3]):
        raise NormalizedV2ContractError("base/joint V2 source session order mismatch")

    # Source-only normalization algebra and global RMS proof.
    normalizer = base_dm.normalizer
    # Use the exact first scheduled source request (window index plus M=4
    # calibration start), not a synthetic or target sample.
    first_index = int(base_dm.train_batch_sampler.batches[0][0])
    first_start = int(base_dm.train_batch_sampler.schedule[0, 0])
    raw_first = base_dm.train_dataset.raw_carrier((first_index, first_start))
    roundtrip = validate_raw_normalized_roundtrip(normalizer, raw_first)
    algebra = validate_intervention_normalization_algebra(normalizer)
    if not np.isclose(normalizer.normalized_global_rms, 1.0, rtol=1e-12, atol=1e-12):
        raise NormalizedV2ContractError("source normalized global RMS is not approximately one")

    # Synchronized base/joint constructor and LazyLinear materialization.
    _seed(42)
    base_model = _new_module("base")
    constructor_rng = _rng_state()
    _seed(42)
    joint_model = _new_module("joint")
    _set_rng_state(constructor_rng)
    _materialize(base_model, base_batch[2])
    post_lazy_rng = _rng_state()
    _set_rng_state(constructor_rng)
    _materialize(joint_model, joint_batch[2])
    if state_hash(base_model.state_dict()) != state_hash(joint_model.state_dict()):
        raise NormalizedV2ContractError("base/joint V2 state differs after synchronized materialization")
    shared_initial_state_sha = state_hash(base_model.state_dict())
    source_forward_rng = post_lazy_rng
    base_backward = _backward(base_model, base_batch, source_forward_rng)
    joint_backward = _backward(joint_model, joint_batch, source_forward_rng)
    if (
        base_backward["loss"] != joint_backward["loss"]
        or base_backward["prediction_sha256"] != joint_backward["prediction_sha256"]
        or base_backward["shared_gradient_sha256"] != joint_backward["shared_gradient_sha256"]
    ):
        raise NormalizedV2ContractError("synchronized normalized V2 source forward/backward parity failed")
    if base_backward["residual_gradient_sha256"] is not None or joint_backward["residual_gradient_sha256"] is None:
        raise NormalizedV2ContractError("V2 base/joint residual trainability contract failed")

    # Numerical-life contract uses a disposable clone; no checkpoint is saved.
    raw_batch_carrier = torch.from_numpy(
        np.stack(
            [
                base_dm.train_dataset.raw_carrier((int(index), int(start)))
                for index, start in zip(
                    base_dm.train_batch_sampler.batches[0],
                    base_dm.train_batch_sampler.schedule[0, : len(base_dm.train_batch_sampler.batches[0])],
                )
            ],
            axis=0,
        )
    )
    normalized_life = _clone_step(joint_model, joint_batch, joint_batch[4], source_forward_rng)
    raw_life = _raw_coordinate_diagnostic(joint_model, joint_batch, raw_batch_carrier, source_forward_rng)
    if joint_backward["residual_grad_rms"] is None or joint_backward["residual_grad_rms"] <= 0:
        raise NormalizedV2ContractError("joint normalized V2 residual gradient is zero")
    if normalized_life["deltaW_rms"] <= 0 or normalized_life["grad_rms_over_adam_eps"] <= 1:
        raise NormalizedV2ContractError("normalized V2 numerical-life thresholds failed")

    source_files = [
        "src/h1_m4_eb_normalized_v2_contract.py",
        "src/data/h1_m4_eb_normalized_v2.py",
        "src/models/components/h1_m4_eb_normalized_v2_residual_spint.py",
        "src/models/h1_m4_eb_normalized_v2_module.py",
        "scripts/h1_m4_eb_normalized_v2_preflight.py",
        "scripts/h1_m4_eb_normalized_v2_evaluate.py",
        "scripts/h1_m4_eb_normalized_v2_paired_launcher.py",
        "scripts/h1_m4_eb_normalized_v2_launcher.py",
        "configs/data/falcon_h1_m4_eb_normalized_v2.yaml",
        "configs/model/falcon_h1_m4_eb_normalized_v2.yaml",
        "configs/experiment/h1_m4_eb_normalized_v2.yaml",
        "configs/callbacks/h1_m4_eb_normalized_v2_terminal.yaml",
        "configs/experiment/h1_m4_eb_normalized_v2_base.yaml",
        "configs/experiment/h1_m4_eb_normalized_v2_joint.yaml",
        "tests/test_h1_m4_eb_normalized_v2_contract.py",
    ]
    source_hashes = {relative: sha256_file(root / relative) for relative in source_files if (root / relative).is_file()}
    future_receipt_name = Path(output_path).name
    future_command = (
        "python scripts/h1_m4_eb_normalized_v2_paired_launcher.py "
        "--output-root pilot_artifacts/h1_m4_eb_normalized_v2/gpu_runs "
        f"--preflight-receipt pilot_artifacts/h1_m4_eb_normalized_v2/{future_receipt_name} "
        "--root-review-marker pilot_artifacts/h1_m4_eb_normalized_v2/ROOT_REVIEW_MARKER.json --execute"
    )
    receipt = {
        "schema": V2_RECEIPT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "claim_status": "engineering numerical-contract evidence only; V1 M4 CPU STOP remains unchanged",
        "scope": {
            "opened": "exactly 11 public held-in-calib source NWBs",
            "source_recordings": 11,
            "target_recordings_opened": 0,
            "target_recordings_enumerated": 0,
            "minival_opened_or_enumerated": False,
            "heldout_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "gpu_constructed_or_launched": False,
            "checkpoint_created": False,
            "root_review_marker_created": False,
        },
        "fixed_v1_contract": {
            "q": 16,
            "lambda": 100.0,
            "support_trials": 4,
            "window_size": 700,
            "batch_size": 32,
            "epochs": 50,
            "seed": 42,
            "optimizer": "Adam",
            "lr": 5.0e-5,
            "weight_decay": 0.0,
            "model_residual_shape": [4, 700],
        },
        "source_manifest": base_dm.pilot_manifest(),
        "source_manifest_sha256": base_dm.pilot_manifest_sha256,
        "source_manifest_base_joint_equal": True,
        "normalizer": {
            **normalizer.manifest,
            "raw_cache_array_sha256": array_sha256(np.stack([e.carrier for e in base_dm.carrier_cache.entries])),
            "normalized_cache_manifest": base_dm.normalized_cache_manifest,
            "source_cache_sha256": base_dm.carrier_cache.manifest["cache_sha256"],
            "raw_to_normalized_roundtrip": roundtrip,
            "intervention_algebra": algebra,
        },
        "first_real_source_batch": {
            "normalized_carrier_rms": float(torch.sqrt(torch.mean(torch.square(base_batch[4].double())))),
            "raw_carrier_rms": float(torch.sqrt(torch.mean(torch.square(raw_batch_carrier.double())))),
            "carrier_shape": list(base_batch[4].shape),
            "session_order": list(base_batch[3]),
            "raw_and_normalized_recorded_same_batch": True,
        },
        "paired_initial_state": {
            "shared_initial_state_sha256_after_lazy_materialization": shared_initial_state_sha,
            "same_ordered_batch": True,
            "same_schedule": True,
            "same_normalizer_sha256": base_dm.normalizer.normalizer_sha256 == joint_dm.normalizer.normalizer_sha256,
        },
        "parity": {"base": base_backward, "joint": joint_backward},
        "numerical_life": {
            "normalized_coordinate": normalized_life,
            "v1_raw_coordinate_diagnostic": raw_life,
            "finite": True,
            "gradient_nonzero": True,
            "deltaW_nonzero": True,
            "gradient_over_adam_eps_gt_one": True,
            "prediction_difference_non_bit_identical": True,
        },
        "source_sha256": source_hashes,
        "launch": {
            "launch_authorized": False,
            "reason": "CPU preflight is engineering evidence only; execute requires explicit immutable root review marker",
            "future_command": future_command,
            "requires_root_review_marker_for_execute": True,
        },
    }
    output, digest = write_immutable_json(output_path, receipt)
    return {**receipt, "receipt_path": str(output), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--eb-receipt", type=Path, default=DEFAULT_EB)
    args = parser.parse_args()
    result = run_preflight(
        project_root=args.project_root,
        cache_dir=args.cache_dir,
        output_path=args.output,
        raw_receipt_path=args.raw_receipt,
        eb_receipt_path=args.eb_receipt,
    )
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
