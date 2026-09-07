#!/usr/bin/env python3
"""Source-only CPU preflight for the fixed-h32 H1 CarrierID pilot.

This script deliberately does not import the target loader.  It binds the
existing normalized-V2 source carrier cache/schedule to two independently
named CarrierID arms and verifies the two important initialisation facts:

* H-C and H-C0 are the identical function on the first real source batch;
* every shared downstream SPINT tensor equals a same-seed matched V2/base
  construction, tensor-by-tensor.

No checkpoint, CUDA context, minival, held-out, formal, or EvalAI path is
created/opened by this preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2DataModule
from src.h1_m4_eb_normalized_v2_contract import (
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    NormalizedV2ContractError,
    array_sha256,
    canonical_sha256,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_POST_PARAMETERS,
    H1_CARRIERID_PRE_PARAMETERS,
    H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
    H1_SPINT_ID_PARAMETERS,
    H1_SPINT_WHOLE_MODEL_PARAMETERS,
    H1CarrierIdSpint,
)
from src.models.components.h1_m4_eb_normalized_v2_residual_spint import H1M4EBNormalizedV2ResidualSpint
from src.models.h1_carrierid_module import H1CarrierIdLitModule
from src.models.h1_m4_eb_normalized_v2_module import H1M4EBNormalizedV2PilotLitModule


PREFLIGHT_STATUS = "PASS_H1_CARRIERID_H32_REAL_SOURCE_CPU_PREFLIGHT_NONLAUNCH"
PREFLIGHT_SCHEMA = "h1_carrierid_h32_fold0_cpu_preflight_v1"
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


def _new_carrierid(arm: str) -> H1CarrierIdLitModule:
    net = H1CarrierIdSpint(
        carrier_hidden_dim=32,
        carrier_dim=4,
        carrier_trial_length=1024,
        zero_carrier=arm == "zero",
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
    return H1CarrierIdLitModule(
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


def _new_v2_base() -> H1M4EBNormalizedV2PilotLitModule:
    net = H1M4EBNormalizedV2ResidualSpint(
        train_residual=False,
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
        pilot_arm="base",
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


def _materialize_v2_identity(model: H1M4EBNormalizedV2PilotLitModule, identity: torch.Tensor) -> None:
    lazy = model.net.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(identity.permute(0, 1, 3, 2))


def _first_batch(dm: H1M4EBNormalizedV2DataModule):
    dm.train_batch_sampler.reset_epoch()
    return next(iter(dm.train_dataloader()))


def _downstream_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    selected = ("net.fc_in.", "net.fc_out.", "net.rep", "net.transformer.")
    values = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.startswith(selected)}
    if not values:
        raise NormalizedV2ContractError("CarrierID downstream state selection is empty")
    return values


def _bitwise_downstream_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, str]:
    if tuple(left) != tuple(right):
        raise NormalizedV2ContractError("CarrierID/V2 downstream tensor-name sets differ")
    hashes: dict[str, str] = {}
    for name in left:
        if not torch.equal(left[name], right[name]):
            delta = float(torch.max(torch.abs(left[name].float() - right[name].float())))
            raise NormalizedV2ContractError(f"CarrierID/V2 shared downstream initialization differs at {name}; max_abs={delta}")
        hashes[name] = array_sha256(left[name].numpy())
    return hashes


def _forward_and_backward(model: H1CarrierIdLitModule, batch, rng_state) -> dict[str, Any]:
    _set_rng_state(rng_state)
    model.train()
    model.zero_grad(set_to_none=True)
    state_before = state_hash(model.state_dict())
    loss, prediction, target, sessions = model.model_step(batch)
    loss.backward()
    state_after = state_hash(model.state_dict())
    if state_before != state_after:
        raise NormalizedV2ContractError("source forward/backward mutated CarrierID parameters before optimizer step")
    shared_grad = {
        name: array_sha256(parameter.grad.detach().cpu().numpy())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and name.startswith(("net.fc_in.", "net.fc_out.", "net.rep", "net.transformer."))
    }
    carrier_grad = {
        name: array_sha256(parameter.grad.detach().cpu().numpy())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and name.startswith(("net.carrier_pre_pool.", "net.carrier_post_pool."))
    }
    carrier_column_grad = model.net.carrier_post_pool[0].weight.grad[:, model.net.carrier_hidden_dim :]
    return {
        "loss": float(loss.detach()),
        "prediction_sha256": array_sha256(prediction.detach().cpu().numpy()),
        "target_sha256": array_sha256(target.detach().cpu().numpy()),
        "sessions": list(sessions),
        "state_sha256_before": state_before,
        "state_sha256_after": state_after,
        "state_immutable_before_optimizer": True,
        "shared_downstream_gradient_sha256": shared_grad,
        "carrier_path_gradient_sha256": carrier_grad,
        "first_post_carrier_column_gradient_nonzero": int(torch.count_nonzero(carrier_column_grad).item()),
    }


def _dense_mac_accounting() -> dict[str, Any]:
    """Count weight multiplies/adds for identity encoders only; bias omitted."""

    trials, neurons, trial_length, hidden, window = 4, 176, 1024, 32, 700
    carrier_pre = trials * neurons * trial_length * hidden
    carrier_post = neurons * ((hidden + 4) * hidden + hidden * hidden + hidden * window)
    carrier_total = carrier_pre + carrier_post
    spint_in = trials * neurons * (trial_length * 1024 + 1024 * 1024 + 1024 * 1024)
    spint_out = neurons * (1024 * 1024 + 1024 * 1024 + 1024 * window)
    spint_total = spint_in + spint_out
    if carrier_total != 27_394_048:
        raise NormalizedV2ContractError("CarrierID dense-MAC arithmetic drift")
    return {
        "convention": "dense Linear weight multiply-adds; excludes bias, nonlinearity, pooling, downstream SPINT decoder",
        "carrierid": {"pre_pool": carrier_pre, "post_pool": carrier_post, "total": carrier_total},
        "spint_identity_mlp": {"fc_id_in": spint_in, "fc_id_out": spint_out, "total": spint_total},
        "spint_to_carrierid_ratio": spint_total / carrier_total,
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
        raise NormalizedV2ContractError("CarrierID preflight requires CUDA_VISIBLE_DEVICES unset or empty")
    root = Path(project_root).resolve()
    cache = Path(cache_dir).resolve()
    if not (root / "data/000954/sub-HumanPitt-held-in-calib").is_dir():
        raise FileNotFoundError("CarrierID preflight requires public H1 held-in-calib source directory")

    data_modules: dict[str, H1M4EBNormalizedV2DataModule] = {}
    for arm in ("full", "zero"):
        dm = H1M4EBNormalizedV2DataModule(
            task="h1",
            data_dir=str(root / "data/000954"),
            raw_receipt_path=str(Path(raw_receipt_path).resolve()),
            eb_receipt_path=str(Path(eb_receipt_path).resolve()),
            cache_dir=str(cache / arm),
        )
        dm.setup("fit")
        data_modules[arm] = dm
    full_dm, zero_dm = data_modules["full"], data_modules["zero"]
    manifest_fields = (
        "source_window_indices_sha256", "batch_order_sha256", "calibration_schedule_sha256",
        "carrier_cache_sha256", "normalized_cache_sha256", "normalizer_sha256", "transform_sha256",
    )
    for field in manifest_fields:
        if full_dm.pilot_manifest()[field] != zero_dm.pilot_manifest()[field]:
            raise NormalizedV2ContractError(f"CarrierID full/zero source binding mismatch at {field}")
    if full_dm.pilot_manifest_sha256 != zero_dm.pilot_manifest_sha256:
        raise NormalizedV2ContractError("CarrierID full/zero complete source manifest mismatch")
    full_batch, zero_batch = _first_batch(full_dm), _first_batch(zero_dm)
    for index in (0, 1, 2, 4):
        if not torch.equal(full_batch[index], zero_batch[index]):
            raise NormalizedV2ContractError(f"CarrierID full/zero first source batch differs at field {index}")
    if list(full_batch[3]) != list(zero_batch[3]):
        raise NormalizedV2ContractError("CarrierID full/zero first source session order differs")
    if torch.count_nonzero(full_batch[4]).item() == 0:
        raise NormalizedV2ContractError("CarrierID Full source carrier unexpectedly contains only zero")

    _seed(42)
    carrier_full = _new_carrierid("full")
    _seed(42)
    carrier_zero = _new_carrierid("zero")
    if state_hash(carrier_full.state_dict()) != state_hash(carrier_zero.state_dict()):
        raise NormalizedV2ContractError("CarrierID H-C/H-C0 initial state differs")
    carrier_state_sha = state_hash(carrier_full.state_dict())
    if sum(parameter.numel() for parameter in carrier_full.net.parameters()) != H1_CARRIERID_WHOLE_MODEL_PARAMETERS:
        raise NormalizedV2ContractError("CarrierID full whole-model parameter count is not registered 10,947,836")

    # The parent SPINT constructor has already created all downstream tensors
    # before CarrierID removes its ID MLP and creates the new small path.
    _seed(42)
    v2_base = _new_v2_base()
    _materialize_v2_identity(v2_base, full_batch[2])
    downstream_hashes = _bitwise_downstream_equal(_downstream_state(v2_base), _downstream_state(carrier_full))

    carrier_full.eval()
    carrier_zero.eval()
    with torch.no_grad():
        full_prediction = carrier_full(
            full_batch[0], calib_trialized_neural_features=full_batch[2], carrier=full_batch[4]
        )
        zero_prediction = carrier_zero(
            zero_batch[0], calib_trialized_neural_features=zero_batch[2], carrier=zero_batch[4]
        )
    if not torch.equal(full_prediction, zero_prediction):
        difference = float(torch.max(torch.abs(full_prediction - zero_prediction)))
        raise NormalizedV2ContractError(f"CarrierID initial Full/H-C0 prediction is not bit-identical (max_abs={difference})")

    _seed(20260807)
    source_rng = _rng_state()
    full_backward = _forward_and_backward(carrier_full, full_batch, source_rng)
    zero_backward = _forward_and_backward(carrier_zero, zero_batch, source_rng)
    for key in ("loss", "prediction_sha256", "target_sha256", "sessions", "shared_downstream_gradient_sha256"):
        if full_backward[key] != zero_backward[key]:
            raise NormalizedV2ContractError(f"CarrierID initial H-C/H-C0 parity differs at {key}")
    if full_backward["first_post_carrier_column_gradient_nonzero"] <= 0:
        raise NormalizedV2ContractError("CarrierID Full has no source-side gradient into initially zero carrier columns")
    if zero_backward["first_post_carrier_column_gradient_nonzero"] != 0:
        raise NormalizedV2ContractError("CarrierID H-C0 carrier columns received a nonzero gradient despite model-bound zero")
    # This is a disposable source-only Adam step, not a target calibration
    # update.  It proves the literal-zero H-C0 boundary is operational rather
    # than merely a metadata label.
    torch.optim.Adam(carrier_full.parameters(), lr=5.0e-5, weight_decay=0.0).step()
    torch.optim.Adam(carrier_zero.parameters(), lr=5.0e-5, weight_decay=0.0).step()
    full_after_step = int(torch.count_nonzero(
        carrier_full.net.carrier_post_pool[0].weight[:, carrier_full.net.carrier_hidden_dim :].detach()
    ).item())
    zero_after_step = int(torch.count_nonzero(
        carrier_zero.net.carrier_post_pool[0].weight[:, carrier_zero.net.carrier_hidden_dim :].detach()
    ).item())
    if full_after_step <= 0 or zero_after_step != 0:
        raise NormalizedV2ContractError("CarrierID one-step Full/H-C0 carrier-boundary proof failed")

    source_files = [
        "src/models/components/h1_carrierid_spint.py",
        "src/models/h1_carrierid_module.py",
        "scripts/h1_carrierid_preflight.py",
        "scripts/h1_carrierid_paired_launcher.py",
        "scripts/h1_carrierid_evaluate.py",
        "configs/model/falcon_h1_carrierid.yaml",
        "configs/experiment/h1_carrierid.yaml",
        "configs/experiment/h1_carrierid_full.yaml",
        "configs/experiment/h1_carrierid_zero.yaml",
        "configs/callbacks/h1_carrierid_terminal.yaml",
        "tests/test_h1_carrierid_contract.py",
        # Read-only dependencies explicitly bind the V2 source/cache topology.
        "src/data/h1_m4_eb_normalized_v2.py",
        "src/data/h1_m4_eb_fold0_datamodule.py",
        "src/data/h1_m4_eb_pilot.py",
        "src/h1_m4_eb_normalized_v2_contract.py",
        "src/models/components/h1_m4_eb_normalized_v2_residual_spint.py",
        "src/models/h1_m4_eb_normalized_v2_module.py",
    ]
    source_hashes = {relative: sha256_file(root / relative) for relative in source_files}
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "opened": "exactly 11 public held-in-calib source NWBs",
            "target_recordings_opened": 0,
            "target_recordings_enumerated": 0,
            "minival_opened_or_enumerated": False,
            "heldout_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "checkpoint_created": False,
        },
        "fixed_protocol": {
            "fold_date": "19250101", "seed": 42, "source_support_trials": 4,
            "query_contract": "not opened in source-only preflight", "window_size": 700,
            "trial_length": 1024, "batch_size": 32, "epochs": 50, "precision": "32-true",
            "optimizer": "Adam", "learning_rate": 5.0e-5, "weight_decay": 0.0,
            "normalizer_formula": NORMALIZER_FORMULA, "normalizer_floor": NORMALIZER_FLOOR,
        },
        "source_manifest": full_dm.pilot_manifest(),
        "source_manifest_sha256": full_dm.pilot_manifest_sha256,
        "full_zero_source_manifest_equal": True,
        "first_real_source_batch": {
            "neural_shape": list(full_batch[0].shape), "identity_shape": list(full_batch[2].shape),
            "carrier_shape": list(full_batch[4].shape), "carrier_rms": float(torch.sqrt(torch.mean(torch.square(full_batch[4].double())))),
            "carrier_nonzero": True, "session_order": list(full_batch[3]),
        },
        "initialization": {
            "h_c_h_c0_complete_state_sha256": carrier_state_sha,
            "h_c_h_c0_state_equal": True,
            "initial_prediction_bit_identical": True,
            "initial_prediction_sha256": array_sha256(full_prediction.detach().cpu().numpy()),
            "shared_downstream_v2_base_bit_identical": True,
            "shared_downstream_tensor_sha256": downstream_hashes,
            "source_forward_backward_parity": {"full": full_backward, "zero": zero_backward},
            "disposable_source_adam_step": {
                "full_first_post_carrier_columns_nonzero_after_step": full_after_step,
                "zero_first_post_carrier_columns_nonzero_after_step": zero_after_step,
                "target_data_or_deployment_update_used": False,
            },
        },
        "parameter_accounting": {
            "static_session_identity_encoder_parameters": {
                "carrierid_pre_pool": H1_CARRIERID_PRE_PARAMETERS,
                "carrierid_post_pool": H1_CARRIERID_POST_PARAMETERS,
                "carrierid_h32_total": H1_CARRIERID_PARAMETERS,
                "original_spint_identity_mlp": H1_SPINT_ID_PARAMETERS,
                "spint_identity_to_carrierid_ratio": H1_SPINT_ID_PARAMETERS / H1_CARRIERID_PARAMETERS,
            },
            "whole_model_parameters": {
                "original_spint": H1_SPINT_WHOLE_MODEL_PARAMETERS,
                "carrierid": H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
                "spint_to_carrierid_ratio": H1_SPINT_WHOLE_MODEL_PARAMETERS / H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
            },
            "target_session_updated_parameters": {"original_spint": 0, "carrierid": 0},
            "deployment_target_optimizer_steps": {"original_spint": 0, "carrierid": 0},
            "deployment_target_backward_steps": {"original_spint": 0, "carrierid": 0},
            "calibration_input_state": {
                "normalized_carrier_elements": 176 * 4,
                "normalized_carrier_bytes_fp32": 176 * 4 * 4,
                "carrier_hidden_dim": 32,
                "persistent_trainable_session_state": 0,
            },
        },
        "dense_mac_accounting": _dense_mac_accounting(),
        "source_sha256": source_hashes,
        "launch": {
            "launch_authorized": False,
            "required_before_execute": "this immutable exact receipt and source-closure recheck",
            "future_command": (
                "python scripts/h1_carrierid_paired_launcher.py --output-root "
                "pilot_artifacts/h1_carrierid/gpu_runs --preflight-receipt "
                f"{Path(output_path).resolve()} --execute"
            ),
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
        project_root=args.project_root, cache_dir=args.cache_dir, output_path=args.output,
        raw_receipt_path=args.raw_receipt, eb_receipt_path=args.eb_receipt,
    )
    print(json.dumps({"status": result["status"], "receipt": result["receipt_path"], "sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
