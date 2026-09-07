#!/usr/bin/env python3
"""CPU-only contract/preflight for H1 activity-only SPINT identity width scaling.

The script opens exactly the eleven H1 fold-0 source recordings through the
new no-carrier DataModule.  It never imports a target loader/evaluator and
never creates a CUDA context.  The output is a write-once receipt that freezes
the three-arm 1024/224/32 routing curve before GPU training.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
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

from src.data.h1_spint_width_fold0 import (
    EXPECTED_BATCH_ORDER_SHA256,
    EXPECTED_CALIBRATION_SCHEDULE_SHA256,
    EXPECTED_SOURCE_WINDOW_SHA256,
    H1SpintWidthFold0DataModule,
    width_accounting_manifest,
)
from src.h1_m4_eb_pilot_contract import state_hash, write_immutable_json
from src.models.components.spint import SpintModel
from src.models.components.spint_identity_width import SpintIdentityWidthModel
from src.models.h1_spint_width_module import WIDTH_ARMS


PREFLIGHT_SCHEMA = "h1_spint_identity_width_fold0_cpu_preflight_v2"
PREFLIGHT_STATUS = "PASS_H1_SPINT_IDENTITY_WIDTH_FOLD0_SOURCE_ONLY_CPU_PREFLIGHT"
ARMS = ("H-S-1024", "H-S-W224", "H-S-W32")
WIDTHS = (1024, 224, 32)
NONINFERIORITY_MARGIN_R2 = 0.03


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _base_kwargs() -> dict[str, Any]:
    return {
        "model_dim": 1024,
        "num_covariates": 7,
        "window_size": 700,
        "num_heads": 64,
        "num_layers": 1,
        "num_id_layers": 3,
        "use_learnable_id": True,
        "learnable_id_type": "mlp",
        "learnable_rep": True,
        "dropout_rate": 0.0,
        "dynamic_dropout": True,
        "dynamic_dropout_low": 0.0,
        "dynamic_dropout_high": 1.0,
        "tf_drop_rate": 0.1,
        "readin_layer_type": "mlp",
    }


def _materialize(model: torch.nn.Module, identity: torch.Tensor) -> None:
    lazy = model.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(identity.permute(0, 1, 3, 2))


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], *, label: str) -> dict[str, str]:
    _need(tuple(left) == tuple(right), f"{label}: state key set/order differs")
    hashes: dict[str, str] = {}
    for name in left:
        if not torch.equal(left[name], right[name]):
            maximum = float(torch.max(torch.abs(left[name].float() - right[name].float())))
            raise ValueError(f"{label}: tensor differs at {name}; max_abs={maximum}")
        hashes[name] = hashlib.sha256(left[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return hashes


def _shared_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    prefixes = ("fc_in.", "fc_out.", "rep", "transformer.")
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name.startswith(prefixes)}
    _need(bool(state), "shared backbone state selection was empty")
    return state


def _id_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in model.named_parameters() if name.startswith(("fc_id_in.", "fc_id_out.")))


def _compose(name: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "configs"), job_name="h1_spint_width_preflight"):
        return compose(config_name="train", overrides=[f"experiment={name}"])


def _check_config(config: Any, *, arm: str, width: int) -> None:
    expected = {
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "width_sweep.arm": arm,
        "width_sweep.identity_width": width,
        "width_sweep.fold_date": "19250101",
        "width_sweep.noninferiority_margin_r2": NONINFERIORITY_MARGIN_R2,
        "data.calibration_n_trials": 4,
        "data.window_size": 700,
        "data.batch_size": 32,
        "data.fixed_epochs": 50,
        "trainer.max_epochs": 50,
        "trainer.min_epochs": 50,
        "trainer.limit_val_batches": 0,
        "trainer.num_sanity_val_steps": 0,
        "model.net.identity_width": width,
        "model.net.model_dim": 1024,
        "model.net.num_id_layers": 3,
        "model.net.learnable_id_type": "mlp",
    }
    for dotted, expected_value in expected.items():
        value = OmegaConf.select(config, dotted)
        _need(value == expected_value, f"{arm}: config drift at {dotted}: {value!r}")
    _need(tuple(config.width_sweep.predeclared_arms) == ARMS, f"{arm}: predeclared arm set/order drift")
    _need(tuple(config.width_sweep.predeclared_widths) == WIDTHS, f"{arm}: predeclared width set/order drift")
    _need(config.model.net._target_ == "src.models.components.spint_identity_width.SpintIdentityWidthModel", f"{arm}: wrong model class")
    _need(config.data._target_ == "src.data.h1_spint_width_fold0.H1SpintWidthFold0DataModule", f"{arm}: wrong activity-only DataModule")
    _need("carrier" not in OmegaConf.to_yaml(config.model, resolve=False).lower(), f"{arm}: model config mentions carrier")


def run_preflight(*, data_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Run source-only real-data preflight and freeze the routing curve receipt."""

    _need(not torch.cuda.is_initialized(), "CPU preflight must begin before any CUDA context exists")
    configs = {}
    for arm, width, name in zip(ARMS, WIDTHS, ("h1_spint_width_fold0", "h1_spint_width_fold0_w224", "h1_spint_width_fold0_w32")):
        config = _compose(name)
        _check_config(config, arm=arm, width=width)
        configs[arm] = OmegaConf.to_container(config, resolve=False)

    # One real source-only setup establishes the exact M=4 activity inputs.
    module = H1SpintWidthFold0DataModule(task="h1", data_dir=str(Path(data_dir).resolve()))
    module.setup("fit")
    module.train_batch_sampler.reset_epoch()
    batch = next(iter(module.train_dataloader()))
    _need(len(batch) == 4, "real source batch contains an unexpected carrier field")
    neural, target, identity, sessions = batch
    _need(tuple(neural.shape) == (32, 700, 176), f"source neural shape drift: {tuple(neural.shape)}")
    _need(tuple(target.shape) == (32, 700, 7), f"source target shape drift: {tuple(target.shape)}")
    _need(tuple(identity.shape) == (32, 4, 1024, 176), f"source M=4 identity shape drift: {tuple(identity.shape)}")
    source = module.source_manifest()
    _need(source["source_window_indices_sha256"] == EXPECTED_SOURCE_WINDOW_SHA256, "source window hash drift")
    _need(source["batch_order_sha256"] == EXPECTED_BATCH_ORDER_SHA256, "source order hash drift")
    _need(source["calibration_schedule_sha256"] == EXPECTED_CALIBRATION_SCHEDULE_SHA256, "source calibration schedule hash drift")
    _need(source["target_nwb_opened_during_training_setup"] is False, "source setup opened target data")
    _need(source["carrier_path"] == "absent_from_dataset_and_model_inputs", "carrier absence contract drift")

    # h=1024 must be a literal Standard-SPINT reference, including its state
    # and its activity-only forward result after lazy materialization.
    _seed()
    standard = SpintModel(**_base_kwargs())
    _materialize(standard, identity)
    _seed()
    reference = SpintIdentityWidthModel(identity_width=1024, **_base_kwargs())
    _materialize(reference, identity)
    standard_state = {name: tensor.detach().cpu().clone() for name, tensor in standard.state_dict().items()}
    reference_state = {name: tensor.detach().cpu().clone() for name, tensor in reference.state_dict().items()}
    h1024_hashes = _state_equal(standard_state, reference_state, label="h=1024 Standard-SPINT equivalence")
    standard.eval()
    reference.eval()
    with torch.no_grad():
        original_output = standard(neural, calib_trialized_neural_features=identity)
        reference_output = reference(neural, calib_trialized_neural_features=identity)
    _need(torch.equal(original_output, reference_output), "h=1024 activity-only forward differs from Standard SPINT")

    arms: dict[str, Any] = {}
    shared_by_arm: dict[str, dict[str, torch.Tensor]] = {}
    for arm, width in zip(ARMS, WIDTHS):
        _seed()
        net = SpintIdentityWidthModel(identity_width=width, **_base_kwargs())
        _materialize(net, identity)
        _need(net.identity_parameter_count() == width_accounting_manifest(WIDTHS)[str(width)]["identity_parameters"], f"{arm}: parameter count drift")
        _need(net.identity_dense_macs_m4_n176() == width_accounting_manifest(WIDTHS)[str(width)]["identity_dense_macs_m4_n176"], f"{arm}: MAC count drift")
        signature = inspect.signature(net.forward)
        _need("carrier" not in signature.parameters, f"{arm}: model forward accepts carrier")
        # Exercise the exact activity-only source batch through a loss and
        # backward pass, before any optimizer step and without opening target
        # recordings.  A no-step hash must remain stable.
        net.train()
        net.zero_grad(set_to_none=True)
        before_backward = state_hash(net.state_dict())
        source_prediction = net(neural, calib_trialized_neural_features=identity)[:, -1:, :] / 20.0
        source_loss = torch.mean(torch.square(source_prediction - target[:, -1:, :]))
        source_loss.backward()
        after_backward = state_hash(net.state_dict())
        _need(before_backward == after_backward, f"{arm}: source forward/backward mutated parameters before optimizer")
        gradients = [parameter.grad for name, parameter in net.named_parameters() if name.startswith(("fc_id_in.", "fc_id_out."))]
        _need(bool(gradients) and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients), f"{arm}: identity gradient is absent/nonfinite")
        _need(any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients), f"{arm}: identity gradient is identically zero")
        shared_by_arm[arm] = _shared_state(net)
        identity_names = _id_parameter_names(net)
        _need(bool(identity_names) and all("carrier" not in name.lower() for name in identity_names), f"{arm}: ID parameters violate activity-only contract")
        arms[arm] = {
            "identity_width": width,
            "identity_parameters": net.identity_parameter_count(),
            "identity_dense_macs_m4_n176": net.identity_dense_macs_m4_n176(),
            "whole_model_parameters": int(sum(parameter.numel() for parameter in net.parameters())),
            "identity_parameter_names": list(identity_names),
            "state_sha256_after_lazy_materialization": state_hash(net.state_dict()),
            "forward_signature": str(signature),
            "real_source_forward_backward": {
                "loss": float(source_loss.detach()),
                "state_immutable_before_optimizer": True,
                "identity_gradient_nonzero": True,
            },
        }
    # Width is the sole intended state/topology intervention: shared backbone
    # initialization must be byte-identical across all predeclared arms.
    shared_hashes = _state_equal(shared_by_arm[ARMS[0]], shared_by_arm[ARMS[1]], label="H-S-1024/H-S-W224 shared backbone")
    _state_equal(shared_by_arm[ARMS[0]], shared_by_arm[ARMS[2]], label="H-S-1024/H-S-W32 shared backbone")

    accounting = width_accounting_manifest(WIDTHS)
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "source_recordings_opened": list(source["source_sessions"]),
            "target_recordings_opened": 0,
            "minival_opened": False,
            "formal_or_organizer_data_opened": False,
            "cuda_initialized": False,
        },
        "scientific_question": "Does a much smaller architecture-preserving activity-derived SPINT NeuronID match H-S on H1?",
        "interpretation_boundary": "This is a capacity/headroom test only. It cannot measure the value of activity-derived identity versus no identity and cannot establish that NeuronID is useless.",
        "predeclared": {
            "arms": list(ARMS),
            "widths": list(WIDTHS),
            "parameter_compression_vs_hs1024": {
                arm: accounting["1024"]["identity_parameters"] / accounting[str(width)]["identity_parameters"]
                for arm, width in zip(ARMS, WIDTHS)
            },
            "noninferiority_margin_r2": NONINFERIORITY_MARGIN_R2,
            "expansion_rule": "Run all three fold0 arms. Only if a compact arm has H-S-width delta >= -0.03 may that predeclared arm advance to the remaining four development dates; otherwise stop expansion.",
            "checkpoint_rule": "fresh seed42, fixed terminal zero-based epoch49 after 50 source-only epochs; no validation/best-checkpoint selection",
        },
        "architecture": {
            "reference": "original H-S identity topology: 1024->h->h->h, mean over M=4 trials, h->h->h->700",
            "only_changed_quantity": "identity MLP internal width h",
            "carrier": "absent from dataset batches, model signature, parameter names, and model call boundary",
            "shared_backbone_initialization_sha256": shared_hashes,
            "h1024_standard_spint_state_sha256": state_hash(standard_state),
            "h1024_width_model_state_sha256": state_hash(reference_state),
            "h1024_state_and_forward_bit_identical_to_standard_spint": True,
            "parameter_and_mac_convention": "identity path only; parameters include biases; MACs count dense Linear weight multiply-adds for M=4,N=176 and exclude bias/nonlinearity/pooling/downstream SPINT decoder",
            "accounting": accounting,
            "arms": arms,
        },
        "fixed_protocol": {
            "fold_date": "19250101",
            "source_sessions": list(source["source_sessions"]),
            "target_sessions_not_opened": list(source["target_sessions_not_opened"]),
            "support_trials": 4,
            "identity_trial_length": 1024,
            "window_size": 700,
            "batch_size": 32,
            "seed": 42,
            "epochs": 50,
            "optimizer": "Adam",
            "learning_rate": 5.0e-5,
            "weight_decay": 0.0,
            "precision": "32-true",
            "source_window_indices_sha256": source["source_window_indices_sha256"],
            "batch_order_sha256": source["batch_order_sha256"],
            "calibration_schedule_sha256": source["calibration_schedule_sha256"],
            "source_manifest_sha256": module.source_manifest_sha256,
            "query_contract": "strict M=4 post-support query remains unopened during this CPU source-only preflight",
        },
        "first_real_source_batch": {
            "neural_shape": list(neural.shape),
            "target_shape": list(target.shape),
            "identity_shape": list(identity.shape),
            "batch_field_count": len(batch),
            "sessions": list(sessions),
        },
        "candidate_audit": {
            "considered": [
                "architecture-preserving width scaling",
                "low-rank factorization",
                "unstructured pruning",
                "teacher distillation",
            ],
            "selected": "architecture-preserving width scaling",
            "reason": "It preserves the original SPINT activity-only MLP/pooling topology and isolates one prespecified capacity variable; the alternatives change parameterization, sparsity/runtime behavior, or training objective.",
        },
    }
    output, digest = write_immutable_json(output_path, receipt)
    receipt["receipt_path"] = str(output)
    receipt["receipt_sha256"] = digest
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "pilot_artifacts/h1_spint_width_fold0/H1_SPINT_IDENTITY_WIDTH_FOLD0_CPU_PREFLIGHT_v2.json",
    )
    args = parser.parse_args()
    result = run_preflight(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
