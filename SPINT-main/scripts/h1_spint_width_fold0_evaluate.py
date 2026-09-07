#!/usr/bin/env python3
"""One-shot strict fold-0 evaluator for the predeclared H-S width curve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET
from src.data.h1_spint_width_fold0 import H1SpintWidthFold0DataModule, load_fold0_strict_target
from src.h1_m4_eb_pilot_contract import assert_state_immutable, sha256_file, state_hash, variance_weighted_r2, write_immutable_json
from src.models.h1_spint_width_module import CHECKPOINT_SCHEMA, FIXED_EPOCHS, FOLD0_DATE, WIDTH_ARMS


ARMS = ("H-S-1024", "H-S-W224", "H-S-W32")
NONINFERIORITY_MARGIN_R2 = 0.03
RESULT_SCHEMA = "h1_spint_identity_width_fold0_terminal_evaluation_v1"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolved_config(path: Path, *, arm: str) -> Any:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"{arm}: resolved Hydra config missing/symlinked")
    config = OmegaConf.load(candidate)
    width = WIDTH_ARMS[arm]
    expected = {
        "seed": 42,
        "train": True,
        "test": False,
        "ckpt_path": None,
        "width_sweep.arm": arm,
        "width_sweep.identity_width": width,
        "width_sweep.fold_date": FOLD0_DATE,
        "width_sweep.noninferiority_margin_r2": NONINFERIORITY_MARGIN_R2,
        "data.calibration_n_trials": 4,
        "data.window_size": 700,
        "data.batch_size": 32,
        "data.fixed_epochs": FIXED_EPOCHS,
        "trainer.max_epochs": FIXED_EPOCHS,
        "trainer.min_epochs": FIXED_EPOCHS,
        "trainer.limit_val_batches": 0,
        "trainer.num_sanity_val_steps": 0,
        "model.net.identity_width": width,
        "model.net.model_dim": 1024,
        "model.net.num_id_layers": 3,
    }
    for dotted, expected_value in expected.items():
        value = OmegaConf.select(config, dotted)
        _need(value == expected_value, f"{arm}: resolved config drift at {dotted}: {value!r}")
    _need(tuple(config.width_sweep.predeclared_arms) == ARMS, f"{arm}: resolved predeclared arm set/order drift")
    _need("carrier" not in OmegaConf.to_yaml(config.model, resolve=False).lower(), f"{arm}: resolved model mentions carrier")
    return config


def _load_checkpoint(path: Path, config_path: Path, *, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = path.resolve()
    _need(candidate.is_file() and not candidate.is_symlink(), f"{arm}: checkpoint missing/symlinked")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    _need(isinstance(payload, dict) and isinstance(payload.get("state_dict"), Mapping), f"{arm}: invalid Lightning checkpoint")
    _need(payload.get("epoch") == 49, f"{arm}: checkpoint is not fixed terminal epoch 49")
    metadata = payload.get("h1_spint_identity_width")
    _need(isinstance(metadata, Mapping), f"{arm}: checkpoint lacks width-sweep metadata")
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "width_arm": arm,
        "identity_width": WIDTH_ARMS[arm],
        "fold_date": FOLD0_DATE,
        "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49,
        "epochs_completed": FIXED_EPOCHS,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "carrier_input": "absent",
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "checkpoint_warm_start": False,
    }
    for key, value in expected.items():
        _need(metadata.get(key) == value, f"{arm}: checkpoint metadata drift at {key}")
    for key in (
        "initial_state_sha256",
        "source_manifest_sha256",
        "source_binding_sha256",
        "config_sha256",
    ):
        value = metadata.get(key)
        _need(isinstance(value, str) and len(value) == 64, f"{arm}: invalid checkpoint metadata {key}")
    _need(metadata["config_sha256"] == sha256_file(config_path), f"{arm}: checkpoint/config SHA mismatch")
    expected_params = 4 * WIDTH_ARMS[arm] ** 2 + 1729 * WIDTH_ARMS[arm] + 700
    _need(metadata.get("identity_parameters") == expected_params, f"{arm}: identity parameter arithmetic drift")
    _need(payload.get("global_step", 0) > 0, f"{arm}: checkpoint has no real source optimizer steps")
    return payload, dict(metadata)


def _instantiate(config: Any, payload: Mapping[str, Any], device: torch.device):
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _evaluate(model: Any, dataset: Any, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    by_session: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {name: ([], []) for name in H1_M4_FOLD0_TARGET}
    batch_sizes: list[int] = []
    before = state_hash(model.state_dict())
    with torch.no_grad():
        for batch in loader:
            _need(len(batch) == 4, "target evaluation batch contains a carrier field")
            neural, target, identity, sessions = batch
            output = model(
                neural.to(device=device, dtype=torch.float32),
                calib_trialized_neural_features=identity.to(device=device, dtype=torch.float32),
            )
            if model.hparams.decode_last_timestep_only:
                output = output[:, -1:, :]
                target = target[:, -1:, :]
            if model.hparams.predict_scaled_behavior:
                output = output / model.hparams.behavior_scaling_factor
            estimate = output[:, -1, :].detach().cpu().numpy()
            truth = target[:, -1, :].detach().cpu().numpy()
            prediction_parts.append(estimate)
            target_parts.append(truth)
            for index, session in enumerate(sessions):
                _need(session in by_session, f"unexpected target session {session}")
                by_session[session][0].append(estimate[index : index + 1])
                by_session[session][1].append(truth[index : index + 1])
            batch_sizes.append(int(estimate.shape[0]))
    after = state_hash(model.state_dict())
    assert_state_immutable(before, after, "strict width target evaluation")
    _need(bool(batch_sizes) and sum(batch_sizes) == len(dataset), "terminal target evaluation omitted samples")
    _need(batch_sizes[-1] == (len(dataset) % 32 or 32), "terminal target evaluation dropped the final remainder")
    prediction, target = np.concatenate(prediction_parts, axis=0), np.concatenate(target_parts, axis=0)
    per_session = {
        name: {**variance_weighted_r2(np.concatenate(truth, axis=0), np.concatenate(estimate, axis=0)), "samples": int(sum(item.shape[0] for item in estimate))}
        for name, (estimate, truth) in by_session.items()
    }
    return {
        **variance_weighted_r2(target, prediction),
        "samples": int(len(dataset)),
        "batches": len(batch_sizes),
        "last_batch_size": batch_sizes[-1],
        "per_session": per_session,
        "r2_accumulator_dtype": "float64",
        "state_immutable": before == after,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "query_window_indices_sha256": dataset.window_indices_sha256,
    }


def evaluate(
    *,
    data_dir: str | Path,
    checkpoint_paths: Mapping[str, str | Path],
    config_paths: Mapping[str, str | Path],
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    _need(device in {"cpu", "cuda"}, "device must be cpu/cuda")
    _need(device != "cuda" or torch.cuda.is_available(), "CUDA evaluation requested but unavailable")
    _need(tuple(checkpoint_paths) == ARMS and tuple(config_paths) == ARMS, "evaluator requires all predeclared arms in canonical order")

    configs: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    # Bind all configs/checkpoints before opening source or target recordings.
    for arm in ARMS:
        config_path = Path(config_paths[arm]).resolve()
        config = _resolved_config(config_path, arm=arm)
        payload, meta = _load_checkpoint(Path(checkpoint_paths[arm]), config_path, arm=arm)
        configs[arm], payloads[arm], metadata[arm] = config, payload, meta
    source_manifests = {row["source_manifest_sha256"] for row in metadata.values()}
    _need(len(source_manifests) == 1, "width arms do not share the exact source-only manifest")

    source_module = H1SpintWidthFold0DataModule(task="h1", data_dir=str(Path(data_dir).resolve()))
    source_module.setup("fit")
    _need(source_module.source_manifest_sha256 == next(iter(source_manifests)), "runtime source manifest differs from terminal checkpoints")
    _need(source_module.source_manifest()["target_nwb_opened_during_training_setup"] is False, "source validation opened target data")
    models = {arm: _instantiate(configs[arm], payloads[arm], torch.device(device)) for arm in ARMS}
    del payloads

    # The first target access occurs only after all three models and their
    # source/checkpoint bindings are complete.
    target = load_fold0_strict_target(str(Path(data_dir).resolve()))
    metrics = {arm: _evaluate(models[arm], target, torch.device(device)) for arm in ARMS}
    _need(len({metric["query_window_indices_sha256"] for metric in metrics.values()}) == 1, "arms use different strict queries")
    reference_r2 = float(metrics["H-S-1024"]["r2"])
    deltas = {arm: float(metrics[arm]["r2"]) - reference_r2 for arm in ARMS}
    compact = {arm: {"delta_r2_vs_hs1024": deltas[arm], "within_noninferiority_margin": deltas[arm] >= -NONINFERIORITY_MARGIN_R2} for arm in ARMS[1:]}
    status = (
        "PASS_H1_SPINT_IDENTITY_WIDTH_FOLD0_ALL_COMPACT_ARMS_WITHIN_0P03"
        if all(item["within_noninferiority_margin"] for item in compact.values())
        else "STOP_H1_SPINT_IDENTITY_WIDTH_FOLD0_COMPACT_ARM_EXCEEDS_0P03"
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "fold_date": FOLD0_DATE,
        "evaluation_device": str(torch.device(device)),
        "checkpoint_binding_completed_before_target_open": True,
        "predeclared_arms": list(ARMS),
        "noninferiority_margin_r2": NONINFERIORITY_MARGIN_R2,
        "checkpoints": {
            arm: {
                "path": str(Path(checkpoint_paths[arm]).resolve()),
                "sha256": sha256_file(checkpoint_paths[arm]),
                "config_path": str(Path(config_paths[arm]).resolve()),
                "config_sha256": sha256_file(config_paths[arm]),
                "metadata": metadata[arm],
            }
            for arm in ARMS
        },
        "source_manifest": source_module.source_manifest(),
        "source_manifest_sha256": source_module.source_manifest_sha256,
        "target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "files": {name: target.records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
            "support": target.support_hashes(),
            "strict_query_window_indices_sha256": target.window_indices_sha256,
            "all_query_histories_start_at_or_after_fifth_trial": True,
            "recordings_concatenated_before_variance_weighted_r2": True,
            "remainder_preserved": True,
            "carrier_input": "absent",
        },
        "metrics": metrics,
        "contrasts": {"reference_arm": "H-S-1024", "delta_r2_vs_hs1024": deltas, "compact_noninferiority": compact},
        "expansion_decision": {
            "rule": "A compact arm advances to remaining development dates only when its fold0 delta versus fresh H-S-1024 is at least -0.03. No result may select a new width.",
            "eligible_compact_arms": [arm for arm, result in compact.items() if result["within_noninferiority_margin"]],
            "stopped_compact_arms": [arm for arm, result in compact.items() if not result["within_noninferiority_margin"]],
        },
        "interpretation_boundary": "A matched compact-versus-large H-S result measures only redundant identity-encoder capacity or limited extra capacity headroom. It does not test activity-derived identity against no identity and cannot establish that NeuronID is useless.",
        "data_scope": {
            "opened": "11 public source held-in-calib recordings, then exactly 2 public fold0 target held-in-calib recordings",
            "minival_opened": False,
            "heldout_opened": False,
            "evalai_opened": False,
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
        },
    }
    output, digest = write_immutable_json(output_path, receipt)
    receipt["receipt_path"], receipt["receipt_sha256"] = str(output), digest
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/000954")
    parser.add_argument("--hs1024-checkpoint", type=Path, required=True)
    parser.add_argument("--w224-checkpoint", type=Path, required=True)
    parser.add_argument("--w32-checkpoint", type=Path, required=True)
    parser.add_argument("--hs1024-config", type=Path, required=True)
    parser.add_argument("--w224-config", type=Path, required=True)
    parser.add_argument("--w32-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = evaluate(
        data_dir=args.data_dir,
        checkpoint_paths={"H-S-1024": args.hs1024_checkpoint, "H-S-W224": args.w224_checkpoint, "H-S-W32": args.w32_checkpoint},
        config_paths={"H-S-1024": args.hs1024_config, "H-S-W224": args.w224_config, "H-S-W32": args.w32_config},
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps({"status": result["status"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()

