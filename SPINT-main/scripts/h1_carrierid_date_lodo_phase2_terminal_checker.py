#!/usr/bin/env python3
"""No-target terminal checker for the H1 date-LODO H-S/H-C source pair.

This module deliberately reads only immutable Phase-1/Phase-2 receipts,
resolved Hydra configs, and source-trained checkpoints.  Its raw-component
preflight hashes differ from the Lightning wrapper hashes captured in source
training: H-C can replay that wrapper domain on CPU, while H-S uses a strictly
pair-bound source-only GPU runtime-init receipt.  It never imports a
DataModule, constructs a Trainer, initialises CUDA, or imports any target
loader.  Consequently it is safe to use as the fail-closed gate between the
H-S and H-C source-training arms, and before a separately designed target
evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import hydra
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, state_hash, write_immutable_json


PAIR_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
PAIR_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED"
LAUNCH_SCHEMA = "h1_carrierid_date_lodo_phase2_paired_source_launch_receipt_v1"
LAUNCH_STATUS = "PASS_PAIRED_SOURCE_TRAINING_PREPARED_NOT_LAUNCHED"
CHECKPOINT_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_checkpoint_v1"
SINGLE_SCHEMA = "h1_carrierid_date_lodo_phase2_single_terminal_check_v1"
SINGLE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_SOURCE_E49_CHECKPOINT_NO_TARGET"
PAIR_SCHEMA = "h1_carrierid_date_lodo_phase2_paired_terminal_check_v1"
PAIR_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIRED_SOURCE_E49_CHECKPOINTS_NO_TARGET"
RUNTIME_INIT_PROBE_SCHEMA = "h1_carrierid_date_lodo_phase2_hs_runtime_init_probe_v1"
RUNTIME_INIT_PROBE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_HS_RUNTIME_INIT_PROBE_SOURCE_ONLY_NO_TARGET"

ARM_TO_CONSUMER = {
    "H-S": "src.models.components.spint.SpintModel",
    "H-C": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
}


class DateLodoTerminalError(ValueError):
    """A source-only terminal checkpoint failed a non-negotiable invariant."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DateLodoTerminalError(message)


def _sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
          f"{label} must be a lowercase SHA-256")
    return value


def _immutable_json(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    _need(resolved.is_file() and stat.S_IMODE(resolved.stat().st_mode) == 0o444,
          f"receipt must be immutable mode 0444: {resolved}")
    try:
        body = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DateLodoTerminalError(f"receipt is not valid JSON: {resolved}") from error
    _need(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
          f"receipt schema/status drift: {resolved}")
    return resolved, body, sha256_file(resolved)


def _finite_state_dict(state_dict: Mapping[str, Any]) -> None:
    _need(bool(state_dict), "checkpoint state_dict is empty")
    for name, tensor in state_dict.items():
        _need(isinstance(tensor, torch.Tensor), f"checkpoint state entry is not a tensor: {name!r}")
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            _need(bool(torch.isfinite(tensor).all().item()), f"checkpoint state tensor is nonfinite: {name!r}")


def _config_for_checkpoint(checkpoint_path: Path) -> Path:
    # .../run/checkpoints/fixed_epoch50/epoch_049.ckpt -> .../run/.hydra/config.yaml
    run_dir = checkpoint_path.parent.parent.parent
    config = run_dir / ".hydra" / "config.yaml"
    _need(config.is_file(), f"resolved Hydra config absent next to checkpoint: {config}")
    return config


def _pair_outer_date(pair_preflight: Mapping[str, Any]) -> str:
    """Take the date authority from the immutable pair preflight, never CLI state."""

    outer_date = str(pair_preflight.get("outer_date", ""))
    _need(outer_date in CONFIRMATORY_DATES, "pair preflight outer_date is not a confirmatory date")
    source = pair_preflight.get("source_binding")
    _need(isinstance(source, Mapping) and source.get("outer_date") == outer_date,
          "pair preflight/source binding outer_date mismatch")
    return outer_date


def _check_resolved_config(
    config_path: Path, *, arm: str, outer_date: str, pair_preflight: Path, phase1_path: Path,
) -> str:
    cfg = OmegaConf.load(config_path)
    checks = (
        cfg.get("protocol_id") == f"h1_carrierid_date_lodo_phase2_{arm.lower().replace('-', '')}_{outer_date}_source_only_v1",
        cfg.get("train") is True, cfg.get("test") is False, cfg.get("ckpt_path") is None,
        str(cfg.seed) == "42", str(cfg.phase2.outer_date) == outer_date, str(cfg.phase2.arm) == arm,
        str(cfg.phase2.phase1_preflight_path) == str(phase1_path),
        str(cfg.phase2.pair_preflight_path) == str(pair_preflight),
        str(cfg.data.phase1_preflight_path) == str(phase1_path),
        str(cfg.data._target_) == "src.data.h1_carrierid_date_lodo_phase2.H1CarrierIdDateLodoSourceDataModule",
        str(cfg.model.net._target_) == ARM_TO_CONSUMER[arm], str(cfg.model.arm) == arm,
        str(cfg.model.fixed_seed) == "42", str(cfg.trainer.max_epochs) == "50", str(cfg.trainer.min_epochs) == "50",
        str(cfg.trainer.limit_val_batches) == "0", str(cfg.trainer.num_sanity_val_steps) == "0",
        str(cfg.trainer.accelerator) == "gpu", str(cfg.trainer.devices) == "1", str(cfg.trainer.precision) == "32-true",
    )
    _need(all(checks), f"resolved config violates fixed source-only {arm} contract")
    terminal = cfg.callbacks.get("fixed_epoch50")
    _need(terminal is not None and terminal.get("monitor") is None and int(terminal.get("every_n_epochs", -1)) == 50
          and int(terminal.get("save_top_k", 0)) == -1 and terminal.get("save_last") is False,
          "resolved config lost fixed e49 callback")
    return sha256_file(config_path)


def _hc_wrapper_initial_hash(*, config_path: Path, fresh_component_hash: Any,
                             pair_preflight: Mapping[str, Any]) -> str:
    """Replay the H-C raw-component -> wrapper hash bridge entirely on CPU.

    v2 published a raw ``H1CarrierIdSpint.state_dict()`` digest, while the
    training wrapper stored ``H1CarrierIdDateLodoPhase2LitModule.state_dict()``
    (whose keys are ``net.*``).  Unlike H-S, H-C has no lazy parameter, so its
    wrapper-domain digest is reproducible on CPU from the resolved config.
    Both hash domains must agree with the pair-preflight raw component before
    this narrow compatibility bridge is accepted.  This is date-generic: all
    Phase-2 pair preflights publish the raw component domain and all H-C
    training checkpoints publish the wrapper domain.
    """

    _need(_pair_outer_date(pair_preflight) in CONFIRMATORY_DATES,
          "H-C wrapper replay requires a confirmatory pair preflight")
    expected_component = _sha(fresh_component_hash, "H-C raw component hash")
    _need(pair_preflight.get("code_sha256", {}).get("model")
          == sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_phase2_module.py"),
          "H-C replay model code differs from immutable pair preflight")
    cfg = OmegaConf.load(config_path)
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(42)
        model = hydra.utils.instantiate(cfg.model)
        component_hash = state_hash(model.net.state_dict())
        wrapper_hash = state_hash(model.state_dict())
    finally:
        torch.random.set_rng_state(rng_state)
    _need(component_hash == expected_component,
          "H-C CPU replay raw component hash differs from immutable pair preflight")
    return wrapper_hash


def _validate_hs_runtime_init_probe(*, path: str | Path, pair_preflight: Path,
                                    pair_preflight_sha256: str, pair: Mapping[str, Any],
                                    config_path: Path, config_sha256: str,
                                    checkpoint_initial_hash: Any) -> dict[str, Any]:
    """Validate the explicit GPU/lazy bridge for the exact receipt-bound H-S date."""

    probe_path, probe, probe_sha = _immutable_json(path, schema=RUNTIME_INIT_PROBE_SCHEMA,
                                                    status=RUNTIME_INIT_PROBE_STATUS)
    outer_date = _pair_outer_date(pair)
    _need(probe.get("arm") == "H-S" and probe.get("outer_date") == outer_date,
          "H-S runtime-init probe outer date/arm differs from the terminal checkpoint pair")
    _need(probe.get("pair_preflight") == {"path": str(pair_preflight), "sha256": pair_preflight_sha256},
          "H-S runtime-init probe binds another pair preflight")
    _need(probe.get("source_binding_sha256") == pair.get("source_binding_sha256"),
          "H-S runtime-init probe source binding differs from pair preflight")
    _need(probe.get("production_hs_config") == {"path": str(config_path), "sha256": config_sha256},
          "H-S runtime-init probe did not use this terminal checkpoint config")
    pair_code = pair.get("code_sha256", {})
    probe_code = probe.get("code_sha256", {})
    _need(isinstance(pair_code, Mapping) and isinstance(probe_code, Mapping)
          and pair_code.get("model") == sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_phase2_module.py")
          and probe_code.get("model") == pair_code.get("model")
          and probe_code.get("data") == pair_code.get("data"),
          "H-S runtime-init probe code closure differs from immutable source path")
    _need(_sha(probe_code.get("runtime_probe"), "H-S runtime-init probe code hash")
          == sha256_file(ROOT / "scripts/h1_carrierid_date_lodo_phase2_preflight.py"),
          "H-S runtime-init probe was produced by a different runtime-probe implementation")
    runtime = probe.get("runtime_initialization")
    _need(isinstance(runtime, Mapping)
          and runtime.get("hash_domain") == "lightning_wrapper_state_dict_after_gpu_transfer_and_lazy_materialization"
          and runtime.get("capture_hook") == "on_train_batch_start(epoch=0,batch_idx=0)"
          and runtime.get("capture_precedes_training_step_and_first_optimizer_step") is True
          and runtime.get("optimizer_steps_before_capture") == 0
          and runtime.get("backward_steps_before_capture") == 0
          and runtime.get("probe_optimizer_steps_after_capture") == 1
          and runtime.get("seed") == 42 and runtime.get("accelerator") == "gpu"
          and runtime.get("precision") == "32-true",
          "H-S runtime-init probe lacks exact pre-optimizer GPU/lazy lifecycle evidence")
    _need(_sha(runtime.get("initial_state_sha256"), "H-S runtime-init probe hash")
          == _sha(checkpoint_initial_hash, "H-S checkpoint initial hash"),
          "H-S runtime-init probe hash does not equal terminal checkpoint metadata")
    scope = probe.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
          and scope.get("target_bytes_read") == 0 and scope.get("formal_heldout_opened") is False
          and scope.get("minival_opened") is False and scope.get("evalai_opened") is False
          and scope.get("checkpoint_loaded") is False and scope.get("checkpoint_created") is False,
          "H-S runtime-init probe violated its source-only/no-checkpoint boundary")
    return {"mode": "hs_gpu_lazy_runtime_probe", "path": str(probe_path), "sha256": probe_sha,
            "initial_state_sha256": str(runtime["initial_state_sha256"])}


def _validate_checkpoint(*, checkpoint_path: Path, arm: str, pair_preflight: Mapping[str, Any],
                         phase1_path: Path, pair_path: Path,
                         pair_preflight_sha256: str | None = None,
                         hs_runtime_init_probe_path: str | Path | None = None) -> dict[str, Any]:
    _need(arm in ARM_TO_CONSUMER, f"unsupported arm: {arm}")
    outer_date = _pair_outer_date(pair_preflight)
    _need(checkpoint_path.is_file(), f"checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict),
          "not a Lightning checkpoint with a state_dict")
    _finite_state_dict(checkpoint["state_dict"])
    _need(int(checkpoint.get("epoch", -1)) == 49, "checkpoint must be terminal epoch 49")
    _need(isinstance(checkpoint.get("global_step"), int) and checkpoint["global_step"] > 0,
          "checkpoint global_step must be a positive integer")
    config_path = _config_for_checkpoint(checkpoint_path)
    config_sha = _check_resolved_config(
        config_path, arm=arm, outer_date=outer_date, pair_preflight=pair_path, phase1_path=phase1_path,
    )
    meta = checkpoint.get("h1_carrierid_date_lodo_phase2")
    _need(isinstance(meta, dict), "checkpoint lacks Phase-2 terminal metadata")
    source = pair_preflight.get("source_binding")
    _need(isinstance(source, Mapping), "pair preflight has no source binding")
    fresh = pair_preflight.get("fresh_models", {}).get("h_s" if arm == "H-S" else "h_c")
    _need(isinstance(fresh, Mapping), f"pair preflight lacks fresh {arm} metadata")
    metadata_initial_hash = meta.get("initial_state_sha256")
    initial_verification: dict[str, Any]
    raw_initial_hash = fresh.get("initial_state_sha256")
    if metadata_initial_hash == raw_initial_hash:
        initial_verification = {"mode": "cpu_raw_component_hash_exact", "initial_state_sha256": metadata_initial_hash}
    elif arm == "H-S":
        _need(hs_runtime_init_probe_path is not None and pair_preflight_sha256 is not None,
              "H-S wrapper/lazy initial hash requires an immutable runtime-init probe")
        initial_verification = _validate_hs_runtime_init_probe(
            path=hs_runtime_init_probe_path, pair_preflight=pair_path,
            pair_preflight_sha256=pair_preflight_sha256, pair=pair_preflight,
            config_path=config_path, config_sha256=config_sha,
            checkpoint_initial_hash=metadata_initial_hash,
        )
    else:
        replay_hash = _hc_wrapper_initial_hash(
            config_path=config_path, fresh_component_hash=raw_initial_hash, pair_preflight=pair_preflight,
        )
        _need(metadata_initial_hash == replay_hash,
              "H-C terminal wrapper initial hash differs from the v2 raw-component + CPU-wrapper replay")
        initial_verification = {"mode": "hc_cpu_wrapper_replay",
                                "raw_component_initial_state_sha256": raw_initial_hash,
                                "wrapper_initial_state_sha256": replay_hash}
    required = {
        "schema": CHECKPOINT_SCHEMA, "arm": arm, "outer_date": outer_date, "fresh_seed": 42,
        "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "phase2_source_binding_sha256": canonical_sha256(source),
        "phase1_source_manifest_sha256": source.get("source_manifest_sha256"),
        "phase1_preflight_sha256": source.get("preflight_sha256"),
        "config_sha256": config_sha, "target_optimizer_steps": 0, "target_backward_steps": 0,
        "checkpoint_warm_start": False,
    }
    for key, value in required.items():
        _need(meta.get(key) == value, f"{arm} terminal metadata drift at {key}")
    for key in ("initial_state_sha256", "phase2_source_binding_sha256", "phase1_source_manifest_sha256",
                "phase1_preflight_sha256", "config_sha256"):
        _sha(meta.get(key), f"{arm} metadata.{key}")
    return {
        "arm": arm, "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "outer_date": outer_date,
        "checkpoint_epoch_zero_based": 49, "global_step": int(checkpoint["global_step"]),
        "config_path": str(config_path), "config_sha256": config_sha, "metadata": meta,
        "initial_state_verification": initial_verification,
        "state_dict_tensor_count": len(checkpoint["state_dict"]), "state_dict_finite": True,
    }


def check_single(*, checkpoint_path: str | Path, arm: str, pair_preflight_path: str | Path,
                 launch_receipt_path: str | Path, output_path: str | Path,
                 hs_runtime_init_probe_path: str | Path | None = None) -> dict[str, Any]:
    pair_path, pair, pair_sha = _immutable_json(pair_preflight_path, schema=PAIR_PREFLIGHT_SCHEMA, status=PAIR_PREFLIGHT_STATUS)
    launch_path, launch, launch_sha = _immutable_json(launch_receipt_path, schema=LAUNCH_SCHEMA, status=LAUNCH_STATUS)
    _need(launch.get("pair_preflight_sha256") == pair_sha and launch.get("pair_preflight_path") == str(pair_path),
          "launch receipt binds a different pair preflight")
    outer_date = _pair_outer_date(pair)
    _need(launch.get("outer_date") == outer_date, "launch receipt outer date differs from pair preflight")
    source = pair.get("source_binding")
    _need(isinstance(source, Mapping), "pair preflight lacks source binding")
    phase1_path = Path(str(source.get("preflight_path", ""))).resolve()
    _need(phase1_path.is_file() and stat.S_IMODE(phase1_path.stat().st_mode) == 0o444,
          "pair preflight source Phase-1 receipt is not immutable")
    _need(arm == "H-S" or hs_runtime_init_probe_path is None,
          "H-S runtime-init probe may only be supplied while checking H-S")
    detail = _validate_checkpoint(checkpoint_path=Path(checkpoint_path).resolve(), arm=arm, pair_preflight=pair,
                                  phase1_path=phase1_path, pair_path=pair_path, pair_preflight_sha256=pair_sha,
                                  hs_runtime_init_probe_path=hs_runtime_init_probe_path)
    body = {
        "schema": SINGLE_SCHEMA, "status": SINGLE_STATUS,
        "outer_date": outer_date,
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0, "cuda_constructed_or_launched": False,
                  "trainer_constructed_or_launched": False,
                  "target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"},
        "pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "source_binding_sha256": pair["source_binding_sha256"], "checkpoint": detail,
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": SINGLE_STATUS, "receipt_path": str(written), "receipt_sha256": digest, "checkpoint": detail}


def check_pair(*, hs_checkpoint_path: str | Path, hc_checkpoint_path: str | Path, pair_preflight_path: str | Path,
               launch_receipt_path: str | Path, output_path: str | Path,
               hs_runtime_init_probe_path: str | Path | None = None) -> dict[str, Any]:
    # Validate both checkpoints in memory first; no receipt is accepted as a substitute for the actual checkpoint.
    pair_path, pair, pair_sha = _immutable_json(pair_preflight_path, schema=PAIR_PREFLIGHT_SCHEMA, status=PAIR_PREFLIGHT_STATUS)
    launch_path, launch, launch_sha = _immutable_json(launch_receipt_path, schema=LAUNCH_SCHEMA, status=LAUNCH_STATUS)
    _need(launch.get("pair_preflight_sha256") == pair_sha and launch.get("pair_preflight_path") == str(pair_path),
          "launch receipt binds a different pair preflight")
    outer_date = _pair_outer_date(pair)
    _need(launch.get("outer_date") == outer_date, "launch receipt outer date differs from pair preflight")
    source = pair.get("source_binding")
    _need(isinstance(source, Mapping), "pair preflight lacks source binding")
    phase1_path = Path(str(source.get("preflight_path", ""))).resolve()
    hs = _validate_checkpoint(checkpoint_path=Path(hs_checkpoint_path).resolve(), arm="H-S", pair_preflight=pair,
                              phase1_path=phase1_path, pair_path=pair_path, pair_preflight_sha256=pair_sha,
                              hs_runtime_init_probe_path=hs_runtime_init_probe_path)
    hc = _validate_checkpoint(checkpoint_path=Path(hc_checkpoint_path).resolve(), arm="H-C", pair_preflight=pair,
                              phase1_path=phase1_path, pair_path=pair_path, pair_preflight_sha256=pair_sha)
    shared = ("phase2_source_binding_sha256", "phase1_source_manifest_sha256", "phase1_preflight_sha256")
    _need(all(hs["metadata"][key] == hc["metadata"][key] for key in shared),
          "H-S/H-C terminal checkpoints do not share the immutable source schedule")
    body = {
        "schema": PAIR_SCHEMA, "status": PAIR_STATUS,
        "outer_date": outer_date,
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0, "cuda_constructed_or_launched": False,
                  "trainer_constructed_or_launched": False,
                  "target_evaluator_status": "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"},
        "pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
        "launch_receipt": {"path": str(launch_path), "sha256": launch_sha},
        "source_binding_sha256": pair["source_binding_sha256"],
        "h_s": hs, "h_c": hc,
        "equal_schedule_verified_fields": list(shared),
        "target_gate": "CLOSED: evaluator code exists but has not run; a separately immutable pre-open receipt remains required.",
    }
    written, digest = write_immutable_json(output_path, body)
    return {"status": PAIR_STATUS, "receipt_path": str(written), "receipt_sha256": digest, "h_s": hs, "h_c": hc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-preflight", required=True, type=Path)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--pair", action="store_true")
    parser.add_argument("--arm", choices=tuple(ARM_TO_CONSUMER))
    parser.add_argument("--hs-checkpoint", type=Path)
    parser.add_argument("--hc-checkpoint", type=Path)
    parser.add_argument("--hs-runtime-init-probe", type=Path,
                        help="immutable exact-date GPU/lazy runtime-init receipt; required for an H-S wrapper-hash mismatch")
    args = parser.parse_args()
    if args.checkpoint is not None:
        _need(args.arm is not None, "--checkpoint requires --arm")
        _need(args.hs_checkpoint is None and args.hc_checkpoint is None, "single check does not accept pair checkpoint arguments")
        result = check_single(checkpoint_path=args.checkpoint, arm=args.arm, pair_preflight_path=args.pair_preflight,
                              launch_receipt_path=args.launch_receipt, output_path=args.output,
                              hs_runtime_init_probe_path=args.hs_runtime_init_probe)
    else:
        _need(args.arm is None and args.hs_checkpoint is not None and args.hc_checkpoint is not None,
              "--pair requires --hs-checkpoint and --hc-checkpoint")
        result = check_pair(hs_checkpoint_path=args.hs_checkpoint, hc_checkpoint_path=args.hc_checkpoint,
                            pair_preflight_path=args.pair_preflight, launch_receipt_path=args.launch_receipt,
                            output_path=args.output, hs_runtime_init_probe_path=args.hs_runtime_init_probe)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
