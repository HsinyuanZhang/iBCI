#!/usr/bin/env python3
"""CPU-only source closure and fresh-model-init preflight for one H1 LODO pair.

The preflight reads one immutable Phase-1 source bundle and its declared
source NWBs, materializes fresh seed-42 H-S/H-C consumers on a common source
batch, and writes a paired launch specification.  It does not construct a
Trainer, create a checkpoint, read an outer-date target, or launch a GPU.
Actual source training remains a separate, explicit later operation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_date_lodo_phase2 import (
    H1CarrierIdDateLodoSchedule,
    H1CarrierIdDateLodoSourceDataset,
    PHASE2_SOURCE_BINDING_SCHEMA,
    load_phase2_source_binding,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, state_hash, write_immutable_json
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint
from src.models.components.spint import SpintModel


PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED"
# The receipt name deliberately identifies the H-S arm, but is *not* tied to
# 19250108.  Every confirmatory date uses the same H-S LazyLinear lifecycle
# and therefore needs its own, pair-bound capture in this wrapper hash domain.
RUNTIME_INIT_PROBE_SCHEMA = "h1_carrierid_date_lodo_phase2_hs_runtime_init_probe_v1"
RUNTIME_INIT_PROBE_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_HS_RUNTIME_INIT_PROBE_SOURCE_ONLY_NO_TARGET"
MODEL_KWARGS = {
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _immutable_json(path: str | Path, *, schema: str, status: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _require(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
             f"immutable mode-0444 receipt required: {candidate}")
    try:
        body = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid immutable JSON receipt: {candidate}") from error
    _require(isinstance(body, dict) and body.get("schema") == schema and body.get("status") == status,
             f"receipt schema/status drift: {candidate}")
    return candidate, body, sha256_file(candidate)


def _require_hs_runtime_probe_config(config_path: Path, *, pair_preflight: Path,
                                     pair: dict[str, Any]) -> Any:
    """Load the resolved production H-S config without constructing its data view."""

    _require(config_path.is_file(), f"H-S resolved config is missing: {config_path}")
    cfg = OmegaConf.load(config_path)
    source = pair.get("source_binding")
    _require(isinstance(source, dict), "pair preflight lacks a source binding")
    outer_date = str(pair.get("outer_date", ""))
    _require(outer_date in CONFIRMATORY_DATES,
             "runtime-init probe pair preflight has no confirmatory outer date")
    checks = (
        cfg.get("protocol_id") == f"h1_carrierid_date_lodo_phase2_hs_{outer_date}_source_only_v1",
        cfg.get("train") is True, cfg.get("test") is False, cfg.get("ckpt_path") is None,
        int(cfg.get("seed")) == 42, str(cfg.phase2.outer_date) == outer_date, str(cfg.phase2.arm) == "H-S",
        str(cfg.phase2.pair_preflight_path) == str(pair_preflight),
        str(cfg.phase2.phase1_preflight_path) == str(source.get("preflight_path")),
        str(cfg.data._target_) == "src.data.h1_carrierid_date_lodo_phase2.H1CarrierIdDateLodoSourceDataModule",
        str(cfg.model._target_) == "src.models.h1_carrierid_date_lodo_phase2_module.H1CarrierIdDateLodoPhase2LitModule",
        str(cfg.model.net._target_) == "src.models.components.spint.SpintModel",
        str(cfg.model.arm) == "H-S", int(cfg.model.fixed_seed) == 42,
        int(cfg.trainer.max_epochs) == int(cfg.trainer.min_epochs) == 50,
        str(cfg.trainer.accelerator) == "gpu", str(cfg.trainer.devices) == "1",
        str(cfg.trainer.precision) == "32-true", int(cfg.trainer.limit_val_batches) == 0,
        int(cfg.trainer.num_sanity_val_steps) == 0,
    )
    _require(all(checks), "H-S runtime-init probe config is not the exact production source-only route")
    return cfg


def run_hs_runtime_init_probe(*, pair_preflight: Path, hs_config: Path, probe_root: Path,
                              output: Path) -> dict[str, Any]:
    """Capture H-S's real GPU/lazy initialization hash before optimiser step 1.

    The CPU pair preflight records a raw consumer hash before the Lightning
    wrapper's GPU-side LazyLinear materialization.  The source-trained terminal
    checkpoint instead records the wrapper hash in ``on_train_batch_start``.
    Those are intentionally distinct hash domains for every H-S date, so this
    one-step source-only probe recreates the production construction order and
    captures the latter before ``training_step`` or the first optimizer update.
    """

    _require(torch.cuda.is_available(), "H-S runtime-init probe requires the production CUDA lifecycle")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""),
             "H-S runtime-init probe requires one explicitly selected CUDA device")
    if Path(output).exists():
        raise FileExistsError(f"refusing to overwrite H-S runtime-init probe receipt: {output}")
    probe_root, output = Path(probe_root).resolve(), Path(output).resolve()
    _require(not probe_root.exists(), f"runtime-init probe root already exists: {probe_root}")
    _require(output.parent == probe_root, "runtime-init probe receipt must be written directly under its new probe root")
    pair_path, pair, pair_sha = _immutable_json(pair_preflight, schema=PREFLIGHT_SCHEMA, status=PREFLIGHT_STATUS)
    outer_date = str(pair.get("outer_date", ""))
    _require(outer_date in CONFIRMATORY_DATES,
             "runtime-init probe pair preflight has no confirmatory outer date")
    source = pair.get("source_binding")
    _require(isinstance(source, dict) and source.get("target_recordings_opened") == 0
             and source.get("target_bytes_read") == 0,
             "pair preflight cannot authorize a runtime probe after target access")
    _require(pair.get("source_binding_sha256") == canonical_sha256(source), "pair source-binding SHA drift")
    _require(pair.get("code_sha256", {}).get("model") == sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_phase2_module.py")
             and pair.get("code_sha256", {}).get("data") == sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_phase2.py"),
             "runtime probe model/data code differs from the immutable pair preflight")
    hs_config = Path(hs_config).resolve()
    cfg = _require_hs_runtime_probe_config(hs_config, pair_preflight=pair_path, pair=pair)

    # The wrapper requires ``default_root_dir/.hydra/config.yaml`` when it
    # records source provenance.  Copying the byte-identical production config
    # to a new probe-only directory leaves the active training run untouched.
    probe_root.mkdir(parents=True)
    copied_config = probe_root / ".hydra" / "config.yaml"
    copied_config.parent.mkdir(parents=True)
    shutil.copyfile(hs_config, copied_config)
    _require(sha256_file(copied_config) == sha256_file(hs_config), "probe config copy byte drift")
    try:
        # Match src/train.py's seed -> DataModule -> model construction order.
        L.seed_everything(42, workers=True)
        datamodule = hydra.utils.instantiate(cfg.data)
        model = hydra.utils.instantiate(cfg.model)
        # No callback is admitted here: the production fixed-e49 checkpoint
        # callback is incompatible with ``enable_checkpointing=False`` and
        # cannot affect model initialization.  The seed/DataModule/model and
        # Trainer attachment order above is the state-bearing path.
        trainer = L.Trainer(
            accelerator="gpu", devices=1, precision="32-true", max_epochs=50, min_epochs=50,
            max_steps=1, logger=False, enable_checkpointing=False, limit_val_batches=0,
            num_sanity_val_steps=0, check_val_every_n_epoch=1, deterministic=False,
            default_root_dir=str(probe_root), use_distributed_sampler=False,
        )
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
        _require(model._initial_state_sha256 is not None, "runtime probe never captured H-S initial state")
        _require(trainer.global_step == 1, "runtime probe did not execute exactly one post-capture source optimizer step")
        observed_source = datamodule.phase2_source_manifest()
        _require(canonical_sha256(observed_source) == pair["source_binding_sha256"],
                 "runtime probe source schedule differs from v2 pair preflight")
        _require(observed_source.get("target_recordings_opened") == 0 and observed_source.get("target_bytes_read") == 0,
                 "runtime probe target boundary drift")
        _require(model._source_binding_sha256 == pair["source_binding_sha256"],
                 "runtime probe model captured another source binding")
        _require(model._config_sha256 == sha256_file(hs_config), "runtime probe did not use the exact H-S resolved config")
        receipt = {
            "schema": RUNTIME_INIT_PROBE_SCHEMA,
            "status": RUNTIME_INIT_PROBE_STATUS,
            "mode": "explicit_gpu_source_only_hs_runtime_lifecycle_one_step_capture_before_first_optimizer_step",
            "arm": "H-S", "outer_date": outer_date,
            "pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
            "source_binding_sha256": pair["source_binding_sha256"],
            "production_hs_config": {"path": str(hs_config), "sha256": sha256_file(hs_config)},
            "runtime_initialization": {
                "hash_domain": "lightning_wrapper_state_dict_after_gpu_transfer_and_lazy_materialization",
                "initial_state_sha256": model._initial_state_sha256,
                "capture_hook": "on_train_batch_start(epoch=0,batch_idx=0)",
                "capture_precedes_training_step_and_first_optimizer_step": True,
                "optimizer_steps_before_capture": 0, "backward_steps_before_capture": 0,
                "probe_optimizer_steps_after_capture": 1,
                "seed": 42, "accelerator": "gpu", "precision": "32-true",
            },
            "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                      "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False,
                      "checkpoint_loaded": False, "checkpoint_created": False},
            "code_sha256": {"runtime_probe": sha256_file(Path(__file__).resolve()),
                              "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_phase2_module.py"),
                              "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_phase2.py")},
        }
        written, digest = write_immutable_json(output, receipt)
        return {"status": RUNTIME_INIT_PROBE_STATUS, "receipt_path": str(written), "receipt_sha256": digest}
    except Exception:
        # Keep the failed probe directory for forensic inspection; never reuse
        # it, because reusing it could blur a fresh-init provenance boundary.
        raise


def _materialize_pair(source_batch: tuple[Any, ...]) -> dict[str, Any]:
    neural, _target, identity, _session, carrier = source_batch
    neural = neural.to(dtype=torch.float32)
    identity = identity.to(dtype=torch.float32)
    carrier = carrier.to(dtype=torch.float32)
    _require(tuple(neural.shape[1:]) == (700, 176), f"source neural shape drift: {tuple(neural.shape)}")
    _require(tuple(identity.shape[1:]) == (4, 1024, 176), f"source identity shape drift: {tuple(identity.shape)}")
    _require(tuple(carrier.shape[1:]) == (176, 4), f"source normalized carrier shape drift: {tuple(carrier.shape)}")
    with torch.no_grad():
        torch.manual_seed(42)
        hs = SpintModel(**MODEL_KWARGS).eval()
        hs_output = hs(neural, calib_trialized_neural_features=identity)
        hs_hash = state_hash(hs.state_dict())
        torch.manual_seed(42)
        hc = H1CarrierIdSpint(carrier_hidden_dim=32, carrier_dim=4, carrier_trial_length=1024, zero_carrier=False,
                              **MODEL_KWARGS).eval()
        hc_output = hc(neural, calib_trialized_neural_features=identity, carrier=carrier)
        hc_hash = state_hash(hc.state_dict())
    _require(tuple(hs_output.shape) == tuple(hc_output.shape) == (neural.shape[0], 700, 7),
             "H-S/H-C fresh consumer output shape mismatch")
    _require(torch.isfinite(hs_output).all().item() and torch.isfinite(hc_output).all().item(),
             "fresh source-only consumer output is nonfinite")
    _require(torch.count_nonzero(hc.carrier_post_pool[0].weight[:, hc.carrier_hidden_dim:]).item() == 0,
             "H-C carrier columns must literal-zero initialize")
    return {
        "h_s": {"component": "SpintModel", "initial_state_sha256": hs_hash,
                "output_shape": list(hs_output.shape), "fresh_seed": 42},
        "h_c": {"component": "H1CarrierIdSpint", "initial_state_sha256": hc_hash,
                "output_shape": list(hc_output.shape), "fresh_seed": 42,
                "carrier_columns_literal_zero_at_init": True},
    }


def run(*, data_dir: Path, phase1_preflight: Path, outer_date: str, output: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("Phase-2 CPU preflight requires CUDA_VISIBLE_DEVICES to be unset")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase-2 pair preflight: {output}")
    if str(outer_date) not in CONFIRMATORY_DATES:
        raise ValueError("outer_date must be one of the five confirmatory dates")
    binding = load_phase2_source_binding(
        data_dir=data_dir, phase1_preflight_path=phase1_preflight, outer_date=outer_date,
    )
    dataset = H1CarrierIdDateLodoSourceDataset(binding)
    sampler = H1CarrierIdDateLodoSchedule(dataset, binding)
    first_requests = next(iter(sampler))
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_sampler=[first_requests], num_workers=0)))
    models = _materialize_pair(batch)
    source_manifest = binding.manifest()
    _require(source_manifest["schema"] == PHASE2_SOURCE_BINDING_SCHEMA, "Phase-2 source binding schema drift")
    _require(source_manifest["target_recordings_opened"] == 0 and source_manifest["target_bytes_read"] == 0,
             "Phase-2 preflight target boundary drift")
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "mode": "cpu_only_source_binding_and_model_materialization_no_trainer_no_gpu_no_target",
        "outer_date": str(outer_date),
        "source_binding": source_manifest,
        "source_binding_sha256": canonical_sha256(source_manifest),
        "paired_source_schedule": {
            "same_for_h_s_and_h_c": True,
            "batch_size": 32,
            "epochs": 50,
            "first_batch_request_count": len(first_requests),
            "batch_order_sha256": binding.batch_order_sha256,
            "calibration_schedule_sha256": binding.calibration_schedule_sha256,
        },
        "fresh_models": models,
        "phase2_training_contract": {
            "arms": ["H-S", "H-C"],
            "fresh_seed": 42,
            "fixed_terminal_epoch_zero_based": 49,
            "epochs": 50,
            "checkpoint_warm_start_forbidden": True,
            "source_only_training": True,
            "target_evaluator_status": "NOT_IMPLEMENTED",
            "gpu_launch_authorized": False,
        },
        "scope": {
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "cuda_constructed_or_launched": False,
        },
        "code_sha256": {
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_phase2.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_phase2_module.py"),
            "preflight": sha256_file(Path(__file__).resolve()),
        },
    }
    write_immutable_json(output, receipt)
    return {**receipt, "output": str(output.resolve()), "output_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument(
        "--phase1-preflight", type=Path,
        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json",
    )
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES)
    parser.add_argument("--run-hs-runtime-init-probe", action="store_true",
                        help="explicit one-step source-only GPU runtime-init probe for one receipt-bound H-S date")
    parser.add_argument("--pair-preflight", type=Path,
                        help="immutable receipt-bound pair preflight required by --run-hs-runtime-init-probe")
    parser.add_argument("--hs-config", type=Path,
                        help="resolved production H-S .hydra/config.yaml required by --run-hs-runtime-init-probe")
    parser.add_argument("--probe-root", type=Path,
                        help="new, empty probe-only root required by --run-hs-runtime-init-probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run_hs_runtime_init_probe:
        _require(args.outer_date is None and args.pair_preflight is not None and args.hs_config is not None
                 and args.probe_root is not None,
                 "runtime-init probe requires --pair-preflight, --hs-config, and --probe-root; do not pass --outer-date")
        result = run_hs_runtime_init_probe(pair_preflight=args.pair_preflight, hs_config=args.hs_config,
                                           probe_root=args.probe_root, output=args.output)
    else:
        _require(args.outer_date is not None and args.pair_preflight is None and args.hs_config is None
                 and args.probe_root is None,
                 "CPU pair preflight requires --outer-date only; runtime-init arguments require --run-hs-runtime-init-probe")
        result = run(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
                     outer_date=args.outer_date, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
