#!/usr/bin/env python3
"""No-target audit of one real H-LS fixed-e49 source checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    SOURCE_PREFLIGHT_SCHEMA, SOURCE_PREFLIGHT_STATUS, SOURCE_TERMINAL_SCHEMA,
    SOURCE_TERMINAL_STATUS, read_immutable_json, sha256_file, write_immutable_json,
)
from src.models.h1_carrierid_date_lodo_hls_module import HLS_CHECKPOINT_SCHEMA


class HlsSourceTerminalAuditError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsSourceTerminalAuditError(message)


def _finite_state(state: Mapping[str, Any]) -> None:
    _need(bool(state), "H-LS terminal state_dict is empty")
    for name, tensor in state.items():
        _need(isinstance(tensor, torch.Tensor), f"H-LS checkpoint state is not tensor: {name}")
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            _need(bool(torch.isfinite(tensor).all().item()), f"H-LS checkpoint state is nonfinite: {name}")


def audit(*, checkpoint: Path, source_preflight: Path, output: Path) -> dict[str, Any]:
    preflight_path, preflight, preflight_sha = read_immutable_json(
        source_preflight, schema=SOURCE_PREFLIGHT_SCHEMA, status=SOURCE_PREFLIGHT_STATUS,
    )
    checkpoint = Path(checkpoint).resolve()
    _need(checkpoint.is_file(), "H-LS terminal checkpoint is missing")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping),
          "H-LS checkpoint is not a Lightning state_dict")
    _finite_state(payload["state_dict"])
    _need(int(payload.get("epoch", -1)) == 49 and int(payload.get("global_step", 0)) > 0,
          "H-LS checkpoint is not the fixed terminal e49 state")
    config_path = checkpoint.parent.parent.parent / ".hydra" / "config.yaml"
    _need(config_path.is_file(), "resolved H-LS Hydra config is missing next to checkpoint")
    cfg = OmegaConf.load(config_path)
    date = str(preflight.get("outer_date", ""))
    checks = (
        cfg.get("protocol_id") == f"h1_carrierid_date_lodo_hls_{date}_source_only_v1",
        cfg.get("train") is True, cfg.get("test") is False, cfg.get("ckpt_path") is None,
        int(cfg.get("seed")) == 42, str(cfg.phase2.outer_date) == date, str(cfg.phase2.arm) == "H-LS",
        Path(str(cfg.phase2.hls_source_preflight_path)).resolve() == preflight_path,
        str(cfg.data._target_) == "src.data.h1_carrierid_date_lodo_hls.H1CarrierIdDateLodoHlsSourceDataModule",
        str(cfg.model._target_) == "src.models.h1_carrierid_date_lodo_hls_module.H1CarrierIdDateLodoHlsLitModule",
        str(cfg.model.net._target_) == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        int(cfg.model.net.carrier_hidden_dim) == 32 and int(cfg.model.net.carrier_dim) == 4,
        bool(cfg.model.net.zero_carrier) is False,
        int(cfg.trainer.max_epochs) == int(cfg.trainer.min_epochs) == 50,
        int(cfg.trainer.limit_val_batches) == int(cfg.trainer.num_sanity_val_steps) == 0,
    )
    _need(all(checks), "resolved H-LS config violates the h=32 matched source contract")
    meta = payload.get("h1_carrierid_date_lodo_hls")
    _need(isinstance(meta, Mapping) and meta.get("schema") == HLS_CHECKPOINT_SCHEMA
          and meta.get("arm") == "H-LS" and meta.get("outer_date") == date,
          "H-LS checkpoint metadata schema/arm/date drift")
    required = {
        "fresh_seed": 42, "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
        "carrier_intervention": "temporal_velocity_label_rotation",
        "hls_source_binding_sha256": preflight.get("source_binding_sha256"),
        "phase1_source_manifest_sha256": preflight.get("source_binding", {}).get("source_manifest_sha256"),
        "phase1_preflight_sha256": preflight.get("source_binding", {}).get("preflight_sha256"),
        "hls_source_preflight_sha256": preflight_sha,
        "config_sha256": sha256_file(config_path),
        "target_optimizer_steps": 0, "target_backward_steps": 0, "checkpoint_warm_start": False,
    }
    for key, value in required.items():
        _need(meta.get(key) == value, f"H-LS checkpoint metadata drift at {key}")
    _need(meta.get("initial_state_sha256") == preflight.get("fresh_model", {}).get("wrapper_initial_state_sha256"),
          "H-LS checkpoint did not start from the preflighted fresh wrapper state")
    body = {
        "schema": SOURCE_TERMINAL_SCHEMA, "status": SOURCE_TERMINAL_STATUS,
        "outer_date": date, "arm": "H-LS",
        "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "source_binding_sha256": preflight["source_binding_sha256"],
        "source_controls": {
            "carrier_intervention": "temporal_velocity_label_rotation", "seed": 42,
            "epochs": 50, "fixed_terminal_epoch_zero_based": 49, "warm_start": False,
        },
        "checkpoint": {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "metadata": dict(meta), "global_step": int(payload["global_step"]),
            "state_dict_tensor_count": len(payload["state_dict"]), "state_dict_finite": True,
        },
        "deployment_updates": {"target_optimizer_steps": 0, "target_backward_steps": 0,
                               "target_model_state_updated": False},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False},
        "code_sha256": {"terminal_audit": sha256_file(Path(__file__).resolve())},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": SOURCE_TERMINAL_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(checkpoint=args.checkpoint, source_preflight=args.source_preflight,
                           output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
