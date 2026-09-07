#!/usr/bin/env python3
"""No-target audit for one actual H-LS early-v2 source e49 checkpoint."""
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

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    EARLY_PREFLIGHT_SCHEMA, EARLY_PREFLIGHT_STATUS, EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS, need, read_immutable_json, sha256_file, write_immutable_json,
)
from src.models.h1_carrierid_date_lodo_hls_module import HLS_CHECKPOINT_SCHEMA


WRITER_CLAIM_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_writer_claim_v1"
WRITER_CLAIM_STATUS = "CLAIMED_BEFORE_SOURCE_PROCESS_START"


def _finite_state(state: Mapping[str, Any]) -> None:
    need(bool(state), "early-v2 terminal state_dict is empty")
    for name, tensor in state.items():
        need(isinstance(tensor, torch.Tensor), f"early-v2 checkpoint state is not tensor: {name}")
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            need(bool(torch.isfinite(tensor).all().item()), f"early-v2 checkpoint state is nonfinite: {name}")


def audit(*, checkpoint: Path, source_preflight: Path, writer_claim: Path,
          output: Path) -> dict[str, Any]:
    preflight_path, preflight, preflight_sha = read_immutable_json(
        source_preflight, schema=EARLY_PREFLIGHT_SCHEMA, status=EARLY_PREFLIGHT_STATUS,
    )
    claim_path, claim, claim_sha = read_immutable_json(
        writer_claim, schema=WRITER_CLAIM_SCHEMA, status=WRITER_CLAIM_STATUS,
    )
    date = str(preflight.get("outer_date", ""))
    need(claim.get("outer_date") == date
         and claim.get("source_preflight", {}).get("path") == str(preflight_path)
         and claim.get("source_preflight", {}).get("sha256") == preflight_sha
         and claim.get("source_binding_sha256") == preflight.get("source_binding_sha256")
         and claim.get("target_recordings_opened") == 0 and claim.get("target_bytes_read") == 0,
         "early-v2 writer claim/preflight/date drift")
    checkpoint = Path(checkpoint).resolve()
    need(checkpoint.is_file(), "early-v2 terminal checkpoint is missing")
    need(Path(str(claim.get("run_dir", ""))).resolve() == checkpoint.parent.parent.parent,
         "early-v2 checkpoint is outside its one-writer claimed run directory")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping),
         "early-v2 checkpoint is not a Lightning state_dict")
    _finite_state(payload["state_dict"])
    need(int(payload.get("epoch", -1)) == 49 and int(payload.get("global_step", 0)) > 0,
         "early-v2 checkpoint is not fixed terminal e49")
    config_path = checkpoint.parent.parent.parent / ".hydra/config.yaml"
    need(config_path.is_file(), "early-v2 resolved config is missing")
    cfg = OmegaConf.load(config_path)
    fixed = (
        cfg.get("protocol_id") == f"h1_carrierid_date_lodo_hls_early_v2_{date}_source_only_v1",
        cfg.get("train") is True, cfg.get("test") is False, cfg.get("ckpt_path") is None,
        int(cfg.get("seed")) == 42, str(cfg.phase2.outer_date) == date, str(cfg.phase2.arm) == "H-LS",
        str(cfg.phase2.target_evaluator_status) == "BLOCKED_UNTIL_COMPLETE_UPSTREAM_BINDER",
        Path(str(cfg.phase2.hls_source_preflight_path)).resolve() == preflight_path,
        str(cfg.data._target_) == "src.data.h1_carrierid_date_lodo_hls_early_v2.H1CarrierIdDateLodoHlsEarlyV2DataModule",
        str(cfg.model._target_) == "src.models.h1_carrierid_date_lodo_hls_module.H1CarrierIdDateLodoHlsLitModule",
        str(cfg.model.net._target_) == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        int(cfg.model.net.carrier_hidden_dim) == 32 and int(cfg.model.net.carrier_dim) == 4,
        bool(cfg.model.net.zero_carrier) is False,
        int(cfg.trainer.max_epochs) == int(cfg.trainer.min_epochs) == 50,
        int(cfg.trainer.limit_val_batches) == int(cfg.trainer.num_sanity_val_steps) == 0,
    )
    need(all(fixed), "early-v2 resolved config violates fixed source-only h32 contract")
    meta = payload.get("h1_carrierid_date_lodo_hls")
    need(isinstance(meta, Mapping) and meta.get("schema") == HLS_CHECKPOINT_SCHEMA
         and meta.get("arm") == "H-LS" and meta.get("outer_date") == date,
         "early-v2 checkpoint metadata schema/arm/date drift")
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
        need(meta.get(key) == value, f"early-v2 checkpoint metadata drift at {key}")
    need(meta.get("initial_state_sha256") == preflight.get("fresh_model", {}).get("wrapper_initial_state_sha256"),
         "early-v2 checkpoint did not start from preflighted fresh wrapper state")
    body = {
        "schema": EARLY_TERMINAL_SCHEMA, "status": EARLY_TERMINAL_STATUS,
        "route": "H1-HLS-EARLY-V2-SOURCE-ONLY", "outer_date": date, "arm": "H-LS",
        "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
        "writer_claim": {"path": str(claim_path), "sha256": claim_sha,
                         "owner": claim.get("owner"), "physical_gpu": claim.get("physical_gpu")},
        "source_binding_sha256": preflight["source_binding_sha256"],
        "base_source_binding_sha256": preflight["matched_h_c_base_source_binding_sha256"],
        "matched_h_c_pair_preflight": dict(preflight["matched_h_c_pair_preflight"]),
        "source_controls": {"carrier_intervention": "temporal_velocity_label_rotation",
                            "seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
                            "warm_start": False},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "config_path": str(config_path), "config_sha256": sha256_file(config_path),
                       "metadata": dict(meta), "global_step": int(payload["global_step"]),
                       "state_dict_tensor_count": len(payload["state_dict"]), "state_dict_finite": True},
        "deployment_updates": {"target_optimizer_steps": 0, "target_backward_steps": 0,
                               "target_model_state_updated": False},
        "post_upstream_state": "NOT_YET_BOUND; TARGET_GATE_CLOSED",
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False},
        "code_sha256": {"terminal_audit": sha256_file(Path(__file__).resolve())},
    }
    written, digest = write_immutable_json(output, body)
    return {"status": EARLY_TERMINAL_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-preflight", required=True, type=Path)
    parser.add_argument("--writer-claim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(checkpoint=args.checkpoint, source_preflight=args.source_preflight,
                           writer_claim=args.writer_claim, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
