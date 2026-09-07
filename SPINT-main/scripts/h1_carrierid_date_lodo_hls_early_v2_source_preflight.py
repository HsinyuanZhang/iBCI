#!/usr/bin/env python3
"""CPU/source-only preflight for one H1 H-LS early-launch-v2 date.

Unlike v1, this source preflight does not require the completed H-S/H-C
five-date aggregate.  It instead binds the already immutable waiting plan,
strong-null audit, exact matched H-C pair preflight, and exact Phase-1 source
binding.  Target access remains impossible in this module.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import hydra
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES, DEFAULT_PHASE1_PREFLIGHT, DEFAULT_WAITING_PLAN, EARLY_PREFLIGHT_SCHEMA,
    EARLY_PREFLIGHT_STATUS, NULL_AUDIT_DEFAULT, PAIR_SCHEMA, PAIR_STATUS,
    need, read_immutable_json, sha256_file, validate_null_strength_audit,
    validate_waiting_plan, write_immutable_json,
)
from src.data.h1_carrierid_date_lodo_hls import (
    H1CarrierIdDateLodoHlsSourceDataset, hls_source_manifest,
)
from src.data.h1_carrierid_date_lodo_phase2 import load_phase2_source_binding
from src.h1_m4_cce_contract import canonical_sha256, state_hash


EXPERIMENT = "h1_carrierid_date_lodo_hls_early_v2"


def _read_pair(path: Path, *, outer_date: str) -> tuple[Path, dict[str, Any], str]:
    pair_path, pair, digest = read_immutable_json(path, schema=PAIR_SCHEMA, status=PAIR_STATUS)
    need(pair.get("outer_date") == outer_date, "early-v2 matched H-C pair outer-date drift")
    scope, contract = pair.get("scope"), pair.get("phase2_training_contract")
    need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
         and scope.get("target_bytes_read") == 0,
         "early-v2 matched H-C pair preflight records target access")
    need(isinstance(contract, Mapping) and contract.get("fresh_seed") == 42
         and contract.get("epochs") == 50 and contract.get("fixed_terminal_epoch_zero_based") == 49
         and contract.get("checkpoint_warm_start_forbidden") is True,
         "early-v2 matched H-C pair contract drift")
    return pair_path, pair, digest


def _compose(*, outer_date: str, phase1_preflight: Path, early_preflight: Path) -> tuple[Any, dict[str, Any]]:
    overrides = [
        f"experiment={EXPERIMENT}", f"phase2.outer_date={outer_date}",
        f"phase2.phase1_preflight_path={phase1_preflight}",
        f"phase2.hls_source_preflight_path={early_preflight}",
        "seed=42", "ckpt_path=null", "train=true", "test=false",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    raw = OmegaConf.to_container(cfg, resolve=False)
    need(isinstance(raw, dict), "early-v2 Hydra composition is malformed")
    model, data, trainer = raw["model"], raw["data"], raw["trainer"]
    fixed = (
        raw["train"] is True, raw["test"] is False, raw["ckpt_path"] is None, int(raw["seed"]) == 42,
        str(raw["phase2"]["outer_date"]) == outer_date, raw["phase2"]["arm"] == "H-LS",
        raw["phase2"]["target_evaluator_status"] == "BLOCKED_UNTIL_COMPLETE_UPSTREAM_BINDER",
        raw["phase2"]["phase1_preflight_path"] == str(phase1_preflight),
        raw["phase2"]["hls_source_preflight_path"] == str(early_preflight),
        data["_target_"] == "src.data.h1_carrierid_date_lodo_hls_early_v2.H1CarrierIdDateLodoHlsEarlyV2DataModule",
        model["_target_"] == "src.models.h1_carrierid_date_lodo_hls_module.H1CarrierIdDateLodoHlsLitModule",
        model["net"]["_target_"] == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        int(model["net"]["carrier_hidden_dim"]) == 32, int(model["net"]["carrier_dim"]) == 4,
        bool(model["net"]["zero_carrier"]) is False,
        int(trainer["max_epochs"]) == int(trainer["min_epochs"]) == 50,
        int(trainer["limit_val_batches"]) == int(trainer["num_sanity_val_steps"]) == 0,
    )
    need(all(fixed), "early-v2 composed config differs from fixed h32 H-C topology")
    terminal = raw["callbacks"]["fixed_epoch50"]
    need(terminal["monitor"] is None and int(terminal["every_n_epochs"]) == 50
         and int(terminal["save_top_k"]) == -1 and terminal["save_last"] is False,
         "early-v2 config lost fixed terminal e49 callback")
    return cfg, raw


def run(*, data_dir: Path, phase1_preflight: Path, pair_preflight: Path,
        waiting_plan: Path, null_strength_audit: Path, outer_date: str,
        output: Path) -> dict[str, Any]:
    need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
         "early-v2 source preflight requires CUDA_VISIBLE_DEVICES unset")
    need(str(outer_date) in DATES, "early-v2 outer date is outside the frozen five-date grid")
    need(not output.exists() and not output.is_symlink() and not os.path.lexists(str(output)),
         "refusing to overwrite early-v2 source preflight")
    waiting_path, _waiting, waiting_sha = validate_waiting_plan(waiting_plan)
    audit_path, _audit, audit_sha = validate_null_strength_audit(null_strength_audit)
    pair_path, pair, pair_sha = _read_pair(pair_preflight, outer_date=str(outer_date))
    phase1_preflight = Path(phase1_preflight).resolve()
    need(phase1_preflight.is_file() and not phase1_preflight.is_symlink()
         and stat.S_IMODE(phase1_preflight.stat().st_mode) == 0o444,
         "early-v2 Phase-1 source preflight must be immutable")
    pair_source = pair.get("source_binding")
    need(isinstance(pair_source, Mapping)
         and pair_source.get("preflight_path") == str(phase1_preflight)
         and pair_source.get("target_recordings_opened") == 0
         and pair_source.get("target_bytes_read") == 0,
         "early-v2 matched pair does not bind the requested Phase-1 source preflight")

    binding = load_phase2_source_binding(
        data_dir=data_dir, phase1_preflight_path=phase1_preflight, outer_date=str(outer_date),
    )
    base = binding.manifest()
    need(pair.get("source_binding_sha256") == canonical_sha256(base) and pair_source == base,
         "early-v2 actual Phase-1 source binding differs from matched H-C pair")
    dataset = H1CarrierIdDateLodoHlsSourceDataset(binding)
    source = hls_source_manifest(binding, dataset)
    need(source.get("effective_source_carriers_nonidentity_all") is True,
         "early-v2 label rotation did not change source carriers")

    cfg, raw = _compose(outer_date=str(outer_date), phase1_preflight=phase1_preflight,
                        early_preflight=output.resolve())
    torch.manual_seed(42)
    component = hydra.utils.instantiate(cfg.model.net).eval()
    component_hash = state_hash(component.state_dict())
    matched = pair.get("fresh_models", {}).get("h_c")
    need(isinstance(matched, Mapping) and matched.get("component") == "H1CarrierIdSpint"
         and matched.get("fresh_seed") == 42
         and matched.get("initial_state_sha256") == component_hash
         and matched.get("carrier_columns_literal_zero_at_init") is True,
         "early-v2 fresh component is not initialization-identical to matched H-C")
    torch.manual_seed(42)
    wrapper_hash = state_hash(hydra.utils.instantiate(cfg.model).eval().state_dict())

    body = {
        "schema": EARLY_PREFLIGHT_SCHEMA, "status": EARLY_PREFLIGHT_STATUS,
        "route": "H1-HLS-EARLY-V2-SOURCE-ONLY", "outer_date": str(outer_date), "arm": "H-LS",
        "upstream_policy": {
            "complete_hs_hc_aggregate_required_for_source_training": False,
            "complete_hs_hc_aggregate_required_before_any_target_or_evaluation": True,
            "post_upstream_binder_required": True,
            "does_not_claim_v1_ready": True,
        },
        "waiting_plan": {"path": str(waiting_path), "sha256": waiting_sha},
        "null_strength_audit": {"path": str(audit_path), "sha256": audit_sha},
        "matched_h_c_pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
        "phase1_preflight": {"path": str(phase1_preflight), "sha256": sha256_file(phase1_preflight)},
        "matched_h_c_base_source_binding": base,
        "matched_h_c_base_source_binding_sha256": canonical_sha256(base),
        "source_binding": source, "source_binding_sha256": canonical_sha256(source),
        "source_controls": {
            "carrier_intervention": "temporal_velocity_label_rotation",
            "same_h_c_source_windows": True, "same_h_c_source_schedule": True,
            "same_h_c_normalizer": True, "same_h_c_h32_topology": True,
            "seed": 42, "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
            "warm_start": False,
        },
        "fresh_model": {
            "component": "H1CarrierIdSpint", "carrier_hidden_dim": 32,
            "component_initial_state_sha256": component_hash,
            "wrapper_initial_state_sha256": wrapper_hash,
            "equal_to_matched_h_c_component_initial_state": True, "ci64_used": False,
        },
        "composed_config": {"canonical_sha256": canonical_sha256(raw), "experiment": EXPERIMENT},
        "scope": {
            "source_recordings_opened": len(binding.source_sessions),
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False, "tmux_started": False,
        },
        "code_sha256": {
            "early_preflight": sha256_file(Path(__file__).resolve()),
            "early_data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_hls_early_v2.py"),
            "hls_data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_hls.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_spint.py"),
            "experiment": sha256_file(ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_early_v2.yaml"),
            "data_config": sha256_file(ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls_early_v2.yaml"),
            "model_config": sha256_file(ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml"),
            "terminal_callback": sha256_file(ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml"),
        },
    }
    written, digest = write_immutable_json(output, body)
    return {"status": EARLY_PREFLIGHT_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--phase1-preflight", type=Path, default=DEFAULT_PHASE1_PREFLIGHT)
    parser.add_argument("--pair-preflight", required=True, type=Path)
    parser.add_argument("--waiting-plan", type=Path, default=DEFAULT_WAITING_PLAN)
    parser.add_argument("--null-strength-audit", type=Path, default=NULL_AUDIT_DEFAULT)
    parser.add_argument("--outer-date", required=True, choices=DATES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
                         pair_preflight=args.pair_preflight, waiting_plan=args.waiting_plan,
                         null_strength_audit=args.null_strength_audit,
                         outer_date=args.outer_date, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
