#!/usr/bin/env python3
"""CPU/source-only preflight for one exact h=32 H-LS date-LODO run.

This is the only preparation step allowed to open source recordings.  It
requires the complete immutable H-S/H-C five-date aggregate, reconstructs the
exact Phase-2 source binding, applies the audited temporal label rotation, and
checks that the fresh H-LS consumer is byte-state-identical to the matched H-C
consumer at seed 42.  It never opens an outer-date target, constructs a
Trainer/CUDA object, loads a checkpoint, or launches a process.
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

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, NULL_AUDIT_DEFAULT, SOURCE_PREFLIGHT_SCHEMA, SOURCE_PREFLIGHT_STATUS,
    UPSTREAM_AGGREGATE_DEFAULT, WAITING_SCHEMA, WAITING_STATUS,
    read_immutable_json, require_complete_upstream, sha256_file,
    validate_null_strength_audit, write_immutable_json,
)
from src.data.h1_carrierid_date_lodo_hls import (
    H1CarrierIdDateLodoHlsSourceDataset, hls_source_manifest,
)
from src.data.h1_carrierid_date_lodo_phase2 import load_phase2_source_binding
from src.h1_m4_cce_contract import canonical_sha256, state_hash


PAIR_SCHEMA = "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
PAIR_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED"
EXPERIMENT = "h1_carrierid_date_lodo_hls_phase2"


class HlsSourcePreflightError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsSourcePreflightError(message)


def _read_pair(path: str | Path, *, outer_date: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and not candidate.is_symlink()
          and stat.S_IMODE(candidate.stat().st_mode) == 0o444,
          "matched H-C pair preflight must be immutable mode 0444")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == PAIR_SCHEMA and body.get("status") == PAIR_STATUS
          and body.get("outer_date") == outer_date,
          "matched H-C pair preflight schema/status/date drift")
    scope = body.get("scope")
    _need(isinstance(scope, Mapping) and scope.get("target_recordings_opened") == 0
          and scope.get("target_bytes_read") == 0,
          "matched H-C pair preflight records target access")
    return candidate, body, sha256_file(candidate)


def _compose(*, outer_date: str, phase1_preflight: Path, hls_source_preflight: Path) -> tuple[Any, dict[str, Any]]:
    overrides = [
        f"experiment={EXPERIMENT}", f"phase2.outer_date={outer_date}",
        f"phase2.phase1_preflight_path={phase1_preflight}",
        f"phase2.hls_source_preflight_path={hls_source_preflight}",
        "ckpt_path=null", "test=false",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str((ROOT / "configs").resolve())):
        cfg = hydra.compose(config_name="train", overrides=overrides)
    raw = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(raw, dict), "H-LS Hydra composition is malformed")
    model, data, trainer = raw["model"], raw["data"], raw["trainer"]
    checks = (
        raw["train"] is True, raw["test"] is False, raw["ckpt_path"] is None, int(raw["seed"]) == 42,
        raw["phase2"]["arm"] == "H-LS", str(raw["phase2"]["outer_date"]) == outer_date,
        raw["phase2"]["phase1_preflight_path"] == str(phase1_preflight),
        raw["phase2"]["hls_source_preflight_path"] == str(hls_source_preflight),
        data["_target_"] == "src.data.h1_carrierid_date_lodo_hls.H1CarrierIdDateLodoHlsSourceDataModule",
        model["_target_"] == "src.models.h1_carrierid_date_lodo_hls_module.H1CarrierIdDateLodoHlsLitModule",
        model["net"]["_target_"] == "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        int(model["net"]["carrier_hidden_dim"]) == 32, int(model["net"]["carrier_dim"]) == 4,
        bool(model["net"]["zero_carrier"]) is False,
        int(trainer["max_epochs"]) == int(trainer["min_epochs"]) == 50,
        int(trainer["limit_val_batches"]) == int(trainer["num_sanity_val_steps"]) == 0,
    )
    _need(all(checks), "H-LS composed config differs from the exact H-C h=32/e49 topology")
    callback = raw["callbacks"]["fixed_epoch50"]
    _need(callback["monitor"] is None and int(callback["every_n_epochs"]) == 50
          and int(callback["save_top_k"]) == -1 and callback["save_last"] is False,
          "H-LS config lost the fixed e49 checkpoint callback")
    return cfg, raw


def run(*, data_dir: Path, phase1_preflight: Path, hc_pair_preflight: Path,
        waiting_plan: Path, upstream_aggregate: Path, null_strength_audit: Path,
        outer_date: str, output: Path) -> dict[str, Any]:
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
          "H-LS source preflight requires CUDA_VISIBLE_DEVICES unset")
    _need(str(outer_date) in DATES, "H-LS source preflight outer date is not in the fixed five-date grid")
    _need(not output.exists() and not output.is_symlink() and not os.path.lexists(str(output)),
          "refusing to overwrite H-LS source preflight")
    waiting_path, waiting, waiting_sha = read_immutable_json(
        waiting_plan, schema=WAITING_SCHEMA, status=WAITING_STATUS,
    )
    _need(tuple(waiting.get("future_complete_grid", {}).get("outer_dates", ())) == DATES,
          "H-LS waiting plan does not bind the exact five-date grid")
    aggregate_path, aggregate_sha = require_complete_upstream(upstream_aggregate)
    audit_path, _audit, audit_sha = validate_null_strength_audit(null_strength_audit)
    pair_path, pair, pair_sha = _read_pair(hc_pair_preflight, outer_date=str(outer_date))
    phase1_preflight = Path(phase1_preflight).resolve()
    _need(phase1_preflight.is_file() and stat.S_IMODE(phase1_preflight.stat().st_mode) == 0o444,
          "Phase-1 source preflight must be immutable")
    pair_source = pair.get("source_binding")
    _need(isinstance(pair_source, Mapping)
          and pair_source.get("preflight_path") == str(phase1_preflight),
          "matched H-C pair does not bind the requested Phase-1 source preflight")

    binding = load_phase2_source_binding(
        data_dir=data_dir, phase1_preflight_path=phase1_preflight, outer_date=str(outer_date),
    )
    _need(pair.get("source_binding_sha256") == canonical_sha256(binding.manifest())
          and pair_source == binding.manifest(),
          "H-LS base source binding differs from the matched H-C source binding")
    dataset = H1CarrierIdDateLodoHlsSourceDataset(binding)
    source = hls_source_manifest(binding, dataset)
    _need(source["effective_source_carriers_nonidentity_all"] is True,
          "H-LS source carrier intervention is identity")

    cfg, raw = _compose(outer_date=str(outer_date), phase1_preflight=phase1_preflight,
                        hls_source_preflight=output.resolve())
    torch.manual_seed(42)
    component = hydra.utils.instantiate(cfg.model.net).eval()
    component_hash = state_hash(component.state_dict())
    matched_hc = pair.get("fresh_models", {}).get("h_c")
    _need(isinstance(matched_hc, Mapping)
          and matched_hc.get("component") == "H1CarrierIdSpint"
          and matched_hc.get("fresh_seed") == 42
          and matched_hc.get("carrier_columns_literal_zero_at_init") is True
          and matched_hc.get("initial_state_sha256") == component_hash,
          "fresh H-LS component is not initialization-identical to matched H-C")
    # Record the wrapper hash in the exact domain later persisted by the
    # checkpoint hook.  H1CarrierIdSpint has no lazy parameters, so this CPU
    # construction is the actual fresh pre-update wrapper state.
    torch.manual_seed(42)
    wrapper = hydra.utils.instantiate(cfg.model).eval()
    wrapper_hash = state_hash(wrapper.state_dict())
    body = {
        "schema": SOURCE_PREFLIGHT_SCHEMA, "status": SOURCE_PREFLIGHT_STATUS,
        "outer_date": str(outer_date), "arm": "H-LS",
        "waiting_plan": {"path": str(waiting_path), "sha256": waiting_sha},
        "upstream_aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha},
        "null_strength_audit": {"path": str(audit_path), "sha256": audit_sha},
        "matched_h_c_pair_preflight": {"path": str(pair_path), "sha256": pair_sha},
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
            "equal_to_matched_h_c_initial_state": True,
            "ci64_used": False,
        },
        "composed_config": {
            "canonical_sha256": canonical_sha256(raw), "experiment": EXPERIMENT,
            "resolved_outer_date": str(outer_date), "resolved_hls_preflight_path": str(output.resolve()),
        },
        "scope": {
            "source_recordings_opened": len(binding.source_sessions),
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False, "tmux_started": False,
        },
        "code_sha256": {
            "source_preflight": sha256_file(Path(__file__).resolve()),
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_hls.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_hls_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_spint.py"),
            "experiment": sha256_file(ROOT / "configs/experiment/h1_carrierid_date_lodo_hls_phase2.yaml"),
            "data_config": sha256_file(ROOT / "configs/data/falcon_h1_carrierid_date_lodo_hls.yaml"),
            "model_config": sha256_file(ROOT / "configs/model/falcon_h1_carrierid_date_lodo_hls.yaml"),
            "terminal_callback": sha256_file(ROOT / "configs/callbacks/h1_carrierid_date_lodo_phase2_terminal.yaml"),
        },
    }
    written, digest = write_immutable_json(output, body)
    return {"status": SOURCE_PREFLIGHT_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--phase1-preflight", required=True, type=Path)
    parser.add_argument("--hc-pair-preflight", required=True, type=Path)
    parser.add_argument("--waiting-plan", required=True, type=Path)
    parser.add_argument("--upstream-aggregate", type=Path, default=UPSTREAM_AGGREGATE_DEFAULT)
    parser.add_argument("--null-strength-audit", type=Path, default=NULL_AUDIT_DEFAULT)
    parser.add_argument("--outer-date", required=True, choices=DATES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
                         hc_pair_preflight=args.hc_pair_preflight, waiting_plan=args.waiting_plan,
                         upstream_aggregate=args.upstream_aggregate, null_strength_audit=args.null_strength_audit,
                         outer_date=args.outer_date, output=args.output), sort_keys=True))


if __name__ == "__main__":
    main()
