#!/usr/bin/env python3
"""Explicit CPU-only source preflight for one fresh H1 CI32/CI64 LODO date.

This is preparatory code, not a launcher.  It opens no target data and, unless
the operator passes ``--run-source-preflight``, does not open an NWB at all.
The resulting receipt binds the five CI arms to one source cache, normalizer,
schedule, seed, and e49 policy.  It deliberately contains no numerical gate
against a sealed development score.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_date_lodo_ci import (
    H1CarrierIdDateLodoCiSourceDataset,
)
from src.data.h1_carrierid_date_lodo_phase2 import (
    H1CarrierIdDateLodoSchedule,
    PHASE2_SOURCE_BINDING_SCHEMA,
    load_phase2_source_binding,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, state_hash, write_immutable_json
from src.models.components.h1_carrierid_ci_spint import H1CarrierIdCiSpint


PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_ci_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_CI_SOURCE_ONLY_NOT_LAUNCHED"
CI_ARMS = ("CI32-FULL", "CI64-FULL", "CI64-C0", "CI64-LS", "CI64-RS")
MODEL_KWARGS = {
    "model_dim": 1024, "num_covariates": 7, "window_size": 700,
    "num_heads": 64, "num_layers": 1, "num_id_layers": 3,
    "use_learnable_id": True, "learnable_id_type": "mlp", "learnable_rep": True,
    "dropout_rate": 0.0, "dynamic_dropout": True, "dynamic_dropout_low": 0.0,
    "dynamic_dropout_high": 1.0, "tf_drop_rate": 0.1, "readin_layer_type": "mlp",
}
ARM_SPECS = {
    "CI32-FULL": {"interface_dim": 32, "intervention": "full", "zero_carrier": False},
    "CI64-FULL": {"interface_dim": 64, "intervention": "full", "zero_carrier": False},
    "CI64-C0": {"interface_dim": 64, "intervention": "c0", "zero_carrier": True},
    "CI64-LS": {"interface_dim": 64, "intervention": "ls", "zero_carrier": False},
    "CI64-RS": {"interface_dim": 64, "intervention": "rs", "zero_carrier": False},
}
CONFIGS = {
    arm: ROOT / "configs/experiment" / f"h1_carrierid_date_lodo_{arm.lower().replace('-', '_')}.yaml"
    for arm in CI_ARMS
}


class CiPreflightError(ValueError):
    """A CI source-only invariant failed before any GPU task could exist."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CiPreflightError(message)


def _arm_model_probe(arm: str, batch: tuple[Any, ...]) -> dict[str, Any]:
    neural, _target, identity, _session, carrier = batch
    spec = ARM_SPECS[arm]
    neural, identity, carrier = neural.float(), identity.float(), carrier.float()
    _need(tuple(neural.shape[1:]) == (700, 176), "CI source neural shape drift")
    _need(tuple(identity.shape[1:]) == (4, 1024, 176), "CI source identity shape drift")
    _need(tuple(carrier.shape[1:]) == (176, 4), "CI source carrier shape drift")
    torch.manual_seed(42)
    model = H1CarrierIdCiSpint(
        carrier_hidden_dim=32, carrier_interface_dim=int(spec["interface_dim"]), carrier_dim=4,
        carrier_trial_length=1024, zero_carrier=bool(spec["zero_carrier"]), **MODEL_KWARGS,
    ).eval()
    with torch.no_grad():
        projection = model.carrierid_identity_projection(identity, carrier)
    _need(tuple(projection.shape) == (neural.shape[0], 176, 700), "CI identity projection shape drift")
    _need(torch.isfinite(projection).all().item(), "CI identity projection is nonfinite")
    carrier_columns = model.carrier_post_pool[0].weight[:, model.carrier_hidden_dim:]
    _need(torch.count_nonzero(carrier_columns).item() == 0, "CI carrier-entry columns must literal-zero initialize")
    return {
        "component": "H1CarrierIdCiSpint",
        "arm": arm,
        "interface_dim": int(spec["interface_dim"]),
        "carrier_intervention": str(spec["intervention"]),
        "zero_carrier_at_model_boundary": bool(spec["zero_carrier"]),
        "fresh_seed": 42,
        "initial_state_sha256": state_hash(model.state_dict()),
        "shared_backbone_initial_state_sha256": model.shared_backbone_state_hash(),
        "carrier_parameters": model.carrier_parameter_count(),
        "whole_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "carrier_columns_literal_zero_at_init": True,
        "identity_projection_shape": list(projection.shape),
    }


def run(*, data_dir: Path, phase1_preflight: Path, outer_date: str, output: Path) -> dict[str, Any]:
    """Run the declared source-only materialisation and publish one receipt."""

    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""), "CI CPU preflight requires CUDA_VISIBLE_DEVICES unset")
    _need(str(outer_date) in CONFIRMATORY_DATES, "CI outer date must be one of canonical five dates")
    _need(not output.exists(), f"CI preflight refuses to overwrite receipt: {output}")
    binding = load_phase2_source_binding(
        data_dir=data_dir, phase1_preflight_path=phase1_preflight, outer_date=outer_date,
    )
    _need(binding.manifest()["schema"] == PHASE2_SOURCE_BINDING_SCHEMA, "CI source binding schema drift")
    datasets = {
        intervention: H1CarrierIdDateLodoCiSourceDataset(binding, carrier_intervention=intervention)
        for intervention in ("full", "c0", "ls", "rs")
    }
    samplers = {key: H1CarrierIdDateLodoSchedule(value, binding) for key, value in datasets.items()}
    full = datasets["full"]
    for key, dataset in datasets.items():
        _need(dataset.window_indices == full.window_indices, f"CI {key} changed source windows")
        _need(np_array_equal(samplers[key].binding.calibration_schedule, samplers["full"].binding.calibration_schedule),
              f"CI {key} changed source schedule")
        _need(samplers[key].binding.batch_order_sha256 == samplers["full"].binding.batch_order_sha256,
              f"CI {key} changed source batch order")
    for key in ("ls", "rs"):
        _need(datasets[key].effective_source_carriers_nonidentity_all is True, f"CI {key} control collapsed")
    batches: dict[str, tuple[Any, ...]] = {}
    for key, dataset in datasets.items():
        request = next(iter(samplers[key]))
        batches[key] = next(iter(torch.utils.data.DataLoader(dataset, batch_sampler=[request], num_workers=0)))
    models = {arm: _arm_model_probe(arm, batches[str(spec["intervention"])]) for arm, spec in ARM_SPECS.items()}
    ci64_states = {models[arm]["initial_state_sha256"] for arm in CI_ARMS if arm.startswith("CI64-")}
    shared_states = {models[arm]["shared_backbone_initial_state_sha256"] for arm in CI_ARMS}
    _need(len(ci64_states) == 1, "CI64 control arms do not share fresh initial state")
    _need(len(shared_states) == 1, "CI32/CI64 do not share decoder-plus-prepool initial backbone")
    source = binding.manifest()
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "mode": "explicit_cpu_source_binding_and_model_materialization_no_trainer_no_gpu_no_target",
        "outer_date": str(outer_date),
        "source_binding": source,
        "source_binding_sha256": canonical_sha256(source),
        "source_controls": {
            "all_arms": list(CI_ARMS), "same_source_windows": True, "same_source_schedule": True,
            "same_source_normalizer": True, "same_fresh_seed": 42,
            "fixed_terminal_epoch_zero_based": 49, "epochs": 50,
            "c0_model_boundary_literal_zero": True,
            "ls_effective_carriers_nonidentity_all": True,
            "rs_effective_carriers_nonidentity_all": True,
        },
        "fresh_models": models,
        "initialization_checks": {
            "all_ci64_controls_full_state_equal": True,
            "ci32_ci64_shared_backbone_state_equal": True,
            "shared_backbone_initial_state_sha256": next(iter(shared_states)),
        },
        "configuration": {arm: {"path": str(path), "sha256": sha256_file(path)} for arm, path in CONFIGS.items()},
        "five_date_aggregate_requirement": {
            "required_before_gpu_launch": True,
            "required_status": "source/date screen complete",
            "automatic_route_selection": "FORBIDDEN",
            "sealed_development_score_numeric_gate": "PROHIBITED",
        },
        "scope": {
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "trainer_constructed_or_launched": False, "checkpoint_created_or_loaded": False,
            "cuda_constructed_or_launched": False,
        },
        "code_sha256": {
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_ci.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_ci_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_ci_spint.py"),
            "preflight": sha256_file(Path(__file__).resolve()),
        },
    }
    written, digest = write_immutable_json(output, receipt)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def np_array_equal(left: Any, right: Any) -> bool:
    """Keep NumPy local to the source path while making the comparison explicit."""

    import numpy as np
    return bool(np.array_equal(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source-preflight", action="store_true",
                        help="required explicit acknowledgement; otherwise this program does nothing")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--phase1-preflight", type=Path,
                        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json")
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_source_preflight:
        raise SystemExit("refusing implicit source/NWB access; pass --run-source-preflight explicitly")
    if args.outer_date is None or args.output is None:
        raise SystemExit("--outer-date and --output are required with --run-source-preflight")
    print(json.dumps(run(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
                         outer_date=args.outer_date, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
