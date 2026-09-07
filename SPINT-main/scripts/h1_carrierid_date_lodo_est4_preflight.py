#!/usr/bin/env python3
"""Explicit CPU-only source preflight for H1-EST4-SLODO.

This program is intentionally not a GPU launcher.  With no flag it refuses to
open anything.  With the explicit source-preflight flag it may open only the
eleven source recordings named by the immutable Phase-1 bundle, after first
binding the completed 5/5 H-S/H-C aggregate and one source-only EST4 route.
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
    FIVE_DATE_AGGREGATE_SCHEMA,
    FIVE_DATE_AGGREGATE_STATUS,
    SOURCE_DATE_SCREEN_COMPLETE,
    _load_bound_frozen_plan,
    _read_immutable_json,
)
from src.data.h1_carrierid_date_lodo_est4 import (
    EST4_ARMS,
    H1CarrierIdDateLodoEst4SourceDataset,
)
from src.data.h1_carrierid_date_lodo_phase2 import H1CarrierIdDateLodoSchedule, load_phase2_source_binding
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, state_hash, write_immutable_json
from src.models.components.h1_carrierid_est4_spint import H1CarrierIdEst4Spint


PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_est4_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_EST4_SOURCE_ONLY_NOT_LAUNCHED"
MODEL_KWARGS = {
    "carrier_hidden_dim": 32, "carrier_dim": 4, "carrier_trial_length": 1024,
    "model_dim": 1024, "num_covariates": 7, "window_size": 700,
    "num_heads": 64, "num_layers": 1, "num_id_layers": 3,
    "use_learnable_id": True, "learnable_id_type": "mlp", "learnable_rep": True,
    "dropout_rate": 0.0, "dynamic_dropout": True, "dynamic_dropout_low": 0.0,
    "dynamic_dropout_high": 1.0, "tf_drop_rate": 0.1, "readin_layer_type": "mlp",
}
ARM_SPECS = {
    "B-C": {"estimator_mode": "baseline", "dataset_arm": "B-C", "zero_carrier": False, "row_shuffle_output": False},
    "B-LS": {"estimator_mode": "baseline", "dataset_arm": "B-LS", "zero_carrier": False, "row_shuffle_output": False},
    "L-C": {"estimator_mode": "learned", "dataset_arm": "L-C", "zero_carrier": False, "row_shuffle_output": False},
    "L-C0": {"estimator_mode": "learned", "dataset_arm": "L-C0", "zero_carrier": True, "row_shuffle_output": False},
    "L-LS": {"estimator_mode": "learned", "dataset_arm": "L-LS", "zero_carrier": False, "row_shuffle_output": False},
    "L-RS": {"estimator_mode": "learned", "dataset_arm": "L-RS", "zero_carrier": False, "row_shuffle_output": True},
}
CONFIGS = {arm: ROOT / "configs/experiment" / f"h1_carrierid_date_lodo_est4_{arm.lower().replace('-', '_')}.yaml" for arm in EST4_ARMS}


class Est4PreflightError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4PreflightError(message)


def _five_date_gate(path: Path) -> tuple[Path, str]:
    aggregate_path, aggregate, digest = _read_immutable_json(
        path, schema=FIVE_DATE_AGGREGATE_SCHEMA, status=FIVE_DATE_AGGREGATE_STATUS,
    )
    route = aggregate.get("route_prerequisite")
    _need(tuple(aggregate.get("required_outer_dates", ())) == CONFIRMATORY_DATES
          and aggregate.get("all_five_date_receipts_present_and_validated") is True
          and isinstance(route, dict) and route.get("status") == SOURCE_DATE_SCREEN_COMPLETE
          and route.get("automatic_route_selection") == "FORBIDDEN",
          "EST4 requires immutable completed non-selecting 5/5 H-S/H-C aggregate")
    return aggregate_path, digest


def _probe(arm: str, batch: tuple[Any, ...], *, frozen_plan_path: Path, normalizer: float) -> dict[str, Any]:
    neural, _target, identity, _session, carrier, rates, labels, mask, permutation = batch
    spec = ARM_SPECS[arm]
    torch.manual_seed(42)
    kwargs: dict[str, Any] = {**MODEL_KWARGS, "estimator_mode": spec["estimator_mode"],
                              "zero_carrier": bool(spec["zero_carrier"]),
                              "row_shuffle_output": bool(spec["row_shuffle_output"])}
    if spec["estimator_mode"] == "learned":
        kwargs.update(frozen_plan_path=str(frozen_plan_path), source_normalizer=float(normalizer))
    model = H1CarrierIdEst4Spint(**kwargs).eval()
    with torch.no_grad():
        if model.estimator is not None:
            projected = model.estimator(rates.float(), labels.float(), mask.bool())
        else:
            projected = carrier.float()
    _need(tuple(projected.shape[1:]) == (176, 4) and torch.isfinite(projected).all().item(),
          "EST4 analytic carrier probe shape/finiteness drift")
    return {
        "arm": arm, "fresh_seed": 42, "estimator_mode": spec["estimator_mode"],
        "model_boundary_c0": bool(spec["zero_carrier"]), "output_row_shuffle": bool(spec["row_shuffle_output"]),
        "initial_state_sha256": state_hash(model.state_dict()),
        "shared_backbone_initial_state_sha256": model.shared_backbone_state_hash(),
        "carrierid_consumer_parameters": model.carrier_parameter_count(),
        "estimator_added_learned_parameters": model.estimator_parameter_count(),
        "combined_identity_path_parameters": model.est4_parameter_accounting()["combined_identity_path_parameters"],
        "analytic_carrier_shape": list(projected.shape),
        "deployment_target_optimizer_steps": 0, "deployment_target_backward_steps": 0,
    }


def run(*, data_dir: Path, phase1_preflight: Path, five_date_aggregate: Path, outer_date: str,
        frozen_plan_path: Path, output: Path) -> dict[str, Any]:
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""), "EST4 CPU preflight requires CUDA_VISIBLE_DEVICES unset")
    _need(str(outer_date) in CONFIRMATORY_DATES and not output.exists(), "invalid EST4 date/output")
    aggregate_path, aggregate_sha = _five_date_gate(five_date_aggregate)
    binding = load_phase2_source_binding(data_dir=data_dir, phase1_preflight_path=phase1_preflight, outer_date=outer_date)
    expected_plan = binding.source_manifest_path.parent / "frozen_m4_plan.npz"
    _need(frozen_plan_path.resolve() == expected_plan.resolve() and expected_plan.is_file(),
          "EST4 frozen plan must be the exact Phase-1 source-bundle plan")
    _load_bound_frozen_plan(binding)  # validates arrays/manifests against the Phase-1 source binding.
    datasets = {arm: H1CarrierIdDateLodoEst4SourceDataset(binding, est4_arm=arm) for arm in EST4_ARMS}
    samplers = {arm: H1CarrierIdDateLodoSchedule(dataset, binding) for arm, dataset in datasets.items()}
    reference = datasets["B-C"]
    for arm, dataset in datasets.items():
        _need(dataset.window_indices == reference.window_indices
              and samplers[arm].binding.batch_order_sha256 == samplers["B-C"].binding.batch_order_sha256,
              f"EST4 {arm} changed matched H-S/H-C source schedule")
    batches = {
        arm: next(iter(torch.utils.data.DataLoader(dataset, batch_sampler=[next(iter(samplers[arm]))], num_workers=0)))
        for arm, dataset in datasets.items()
    }
    models = {arm: _probe(arm, batches[arm], frozen_plan_path=expected_plan,
                          normalizer=float(binding.normalizer.denominator)) for arm in EST4_ARMS}
    _need(len({models[arm]["initial_state_sha256"] for arm in ("B-C", "B-LS")}) == 1,
          "B-C/B-LS must share one fresh initial state")
    _need(len({models[arm]["initial_state_sha256"] for arm in ("L-C", "L-C0", "L-LS", "L-RS")}) == 1,
          "L controls must share one fresh initial state")
    _need(len({models[arm]["shared_backbone_initial_state_sha256"] for arm in EST4_ARMS}) == 1,
          "EST4 B/L arms diverged from matched H-S/H-C consumer backbone")
    source = binding.manifest()
    receipt = {
        "schema": PREFLIGHT_SCHEMA, "status": PREFLIGHT_STATUS,
        "mode": "explicit_cpu_source_binding_and_estimator_materialization_no_trainer_no_gpu_no_target",
        "outer_date": str(outer_date), "source_binding": source, "source_binding_sha256": canonical_sha256(source),
        "five_date_aggregate": {"path": str(aggregate_path), "sha256": aggregate_sha,
                                "all_five_date_receipts_present_and_validated": True,
                                "automatic_route_selection": "FORBIDDEN"},
        "frozen_estimator_initialization": {"path": str(expected_plan), "sha256": sha256_file(expected_plan),
                                              "q": 16, "carrier_dim": 4, "normalizer": float(binding.normalizer.denominator)},
        "source_controls": {"all_arms": list(EST4_ARMS), "same_hs_hc_source_partition": True,
                            "same_source_windows": True, "same_source_schedule": True,
                            "same_source_normalizer": True, "same_fresh_seed": 42,
                            "fixed_terminal_epoch_zero_based": 49, "epochs": 50,
                            "target_score_selection": "FORBIDDEN", "warm_start": "FORBIDDEN"},
        "fresh_models": models,
        "initialization_checks": {"b_c_b_ls_full_state_equal": True,
                                    "learned_controls_full_state_equal": True,
                                    "all_arms_shared_consumer_backbone_equal": True},
        "configuration": {arm: {"path": str(path), "sha256": sha256_file(path)} for arm, path in CONFIGS.items()},
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "trainer_constructed_or_launched": False, "checkpoint_created_or_loaded": False,
                  "cuda_constructed_or_launched": False},
        "code_sha256": {"data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_est4.py"),
                        "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_est4_module.py"),
                        "component": sha256_file(ROOT / "src/models/components/h1_carrierid_est4_spint.py"),
                        "preflight": sha256_file(Path(__file__).resolve())},
    }
    written, digest = write_immutable_json(output, receipt)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source-preflight", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--phase1-preflight", type=Path, default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json")
    parser.add_argument("--five-date-aggregate", type=Path, required=False)
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES)
    parser.add_argument("--frozen-plan", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_source_preflight:
        raise SystemExit("refusing implicit source/NWB access; pass --run-source-preflight explicitly")
    if None in (args.five_date_aggregate, args.outer_date, args.frozen_plan, args.output):
        raise SystemExit("--five-date-aggregate, --outer-date, --frozen-plan and --output are required")
    print(json.dumps(run(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
                         five_date_aggregate=args.five_date_aggregate, outer_date=args.outer_date,
                         frozen_plan_path=args.frozen_plan, output=args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

