#!/usr/bin/env python3
"""CPU-only preflight for the minimal fresh H1-EST4 B-C/L-C pair.

This is intentionally *not* a shortened version of the historical six-arm
attribution protocol.  It binds only a fresh ordinary-carrier B-C reference
and a learned-estimator L-C arm to the exact same already-frozen Phase-1
source roster, windows, M=4 schedule, normalizer, seed, and common CarrierID
consumer initialization.  The five-date H-S/H-C aggregate remains a hard
precondition.  The program has no Trainer, checkpoint, CUDA, target, minival,
formal-test, or EvalAI path.

The resulting receipt is training provenance only.  It deliberately records
no source-training loss as a held-out result: a later outer-date source-LODO
score must be produced by its separately declared evaluator.
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

from scripts.h1_carrierid_date_lodo_est4_preflight import (
    ARM_SPECS,
    CONFIGS,
    MODEL_KWARGS,
    _five_date_gate,
    _probe,
)
from src.data.h1_carrierid_date_lodo_ci import _load_bound_frozen_plan
from src.data.h1_carrierid_date_lodo_est4 import (
    EST4_PAIR_ARMS,
    EST4_PAIR_PREFLIGHT_SCHEMA,
    EST4_PAIR_PREFLIGHT_STATUS,
    H1CarrierIdDateLodoEst4SourceDataset,
)
from src.data.h1_carrierid_date_lodo_phase2 import H1CarrierIdDateLodoSchedule, load_phase2_source_binding
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, sha256_file, write_immutable_json


PREFLIGHT_SCHEMA = EST4_PAIR_PREFLIGHT_SCHEMA
PREFLIGHT_STATUS = EST4_PAIR_PREFLIGHT_STATUS
PAIR_ARMS = EST4_PAIR_ARMS


class Est4PairPreflightError(ValueError):
    """The minimal B-C/L-C source-pair contract is incomplete."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Est4PairPreflightError(message)


def run(
    *, data_dir: Path, phase1_preflight: Path, five_date_aggregate: Path,
    outer_date: str, frozen_plan_path: Path, output: Path,
) -> dict[str, Any]:
    """Materialize one exact source-only paired preflight.

    ``outer_date`` has no default on purpose.  Calling code must name it only
    after the upstream aggregate exists; this function never selects a date
    from measurements.
    """

    _need(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""),
          "EST4 pair preflight requires CUDA_VISIBLE_DEVICES unset/empty")
    _need(str(outer_date) in CONFIRMATORY_DATES, "outer date is not confirmatory")
    _need(not output.exists() and not os.path.lexists(str(output)), "refusing to overwrite EST4 pair preflight")
    aggregate_path, aggregate_sha = _five_date_gate(five_date_aggregate)
    binding = load_phase2_source_binding(
        data_dir=data_dir, phase1_preflight_path=phase1_preflight, outer_date=str(outer_date),
    )
    expected_plan = binding.source_manifest_path.parent / "frozen_m4_plan.npz"
    _need(frozen_plan_path.resolve() == expected_plan.resolve() and expected_plan.is_file(),
          "EST4 pair frozen plan must be the exact Phase-1 source-bundle plan")
    _load_bound_frozen_plan(binding)

    datasets = {
        arm: H1CarrierIdDateLodoEst4SourceDataset(binding, est4_arm=arm)
        for arm in PAIR_ARMS
    }
    samplers = {arm: H1CarrierIdDateLodoSchedule(dataset, binding) for arm, dataset in datasets.items()}
    reference = datasets["B-C"]
    for arm, dataset in datasets.items():
        _need(dataset.window_indices == reference.window_indices
              and samplers[arm].binding.batch_order_sha256 == samplers["B-C"].binding.batch_order_sha256,
              f"EST4 pair {arm} altered the frozen source schedule")
    batches = {
        arm: next(iter(torch.utils.data.DataLoader(
            dataset, batch_sampler=[next(iter(samplers[arm]))], num_workers=0,
        )))
        for arm, dataset in datasets.items()
    }
    models = {
        arm: _probe(arm, batches[arm], frozen_plan_path=expected_plan,
                    normalizer=float(binding.normalizer.denominator))
        for arm in PAIR_ARMS
    }
    _need(models["B-C"]["shared_backbone_initial_state_sha256"] ==
          models["L-C"]["shared_backbone_initial_state_sha256"],
          "B-C/L-C diverged from a common fresh CarrierID consumer backbone")
    _need(models["B-C"]["estimator_mode"] == "baseline" and models["L-C"]["estimator_mode"] == "learned",
          "B-C/L-C estimator-mode contract drift")

    source = binding.manifest()
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "mode": "explicit_cpu_source_binding_and_pair_estimator_materialization_no_trainer_no_gpu_no_target",
        "outer_date": str(outer_date),
        "source_binding": source,
        "source_binding_sha256": canonical_sha256(source),
        "five_date_aggregate": {
            "path": str(aggregate_path), "sha256": aggregate_sha,
            "all_five_date_receipts_present_and_validated": True,
            "automatic_route_selection": "FORBIDDEN",
        },
        "frozen_estimator_initialization": {
            "path": str(expected_plan), "sha256": sha256_file(expected_plan),
            "q": 16, "carrier_dim": 4, "normalizer": float(binding.normalizer.denominator),
        },
        "source_controls": {
            "all_arms": list(PAIR_ARMS),
            "same_hs_hc_source_partition": True,
            "same_source_windows": True,
            "same_source_schedule": True,
            "same_source_normalizer": True,
            "same_fresh_seed": 42,
            "fixed_terminal_epoch_zero_based": 49,
            "epochs": 50,
            "target_score_selection": "FORBIDDEN",
            "warm_start": "FORBIDDEN",
            "source_training_loss_is_heldout_evidence": False,
            "future_prediction_endpoint": "separately declared outer-date source-LODO evaluator; target is closed here",
        },
        "fresh_models": models,
        "initialization_checks": {
            "b_c_l_c_common_fresh_consumer_backbone_equal": True,
            "full_model_state_equality_not_expected_because_l_c_adds_estimator": True,
        },
        "configuration": {
            arm: {"path": str(CONFIGS[arm]), "sha256": sha256_file(CONFIGS[arm])}
            for arm in PAIR_ARMS
        },
        "scope": {
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "minival_opened": 0, "formal_test_opened": 0, "evalai_opened": 0,
            "trainer_constructed_or_launched": False, "checkpoint_created_or_loaded": False,
            "cuda_constructed_or_launched": False,
        },
        "code_sha256": {
            "data": sha256_file(ROOT / "src/data/h1_carrierid_date_lodo_est4.py"),
            "model": sha256_file(ROOT / "src/models/h1_carrierid_date_lodo_est4_module.py"),
            "component": sha256_file(ROOT / "src/models/components/h1_carrierid_est4_spint.py"),
            "preflight": sha256_file(Path(__file__).resolve()),
        },
    }
    written, digest = write_immutable_json(output, receipt)
    return {"status": PREFLIGHT_STATUS, "receipt_path": str(written), "receipt_sha256": digest}


def main() -> None:
    # Check this before argparse's required-path validation so an accidental
    # bare invocation cannot be mistaken for permission to enumerate source
    # recordings merely because it omitted other arguments.
    if "--run-source-preflight" not in sys.argv:
        raise SystemExit("refusing implicit source/NWB access; pass --run-source-preflight explicitly")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source-preflight", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--phase1-preflight", type=Path,
                        default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json")
    parser.add_argument("--five-date-aggregate", type=Path, required=True)
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        data_dir=args.data_dir, phase1_preflight=args.phase1_preflight,
        five_date_aggregate=args.five_date_aggregate, outer_date=args.outer_date,
        frozen_plan_path=args.frozen_plan, output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
