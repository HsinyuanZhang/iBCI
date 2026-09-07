#!/usr/bin/env python3
"""Leakage-marked query-carrier diagnostic for the sealed H1 CarrierID H-C.

This evaluator intentionally answers a narrow *diagnostic* question: how does
the fixed, already-trained H-C checkpoint respond if its four-dimensional
carrier is fitted from four **query** trials, while scoring the same ordinary
post-support query window set?  This leaks target/query labels by design.

It is therefore hard-coded as ``LEAKAGE_DIAGNOSTIC_ONLY``.  The receipt cannot
be used for model selection, checkpoint selection, hyperparameter selection,
or a paper's main performance table.  It does not train, backpropagate, or
change the sealed checkpoint.  The ordinary support-carrier H-C comparison is
performed with the exact same checkpoint and exact same strict query windows.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import (
    H1_M4_FOLD0_TARGET,
    carrier_sha256,
    fit_frozen_carrier,
    load_target_records,
    validate_target_receipt_binding,
)
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    array_sha256,
    assert_immutable_receipt,
    assert_state_immutable,
    sha256_file,
    state_hash,
    write_immutable_json,
)
from scripts.h1_carrierid_evaluate import (
    _evaluate,
    _instantiate,
    _load_carrierid_checkpoint,
    _validate_carrierid_config,
)


DIAGNOSTIC_SCHEMA = "h1_carrierid_query_oracle_carrier_leakage_diagnostic_v1"
DIAGNOSTIC_STATUS = "LEAKAGE_DIAGNOSTIC_ONLY_NOT_FOR_SELECTION_OR_PAPER_MAIN_RESULT"
ORACLE_FIT_TRIALS = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NormalizedV2ContractError(message)


class _OracleCarrierDataset(Dataset):
    """The standard strict target dataset with only the carrier replaced.

    It delegates every query neural/target/identity lookup to the ordinary
    support-carrier dataset.  Thus a mismatch in query windows, query samples,
    or support identity cannot silently enter the oracle comparison.
    """

    def __init__(
        self,
        base: H1M4EBNormalizedV2StrictTargetDataset,
        oracle_carriers: Mapping[str, np.ndarray],
    ) -> None:
        self.base = base
        self.window_indices = base.window_indices
        self.window_indices_sha256 = base.window_indices_sha256
        self.oracle_carriers = {
            name: np.asarray(oracle_carriers[name], dtype=np.float32)
            for name in H1_M4_FOLD0_TARGET
        }
        for name in H1_M4_FOLD0_TARGET:
            expected = np.asarray(base.support[name].carriers["full"], dtype=np.float32).shape
            value = self.oracle_carriers[name]
            if value.shape != expected or not np.isfinite(value).all():
                raise NormalizedV2ContractError(
                    f"oracle carrier for {name} has shape {value.shape}; expected finite {expected}"
                )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        neural, target, identity, session, _support_carrier = self.base[index]
        return neural, target, identity, session, self.oracle_carriers[str(session)]


def _query_trial_values(record: Any) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in record.trial_values[4 : 4 + ORACLE_FIT_TRIALS])
    if len(values) != ORACLE_FIT_TRIALS or len(set(values)) != ORACLE_FIT_TRIALS:
        raise NormalizedV2ContractError(
            f"{record.session_name}: oracle needs four distinct query TrialNum values after support"
        )
    return values


def _query_fit_audit(
    record: Any,
    dataset: H1M4EBNormalizedV2StrictTargetDataset,
    trial_values: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Expose the exact query-label / query-window overlap of the leakage."""

    fit_bins = np.concatenate(
        [np.asarray(record.blocks_for(value).block_indices, dtype=np.int64).reshape(-1) for value in trial_values]
    )
    unique_fit_bins = np.unique(fit_bins)
    session_windows = [start for session, start in dataset.window_indices if session == record.session_name]
    output_bins = np.asarray([start + 700 - 1 for start in session_windows], dtype=np.int64)
    output_trials = np.asarray(record.trial_num[output_bins], dtype=np.float64)
    return {
        "oracle_fit_trial_values": list(trial_values),
        "oracle_fit_velocity_bins": int(unique_fit_bins.size),
        "oracle_fit_100ms_blocks": int(sum(record.blocks_for(value).rates.shape[0] for value in trial_values)),
        "ordinary_query_windows_same_recording": int(len(session_windows)),
        "ordinary_query_outputs_in_oracle_fit_trials": int(np.isin(output_trials, np.asarray(trial_values)).sum()),
        "ordinary_query_outputs_in_oracle_fit_velocity_bins": int(np.isin(output_bins, unique_fit_bins).sum()),
        "query_label_reuse_present": True,
    }


def _bind_to_sealed_terminal_receipt(
    terminal_receipt_path: Path,
    *,
    full_checkpoint: Path,
    full_config: Path,
) -> dict[str, Any]:
    receipt = assert_immutable_receipt(terminal_receipt_path)
    checkpoint = receipt.get("checkpoints", {}).get("h_c_full", {})
    target = receipt.get("target", {})
    _require(
        checkpoint.get("sha256") == sha256_file(full_checkpoint),
        "oracle full checkpoint does not equal the sealed H-C terminal checkpoint",
    )
    _require(
        checkpoint.get("config_sha256") == sha256_file(full_config),
        "oracle full config does not equal the sealed H-C terminal config",
    )
    query_hash = target.get("strict_query_window_indices_sha256")
    _require(isinstance(query_hash, str) and len(query_hash) == 64, "sealed terminal receipt lacks strict query hash")
    return {
        "path": str(terminal_receipt_path.resolve()),
        "sha256": sha256_file(terminal_receipt_path),
        "status": receipt.get("status"),
        "h_c_checkpoint_sha256": checkpoint["sha256"],
        "h_c_config_sha256": checkpoint["config_sha256"],
        "strict_query_window_indices_sha256": query_hash,
        "support_h_c_pooled_r2_reference": receipt.get("metrics", {}).get("h_c_interventions", {}).get("full", {}).get("pooled_r2"),
    }


def run(
    *,
    data_dir: Path,
    raw_receipt: Path,
    eb_receipt: Path,
    shared_cache_dir: Path,
    full_checkpoint: Path,
    full_config: Path,
    sealed_terminal_receipt: Path,
    output: Path,
    device: str,
    leakage_diagnostic_only: bool,
) -> dict[str, Any]:
    if not leakage_diagnostic_only:
        raise PermissionError("refusing target access without --leakage-diagnostic-only")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable oracle diagnostic: {output}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA oracle diagnostic requested but CUDA is unavailable")
    # No CUDA may be accidentally created by a CPU diagnostic.
    cuda_before = bool(torch.cuda.is_initialized())

    # Bind the immutable H-C terminal state and its source provenance before
    # creating a source cache or opening either target recording.
    terminal_binding = _bind_to_sealed_terminal_receipt(
        sealed_terminal_receipt, full_checkpoint=full_checkpoint, full_config=full_config
    )
    config = _validate_carrierid_config(full_config, "full")
    checkpoint, metadata = _load_carrierid_checkpoint(full_checkpoint, full_config, "full")
    source = H1M4EBNormalizedV2DataModule(
        task="h1",
        data_dir=str(data_dir.resolve()),
        raw_receipt_path=str(raw_receipt.resolve()),
        eb_receipt_path=str(eb_receipt.resolve()),
        cache_dir=str(shared_cache_dir.resolve()),
    )
    source.setup("fit")
    source_manifest = source.pilot_manifest()
    _require(
        metadata.get("source_manifest_sha256") == source.pilot_manifest_sha256,
        "sealed H-C checkpoint does not bind the runtime source manifest",
    )
    _require(
        metadata.get("normalizer_sha256") == source.normalizer.normalizer_sha256,
        "sealed H-C checkpoint does not bind the runtime source normalizer",
    )

    evaluation_device = torch.device(device)
    model = _instantiate(config, checkpoint, evaluation_device)
    del checkpoint

    # This is the intentional target-opening boundary.  No optimizer, model
    # update, backward pass, or selection function exists below this line.
    target_records = load_target_records(data_dir)
    validate_target_receipt_binding(target_records, source.plan, raw_receipt, eb_receipt)
    support_dataset = H1M4EBNormalizedV2StrictTargetDataset(
        target_records, source.plan, source.normalizer, "full"
    )
    _require(
        support_dataset.window_indices_sha256 == terminal_binding["strict_query_window_indices_sha256"],
        "ordinary support-carrier query differs from sealed H-C terminal query",
    )

    oracle_carriers: dict[str, np.ndarray] = {}
    query_fit: dict[str, Any] = {}
    support_identity_binding: dict[str, Any] = {}
    for name in H1_M4_FOLD0_TARGET:
        record = target_records[name]
        values = _query_trial_values(record)
        fit = fit_frozen_carrier(record, source.plan, values)
        normalized = source.normalizer.normalize(fit["carrier"])
        oracle_carriers[name] = normalized
        query_fit[name] = {
            **_query_fit_audit(record, support_dataset, values),
            "oracle_raw_carrier_sha256": carrier_sha256(np.asarray(fit["carrier"], dtype=np.float64)),
            "oracle_normalized_carrier_sha256": carrier_sha256(np.asarray(normalized, dtype=np.float64)),
        }
        support = support_dataset.support[name]
        support_identity_binding[name] = {
            "support_trial_values": list(support.trial_values),
            "support_sha256": support.support_sha256,
            "identity_sha256": array_sha256(np.asarray(support.identity, dtype=np.float32)),
            "ordinary_support_carrier_sha256": support.carrier_sha256["full"],
        }

    oracle_dataset = _OracleCarrierDataset(support_dataset, oracle_carriers)
    _require(len(oracle_dataset) == len(support_dataset), "oracle changed ordinary query sample count")
    _require(
        oracle_dataset.window_indices_sha256 == support_dataset.window_indices_sha256,
        "oracle changed strict query window identity",
    )
    state_before = state_hash(model.state_dict())
    support_metrics = _evaluate(model, support_dataset, evaluation_device, "H-C/support-carrier")
    oracle_metrics = _evaluate(model, oracle_dataset, evaluation_device, "H-C/query-oracle-carrier")
    state_after = state_hash(model.state_dict())
    assert_state_immutable(state_before, state_after, "CarrierID query-oracle diagnostic")
    _require(
        support_metrics["query_window_indices_sha256"] == oracle_metrics["query_window_indices_sha256"],
        "oracle and support H-C evaluated different query windows",
    )
    _require(
        support_metrics["samples"] == oracle_metrics["samples"]
        and support_metrics["session_samples"] == oracle_metrics["session_samples"],
        "oracle and support H-C evaluated different sample counts",
    )
    if device == "cpu":
        _require(bool(torch.cuda.is_initialized()) == cuda_before, "CPU oracle diagnostic initialized CUDA")

    per_recording_delta = {
        name: {
            "support_carrier_r2": support_metrics["per_session"][name]["r2"],
            "query_oracle_carrier_r2": oracle_metrics["per_session"][name]["r2"],
            "oracle_minus_support_r2": (
                oracle_metrics["per_session"][name]["r2"] - support_metrics["per_session"][name]["r2"]
            ),
            "samples": support_metrics["per_session"][name]["samples"],
        }
        for name in H1_M4_FOLD0_TARGET
    }
    receipt = {
        "schema": DIAGNOSTIC_SCHEMA,
        "status": DIAGNOSTIC_STATUS,
        "LEAKAGE_DIAGNOSTIC_ONLY": True,
        "eligibility": {
            "model_selection": False,
            "checkpoint_or_epoch_selection": False,
            "hyperparameter_selection": False,
            "paper_main_result": False,
            "formal_or_evalai_claim": False,
        },
        "interpretation": {
            "question": "diagnostic response of sealed H-C to a query-label-fitted carrier on the same strict query pool",
            "carrier_channel_ceiling_language": "approximate leakage diagnostic only",
            "not_a_strict_upper_bound": "The sealed H-C was trained with ordinary support carriers; this intervention changes only its target-time carrier. No observed delta, including a previously discussed 0.0102 value, is a mathematical upper bound on an honest source-trained feature.",
            "deployment_valid": False,
        },
        "checkpoint_and_query_binding": {
            "same_sealed_h_c_checkpoint_for_support_and_oracle": True,
            "checkpoint_path": str(full_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(full_checkpoint),
            "config_path": str(full_config.resolve()),
            "config_sha256": sha256_file(full_config),
            "sealed_terminal_receipt": terminal_binding,
            "strict_query_window_indices_sha256": support_dataset.window_indices_sha256,
            "same_query_windows": True,
            "same_query_sample_counts": True,
            "same_support_identity": True,
        },
        "source_binding": {
            "source_manifest_sha256": source.pilot_manifest_sha256,
            "normalizer_sha256": source.normalizer.normalizer_sha256,
            "source_manifest": source_manifest,
        },
        "intentional_leakage": {
            "fit_trials_per_recording": ORACLE_FIT_TRIALS,
            "fit_scope": "first four query TrialNum values after the ordinary support four",
            "evaluation_scope": "the unchanged ordinary strict post-support query windows, including fit-query trials",
            "per_recording": query_fit,
        },
        "support_identity_binding": support_identity_binding,
        "metrics": {
            "support_carrier_h_c": support_metrics,
            "query_oracle_carrier_h_c": oracle_metrics,
            "pooled": {
                "support_carrier_r2": support_metrics["pooled_r2"],
                "query_oracle_carrier_r2": oracle_metrics["pooled_r2"],
                "oracle_minus_support_r2": oracle_metrics["pooled_r2"] - support_metrics["pooled_r2"],
                "samples": support_metrics["samples"],
                "r2_accumulator_dtype": "float64",
            },
            "per_recording": per_recording_delta,
        },
        "execution": {
            "device": str(evaluation_device),
            "model_state_immutable": state_before == state_after,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "trainer_launched": False,
            "checkpoint_created_or_selected": False,
            "formal_heldout_opened": False,
            "evalai_opened": False,
        },
    }
    receipt_path, digest = write_immutable_json(output, receipt)
    return {"receipt_path": str(receipt_path), "receipt_sha256": digest, "status": DIAGNOSTIC_STATUS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "000954")
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--eb-receipt", type=Path, required=True)
    parser.add_argument("--shared-cache-dir", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--sealed-terminal-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--leakage-diagnostic-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
