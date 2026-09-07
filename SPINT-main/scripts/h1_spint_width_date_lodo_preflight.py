#!/usr/bin/env python3
"""CPU-only, source-only gate for conditional H1 SPINT width date follow-up.

It is deliberately unusable until the immutable fold-0 routing receipt has
identified eligible predeclared compact arms.  It validates all four remaining
development source schedules and their sealed H-S references without opening a
single outer-date target recording.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random
import stat
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from src.data.h1_spint_width_date_lodo import H1SpintWidthDateLodoSourceDataModule
from src.h1_m4_cce_contract import sha256_file, write_immutable_json
from src.h1_m4_eb_pilot_contract import state_hash
from src.models.components.spint import SpintModel
from src.models.components.spint_identity_width import SpintIdentityWidthModel, identity_accounting
from src.models.h1_spint_width_module import WIDTH_ARMS


FOLD0_SCHEMA = "h1_spint_identity_width_fold0_terminal_evaluation_v1"
DATE_PREFLIGHT_SCHEMA = "h1_spint_identity_width_date_lodo_cpu_preflight_v1"
DATE_PREFLIGHT_STATUS = "PASS_H1_SPINT_IDENTITY_WIDTH_DATE_LODO_SOURCE_ONLY_CPU_PREFLIGHT"
NONINFERIORITY_MARGIN_R2 = 0.03
# The user-authorized conditional expansion is deliberately four dates, not a
# new five-date exploration.  This ordered set is frozen before fold0 results.
FOLLOWUP_DATES = ("19250108", "19250113", "19250115", "19250119")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _seed() -> None:
    random.seed(42); np.random.seed(42); torch.manual_seed(42)


def _base_kwargs() -> dict[str, Any]:
    return {
        "model_dim": 1024, "num_covariates": 7, "window_size": 700, "num_heads": 64,
        "num_layers": 1, "num_id_layers": 3, "use_learnable_id": True, "learnable_id_type": "mlp",
        "learnable_rep": True, "dropout_rate": 0.0, "dynamic_dropout": True,
        "dynamic_dropout_low": 0.0, "dynamic_dropout_high": 1.0, "tf_drop_rate": 0.1,
        "readin_layer_type": "mlp",
    }


def _materialize(net: torch.nn.Module, identity: torch.Tensor) -> None:
    lazy = net.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(identity.permute(0, 1, 3, 2))


def _read_immutable(path: str | Path, *, schema: str) -> tuple[Path, dict[str, Any], str]:
    candidate = Path(path).resolve()
    _need(candidate.is_file() and stat.S_IMODE(candidate.stat().st_mode) == 0o444, f"need immutable 0444 receipt: {candidate}")
    body = json.loads(candidate.read_text(encoding="utf-8"))
    _need(isinstance(body, dict) and body.get("schema") == schema, f"receipt schema mismatch: {candidate}")
    return candidate, body, sha256_file(candidate)


def _eligible(fold0: Mapping[str, Any]) -> tuple[str, ...]:
    _need(tuple(fold0.get("predeclared_arms", ())) == ("H-S-1024", "H-S-W224", "H-S-W32"), "fold0 arm order drift")
    _need(fold0.get("noninferiority_margin_r2") == NONINFERIORITY_MARGIN_R2, "fold0 noninferiority margin drift")
    contrast = fold0.get("contrasts", {})
    compact = contrast.get("compact_noninferiority", {}) if isinstance(contrast, Mapping) else {}
    decision = fold0.get("expansion_decision", {})
    _need(isinstance(compact, Mapping) and isinstance(decision, Mapping), "fold0 compact decision absent")
    eligible = tuple(str(item) for item in decision.get("eligible_compact_arms", ()))
    for arm in ("H-S-W224", "H-S-W32"):
        row = compact.get(arm)
        _need(isinstance(row, Mapping) and isinstance(row.get("delta_r2_vs_hs1024"), (float, int)), f"fold0 {arm} delta absent")
        _need(bool(row.get("within_noninferiority_margin")) == (arm in eligible), f"fold0 {arm} eligibility drift")
        _need(bool(row.get("within_noninferiority_margin")) == (float(row["delta_r2_vs_hs1024"]) >= -NONINFERIORITY_MARGIN_R2),
              f"fold0 {arm} margin arithmetic drift")
    _need(eligible, "no compact arm passed fold0; date expansion is forbidden")
    _need(all(arm in WIDTH_ARMS and arm != "H-S-1024" for arm in eligible), "fold0 selected a noncompact/unregistered arm")
    return eligible


def _reference_receipt_path(date: str) -> Path:
    return ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2/terminal_evaluations" / f"H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_TERMINAL_EVALUATION_v1.json"


def _reference(date: str) -> dict[str, Any]:
    path, body, digest = _read_immutable(path=_reference_receipt_path(date), schema="h1_carrierid_date_lodo_phase2_terminal_evaluation_v1")
    _need(body.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED" and body.get("outer_date") == date,
          f"{date}: H-S reference receipt status/date drift")
    checkpoint = body.get("checkpoints", {}).get("H-S", {})
    meta = checkpoint.get("metadata", {}) if isinstance(checkpoint, Mapping) else {}
    expected = {"schema": "h1_carrierid_date_lodo_phase2_terminal_checkpoint_v1", "arm": "H-S", "outer_date": date,
                "fresh_seed": 42, "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
                "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection", "target_optimizer_steps": 0,
                "target_backward_steps": 0, "checkpoint_warm_start": False}
    _need(all(meta.get(key) == value for key, value in expected.items()), f"{date}: reusable H-S reference checkpoint contract drift")
    metric = body.get("metrics", {}).get("h_s", {})
    target = body.get("target", {}).get("strict_dataset", {})
    _need(isinstance(metric.get("pooled_r2"), (float, int)) and isinstance(target.get("window_indices_sha256"), str),
          f"{date}: H-S reference metric/target binding absent")
    return {"path": str(path), "sha256": digest, "source_manifest_sha256": body.get("source_manifest_sha256"),
            "source_binding_sha256": meta.get("phase2_source_binding_sha256"), "h_s_pooled_r2": float(metric["pooled_r2"]),
            "target_query_window_indices_sha256": target["window_indices_sha256"], "target_support": target.get("support"),
            "target_files": body.get("target", {}).get("files"), "h_s_metadata": dict(meta)}


def run(*, data_dir: str | Path, phase1_preflight: str | Path, fold0_receipt: str | Path, output: str | Path) -> dict[str, Any]:
    _need(not torch.cuda.is_initialized(), "date CPU preflight must start before CUDA initialization")
    fold_path, fold, fold_sha = _read_immutable(fold0_receipt, schema=FOLD0_SCHEMA)
    arms = _eligible(fold)
    records: dict[str, Any] = {}
    # Static equivalence is an additional local proof: the h=1024 form in the
    # compact implementation remains literal current SpintModel, even though
    # the sealed H-S date references are reused rather than retrained.
    identity_probe = torch.randn(1, 4, 1024, 176)
    neural_probe = torch.randn(1, 700, 176)
    _seed(); standard = SpintModel(**_base_kwargs()); _materialize(standard, identity_probe)
    _seed(); scaled = SpintIdentityWidthModel(identity_width=1024, **_base_kwargs()); _materialize(scaled, identity_probe)
    _need(all(torch.equal(value, scaled.state_dict()[key]) for key, value in standard.state_dict().items()),
          "h=1024 date preflight state differs from current standard SPINT")
    standard.eval(); scaled.eval()
    with torch.no_grad():
        _need(torch.equal(standard(neural_probe, calib_trialized_neural_features=identity_probe),
                          scaled(neural_probe, calib_trialized_neural_features=identity_probe)),
              "h=1024 date preflight forward differs from current standard SPINT")
    for date in FOLLOWUP_DATES:
        reference = _reference(date)
        module = H1SpintWidthDateLodoSourceDataModule(
            task="h1", data_dir=str(Path(data_dir).resolve()), phase1_preflight_path=str(Path(phase1_preflight).resolve()), outer_date=date,
        )
        module.setup("fit")
        module.train_batch_sampler.reset_epoch()
        batch = next(iter(module.train_dataloader()))
        _need(len(batch) == 4, f"{date}: date source batch has carrier field")
        neural, target, identity, sessions = batch
        _need(tuple(neural.shape) == (32, 700, 176) and tuple(target.shape) == (32, 700, 7)
              and tuple(identity.shape) == (32, 4, 1024, 176), f"{date}: source batch shape drift")
        source = module.source_manifest()
        _need(source["phase1_source_manifest_sha256"] == reference["source_manifest_sha256"] == module.phase1_manifest_sha256,
              f"{date}: compact run and reusable H-S reference do not share source manifest")
        _need(source["phase2_source_binding_sha256"] == reference["source_binding_sha256"],
              f"{date}: compact run and reusable H-S reference do not share M4 schedule binding")
        _need(source["carrier_path"] == "absent_from_dataset_and_model_inputs" and source["target_recordings_opened_during_training_setup"] is False
              and source["target_bytes_read_during_training_setup"] == 0, f"{date}: source-only carrier/target contract drift")
        arm_rows: dict[str, Any] = {}
        for arm in arms:
            _seed(); net = SpintIdentityWidthModel(identity_width=WIDTH_ARMS[arm], **_base_kwargs()); _materialize(net, identity)
            _need("carrier" not in inspect.signature(net.forward).parameters, f"{date}/{arm}: carrier appears in forward signature")
            before = state_hash(net.state_dict()); net.train(); net.zero_grad(set_to_none=True)
            prediction = net(neural, calib_trialized_neural_features=identity)[:, -1:, :] / 20.0
            loss = torch.mean(torch.square(prediction - target[:, -1:, :])); loss.backward()
            _need(state_hash(net.state_dict()) == before, f"{date}/{arm}: preoptimizer source pass mutated model")
            gradients = [p.grad for name, p in net.named_parameters() if name.startswith(("fc_id_in.", "fc_id_out."))]
            _need(all(grad is not None and torch.isfinite(grad).all() for grad in gradients) and any(torch.count_nonzero(grad).item() for grad in gradients),
                  f"{date}/{arm}: identity gradient absent/nonfinite")
            account = identity_accounting(WIDTH_ARMS[arm])
            arm_rows[arm] = {"identity_width": WIDTH_ARMS[arm], "identity_parameters": account.parameters,
                              "identity_dense_macs_m4_n176": account.dense_macs_m4_n176,
                              "real_source_loss": float(loss.detach()), "state_immutable_before_optimizer": True,
                              "identity_gradient_nonzero": True}
        records[date] = {"source_manifest": source, "source_manifest_sha256": module.source_manifest_sha256,
                         "reused_hs_reference": reference, "first_source_batch": {"field_count": len(batch),
                         "neural_shape": list(neural.shape), "target_shape": list(target.shape), "identity_shape": list(identity.shape),
                         "sessions": list(sessions)}, "compact_arms": arm_rows}
    receipt = {
        "schema": DATE_PREFLIGHT_SCHEMA, "status": DATE_PREFLIGHT_STATUS,
        "fold0_gate": {"path": str(fold_path), "sha256": fold_sha, "eligible_compact_arms": list(arms),
                       "noninferiority_margin_r2": NONINFERIORITY_MARGIN_R2,
                       "rule": "Eligibility is inherited verbatim from the immutable no-selection fold0 receipt."},
        "scope": {"source_only": True, "target_recordings_opened": 0, "target_bytes_read": 0, "minival_opened": False,
                  "formal_or_organizer_data_opened": False, "cuda_initialized": False},
        "protocol": {"dates": list(FOLLOWUP_DATES), "seed": 42, "epochs": 50, "checkpoint_epoch_zero_based": 49,
                     "support_trials": 4, "window_size": 700, "batch_size": 32, "carrier_input": "absent",
                     "h1024_state_and_forward_bit_identical_to_current_standard_spint": True,
                     "reference_policy": "reuse immutable existing H-S date terminal receipts only after source manifest, schedule binding, fixed e49, seed42, and strict query/support bindings match"},
        "date_bindings": records,
        "interpretation_boundary": "A compact-versus-H-S match establishes only redundant identity-encoder capacity or limited extra capacity headroom. It does not test activity-derived identity versus no identity.",
    }
    written, digest = write_immutable_json(output, receipt)
    receipt["receipt_path"], receipt["receipt_sha256"] = str(written), digest
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/000954")
    parser.add_argument("--phase1-preflight", type=Path, default=ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json")
    parser.add_argument("--fold0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(data_dir=args.data_dir, phase1_preflight=args.phase1_preflight, fold0_receipt=args.fold0_receipt, output=args.output)
    print(json.dumps({"status": result["status"], "eligible_compact_arms": result["fold0_gate"]["eligible_compact_arms"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
