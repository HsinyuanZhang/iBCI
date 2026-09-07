#!/usr/bin/env python3
"""Opened-development carrier-corruption dose response on sealed fold-0 H-C.

This diagnostic never trains a model.  It reuses the sealed epoch-49 H-C
checkpoint and the exact strict fold-0 query windows, then changes only the
carrier supplied at the model boundary.  Its evidence is therefore limited to
same-checkpoint sensitivity; it cannot establish that carrier content was
necessary during source training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_dose_response import (
    CarrierOverrideTargetDataset,
    DOSE_GRID,
    REPEAT_SEEDS,
    partial_label_carrier,
    partial_row_corruption,
)
from src.data.h1_m4_eb_normalized_v2 import (
    H1M4EBNormalizedV2DataModule,
    H1M4EBNormalizedV2StrictTargetDataset,
)
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET
from src.h1_m4_eb_normalized_v2_contract import (
    NormalizedV2ContractError,
    assert_immutable_receipt,
    immutable_mode_0444,
    sha256_file,
    write_immutable_json,
)
from scripts.h1_carrierid_evaluate import (
    TERMINAL_SCHEMA as H_C_GATE_SCHEMA,
    _evaluate,
    _load_carrierid_checkpoint,
    _validate_carrierid_config,
)


ARTIFACT_DIR = ROOT / "pilot_artifacts/h1_carrierid_dose_response"
PREFLIGHT = ARTIFACT_DIR / "H1_CARRIERID_HC_FOLD0_SAME_CHECKPOINT_DOSE_RESPONSE_PREFLIGHT_v2.json"
RUNTIME_P0_GATE = ARTIFACT_DIR / "H1_CARRIERID_HC_FOLD0_RUNTIME_P0_REPRODUCTION_GATE_v2.json"
RESULT = ARTIFACT_DIR / "H1_CARRIERID_HC_FOLD0_SAME_CHECKPOINT_DOSE_RESPONSE_RESULT_v2.json"
SHARD_DIR = ARTIFACT_DIR / "shards_v2"
SUPERSEDED_V1_PREFLIGHT = ARTIFACT_DIR / "H1_CARRIERID_HC_FOLD0_SAME_CHECKPOINT_DOSE_RESPONSE_PREFLIGHT_v1.json"
SUPERSEDED_V1_SHARDS = ARTIFACT_DIR / "shards_v1"
H_C_GATE = (
    ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/"
    "H1_CARRIERID_H32_FOLD0_TERMINAL_GATE_FLOAT64_R2.json"
)
H_C_CHECKPOINT = (
    ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/"
    "checkpoints/fixed_epoch50/epoch_049.ckpt"
)
H_C_CONFIG = ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/.hydra/config.yaml"

PREFLIGHT_SCHEMA = "h1_carrierid_hc_fold0_same_checkpoint_dose_response_preflight_v2"
PREFLIGHT_STATUS = "READY_H1_OPENED_DEVELOPMENT_RUNTIME_P0_GATED_SAME_CHECKPOINT_DOSE_RESPONSE"
RUNTIME_P0_SCHEMA = "h1_carrierid_hc_fold0_runtime_p0_reproduction_gate_v2"
RUNTIME_P0_PASS = "PASS_H1_RUNTIME_P0_REPRODUCES_SEALED_HC_NONZERO_DOSES_AUTHORIZED"
RUNTIME_P0_FAIL = "STOP_H1_RUNTIME_P0_REPRODUCTION_FAILED_NONZERO_DOSES_FORBIDDEN"
SHARD_SCHEMA = "h1_carrierid_hc_fold0_same_checkpoint_dose_response_shard_v2"
SHARD_STATUS = "COMPLETE_H1_OPENED_DEVELOPMENT_SAME_CHECKPOINT_DOSE_SHARD"
RESULT_SCHEMA = "h1_carrierid_hc_fold0_same_checkpoint_dose_response_result_v2"
H_C_GATE_STATUS = "PASS_H1_CARRIERID_H32_EXPANSION_AUTHORIZED"
EXPECTED_QUERY_HASH = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
CORRUPTION_KINDS = ("label", "row")
P0_R2_ABSOLUTE_TOLERANCE = 1.0e-6
CODE_PATHS = (
    Path(__file__).resolve(),
    ROOT / "src/data/h1_carrierid_dose_response.py",
    ROOT / "src/data/h1_m4_eb_normalized_v2.py",
    ROOT / "src/data/h1_m4_eb_pilot.py",
    ROOT / "scripts/h1_carrierid_evaluate.py",
    ROOT / "src/models/components/h1_carrierid_spint.py",
)


def _gate_and_full_metric() -> tuple[dict[str, Any], Mapping[str, Any]]:
    gate = assert_immutable_receipt(H_C_GATE, H_C_GATE_STATUS)
    if gate.get("schema") != H_C_GATE_SCHEMA:
        raise NormalizedV2ContractError("sealed H-C terminal schema drift")
    checkpoint = gate.get("checkpoints", {}).get("h_c_full", {})
    if Path(str(checkpoint.get("path", ""))).resolve() != H_C_CHECKPOINT.resolve():
        raise NormalizedV2ContractError("sealed gate points to another H-C checkpoint")
    if checkpoint.get("sha256") != sha256_file(H_C_CHECKPOINT):
        raise NormalizedV2ContractError("sealed H-C checkpoint bytes drifted")
    if checkpoint.get("config_sha256") != sha256_file(H_C_CONFIG):
        raise NormalizedV2ContractError("sealed H-C config bytes drifted")
    target = gate.get("target", {})
    if tuple(target.get("sessions", ())) != tuple(H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError("sealed H-C target recording set drifted")
    if target.get("strict_query_window_indices_sha256") != EXPECTED_QUERY_HASH:
        raise NormalizedV2ContractError("sealed H-C query-window hash drifted")
    full = gate.get("metrics", {}).get("h_c_interventions", {}).get("full", {})
    if (
        full.get("query_window_indices_sha256") != EXPECTED_QUERY_HASH
        or full.get("r2_accumulator_dtype") != "float64"
        or full.get("state_immutable") is not True
        or full.get("state_sha256_before") != full.get("state_sha256_after")
    ):
        raise NormalizedV2ContractError("sealed H-C p=0 metric contract drifted")
    if set(full.get("per_session", {})) != set(H1_M4_FOLD0_TARGET):
        raise NormalizedV2ContractError("sealed H-C p=0 metric omits a recording")
    return gate, full


def _code_hashes() -> dict[str, str]:
    missing = [str(path) for path in CODE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dose-response code closure is incomplete: {missing}")
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in CODE_PATHS}


def write_preflight() -> dict[str, Any]:
    """Bind the diagnostic before any target NWB is opened."""

    gate, full = _gate_and_full_metric()
    # Load/validate source-side checkpoint metadata here, before the target
    # execution boundary, so the preflight binds more than a filename/hash.
    cfg = _validate_carrierid_config(H_C_CONFIG, "full")
    _checkpoint, meta = _load_carrierid_checkpoint(H_C_CHECKPOINT, H_C_CONFIG, "full")
    del _checkpoint
    target = gate["target"]
    superseded_shards = []
    if SUPERSEDED_V1_SHARDS.is_dir():
        superseded_shards = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(SUPERSEDED_V1_SHARDS.glob("*.json"))
        ]
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "created_for": "opened fold0 development diagnostic only",
        "supersedes": {
            "v1_preflight": {
                "path": str(SUPERSEDED_V1_PREFLIGHT),
                "sha256": sha256_file(SUPERSEDED_V1_PREFLIGHT),
            },
            "reason": "v1 did not require an actual current-runtime p=0 forward before nonzero-dose execution",
            "historical_v1_nonzero_shards": superseded_shards,
            "v1_shards_admitted_to_v2_final": False,
            "policy": "preserve v1 bytes, but rerun every included dose under v2 after the p=0 gate",
        },
        "checkpoint": {
            "path": str(H_C_CHECKPOINT),
            "sha256": sha256_file(H_C_CHECKPOINT),
            "config_path": str(H_C_CONFIG),
            "config_sha256": sha256_file(H_C_CONFIG),
            "epoch_zero_based": 49,
            "frozen_eval_only": True,
            "metadata": meta,
        },
        "sealed_terminal_gate": {"path": str(H_C_GATE), "sha256": sha256_file(H_C_GATE)},
        "source_contract": {
            "manifest_sha256": meta["source_manifest_sha256"],
            "normalizer_sha256": meta["normalizer_sha256"],
            "normalizer_scalar": float(meta["s_src"]),
            "calibration_n_trials": int(cfg.data.calibration_n_trials),
        },
        "target_contract_copied_without_opening_target": {
            "sessions": list(H1_M4_FOLD0_TARGET),
            "file_sha256": dict(target["files"]),
            "strict_query_window_indices_sha256": EXPECTED_QUERY_HASH,
            "samples": int(full["samples"]),
            "session_samples": dict(full["session_samples"]),
            "target_opened_during_preflight": False,
        },
        "frozen_design": {
            "corruption_kinds": list(CORRUPTION_KINDS),
            "nominal_p_grid": list(DOSE_GRID),
            "repeat_seeds": list(REPEAT_SEEDS),
            "logical_repeats_at_p0": len(REPEAT_SEEDS),
            "p0_metric_source": "mandatory current-runtime forward on the unmodified strict target dataset",
            "p0_must_complete_before_nonzero_forward": True,
            "p0_reproduction_absolute_r2_tolerance": P0_R2_ABSOLUTE_TOLERANCE,
            "p0_exact_checks": [
                "checkpoint hash", "carrier hashes", "query hash", "sample counts",
                "per-recording set", "state immutability",
            ],
            "p0_tolerance_checks": ["pooled R2", "each per-recording R2"],
            "fixed_count_rule": "floor(p*N+0.5)",
            "dose_wording": "p is nominal; each session records actual_selected_fraction after integer rounding",
            "nested_selection": "one deterministic row ranking per kind/session/repeat; p selects its prefix",
            "assignment": "deterministic cyclic derangement within selected rows",
            "row_corruption": "reassign selected normalized carrier rows across neural channels; preserve selected row multiset",
            "label_corruption": "reassign selected velocity-label rows across fixed support blocks; keep rates fixed; refit frozen EB carrier",
            "aggregation": "float64 pooled and per-recording R2; mean/SD/min/max over four fixed repeats; delta relative to p=0",
            "monotonicity": "nonincreasing mean curve and per-repeat adjacent-step audit, with zero numerical tolerance",
        },
        "claim_boundary": {
            "permitted": "same-checkpoint forward sensitivity to increasing carrier corruption on opened fold0 development data",
            "forbidden": [
                "training-time necessity",
                "source-training causality",
                "formal held-out or EvalAI evidence",
                "model selection or routing of EST4/CI64",
            ],
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
        "code_sha256": _code_hashes(),
        "data_scope": {
            "preflight_opened": "checkpoint/config and immutable existing receipts only",
            "public_fold0_target_opened": False,
            "formal_hidden_opened": False,
            "evalai_opened": False,
        },
    }
    path, digest = write_immutable_json(PREFLIGHT, body)
    return {"status": PREFLIGHT_STATUS, "receipt": str(path), "sha256": digest}


def _require_preflight() -> dict[str, Any]:
    receipt = assert_immutable_receipt(PREFLIGHT, PREFLIGHT_STATUS)
    if receipt.get("schema") != PREFLIGHT_SCHEMA:
        raise NormalizedV2ContractError("dose-response preflight schema drift")
    if receipt.get("code_sha256") != _code_hashes():
        raise NormalizedV2ContractError("dose-response code changed after immutable preflight")
    design = receipt.get("frozen_design", {})
    if (
        tuple(design.get("nominal_p_grid", ())) != DOSE_GRID
        or tuple(design.get("repeat_seeds", ())) != REPEAT_SEEDS
        or tuple(design.get("corruption_kinds", ())) != CORRUPTION_KINDS
    ):
        raise NormalizedV2ContractError("dose-response design differs from immutable preflight")
    gate, _full = _gate_and_full_metric()
    if receipt["sealed_terminal_gate"]["sha256"] != sha256_file(H_C_GATE):
        raise NormalizedV2ContractError("sealed terminal receipt changed after preflight")
    if receipt["target_contract_copied_without_opening_target"]["file_sha256"] != gate["target"]["files"]:
        raise NormalizedV2ContractError("preflight target-file binding differs from sealed gate")
    return receipt


def _r2_reproduction(runtime: Mapping[str, Any], sealed: Mapping[str, Any]) -> dict[str, Any]:
    pooled_delta = float(runtime["pooled_r2"]) - float(sealed["pooled_r2"])
    per_recording_delta = {
        name: float(runtime["per_session"][name]["r2"]) - float(sealed["per_session"][name]["r2"])
        for name in H1_M4_FOLD0_TARGET
    }
    return {
        "absolute_tolerance": P0_R2_ABSOLUTE_TOLERANCE,
        "pooled_delta": pooled_delta,
        "pooled_within_tolerance": abs(pooled_delta) <= P0_R2_ABSOLUTE_TOLERANCE,
        "per_recording_delta": per_recording_delta,
        "per_recording_within_tolerance": {
            name: abs(delta) <= P0_R2_ABSOLUTE_TOLERANCE
            for name, delta in per_recording_delta.items()
        },
    }


def execute_runtime_p0_gate(*, device: str, torch_threads: int) -> dict[str, Any]:
    """Run and seal the mandatory current-runtime p=0 comparability gate."""

    if RUNTIME_P0_GATE.exists():
        raise FileExistsError(f"runtime p=0 gate already exists: {RUNTIME_P0_GATE}")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(int(torch_threads))
    preflight, _cfg, _source, model, base, evaluation_device, observed_files = _prepare_opened_development(device)
    gate, sealed = _gate_and_full_metric()
    runtime_carrier_hashes = {
        name: base.support[name].carrier_sha256["full"] for name in H1_M4_FOLD0_TARGET
    }
    sealed_carrier_hashes = {
        name: gate["target"]["support_and_carrier_hashes"][name]["carrier_sha256"]["full"]
        for name in H1_M4_FOLD0_TARGET
    }
    exact = {
        "target_files": observed_files == gate["target"]["files"],
        "query_window_indices_sha256": base.window_indices_sha256 == EXPECTED_QUERY_HASH,
        "carrier_sha256": runtime_carrier_hashes == sealed_carrier_hashes,
        "samples": len(base) == int(sealed["samples"]),
        "session_set": set(base.records) == set(sealed["per_session"]),
    }
    started = time.monotonic()
    runtime = _evaluate(model, base, evaluation_device, "H-C/runtime-p0-reproduction")
    elapsed = time.monotonic() - started
    exact.update(
        {
            "runtime_query_hash": runtime["query_window_indices_sha256"] == EXPECTED_QUERY_HASH,
            "runtime_samples": int(runtime["samples"]) == int(sealed["samples"]),
            "runtime_session_samples": runtime["session_samples"] == sealed["session_samples"],
            "runtime_state_immutable": runtime["state_immutable"] is True
            and runtime["state_sha256_before"] == runtime["state_sha256_after"]
            and runtime["state_sha256_before"] == sealed["state_sha256_before"],
        }
    )
    reproduction = _r2_reproduction(runtime, sealed)
    passed = (
        all(exact.values())
        and reproduction["pooled_within_tolerance"]
        and all(reproduction["per_recording_within_tolerance"].values())
    )
    body = {
        "schema": RUNTIME_P0_SCHEMA,
        "status": RUNTIME_P0_PASS if passed else RUNTIME_P0_FAIL,
        "preflight": {"path": str(PREFLIGHT), "sha256": sha256_file(PREFLIGHT)},
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "device": str(evaluation_device),
        "exact_checks": exact,
        "runtime_carrier_sha256": runtime_carrier_hashes,
        "sealed_carrier_sha256": sealed_carrier_hashes,
        "runtime_metrics": runtime,
        "sealed_metrics": sealed,
        "r2_reproduction": reproduction,
        "elapsed_seconds": elapsed,
        "nonzero_dose_authorized": passed,
        "target_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "claim_boundary": "runtime comparability gate only; not a new model result",
    }
    path, digest = write_immutable_json(RUNTIME_P0_GATE, body)
    return {"status": body["status"], "receipt": str(path), "sha256": digest, "r2_reproduction": reproduction}


def _require_runtime_p0_gate(preflight: Mapping[str, Any]) -> dict[str, Any]:
    receipt = assert_immutable_receipt(RUNTIME_P0_GATE, RUNTIME_P0_PASS)
    if receipt.get("schema") != RUNTIME_P0_SCHEMA:
        raise NormalizedV2ContractError("runtime p=0 gate schema drift")
    if receipt.get("preflight") != {"path": str(PREFLIGHT), "sha256": sha256_file(PREFLIGHT)}:
        raise NormalizedV2ContractError("runtime p=0 gate does not bind current v2 preflight")
    if receipt.get("checkpoint_sha256") != preflight["checkpoint"]["sha256"]:
        raise NormalizedV2ContractError("runtime p=0 gate binds another checkpoint")
    if receipt.get("nonzero_dose_authorized") is not True or not all(receipt.get("exact_checks", {}).values()):
        raise NormalizedV2ContractError("runtime p=0 gate did not authorize nonzero doses")
    reproduction = receipt.get("r2_reproduction", {})
    if (
        reproduction.get("absolute_tolerance") != P0_R2_ABSOLUTE_TOLERANCE
        or reproduction.get("pooled_within_tolerance") is not True
        or not all(reproduction.get("per_recording_within_tolerance", {}).values())
    ):
        raise NormalizedV2ContractError("runtime p=0 R2 reproduction is outside frozen tolerance")
    return receipt


def _source_datamodule(cfg: Any) -> H1M4EBNormalizedV2DataModule:
    data = cfg.data
    source = H1M4EBNormalizedV2DataModule(
        task=str(data.task),
        data_dir=str(data.data_dir),
        raw_receipt_path=str(data.raw_receipt_path),
        eb_receipt_path=str(data.eb_receipt_path),
        cache_dir=str(data.cache_dir),
        batch_size=int(data.batch_size),
        window_size=int(data.window_size),
        calibration_n_trials=int(data.calibration_n_trials),
        max_trial_length=int(data.max_trial_length),
        random_calibration=bool(data.random_calibration),
        smooth_calibration=bool(data.smooth_calibration),
        interpolate_trials=bool(data.interpolate_trials),
        interpolate_trials_kind=str(data.interpolate_trials_kind),
        num_workers=0,
        pin_memory=False,
        seed=int(data.seed),
        fixed_epochs=int(data.fixed_epochs),
        normalizer_floor=float(data.normalizer_floor),
    )
    source.setup("fit")
    return source


def _prepare_opened_development(device: str):
    """Bind source and model, then perform the sole target-opening step."""

    preflight = _require_preflight()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = _validate_carrierid_config(H_C_CONFIG, "full")
    checkpoint, meta = _load_carrierid_checkpoint(H_C_CHECKPOINT, H_C_CONFIG, "full")
    source = _source_datamodule(cfg)
    if meta["source_manifest_sha256"] != source.pilot_manifest_sha256:
        raise NormalizedV2ContractError("runtime source manifest differs from sealed checkpoint")
    if meta["normalizer_sha256"] != source.normalizer.normalizer_sha256:
        raise NormalizedV2ContractError("runtime source normalizer differs from sealed checkpoint")
    evaluation_device = torch.device(device)
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(evaluation_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del checkpoint
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise NormalizedV2ContractError("same-checkpoint diagnostic model is not frozen")

    # This import and call are deliberately below every preflight/checkpoint/
    # source binding above.  The loader indexes only public held-in-calib NWBs.
    from src.data.h1_m4_eb_pilot import load_target_records, validate_target_receipt_binding

    target_records = load_target_records(cfg.data.data_dir)
    validate_target_receipt_binding(
        target_records, source.plan, cfg.data.raw_receipt_path, cfg.data.eb_receipt_path
    )
    expected_files = preflight["target_contract_copied_without_opening_target"]["file_sha256"]
    observed_files = {name: target_records[name].input_sha256 for name in H1_M4_FOLD0_TARGET}
    if observed_files != expected_files:
        raise NormalizedV2ContractError("opened fold0 target bytes differ from immutable preflight")
    base = H1M4EBNormalizedV2StrictTargetDataset(
        target_records, source.plan, source.normalizer, "full"
    )
    gate, _full = _gate_and_full_metric()
    if base.window_indices_sha256 != EXPECTED_QUERY_HASH:
        raise NormalizedV2ContractError("runtime strict target query windows differ from sealed H-C")
    if base.support_and_carrier_hashes() != gate["target"]["support_and_carrier_hashes"]:
        raise NormalizedV2ContractError("runtime target support/carriers differ from sealed H-C")
    return preflight, cfg, source, model, base, evaluation_device, observed_files


def _override_dataset(base: Any, source: Any, *, kind: str, p: float, repeat_seed: int):
    if kind not in CORRUPTION_KINDS or p not in DOSE_GRID or p == 0.0 or repeat_seed not in REPEAT_SEEDS:
        raise NormalizedV2ContractError("requested shard lies outside the frozen design")
    carriers: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    for name in H1_M4_FOLD0_TARGET:
        support = base.support[name]
        full = np.asarray(support.carriers["full"], dtype=np.float64)
        if kind == "row":
            corrupted, audit = partial_row_corruption(
                full, session_name=name, p=p, repeat_seed=repeat_seed
            )
        else:
            raw, audit = partial_label_carrier(
                base.records[name], source.plan, support.trial_values, p=p, repeat_seed=repeat_seed
            )
            corrupted = source.normalizer.normalize(raw)
        if np.array_equal(corrupted, full):
            raise NormalizedV2ContractError(f"{kind} p={p} did not alter {name} carrier")
        carriers[name] = np.asarray(corrupted, dtype=np.float32)
        audits[name] = audit.manifest()
    overlay = CarrierOverrideTargetDataset(
        base,
        carriers,
        audit={"kind": kind, "nominal_p": p, "repeat_seed": repeat_seed, "per_session": audits},
    )
    if overlay.window_indices is not base.window_indices or overlay.window_indices_sha256 != base.window_indices_sha256:
        raise NormalizedV2ContractError("carrier overlay changed query-window identity")
    return overlay, audits


def _shard_path(kind: str, p: float, repeat_seed: int) -> Path:
    return SHARD_DIR / f"{kind}_p{int(round(p * 100)):03d}_seed{int(repeat_seed)}.json"


def _expected_shard_binding(
    preflight: Mapping[str, Any], p0_gate: Mapping[str, Any], kind: str, p: float, repeat_seed: int
) -> dict[str, Any]:
    return {
        "preflight_sha256": sha256_file(PREFLIGHT),
        "runtime_p0_gate_sha256": sha256_file(RUNTIME_P0_GATE),
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "kind": kind,
        "nominal_p": p,
        "repeat_seed": repeat_seed,
        "query_window_indices_sha256": EXPECTED_QUERY_HASH,
        "runtime_p0_authorized_nonzero_dose": p0_gate["nonzero_dose_authorized"],
    }


def _load_shard(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or not immutable_mode_0444(path):
        raise NormalizedV2ContractError(f"dose-response shard missing/not immutable: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SHARD_SCHEMA or receipt.get("status") != SHARD_STATUS:
        raise NormalizedV2ContractError(f"invalid dose-response shard status: {path}")
    if receipt.get("binding") != dict(expected):
        raise NormalizedV2ContractError(f"dose-response shard binding drift: {path}")
    metric = receipt.get("metrics", {})
    if (
        metric.get("query_window_indices_sha256") != EXPECTED_QUERY_HASH
        or metric.get("r2_accumulator_dtype") != "float64"
        or metric.get("state_immutable") is not True
    ):
        raise NormalizedV2ContractError(f"dose-response shard metric contract drift: {path}")
    return receipt


def _run_one_prepared(
    *, preflight: Mapping[str, Any], p0_gate: Mapping[str, Any], source: Any, model: Any, base: Any,
    device: torch.device, kind: str, p: float, repeat_seed: int,
) -> dict[str, Any]:
    expected = _expected_shard_binding(preflight, p0_gate, kind, p, repeat_seed)
    path = _shard_path(kind, p, repeat_seed)
    if path.exists():
        return _load_shard(path, expected)
    overlay, audits = _override_dataset(base, source, kind=kind, p=p, repeat_seed=repeat_seed)
    started = time.monotonic()
    metrics = _evaluate(model, overlay, device, f"H-C/{kind}/p={p}/seed={repeat_seed}")
    elapsed = time.monotonic() - started
    body = {
        "schema": SHARD_SCHEMA,
        "status": SHARD_STATUS,
        "binding": expected,
        "carrier_sha256": overlay.carrier_sha256,
        "corruption_audit": audits,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "target_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged": True},
        "claim_boundary": "same-checkpoint opened-development sensitivity only; not training-time necessity",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    output, digest = write_immutable_json(path, body)
    print(
        json.dumps(
            {"completed": str(output), "sha256": digest, "pooled_r2": metrics["pooled_r2"], "elapsed_s": elapsed},
            sort_keys=True,
        ),
        flush=True,
    )
    return body


def execute_all_missing(*, device: str, torch_threads: int) -> dict[str, Any]:
    if RESULT.exists():
        raise FileExistsError(f"final result already exists: {RESULT}")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(int(torch_threads))
    preflight = _require_preflight()
    p0_gate = _require_runtime_p0_gate(preflight)
    prepared = _prepare_opened_development(device)
    prepared_preflight, _cfg, source, model, base, evaluation_device, _files = prepared
    if prepared_preflight != preflight:
        raise NormalizedV2ContractError("prepared execution preflight drift")
    completed = 0
    for kind in CORRUPTION_KINDS:
        for repeat_seed in REPEAT_SEEDS:
            for p in DOSE_GRID[1:]:
                _run_one_prepared(
                    preflight=preflight, p0_gate=p0_gate, source=source, model=model, base=base,
                    device=evaluation_device, kind=kind, p=p, repeat_seed=repeat_seed,
                )
                completed += 1
    return {"status": "SHARDS_COMPLETE_READY_TO_FINALIZE", "shards": completed}


def execute_one(*, device: str, torch_threads: int, kind: str, p: float, repeat_seed: int) -> dict[str, Any]:
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(int(torch_threads))
    preflight = _require_preflight()
    p0_gate = _require_runtime_p0_gate(preflight)
    prepared_preflight, _cfg, source, model, base, evaluation_device, _files = _prepare_opened_development(device)
    if prepared_preflight != preflight:
        raise NormalizedV2ContractError("prepared execution preflight drift")
    return _run_one_prepared(
        preflight=preflight, p0_gate=p0_gate, source=source, model=model, base=base,
        device=evaluation_device, kind=kind, p=p, repeat_seed=repeat_seed,
    )


def _metric_scalar(metric: Mapping[str, Any], recording: str | None) -> float:
    if recording is None:
        return float(metric["pooled_r2"])
    return float(metric["per_session"][recording]["r2"])


def _curve_summary(
    base_metric: Mapping[str, Any], shards: Mapping[tuple[str, float, int], Mapping[str, Any]],
    *, kind: str, recording: str | None,
) -> dict[str, Any]:
    base = _metric_scalar(base_metric, recording)
    rows: list[dict[str, Any]] = []
    mean_values: list[float] = []
    for p in DOSE_GRID:
        values = [base] * len(REPEAT_SEEDS) if p == 0.0 else [
            _metric_scalar(shards[(kind, p, seed)]["metrics"], recording) for seed in REPEAT_SEEDS
        ]
        mean = float(np.mean(values, dtype=np.float64))
        mean_values.append(mean)
        rows.append(
            {
                "nominal_p": p,
                "logical_repeats": len(values),
                "forward_evaluations": 0 if p == 0.0 else len(values),
                "mean_r2": mean,
                "sd_r2_population": float(np.std(values, ddof=0, dtype=np.float64)),
                "min_r2": float(min(values)),
                "max_r2": float(max(values)),
                "mean_delta_vs_p0": mean - base,
                "repeat_r2": {str(seed): float(value) for seed, value in zip(REPEAT_SEEDS, values)},
            }
        )
    mean_steps = [mean_values[index + 1] - mean_values[index] for index in range(len(mean_values) - 1)]
    repeats: dict[str, Any] = {}
    for seed in REPEAT_SEEDS:
        values = [base] + [
            _metric_scalar(shards[(kind, p, seed)]["metrics"], recording) for p in DOSE_GRID[1:]
        ]
        steps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
        repeats[str(seed)] = {
            "r2": values,
            "adjacent_steps": steps,
            "nonincreasing": all(step <= 0.0 for step in steps),
            "upward_step_count": sum(step > 0.0 for step in steps),
            "max_upward_step": max([0.0, *steps]),
        }
    return {
        "scope": "pooled" if recording is None else recording,
        "by_p": rows,
        "mean_curve_adjacent_steps": mean_steps,
        "mean_curve_nonincreasing": all(step <= 0.0 for step in mean_steps),
        "mean_curve_upward_step_count": sum(step > 0.0 for step in mean_steps),
        "mean_curve_max_upward_step": max([0.0, *mean_steps]),
        "per_repeat": repeats,
        "nonincreasing_repeat_count": sum(value["nonincreasing"] for value in repeats.values()),
    }


def finalize() -> dict[str, Any]:
    if RESULT.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {RESULT}")
    preflight = _require_preflight()
    p0_gate = _require_runtime_p0_gate(preflight)
    gate, sealed_base_metric = _gate_and_full_metric()
    runtime_base_metric = p0_gate["runtime_metrics"]
    shards: dict[tuple[str, float, int], Mapping[str, Any]] = {}
    shard_bindings: list[dict[str, Any]] = []
    for kind in CORRUPTION_KINDS:
        for repeat_seed in REPEAT_SEEDS:
            for p in DOSE_GRID[1:]:
                expected = _expected_shard_binding(preflight, p0_gate, kind, p, repeat_seed)
                path = _shard_path(kind, p, repeat_seed)
                receipt = _load_shard(path, expected)
                shards[(kind, p, repeat_seed)] = receipt
                shard_bindings.append({"path": str(path), "sha256": sha256_file(path), **expected})

    curves: dict[str, Any] = {}
    for kind in CORRUPTION_KINDS:
        curves[kind] = {
            "pooled": _curve_summary(runtime_base_metric, shards, kind=kind, recording=None),
            "per_recording": {
                name: _curve_summary(runtime_base_metric, shards, kind=kind, recording=name)
                for name in H1_M4_FOLD0_TARGET
            },
        }
    pooled_monotone = {kind: bool(curves[kind]["pooled"]["mean_curve_nonincreasing"]) for kind in CORRUPTION_KINDS}
    body = {
        "schema": RESULT_SCHEMA,
        "status": (
            "COMPLETE_H1_SAME_CHECKPOINT_DOSE_RESPONSE_BOTH_POOLED_MEAN_CURVES_MONOTONE"
            if all(pooled_monotone.values())
            else "COMPLETE_H1_SAME_CHECKPOINT_DOSE_RESPONSE_WITH_POOLED_MONOTONICITY_VIOLATION"
        ),
        "preflight": {"path": str(PREFLIGHT), "sha256": sha256_file(PREFLIGHT)},
        "runtime_p0_gate": {"path": str(RUNTIME_P0_GATE), "sha256": sha256_file(RUNTIME_P0_GATE)},
        "sealed_h_c_terminal_gate": {"path": str(H_C_GATE), "sha256": sha256_file(H_C_GATE)},
        "checkpoint": dict(preflight["checkpoint"]),
        "frozen_design": dict(preflight["frozen_design"]),
        "target": {
            **dict(preflight["target_contract_copied_without_opening_target"]),
            "same_query_windows_for_every_shard": True,
        },
        "p0": {
            "origin": "mandatory current-runtime forward, gated against canonical sealed float64 H-C metric",
            "runtime_metrics": runtime_base_metric,
            "sealed_metrics": sealed_base_metric,
            "r2_reproduction": p0_gate["r2_reproduction"],
            "carrier_sha256": {
                name: gate["target"]["support_and_carrier_hashes"][name]["carrier_sha256"]["full"]
                for name in H1_M4_FOLD0_TARGET
            },
        },
        "shards": shard_bindings,
        "superseded_v1": dict(preflight["supersedes"]),
        "curves": curves,
        "pooled_mean_curve_nonincreasing": pooled_monotone,
        "claim_boundary": {
            "supported_if_monotone": "the sealed H-C checkpoint is progressively sensitive to corruption of carrier attachment/pairing on opened fold0 development data",
            "not_supported": "carrier content was necessary during source training",
            "reason": "all nonzero doses are deployment-time interventions on one already-trained checkpoint",
            "formal_endpoint": False,
        },
        "target_updates": {"optimizer_steps": 0, "backward_steps": 0, "model_state_unchanged_each_shard": True},
        "data_scope": {
            "opened": "11 public source NWBs then exactly 2 public fold0 held-in-calib development NWBs per execution process",
            "formal_hidden_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        },
        "code_sha256": _code_hashes(),
    }
    path, digest = write_immutable_json(RESULT, body)
    return {"status": body["status"], "receipt": str(path), "sha256": digest, "monotone": pooled_monotone}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-preflight", action="store_true")
    mode.add_argument("--execute-runtime-p0-gate", action="store_true")
    mode.add_argument("--execute-all-missing-opened-development", action="store_true")
    mode.add_argument("--execute-one-opened-development", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--kind", choices=CORRUPTION_KINDS)
    parser.add_argument("--p", type=float, choices=DOSE_GRID[1:])
    parser.add_argument("--repeat-seed", type=int, choices=REPEAT_SEEDS)
    args = parser.parse_args()
    if args.write_preflight:
        output = write_preflight()
    elif args.execute_runtime_p0_gate:
        output = execute_runtime_p0_gate(device=args.device, torch_threads=args.torch_threads)
    elif args.finalize:
        output = finalize()
    elif args.execute_all_missing_opened_development:
        output = execute_all_missing(device=args.device, torch_threads=args.torch_threads)
    else:
        if args.kind is None or args.p is None or args.repeat_seed is None:
            parser.error("--execute-one requires --kind, --p and --repeat-seed from the frozen design")
        output = execute_one(
            device=args.device, torch_threads=args.torch_threads,
            kind=args.kind, p=args.p, repeat_seed=args.repeat_seed,
        )
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
