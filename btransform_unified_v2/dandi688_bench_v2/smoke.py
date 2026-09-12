"""Two-session, source-only end-to-end smoke for DANDI688 v2.

The smoke is deliberately ineligible for any scientific selection: it uses
only 2015 source sessions, two optimizer updates per stage, and writes an
explicit ``SMOKE`` receipt.  Its purpose is to exercise cache loading,
representation pairing, encoder handoff, decoder checkpoint reload, and the
CPU baseline interfaces before a formal command is considered.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

# The smoke is an executable entry point as well as an importable function.
# Keep it runnable in the isolated environment without inheriting PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "learnable_recency_v1" / "src", _ROOT.parent / "btransform_unified_v1" / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from . import protocol
from .baseline_runner import PMUA_METHODS, run_baselines
from .baselines import fitstatic_adapter
from .common import (SCHEMA, atomic_json, fresh_directory, load_records, record_binding,
                     sha256, source_hashes, score_predictions)
from .data import load_pair, save_session
from .training import (load_encoder, load_trained_model, predict_network,
                       calibration_inputs, run_training)

SMOKE_IDS = ("sub-C_ses-CO-20150309", "sub-C_ses-CO-20150716")
QUERY_CAP = 8


def _prepare_cache(dest: Path) -> Path:
    cache = dest / "prepared_two_source_smoke"
    cache.mkdir(parents=True, exist_ok=False)
    for session_id in SMOKE_IDS:
        pair = load_pair(session_id, purpose="source")
        for representation, record in pair.items():
            save_session(record, cache / f"{session_id}.{representation}.npz")
    return cache


def _short(record):
    return replace(record, query_indices=np.asarray(record.query_indices[:QUERY_CAP], dtype=np.int64))


def _assert_completed(receipt: dict, *, representation: str, arm: str, stage: str) -> None:
    if (receipt.get("status"), receipt.get("completed"), receipt.get("representation"),
            receipt.get("arm"), receipt.get("stage"), receipt.get("global_step")) != ("SMOKE", True, representation, arm, stage, 2):
        raise RuntimeError("smoke training receipt contract drift")
    if receipt.get("final_sessions_opened") != 0:
        raise RuntimeError("smoke must never open final sessions")
    if receipt.get("device") != "cpu":
        raise RuntimeError("DANDI smoke is CPU-only")


def _carrier_route_evidence(encoder, record, stats: dict) -> dict[str, Any]:
    """Prove a trained Full encoder consumes real MOVE--T4 through concat."""
    calibration = calibration_inputs(record, "full", stats, torch.device("cpu"), n_pad=100)
    activity, carrier = calibration["activity"], calibration["carrier"]
    with torch.inference_mode():
        real_e0 = encoder(activity, carrier)
        zero_e0 = encoder(activity, torch.zeros_like(carrier))
    if not bool(torch.isfinite(real_e0).all()) or not bool(torch.isfinite(zero_e0).all()):
        raise RuntimeError("Full encoder emitted non-finite E0 during carrier-route smoke")
    difference = float(torch.linalg.vector_norm(real_e0 - zero_e0).cpu())
    if difference <= 0.0:
        raise RuntimeError("two-step Full pretrain did not make real MOVE-T4 affect E0")
    first = encoder.post_pool[0]
    if not isinstance(first, torch.nn.Linear) or first.in_features - encoder.hidden_dim != 4:
        raise RuntimeError("Full encoder lost the B3S post-pool concat interface")
    side = first.weight[:, encoder.hidden_dim:]
    side_norm = float(torch.linalg.vector_norm(side).cpu())
    if side_norm <= 0.0:
        raise RuntimeError("two-step Full pretrain did not update zero-initialized MOVE-T4 columns")
    return {"actual_carrier_routed": True, "e0_shape": list(real_e0.shape),
            "real_t4_vs_zero_t4_e0_l2": difference, "side_column_l2": side_norm,
            "side_dim": 4, "concat_line": "per_unit_pre_pool_mean + MOVE_T4 -> post_pool"}


def _assert_activity_excludes_carrier(model, record, stats: dict) -> None:
    calibration = calibration_inputs(record, "activity", stats, torch.device("cpu"), n_pad=model.units)
    try:
        model.encode(calibration["activity"], torch.zeros(model.units, 4))
    except ValueError:
        return
    raise RuntimeError("activity checkpoint accepted a MOVE-T4 carrier")


def run_smoke(dest: Path, cache: Path | None = None) -> dict[str, Any]:
    """Execute the source-only two-session smoke and write ``receipt.json``.

    With ``cache=None`` this opens exactly the two authorized source NWBs and
    saves both representations into an owned cache below ``dest``.  Supplying
    a cache performs no NWB I/O and is preferred by CI/review reruns.
    """
    destination = fresh_directory(dest)
    started = time.monotonic()
    if cache is None:
        cache_path, cache_mode = _prepare_cache(destination), "built_two_source"
    else:
        cache_path, cache_mode = Path(cache).resolve(), "provided"
    if not cache_path.is_dir():
        raise FileNotFoundError(cache_path)

    results: dict[str, Any] = {"pretrain": {}, "network": {}, "checkpoint_reload": {}, "baselines": {}}
    sampler_hashes: list[str] = []
    records_by_rep = {representation: load_records(cache_path, representation, "train", session_ids=SMOKE_IDS)
                      for representation in ("sua", "pmua")}
    for representation, records in records_by_rep.items():
        pretrain_dir = destination / f"pretrain_{representation}"
        pretrain = run_training(cache_path, pretrain_dir, representation=representation,
                                stage="pretrain", arm="full", smoke_updates=2,
                                source_ids=SMOKE_IDS)
        _assert_completed(pretrain, representation=representation, arm="full", stage="pretrain")
        encoder_path = pretrain_dir / "encoder.pt"
        # This also checks representation/statistics binding and permits only
        # this explicitly marked smoke handoff.
        encoder_stats = json.loads((pretrain_dir / "source_stats.json").read_text())
        encoder = load_encoder(encoder_path, representation=representation, stats=encoder_stats, allow_smoke=True)
        results["pretrain"][representation] = {"receipt": str(pretrain_dir / "receipt.json"),
                                                 "encoder": str(encoder_path), "encoder_sha256": sha256(encoder_path),
                                                 "carrier_route": _carrier_route_evidence(encoder, records[0],
                                                                                           encoder_stats)}
        sampler_hashes.append(pretrain["sampler"]["sha256"])
        for arm in ("full", "activity", "raw_set"):
            run_dir = destination / f"train_{representation}_{arm}"
            receipt = run_training(cache_path, run_dir, representation=representation, arm=arm,
                                   stage="train", encoder_path=encoder_path if arm == "full" else None,
                                   smoke_updates=2, source_ids=SMOKE_IDS, eval_records=[_short(records[1])])
            _assert_completed(receipt, representation=representation, arm=arm, stage="train")
            roundtrip = receipt.get("checkpoint_roundtrip")
            if not isinstance(roundtrip, dict) or roundtrip.get("in_memory_ema_vs_reload") is not True or roundtrip.get("max_abs_delta") != 0.0:
                raise RuntimeError("decoder smoke receipt lacks an exact in-memory EMA checkpoint roundtrip")
            sampler_hashes.append(receipt["sampler"]["sha256"])
            checkpoint = run_dir / "segment_01.pt"
            model_a, stats_a = load_trained_model(checkpoint, allow_smoke=True)
            model_b, stats_b = load_trained_model(checkpoint, allow_smoke=True)
            probe = _short(records[0])
            first, second = predict_network(model_a, probe, stats_a), predict_network(model_b, probe, stats_b)
            if first.shape != (QUERY_CAP, 2) or not np.array_equal(first, second):
                raise RuntimeError("checkpoint reload prediction parity failed")
            if arm == "full":
                exported = load_encoder(encoder_path, representation=representation,
                                        stats=encoder_stats, allow_smoke=True)
                if any(not torch.equal(value.cpu(), model_a.encoder.state_dict()[name].cpu())
                       for name, value in exported.state_dict().items()):
                    raise RuntimeError("frozen Full encoder changed during decoder smoke")
            elif arm == "activity":
                _assert_activity_excludes_carrier(model_a, records[0], stats_a)
            evidence = {"receipt": str(run_dir / "receipt.json"), "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint), "prediction_shape": list(first.shape),
                        "training_checkpoint_roundtrip": roundtrip}
            if arm == "full":
                evidence["encoder_frozen"] = True
            elif arm == "activity":
                evidence["carrier_rejected"] = True
            results["network"][f"{representation}:{arm}"] = evidence
            results["checkpoint_reload"][f"{representation}:{arm}"] = True

    if len(set(sampler_hashes)) != 1:
        raise RuntimeError("all six decoder arms and representation pretrains must share one sampler stream")

    source_short = {name: [_short(records_by_rep[name][0])] for name in ("sua", "pmua")}
    target_short = {name: [_short(records_by_rep[name][1])] for name in ("sua", "pmua")}
    baseline_dir = destination / "baselines_pmua_source_free"
    baseline_methods = (*PMUA_METHODS, "wf_fss_sua", "wf_fss_pmua")
    selection = run_baselines(cache_path, baseline_dir, methods=baseline_methods, smoke=True,
                              source_records=source_short, dev_records=target_short)
    if set(selection["artifacts"]) != set(baseline_methods):
        raise RuntimeError("all five source-free and both FSS smoke baselines must produce artifacts")

    # Replay the PMUA raw checkpoint with target-neural-only adapters.  Scores
    # are diagnostic only; target velocity remains untouched except for metric
    # calculation and its pre/post digest is recorded below.
    raw_checkpoint = destination / "train_pmua_raw_set" / "segment_01.pt"
    raw_model, raw_stats = load_trained_model(raw_checkpoint, allow_smoke=True)
    source, target = records_by_rep["pmua"]
    target_probe = _short(target)
    before = target_probe.metadata["array_sha256"]["velocity"]
    identity = score_predictions(target_probe, predict_network(raw_model, target_probe, raw_stats))
    adapters = {}
    for kind in ("diag_z", "coral"):
        adapter = fitstatic_adapter([source], target_probe, kind=kind, reference_session=source.session_id)
        adapters[kind] = {"metrics": score_predictions(target_probe, predict_network(raw_model, target_probe, raw_stats, adapter=adapter)),
                          "diagnostics": adapter.diagnostics}
    if before != target_probe.metadata["array_sha256"]["velocity"]:
        raise RuntimeError("baseline smoke mutated target velocity provenance")
    results["baselines"] = {"selection": str(baseline_dir / "selection.json"), "methods": list(baseline_methods),
                            "raw_set_pmua": {"identity": identity, "adapters": adapters,
                                                               "query_cap": QUERY_CAP, "target_velocity_sha256": before}}

    receipt = {"schema": SCHEMA + "_two_source_smoke_v1", "status": "SMOKE", "ineligible_for_formal": True,
               "cache": str(cache_path), "cache_mode": cache_mode, "source_sessions": list(SMOKE_IDS),
               "forbidden": {"non_2015": True, "development": True, "final": True, "full_training": True},
               "cpu_only": True, "device": "cpu", "final_sessions_opened": 0, "no_development_access": True,
               "architecture": {"identity_encoder": "B3S", "full_encoder_fusion": "early_pool_then_concat_MOVE_T4_then_post_pool",
                                "concat_line": "per_unit_pre_pool_mean + MOVE_T4 -> post_pool", "decoder_identity": "proj_add",
                                "film": False, "carrier_side_dim": 4, "activity_side_dim": 0},
               "sampler_sha256": sampler_hashes[0], "source_binding": {name: record_binding(rows) for name, rows in records_by_rep.items()},
               "results": results, "code_hashes": source_hashes(), "elapsed_seconds": time.monotonic() - started}
    atomic_json(destination / "receipt.json", receipt)
    return receipt


__all__ = ["SMOKE_IDS", "QUERY_CAP", "run_smoke"]
