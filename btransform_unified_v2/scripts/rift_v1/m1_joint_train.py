#!/usr/bin/env python3
"""Formal M1 RIFT R100/D4 full-concat runner over the frozen four-session M10 face."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT.parent / "btransform_unified_v1"
WS = ROOT.parent
for candidate in (ROOT / "src", V1 / "src", V1 / "scripts", WS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from btransform_unified_v1 import m1_projadd as m1_plan
from btransform_unified_v1 import plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v1.schedule import warmup_cosine_lr
from tfpd_exploration.src.m2_dual_track_v1 import training as optimizer_factory
import m1_projadd_depth2_series as legacy

CELL = "M1-RIFT-R100-D4-JOINT-B3S-CONCAT-V1"
SEED, CONTEXT, PROJ_DIM, EPOCHS, BATCH, LR = 42, 100, 16, 24, 32, 1e-4
QUERY_PAD_BINS = CONTEXT - 1
SOURCE_SESSIONS = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
EXPECTED_WINDOWS, UPDATES_PER_EPOCH = 213336, 6665


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)
    return _sha_file(path)


def _source_hashes() -> dict[str, str]:
    from tfpd_exploration.src.m1_optimized_v2 import plan as carrier_plan
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as raw_plan
    paths = (Path(__file__), ROOT / "src/btransform_unified_v2/joint_m1_model.py", ROOT / "src/btransform_unified_v2/model.py", ROOT / "src/btransform_unified_v2/concat_model.py", ROOT / "src/btransform_unified_v2/temporal.py",
             ROOT / "src/btransform_unified_v2/config.py", V1 / "scripts/m1_projadd_depth2_series.py",
             V1 / "src/btransform_unified_v1/m1_projadd.py", V1 / "src/btransform_unified_v1/m1_b3s_joint.py", V1 / "src/btransform_unified_v1/identity_variant.py",
             V1 / "src/btransform_unified_v1/bank.py", WS / "streaming_calibration_exp/src/data/falcon_datamodule.py",
             WS / "tfpd_exploration/src/m1_b3_allsource_v1/rsyn3_bank.py", WS / "tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py",
             WS / "tfpd_exploration/src/m1_optimized_v2/bank.py", WS / "tfpd_exploration/src/m1_optimized_v2/calibration.py",
             WS / "streaming_calibration_exp/src/models/components/streaming_encoders.py",
             WS / "tfpd_exploration/src/two_mainlines_long_v1/decoder/m1_config.py",
             V1 / "scripts/m1_family_loso_outer20120924.py", Path(carrier_plan.BANK_NPZ), Path(carrier_plan.BANK_RECEIPT),
             *(WS / raw_plan.SOURCE_RELATIVE[name] for name in SOURCE_SESSIONS))
    return {str(path): _sha_file(path) for path in paths}


def _b3s_provenance() -> dict[str, Any]:
    """Bind the live encoder to its frozen B3 source without copying decoder weights."""
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import S_FIX_PATH, S_FIX_SHA256
    from btransform_unified_v1.m1_b3s_joint import B3S_HIDDEN, B3S_POST_LAYERS, B3S_SIDE_DIM
    path = Path(S_FIX_PATH)
    digest = _sha_file(path)
    if digest != S_FIX_SHA256:
        raise RuntimeError("B3 Sfix checkpoint checksum drift")
    return {"sfix_path": str(path), "sfix_sha256": digest, "encoder": "SideFeatureEarlyPoolEncoder",
            "hidden_dim": B3S_HIDDEN, "side_dim": B3S_SIDE_DIM, "post_layers": B3S_POST_LAYERS,
            "decoder_weights_copied": False, "side_columns_initialized_zero": True}


def _decoder(device: torch.device, arm: str, seed: int):
    from btransform_unified_v2.joint_m1_model import JointM1ConcatDecoder
    model = JointM1ConcatDecoder(arm, seed=seed).to(device)
    model.temporal.set_attention_backend("local")
    if tuple(model.temporal_config.windows) != (25, 25, 25, 24):
        raise RuntimeError("M1 R100 D4 layer windows drifted")
    return model


def _initialization_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def valid_mask_from_padded_starts(starts: tuple[int, ...] | list[int], *, device: torch.device) -> torch.Tensor:
    """M1 raw W100 mask from padded-timeline starts; never infer validity from X values."""
    begin = torch.as_tensor(starts, dtype=torch.long, device=device).unsqueeze(1)
    offsets = torch.arange(CONTEXT, dtype=torch.long, device=device).unsqueeze(0)
    return begin + offsets >= QUERY_PAD_BINS


def _source_contract(dataset: Any, sampler: Any, banks: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    counts = {session: sum(1 for row in dataset.window_indices if row[0] == session) for session in SOURCE_SESSIONS}
    if tuple(counts) != SOURCE_SESSIONS or sum(counts.values()) != EXPECTED_WINDOWS:
        raise RuntimeError(f"M1 source window inventory drifted: {counts}")
    if len(sampler) != UPDATES_PER_EPOCH:
        raise RuntimeError(f"M1 updates/epoch drifted: {len(sampler)}")
    queries: dict[str, Any] = {}
    bank_hashes: dict[str, Any] = {}
    for session in SOURCE_SESSIONS:
        starts = np.asarray([start for name, start in dataset.window_indices if name == session], dtype=np.int64)
        neural = np.ascontiguousarray(dataset.neural_data[session], dtype=np.float32)
        covariates = np.ascontiguousarray(dataset.covariate_data[session], dtype=np.float32)
        targets = np.ascontiguousarray(covariates[starts + QUERY_PAD_BINS], dtype=np.float32)
        calib = np.ascontiguousarray(dataset.calib_trialized_neural_features[session][:10], dtype=np.float32)
        queries[session] = {"window_count": int(len(starts)), "window_starts_sha256": _array_sha(starts),
                            "eval_mask_sha256": _array_sha(dataset.eval_mask[session]),
                            "padded_neural_sha256": _array_sha(neural), "padded_covariate_sha256": _array_sha(covariates),
                            "query_target_sha256": _array_sha(targets), "m10_calib_sha256": _array_sha(calib),
                            "query_pad_bins": QUERY_PAD_BINS, "validity": "all source windows post-M10 legal; explicit all-true mask"}
        bank = banks[session]
        bank_hashes[session] = {"e0_sha256": str(bank.calibration_meta["array_sha256"]),
                                "carrier_sha256": str(bank.calibration_meta["carrier_sha256"]),
                                "budget": int(bank.calibration_meta["budget"])}
    return {"sessions": list(SOURCE_SESSIONS), "heldout_sessions": [], "total_windows": EXPECTED_WINDOWS,
            "windows_by_session": counts, "updates_per_epoch": UPDATES_PER_EPOCH,
            "sampler_batch_sha256": m1_plan.sampler_digest(sampler), "sampler": {"batch": BATCH, "seed": SEED, "shuffle": True, "balance": False, "reshuffle_each_epoch": False},
            "bank_hashes": bank_hashes, "query_hashes": queries, "bank_report": dict(report)}


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def _restore_rng(payload: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(payload["python"]); np.random.set_state(payload["numpy"]); torch.set_rng_state(payload["torch_cpu"].cpu())
    if device.type == "cuda" and payload.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in payload["torch_cuda"]])


def _with_ema_eval(model: nn.Module, ema: DecoderEMA, fn):
    named = dict(model.named_parameters())
    saved = {name: value.detach().clone() for name, value in named.items()}
    was_training = model.training
    try:
        with torch.no_grad():
            for name, value in named.items():
                value.copy_(ema.shadow[name].to(value.device, dtype=value.dtype))
        model.eval()
        return fn()
    finally:
        with torch.no_grad():
            for name, value in named.items():
                value.copy_(saved[name])
        model.train(was_training)


def _ho_material() -> dict[str, Any]:
    provider = legacy.mp.default_identity_provider()
    result: dict[str, Any] = {}
    for session in HO:
        opened = legacy.open_heldout_calib_session(session)
        carrier, carrier_meta = legacy.encode_heldout_calib_carrier(session)
        bank = legacy.make_pick_bank(session, provider(opened["calib10"]), carrier)
        dataset = opened["dataset"]
        starts = tuple(int(start) for _name, start in dataset.window_indices)
        target = np.ascontiguousarray(dataset.covariate_data[session][np.asarray(starts, dtype=np.int64) + QUERY_PAD_BINS], dtype=np.float32)
        result[session] = {"dataset": dataset, "bank": bank, "calib10": opened["calib10"], "starts": starts,
                           "body_sha256": opened["body_sha256"], "carrier_sha256": str(bank.calibration_meta["carrier_sha256"]),
                           "e0_sha256": str(bank.calibration_meta["array_sha256"]), "starts_sha256": _array_sha(np.asarray(starts, dtype=np.int64)),
                           "target_sha256": _array_sha(target), "neural_sha256": _array_sha(dataset.neural_data[session]),
                           "covariate_sha256": _array_sha(dataset.covariate_data[session]), "calib10_sha256": _array_sha(opened["calib10"]),
                           "window_count": len(starts), "carrier_meta": carrier_meta}
    return result


def _ho_contract(material: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
    rows = {}
    for session in HO:
        item = material[session]
        if int(item["window_count"]) != expected[session]:
            raise RuntimeError(f"held-out window inventory drifted for {session}")
        rows[session] = {key: item[key] for key in ("body_sha256", "e0_sha256", "carrier_sha256", "starts_sha256", "target_sha256", "neural_sha256", "covariate_sha256", "calib10_sha256", "window_count")}
    return {"sessions": list(HO), "per_session": rows, "total_windows": sum(expected.values()), "query_pad_bins": QUERY_PAD_BINS}


def _validate_scored_report(report: Mapping[str, Any]) -> None:
    expected = {"20121004": 1305, "20121017": 1295, "20121024": 1281}
    if report.get("partial") is not False or set(report.get("per_session", {})) != set(HO):
        raise RuntimeError("scored held-out report is incomplete")
    if int(report.get("n_windows", -1)) != 3881 or not np.isfinite(float(report.get("equal_session_mean", np.nan))):
        raise RuntimeError("scored held-out report aggregate drifted")
    for session, count in expected.items():
        row = report["per_session"][session]
        if int(row.get("window_count", -1)) != count or not np.isfinite(float(row.get("r2", np.nan))):
            raise RuntimeError(f"scored held-out report invalid for {session}")


def _ho_score(model: nn.Module, material: Mapping[str, Any], device: torch.device, *, max_batches: int | None = None) -> dict[str, Any]:
    """Score full padded W100 contexts with model.eval(), preserving real zero bins."""
    was_training = model.training
    model.eval()
    try:
        rows: dict[str, Any] = {}
        for session in HO:
            item = material[session]; dataset = item["dataset"]
            predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []
            loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                x, y = batch[0], batch[1]
                offset = batch_index * BATCH
                starts = item["starts"][offset:offset + len(x)]
                valid = valid_mask_from_padded_starts(starts, device=device)
                with torch.inference_mode():
                    prediction = model(x.float().to(device), item["bank"], input_valid_mask=valid)
                predictions.append(np.ascontiguousarray(prediction.float().cpu().numpy(), dtype=np.float32))
                targets.append(np.ascontiguousarray(y[:, -1, :].numpy(), dtype=np.float32))
            pred = np.concatenate(predictions); target = np.concatenate(targets)
            rows[session] = {"r2": float(variance_weighted_r2(target, pred)), "window_count": int(len(target)),
                             "prediction_sha256": _array_sha(pred)}
        return {"per_session": rows, "equal_session_mean": float(np.mean([rows[key]["r2"] for key in HO])),
                "partial": max_batches is not None, "n_windows": int(sum(row["window_count"] for row in rows.values()))}
    finally:
        model.train(was_training)


def _assert_full_stream_parity(model: nn.Module, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Check full-window and token-by-token paths on a left-padded W100 context."""
    from btransform_unified_v2.streaming import RiftStreamDecoder
    item = material[HO[0]]
    x = np.ascontiguousarray(item["dataset"][0][0], dtype=np.float32).copy()
    # A synthetic padded coordinate exercises the 99-bin law using an actual
    # raw window. It does not alter the held-out scoring inventory.
    valid = valid_mask_from_padded_starts((97,), device=device)
    # This zero is in a *valid* final-bin coordinate, so it proves zero-valued
    # activity is kept separate from the startup validity mask.
    x[-1, 0] = 0.0
    raw = torch.from_numpy(x).unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            full = model(raw, item["bank"], input_valid_mask=valid)
        stream = RiftStreamDecoder(model)
        streamed = None
        for offset in range(CONTEXT):
            streamed = stream.stream_step(raw[:, offset], item["bank"], ["m1-parity"], valid_mask=valid[:, offset])
        if streamed is None or not torch.allclose(full, streamed, rtol=2e-5, atol=2e-5):
            raise RuntimeError("M1 full-window versus streaming parity failed")
        return {"status": "PASSED", "query_start": 97, "valid_bins": int(valid.sum()), "startup_masked_bins": int((~valid).sum()),
                "true_zero_valid_coordinate": [CONTEXT - 1, 0], "max_abs": float((full - streamed).abs().max())}
    finally:
        model.train(was_training)


def _assert_ho_repeatable(model: nn.Module, ema: DecoderEMA, material: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    scorer = (lambda: _with_ema_eval(model, ema, lambda: _ho_score(model, material, device, max_batches=1))) if ema.n_updates > 0 else (lambda: _ho_score(model, material, device, max_batches=1))
    first = scorer()
    second = scorer()
    if first != second:
        raise RuntimeError("M1 held-out evaluation is not repeatable in eval mode")
    return {"status": "PASSED", "batches_per_session": 1, "prediction_sha256": {s: first["per_session"][s]["prediction_sha256"] for s in HO}}


def _init_parity(dataset: Any, banks: Mapping[str, Any], source_calib: Mapping[str, Any], device: torch.device, *, seed: int) -> dict[str, Any]:
    """Joint-D and frozen concat give the same source-bank forward at init.

    This is stronger than affine-fold parity: it proves the live B3S E0 (whose
    side columns start at zero) equals the bank's frozen B3 E0 on actual M10.
    """
    from btransform_unified_v2.joint_m1_model import JointM1ConcatDecoder, ARM_D
    from btransform_unified_v2.concat_model import RiftConcatDecoder
    x, _y, sessions = legacy._collate([dataset[0], dataset[1]])
    session = sessions[0]
    if any(name != session for name in sessions): raise RuntimeError("parity batch session drift")
    x = x.clone(); zero_count = int((x == 0).sum())
    # A zero-valued sample is valid evidence, never a padding sentinel.
    if zero_count == 0: x[0, 0, 0] = 0.; zero_count = 1
    valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
    frozen = RiftConcatDecoder("m1", context_bins=CONTEXT, bias_mode="recency", seed=seed).to(device).eval()
    joint = JointM1ConcatDecoder(ARM_D, seed=seed).to(device)
    joint.install_session_memory(banks, source_calib); joint.to(device); joint.eval()
    shared = dict(frozen.named_parameters()); pairs = [(n,v) for n,v in joint.named_parameters() if n in shared and v.shape == shared[n].shape]
    with torch.inference_mode():
        left = frozen(x.to(device), banks[session], input_valid_mask=valid)
        right = joint(x.to(device), banks[session], input_valid_mask=valid)
    maximum = float((left-right).abs().max())
    if maximum >= 2e-6 or not all(torch.equal(v, shared[n]) for n,v in pairs): raise RuntimeError("full-forward concat init parity failed")
    digest=lambda m: hashlib.sha256(b"".join(v.detach().cpu().contiguous().numpy().tobytes() for _n,v in sorted(m.named_parameters()))).hexdigest()
    return {"status":"PASSED","comparison":"frozen_concat_vs_live_joint_d","seed":seed,"batch_size":len(x),"session":session,"valid_zero_bin_count":zero_count,"output_max_abs":maximum,"threshold":2e-6,"shared_same_parameter_count":len(pairs),"shared_all_byte_equal":True,"frozen_total_parameters":sum(v.numel() for v in frozen.parameters()),"joint_total_parameters":sum(v.numel() for v in joint.parameters()),"frozen_init_sha256":digest(frozen),"joint_init_sha256":digest(joint),"head_alias":{"owner_final_norm_is_model":joint._frontend_owner.final_norm is joint.final_norm,"owner_readout_is_model":joint._frontend_owner.readout is joint.readout}}

def _checkpoint_payload(model, optimizer, ema, *, epoch: int, step: int, smoke: bool, meta: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {"schema": "m1_rift_joint_epoch_checkpoint_v1", "cell": CELL, "epoch": epoch, "global_step": step, "smoke": smoke,
            "config": {"arm":meta["arm"], "seed": meta["seed"], "sampler_seed":SEED, "context_bins": CONTEXT, "proj_dim": None, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "bias_mode": "recency", "attention_backend": "local", "b3s": meta["b3s"]},
            "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "initialization_sha256": meta["initialization_sha256"],
            "raw_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.state_dict(), "rng": _rng_state(device)}


def _validate_checkpoint(path: Path, meta: Mapping[str, Any], *, expected_epoch: int | None = None) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"schema": "m1_rift_joint_epoch_checkpoint_v1", "cell": CELL, "smoke": False}
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    if expected_epoch is not None and int(state.get("epoch", 0)) != expected_epoch:
        raise RuntimeError(f"checkpoint epoch mismatch: {path}")
    if int(state.get("epoch",0)) < 1 or int(state.get("global_step",-1)) != int(state.get("epoch"))*UPDATES_PER_EPOCH:
        raise RuntimeError(f"checkpoint step/epoch mismatch: {path}")
    config = state.get("config", {})
    if config != {"arm":meta["arm"], "seed":meta["seed"], "sampler_seed":SEED, "context_bins": CONTEXT, "proj_dim": None, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "bias_mode": "recency", "attention_backend": "local", "b3s": meta["b3s"]}:
        raise RuntimeError(f"checkpoint configuration mismatch: {path}")
    for key in ("source_hashes", "source_contract", "initialization_sha256"):
        if state.get(key) != meta.get(key):
            raise RuntimeError(f"checkpoint {key} mismatch: {path}")
    return state


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    smoke = args.max_updates_smoke is not None
    if not smoke and args.epochs != EPOCHS:
        raise ValueError("formal M1 RIFT requires exactly 24 epochs")
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dest = args.dest.resolve()
    if args.resume is None and dest.exists() and any(dest.iterdir()):
        raise FileExistsError("new destination must be empty")
    dataset, sampler = legacy.build_fullsession_face(); banks, report = legacy.build_fullsession_banks(dataset)
    contract = _source_contract(dataset, sampler, banks, report)
    source_calib = m1_plan.calib_trials_from_dataset(dataset)
    contract["raw_m10_calib_sha256"] = {name:_array_sha(value) for name,value in source_calib.items()}
    model = _decoder(device, args.arm, args.seed)
    model.install_session_memory(banks, source_calib); model.to(device)
    init_sha = _initialization_sha(model); ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    optimizer = optimizer_factory.build_optimizer(model.named_parameters(), lr=LR, weight_decay=plan.WEIGHT_DECAY)
    smoke_preflight = None
    smoke_ho_contract = None
    if smoke:
        _material = _ho_material()
        smoke_ho_contract = _ho_contract(_material)
        model.install_session_memory({s:_material[s]["bank"] for s in HO},{s:_material[s]["calib10"] for s in HO}); model.to(device)
        smoke_preflight = {"init_parity": _init_parity(dataset, banks, source_calib, device, seed=args.seed), "streaming_parity": _assert_full_stream_parity(model, _material, device), "repeatable_eval": _assert_ho_repeatable(model, ema, _material, device), "live_encoder_parameter_sha256": _initialization_sha(model.encoder)}
    meta = {"schema": "m1_rift_joint_train_v1", "status": "SMOKE" if smoke else "FORMAL", "cell": CELL, "task": "m1", "arm":args.arm, "variant": "recency", "identity_interface": "live_b3s_concat", "identity_e0_dim": 100, "concat_token_width": 120, "seed": args.seed, "sampler_seed":SEED,
            "context_bins": CONTEXT, "query_pad_bins": QUERY_PAD_BINS, "layer_windows": [25, 25, 25, 24], "depth": 4, "width": 256, "proj_dim": None, "attention_backend": "local",
            "epochs": args.epochs, "batch": BATCH, "updates_per_epoch": UPDATES_PER_EPOCH, "total_updates": EPOCHS * UPDATES_PER_EPOCH, "optimizer": {"name": "AdamW", "weight_decay": plan.WEIGHT_DECAY, "clip": 1.0}, "lr": {"peak": LR, "min": LR * plan.LR_MIN_FACTOR, "warmup_updates": UPDATES_PER_EPOCH}, "ema_decay": plan.EMA_DECAY, "unit_dropout": 0.1,
            "source_train_only_for_gradients": True, "development_surface": "held-out-calib M10, post-training EMA scan only", "official_test_used": False,
            "source_hashes": _source_hashes(), "source_contract": contract, "b3s": _b3s_provenance(), "initialization_sha256": init_sha,
            "launch": {"argv": sys.argv, "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "python_no_user_site": os.environ.get("PYTHONNOUSERSITE")}, "utc": datetime.now(timezone.utc).isoformat()}
    dest.mkdir(parents=True, exist_ok=True)
    if smoke: _atomic_json(dest / "init_parity_receipt.json", smoke_preflight["init_parity"])
    step, first_epoch = 0, 1
    if args.resume is None:
        _atomic_json(dest / "run_meta.json", meta)
    else:
        existing = json.loads((dest / "run_meta.json").read_text())
        if existing.get("status") != "FORMAL" or any(existing.get(key) != meta[key] for key in ("cell", "arm", "seed", "sampler_seed", "source_hashes", "source_contract", "initialization_sha256")):
            raise RuntimeError("resume run metadata mismatch")
        if args.resume.resolve().parent != dest:
            raise RuntimeError("resume checkpoint must belong directly to --dest")
        state = _validate_checkpoint(args.resume.resolve(), existing)
        if int(state["epoch"]) >= EPOCHS:
            raise RuntimeError("epoch 24 is complete; run --stage score instead")
        model.load_state_dict(state["raw_state_dict"]); optimizer.load_state_dict(state["optimizer"]); ema.load_state_dict(state["ema"]); _restore_rng(state["rng"], device)
        step, first_epoch = int(state["global_step"]), int(state["epoch"]) + 1
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=legacy._collate, num_workers=0)
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train(); losses: list[float] = []
        for batch_id, (x, y, sessions) in enumerate(loader):
            if any(session != sessions[0] for session in sessions):
                raise RuntimeError("M1 source sampler produced mixed-session batch")
            step += 1
            lr = warmup_cosine_lr(step, total_steps=EPOCHS * UPDATES_PER_EPOCH, warmup_steps=UPDATES_PER_EPOCH, peak=LR, min_factor=plan.LR_MIN_FACTOR)
            for group in optimizer.param_groups: group["lr"] = lr
            keep_rng = torch.Generator(device="cpu"); keep_rng.manual_seed(unit_dropout_seed(args.seed, epoch, batch_id))
            keep = whole_unit_dropout(banks[sessions[0]].unit_mask, p=0.1, generator=keep_rng)
            valid = torch.ones((len(x), CONTEXT), dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(x.float().to(device), banks[sessions[0]], dropout_keep=keep, input_valid_mask=valid)
                loss = nn.functional.mse_loss(prediction.float(), y.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss epoch={epoch} batch={batch_id}")
            loss.backward(); grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)); grad_finite = True
            if smoke:
                grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not grad_finite: raise FloatingPointError("nonfinite concat gradient")
            optimizer.step(); ema.update_after_step(model)
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 100 == 0:
                _atomic_json(dest / "heartbeat.json", {"status": "TRAINING", "pid": os.getpid(), "event": "step", "epoch": epoch, "global_step": step, "loss": losses[-1], "lr": lr, "grad_norm": grad_norm, "elapsed_seconds": time.monotonic() - started, "utc": datetime.now(timezone.utc).isoformat()})
            if smoke and step >= args.max_updates_smoke:
                break
        postupdate = None
        if smoke and step >= args.max_updates_smoke:
            # Check the *updated* live encoder path, not merely the frozen init.
            postupdate = {"full_vs_stream_startup_truezero": _assert_full_stream_parity(model, _material, device),
                          "repeatable_eval": _assert_ho_repeatable(model, ema, _material, device),
                          "live_encoder_parameter_sha256": _initialization_sha(model.encoder)}
        checkpoint = _checkpoint_payload(model, optimizer, ema, epoch=epoch, step=step, smoke=smoke, meta=meta, device=device)
        checkpoint_sha = _atomic_checkpoint(dest / f"epoch_{epoch:03d}.pt", checkpoint)
        row = {"status": "SMOKE" if smoke else "TRAINING", "event": "epoch", "epoch": epoch, "global_step": step, "train_mse": float(np.mean(losses)), "checkpoint_sha256": checkpoint_sha, "utc": datetime.now(timezone.utc).isoformat()}
        _atomic_json(dest / "heartbeat.json", row); _append_jsonl(dest / "metrics.jsonl", row)
        if smoke:
            _atomic_json(dest / "smoke_receipt.json", {"schema": "m1_rift_joint_smoke_receipt_v1", "status": "COMPLETED", "cell": CELL,
                         "arm": args.arm, "seed": args.seed, "sampler_seed": SEED, "steps": step,
                         "checkpoint_sha256": checkpoint_sha, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"],
                         "b3s": meta["b3s"], "ho_contract": smoke_ho_contract, "cuda_initialized": torch.cuda.is_initialized(), "finite_loss": bool(np.isfinite(losses[-1])),
                         "all_gradients_finite": grad_finite, "preflight": smoke_preflight, "postupdate": postupdate})
            return {"status": "SMOKE_COMPLETED", "steps": step}
        if len(losses) != UPDATES_PER_EPOCH:
            raise RuntimeError(f"epoch {epoch} had {len(losses)} updates, expected {UPDATES_PER_EPOCH}")
    if step != EPOCHS * UPDATES_PER_EPOCH:
        raise RuntimeError(f"formal total updates {step} != {EPOCHS * UPDATES_PER_EPOCH}")
    _atomic_json(dest / "train_receipt.json", {"schema": "m1_rift_joint_train_receipt_v1", "status": "COMPLETED", "cell": CELL,
                 "arm": args.arm, "seed": args.seed, "sampler_seed": SEED, "epochs": EPOCHS, "steps": step,
                 "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"],
                 "post_training_scoring_required": True})
    return {"status": "TRAIN_COMPLETED", "steps": step}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    dest = args.dest.resolve(); meta = json.loads((dest / "run_meta.json").read_text())
    if meta.get("status") != "FORMAL" or meta.get("cell") != CELL or meta.get("arm") != args.arm or meta.get("seed") != args.seed or meta.get("source_hashes") != _source_hashes() or meta.get("b3s") != _b3s_provenance():
        raise RuntimeError("score requires matching formal M1 RIFT run")
    if meta.get("source_contract", {}).get("total_windows") != EXPECTED_WINDOWS:
        raise RuntimeError("score source contract mismatch")
    receipt = json.loads((dest / "train_receipt.json").read_text())
    required_receipt = {"schema": "m1_rift_joint_train_receipt_v1", "status": "COMPLETED", "cell": CELL, "arm": args.arm,
                        "seed": args.seed, "sampler_seed": SEED, "epochs": EPOCHS, "steps": EPOCHS * UPDATES_PER_EPOCH,
                        "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"]}
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise RuntimeError("score requires a completed matching formal train receipt")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device); material = _ho_material(); model = _decoder(device,args.arm,args.seed)
    model.install_session_memory({s:material[s]['bank'] for s in HO},{s:material[s]['calib10'] for s in HO}); model.to(device)
    ema = DecoderEMA(model, decay=plan.EMA_DECAY); ho_contract = _ho_contract(material)
    progress_path = dest / "score_progress.json"
    initial_progress = {"schema": "m1_rift_joint_score_progress_v1", "cell": CELL, "arm": args.arm, "seed": args.seed,
                        "sampler_seed": SEED, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"],
                        "b3s": meta["b3s"], "ho_contract": ho_contract, "completed": {}}
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else initial_progress
    if any(progress.get(key) != initial_progress[key] for key in ("schema", "cell", "arm", "seed", "sampler_seed", "source_hashes", "source_contract", "b3s", "ho_contract")):
        raise RuntimeError("score progress provenance mismatch")
    completed = progress.get("completed", {})
    if not isinstance(completed, dict) or not set(completed).issubset({str(epoch) for epoch in range(1, EPOCHS + 1)}):
        raise RuntimeError("malformed score progress")
    repeat: dict[str, Any] | None = None
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"; state = _validate_checkpoint(path, meta, expected_epoch=epoch); checkpoint_sha = _sha_file(path)
        model.load_state_dict(state["raw_state_dict"]); ema.load_state_dict(state["ema"])
        if repeat is None:
            repeat = _assert_ho_repeatable(model, ema, material, device)
            repeat["full_vs_stream_parity"] = _assert_full_stream_parity(model, material, device)
        previous = completed.get(str(epoch))
        if previous is not None:
            if previous.get("checkpoint_sha256") != checkpoint_sha:
                raise RuntimeError(f"checkpoint hash drift after scored epoch {epoch}")
            _validate_scored_report(previous.get("ema_ho_calib", {}))
            continue
        report = _with_ema_eval(model, ema, lambda: _ho_score(model, material, device))
        _validate_scored_report(report)
        completed[str(epoch)] = {"checkpoint_sha256": checkpoint_sha, "ema_ho_calib": report}
        _atomic_json(progress_path, {**initial_progress, "status": "SCORING", "completed": completed, "last_completed_epoch": epoch, "repeatability": repeat})
        _atomic_json(dest / "heartbeat.json", {"status": "SCORING", "event": "ho_calib_score", "epoch": epoch, "completed_epochs": len(completed), "utc": datetime.now(timezone.utc).isoformat()})
    if set(completed) != {str(epoch) for epoch in range(1, EPOCHS + 1)}:
        raise RuntimeError("score scan did not produce exactly 24 epochs")
    scores = {epoch: row["ema_ho_calib"] for epoch, row in completed.items()}
    best = min(range(1, EPOCHS + 1), key=lambda epoch: (-scores[str(epoch)]["equal_session_mean"], epoch))
    receipt = {"schema": "m1_rift_joint_ho_calib_epoch_scan_v1", "status": "COMPLETED", "cell": CELL, "arm": args.arm,
               "seed": args.seed, "sampler_seed": SEED, "source_hashes": meta["source_hashes"], "source_contract": meta["source_contract"], "b3s": meta["b3s"], "ema_by_epoch": scores,
               "checkpoint_sha256_by_epoch": {epoch: completed[epoch]["checkpoint_sha256"] for epoch in sorted(completed, key=int)},
               "selection": {"epoch": best, "rule": "earliest maximum equal-session mean EMA"}, "repeatability": repeat, "ho_contract": ho_contract, "official_test_used": False}
    _atomic_json(dest / "score_receipt.json", receipt); _atomic_json(dest / "ho_calib_epoch_scan.json", receipt)
    _atomic_json(dest / "heartbeat.json", {"status": "COMPLETED", "event": "score_complete", "epoch": best, "completed_epochs": EPOCHS, "utc": datetime.now(timezone.utc).isoformat()})
    return {"status": "SCORE_COMPLETED", "best_epoch": best}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    from btransform_unified_v2.joint_m1_model import ARMS
    parser.add_argument("--dest", required=True, type=Path); parser.add_argument("--arm", choices=ARMS, required=True); parser.add_argument("--seed",type=int,choices=(42,43,44),default=42); parser.add_argument("--stage", choices=("train", "score", "all"), default="train")
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates-smoke", type=int); parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.epochs > EPOCHS: parser.error("--epochs must be 1..24")
    if args.max_updates_smoke is not None and (args.max_updates_smoke < 1 or args.stage != "train"): parser.error("smoke requires positive updates and --stage train")
    if args.max_updates_smoke is None and args.stage == "train" and args.epochs != EPOCHS: parser.error("formal M1 RIFT requires exactly 24 epochs")
    if args.stage == "all": parser.error("formal training and held-out scoring are separate stages; run --stage score after training")
    if args.resume is not None and args.stage != "train": parser.error("--resume is valid only with --stage train")
    result = run_train(args) if args.stage == "train" else run_score(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
