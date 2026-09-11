#!/usr/bin/env python3
"""Isolated current-H1 signed_state14 R300 flat control.

This runner deliberately imports the current signed-state training builder; it
does not route through the historical H1 pair recipe.  It is a fresh-result
family and writes only under results/recency_flat_ablation_v1.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = ROOT / "scripts" / "h1_signed_state_r300_v1"
CURRENT_TRAIN = CURRENT_DIR / "train.py"
RESULT_ROOT = ROOT / "results" / "recency_flat_ablation_v1"
FORMAL_DEST = RESULT_ROOT / "formal_h1_signedstate14_flat_s42"
SMOKE_DEST = RESULT_ROOT / "root_smoke_h1_flat_s42"
REF_DEST = ROOT / "results" / "h1_signed_state_r300_v1" / "recency_s42_formal_20260909"
REF_BANKS = ROOT / "results" / "h1_signed_state_r300_v1" / "banks_official13_20260909"
EPOCHS, UPDATES, SEED, BATCH, MICRO = 32, 731, 42, 32, 32


def _load_current():
    # Current train.py imports its sibling common.py/build_banks.py by name.
    sys.path.insert(0, str(CURRENT_DIR))
    spec = importlib.util.spec_from_file_location("_signed_state14_current", CURRENT_TRAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load current signed-state trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


signed = _load_current()
np, torch, nn, ht = signed.np, signed.torch, signed.nn, signed.ht


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def params_sha(model: Any) -> str:
    """Shared trainable state only; slope buffers intentionally excluded."""
    return ht._sha_state(model)


def slopes(model: Any, label: str, *, flat: bool) -> str:
    value = dict(model.named_buffers()).get("temporal.recency_slopes")
    if value is None or tuple(value.shape) != (8,) or value.dtype != torch.float32:
        raise RuntimeError(f"{label}: recency_slopes must be float32[8]")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{label}: nonfinite recency_slopes")
    if flat and not bool(torch.equal(value.detach().cpu(), torch.zeros(8, dtype=torch.float32))):
        raise RuntimeError(f"{label}: flat recency_slopes are not exact zero")
    if not flat and bool(torch.equal(value.detach().cpu(), torch.zeros(8, dtype=torch.float32))):
        raise RuntimeError(f"{label}: recency slopes unexpectedly all zero")
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    paths = [
        Path(__file__), CURRENT_TRAIN, CURRENT_DIR / "common.py", CURRENT_DIR / "build_banks.py",
        ROOT / "scripts" / "rift_v1" / "h1_train.py",
        ROOT / "src" / "btransform_unified_v2" / "model.py",
        ROOT / "src" / "btransform_unified_v2" / "temporal.py",
        ROOT / "src" / "btransform_unified_v2" / "config.py",
    ]
    return {str(path): sha_file(path) for path in paths}


def _read_epoch_rows(path: Path, *, expected_epochs: int) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing epoch metrics: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    epoch_rows = [row for row in rows if row.get("event") == "epoch"]
    by_epoch = {int(row.get("epoch", -1)): row for row in epoch_rows}
    if len(by_epoch) != len(epoch_rows) or set(by_epoch) != set(range(1, expected_epochs + 1)):
        raise RuntimeError(f"epoch metric coverage drift: {path}")
    return by_epoch


def trainable_names(model: Any) -> set[str]:
    named = model.trainable_parameters() if hasattr(model, "trainable_parameters") else ((name, value) for name, value in model.named_parameters() if value.requires_grad)
    return set(dict(named))


def _ema_shadow_names(state: dict[str, Any], trainable_names: set[str], label: str, expected_updates: int) -> None:
    ema_state = state.get("ema")
    if not isinstance(ema_state, dict) or set(ema_state) != {"decay", "n_updates", "shadow"}:
        raise RuntimeError(f"{label}: malformed DecoderEMA state")
    if float(ema_state["decay"]) != 0.9995 or int(ema_state["n_updates"]) != expected_updates:
        raise RuntimeError(f"{label}: DecoderEMA decay/update drift")
    shadow = ema_state["shadow"]
    if not isinstance(shadow, dict) or set(shadow) != trainable_names:
        raise RuntimeError(f"{label}: DecoderEMA shadow trainable-name drift")
    if "temporal.recency_slopes" in shadow:
        raise RuntimeError(f"{label}: DecoderEMA must not contain slope buffers")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in shadow.values()):
        raise RuntimeError(f"{label}: invalid DecoderEMA shadow tensor")


def _raw_flat_slopes(state: dict[str, Any], label: str) -> str:
    raw = state.get("raw_state_dict")
    slope = raw.get("temporal.recency_slopes") if isinstance(raw, dict) else None
    if slope is None or tuple(slope.shape) != (8,) or slope.dtype != torch.float32 or not bool(torch.equal(slope.cpu(), torch.zeros(8, dtype=torch.float32))):
        raise RuntimeError(f"{label}: raw flat slope buffer drift")
    return hashlib.sha256(slope.cpu().numpy().tobytes()).hexdigest()


def validate_reference(banks: Path) -> dict[str, Any]:
    meta_path, receipt_path = REF_DEST / "run_meta.json", banks / "receipt.json"
    meta = json.loads(meta_path.read_text()); receipt = json.loads(receipt_path.read_text())
    if meta.get("schema") != "rift_h1_signed_state_r300_v1" or meta.get("status") != "FORMAL" or meta.get("variant") != "recency":
        raise RuntimeError("reference is not completed current signed_state14 formal recency")
    required = {"context_bins": 300, "seed": SEED, "epochs": EPOCHS, "updates_per_epoch": UPDATES,
                "batch": BATCH, "microbatch": MICRO, "attention_backend": "dense", "ema": 0.9995,
                "unit_dropout": 0.1, "peak_lr": 1e-4, "floor_lr": 1e-5, "warmup_epochs": 1}
    if any(meta.get(k) != v for k, v in required.items()):
        raise RuntimeError("reference frozen recipe/optimizer/EMA/dropout drift")
    if sha_file(receipt_path) != meta.get("banks_receipt_sha256") or banks.resolve() != REF_BANKS.resolve():
        raise RuntimeError("flat run requires the exact current signed-state frozen banks")
    train_receipt = json.loads((REF_DEST / "train_receipt.json").read_text())
    if train_receipt.get("status") != "COMPLETED" or train_receipt.get("epochs") != EPOCHS or train_receipt.get("updates") != EPOCHS * UPDATES:
        raise RuntimeError("reference formal training is incomplete")
    selection = json.loads((REF_DEST / "ho_m3_selection.json").read_text())
    curve = selection.get("curve", [])
    if selection.get("status") != "HO_M3_DEVELOPMENT_SELECTION" or len(curve) != EPOCHS or {row.get("epoch") for row in curve} != set(range(1, EPOCHS + 1)):
        raise RuntimeError("reference selection is incomplete")
    metric_rows = _read_epoch_rows(REF_DEST / "metrics.jsonl", expected_epochs=EPOCHS)
    reference_decoder = ht._decoder("recency", torch.device("cpu"), 300, "dense")
    reference_trainable_names = trainable_names(reference_decoder)
    del reference_decoder
    for epoch, row in metric_rows.items():
        step = epoch * UPDATES
        if row.get("smoke") or int(row.get("global_step", -1)) != step or not row.get("endpoint_bank_mask_sequence_sha256"):
            raise RuntimeError(f"reference metrics epoch {epoch} drift")
        state = torch.load(REF_DEST / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=False)
        raw = state.get("raw_state_dict", {}); slope = raw.get("temporal.recency_slopes")
        if state.get("smoke") or state.get("epoch") != epoch or state.get("global_step") != step or slope is None or tuple(slope.shape) != (8,) or slope.dtype != torch.float32 or bool(torch.equal(slope.cpu(), torch.zeros(8, dtype=torch.float32))):
            raise RuntimeError(f"reference checkpoint epoch {epoch} identity/slope drift")
        _ema_shadow_names(state, reference_trainable_names, f"reference epoch {epoch}", step)
        optimizer = state.get("optimizer", {}); groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
        expected_lr = signed.warmup_cosine_lr(step, EPOCHS * UPDATES, UPDATES, peak=1e-4, min_factor=.1)
        if len(groups) != 1 or abs(float(groups[0].get("lr", -1.0)) - expected_lr) > 1e-14 or groups[0].get("betas") != (0.9, 0.999) or float(groups[0].get("weight_decay", -1.0)) != 0.01 or float(groups[0].get("eps", -1.0)) != 1e-8:
            raise RuntimeError(f"reference checkpoint epoch {epoch} optimizer/LR drift")
    current = source_manifest(); ref_sources = meta.get("source_manifest_sha256", {})
    for path in (ROOT / "scripts" / "rift_v1" / "h1_train.py", ROOT / "src" / "btransform_unified_v2" / "model.py", ROOT / "src" / "btransform_unified_v2" / "temporal.py", ROOT / "src" / "btransform_unified_v2" / "config.py"):
        if ref_sources.get(str(path)) != current[str(path)]:
            raise RuntimeError(f"reference shared source drift: {path}")
    return {"reference_run_meta_sha256": sha_file(meta_path), "reference_banks_receipt_sha256": sha_file(receipt_path),
            "reference_recipe": required, "reference_pairing_digests": meta["pairing_digests"],
            "reference_initialization_sha256": meta["initialization_sha256"],
            "reference_metrics_sha256": sha_file(REF_DEST / "metrics.jsonl"),
            "reference_endpoint_sequences": {str(epoch): row["endpoint_bank_mask_sequence_sha256"] for epoch, row in metric_rows.items()}}

def contract_hash(ho: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ho["keys"]:
        h.update(str(key).encode())
        for name in ("X", "valid", "y"):
            array = np.ascontiguousarray(ho[name][key])
            h.update(name.encode()); h.update(str(array.shape).encode()); h.update(array.dtype.str.encode()); h.update(array.tobytes())
        bank = ho["banks"][key]
        h.update(str(bank.calibration_meta.get("array_sha256")).encode())
        h.update(str(bank.calibration_meta.get("carrier_sha256")).encode())
    return h.hexdigest()


def checkpoint_binding(dest: Path, epoch: int, step: int, model: Any, ema: Any, meta: dict[str, Any]) -> None:
    path = dest / f"epoch_{epoch:03d}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    slope_sha = _raw_flat_slopes(state, f"epoch {epoch}")
    _ema_shadow_names(state, trainable_names(model), f"epoch {epoch}", step)
    buffer_before = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    raw_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    ema.apply_to(model)
    if any(not torch.equal(value.detach().cpu(), buffer_before[name]) for name, value in model.named_buffers()):
        raise RuntimeError(f"epoch {epoch}: applying EMA mutated a model buffer")
    with torch.no_grad():
        for name, value in raw_before.items():
            dict(model.named_parameters())[name].copy_(value)
    if state.get("variant") != "flat" or state.get("epoch") != epoch or state.get("global_step") != step or state.get("smoke") != bool(meta["smoke"]):
        raise RuntimeError(f"epoch {epoch}: checkpoint identity drift")
    atomic(path.with_suffix(".pt.binding.json"), {"schema": "h1_signedstate14_flat_checkpoint_binding_v2", "epoch": epoch, "global_step": step,
        "variant": "flat", "checkpoint_sha256": sha_file(path), "run_meta_sha256": sha_file(dest / "run_meta.json"),
        "source_manifest_sha256": meta["source_manifest_sha256"], "reference": meta["reference"],
        "banks_receipt_sha256": meta["banks_receipt_sha256"], "pairing_digests": meta["pairing_digests"],
        "raw_slope_sha256": slope_sha, "ema_shadow_names_sha256": hashlib.sha256("\n".join(sorted(dict(state["ema"]["shadow"]))).encode()).hexdigest(),
        "ema_buffer_policy": "DecoderEMA shadows trainable parameters only; model buffers remain raw/flat."})


def shared_validate(args: Any, meta: dict[str, Any], reference: dict[str, Any], model: Any) -> None:
    frozen = {"schema": "h1_signedstate14_flat_v1", "status": "FORMAL", "smoke": False, "variant": "flat", "context_bins": 300,
              "attention_backend": "dense", "seed": SEED, "batch": BATCH, "microbatch": MICRO, "updates_per_epoch": UPDATES, "epochs": EPOCHS,
              "ema": 0.9995, "unit_dropout": 0.1}
    if any(meta.get(k) != v for k, v in frozen.items()) or meta.get("source_manifest_sha256") != source_manifest():
        raise RuntimeError("formal flat metadata/source/config drift")
    if meta.get("banks_receipt_sha256") != sha_file(args.banks / "receipt.json") or meta.get("reference") != reference:
        raise RuntimeError("formal flat bank/reference binding drift")
    if meta.get("pairing_digests") != reference["reference_pairing_digests"]:
        raise RuntimeError("formal flat pairing binding drift")
    for epoch in range(1, EPOCHS + 1):
        path = args.dest / f"epoch_{epoch:03d}.pt"; bind_path = path.with_suffix(".pt.binding.json")
        if not path.is_file() or not bind_path.is_file():
            raise RuntimeError(f"missing formal checkpoint/binding epoch {epoch}")
        binding = json.loads(bind_path.read_text())
        if binding.get("checkpoint_sha256") != sha_file(path) or binding.get("run_meta_sha256") != sha_file(args.dest / "run_meta.json") or binding.get("global_step") != epoch * UPDATES or binding.get("source_manifest_sha256") != meta["source_manifest_sha256"] or binding.get("banks_receipt_sha256") != meta["banks_receipt_sha256"] or binding.get("pairing_digests") != meta["pairing_digests"] or binding.get("reference") != reference:
            raise RuntimeError(f"epoch {epoch}: binding step/provenance drift")
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected_state = {"schema": "rift_h1_context_train_v1", "variant": "flat", "context_bins": 300, "attention_backend": "dense", "microbatch": MICRO, "smoke": False, "epoch": epoch, "global_step": epoch * UPDATES}
        if any(state.get(key) != value for key, value in expected_state.items()):
            raise RuntimeError(f"epoch {epoch}: formal checkpoint schema/config/step/smoke drift")
        _raw_flat_slopes(state, f"formal epoch {epoch}"); _ema_shadow_names(state, trainable_names(model), f"formal epoch {epoch}", epoch * UPDATES)

def score(args: Any, carriers: dict[str, Any] | None = None) -> dict[str, Any]:
    torch.set_num_threads(2)
    dest = args.dest; meta = json.loads((dest / "run_meta.json").read_text())
    reference = validate_reference(args.banks)
    if carriers is None:
        _, _, carriers = signed._load_banks(args.banks)
    device = torch.device(args.device); model = ht._decoder("flat", device, 300, "dense"); slopes(model, "score-init", flat=True)
    shared_validate(args, meta, reference, model)
    ema = signed.DecoderEMA(model, decay=0.9995); ho = signed.build_ho_signed(300, carriers); input_sha = contract_hash(ho)
    progress_path = dest / "score_progress.json"; prior = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    if prior and (prior.get("input_contract_sha256") != input_sha or prior.get("run_meta_sha256") != sha_file(dest / "run_meta.json") or prior.get("reference") != reference):
        raise RuntimeError("score progress provenance drift")
    curve = list(prior.get("curve", [])); done = {int(row["epoch"]) for row in curve}
    if len(done) != len(curve) or not done.issubset(set(range(1, EPOCHS + 1))):
        raise RuntimeError("score progress epoch coverage drift")
    score_started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        if epoch in done: continue
        path = dest / f"epoch_{epoch:03d}.pt"; state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["raw_state_dict"], strict=True); slopes(model, f"score-raw-e{epoch}", flat=True); ema.load_state_dict(state["ema"])
        report = signed.score_ho_m3(model, ema, ho, device)
        curve.append({"epoch": epoch, "epoch_zero_based": epoch - 1, signed.HO_SELECTION_METRIC: report["r2_mean"], "worst_session_r2": report["worst_session_r2"], "session_std_population": report["r2_std_population"], "per_session_r2": report["per_session_r2"]})
        curve.sort(key=lambda row: row["epoch"])
        atomic(progress_path, {"schema": "h1_signedstate14_flat_score_progress_v1", "status": "IN_PROGRESS", "run_meta_sha256": sha_file(dest / "run_meta.json"), "reference": reference, "input_contract_sha256": input_sha, "curve": curve, "last_completed_epoch": epoch})
        atomic(dest / "heartbeat.json", {"event": "score", "epoch": epoch, "epochs": EPOCHS, "elapsed_seconds": time.monotonic() - score_started})
    if len(curve) != EPOCHS: raise RuntimeError("score did not cover all formal epochs")
    selected = signed.select_epoch(curve); elapsed = time.monotonic() - score_started
    atomic(dest / "ho_m3_selection.json", {"status": "HO_M3_DEVELOPMENT_SELECTION", "selected": selected, "curve": curve, "current_query_input_contract_sha256": input_sha, "score_elapsed_seconds": elapsed, "selection_rule": "same current HO-M3 grouped-seven, all32, earliest maximum"})
    atomic(progress_path, {"schema": "h1_signedstate14_flat_score_progress_v1", "status": "COMPLETED", "run_meta_sha256": sha_file(dest / "run_meta.json"), "reference": reference, "input_contract_sha256": input_sha, "curve": curve, "last_completed_epoch": EPOCHS})
    timing = json.loads((dest / "timing.json").read_text()); timing["score_elapsed_seconds"] = elapsed; atomic(dest / "timing.json", timing)
    atomic(dest / "train_receipt.json", {"status": "COMPLETED", "epochs": EPOCHS, "updates": EPOCHS * UPDATES, "selected_epoch": selected["epoch"], "official_test_used": False, "train_elapsed_seconds": timing["train_elapsed_seconds"], "score_elapsed_seconds": elapsed})
    return {"status": "SCORE_COMPLETED", "selected_epoch": selected["epoch"], "score_elapsed_seconds": elapsed}


def runtime_parity(model: Any, train_data: dict[str, Any], cal1: dict[str, Any], device: Any, label: str) -> dict[str, Any]:
    """Small inference-only dense/local/stream check using one real prefix and valid mask."""
    from btransform_unified_v2.streaming import RiftStreamDecoder
    session = train_data["sessions"][0]; budget = int(signed.prefix_schedule(0, UPDATES)[0]); starts = signed.cal1_b2.legal_starts(cal1["starts"][session], cal1["n_trials"][session], budget)
    bank = cal1["banks"][(session, signed.pick_m7_start(session, epoch0=0, step=0, starts=starts), budget)]
    xb = torch.from_numpy(train_data["X"][session][:1]).to(device); valid = torch.from_numpy(train_data["valid"][session][:1]).to(device)
    was = model.training; model.eval(); before = {name: value.detach().clone() for name, value in model.named_parameters()}
    try:
        model.temporal.set_attention_backend("dense")
        with torch.inference_mode(): dense = ht._forward(model, xb, bank, None, valid)
        model.temporal.set_attention_backend("local")
        with torch.inference_mode(): local = ht._forward(model, xb, bank, None, valid)
        stream = RiftStreamDecoder(model)
        for index in range(xb.shape[1]):
            got = stream.stream_step(xb[:, index], bank, ["flat-parity"], valid_mask=valid[:, index])
        stream_last = stream.predict("flat-parity")
        if not torch.allclose(dense, local, rtol=2e-4, atol=2e-5) or not torch.allclose(local[0], stream_last, rtol=2e-4, atol=2e-5):
            raise RuntimeError(f"{label}: dense/local/stream prefix parity drift")
        return {"label": label, "prefix_bins": int(xb.shape[1]), "valid_bins": int(valid.sum().item()), "dense_local_max_abs": float((dense-local).abs().max().cpu()), "local_stream_max_abs": float((local[0]-stream_last).abs().max().cpu())}
    finally:
        model.temporal.set_attention_backend("dense"); model.train(was)
        with torch.no_grad():
            for name, value in before.items(): dict(model.named_parameters())[name].copy_(value)

def train(args: Any) -> dict[str, Any]:
    reference = validate_reference(args.banks)
    device = torch.device(args.device)
    torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dest = args.dest
    if dest.exists() and any(dest.iterdir()) and not args.resume:
        raise FileExistsError(f"fresh destination required: {dest}")
    existing = None
    if args.resume:
        if not args.resume.is_file() or not (dest / "run_meta.json").is_file():
            raise RuntimeError("--resume requires an existing own flat checkpoint and destination")
        existing = json.loads((dest / "run_meta.json").read_text())
        frozen = {"schema": "h1_signedstate14_flat_v1", "status": "FORMAL", "smoke": False, "variant": "flat", "context_bins": 300,
                  "attention_backend": "dense", "seed": SEED, "batch": BATCH, "microbatch": MICRO,
                  "updates_per_epoch": UPDATES, "epochs": EPOCHS, "banks_receipt_sha256": sha_file(args.banks / "receipt.json")}
        if args.smoke_steps is not None:
            raise RuntimeError("smoke checkpoints are not resumable into a formal flat run")
        if args.resume.resolve().parent != dest.resolve():
            raise RuntimeError("resume checkpoint must be an epoch checkpoint owned by --dest")
        if any(existing.get(k) != v for k, v in frozen.items()) or existing.get("source_manifest_sha256") != source_manifest() or existing.get("reference") != reference or existing.get("pairing_digests") != reference["reference_pairing_digests"]:
            raise RuntimeError("resume frozen metadata/reference/pairing binding drift")
    plan, bank_receipt, carriers = signed._load_banks(args.banks)
    train_data = ht.build_train(300); cal1 = signed.build_cal1_signed(train_data["banks"], plan)
    digests = ht._digest_train_contract(train_data, cal1)
    if digests != reference["reference_pairing_digests"]:
        raise RuntimeError("flat endpoint/bank/valid-mask contract differs from current recency reference")
    flat = ht._decoder("flat", device, 300, "dense"); flat_sha = params_sha(flat); flat_slope_sha = slopes(flat, "flat-init", flat=True)
    recency = ht._decoder("recency", device, 300, "dense")
    if params_sha(recency) != flat_sha or params_sha(recency) != reference["reference_initialization_sha256"]:
        raise RuntimeError("shared named-parameter initialization is not paired to current recency")
    slopes(recency, "fresh-recency-init", flat=False); del recency
    ema = signed.DecoderEMA(flat, decay=0.9995)
    opt = torch.optim.AdamW(flat.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(SEED); dest.mkdir(parents=True, exist_ok=True)
    smoke = args.smoke_steps is not None
    meta = {"schema": "h1_signedstate14_flat_v1", "cell": "CURRENT_H1_SIGNEDSTATE14_R300_FLAT_S42",
            "status": "SMOKE" if smoke else "FORMAL", "smoke": smoke, "variant": "flat", "context_bins": 300,
            "attention_backend": "dense", "epochs": EPOCHS if not smoke else 1, "updates_per_epoch": UPDATES,
            "batch": BATCH, "microbatch": MICRO, "seed": SEED, "ema": 0.9995, "unit_dropout": 0.1,
            "initialization_shared_named_parameters_sha256": flat_sha, "flat_recency_slopes_sha256": flat_slope_sha,
            "banks_receipt_sha256": sha_file(args.banks / "receipt.json"), "pairing_digests": digests,
            "source_manifest_sha256": source_manifest(), "reference": reference, "official_test_used": False,
            "selection_surface": "HO-M3 development only; score all 32 after training", "launch": {"argv": sys.argv, "pid": os.getpid()},
            "utc": datetime.now(timezone.utc).isoformat()}
    if existing is None:
        atomic(dest / "run_meta.json", meta)
    else:
        meta = existing
    parity_pre = runtime_parity(flat, train_data, cal1, device, "pre-train")
    atomic(dest / "runtime_parity_pre.json", parity_pre)
    total = EPOCHS * UPDATES; step = 0; first_epoch = 1; started = time.monotonic()
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        binding_path = args.resume.with_suffix(".pt.binding.json")
        if not binding_path.is_file():
            raise RuntimeError("resume checkpoint lacks own provenance binding")
        binding = json.loads(binding_path.read_text())
        if binding.get("checkpoint_sha256") != sha_file(args.resume) or binding.get("run_meta_sha256") != sha_file(dest / "run_meta.json") or binding.get("reference") != reference or binding.get("pairing_digests") != digests:
            raise RuntimeError("resume checkpoint binding/reference/pairing drift")
        if state.get("variant") != "flat" or state.get("context_bins") != 300 or state.get("attention_backend") != "dense" or state.get("microbatch") != MICRO or state.get("smoke"):
            raise RuntimeError("resume checkpoint identity drift")
        if int(state.get("global_step", -1)) != int(state.get("epoch", -1)) * UPDATES or binding.get("global_step") != state.get("global_step"):
            raise RuntimeError("resume checkpoint must end at a frozen complete epoch")
        _raw_flat_slopes(state, "resume-raw"); _ema_shadow_names(state, trainable_names(flat), "resume", int(state["global_step"]))
        flat.load_state_dict(state["raw_state_dict"], strict=True); slopes(flat, "resume-raw", flat=True)
        ema.load_state_dict(state["ema"]); opt.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]; torch.set_rng_state(state["torch_rng_cpu"].cpu()); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("torch_rng_cuda") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_rng_cuda"]])
        step, first_epoch = int(state["global_step"]), int(state["epoch"]) + 1
        if first_epoch > EPOCHS: raise RuntimeError("resume checkpoint already completes formal run")
    for epoch in range(first_epoch, EPOCHS + 1):
        flat.train(); losses = []; schedule = signed.prefix_schedule(epoch - 1, UPDATES); order = list(train_data["sessions"]); rng.shuffle(order); si = 0; pairing = hashlib.sha256()
        for session in order:
            index = rng.permutation(len(train_data["X"][session]))
            for off in range(0, len(index), BATCH):
                take = index[off:off + BATCH]; budget = int(schedule[si]); starts = signed.cal1_b2.legal_starts(cal1["starts"][session], cal1["n_trials"][session], budget)
                bank = cal1["banks"][(session, signed.pick_m7_start(session, epoch0=epoch - 1, step=si, starts=starts), budget)]
                pairing.update(session.encode()); pairing.update(np.asarray(train_data["ids"][session][take], np.int64).tobytes()); pairing.update(str(bank.calibration_meta.get("array_sha256")).encode())
                step += 1; opt.param_groups[0]["lr"] = signed.warmup_cosine_lr(step, total, UPDATES, peak=1e-4, min_factor=.1)
                keep = signed.whole_unit_dropout(torch.from_numpy(bank.unit_mask.copy()), p=.1, generator=torch.Generator().manual_seed(signed.unit_dropout_seed(SEED, epoch, si))); pairing.update(keep.numpy().tobytes())
                opt.zero_grad(set_to_none=True); batch_loss = 0.0
                for moff in range(0, len(take), MICRO):
                    subset = take[moff:moff + MICRO]; xb = torch.from_numpy(train_data["X"][session][subset]).to(device); valid = torch.from_numpy(train_data["valid"][session][subset]).to(device); yb = torch.from_numpy(train_data["y"][session][subset] * signed.SCALE).to(device)
                    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                    with amp: loss = nn.functional.mse_loss(ht._forward(flat, xb, bank, keep, valid).float(), yb)
                    if not bool(torch.isfinite(loss)): raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
                    (loss * (len(subset) / len(take))).backward(); batch_loss += float(loss.detach()) * len(subset) / len(take)
                grad = float(nn.utils.clip_grad_norm_(flat.parameters(), 1., error_if_nonfinite=True)); opt.step(); ema.update_after_step(flat); losses.append(batch_loss); si += 1
                if step == 1 or step % 50 == 0:
                    atomic(dest / "heartbeat.json", {"event": "step", "epoch": epoch, "global_step": step, "loss": batch_loss, "grad_norm": grad, "elapsed_seconds": time.monotonic() - started})
                if args.smoke_steps and step >= args.smoke_steps: break
            if args.smoke_steps and step >= args.smoke_steps: break
        ht._checkpoint(dest / f"epoch_{epoch:03d}.pt", flat, opt, ema, epoch, step, rng, variant="flat", context_bins=300, attention_backend="dense", microbatch=MICRO, smoke=smoke)
        checkpoint_binding(dest, epoch, step, flat, ema, meta)
        row = {"event": "epoch", "epoch": epoch, "global_step": step, "train_mse": float(np.mean(losses)), "lr": float(opt.param_groups[0]["lr"]), "smoke": smoke, "endpoint_bank_mask_sequence_sha256": pairing.hexdigest()}
        if not smoke:
            if step != epoch * UPDATES or si != UPDATES:
                raise RuntimeError(f"epoch {epoch} update count/step drift")
            if row["endpoint_bank_mask_sequence_sha256"] != reference["reference_endpoint_sequences"][str(epoch)]:
                raise RuntimeError(f"epoch {epoch}: endpoint/bank/mask sequence differs from current recency reference")
        with (dest / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush()
        atomic(dest / "heartbeat.json", row)
        if smoke: break
    train_elapsed = time.monotonic() - started
    atomic(dest / "timing.json", {"train_elapsed_seconds": train_elapsed, "score_elapsed_seconds": None,
                                   "scope": "monotonic train timer excludes post-training HO-M3 scoring"})
    parity_post = runtime_parity(flat, train_data, cal1, device, "post-train")
    atomic(dest / "runtime_parity_post.json", parity_post)
    if smoke:
        receipt = {"schema": "h1_signedstate14_flat_smoke_receipt_v1", "status": "SMOKE_COMPLETED", "steps": step, "finite_loss_and_grad": True, "checkpoint_binding": json.loads((dest / "epoch_001.pt.binding.json").read_text()), "runtime_parity_pre": parity_pre, "runtime_parity_post": parity_post, "train_elapsed_seconds": train_elapsed}
        atomic(dest / "smoke_receipt.json", receipt)
        return {"status": "SMOKE_COMPLETED", "steps": step, "train_elapsed_seconds": train_elapsed}
    if step != total: raise RuntimeError("formal update total drift")
    _read_epoch_rows(dest / "metrics.jsonl", expected_epochs=EPOCHS)
    return score(args, carriers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banks", type=Path, default=REF_BANKS); parser.add_argument("--dest", type=Path)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--smoke1", action="store_true"); parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", type=Path); args = parser.parse_args()
    if args.smoke1:
        if args.smoke_steps not in (None, 1): parser.error("--smoke1 conflicts with --smoke-steps")
        args.smoke_steps = 1
    if args.smoke_steps is not None and args.smoke_steps < 1: parser.error("positive --smoke-steps required")
    args.banks = args.banks.resolve(); args.dest = (args.dest or (SMOKE_DEST if args.smoke_steps else FORMAL_DEST)).resolve()
    if args.stage == "score" and args.smoke_steps: parser.error("score cannot use smoke")
    print(json.dumps(score(args) if args.stage == "score" else train(args), indent=2))


if __name__ == "__main__":
    main()
