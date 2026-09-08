"""Train, preflight, and score strict H1 six-fold LODO Z/B/D RIFT arms.

No target array is opened in ``train`` or ``preflight``.  Only ``score`` opens
prepared target queries, and it emits independently selected and fixed-e32
EMA scores.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, random, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT.parent / "btransform_unified_v1/src", ROOT.parent):
    sys.path.insert(0, str(path))
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import whole_unit_dropout, unit_dropout_seed
from btransform_unified_v1.schedule import warmup_cosine_lr
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v2.cross_session_h1_model import ARMS, ARM_B, ARM_D, ARM_Z, CrossSessionH1Decoder

BATCH, EPOCHS, SCALE, EMA_DECAY = 32, 32, 20.0, 0.9995


def json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_bytes(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepared_array_sha(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def audit_npz(path: Path, row: dict, *, arm: str) -> None:
    """Verify the materialized array payload, not merely manifest assertions."""
    with np.load(path, allow_pickle=False) as z:
        required = {"X", "valid", "y", "ends", "starts", "segment_starts", "support", "query", "native_query_indices", "stride"}
        if not required.issubset(set(z.files)): raise RuntimeError(f"{path}: missing prepared fields")
        X, valid, y = z["X"], z["valid"], z["y"]
        starts, ends, segment_starts = z["starts"], z["ends"], z["segment_starts"]
        support, query, native_indices, stride = z["support"], z["query"], z["native_query_indices"], int(z["stride"])
        if prepared_array_sha(X) != row["X_sha256"] or prepared_array_sha(y) != row["y_sha256"]: raise RuntimeError(f"{path}: X/y hash mismatch")
        if int(stride) != int(row["stride"]) or set(map(float, support)) & set(map(float, query)) or not np.array_equal(native_indices, np.asarray(row["query_indices"])): raise RuntimeError(f"{path}: support/query contract mismatch")
        if prepared_array_sha(ends) != row["endpoint_sha256"] or prepared_array_sha(starts) != row["starts_sha256"] or prepared_array_sha(segment_starts) != row["segment_starts_sha256"]: raise RuntimeError(f"{path}: endpoint hash mismatch")
        if not (len(X)==len(y)==len(valid)==len(starts)==len(ends)==len(segment_starts) and np.all(starts >= segment_starts) and np.all(ends >= starts)):
            raise RuntimeError(f"{path}: invalid endpoint bounds")
        widths = ends - starts + 1
        if np.any(widths > 300) or np.any(valid.sum(axis=1) != widths) or np.any(valid[:, :-1] & ~valid[:, 1:]):
            raise RuntimeError(f"{path}: invalid reset/padding mask")
        if arm in (ARM_B, ARM_D):
            if "activity" not in z.files or prepared_array_sha(z["activity"]) != row["activity_sha256"]: raise RuntimeError(f"{path}: activity binding mismatch")
        if arm == ARM_D:
            if "carrier" not in z.files or prepared_array_sha(z["carrier"]) != row["carrier_sha256"]: raise RuntimeError(f"{path}: carrier binding mismatch")


def audit_surface(root: Path, fold: str, surface: str, arm: str, *, validation: bool = False) -> dict:
    directory = root / fold / surface; manifest = json.loads((directory / "manifest.json").read_text())
    for session, primary in manifest["records"].items():
        row = primary["validation"] if validation else primary
        suffix = ".val.npz" if validation else ".npz"
        audit_npz(directory / f"{session}{suffix}", row, arm=arm)
        if validation and set(map(float, primary["query_trials"])) & set(map(float, row["query_trials"])):
            raise RuntimeError(f"{session}: source gradient/validation trial overlap")
    return manifest


def load_surface(root: Path, fold: str, surface: str, arm: str, *, validation: bool = False):
    directory = root / fold / surface
    manifest = json.loads((directory / "manifest.json").read_text())
    banks: dict[str, TaskBank] = {}
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    activity: dict[str, torch.Tensor] = {}
    records = manifest["records"]
    for session in sorted(records):
        suffix = ".val.npz" if validation else ".npz"
        with np.load(directory / f"{session}{suffix}", allow_pickle=False) as z:
            X = np.ascontiguousarray(z["X"], dtype=np.float32)
            y = np.ascontiguousarray(z["y"], dtype=np.float32)
            valid = np.ascontiguousarray(z["valid"], dtype=np.bool_)
            ends = np.ascontiguousarray(z["ends"], dtype=np.int64)
            # Z deliberately consumes neither target/source M3 nor H-C.  B
            # consumes only M3. D alone reads H-C.
            carrier = np.ascontiguousarray(z["carrier"], dtype=np.float32) if arm == ARM_D else np.zeros((176, 4), np.float32)
            e0 = np.zeros((176, 700), np.float32)
            banks[session] = TaskBank(session, e0, carrier, np.ones(176, bool), X, y, ends,
                {"shape": (176, 700), "trial_count": 3, "budget": 3,
                 "estimator": "fresh random C2 M3 + source-only H-C", "array_sha256": array_sha256(e0)})
            data[session] = (X, y, valid)
            if arm in (ARM_B, ARM_D):
                activity[session] = torch.from_numpy(np.ascontiguousarray(z["activity"], dtype=np.float32))
    return manifest, banks, data, activity


def make_model(arm: str, seed: int, device: torch.device, banks, activity) -> CrossSessionH1Decoder:
    model = CrossSessionH1Decoder(arm, seed=seed).to(device)
    model.install_memory(banks, activity)
    return model


def strict_ema(module: nn.Module, ema: DecoderEMA) -> None:
    parameters = dict(module.named_parameters())
    if set(parameters) != set(ema.shadow):
        raise RuntimeError("EMA key set does not equal the complete unique parameter key set")
    pointers: set[int] = set()
    for name, shadow in ema.shadow.items():
        if tuple(shadow.shape) != tuple(parameters[name].shape) or not torch.isfinite(shadow).all():
            raise RuntimeError(f"EMA invalid tensor {name}")
        if shadow.data_ptr() in pointers:
            raise RuntimeError(f"EMA shadow aliases another tensor: {name}")
        pointers.add(shadow.data_ptr())


def swap_ema(module: nn.Module, ema: DecoderEMA):
    strict_ema(module, ema)
    params = dict(module.named_parameters())
    saved = {name: value.detach().clone() for name, value in params.items()}
    buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    with torch.no_grad():
        for name, value in params.items():
            value.copy_(ema.shadow[name].to(value.device, value.dtype))
    return saved, buffers


def restore_raw(module: nn.Module, saved, buffers) -> None:
    with torch.no_grad():
        for name, value in module.named_parameters(): value.copy_(saved[name])
        for name, value in module.named_buffers():
            if not torch.equal(value, buffers[name]):
                raise RuntimeError(f"EMA evaluation mutated fixed buffer {name}")


def r2(y_rows: list[np.ndarray], p_rows: list[np.ndarray]) -> float:
    """Canonical float64 R² surface shared with artifacts and final aggregation."""
    return float(variance_weighted_r2(np.concatenate(y_rows), np.concatenate(p_rows)))


def evaluate(model, ema, banks, data, device, *, artifact_dir: Path | None = None) -> dict:
    saved, buffers = swap_ema(model, ema); old_training = model.training; model.eval(); rows = {}
    try:
        for session, (X, y, valid) in data.items():
            predicted = []
            for offset in range(0, len(X), BATCH):
                with torch.inference_mode():
                    output = model(torch.from_numpy(X[offset:offset+BATCH]).to(device), banks[session],
                                   input_valid_mask=torch.from_numpy(valid[offset:offset+BATCH]).to(device))
                    predicted.append((output / SCALE).float().cpu().numpy())
            prediction = np.ascontiguousarray(np.concatenate(predicted), dtype=np.float32)
            artifact = None
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True); artifact = artifact_dir / f"{session}.npz"
                np.savez_compressed(artifact, prediction=prediction, target=y, endpoints=banks[session].window_ids)
            rows[session] = {"r2": r2([y], [prediction]), "windows": int(len(y)),
                             "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                             "endpoint_sha256": hashlib.sha256(banks[session].window_ids.tobytes()).hexdigest(),
                             "artifact": None if artifact is None else str(artifact), "artifact_sha256": None if artifact is None else sha_file(artifact)}
    finally:
        restore_raw(model, saved, buffers); model.train(old_training)
    return {"per_session": rows, "equal_session_mean": float(np.mean([row["r2"] for row in rows.values()])),
            "n_windows": int(sum(row["windows"] for row in rows.values()))}


def raw_disjoint(root: Path, fold: str, manifest: dict, *, source: bool, arm: str) -> None:
    expected_stride = 4 if source else 1
    for session, row in manifest["records"].items():
        if row["stride"] != expected_stride or not row["raw_interval_disjoint"]: raise RuntimeError(f"{session}: manifest stride/interval failure")
        if source:
            validation = row["validation"]
            if validation["stride"] != 4 or not validation["raw_interval_disjoint"]: raise RuntimeError(f"{session}: validation manifest failure")
    audit_surface(root, fold, "source" if source else "target", arm, validation=False)
    if source: audit_surface(root, fold, "source", arm, validation=True)


def perturbation_preflight(args, device: torch.device, source_banks, source_data, source_activity) -> dict:
    models = {arm: make_model(arm, args.seed, device, source_banks if arm == ARM_D else source_banks,
                              source_activity if arm in (ARM_B, ARM_D) else {}) for arm in ARMS}
    params = {arm: sum(value.numel() for value in model.parameters()) for arm, model in models.items()}
    first = next(iter(source_data)); X, y, valid = source_data[first]; raw = torch.from_numpy(X[:2]).to(device); mask = torch.from_numpy(valid[:2]).to(device)
    states = {arm: dict(model.named_parameters()) for arm, model in models.items()}
    common_keys = set(states[ARM_Z])
    if common_keys != (set(states[ARM_B]) - {name for name in states[ARM_B] if name.startswith("encoder.")}) or common_keys != (set(states[ARM_D]) - {name for name in states[ARM_D] if name.startswith("encoder.")}):
        raise RuntimeError("Z must contain exactly the shared decoder keys, with encoder only in B/D")
    shared = all(torch.equal(states[ARM_Z][key].cpu(), states[ARM_B][key].cpu()) and torch.equal(states[ARM_Z][key].cpu(), states[ARM_D][key].cpu()) for key in common_keys)
    if not shared: raise RuntimeError("shared decoder initial states are not byte-identical")
    encoder_keys = {name for name in states[ARM_B] if name.startswith("encoder.")}
    if not encoder_keys or encoder_keys != {name for name in states[ARM_D] if name.startswith("encoder.")}:
        raise RuntimeError("B/D encoder key sets mismatch")
    if not all(torch.equal(states[ARM_B][key].cpu(), states[ARM_D][key].cpu()) for key in encoder_keys):
        raise RuntimeError("B/D random C2 encoder initial state mismatch")
    if max(params.values()) / min(params.values()) - 1.0 >= .05: raise RuntimeError("actual arm parameter difference >=5%")
    finite_backward, nonzero_gradient_parameters = {}, {}
    for arm, model in models.items():
        model.train(); model.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(raw, source_banks[first], input_valid_mask=mask).float(), torch.from_numpy(y[:2] * SCALE).to(device))
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        active = [g for g in gradients if g is not None and torch.isfinite(g).all() and g.abs().sum() > 0]
        finite_backward[arm] = bool(torch.isfinite(loss) and active)
        nonzero_gradient_parameters[arm] = int(sum(g.numel() for g in active))
        if not finite_backward[arm]: raise RuntimeError(f"{arm}: no finite nonzero parameter gradient")
    # Exact comparisons use eval mode: no unit dropout may manufacture effects.
    for model in models.values(): model.eval()
    def altered_bank(bank: TaskBank) -> TaskBank:
        return TaskBank(bank.session_id, bank.E0, bank.carrier + .25, bank.unit_mask, bank.X_store, bank.target_store, bank.window_ids, bank.calibration_meta)
    with torch.inference_mode():
        z0 = models[ARM_Z](raw, source_banks[first], input_valid_mask=mask)
        z_changed_bank = models[ARM_Z](raw, altered_bank(source_banks[first]), input_valid_mask=mask)
        b0 = models[ARM_B](raw, source_banks[first], input_valid_mask=mask)
        b_changed_bank = models[ARM_B](raw, altered_bank(source_banks[first]), input_valid_mask=mask)
        d0 = models[ARM_D](raw, source_banks[first], input_valid_mask=mask)
        bbuf, dbuf, cbuf = next(iter(models[ARM_B]._activity.values())), next(iter(models[ARM_D]._activity.values())), next(iter(models[ARM_D]._carrier.values()))
        old_b, old_d, old_c = bbuf.clone(), dbuf.clone(), cbuf.clone()
        bbuf.add_(.25); b_activity = models[ARM_B](raw, source_banks[first], input_valid_mask=mask); bbuf.copy_(old_b)
        dbuf.add_(.25); d_activity = models[ARM_D](raw, source_banks[first], input_valid_mask=mask); dbuf.copy_(old_d)
        cbuf.add_(.25); d_carrier = models[ARM_D](raw, source_banks[first], input_valid_mask=mask); cbuf.copy_(old_c)
        dbuf.add_(.25); cbuf.add_(.25); d_both = models[ARM_D](raw, source_banks[first], input_valid_mask=mask); dbuf.copy_(old_d); cbuf.copy_(old_c)
    changed = lambda a, b: bool(not torch.equal(a, b))
    effects = {"z_calibration_insensitive": not changed(z0, z_changed_bank), "b_activity_sensitive": changed(b0, b_activity),
               "b_carrier_insensitive": not changed(b0, b_changed_bank), "d_activity_sensitive": changed(d0, d_activity),
               "d_carrier_sensitive": changed(d0, d_carrier), "d_dual_path_sensitive": changed(d0, d_both)}
    if not all(effects.values()): raise RuntimeError(f"invalid arm dataflow: {effects}")
    return {"trainable_parameter_counts": params, "parameter_delta_fraction": max(params.values()) / min(params.values()) - 1.,
            "shared_decoder_initial_states_byte_equal": True, "b_d_encoder_initial_state_equal": True,
            "z_has_no_encoder": not hasattr(models[ARM_Z], "encoder"), "finite_backward": finite_backward,
            "nonzero_gradient_parameter_counts": nonzero_gradient_parameters, "effects": effects,
            "z_output_sha256": hashlib.sha256(z0.cpu().numpy().tobytes()).hexdigest()}


def preflight(args) -> dict:
    device = torch.device(args.device)
    manifest, banks, data, activity = load_surface(args.prepared, args.fold, "source", ARM_D)
    raw_disjoint(args.prepared, args.fold, manifest, source=True, arm=ARM_D)
    # Use ARM_D load to make concrete source H-C available to all tests, while
    # Z/B model construction itself denies carrier/activity as appropriate.
    source_activity = activity
    checks = perturbation_preflight(args, device, banks, data, source_activity)
    row = {"schema": "h1_lodo_preflight_v2", "fold": args.fold, "arm": args.arm, "source_only": True,
           "target_arrays_opened": 0, "dataflow": {"support": "first 3 available eval-valid native trials", "gradient": "source available-trial indices[3:-2], stride4", "selection": "source available-trial indices[-2:], stride4 EMA", "target": "score-only target available-trial indices[3:] all valid bins", "native_trial_ids": "recorded per session in prepared manifest"}, **checks}
    json_atomic(args.dest / "preflight.json", row); return row


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def recipe_binding(meta: dict) -> str:
    return hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fixed_buffer_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_buffers():
        if not name.startswith(("activity_", "carrier_")):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(args) -> dict:
    device = torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    manifest, banks, data, activity = load_surface(args.prepared, args.fold, "source", args.arm)
    raw_disjoint(args.prepared, args.fold, manifest, source=True, arm=args.arm)
    _, val_banks, val_data, _ = load_surface(args.prepared, args.fold, "source", args.arm, validation=True)
    # Same sessions / frozen memories; only X/y differ. Never reinstall buffers.
    if set(banks) != set(val_banks): raise RuntimeError("source train/val session mismatch")
    model = make_model(args.arm, args.seed, device, banks, activity)
    ema = DecoderEMA(model, decay=EMA_DECAY); strict_ema(model, ema)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    run = args.dest
    updates_per_epoch = sum((len(X) + BATCH - 1) // BATCH for X, _, _ in data.values())
    meta = {"schema": "h1_lodo_train_v2", "fold": args.fold, "arm": args.arm, "seed": args.seed,
            "epochs": EPOCHS, "batch": BATCH, "optimizer": "AdamW(lr=1e-4,wd=.01)", "clip_norm": 1.,
            "ema": EMA_DECAY, "schedule": "warm1epoch cosine floor.1", "precision": "bf16 RIFT trunk; C2 fp32", "backend": "local",
            "unit_dropout": {"p": .1, "paired_seed_recipe": "unit_dropout_seed(seed,epoch,global_step)"}, "behavior_scale": SCALE,
            "support": "first 3 available eval-valid native trials", "source_train": "available-trial indices[3:-2], stride4", "source_val": "available-trial indices[-2:], stride4", "target": "score-only available-trial indices[3:] all valid bins", "native_trial_ids": "per-session prepared manifest",
            "initial_parameter_sha256": parameter_bytes(model), "source_manifest_sha256": sha_file(args.prepared / args.fold / "source" / "manifest.json"),
            "source_code_sha256": {"cross_session_h1_model.py": sha_file(ROOT / "src/btransform_unified_v2/cross_session_h1_model.py"), "h1_prepare.py": sha_file(ROOT / "scripts/cross_session_v1/h1_prepare.py"), "h1_train.py": sha_file(ROOT / "scripts/cross_session_v1/h1_train.py"), "h1_m4_eb_pilot.py": sha_file(ROOT.parent / "SPINT-main/src/data/h1_m4_eb_pilot.py")}}
    binding = recipe_binding(meta)
    allowed_fresh = {"preflight.json"}
    if run.exists() and args.resume is None:
        present = {item.name for item in run.iterdir()}
        if present - allowed_fresh: raise FileExistsError(f"fresh run refuses existing artifacts: {sorted(present - allowed_fresh)}")
        if "preflight.json" in present:
            check = json.loads((run / "preflight.json").read_text())
            if check.get("fold") != args.fold or check.get("arm") != args.arm: raise RuntimeError("preflight fold/arm mismatch")
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "run_meta.json").exists(): json_atomic(run / "run_meta.json", {**meta, "recipe_binding": binding})
    start, step, curve = 1, 0, []
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("model", "opt", "ema", "epoch", "step", "curve", "rng", "recipe_binding"):
            if key not in state: raise RuntimeError(f"resume missing {key}")
        if state["recipe_binding"] != binding: raise RuntimeError("resume fold/arm/recipe/source binding mismatch")
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["opt"]); ema.load_state_dict(state["ema"]); strict_ema(model, ema)
        restore_rng(state["rng"]); start, step, curve = int(state["epoch"]) + 1, int(state["step"]), list(state["curve"])
    began = time.monotonic()
    for epoch in range(start, EPOCHS + 1):
        model.train(); losses = []; sampler = np.random.default_rng(args.seed + epoch); sessions = sorted(data); sampler.shuffle(sessions)
        sampler_digest = hashlib.sha256(); sampler_digest.update(np.asarray(sessions, dtype="S").tobytes())
        for session in sessions:
            X, y, valid = data[session]; order = sampler.permutation(len(X)); sampler_digest.update(session.encode()); sampler_digest.update(order.astype(np.int64).tobytes())
            for offset in range(0, len(order), BATCH):
                index = order[offset:offset+BATCH]; step += 1
                optimizer.param_groups[0]["lr"] = warmup_cosine_lr(step, EPOCHS * updates_per_epoch, updates_per_epoch, peak=1e-4, min_factor=.1)
                keep = whole_unit_dropout(torch.ones(176, dtype=torch.bool), p=.1, generator=torch.Generator().manual_seed(unit_dropout_seed(args.seed, epoch, step))); sampler_digest.update(keep.numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                with amp:
                    prediction = model(torch.from_numpy(X[index]).to(device), banks[session], dropout_keep=keep, input_valid_mask=torch.from_numpy(valid[index]).to(device))
                    loss = nn.functional.mse_loss(prediction.float(), torch.from_numpy(y[index] * SCALE).to(device))
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                loss.backward(); grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)); optimizer.step(); ema.update_after_step(model); losses.append(float(loss.detach()))
                if step == 1 or step % 50 == 0:
                    heartbeat = {"event": "step", "epoch": epoch, "step": step, "loss": losses[-1], "grad_norm": grad_norm, "elapsed_seconds": time.monotonic()-began}
                    json_atomic(run / "heartbeat.json", heartbeat); print(json.dumps(heartbeat, sort_keys=True), flush=True)
        validation = evaluate(model, ema, banks, val_data, device)
        strict_ema(model, ema)
        epoch_row = {"event": "epoch", "epoch": epoch, "train_mse": float(np.mean(losses)), "source_val_ema": validation, "sampler_endpoint_keep_sha256": sampler_digest.hexdigest(), "elapsed_seconds": time.monotonic()-began}
        curve.append(epoch_row); json_atomic(run / "heartbeat.json", epoch_row); print(json.dumps(epoch_row, sort_keys=True), flush=True)
        state = {"schema": "h1_lodo_resume_v3", "epoch": epoch, "step": step, "model": model.state_dict(), "opt": optimizer.state_dict(), "ema": ema.state_dict(), "curve": curve, "rng": rng_state(), "recipe_binding": binding}
        torch_atomic(run / "resume_latest.pt", state)
        torch_atomic(run / f"ema_epoch_{epoch:03d}.pt", {"schema": "h1_lodo_ema_only_v1", "epoch": epoch, "ema": ema.state_dict(), "recipe_binding": binding, "fixed_buffer_sha256": fixed_buffer_hash(model)})
    selected = max(curve, key=lambda row: (row["source_val_ema"]["equal_session_mean"], -row["epoch"]))
    receipt = {"schema": "h1_lodo_train_receipt_v2", "meta": meta, "selection": "earliest equal-session mean source-validation EMA", "selected": selected,
               "curve": curve, "recipe_binding": binding, "source_files": {session: {"train": sha_file(args.prepared / args.fold / "source" / f"{session}.npz"), "validation": sha_file(args.prepared / args.fold / "source" / f"{session}.val.npz")} for session in sorted(data)}, "sampler": "per-epoch PCG64(seed+epoch), sessions/order/endpoint/keep bytes hashed in each curve row", "checkpoints": {f"ema_epoch_{epoch:03d}": sha_file(run / f"ema_epoch_{epoch:03d}.pt") for epoch in range(1, EPOCHS+1)}, "resume_latest_sha256": sha_file(run / "resume_latest.pt")}
    json_atomic(run / "selection.json", {"selection": receipt["selection"], **selected}); json_atomic(run / "train_receipt.json", receipt)
    return {"status": "TRAIN_COMPLETED", "selected_epoch": selected["epoch"], "receipt": str(run / "train_receipt.json")}


def score_epoch(args, epoch: int, target_banks, target_data, target_activity, device, *, label: str) -> dict:
    path = args.dest / f"ema_epoch_{epoch:03d}.pt"
    if not path.is_file(): raise FileNotFoundError(path)
    model = make_model(args.arm, args.seed, device, target_banks, target_activity)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != epoch or state.get("schema") != "h1_lodo_ema_only_v1": raise RuntimeError("EMA-only checkpoint binding failure")
    meta = json.loads((args.dest / "run_meta.json").read_text())
    if state.get("recipe_binding") != meta.get("recipe_binding"): raise RuntimeError("EMA-only recipe binding mismatch")
    if state.get("fixed_buffer_sha256") != fixed_buffer_hash(model): raise RuntimeError("fixed decoder buffer contract mismatch")
    ema = DecoderEMA(model, decay=EMA_DECAY); ema.load_state_dict(state["ema"]); strict_ema(model, ema)
    return {"epoch": epoch, "checkpoint_sha256": sha_file(path), "target": evaluate(model, ema, target_banks, target_data, device, artifact_dir=args.dest / "target_score_artifacts" / label)}


def score(args) -> dict:
    selected = json.loads((args.dest / "selection.json").read_text()); epoch = int(selected["epoch"])
    device = torch.device(args.device); target_manifest, banks, data, activity = load_surface(args.prepared, args.fold, "target", args.arm)
    raw_disjoint(args.prepared, args.fold, target_manifest, source=False, arm=args.arm)
    result = {"schema": "h1_lodo_target_score_v2", "fold": args.fold, "arm": args.arm, "target_query_labels_used_for_gradients": False,
              "target_query_labels_used_for_selection": False, "target_calibration_consumption": "none" if args.arm == ARM_Z else ("M3 activity only" if args.arm == ARM_B else "M3 activity plus H-C carrier"),
              "selected_source_epoch": epoch, "selected_ema": score_epoch(args, epoch, banks, data, activity, device, label="selected_ema"), "fixed_e32_ema": score_epoch(args, 32, banks, data, activity, device, label="fixed_e32_ema"),
              "target_manifest_sha256": sha_file(args.prepared / args.fold / "target" / "manifest.json")}
    json_atomic(args.dest / "target_score.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True); parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--fold", required=True, choices=("1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15", "1925-01-19", "1925-01-20"))
    parser.add_argument("--arm", required=True, choices=ARMS); parser.add_argument("--stage", required=True, choices=("preflight", "train", "score"))
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--cpu-threads", type=int, default=2); parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = preflight(args) if args.stage == "preflight" else train(args) if args.stage == "train" else score(args)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__": main()
