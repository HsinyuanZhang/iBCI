"""Hash-authorized CRST-B4 split-arm training and post-freeze evaluation.

Importing this module performs no training or data access. The separate
supervisor must bind the exact source authority, code, and recipe before
calling workers. No automatic resume or mid-epoch resume is implemented.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
EPOCHS = 12
EPOCH_UPDATES = 731
MICRO = 8
EFFECTIVE = 32
LR = 1e-4
EMA = .9995
DROP_P = .10
ARM_DEVICE = {"flat": 0, "route": 1}


PROSPECTIVE_PROTOCOL = {
    "schema": "h1_crst_b4_localbalanced_splitarm_formal_prospective_v1",
    "status": "FROZEN_RECIPE_REQUIRES_EXTERNAL_HASH_AUTHORIZATION",
    "arms": {"flat": {"physical_gpu": 0}, "route": {"physical_gpu": 1}},
    "initialization": "each subprocess calls the same fresh make_v2_unscaled_dot_localbalanced_pair(seed=42), proves g0 shared parity, records common shared-state SHA, then discards the other arm",
    "source_sampler": "each arm independently calls the identical h1_optimized_v4.paired_train.batches(cache, epoch); assert exactly 731 ordered effective batches and byte-identical (session, starts) identity digest across split-arm receipts",
    "dropout": "for every (epoch, batch_index), keep = stateless_uniform(seed=42|keep|epoch|batch_index)>=.10 AND bank.unit_mask; same identities/mask bytes required for FLAT and ROUTE",
    "recipe": {"epochs": EPOCHS, "source_windows": 23212, "updates_per_epoch_per_arm": EPOCH_UPDATES, "total_updates_per_arm": EPOCH_UPDATES * EPOCHS, "microbatch": MICRO, "effective_batch": EFFECTIVE, "lr": LR, "warmup": "linear on steps 1..731 of epoch1 then constant", "optimizer": "AdamW trusted H1 decay grouping, wd=.01, clip_norm=1", "ema": EMA, "fresh_seed": SEED, "warmstart_from_source1040": False},
    "selection": "after every completed epoch, score EMA only on frozen 2908 minival endpoints; governing pooled R2; choose earliest epoch at maximum",
    "complete": "only after selection freeze: report selected EMA and independent epoch12 EMA on frozen 20325 complete bins",
    "checkpoint": "end-of-epoch per arm; atomic payload model+optimizer+EMA+CPU/CUDA/numpy/python RNG, epoch and next_batch_index=731; strict restore; no mid-epoch resume claim",
    "provenance": "freeze source cache authority and code hashes before launch; each arm binds protocol hash, sampler digest, matching peer shared-state hash, and device assignment",
    "resource": "prospective hard wall 21600 seconds; split-smoke estimates must be reviewed with validation/export allowance before authorization",
}


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def protocol_sha256() -> str:
    return sha_bytes((json.dumps(PROSPECTIVE_PROTOCOL, sort_keys=True, separators=(",", ":")) + "\n").encode())


def dropout_keep(*, n: int, epoch: int, batch_index: int, bank_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The exact paired split-arm mask law; invalid bank units can never revive."""
    if not (1 <= epoch <= EPOCHS and 0 <= batch_index < EPOCH_UPDATES):
        raise ValueError("epoch/batch index outside frozen formal schedule")
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|keep|{epoch}|{batch_index}".encode()).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keep = (torch.rand((n, bank_mask.numel()), generator=generator) >= DROP_P) & bank_mask.detach().cpu().to(torch.bool).view(1, -1)
    if not bool(keep.any(dim=1).all()):
        raise RuntimeError("empty unit row")
    return keep.to(device)


def warmup_lr(*, epoch: int, global_step: int) -> float:
    """Epoch-one per-step linear warmup, then fixed LR, without scheduler state."""
    if epoch == 1:
        if not 1 <= global_step <= EPOCH_UPDATES:
            raise ValueError("epoch1 global step inconsistent with 731-update sampler")
        return LR * global_step / EPOCH_UPDATES
    if global_step < EPOCH_UPDATES:
        raise ValueError("post-warmup epoch cannot precede complete epoch1")
    return LR


def sampler_identity_digest(ordered_batches: list[tuple[str, np.ndarray]]) -> str:
    """Bind split workers to equal session/start identities, not merely counts."""
    h = hashlib.sha256()
    if len(ordered_batches) != EPOCH_UPDATES:
        raise RuntimeError("expected exactly 731 effective batches")
    for index, (session, starts) in enumerate(ordered_batches):
        starts = np.asarray(starts, dtype=np.int64)
        h.update(f"{index}|{session}|".encode()); h.update(starts.tobytes())
    return h.hexdigest()


def checkpoint_payload(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: Any, epoch: int, global_step: int, sampler_digest: str, source_authority_sha256: str, code_sha256: dict[str, str], arm: str, shared_init_sha256: str, dropout_digest: str) -> dict[str, Any]:
    """End-of-epoch-only resume payload; all inputs must be frozen externally."""
    if not (1 <= epoch <= EPOCHS and global_step == epoch * EPOCH_UPDATES):
        raise RuntimeError("only a completed epoch is checkpointable")
    if arm not in ARM_DEVICE or ema.n_updates != global_step:
        raise RuntimeError("checkpoint arm or EMA update count drift")
    return {"schema": "h1_crst_b4_splitarm_end_epoch_checkpoint_v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema.checkpoint_state(), "epoch": epoch, "next_batch_index": EPOCH_UPDATES, "global_step": global_step, "sampler_identity_sha256": sampler_digest, "source_authority_sha256": source_authority_sha256, "code_sha256": code_sha256, "protocol_sha256": protocol_sha256(), "arm": arm, "shared_init_sha256": shared_init_sha256, "dropout_identity_sha256": dropout_digest, "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(), "python": random.getstate()}}


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Atomic end-of-epoch serialization contract for a future authorized run."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def strict_restore(*, payload: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: Any, expected_epoch: int, expected_sampler_digest: str, expected_source_authority_sha256: str, expected_code_sha256: dict[str, str], expected_arm: str, expected_shared_init_sha256: str, expected_dropout_digest: str) -> int:
    """Refuse a resume whose stateless schedule or immutable provenance drifted."""
    if payload.get("protocol_sha256") != protocol_sha256() or payload.get("epoch") != expected_epoch or payload.get("next_batch_index") != EPOCH_UPDATES or payload.get("global_step") != expected_epoch * EPOCH_UPDATES:
        raise RuntimeError("checkpoint is not the required end-of-epoch split-arm state")
    if payload.get("sampler_identity_sha256") != expected_sampler_digest or payload.get("source_authority_sha256") != expected_source_authority_sha256 or payload.get("code_sha256") != expected_code_sha256:
        raise RuntimeError("checkpoint provenance does not match frozen split-arm inputs")
    if (payload.get("arm") != expected_arm or payload.get("shared_init_sha256") != expected_shared_init_sha256
            or payload.get("dropout_identity_sha256") != expected_dropout_digest
            or payload["ema"]["n_updates"] != expected_epoch * EPOCH_UPDATES):
        raise RuntimeError("checkpoint paired-arm, dropout, or EMA provenance drift")
    model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"]); ema.load_checkpoint_state(payload["ema"])
    rng = payload["rng"]; torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state_all(rng["cuda"]); np.random.set_state(rng["numpy"]); random.setstate(rng["python"])
    return expected_epoch + 1


def earliest_argmax(epoch_scores: list[tuple[int, float]]) -> tuple[int, float]:
    """Stable first maximum enforces the frozen earliest-epoch tie rule."""
    if len(epoch_scores) != EPOCHS or [epoch for epoch, _ in epoch_scores] != list(range(1, EPOCHS + 1)):
        raise RuntimeError("selection requires all twelve predeclared EMA scores")
    if not all(np.isfinite(score) for _, score in epoch_scores):
        raise RuntimeError("primary EMA selection contains nonfinite score")
    return max(epoch_scores, key=lambda item: item[1])


# -- Runnable path below ----------------------------------------------------
# It is deliberately reachable only through the separately authorized
# supervisor.  Keeping the implementation here lets the supervisor bind one
# immutable code closure without creating a formal root during review.

def _code_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_closure() -> dict[str, str]:
    src=Path(__file__).parents[1]
    files={"split_train":Path(__file__),"launcher":Path(__file__).with_name("familyformal_split_launcher.py"),"family_model":Path(__file__).with_name("model.py"),"h1v2_model":src/"h1_optimized_v2/model.py","cache":src/"h1_optimized_v2/cache.py","data":src/"h1_optimized_v2/data.py","groups":src/"h1_optimized_v2/paired_train.py","score":src/"h1_optimized_v2/score.py","v4_sampler":src/"h1_optimized_v4/paired_train.py","h1_config":src/"two_mainlines_long_v1/decoder/h1_config.py","temporal":src/"two_mainlines_long_v1/decoder/h1_temporal.py","current_query_core":src/"two_mainlines_long_v1/current_query_v2/core.py","ema":src/"h1_temporal_decoder_quick_product_v1/ema.py"}
    return {name:_code_sha(path) for name,path in files.items()}


def require_frozen_inputs(output: Path) -> dict[str, Any]:
    from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, CACHE
    frozen = json.loads((output / "input_authority.json").read_text())
    if frozen["code_closure"] != {**code_closure(), "protocol": protocol_sha256()}:
        raise RuntimeError("current code differs from frozen execution authority")
    if (_code_sha(H1_ROOT / "source_cache_authority.json") != frozen["bindings"]["source_authority_sha256"]
            or _code_sha(CACHE) != frozen["bindings"]["source_cache_sha256"]):
        raise RuntimeError("source cache or authority differs from frozen inputs")
    return frozen


def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name+".", suffix=".tmp", mode="w", delete=False) as handle:
        temporary=Path(handle.name); json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    try: os.replace(temporary,destination)
    finally:
        if temporary.exists(): temporary.unlink()


def _state_sha(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _shared_state_sha(flat: torch.nn.Module, route: torch.nn.Module) -> str:
    # shared equality is proved separately by initialization_receipt; do not
    # silently discard unequal names while composing this cross-process digest.
    return _state_sha(dict(flat.state_dict()))


def _all_epoch_identities(cache: dict, *, device: torch.device) -> dict[str, dict[str, str]]:
    """Digest all twelve ordered batches and all paired keep masks pre-update."""
    from tfpd_exploration.src.h1_optimized_v4.paired_train import batches
    result: dict[str, dict[str, str]] = {}
    for epoch in range(1, EPOCHS + 1):
        ordered = batches(cache, epoch)
        if sum(len(starts) for _, starts in ordered) != 23212:
            raise RuntimeError("source sampler must visit all 23212 starts per epoch")
        sampler = sampler_identity_digest(ordered)
        masks = hashlib.sha256()
        for batch_index, (session, starts) in enumerate(ordered):
            mask = dropout_keep(n=len(starts), epoch=epoch, batch_index=batch_index, bank_mask=cache["train"][session]["bank"]["unit_mask"], device=torch.device("cpu"))
            masks.update(session.encode()); masks.update(np.asarray(starts, dtype=np.int64).tobytes()); masks.update(mask.numpy().tobytes())
        result[str(epoch)] = {"sampler_sha256": sampler, "keep_sha256": masks.hexdigest()}
    return result


def _r2_float64(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(np.float64, copy=False), target.astype(np.float64, copy=False)
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("nonfinite or mismatched native metric arrays")
    denom = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ((prediction - target) ** 2).sum() / denom) if denom else float("nan")


def score_selection_cached_ema(*, model: torch.nn.Module, ema: Any, cache: dict, device: torch.device) -> dict[str, Any]:
    """Frozen minival 2,908 endpoint primary score; cache is already validated."""
    from tfpd_exploration.src.h1_optimized_v2.score import _windows
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    pieces, targets = [], []
    with torch.inference_mode():
        def score(candidate: torch.nn.Module) -> None:
            candidate.eval()
            for session, row in sorted(cache["minival"].items()):
                ends = row["query_starts"] + 699
                bank = H1Bank(row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device))
                for offset in range(0, len(ends), 8):
                    x = torch.as_tensor(_windows(row["neural"], ends[offset:offset + 8]), device=device)
                    pieces.append((candidate.forward_last(x, bank) / 20.0).cpu().numpy().astype(np.float64))
                targets.append(row["velocity"][ends].astype(np.float64))
        ema.score_with_ema(model, score)
    pred, target = np.concatenate(pieces), np.concatenate(targets)
    if len(pred) != 2908 or not np.isfinite(pred).all():
        raise RuntimeError("selection surface must be exactly 2908 finite predictions")
    return {"n_bins": int(len(pred)), "r2_concat_float64": _r2_float64(pred, target), "finite": True}


def score_complete_cached_ema(*, model: torch.nn.Module, ema: Any, cache: dict, device: torch.device) -> dict[str, Any]:
    """Post-freeze complete score: exactly 20,325 masked bins, native FP64."""
    from tfpd_exploration.src.h1_optimized_v2.score import _windows
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    pieces, targets, sessions, ends_all = [], [], [], []
    with torch.inference_mode():
        def score(candidate: torch.nn.Module) -> None:
            candidate.eval()
            for session, row in sorted(cache["minival"].items()):
                ends = np.flatnonzero(row["eval_mask"])
                bank = H1Bank(row["bank"]["E0"].to(device), row["bank"]["T"].to(device), row["bank"]["unit_mask"].to(device))
                for offset in range(0, len(ends), 8):
                    x = torch.as_tensor(_windows(row["neural"], ends[offset:offset + 8]), device=device)
                    pieces.append((candidate.forward_last(x, bank) / 20.0).cpu().numpy().astype(np.float64))
                targets.append(row["velocity"][ends].astype(np.float64))
                sessions.extend([session]*len(ends)); ends_all.append(ends.astype(np.int64))
        ema.score_with_ema(model, score)
    pred, target = np.concatenate(pieces), np.concatenate(targets)
    if len(pred) != 20325 or not np.isfinite(pred).all(): raise RuntimeError("complete surface must be exactly 20325 finite predictions")
    sid=np.asarray(sessions); end=np.concatenate(ends_all)
    per={name:_r2_float64(pred[sid==name],target[sid==name]) for name in sorted(set(sid.tolist()))}
    return {"n_bins": int(len(pred)), "r2_concat_float64": _r2_float64(pred, target), "equal_session_mean_r2_float64":float(np.mean(list(per.values()))),"worst_session":min(per,key=per.get),"worst_session_r2_float64":float(min(per.values())),"per_session_r2_float64":per,"finite":True,"_prediction":pred,"_target":target,"_session_id":sid,"_end":end}


def _normalised_micro_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, bank: Any, keep: torch.Tensor) -> float:
    """Backprop exactly summed size-normalised micro losses, never their mean."""
    total = 0.0
    for offset in range(0, len(x), MICRO):
        count = len(x[offset:offset + MICRO])
        loss = F.mse_loss(model.forward_last(x[offset:offset + MICRO], bank, dropout_keep=keep[offset:offset + MICRO]), y[offset:offset + MICRO])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite training loss before optimizer update")
        (loss * (count / len(x))).backward()
        total += float(loss.detach()) * (count / len(x))
    return total


def worker_run(*, arm: str, output: Path, physical_gpu: int, start_marker: Path) -> None:
    """Worker called only by `familyformal_split_launcher`; no auto-resume."""
    from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, build_or_load, validate_authority
    from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
    from tfpd_exploration.src.h1_optimized_v4.paired_train import batches, collate
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from .model import initialization_receipt, make_v2_unscaled_dot_localbalanced_pair, route_gate_gradient_l1, zero_gate_parity
    if arm not in ARM_DEVICE or physical_gpu != ARM_DEVICE[arm]: raise RuntimeError("arm/device contract mismatch")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    frozen = require_frozen_inputs(output)
    cache = build_or_load(); authority_path = H1_ROOT / "source_cache_authority.json"; authority = json.loads(authority_path.read_text()); validate_authority(cache, authority)
    identities = _all_epoch_identities(cache, device=torch.device("cpu"))
    device = torch.device("cuda:0"); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=SEED); init = initialization_receipt(flat, route)
    shared = _shared_state_sha(flat, route)
    selected, other = (flat, route) if arm == "flat" else (route, flat); selected = selected.to(device); del other
    # CPU construction evidence closes g0 pairing before the processes divide arms.
    pf, pr = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
    first_session, first_starts = batches(cache, 1)[0]; row = cache["train"][first_session]
    x_cpu, y_cpu = collate(row, first_starts, torch.device("cpu")); bank_cpu = H1Bank(row["bank"]["E0"], row["bank"]["T"], row["bank"]["unit_mask"])
    parity = zero_gate_parity(pf, pr, x_cpu[:MICRO], bank_cpu); pr.zero_grad(set_to_none=True); keep_cpu = dropout_keep(n=MICRO, epoch=1, batch_index=0, bank_mask=bank_cpu.unit_mask, device=torch.device("cpu")); F.mse_loss(pr.forward_last(x_cpu[:MICRO], bank_cpu, dropout_keep=keep_cpu), y_cpu[:MICRO]).backward(); gate_grad = route_gate_gradient_l1(pr); del pf, pr
    if parity["max_abs_diff"] != 0.0 or (arm=="route" and (gate_grad is None or not np.isfinite(gate_grad) or gate_grad<=0)):
        raise RuntimeError("invalid g0 matched-pair preflight")
    atomic_json({"arm": arm, "shared_sha256": shared, "identities": identities, "init": init, "g0_parity": parity, "route_gate_gradient_l1": gate_grad if arm == "route" else None}, output / "barrier" / f"{arm}.ready.json")
    while not start_marker.exists(): time.sleep(.05)
    opt, ema, records = torch.optim.AdamW(groups(selected), lr=LR), DecoderEMA(selected, decay=EMA), []
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        ordered = batches(cache, epoch)
        if sampler_identity_digest(ordered) != identities[str(epoch)]["sampler_sha256"]: raise RuntimeError("sampler drift after preflight")
        losses = []
        for batch_index, (session, starts) in enumerate(ordered):
            row = cache["train"][session]; x, y = collate(row, starts, device); bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")]); keep = dropout_keep(n=len(x), epoch=epoch, batch_index=batch_index, bank_mask=bank.unit_mask, device=device)
            selected.train(); opt.zero_grad(set_to_none=True); step = (epoch - 1) * EPOCH_UPDATES + batch_index + 1
            for group in opt.param_groups: group["lr"] = warmup_lr(epoch=epoch, global_step=step)
            losses.append(_normalised_micro_loss(selected, x, y, bank, keep)); torch.nn.utils.clip_grad_norm_(selected.parameters(), 1.0, error_if_nonfinite=True); opt.step(); ema.update_after_step(selected)
            if step % 64 == 0:
                atomic_json({"status": "TRAINING", "arm": arm, "epoch": epoch, "batch_index": batch_index, "global_step": step, "last_loss": losses[-1], "lr": opt.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - started}, output / "workers" / f"{arm}_live.json")
        require_frozen_inputs(output)
        payload = checkpoint_payload(model=selected, optimizer=opt, ema=ema, epoch=epoch, global_step=epoch * EPOCH_UPDATES, sampler_digest=identities[str(epoch)]["sampler_sha256"], source_authority_sha256=frozen["bindings"]["source_authority_sha256"], code_sha256=frozen["code_closure"], arm=arm, shared_init_sha256=shared, dropout_digest=identities[str(epoch)]["keep_sha256"])
        checkpoint = output / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"
        if checkpoint.exists(): raise FileExistsError(checkpoint)
        atomic_torch_save(payload, checkpoint)  # preserve completed epoch if scoring fails
        selection = score_selection_cached_ema(model=selected, ema=ema, cache=cache, device=device)
        records.append({"epoch": epoch, "selection": selection, "checkpoint": str(checkpoint), "checkpoint_sha256": _code_sha(checkpoint), "mean_normalised_loss": float(np.mean(losses)), "elapsed_seconds": time.monotonic() - started})
        atomic_json(records[-1], output / "workers" / f"{arm}_epoch_{epoch:03d}_metrics.json")
        atomic_json({"status": "RUNNING", "arm": arm, "epochs": records}, output / "workers" / f"{arm}_progress.json")
    if any(not np.isfinite(entry["selection"]["r2_concat_float64"]) for entry in records): raise RuntimeError("nonfinite primary selection metric")
    selected_epoch, selected_score = earliest_argmax([(entry["epoch"], entry["selection"]["r2_concat_float64"]) for entry in records])
    atomic_json({"status": "EPOCHS_COMPLETE_AWAITING_SUPERVISOR_FREEZE", "arm": arm, "shared_sha256": shared, "identities": identities, "epochs": records, "selected_epoch": selected_epoch, "selected_ema_r2_float64": selected_score}, output / "workers" / f"{arm}_complete.json")


def finalizer_run(*, arm: str, output: Path, physical_gpu: int) -> None:
    """Strict-reload post-freeze selected and epoch12 EMA complete reports."""
    from tfpd_exploration.src.h1_optimized_v2.cache import build_or_load, validate_authority, ROOT as H1_ROOT
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
    from .model import make_v2_unscaled_dot_localbalanced_pair
    if arm not in ARM_DEVICE or physical_gpu != ARM_DEVICE[arm]:
        raise RuntimeError("finalizer arm/device mismatch")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    frozen = require_frozen_inputs(output)
    cache = build_or_load(); auth = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, auth)
    selection_freeze = json.loads((output / "selection_freeze.json").read_text())
    ready = json.loads((output / "barrier" / f"{arm}.ready.json").read_text())
    device = torch.device("cuda:0")
    pair = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
    model = pair[0 if arm == "flat" else 1].to(device); del pair
    out = {}
    for label, record in (("selected", selection_freeze["selected"][arm]), ("epoch12", selection_freeze["endpoints"][arm])):
        path, epoch = Path(record["checkpoint"]), int(record["epoch"])
        if _code_sha(path) != record["checkpoint_sha256"] or (label == "epoch12" and epoch != 12):
            raise RuntimeError("freeze checkpoint SHA or exact endpoint epoch mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        identity = ready["identities"][str(epoch)]
        expected_fields = {
            "arm": arm, "epoch": epoch, "global_step": epoch * EPOCH_UPDATES,
            "next_batch_index": EPOCH_UPDATES, "protocol_sha256": protocol_sha256(),
            "source_authority_sha256": frozen["bindings"]["source_authority_sha256"],
            "code_sha256": frozen["code_closure"], "shared_init_sha256": ready["shared_sha256"],
            "sampler_identity_sha256": identity["sampler_sha256"], "dropout_identity_sha256": identity["keep_sha256"],
        }
        if any(payload.get(name) != value for name, value in expected_fields.items()):
            raise RuntimeError("checkpoint frozen arm/epoch/input/recipe provenance mismatch")
        model.load_state_dict(payload["model"], strict=True)
        ema = DecoderEMA(model, decay=EMA); ema.load_checkpoint_state(payload["ema"])
        expected = {name: param for name, param in model.named_parameters() if param.requires_grad}
        if (ema.decay != EMA or ema.n_updates != epoch * EPOCH_UPDATES or set(ema.shadow) != set(expected)
                or any(ema.shadow[name].shape != param.shape or not bool(torch.isfinite(ema.shadow[name]).all()) for name, param in expected.items())):
            raise RuntimeError("EMA decay, exact update count, keys, shape, or finite contract mismatch")
        # Construct a self-contained CPU EMA state and strictly reload those
        # exact exported tensors for both selection reproduction and reporting.
        plain = {name: ema.shadow.get(name, value).detach().cpu().clone() for name, value in model.state_dict().items()}
        export = output / "exports" / f"{arm}_{label}_plain_ema.pt"
        if export.exists(): raise FileExistsError(export)
        atomic_torch_save(plain, export)
        model.load_state_dict(torch.load(export, map_location="cpu", weights_only=True), strict=True)
        reproduced = score_selection_cached_ema(model=model, ema=ema, cache=cache, device=device)
        if abs(reproduced["r2_concat_float64"] - record["ema_r2_float64"]) > 1e-5:
            raise RuntimeError(f"{label} 2908 reproduction mismatch")
        complete = score_complete_cached_ema(model=model, ema=ema, cache=cache, device=device)
        archive = output / "exports" / f"{arm}_{label}_complete_native_float64.npz"
        if archive.exists(): raise FileExistsError(archive)
        with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=archive.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, prediction=complete.pop("_prediction"), target=complete.pop("_target"), session_id=complete.pop("_session_id"), end=complete.pop("_end"))
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, archive)
        out[label] = {"epoch": epoch, "checkpoint_sha256": _code_sha(path), "plain_ema_sha256": _code_sha(export), "complete_archive_sha256": _code_sha(archive), "selection_reproduced": reproduced, "complete": complete}
    require_frozen_inputs(output)
    atomic_json({"status": "COMPLETE_POST_FREEZE", "arm": arm, "reports": out, "numeric_precision": "FP32 model forward; native predictions promoted to float64 for archive and metrics"}, output / "workers" / f"{arm}_final.json")


if __name__ == "__main__":
    raise SystemExit("Prospective-only split-arm trainer draft: explicit formal authorization and a separate launcher are required.")
