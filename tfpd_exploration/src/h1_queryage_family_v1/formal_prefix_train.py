"""Prospective H1 QueryAge16+p=.5 formal contract; no CLI or auto-launch."""
from __future__ import annotations

import copy
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

SEED, EPOCHS, EPOCH_UPDATES, MICRO, EFFECTIVE, LR, EMA, DROP_P = 42, 12, 731, 8, 32, 1e-4, .9995, .10
DEADLINE_SECONDS, MEMORY_LIMIT = 21600, 22 << 30
ARMS = ("flat", "route")
ARM_DEVICE = {"flat": 0, "route": 1}
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
FORMAL_LAUNCHER = Path(__file__).with_name("formal_prefix_launcher.py")
FORMAL_SMOKE = Path(__file__).with_name("formal_prefix_smoke.py")
CACHE = ROOT / "results/decoder_validation_v2/20260905_190000/h1/source_cache.pt"
CACHE_AUTHORITY = CACHE.with_name("source_cache_authority.json")
PROTOCOL_DOC = ROOT / "docs/PROTOCOL_H1_QUERYAGE_FORMAL_PREFIX_V1_20260906.md"

PROTOCOL = {
    "schema": "h1_queryage_formal_prefix_v1",
    "status": "PROSPECTIVE_REQUIRES_ROOT_AUTHORIZATION",
    "candidate": "fresh FW-QueryAge16 with deterministic p=.5 left-zero cold prefix; compound quality candidate",
    "arms": {"flat": {"physical_gpu": 0}, "route": {"physical_gpu": 1}},
    "recipe": {"seed": SEED, "epochs": EPOCHS, "updates_per_epoch": EPOCH_UPDATES, "source_windows_per_epoch": 23212,
               "microbatch": MICRO, "effective_batch": EFFECTIVE, "optimizer": "fresh AdamW trusted H1 groups wd=.01 clip1",
               "lr": "linear 1..731 to 1e-4 then constant", "ema": EMA, "fresh_no_capacity_state": True},
    "input": "p=.5 cold prefix before stateless p=.1 unit dropout AND immutable bank mask",
    "selection": "EMA pooled native float64 R2 over frozen 2908 minival endpoints after every epoch; earliest maximum",
    "complete": "post-freeze selected and epoch12 EMA frozen 20325 complete bins; no selection on complete",
    "admission": "both-arm source1040 R2>=.5 and prediction_std>=.5*target_std, fresh exact smoke/resource pass, root GO",
}

def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def protocol_sha256() -> str: return digest(PROTOCOL)

def code_closure() -> dict[str, str]:
    """Hash every executable dependency, including the additive launcher/smoke."""
    paths = {
        "train": Path(__file__), "launcher": FORMAL_LAUNCHER, "smoke": FORMAL_SMOKE,
        "formal_score": Path(__file__).with_name("formal_prefix_score.py"),
        "queryage_package": Path(__file__).with_name("__init__.py"), "queryage_model": Path(__file__).with_name("model.py"),
        "family_model": SRC / "h1_family_v1/model.py", "prefix": SRC / "h1_family_v1/cold_history.py",
        "old_formal_score": SRC / "h1_family_v1/familyformal_split_train.py", "h1v2_model": SRC / "h1_optimized_v2/model.py",
        "cache": SRC / "h1_optimized_v2/cache.py", "cache_data": SRC / "h1_optimized_v2/data.py",
        "groups": SRC / "h1_optimized_v2/paired_train.py", "score": SRC / "h1_optimized_v2/score.py",
        "sampler": SRC / "h1_optimized_v4/paired_train.py", "ema": SRC / "h1_temporal_decoder_quick_product_v1/ema.py",
        "query_core": SRC / "two_mainlines_long_v1/current_query_v2/core.py", "query_init": SRC / "two_mainlines_long_v1/current_query_v2/__init__.py",
        "h1_config": SRC / "two_mainlines_long_v1/decoder/h1_config.py", "h1_temporal": SRC / "two_mainlines_long_v1/decoder/h1_temporal.py",
    }
    return {name: sha(path) for name, path in paths.items()}

def collect_bindings(output: Path, *, capacity_receipt: Path, capacity_checkpoint: Path, smoke_receipt: Path | None = None) -> dict[str, Any]:
    """Pure pre-runtime binding builder for the root authorizer and launcher."""
    output, capacity_receipt, capacity_checkpoint = Path(output), Path(capacity_receipt), Path(capacity_checkpoint)
    if not all(path.is_absolute() for path in (output, capacity_receipt, capacity_checkpoint)) or (smoke_receipt is not None and not Path(smoke_receipt).is_absolute()):
        raise RuntimeError("all formal authority paths must be absolute")
    inputs = {str(path): sha(path) for path in (PROTOCOL_DOC, CACHE, CACHE_AUTHORITY, capacity_receipt, capacity_checkpoint)}
    if smoke_receipt is not None: inputs[str(Path(smoke_receipt))] = sha(Path(smoke_receipt))
    return {"protocol": PROTOCOL, "protocol_sha256": protocol_sha256(), "code_closure": code_closure(), "inputs": inputs,
            "output": str(output), "physical_devices": ARM_DEVICE, "threads": 1,
            "capacity_state_used_for_warmstart": False}

def validate_authority_payload(authority: dict[str, Any], *, output: Path, bindings: dict[str, Any]) -> None:
    """Stdlib-only structural check for a launcher before runtime imports."""
    if Path(output).exists(): raise FileExistsError(output)
    if authority.get("schema") != "h1_queryage_formal_prefix_root_authorization_v1" or authority.get("status") != "ROOT_REVIEW_GO" or authority.get("bindings") != bindings:
        raise RuntimeError("root authorization does not bind exact formal closure/input/output")

def warmup_lr(*, epoch: int, global_step: int) -> float:
    if epoch == 1:
        if not 1 <= global_step <= EPOCH_UPDATES: raise ValueError("invalid epoch-one step")
        return LR * global_step / EPOCH_UPDATES
    if global_step < EPOCH_UPDATES: raise ValueError("post-warmup before epoch one")
    return LR

def dropout_keep(*, n: int, epoch: int, batch_index: int, bank_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not (1 <= epoch <= EPOCHS and 0 <= batch_index < EPOCH_UPDATES and n > 0): raise ValueError("dropout schedule geometry")
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|keep|{epoch}|{batch_index}".encode()).digest()[:8], "little")
    keep = torch.rand((n, bank_mask.numel()), generator=torch.Generator(device="cpu").manual_seed(seed)) >= DROP_P
    keep &= bank_mask.detach().cpu().bool().view(1, -1)
    if not bool(keep.any(dim=1).all()): raise RuntimeError("dropout produced empty row")
    return keep.to(device)

def prefix_and_keep(x: torch.Tensor, *, epoch: int, batch_index: int, bank_mask: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The sole formal ordering: prefix x first, then construct unit keep."""
    from tfpd_exploration.src.h1_family_v1.cold_history import apply_cold_history
    prefixed, lengths = apply_cold_history(x, seed=SEED, epoch=epoch, batch_id=batch_index, probability=.5)
    if not torch.equal(lengths.detach().cpu(), prefix_lengths(len(x), epoch=epoch, batch_index=batch_index)):
        raise RuntimeError("actual prefix differs from precommitted stateless length law")
    keep = dropout_keep(n=len(x), epoch=epoch, batch_index=batch_index, bank_mask=bank_mask, device=device)
    if not torch.equal(prefixed[:, -1], x[:, -1]): raise RuntimeError("prefix changed current bin")
    return prefixed.to(device), lengths.detach().cpu(), keep

def sampler_identity_digest(ordered: list[tuple[str, np.ndarray]]) -> str:
    if len(ordered) != EPOCH_UPDATES: raise RuntimeError("requires exactly 731 ordered batches")
    h = hashlib.sha256()
    for index, (session, starts) in enumerate(ordered):
        a = np.asarray(starts, dtype=np.int64); h.update(f"{index}|{session}|".encode()); h.update(a.tobytes())
    return h.hexdigest()

def prefix_lengths(n: int, *, epoch: int, batch_index: int) -> torch.Tensor:
    """Exactly the cold-prefix helper's stateless lengths, without W700 data."""
    from tfpd_exploration.src.h1_family_v1.cold_history import _seed
    generator = torch.Generator(device="cpu").manual_seed(_seed(SEED, epoch, batch_index))
    chosen = torch.rand((n,), generator=generator) < .5
    lengths = torch.full((n,), 700, dtype=torch.int64)
    if bool(chosen.any()):
        lengths[chosen] = torch.randint(1, 700, (int(chosen.sum()),), generator=generator, dtype=torch.int64)
    return lengths

def identity_digest(ordered: list[tuple[str, np.ndarray]], *, cache: dict[str, Any], epoch: int) -> dict[str, str]:
    """CPU-only schedule evidence; no model and no data mutation."""
    sampler, keeps, prefixes = sampler_identity_digest(ordered), hashlib.sha256(), hashlib.sha256()
    for batch_index, (session, starts) in enumerate(ordered):
        row, a = cache["train"][session], np.asarray(starts, dtype=np.int64)
        # Prefix lengths are a stateless function of (seed, epoch, batch,
        # batch size), not neural values.  Avoid materializing all W700 source
        # windows merely to precommit the treatment schedule.
        lengths = prefix_lengths(len(a), epoch=epoch, batch_index=batch_index)
        keep = dropout_keep(n=len(a), epoch=epoch, batch_index=batch_index, bank_mask=row["bank"]["unit_mask"], device=torch.device("cpu"))
        for h in (keeps, prefixes): h.update(session.encode()); h.update(a.tobytes())
        keeps.update(keep.numpy().tobytes()); prefixes.update(lengths.numpy().tobytes())
    return {"sampler_sha256": sampler, "keep_sha256": keeps.hexdigest(), "prefix_sha256": prefixes.hexdigest()}

def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy(); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, dtype=np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()

def same(left: Any, right: Any) -> bool:
    if type(left) is not type(right): return False
    if isinstance(left, torch.Tensor): return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and bool(torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray): return left.dtype == right.dtype and left.shape == right.shape and bool(np.array_equal(left, right))
    if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)): return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right

def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle: temporary = Path(handle.name)
    try:
        torch.save(payload, temporary); os.replace(temporary, destination)
    finally: temporary.unlink(missing_ok=True)

def checkpoint_payload(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: Any, epoch: int, identities: dict[str, str], arm: str, shared_init_sha256: str, bindings: dict[str, Any]) -> dict[str, Any]:
    step = epoch * EPOCH_UPDATES
    if arm not in ARMS or epoch not in range(1, EPOCHS + 1) or ema.n_updates != step: raise RuntimeError("formal checkpoint schedule/EMA drift")
    return {"schema": "h1_queryage_formal_prefix_end_epoch_checkpoint_v1", "arm": arm, "epoch": epoch, "next_batch_index": EPOCH_UPDATES,
            "global_step": step, "models": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "ema": copy.deepcopy(ema.checkpoint_state()), "identities": identities, "shared_init_sha256": shared_init_sha256,
            "bindings_sha256": digest(bindings), "raw_state_sha256": state_digest(model.state_dict()), "ema_state_sha256": state_digest(ema.shadow),
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(), "python": random.getstate()}}

def strict_restore(*, payload: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: Any, epoch: int, identities: dict[str, str], arm: str, shared_init_sha256: str, bindings: dict[str, Any]) -> None:
    step = epoch * EPOCH_UPDATES
    expected = {"schema": "h1_queryage_formal_prefix_end_epoch_checkpoint_v1", "arm": arm, "epoch": epoch, "next_batch_index": EPOCH_UPDATES,
                "global_step": step, "identities": identities, "shared_init_sha256": shared_init_sha256, "bindings_sha256": digest(bindings)}
    if any(payload.get(k) != v for k, v in expected.items()) or payload.get("ema", {}).get("n_updates") != step: raise RuntimeError("checkpoint identity drift")
    if state_digest(payload["models"]) != payload.get("raw_state_sha256") or state_digest(payload["ema"]["shadow"]) != payload.get("ema_state_sha256"): raise RuntimeError("checkpoint disk state drift")
    model.load_state_dict(payload["models"], strict=True); optimizer.load_state_dict(payload["optimizer"]); ema.load_checkpoint_state(payload["ema"])
    ema.shadow = {key: value.to(next(model.parameters()).device) for key, value in ema.shadow.items()}
    if state_digest(model.state_dict()) != payload["raw_state_sha256"] or state_digest(ema.shadow) != payload["ema_state_sha256"]: raise RuntimeError("restored state differs from payload")
    torch.set_rng_state(payload["rng"]["torch"]); torch.cuda.set_rng_state_all(payload["rng"]["cuda"]); np.random.set_state(payload["rng"]["numpy"]); random.setstate(payload["rng"]["python"])

def earliest_argmax(scores: list[tuple[int, float]]) -> tuple[int, float]:
    if len(scores) != EPOCHS or [epoch for epoch, _ in scores] != list(range(1, EPOCHS + 1)) or not np.isfinite([score for _, score in scores]).all(): raise RuntimeError("requires twelve finite ordered EMA scores")
    return max(scores, key=lambda item: item[1])

def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, mode="w", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name); json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    try: os.replace(temporary, destination)
    finally: temporary.unlink(missing_ok=True)

def require_frozen_authority(output: Path) -> dict[str, Any]:
    frozen = json.loads((Path(output) / "input_authority.json").read_text()); bound = frozen.get("bindings")
    if not isinstance(bound, dict) or bound.get("protocol_sha256") != protocol_sha256() or bound.get("code_closure") != code_closure(): raise RuntimeError("formal code/protocol closure drift")
    if (bound.get("protocol") != PROTOCOL or bound.get("output") != str(Path(output))
            or bound.get("physical_devices") != ARM_DEVICE or bound.get("threads") != 1
            or bound.get("capacity_state_used_for_warmstart") is not False
            or not {str(PROTOCOL_DOC),str(CACHE),str(CACHE_AUTHORITY)}.issubset(bound.get("inputs", {}))):
        raise RuntimeError("formal complete recipe/output/input authority required")
    if any(sha(Path(path)) != value for path, value in bound.get("inputs", {}).items()): raise RuntimeError("formal immutable input drift")
    authorization = Path(frozen.get("authorization_path", ""))
    if not authorization.is_absolute() or sha(authorization) != frozen.get("authorization_sha256"):
        raise RuntimeError("external root authorization path/SHA drift")
    actual = json.loads(authorization.read_text())
    if (actual.get("schema") != "h1_queryage_formal_prefix_root_authorization_v1"
            or actual.get("mode") != "formal" or actual.get("status") != "ROOT_REVIEW_GO" or actual.get("bindings") != bound):
        raise RuntimeError("external root authorization status/bindings drift")
    return frozen

def load_readonly_cache(torch_module=torch, *, source_loader=None, allow_cpu_fixture=False) -> dict[str, Any]:
    """Load only an existing immutable cache; injection is test-only."""
    if source_loader is not None:
        if not allow_cpu_fixture: raise RuntimeError("source loader injection is test-only")
        cache, authority = source_loader()
    else:
        if not CACHE.is_file() or not CACHE_AUTHORITY.is_file(): raise FileNotFoundError("existing source cache/authority required")
        cache, authority = torch_module.load(CACHE, map_location="cpu", weights_only=False), json.loads(CACHE_AUTHORITY.read_text())
    from tfpd_exploration.src.h1_optimized_v2.cache import validate_authority
    validate_authority(cache, authority)
    if not {"train", "minival"}.issubset(cache) or len(cache["train"]) != 13: raise RuntimeError("immutable H1 cache topology drift")
    return cache

def _all_epoch_identities(cache: dict[str, Any]) -> dict[str, dict[str, str]]:
    from tfpd_exploration.src.h1_optimized_v4.paired_train import batches
    result = {}
    for epoch in range(1, EPOCHS + 1):
        ordered = batches(cache, epoch)
        if sum(len(starts) for _, starts in ordered) != 23212: raise RuntimeError("source sampler cardinality drift")
        result[str(epoch)] = identity_digest(ordered, cache=cache, epoch=epoch)
    return result

def _guarded_score(model: torch.nn.Module, callback, *, guard=lambda: None):
    guard(); before = state_digest(model.state_dict())
    try: result = callback()
    finally:
        if state_digest(model.state_dict()) != before: raise RuntimeError("scoring mutated RAW model")
    guard()
    return result

def resource_guard(started, device):
    """Whole-lifecycle wall, peak allocation, resident memory and thread limit."""
    if device.type == "cuda": torch.cuda.synchronize(device)
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if (time.monotonic() - started > DEADLINE_SECONDS or rss > MEMORY_LIMIT
            or (device.type == "cuda" and torch.cuda.max_memory_allocated() > MEMORY_LIMIT)
            or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1):
        raise RuntimeError("formal wall/GPU/RSS/thread guard")

def _train_update(model, optimizer, ema, x, y, bank, *, epoch, batch_index, device: torch.device, guard=lambda: None) -> tuple[float, torch.Tensor]:
    prefixed, lengths, keep = prefix_and_keep(x, epoch=epoch, batch_index=batch_index, bank_mask=bank.unit_mask, device=device)
    model.train(); optimizer.zero_grad(set_to_none=True); total = 0.0
    for offset in range(0, len(prefixed), MICRO):
        guard(); n = len(prefixed[offset:offset + MICRO]); loss = F.mse_loss(model.forward_last(prefixed[offset:offset + MICRO], bank, dropout_keep=keep[offset:offset + MICRO]), y[offset:offset + MICRO]); guard()
        if not bool(torch.isfinite(loss)): raise RuntimeError("nonfinite formal loss")
        (loss * (n / len(prefixed))).backward(); total += float(loss.detach()) * (n / len(prefixed)); guard()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); guard(); optimizer.step(); ema.update_after_step(model); guard()
    return total, lengths

def worker_run(*, arm: str, output: Path, physical_gpu: int, start_marker: Path, allow_cpu_fixture=False, source_loader=None) -> None:
    """Authorized one-arm formal worker; no CLI and no automatic resume."""
    started = time.monotonic()
    if arm not in ARMS or physical_gpu != ARM_DEVICE[arm]: raise RuntimeError("arm/device contract drift")
    frozen = require_frozen_authority(output)
    if not allow_cpu_fixture and (os.environ.get("H1_QUERYAGE_FORMAL_PREFIX_GO") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu)):
        raise RuntimeError("explicit GO and exact physical GPU visibility required")
    if not allow_cpu_fixture and (not torch.cuda.is_available() or torch.cuda.current_device() != 0): raise RuntimeError("visible cuda:0 required")
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1: raise
    from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
    from tfpd_exploration.src.h1_optimized_v4.paired_train import batches, collate
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from tfpd_exploration.src.h1_family_v1.model import initialization_receipt, route_gate_gradient_l1, zero_gate_parity
    from tfpd_exploration.src.h1_queryage_family_v1.model import make_queryage_localbalanced_pair
    device = torch.device("cpu" if allow_cpu_fixture else "cuda:0")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    guard = lambda: resource_guard(started, device)
    guard(); cache = load_readonly_cache(source_loader=source_loader, allow_cpu_fixture=allow_cpu_fixture); guard(); identities = _all_epoch_identities(cache); guard()
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    flat, route = make_queryage_localbalanced_pair(seed=SEED); init = initialization_receipt(flat, route); shared = state_digest(flat.state_dict())
    session, starts = batches(cache, 1)[0]; row = cache["train"][session]; x_probe, y_probe = collate(row, starts, device); bank_probe = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
    flat, route = flat.to(device), route.to(device)
    # Generate the actual first effective-batch treatment first; slicing a
    # separately generated B=8 would advance the prefix RNG differently.
    probe_full, _, keep_full = prefix_and_keep(x_probe, epoch=1, batch_index=0, bank_mask=bank_probe.unit_mask, device=device); probe, probe_keep = probe_full[:MICRO], keep_full[:MICRO]
    if not allow_cpu_fixture: torch.cuda.synchronize(device)
    guard(); parity = zero_gate_parity(flat, route, probe, bank_probe); guard(); route.zero_grad(set_to_none=True); F.mse_loss(route.forward_last(probe, bank_probe, dropout_keep=probe_keep), y_probe[:MICRO]).backward(); guard(); gate = route_gate_gradient_l1(route)
    if not allow_cpu_fixture: torch.cuda.synchronize(device)
    if parity["max_abs_diff"] != 0.0 or not np.isfinite(gate) or gate <= 0: raise RuntimeError("QueryAge paired g0 contract drift")
    # The probe graph and its RNG effects never enter the formal trajectory.
    del flat, route, probe_full, keep_full, probe, probe_keep, x_probe, y_probe, bank_probe; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    flat, route = make_queryage_localbalanced_pair(seed=SEED)
    if state_digest(flat.state_dict()) != shared: raise RuntimeError("post-probe fresh initialization drift")
    if arm == "flat": selected = flat.to(device); del route
    else: selected = route.to(device); del flat
    atomic_json({"arm": arm, "physical_gpu": physical_gpu, "shared_init_sha256": shared, "identities": identities, "g0_parity": parity, "route_gate_gradient_l1": gate, "initialization": init}, Path(output) / "barrier" / f"{arm}.ready.json")
    barrier_deadline = time.monotonic() + DEADLINE_SECONDS
    while not Path(start_marker).is_file():
        guard()
        if time.monotonic() > barrier_deadline: raise TimeoutError("formal start barrier timeout")
        time.sleep(.05)
    optimizer, ema, records = torch.optim.AdamW(groups(selected), lr=LR, weight_decay=.01), DecoderEMA(selected, decay=EMA), []
    from .formal_prefix_score import score_selection_cached_ema
    for epoch in range(1, EPOCHS + 1):
        losses = []; ordered = batches(cache, epoch)
        if identity_digest(ordered, cache=cache, epoch=epoch) != identities[str(epoch)]: raise RuntimeError("sampler/dropout/prefix schedule drift")
        for batch_index, (session, starts) in enumerate(ordered):
            if not allow_cpu_fixture and (time.monotonic() > barrier_deadline or torch.cuda.max_memory_allocated() > MEMORY_LIMIT): raise RuntimeError("formal wall or memory guard")
            row = cache["train"][session]; x, y = collate(row, starts, device); bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
            step = (epoch - 1) * EPOCH_UPDATES + batch_index + 1
            for group in optimizer.param_groups: group["lr"] = warmup_lr(epoch=epoch, global_step=step)
            loss, _ = _train_update(selected, optimizer, ema, x, y, bank, epoch=epoch, batch_index=batch_index, device=device, guard=guard); losses.append(loss)
        payload = checkpoint_payload(model=selected, optimizer=optimizer, ema=ema, epoch=epoch, identities=identities[str(epoch)], arm=arm, shared_init_sha256=shared, bindings=frozen["bindings"])
        checkpoint = Path(output) / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"
        if checkpoint.exists(): raise FileExistsError(checkpoint)
        guard(); atomic_torch_save(payload, checkpoint); guard(); loaded = torch.load(checkpoint, map_location="cpu", weights_only=False); guard()
        if not same(payload, loaded): raise RuntimeError("checkpoint disk recursive equality drift")
        strict_restore(payload=loaded, model=selected, optimizer=optimizer, ema=ema, epoch=epoch, identities=identities[str(epoch)], arm=arm, shared_init_sha256=shared, bindings=frozen["bindings"])
        guard(); selection = _guarded_score(selected, lambda: score_selection_cached_ema(model=selected, ema=ema, cache=cache, device=device, guard=guard), guard=guard)
        record = {"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "selection": selection, "mean_loss": float(np.mean(losses)), "identities": identities[str(epoch)]}; records.append(record); atomic_json(record, Path(output) / "workers" / f"{arm}_epoch_{epoch:03d}.json")
        require_frozen_authority(output)
    selected_epoch, selected_score = earliest_argmax([(r["epoch"], r["selection"]["r2_concat_float64"]) for r in records])
    atomic_json({"arm": arm, "identities": identities, "shared_init_sha256": shared, "epochs": records, "selected_epoch": selected_epoch, "selected_ema_r2_float64": selected_score}, Path(output) / "workers" / f"{arm}_complete.json")

def finalizer_run(*, arm: str, output: Path, physical_gpu: int, allow_cpu_fixture=False, source_loader=None) -> None:
    """Strict post-freeze reload/export path; launcher writes selection_freeze."""
    started = time.monotonic()
    if arm not in ARMS or physical_gpu != ARM_DEVICE[arm]: raise RuntimeError("finalizer arm/device drift")
    frozen = require_frozen_authority(output)
    if not allow_cpu_fixture and (os.environ.get("H1_QUERYAGE_FORMAL_PREFIX_GO") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu)):
        raise RuntimeError("explicit GO and exact physical GPU visibility required")
    if not allow_cpu_fixture and (not torch.cuda.is_available() or torch.cuda.current_device() != 0): raise RuntimeError("visible cuda:0 required")
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1: raise
    from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
    from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
    from tfpd_exploration.src.h1_queryage_family_v1.model import make_queryage_localbalanced_pair
    from .formal_prefix_score import score_selection_cached_ema, score_complete_cached_ema
    device = torch.device("cpu" if allow_cpu_fixture else "cuda:0")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    guard = lambda: resource_guard(started, device)
    guard(); cache = load_readonly_cache(source_loader=source_loader, allow_cpu_fixture=allow_cpu_fixture); guard()
    freeze = json.loads((Path(output) / "selection_freeze.json").read_text()); ready = json.loads((Path(output) / "barrier" / f"{arm}.ready.json").read_text())
    reports = {}
    for label, row in (("selected", freeze["selected"][arm]), ("epoch12", freeze["epoch12"][arm])):
        epoch, path = int(row["epoch"]), Path(row["checkpoint"])
        if sha(path) != row["checkpoint_sha256"]: raise RuntimeError("frozen checkpoint hash drift")
        pair = make_queryage_localbalanced_pair(seed=SEED); model = (pair[0] if arm == "flat" else pair[1]).to(device); del pair
        optimizer = torch.optim.AdamW(groups(model), lr=LR, weight_decay=.01); ema = DecoderEMA(model, decay=EMA)
        guard(); payload = torch.load(path, map_location="cpu", weights_only=False); guard()
        strict_restore(payload=payload, model=model, optimizer=optimizer, ema=ema, epoch=epoch, identities=ready["identities"][str(epoch)], arm=arm, shared_init_sha256=ready["shared_init_sha256"], bindings=frozen["bindings"])
        guard(); selection = _guarded_score(model, lambda: score_selection_cached_ema(model=model, ema=ema, cache=cache, device=device, guard=guard), guard=guard)
        if abs(selection["r2_concat_float64"] - row["ema_r2_float64"]) > 1e-5: raise RuntimeError("selected minival reproduction drift")
        complete = _guarded_score(model, lambda: score_complete_cached_ema(model=model, ema=ema, cache=cache, device=device, guard=guard), guard=guard)
        arrays = {key.removeprefix("_"): complete.pop(key) for key in ("_prediction", "_target", "_session_id", "_end")}
        export = Path(output) / "exports" / f"{arm}_{label}_complete_native_float64.npz"
        export.parent.mkdir(parents=True, exist_ok=True)
        if export.exists(): raise FileExistsError(export)
        with tempfile.NamedTemporaryFile(dir=export.parent, suffix=".tmp", delete=False) as handle: temporary = Path(handle.name)
        try:
            with temporary.open("wb") as handle: np.savez_compressed(handle, **arrays)
            os.replace(temporary, export)
        finally: temporary.unlink(missing_ok=True)
        plain = export.with_name(f"{arm}_{label}_plain_ema.pt")
        if plain.exists(): raise FileExistsError(plain)
        plain_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for k, v in ema.shadow.items(): plain_state[k] = v.detach().cpu().clone()
        if any(not bool(torch.isfinite(v).all()) for v in plain_state.values()): raise RuntimeError("nonfinite plain EMA export")
        guard(); atomic_torch_save(plain_state, plain); guard()
        if not same(plain_state, torch.load(plain, map_location="cpu", weights_only=True)): raise RuntimeError("plain EMA disk reproduction drift")
        with np.load(export, allow_pickle=False) as check:
            if set(check.files) != set(arrays) or any(not np.array_equal(check[k], arrays[k]) for k in arrays): raise RuntimeError("native complete archive disk equality drift")
        guard(); reports[label] = {"epoch": epoch, "checkpoint_sha256": sha(path), "selection_reproduced": selection, "complete": complete, "complete_archive": str(export), "complete_archive_sha256": sha(export), "plain_ema_path": str(plain), "plain_ema_sha256": sha(plain)}
        del model, optimizer, ema, payload
    require_frozen_authority(output)
    atomic_json({"status": "COMPLETE_POST_FREEZE", "arm": arm, "reports": reports}, Path(output) / "workers" / f"{arm}_final.json")
