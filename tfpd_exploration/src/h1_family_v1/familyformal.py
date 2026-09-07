"""Disposable, source-only CRST-B4 paired-formal resource/serialization smoke.

This module is deliberately *not* a formal trainer.  It creates no formal
output root, does not iterate or score minival, retains no checkpoint, and
uses exactly eight deterministic effective batches solely to measure the
proposed paired recipe's resource envelope.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT as H1_ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v4.paired_train import batches, collate
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .model import initialization_receipt, make_v2_unscaled_dot_localbalanced_pair, route_gate_gradient_l1, zero_gate_parity

OUT = H1_ROOT / "family_v1" / "crst_b4_localbalanced_paired_recipe_smoke_v1"
SEED = 42
MICRO = 8
EFFECTIVE = 32
LR = 1e-4
EMA_DECAY = 0.9995
DROP_P = 0.10
PAIRS = 8
WARM_PAIRS = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dropout_keep(*, n: int, epoch: int, batch: int, bank_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Proposed formal dropout law, explicitly intersected with bank validity."""
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|keep|{epoch}|{batch}".encode()).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_keep = torch.rand((n, bank_mask.numel()), generator=generator) >= DROP_P
    valid = bank_mask.detach().cpu().to(dtype=torch.bool).view(1, -1)
    keep = random_keep & valid  # Never revive a masked bank unit.
    if not bool(keep.any(dim=1).all()):
        raise RuntimeError("dropout law produced an empty valid-unit row")
    return keep.to(device=device)


def _process_memory_mib() -> float | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        pid, memory = [part.strip() for part in line.split(",", 1)]
        if int(pid) == os.getpid():
            return float(memory)
    return None


def _step(model: torch.nn.Module, opt: torch.optim.Optimizer, ema: DecoderEMA, x: torch.Tensor, y: torch.Tensor, bank: H1Bank, keep: torch.Tensor) -> float:
    model.train()
    opt.zero_grad(set_to_none=True)
    parts: list[float] = []
    for offset in range(0, len(x), MICRO):
        pred = model.forward_last(x[offset : offset + MICRO], bank, dropout_keep=keep[offset : offset + MICRO])
        loss = F.mse_loss(pred, y[offset : offset + MICRO]) * (len(pred) / len(x))
        loss.backward()
        parts.append(float(loss.detach().item()))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    ema.update_after_step(model)
    return float(np.mean(parts))


def _strict_serialization_and_finite(arms: dict[str, torch.nn.Module], emas: dict[str, DecoderEMA], x: torch.Tensor, bank: H1Bank) -> dict[str, dict[str, object]]:
    """All serialization is in-memory; no learned state is persisted."""
    out: dict[str, dict[str, object]] = {}
    for arm, model in arms.items():
        # Serialization occurs after training.  Park the inactive formal arm
        # and its EMA shadow on CPU so a B=16 finite check is a one-arm
        # measurement, not a spurious three-model shared-GPU allocation.
        for other, other_model in arms.items():
            target = x.device if other == arm else torch.device("cpu")
            other_model.to(target)
            emas[other].shadow = {name: value.to(target) for name, value in emas[other].shadow.items()}
        torch.cuda.empty_cache()
        torch.cuda.synchronize(x.device)
        with torch.inference_mode():
            raw = model.eval().forward_last(x, bank)
        if not bool(torch.isfinite(raw).all()):
            raise RuntimeError(f"{arm} raw source batch is non-finite")
        raw_bytes = io.BytesIO(); torch.save(model.state_dict(), raw_bytes)
        clone_flat, clone_route = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
        # Keep the strict reload off the shared GPU.  The actual B=16 raw and
        # EMA forwards above are GPU finite checks; a third W=700 model would
        # turn a serialization smoke into an artificial co-schedule OOM.
        clone = {"flat": clone_flat, "route": clone_route}[arm]
        loaded_raw = torch.load(io.BytesIO(raw_bytes.getvalue()), map_location="cpu", weights_only=True)
        clone.load_state_dict(loaded_raw, strict=True)
        if any(not torch.equal(value.cpu(), clone.state_dict()[name]) for name, value in model.state_dict().items()):
            raise RuntimeError(f"{arm} strict raw CPU reload changed a state entry")

        ema_bytes = io.BytesIO(); torch.save(emas[arm].checkpoint_state(), ema_bytes)
        clone_ema = DecoderEMA(clone, decay=EMA_DECAY)
        clone_ema.load_checkpoint_state(torch.load(io.BytesIO(ema_bytes.getvalue()), map_location="cpu", weights_only=True))
        source_ema = emas[arm].checkpoint_state()
        loaded_ema = clone_ema.checkpoint_state()
        if source_ema["decay"] != loaded_ema["decay"] or source_ema["n_updates"] != loaded_ema["n_updates"] or set(source_ema["shadow"]) != set(loaded_ema["shadow"]) or any(not torch.equal(source_ema["shadow"][name].cpu(), loaded_ema["shadow"][name]) for name in source_ema["shadow"]):
            raise RuntimeError(f"{arm} strict EMA CPU reload changed a state entry")
        with torch.inference_mode():
            ema_pred = emas[arm].score_with_ema(model, lambda candidate: candidate.forward_last(x, bank))
        if not bool(torch.isfinite(ema_pred).all()):
            raise RuntimeError(f"{arm} EMA source batch is non-finite")
        out[arm] = {
            "source_batch": int(len(x)),
            "raw_finite": True,
            "ema_finite": True,
            "raw_state_strict_reload": True,
            "ema_state_strict_reload": True,
            "raw_serialized_bytes": len(raw_bytes.getvalue()),
            "ema_serialized_bytes": len(ema_bytes.getvalue()),
        }
        del clone, clone_ema
    return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite smoke receipt root: {OUT}")
    # build_or_load validates cache provenance and may deserialize cached
    # minival arrays for integrity.  This smoke only indexes cache['train']; it
    # never constructs a minival iterator or invokes scoring/selection.
    cache = build_or_load()
    recorded = json.loads((H1_ROOT / "source_cache_authority.json").read_text())
    validate_authority(cache, recorded)
    source_batches = batches(cache, 1)[:PAIRS]
    if len(source_batches) != PAIRS:
        raise RuntimeError("expected eight available full-source sampler batches")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED); np.random.seed(SEED)
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=SEED)
    arms = {"flat": flat.to(device), "route": route.to(device)}
    initial_receipt = initialization_receipt(arms["flat"], arms["route"])
    optimizers = {name: torch.optim.AdamW(groups(model), lr=LR) for name, model in arms.items()}
    emas = {name: DecoderEMA(model, decay=EMA_DECAY) for name, model in arms.items()}
    torch.cuda.reset_peak_memory_stats(device)

    name0, starts0 = source_batches[0]
    row0 = cache["train"][name0]
    x0, y0 = collate(row0, starts0, device)
    bank0 = H1Bank(*[row0["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
    parity = zero_gate_parity(arms["flat"], arms["route"], x0[:MICRO], bank0)
    keep0 = dropout_keep(n=len(x0), epoch=1, batch=0, bank_mask=bank0.unit_mask, device=device)
    if bool((keep0 & ~bank0.unit_mask.to(dtype=torch.bool).view(1, -1)).any()):
        raise RuntimeError("dropout_keep revived a masked bank unit")
    arms["route"].train(); optimizers["route"].zero_grad(set_to_none=True)
    F.mse_loss(arms["route"].forward_last(x0[:MICRO], bank0, dropout_keep=keep0[:MICRO]), y0[:MICRO]).backward()
    gate_grad = route_gate_gradient_l1(arms["route"])
    optimizers["route"].zero_grad(set_to_none=True)
    if gate_grad <= 0.0:
        raise RuntimeError("ROUTE gate gradient is zero at g=0")

    pair_losses: list[dict[str, float]] = []
    timed: list[float] = []
    for index, (name, starts) in enumerate(source_batches):
        row = cache["train"][name]
        x, y = collate(row, starts, device)
        bank = H1Bank(*[row["bank"][key].to(device) for key in ("E0", "T", "unit_mask")])
        keep = dropout_keep(n=len(x), epoch=1, batch=index, bank_mask=bank.unit_mask, device=device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses = {arm: _step(model, optimizers[arm], emas[arm], x, y, bank, keep) for arm, model in arms.items()}
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        pair_losses.append(losses)
        if index >= WARM_PAIRS:
            timed.append(elapsed)

    # A source-only B=16 inference checks raw and EMA finite outputs and the
    # strict in-memory serialization path.  No old score evaluator is called.
    # Optimizer moments are irrelevant after the eighth disposable update and
    # would otherwise distort the B=16 serialization memory check.
    del optimizers
    torch.cuda.empty_cache()
    serial = _strict_serialization_and_finite(arms, emas, x0[:16], bank0)
    torch.cuda.synchronize(device)
    mean_pair_s = float(np.mean(timed))
    report = {
        "schema": "h1_crst_b4_localbalanced_paired_recipe_disposable_smoke_v1",
        "status": "COMPLETE_DISPOSABLE_SOURCE_ONLY",
        "formal_training_launched": False,
        "formal_output_root_created": False,
        "retained_learned_candidate": False,
        "source_only": True,
        "cache_open_disclosure": "build_or_load plus validate_authority deserialize cached minival arrays for cache-integrity validation; no minival iterator, scoring, or selection was invoked",
        "cache_authority": recorded,
        "operator_sha256": {"familyformal.py": _sha(Path(__file__)), "model.py": _sha(Path(__file__).with_name("model.py"))},
        "proposed_recipe_frozen_only": {"seed": SEED, "epochs": 12, "source_windows": 23212, "effective_updates_per_arm_per_epoch": 731, "effective_updates_per_arm_total": 8772, "lr": LR, "warmup": "linear through epoch 1 then constant", "microbatch": MICRO, "effective_batch": EFFECTIVE, "dropout": "p=.10 stateless paired random mask AND bank.unit_mask", "ema_decay": EMA_DECAY, "selection": "EMA pooled R2 on frozen 2908 minival endpoints, earliest maximum tie", "complete": "selected and epoch12 EMA on 20325 bins", "fresh_seed": SEED, "warmstart_from_1040": False, "prospective_hard_wall_bound_s": 21600},
        "sampler": {"epoch": 1, "first_effective_batches": PAIRS, "full_source_sampler": "h1_optimized_v4.paired_train.batches", "batch_sessions": [name for name, _ in source_batches], "batch_sizes": [int(len(starts)) for _, starts in source_batches]},
        "start_invariants": {"initialization": initial_receipt, "zero_gate_parity": parity, "route_gate_gradient_l1_at_g0": gate_grad, "dropout_mask_preserves_bank_mask": True},
        "timing": {"warm_pairs": WARM_PAIRS, "timed_pairs": len(timed), "per_pair_seconds": timed, "mean_pair_seconds": mean_pair_s, "projected_train_only_seconds_paired_sequential": mean_pair_s * 731 * 12, "projection_note": "pair time includes both sequential arms; multiply mean paired effective-batch time by 731x12. Individual-arm time was not measured. Validation is excluded and must be measured separately."},
        "losses": pair_losses,
        "memory": {"peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20, "process_memory_mib_after_sync": _process_memory_mib()},
        "source_batch16_raw_ema_serialization": serial,
    }
    OUT.mkdir(parents=True)
    (OUT / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # Deliberately no torch.save: learned model, optimizer, and EMA state die
    # with this disposable process.


if __name__ == "__main__":
    main()
