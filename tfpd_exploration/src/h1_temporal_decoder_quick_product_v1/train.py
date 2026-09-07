"""H1 FLAT/ROUTE training. One process per arm; CUDA_VISIBLE_DEVICES selects the card."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1Bank,
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
    adamw_param_groups,
    whole_unit_dropout,
)

from .config import (
    EFFECTIVE_BATCH,
    EPOCHS_TARGET,
    GRAD_CLIP,
    LR,
    RESULT_ROOT,
    SEED,
    SLOT_ROOT,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from .data import build_window_manifest, collate_batch, materialize_banks, shuffled_batches
from .ema import DecoderEMA
from .profile import PROFILE_PATH

MICROBATCH = 8
ACCUM = EFFECTIVE_BATCH // MICROBATCH


def _require_profile() -> dict[str, Any]:
    if not PROFILE_PATH.is_file():
        raise RuntimeError("refusing to train before h1_cost_profile.json exists")
    report = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "PROFILE_OK":
        raise RuntimeError(f"refusing to train: profile status {report.get('status')}")
    return report


def _lr_at_step(step: int, updates_per_epoch: int) -> float:
    if step < 1:
        raise ValueError("step is 1-based")
    warmup_steps = max(int(updates_per_epoch) * int(WARMUP_EPOCHS), 1)
    if step <= warmup_steps:
        return LR * float(step) / float(warmup_steps)
    return LR


def _dropout_keep(batch: int, seed: int, epoch: int, batch_id: int, device: torch.device) -> torch.Tensor:
    import hashlib

    digest = hashlib.sha256(f"h1_temporal_unit_dropout|{seed}|{epoch}|{batch_id}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    keep = whole_unit_dropout(torch.ones(176, dtype=torch.bool), p=0.10, generator=generator)
    keep = keep.reshape(-1)
    return keep.unsqueeze(0).expand(batch, -1).contiguous().to(device)


def _heartbeat(arm: str, dest: Path, payload: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = {"arm": arm, "updated": datetime.now(timezone.utc).isoformat(), **payload}
    dest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_slot(arm: str, epoch: int, status: str) -> None:
    import json as _json

    slot = "S3" if arm == "flat" else "S4"
    name = "H1-TEMPORAL-TRF-FLAT" if arm == "flat" else "H1-TEMPORAL-TRF-ROUTE"
    dest = SLOT_ROOT / f"slot_{slot}.json"
    SLOT_ROOT.mkdir(parents=True, exist_ok=True)
    current: dict = {}
    if dest.is_file():
        current = _json.loads(dest.read_text(encoding="utf-8"))
    if current.get("submission_id") or current.get("status") == "REGISTERED":
        current["train_epoch"] = epoch
        current["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        current["updated"] = datetime.now(timezone.utc).isoformat()
        dest.write_text(_json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    current.update(
        {
            "slot": slot,
            "name": name,
            "owner": "H1",
            "status": status,
            "epoch": epoch,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "disclosure": current.get("disclosure") or "known-source development; early snapshot is not 12ep",
            "updated": datetime.now(timezone.utc).isoformat(),
        }
    )
    dest.write_text(_json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_model(arm: str, device: torch.device):
    if arm == "flat":
        return H1TemporalFlatDecoder(seed=SEED).to(device)
    template = H1TemporalFlatDecoder(seed=SEED)
    return H1TemporalRouteDecoder(seed=SEED, flat_template=template).to(device)


def main(arm: str = "flat", max_epochs: int | None = None, start_epoch: int = 1) -> None:
    if arm not in {"flat", "route"}:
        raise ValueError(arm)
    _require_profile()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    dest = RESULT_ROOT / arm
    dest.mkdir(parents=True, exist_ok=True)
    _write_slot(arm, start_epoch - 1, "TRAINING")
    pack = build_window_manifest()
    train_sessions = pack["train_sessions"]
    banks = materialize_banks(train_sessions)
    gpu_banks = {
        name: H1Bank(
            E0=bank.E0.to(device),
            T=bank.T.to(device),
            unit_mask=bank.unit_mask.to(device),
        )
        for name, bank in banks.items()
    }
    model = _build_model(arm, device)
    model.train()
    opt = torch.optim.AdamW(adamw_param_groups(model, WEIGHT_DECAY), lr=LR)
    ema = DecoderEMA(model)
    global_step = 0
    if start_epoch > 1:
        prev = dest / f"epoch_{start_epoch - 1:03d}.pt"
        payload = torch.load(prev, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["raw_state_dict"])
        opt.load_state_dict(payload["optimizer"])
        ema.load_checkpoint_state(payload["ema"])
        global_step = int(payload["global_step"])
        model.to(device)
    start_wall = time.time()
    probe = shuffled_batches(train_sessions, seed=SEED, epoch=1, batch_size=EFFECTIVE_BATCH)
    updates_per_epoch = len(probe)
    last_epoch = int(max_epochs or EPOCHS_TARGET)
    for epoch in range(start_epoch, last_epoch + 1):
        batches = shuffled_batches(train_sessions, seed=SEED, epoch=epoch, batch_size=EFFECTIVE_BATCH)
        epoch_loss = 0.0
        for batch_id, items in enumerate(batches):
            session, neural, target, _starts = collate_batch(train_sessions, items)
            neural = neural.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            bank = gpu_banks[session]
            opt.zero_grad(set_to_none=True)
            n = int(neural.size(0))
            global_step += 1
            lr = _lr_at_step(global_step, updates_per_epoch)
            for group in opt.param_groups:
                group["lr"] = lr
            keep = _dropout_keep(n, SEED, epoch, batch_id, device)
            running = 0.0
            for offset in range(0, n, MICROBATCH):
                sl = slice(offset, min(offset + MICROBATCH, n))
                chunk = int(neural[sl].size(0))
                pred = model.forward_last(neural[sl], bank, dropout_keep=keep[sl])
                loss = F.mse_loss(pred, target[sl]) * (chunk / float(n))
                loss.backward()
                running += float(loss.detach().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ema.update_after_step(model)
            epoch_loss += running
            if global_step % 20 == 0:
                _heartbeat(
                    arm,
                    dest / "heartbeat.json",
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "lr": lr,
                        "last_loss": running,
                        "visible": visible,
                        "device": str(device),
                    },
                )
        mean_loss = epoch_loss / max(len(batches), 1)
        ckpt = {
            "schema": "h1_temporal_decoder_quick_product_v1_ckpt",
            "arm": arm,
            "epoch": epoch,
            "global_step": global_step,
            "seed": SEED,
            "lr": lr,
            "raw_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": ema.checkpoint_state(),
            "optimizer": opt.state_dict(),
            "train_mse": mean_loss,
            "visible": visible,
            "updates_per_epoch": updates_per_epoch,
        }
        torch.save(ckpt, dest / f"epoch_{epoch:03d}.pt")
        _heartbeat(
            arm,
            dest / "heartbeat.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "lr": lr,
                "epoch_mse": mean_loss,
                "elapsed_s": time.time() - start_wall,
                "visible": visible,
            },
        )
        _write_slot(arm, epoch, "TRAINING")
        (dest / "summary.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "last_epoch": epoch,
                    "global_step": global_step,
                    "epoch_mse": mean_loss,
                    "elapsed_s": time.time() - start_wall,
                    "updates_per_epoch": updates_per_epoch,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_slot(arm, last_epoch, "TRAIN_COMPLETE" if last_epoch >= EPOCHS_TARGET else "EPOCH_SNAPSHOT")


def train_pair(max_epochs: int | None = None) -> None:
    last = int(max_epochs or EPOCHS_TARGET)
    for epoch in range(1, last + 1):
        main(arm="flat", max_epochs=epoch, start_epoch=epoch)
        main(arm="route", max_epochs=epoch, start_epoch=epoch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("flat", "route", "pair"), default="flat")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    args = parser.parse_args()
    if args.arm == "pair":
        train_pair(max_epochs=args.max_epochs)
    else:
        main(arm=args.arm, max_epochs=args.max_epochs, start_epoch=args.start_epoch)
