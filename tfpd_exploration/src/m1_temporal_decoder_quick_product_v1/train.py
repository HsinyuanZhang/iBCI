"""M1 FLAT/ROUTE training. Do not start while H1 still holds both GPUs."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import plan as fold_plan
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
    M1TemporalFlatDecoder,
    M1TemporalRouteDecoder,
    adamw_param_groups,
    whole_unit_dropout,
)

from .config import (
    EFFECTIVE_BATCH,
    EPOCHS_TARGET,
    GRAD_CLIP,
    LR,
    N_UNITS,
    PROFILE_PATH,
    RESULT_ROOT,
    SEED,
    SLOT_ROOT,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from tfpd_exploration.src.m1_temporal_v2 import plan as v2_plan

from .data import build_fold0_train_datamodule, materialize_source_banks_for_train as materialize_source_banks
from .ema import DecoderEMA

MICROBATCH = 8


def _require_profile() -> dict[str, Any]:
    if not PROFILE_PATH.is_file():
        raise RuntimeError("refusing to train before m1_cost_profile.json exists")
    report = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if report.get("status") != "PROFILE_OK":
        raise RuntimeError(f"refusing to train: profile status {report.get('status')}")
    return report


def _lr_at_step(step: int, updates_per_epoch: int) -> float:
    warmup_steps = max(int(updates_per_epoch) * int(WARMUP_EPOCHS), 1)
    if step <= warmup_steps:
        return LR * float(step) / float(warmup_steps)
    return LR


def _dropout_keep(batch: int, seed: int, epoch: int, batch_id: int, device: torch.device) -> torch.Tensor:
    import hashlib

    digest = hashlib.sha256(f"m1_temporal_unit_dropout|{seed}|{epoch}|{batch_id}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    keep = whole_unit_dropout(torch.ones(N_UNITS, dtype=torch.bool), p=0.10, generator=generator)
    return keep.reshape(-1).unsqueeze(0).expand(batch, -1).contiguous().to(device)


def _heartbeat(arm: str, dest: Path, payload: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps({"arm": arm, "updated": datetime.now(timezone.utc).isoformat(), **payload}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_slot(arm: str, epoch: int, status: str) -> None:
    slot = "S5" if arm == "flat" else "S6"
    name = "M1-TEMPORAL-TRF-FLAT" if arm == "flat" else "M1-TEMPORAL-TRF-ROUTE"
    SLOT_ROOT.mkdir(parents=True, exist_ok=True)
    dest = SLOT_ROOT / f"slot_{slot}.json"
    prior_id = None
    if dest.is_file():
        try:
            prior_id = json.loads(dest.read_text(encoding="utf-8")).get("submission_id")
        except (OSError, json.JSONDecodeError):
            prior_id = None
    dest.write_text(
        json.dumps(
            {
                "slot": slot,
                "name": name,
                "owner": "M1-D",
                "status": status,
                "epoch": epoch,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "submission_id": prior_id,
                "disclosure": (
                    "M1-TEMPORAL-v2 / rSyn3-refit-v1; fold0 source-only new decoder; "
                    "frozen S-Fix B3 + named refit bank; not the sealed Stage-0 P carrier"
                ),
                "carrier_revision": v2_plan.REVISION,
                "carrier_name": v2_plan.CARRIER_NAME,
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_model(arm: str, device: torch.device):
    if arm == "flat":
        return M1TemporalFlatDecoder(seed=SEED).to(device)
    template = M1TemporalFlatDecoder(seed=SEED)
    return M1TemporalRouteDecoder(seed=SEED, flat_template=template).to(device)


def _session_name(session: Any) -> str:
    text = session.decode("ascii") if isinstance(session, bytes) else str(session)
    return text if text.startswith("ses-") else str(session)


def main(
    arm: str = "flat",
    max_epochs: int | None = None,
    start_epoch: int = 1,
    max_steps: int | None = None,
) -> None:
    if arm not in {"flat", "route"}:
        raise ValueError(arm)
    if not v2_plan.BANK_NPZ.is_file() or not v2_plan.BANK_RECEIPT.is_file():
        raise RuntimeError("refusing to train before M1-TEMPORAL-v2 / rSyn3-refit-v1 is sealed")
    _require_profile()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    dest = RESULT_ROOT / arm
    dest.mkdir(parents=True, exist_ok=True)
    _write_slot(arm, start_epoch - 1, "TRAINING")
    data = build_fold0_train_datamodule()
    banks = materialize_source_banks()
    gpu_banks = {
        name: type(bank)(
            E0=bank.E0.to(device),
            T=bank.T.to(device),
            unit_mask=bank.unit_mask.to(device),
        )
        for name, bank in banks.items()
    }
    loader = data.train_dataloader()
    updates_per_epoch = len(loader)
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
    last_epoch = int(max_epochs or EPOCHS_TARGET)
    window = int(fold_plan.WINDOW_SIZE)
    for epoch in range(start_epoch, last_epoch + 1):
        epoch_loss = 0.0
        for batch_id, batch in enumerate(loader):
            neural, target, _calib, session, _carrier = batch[:5]
            name = _session_name(session[0] if not isinstance(session, str) else session)
            if name not in gpu_banks:
                raise RuntimeError(f"batch session {name} is not a fold0 source")
            neural = neural.to(device, dtype=torch.float32)
            if neural.dim() == 4:
                neural = neural.squeeze(-1) if neural.size(-1) == 1 else neural.mean(-1)
            if neural.size(-1) != N_UNITS and neural.size(1) == N_UNITS:
                neural = neural.transpose(1, 2)
            target = target.to(device, dtype=torch.float32)
            if target.dim() == 3:
                target = target[:, -1, :]
            bank = gpu_banks[name]
            n = int(neural.size(0))
            global_step += 1
            lr = _lr_at_step(global_step, updates_per_epoch)
            for group in opt.param_groups:
                group["lr"] = lr
            keep = _dropout_keep(n, SEED, epoch, batch_id, device)
            opt.zero_grad(set_to_none=True)
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
            if max_steps is not None and global_step >= int(max_steps):
                dest.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema": "m1_temporal_v2_smoke_ckpt",
                        "arm": arm,
                        "epoch": epoch,
                        "global_step": global_step,
                        "seed": SEED,
                        "raw_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "ema": ema.checkpoint_state(),
                        "optimizer": opt.state_dict(),
                        "train_mse": running,
                        "carrier_revision": v2_plan.REVISION,
                    },
                    dest / "smoke_100.pt",
                )
                _heartbeat(
                    arm,
                    dest / "heartbeat.json",
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "last_loss": running,
                        "smoke": True,
                        "visible": visible,
                    },
                )
                return
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
                        "window": window,
                    },
                )
        mean_loss = epoch_loss / max(updates_per_epoch, 1)
        torch.save(
            {
                "schema": "m1_temporal_decoder_quick_product_v1_ckpt",
                "arm": arm,
                "epoch": epoch,
                "global_step": global_step,
                "seed": SEED,
                "raw_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "ema": ema.checkpoint_state(),
                "optimizer": opt.state_dict(),
                "train_mse": mean_loss,
            },
            dest / f"epoch_{epoch:03d}.pt",
        )
        _write_slot(arm, epoch, "TRAINING")
        _heartbeat(
            arm,
            dest / "heartbeat.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "epoch_mse": mean_loss,
                "elapsed_s": time.time() - start_wall,
                "visible": visible,
            },
        )
    _write_slot(arm, last_epoch, "TRAIN_COMPLETE" if last_epoch >= EPOCHS_TARGET else "EPOCH_SNAPSHOT")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("flat", "route"), default="flat")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    main(arm=args.arm, max_epochs=args.max_epochs, start_epoch=args.start_epoch, max_steps=args.max_steps)
