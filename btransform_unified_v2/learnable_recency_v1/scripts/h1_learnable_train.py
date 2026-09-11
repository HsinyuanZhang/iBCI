#!/usr/bin/env python3
"""H1 R300 learnable-recency trainer. Mirrors h1_flat_train.py; does not edit it."""
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

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
ROOT = PKG.parent
FLAT = ROOT / "scripts/recency_flat_ablation_v1/h1_flat_train.py"
for path in (PKG / "src", ROOT / "src", ROOT.parent / "btransform_unified_v1/src", ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location("_h1_flat_learnable", FLAT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot privately load h1_flat_train.py")
h1_flat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h1_flat)
signed, ht = h1_flat.signed, h1_flat.ht
np, torch, nn = h1_flat.np, h1_flat.torch, h1_flat.nn

from btransform_unified_v2.model import RiftDecoder
from learnable_recency_v1.config import add_learnable_flags, config_from_args, config_from_run_meta, dataset_config
from learnable_recency_v1.wrap import (
    LearnableRiftStreamDecoder,
    apply_group_lrs,
    assert_shared_byte_equal,
    install_temporal,
    new_parameter_names,
    recency_bias_snapshot,
    split_optimizer_parameters,
    trainable_new_parameter_count,
)

EPOCHS, UPDATES, SEED, BATCH, MICRO = h1_flat.EPOCHS, h1_flat.UPDATES, h1_flat.SEED, h1_flat.BATCH, h1_flat.MICRO
REF_BANKS = h1_flat.REF_BANKS
PROJ_DIM = 16
RESULTS = PKG / "results"


def sha_file(path: Path) -> str:
    return h1_flat.sha_file(path)


def atomic(path: Path, value: Any) -> None:
    h1_flat.atomic(path, value)


def stock_decoder(variant: str, device: torch.device, backend: str, seed: int, proj_dim: int = PROJ_DIM) -> Any:
    """``ht._decoder`` with explicit construction knobs (ht hardcodes SEED=42/P16).

    Bit-identical to ``ht._decoder(variant, device, 300, backend)`` at seed 42,
    proj_dim 16: same constructor sequence, same RNG domains.
    """
    model = RiftDecoder(task="h1", context_bins=300, bias_mode=variant, seed=seed, proj_dim=proj_dim).to(device)
    model.temporal.set_attention_backend(backend)
    return model


def learnable_decoder(
    device: torch.device, recency_cfg, backend: str = "dense", seed: int = SEED, proj_dim: int = PROJ_DIM
):
    model = stock_decoder("recency", device, backend, seed, proj_dim)
    install_temporal(model, recency_cfg, seed)
    model.temporal.set_attention_backend(backend)
    return model


def shared_sha(model: Any, names: list[str]) -> str:
    digest = hashlib.sha256()
    named = dict(model.named_parameters())
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_paired(
    model: Any,
    device: torch.device,
    reference: dict[str, Any],
    recency_cfg,
    seed: int = SEED,
    proj_dim: int = PROJ_DIM,
) -> dict[str, Any]:
    # Reference binding stays on the frozen seed-42 P16 recency reference; pairing means
    # "same seed + same structure", so a non-42 run re-pairs against its own seed and a
    # non-16 run re-pairs against its own proj_dim.
    stock = stock_decoder("recency", device, "dense", SEED)
    if recency_cfg.layers == 4:
        if ht._sha_state(stock) != reference["reference_initialization_sha256"]:
            raise RuntimeError("fresh recency init SHA differs from current signed-state reference")
        if seed != SEED or proj_dim != PROJ_DIM:
            stock = stock_decoder("recency", device, "dense", seed, proj_dim)
        shared = assert_shared_byte_equal(model, stock)
        recency_sha = shared_sha(stock, shared)
        if shared_sha(model, shared) != recency_sha:
            raise RuntimeError("learnable shared named parameters are not paired to recency")
    else:
        if proj_dim != PROJ_DIM:
            stock = stock_decoder("recency", device, "dense", SEED, proj_dim)
        shared_cfg = dataset_config("h1", tier="fixed", layers=recency_cfg.layers, ladder="default")
        install_temporal(stock, shared_cfg, seed)
        stock.temporal.set_attention_backend("dense")
        shared = assert_shared_byte_equal(model, stock)
        recency_sha = shared_sha(model, shared)
    extra = new_parameter_names(model)
    ladder_cfg = dataset_config(
        "h1", tier="fixed", layers=recency_cfg.layers, half_life_seconds=recency_cfg.half_life_seconds
    )
    ladder_ref = stock_decoder("recency", device, "dense", seed, proj_dim)
    install_temporal(ladder_ref, ladder_cfg, seed)
    ladder_ref.temporal.set_attention_backend("dense")
    z = torch.randn(1, 32, 256, device=device)
    mask = torch.ones(1, 32, dtype=torch.bool, device=device)
    with torch.inference_mode():
        left = model.temporal(z, mask)
        right = ladder_ref.temporal(z, mask)
    if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
        raise RuntimeError("H1 learnable/same-ladder init forward drift")
    del stock, ladder_ref
    return {
        "shared_parameter_names": shared,
        "new_parameter_names": extra,
        "new_parameter_counts": {name: int(dict(model.named_parameters())[name].numel()) for name in extra},
        "trainable_new_parameter_count": trainable_new_parameter_count(model),
        "shared_initialization_sha256": recency_sha,
        "init_forward_max_abs": float((left - right).abs().max().cpu()),
        "ladder": recency_cfg.ladder_metadata(),
    }


def runtime_parity(model: Any, train_data: dict[str, Any], cal1: dict[str, Any], device: Any, label: str) -> dict[str, Any]:
    session = train_data["sessions"][0]
    budget = int(signed.prefix_schedule(0, UPDATES)[0])
    starts = signed.cal1_b2.legal_starts(cal1["starts"][session], cal1["n_trials"][session], budget)
    bank = cal1["banks"][(session, signed.pick_m7_start(session, epoch0=0, step=0, starts=starts), budget)]
    xb = torch.from_numpy(train_data["X"][session][:1]).to(device)
    valid = torch.from_numpy(train_data["valid"][session][:1]).to(device)
    was = model.training
    model.eval()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    try:
        model.temporal.set_attention_backend("dense")
        with torch.inference_mode():
            dense = ht._forward(model, xb, bank, None, valid)
        model.temporal.set_attention_backend("local")
        with torch.inference_mode():
            local = ht._forward(model, xb, bank, None, valid)
        stream = LearnableRiftStreamDecoder(model)
        for index in range(xb.shape[1]):
            stream.stream_step(xb[:, index], bank, ["learnable-parity"], valid_mask=valid[:, index])
        stream_last = stream.predict("learnable-parity")
        if not torch.allclose(dense, local, rtol=2e-4, atol=2e-5) or not torch.allclose(local[0], stream_last, rtol=2e-4, atol=2e-5):
            raise RuntimeError(f"{label}: dense/local/stream prefix parity drift")
        return {
            "label": label,
            "prefix_bins": int(xb.shape[1]),
            "valid_bins": int(valid.sum().item()),
            "dense_local_max_abs": float((dense - local).abs().max().cpu()),
            "local_stream_max_abs": float((local[0] - stream_last).abs().max().cpu()),
        }
    finally:
        model.temporal.set_attention_backend("dense")
        model.train(was)
        with torch.no_grad():
            for name, value in before.items():
                dict(model.named_parameters())[name].copy_(value)


def bias_from_prefix(model: Any, train_data: dict[str, Any], device: Any) -> dict[str, Any]:
    session = train_data["sessions"][0]
    xb = torch.from_numpy(train_data["X"][session][:1]).to(device)
    valid = torch.from_numpy(train_data["valid"][session][:1]).to(device)
    was = model.training
    model.eval()
    with torch.inference_mode():
        bank = next(iter(train_data["banks"].values()))
        tokens = model.frontend_tokens(xb, bank)
        stats = recency_bias_snapshot(model.temporal, tokens, valid)
    model.train(was)
    return stats


def rebuild_model(meta: dict[str, Any], device: torch.device) -> Any:
    """Rebuild the learnable decoder for scoring from the run's recorded seed/proj_dim."""
    recency_cfg = config_from_run_meta(meta, "h1")
    return learnable_decoder(
        device,
        recency_cfg,
        str(meta.get("attention_backend", "dense")),
        int(meta.get("seed", SEED)),
        int(meta.get("proj_dim", PROJ_DIM)),
    )


def score(args: Any, recency_cfg, carriers: dict[str, Any] | None = None) -> dict[str, Any]:
    torch.set_num_threads(2)
    dest = args.dest
    meta = json.loads((dest / "run_meta.json").read_text())
    reference = h1_flat.validate_reference(args.banks)
    if carriers is None:
        _, _, carriers = signed._load_banks(args.banks)
    device = torch.device(args.device)
    model = rebuild_model(meta, device)
    ema = signed.DecoderEMA(model, decay=0.9995)
    ho = signed.build_ho_signed(300, carriers)
    curve = []
    for epoch in range(1, EPOCHS + 1):
        path = dest / f"epoch_{epoch:03d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["raw_state_dict"], strict=True)
        ema.load_state_dict(state["ema"])
        report = signed.score_ho_m3(model, ema, ho, device)
        curve.append(
            {
                "epoch": epoch,
                "epoch_zero_based": epoch - 1,
                signed.HO_SELECTION_METRIC: report["r2_mean"],
                "worst_session_r2": report["worst_session_r2"],
                "session_std_population": report["r2_std_population"],
                "per_session_r2": report["per_session_r2"],
            }
        )
    selected = signed.select_epoch(curve)
    atomic(
        dest / "ho_m3_selection.json",
        {
            "status": "HO_M3_DEVELOPMENT_SELECTION",
            "selected": selected,
            "curve": curve,
            "selection_rule": "same current HO-M3 grouped-seven, all32, earliest maximum",
        },
    )
    atomic(
        dest / "train_receipt.json",
        {
            "status": "COMPLETED",
            "epochs": EPOCHS,
            "updates": EPOCHS * UPDATES,
            "selected_epoch": selected["epoch"],
            "official_test_used": False,
            "reference": reference,
        },
    )
    return {"status": "SCORE_COMPLETED", "selected_epoch": selected["epoch"]}


def train(args: Any) -> dict[str, Any]:
    recency_cfg = config_from_args(args, "h1")
    reference = h1_flat.validate_reference(args.banks)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    seed = int(args.seed)
    proj_dim = int(args.proj_dim)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dest = args.dest
    if dest.exists() and any(dest.iterdir()) and not args.resume:
        raise FileExistsError(f"fresh destination required: {dest}")
    plan, bank_receipt, carriers = signed._load_banks(args.banks)
    train_data = ht.build_train(300)
    cal1 = signed.build_cal1_signed(train_data["banks"], plan)
    digests = ht._digest_train_contract(train_data, cal1)
    if digests != reference["reference_pairing_digests"]:
        raise RuntimeError("learnable endpoint/bank/valid-mask contract differs from current recency reference")
    model = learnable_decoder(device, recency_cfg, "dense", seed, proj_dim)
    init = assert_paired(model, device, reference, recency_cfg, seed, proj_dim)
    ema = signed.DecoderEMA(model, decay=0.9995)
    groups = split_optimizer_parameters(model, peak_lr=1e-4, weight_decay=0.01, lr_multiplier=recency_cfg.lr_multiplier)
    opt = torch.optim.AdamW(groups, lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(seed)
    dest.mkdir(parents=True, exist_ok=True)
    smoke = args.max_updates_smoke is not None
    meta = {
        "schema": "h1_signedstate14_learnable_v1",
        "cell": f"CURRENT_H1_SIGNEDSTATE14_R300_LEARNABLE_{recency_cfg.tier.upper()}_S{seed}",
        "status": "SMOKE" if smoke else "FORMAL",
        "smoke": smoke,
        "tier": recency_cfg.tier,
        "learnable_config": recency_cfg.__dict__,
        "new_parameter_names": init["new_parameter_names"],
        "new_parameter_counts": init["new_parameter_counts"],
        "trainable_new_parameter_count": init["trainable_new_parameter_count"],
        "context_bins": 300,
        "proj_dim": proj_dim,
        "layer_windows": list(recency_cfg.temporal_config.windows),
        "depth": recency_cfg.layers,
        "ladder": recency_cfg.ladder_metadata(),
        "attention_backend": "dense",
        "epochs": EPOCHS if not smoke else 1,
        "updates_per_epoch": UPDATES,
        "batch": BATCH,
        "microbatch": MICRO,
        "seed": seed,
        "ema": 0.9995,
        "unit_dropout": 0.1,
        "initialization_pairing": init,
        "banks_receipt_sha256": sha_file(args.banks / "receipt.json"),
        "pairing_digests": digests,
        "reference": reference,
        "official_test_used": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic(dest / "run_meta.json", meta)
    parity_pre = runtime_parity(model, train_data, cal1, device, "pre-train")
    atomic(dest / "runtime_parity_pre.json", parity_pre)
    total = EPOCHS * UPDATES
    step = 0
    started = time.monotonic()
    last_bias = None
    last_loss = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        schedule = signed.prefix_schedule(epoch - 1, UPDATES)
        order = list(train_data["sessions"])
        rng.shuffle(order)
        si = 0
        for session in order:
            index = rng.permutation(len(train_data["X"][session]))
            for off in range(0, len(index), BATCH):
                take = index[off : off + BATCH]
                budget = int(schedule[si])
                starts = signed.cal1_b2.legal_starts(cal1["starts"][session], cal1["n_trials"][session], budget)
                bank = cal1["banks"][(session, signed.pick_m7_start(session, epoch0=epoch - 1, step=si, starts=starts), budget)]
                step += 1
                apply_group_lrs(opt, signed.warmup_cosine_lr(step, total, UPDATES, peak=1e-4, min_factor=0.1))
                keep = signed.whole_unit_dropout(
                    torch.from_numpy(bank.unit_mask.copy()),
                    p=0.1,
                    generator=torch.Generator().manual_seed(signed.unit_dropout_seed(seed, epoch, si)),
                )
                opt.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for moff in range(0, len(take), MICRO):
                    subset = take[moff : moff + MICRO]
                    xb = torch.from_numpy(train_data["X"][session][subset]).to(device)
                    valid = torch.from_numpy(train_data["valid"][session][subset]).to(device)
                    yb = torch.from_numpy(train_data["y"][session][subset] * signed.SCALE).to(device)
                    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
                    with amp:
                        loss = nn.functional.mse_loss(ht._forward(model, xb, bank, keep, valid).float(), yb)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"nonfinite loss epoch={epoch} step={step}")
                    (loss * (len(subset) / len(take))).backward()
                    batch_loss += float(loss.detach()) * len(subset) / len(take)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                ema.update_after_step(model)
                losses.append(batch_loss)
                last_loss = batch_loss
                si += 1
                if args.max_updates_smoke and step >= args.max_updates_smoke:
                    break
            if args.max_updates_smoke and step >= args.max_updates_smoke:
                break
        last_bias = bias_from_prefix(model, train_data, device)
        ht._checkpoint(
            dest / f"epoch_{epoch:03d}.pt",
            model,
            opt,
            ema,
            epoch,
            step,
            rng,
            variant=f"learnable_{recency_cfg.tier}",
            context_bins=300,
            attention_backend="dense",
            microbatch=MICRO,
            smoke=smoke,
        )
        row = {
            "event": "epoch",
            "seed": seed,
            "epoch": epoch,
            "global_step": step,
            "train_mse": float(np.mean(losses)),
            "smoke": smoke,
            "bias": last_bias,
        }
        with (dest / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        atomic(dest / "heartbeat.json", row)
        if smoke:
            break
    train_elapsed = time.monotonic() - started
    parity_post = runtime_parity(model, train_data, cal1, device, "post-train")
    atomic(dest / "runtime_parity_post.json", parity_post)
    if smoke:
        receipt = {
            "schema": "h1_signedstate14_learnable_smoke_receipt_v1",
            "status": "SMOKE_COMPLETED",
            "seed": seed,
            "tier": recency_cfg.tier,
            "steps": step,
            "finite_loss": last_loss is not None and bool(np.isfinite(last_loss)),
            "runtime_parity_pre": parity_pre,
            "runtime_parity_post": parity_post,
            "bias": last_bias,
            "new_parameter_names": init["new_parameter_names"],
            "new_parameter_counts": init["new_parameter_counts"],
            "trainable_new_parameter_count": init["trainable_new_parameter_count"],
            "train_elapsed_seconds": train_elapsed,
        }
        atomic(dest / "smoke_receipt.json", receipt)
        return receipt
    if step != total:
        raise RuntimeError("formal update total drift")
    return score(args, recency_cfg, carriers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_learnable_flags(parser)
    parser.add_argument("--config", default="h1")
    parser.add_argument("--banks", type=Path, default=REF_BANKS)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("train", "score"), default="train")
    parser.add_argument("--max-updates-smoke", type=int)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM)
    return parser


def default_dest(args: Any, recency_cfg) -> Path:
    """Seed 42 / P16 keep the historical default name; other values carry run knobs."""
    if args.seed == SEED and args.proj_dim == PROJ_DIM:
        name = f"h1_{args.tier}_s{SEED}"
    elif args.proj_dim == PROJ_DIM:
        name = f"h1_{args.tier}_{recency_cfg.ladder}_s{args.seed}"
    else:
        name = f"h1_{args.tier}_{recency_cfg.ladder}_p{args.proj_dim}_s{args.seed}"
    if args.max_updates_smoke:
        extra = "" if int(args.layers) == 4 else f"_layers{int(args.layers)}"
        return (RESULTS / "smoke" / f"{name}{extra}_v2").resolve()
    return (RESULTS / name).resolve()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config != "h1":
        parser.error("this runner is the H1 config")
    if args.seed <= 0:
        parser.error("seed must be positive")
    if args.proj_dim <= 0:
        parser.error("proj_dim must be positive")
    if args.smoke_steps is not None:
        args.max_updates_smoke = args.smoke_steps
    recency_cfg = config_from_args(args, "h1")
    args.banks = args.banks.resolve()
    if args.dest is None:
        args.dest = default_dest(args, recency_cfg)
    else:
        args.dest = args.dest.resolve()
    if args.max_updates_smoke is not None and args.device == "cuda:0":
        args.device = "cpu"
    payload = score(args, recency_cfg) if args.stage == "score" else train(args)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
