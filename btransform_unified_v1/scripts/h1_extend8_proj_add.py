"""Continue 8 epochs from a sealed e24 proj_add checkpoint (floor LR 1e-5).

Does not overwrite the 24-ep historical root. Not EvalAI.
Supports P16/P32 via --proj-dim. Pin GPU with --cuda-visible {0,1}.
After stage-2 +8, write epoch_pick_24_32.json (earliest max minival13 EMA
equal_session_mean on epochs 24-32). That pick is diagnostic; do not
auto-submit over the e24 official candidate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SCRIPTS = Path(__file__).resolve().parent
for _p in (str(PACKAGE_ROOT / "src"), str(WORKSPACE_ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from btransform_unified_v1 import h1_config, plan, receipts  # noqa: E402
from btransform_unified_v1.ema import DecoderEMA  # noqa: E402
from btransform_unified_v1.identity_variant import BTransformerUnifiedDecoderIdentity  # noqa: E402
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout  # noqa: E402

import h1_matrix_mf250_proj_add as mf250  # noqa: E402
import h1_stage2_full13_proj_add as s2  # noqa: E402

TRAIN_ENV_FLAG = "BTRANSFORM_H1_EXTEND8_TRAIN"
SEED = plan.SEED
EXTRA_EPOCHS = 8
FROM_EPOCH = 24
FLOOR_LR = 1.0e-5
FORBIDDEN_ROOTS = {
    PACKAGE_ROOT / "results/h1_matrix/M_F250_lr1e4_20260906T094633Z",
    PACKAGE_ROOT / "results/h1_matrix/M_F150_lr1e4_20260906T125517Z",
}


def _seal(path: Path, payload: Any) -> str:
    return receipts.seal_json(path, payload)


def _copy_prior_metrics(source: Path, dest: Path) -> None:
    src = source / "metrics.jsonl"
    if src.is_file():
        shutil.copy2(src, dest / "metrics.jsonl")


def epoch_pick_24_32(metrics: dict[int, dict[str, Any]], dest: Path, *, proj_dim: int) -> dict[str, Any]:
    """Earliest max minival13 EMA eq on epochs 24-32. Diagnostic only."""
    window = list(range(FROM_EPOCH, FROM_EPOCH + EXTRA_EPOCHS + 1))
    missing = [e for e in window if e not in metrics]
    plan.require(not missing, f"epoch-pick missing epochs {missing}")
    mini = {e: float(metrics[e]["minival13_ema"]["equal_session_mean"]) for e in window}
    exam: dict[int, float] = {}
    for e in window:
        row = metrics[e]
        exam_row = row.get("exam_0120_ema") or row.get("lodo_exam_ema")
        if exam_row:
            exam[e] = float(exam_row["equal_session_mean"])
    best = max(mini.values())
    pick = min(e for e, v in mini.items() if abs(v - best) <= 1e-10)
    exam_pick = None
    if exam:
        exam_best = max(exam.values())
        exam_pick = min(e for e, v in exam.items() if abs(v - exam_best) <= 1e-10)
    payload = {
        "schema": "btransform_unified_v1_h1_stage2_extend8_epoch_pick",
        "rule": "earliest max minival13 EMA equal_session_mean over epochs 24-32",
        "window_epochs": window,
        "proj_dim": int(proj_dim),
        "pick_epoch": pick,
        "pick_ckpt": str(dest / f"epoch_{pick:03d}.pt"),
        "pick_minival13_ema_equal_mean": mini[pick],
        "e24_minival13_ema_equal_mean": mini[FROM_EPOCH],
        "endpoint32_minival13_ema_equal_mean": mini[FROM_EPOCH + EXTRA_EPOCHS],
        "delta_vs_e24": mini[pick] - mini[FROM_EPOCH],
        "beats_e24": bool(mini[pick] > mini[FROM_EPOCH] + 1e-10),
        "minival13_series": mini,
        "exam_0120_series": exam,
        "exam_pick_epoch": exam_pick,
        "exam_pick_equal_mean": exam.get(exam_pick) if exam_pick is not None else None,
        "note": (
            "stage-2 minival13 and exam 01-20 are polluted (all 13 in train). "
            "Official HO is the only selector-bearing number. Do not auto-submit the pick."
        ),
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "epoch_pick_24_32.json", payload)
    return payload


def run_extend(
    *, kind: str, source: Path, dest: Path, window: int, micro: int, proj_dim: int = 16
) -> dict[str, Any]:
    s2.PROJ_DIM = int(proj_dim)
    s2.IDENTITY_MODE = "proj_add"
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    ckpt_path = source / f"epoch_{FROM_EPOCH:03d}.pt"
    plan.require(ckpt_path.is_file(), f"missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if kind == "lodo_f250":
        plan.require(window == 250, "LODO F250 extend is L=250 only")
        faces = mf250.build_faces()
        train_sessions = faces["train"]["sessions"]
        plan.require(len(train_sessions) == 11, "LODO train must stay 11 sessions")
        geometry = s2._make_geometry(window)
        score_primary = ("exam", faces["exam"]["sessions"], "lodo_exam")
    elif kind == "stage2":
        faces = s2.build_faces(window)
        train_sessions = faces["train"]["sessions"]
        plan.require(len(train_sessions) == 13, "stage-2 extend must keep all 13")
        geometry = s2._make_geometry(window)
        score_primary = ("minival", faces["minival"]["sessions"], "minival13")
    else:
        raise ValueError(kind)

    model = BTransformerUnifiedDecoderIdentity(
        geometry,
        seed=SEED,
        override_prefix=0,
        override_window=window,
        identity_mode="proj_add",
        **s2._proj_kwargs(),
    ).to(device)
    plan.require(
        int(model.init_meta["token_in"]) == s2._expected_token_in(),
        f"token_in drift: {model.init_meta['token_in']} != {s2._expected_token_in()}",
    )
    model.load_state_dict(ckpt["raw_state_dict"])
    ema = DecoderEMA(model, decay=plan.EMA_DECAY)
    ema.load_state_dict(ckpt["ema"])

    updates_per_epoch = int(
        sum(math.ceil(len(faces["train"]["X"][s]) / s2.EFFECTIVE_BATCH) for s in train_sessions)
    )
    expected = 597 if kind == "lodo_f250" else s2.FORMAL12_UPDATES_PER_EPOCH
    plan.require(updates_per_epoch == expected, f"caliber drift {updates_per_epoch} != {expected}")
    global_step = int(ckpt.get("global_step", FROM_EPOCH * updates_per_epoch))
    plan.require(global_step == FROM_EPOCH * updates_per_epoch, f"e24 step {global_step}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FLOOR_LR, weight_decay=plan.WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8
    )
    rng = np.random.default_rng(SEED + FROM_EPOCH)
    accum = s2.EFFECTIVE_BATCH // micro
    plan.require(micro * accum == s2.EFFECTIVE_BATCH, "micro x accum != 32")

    _seal(
        dest / "extend_meta.json",
        {
            "schema": "btransform_unified_v1_h1_extend8_meta",
            "kind": kind,
            "source": str(source),
            "window": window,
            "proj_dim": int(proj_dim),
            "token_in": s2._expected_token_in(),
            "from_epoch": FROM_EPOCH,
            "extra_epochs": EXTRA_EPOCHS,
            "lr": FLOOR_LR,
            "lr_note": "cosine already at floor after 24ep; supplement is constant 1e-5",
            "adamw_moments": "reinitialized (e24 ckpt has no optimizer state)",
            "rng_note": "new Generator(SEED+24); not a replay of the original 24-ep consumption walk",
            "updates_per_epoch": updates_per_epoch,
            "micro": micro,
            "utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    (dest / "PROGRESS_10MIN.md").write_text(
        f"# extend8 {kind} L={window} from {source.name}\n\n"
        "| ep | train_mse | primaryEMA eq | primaryEMA pool | ruling |\n"
        "|----|-----------|---------------|-----------------|--------|\n",
        encoding="utf-8",
    )

    metrics_path = dest / "metrics.jsonl"
    heartbeat = dest / "heartbeat.json"
    started = time.monotonic()
    deadline = started + s2.BUDGET_SECONDS
    train_mse_series: dict[int, float] = {}
    primary_ema: dict[int, dict[str, Any]] = {}
    primary_raw: dict[int, dict[str, Any]] = {}

    for epoch in range(FROM_EPOCH + 1, FROM_EPOCH + EXTRA_EPOCHS + 1):
        model.train()
        epoch_t0 = time.monotonic()
        order = list(train_sessions)
        rng.shuffle(order)
        losses: list[float] = []
        batch_id = -1
        lr = FLOOR_LR
        for session in order:
            X, y = faces["train"]["X"][session], faces["train"]["y"][session]
            bank = faces["train"]["banks"][session]
            idx = rng.permutation(len(X))
            for offset in range(0, len(idx), s2.EFFECTIVE_BATCH):
                if time.monotonic() >= deadline:
                    raise RuntimeError("extend8 4h budget hit")
                take = idx[offset : offset + s2.EFFECTIVE_BATCH]
                micro_losses = []
                for m_off in range(0, len(take), micro):
                    m_take = take[m_off : m_off + micro]
                    batch_id += 1
                    global_step += 1
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(unit_dropout_seed(SEED, epoch, batch_id))
                    keep = whole_unit_dropout(
                        torch.from_numpy(bank.unit_mask.copy()), p=plan.UNIT_DROPOUT, generator=generator
                    )
                    xb = torch.from_numpy(X[m_take]).to(device)
                    yb = torch.from_numpy(y[m_take] * s2.SCALE).to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = model(xb, bank, dropout_keep=keep)
                        loss = nn.functional.mse_loss(pred.float(), yb) / accum
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    micro_losses.append(float(loss.detach().cpu()) * accum)
                nn.utils.clip_grad_norm_(list(model.trainable_parameters().values()), plan.GRAD_CLIP)
                optimizer.step()
                ema.update_after_step(model)
                losses.extend(micro_losses)

        raw = s2._score_view(model, ema, "RAW", faces[score_primary[0]], score_primary[1], device)
        ema_rep = s2._score_view(
            model, ema, "EMA", faces[score_primary[0]], score_primary[1], device, bridge_check=True
        )
        extra = {}
        if kind == "lodo_f250":
            extra["sel2908_ema"] = s2._score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
        else:
            extra["exam_0120_ema"] = s2._score_view(model, ema, "EMA", faces["exam"], faces["exam"]["sessions"], device)
            extra["sel2908_ema"] = s2._score_view(model, ema, "EMA", faces["sel"], faces["sel"]["sessions"], device)
        train_mse_series[epoch] = float(np.mean(losses)) if losses else float("nan")
        primary_raw[epoch] = raw
        primary_ema[epoch] = ema_rep
        row = {
            "event": "epoch",
            "epoch": epoch,
            "kind": kind,
            "window": window,
            "train_mse": train_mse_series[epoch],
            f"{score_primary[2]}_raw": raw,
            f"{score_primary[2]}_ema": ema_rep,
            **extra,
            "lr": lr,
            "seconds": time.monotonic() - epoch_t0,
            "global_step": global_step,
            "ema_updates": ema.n_updates,
            "unix": time.time(),
            "extend8": True,
        }
        if kind == "lodo_f250":
            row["lodo_exam_raw"] = raw
            row["lodo_exam_ema"] = ema_rep
        else:
            row["minival13_raw"] = raw
            row["minival13_ema"] = ema_rep
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        dest.joinpath("heartbeat.json").write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
        with (dest / "PROGRESS_10MIN.md").open("a", encoding="utf-8") as prog:
            prog.write(
                f"| {epoch} | {row['train_mse']:.6f} | {ema_rep['equal_session_mean']:.4f} | "
                f"{ema_rep['pooled_r2']:.4f} | EXTEND |\n"
            )
        print(
            f"[extend8 {kind} L={window}] ep{epoch} loss={train_mse_series[epoch]:.5f} "
            f"EMA={ema_rep['equal_session_mean']:.4f}/{ema_rep['pooled_r2']:.4f} "
            f"({row['seconds']:.0f}s)",
            flush=True,
        )
        torch.save(
            {
                "schema": "btransform_unified_v1_h1_extend8_ckpt",
                "kind": kind,
                "window": window,
                "epoch": epoch,
                "global_step": global_step,
                "seed": SEED,
                "raw_state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "ema": ema.state_dict(),
                "lr": lr,
            },
            dest / f"epoch_{epoch:03d}.pt",
        )

    end_epoch = FROM_EPOCH + EXTRA_EPOCHS
    summary = {
        "schema": "btransform_unified_v1_h1_extend8_train_receipt",
        "status": "COMPLETED",
        "kind": kind,
        "window": window,
        "source": str(source),
        "epochs": list(range(FROM_EPOCH + 1, end_epoch + 1)),
        "endpoint_epoch": end_epoch,
        "primary_ema_equal_mean_by_epoch": {e: primary_ema[e]["equal_session_mean"] for e in primary_ema},
        "endpoint_primary_ema_equal_mean": primary_ema[end_epoch]["equal_session_mean"],
        "endpoint_primary_ema_pooled": primary_ema[end_epoch]["pooled_r2"],
        "train_mse": train_mse_series,
        "elapsed_s": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    _seal(dest / "train_receipt.json", summary)

    if kind == "stage2":
        faces_s2 = faces
        faces_s2["window"] = window
        # temporarily point s2.EPOCHS so receipt uses e32
        old = s2.EPOCHS
        s2.EPOCHS = end_epoch
        try:
            receipt = s2.run_cell_receipt(dest, faces_s2)
        finally:
            s2.EPOCHS = old
        receipt["extend8"] = True
        receipt["legal_checkpoint"]["epoch"] = end_epoch
        metrics = s2._read_epoch_metrics(dest)
        pick = epoch_pick_24_32(metrics, dest, proj_dim=int(proj_dim))
        receipt["epoch_pick_24_32"] = {
            "pick_epoch": pick["pick_epoch"],
            "pick_minival13_ema_equal_mean": pick["pick_minival13_ema_equal_mean"],
            "delta_vs_e24": pick["delta_vs_e24"],
            "beats_e24": pick["beats_e24"],
            "rule": pick["rule"],
        }
        dest.joinpath("cell_receipt_content.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
        )
        _seal(dest / "cell_receipt.json", receipt)
    else:
        metrics = s2._read_epoch_metrics(dest)
        exam_ema_eq = {e: float(metrics[e]["lodo_exam_ema"]["equal_session_mean"]) for e in metrics}
        pick = min(
            e
            for e, v in exam_ema_eq.items()
            if abs(v - max(exam_ema_eq.values())) <= 1e-10
        )
        receipt = {
            "schema": "btransform_unified_v1_mf250_extend8_cell_receipt",
            "cell": "M-F250-PROJ-ADD-EXT8",
            "source_24ep": str(source),
            "sel2_rule": "earliest max on LODO exam EMA equal_session_mean over the 32-epoch series",
            "SEL-2_epoch_pick": pick,
            "SEL-2_pick_exam_ema_equal_mean": exam_ema_eq[pick],
            "SEL-1_endpoint32_exam_ema_equal_mean": exam_ema_eq.get(end_epoch),
            "exam_ema_equal_mean_series": exam_ema_eq,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        _seal(dest / "cell_receipt.json", receipt)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lodo_f250", "stage2"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--micro", type=int, default=32)
    parser.add_argument("--proj-dim", type=int, default=16, choices=[16, 32])
    parser.add_argument("--cuda-visible", default="0", choices=["0", "1"])
    args = parser.parse_args()
    if os.environ.get(TRAIN_ENV_FLAG) != "1":
        print(f"REFUSED: {TRAIN_ENV_FLAG}=1 required", file=sys.stderr)
        return 2
    pin = str(args.cuda_visible)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != pin:
        print(f"REFUSED: CUDA_VISIBLE_DEVICES must be pinned to {pin!r}", file=sys.stderr)
        return 2
    dest = args.dest.resolve()
    source = args.source.resolve()
    plan.require(dest not in FORBIDDEN_ROOTS, f"refusing to write into {dest}")
    plan.require(source != dest, "dest must be a new directory (historical 24-ep root stays sealed)")
    dest.mkdir(parents=True, exist_ok=True)
    s2.WINDOW = int(args.window)
    s2.PROJ_DIM = int(args.proj_dim)
    s2.IDENTITY_MODE = "proj_add"
    s2.CUDA_PIN = pin
    pre = s2.gpu_preflight(dest / "preflight_gpu.json")
    if not pre["ok"]:
        _seal(dest / "BLOCKED.json", {"reason": f"GPU{pin} not clean", "preflight": pre})
        return 2
    _copy_prior_metrics(source, dest)
    # also copy epoch_024 so receipts can reload it if needed
    src_ckpt = source / f"epoch_{FROM_EPOCH:03d}.pt"
    if src_ckpt.is_file() and not (dest / src_ckpt.name).exists():
        os.symlink(src_ckpt, dest / src_ckpt.name)
    summary = run_extend(
        kind=args.kind,
        source=source,
        dest=dest,
        window=int(args.window),
        micro=int(args.micro),
        proj_dim=int(args.proj_dim),
    )
    print(json.dumps({"status": summary["status"], "endpoint": summary["endpoint_primary_ema_equal_mean"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
