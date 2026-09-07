#!/usr/bin/env python3
"""TFAP Stage-2: strict-27 whole-model fine-tune from a Stage-1 pretrained SWA.

TFAP_CONTRACT_20260817.md §3:

- whole-model `load_state_dict(stage1_swa_state, strict=True)` — encoder AND
  decoder, no partial load, no reset, no freeze;
- FRESH Adam (optimizer state cleared at the phase boundary: nothing carries
  over from Stage 1), then EXACTLY arm A's recipe: step-level linear warmup
  1e-5 -> 1e-4 over the first two epochs, cosine to 1e-6 at the final step of
  epoch 47, 48 epochs on the strict-27 full window set (33,925 steps/epoch),
  the same SessionBatchSampler (batch 32, seed 42);
- visible side: canonical normalized T4 under the strict-27 source-only
  normalizer (deployment contract); within/external/formal data never opened;
- predeclared final-four SWA (epochs 44-47) + final checkpoint, immutable;
- 0444 launch/terminal receipts, per-epoch §9 diagnostics, closure equality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

PAD_VALUE = -1.0
BOUND_PATTERNS = (
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_tfap_finetune.py",
    "scripts/run_tfap_pretrain.py",
    "scripts/run_admission_arm.py",
    "src/tfpd/spintshape_module.py",
)
STAGE1_DIRS = {
    "pt4": ROOT / "results/tfap_stage1_v1/pretrain_pt4",
    "pz4": ROOT / "results/tfap_stage1_v1/pretrain_pz4",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
matched_scorer = _load_module("tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py")
receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-arm", required=True, choices=["pt4", "pz4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tfap_stage2_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument(
        "--stage1-dir", type=Path, default=None,
        help="override the stage-1 arm directory (relaunch roots); default from arm name",
    )
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    out_dir = Path(args.output_root) / f"finetune_{args.pretrain_arm}"
    if args.smoke:
        out_dir = Path(str(out_dir) + "_smoke")
    if out_dir.exists():
        print(f"fresh finetune output directory required: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        return _run(args, out_dir, device, started)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        receipt_mod.write_receipt_transactionally(
            out_dir / "terminal_receipt.json",
            {
                "schema": "tfap_stage2_v1",
                "pretrain_arm": args.pretrain_arm,
                "status": "FINETUNE_FAILED",
                "started_utc": started,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "failure": {"kind": type(exc).__name__, "detail": str(exc),
                            "traceback": traceback.format_exc()},
            },
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def _moment_norm(model, optimizer, name):
    tensor = arm_common.w_side_moment(model, optimizer, name)
    return 0.0 if tensor is None else float(tensor.norm().item())


def _run(args, out_dir: Path, device, started: str) -> int:
    import lightning.pytorch as pl
    from mc_maze.multisession_datamodule import SessionBatchSampler
    from torch.nn.parameter import UninitializedParameter
    from torch.utils.data import default_collate

    spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")
    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    stage1_dir = Path(args.stage1_dir) if args.stage1_dir else STAGE1_DIRS[args.pretrain_arm]
    stage1_terminal = json.loads((stage1_dir / "terminal_receipt.json").read_text())
    accepted = ("PRETRAIN_TERMINAL",) + (
        ("PRETRAIN_SMOKE_COMPLETE__NON_AUTHORITATIVE",) if args.smoke else ()
    )
    if stage1_terminal["status"] not in accepted:
        raise SystemExit(f"stage-1 arm not terminal: {stage1_terminal['status']}")
    stage1_receipt_sha = receipt_mod.sha256_file(stage1_dir / "terminal_receipt.json")
    swa_path = stage1_dir / "swa_final4.pt"
    swa_sidecar = Path(str(swa_path) + ".sha256")
    if not swa_path.is_file() or not swa_sidecar.is_file():
        raise SystemExit(f"stage-1 SWA missing: {swa_path}")
    swa_sha = receipt_mod.sha256_file(swa_path)
    if swa_sha != swa_sidecar.read_text().split()[0]:
        raise SystemExit("stage-1 SWA SHA mismatch against sidecar")
    swa_payload = torch.load(swa_path, map_location="cpu", weights_only=False)
    state = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in swa_payload["state_dict"].items()
    }

    epochs = args.smoke_epochs if args.smoke else args.epochs
    if not args.smoke and epochs != 48:
        raise SystemExit("stage-2 budget is frozen at 48 epochs")

    # ---- strict-27 data (same contract as arm A; dev rosters never opened) --
    dm, a2 = arm_runner.build_datamodule(args)
    train_dataset = dm.train_dataset
    roster = tuple(dm.session_splits["train"])
    if len(roster) != 27:
        raise SystemExit("strict-27 roster drift")
    manifest_sha = receipt_mod.sha256_file(a2.MANIFEST_PATH)
    if manifest_sha != a2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest SHA drift")
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    if not behavior_semantic.startswith("f062506c"):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    pl.seed_everything(args.seed, workers=True)
    model = spintshape.build_spintshape_model(seed=args.seed)
    model.load_state_dict(state, strict=True)  # whole model, no partial load
    state_sha_loaded = arm_common.state_sha256(model)
    model.to(device)

    sampler = SessionBatchSampler(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, seed=args.seed
    )
    steps_per_epoch = len(sampler)
    if not args.smoke and steps_per_epoch != 33925:
        raise SystemExit(f"steps/epoch drift: {steps_per_epoch} != 33925")
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, _fx_sess, fx_side = fixed_batch[:5]
    fx_neural, fx_calib, fx_side = fx_neural.to(device), fx_calib.to(device), fx_side.to(device)

    optimizer = torch.optim.Adam(  # fresh: stage-1 optimizer state NOT carried over
        model.parameters(),
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    schedule = arm_common.schedule_params(epochs, steps_per_epoch)
    lr_fn = lambda step: arm_common.lr_at_step(step, epochs, steps_per_epoch)

    launch = {
        "schema": "tfap_stage2_v1_launch",
        "status": "FINETUNE_LAUNCHED",
        "pretrain_arm": args.pretrain_arm,
        "smoke": args.smoke,
        "started_utc": started,
        "recipe": "arm A verbatim: warmup 1e-5->1e-4 over two epochs, cosine to 1e-6, 48 epochs, strict-27 full window set",
        "stage1_provenance": {
            "terminal_receipt": str(stage1_dir / "terminal_receipt.json"),
            "terminal_receipt_sha256": stage1_receipt_sha,
            "swa_path": str(swa_path), "swa_sha256": swa_sha,
            "visible_side_in_stage1": stage1_terminal["visible_side"],
        },
        "initial_state": {
            "source": "stage-1 final-four SWA (whole model)",
            "strict_load": True,
            "state_dict_sha256_after_load": state_sha_loaded,
            "optimizer_state_cleared": True,
            "no_weight_reset": True,
        },
        "budget": {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "schedule": schedule,
            "optimizer": arm_common.ADAM_CONSTRUCTOR,
            "sampler": "mc_maze SessionBatchSampler(batch=32, shuffle, seed=42)",
        },
        "data_contract": {
            "roster_n": len(roster), "manifest_sha256": manifest_sha,
            "within_dev_sessions_opened": False,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "behavior_normalizer_semantic_sha256": behavior_semantic,
            "visible_side": "canonical normalized T4 (strict-27 source-only normalizer)",
        },
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "dev_validation_early_stopping_or_checkpoint_selection": False,
            "clipping": "none",
        },
        "source_closure": closure_launch,
        "environment": {
            "device": str(device), "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "launch_receipt.json", launch)

    swa_local = set(range(max(0, epochs - 4), epochs))
    diagnostics, checkpoints, invariant_failures = [], [], []
    phase_step = 0
    for epoch in range(epochs):
        t0 = time.time()
        stats = arm_runner.train_epoch(
            model, optimizer, loader, "t4", lr_fn, device, phase_step,
            max_steps=args.max_train_steps,
        )
        phase_step += stats["optimizer_steps"]
        w_side = arm_common.w_side_block(model)
        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter) and p.numel()
        )
        with torch.no_grad():
            contribution = arm_common.post_pool_contribution(model, fx_calib[:8], fx_side[:8])
            attention = arm_common.attention_summary(
                model, fx_neural[:4], fx_calib[:4], arm_common.admit_side(fx_side[:4], "t4")
            )
        authority_ok = arm_common.t4_authority_fingerprint(train_dataset.sessions) == t4_authority
        diag = {
            "epoch": epoch,
            "visible_side_mode": "t4",
            "duration_s": round(time.time() - t0, 3),
            **stats,
            "optimizer_steps_total": phase_step,
            "lr_expected_first": schedule["lr_at_step_0"],
            "lr_expected_last": schedule["lr_at_final_step"],
            "clipping_authorized": False,
            "w_side_norm": float(w_side.norm().item()),
            "norm_alpha_times_w_side": float(w_side.norm().item()),
            "w_side_exp_avg_norm": _moment_norm(model, optimizer, "exp_avg"),
            "w_side_exp_avg_sq_norm": _moment_norm(model, optimizer, "exp_avg_sq"),
            **contribution,
            **attention,
            "t4_authority_unchanged": authority_ok,
            "parameters_finite": params_finite,
            "optimizer_state_finite": arm_runner.optimizer_state_finite(optimizer),
            "state_dict_sha256": arm_common.state_sha256(model),
            "optimizer_state_sha256": arm_common.optimizer_sha256(optimizer),
            "checkpoint_saved": None,
        }
        ok_steps = (
            stats["optimizer_steps"] == min(args.max_train_steps, steps_per_epoch)
            if args.max_train_steps is not None
            else stats["optimizer_steps"] == steps_per_epoch
        )
        if not all([
            stats["visible_side_violation_count"] == 0,
            stats["nonfinite_loss_steps"] == 0,
            stats["nonfinite_grad_steps"] == 0,
            params_finite, diag["optimizer_state_finite"], authority_ok, ok_steps,
        ]):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfap_stage2_ckpt_v1",
                    "pretrain_arm": args.pretrain_arm,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "state_dict_sha256": diag["state_dict_sha256"],
                    "optimizer_state_sha256": diag["optimizer_state_sha256"],
                },
                path,
            )
            sha = arm_runner.seal_file(path)
            diag["checkpoint_saved"] = name
            checkpoints.append({"file": name, "epoch": epoch, "sha256": sha})
        diagnostics.append(diag)
        print(json.dumps({
            "pretrain_arm": args.pretrain_arm, "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "lr_last": stats["lr_last"], "w_side": round(diag["w_side_norm"], 6),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa2_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa2_path
    )
    swa2_sha = arm_runner.seal_file(swa2_path)
    swa2_state = torch.load(swa2_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = spintshape.build_spintshape_model(seed=args.seed)
    swa_model.load_state_dict(swa2_state, strict=True)
    swa_model.to(device).eval()
    with torch.no_grad():
        pred, _ = swa_model(fx_neural[:4], calib_trials=fx_calib[:4], side_features=fx_side[:4])
        swa_finite = bool(torch.isfinite(pred).all().item())
    del swa_model

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure_launch["closure_sha256"]
    status = (
        ("FINETUNE_SMOKE_COMPLETE__NON_AUTHORITATIVE" if args.smoke else "FINETUNE_TERMINAL")
        if not invariant_failures and closure_equal and len(diagnostics) == epochs
        else ("FINETUNE_SMOKE_INVARIANT_FAILURE" if args.smoke else "FINETUNE_INVARIANT_OR_CLOSURE_FAILURE")
    )
    receipt_mod.write_receipt_transactionally(
        out_dir / "terminal_receipt.json",
        {
            "schema": "tfap_stage2_v1",
            "status": status,
            "pretrain_arm": args.pretrain_arm,
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recipe": launch["recipe"],
            "stage1_provenance": launch["stage1_provenance"],
            "initial_state": launch["initial_state"],
            "budget": launch["budget"],
            "data_contract": launch["data_contract"],
            "epochs_run": len(diagnostics),
            "diagnostics_per_epoch": diagnostics,
            "invariant_failures": invariant_failures,
            "checkpoints": checkpoints,
            "swa": {
                "path": str(swa2_path), "sha256": swa2_sha,
                "window_epochs": [c["epoch"] for c in checkpoints],
                "manifest": swa_manifest,
                "strict_reload_finite_forward_smoke": swa_finite,
            },
            "disclosures": launch["disclosures"],
            "source_closure": {"launch": closure_launch, "final": closure_final,
                               "launch_final_closure_equal": closure_equal},
            "environment": launch["environment"],
        },
    )
    print(json.dumps({
        "pretrain_arm": args.pretrain_arm, "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "w_side_final": round(diagnostics[-1]["w_side_norm"], 6),
        "swa_sha256": swa2_sha[:16],
    }, indent=1))
    return 0 if status in ("FINETUNE_TERMINAL", "FINETUNE_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
