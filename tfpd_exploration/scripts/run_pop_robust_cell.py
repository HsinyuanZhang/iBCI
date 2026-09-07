#!/usr/bin/env python3
"""Population-robustness cell runner: D (2 heads) and DH (64 heads).

HANDOFF_POPULATION_ROBUSTNESS_NEXT_ROUND_20260817.md: both cells are EXACT
arm-A replicas (strict-27 full window set, 33,925 steps/epoch, batch-32
seed-42 session sampler, fresh Adam, step-level warmup 1e-5 -> 1e-4 over two
epochs then cosine to 1e-6 at the final step of epoch 47, 48 epochs,
predeclared final-four SWA) with ONLY two changes: the decoder head count and
`dynamic_dropout=True` (low/high 0.0/1.0, the existing implementation
verbatim — one `random.uniform` p and one PyTorch unit-mask dropout per
training forward; never active in eval/scoring).

Both cells strict-load the SAME canonical_initial_state.pt artifact
(state sha 65bacb85…): MHA parameter shapes are head-count independent, only
the head partition changes, and construction consumes the identical RNG
stream — the launch receipt carries the bitwise equality proof.

Per-epoch diagnostics (0444 receipts): the §9 subset used across this lane
(loss, steps, LR, global/per-branch grad norms, W_side norms, T4/calibration
contribution ratio, finiteness, state/optimizer SHAs) PLUS the population
instrumentation: sampled-p summary (the model's own draws, never re-sampled),
realized retained-unit fraction, all-zero-population sample count, pre/post
dropout token norms, fc_in input norm, and (both cells) per-head attention
entropies with the across-head diversity statistic.  Launch/terminal receipts
record the head count and dropout configuration explicitly.
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
    "src/tfpd_lane/pop_robust.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_pop_robust_cell.py",
    "scripts/run_admission_arm.py",
)
CELL_NAMES = {"D": "cellD_2heads_dynamic_dropout", "DH": "cellDH_64heads_dynamic_dropout"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=["D", "DH"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/pop_robust_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument(
        "--initial-state", type=Path,
        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
    )
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    out_dir = Path(args.output_root) / CELL_NAMES[args.cell]
    if args.smoke:
        out_dir = Path(str(out_dir) + "_smoke")
    if out_dir.exists():
        print(f"fresh cell output directory required: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        return _run(args, out_dir, device, started)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
        receipt_mod.write_receipt_transactionally(
            out_dir / "terminal_receipt.json",
            {
                "schema": "tfpd_pop_robust_cell_v1",
                "cell": args.cell,
                "status": "CELL_FAILED",
                "smoke": args.smoke,
                "started_utc": started,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "failure": {"kind": type(exc).__name__, "detail": str(exc),
                            "traceback": traceback.format_exc()},
            },
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def _run(args, out_dir: Path, device, started: str) -> int:
    import lightning.pytorch as pl
    from mc_maze.multisession_datamodule import SessionBatchSampler
    from torch.nn.parameter import UninitializedParameter
    from torch.utils.data import default_collate

    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    pop_robust = _load_module("tfpd_lane_pop_robust", ROOT / "src/tfpd_lane/pop_robust.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    config = pop_robust.CELLS[args.cell]

    # ---- canonical initial state: strict load + bitwise equality proof -----
    initial_sidecar = Path(str(args.initial_state) + ".sha256")
    if not args.initial_state.is_file() or not initial_sidecar.is_file():
        raise SystemExit("canonical initial state artifact missing")
    initial_sha = receipt_mod.sha256_file(args.initial_state)
    if initial_sha != initial_sidecar.read_text().split()[0]:
        raise SystemExit("canonical initial state SHA mismatch")
    initial_payload = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    canonical_state = initial_payload["state_dict"]

    # ---- data (arm-A contract; development rosters never opened) ------------
    dm, a2 = arm_runner.build_datamodule(args)
    train_dataset = dm.train_dataset
    if len(dm.session_splits["train"]) != 27:
        raise SystemExit("strict-27 roster drift")
    if receipt_mod.sha256_file(a2.MANIFEST_PATH) != a2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest SHA drift")
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    if not behavior_semantic.startswith("f062506c"):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    pl.seed_everything(args.seed, workers=True)
    model = pop_robust.build_population_robustness_model(seed=args.seed, cell=args.cell)
    model.load_state_dict(canonical_state, strict=True)
    state_sha_loaded = arm_common.state_sha256(model)
    if state_sha_loaded != initial_payload["state_sha256"]:
        raise SystemExit("loaded state SHA != canonical artifact state SHA")
    proof = pop_robust.initial_state_equality_proof(canonical_state, seed=args.seed)
    for cell_proof in proof.values():
        if not cell_proof["state_keys_equal_to_canonical"] or cell_proof["shape_mismatches"]:
            raise SystemExit(f"initial-state equality proof failed for {cell_proof}")
    if len({p["trainable_parameters"] for p in proof.values()}) != 1:
        raise SystemExit("parameter-count drift across cells")
    model.to(device)

    sampler = SessionBatchSampler(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, seed=args.seed
    )
    steps_per_epoch = len(sampler)
    epochs = args.smoke_epochs if args.smoke else args.epochs
    if not args.smoke and (steps_per_epoch != 33925 or epochs != 48):
        raise SystemExit("budget drift: expected 33,925 steps/epoch over 48 epochs")
    loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, _fx_sess, fx_side = fixed_batch[:5]
    fx_neural, fx_calib, fx_side = (
        fx_neural.to(device), fx_calib.to(device), fx_side.to(device),
    )

    optimizer = torch.optim.Adam(
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
        "schema": "tfpd_pop_robust_cell_v1_launch",
        "status": "CELL_LAUNCHED",
        "cell": args.cell,
        "cell_name": CELL_NAMES[args.cell],
        "head_and_dropout_config": {
            "num_heads": config["num_heads"],
            "dynamic_dropout": config["dynamic_dropout"],
            "dynamic_dropout_low": pop_robust.DYNAMIC_DROPOUT_LOW,
            "dynamic_dropout_high": pop_robust.DYNAMIC_DROPOUT_HIGH,
            "note": "the ONLY changes vs arm A; everything else is the exact arm-A recipe",
        },
        "smoke": args.smoke,
        "started_utc": started,
        "initial_state": {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "loaded_state_sha256": state_sha_loaded,
            "strict_load": True,
            "bitwise_equality_proof": proof,
        },
        "budget": {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "schedule": schedule, "optimizer": arm_common.ADAM_CONSTRUCTOR,
            "sampler": "mc_maze SessionBatchSampler(batch=32, shuffle, seed=42)",
        },
        "data_contract": {
            "roster_n": 27,
            "within_dev_sessions_opened": False,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "visible_side": "canonical normalized T4",
            "behavior_normalizer_semantic_sha256": behavior_semantic,
        },
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "pretraining_used": False,
            "width_changed": False,
            "extra_seed": False,
            "clipping": "none",
            "prohibited_arms_present": ["auto 64-head without dropout", "no-dropout cell"],
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
        with pop_robust.dynamic_dropout_recorder() as record, \
                pop_robust.fc_in_token_norm_recorder(model) as token_norms:
            stats = arm_runner.train_epoch(
                model, optimizer, loader, "t4", lr_fn, device, phase_step,
                max_steps=args.max_train_steps,
            )
        record["fc_in_token_norms"] = token_norms
        phase_step += stats["optimizer_steps"]
        dropout_summary = pop_robust.summarize_dropout_record(record, config["num_heads"])
        fixed_diag = pop_robust.fixed_batch_dropout_diagnostic(
            model, fx_neural[:8], fx_calib[:8], fx_side[:8]
        )
        if dropout_summary["n_forwards_with_sampled_p"] != stats["optimizer_steps"]:
            raise SystemExit(
                "dynamic-dropout sampling drift: sampled-p count != optimizer steps"
            )

        w_side = arm_common.w_side_block(model)
        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter) and p.numel()
        )
        with torch.no_grad():
            contribution = arm_common.post_pool_contribution(model, fx_calib[:8], fx_side[:8])
            model.eval()
            head_summary = pop_robust.per_head_attention_summary(
                model, fx_neural[:4], fx_calib[:4], fx_side[:4]
            )
            model.train()
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
            **contribution,
            "head_and_dropout_config": launch["head_and_dropout_config"],
            "dynamic_dropout_summary": dropout_summary,
            "fixed_batch_dropout_diagnostic": fixed_diag,
            "per_head_attention": head_summary,
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
            head_summary["n_heads"] == config["num_heads"],
            dropout_summary["n_recorded_unit_mask_calls"] == stats["optimizer_steps"],
        ]):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfpd_pop_robust_ckpt_v1",
                    "cell": args.cell,
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
            "cell": args.cell, "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "lr_last": stats["lr_last"],
            "p_mean": round(dropout_summary["sampled_p_mean"] or 0.0, 4),
            "retained": round(dropout_summary.get("retained_unit_fraction_mean") or 0.0, 4),
            "head_entropy_std": round(head_summary["across_head_entropy_std"] or 0.0, 4),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = pop_robust.build_population_robustness_model(seed=args.seed, cell=args.cell)
    swa_model.load_state_dict(swa_state, strict=True)
    swa_model.to(device).eval()
    with torch.no_grad():
        pred, _ = swa_model(fx_neural[:4], calib_trials=fx_calib[:4], side_features=fx_side[:4])
        swa_finite = bool(torch.isfinite(pred).all().item())
    del swa_model

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure_launch["closure_sha256"]
    status = (
        ("CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE" if args.smoke else "CELL_TERMINAL")
        if not invariant_failures and closure_equal and len(diagnostics) == epochs
        else ("CELL_SMOKE_INVARIANT_FAILURE" if args.smoke else "CELL_INVARIANT_OR_CLOSURE_FAILURE")
    )
    receipt_mod.write_receipt_transactionally(
        out_dir / "terminal_receipt.json",
        {
            "schema": "tfpd_pop_robust_cell_v1",
            "status": status,
            "cell": args.cell,
            "cell_name": CELL_NAMES[args.cell],
            "head_and_dropout_config": launch["head_and_dropout_config"],
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "initial_state": launch["initial_state"],
            "budget": launch["budget"],
            "data_contract": launch["data_contract"],
            "epochs_run": len(diagnostics),
            "diagnostics_per_epoch": diagnostics,
            "invariant_failures": invariant_failures,
            "checkpoints": checkpoints,
            "swa": {
                "path": str(swa_path), "sha256": swa_sha,
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
        "cell": args.cell, "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "heads": config["num_heads"],
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("CELL_TERMINAL", "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
