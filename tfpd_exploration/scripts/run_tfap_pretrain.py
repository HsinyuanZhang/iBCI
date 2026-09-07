#!/usr/bin/env python3
"""TFAP Stage-1: 000128 whole-model pretraining (P-T4 / P-Z4). GPU, contract run.

TFAP_CONTRACT_20260817.md §2 (with the sealed 2026-08-17 revision):

- initial state: the SAME canonical artifact as arms A/B/C
  (canonical_initial_state.pt, state sha 65bacb85…), strict=True, no rebuild;
- data: EXACTLY the sealed Stage-0 payload (jenkins_derived_payload.npz,
  sidecar-verified) — binned 000128 neural, own-normalized behavior, M30
  calib, closed-form T4 standardized by 000128's own stats, 171,935 train
  windows (5,372 steps/epoch, drop-partial, frozen seed-42 permutation);
- schedule: arm A's recipe applied phase-locally — linear warmup 1e-5 -> 1e-4
  over the first two epochs' steps, cosine to 1e-6 at the final step of epoch
  47 (arm_common.lr_at_step, n_epochs=48, steps_per_epoch=5372); fresh Adam
  betas=(0.9,0.999) eps=1e-8 weight_decay=0 amsgrad=False; no clipping;
- P-T4 visible side: the standardized T4 tensor itself. P-Z4: `zeros_like` of
  it (exact positive zero, admission after normalization — never 0*raw). With
  the z4-visible control, W_side / exp_avg / exp_avg_sq must stay elementwise
  exactly zero every epoch (fail closed), exactly as in Gate-2 arm B phase 1;
- per-epoch §9 diagnostics (loss, steps, LR, global/per-branch grad norms,
  W_side norms and moments, fixed-batch T4/calibration contribution ratio,
  attention entropy/concentration, state/optimizer SHAs, finiteness);
- predeclared final-four SWA (epochs 44-47) + finite forward smoke; 0444
  launch/terminal receipts with closure equality.
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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

PAD_VALUE = -1.0
BOUND_PATTERNS = (
    "src/tfpd_lane/tfap_stage0.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_tfap_pretrain.py",
    "scripts/run_admission_arm.py",
    "src/tfpd/spintshape_module.py",
)
ARM_NAMES = {"t4": "P-T4", "z4": "P-Z4"}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
matched_scorer = _load_module("tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py")
receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
tfap = _load_module("tfpd_lane_tfap_stage0", ROOT / "src/tfpd_lane/tfap_stage0.py")
arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")


class JenkinsPretrainDataset(Dataset):
    """688-contract batch shape from the sealed Stage-0 payload arrays."""

    def __init__(self, neural, behavior, calib, side, starts, window_size=50):
        self.neural = neural
        self.behavior = behavior
        self.calib = calib
        self.side = side
        self.starts = starts
        self.window_size = window_size
        self.name = "sub-Jenkins_ses-full"

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        end = start + self.window_size
        return (
            torch.from_numpy(self.neural[start:end]).float(),
            torch.from_numpy(self.behavior[start:end]).float(),
            torch.from_numpy(self.calib.copy()).float(),
            self.name,
            torch.from_numpy(self.side).float(),
        )


class FixedSingleSessionBatchSampler:
    """Drop-partial fixed permutation (the three-arm sampler semantics)."""

    def __init__(self, n_windows, batch_size=32, seed=42):
        import random as _random

        order = list(range(n_windows))
        order = _random.Random(seed).sample(order, len(order))
        self.batches = [
            order[i : i + batch_size]
            for i in range(0, len(order), batch_size)
            if len(order[i : i + batch_size]) == batch_size
        ]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible", required=True, choices=["t4", "z4"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/tfap_stage1_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument(
        "--payload", type=Path, default=ROOT / "results/tfap_stage0_v1/jenkins_derived_payload.npz"
    )
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

    out_dir = Path(args.output_root) / f"pretrain_{'pt4' if args.visible == 't4' else 'pz4'}"
    if args.smoke:
        out_dir = Path(str(out_dir) + "_smoke")
    if out_dir.exists():
        print(f"fresh pretrain output directory required: {out_dir}", file=sys.stderr)
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
                "schema": "tfap_stage1_v1",
                "arm": ARM_NAMES[args.visible],
                "status": "PRETRAIN_FAILED",
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
    from torch.nn.parameter import UninitializedParameter
    from torch.utils.data import default_collate

    spintshape = _load_module("tfpd_spintshape_module", ROOT / "src/tfpd/spintshape_module.py")
    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- sealed Stage-0 payload (sidecar-verified) ---------------------------
    payload_path = Path(args.payload)
    payload_sidecar = Path(str(payload_path) + ".sha256")
    if not payload_path.is_file() or not payload_sidecar.is_file():
        raise SystemExit(f"Stage-0 payload or sidecar missing: {payload_path}")
    payload_sha = receipt_mod.sha256_file(payload_path)
    if payload_sha != payload_sidecar.read_text().split()[0]:
        raise SystemExit("Stage-0 payload SHA mismatch against sidecar")
    stage0_receipt = json.loads(
        (payload_path.parent / "stage0_receipt.json").read_text()
    )
    if stage0_receipt["status"] != "STAGE0_PREFLIGHT_PASSED":
        raise SystemExit("Stage-0 receipt not PASSED")
    if stage0_receipt["derived_payload"]["sha256"] != payload_sha:
        raise SystemExit("Stage-0 receipt/payload SHA mismatch")
    with np.load(payload_path, allow_pickle=False) as data:
        neural = data["neural"]
        behavior = data["behavior"]
        calib = data["calib_trials"]
        t4_std = data["t4_standardized"]
        starts = data["valid_train_starts"]
    n_units = neural.shape[1]
    t4_authority_sha = tfap.array_sha256(t4_std)
    calib_sha = tfap.array_sha256(calib)

    epochs = args.smoke_epochs if args.smoke else args.epochs
    n_windows = len(starts)
    sampler = FixedSingleSessionBatchSampler(n_windows, args.train_batch_size, args.seed)
    steps_per_epoch = len(sampler)
    if not args.smoke:
        expected_spe = stage0_receipt["budget_frozen"]["steps_per_epoch"]
        if steps_per_epoch != expected_spe or epochs != 48:
            raise SystemExit(
                f"budget drift: steps/epoch {steps_per_epoch} != {expected_spe} or epochs {epochs} != 48"
            )
    schedule = arm_common.schedule_params(epochs, steps_per_epoch)
    lr_fn = lambda step: arm_common.lr_at_step(step, epochs, steps_per_epoch)

    # ---- canonical initial state, strict=True -------------------------------
    initial_sidecar = Path(str(args.initial_state) + ".sha256")
    if not args.initial_state.is_file() or not initial_sidecar.is_file():
        raise SystemExit("canonical initial state artifact missing")
    initial_sha = receipt_mod.sha256_file(args.initial_state)
    if initial_sha != initial_sidecar.read_text().split()[0]:
        raise SystemExit("canonical initial state SHA mismatch")
    initial_payload = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    pl.seed_everything(args.seed, workers=True)
    model = spintshape.build_spintshape_model(seed=args.seed)
    model.load_state_dict(initial_payload["state_dict"], strict=True)
    if arm_common.state_sha256(model) != initial_payload["state_sha256"]:
        raise SystemExit("loaded initial state SHA != artifact state SHA")
    w_side = arm_common.w_side_block(model)
    if int(torch.count_nonzero(w_side).item()) != 0 or bool(w_side.signbit().any().item()):
        raise SystemExit("W_side not exactly positive zero after initial-state load")
    model.to(device)

    dataset = JenkinsPretrainDataset(neural, behavior, calib, t4_std, starts)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    fixed_batch = default_collate([dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, _fx_sess, fx_side = fixed_batch[:5]
    fx_neural = fx_neural.to(device)
    fx_calib = fx_calib.to(device)
    fx_side = fx_side.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )

    launch = {
        "schema": "tfap_stage1_v1_launch",
        "status": "PRETRAIN_LAUNCHED",
        "arm": ARM_NAMES[args.visible],
        "visible_side": args.visible,
        "visible_side_semantics": (
            "standardized 000128 T4 (identity)" if args.visible == "t4"
            else "zeros_like(standardized 000128 T4): exact positive zero after normalization"
        ),
        "smoke": args.smoke,
        "started_utc": started,
        "contract": "docs/TFAP_CONTRACT_20260817.md (+ sealed 2026-08-17 LR revision)",
        "initial_state": {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "strict_load": True,
        },
        "stage0_payload": {
            "path": str(payload_path), "sha256": payload_sha,
            "n_units": int(n_units), "n_train_windows": int(n_windows),
            "calib_sha256": calib_sha, "t4_standardized_sha256": t4_authority_sha,
        },
        "budget": {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "schedule": schedule, "sampler": "fixed seed-42 permutation, drop-partial",
            "optimizer": arm_common.ADAM_CONSTRUCTOR,
        },
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "dandi_000688_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "desc_test_nwb_opened": False,
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
    diagnostics = []
    checkpoints = []
    invariant_failures = []
    phase_step = 0
    for epoch in range(epochs):
        t0 = time.time()
        stats = arm_runner.train_epoch(
            model, optimizer, loader, args.visible, lr_fn, device,
            phase_step, max_steps=args.max_train_steps,
        )
        phase_step += stats["optimizer_steps"]

        w_side = arm_common.w_side_block(model)
        w_zero = int(torch.count_nonzero(w_side).item()) == 0
        w_positive = not bool(w_side.signbit().any().item())
        exp_avg = arm_common.w_side_moment(model, optimizer, "exp_avg")
        exp_avg_sq = arm_common.w_side_moment(model, optimizer, "exp_avg_sq")
        moments_zero = (exp_avg is None or int(torch.count_nonzero(exp_avg).item()) == 0) and (
            exp_avg_sq is None or int(torch.count_nonzero(exp_avg_sq).item()) == 0
        )
        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter) and p.numel()
        )
        with torch.no_grad():
            contribution = arm_common.post_pool_contribution(model, fx_calib[:8], fx_side[:8])
            attention = arm_common.attention_summary(
                model, fx_neural[:4], fx_calib[:4],
                arm_common.admit_side(fx_side[:4], args.visible),
            )
        authority_ok = (
            tfap.array_sha256(t4_std) == t4_authority_sha
            and tfap.array_sha256(calib) == calib_sha
        )
        diag = {
            "epoch": epoch,
            "visible_side_mode": args.visible,
            "duration_s": round(time.time() - t0, 3),
            **stats,
            "optimizer_steps_total": phase_step,
            "lr_expected_first": schedule["lr_at_step_0"],
            "lr_expected_last": schedule["lr_at_final_step"],
            "clipping_authorized": False,
            "w_side_norm": float(w_side.norm().item()),
            "norm_alpha_times_w_side": float(w_side.norm().item()) if args.visible == "t4" else 0.0,
            "w_side_exact_zero_magnitude": w_zero,
            "w_side_positive_zero_bitwise": w_positive,
            "w_side_exp_avg_norm": 0.0 if exp_avg is None else float(exp_avg.norm().item()),
            "w_side_exp_avg_sq_norm": 0.0 if exp_avg_sq is None else float(exp_avg_sq.norm().item()),
            "w_side_moments_exact_zero_magnitude": moments_zero,
            **contribution,
            **attention,
            "t4_authority_unchanged": authority_ok,
            "parameters_finite": params_finite,
            "optimizer_state_finite": arm_runner.optimizer_state_finite(optimizer),
            "state_dict_sha256": arm_common.state_sha256(model),
            "optimizer_state_sha256": arm_common.optimizer_sha256(optimizer),
            "checkpoint_saved": None,
        }
        checks = [
            stats["visible_side_violation_count"] == 0,
            stats["nonfinite_loss_steps"] == 0,
            stats["nonfinite_grad_steps"] == 0,
            params_finite,
            diag["optimizer_state_finite"],
            authority_ok,
            attention["n_layers_captured"] > 0,
            steps_this_epoch_ok(stats, steps_per_epoch, args.max_train_steps),
        ]
        if args.visible == "z4":
            checks += [w_zero, w_positive, moments_zero, stats["w_side_grad_exact_zero_all_steps"],
                       diag["norm_w_side_times_t4_fixed_batch"] == 0.0]
        if not all(checks):
            invariant_failures.append(epoch)

        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfap_stage1_ckpt_v1",
                    "arm": ARM_NAMES[args.visible],
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
            "arm": ARM_NAMES[args.visible], "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "lr_last": stats["lr_last"], "w_side": round(diag["w_side_norm"], 6),
            "ratio_t4_calib": round(diag["ratio_t4_to_calibration_at_post_pool0"], 6),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA-window checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = spintshape.build_spintshape_model(seed=args.seed)
    swa_model.load_state_dict(swa_state, strict=True)
    swa_model.to(device).eval()
    with torch.no_grad():
        swa_pred, _ = swa_model(
            fx_neural[:4], calib_trials=fx_calib[:4], side_features=arm_common.admit_side(fx_side[:4], "t4"),
        )
        swa_finite = bool(torch.isfinite(swa_pred).all().item())
    del swa_model

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure_launch["closure_sha256"]
    status = (
        ("PRETRAIN_SMOKE_COMPLETE__NON_AUTHORITATIVE" if args.smoke else "PRETRAIN_TERMINAL")
        if not invariant_failures and closure_equal and len(diagnostics) == epochs
        else ("PRETRAIN_SMOKE_INVARIANT_FAILURE" if args.smoke else "PRETRAIN_INVARIANT_OR_CLOSURE_FAILURE")
    )
    receipt_mod.write_receipt_transactionally(
        out_dir / "terminal_receipt.json",
        {
            "schema": "tfap_stage1_v1",
            "status": status,
            "arm": ARM_NAMES[args.visible],
            "visible_side": args.visible,
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "budget": launch["budget"],
            "initial_state": launch["initial_state"],
            "stage0_payload": launch["stage0_payload"],
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
        "arm": ARM_NAMES[args.visible], "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "w_side_final": round(diagnostics[-1]["w_side_norm"], 6),
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("PRETRAIN_TERMINAL", "PRETRAIN_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


def steps_this_epoch_ok(stats, steps_per_epoch, max_steps):
    if max_steps is not None:
        return stats["optimizer_steps"] == min(max_steps, steps_per_epoch)
    return stats["optimizer_steps"] == steps_per_epoch


if __name__ == "__main__":
    raise SystemExit(main())
