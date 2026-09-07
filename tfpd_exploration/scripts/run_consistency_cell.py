#!/usr/bin/env python3
"""Route-owned Cell C runner: paired-subset consistency (handoff 2026-08-18 §3).

Exact Arm A / D training replica (canonical initial tensor bytes strict=True,
2 heads, identical state keys/shapes/parameter count, strict-27 full window
set, 33,925 steps/epoch, batch-32 seed-42 session sampler, warmup 1e-5->1e-4
over two epochs then cosine to 1e-6, 48 epochs, no clipping, final-four SWA,
seed 42) with ONE change per step: TWO whole-unit subset masks, each identical
in law to D's mask (per-(batch, unit) Bernoulli(1-p_i), one p_i per branch
forward from the shared PCG64(42) stream -- two draws per step in fixed
order), applied at the exact D site (after `src = activity + identity`,
before `fc_in`) with gain 1/(1-p_i); p_i == 1 yields the all-zero branch
without division.  Both forwards run in train() mode (each internal decoder
dropout pass is independent -- the R-Drop structure), both branches are
supervised with Arm A's exact masked MSE against the behaviour target, and a
frozen lambda = 0.1 consistency term (same functional form on the behaviour
predictions) ties them together; ONE optimizer step per batch (single
backward on the summed loss).  Skip floor 1 per (sample, branch).  Compute
option "two-view same-batch" (below).  Evaluation/scoring forwards run with
the mask as exact ones -- bitwise equal to the unsparsified parent path
(asserted at launch and in tests).

Integrity block (launch + terminal): num_heads, dropout structure, frozen
lambda, compute option + pre-registration, p law + draws-per-step (the SHA
differs from R/S2/T/G by construction), generator namespaces, skip floor and
rule, R-Drop structure note, eval policy, behaviour scaling convention.
Per-epoch receipt fields: the two loss components logged SEPARATELY
(behavior_loss_sum, consistency_loss_sum, branch losses, total), per-branch
kept-fraction distributions, all-zero branch counts, per-step Jaccard overlap
of the two keep sets (mean/quantiles, both-empty skip count), p stream stats,
gradient finiteness, state/optimizer SHAs.  The launch receipt records the
per-session unit counts (and window counts) of the strict-27 roster once.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))

CELL_NAME = "cellC_paired_consistency"
BOUND_PATTERNS = (
    "src/tfpd_lane/consistency_cell.py",
    "src/tfpd_lane/sparsification.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_consistency_cell.py",
    "scripts/run_admission_arm.py",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/subpop_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument("--initial-state", type=Path,
                        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3
    out_dir = Path(args.output_root) / CELL_NAME
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
                "schema": "tfpd_consistency_cell_v1",
                "cell": "C",
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

    from src.tfpd_lane import (
        arm_common,
        consistency_cell,
        matched_scorer,
        receipt as receipt_mod,
        sparsification,
    )

    arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- canonical initial state: strict load + SHA + graph parity ----------
    initial_sidecar = Path(str(args.initial_state) + ".sha256")
    if not args.initial_state.is_file() or not initial_sidecar.is_file():
        raise SystemExit("canonical initial state artifact missing")
    initial_sha = receipt_mod.sha256_file(args.initial_state)
    if initial_sha != initial_sidecar.read_text().split()[0]:
        raise SystemExit("canonical initial state SHA mismatch")
    initial_payload = torch.load(args.initial_state, map_location="cpu", weights_only=False)
    canonical_state = initial_payload["state_dict"]

    # ---- data (arm-A contract; development rosters never opened) ------------
    ns = type("NS", (), {"train_batch_size": 32, "num_workers": args.num_workers,
                         "seed": args.seed})()
    dm, a2 = arm_runner.build_datamodule(ns)
    train_dataset = dm.train_dataset
    if len(dm.session_splits["train"]) != 27:
        raise SystemExit("strict-27 roster drift")
    if receipt_mod.sha256_file(a2.MANIFEST_PATH) != a2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest SHA drift")
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    if not behavior_semantic.startswith("f062506c"):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    # per-session unit counts of the training roster, recorded once (handoff
    # item 6): unit counts drive the skip-floor trigger rate E[p^N] = 1/(N+1)
    windows_per_session: dict[str, int] = {}
    for session_name, _start in train_dataset.window_indices:
        windows_per_session[session_name] = windows_per_session.get(session_name, 0) + 1
    per_session_units = {
        name: {
            "n_units": int(record.side_features.shape[0]),
            "n_windows": windows_per_session.get(name, 0),
        }
        for name, record in sorted(train_dataset.sessions.items())
    }
    roster_units = [entry["n_units"] for entry in per_session_units.values()]
    if len(per_session_units) != 27:
        raise SystemExit("per-session unit roster drift (expected 27 sessions)")

    pl.seed_everything(args.seed, workers=True)
    model = consistency_cell.build_consistency_model(seed=args.seed)
    model.load_state_dict(canonical_state, strict=True)
    if arm_common.state_sha256(model) != initial_payload["state_sha256"]:
        raise SystemExit("loaded state SHA != canonical artifact state SHA")
    # graph parity proof vs the canonical initial state (keys/shapes)
    shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter) else tuple(v.shape))
        for k, v in model.state_dict().items()
    }
    canonical_shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter) else tuple(v.shape))
        for k, v in canonical_state.items()
    }
    if sorted(shapes) != sorted(canonical_shapes) or shapes != canonical_shapes:
        raise SystemExit("graph parity failure vs canonical initial state")
    model.to(device)

    sampler = SessionBatchSampler(train_dataset, batch_size=32, shuffle=True, seed=args.seed)
    steps_per_epoch = len(sampler)
    epochs = args.smoke_epochs if args.smoke else args.epochs
    if not args.smoke and (steps_per_epoch != 33925 or epochs != 48):
        raise SystemExit("budget drift: expected 33,925 steps/epoch over 48 epochs")
    loader = DataLoader(train_dataset, batch_sampler=sampler,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, fx_sess, fx_side = fixed_batch[:5]
    fx_neural, fx_calib, fx_side = (
        fx_neural.to(device), fx_calib.to(device), fx_side.to(device),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    schedule = arm_common.schedule_params(epochs, steps_per_epoch)
    lr_fn = lambda step: arm_common.lr_at_step(step, epochs, steps_per_epoch)

    # eval-path parity assertion: with sparsification disabled the decode is
    # bitwise equal to the unsparsified parent path
    model.eval()
    with torch.no_grad():
        consistency_eval, _ = model(fx_neural[:4], calib_trials=fx_calib[:4],
                                    side_features=fx_side[:4])
        # the unsparsified parent path through the same modules, same order
        src = fx_neural[:4].permute(0, 2, 1)
        identity = model.compute_identity(fx_calib[:4], side_features=fx_side[:4])
        src = src + identity
        src = model.decoder.fc_in(src)
        rep = model.decoder.fc_in(model.decoder.rep).to(src)
        out, _ = model.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        parent_eval = model.decoder.fc_out(out).permute(0, 2, 1)
    if not torch.equal(consistency_eval, parent_eval):
        raise SystemExit("eval-path parity failure: paired-consistency decode != parent path")
    eval_parity_bitwise = True

    p_stream = sparsification.PStream(seed=consistency_cell.P_STREAM_SEED)
    masker = consistency_cell.PairedSubsetMasker()
    jaccard = consistency_cell.JaccardRecorder()

    integrity_block = {
        "cell": "C",
        "num_heads": consistency_cell.NUM_HEADS,
        "dropout_structure": (
            "whole-unit Bernoulli per branch, D law: per-(batch, unit) keep ~ "
            "Bernoulli(1-p_i), one p_i per branch forward, applied at the exact "
            "D site (after `src = activity + identity`, before `fc_in`) with "
            "gain 1/(1-p_i); p_i == 1 -> all-zero branch without division"
        ),
        "lambda": consistency_cell.LAMBDA,
        "lambda_frozen": True,
        "lambda_sweep": "none",
        "consistency_site": "behaviour predictions (never a latent)",
        "loss": (
            "behavior_loss(pred_1, target) + behavior_loss(pred_2, target) + "
            "0.1 * consistency(pred_1, pred_2); behavior_loss is EXACTLY Arm A's "
            "masked MSE ((diff2 * valid_rows).sum() / (valid_rows.sum() * C)) per "
            "branch; consistency = (((pred_1 - pred_2)**2) * valid_rows).sum() / "
            "(valid_rows.sum() * C) -- same scale, on the behaviour predictions"
        ),
        "compute_option": consistency_cell.COMPUTE_OPTION,
        "preregistration": consistency_cell.PREREGISTRATION,
        "p_stream": {
            "law": ("p ~ Uniform(0,1) from numpy PCG64(42); TWO draws per training "
                    "step in fixed order (branch 1 then branch 2)"),
            "draws_per_step": consistency_cell.DRAWS_PER_STEP,
            "seed": consistency_cell.P_STREAM_SEED,
            "distribution_matched_to_D": True,
            "bitwise_matched_to_D": False,
            "sha_note": ("Cell C consumes 2 draws per step, so p_sequence_sha256 "
                         "differs from R/S2/T/G by construction; total draws are "
                         "recorded as 2 x optimizer steps"),
        },
        "generator_namespaces": {
            "p_stream": f"numpy PCG64({consistency_cell.P_STREAM_SEED})",
            "branch1_keep_mask": (
                f"torch.Generator CPU seeded {consistency_cell.BRANCH1_KEEP_SEED} "
                "(per-(batch, unit) Bernoulli keep, branch 1)"
            ),
            "branch2_keep_mask": (
                f"torch.Generator CPU seeded {consistency_cell.BRANCH2_KEEP_SEED} "
                "(per-(batch, unit) Bernoulli keep, branch 2)"
            ),
            "isolation": ("route-owned generator objects only; data ordering and "
                          "unrelated RNG (including the global torch RNG that feeds "
                          "the decoder's internal dropout) are untouched"),
        },
        "rdrop_structure": (
            "both forwards run in train() mode; each forward's internal decoder "
            "dropout (cross-attention/FFN/residual, p=0.1, global torch RNG) is an "
            "independent stochastic pass -- the R-Drop structure (Liang et al., "
            "NeurIPS 2021, arXiv:2106.14448; MSE replaces KL in the regression case)"
        ),
        "skip_floor": consistency_cell.SKIP_FLOOR,
        "skip_rule": (
            "a branch whose keep count is 0 for a sample is skipped for that sample "
            "(its behavior loss and the consistency term drop out for that sample; "
            "both branches empty -> zero loss for that sample); triggers counted "
            "per epoch; marginal trigger probability under D's law is 1/(N+1)"
        ),
        "eval_mask": ("exact ones; single forward; bitwise equal to the "
                      "unsparsified parent path (asserted at launch and in tests)"),
        "behavior_scaling_convention": "unscaled standardized behavior (exact Arm A replica)",
        "prior_art": (
            "the objective IS R-Drop and two-view agreement is co-training; the "
            "route-owned claim is that a whole-unit perturbation (not R-Drop's "
            "elementwise one) is right for population decoding -- evidenced by "
            "R = 0.1877 vs D = 0.4179 at matched removal amount (0/15 sessions)"
        ),
    }

    launch = {
        "schema": "tfpd_consistency_cell_v1_launch",
        "status": "CELL_LAUNCHED",
        "cell": "C",
        "cell_name": CELL_NAME,
        "integrity": integrity_block,
        "smoke": args.smoke,
        "started_utc": started,
        "initial_state": {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "loaded_state_sha256": arm_common.state_sha256(model),
            "strict_load": True,
            "graph_parity_vs_canonical": True,
            "eval_path_bitwise_parent_equal": eval_parity_bitwise,
        },
        "budget": {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "forwards_per_step": 2,
            "total_forwards": 2 * epochs * steps_per_epoch,
            "schedule": schedule, "optimizer": arm_common.ADAM_CONSTRUCTOR,
            "sampler": "mc_maze SessionBatchSampler(batch=32, shuffle, seed=42)",
            "clipping": "none",
        },
        "roster_unit_counts": {
            "per_session": per_session_units,
            "n_sessions": len(per_session_units),
            "n_units_min": int(min(roster_units)),
            "n_units_median": float(np.median(roster_units)),
            "n_units_max": int(max(roster_units)),
            "total_train_windows": int(sum(windows_per_session.values())),
            "note": ("unit counts drive the skip-floor trigger rate "
                     "E[p^N] = 1/(N+1) per (sample, branch) under D's p ~ U(0,1) law"),
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
            "target_based_selection": False,
            "checkpoint_selection": "predeclared final-four SWA window only; no selection",
            "clipping": "none",
            "lambda_sweep": "none (frozen at 0.1)",
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
        model.mask_stats = []
        masker.reset_epoch()
        jaccard.reset()
        stats = consistency_cell.train_epoch_paired_consistency(
            model, optimizer, loader, lr_fn, device, p_stream, masker, jaccard,
            phase_step, num_heads=consistency_cell.NUM_HEADS,
            max_steps=args.max_train_steps,
        )
        phase_step += stats["optimizer_steps"]
        p_epoch_stats = p_stream.stats()
        mask_summary = masker.summary()
        jaccard_summary = jaccard.summary()
        branch1_forwards = sum(1 for m in model.mask_stats if m["branch"] == 1)
        branch2_forwards = sum(1 for m in model.mask_stats if m["branch"] == 2)

        w_side = arm_common.w_side_block(model)
        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter) and p.numel()
        )
        with torch.no_grad():
            contribution = arm_common.post_pool_contribution(model, fx_calib[:8], fx_side[:8])
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
            **contribution,
            "p_stream_epoch_stats": p_epoch_stats,
            "p_stream_draws_cumulative": p_epoch_stats["n"],
            "mask_summary": mask_summary,
            "jaccard_overlap": jaccard_summary,
            "branch_forwards": {"branch1": branch1_forwards, "branch2": branch2_forwards},
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
            stats["nonfinite_loss_steps"] == 0,
            stats["nonfinite_grad_steps"] == 0,
            params_finite, diag["optimizer_state_finite"], authority_ok, ok_steps,
            p_epoch_stats["n"] == consistency_cell.DRAWS_PER_STEP * phase_step,
            stats["forwards"] == 2 * stats["optimizer_steps"],
            branch1_forwards == stats["optimizer_steps"],
            branch2_forwards == stats["optimizer_steps"],
            mask_summary["branch1"]["n_draws"] == stats["optimizer_steps"],
            mask_summary["branch2"]["n_draws"] == stats["optimizer_steps"],
            (jaccard_summary["n"] + jaccard_summary["skipped_both_empty"]
             == stats["train_example_windows"]),
        ]):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfpd_consistency_ckpt_v1",
                    "cell": "C",
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
            "cell": "C", "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "behavior_sum": round(stats["behavior_loss_sum_mean_per_step"], 6),
            "consistency": round(stats["consistency_loss_mean_per_step"], 6),
            "jaccard_mean": round(jaccard_summary.get("mean") or 0.0, 4),
            "skip1": stats["branch1_skipped_samples"],
            "skip2": stats["branch2_skipped_samples"],
            "lr_last": lr_fn(phase_step - 1),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = consistency_cell.build_consistency_model(seed=args.seed)
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
            "schema": "tfpd_consistency_cell_v1",
            "status": status,
            "cell": "C",
            "cell_name": CELL_NAME,
            "integrity": integrity_block,
            "p_sequence_sha256": p_stream.sha256(),
            "p_stream_total_draws": consistency_cell.DRAWS_PER_STEP * phase_step,
            "p_stream_draws_note": (
                "two draws per optimizer step (branch 1 then branch 2); the SHA "
                "therefore differs from R/S2/T/G by construction"
            ),
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "initial_state": launch["initial_state"],
            "budget": launch["budget"],
            "roster_unit_counts": launch["roster_unit_counts"],
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
        "cell": "C", "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "final_behavior_sum": round(
            diagnostics[-1]["behavior_loss_sum_mean_per_step"], 6),
        "final_consistency": round(
            diagnostics[-1]["consistency_loss_mean_per_step"], 6),
        "p_sequence_sha256": p_stream.sha256()[:16],
        "p_stream_total_draws": consistency_cell.DRAWS_PER_STEP * phase_step,
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("CELL_TERMINAL", "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
