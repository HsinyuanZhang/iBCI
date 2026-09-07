#!/usr/bin/env python3
"""Route-owned Cell W runner: temporal latent residual decoder (handoff 2026-08-18 §3).

Exact Arm A training replica (T4 visible side, B3S encoder, coupled decoder,
2 heads, batch-32 seed-42 session sampler, strict-27 full window set, 33,925
steps/epoch, warmup 1e-5->1e-4 over two epochs then cosine to 1e-6, 48 epochs,
no clipping, no pretraining/teacher, final-four SWA, seed 42) with exactly ONE
change: the output head gains a zero-initialized temporal latent residual -

    unit_tokens = fc_in(activity + identity)                     # unchanged
    latent_k    = cross_attention(K learned queries, unit_tokens)  # K = 8 FROZEN
    delta[t, c] = sum_k temporal_basis[t, k] * value_head(latent_k)[k, c]
    prediction  = base_output + delta                             # delta == 0 at init

The new cross-attention is ONE nn.MultiheadAttention(512, 2, batch_first=True)
(dropout 0.0, PyTorch default init); the existing decoder transformer is
neither reused nor modified.  value_head = nn.Linear(512, 2) is zero-
initialized in weight AND bias, so delta is EXACTLY zero at initialization and
the model is bitwise equal to Arm A at initialization (asserted at launch and
in tests).  The loss is EXACTLY Arm A's masked MSE - no masking, no
consistency, no dropout beyond the built-in decoder train-mode dropout.

Canonical-initial-state discipline (W ADDS 1,056,146 parameters, so the
T/C/G strict=True template does not apply): the canonical artifact is loaded
with strict=False, the missing keys are asserted to be EXACTLY the eight
W-head keys (sorted equality), the unexpected keys are asserted empty, and the
shared-key state is proven byte-exact by recomputing state_sha256 semantics
over the canonical-named subset against the artifact's recorded state_sha256.

Integrity block (launch + terminal): num_heads, head structure, K, inits,
added-parameter manifest, zero-init guarantee, non-causal caveat, loss,
gradient-flow note, behavior-scaling convention.  Per-epoch receipt fields:
delta magnitude statistics on a fixed probe batch (the residual waking up from
zero), W-head parameter norms, value_head weight/bias max-abs, gradient
finiteness, loss, state/optimizer SHAs, t4 authority fingerprint.
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
sys.path.insert(0, str(ROOT))

PAD_VALUE = -1.0
BOUND_PATTERNS = (
    "src/tfpd_lane/temporal_residual_cell.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_temporal_residual_cell.py",
    "scripts/run_admission_arm.py",
)
CELL_NAME = "cellW_temporal_residual"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def train_epoch_temporal_residual(model, optimizer, loader, lr_fn, device,
                                  phase_step, num_heads, max_steps=None):
    """Route-owned training step: the EXACT Arm A loss on the W-headed model.

    No masking, no p stream, no consistency term, no extra dropout: the only
    stochasticity is the decoder's built-in train-mode dropout
    (tf_drop_rate = 0.1, unchanged); the W head consumes no RNG.
    """
    model.train()
    arm_common = sys.modules["tfpd_lane_arm_common"]
    w_param = model.id_encoder.post_pool[0].weight
    encoder = model.id_encoder
    hidden, side_dim = encoder.hidden_dim, encoder.side_dim
    acc = {
        "loss_sum": torch.zeros((), device=device),
        "nonfinite_loss": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_grad": torch.zeros((), device=device, dtype=torch.long),
        "wside_grad_nonzero": torch.zeros((), device=device, dtype=torch.long),
    }
    steps = examples = 0
    for batch in loader:
        neural, behavior, calib, _sessions, side = batch[:5]
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)
        lr = lr_fn(phase_step + steps)
        optimizer.param_groups[0]["lr"] = float(lr)
        prediction, _identity = model(neural, calib_trials=calib, side_features=side)
        valid_rows = (behavior != PAD_VALUE).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid_rows).sum() / (valid_rows.sum() * behavior.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grads = [q.grad for q in model.parameters() if q.grad is not None]
            total_sq = (
                torch.stack(torch._foreach_norm(grads)).pow(2).sum()
                if grads else torch.zeros((), device=device)
            )
            acc["nonfinite_grad"] += (~torch.isfinite(total_sq)).long()
            grad = w_param.grad
            if grad is not None:
                acc["wside_grad_nonzero"] += (grad[:, hidden : hidden + side_dim] != 0).sum()
            acc["nonfinite_loss"] += (~torch.isfinite(loss)).long()
        optimizer.step()
        acc["loss_sum"] += loss.detach()
        steps += 1
        examples += int(neural.shape[0])
        if max_steps is not None and steps >= max_steps:
            break
    return {
        "optimizer_steps": steps,
        "train_example_windows": examples,
        "train_loss_mean_per_step": float(acc["loss_sum"].item()) / max(steps, 1),
        "nonfinite_loss_steps": int(acc["nonfinite_loss"].item()),
        "nonfinite_grad_steps": int(acc["nonfinite_grad"].item()),
        "num_heads": num_heads,
    }


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
                "schema": "tfpd_temporal_residual_cell_v1",
                "cell": "W",
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
        matched_scorer,
        receipt as receipt_mod,
        temporal_residual_cell as trc,
    )

    arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- canonical initial state: strict=False + exact W-load discipline --
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
    if len(per_session_units) != 27:
        raise SystemExit("per-session unit roster drift (expected 27 sessions)")

    pl.seed_everything(args.seed, workers=True)
    model = trc.build_temporal_residual_model(seed=args.seed)
    # W adds parameters: strict=False + exact missing-key set + shared-subset
    # byte equality against the artifact's recorded state SHA
    load_report = trc.load_canonical_initial_state(
        model, canonical_state, initial_payload["state_sha256"]
    )
    # graph parity proof vs the canonical initial state (shared keys/shapes)
    shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter) else tuple(v.shape))
        for k, v in model.state_dict().items()
    }
    canonical_shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter) else tuple(v.shape))
        for k, v in canonical_state.items()
    }
    if sorted(set(shapes) - set(canonical_shapes)) != sorted(trc.W_HEAD_STATE_KEYS):
        raise SystemExit("graph drift: added keys are not exactly the W-head keys")
    for key, shape in shapes.items():
        if key in canonical_shapes and shape != canonical_shapes[key]:
            raise SystemExit(f"graph parity failure at shared key {key}")
    head_manifest = trc.head_parameter_manifest(model)
    if not head_manifest["counts_match"]:
        raise SystemExit("W-head parameter count drift")
    model.to(device)

    sampler = SessionBatchSampler(train_dataset, batch_size=32, shuffle=True, seed=args.seed)
    steps_per_epoch = len(sampler)
    epochs = args.smoke_epochs if args.smoke else args.epochs
    if not args.smoke and (steps_per_epoch != 33925 or epochs != 48):
        raise SystemExit("budget drift: expected 33,925 steps/epoch over 48 epochs")
    loader = DataLoader(train_dataset, batch_sampler=sampler,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, _fx_sess, fx_side = fixed_batch[:5]
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

    # zero-init guarantee at launch: with the head at its canonical-loaded
    # bytes the decode is bitwise equal to the plain parent decode path
    proof = trc.prove_zero_init_bitwise(model, fx_neural[:8], fx_calib[:8], fx_side[:8])
    if not (proof["torch_equal"] and proof["tensor_sha256_equal"]
            and proof["delta_exactly_zero"]):
        raise SystemExit("zero-init proof failure: W decode != parent path at init")
    eval_parity_bitwise = True

    integrity_block = {
        "cell": "W",
        "num_heads": trc.NUM_HEADS,
        "head_structure": {
            "unit_tokens": "fc_in(activity + identity) - the existing path, unchanged",
            "latent_k": (
                f"cross_attention(K={trc.K_LATENT} learned queries, unit_tokens); "
                "ONE nn.MultiheadAttention(512, 2, batch_first=True), "
                "dropout=0.0, PyTorch default init; the existing decoder "
                "transformer is NOT reused and NOT modified"
            ),
            "delta": (
                "delta[t, c] = sum_k temporal_basis[t, k] * "
                "value_head(latent_k)[k, c], computed as [B, W, C] and added "
                "AFTER the existing permute to the base output [B, W, C]"
            ),
            "prediction": "base_output + delta",
            "k_latent_frozen": trc.K_LATENT,
            "k_sweep": "none (K = 8 frozen)",
            "value_head": "nn.Linear(512, 2), weight AND bias zero-initialized",
        },
        "initialization": {
            "value_head.weight": "zeros -> delta EXACTLY zero at init",
            "value_head.bias": "zeros -> delta EXACTLY zero at init",
            "queries": f"normal(std=512**-0.5) [K={trc.K_LATENT}, 512] (the CalibrationFixedSlotRouter.slot_queries scaled-normal convention); dead at init",
            "temporal_basis": f"standard normal [window_size=50, K={trc.K_LATENT}]; dead at init",
            "cross_attention": "PyTorch default (xavier-uniform projections, uniform biases), dropout=0.0",
            "head_draw_order": "after the parent decoder/encoder draws, so the shared canonical bytes are unchanged; head bytes are seed-42-determined, not canonical",
        },
        "zero_init_guarantee": (
            "delta is exactly zero at initialization, so the W forward is "
            "bitwise equal to the Arm A parent path (torch.equal AND equal "
            "arm_common.tensor_sha256; asserted at launch and in tests)"
        ),
        "non_causal_caveat": trc.NON_CAUSAL_CAVEAT,
        "gradient_flow": trc.GRADIENT_FLOW_NOTE,
        "rng_note": (
            "the new head consumes no RNG (cross_attention dropout = 0.0), so "
            "the decoder's built-in train-mode dropout stream (tf_drop_rate = "
            "0.1) is identical to Arm A's"
        ),
        "loss": (
            "EXACTLY Arm A's masked MSE: ((diff2 * valid_rows).sum() / "
            "(valid_rows.sum() * C)) with valid_rows = (behavior != -1).all(-1); "
            "no masking, no consistency term, no dropout beyond the built-in "
            "decoder train-mode dropout"
        ),
        "dropout_structure": {
            "tf_drop_rate": 0.1,
            "sites": (
                "CrossAttentionLayer MHA dropout, FFN dropout and residual "
                "dropout (the exact parent graph, unchanged)"
            ),
            "decoder_dropout_rate": 0.0,
            "dynamic_dropout": False,
            "temporal_head_cross_attention_dropout": 0.0,
        },
        "w_head_parameters": head_manifest,
        "behavior_scaling_convention": "unscaled standardized behavior (exact Arm A replica)",
    }

    launch = {
        "schema": "tfpd_temporal_residual_cell_v1_launch",
        "status": "CELL_LAUNCHED",
        "cell": "W",
        "cell_name": CELL_NAME,
        "integrity": integrity_block,
        "smoke": args.smoke,
        "started_utc": started,
        "initial_state": {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            **load_report,
            "graph_parity_vs_canonical": {
                "shared_keys_shapes_equal": True,
                "added_keys": sorted(trc.W_HEAD_STATE_KEYS),
            },
            "eval_path_bitwise_parent_equal": eval_parity_bitwise,
            "zero_init_delta_exactly_zero": True,
        },
        "launch_proofs": {"zero_init_bitwise": proof},
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
            "per_session_unit_counts": per_session_units,
        },
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "pretraining_used": False,
            "target_based_selection": False,
            "clipping": "none",
            "masking": "none (the ONE change is the output head)",
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
        stats = train_epoch_temporal_residual(
            model, optimizer, loader, lr_fn, device,
            phase_step, num_heads=2, max_steps=args.max_train_steps,
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
        probe = trc.head_probe_diagnostics(model, fx_neural[:8], fx_calib[:8], fx_side[:8])
        authority_ok = arm_common.t4_authority_fingerprint(train_dataset.sessions) == t4_authority
        value_head_woke = probe["delta_abs_max"] > 0.0
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
            "temporal_head_probe": probe,
            "value_head_woke": value_head_woke,
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
            value_head_woke,
        ]):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfpd_temporal_residual_ckpt_v1",
                    "cell": "W",
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
            "cell": "W", "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "lr_last": lr_fn(phase_step - 1),
            "delta_abs_mean": round(probe["delta_abs_mean"], 8),
            "delta_abs_max": round(probe["delta_abs_max"], 8),
            "value_head_max_abs": probe["value_head_weight_max_abs"],
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = trc.build_temporal_residual_model(seed=args.seed)
    swa_model.load_state_dict(swa_state, strict=True)  # W keys present
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
            "schema": "tfpd_temporal_residual_cell_v1",
            "status": status,
            "cell": "W",
            "cell_name": CELL_NAME,
            "integrity": integrity_block,
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "initial_state": launch["initial_state"],
            "launch_proofs": launch["launch_proofs"],
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
        "cell": "W", "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "final_delta_abs_max": round(
            diagnostics[-1]["temporal_head_probe"]["delta_abs_max"], 8
        ),
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("CELL_TERMINAL", "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
