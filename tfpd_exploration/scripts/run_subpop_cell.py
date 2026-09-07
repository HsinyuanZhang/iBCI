#!/usr/bin/env python3
"""Route-owned T / G sub-population cell runner (handoff 2026-08-18 §3/§5).

Exact Arm A / D training replica (canonical initial tensor bytes strict=True,
2 heads, identical state keys/shapes/parameter count, strict-27 full window
set, 33,925 steps/epoch, batch-32 seed-42 session sampler, warmup 1e-5->1e-4
over two epochs then cosine to 1e-6, 48 epochs, no clipping, final-four SWA,
seed 42) with exactly ONE change at the exact D site (after
`src = activity + identity`, before `fc_in`), driven by the exact shared p
stream (one p ~ U(0,1) per forward, SHA-recorded, imported from
src/tfpd_lane/sparsification.py — never duplicated — so T and G draw the same
sequence as R / S2):

- cell T: per-(batch, unit) whole-unit Bernoulli keep mask from a route-owned
  torch.Generator namespace (CPU seed 42_003); the boolean complement is passed
  as `key_padding_mask` to `decoder.transformer(...)`, so dropped units are
  genuinely absent from the attention softmax.  `src` is never zeroed and no
  1/(1-p) rescaling is applied.  `min_keep = 4` per row restores randomly
  chosen dropped units whenever a row would fall below the floor (guarding the
  fully-masked-row NaN softmax); triggering rows are recorded per epoch.
- cell G: nothing is dropped; every unit window is multiplied by
  1/(1-clamp(p, 0, 0.95)).  The realized gain distribution and the clamp
  trigger rate (raw p > 0.95) are recorded per epoch.

Evaluation/scoring forwards run with the perturbation disabled: no mask and no
gain, `key_padding_mask` is not passed at all, and the forward is bitwise
equal to the unsparsified parent path (asserted at launch and in tests).

Integrity block (launch + terminal): num_heads, dropout structure, p
distribution and clamp policy, min_keep, rescaling policy, generator
namespaces, eval-mask policy, behavior-scaling convention.  Per-epoch receipt
fields: realized surviving-unit count distribution / gain distribution, the
min_keep or clamp trigger counts, p stream statistics, gradient finiteness and
loss.
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

PAD_VALUE = -1.0
BOUND_PATTERNS = (
    "src/tfpd_lane/sparsification.py",
    "src/tfpd_lane/subpop_cells.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_subpop_cell.py",
    "scripts/run_admission_arm.py",
)
CELL_NAMES = {
    "T": "cellT_true_removal",
    "G": "cellG_global_gain",
}
THETA_AUTHORITY = ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def train_epoch_perturbed(model, optimizer, loader, lr_fn, device, p_stream,
                          session_authority, phase_step, num_heads,
                          max_steps=None):
    """Route-owned training step: one shared p per forward, exact D loss/site."""
    model.train()
    arm_common = sys.modules["tfpd_lane_arm_common"]
    encoder_params, decoder_params = arm_common.param_groups_by_branch(model)
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
        neural, behavior, calib, sessions, side = batch[:5]
        session = sessions[0] if isinstance(sessions, (list, tuple)) else sessions
        theta, valid = session_authority[session]
        model.set_session_authority(theta, valid)
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)
        lr = lr_fn(phase_step + steps)
        optimizer.param_groups[0]["lr"] = float(lr)
        p = p_stream.next()
        model.current_p = p
        model.perturbation_enabled = True
        prediction, _identity = model(neural, calib_trials=calib, side_features=side)
        model.perturbation_enabled = False  # never leak into eval paths
        valid_rows = (behavior != PAD_VALUE).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid_rows).sum() / (valid_rows.sum() * behavior.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grads = [q.grad for q in encoder_params + decoder_params if q.grad is not None]
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
    parser.add_argument("--cell", required=True, choices=["T", "G"])
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
                "schema": "tfpd_subpop_cell_v1",
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
    sparsification = _load_module(
        "tfpd_lane_sparsification", ROOT / "src/tfpd_lane/sparsification.py"
    )
    subpop = _load_module("tfpd_lane_subpop_cells", ROOT / "src/tfpd_lane/subpop_cells.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    config = subpop.CELLS[args.cell]

    # ---- authorities --------------------------------------------------------
    initial_sidecar = Path(str(args.initial_state) + ".sha256")
    if not args.initial_state.is_file() or not initial_sidecar.is_file():
        raise SystemExit("canonical initial state artifact missing")
    initial_sha = receipt_mod.sha256_file(args.initial_state)
    if initial_sha != initial_sidecar.read_text().split()[0]:
        raise SystemExit("canonical initial state SHA mismatch")
    initial_payload = torch.load(args.initial_state, map_location="cpu", weights_only=False)

    theta_sidecar = Path(str(THETA_AUTHORITY) + ".sha256")
    if not THETA_AUTHORITY.is_file() or not theta_sidecar.is_file():
        raise SystemExit("theta authority artifact missing")
    theta_sha = receipt_mod.sha256_file(THETA_AUTHORITY)
    if theta_sha != theta_sidecar.read_text().split()[0]:
        raise SystemExit("theta authority SHA mismatch")
    theta_payload = torch.load(THETA_AUTHORITY, map_location="cpu", weights_only=False)
    if theta_payload["kind"] != "tfpd_sparsification_theta_authority_v1":
        raise SystemExit("theta authority kind drift")
    session_authority = {}
    per_session_valid = {}
    for name, entry in sorted(theta_payload["authority"].items()):
        session_authority[name] = (
            entry["theta"].to(device), entry["valid"].to(device)
        )
        per_session_valid[name] = {
            "n_units": int(entry["n_units"]),
            "n_valid": int(entry["valid"].sum().item()),
            "n_undefined": int((~entry["valid"]).sum().item()),
        }

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
    missing = set(dm.session_splits["train"]) - set(session_authority)
    if missing:
        raise SystemExit(f"theta authority missing sessions: {sorted(missing)}")

    pl.seed_everything(args.seed, workers=True)
    model = subpop.build_subpop_model(seed=args.seed, cell=args.cell)
    model.load_state_dict(initial_payload["state_dict"], strict=True)
    if arm_common.state_sha256(model) != initial_payload["state_sha256"]:
        raise SystemExit("loaded state SHA != canonical artifact state SHA")
    # graph parity proof vs the arm-A builder (keys/shapes/param count)
    arm_a_state = initial_payload["state_dict"]
    from torch.nn.parameter import UninitializedParameter as _Lazy

    shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, _Lazy) else tuple(v.shape))
        for k, v in model.state_dict().items()
    }
    canonical_shapes = {
        k: (("uninitialized-lazy",) if isinstance(v, _Lazy) else tuple(v.shape))
        for k, v in arm_a_state.items()
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

    # eval-path parity assertion: with the perturbation disabled the decode is
    # bitwise equal to the unsparsified parent path (no mask, no gain)
    model.eval()
    with torch.no_grad():
        perturbed_eval, _ = model(fx_neural[:4], calib_trials=fx_calib[:4],
                                  side_features=fx_side[:4])
        # the unsparsified parent path through the same modules, same order
        src = fx_neural[:4].permute(0, 2, 1)
        identity = model.compute_identity(fx_calib[:4], side_features=fx_side[:4])
        src = src + identity
        src = model.decoder.fc_in(src)
        rep = model.decoder.fc_in(model.decoder.rep).to(src)
        out, _ = model.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        parent_eval = model.decoder.fc_out(out).permute(0, 2, 1)
    if not torch.equal(perturbed_eval, parent_eval):
        raise SystemExit("eval-path parity failure: perturbed decode != parent path")
    eval_parity_bitwise = True
    assert model.perturbation_enabled is False  # never leaked into the eval pass

    # cell-specific launch proofs on the same fixed batch
    with torch.no_grad():
        identity_probe = model.compute_identity(fx_calib[:4], side_features=fx_side[:4])
        src_probe = fx_neural[:4].permute(0, 2, 1) + identity_probe
        if args.cell == "T":
            removal_proof = subpop.prove_true_removal(model, src_probe, excluded_unit=0)
            gain_proof = None
            if not (
                removal_proof["excluded_unit_change_bitwise_invisible"]
                and removal_proof["kept_unit_change_visible"]
                and removal_proof["output_finite"]
            ):
                raise SystemExit("true-removal proof failure at launch")
        else:
            removal_proof = None
            gain_proof = subpop.prove_gain_rule(
                model, fx_neural[:4], identity_probe, p=0.5
            )
            if not gain_proof["bitwise_equal_to_parent_times_gain"]:
                raise SystemExit("gain-rule proof failure at launch")
    assert model.perturbation_enabled is False  # the proofs never leak either

    p_stream = subpop.pstream()
    integrity_block = {
        "cell": args.cell,
        "num_heads": 2,
        "dropout_structure": {
            "tf_drop_rate": 0.1,
            "sites": (
                "CrossAttentionLayer MHA dropout, FFN dropout and residual "
                "dropout (the exact parent graph, unchanged)"
            ),
            "decoder_dropout_rate": 0.0,
            "dynamic_dropout": False,
            "note": (
                "the route perturbation replaces D's built-in dynamic dropout "
                "at the same site; the flag itself carries no parameters"
            ),
        },
        "mask_structure": config["mask_structure"],
        "sparsification_site": "after src = activity + identity, before fc_in",
        "p_distribution_and_clamp_policy": {
            "law": "one p ~ Uniform(0,1) per training forward, PCG64(42), exact stream shared with the R / S2 / T / G cells",
            "distribution_matched_to_D": True,
            "bitwise_matched_to_D": False,
            "clamp": (
                f"p clamped to [0, {subpop.GAIN_CLAMP}] before the gain"
                if args.cell == "G" else "none"
            ),
            "seed": subpop.P_STREAM_SEED,
        },
        "min_keep": config["min_keep"],
        "rescaling_policy": config["rescaling_policy"],
        "key_padding_mask_semantics": (
            "boolean complement of the whole-unit keep mask, [B,N], True = "
            "excluded from the attention softmax; src is never zeroed"
        ) if args.cell == "T" else "n/a (no masking)",
        "generator_namespaces": {
            "p_stream": f"numpy PCG64({subpop.P_STREAM_SEED}) (imported from src/tfpd_lane/sparsification.py, never duplicated)",
            "unit_mask": f"torch.Generator CPU seeded {subpop.UNIT_MASK_SEED} (cell T only, whole-unit draws + min_keep restoration)",
        },
        "eval_mask_policy": (
            "perturbation disabled: no mask and no gain applied, "
            "key_padding_mask is not passed at all; forward bitwise equal to "
            "the unsparsified parent path"
        ),
        "theta_authority": {"path": str(THETA_AUTHORITY), "sha256": theta_sha,
                            "authority_sha256": theta_payload["authority_sha256"],
                            "used_for_masking": False,
                            "note": "per-session unit counts only; T/G masks are direction-free"},
        "behavior_scaling_convention": "unscaled standardized behavior (exact Arm A replica)",
    }

    launch = {
        "schema": "tfpd_subpop_cell_v1_launch",
        "status": "CELL_LAUNCHED",
        "cell": args.cell,
        "cell_name": CELL_NAMES[args.cell],
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
        "launch_proofs": {
            "true_removal": removal_proof,
            "gain_rule": gain_proof,
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
            "per_session_unit_counts": per_session_valid,
        },
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "pretraining_used": False,
            "target_based_selection": False,
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
        model.perturbation_stats = []
        if args.cell == "T":
            model.masker.reset_epoch()
        else:
            model.gain_recorder.reset_epoch()
        stats = train_epoch_perturbed(
            model, optimizer, loader, lr_fn, device, p_stream, session_authority,
            phase_step, num_heads=2, max_steps=args.max_train_steps,
        )
        phase_step += stats["optimizer_steps"]
        p_epoch_stats = p_stream.stats()
        perturbation_agg = subpop.summarize_perturbation(model)
        if perturbation_agg["n_forwards"] != stats["optimizer_steps"]:
            raise SystemExit("perturbation/forward count drift")
        if args.cell == "T" and (
            perturbation_agg.get("rows_below_min_keep_after_guard", 0) != 0
            or perturbation_agg.get("all_masked_rows_after_guard", 0) != 0
        ):
            raise SystemExit("min_keep guard failed: a row fell below the floor")

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
            "perturbation_summary": perturbation_agg,
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
            p_epoch_stats["n"] == phase_step,
        ]):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfpd_subpop_ckpt_v1",
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
            "lr_last": lr_fn(phase_step - 1),
            "p_median": round(p_epoch_stats.get("median") or 0.0, 4),
            "surviving_min": perturbation_agg.get("surviving_units_min"),
            "min_keep_rows": perturbation_agg.get("min_keep_trigger_rows"),
            "gain_max": perturbation_agg.get("gain_max"),
            "clamp_rate": perturbation_agg.get("clamp_trigger_rate"),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = subpop.build_subpop_model(seed=args.seed, cell=args.cell)
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
            "schema": "tfpd_subpop_cell_v1",
            "status": status,
            "cell": args.cell,
            "cell_name": CELL_NAMES[args.cell],
            "integrity": integrity_block,
            "p_sequence_sha256": p_stream.sha256(),
            "p_stream_total_draws": phase_step,
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
        "cell": args.cell, "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "p_sequence_sha256": p_stream.sha256()[:16],
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("CELL_TERMINAL", "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
