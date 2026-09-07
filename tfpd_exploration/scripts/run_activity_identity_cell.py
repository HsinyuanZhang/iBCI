#!/usr/bin/env python3
"""Route-owned AM / IM activity-vs-identity cell runner (handoff 2026-08-19).

Exact Arm A / D training replica (canonical initial tensor bytes strict=True,
2 heads, identical state keys/shapes/parameter count, strict-27 full window
set, 33,925 steps/epoch, batch-32 seed-42 session sampler, warmup 1e-5->1e-4
over two epochs then cosine to 1e-6, 48 epochs, no clipping, final-four SWA,
seed 42) with exactly ONE change at the exact D site
(before `src = activity + identity` is reassembled, before `fc_in`), driven by
the exact shared p stream (one p ~ U(0,1) per forward, SHA-recorded, imported
from src/tfpd_lane/sparsification.py — never duplicated — so AM and IM draw
the same sequence as R / S2 / T / G, `p_sequence_sha256` = e62fc92f...):

- cell AM: `src = activity * mask + identity` — a dropped unit is present and
  identifiable, but silent.
- cell IM: `src = activity + identity * mask` — a dropped unit is audible but
  anonymous.
- D (sealed reference) masks the SUM: `(activity + identity) * mask`.

The mask is drawn by the SAME F.dropout call form D uses
(`torch.nn.functional.dropout(torch.ones(B, N).to(ref), p=p, training=True)`)
on the GLOBAL torch RNG, so the 1/(1-p) survivor gain is INHERITED from the
kernel rather than reimplemented, and the mask law is code-path-matched to D
(unlike R / S2 / T, which used route-owned generators).  No min_keep, no
clamp, no floor: nothing is excluded from the attention softmax, so an
all-masked row or an all-zero-mask forward (p -> 1) has no NaN trap — exactly
D's own edge.

Evaluation/scoring forwards run with the perturbation disabled: NO mask at
all, `src = activity + identity` literally, bitwise equal to the unsparsified
parent path (asserted at launch and in tests).

Integrity block (launch + terminal): cell, num_heads, mask_structure, the
code-path match to D, p law + stream seed + draws-per-forward, gain rule,
min_keep policy, eval policy, application site, supersedes (null: first
round), behavior-scaling convention.  Per-epoch receipt fields: realized p
quantiles, the exact surviving-unit-count histogram over rows, all-zero-mask
row/forward counts, the bitwise mask-value law check, p stream statistics,
gradient finiteness and loss.
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
    "src/tfpd_lane/activity_identity_cells.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_activity_identity_cell.py",
    "scripts/run_admission_arm.py",
)
CELL_NAMES = {
    "AM": "cellAM_activity_mask",
    "IM": "cellIM_identity_mask",
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
    parser.add_argument("--cell", required=True, choices=["AM", "IM"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/aimask_v1")
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
                "schema": "tfpd_aimask_cell_v1",
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
    aimask = _load_module(
        "tfpd_lane_activity_identity_cells", ROOT / "src/tfpd_lane/activity_identity_cells.py"
    )
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    config = aimask.CELLS[args.cell]

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
    model = aimask.build_activity_identity_model(seed=args.seed, cell=args.cell)
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
    aimask.assert_full_budget(steps_per_epoch, epochs, smoke=args.smoke)
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
    # bitwise equal to the unsparsified parent path (no mask at all, the
    # literal `src = activity + identity` sum)
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

    # cell-specific launch proofs on the same fixed batch: the mask hit ONLY
    # the intended component, bitwise
    with torch.no_grad():
        identity_probe = model.compute_identity(fx_calib[:4], side_features=fx_side[:4])
        activity_probe = fx_neural[:4].permute(0, 2, 1)
        site_proof = aimask.prove_site(model, activity_probe, identity_probe,
                                       dropped_unit=0)
    if not (
        site_proof["masked_component_change_bitwise_invisible"]
        and site_proof["unmasked_component_change_visible"]
        and site_proof["survivor_masked_component_change_visible"]
        and site_proof["output_finite"]
    ):
        raise SystemExit(f"site proof failure at launch: {site_proof}")
    assert model.perturbation_enabled is False  # the proofs never leak either

    # code-path proof: the mask call is the same F.dropout call form D uses
    code_path_proof = aimask.probe_mask_code_path(
        model, fx_neural[:4], identity_probe, p=0.5
    )
    if not code_path_proof["matches_D_call_form"]:
        raise SystemExit(f"mask code-path proof failure at launch: {code_path_proof}")
    assert model.perturbation_enabled is False
    d_site = aimask.d_reference_site()  # read-only fingerprint of the D site
    model.recorder.reset_epoch()  # the proofs above never reach the receipts

    p_stream = aimask.pstream()
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
        "mask_structure_note": (
            "whole-unit [B, N] F.dropout mask broadcast over W (mask."
            "unsqueeze(-1)) applied to ONE component only — the exact D code "
            "path, which masks the sum"
        ),
        "mask_code_path_matches_D": True,
        "d_reference_site": d_site,
        "sparsification_site": (
            "pre-sum, before fc_in: activity * mask + identity (AM) / "
            "activity + identity * mask (IM); D masks (activity + identity)"
        ),
        "application_site": config["application_site"],
        "dropped_unit_becomes": config["dropped_unit_becomes"],
        "p_law": {
            "law": (
                "one p ~ Uniform(0,1) per training forward, PCG64(42), exact "
                "stream shared with the R / S2 / T / G cells"
            ),
            "seed": aimask.P_STREAM_SEED,
            "draws_per_forward": 1,
            "distribution_matched_to_D": True,
            "bitwise_matched_to_D": False,
            "note": (
                "D sampled p on the Python random module without recording "
                "draws; these cells take it from the SHA-recorded PCG64(42) "
                "stream, so a full run's p_sequence_sha256 equals the "
                f"R/S2/T/G value {aimask.EXPECTED_P_SEQUENCE_SHA256[:16]}..."
            ),
            "clamp": "none",
        },
        "gain_rule": config["gain_rule"],
        "min_keep": config["min_keep"],
        "key_padding_mask_semantics": (
            "n/a — nothing is excluded from the attention softmax; masked "
            "units still contribute a token (their unmasked component)"
        ),
        "generator_namespaces": {
            "p_stream": (
                f"numpy PCG64({aimask.P_STREAM_SEED}) (imported from "
                "src/tfpd_lane/sparsification.py, never duplicated)"
            ),
            "unit_mask": (
                "the GLOBAL torch RNG inside torch.nn.functional.dropout — "
                "exactly D's own code path (unlike R / S2 / T, which used "
                "route-owned torch.Generator namespaces)"
            ),
        },
        "eval_mask_policy": aimask.EVAL_POLICY,
        "theta_authority": {"path": str(THETA_AUTHORITY), "sha256": theta_sha,
                            "authority_sha256": theta_payload["authority_sha256"],
                            "used_for_masking": False,
                            "note": "per-session unit counts only; AM/IM masks are direction-free"},
        "behavior_scaling_convention": "unscaled standardized behavior (exact Arm A replica)",
        "supersedes": None,
        "supersedes_note": "first round of the activity/identity decomposition",
    }

    launch = {
        "schema": "tfpd_aimask_cell_v1_launch",
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
            "site": site_proof,
            "mask_code_path": code_path_proof,
        },
        "budget": {
            "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "p_draws_per_forward": 1,
            "expected_full_budget_p_draws": aimask.FULL_BUDGET_P_DRAWS,
            "expected_full_budget_p_sequence_sha256": aimask.EXPECTED_P_SEQUENCE_SHA256,
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
            "mixing_coefficient_between_AM_and_IM": "none (explicitly prohibited)",
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
        model.recorder.reset_epoch()
        stats = train_epoch_perturbed(
            model, optimizer, loader, lr_fn, device, p_stream, session_authority,
            phase_step, num_heads=2, max_steps=args.max_train_steps,
        )
        phase_step += stats["optimizer_steps"]
        p_epoch_stats = p_stream.stats()
        perturbation_agg = aimask.summarize_perturbation(model)
        if perturbation_agg["n_forwards"] != stats["optimizer_steps"]:
            raise SystemExit("perturbation/forward count drift")
        if perturbation_agg["mask_value_law_violations"] != 0:
            raise SystemExit(
                "mask value law violated: a mask entry was neither exactly 0 "
                "nor exactly 1/(1-p)"
            )
        if perturbation_agg["rows_total"] != stats["optimizer_steps"] * 32:
            raise SystemExit("mask row count drift")

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
            "mask_code_path_matches_D": True,
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
                    "kind": "tfpd_aimask_ckpt_v1",
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
            "surviving_median": perturbation_agg.get("surviving_units_median"),
            "all_zero_mask_rows": perturbation_agg.get("all_zero_mask_rows"),
            "all_zero_mask_forwards": perturbation_agg.get("all_zero_mask_forwards"),
            "mask_value_law_violations": perturbation_agg.get("mask_value_law_violations"),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = aimask.build_activity_identity_model(seed=args.seed, cell=args.cell)
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
            "schema": "tfpd_aimask_cell_v1",
            "status": status,
            "cell": args.cell,
            "cell_name": CELL_NAMES[args.cell],
            "integrity": integrity_block,
            "p_sequence_sha256": p_stream.sha256(),
            "p_stream_total_draws": phase_step,
            "p_sequence_matches_shared_stream": (
                p_stream.sha256() == aimask.EXPECTED_P_SEQUENCE_SHA256
                if phase_step == aimask.FULL_BUDGET_P_DRAWS else
                f"prefix of the shared stream ({phase_step} of "
                f"{aimask.FULL_BUDGET_P_DRAWS} draws; smoke or truncated run)"
            ),
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
