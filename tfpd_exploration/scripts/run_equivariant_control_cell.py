#!/usr/bin/env python3
"""Route-owned matched-capacity NON-equivariant control cell (arm C-prime).

Operator follow-up: the C - A gate is confounded by construction (exact
equivariance disconnected/froze 3,203,684 ordinary-consumer parameters, so arm
C trains 492,951 parameters vs sealed Cell D's 3,510,842).  This cell is the
matched-capacity control: IDENTICAL to arm C in every respect — same
``EquivariantCarrierConsumer`` class and 185,793-parameter head, same
carrier-blind canonical-Z4 fused path, same frozen inactive modules, same
492,951 trainable set, same raw-T4 carrier authority and frozen m_scale, same
training contract (canonical strict load, strict-27, 33,925 x 48 budget guard,
batch-32 seed-42 session sampler, warmup->cosine, fresh Adam, no clipping,
final-four SWA, launch/terminal receipts with closure equality, seed 42) —
EXCEPT the carrier enters as ordinary real per-unit features: the
per-component z-scored RAW (a, c) with the source-fit statistics (the ordinary
pipeline's own quantity) plus the same invariants (m, b z-scores,
log1p(|beta|/m_scale), validity).  No complex-unit value stream, no
equivariance claim; a launch probe must show the rotation property FAILS
grossly (the mirror of arm C's equivariance proof).

Pre-registered reading (launch + terminal receipts): C - C-prime =
equivariance at matched capacity; C-prime - A = capacity-confound bound;
C-prime - B also reported.  Any superiority claim keeps the external-governing
gate (mean paired delta >= +0.03 AND >= 10/15 positive).

This is a NEW runner by construction: the launched arm B / C runs hash
``src/tfpd_lane/equivariant_cell.py`` and ``scripts/run_equivariant_cell.py``
in their source closures, so those files stay byte-frozen and are reused here
by import only (their bytes are hash-bound in this runner's own closure as
authorities).
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
    "src/tfpd_lane/equivariant_control.py",
    "src/tfpd_lane/equivariant_cell.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_equivariant_control_cell.py",
    "scripts/run_admission_arm.py",
)
CELL_KEY = "Cprime_matched_control"
CELL_NAME = "cellCprime_matched_control"
AUTHORITY_ROOT = ROOT / "results/equivariant_v1/carrier_authority"
# the matched control must violate the rotation property grossly (relative to
# its own output scale); arm C satisfies it to 1e-5
NONEQUIVARIANCE_RELATIVE_FLOOR = 1e-2


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_sealed_authority(equivariant, receipt_mod, authority_root: Path):
    """Load + verify the SHARED sealed raw-T4 carrier authority (never rebuilt).

    Returns (receipt_info, payload): the info dict is JSON-safe for receipts;
    the payload (torch tensors) never enters a receipt.
    """
    artifact = authority_root / "carrier_authority.pt"
    sidecar = Path(str(artifact) + ".sha256")
    receipt_path = authority_root / "carrier_authority_receipt.json"
    if not artifact.is_file() or not sidecar.is_file() or not receipt_path.is_file():
        raise SystemExit(
            "sealed carrier authority missing (built by the arm C run): "
            f"{authority_root}"
        )
    if receipt_mod.sha256_file(artifact) != sidecar.read_text().split()[0]:
        raise SystemExit("carrier authority SHA mismatch against sidecar")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "CARRIER_AUTHORITY_SEALED":
        raise SystemExit("carrier authority receipt not sealed")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if payload["kind"] != equivariant.CARRIER_AUTHORITY_KIND:
        raise SystemExit("carrier authority kind drift")
    if payload["authority_sha256"] != receipt["authority_sha256"]:
        raise SystemExit("carrier authority receipt/artifact SHA mismatch")
    info = {
        "path": str(artifact),
        "sha256": sidecar.read_text().split()[0],
        "receipt": receipt_path.name,
        "m_scale": float(payload["m_scale"]),
        "shared_with_arm_c": True,
    }
    return info, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default=CELL_KEY, choices=[CELL_KEY])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/equivariant_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=5)
    parser.add_argument("--initial-state", type=Path,
                        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt")
    parser.add_argument("--authority-root", type=Path, default=AUTHORITY_ROOT)
    parser.add_argument("--launch-probe-rotations", type=int, default=8)
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
                "schema": "tfpd_equivariant_control_cell_v1",
                "cell": CELL_KEY,
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
    equivariant = _load_module(
        "tfpd_lane_equivariant", ROOT / "src/tfpd_lane/equivariant_cell.py"
    )
    control = _load_module(
        "tfpd_lane_equivariant_control", ROOT / "src/tfpd_lane/equivariant_control.py"
    )
    equiv_runner = _load_module(
        "tfpd_equivariant_runner", ROOT / "scripts/run_equivariant_cell.py"
    )
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_runner = _load_module("tfpd_admission_runner", ROOT / "scripts/run_admission_arm.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- canonical initial state artifact ---------------------------------
    initial_sidecar = Path(str(args.initial_state) + ".sha256")
    if not args.initial_state.is_file() or not initial_sidecar.is_file():
        raise SystemExit("canonical initial state artifact missing")
    initial_sha = receipt_mod.sha256_file(args.initial_state)
    if initial_sha != initial_sidecar.read_text().split()[0]:
        raise SystemExit("canonical initial state SHA mismatch")
    initial_payload = torch.load(args.initial_state, map_location="cpu",
                                 weights_only=False)
    canonical_state = initial_payload["state_dict"]

    # ---- data (arm-A contract; development rosters never opened) -----------
    dm, a2 = arm_runner.build_datamodule(args)
    train_dataset = dm.train_dataset
    if len(dm.session_splits["train"]) != 27:
        raise SystemExit("strict-27 roster drift")
    if receipt_mod.sha256_file(a2.MANIFEST_PATH) != a2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit("manifest SHA drift")
    behavior_semantic = a2.normalizer_value_sha256(*dm._behavior_stats)
    if not behavior_semantic.startswith("f062506c"):
        raise SystemExit("source behavior normalizer semantic SHA drift")
    side_mean, side_std = dm._side_feature_stats
    side_semantic = a2.normalizer_value_sha256(side_mean, side_std)
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    pl.seed_everything(args.seed, workers=True)

    # ---- model: the arm C graph verbatim + the matched-control carrier view -
    model = control.build_matched_control_model(seed=args.seed)
    prefix_proof = equivariant.load_canonical_prefix(model, canonical_state)
    if prefix_proof["canonical_subset_sha256"] != initial_payload["state_sha256"]:
        raise SystemExit("canonical-prefix subset SHA != canonical artifact SHA")
    freeze = equivariant.freeze_inactive_consumer_modules(model)
    accounting = equivariant.parameter_accounting(model)

    # exact capacity parity vs arm C (a second graph built and counted here)
    arm_c_reference = equivariant.build_equivariant_model(seed=args.seed)
    equivariant.load_canonical_prefix(arm_c_reference, canonical_state)
    equivariant.freeze_inactive_consumer_modules(arm_c_reference)
    parity = control.matched_parameter_parity(model, arm_c_reference)
    del arm_c_reference

    # ---- the shared sealed raw-T4 carrier authority -------------------------
    authority_info, payload = _load_sealed_authority(
        equivariant, receipt_mod, Path(args.authority_root)
    )
    authority_np = {
        name: {
            "raw": entry["raw"].numpy(),
            "valid": entry["valid"].numpy(),
            "n_units": int(entry["n_units"]),
            "raw_t4_sha256": entry["raw_t4_sha256"],
        }
        for name, entry in payload["authority"].items()
    }
    missing = set(dm.session_splits["train"]) - set(authority_np)
    if missing:
        raise SystemExit(f"carrier authority missing sessions: {sorted(missing)}")
    # the per-component z-scores are the SOURCE-FIT statistics — the ordinary
    # pipeline's own normalizer values for the raw (a,c) columns
    mean_ac = np.asarray(side_mean, dtype=np.float64)[0:2]
    std_ac = np.asarray(side_std, dtype=np.float64)[0:2]
    bundles = control.build_matched_control_bundles(
        authority_np, train_dataset.sessions, authority_info["m_scale"],
        mean_ac, std_ac,
    )

    z4_authority = arm_runner.verify_z4_is_zeros_like_t4(dm, a2.SOURCE_CACHE_ROOT)
    if not (z4_authority["all_bitwise_equal"] and z4_authority["all_exact_positive_zero"]):
        raise SystemExit("z4/zeros_like(normalized T4) authority check failed")
    model.to(device)

    # ---- sampler / budget / fixed batch ------------------------------------
    sampler = SessionBatchSampler(train_dataset, batch_size=args.train_batch_size,
                                  shuffle=True, seed=args.seed)
    steps_per_epoch = len(sampler)
    epochs = args.smoke_epochs if args.smoke else args.epochs
    budget = equivariant.assert_training_budget(steps_per_epoch, epochs, args.smoke)
    loader = DataLoader(train_dataset, batch_sampler=sampler,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, _fx_beh, fx_calib, fx_sess, fx_side = fixed_batch[:5]
    fx_neural, fx_calib, fx_side = (
        fx_neural.to(device), fx_calib.to(device), fx_side.to(device),
    )
    fx_session = fx_sess[0] if isinstance(fx_sess, (list, tuple)) else fx_sess

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    schedule = arm_common.schedule_params(epochs, steps_per_epoch)
    lr_fn = lambda step: arm_common.lr_at_step(step, epochs, steps_per_epoch)

    # ---- launch proofs ------------------------------------------------------
    launch_proofs = {}
    model.set_carrier(bundles[fx_session])
    thetas = [0.0] + equivariant.probe_thetas(args.launch_probe_rotations)
    probe = control.nonequivariance_probe(
        model, fx_neural[:4], fx_calib[:4], fx_side[:4], thetas)
    if not probe["identity_rotation_bitwise_equal"]:
        raise SystemExit("matched control: identity rotation is not a bitwise baseline")
    if probe["relative_max_violation"] is None or \
            probe["relative_max_violation"] <= NONEQUIVARIANCE_RELATIVE_FLOOR:
        raise SystemExit(
            "matched control failed to violate the rotation property "
            f"(relative max violation {probe['relative_max_violation']})"
        )
    launch_proofs["nonequivariance_probe"] = probe
    launch_proofs["identity_rotation_bitwise_baseline"] = True
    # carrier blindness of the fused path (identical construction to arm C)
    alt_side = fx_side.clone()
    alt_side[..., 2:4] = torch.randn_like(alt_side[..., 2:4]) * 5.0
    model.eval()
    with torch.no_grad():
        base_out, _ = model(fx_neural[:2], calib_trials=fx_calib[:2],
                            side_features=fx_side[:2])
        alt_out, _ = model(fx_neural[:2], calib_trials=fx_calib[:2],
                           side_features=alt_side[:2])
    if not torch.equal(base_out, alt_out):
        raise SystemExit("fused path is not carrier-blind (side (m,b) leaked)")
    launch_proofs["fused_path_blind_to_unused_side_columns"] = True
    launch_proofs["parameter_parity_vs_arm_c"] = parity

    integrity = control.matched_control_integrity_block(
        num_heads=2,
        canonical_parameters=accounting["canonical_tensors_strict_loaded"],
        consumer_parameters=accounting["new_consumer_parameters"],
        total_trainable=accounting["total_trainable"],
        m_scale=authority_info["m_scale"],
        launch_max_violation=probe["max_violation"],
        launch_relative_violation=probe["relative_max_violation"],
    )

    launch = {
        "schema": "tfpd_equivariant_control_cell_v1_launch",
        "status": "CELL_LAUNCHED",
        "cell": CELL_KEY,
        "cell_name": CELL_NAME,
        "integrity": integrity,
        "launch_proofs": launch_proofs,
        "smoke": args.smoke,
        "started_utc": started,
        "initial_state": {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "canonical_prefix_proof": prefix_proof,
            "inactive_modules_frozen": freeze,
            "bitwise_parity_at_init_vs_cell_d": False,
            "disclosed": (
                "matched-capacity control: canonical tensors bitwise "
                "strict-loaded, consumer head identical to arm C's, carrier "
                "view ordinary z-scored reals (not covariant)"
            ),
        },
        "budget": {
            **budget,
            "total_optimizer_steps": epochs * steps_per_epoch,
            "schedule": schedule, "optimizer": arm_common.ADAM_CONSTRUCTOR,
            "sampler": "mc_maze SessionBatchSampler(batch=32, shuffle, seed=42)",
        },
        "data_contract": {
            "roster_n": 27,
            "within_dev_sessions_opened": False,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "visible_side": (
                "canonical Z4 fused path + RAW carrier authority through the "
                "ordinary z-scored view (per-component source-fit statistics)"
            ),
            "behavior_normalizer_semantic_sha256": behavior_semantic,
            "side_feature_semantic_sha256": side_semantic,
            "per_component_zscore_statistics": {
                "mean_ac": [float(v) for v in mean_ac],
                "std_ac": [float(v) for v in std_ac],
                "source": "dm._side_feature_stats columns 0:2 (the ordinary pipeline's normalizer)",
            },
        },
        "carrier_authority": authority_info,
        "z4_authority": z4_authority,
        "t4_authority_sha256": t4_authority,
        "parameter_accounting": accounting,
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "pretraining_used": False,
            "target_based_selection": False,
            "clipping": "none",
            "gates_preregistered": True,
            "equivariance_claim": False,
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
        stats = equiv_runner.train_epoch_equivariant(
            model, optimizer, loader, lr_fn, device, bundles,
            phase_step, num_heads=2, max_steps=args.max_train_steps,
        )
        phase_step += stats["optimizer_steps"]

        w_side = arm_common.w_side_block(model)
        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter) and p.numel()
        )
        authority_ok = (
            arm_common.t4_authority_fingerprint(train_dataset.sessions) == t4_authority
        )
        model.set_carrier(bundles[fx_session])
        rot_probe = control.nonequivariance_probe(
            model, fx_neural[:4], fx_calib[:4], fx_side[:4],
            [0.0] + equivariant.probe_thetas(
                4, seed=equivariant.EQUIVARIANCE_PROBE_SEED + epoch
            ),
        )
        diag = {
            "epoch": epoch,
            "visible_side_mode": "z4_fused_raw_carrier_matched_control",
            "duration_s": round(time.time() - t0, 3),
            **stats,
            "matched_control_rotation_probe": rot_probe,
            "optimizer_steps_total": phase_step,
            "lr_expected_first": schedule["lr_at_step_0"],
            "lr_expected_last": schedule["lr_at_final_step"],
            "clipping_authorized": False,
            "w_side_norm": float(w_side.norm().item()),
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
        checks = [
            stats["nonfinite_loss_steps"] == 0,
            stats["nonfinite_grad_steps"] == 0,
            params_finite, diag["optimizer_state_finite"], authority_ok, ok_steps,
            stats["w_side_grad_exact_zero_all_steps"],
            rot_probe["identity_rotation_bitwise_equal"],
            rot_probe["relative_max_violation"] is not None
            and rot_probe["relative_max_violation"] > NONEQUIVARIANCE_RELATIVE_FLOOR,
        ]
        if not all(checks):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfpd_equivariant_control_cell_ckpt_v1",
                    "cell": CELL_KEY,
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
            "cell": CELL_KEY, "epoch": epoch,
            "loss": round(stats["train_loss_mean_per_step"], 6),
            "lr_last": lr_fn(phase_step - 1),
            "nonequivariance_relative_max": round(
                rot_probe["relative_max_violation"], 6),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = control.build_matched_control_model(seed=args.seed)
    equivariant.freeze_inactive_consumer_modules(swa_model)
    swa_model.load_state_dict(swa_state, strict=True)
    swa_model.to(device).eval()
    with torch.no_grad():
        swa_model.set_carrier(bundles[fx_session])
        swa_pred, _ = swa_model(fx_neural[:4], calib_trials=fx_calib[:4],
                                side_features=fx_side[:4])
        swa_finite = bool(torch.isfinite(swa_pred).all().item())
        swa_probe = control.nonequivariance_probe(
            swa_model, fx_neural[:4], fx_calib[:4], fx_side[:4],
            [0.0] + equivariant.probe_thetas(args.launch_probe_rotations),
        )
        if (not swa_finite
                or not swa_probe["identity_rotation_bitwise_equal"]
                or swa_probe["relative_max_violation"] <= NONEQUIVARIANCE_RELATIVE_FLOOR):
            raise SystemExit("SWA matched-control model failed its probe")
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
            "schema": "tfpd_equivariant_control_cell_v1",
            "status": status,
            "cell": CELL_KEY,
            "cell_name": CELL_NAME,
            "integrity": integrity,
            "launch_proofs": launch_proofs,
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "initial_state": launch["initial_state"],
            "budget": launch["budget"],
            "data_contract": launch["data_contract"],
            "carrier_authority": authority_info,
            "z4_authority": z4_authority,
            "t4_authority_sha256": t4_authority,
            "parameter_accounting": accounting,
            "epochs_run": len(diagnostics),
            "diagnostics_per_epoch": diagnostics,
            "invariant_failures": invariant_failures,
            "checkpoints": checkpoints,
            "swa": {
                "path": str(swa_path), "sha256": swa_sha,
                "window_epochs": [c["epoch"] for c in checkpoints],
                "manifest": swa_manifest,
                "strict_reload_finite_forward_smoke": swa_finite,
                "nonequivariance_probe_after_swa": swa_probe,
            },
            "disclosures": launch["disclosures"],
            "source_closure": {"launch": closure_launch, "final": closure_final,
                               "launch_final_closure_equal": closure_equal},
            "environment": launch["environment"],
        },
    )
    print(json.dumps({
        "cell": CELL_KEY, "status": status,
        "epochs_run": len(diagnostics),
        "final_loss": round(diagnostics[-1]["train_loss_mean_per_step"], 6),
        "nonequivariance_relative_max_final": round(
            diagnostics[-1]["matched_control_rotation_probe"]["relative_max_violation"], 6),
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("CELL_TERMINAL", "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
