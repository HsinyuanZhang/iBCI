#!/usr/bin/env python3
"""Gate-2 three-arm admission trainer (HANDOFF_..._20260816.md §3/§4/§6/§9/§15).

Arms (all strict recombinations of one implementation surface):

  A  direct_t4_48            : canonical full T4, 48 epochs, one fresh Adam,
                               step-level linear warmup 1e-5 -> 1e-4 over the
                               first two epochs, cosine decay to 1e-6 at the
                               final step of epoch 47.
  B  z4_pretrain_then_t4     : canonical Z4 for exactly T_pre epochs at
                               constant lr=1e-4 (identical to the sealed
                               boundary pilot; W_side / exp_avg / exp_avg_sq
                               stay elementwise exactly zero and the visible
                               side stays bitwise canonical Z4 every epoch),
                               then an ATOMIC admission switch to the aligned
                               normalized T4, a FRESH Adam, and arm C's exact
                               T4-phase schedule for E_t4 epochs.  No weight
                               is reset; the T4 branch starts from W_side=0.
  C  direct_t4_exposure_matched: same canonical initial state, canonical full
                               T4 from its first batch, exactly E_t4 epochs;
                               the T4 phase is step-for-step equal to B's
                               phase 2 (constructor, warmup, cosine endpoint,
                               LR at every T4-phase step, batch order, step
                               count).

Common invariants: strict-27 source roster, M30 chronological calibration,
seed 42, one immutable canonical initial state loaded strictly, identical
training examples and sampler order across arms (the official cells'
SessionBatchSampler: batch 32, session-grouped, frozen seed-42 permutation),
Adam lr=1e-4 betas=(0.9,0.999) eps=1e-8 weight_decay=0 amsgrad=False before
the schedule modifies LR, no clipping, no dev-session validation, no
checkpoint selection, no formal/organizer-held data.

Receipts (0444, transactional): launch (sealed before the first gradient
step, full §15 provenance), terminal (per-epoch §9 diagnostics, transition
record, checkpoint/SWA SHAs, disclosures, closure equality).  The final
checkpoint and the predeclared final-four SWA (matched_scorer.
build_swa_final_four) are sealed read-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

import numpy as np
import torch
from torch.utils.data import DataLoader

PAD_VALUE = -1.0
WINDOW_SIZE = 50

BOUND_PATTERNS = (
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_admission_arm.py",
    "scripts/make_canonical_initial_state.py",
    "src/tfpd/spintshape_module.py",
)
AUTHORITY_PATTERNS = (
    "../sua_exploration/mc_maze/multisession_datamodule.py",
    "../sua_exploration/mc_maze/unit_side_features.py",
    "../streaming_calibration_exp/src/models/components/spint.py",
    "../streaming_calibration_exp/src/models/components/streaming_spint.py",
    "../streaming_calibration_exp/src/models/components/streaming_encoders.py",
)
ARM_NAMES = {
    "A": "armA_direct_t4_48",
    "B": "armB_z4_pretrain_then_t4",
    "C": "armC_direct_t4_exposure_matched",
}


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


arm_common = _load_module("tfpd_lane_arm_common", "src/tfpd_lane/arm_common.py")
matched_scorer = _load_module("tfpd_lane_matched_scorer", "src/tfpd_lane/matched_scorer.py")
receipt_mod = _load_module("tfpd_lane_receipt", "src/tfpd_lane/receipt.py")


def seal_file(path: Path) -> str:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    digest = arm_common.sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "  " + path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return digest


def load_boundary(path: Path) -> dict:
    """Verify the sealed boundary receipt (tamper-evident) and return T_pre/E_t4."""
    payload = json.loads(Path(path).read_text())
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise SystemExit(f"boundary receipt sidecar missing: {sidecar}")
    expected = sidecar.read_text().split()[0]
    if arm_common.sha256_file(path) != expected:
        raise SystemExit("boundary receipt SHA mismatch against its sidecar")
    if payload.get("status") != "Z4_BOUNDARY_PILOT_TERMINAL":
        raise SystemExit(f"boundary pilot not terminal: {payload.get('status')}")
    if not payload.get("source_closure", {}).get("launch_final_closure_equal", False):
        raise SystemExit("boundary pilot closure inequality")
    if payload.get("invariant_failures"):
        raise SystemExit("boundary pilot invariant failures present")
    boundary = payload.get("boundary") or {}
    t_pre, e_t4 = boundary.get("t_pre"), boundary.get("e_t4")
    if not isinstance(t_pre, int) or not isinstance(e_t4, int):
        raise SystemExit("boundary receipt lacks integer t_pre/e_t4")
    if t_pre + e_t4 != arm_common.TOTAL_BUDGET_EPOCHS:
        raise SystemExit("boundary t_pre + e_t4 != 48")
    return {
        "path": str(path),
        "receipt_sha256": expected,
        "t_pre": t_pre,
        "e_t4": e_t4,
        "selected_epoch": boundary.get("selected_epoch"),
        "s_max_source_audit": boundary.get("s_max_source_audit"),
    }


def build_datamodule(args):
    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=args.train_batch_size,
        window_size=WINDOW_SIZE,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        random_calibration=False,
        seed=args.seed,
        max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT),
        signal_view="sua",
        side_feature_group="t4",
        side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    # The arms never touch development data: resolve the frozen manifest, then
    # drop the validation roster before any session array is opened.  No
    # within-dev/external/formal session is loaded anywhere in this trainer.
    dm._initialize_splits()
    dm.session_files["val"] = []
    dm.setup("fit")
    return dm, a2


def verify_z4_is_zeros_like_t4(datamodule, cache_dir) -> dict:
    """§4: production z4 loader output is bitwise zeros_like(normalized T4)."""
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import load_unit_side_features

    side_mean, side_std = datamodule._side_feature_stats
    rows = []
    for nwb_path in datamodule.session_files["train"]:
        name = session_name_from_path(nwb_path)
        record = datamodule.train_dataset.sessions[name]
        z4, _meta = load_unit_side_features(
            nwb_path,
            feature_group="z4",
            pool_size=30,
            mean=side_mean,
            std=side_std,
            cache_dir=cache_dir,
            bin_size_ms=20,
            window_size=WINDOW_SIZE,
            trial_result_filter="R",
            signal_view="sua",
        )
        zeros = np.zeros_like(record.side_features, dtype=np.float32)
        rows.append(
            {
                "session": name,
                "production_z4_bitwise_equal_to_zeros_like_normalized_t4": bool(
                    z4.shape == zeros.shape and np.array_equal(z4, zeros)
                ),
                "production_z4_exact_positive_zero": bool(
                    np.all(z4 == 0) and not np.any(np.signbit(z4))
                ),
            }
        )
    return {
        "rule": (
            "phase-1 visible side = zeros_like(aligned normalized T4), proven bitwise "
            "equal to load_unit_side_features(group='z4') for every source session"
        ),
        "n_sessions_checked": len(rows),
        "all_bitwise_equal": all(r["production_z4_bitwise_equal_to_zeros_like_normalized_t4"] for r in rows),
        "all_exact_positive_zero": all(r["production_z4_exact_positive_zero"] for r in rows),
        "per_session": rows,
    }


def train_epoch(
    model,
    optimizer,
    loader,
    visible_mode: str,
    lr_fn,
    device,
    phase_local_step: int,
    max_steps=None,
):
    """One epoch.  `lr_fn(step)` is None for the constant-lr phase-1 schedule."""
    model.train()
    encoder_params, decoder_params = arm_common.param_groups_by_branch(model)
    all_params = encoder_params + decoder_params
    w_param = model.id_encoder.post_pool[0].weight
    encoder_mod = model.id_encoder
    hidden, side_dim = encoder_mod.hidden_dim, encoder_mod.side_dim
    acc = {
        "loss_sum": torch.zeros((), device=device),
        "valid_bins": torch.zeros((), device=device),
        "grad_sq_global": torch.zeros((), device=device),
        "grad_sq_encoder": torch.zeros((), device=device),
        "grad_sq_decoder": torch.zeros((), device=device),
        "grad_sq_w_side": torch.zeros((), device=device),
        "grad_max": torch.zeros((), device=device),
        "wside_grad_nonzero": torch.zeros((), device=device, dtype=torch.long),
        "side_violations": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_loss": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_grad": torch.zeros((), device=device, dtype=torch.long),
    }
    steps = 0
    examples = 0
    lr_values = []
    for batch in loader:
        neural, behavior, calib, _session, side = batch[:5]
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)
        visible = arm_common.admit_side(side, visible_mode)
        with torch.no_grad():
            if visible_mode == "z4":
                acc["side_violations"] += (
                    (visible != 0).sum() + visible.signbit().sum() + (~torch.isfinite(visible)).sum()
                )
            else:
                acc["side_violations"] += (
                    (visible != side).sum()
                    + (~torch.isfinite(visible)).sum()
                    + (visible.shape[-1] != side_dim)
                )
        lr = arm_common.PHASE1_LR if lr_fn is None else lr_fn(phase_local_step + steps)
        optimizer.param_groups[0]["lr"] = float(lr)
        lr_values.append(float(lr))

        prediction, _identity = model(neural, calib_trials=calib, side_features=visible)
        valid = (behavior != PAD_VALUE).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            enc_grads = [p.grad for p in encoder_params if p.grad is not None]
            dec_grads = [p.grad for p in decoder_params if p.grad is not None]
            enc_sq = (
                torch.stack(torch._foreach_norm(enc_grads)).pow(2).sum()
                if enc_grads
                else torch.zeros((), device=device)
            )
            dec_sq = (
                torch.stack(torch._foreach_norm(dec_grads)).pow(2).sum()
                if dec_grads
                else torch.zeros((), device=device)
            )
            acc["grad_sq_encoder"] += enc_sq
            acc["grad_sq_decoder"] += dec_sq
            acc["grad_sq_global"] += enc_sq + dec_sq
            acc["grad_max"] = torch.maximum(acc["grad_max"], (enc_sq + dec_sq).sqrt())
            acc["nonfinite_grad"] += (~(torch.isfinite(enc_sq) and torch.isfinite(dec_sq))).long()
            grad = w_param.grad
            if grad is None:
                acc["wside_grad_nonzero"] += 1
            else:
                block = grad[:, hidden : hidden + side_dim]
                acc["wside_grad_nonzero"] += (block != 0).sum()
                acc["grad_sq_w_side"] += block.pow(2).sum()
            acc["nonfinite_loss"] += (~torch.isfinite(loss)).long()
        optimizer.step()
        acc["loss_sum"] += loss.detach()
        acc["valid_bins"] += valid.sum()
        steps += 1
        examples += int(neural.shape[0])
        if max_steps is not None and steps >= max_steps:
            break
    n = max(steps, 1)
    return {
        "optimizer_steps": steps,
        "train_example_windows": examples,
        "train_valid_bins": int(acc["valid_bins"].item()),
        "train_loss_mean_per_step": float(acc["loss_sum"].item()) / n,
        "lr_first": lr_values[0] if lr_values else None,
        "lr_last": lr_values[-1] if lr_values else None,
        "lr_mean": float(np.mean(lr_values)) if lr_values else None,
        "grad_norm_global_mean": float((acc["grad_sq_global"] / n).sqrt().item()),
        "grad_norm_global_max": float(acc["grad_max"].item()),
        "grad_norm_encoder_mean": float((acc["grad_sq_encoder"] / n).sqrt().item()),
        "grad_norm_decoder_mean": float((acc["grad_sq_decoder"] / n).sqrt().item()),
        "grad_norm_w_side_mean": float((acc["grad_sq_w_side"] / n).sqrt().item()),
        "w_side_grad_exact_zero_all_steps": int(acc["wside_grad_nonzero"].item()) == 0,
        "visible_side_violation_count": int(acc["side_violations"].item()),
        "nonfinite_loss_steps": int(acc["nonfinite_loss"].item()),
        "nonfinite_grad_steps": int(acc["nonfinite_grad"].item()),
        "clip_events": 0,
    }


def optimizer_state_finite(optimizer) -> bool:
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            tensor = state.get(key)
            if tensor is not None and not bool(torch.isfinite(tensor).all().item()):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate-2 three-arm admission trainer")
    parser.add_argument("--arm", required=True, choices=["A", "B", "C"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/admission_arms_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--boundary-receipt",
        type=Path,
        default=ROOT / "results/z4_boundary_pilot_v1/terminal_receipt.json",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="non-authoritative CPU smoke: reduced epochs")
    parser.add_argument("--smoke-t4-phase-epochs", type=int, default=4)
    parser.add_argument("--smoke-phase1-epochs", type=int, default=2)
    parser.add_argument("--smoke-arm-a-epochs", type=int, default=6)
    parser.add_argument("--max-train-steps", type=int, default=None)
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA requested but unavailable ({args.device})", file=sys.stderr)
        return 3

    out_dir = Path(args.output_root) / ARM_NAMES[args.arm]
    if args.preflight_only:
        out_dir = Path(args.output_root)
    else:
        if out_dir.exists():
            print(f"fresh arm output directory already exists: {out_dir}", file=sys.stderr)
            return 2
        out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        return _run(args, out_dir, device, started)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - failure receipts are mandatory
        if not args.preflight_only:
            receipt_mod.write_receipt_transactionally(
                out_dir / "terminal_receipt.json",
                {
                    "schema": "tfpd_admission_arm_v1",
                    "arm": args.arm,
                    "status": "ARM_FAILED",
                    "smoke": args.smoke,
                    "started_utc": started,
                    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "failure": {
                        "kind": type(exc).__name__,
                        "detail": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def _run(args, out_dir: Path, device, started: str) -> int:
    import lightning.pytorch as pl
    from mc_maze.multisession_datamodule import SessionBatchSampler
    from torch.nn.parameter import UninitializedParameter
    from torch.utils.data import default_collate

    spintshape = _load_module("tfpd_spintshape_module", "src/tfpd/spintshape_module.py")

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    authorities = {
        rel: receipt_mod.sha256_file(REPO / rel.split("../", 1)[1]) for rel in AUTHORITY_PATTERNS
    }

    boundary = load_boundary(args.boundary_receipt)
    t_pre, e_t4 = boundary["t_pre"], boundary["e_t4"]
    plan = arm_common.build_arm_plan(args.arm, t_pre, e_t4)
    if args.arm == "B":
        # §6.4 equality is asserted from B's side here and from C's side in C's
        # own preflight/launch; arm A is the equal-total-compute baseline.
        bc_equality = arm_common.assert_b_phase2_matches_c_plan(
            plan, arm_common.build_arm_plan("C", t_pre, e_t4)
        )
    elif args.arm == "C":
        bc_equality = arm_common.assert_b_phase2_matches_c_plan(
            arm_common.build_arm_plan("B", t_pre, e_t4), plan
        )
    else:
        bc_equality = {
            "applicability": (
                "arm A is the 48-epoch equal-total-compute baseline; the step-level "
                "B/C T4-phase equality is asserted separately in arms B and C"
            )
        }
    if args.smoke:
        if args.arm == "A":
            plan["phases"][0]["epochs"] = args.smoke_arm_a_epochs
        elif args.arm == "B":
            plan["phases"][0]["epochs"] = args.smoke_phase1_epochs
            plan["phases"][1]["epochs"] = args.smoke_t4_phase_epochs
        else:
            plan["phases"][0]["epochs"] = args.smoke_t4_phase_epochs
        plan["total_model_training_epochs"] = sum(p["epochs"] for p in plan["phases"])
        plan["smoke_epoch_override"] = True
        bc_equality = dict(
            bc_equality,
            smoke_note="smoke run: epochs reduced; schedule shape unchanged (non-authoritative)",
        )

    # ---- canonical initial state, loaded strictly --------------------------
    initial_path = Path(args.initial_state)
    initial_sidecar = Path(str(initial_path) + ".sha256")
    if not initial_path.is_file() or not initial_sidecar.is_file():
        raise SystemExit(f"canonical initial state artifact missing: {initial_path}")
    initial_artifact_sha = initial_sidecar.read_text().split()[0]
    if arm_common.sha256_file(initial_path) != initial_artifact_sha:
        raise SystemExit("canonical initial state artifact SHA mismatch against sidecar")
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)

    # ---- data (train-only; development rosters never opened) ---------------
    dm, a2 = build_datamodule(args)
    train_dataset = dm.train_dataset
    roster = tuple(dm.session_splits["train"])
    formal_names = tuple(dm.session_splits["test"])
    if len(roster) != 27:
        raise SystemExit("strict source roster drift (expected 27 train sessions)")
    manifest_sha = receipt_mod.sha256_file(a2.MANIFEST_PATH)
    if manifest_sha != a2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit(f"strict manifest SHA drift: {manifest_sha}")
    behavior_mean, behavior_std = dm._behavior_stats
    behavior_semantic = a2.normalizer_value_sha256(behavior_mean, behavior_std)
    if not behavior_semantic.startswith("f062506c"):
        raise SystemExit(f"source behavior normalizer semantic SHA drift: {behavior_semantic}")
    side_mean, side_std = dm._side_feature_stats
    side_semantic = a2.normalizer_value_sha256(side_mean, side_std)

    z4_authority = verify_z4_is_zeros_like_t4(dm, a2.SOURCE_CACHE_ROOT)
    if not (z4_authority["all_bitwise_equal"] and z4_authority["all_exact_positive_zero"]):
        raise SystemExit("z4/zeros_like(normalized T4) authority check failed")
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    pl.seed_everything(args.seed, workers=True)
    model = spintshape.build_spintshape_model(seed=args.seed)
    model.load_state_dict(initial_payload["state_dict"], strict=True)
    state_sha_loaded = arm_common.state_sha256(model)
    if state_sha_loaded != initial_payload["state_sha256"]:
        raise SystemExit("loaded initial state SHA != artifact state SHA")
    w_side = arm_common.w_side_block(model)
    if int(torch.count_nonzero(w_side).item()) != 0 or bool(w_side.signbit().any().item()):
        raise SystemExit("W_side is not exactly positive zero after strict initial-state load")
    model.to(device)

    # ---- sampler shared verbatim with the official cells -------------------
    sampler = SessionBatchSampler(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, seed=args.seed
    )
    steps_per_epoch = len(sampler)
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    # fixed source-only diagnostic batch: the sampler's first batch (frozen order)
    fixed_batch = default_collate([train_dataset[i] for i in next(iter(sampler))])
    fx_neural, fx_behavior, fx_calib, _fx_session, fx_side = fixed_batch[:5]
    fx_neural = fx_neural.to(device)
    fx_calib = fx_calib.to(device)
    fx_side_t4 = fx_side.to(device)

    initial_receipt_path = initial_path.parent / "canonical_initial_state_receipt.json"
    if not initial_receipt_path.is_file():
        raise SystemExit(f"canonical initial state receipt missing: {initial_receipt_path}")
    initial_receipt = json.loads(initial_receipt_path.read_text())
    if initial_receipt.get("state_dict_sha256") != initial_payload["state_sha256"]:
        raise SystemExit("initial-state receipt/artifact SHA mismatch")

    t4_phase = [p for p in plan["phases"] if p["visible_side"] == "t4"][-1]
    t4_n = t4_phase["epochs"]
    swa_local_epochs = set(range(max(0, t4_n - 4), t4_n))

    schedules = {}
    for phase in plan["phases"]:
        if phase["lr_schedule"] == "warmup_then_cosine":
            schedules[phase["phase"]] = arm_common.schedule_params(
                phase["epochs"], steps_per_epoch
            )
        else:
            schedules[phase["phase"]] = {
                "kind": "constant", "lr": arm_common.PHASE1_LR,
                "n_epochs": phase["epochs"], "steps_per_epoch": steps_per_epoch,
            }

    preflight_body = {
        "arm": args.arm,
        "arm_name": ARM_NAMES[args.arm],
        "smoke": args.smoke,
        "started_utc": started,
        "boundary_provenance": boundary,
        "arm_plan": plan,
        "b_phase2_vs_c_equality": bc_equality,
        "lr_schedules": schedules,
        "adam_constructor": arm_common.ADAM_CONSTRUCTOR,
        "initial_state": {
            "path": str(initial_path),
            "artifact_sha256": initial_artifact_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "loaded_state_sha256": state_sha_loaded,
            "strict_load": True,
            "w_side_exact_positive_zero": True,
        },
        "data_contract": {
            "roster": list(roster),
            "n_train_sessions": len(roster),
            "manifest_path": str(a2.MANIFEST_PATH),
            "manifest_sha256": manifest_sha,
            "formal_test_names_inert": list(formal_names),
            "within_dev_sessions_opened": False,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
            "n_train_windows": len(train_dataset.window_indices),
            "audit_windows_excluded": False,
            "audit_note": "audit exclusion was pilot-only; official arms train on the full strict-27 window set",
            "sampler": "mc_maze SessionBatchSampler(batch=32, shuffle, seed=42); identical constructor for A/B/C",
            "steps_per_epoch": steps_per_epoch,
            "normalizers": {
                "behavior_semantic_sha256": behavior_semantic,
                "side_feature_semantic_sha256": side_semantic,
            },
            "unit_order": "datamodule channel order (sorted units, units<100)",
        },
        "z4_authority": z4_authority,
        "t4_authority_sha256": t4_authority,
        "swa_window_t4_phase_local_epochs": sorted(swa_local_epochs),
        "source_closure": closure_launch,
        "authority_sha256": authorities,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }

    if args.preflight_only:
        receipt_mod.write_receipt_transactionally(
            Path(args.output_root) / f"preflight_arm{args.arm}.json",
            {
                "schema": "tfpd_admission_arm_v1_preflight",
                "status": "ARM_PREFLIGHT_PASSED",
                **preflight_body,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        print(json.dumps({"arm": args.arm, "status": "ARM_PREFLIGHT_PASSED",
                          "steps_per_epoch": steps_per_epoch,
                          "t_pre": t_pre, "e_t4": e_t4}, indent=1))
        return 0

    receipt_mod.write_receipt_transactionally(
        out_dir / "launch_receipt.json",
        {
            "schema": "tfpd_admission_arm_v1_launch",
            "status": "ARM_LAUNCHED",
            **preflight_body,
        },
    )

    # ---- training ----------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    diagnostics = []
    checkpoints = []
    transition_record = None
    invariant_failures = []
    global_epoch = 0
    phase_local_step = 0
    phase_total_steps = 0

    for phase_index, phase in enumerate(plan["phases"]):
        visible_mode = phase["visible_side"]
        lr_fn = None
        if phase["lr_schedule"] == "warmup_then_cosine":
            n_epochs_phase = phase["epochs"]
            lr_fn = (
                lambda step, n=n_epochs_phase: arm_common.lr_at_step(
                    step, n, steps_per_epoch
                )
            )
        if phase_index > 0:
            # ---- atomic admission transition (§6.3) -------------------------
            state_sha_before = arm_common.state_sha256(model)
            old_optimizer_sha = arm_common.optimizer_sha256(optimizer)
            phase1_path = out_dir / "phase1_final.pt"
            if not phase1_path.exists():
                torch.save(
                    {
                        "kind": "tfpd_admission_armB_phase1_final_v1",
                        "state_dict": model.state_dict(),
                        "state_dict_sha256": state_sha_before,
                        "z4_epochs_completed": plan["phases"][0]["epochs"],
                        "optimizer_state_sha256": old_optimizer_sha,
                    },
                    phase1_path,
                )
                phase1_sha = seal_file(phase1_path)
            else:
                phase1_sha = arm_common.sha256_file(phase1_path)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=arm_common.ADAM_CONSTRUCTOR["lr"],
                betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
                eps=arm_common.ADAM_CONSTRUCTOR["eps"],
                weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
                amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
            )
            state_sha_after = arm_common.state_sha256(model)
            w_side_now = arm_common.w_side_block(model)
            w_side_zero_before_t4 = (
                int(torch.count_nonzero(w_side_now).item()) == 0
                and not bool(w_side_now.signbit().any().item())
            )
            with torch.no_grad():
                model.eval()
                visible_probe = arm_common.admit_side(fx_side_t4, "t4")
                probe_pred, _ = model(
                    fx_neural[:4], calib_trials=fx_calib[:4], side_features=visible_probe[:4]
                )
                first_t4_forward_finite = bool(
                    torch.isfinite(probe_pred).all().item()
                )
                model.train()
            phase_local_step = 0
            phase_total_steps = 0
            transition_record = {
                "z4_epochs_completed": plan["phases"][0]["epochs"],
                "phase1_checkpoint": str(phase1_path),
                "phase1_checkpoint_sha256": phase1_sha,
                "phase1_state_sha256": state_sha_before,
                "phase1_optimizer_state_sha256": old_optimizer_sha,
                "switch": "z4 -> t4 on the aligned normalized T4 tensor (post-normalization admission)",
                "fresh_adam_constructor": arm_common.ADAM_CONSTRUCTOR,
                "fresh_optimizer_state_empty": len(optimizer.state) == 0,
                "no_weight_reset_state_sha_equal": state_sha_before == state_sha_after,
                "state_sha256_at_transition": state_sha_after,
                "first_t4_forward_finite": first_t4_forward_finite,
                "w_side_exact_zero_before_first_t4_step": w_side_zero_before_t4,
                "t4_authority_sha_unchanged_at_transition": (
                    arm_common.t4_authority_fingerprint(train_dataset.sessions) == t4_authority
                ),
            }
            if not (
                transition_record["no_weight_reset_state_sha_equal"]
                and first_t4_forward_finite
                and w_side_zero_before_t4
                and transition_record["fresh_optimizer_state_empty"]
            ):
                raise SystemExit("admission transition invariant failed")

        for local_epoch in range(phase["epochs"]):
            epoch_started = time.time()
            stats = train_epoch(
                model,
                optimizer,
                loader,
                visible_mode,
                lr_fn,
                device,
                phase_local_step,
                max_steps=args.max_train_steps,
            )
            steps_this_epoch = stats["optimizer_steps"]
            phase_local_step += steps_this_epoch
            phase_total_steps += steps_this_epoch
            schedule = schedules[phase["phase"]]

            w_side = arm_common.w_side_block(model)
            w_zero_mag = int(torch.count_nonzero(w_side).item()) == 0
            w_positive_zero = not bool(w_side.signbit().any().item())
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
                contribution = arm_common.post_pool_contribution(
                    model, fx_calib[:8], fx_side_t4[:8]
                )
                attention = arm_common.attention_summary(
                    model, fx_neural[:4], fx_calib[:4],
                    arm_common.admit_side(fx_side_t4[:4], visible_mode),
                )
            authority_ok = (
                arm_common.t4_authority_fingerprint(train_dataset.sessions) == t4_authority
            )

            alpha = phase["alpha"]
            diag = {
                "global_epoch": global_epoch,
                "phase_label": phase["phase"],
                "phase_local_epoch": local_epoch,
                "visible_side_mode": visible_mode,
                "alpha": alpha,
                "duration_s": round(time.time() - epoch_started, 3),
                **stats,
                "optimizer_steps_phase_total": phase_total_steps,
                "lr_expected_first": schedule.get("lr_at_step_0", schedule.get("lr")),
                "lr_expected_last": schedule.get("lr_at_final_step", schedule.get("lr")),
                "clipping_authorized": False,
                "w_side_norm": float(w_side.norm().item()),
                "norm_alpha_times_w_side": float(w_side.norm().item()) if alpha else 0.0,
                "w_side_exact_zero_magnitude": w_zero_mag,
                "w_side_positive_zero_bitwise": w_positive_zero,
                "w_side_exp_avg_norm": (
                    0.0 if exp_avg is None else float(exp_avg.norm().item())
                ),
                "w_side_exp_avg_sq_norm": (
                    0.0 if exp_avg_sq is None else float(exp_avg_sq.norm().item())
                ),
                "w_side_moments_exact_zero_magnitude": moments_zero,
                **contribution,
                **attention,
                "t4_authority_unchanged": authority_ok,
                "parameters_finite": params_finite,
                "optimizer_state_finite": optimizer_state_finite(optimizer),
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
            ]
            if visible_mode == "z4":
                checks += [
                    w_zero_mag,
                    w_positive_zero,
                    moments_zero,
                    stats["w_side_grad_exact_zero_all_steps"],
                    stats["lr_first"] == arm_common.PHASE1_LR,
                    abs(diag["norm_w_side_times_t4_fixed_batch"]) == 0.0,
                ]
            else:
                checks += [steps_this_epoch in (steps_per_epoch, args.max_train_steps or 0)]
            if not all(checks):
                invariant_failures.append(global_epoch)

            save_here = visible_mode == "t4" and local_epoch in swa_local_epochs
            if save_here:
                ckpt_name = f"phase{phase_index}_epoch{local_epoch:03d}.ckpt"
                ckpt_path = out_dir / ckpt_name
                torch.save(
                    {
                        "kind": "tfpd_admission_arm_ckpt_v1",
                        "arm": args.arm,
                        "phase_label": phase["phase"],
                        "phase_local_epoch": local_epoch,
                        "global_epoch": global_epoch,
                        "state_dict": model.state_dict(),
                        "state_dict_sha256": diag["state_dict_sha256"],
                        "optimizer_state_sha256": diag["optimizer_state_sha256"],
                    },
                    ckpt_path,
                )
                ckpt_sha = seal_file(ckpt_path)
                diag["checkpoint_saved"] = ckpt_name
                checkpoints.append(
                    {
                        "file": ckpt_name,
                        "phase_local_epoch": local_epoch,
                        "global_epoch": global_epoch,
                        "sha256": ckpt_sha,
                    }
                )
            diagnostics.append(diag)
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "global_epoch": global_epoch,
                        "phase": phase["phase"],
                        "loss": round(stats["train_loss_mean_per_step"], 6),
                        "lr_last": stats["lr_last"],
                        "w_side_norm": diag["w_side_norm"],
                        "ratio_t4_calib": round(
                            diag["ratio_t4_to_calibration_at_post_pool0"], 6
                        ),
                    }
                ),
                flush=True,
            )
            global_epoch += 1

    # ---- final-four SWA + finite forward smoke ------------------------------
    if len(checkpoints) != 4:
        raise SystemExit(f"expected exactly four SWA-window checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]
    swa_model = spintshape.build_spintshape_model(seed=args.seed)
    swa_model.load_state_dict(swa_state, strict=True)
    swa_model.to(device).eval()
    with torch.no_grad():
        swa_pred, _ = swa_model(
            fx_neural[:4],
            calib_trials=fx_calib[:4],
            side_features=arm_common.admit_side(fx_side_t4[:4], "t4"),
        )
        swa_forward_finite = bool(torch.isfinite(swa_pred).all().item())
    del swa_model

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure_launch["closure_sha256"]
    n_epochs_expected = sum(p["epochs"] for p in plan["phases"])
    epochs_run = len(diagnostics)

    terminal = {
        "schema": "tfpd_admission_arm_v1",
        "status": (
            ("ARM_SMOKE_COMPLETE__NON_AUTHORITATIVE" if args.smoke else "ARM_TERMINAL")
            if not invariant_failures and closure_equal and epochs_run == n_epochs_expected
            else ("ARM_SMOKE_INVARIANT_FAILURE" if args.smoke else "ARM_INVARIANT_OR_CLOSURE_FAILURE")
        ),
        "arm": args.arm,
        "arm_name": ARM_NAMES[args.arm],
        "smoke": args.smoke,
        "max_train_steps": args.max_train_steps,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary_provenance": boundary,
        "arm_plan": plan,
        "b_phase2_vs_c_equality": bc_equality,
        "lr_schedules": schedules,
        "initial_state": preflight_body["initial_state"],
        "data_contract": preflight_body["data_contract"],
        "z4_authority": z4_authority,
        "t4_authority_sha256": t4_authority,
        "transition": transition_record,
        "epochs_run": epochs_run,
        "epochs_expected": n_epochs_expected,
        "diagnostics_per_epoch": diagnostics,
        "invariant_failures": invariant_failures,
        "checkpoints": checkpoints,
        "swa": {
            "path": str(swa_path),
            "sha256": swa_sha,
            "window_phase_local_epochs": [c["phase_local_epoch"] for c in checkpoints],
            "manifest": swa_manifest,
            "strict_reload_finite_forward_smoke": swa_forward_finite,
        },
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "optimizer": arm_common.ADAM_CONSTRUCTOR,
            "weight_decay": 0.0,
            "clipping": "none authorized, none applied",
            "no_arm_specific_regularization_loss_weight_batch_size_or_augmentation": True,
            "dev_session_validation_early_stopping_or_checkpoint_selection_during_fitting": False,
            "checkpoint_policy": "predeclared final-four T4-phase window only; no selection",
            "formal_or_organizer_held_data_opened": False,
            "external_sub_m_opened": False,
            "within_dev_sessions_opened": False,
            "target_updates_gradients_or_optimizer_steps_during_scoring": 0,
            "swa_forward_smoke_target_updates": 0,
            "w_side_and_moment_invariants": (
                "phase 1 (arm B): W_side, exp_avg, exp_avg_sq elementwise exactly zero every epoch; "
                "verified bitwise and fail-closed"
            ),
            "admission_order": "raw -> canonical T4 builder -> source-only normalizer -> aligned normalized T4 -> admission (zeros_like/Z4 or identity/T4); never alpha*raw_T4",
            "phase_and_visible_side_mode_per_epoch": [
                {"global_epoch": d["global_epoch"], "phase": d["phase_label"], "mode": d["visible_side_mode"]}
                for d in diagnostics
            ],
        },
        "source_closure": {
            "launch": closure_launch,
            "final": closure_final,
            "launch_final_closure_equal": closure_equal,
        },
        "authority_sha256": authorities,
        "environment": preflight_body["environment"],
    }
    receipt_mod.write_receipt_transactionally(out_dir / "terminal_receipt.json", terminal)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "status": terminal["status"],
                "epochs_run": epochs_run,
                "w_side_norm_final": diagnostics[-1]["w_side_norm"],
                "swa_sha256": swa_sha[:16],
            },
            indent=1,
        )
    )
    return 0 if terminal["status"] in ("ARM_TERMINAL", "ARM_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
