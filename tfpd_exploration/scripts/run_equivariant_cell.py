#!/usr/bin/env python3
"""Route-owned SO(2)-equivariance cells (family B): B rotation augmentation,
C equivariant decoder.

The Arm A/D training contract VERBATIM (canonical initial tensor bytes strict
load where the graph allows it, strict-27 full window set, 33,925 steps/epoch,
batch-32 seed-42 session sampler, warmup 1e-5 -> 1e-4 over two epochs then
cosine to 1e-6, 48 epochs, fresh Adam, no clipping, predeclared final-four
SWA, launch/terminal receipts with closure equality, seed 42, M30 calibration,
B3S encoder, whole-unit dynamic dropout law unchanged) with ONE factor:

- ``--cell B_augmentation``: the sealed Cell D graph (2 heads, dynamic
  dropout) strict-loads the canonical initial state bitwise; the ONE factor is
  training-time joint SO(2) rotation augmentation — one theta ~ U(-pi, pi) per
  batch (numpy PCG64(42010) dedicated namespace, SHA-recorded), shared across
  the batch, rotating the normalized T4 side columns (a,c) and the
  standardized behavior labels jointly; activity/calibration/(m,b) invariant;
  never applied at eval (eval forward is the plain Cell-D path).

- ``--cell C_equivariant``: the equivariant consumer of the carrier replaces
  the ordinary fused-token consumer.  Every canonical tensor is strict-loaded
  bitwise (the ordinary consumer modules stay present, frozen, disconnected —
  disclosed); the consumer head is built in its own seeded namespace; the
  fused path receives the canonical Z4, so the carrier enters ONLY through the
  rotation-covariant complex-unit stream and the equivariance property
  f(x, R*beta) = R * f(x, beta) is structural.  Parameter count differs from
  canonical — a DISCLOSED family-B deviation.  Parity claims proven at launch:
  identity rotation is the model's own bitwise baseline, and max equivariance
  violation over seeded random rotations <= 1e-5 (fail-closed).

Arm A of the comparison is the SEALED Cell D
(results/pop_robust_v1/cellD_2heads_dynamic_dropout) — no run needed.  The
pre-registered gates (C - A external governing >= +0.03 AND >= 10/15
positive; also report C - B and B - A; both granularities, date blocks,
paired session statistics) live in the launch and terminal receipts.
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
WINDOW_SIZE = 50
BOUND_PATTERNS = (
    "src/tfpd_lane/equivariant_cell.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/pop_robust.py",
    "scripts/run_equivariant_cell.py",
    "scripts/run_admission_arm.py",
)
CELL_NAMES = {
    "B_augmentation": "cellB_rotation_augmentation",
    "C_equivariant": "cellC_equivariant",
}
THETA_AUTHORITY = ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seal_artifact(path: Path, sha256_of) -> str:
    """0444 + sidecar, the theta-authority sealing law."""
    import stat

    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    digest = sha256_of(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "  " + path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return digest


# ---- cell C: the raw carrier authority --------------------------------------
def ensure_carrier_authority(equivariant, receipt_mod, arm_runner, a2, dm,
                             authority_root: Path, theta_authority_path: Path,
                             context: dict) -> dict:
    """Load-or-build the sealed raw-T4 carrier authority (theta-authority rule).

    The build is a deterministic, training-only derivation: the same
    ``compute_unit_side_features_uncached`` call the sealed theta authority
    used, plus a per-session ``raw_t4_sha256`` binding that must EQUAL the
    theta authority's value, plus a canonical-unit-order alignment proof and a
    normalized-side roundtrip check against the datamodule.
    """
    import mc_maze.a2_matched_subject_shift_v2_core as a2core

    artifact = authority_root / "carrier_authority.pt"
    receipt_path = authority_root / "carrier_authority_receipt.json"
    theta_sidecar = Path(str(theta_authority_path) + ".sha256")
    if not theta_authority_path.is_file() or not theta_sidecar.is_file():
        raise SystemExit("theta authority artifact missing (binding required)")
    theta_sha = receipt_mod.sha256_file(theta_authority_path)
    if theta_sha != theta_sidecar.read_text().split()[0]:
        raise SystemExit("theta authority SHA mismatch against sidecar")
    theta_payload = torch.load(theta_authority_path, map_location="cpu",
                               weights_only=False)
    if theta_payload["kind"] != "tfpd_sparsification_theta_authority_v1":
        raise SystemExit("theta authority kind drift")

    if artifact.is_file():
        sidecar = Path(str(artifact) + ".sha256")
        if not sidecar.is_file():
            raise SystemExit("carrier authority sidecar missing")
        if receipt_mod.sha256_file(artifact) != sidecar.read_text().split()[0]:
            raise SystemExit("carrier authority SHA mismatch against sidecar")
        if not receipt_path.is_file():
            raise SystemExit("carrier authority receipt missing")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "CARRIER_AUTHORITY_SEALED":
            raise SystemExit("carrier authority receipt not sealed")
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        if payload["kind"] != equivariant.CARRIER_AUTHORITY_KIND:
            raise SystemExit("carrier authority kind drift")
        if payload["authority_sha256"] != receipt["authority_sha256"]:
            raise SystemExit("carrier authority receipt/artifact SHA mismatch")
        # re-verify the theta binding against the LIVE theta artifact
        for name, entry in sorted(payload["authority"].items()):
            theta_entry = theta_payload["authority"].get(name)
            if theta_entry is None or entry["raw_t4_sha256"] != theta_entry["raw_t4_sha256"]:
                raise SystemExit(f"carrier/theta raw-T4 SHA binding failed at {name}")
        return {"path": str(artifact), "sha256": sidecar.read_text().split()[0],
                "receipt": receipt_path.name, "built_this_run": False,
                "theta_authority": {"path": str(theta_authority_path),
                                    "sha256": theta_sha,
                                    "raw_t4_sha_binding_verified": True},
                "m_scale": float(payload["m_scale"])}

    if authority_root.exists():
        raise SystemExit(
            f"carrier authority root exists without a sealed artifact: {authority_root}"
        )
    authority_root.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    train_paths, _val, _test = a2core.active_source_session_paths()
    authority = equivariant.build_carrier_authority(train_paths)

    # theta-authority raw-byte binding (fail-closed)
    for name, entry in sorted(authority.items()):
        theta_entry = theta_payload["authority"].get(name)
        if theta_entry is None:
            raise SystemExit(f"theta authority missing session {name}")
        if entry["raw_t4_sha256"] != theta_entry["raw_t4_sha256"]:
            raise SystemExit(f"raw T4 bytes differ from the theta authority at {name}")

    # alignment + normalized-side roundtrip proof against the datamodule
    mean_side, std_side = dm._side_feature_stats
    alignment = {}
    for name, entry in sorted(authority.items()):
        record = dm.train_dataset.sessions[name]
        raw32 = entry["raw"].astype(np.float32)
        renorm = ((raw32 - mean_side) / std_side).astype(np.float32)
        side_delta = float(np.abs(renorm - record.side_features).max())
        alignment[name] = {
            "authority_units": entry["n_units"],
            "datamodule_channels": int(record.neural.shape[1]),
            "side_rows": int(record.side_features.shape[0]),
            "aligned": bool(
                entry["n_units"] == record.neural.shape[1]
                == record.side_features.shape[0]
            ),
            "normalized_side_roundtrip_max_abs_delta": side_delta,
            "n_valid_directions": int(entry["valid"].sum()),
            "n_undefined": int((~entry["valid"]).sum()),
        }
        if not alignment[name]["aligned"]:
            raise SystemExit(f"carrier authority failed alignment at {name}")
        if side_delta > 1e-6:
            raise SystemExit(f"carrier authority normalized roundtrip drift at {name}")

    m_scale = equivariant.global_m_scale(authority)
    payload_out = {
        "kind": equivariant.CARRIER_AUTHORITY_KIND,
        "authority": {
            name: {
                "raw": torch.from_numpy(entry["raw"]),
                "valid": torch.from_numpy(entry["valid"]),
                "n_units": entry["n_units"],
                "raw_t4_sha256": entry["raw_t4_sha256"],
            }
            for name, entry in sorted(authority.items())
        },
        "authority_sha256": equivariant.authority_sha256(authority),
        "m_scale": m_scale,
        "rule": (
            "raw T4 = compute_unit_side_features_uncached(feature_group='t4', "
            "pool_size=30, bin 20ms, window 50, 'R', 'sua'); beta=(a,c) raw, "
            "never z-scored; valid = raw m > MODULATION_EPS (theta-authority "
            "law); per-session raw_t4_sha256 must equal the sealed theta "
            "authority's"
        ),
        "m_scale_rule": (
            "frozen rotation-invariant global scalar = mean raw |beta| over "
            "all units of all strict-27 sessions"
        ),
    }
    torch.save(payload_out, artifact)
    artifact_sha = _seal_artifact(artifact, receipt_mod.sha256_file)
    receipt_mod.write_receipt_transactionally(
        receipt_path,
        {
            "schema": equivariant.CARRIER_AUTHORITY_KIND,
            "status": "CARRIER_AUTHORITY_SEALED",
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifact": {"path": str(artifact), "sha256": artifact_sha},
            "authority_sha256": payload_out["authority_sha256"],
            "rule": payload_out["rule"],
            "m_scale": m_scale,
            "m_scale_rule": payload_out["m_scale_rule"],
            "n_sessions": len(authority),
            "alignment_proof": alignment,
            "built_during": context,
            "theta_authority": {"path": str(theta_authority_path),
                                "sha256": theta_sha,
                                "raw_t4_sha_binding_verified": True},
            "disclosures": {
                "z_scored_values_used": False,
                "within_dev_or_external_opened": False,
                "formal_or_organizer_held_data_opened": False,
                "training_only_derivation": True,
            },
            "environment": {"no_user_site": bool(sys.flags.no_user_site)},
        },
    )
    return {"path": str(artifact), "sha256": artifact_sha,
            "receipt": receipt_path.name, "built_this_run": True,
            "theta_authority": {"path": str(theta_authority_path),
                                "sha256": theta_sha,
                                "raw_t4_sha_binding_verified": True},
            "m_scale": m_scale}


# ---- training epochs --------------------------------------------------------
def _grad_sq(params, device):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros((), device=device)
    return torch.stack(torch._foreach_norm(grads)).pow(2).sum()


def train_epoch_rotation_augmented(model, optimizer, loader, lr_fn, device,
                                   rotation_stream, phase_step, num_heads,
                                   max_steps=None):
    """Arm B: the exact Cell-D training step with the ONE augmentation factor."""
    model.train()
    arm_common = sys.modules["tfpd_lane_arm_common"]
    equivariant = sys.modules["tfpd_lane_equivariant"]
    encoder_params, decoder_params = arm_common.param_groups_by_branch(model)
    acc = {
        "loss_sum": torch.zeros((), device=device),
        "nonfinite_loss": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_grad": torch.zeros((), device=device, dtype=torch.long),
        "grad_sq_encoder": torch.zeros((), device=device),
        "grad_sq_decoder": torch.zeros((), device=device),
    }
    steps = examples = 0
    for batch in loader:
        neural, behavior, calib, sessions, side = batch[:5]
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)
        theta = rotation_stream.next()
        rot_side, rot_behavior = equivariant.rotate_side_behavior(
            side, behavior, theta, pad_value=PAD_VALUE
        )
        lr = lr_fn(phase_step + steps)
        optimizer.param_groups[0]["lr"] = float(lr)
        prediction, _identity = model(neural, calib_trials=calib,
                                      side_features=rot_side)
        valid_rows = (rot_behavior != PAD_VALUE).all(dim=-1)
        diff2 = ((prediction - rot_behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid_rows).sum() / (valid_rows.sum() * rot_behavior.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            acc["grad_sq_encoder"] += _grad_sq(encoder_params, device)
            acc["grad_sq_decoder"] += _grad_sq(decoder_params, device)
            total_sq = acc["grad_sq_encoder"] + acc["grad_sq_decoder"]
            acc["nonfinite_grad"] += (~torch.isfinite(total_sq)).long()
            acc["nonfinite_loss"] += (~torch.isfinite(loss)).long()
        optimizer.step()
        acc["loss_sum"] += loss.detach()
        steps += 1
        examples += int(neural.shape[0])
        if max_steps is not None and steps >= max_steps:
            break
    n = max(steps, 1)
    return {
        "optimizer_steps": steps,
        "train_example_windows": examples,
        "train_loss_mean_per_step": float(acc["loss_sum"].item()) / n,
        "nonfinite_loss_steps": int(acc["nonfinite_loss"].item()),
        "nonfinite_grad_steps": int(acc["nonfinite_grad"].item()),
        "grad_norm_encoder_mean": float((acc["grad_sq_encoder"] / n).sqrt().item()),
        "grad_norm_decoder_mean": float((acc["grad_sq_decoder"] / n).sqrt().item()),
        "num_heads": num_heads,
        "augmented_batches": steps,
    }


def train_epoch_equivariant(model, optimizer, loader, lr_fn, device, bundles,
                            phase_step, num_heads, max_steps=None):
    """Arm C: the exact Cell-D training step on the equivariant decode path."""
    model.train()
    arm_common = sys.modules["tfpd_lane_arm_common"]
    encoder_params, decoder_params = arm_common.param_groups_by_branch(model)
    consumer_params = [p for p in model.consumer.parameters() if p.requires_grad]
    w_param = model.id_encoder.post_pool[0].weight
    encoder = model.id_encoder
    hidden, side_dim = encoder.hidden_dim, encoder.side_dim
    acc = {
        "loss_sum": torch.zeros((), device=device),
        "nonfinite_loss": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_grad": torch.zeros((), device=device, dtype=torch.long),
        "grad_sq_encoder": torch.zeros((), device=device),
        "grad_sq_decoder": torch.zeros((), device=device),
        "grad_sq_consumer": torch.zeros((), device=device),
        "wside_grad_nonzero": torch.zeros((), device=device, dtype=torch.long),
    }
    steps = examples = 0
    for batch in loader:
        neural, behavior, calib, sessions, side = batch[:5]
        session = sessions[0] if isinstance(sessions, (list, tuple)) else sessions
        model.set_carrier(bundles[session])
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
            acc["grad_sq_encoder"] += _grad_sq(encoder_params, device)
            acc["grad_sq_decoder"] += _grad_sq(decoder_params, device)
            acc["grad_sq_consumer"] += _grad_sq(consumer_params, device)
            total_sq = (
                acc["grad_sq_encoder"] + acc["grad_sq_decoder"] + acc["grad_sq_consumer"]
            )
            acc["nonfinite_grad"] += (~torch.isfinite(total_sq)).long()
            grad = w_param.grad
            if grad is not None:
                acc["wside_grad_nonzero"] += (grad[:, hidden:hidden + side_dim] != 0).sum()
            acc["nonfinite_loss"] += (~torch.isfinite(loss)).long()
        optimizer.step()
        acc["loss_sum"] += loss.detach()
        steps += 1
        examples += int(neural.shape[0])
        if max_steps is not None and steps >= max_steps:
            break
    n = max(steps, 1)
    return {
        "optimizer_steps": steps,
        "train_example_windows": examples,
        "train_loss_mean_per_step": float(acc["loss_sum"].item()) / n,
        "nonfinite_loss_steps": int(acc["nonfinite_loss"].item()),
        "nonfinite_grad_steps": int(acc["nonfinite_grad"].item()),
        "grad_norm_encoder_mean": float((acc["grad_sq_encoder"] / n).sqrt().item()),
        "grad_norm_decoder_mean": float((acc["grad_sq_decoder"] / n).sqrt().item()),
        "grad_norm_consumer_mean": float((acc["grad_sq_consumer"] / n).sqrt().item()),
        "w_side_grad_exact_zero_all_steps": int(acc["wside_grad_nonzero"].item()) == 0,
        "num_heads": num_heads,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=list(CELL_NAMES))
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
    parser.add_argument("--authority-root", type=Path,
                        default=ROOT / "results/equivariant_v1/carrier_authority")
    parser.add_argument("--theta-authority", type=Path, default=THETA_AUTHORITY)
    parser.add_argument("--launch-probe-rotations", type=int, default=8)
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
                "schema": "tfpd_equivariant_cell_v1",
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
    equivariant = _load_module(
        "tfpd_lane_equivariant", ROOT / "src/tfpd_lane/equivariant_cell.py"
    )
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    pop_robust = _load_module("tfpd_lane_pop_robust", ROOT / "src/tfpd_lane/pop_robust.py")
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
    side_semantic = a2.normalizer_value_sha256(*dm._side_feature_stats)
    t4_authority = arm_common.t4_authority_fingerprint(train_dataset.sessions)

    pl.seed_everything(args.seed, workers=True)

    # ---- model + graph proofs ---------------------------------------------
    carrier_authority_info = None
    z4_authority = None
    if args.cell == "B_augmentation":
        model = pop_robust.build_population_robustness_model(seed=args.seed, cell="D")
        model.load_state_dict(canonical_state, strict=True)
        state_sha_loaded = arm_common.state_sha256(model)
        if state_sha_loaded != initial_payload["state_sha256"]:
            raise SystemExit("loaded state SHA != canonical artifact state SHA")
        shapes = {
            k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter)
                else tuple(v.shape))
            for k, v in model.state_dict().items()
        }
        canonical_shapes = {
            k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter)
                else tuple(v.shape))
            for k, v in canonical_state.items()
        }
        if shapes != canonical_shapes:
            raise SystemExit("graph parity failure vs canonical initial state")
        if model.decoder.transformer.layers[0].cross_attn.num_heads != 2:
            raise SystemExit("sealed Cell D head-count drift")
        if not model.decoder.dynamic_dropout:
            raise SystemExit("sealed Cell D dynamic-dropout flag drift")
        initial_state_block = {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "loaded_state_sha256": state_sha_loaded,
            "strict_load": True, "graph_parity_vs_canonical": True,
            "bitwise_parity_at_init_vs_cell_d": True,
        }
        param_proof = pop_robust.trainable_parameter_count(model)
    else:
        model = equivariant.build_equivariant_model(seed=args.seed)
        prefix_proof = equivariant.load_canonical_prefix(model, canonical_state)
        if prefix_proof["canonical_subset_sha256"] != initial_payload["state_sha256"]:
            raise SystemExit("canonical-prefix subset SHA != canonical artifact SHA")
        freeze = equivariant.freeze_inactive_consumer_modules(model)
        accounting = equivariant.parameter_accounting(model)
        carrier_authority_info = ensure_carrier_authority(
            equivariant, receipt_mod, arm_runner, a2, dm,
            Path(args.authority_root), Path(args.theta_authority),
            context={"cell": args.cell, "smoke": args.smoke},
        )
        authority_payload = torch.load(carrier_authority_info["path"],
                                       map_location="cpu", weights_only=False)
        authority_np = {
            name: {
                "raw": entry["raw"].numpy(),
                "valid": entry["valid"].numpy(),
                "n_units": int(entry["n_units"]),
                "raw_t4_sha256": entry["raw_t4_sha256"],
                "beta_norm": np.hypot(entry["raw"][:, 0].numpy(),
                                      entry["raw"][:, 1].numpy()),
            }
            for name, entry in authority_payload["authority"].items()
        }
        missing = set(dm.session_splits["train"]) - set(authority_np)
        if missing:
            raise SystemExit(f"carrier authority missing sessions: {sorted(missing)}")
        bundles = equivariant.build_carrier_bundles(
            authority_np, train_dataset.sessions, carrier_authority_info["m_scale"]
        )
        initial_state_block = {
            "path": str(args.initial_state), "artifact_sha256": initial_sha,
            "state_dict_sha256": initial_payload["state_sha256"],
            "canonical_prefix_proof": prefix_proof,
            "inactive_modules_frozen": freeze,
            "bitwise_parity_at_init_vs_cell_d": False,
            "disclosed": (
                "family-B new-architecture cell: canonical tensors bitwise "
                "strict-loaded, consumer head newly seeded (parameter count "
                "differs from canonical — disclosed deviation)"
            ),
        }
        param_proof = accounting
        z4_authority = arm_runner.verify_z4_is_zeros_like_t4(dm, a2.SOURCE_CACHE_ROOT)
        if not (z4_authority["all_bitwise_equal"]
                and z4_authority["all_exact_positive_zero"]):
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
    fx_behavior = _fx_beh.to(device)

    optimizer_params = (
        model.parameters()
        if args.cell == "B_augmentation"
        else (p for p in model.parameters() if p.requires_grad)
    )
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=arm_common.ADAM_CONSTRUCTOR["lr"],
        betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
        eps=arm_common.ADAM_CONSTRUCTOR["eps"],
        weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
        amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
    )
    schedule = arm_common.schedule_params(epochs, steps_per_epoch)
    lr_fn = lambda step: arm_common.lr_at_step(step, epochs, steps_per_epoch)

    # ---- launch parity proofs ----------------------------------------------
    launch_proofs = {}
    if args.cell == "B_augmentation":
        # plain (unaugmented) eval forward == the parent decode path, bitwise
        model.eval()
        with torch.no_grad():
            mine, _ = model(fx_neural[:4], calib_trials=fx_calib[:4],
                            side_features=fx_side[:4])
            identity = model.compute_identity(fx_calib[:4], side_features=fx_side[:4])
            src = fx_neural[:4].permute(0, 2, 1) + identity
            rep_src = model.decoder.fc_in(src)
            rep_q = model.decoder.fc_in(model.decoder.rep).to(rep_src)
            out, _ = model.decoder.transformer(rep_q.repeat(rep_src.size(0), 1, 1), rep_src)
            parent = model.decoder.fc_out(out).permute(0, 2, 1)
        if not torch.equal(mine, parent):
            raise SystemExit("arm B eval-path parity failure vs the Cell-D parent path")
        launch_proofs["eval_path_bitwise_parent_equal"] = True
        # theta == 0: bitwise own baseline for the augmentation
        side0, beh0 = equivariant.rotate_side_behavior(
            fx_side, fx_behavior, 0.0, pad_value=PAD_VALUE)
        if not (torch.equal(side0, fx_side) and torch.equal(beh0, fx_behavior)):
            raise SystemExit("arm B identity rotation is not a bitwise own baseline")
        launch_proofs["identity_rotation_bitwise_baseline"] = True
        # a nonzero rotation moves (a,c) and the labels jointly, nothing else
        valid_fx = (fx_behavior != PAD_VALUE).all(dim=-1)
        theta_check = 0.7
        side_r, beh_r = equivariant.rotate_side_behavior(
            fx_side, fx_behavior, theta_check, pad_value=PAD_VALUE)
        rot = torch.from_numpy(equivariant.rotation_matrix_2d(theta_check)).to(
            device=fx_side.device, dtype=fx_side.dtype)
        launch_proofs["augmentation_joint_rotation_check"] = {
            "theta": theta_check,
            "side_ac_rotated": bool(
                torch.allclose(side_r[..., 0:2], fx_side[..., 0:2] @ rot.t(), atol=1e-6)
            ),
            "side_mb_bitwise_unchanged": bool(
                torch.equal(side_r[..., 2:4], fx_side[..., 2:4])
            ),
            "behavior_valid_rows_rotated": bool(
                torch.allclose(
                    beh_r[..., 0:2][valid_fx],
                    fx_behavior[..., 0:2][valid_fx] @ rot.t(),
                    atol=1e-6,
                )
            ),
            "pad_rows_restored_bitwise": bool(
                torch.equal(
                    beh_r[..., 0:2][~valid_fx], fx_behavior[..., 0:2][~valid_fx]
                ),
            ),
        }
        integrity = equivariant.arm_b_integrity_block(
            num_heads=2, total_parameters=param_proof,
            canonical_parameters=pop_robust.trainable_parameter_count(model),
        )
        rotation_stream = equivariant.RotationStream()
    else:
        model.set_carrier(bundles[fx_session])
        thetas = [0.0] + equivariant.probe_thetas(args.launch_probe_rotations)
        probe = equivariant.equivariance_probe(
            model, fx_neural[:4], fx_calib[:4], fx_side[:4], thetas)
        if not probe["identity_rotation_bitwise_equal"]:
            raise SystemExit("arm C identity rotation is not a bitwise own baseline")
        if probe["max_violation"] > equivariant.EQUIVARIANCE_TOLERANCE:
            raise SystemExit(
                f"arm C equivariance launch probe failed: {probe['max_violation']}"
            )
        launch_proofs["equivariance_probe"] = probe
        launch_proofs["identity_rotation_bitwise_baseline"] = True
        # carrier blindness of the fused path: changing the side (a,c) columns
        # cannot move the output (they are zeroed before the B3S encoder)
        alt_side = fx_side.clone()
        alt_side[..., 0:2] = torch.randn_like(alt_side[..., 0:2]) * 5.0
        model.eval()
        with torch.no_grad():
            base_out, _ = model(fx_neural[:2], calib_trials=fx_calib[:2],
                                side_features=fx_side[:2])
            alt_out, _ = model(fx_neural[:2], calib_trials=fx_calib[:2],
                               side_features=alt_side[:2])
        if not torch.equal(base_out, alt_out):
            raise SystemExit("fused path is not carrier-blind (side (a,c) leaked)")
        launch_proofs["fused_path_carrier_blind"] = True
        # the attention weights are invariant: bitwise unchanged under rotation
        with torch.no_grad():
            identity_z4 = model.compute_identity(
                fx_calib[:4], side_features=torch.zeros_like(fx_side[:4]))
            src = fx_neural[:4].permute(0, 2, 1) + identity_z4
            tokens = model.decoder.fc_in(src)
            queries = model.decoder.fc_in(model.decoder.rep).to(tokens)
            queries = queries.repeat(tokens.size(0), 1, 1)
            w_base = model.consumer.attention_weights(tokens, queries, model.carrier)
            w_rot = model.consumer.attention_weights(
                tokens, queries, model.carrier.rotated(1.234))
        weight_delta = float((w_base - w_rot).abs().max().item())
        if weight_delta > 1e-6:
            raise SystemExit(
                f"attention weights are not rotation-invariant: {weight_delta}"
            )
        launch_proofs["attention_weights_rotation_invariant"] = {
            "max_abs_delta": weight_delta,
            "bitwise_equal": bool(torch.equal(w_base, w_rot)),
        }
        integrity = equivariant.arm_c_integrity_block(
            num_heads=2,
            canonical_parameters=accounting["canonical_tensors_strict_loaded"],
            consumer_parameters=accounting["new_consumer_parameters"],
            total_trainable=accounting["total_trainable"],
            m_scale=carrier_authority_info["m_scale"],
            launch_max_violation=probe["max_violation"],
            normalization_commutes=True,
        )

    launch = {
        "schema": "tfpd_equivariant_cell_v1_launch",
        "status": "CELL_LAUNCHED",
        "cell": args.cell,
        "cell_name": CELL_NAMES[args.cell],
        "integrity": integrity,
        "launch_proofs": launch_proofs,
        "smoke": args.smoke,
        "started_utc": started,
        "initial_state": initial_state_block,
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
                "canonical normalized T4 (rotated per batch)" if args.cell == "B_augmentation"
                else "canonical Z4 fused path + RAW carrier authority to the consumer"
            ),
            "behavior_normalizer_semantic_sha256": behavior_semantic,
            "side_feature_semantic_sha256": side_semantic,
        },
        "carrier_authority": carrier_authority_info,
        "z4_authority": z4_authority,
        "t4_authority_sha256": t4_authority,
        "parameter_accounting": (
            param_proof if args.cell == "C_equivariant"
            else {"trainable_parameters": param_proof}
        ),
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "pretraining_used": False,
            "target_based_selection": False,
            "clipping": "none",
            "gates_preregistered": True,
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
        if args.cell == "B_augmentation":
            stats = train_epoch_rotation_augmented(
                model, optimizer, loader, lr_fn, device, rotation_stream,
                phase_step, num_heads=2, max_steps=args.max_train_steps,
            )
            stream_stats = rotation_stream.stats()
            extra = {
                "rotation_stream_epoch_stats": stream_stats,
                "rotation_sequence_sha256": rotation_stream.sha256(),
            }
        else:
            stats = train_epoch_equivariant(
                model, optimizer, loader, lr_fn, device, bundles,
                phase_step, num_heads=2, max_steps=args.max_train_steps,
            )
            stream_stats = None
            extra = {}
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
        diag = {
            "epoch": epoch,
            "visible_side_mode": "t4_rotated" if args.cell == "B_augmentation" else "z4_fused_raw_carrier_consumer",
            "duration_s": round(time.time() - t0, 3),
            **stats,
            **extra,
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
        if args.cell == "C_equivariant":
            model.set_carrier(bundles[fx_session])
            probe = equivariant.equivariance_probe(
                model, fx_neural[:4], fx_calib[:4], fx_side[:4],
                [0.0] + equivariant.probe_thetas(4, seed=equivariant.EQUIVARIANCE_PROBE_SEED + epoch),
            )
            diag["equivariance_probe"] = probe
        ok_steps = (
            stats["optimizer_steps"] == min(args.max_train_steps, steps_per_epoch)
            if args.max_train_steps is not None
            else stats["optimizer_steps"] == steps_per_epoch
        )
        checks = [
            stats["nonfinite_loss_steps"] == 0,
            stats["nonfinite_grad_steps"] == 0,
            params_finite, diag["optimizer_state_finite"], authority_ok, ok_steps,
        ]
        if args.cell == "B_augmentation":
            checks.append(stream_stats["n"] == phase_step)
        else:
            checks += [
                diag["equivariance_probe"]["identity_rotation_bitwise_equal"],
                diag["equivariance_probe"]["max_violation"] <= equivariant.EQUIVARIANCE_TOLERANCE,
                stats["w_side_grad_exact_zero_all_steps"],
            ]
        if not all(checks):
            invariant_failures.append(epoch)
        if epoch in swa_local:
            name = f"epoch{epoch:03d}.ckpt"
            path = out_dir / name
            torch.save(
                {
                    "kind": "tfpd_equivariant_cell_ckpt_v1",
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
            **({"rotation_median_abs": round(stream_stats["abs_median"], 4)}
               if stream_stats else {}),
            **({"equivariance_max_violation": diag["equivariance_probe"]["max_violation"]}
               if args.cell == "C_equivariant" else {}),
        }), flush=True)

    if len(checkpoints) != 4:
        raise SystemExit(f"expected 4 SWA checkpoints, got {len(checkpoints)}")
    swa_path = out_dir / "swa_final4.pt"
    swa_manifest = matched_scorer.build_swa_final_four(
        [out_dir / c["file"] for c in checkpoints], swa_path
    )
    swa_sha = arm_runner.seal_file(swa_path)
    swa_state = torch.load(swa_path, map_location="cpu", weights_only=False)["state_dict"]

    if args.cell == "B_augmentation":
        swa_model = pop_robust.build_population_robustness_model(seed=args.seed, cell="D")
    else:
        swa_model = equivariant.build_equivariant_model(seed=args.seed)
        equivariant.freeze_inactive_consumer_modules(swa_model)
    swa_model.load_state_dict(swa_state, strict=True)
    swa_model.to(device).eval()
    with torch.no_grad():
        if args.cell == "B_augmentation":
            swa_pred, _ = swa_model(fx_neural[:4], calib_trials=fx_calib[:4],
                                    side_features=fx_side[:4])
            swa_finite = bool(torch.isfinite(swa_pred).all().item())
            swa_probe = None
        else:
            swa_model.set_carrier(bundles[fx_session])
            swa_pred, _ = swa_model(fx_neural[:4], calib_trials=fx_calib[:4],
                                    side_features=fx_side[:4])
            swa_finite = bool(torch.isfinite(swa_pred).all().item())
            swa_probe = equivariant.equivariance_probe(
                swa_model, fx_neural[:4], fx_calib[:4], fx_side[:4],
                [0.0] + equivariant.probe_thetas(args.launch_probe_rotations),
            )
            if (swa_probe["max_violation"] > equivariant.EQUIVARIANCE_TOLERANCE
                    or not swa_probe["identity_rotation_bitwise_equal"]):
                raise SystemExit("SWA model failed the equivariance probe")
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
            "schema": "tfpd_equivariant_cell_v1",
            "status": status,
            "cell": args.cell,
            "cell_name": CELL_NAMES[args.cell],
            "integrity": integrity,
            "launch_proofs": launch_proofs,
            "smoke": args.smoke,
            "max_train_steps": args.max_train_steps,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "initial_state": launch["initial_state"],
            "budget": launch["budget"],
            "data_contract": launch["data_contract"],
            "carrier_authority": carrier_authority_info,
            "z4_authority": z4_authority,
            "t4_authority_sha256": t4_authority,
            "parameter_accounting": launch["parameter_accounting"],
            "epochs_run": len(diagnostics),
            "diagnostics_per_epoch": diagnostics,
            "invariant_failures": invariant_failures,
            "checkpoints": checkpoints,
            "swa": {
                "path": str(swa_path), "sha256": swa_sha,
                "window_epochs": [c["epoch"] for c in checkpoints],
                "manifest": swa_manifest,
                "strict_reload_finite_forward_smoke": swa_finite,
                "equivariance_probe_after_swa": swa_probe,
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
        **({"rotation_sequence_sha256": rotation_stream.sha256()[:16]}
           if args.cell == "B_augmentation" else {}),
        "swa_sha256": swa_sha[:16],
    }, indent=1))
    return 0 if status in ("CELL_TERMINAL", "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
