#!/usr/bin/env python3
"""Z4 boundary pilot: source-only, dual-track T_pre freeze (handoff 20260816 §5).

Runs the phase-1 recipe intended for the official curriculum arm — canonical
Z4 visible side, `build_spintshape_model(seed=42)` standard initialization,
fresh Adam with constant lr=1e-4 (betas=(0.9,0.999), eps=1e-8, weight_decay=0,
amsgrad=False) — over the strict-27 source-train roster, with a deterministic
source-audit split carved out of the gradient set.

DUAL-TRACK JUDGEMENT (the held-in column is never the sole judge):

  (a) source-audit track — 5% of each source session's query windows selected
      by the frozen sha256(session, window_start, "tfpd_pilot_v1") ordering,
      excluded from every pilot gradient.  THIS COLUMN ALONE freezes the
      boundary via the sealed earliest-near-best rule
      (selected_epoch = earliest e in 0..15 with S_e >= S_max - 0.005,
      T_pre = selected_epoch + 1, E_t4 = 48 - T_pre).
  (b) within-development track — all six within-dev sessions, full window
      sets, identical matched scorer.  REPORT ONLY: it never enters T_pre
      selection, early stopping, checkpoint selection, or any training
      decision, and the receipts say so explicitly.

Endpoint convention: endpoint e is the model state after e+1 completed Z4
epochs (epochs are 0-indexed inside the pilot), so T_pre = selected_epoch + 1
is exactly the number of Z4 pretraining epochs the official arm will run.

Receipts (0444, transactional, never overwritten): preflight (split + byte
binding + frozen rules, sealed before the first gradient step) and terminal
(16 endpoints x two columns + T_pre/E_t4 + per-epoch §9 diagnostics +
closure), under results/z4_boundary_pilot_v1 (dry-run under
results/z4_boundary_pilot_v1_dryrun, explicitly non-authoritative).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
# Path order mirrors scripts/run_stage1_spintshape_cell.py: streaming_calibration_exp
# owns src.models.* (imported inside build_spintshape_model); tfpd's own src must
# not shadow it, so tfpd_lane modules are loaded by file path below.
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

PAD_VALUE = -1.0

# Implementation closure: the pilot-owned files plus the read-only model builder.
BOUND_PATTERNS = (
    "src/tfpd_lane/source_audit.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_z4_boundary_pilot.py",
    "src/tfpd/spintshape_module.py",
)
# Read-only authorities recorded (not pilot-owned): data + model surfaces.
AUTHORITY_PATTERNS = (
    "../sua_exploration/mc_maze/multisession_datamodule.py",
    "../sua_exploration/mc_maze/unit_side_features.py",
    "../streaming_calibration_exp/src/models/components/spint.py",
    "../streaming_calibration_exp/src/models/components/streaming_spint.py",
    "../streaming_calibration_exp/src/models/components/streaming_encoders.py",
)


def _load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source_audit = _load_module("tfpd_lane_source_audit", "src/tfpd_lane/source_audit.py")
matched_scorer = _load_module("tfpd_lane_matched_scorer", "src/tfpd_lane/matched_scorer.py")
receipt_mod = _load_module("tfpd_lane_receipt", "src/tfpd_lane/receipt.py")


# ---------------------------------------------------------------------------
# Deterministic audit-excluded training sampler (session-grouped, like the
# official cells: fixed permutation built once, reused every epoch).
# ---------------------------------------------------------------------------
class SourceTrainBatchSampler(Sampler):
    """Session-grouped batches over the audit-excluded positions only.

    Mirrors mc_maze.multisession_datamodule.SessionBatchSampler semantics
    (session-grouped batches of `batch_size`, partial batches dropped, frozen
    seed permutation built once) with the source-audit positions removed from
    the index set entirely.
    """

    def __init__(
        self,
        dataset,
        train_positions,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        import random as _random

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        allowed = list(train_positions)
        session_to_indices: dict[str, list[int]] = {}
        for idx in allowed:
            session = dataset.window_indices[idx][0]
            session_to_indices.setdefault(session, []).append(idx)
        session_batches: dict[str, list[list[int]]] = {}
        for session, indices in session_to_indices.items():
            ordered = list(indices)
            if shuffle:
                ordered = _random.Random(seed).sample(ordered, len(ordered))
            batches = [
                ordered[i : i + batch_size]
                for i in range(0, len(ordered), batch_size)
                if len(ordered[i : i + batch_size]) == batch_size
            ]
            session_batches[session] = batches
        batched = [b for batches in session_batches.values() for b in batches]
        if shuffle:
            batched = _random.Random(seed).sample(batched, len(batched))
        self.batched_indices = batched

    def __iter__(self):
        yield from self.batched_indices

    def __len__(self) -> int:
        return len(self.batched_indices)


# ---------------------------------------------------------------------------
# Hash helpers (§3: signed zero is normalized to +0 before hashing).
# ---------------------------------------------------------------------------
def tensor_sha256(tensor) -> str:
    digest = hashlib.sha256()
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.is_floating_point():
        flat = flat + 0  # IEEE: -0.0 + 0.0 == +0.0; identity for everything else
    digest.update(str(flat.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    if flat.numel():
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(model) -> str:
    from torch.nn.parameter import UninitializedParameter

    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        tensor = state[key]
        if isinstance(tensor, UninitializedParameter):
            digest.update(b"|uninitialized-lazy|")
            continue
        digest.update(tensor_sha256(tensor).encode("utf-8"))
    return digest.hexdigest()


def optimizer_sha256(optimizer) -> str:
    payload = optimizer.state_dict()
    digest = hashlib.sha256()
    digest.update(json.dumps(payload.get("param_groups", []), sort_keys=True).encode("utf-8"))
    for index in sorted(payload.get("state", {}), key=int):
        entry = payload["state"][index]
        digest.update(f"|{index}|".encode("utf-8"))
        step = entry.get("step")
        step_bytes = (
            tensor_sha256(step).encode("utf-8")
            if torch.is_tensor(step)
            else repr(step).encode("utf-8")
        )
        digest.update(b"step:" + step_bytes)
        for moment in ("exp_avg", "exp_avg_sq"):
            if moment in entry:
                digest.update(moment.encode("utf-8") + b":" + tensor_sha256(entry[moment]).encode("utf-8"))
            else:
                digest.update(moment.encode("utf-8") + b":absent")
    return digest.hexdigest()


def w_side_block(model) -> torch.Tensor:
    encoder = model.id_encoder
    weight = encoder.post_pool[0].weight
    return weight[:, encoder.hidden_dim : encoder.hidden_dim + encoder.side_dim]


def _moment_side(model, optimizer, name: str):
    param = model.id_encoder.post_pool[0].weight
    encoder = model.id_encoder
    state = optimizer.state.get(param, {})
    tensor = state.get(name)
    if tensor is None:
        return None
    return tensor[:, encoder.hidden_dim : encoder.hidden_dim + encoder.side_dim]


# ---------------------------------------------------------------------------
# Training / scoring
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device, max_steps=None) -> dict:
    model.train()
    encoder = model.id_encoder
    hidden, side_dim = encoder.hidden_dim, encoder.side_dim
    w_param = encoder.post_pool[0].weight
    params = [p for p in model.parameters() if p.requires_grad]
    acc = {
        "loss_sum": torch.zeros((), device=device),
        "valid_bins": torch.zeros((), device=device),
        "grad_sq_sum": torch.zeros((), device=device),
        "grad_max": torch.zeros((), device=device),
        "wside_grad_nonzero": torch.zeros((), device=device, dtype=torch.long),
        "wside_grad_missing": torch.zeros((), device=device, dtype=torch.long),
        "side_violations": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_loss": torch.zeros((), device=device, dtype=torch.long),
        "nonfinite_grad_steps": torch.zeros((), device=device, dtype=torch.long),
    }
    steps = 0
    examples = 0
    for batch in loader:
        neural, behavior, calib, _session, side = batch[:5]
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        side = side.to(device)
        with torch.no_grad():
            # phase-1 visible side must be exact canonical Z4: finite, all
            # zero, and never negative zero
            acc["side_violations"] += (
                (side != 0).sum() + side.signbit().sum() + (~torch.isfinite(side)).sum()
            )
        prediction, _identity = model(neural, calib_trials=calib, side_features=side)
        valid = (behavior != PAD_VALUE).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            grads = [p.grad for p in params if p.grad is not None]
            norms = torch._foreach_norm(grads)
            total_sq = torch.stack(norms).pow(2).sum()
            acc["grad_sq_sum"] += total_sq
            acc["grad_max"] = torch.maximum(acc["grad_max"], total_sq.sqrt())
            acc["nonfinite_grad_steps"] += (~torch.isfinite(total_sq)).long()
            grad = w_param.grad
            if grad is None:
                acc["wside_grad_missing"] += 1
            else:
                acc["wside_grad_nonzero"] += (grad[:, hidden : hidden + side_dim] != 0).sum()
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
        "grad_norm_global_mean": float((acc["grad_sq_sum"] / steps).sqrt().item()) if steps else 0.0,
        "grad_norm_global_max": float(acc["grad_max"].item()),
        "w_side_grad_exact_zero_all_steps": int(acc["wside_grad_nonzero"].item()) == 0
        and int(acc["wside_grad_missing"].item()) == 0,
        "w_side_grad_nonzero_count": int(acc["wside_grad_nonzero"].item()),
        "visible_side_violation_count": int(acc["side_violations"].item()),
        "nonfinite_loss_steps": int(acc["nonfinite_loss"].item()),
        "nonfinite_grad_steps": int(acc["nonfinite_grad_steps"].item()),
    }


def score_track(
    model,
    dataset,
    positions,
    device,
    batch_size: int,
    window_size: int,
    cap_per_session=None,
    forward_mode: str = "identity_cached",
) -> dict:
    """Score a fixed position set with the matched per-session scorer.

    Equal weight per session; per-window semantics identical to the Stage-1
    evaluators (row-masked padding, [n, 2] prediction/target pairs).  No
    gradients, no optimizer, no state mutation: the model stays in eval mode
    under torch.no_grad().
    """
    model.eval()
    starts_by_session: dict[str, list[int]] = {}
    for position in positions:
        session, start = dataset.window_indices[position]
        starts_by_session.setdefault(session, []).append(int(start))
    per_session = []
    with torch.no_grad():
        for session in sorted(starts_by_session):
            starts = starts_by_session[session]
            if cap_per_session is not None:
                starts = starts[:cap_per_session]
            record = dataset.sessions[session]
            calib = (
                torch.from_numpy(record.calib_trials.copy()).float().unsqueeze(0).to(device)
            )
            side = (
                torch.from_numpy(record.side_features.copy()).float().unsqueeze(0).to(device)
            )
            identity = None
            if forward_mode == "identity_cached":
                identity = model.compute_identity(calib, side_features=side)
            predictions, targets = [], []
            for i in range(0, len(starts), batch_size):
                chunk = starts[i : i + batch_size]
                neural = (
                    torch.from_numpy(
                        np.stack([record.neural[s : s + window_size] for s in chunk])
                    )
                    .float()
                    .to(device)
                )
                behavior = (
                    torch.from_numpy(
                        np.stack([record.behavior[s : s + window_size] for s in chunk])
                    )
                    .float()
                    .to(device)
                )
                if forward_mode == "identity_cached":
                    prediction = model.decode_with_identity(neural, identity)
                else:
                    prediction, _ = model(
                        neural,
                        calib_trials=calib.expand(neural.shape[0], -1, -1, -1),
                        side_features=side.expand(neural.shape[0], -1, -1),
                    )
                valid = (behavior != PAD_VALUE).all(dim=-1)
                predictions.append(prediction[valid].cpu())
                targets.append(behavior[valid].cpu())
            per_session.append(
                {
                    "session": session,
                    "r2": matched_scorer.session_r2(
                        torch.cat(predictions), torch.cat(targets)
                    ),
                    "n_windows_scored": len(starts),
                }
            )
    return {
        "mean_r2": float(np.mean([row["r2"] for row in per_session])),
        "per_session": per_session,
        "n_windows_scored": int(sum(row["n_windows_scored"] for row in per_session)),
        "n_sessions": len(per_session),
        "equal_weight_per_session": True,
    }


def probe_scoring_forward_mode(model, dataset, positions, device, batch_size, window_size) -> dict:
    """Verify the identity-cached decode path against the full forward path.

    The encoder output for a session's calibration is constant across its
    windows, so hoisting compute_identity out of the batch loop is
    mathematically identical; the probe bounds the numerical deviation once,
    before any endpoint is scored, and the receipt records it.
    """
    position = positions[0]
    session, start = dataset.window_indices[position]
    record = dataset.sessions[session]
    chunk = [int(start)] * min(batch_size, 8)
    neural = (
        torch.from_numpy(np.stack([record.neural[s : s + window_size] for s in chunk]))
        .float()
        .to(device)
    )
    behavior = (
        torch.from_numpy(np.stack([record.behavior[s : s + window_size] for s in chunk]))
        .float()
        .to(device)
    )
    calib = torch.from_numpy(record.calib_trials.copy()).float().unsqueeze(0).to(device)
    side = torch.from_numpy(record.side_features.copy()).float().unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        fast = model.decode_with_identity(neural, identity)
        full, _ = model(
            neural,
            calib_trials=calib.expand(neural.shape[0], -1, -1, -1),
            side_features=side.expand(neural.shape[0], -1, -1),
        )
        max_abs_diff = float((fast - full).abs().max().item())
    if was_training:
        model.train()
    return {
        "probe_session": session,
        "probe_window_start": int(start),
        "max_abs_diff": max_abs_diff,
        "tolerance": 1e-5,
        "identity_cached_path_accepted": max_abs_diff <= 1e-5,
    }


def verify_z4_authority(datamodule, cache_dir) -> dict:
    """Bitwise z4 authority check against the production side-feature loader.

    Handoff §4: the phase-1 visible side must be bitwise equal to
    load_unit_side_features(..., group="z4") for the same session and unit
    order, after normalization.  Recomputed here for every train and val
    session from the cache-backed production loader.
    """
    from mc_maze.multisession_datamodule import session_name_from_path
    from mc_maze.unit_side_features import load_unit_side_features

    side_mean, side_std = datamodule._side_feature_stats
    rows = []
    for split, dataset in (("train", datamodule.train_dataset), ("val", datamodule.val_dataset)):
        for nwb_path in datamodule.session_files[split]:
            name = session_name_from_path(nwb_path)
            features, _meta = load_unit_side_features(
                nwb_path,
                feature_group="z4",
                pool_size=30,
                mean=side_mean,
                std=side_std,
                cache_dir=cache_dir,
                bin_size_ms=20,
                window_size=50,
                trial_result_filter="R",
                signal_view="sua",
            )
            record = dataset.sessions[name]
            bitwise_equal = features.shape == record.side_features.shape and np.array_equal(
                features, record.side_features
            )
            exact_positive_zero = bool(
                np.all(features == 0) and not np.any(np.signbit(features)) and np.isfinite(features).all()
            )
            rows.append(
                {
                    "split": split,
                    "session": name,
                    "shape": list(features.shape),
                    "bitwise_equal_to_dataset_record": bool(bitwise_equal),
                    "exact_positive_zero": exact_positive_zero,
                    "sha256": source_audit._float32_sha256(features),
                }
            )
    return {
        "loader": "mc_maze.unit_side_features.load_unit_side_features(feature_group='z4')",
        "all_bitwise_equal": all(r["bitwise_equal_to_dataset_record"] for r in rows),
        "all_exact_positive_zero": all(r["exact_positive_zero"] for r in rows),
        "n_sessions_checked": len(rows),
        "per_session": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--smoke-windows-per-session", type=int, default=96)
    parser.add_argument(
        "--eval-forward-mode", choices=["auto", "identity_cached", "full"], default="auto"
    )
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3

    dry_run = args.dry_run
    epochs = 1 if dry_run else args.epochs
    max_train_steps = 1 if dry_run else args.max_train_steps
    cap = args.smoke_windows_per_session if dry_run else None
    if not dry_run and epochs != len(source_audit.CANDIDATE_EPOCHS):
        print(
            f"the frozen candidate range is epochs 0..15 (16 endpoints); refusing epochs={epochs}",
            file=sys.stderr,
        )
        return 3

    out_dir = args.output_root or (
        ROOT / "results/z4_boundary_pilot_v1_dryrun"
        if dry_run
        else ROOT / "results/z4_boundary_pilot_v1"
    )
    out_dir = Path(out_dir)
    if out_dir.exists():
        print(f"fresh pilot output directory already exists: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = "DRY_RUN_SMOKE" if dry_run else "Z4_BOUNDARY_PILOT_RUNNING"
    failure: dict | None = None
    try:
        _run_pilot(args, out_dir, dry_run, epochs, max_train_steps, cap, started)
    except SystemExit as exc:  # pragma: no cover - argparse/env paths
        failure = {"kind": "SystemExit", "detail": str(exc.code)}
        raise
    except BaseException as exc:  # noqa: BLE001 - failure receipts are mandatory
        failure = {"kind": type(exc).__name__, "detail": str(exc), "traceback": traceback.format_exc()}
        terminal = {
            "schema": "tfpd_z4_boundary_pilot_v1",
            "status": "PILOT_FAILED",
            "dry_run": dry_run,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "failure": failure,
        }
        receipt_mod.write_receipt_transactionally(out_dir / "terminal_receipt.json", terminal)
        print(json.dumps(terminal, indent=1), file=sys.stderr)
        return 1
    return 0


def _run_pilot(args, out_dir: Path, dry_run: bool, epochs: int, max_train_steps, cap, started: str) -> None:
    import lightning.pytorch as pl

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA requested but unavailable ({args.device})", file=sys.stderr)
        raise SystemExit(3)

    closure_launch = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    authorities = {
        rel: receipt_mod.sha256_file(REPO / rel.split("../", 1)[1])
        for rel in AUTHORITY_PATTERNS
    }

    import mc_maze.a2_matched_subject_shift_v2_core as a2
    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    spintshape = _load_module("tfpd_spintshape_module", "src/tfpd/spintshape_module.py")

    pl.seed_everything(args.seed, workers=True)

    datamodule = Dandi688MultiSessionDataModule(
        data_dir=str(a2.SUBC_DATA_ROOT),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=args.train_batch_size,
        window_size=50,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        random_calibration=False,
        seed=args.seed,
        max_units_exclusive=100,
        cache_dir=str(a2.SOURCE_CACHE_ROOT),
        signal_view="sua",
        side_feature_group="z4",
        side_feature_pool_size=30,
        train_val_manifest_path=str(a2.MANIFEST_PATH),
    )
    datamodule.setup("fit")
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    roster = tuple(datamodule.session_splits["train"])
    within_roster = tuple(datamodule.session_splits["val"])
    formal_names = tuple(datamodule.session_splits["test"])  # inert strings, never resolved
    if len(roster) != 27 or len(within_roster) != 6 or len(formal_names) != 6:
        raise SystemExit("strict manifest roster drift: expected 27/6/6")

    # ---- deterministic source-audit split (source sessions only) ----------
    window_keys = [(s, int(t)) for s, t in train_dataset.window_indices]
    plan = source_audit.build_audit_plan(window_keys, roster)
    source_audit.assert_no_forbidden_sessions(
        [s for s, _ in window_keys], within_roster + formal_names
    )
    binding = source_audit.bind_audit_windows(plan, train_dataset, window_size=50)

    behavior_mean, behavior_std = datamodule._behavior_stats
    side_mean, side_std = datamodule._side_feature_stats
    behavior_semantic = a2.normalizer_value_sha256(behavior_mean, behavior_std)
    side_semantic = a2.normalizer_value_sha256(side_mean, side_std)
    if not behavior_semantic.startswith("f062506c"):
        raise SystemExit(f"source behavior normalizer semantic SHA drift: {behavior_semantic}")
    manifest_sha = receipt_mod.sha256_file(a2.MANIFEST_PATH)
    if manifest_sha != a2.EXPECTED_MANIFEST_SHA256:
        raise SystemExit(f"strict manifest SHA drift: {manifest_sha}")

    z4_authority = verify_z4_authority(datamodule, a2.SOURCE_CACHE_ROOT)
    if not (z4_authority["all_bitwise_equal"] and z4_authority["all_exact_positive_zero"]):
        raise SystemExit("canonical Z4 authority check failed (bitwise loader equality)")

    preflight = {
        "schema": "tfpd_z4_boundary_pilot_v1_preflight",
        "status": "PREFLIGHT_SEALED" if not dry_run else "PREFLIGHT_SEALED_DRY_RUN",
        "dry_run": dry_run,
        "sealed_before_first_gradient_step": True,
        "started_utc": started,
        "namespace": source_audit.NAMESPACE,
        "audit_rule": {
            "selection": (
                "per strict-27 train session, eligible query windows ranked ascending by "
                "sha256('<session_id>|<window_start>|<namespace>') with exact "
                "(window_start, position) tie-break; first floor(5%*n) windows (min 1)"
            ),
            "fraction": source_audit.AUDIT_FRACTION,
            "gradient_exclusion": "audit positions are removed from the pilot training index set",
            "plan_sha256": plan.plan_sha256,
            "per_session": [dict(row) for row in plan.per_session],
        },
        "selector_rule_frozen": {
            "candidate_epochs": list(source_audit.CANDIDATE_EPOCHS),
            "endpoint_convention": "endpoint e = model state after e+1 completed Z4 epochs",
            "tolerance": source_audit.SELECTOR_TOLERANCE,
            "rule": "selected_epoch = earliest e with S_e >= S_max - 0.005; T_pre = e + 1; E_t4 = 48 - T_pre",
            "input_column": "source_audit_only",
        },
        "track_roles": {
            "source_audit": "SELECTS T_pre (this column alone freezes the boundary)",
            "within_dev_report_only": (
                "REPORT ONLY: never used for T_pre, early stopping, checkpoint "
                "selection, or any training decision"
            ),
        },
        "roster": {
            "train": list(roster),
            "within_dev": list(within_roster),
            "formal_test_names_inert": list(formal_names),
            "manifest_path": str(a2.MANIFEST_PATH),
            "manifest_sha256": manifest_sha,
        },
        "no_within_external_formal_rows": {
            "assertion": "audit windows drawn exclusively from the strict-27 train roster",
            "within_dev_sessions_in_audit": 0,
            "external_sub_m_opened": False,
            "formal_or_organizer_held_data_opened": False,
        },
        "normalizers": {
            "behavior_semantic_sha256": behavior_semantic,
            "side_feature_semantic_sha256": side_semantic,
        },
        "z4_authority": z4_authority,
        "audit_binding": binding,
        "counts": {
            "n_train_windows": len(window_keys),
            "n_audit_windows": len(plan.audit_keys),
            "n_gradient_windows": len(plan.train_positions),
            "n_within_dev_windows": len(val_dataset.window_indices),
        },
        "model_and_optimizer_contract": {
            "builder": "src/tfpd/spintshape_module.py:build_spintshape_model(seed=42)",
            "visible_side": "canonical_z4",
            "optimizer": {
                "name": "Adam", "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0.0, "amsgrad": False, "schedule": "constant",
            },
            "train_batch_size": args.train_batch_size,
            "sampler": "session-grouped, drop-partial, frozen seed permutation, audit positions excluded",
        },
        "source_closure": closure_launch,
        "authority_sha256": authorities,
        "environment": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "preflight_receipt.json", preflight)

    # ---- model, optimizer, phase-1 invariants ----------------------------
    model = spintshape.build_spintshape_model(seed=args.seed).to(device)
    encoder = model.id_encoder
    if getattr(encoder, "variant", "") != "B3S":
        raise SystemExit("unexpected encoder variant")
    if tuple(encoder.post_pool[0].weight.shape) != (
        encoder.hidden_dim,
        encoder.hidden_dim + encoder.side_dim,
    ):
        raise SystemExit("post_pool[0] geometry drift (expected Linear(68,64))")
    w_side = w_side_block(model)
    if int(torch.count_nonzero(w_side).item()) != 0 or bool(w_side.signbit().any().item()):
        raise SystemExit("W_side is not exactly positive zero at initialization")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False,
    )
    if optimizer.state:
        raise SystemExit("optimizer state must be absent before the first step")

    train_sampler = SourceTrainBatchSampler(
        train_dataset, plan.train_positions, batch_size=args.train_batch_size,
        shuffle=True, seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    forward_mode = (
        "identity_cached"
        if args.eval_forward_mode in ("auto", "identity_cached")
        else "full"
    )
    probe = probe_scoring_forward_mode(
        model, train_dataset, plan.audit_positions, device,
        args.eval_batch_size, 50,
    )
    if args.eval_forward_mode == "auto" and not probe["identity_cached_path_accepted"]:
        forward_mode = "full"
    probe["forward_mode_selected"] = forward_mode

    # §3 preflight gradient probe: with canonical Z4, grad(W_side) is bitwise zero
    probe_position = plan.train_positions[0]
    p_neural, p_behavior, p_calib, _p_session, p_side = train_dataset[probe_position]
    p_neural = p_neural.unsqueeze(0).to(device)
    p_behavior = p_behavior.unsqueeze(0).to(device)
    p_calib = p_calib.unsqueeze(0).to(device)
    p_side = p_side.unsqueeze(0).to(device)
    model.train()
    _pred, _ident = model(p_neural, calib_trials=p_calib, side_features=p_side)
    valid = (p_behavior != PAD_VALUE).all(dim=-1)
    diff2 = ((_pred - p_behavior) ** 2).sum(dim=-1)
    probe_loss_tensor = (diff2 * valid).sum() / (valid.sum() * p_behavior.shape[-1])
    probe_loss_tensor.backward()
    probe_loss = float(probe_loss_tensor.item())
    grad_side = encoder.post_pool[0].weight.grad[
        :, encoder.hidden_dim : encoder.hidden_dim + encoder.side_dim
    ]
    probe_grad_zero = int(torch.count_nonzero(grad_side).item()) == 0
    optimizer.zero_grad(set_to_none=True)
    if optimizer.state:
        raise SystemExit("optimizer state must remain absent after the probe (no step taken)")
    if not probe_grad_zero:
        raise SystemExit("grad(W_side) is not bitwise zero under canonical Z4")
    gradient_probe = {
        "probe_position": int(probe_position),
        "probe_session": train_dataset.window_indices[probe_position][0],
        "probe_loss": probe_loss,
        "grad_w_side_bitwise_zero_magnitude": True,
        "optimizer_state_absent_after_probe": True,
        "w_side_exact_positive_zero_at_init": True,
    }
    del p_neural, p_behavior, p_calib, p_side, _pred, _ident

    endpoints = []
    diagnostics = []
    invariant_failures = []
    total_steps = 0
    for epoch in range(epochs):
        epoch_started = time.time()
        stats = train_one_epoch(
            model, optimizer, train_loader, device, max_steps=max_train_steps
        )
        total_steps += stats["optimizer_steps"]

        w_side = w_side_block(model)
        w_zero_mag = int(torch.count_nonzero(w_side).item()) == 0
        w_positive_zero = not bool(w_side.signbit().any().item())
        exp_avg_side = _moment_side(model, optimizer, "exp_avg")
        exp_avg_sq_side = _moment_side(model, optimizer, "exp_avg_sq")
        moments_zero = (exp_avg_side is None or int(torch.count_nonzero(exp_avg_side).item()) == 0) and (
            exp_avg_sq_side is None or int(torch.count_nonzero(exp_avg_sq_side).item()) == 0
        )
        from torch.nn.parameter import UninitializedParameter

        params_finite = all(
            bool(torch.isfinite(p.detach()).all().item())
            for p in model.parameters()
            if p.requires_grad
            and not isinstance(p, UninitializedParameter)
            and p.numel()
        )
        diag = {
            "epoch": epoch,
            "epochs_completed": epoch + 1,
            "duration_s": round(time.time() - epoch_started, 3),
            "visible_side_mode": "canonical_z4",
            **stats,
            "optimizer_steps_total": total_steps,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "w_side_norm": float(w_side.norm().item()),
            "w_side_exact_zero_magnitude": w_zero_mag,
            "w_side_positive_zero_bitwise": w_positive_zero,
            "w_side_exp_avg_norm": (
                0.0 if exp_avg_side is None else float(exp_avg_side.norm().item())
            ),
            "w_side_exp_avg_sq_norm": (
                0.0 if exp_avg_sq_side is None else float(exp_avg_sq_side.norm().item())
            ),
            "w_side_moments_exact_zero_magnitude": moments_zero,
            "norm_alpha_times_w_side": float(w_side.norm().item()),
            "norm_w_side_times_t4_fixed_batch": 0.0,
            "parameters_finite": params_finite,
            "state_dict_sha256": state_sha256(model),
            "optimizer_state_sha256": optimizer_sha256(optimizer),
        }
        diagnostics.append(diag)
        if not (
            w_zero_mag
            and moments_zero
            and stats["w_side_grad_exact_zero_all_steps"]
            and stats["visible_side_violation_count"] == 0
            and stats["nonfinite_loss_steps"] == 0
            and stats["nonfinite_grad_steps"] == 0
            and params_finite
        ):
            invariant_failures.append(epoch)

        audit_score = score_track(
            model, train_dataset, plan.audit_positions, device,
            args.eval_batch_size, 50, cap_per_session=cap, forward_mode=forward_mode,
        )
        within_score = score_track(
            model, val_dataset, list(range(len(val_dataset.window_indices))), device,
            args.eval_batch_size, 50, cap_per_session=cap, forward_mode=forward_mode,
        )
        endpoints.append(
            {
                "epoch": epoch,
                "epochs_completed": epoch + 1,
                "source_audit_selects_t_pre": audit_score,
                "within_dev_report_only": within_score,
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "source_audit_r2": round(audit_score["mean_r2"], 6),
                    "within_dev_r2": round(within_score["mean_r2"], 6),
                    "train_loss": round(stats["train_loss_mean_per_step"], 6),
                    "w_side_norm": diag["w_side_norm"],
                }
            ),
            flush=True,
        )

    closure_final = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)
    closure_equal = closure_final["closure_sha256"] == closure_launch["closure_sha256"]

    terminal = {
        "schema": "tfpd_z4_boundary_pilot_v1",
        "status": (
            "DRY_RUN_SMOKE__NON_AUTHORITATIVE"
            if dry_run
            else (
                "Z4_BOUNDARY_PILOT_TERMINAL"
                if not invariant_failures and closure_equal
                else "Z4_BOUNDARY_PILOT_INVARIANT_OR_CLOSURE_FAILURE"
            )
        ),
        "dry_run": dry_run,
        "smoke_cap_windows_per_session": cap,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track_roles": {
            "source_audit": (
                "SELECTS T_pre: matched per-session variance-weighted R2, equal weight "
                "per session, over the deterministic 5% source-audit windows; this "
                "column alone freezes the boundary"
            ),
            "within_dev_report_only": (
                "REPORT ONLY (dual-track judgement): all six within-dev sessions, full "
                "window sets, same scorer; never used for T_pre, checkpoint selection, "
                "early stopping, or any training decision"
            ),
        },
        "scoring": {
            "scorer": "src/tfpd_lane/matched_scorer.session_r2 (torchmetrics R2Score, variance_weighted)",
            "forward_mode": forward_mode,
            "identity_cached_probe": probe,
            "w_side_gradient_probe": gradient_probe,
            "eval_batch_size": args.eval_batch_size,
        },
        "model_and_optimizer": preflight["model_and_optimizer_contract"],
        "audit_plan_sha256": plan.plan_sha256,
        "audit_binding_sha256": binding["binding_sha256"],
        "counts": preflight["counts"],
        "endpoints": endpoints,
        "diagnostics_per_epoch": diagnostics,
        "invariant_failures": invariant_failures,
        "disclosures": {
            "teacher_checkpoint_logits_or_loss_used": False,
            "t4_tensors_loaded": False,
            "formal_or_organizer_held_data_opened": False,
            "external_sub_m_opened": False,
            "audit_windows_excluded_from_gradients": True,
            "scoring_target_updates_or_gradients_or_optimizer_steps": 0,
            "within_dev_influence_on_t_pre": "none (report-only column)",
            "w_side_zero_column_and_zero_moment_invariants_held": not invariant_failures,
            "source_roster_and_manifest": preflight["roster"],
            "unit_order": "datamodule channel order (sorted units, units<100)",
        },
        "source_closure": {"launch": closure_launch, "final": closure_final, "launch_final_closure_equal": closure_equal},
        "authority_sha256": authorities,
        "environment": preflight["environment"],
    }

    if not dry_run:
        if invariant_failures or not closure_equal:
            terminal["t_pre_selection"] = None
        else:
            selection = source_audit.select_t_pre(
                {row["epoch"]: row["source_audit_selects_t_pre"]["mean_r2"] for row in endpoints}
            )
            within_best = max(row["within_dev_report_only"]["mean_r2"] for row in endpoints)
            within_best_epoch = [
                row["epoch"]
                for row in endpoints
                if row["within_dev_report_only"]["mean_r2"] == within_best
            ][0]
            terminal["t_pre_selection"] = selection
            terminal["boundary"] = {
                "t_pre": selection["t_pre"],
                "e_t4": selection["e_t4"],
                "selected_epoch": selection["selected_epoch"],
                "s_max_source_audit": selection["s_max"],
                "within_dev_report_only_best": {
                    "epoch": within_best_epoch,
                    "mean_r2": within_best,
                    "role": "descriptive only; NOT the selector column",
                },
                "columns_agree_on_selected_epoch": within_best_epoch == selection["selected_epoch"],
            }
    receipt_mod.write_receipt_transactionally(out_dir / "terminal_receipt.json", terminal)

    summary = {
        "status": terminal["status"],
        "output_root": str(out_dir),
        "endpoints": [
            {
                "epoch": row["epoch"],
                "source_audit_r2": round(row["source_audit_selects_t_pre"]["mean_r2"], 6),
                "within_dev_r2_report_only": round(row["within_dev_report_only"]["mean_r2"], 6),
            }
            for row in endpoints
        ],
        "t_pre_selection": terminal.get("t_pre_selection"),
    }
    print(json.dumps(summary, indent=1))
    if invariant_failures or not closure_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
