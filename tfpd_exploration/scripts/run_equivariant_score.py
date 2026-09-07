#!/usr/bin/env python3
"""Matched scorer for the SO(2) equivariance arms (B / C / C-prime) + references.

Lane conventions (``run_subpop_score.py`` pattern), read-only, zero GPU
training, zero target updates:

- same-engine live scoring of every landed arm and of the sealed references
  (arm A and sealed Cell D via the Step 0C SHA-bound checkpoint machinery) —
  no hand-copied numbers anywhere;
- strict-27 source normalizers (behavior semantic SHA f062506c; source-fit side
  statistics) injected and SHA-checked exactly as ``build_surfaces`` does;
- governing = last-bin, per-session variance-weighted R2, equal session
  weight, ``n_windows`` in every per-session entry; full-window + window
  weighted as DIAGNOSTICS ONLY;
- both surfaces (within 6 sub-C dev, external 15 sub-M), both granularities;
- external date blocks (2014 / 2015) with ``sub-M_ses-CO-20141203`` reported
  separately, paired deltas vs arm A and D per block, window share;
- ``matched_scorer.paired_session_stats`` paired per-session deltas with the
  10,000-draw bootstrap CI (DESCRIPTIVE ONLY) and exact sign patterns;
- every landed cell is independently gated on its own terminal receipt
  (CELL_TERMINAL, smoke false, 48 epochs at 33,925 steps/epoch, empty
  invariant failures, launch/final closure equality, SWA SHA vs receipt,
  0444 + sidecar, non-symlink); a cell that has not landed is omitted with a
  ``not_landed`` note, never scored from a partial artifact (C-prime in this
  pass: its contrast cells are emitted as PENDING until it lands).

Model-specific loading:

- armA_swa / D_swa: the standard lane loading path
  (``step0c.build_step0c_model`` + conditional ``model.``-prefix strip +
  strict load + ``UninitializedParameter`` safe globals), SHA-bound by
  ``step0c.verify_checkpoints`` before any data is opened.
- B_swa: the sealed Cell-D graph through the SAME standard loading path (B is
  an exact Cell-D replica; the dynamic-dropout flag is train-mode-only and
  carries no parameters), plus a one-batch bitwise cross-check against the
  D-shaped (pop_robust) build to prove the eval path is the standard one.
- C_swa / Cprime_swa: the frozen builders (``equivariant_cell`` /
  ``equivariant_control``) + the sealed shared raw-T4 carrier authority and
  its frozen m_scale; eval = PLAIN forward (no rotation, no augmentation).
  The lane's ``score_last_bin`` / ``score_track`` engines are extended by
  VERBATIM COPIES whose ONLY additions are the per-session carrier
  ``set_carrier`` and (full-window track) the zeros_like fused-path side; for
  carrier-free models the copies reduce to the originals bit-for-bit, which is
  asserted live for arm B on the within surface (same-engine proof).

Pre-registered contrasts with the gates LOADED from the landed training
receipts (single source of truth): B - A external governing (>= +0.03 AND
>= 10/15 = the augmentation win), C - A (preregistered, capacity-confounded —
recorded as such), C - B, plus D - A as the reference gain.  C - C-prime /
C-prime - A / C-prime - B are the matched-capacity readings and stay PENDING
until C-prime lands; scoring it later is one command:
``--cells Cprime --output-root results/equivariant_v1/score_cprime``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))

PAD_VALUE = -1.0
AUTH_VALUE = "I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS"
SEPARATE_SESSION = "sub-M_ses-CO-20141203"
FULL_BUDGET_EPOCHS = 48
FULL_BUDGET_STEPS = 33_925
EQUIVARIANCE_TOL = 1e-5
BOUND_PATTERNS = (
    "scripts/run_equivariant_score.py",
    "tests/test_equivariant_score.py",
    "src/tfpd_lane/equivariant_cell.py",
    "src/tfpd_lane/equivariant_control.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "src/tfpd_lane/arm_common.py",
    "scripts/run_a2_matched_rescore.py",
    "scripts/run_z4_boundary_pilot.py",
    "scripts/run_subpop_step0c.py",
)
CELL_ROOT = ROOT / "results/equivariant_v1"
CANONICAL_INITIAL_STATE = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
CARRIER_AUTHORITY = CELL_ROOT / "carrier_authority/carrier_authority.pt"
CELL_SPECS = {
    "B": {"cell_dir": "cellB_rotation_augmentation"},
    "C": {"cell_dir": "cellC_equivariant"},
    "Cprime": {"cell_dir": "cellCprime_matched_control"},
}
CONTRAST_GATES = {
    "B_minus_A": "augmentation win: external governing mean >= +0.03 AND >= 10/15 positive",
    "C_minus_A": (
        "preregistered as launched; CAPACITY-CONFOUNDED (C trains 492,951 of "
        "3,510,842 parameters) — decompose with C - Cprime when it lands"
    ),
    "C_minus_B": "descriptive: symmetry vs augmentation",
    "C_minus_Cprime": "PENDING until Cprime lands (equivariance at matched capacity)",
    "Cprime_minus_A": "PENDING (capacity-confound bound)",
    "Cprime_minus_B": "PENDING (matched-capacity control vs augmentation)",
    "D_minus_A": "reference gain (the lane's calibration bar), scored live",
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---- terminal-receipt validation (fail-closed; mirrors the lane pattern) ----
def validate_equivariant_terminal(cell: str, arm_common, root: Path | None = None):
    """Validate one landed arm's terminal receipt + sealed SWA.

    Returns None when the terminal receipt does not exist yet (not landed);
    raises SystemExit on ANY defect in a receipt that DOES exist.  ``root``
    overrides the receipt family root (test fixtures use it; the production
    path always reads results/equivariant_v1).
    """
    import stat as _stat

    directory = (Path(root) if root is not None else CELL_ROOT) / CELL_SPECS[cell]["cell_dir"]
    receipt_path = directory / "terminal_receipt.json"
    swa_path = directory / "swa_final4.pt"
    if not receipt_path.is_file():
        return None
    problems: list[str] = []
    sidecar = Path(str(receipt_path) + ".sha256")
    if not sidecar.is_file():
        problems.append("terminal sidecar missing")
    if receipt_path.is_symlink() or (sidecar.is_file() and sidecar.is_symlink()):
        problems.append("symlinked receipt/sidecar")
    body_sha = arm_common.sha256_file(receipt_path)
    if sidecar.is_file() and body_sha != sidecar.read_text().split()[0]:
        problems.append("sidecar SHA mismatch")
    mode = _stat.S_IMODE(receipt_path.stat().st_mode)
    if mode != 0o444:
        problems.append(f"receipt mode {oct(mode)} != 0444")
    payload = json.loads(receipt_path.read_text())
    if payload.get("status") != "CELL_TERMINAL":
        problems.append(f"status {payload.get('status')!r}")
    if payload.get("smoke") is not False:
        problems.append("smoke receipt (non-authoritative)")
    if payload.get("max_train_steps") is not None:
        problems.append("truncated-step run")
    if payload.get("epochs_run") != FULL_BUDGET_EPOCHS:
        problems.append(f"epochs_run {payload.get('epochs_run')} != {FULL_BUDGET_EPOCHS}")
    budget = payload.get("budget", {})
    if budget.get("steps_per_epoch") != FULL_BUDGET_STEPS or budget.get("epochs") != FULL_BUDGET_EPOCHS:
        problems.append(
            f"budget {budget.get('steps_per_epoch')}x{budget.get('epochs')} != "
            f"{FULL_BUDGET_STEPS}x{FULL_BUDGET_EPOCHS}"
        )
    if payload.get("invariant_failures"):
        problems.append("invariant failures recorded")
    if not payload.get("source_closure", {}).get("launch_final_closure_equal"):
        problems.append("launch/final closure inequality")
    canonical = torch.load(CANONICAL_INITIAL_STATE, map_location="cpu",
                           weights_only=False)
    initial = payload.get("initial_state", {})
    if initial.get("state_dict_sha256") != canonical["state_sha256"]:
        problems.append("terminal-bound initial-state state SHA != canonical")
    if initial.get("artifact_sha256") != arm_common.sha256_file(CANONICAL_INITIAL_STATE):
        problems.append("terminal-bound initial-state artifact SHA drift")
    normalizer = payload.get("data_contract", {}).get(
        "behavior_normalizer_semantic_sha256", "")
    if not str(normalizer).startswith("f062506c"):
        problems.append("terminal-bound normalizer semantic SHA drift")
    integrity = payload.get("integrity", {})
    heads = integrity.get("num_heads")
    heads_ok = (heads == 2) or (
        isinstance(heads, dict)
        and heads.get("consumer_attention_heads") == 2
        and heads.get("decoder_num_heads") == 2
    )
    if not heads_ok:
        problems.append(f"num_heads {heads!r} invalid")
    if not payload.get("t4_authority_sha256"):
        problems.append("no T4 authority fingerprint in the receipt")
    # the sealed SWA artifact itself
    swa_sha = payload.get("swa", {}).get("sha256")
    if not swa_sha:
        problems.append("terminal receipt carries no swa sha256")
    if not swa_path.is_file():
        problems.append(f"SWA missing: {swa_path}")
    else:
        if swa_path.is_symlink():
            problems.append("symlinked SWA")
        live_sha = arm_common.sha256_file(swa_path)
        if swa_sha and live_sha != swa_sha:
            problems.append("SWA SHA mismatch vs terminal receipt")
        swa_sidecar = Path(str(swa_path) + ".sha256")
        if swa_sidecar.is_file() and swa_sidecar.read_text().split()[0] != live_sha:
            problems.append("SWA sidecar SHA mismatch")

    gates_from_receipt = integrity.get("gates", {})

    # ---- cell-specific law checks -----------------------------------------
    per_epoch = payload.get("diagnostics_per_epoch", [])
    if cell == "B":
        law = integrity.get("augmentation_law", {})
        if law.get("shared_across_batch") is not True:
            problems.append("B: rotation not shared across the batch")
        if "never applied" not in str(law.get("eval", "")):
            problems.append("B: eval policy does not forbid augmentation")
        if not payload.get("rotation_sequence_sha256") and not any(
            "rotation_sequence_sha256" in d for d in per_epoch
        ):
            problems.append("B: no SHA-recorded rotation stream")
    elif cell == "C":
        authority = payload.get("carrier_authority") or {}
        if authority.get("sha256") != arm_common.sha256_file(CARRIER_AUTHORITY):
            problems.append("C: carrier-authority SHA drift vs the sealed artifact")
        probes = [d.get("equivariance_probe") for d in per_epoch]
        if not probes or any(p is None for p in probes):
            problems.append("C: missing per-epoch equivariance records")
        else:
            worst = max(p["max_violation"] for p in probes)
            if worst > EQUIVARIANCE_TOL:
                problems.append(f"C: per-epoch equivariance violation {worst}")
            if not all(p["identity_rotation_bitwise_equal"] for p in probes):
                problems.append("C: identity rotation not a bitwise baseline")
        swa_probe = payload.get("swa", {}).get("equivariance_probe_after_swa")
        if not swa_probe or swa_probe["max_violation"] > EQUIVARIANCE_TOL:
            problems.append("C: post-SWA equivariance probe missing or failing")
    elif cell == "Cprime":
        matched = integrity.get("matched_to_arm_c", {})
        if matched.get("consumer_parameters") != 185_793 or \
                matched.get("total_trainable") != 492_951:
            problems.append("Cprime: matched-capacity numbers drift from arm C's")
        if str(integrity.get("symmetry_claim", "")).startswith("none") is False:
            problems.append("Cprime: symmetry claim is not 'none'")
        probes = [d.get("matched_control_rotation_probe") for d in per_epoch]
        if not probes or any(p is None for p in probes):
            problems.append("Cprime: missing per-epoch rotation records")
        else:
            if not all(p["identity_rotation_bitwise_equal"] for p in probes):
                problems.append("Cprime: identity rotation not a bitwise baseline")

    if problems:
        raise SystemExit(f"cell {cell} terminal validation failed: {problems}")
    return {
        "cell": cell,
        "cell_dir": CELL_SPECS[cell]["cell_dir"],
        "terminal_receipt": str(receipt_path),
        "terminal_receipt_sha256": body_sha,
        "mode_0444": True,
        "non_symlink": True,
        "launch_final_closure_equal": True,
        "initial_state_reconciled": True,
        "normalizer_reconciled_f062506c": True,
        "budget_reconciled_33925x48": True,
        "swa_path": str(swa_path),
        "swa_sha256": swa_sha,
        "epochs_run": payload.get("epochs_run"),
        "gates_from_receipt": gates_from_receipt,
        "integrity_summary": {
            k: integrity.get(k) for k in (
                "num_heads", "symmetry_statement", "symmetry_claim",
                "carrier_normalization", "augmentation_law", "matched_to_arm_c",
                "parameter_disclosure", "eval_policy",
            ) if k in integrity
        },
    }


# ---- the carrier-aware scoring engines (verbatim copies + carrier lines) ----
def score_last_bin_carrier(model, dataset, starts_by_session, device, scorer,
                           output_scale: float = 1.0, carriers=None,
                           eval_batch_size: int = 128):
    """VERBATIM copy of run_a2_matched_rescore.score_last_bin with ONE addition:
    when ``carriers`` is given, ``model.set_carrier(carriers[session])`` is
    called per session (the equivariant consumer's per-session carrier view).
    With ``carriers=None`` the computation is bit-identical to the original.
    """
    per_session = []
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise SystemExit("scoring must run with autograd disabled")
        for session in sorted(starts_by_session):
            record = dataset.sessions[session]
            starts = starts_by_session[session]
            calib = torch.from_numpy(record.calib_trials.copy()).float().unsqueeze(0).to(device)
            side = torch.from_numpy(record.side_features.copy()).float().unsqueeze(0).to(device)
            if carriers is not None:
                model.set_carrier(carriers[session])
            preds, tgts = [], []
            for i in range(0, len(starts), eval_batch_size):
                chunk = starts[i : i + eval_batch_size]
                neural = (
                    torch.from_numpy(np.stack([record.neural[s : s + 50] for s in chunk]))
                    .float()
                    .to(device)
                )
                behavior = (
                    torch.from_numpy(np.stack([record.behavior[s : s + 50] for s in chunk]))
                    .float()
                    .to(device)
                )
                out = model(neural, calib_trials=calib.expand(neural.shape[0], -1, -1, -1),
                            side_features=side.expand(neural.shape[0], -1, -1))
                raw = out[0] if isinstance(out, tuple) else out
                pred = raw[:, -1, :] / output_scale
                target = behavior[:, -1, :]
                valid = (behavior[:, -1, :] != PAD_VALUE).all(dim=-1)
                preds.append(pred[valid].cpu())
                tgts.append(target[valid].cpu())
            per_session.append(
                {"session": session, "r2": scorer(torch.cat(preds), torch.cat(tgts)),
                 "n_windows": len(starts)}
            )
    return {
        "per_session": per_session,
        "mean_r2": float(np.mean([row["r2"] for row in per_session])),
        "n_sessions": len(per_session),
    }


def score_track_carrier(model, dataset, positions, device, batch_size,
                        window_size, carriers=None, cap_per_session=None):
    """VERBATIM copy of run_z4_boundary_pilot.score_track (identity_cached
    mode) with TWO additions for carrier models: per-session
    ``model.set_carrier`` and the canonical-Z4 zeros_like side for the fused
    identity path.  With ``carriers=None`` the computation (including the REAL
    side fed to compute_identity) is bit-identical to the original.
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
            carrier_for_decode = None
            if carriers is not None:
                model.set_carrier(carriers[session])
                side = torch.zeros_like(side)  # canonical Z4 fused path
                carrier_for_decode = carriers[session]
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
                if carrier_for_decode is not None:
                    prediction = model.decode_with_identity(
                        neural, identity, carrier_for_decode
                    )
                else:
                    prediction = model.decode_with_identity(neural, identity)
                valid = (behavior != PAD_VALUE).all(dim=-1)
                predictions.append(prediction[valid].cpu())
                targets.append(behavior[valid].cpu())
            per_session.append(
                {"session": session, "r2": _matched_session_r2(
                    torch.cat(predictions), torch.cat(targets)),
                 "n_windows_scored": len(starts)}
            )
    return {
        "mean_r2": float(np.mean([row["r2"] for row in per_session])),
        "per_session": per_session,
        "n_windows_scored": int(sum(row["n_windows_scored"] for row in per_session)),
        "n_sessions": len(per_session),
        "equal_weight_per_session": True,
    }


_MATCHED_SCORER = None


def _matched_session_r2(predictions, targets):
    global _MATCHED_SCORER
    if _MATCHED_SCORER is None:
        _MATCHED_SCORER = _load_module(
            "tfpd_lane_matched_scorer_score", ROOT / "src/tfpd_lane/matched_scorer.py"
        )
    return _MATCHED_SCORER.session_r2(predictions, targets)


def all_positions(dataset, starts_by_session):
    positions = []
    for idx, (session, _start) in enumerate(dataset.window_indices):
        if session in starts_by_session:
            positions.append(idx)
    return positions


def annotate_n_windows(block: dict) -> dict:
    for row in block["per_session"]:
        row.setdefault("n_windows", row.get("n_windows_scored"))
    return block


# ---- gates + contrasts ------------------------------------------------------
def gate_result(stats: dict) -> dict:
    return {
        "rule": "external governing mean paired delta >= +0.03 AND >= 10/15 positive",
        "mean_delta": stats["mean"],
        "n_positive": stats["n_positive"],
        "n_total": stats["n_total"],
        "mean_gate_pass": bool(stats["mean"] >= 0.03),
        "positive_gate_pass": bool(stats["n_positive"] >= 10),
        "gate_pass": bool(stats["mean"] >= 0.03 and stats["n_positive"] >= 10),
        "bootstrap_ci_descriptive_only": stats["bootstrap_95_interval"],
    }


def date_bucket(session: str) -> str:
    if session == SEPARATE_SESSION:
        return "separate_20141203"
    return "2014" if "-2014" in session else ("2015" if "-2015" in session else "other")


# ---- eval-surface carrier views --------------------------------------------
def nwb_path_for_session(session: str, a2) -> Path:
    root = a2.SUBC_DATA_ROOT if session.startswith("sub-C") else a2.SUBM_DATA_ROOT
    path = Path(root) / f"{session}_behavior+ecephys.nwb"
    if not path.is_file():
        raise SystemExit(f"eval session NWB missing: {path}")
    return path


def build_surface_carriers(equivariant, control, a2, surfaces, side_mean, side_std,
                           m_scale: float, want_control: bool):
    """Raw-T4 carrier views for the EVAL sessions (same uncached rule as the
    sealed authority; m/b z-scores from the source-normalized side rows).

    Returns (carrier_views, control_views): {surface: {session: bundle}}.
    """
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    side_mean = np.asarray(side_mean, dtype=np.float64)
    side_std = np.asarray(side_std, dtype=np.float64)
    views: dict[str, dict] = {}
    control_views: dict[str, dict] = {}
    for surface, (ds, _starts) in surfaces.items():
        views[surface] = {}
        control_views[surface] = {}
        for session, record in sorted(ds.sessions.items()):
            raw, _meta = compute_unit_side_features_uncached(
                nwb_path_for_session(session, a2),
                feature_group="t4", pool_size=30, bin_size_ms=20, window_size=50,
                trial_result_filter="R", signal_view="sua",
            )
            raw = np.asarray(raw, dtype=np.float64)
            side = np.asarray(record.side_features)
            if raw.shape[0] != side.shape[0]:
                raise SystemExit(
                    f"{session}: raw T4 rows {raw.shape[0]} != side rows {side.shape[0]}"
                )
            renorm = ((raw - side_mean) / side_std).astype(np.float32)
            delta = float(np.abs(renorm - side).max())
            if delta > 1e-6:
                raise SystemExit(
                    f"{session}: source-normalized roundtrip drift ({delta})"
                )
            m_norm = torch.from_numpy(np.ascontiguousarray(side[:, 2], dtype=np.float32))
            b_norm = torch.from_numpy(np.ascontiguousarray(side[:, 3], dtype=np.float32))
            raw_t = torch.from_numpy(raw)
            views[surface][session] = equivariant.CarrierBundle.from_raw(
                raw_t, m_norm, b_norm, m_scale,
            )
            if want_control:
                control_views[surface][session] = (
                    control.MatchedControlCarrierBundle.from_raw(
                        raw_t,
                        torch.from_numpy(side_mean[0:2].copy()),
                        torch.from_numpy(side_std[0:2].copy()),
                        m_norm, b_norm, m_scale,
                    )
                )
    return views, control_views


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/equivariant_v1/score_b_c")
    parser.add_argument("--authorize-target", default="")
    parser.add_argument("--cells", nargs="+", default=["B", "C"],
                        choices=sorted(CELL_SPECS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    authorized = args.authorize_target == AUTH_VALUE
    if not args.dry_run and not authorized:
        print(
            "external scoring requires --authorize-target " + AUTH_VALUE,
            file=sys.stderr,
        )
        return 3

    receipt_mod = _load_module("tfpd_lane_receipt", ROOT / "src/tfpd_lane/receipt.py")
    arm_common = _load_module("tfpd_lane_arm_common", ROOT / "src/tfpd_lane/arm_common.py")
    matched_scorer = _load_module(
        "tfpd_lane_matched_scorer", ROOT / "src/tfpd_lane/matched_scorer.py"
    )
    rescorer = _load_module(
        "tfpd_a2_rescorer_equivariant", ROOT / "scripts/run_a2_matched_rescore.py"
    )
    pilot = _load_module(
        "tfpd_z4_pilot_equivariant", ROOT / "scripts/run_z4_boundary_pilot.py"
    )
    step0c = _load_module(
        "tfpd_subpop_step0c_score", ROOT / "scripts/run_subpop_step0c.py"
    )
    equivariant = _load_module(
        "tfpd_lane_equivariant", ROOT / "src/tfpd_lane/equivariant_cell.py"
    )
    control = _load_module(
        "tfpd_lane_equivariant_control", ROOT / "src/tfpd_lane/equivariant_control.py"
    )

    # ---- landed cells + sealed references, verified BEFORE any data open ----
    provenance: dict[str, dict] = {}
    not_landed: dict[str, dict] = {}
    for cell in args.cells:
        record = validate_equivariant_terminal(cell, arm_common)
        if record is None:
            not_landed[cell] = {
                "note": (
                    f"terminal receipt not present yet ({CELL_ROOT}/"
                    f"{CELL_SPECS[cell]['cell_dir']}); cell omitted — never "
                    f"scored from a partial artifact"
                )
            }
        else:
            provenance[cell] = record
    landed = [c for c in args.cells if c in provenance]
    checkpoint_integrity = step0c.verify_checkpoints(arm_common.sha256_file)
    torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
    authority_payload = torch.load(CARRIER_AUTHORITY, map_location="cpu",
                                   weights_only=False)
    m_scale = float(authority_payload["m_scale"])

    plan = {
        "schema": "tfpd_equivariant_score_v1",
        "status": "PLAN",
        "cells_requested": list(args.cells),
        "cells_landed": landed,
        "cells_not_landed": sorted(not_landed),
        "surfaces": ["within 6 sub-C dev", "external 15 sub-M"],
        "granularity": {
            "governing_last_bin": (
                "GOVERNING: last timestep of window, variance-weighted R2, "
                "equal session weight"
            ),
            "diagnostic_full_window": (
                "DIAGNOSTIC ONLY: cannot rescue a failed governing mean"
            ),
        },
        "references_scored_live": ["armA_swa", "D_swa"],
        "contrasts": sorted(CONTRAST_GATES),
        "pending_until_cprime_lands": [
            k for k in CONTRAST_GATES if "Cprime" in k and k != "Cprime_minus_A"
        ] + (["Cprime_minus_A"] if "Cprime" not in landed else []),
        "authorized": authorized,
        "output_root": str(args.output_root),
    }
    if args.dry_run:
        print(json.dumps({
            **plan,
            "status": "DRY_RUN__NO_NWB_OPENED",
            "landed_cell_provenance": {
                cell: {
                    "terminal_receipt": provenance[cell]["terminal_receipt"],
                    "terminal_receipt_sha256": provenance[cell]["terminal_receipt_sha256"],
                    "swa_sha256": provenance[cell]["swa_sha256"],
                    "epochs_run": provenance[cell]["epochs_run"],
                } for cell in landed
            },
            "not_landed": not_landed,
            "reference_checkpoint_integrity": {
                k: {"sha256": v["sha256"], "bound_from": v.get("bound_from")}
                for k, v in checkpoint_integrity.items()
                if isinstance(v, dict)
            },
        }, indent=1))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3
    out_dir = Path(args.output_root)
    if out_dir.exists():
        print(f"fresh output root required: {out_dir}", file=sys.stderr)
        return 2
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True)
    closure = receipt_mod.source_closure(ROOT, BOUND_PATTERNS)

    # ---- surfaces (exact A2-matched convention, source normalizers) ---------
    import mc_maze.a2_matched_subject_shift_v2_core as a2

    within_surface, external_surface = rescorer.build_surfaces(args, a2, "t4")
    surfaces = {"within": within_surface, "external": external_surface}
    from mc_maze.unit_side_features import fit_side_feature_stats

    train_paths, _v, _n = a2.active_source_session_paths()
    side_mean, side_std = fit_side_feature_stats(
        train_paths, feature_group="t4", pool_size=30,
        cache_dir=a2.SOURCE_CACHE_ROOT, signal_view="sua",
    )
    want_control = "Cprime" in landed
    carrier_views, control_views = build_surface_carriers(
        equivariant, control, a2, surfaces, side_mean, side_std, m_scale,
        want_control,
    )
    print(json.dumps({
        "surface": "within", "n_sessions": len(surfaces["within"][1]),
        "sessions": sorted(surfaces["within"][1])}), flush=True)
    print(json.dumps({
        "surface": "external", "n_sessions": len(surfaces["external"][1]),
        "sessions": sorted(surfaces["external"][1])}), flush=True)

    # ---- the model table ----------------------------------------------------
    model_specs: dict[str, dict] = {
        "armA_swa": {"path": step0c.CHECKPOINTS["armA"]["path"], "cell": None},
        "D_swa": {"path": step0c.CHECKPOINTS["D"]["path"], "cell": None},
    }
    for cell in landed:
        model_specs[f"{cell}_swa"] = {
            "path": provenance[cell]["swa_path"], "cell": cell,
        }

    results: dict[str, dict] = {}
    integrity_blocks: dict[str, dict] = {}
    engine_parity: dict[str, dict] = {}
    for name, spec in model_specs.items():
        path = Path(spec["path"])
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {(k[len("model."):] if k.startswith("model.") else k): v
                 for k, v in state.items()}
        cell = spec["cell"]
        if cell is None:
            model = step0c.build_step0c_model(seed=42)
            model.load_state_dict(state, strict=True)
            carriers = None
            graph_note = "the standard lane loading path (step0c.build_step0c_model)"
        elif cell == "B":
            model = step0c.build_step0c_model(seed=42)
            model.load_state_dict(state, strict=True)
            carriers = None
            graph_note = (
                "sealed Cell-D graph via the standard lane loading path; the "
                "dynamic-dropout flag is train-mode-only and carries no "
                "parameters (cross-checked bitwise vs the D-shaped build)"
            )
        elif cell == "C":
            model = equivariant.build_equivariant_model(seed=42)
            equivariant.freeze_inactive_consumer_modules(model)
            model.load_state_dict(state, strict=True)
            carriers = None  # set per surface below
            graph_note = (
                "the frozen equivariant builder + sealed raw-T4 carrier "
                "authority (m_scale frozen); eval = plain forward"
            )
        else:  # Cprime
            model = control.build_matched_control_model(seed=42)
            equivariant.freeze_inactive_consumer_modules(model)
            model.load_state_dict(state, strict=True)
            carriers = None
            graph_note = (
                "the frozen matched-control builder + sealed raw-T4 carrier "
                "authority; eval = plain forward (no covariant view)"
            )
        del state
        state_before = arm_common.state_sha256(model)
        model.to(device).eval()

        governing = {}
        diagnostic = {}
        for surface, (ds, starts) in surfaces.items():
            surface_carriers = None
            if cell == "C":
                surface_carriers = carrier_views[surface]
            elif cell == "Cprime":
                surface_carriers = control_views[surface]
            governing[surface] = score_last_bin_carrier(
                model, ds, starts, device, matched_scorer.session_r2,
                output_scale=1.0, carriers=surface_carriers,
            )
            diagnostic[surface] = annotate_n_windows(score_track_carrier(
                model, ds, all_positions(ds, starts), device, 128, 50,
                carriers=surface_carriers,
            ))

        # same-engine parity proof: for a carrier-free model the copies must
        # reproduce the lane engines bit-for-bit (asserted live for arm B)
        if cell == "B":
            ds, starts = surfaces["within"]
            mine = score_last_bin_carrier(
                model, ds, starts, device, matched_scorer.session_r2,
                output_scale=1.0, carriers=None,
            )
            theirs = rescorer.score_last_bin(
                lambda n, c, s, _m=model: _m(n, calib_trials=c, side_features=s),
                ds, starts, device, matched_scorer.session_r2, output_scale=1.0,
            )
            last_bin_equal = all(
                a["r2"] == b["r2"] and a["n_windows"] == b["n_windows"]
                for a, b in zip(mine["per_session"], theirs["per_session"])
            )
            mine_track = annotate_n_windows(score_track_carrier(
                model, ds, all_positions(ds, starts), device, 128, 50,
                carriers=None,
            ))
            theirs_track = pilot.score_track(
                model, ds, all_positions(ds, starts), device, 128, 50,
                cap_per_session=None, forward_mode="identity_cached",
            )
            track_equal = all(
                a["r2"] == b["r2"] for a, b in zip(
                    mine_track["per_session"], theirs_track["per_session"]
                )
            )
            if not (last_bin_equal and track_equal):
                raise SystemExit(
                    f"engine parity failure for {name}: last_bin={last_bin_equal} "
                    f"track={track_equal}"
                )
            engine_parity[name] = {
                "last_bin_bitwise_equal_per_session": last_bin_equal,
                "full_window_bitwise_equal_per_session": track_equal,
                "surface": "within",
                "note": (
                    "carrier-free models: the carrier-aware copies reproduce "
                    "the lane engines exactly, so all arms share one engine"
                ),
            }
            # D-shaped build cross-check (the standard path IS the eval path)
            d_shaped = _load_module(
                "tfpd_lane_pop_robust_score_eq", ROOT / "src/tfpd_lane/pop_robust.py"
            ).build_population_robustness_model(seed=42, cell="D")
            d_shaped.load_state_dict(model.state_dict(), strict=True)
            d_shaped.to(device).eval()
            record = surfaces["within"][0].sessions[
                sorted(surfaces["within"][1])[0]
            ]
            probe_neural = (
                torch.from_numpy(record.neural[: 2 * 50]).float()
                .view(2, 50, -1).to(device)
            )
            probe_calib = (
                torch.from_numpy(record.calib_trials.copy()).float()
                .unsqueeze(0).expand(2, -1, -1, -1).to(device)
            )
            probe_side = (
                torch.from_numpy(record.side_features.copy()).float()
                .unsqueeze(0).expand(2, -1, -1).to(device)
            )
            with torch.no_grad():
                out_standard, _ = model(probe_neural, calib_trials=probe_calib,
                                        side_features=probe_side)
                out_dshaped, _ = d_shaped(probe_neural, calib_trials=probe_calib,
                                          side_features=probe_side)
            engine_parity[name]["d_shaped_build_bitwise_equal"] = bool(
                torch.equal(out_standard, out_dshaped)
            )
            if not engine_parity[name]["d_shaped_build_bitwise_equal"]:
                raise SystemExit("arm B: D-shaped build forward mismatch")
            del d_shaped

        state_after = arm_common.state_sha256(model)
        if state_before != state_after:
            raise SystemExit(f"scoring mutated state: {name}")
        results[name] = {
            "governing_last_bin": governing,
            "diagnostic_full_window": diagnostic,
        }
        block = {
            "path": str(path),
            "sha256": (
                provenance[cell]["swa_sha256"] if cell
                else checkpoint_integrity[
                    "armA" if name == "armA_swa" else "D"
                ]["sha256"]
            ),
            "loaded_graph": graph_note,
            "strict_load": True,
            "state_unchanged_during_scoring": True,
            "target_updates_gradients_or_optimizer_steps": 0,
            "grads_all_none_after": all(
                p.grad is None for p in model.parameters()
                if not isinstance(p, torch.nn.parameter.UninitializedParameter)
            ),
            "eval_policy": (
                "plain forward; no rotation, no augmentation"
                if cell in ("C", "Cprime")
                else "plain forward (the unaugmented Cell-D path)"
            ),
        }
        if cell:
            block.update({
                "cell": cell,
                "terminal_receipt": provenance[cell]["terminal_receipt"],
                "terminal_receipt_sha256": provenance[cell]["terminal_receipt_sha256"],
                "launch_final_closure_equal": True,
                "epochs_run": provenance[cell]["epochs_run"],
                "gates_from_training_receipt": provenance[cell][
                    "gates_from_receipt"
                ],
            })
        else:
            block.update({
                "bound_from": checkpoint_integrity[
                    "armA" if name == "armA_swa" else "D"
                ]["bound_from"],
                "terminal_receipt": checkpoint_integrity[
                    "armA" if name == "armA_swa" else "D"
                ]["terminal_receipt"],
            })
        integrity_blocks[name] = block
        print(json.dumps({
            "model": name,
            "within_gov": round(governing["within"]["mean_r2"], 6),
            "external_gov": round(governing["external"]["mean_r2"], 6),
            "within_full": round(diagnostic["within"]["mean_r2"], 6),
            "external_full": round(diagnostic["external"]["mean_r2"], 6),
        }), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- paired contrasts (same-engine live pairing) -------------------------
    def per_session(model, surface, granularity="governing_last_bin"):
        return {
            row["session"]: row["r2"]
            for row in results[model][granularity][surface]["per_session"]
        }

    def paired(model_a, model_b, surface, granularity="governing_last_bin"):
        table_a = per_session(model_a, surface, granularity)
        table_b = per_session(model_b, surface, granularity)
        sessions = sorted(set(table_a) & set(table_b))
        if len(sessions) != len(table_a) or len(sessions) != len(table_b):
            raise SystemExit(f"session-roster mismatch: {model_a} vs {model_b}")
        deltas = [table_a[s] - table_b[s] for s in sessions]
        stats = matched_scorer.paired_session_stats(deltas, seed=42, n_boot=10000)
        stats["contrast"] = f"{model_a} - {model_b} ({surface}, {granularity})"
        return stats

    contrast_pairs = [("D_swa", "armA_swa")]
    for cell in landed:
        contrast_pairs.append((f"{cell}_swa", "armA_swa"))
        contrast_pairs.append((f"{cell}_swa", "D_swa"))
    if "C" in landed and "B" in landed:
        contrast_pairs.append(("C_swa", "B_swa"))
    if "C" in landed and "Cprime" in landed:
        contrast_pairs.append(("C_swa", "Cprime_swa"))
    if "Cprime" in landed and "B" in landed:
        contrast_pairs.append(("Cprime_swa", "B_swa"))
    contrasts: dict[str, dict] = {}
    for granularity in ("governing_last_bin", "diagnostic_full_window"):
        for model_a, model_b in contrast_pairs:
            for surface in ("within", "external"):
                contrasts[f"{model_a}_minus_{model_b}_{surface}_{granularity}"] = (
                    paired(model_a, model_b, surface, granularity)
                )

    # governing gate evaluations (external governing only, per pre-registration)
    gate_key = {
        ("B_swa", "armA_swa"): "B_minus_A",
        ("C_swa", "armA_swa"): "C_minus_A",
        ("Cprime_swa", "armA_swa"): "Cprime_minus_A",
    }
    gates: dict[str, dict] = {}
    for (model_a, model_b), label in gate_key.items():
        key = f"{model_a}_minus_{model_b}_external_governing_last_bin"
        if key in contrasts:
            gates[label] = {
                "reading": CONTRAST_GATES[label],
                "stats": contrasts[key],
                "gate": gate_result(contrasts[key]),
            }
    gates["C_minus_B"] = {
        "reading": CONTRAST_GATES["C_minus_B"],
        "stats": contrasts.get("C_swa_minus_B_swa_external_governing_last_bin"),
        "note": "descriptive; no gate",
    }
    pending = {
        label: {
            "reading": CONTRAST_GATES[label],
            "status": "PENDING",
            "note": (
                "Cprime has not landed; score it with --cells Cprime "
                "--output-root results/equivariant_v1/score_cprime"
            ),
        }
        for label in ("C_minus_Cprime", "Cprime_minus_A", "Cprime_minus_B")
        if "Cprime" not in landed
    }

    # ---- date blocks + the separately reported session (external, governing)
    n_windows_reference = {
        row["session"]: row.get("n_windows")
        for row in results["armA_swa"]["governing_last_bin"]["external"]["per_session"]
    }
    blocks: dict[str, dict] = {}
    armA_table = per_session("armA_swa", "external")
    d_table = per_session("D_swa", "external")
    for model in model_specs:
        table = per_session(model, "external")
        for session, value in table.items():
            bucket = date_bucket(session)
            blocks.setdefault(model, {}).setdefault(bucket, {})[session] = value
        for bucket, entries in blocks[model].items():
            sessions = sorted(entries)
            values = [entries[s] for s in sessions]

            def _delta(reference):
                return [entries[s] - reference[s] for s in sessions]

            blocks[model][bucket] = {
                "n_sessions": len(sessions),
                "block_mean": float(np.mean(values)),
                "paired_delta_vs_armA_mean": float(np.mean(_delta(armA_table))),
                "paired_delta_vs_armA_n_positive": int(
                    sum(d > 0 for d in _delta(armA_table))
                ),
                "paired_delta_vs_D_mean": float(np.mean(_delta(d_table))),
                "paired_delta_vs_D_n_positive": int(
                    sum(d > 0 for d in _delta(d_table))
                ),
                "window_share_of_external": float(
                    sum(n_windows_reference.get(s, 0) for s in sessions)
                    / max(sum(n_windows_reference.values()), 1)
                ),
                "sessions": sessions,
            }

    receipt = {
        "schema": "tfpd_equivariant_score_v1",
        "status": "SCORED",
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells_landed": landed,
        "cells_not_landed": not_landed,
        "granularity": plan["granularity"],
        "models": results,
        "integrity": integrity_blocks,
        "engine_parity": engine_parity,
        "contrasts": contrasts,
        "gates": gates,
        "gates_pending_until_cprime_lands": pending,
        "contrast_readings": CONTRAST_GATES,
        "date_blocks_external_governing": blocks,
        "separate_session": SEPARATE_SESSION,
        "carrier_authority": {
            "path": str(CARRIER_AUTHORITY),
            "sha256": arm_common.sha256_file(CARRIER_AUTHORITY),
            "m_scale": m_scale,
            "eval_rule": (
                "raw T4 recomputed per eval session with the sealed authority's "
                "uncached rule; source-normalized roundtrip asserted; m/b "
                "z-scores from the source-normalized side rows"
            ),
        },
        "normalizers": {
            "behavior_semantic_sha256": "f062506c (asserted by build_surfaces)",
            "side_feature_stats": "source-fit (fit_side_feature_stats on the strict-27)",
        },
        "disclosures": {
            "gpu_training": 0,
            "target_updates_gradients_or_optimizer_steps": 0,
            "scoring_only": True,
            "external_development_target_authorized": authorized,
            "bootstrap_ci_descriptive_only": True,
            "hand_copied_numbers": "none — every reference scored live",
        },
        "source_closure": closure,
        "environment": {
            "device": str(device), "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }
    receipt_mod.write_receipt_transactionally(out_dir / "score_receipt.json", receipt)
    print(json.dumps({
        "status": "SCORED",
        "cells_landed": landed,
        **{
            f"{label}_external_gov": (
                round(gates[label]["stats"]["mean"], 4),
                f"{gates[label]['stats']['n_positive']}/"
                f"{gates[label]['stats']['n_total']}",
                (gates[label]["gate"]["gate_pass"]
                 if "gate" in gates[label] else "descriptive (no gate)"),
            )
            for label in gates if gates[label].get("stats")
        },
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
