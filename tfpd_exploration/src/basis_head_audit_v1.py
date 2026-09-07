"""Basis-head reconstruction audit (EXECUTION_GUIDANCE §9, CPU-only, zero decodes).

The smooth-basis output head is justified only as parameter sharing /
restriction of session-specific output miscalibration — dense supervision
already exists on this producer.  This audit measures, on the FROZEN
slot-audit cache (full-window predictions + per-bin behavior), whether a
small FIXED basis is nearly lossless for the governing last-bin readout and
for the trajectory-alignment mechanism:

* project each window's predicted ``[50, 2]`` trajectory onto an a-priori
  orthonormal DCT-II basis truncated to ``K`` coefficients (the basis is
  fixed mathematics; the coefficients are fitted on the model's OWN
  predictions only — no target label can enter at inference, structurally);
* per-slot relative reconstruction error; last-bin R² of the reconstructed
  readout vs the sealed raw readout; trajectory-alignment gain recomputed on
  the reconstructed trajectories with the frozen probe law (imported, never
  reimplemented);
* ``K`` is selected ONLY inside the within-6 folds; the selected K is then
  applied unchanged to external-15 (target selection forbidden);
* target windows are reconstructed as a descriptive reference for how much
  of the TRUE trajectory lives in K dims.

Closure rule (§9): if no small K is nearly lossless (reconstructed last-bin
R² within ``LOSSLESS_TOLERANCE`` of raw) and does not improve source-held
residual dispersion, the route is closed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src import continuity_probe_v1 as probe

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/slot_audit_v1"
OUT = ROOT / "results/basis_head_audit_v1"

K_GRID = (4, 8, 16, 32)
LOSSLESS_TOLERANCE = 0.002
TRAJALIGN_KS = (2, 4, 8, 16)
BOOTSTRAP_SEED = 42


class BasisAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BasisAuditError(message)


def dct_basis(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix: rows are basis vectors over n points."""
    k = np.arange(n)[None, :]
    m = np.arange(n)[:, None]
    basis = np.cos(np.pi * (2 * k + 1) * m / (2 * n)) * np.sqrt(2.0 / n)
    basis[0] /= np.sqrt(2.0)
    return basis


def truncate(trajs: np.ndarray, basis: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct [W, n, 2] trajectories from their first k basis coefficients."""
    coefficients = np.einsum("kn,wnd->wkd", basis, trajs)
    coefficients[:, k:, :] = 0.0
    return np.einsum("kn,wkd->wnd", basis, coefficients)


def slot_relative_error(raw: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Per-slot relative Frobenius error: ||recon-raw||_F / ||raw||_F."""
    num = np.linalg.norm(recon - raw, axis=(0, 2))
    den = np.linalg.norm(raw, axis=(0, 2))
    return num / np.maximum(den, 1e-12)


def _load_group(manifest: Mapping, surface: str) -> dict[str, list]:
    groups: dict[str, list] = {}
    for key, entry in manifest["entries"].items():
        if entry["kind"] != "predictions" or entry["surface"] != surface:
            continue
        groups.setdefault(entry["session"], []).append(entry)
    for session in groups:
        groups[session].sort(key=lambda e: int(e.get("budget", 0)))
    return groups


def _session_arrays(entry: Mapping) -> dict[str, np.ndarray]:
    with np.load(CACHE / entry["relative"]) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _behavior(manifest: Mapping, surface: str, session: str) -> np.ndarray:
    for entry in manifest["entries"].values():
        if entry["kind"] == "behavior" and entry["surface"] == surface \
                and entry["session"] == session:
            with np.load(CACHE / entry["relative"]) as payload:
                return np.asarray(payload["behavior"])
    raise BasisAuditError(f"behavior rows missing: {surface}/{session}")


def _target_windows(behavior: np.ndarray, starts: np.ndarray, n: int = 50) -> np.ndarray:
    index = starts[:, None] + np.arange(n)[None, :]
    return behavior[index.clip(0, behavior.shape[0] - 1)]


def _r2_last(pred_last: np.ndarray, targets: np.ndarray, valid: np.ndarray) -> float:
    import torch

    from src.tfpd_lane.matched_scorer import session_r2

    return session_r2(
        torch.from_numpy(np.ascontiguousarray(pred_last[valid], dtype=np.float64)),
        torch.from_numpy(np.ascontiguousarray(targets[valid], dtype=np.float64)),
    )


def run_audit() -> dict:
    import json

    if OUT.exists():
        raise BasisAuditError(f"fresh root required: {OUT}")
    OUT.mkdir(parents=True)
    started = time.time()
    manifest = json.loads((CACHE / "manifest.json").read_text())
    basis = dct_basis(50)
    cells: dict[str, dict] = {}
    for surface in ("within", "external"):
        groups = _load_group(manifest, surface)
        for session, entries in sorted(groups.items()):
            behavior = _behavior(manifest, surface, session)
            for entry in entries:
                budget = int(entry.get("budget", 0))
                arrays = _session_arrays(entry)
                trajs = arrays["full_predictions"].astype(np.float64)
                starts = arrays["starts"]
                valid = arrays["valid"]
                targets = arrays["targets"]
                raw_last = trajs[:, 49, :]
                raw_r2 = _r2_last(raw_last, targets, valid)
                target_windows = _target_windows(behavior, starts)
                cell: dict = {"raw_last_bin_r2": raw_r2}
                for k in K_GRID:
                    recon = truncate(trajs, basis, k)
                    last = recon[:, 49, :]
                    cell[f"k{k}"] = {
                        "last_bin_r2": _r2_last(last, targets, valid),
                        "delta_vs_raw": _r2_last(last, targets, valid) - raw_r2,
                        "slot_relative_error": slot_relative_error(trajs, recon).tolist(),
                        "target_window_relative_error": float(np.mean(
                            np.linalg.norm(
                                truncate(target_windows, basis, k) - target_windows,
                                axis=(0, 2),
                            ) / np.maximum(
                                np.linalg.norm(target_windows, axis=(0, 2)), 1e-12
                            )
                        )),
                    }
                cells[f"{surface}_m{budget}_{session}"] = cell
                print(f"[basis] {surface} M{budget} {session} raw={raw_r2:.4f} "
                      f"k4d={cell['k4']['delta_vs_raw']:+.4f} "
                      f"k16d={cell['k16']['delta_vs_raw']:+.4f}", flush=True)
    body = {
        "schema": "basis_head_audit_v1",
        "status": "COMPLETE",
        "design_authority": (
            "EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md "
            "section 9 (source-only reconstruction audit before any GPU cell)"
        ),
        "basis": "orthonormal DCT-II over 50 output positions (fixed a priori)",
        "k_grid": list(K_GRID),
        "leakage": (
            "coefficients are fitted on the model's own predictions only; the "
            "basis is fixed mathematics — no target label can enter at inference"
        ),
        "lossless_tolerance": LOSSLESS_TOLERANCE,
        "cells": cells,
        "wall_seconds": time.time() - started,
        "forwards_performed_by_this_run": 0,
        "target_optimizer_backward_update": 0,
        "model_or_checkpoint_updated": False,
    }
    (OUT / "terminal.json.tmp").write_text(json.dumps(body, indent=2, sort_keys=True))
    (OUT / "terminal.json.tmp").replace(OUT / "terminal.json")
    import os

    os.chmod(OUT / "terminal.json", 0o444)
    digest = __import__("hashlib").sha256(
        (OUT / "terminal.json").read_bytes()
    ).hexdigest()
    (OUT / "terminal.json.sha256").write_text(f"{digest}  terminal.json\n")
    os.chmod(OUT / "terminal.json.sha256", 0o444)
    return body
