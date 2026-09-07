"""Evaluation for FABLE TKD M2 runs (Wave 2, spec section 3)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as screen_core

from . import plan
from .model import TKD
from .train import (
    _assert_tf32_off,
    _build_arm_model,
    gather_windows,
    load_session_plane,
    post30_starts,
    state_dict_sha256,
)


def run_dir_for(repo_root: Path, arm: str, seed: int) -> Path:
    """The LATEST run directory for (arm, seed): the highest _a<n> retry
    suffix when retries exist (the base name is attempt 1).  Latest-wins is
    required so a retried pilot never evaluates a stale earlier checkpoint."""
    runs_root = plan.result_root(repo_root) / "runs"
    candidates: list[tuple[int, Path]] = []
    base = runs_root / f"{arm}_s{seed}"
    if (base / "train.json").is_file():
        candidates.append((1, base))
    for path in runs_root.glob(f"{arm}_s{seed}_a*"):
        if path.is_dir() and (path / "train.json").is_file():
            candidates.append((int(path.name.rsplit("_a", 1)[1]), path))
    plan.require(bool(candidates), f"no run directory for {arm} s{seed}")
    return max(candidates, key=lambda item: item[0])[1]


def _score_surface(
    model: TKD,
    dataset: Any,
    sessions: tuple[str, ...],
    fits: dict[str, Any],
    device: torch.device,
    *,
    surface: str,
    batch_size: int = plan.EVAL_BATCH,
) -> dict[str, Any]:
    model.eval()
    per_session: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for session in sessions:
            fit = fits[session]
            if surface == "within_post30":
                starts = post30_starts(dataset, session)
            else:
                starts = data_starts(dataset, session)
            tensor_t = torch.from_numpy(np.ascontiguousarray(fit.t)).to(device)
            tensor_rho = torch.from_numpy(np.ascontiguousarray(fit.rho)).to(device)
            predictions: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for offset in range(0, starts.size, batch_size):
                chunk = starts[offset : offset + batch_size]
                windows, window_targets = gather_windows(dataset, session, chunk)
                prediction = model(
                    torch.from_numpy(windows).to(device), tensor_t, tensor_rho
                )
                predictions.append(prediction.cpu().numpy())
                targets.append(window_targets)
            prediction_np = np.ascontiguousarray(np.concatenate(predictions, axis=0))
            target_np = np.ascontiguousarray(np.concatenate(targets, axis=0))
            plan.require(prediction_np.shape == target_np.shape, "eval shape drift")
            r2 = screen_core.variance_weighted_r2(target_np, prediction_np)
            per_session[session] = r2
            rows.append(
                {
                    "session": session,
                    "window_count": int(starts.size),
                    "ordered_window_starts_sha256": screen_core.array_sha256(starts),
                    "target_sha256": screen_core.array_sha256(target_np),
                    "prediction_sha256": screen_core.array_sha256(prediction_np),
                    "r2": r2,
                    "raw_t4_sha256": fit.evidence["raw_t4_sha256"],
                    "rho_sha256": fit.evidence["rho_sha256"],
                }
            )
    summary = screen_core.summarize_sessions(per_session)
    return {"surface": surface, "per_session_r2": per_session, "rows": rows, **summary}


def data_starts(dataset: Any, session: str) -> np.ndarray:
    """All official query windows for a session (external face)."""
    from . import data as data_module

    return data_module.session_window_starts(dataset, session)


def _static_alpha_summary(
    model: TKD,
    dataset: Any,
    sessions: tuple[str, ...],
    fits: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Static attention mass summary + bit-exact cache assertion (spec 3.4).

    The cached static forward is bit-exact only for the epsilon='zero' arm
    (arm A); learnable-epsilon arms report the t-only static alpha and the
    trained epsilon value without the cache assertion.
    """
    epsilon_value = float(model.epsilon.detach().cpu())
    payload: dict[str, Any] = {
        "epsilon_value": epsilon_value,
        "epsilon_mode": model.epsilon_mode,
        "static_cache_bitexact_asserted": model.epsilon_mode == "zero",
    }
    summaries: dict[str, Any] = {}
    for session in sessions:
        fit = fits[session]
        tensor_t = torch.from_numpy(np.ascontiguousarray(fit.t)).to(device)
        tensor_rho = torch.from_numpy(np.ascontiguousarray(fit.rho)).to(device)
        with torch.no_grad():
            key = model.compute_key(tensor_t)
            scores = torch.einsum("qd,nd->qn", model.queries, key) / 8.0
            alpha = torch.softmax(scores, dim=-1)
            entry: dict[str, Any] = {}
            if model.epsilon_mode == "zero":
                cached = model(
                    _probe_batch(dataset, session, fits, device),
                    tensor_t,
                    tensor_rho,
                    static={"key": key, "alpha": alpha},
                )
                recomputed = model(
                    _probe_batch(dataset, session, fits, device), tensor_t, tensor_rho
                )
                plan.require(bool(torch.equal(cached, recomputed)),
                             f"static cache not bit-exact for {session}")
                entry["cache_bitexact"] = True
        alpha_np = alpha.detach().cpu().numpy().astype(np.float64)
        row_sums = alpha_np.sum(axis=1)
        plan.require(float(np.abs(row_sums - 1.0).max()) < 1.0e-5,
                     "alpha rows do not sum to 1")
        hard = np.argmax(alpha_np, axis=0)
        entry.update(
            {
                "soft_bin_mass": alpha_np.sum(axis=0).tolist(),
                "hard_bin_counts": np.bincount(
                    hard, minlength=plan.N_QUERIES
                ).astype(np.int64).tolist(),
                "per_query_max_weight": alpha_np.max(axis=1).tolist(),
            }
        )
        summaries[session] = entry
    payload["per_session"] = summaries
    return payload


def _probe_batch(
    dataset: Any, session: str, fits: dict[str, Any], device: torch.device
) -> torch.Tensor:
    """First 256 query windows of a session, cached on device for the probe."""
    cache_key = ("__probe__", session)
    probe_cache = getattr(_probe_batch, "_cache", None)
    if probe_cache is None:
        probe_cache = {}
        _probe_batch._cache = probe_cache
    if cache_key not in probe_cache:
        starts = data_starts(dataset, session)[:256]
        windows, _targets = gather_windows(dataset, session, starts)
        probe_cache[cache_key] = torch.from_numpy(windows)
    return probe_cache[cache_key].to(device)


def evaluate_run(
    repo_root: Path,
    arm: str,
    seed: int,
    device: torch.device,
    *,
    plane: dict[str, Any] | None = None,
    receipt_name: str = "eval.json",
    run_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo_root = Path(repo_root)
    run_dir = Path(run_dir) if run_dir is not None else run_dir_for(repo_root, arm, seed)
    train_path = run_dir / "train.json"
    plan.require(train_path.is_file(), f"missing train receipt: {train_path}")
    plan.verify_sidecar(train_path)
    train_receipt = json.loads(train_path.read_text(encoding="utf-8"))
    best_path = run_dir / "best.pt"
    plan.require(best_path.is_file(), f"missing checkpoint: {best_path}")
    best_bytes = best_path.read_bytes()
    file_sha = hashlib.sha256(best_bytes).hexdigest()
    plan.require(file_sha == train_receipt["best_pt_sha256"],
                 "best.pt file sha drift vs train receipt")
    _assert_tf32_off()
    plane = plane if plane is not None else load_session_plane(repo_root)
    model = _build_arm_model(arm, plane, device)
    state = torch.load(
        __import__("io").BytesIO(best_bytes), map_location="cpu", weights_only=False
    )
    model.load_state_dict(state, strict=True)
    plan.require(state_dict_sha256(state) == train_receipt["best_state_sha256"],
                 "state_dict sha drift vs train receipt")
    model.eval()

    bundle, fits, authority = plane["bundle"], plane["fits"], plane["authority"]
    # The sealed authority pins the held-in digests; external digests are
    # recomputed and verified transitively by build_or_verify_authority's
    # full-payload bit-equality check and recorded in the receipt rows.
    for session in plan.HELDIN_SESSIONS:
        plan.require(
            fits[session].evidence["raw_t4_sha256"]
            == authority["per_session_raw_t4_sha256"][session],
            f"raw T4 digest drift for {session}",
        )

    external = _score_surface(
        model, bundle["external"], plan.EXTERNAL_SESSIONS, fits, device,
        surface="external_official_query",
    )
    within = _score_surface(
        model, bundle["within"], plan.HELDIN_SESSIONS, fits, device,
        surface="within_post30",
    )
    vs_ref = screen_core.paired_contrast(
        external["per_session_r2"], plan.REF_EXTERNAL_PER_SESSION
    )
    alpha_summary = _static_alpha_summary(
        model, bundle["external"], plan.EXTERNAL_SESSIONS, fits, device
    )

    receipt = {
        "schema": f"{plan.SCHEMA}:eval",
        "arm": arm,
        "seed": seed,
        "device": str(device),
        "gpu_uuid": (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"),
        "reference_chain": {
            "train_json_sha256": plan.sha256_bytes(train_path.read_bytes()),
            "best_pt_sha256": file_sha,
            "best_state_sha256": train_receipt["best_state_sha256"],
            "authority_sha256": plan.sha256_bytes(plan.receipt_body(authority)),
        },
        "pv_init_config": plan.PV_INIT_CONFIG,
        "external": {
            "equal_session_mean": external["equal_session_mean"],
            "per_session_r2": external["per_session_r2"],
            "rows": external["rows"],
        },
        "within": {
            "equal_session_mean": within["equal_session_mean"],
            "per_session_r2": within["per_session_r2"],
            "rows": within["rows"],
        },
        "contrast_vs_ref_external": vs_ref,
        "static_alpha": alpha_summary,
        "elapsed_seconds": time.monotonic() - started,
    }
    plan.atomic_receipt(run_dir / receipt_name, receipt)
    return receipt
