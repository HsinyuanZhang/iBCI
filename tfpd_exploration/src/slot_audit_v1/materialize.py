"""SLOT-AUDIT V1 materialization — one frozen CPU decode per (session, budget).

Work order section 3.  The decode is the frozen static deployment recipe
through ``p4_stream_stats.materialize_session`` + ``_decode_static`` — the
same call chain as the continuity probe's baseline row, one decode per
(session, budget) exactly like the probe (the budget changes the calibration
activity AND the T4 side, so the decode is per budget).  Surfaces
external-15 + within-6, budgets M4/M10/M30.

Parity law (the anchor): for every (surface, budget, session) the last-bin
house R2 must equal the SEALED continuity-probe baseline row
(``variance_weighted_r2``, tolerance 1e-12 — CPU-to-CPU on bit-identical
inputs, so any drift is a hard failure), ``n_windows`` must match, the FULL
[W, 50, 2] prediction SHA must be bit-exact, and the input SHAs (targets,
neural, calibration, carrier side, selection, ridge T4, identity) must
equal the sealed row field-for-field.

The full prediction tensors are cached ONCE under ``cache/slot_audit_v1/``
(digest-bound manifest + sidecar, the filter-line law, this package's own
format).  The session behavior-bin target source is cached per session with
its own parity evidence (see ``targets.behavior_source``).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import cache_store, plan, targets


class SlotAuditMaterializeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlotAuditMaterializeError(message)


def load_sealed_probe(root: Path = plan.ROOT) -> dict:
    """SHA-verified load of the sealed continuity-probe receipt (the anchor)."""
    path = Path(root) / plan.SEALED_PROBE_REL
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(
        digest == plan.SEALED_PROBE_SHA256,
        f"sealed probe receipt SHA drift: expected {plan.SEALED_PROBE_SHA256}, got {digest}",
    )
    return json.loads(path.read_text(encoding="utf-8"))


def sealed_baseline_rows(sealed_probe: dict) -> dict[tuple[str, int, str], dict]:
    rows: dict[tuple[str, int, str], dict] = {}
    for cell in sealed_probe["cells"]:
        if cell["arm"] != "baseline":
            continue
        for row in cell["sessions"]:
            key = (str(row["surface"]), int(row["budget"]), str(row["session"]))
            _require(key not in rows, f"sealed baseline row duplication: {key}")
            rows[key] = row
    _require(bool(rows), "sealed probe has no baseline rows")
    return rows


def _verify_bin_size_provenance() -> dict[str, Any]:
    """Mechanically verify the frozen loader's bin size (never guessed)."""
    from src.calibration_gap_v1 import p4_stream_stats as p4

    source = inspect.getsource(p4.materialize_session)
    _require(
        "bin_size_ms=20" in source,
        "frozen materialize_session no longer parses with bin_size_ms=20 "
        "(latency provenance drift — re-read the loader before citing ms)",
    )
    return {
        "bin_size_ms": plan.BIN_SIZE_MS,
        "provenance": plan.BIN_SIZE_PROVENANCE,
        "verified_from_frozen_source": True,
        "verification": (
            "inspect.getsource(p4_stream_stats.materialize_session) contains "
            "'bin_size_ms=20' at materialization time"
        ),
    }


def run_materialization(
    root: Path = plan.ROOT, *, budgets: Sequence[int] = plan.BUDGETS,
    surfaces: Sequence[str] = plan.SURFACES, on_progress: Callable | None = None,
    smoke_sessions: int | None = None,
) -> dict[str, Any]:
    """Decode every (surface, session, budget), anchor it, cache it."""
    import torch

    from src import low_cost_calibration_v1 as lc
    from src.calibration_gap_v1 import p4_stream_stats as p4
    from src.calibration_gap_v1 import z1_oracle_cells as z1

    from .scoring import house_session_r2

    cache_root = Path(root) / plan.CACHE_ROOT_RELATIVE
    _require(
        not cache_store.manifest_path(cache_root).exists()
        or not load_manifest_entries(cache_root),
        f"slot audit cache is not fresh: {cache_root} (a fresh root is required)",
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    sealed_probe = load_sealed_probe(root)
    baseline = sealed_baseline_rows(sealed_probe)

    runtime = z1.HonestOracleRuntime(plan.REPO_ROOT)
    started = time.perf_counter()
    state_before = runtime.state_digest()
    parity: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    rosters: dict[str, list[str]] = {}
    try:
        for surface in surfaces:
            roster = (
                runtime.external_roster if surface == "external"
                else runtime.within_roster
            )
            if smoke_sessions is not None:
                roster = roster[: int(smoke_sessions)]
            rosters[surface] = list(roster)
            for session_name in roster:
                t_parse = time.perf_counter()
                inputs = p4.materialize_session(runtime, surface, session_name)
                parse_s = time.perf_counter() - t_parse
                behavior = targets.behavior_source(
                    runtime, surface, session_name, inputs,
                )
                target_rows.append({
                    "surface": surface, "session": session_name,
                    "n_bins": behavior["n_bins"],
                    "bin_valid_fraction": behavior["bin_valid_fraction"],
                    "in_window_bin_valid_fraction": behavior[
                        "in_window_bin_valid_fraction"
                    ],
                    "n_query_trials": behavior["n_query_trials"],
                    "last_bin_reproduces_governing_target_bitexact": behavior[
                        "last_bin_reproduces_governing_target_bitexact"
                    ],
                })
                cache_store.cache_behavior(
                    cache_root, surface=surface, session=session_name,
                    behavior=behavior["behavior"], bin_valid=behavior["bin_valid"],
                )
                heads = targets.trial_heads_from_starts(inputs.starts)
                _require(
                    int(heads.sum()) == behavior["n_query_trials"],
                    f"{session_name}: cached trial heads disagree with query trials",
                )
                for budget in budgets:
                    selected = inputs.selected_by_budget[budget]
                    activity = inputs.calib[list(selected)]
                    t0 = time.perf_counter()
                    prediction, identities = p4._decode_static(
                        runtime, inputs, activity, inputs.side_by_budget[budget],
                    )
                    wall_s = time.perf_counter() - t0
                    full = np.ascontiguousarray(
                        prediction.detach().contiguous().numpy(), dtype=np.float32,
                    )
                    _require(
                        full.shape == (inputs.n_windows, plan.WINDOW_BINS, 2)
                        and bool(np.isfinite(full).all()),
                        f"{session_name} M{budget}: full prediction shape/nonfinite drift",
                    )
                    last32 = np.ascontiguousarray(
                        full[:, plan.GOVERNING_BIN, :], dtype=np.float32,
                    )
                    target32 = np.ascontiguousarray(
                        inputs.last_targets, dtype=np.float32,
                    )
                    house_full = house_session_r2(last32, target32)
                    full_sha = hashlib.sha256(full.tobytes()).hexdigest()
                    anchor = baseline[(surface, int(budget), session_name)]
                    for field in (
                        "target_sha256", "neural_sha256",
                        "calibration_m30_sha256", "normalized_side_sha256",
                        "selected_indices_sha256", "raw_t4_sha256",
                    ):
                        _require(
                            {
                                "target_sha256": inputs.target_sha256,
                                "neural_sha256": inputs.neural_sha256,
                                "calibration_m30_sha256": inputs.calibration_m30_sha256,
                                "normalized_side_sha256": inputs.side_sha_by_budget[budget],
                                "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
                                "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
                            }[field] == anchor[field],
                            f"{surface} M{budget} {session_name}: sealed-probe anchor "
                            f"SHA drift at {field}",
                        )
                    _require(
                        int(anchor["n_windows"]) == int(full.shape[0]),
                        f"{surface} M{budget} {session_name}: n_windows drift",
                    )
                    delta = house_full - float(anchor["variance_weighted_r2"])
                    _require(
                        abs(delta) <= plan.BASELINE_R2_TOLERANCE,
                        f"{surface} M{budget} {session_name}: baseline parity drift "
                        f"{delta:.3e} exceeds {plan.BASELINE_R2_TOLERANCE:.1e}",
                    )
                    _require(
                        anchor["prediction_sha256"] == full_sha,
                        f"{surface} M{budget} {session_name}: full prediction SHA is "
                        "not bit-exact vs the sealed probe baseline row",
                    )
                    identity_sha = lc._array_sha(
                        identities[0].detach().numpy()
                    )
                    _require(
                        anchor["identity_sha256_by_block"][0] == identity_sha,
                        f"{surface} M{budget} {session_name}: identity SHA drift",
                    )
                    entry = cache_store.cache_prediction(
                        cache_root, surface=surface, session=session_name,
                        budget=budget, full_predictions=full, targets=target32,
                        starts=inputs.starts, valid=inputs.last_valid_mask,
                        trial_heads=heads,
                    )
                    parity.append({
                        "surface": surface, "session": session_name,
                        "budget": int(budget),
                        "probe_baseline_r2": float(anchor["variance_weighted_r2"]),
                        "recomputed_house_r2": house_full,
                        "delta_r2": delta,
                        "prediction_sha256_full_bitexact": True,
                        "cached_full_prediction_sha256": entry["full_prediction_sha256"],
                        "cached_full_prediction_bytes_sha256": entry[
                            "full_prediction_bytes_sha256"
                        ],
                        "cached_last_bin_prediction_sha256": entry[
                            "last_bin_prediction_sha256"
                        ],
                        "cached_last_bin_prediction_bytes_sha256": entry[
                            "last_bin_prediction_bytes_sha256"
                        ],
                        "probe_prediction_sha256": anchor["prediction_sha256"],
                        "identity_sha256": identity_sha,
                        "n_windows": int(full.shape[0]),
                        "n_query_trials": behavior["n_query_trials"],
                        "decode_wall_seconds": wall_s,
                    })
                if on_progress is not None:
                    on_progress(surface, session_name, parse_s, parity[-len(budgets):])
    finally:
        runtime.close()
    state_after = runtime.state_digest()
    _require(
        state_after == state_before,
        "sealed Cell-D state changed during slot-audit materialization",
    )
    return {
        "schema": "slot_audit_v1_materialize_v1",
        "status": "COMPLETE",
        "smoke": smoke_sessions is not None,
        "smoke_sessions_per_surface": smoke_sessions,
        "surfaces": list(surfaces),
        "budgets": [int(b) for b in budgets],
        "rosters": rosters,
        "sessions": len(target_rows),
        "decodes": len(parity),
        "sealed_anchor": {
            "rel": plan.SEALED_PROBE_REL,
            "sha256": plan.SEALED_PROBE_SHA256,
            "baseline_rows": len(baseline),
        },
        "parity_tolerance": plan.BASELINE_R2_TOLERANCE,
        "parity_sessions": parity,
        "max_abs_r2_drift": max(abs(row["delta_r2"]) for row in parity),
        "all_prediction_shas_bitexact": all(
            row["prediction_sha256_full_bitexact"] for row in parity
        ),
        "target_source": {
            "description": (
                "session-level behavior-bin array [n_bins, 2] from the SAME "
                "frozen loader call as the probe session parse, on the same "
                "verified held-data snapshot; per-bin targets gathered at the "
                "absolute bin index; last-bin parity asserted bit-exact"
            ),
            "sessions": target_rows,
            "all_last_bin_bitexact": all(
                row["last_bin_reproduces_governing_target_bitexact"]
                for row in target_rows
            ),
        },
        "bin_size": _verify_bin_size_provenance(),
        "cache_manifest": cache_store.manifest_digest(cache_root),
        "throughput": {
            **runtime.timing,
            "wall_seconds_total": time.perf_counter() - started,
            "sessions": len(target_rows),
            "decodes": len(parity),
        },
        "sealed_state_sha256": state_before,
        "sealed_state_sha256_after": state_after,
        "sealed_state_unchanged": True,
        "boundaries": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "zero_target_optimizer_steps": True,
            "zero_target_backward_calls": True,
            "zero_target_update_calls": True,
            "training_authorized": False,
            "formal_opened": False,
            "frozen_sealed_cell_d_swa_strict_loaded": True,
            "deployment_recipe_unchanged": True,
            "strict27_source_roster_decision": (
                "NOT RUN: the frozen runtime exposes only the within-6 and "
                "external-15 evaluation rosters; a strict-27 source roster is "
                "not exposed without new parsing (work order section 3)"
            ),
        },
        "target_optimizer_backward_update": 0,
        "torch_version": torch.__version__,
    }


def load_manifest_entries(cache_root: Path) -> dict:
    """Empty dict when no manifest exists yet (fresh-root check helper)."""
    path = cache_store.manifest_path(cache_root)
    if not path.exists():
        return {}
    return cache_store.load_manifest(cache_root)["entries"]
