"""Raw frozen prediction streams: one-time materialization + cached reuse.

Both surfaces' raw streams are produced ONCE and cached under a route-owned
cache directory with digests; every later filter stage (P0-P4) consumes the
cached bytes and performs ZERO further decoder forwards.

* ``static`` — the frozen static T4/Cell-D deployment recipe (ridge-T4 0.1;
  D-opt-first-30 @M4, chronological @M10/M30), CPU-only, reproduced through
  the sealed Z1/P4 harness exactly like the continuity probe.  Parity anchors:
  the continuity-probe baseline rows (float32 full-stream R2 equality and
  full [W,50,2] prediction-SHA bit-exactness).
* ``cdm`` — the activity-only CDM stream (the P2' stage-A A0 rollouts),
  GPU inference only, reproduced through the sealed P2' runtime.  Parity
  anchors: the sealed stage-A A0 rows (float-equal matrix R2 and bit-equal
  prediction SHA over the joined valid rows).

TRIAL_RESET blocks are derived fail-closed at materialization time: the static
block count must equal the loader's query-trial count; the CDM trial order
comes from the immutable input authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import ladder, plan


class StreamsError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StreamsError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


GOVERNING_BIN = 49


# ---------------------------------------------------------------------------
# Manifest (cumulative, digest-bound).
# ---------------------------------------------------------------------------


def manifest_path(cache_root: Path) -> Path:
    return cache_root / "manifest.json"


def load_manifest(cache_root: Path) -> dict[str, Any]:
    path = manifest_path(cache_root)
    _require(path.exists(), f"stream cache manifest missing: {path}")
    body = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    _require(sidecar.exists(), "stream cache manifest sidecar missing")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {path.name}",
        "stream cache manifest sidecar drift",
    )
    return body


def _add_entry(cache_root: Path, key: str, entry: dict[str, Any]) -> None:
    def mutate(body: dict[str, Any]) -> dict[str, Any]:
        _require(key not in body["entries"], f"stream cache entry already present: {key}")
        body["entries"][key] = entry
        return body

    _update_manifest(cache_root, mutate)


def _update_manifest(cache_root: Path, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    path = manifest_path(cache_root)
    if path.exists():
        body = json.loads(path.read_text(encoding="utf-8"))
    else:
        body = {
            "schema": "learnable_output_filter_v1_stream_cache_v1",
            "entries": {},
        }
    body = mutate(body)
    text = json.dumps(body, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")


def _npz_relative(kind: str, surface: str, session: str, budget: int) -> str:
    return f"{kind}/{surface}/{session}/m{int(budget)}.npz"


def _cache_stream(
    cache_root: Path, *, kind: str, surface: str, session: str, budget: int,
    bins: np.ndarray, heads: np.ndarray, raw: np.ndarray, target: np.ndarray,
    valid: np.ndarray, trial_ids: Sequence[str], stream: ladder.SessionStream,
) -> None:
    relative = _npz_relative(kind, surface, session, budget)
    path = cache_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary, bins=np.asarray(bins, dtype=np.int64), heads=np.asarray(heads, dtype=bool),
        raw=np.asarray(raw, dtype=np.float32), target=np.asarray(target, dtype=np.float32),
        valid=np.asarray(valid, dtype=bool), trial_ids=np.asarray(list(trial_ids)),
    )
    os.replace(temporary, path)
    proof = stream.chronology_proof()
    _add_entry(cache_root, f"{kind}:{surface}:{session}:m{int(budget)}", {
        "kind": kind, "surface": surface, "session": session, "budget": int(budget),
        "relative": relative, "sha256": _sha256_file(path),
        "n_windows": int(np.asarray(valid).size), "n_trials": len(trial_ids),
        "chronological_row_digest": proof["chronological_row_digest"],
        "reset_event_digest": proof["reset_event_digest"],
    })


# ---------------------------------------------------------------------------
# Loading cached streams back into the ladder's stream model.
# ---------------------------------------------------------------------------


def cached_roster(cache_root: Path, kind: str) -> dict[str, list[str]]:
    manifest = load_manifest(cache_root)
    out: dict[str, list[str]] = {}
    for entry in manifest["entries"].values():
        if entry["kind"] != kind:
            continue
        out.setdefault(entry["surface"], [])
        if entry["session"] not in out[entry["surface"]]:
            out[entry["surface"]].append(entry["session"])
    return out


def load_stream(
    cache_root: Path, kind: str, surface: str, session: str, budget: int,
) -> ladder.SessionStream:
    manifest = load_manifest(cache_root)
    key = f"{kind}:{surface}:{session}:m{int(budget)}"
    _require(key in manifest["entries"], f"stream not cached: {key}")
    entry = manifest["entries"][key]
    path = cache_root / str(entry["relative"])
    _require(_sha256_file(path) == entry["sha256"], f"cached stream digest drift: {key}")
    with np.load(path) as payload:
        bins = np.asarray(payload["bins"], dtype=np.int64)
        heads = np.asarray(payload["heads"], dtype=bool)
        raw = np.asarray(payload["raw"], dtype=np.float32)
        target = np.asarray(payload["target"], dtype=np.float32)
        valid = np.asarray(payload["valid"], dtype=bool)
        trial_ids = [str(item) for item in payload["trial_ids"]]
    _require(
        bool(np.array_equal(heads, ladder.trial_heads_from_flat_bins(bins))),
        f"{key}: cached TRIAL_RESET heads disagree with the bin stream",
    )
    blocks = ladder.blocks_from_flat(
        bins=bins, trial_labels=np.cumsum(heads) - 1, raw=raw, target=target,
        valid=valid, expected_blocks=len(trial_ids), trial_ids=trial_ids,
    )
    stream = ladder.SessionStream(surface, session, int(budget), blocks)
    stream.validate()
    proof = stream.chronology_proof()
    _require(
        proof["chronological_row_digest"] == entry["chronological_row_digest"]
        and proof["reset_event_digest"] == entry["reset_event_digest"],
        f"{key}: chronology/reset digest drift",
    )
    return stream


def load_all(
    cache_root: Path, kind: str, *, surfaces=plan.SURFACES,
) -> dict[tuple[str, str, int], ladder.SessionStream]:
    """Load every cached stream of ``kind`` (all surfaces, all cached budgets)."""
    manifest = load_manifest(cache_root)
    out: dict[tuple[str, str, int], ladder.SessionStream] = {}
    for key, entry in manifest["entries"].items():
        if entry["kind"] != kind:
            continue
        surface, session, budget = entry["surface"], entry["session"], int(entry["budget"])
        if surfaces is not None and surface not in surfaces:
            continue
        out[(surface, session, budget)] = load_stream(cache_root, kind, surface, session, budget)
    return out


def _verify_sealed(relative: str, expected_sha: str) -> dict[str, Any]:
    path = plan.REPO_ROOT / relative
    _require(path.exists(), f"sealed receipt missing: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(digest == expected_sha, f"sealed receipt SHA drift: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Static surface materialization (CPU-only).
# ---------------------------------------------------------------------------


def materialize_static(
    root: Path, *, budgets: Sequence[int] = plan.BUDGETS, surfaces: Sequence[str] = plan.SURFACES,
    smoke_sessions: int | None = None,
) -> dict[str, Any]:
    """Reproduce the frozen static deployment stream once and cache it.

    ``smoke_sessions`` limits the roster per surface (never governing); with a
    smoke subset the budget list is forced to M4.
    """
    import torch

    from src.tfpd_lane.matched_scorer import session_r2
    from src.calibration_gap_v1 import ledger as gap_ledger
    from src.calibration_gap_v1 import p4_stream_stats as p4
    from src.calibration_gap_v1 import z1_oracle_cells as z1

    if smoke_sessions is not None:
        budgets = (4,)
    cache_root = Path(root) / plan.CACHE_ROOT_RELATIVE
    cache_root.mkdir(parents=True, exist_ok=True)
    probe_body = _verify_sealed(plan.SEALED_PROBE_REL, plan.SEALED_PROBE_SHA256)
    baseline = {
        (row["surface"], int(row["budget"]), row["session"]): row
        for cell in probe_body["cells"] for row in cell["sessions"] if row["arm"] == "baseline"
    }
    ledger = gap_ledger.load_ledger(plan.ROOT)
    runtime = z1.HonestOracleRuntime(plan.REPO_ROOT)
    started = time.perf_counter()
    state_before = runtime.state_digest()
    parity: list[dict[str, Any]] = []
    try:
        for surface in surfaces:
            roster = runtime.external_roster if surface == "external" else runtime.within_roster
            if smoke_sessions is not None:
                roster = roster[: int(smoke_sessions)]
            for session_name in roster:
                inputs = p4.materialize_session(runtime, surface, session_name)
                bins = np.asarray(inputs.starts, dtype=np.int64) + GOVERNING_BIN
                expected_blocks = int(inputs.n_trials) - 30
                heads = ladder.trial_heads_from_flat_bins(bins)
                _require(
                    int(heads.sum()) == expected_blocks,
                    f"{session_name}: recovered {int(heads.sum())} static trial blocks, "
                    f"expected {expected_blocks} query trials",
                )
                trial_labels = np.cumsum(heads) - 1
                trial_ids = [f"static-query-{index:03d}" for index in range(int(heads.sum()))]
                for budget in budgets:
                    selected = inputs.selected_by_budget[budget]
                    activity = inputs.calib[list(selected)]
                    t0 = time.perf_counter()
                    prediction, _identities = p4._decode_static(
                        runtime, inputs, activity, inputs.side_by_budget[budget],
                    )
                    wall_s = time.perf_counter() - t0
                    raw = np.ascontiguousarray(
                        prediction[:, GOVERNING_BIN, :].detach().contiguous().numpy(), dtype=np.float32,
                    )
                    target = np.ascontiguousarray(inputs.last_targets, dtype=np.float32)
                    valid = np.ascontiguousarray(inputs.last_valid_mask, dtype=bool)
                    full_prediction_sha = hashlib.sha256(
                        prediction.detach().contiguous().numpy().tobytes(),
                    ).hexdigest()
                    house_full = float(session_r2(
                        torch.from_numpy(raw.copy()), torch.from_numpy(target.copy()),
                    ))
                    anchor = baseline[(surface, int(budget), session_name)]
                    _require(
                        anchor["prediction_sha256"] == full_prediction_sha,
                        f"{surface} M{budget} {session_name}: static prediction SHA drift vs probe baseline",
                    )
                    _require(
                        abs(float(anchor["variance_weighted_r2"]) - house_full)
                        <= plan.STATIC_ANCHOR_TOLERANCE,
                        f"{surface} M{budget} {session_name}: static R2 drift vs probe baseline",
                    )
                    _require(
                        anchor["target_sha256"] == inputs.target_sha256,
                        f"{surface} M{budget} {session_name}: static target SHA drift",
                    )
                    _require(int(anchor["n_windows"]) == int(raw.shape[0]), "static n_windows drift")
                    blocks = ladder.blocks_from_flat(
                        bins=bins, trial_labels=trial_labels, raw=raw, target=target,
                        valid=valid, expected_blocks=expected_blocks, trial_ids=trial_ids,
                    )
                    stream = ladder.SessionStream(surface, session_name, int(budget), blocks)
                    _cache_stream(
                        cache_root, kind="static", surface=surface, session=session_name,
                        budget=budget, bins=bins, heads=heads, raw=raw, target=target,
                        valid=valid, trial_ids=trial_ids, stream=stream,
                    )
                    parity.append({
                        "surface": surface, "session": session_name, "budget": int(budget),
                        "probe_baseline_r2": float(anchor["variance_weighted_r2"]),
                        "recomputed_house_r2_full_stream": house_full,
                        "prediction_sha256_full_bitexact": True,
                        "target_sha256": inputs.target_sha256,
                        "n_windows": int(raw.shape[0]),
                        "n_query_trials": expected_blocks,
                        "all_rows_valid": bool(valid.all()),
                        "selected_indices_sha256": inputs.selected_sha_by_budget[budget],
                        "raw_t4_sha256": inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"],
                        "decode_wall_seconds": wall_s,
                    })
                print(f"[static] {surface} {session_name} cached", flush=True)
    finally:
        runtime.close()
    state_after = runtime.state_digest()
    _require(state_before == state_after, "sealed Cell-D state changed during static materialization")
    return {
        "kind": "static",
        "smoke": smoke_sessions is not None,
        "smoke_sessions_per_surface": smoke_sessions,
        "sealed_anchor": {"rel": plan.SEALED_PROBE_REL, "sha256": plan.SEALED_PROBE_SHA256},
        "parity_sessions": parity,
        "sessions": len(parity),
        "max_abs_r2_drift": max(
            abs(item["probe_baseline_r2"] - item["recomputed_house_r2_full_stream"]) for item in parity
        ),
        "all_prediction_shas_bitexact": True,
        "sealed_state_sha256": state_before,
        "sealed_state_unchanged": True,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "wall_seconds": time.perf_counter() - started,
        "target_optimizer_backward_update": 0,
    }


# ---------------------------------------------------------------------------
# Activity-only CDM materialization (GPU inference, frozen weights).
# ---------------------------------------------------------------------------


def materialize_cdm(
    root: Path, *, gpu_index: int, budgets: Sequence[int] = plan.BUDGETS,
    surfaces: Sequence[str] = plan.SURFACES, smoke_sessions: int | None = None,
) -> dict[str, Any]:
    """Reproduce the P2' stage-A A0 rollouts once and cache the raw streams.

    ``smoke_sessions`` parses only the first N sessions per surface through the
    P2' smoke materialization path (never governing); budgets forced to M4.
    """
    from src.causal_dual_memory_cell_d_score_v1 import score as v1score
    from src.learned_gate_p2prime_v1 import physical as p2p

    _require(gpu_index in plan.GPUS_ALLOWED_FOR_CDM_CACHE, "CDM cache GPU index drift")
    if smoke_sessions is not None:
        budgets = (4,)
    cache_root = Path(root) / plan.CACHE_ROOT_RELATIVE
    cache_root.mkdir(parents=True, exist_ok=True)
    stage_a = _verify_sealed(plan.SEALED_STAGE_A_REL, plan.SEALED_STAGE_A_SHA256)
    anchors: dict[tuple[str, int, str], dict[str, Any]] = {}
    for budget_key, surface_map in stage_a["matrix"].items():
        for surface, rows in surface_map.items():
            for row in rows["A0"]["sessions"]:
                anchors[(surface, int(row["budget"]), str(row["session"]))] = row
    base = plan.REPO_ROOT
    runtime, meta, identity = p2p._runtime_for(base, gpu_index=gpu_index)
    started = time.perf_counter()
    parity: list[dict[str, Any]] = []
    model_state_before: str | None = None
    model_state_after: str | None = None
    try:
        runtime.prepare(identity=identity)
        state = runtime._require_state()
        arm_state = state.modules["arm_common"]
        model_state_before = arm_state.state_sha256(state.model)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        if smoke_sessions is None:
            runtime.materialize_inputs(identity=identity, authority=fixed)
        else:
            runtime._materialize_subset(authority=fixed, per_surface=int(smoke_sessions))
        for budget in budgets:
            for surface in surfaces:
                for key in p2p._ordered_session_keys(runtime, surface):
                    session = runtime._require_state().sessions[key]
                    rollout = runtime._rollout_activity(session=session, budget=budget)
                    row = rollout["rows"]["A0"]
                    anchor = anchors[(surface, int(budget), session.session)]
                    _require(
                        float(anchor["matrix_r2"]) == float(row["matrix_r2"]),
                        f"{surface} M{budget} {session.session}: CDM matrix R2 drift vs sealed stage A",
                    )
                    _require(
                        anchor["prediction_sha256_raw"] == row["prediction_sha256_raw"],
                        f"{surface} M{budget} {session.session}: CDM raw SHA drift vs sealed stage A",
                    )
                    _require(
                        int(anchor["carrier_transitions_committed"]) == 0,
                        "CDM activity-only rollout committed a carrier transition",
                    )
                    targets, masks, _sst = runtime._session_target_views(session, budget)
                    query_ids = list(session.query_trial_ids[budget])
                    raw_blocks = rollout["per_trial_raw"]
                    _require(len(raw_blocks) == len(query_ids) == len(targets) == len(masks),
                             "CDM per-trial stream length drift")
                    bins = np.concatenate([
                        np.asarray(session.trials_by_id[trial_id].endpoint_bins, dtype=np.int64)
                        for trial_id in query_ids
                    ])
                    raw = np.ascontiguousarray(np.concatenate(
                        [np.asarray(item, dtype=np.float32) for item in raw_blocks], axis=0,
                    ))
                    target = np.ascontiguousarray(np.concatenate(
                        [np.asarray(item, dtype=np.float32) for item in targets], axis=0,
                    ))
                    valid = np.ascontiguousarray(np.concatenate(
                        [np.asarray(item, dtype=bool) for item in masks], axis=0,
                    ))
                    heads = np.zeros(bins.shape, dtype=bool)
                    lengths = [int(np.asarray(item).shape[0]) for item in raw_blocks]
                    heads[np.cumsum([0] + lengths[:-1])] = True
                    stream = ladder.SessionStream(
                        surface, session.session, int(budget),
                        ladder.blocks_from_flat(
                            bins=bins, trial_labels=np.cumsum(heads) - 1, raw=raw,
                            target=target, valid=valid, expected_blocks=len(query_ids),
                            trial_ids=query_ids,
                        ),
                    )
                    _cache_stream(
                        cache_root, kind="cdm", surface=surface, session=session.session,
                        budget=budget, bins=bins, heads=heads, raw=raw, target=target,
                        valid=valid, trial_ids=query_ids, stream=stream,
                    )
                    parity.append({
                        "surface": surface, "session": session.session, "budget": int(budget),
                        "sealed_stage_a_matrix_r2": float(anchor["matrix_r2"]),
                        "recomputed_matrix_r2": float(row["matrix_r2"]),
                        "prediction_sha256_raw_bitexact": True,
                        "n_valid_rows": int(row["n_windows"]),
                        "n_query_trials": len(query_ids),
                        "initial_activity_sha256": row["initial_activity_sha256"],
                        "initial_carrier_sha256": row["initial_carrier_sha256"],
                        "final_carrier_sha256": row["final_carrier_sha256"],
                        "carrier_transitions_committed": int(row["carrier_transitions_committed"]),
                        "activity_transitions_committed": int(row["activity_transitions_committed"]),
                    })
                print(f"[cdm] M{budget} {surface} {session.session} cached", flush=True)
        model_state_after = arm_state.state_sha256(runtime._require_state().model)
        _require(
            model_state_before == model_state_after,
            "sealed Cell-D model state changed during CDM materialization",
        )
    finally:
        runtime.close()
    return {
        "kind": "cdm",
        "smoke": smoke_sessions is not None,
        "smoke_sessions_per_surface": smoke_sessions,
        "sealed_anchor": {"rel": plan.SEALED_STAGE_A_REL, "sha256": plan.SEALED_STAGE_A_SHA256},
        "gpu_inference_only": True,
        "gpu_index": int(gpu_index),
        "environment": meta["environment"],
        "identity_sha256": identity.sha256,
        "parity_sessions": parity,
        "sessions": len(parity),
        "all_matrix_r2_equal": True,
        "all_raw_shas_bitexact": True,
        "model_state_sha256_before": model_state_before,
        "model_state_sha256_after": model_state_after,
        "model_state_unchanged_by_wrapper": True,
        "carrier_transitions_committed_total": sum(
            item["carrier_transitions_committed"] for item in parity
        ),
        "wall_seconds": time.perf_counter() - started,
        "target_optimizer_backward_update": 0,
    }
