"""Scoring: sealed T0/C1 chunk-CDM plus trained-arm static / trial-CDM / chunk-CDM.

Reuses m1_t0c1_prefix_v1.phase3 helpers (open_session_dataset, score_static,
score_cdm_fifo, strict_load_arm_model) via import. Adds score_cdm_chunk_fifo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical as base_physical
from tfpd_exploration.src.m1_h1_activity_headroom_v1 import core as headroom_core
from tfpd_exploration.src.m1_t0c1_prefix_v1 import phase3 as t0c1_phase3

from . import chunk as chunk_module
from . import plan


class ScoreError(RuntimeError):
    """Fail closed for chunk-CDM or trained-arm scoring."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


WINDOW = t0c1_phase3.WINDOW
UNITS = t0c1_phase3.UNITS
OUTPUTS = t0c1_phase3.OUTPUTS
BATCH = t0c1_phase3.BATCH
SUPPORT_TRIALS = t0c1_phase3.SUPPORT_TRIALS

open_session_dataset = t0c1_phase3.open_session_dataset
score_static = t0c1_phase3.score_static
score_cdm_fifo = t0c1_phase3.score_cdm_fifo
strict_load_arm_model = t0c1_phase3.strict_load_arm_model
load_arm_binding = t0c1_phase3.load_arm_binding
output_trial_indices = t0c1_phase3.output_trial_indices


class _DatasetQueryView:
    """Read-only window subset. Identity/calibration tensors stay full-session."""

    def __init__(self, dataset: Any, keep: Sequence[int]) -> None:
        indices = list(dataset.window_indices)
        self.window_indices = [indices[int(row)] for row in keep]
        for name in (
            "neural_data", "covariate_data", "trial_start_indices",
            "calib_trialized_neural_features",
        ):
            setattr(self, name, getattr(dataset, name))

    def __len__(self) -> int:
        return len(self.window_indices)


def query_row_mask(
    output_trials: Sequence[int], *, start: int, stop: int | None,
) -> np.ndarray:
    """Keep rows whose output trial is in [start, stop). stop=None is [start, end)."""
    _require(type(start) is int and start >= 0, "query start")
    _require(stop is None or (type(stop) is int and stop > start), "query stop")
    trials = np.asarray(output_trials, dtype=np.int64)
    _require(trials.ndim == 1 and trials.size > 0, "output-trial vector")
    mask = trials >= start
    if stop is not None:
        mask &= trials < stop
    return mask


def restrict_opened_to_query(
    opened: Mapping[str, Any], *, start: int, stop: int | None,
) -> dict[str, Any]:
    """Filter query windows by output trial. Support trials [0,10) stay available."""
    dataset = opened["dataset"]
    session_id = str(opened["session_id"])
    trials = output_trial_indices(dataset, session_id)
    mask = query_row_mask(trials, start=start, stop=stop)
    keep = [int(row) for row in np.flatnonzero(mask)]
    _require(len(keep) > 0, f"query window empty for {session_id}")
    kept = [int(trials[row]) for row in keep]
    return {
        "dataset": _DatasetQueryView(dataset, keep),
        "trials": opened["trials"],
        "session_id": session_id,
        "query": [start, stop],
        "query_label": "q10_end" if start == 10 and stop is None else f"q{start}_{stop}",
        "n_windows_full": int(len(trials)),
        "n_windows_query": int(len(keep)),
        "output_trial_min": int(min(kept)),
        "output_trial_max": int(max(kept)),
    }


def load_sealed_t0_anchors(repo_root: Path) -> dict[str, float]:
    """Read sealed T0/C1 phase-3 floats; fall back to spec literals."""
    table_path = Path(repo_root) / plan.SEALED_T0C1_PHASE3_TABLE_RELATIVE
    anchors = {
        "t0_static_20120924": plan.ANCHOR_T0_STATIC_20120924_LITERAL,
        "t0_trial_cdm_20120924": plan.ANCHOR_T0_TRIAL_CDM_20120924_LITERAL,
        "source": "literal",
    }
    if not table_path.is_file():
        return anchors
    table = json.loads(table_path.read_bytes().decode("utf-8"))
    _require(isinstance(table, Mapping) and isinstance(table.get("rows"), list),
             "sealed phase-3 table topology drift")
    extras: dict[str, float] = {}
    for row in table["rows"]:
        if not isinstance(row, Mapping):
            continue
        per_session = row.get("per_session")
        if not isinstance(per_session, Mapping):
            continue
        key = (row.get("arm"), row.get("deployment"), row.get("surface"))
        if "20120924" in per_session:
            extras[f"{row.get('arm')}_{row.get('deployment')}_{row.get('surface')}_20120924"] = float(
                per_session["20120924"])
        if key == ("t0", "static_m10", "heldout_fold") and "20120924" in per_session:
            anchors["t0_static_20120924"] = float(per_session["20120924"])
            anchors["source"] = str(plan.SEALED_T0C1_PHASE3_TABLE_RELATIVE)
        if key == ("t0", "cdm_activity_fifo_m10", "heldout_fold") and "20120924" in per_session:
            anchors["t0_trial_cdm_20120924"] = float(per_session["20120924"])
        if key == ("c1", "static_m10", "heldout_fold") and "20120924" in per_session:
            anchors["c1_static_20120924"] = float(per_session["20120924"])
        if key == ("t0", "static_m10", "heldout_fold"):
            pass
    anchors["sealed_cells"] = extras  # type: ignore[assignment]
    return anchors


def score_cdm_chunk_fifo(model: Any, opened: Mapping[str, Any], *, device: str) -> dict[str, object]:
    """Chunk-CDM: causal last-10-completed-chunks identity, support-trial bootstrap."""
    import torch

    dataset = opened["dataset"]
    session_id = opened["session_id"]
    trials = opened["trials"]
    neural = np.ascontiguousarray(dataset.neural_data[session_id])
    chunks = chunk_module.extract_chunks(neural)
    n_complete = int(chunks.shape[0])
    bootstrap_rows: list[int] = []
    chunk_groups: dict[tuple[int, ...], list[int]] = {}
    decisions: list[dict[str, object]] = []
    for row_index, (_session, start) in enumerate(dataset.window_indices):
        last_bin = int(start) + WINDOW - 1
        decision = chunk_module.selection_for_last_bin(last_bin, n_complete)
        decisions.append(decision)
        if decision["bootstrap"] is True:
            bootstrap_rows.append(row_index)
        else:
            selection = tuple(decision["chunk_selection"])  # type: ignore[arg-type]
            chunk_groups.setdefault(selection, []).append(row_index)
    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    forwards = 0

    def _fill(identity: Any, rows: list[int]) -> None:
        nonlocal forwards
        for offset in range(0, len(rows), BATCH):
            batch_rows = rows[offset:offset + BATCH]
            xs, ys = [], []
            for row in batch_rows:
                _session, start = dataset.window_indices[row]
                end = int(start) + WINDOW
                xs.append(dataset.neural_data[session_id][int(start):end])
                ys.append(dataset.covariate_data[session_id][end - 1])
            x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
            output = headroom_core.forward_with_cached_identity(model, x, identity)
            prediction[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(
                output[:, -1, :].detach().cpu().numpy(), dtype=np.float32)
            target[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(
                np.stack(ys), dtype=np.float32)
            forwards += 1

    with torch.no_grad():
        if bootstrap_rows:
            identity = headroom_core.identity_from_raw_trials(
                model, trials, tuple(range(SUPPORT_TRIALS)), family="m1", device=device)
            _fill(identity, bootstrap_rows)
        for selection, rows in chunk_groups.items():
            identity = headroom_core.identity_from_raw_trials(
                model, chunks, selection, family="m1", device=device)
            _fill(identity, list(rows))
    extra = chunk_module.summarize_decisions(decisions)
    extra.update({
        "forward_path": "cdm_chunk_fifo_cached_identity",
        "forward_batches": forwards,
        "n_complete_chunks": n_complete,
        "unique_chunk_activity_states": len(chunk_groups) + (1 if bootstrap_rows else 0),
        "causal": True,
        "label_free": True,
        "activity_cardinality": SUPPORT_TRIALS,
        "chunk_bins": plan.CHUNK_BINS,
    })
    return t0c1_phase3._cell_result(prediction, target, extra)


def unwrap_state_dict(payload: Any) -> Any:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def load_state_bytes(path: Path) -> bytes:
    body = Path(path).read_bytes()
    _require(body, f"checkpoint empty: {path}")
    return body


def construct_model_for_arm(repo_root: Path, arm: str, *, device: str) -> Any:
    from . import train as train_module

    if arm in plan.CHUNK_CDM_ARMS or plan.ARM_SPECS.get(arm, {}).get("stock_spint"):
        model = base_physical.load_exact_m1_spint_model(Path(repo_root)).to(device)
        base_physical.materialize_exact_m1_model(model, device=device)
        return model
    model = train_module.load_w32_model(Path(repo_root)).to(device)
    train_module.materialize_w32_model(model, device=device)
    return model


def strict_load_bytes(repo_root: Path, arm: str, body: bytes, *, device: str,
                      expected_state_sha256: str | None = None) -> tuple[Any, str]:
    import torch

    captured: list[Any] = []

    def factory() -> Any:
        model = construct_model_for_arm(Path(repo_root), arm, device=device)
        captured.append(model)
        return model

    if expected_state_sha256:
        # Sealed T0/C1 best checkpoints are bare state_dicts.
        observed = base_physical.strict_reload_checkpoint_bytes(
            body, expected_state_sha256=expected_state_sha256, model_factory=factory, device=device)
        model = captured[0]
        model.eval()
        return model, observed
    model = factory()
    payload = torch.load(__import__("io").BytesIO(body), map_location=device, weights_only=False)
    state = unwrap_state_dict(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    observed = base_physical._state_digest(model)
    return model, observed


def score_three_deployments(
    model: Any, opened: Mapping[str, Any], *, device: str,
) -> dict[str, dict[str, object]]:
    return {
        "static_m10": score_static(model, opened, device=device),
        "cdm_trial_fifo_m10": score_cdm_fifo(model, opened, device=device),
        "cdm_chunk_fifo_m10": score_cdm_chunk_fifo(model, opened, device=device),
    }


def check_t0_anchors(cells: Mapping[str, Mapping[str, object]], anchors: Mapping[str, object]) -> None:
    static_key = "t0_static_m10_20120924"
    trial_key = "t0_cdm_trial_fifo_m10_20120924"
    _require(static_key in cells and trial_key in cells, "T0 20120924 score cells missing for anchor gate")
    static_r2 = float(cells[static_key]["governing_r2"])
    trial_r2 = float(cells[trial_key]["governing_r2"])
    expected_static = float(anchors["t0_static_20120924"])
    expected_trial = float(anchors["t0_trial_cdm_20120924"])
    _require(abs(static_r2 - expected_static) <= plan.ANCHOR_TOLERANCE,
             f"T0 static 20120924 anchor broken: {static_r2} vs {expected_static}")
    _require(abs(trial_r2 - expected_trial) <= plan.ANCHOR_TOLERANCE,
             f"T0 trial-CDM 20120924 anchor broken: {trial_r2} vs {expected_trial}")


def sessions_for_chunk_cdm(arm: str) -> tuple[str, ...]:
    if arm == "t0":
        return tuple(plan.SCORE_ORDER)
    return tuple(plan.SCORE_ORDER)


def publish_score_cells(
    artifact: Any,
    *,
    arm: str,
    cells: Mapping[str, Mapping[str, object]],
) -> dict[str, str]:
    shas: dict[str, str] = {}
    for key, cell in cells.items():
        name = f"score_{key}.json"
        shas[name] = artifact.publish_json(name, dict(cell))
    return shas


def build_arm_score_table(arm: str, cells: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    rows = []
    for deployment in ("static_m10", "cdm_trial_fifo_m10", "cdm_chunk_fifo_m10"):
        for surface, sessions in (
            ("heldout_fold", plan.HELDOUT_FOLD_SESSIONS),
            ("heldin_training", plan.HELDIN_TRAINING_SESSIONS),
            ("heldout_fold_q10_end", plan.QUERY_Q10_END_SESSIONS),
        ):
            values = {}
            missing = False
            for session_id in sessions:
                if surface.endswith("_q10_end"):
                    key = f"{arm}_{deployment}_q10_end_{session_id}"
                else:
                    key = f"{arm}_{deployment}_{session_id}"
                if key not in cells:
                    missing = True
                    break
                values[session_id] = float(cells[key]["governing_r2"])
            if missing:
                continue
            row = {
                "arm": arm, "deployment": deployment, "surface": surface,
                "sessions": list(sessions),
                "equal_session_mean": float(np.mean(list(values.values()))),
                "per_session": values,
            }
            if surface.endswith("_q10_end"):
                row["query"] = [plan.QUERY_Q10_END[0], plan.QUERY_Q10_END[1]]
                row["governing"] = False
            else:
                row["governing"] = True
            rows.append(row)
    return {
        "schema": "m1_tier12_arm_score_table_v1",
        "arm": arm,
        "rows": rows,
        "governing_window": "full_session",
        "comparison_window": "q10_end",
        "formal_benchmark_verdict": False,
        "metric": plan.METRIC_LABEL,
    }


def evaluate_read_rules(
    *,
    sealed: Mapping[str, object],
    cells: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate frozen read rules once the relevant score cells exist."""
    out: dict[str, object] = {
        "schema": "m1_tier12_read_rules_eval_v1",
        "formal_benchmark_verdict": False,
        "primary_surface": "20120924",
    }
    t0_static = sealed.get("t0_static_20120924")
    t0_trial = sealed.get("t0_trial_cdm_20120924")
    t0_chunk_cell = cells.get("t0_cdm_chunk_fifo_m10_20120924")
    if t0_static is not None and t0_trial is not None and t0_chunk_cell is not None:
        chunk_r2 = float(t0_chunk_cell["governing_r2"])
        trial_delta = float(t0_trial) - float(t0_static)
        chunk_delta = chunk_r2 - float(t0_static)
        preserves = chunk_delta >= 0.5 * trial_delta
        out["chunk_cdm"] = {
            "t0_static": float(t0_static),
            "t0_trial_cdm": float(t0_trial),
            "t0_chunk_cdm": chunk_r2,
            "trial_minus_static": trial_delta,
            "chunk_minus_static": chunk_delta,
            "threshold": 0.5 * trial_delta,
            "verdict": "PRESERVES" if preserves else "DROPS",
        }
    for sealed_arm, trained in (("t0", "t0_swa"), ("c1", "c1_swa")):
        sealed_key = f"{sealed_arm}_static_20120924"
        trained_key = f"{trained}_static_m10_20120924"
        if sealed_key not in sealed and sealed_arm == "c1":
            continue
        if trained_key not in cells:
            continue
        sealed_static = float(sealed[sealed_key]) if sealed_arm == "t0" or sealed_key in sealed else None
        if sealed_arm == "t0":
            sealed_static = float(sealed["t0_static_20120924"])
        elif "c1_static_20120924" in sealed:
            sealed_static = float(sealed["c1_static_20120924"])
        else:
            continue
        delta = float(cells[trained_key]["governing_r2"]) - sealed_static
        if delta >= 0.005:
            verdict = "HELPS"
        elif delta <= -0.005:
            verdict = "HURTS"
        else:
            verdict = "NEUTRAL"
        out[f"swa_{sealed_arm}"] = {
            "sealed_static": sealed_static,
            "swa_static": float(cells[trained_key]["governing_r2"]),
            "delta": delta,
            "verdict": verdict,
        }
    w32_key = "t0_w32_static_m10_20120924"
    if w32_key in cells and t0_static is not None:
        delta = float(cells[w32_key]["governing_r2"]) - float(t0_static)
        if delta >= 0.01:
            verdict = "WINS"
        elif delta >= -0.03:
            verdict = "NONINFERIOR"
        else:
            verdict = "INFERIOR"
        out["w32_t0"] = {
            "sealed_t0_static": float(t0_static),
            "t0_w32_static": float(cells[w32_key]["governing_r2"]),
            "delta": delta,
            "verdict": verdict,
        }
    return out


def trained_checkpoint_filename(arm: str) -> str:
    spec = plan.ARM_SPECS[arm]
    if spec["swa"]:
        return "swa_final4.pt"
    return "checkpoint_best_source_train_loss.pt"


def trained_arm_checkpoint_path(repo_root: Path, arm: str) -> Path:
    return Path(repo_root) / plan.TRAIN_ROOT_RELATIVE[arm] / trained_checkpoint_filename(arm)
