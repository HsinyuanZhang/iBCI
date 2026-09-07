"""Frozen-model native-M2 Precision-CDM V2 local evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import core as cdm
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import physical as shared_physical

from . import plan


def _finite_m10_indices(theta_first30: np.ndarray) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(theta_first30)).astype(np.int64)
    core.require(finite.size >= 10, "native M2 first30 has fewer than ten directional trials")
    return np.ascontiguousarray(finite[:10], dtype=np.int64)


def _raw_fixed_ridge_m30(
    support_rates_hz30: np.ndarray, theta_first30: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    rates = np.ascontiguousarray(np.asarray(support_rates_hz30, dtype=np.float64))
    theta = np.ascontiguousarray(np.asarray(theta_first30, dtype=np.float64))
    core.require(rates.ndim == 2 and rates.shape[0] == plan.ACTIVITY_STACK_LIMIT
                 and theta.shape == (plan.ACTIVITY_STACK_LIMIT,),
                 "native M2 full-trial M30 support topology drift")
    usable = np.isfinite(theta)
    core.require(int(usable.sum()) >= 10, "native M2 M30 lacks directional support")
    raw, _evidence = fit_ridge_t4(
        rates[usable], theta[usable], normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    valid = np.ascontiguousarray(
        raw[:, 2] * plan.MODEL_BIN_SECONDS > 1.0e-6, dtype=np.bool_,
    )
    core.require(raw.shape[1] == 4 and int(valid.sum()) >= 4, "native M2 M30 T4 validity drift")
    return raw, valid


def _interpolate_raw_trial_activity(
    neural: np.ndarray, eval_mask: np.ndarray, *, start: int, stop: int,
) -> np.ndarray:
    """Rebuild one B3S row without conflating the raw and filtered axes."""
    from scipy.interpolate import interp1d

    trial = np.ascontiguousarray(neural[start:stop][eval_mask[start:stop]], dtype=np.float32)
    core.require(trial.ndim == 2 and trial.shape[0] >= 2,
                 "native M2 trial has fewer than two calibration-visible bins")
    source = np.linspace(0.0, 1.0, trial.shape[0])
    target = np.linspace(0.0, 1.0, 100)
    result = interp1d(source, trial, axis=0, kind="linear", fill_value="extrapolate")(target)
    result = np.ascontiguousarray(result, dtype=np.float32)
    core.require(result.shape == (100, neural.shape[1]) and np.isfinite(result).all(),
                 "native M2 B3S raw-trial reconstruction drift")
    return result


def _native_trial_views(
    raw_session: Mapping[str, object], *, session: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw neural/starts, first30 Hz rates+labels, and all B3S rows.

    Carrier statistics use complete rewarded-trial spike counts.  B3S uses the
    frozen calibration-visible mask inside each independently held raw trial.
    Query target angles beyond first30 are never materialized.
    """
    neural = np.ascontiguousarray(np.asarray(raw_session["neural"], dtype=np.float32))
    trial_change = np.ascontiguousarray(np.asarray(raw_session["trial_change"], dtype=np.bool_))
    eval_mask = np.ascontiguousarray(np.asarray(raw_session["eval_mask"], dtype=np.bool_))
    core.require(neural.ndim == 2 and trial_change.shape == eval_mask.shape == (neural.shape[0],),
                 f"{session} native M2 raw session topology drift")
    starts = np.ascontiguousarray(np.flatnonzero(trial_change), dtype=np.int64)
    core.require(starts.size > plan.ACTIVITY_STACK_LIMIT and np.all(np.diff(starts) > 0),
                 f"{session} native M2 lacks post30 raw trials")
    raw_angles = raw_session.get("trial_target_angles")
    core.require(raw_angles is not None and np.asarray(raw_angles).shape == (starts.size,),
                 f"{session} native M2 support label topology drift")
    # The slice is the only angle value access; query angles are not read.
    theta_first30 = np.ascontiguousarray(
        np.asarray(raw_angles[: plan.ACTIVITY_STACK_LIMIT], dtype=np.float64)
    )
    rates30: list[np.ndarray] = []
    activities: list[np.ndarray] = []
    for position, start_value in enumerate(starts):
        start = int(start_value)
        stop = int(starts[position + 1]) if position + 1 < starts.size else int(neural.shape[0])
        core.require(stop > start, f"{session} native M2 empty raw trial")
        if position < plan.ACTIVITY_STACK_LIMIT:
            rates30.append(
                np.asarray(neural[start:stop], dtype=np.float64).mean(axis=0)
                / plan.MODEL_BIN_SECONDS
            )
        activities.append(
            _interpolate_raw_trial_activity(neural, eval_mask, start=start, stop=stop)
        )
    rates_hz30 = np.ascontiguousarray(np.stack(rates30), dtype=np.float64)
    activity_rows = np.ascontiguousarray(np.stack(activities), dtype=np.float32)
    core.require(rates_hz30.shape == (30, neural.shape[1])
                 and activity_rows.shape == (starts.size, 100, neural.shape[1]),
                 f"{session} native M2 raw trial view axes drift")
    return neural, starts, theta_first30, rates_hz30, activity_rows


def _surface_starts(dataset: Any, session: str) -> np.ndarray:
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core as activity_core

    starts = np.asarray(
        [start for name, start in dataset.window_indices if name == session], dtype=np.int64,
    )
    core.require(starts.size > 0 and np.all(np.diff(starts) > 0), "native M2 query order drift")
    return activity_core.select_common_post30_window_starts(
        starts, np.asarray(dataset.trial_start_indices[session], dtype=np.int64),
    )


def _query_trial_rows(
    dataset: Any, session: str, *, raw_neural: np.ndarray,
    raw_starts: np.ndarray, activities: np.ndarray,
) -> tuple[dict[str, object], ...]:
    neural_padded = np.asarray(dataset.neural_data[session], dtype=np.float32)
    padded_starts = np.asarray(dataset.trial_start_indices[session], dtype=np.int64)
    common = _surface_starts(dataset, session)
    core.require(
        padded_starts.shape == raw_starts.shape and activities.shape[0] == raw_starts.size
        and np.array_equal(padded_starts, raw_starts + (plan.WINDOW_BINS - 1)),
        "native M2 padded/raw trial chronology drift",
    )
    rows: list[dict[str, object]] = []
    joined: list[np.ndarray] = []
    for position in range(plan.ACTIVITY_STACK_LIMIT, raw_starts.size):
        raw_start = int(raw_starts[position])
        raw_stop = int(raw_starts[position + 1]) if position + 1 < raw_starts.size else int(raw_neural.shape[0])
        padded_start = int(padded_starts[position])
        padded_stop = (
            int(padded_starts[position + 1]) if position + 1 < padded_starts.size else int(neural_padded.shape[0])
        )
        endpoints = common + plan.WINDOW_BINS - 1
        metric = np.ascontiguousarray(
            common[(endpoints >= padded_start) & (endpoints < padded_stop)], dtype=np.int64,
        )
        causal = np.arange(padded_start, padded_stop - plan.WINDOW_BINS + 1, dtype=np.int64)
        core.require(raw_stop > raw_start, "native M2 query trial interval drift")
        joined.append(metric)
        rows.append({
            "position": position, "trial_id": f"{session}:trial:{position}",
            "raw_start": raw_start, "raw_stop": raw_stop,
            "padded_start": padded_start, "padded_stop": padded_stop,
            "metric_starts": metric, "causal_starts": causal,
            "activity": np.ascontiguousarray(activities[position], dtype=np.float32),
        })
    core.require(rows and np.array_equal(np.concatenate(joined), common),
                 "native M2 query trials do not partition common post30 metric surface")
    return tuple(rows)


def _windows(neural: np.ndarray, starts: np.ndarray) -> np.ndarray:
    indices = starts[:, None] + np.arange(plan.WINDOW_BINS, dtype=np.int64)[None, :]
    return np.ascontiguousarray(neural[indices], dtype=np.float32)


def _trial_capabilities(
    *, session: str, row: Mapping[str, object], raw_neural: np.ndarray,
    channel_ids: np.ndarray,
) -> tuple[
    cdm.B3SInterpolatedSpikeCountTrial,
    cdm.NativeRewardedTrialSpikeCounts,
    cdm.VelocityValidityEvidence | None,
]:
    trial_id = str(row["trial_id"])
    channel_sha = cdm.channel_order_digest(channel_ids)
    b3s = cdm.B3SInterpolatedSpikeCountTrial(
        activity=np.asarray(row["activity"], dtype=np.float32), session_id=session,
        trial_id=trial_id, channel_order_sha256=channel_sha,
    )
    raw_start, raw_stop = int(row["raw_start"]), int(row["raw_stop"])
    native = cdm.NativeRewardedTrialSpikeCounts(
        counts=np.ascontiguousarray(raw_neural[raw_start:raw_stop], dtype=np.float32),
        session_id=session, trial_id=trial_id, channel_order_sha256=channel_sha,
        rewarded_interval_start_bin=raw_start, rewarded_interval_stop_bin=raw_stop,
    )
    causal_starts = np.asarray(row["causal_starts"], dtype=np.int64)
    if causal_starts.size == 0:
        return b3s, native, None
    first_endpoint = int(causal_starts[0] + plan.WINDOW_BINS - 1)
    validity = cdm.VelocityValidityEvidence(
        valid_mask=np.ones(causal_starts.size, dtype=np.bool_), session_id=session, trial_id=trial_id,
        prediction_interval_start_bin=first_endpoint,
        prediction_interval_stop_bin=first_endpoint + int(causal_starts.size),
    )
    return b3s, native, validity


def _held_predictions(
    *, torch: Any, model: Any, neural_windows: np.ndarray, activity: np.ndarray,
    carrier_hz: np.ndarray, groups: cdm.ComplementaryGroups,
    validity: cdm.VelocityValidityEvidence, side_mean: np.ndarray, side_std: np.ndarray,
    device: Any, batch_size: int,
) -> tuple[cdm.CompletedVelocityPrediction, ...]:
    values: list[cdm.CompletedVelocityPrediction] = []
    model_t4 = np.ascontiguousarray(carrier_hz * plan.MODEL_BIN_SECONDS, dtype=np.float32)
    for group in range(cdm.GROUP_COUNT):
        held = np.asarray(groups.held_mask(group), dtype=np.bool_)
        keep = ~held
        prediction = shared_physical._predict(
            torch=torch, model=model,
            neural_windows=np.ascontiguousarray(neural_windows[:, :, keep], dtype=np.float32),
            activity=np.ascontiguousarray(activity[:, :, keep], dtype=np.float32),
            raw_t4=np.ascontiguousarray(model_t4[keep], dtype=np.float32),
            side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
        )
        values.append(cdm.CompletedVelocityPrediction(
            np.ascontiguousarray(prediction, dtype=np.float64), validity,
        ))
    return tuple(values)


def _build_memory(
    *, system: str, budget: int, selected: np.ndarray, support_rates_hz30: np.ndarray,
    theta_first30: np.ndarray,
    support_b3s: Sequence[cdm.B3SInterpolatedSpikeCountTrial], channel_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> core.RuntimeMemory:
    rates_hz = np.ascontiguousarray(support_rates_hz30[selected], dtype=np.float64)
    theta = np.ascontiguousarray(theta_first30[selected], dtype=np.float64)
    core.require(np.isfinite(theta).all(), "native M2 selected support includes non-directional row")
    directions = core.canonical_direction_indices(theta)
    carrier, posterior = core.fit_initial_carrier(
        support_rates=rates_hz, direction_indices=directions,
        channel_ids=channel_ids, valid_mask=valid_mask,
    )
    return core.build_runtime_memory(
        system=system, budget=budget,
        support_trials=tuple(support_b3s[index] for index in selected),
        carrier=carrier, posterior=posterior,
    )


def _score_system(
    *, torch: Any, model: Any, dataset: Any, session: str, memory: core.RuntimeMemory,
    query_rows: Sequence[Mapping[str, object]], side_mean: np.ndarray, side_std: np.ndarray,
    raw_neural: np.ndarray, device: Any, batch_size: int,
) -> dict[str, object]:
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    channel_ids = np.arange(neural.shape[1], dtype=np.int64)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts_joined: list[np.ndarray] = []
    transition_sha: list[str] = []
    for row in query_rows:
        metric_starts = np.asarray(row["metric_starts"], dtype=np.int64)
        activity, carrier_hz = memory.prediction_inputs()
        if metric_starts.size:
            prediction = shared_physical._predict(
                torch=torch, model=model, neural_windows=_windows(neural, metric_starts),
                activity=activity,
                raw_t4=np.ascontiguousarray(carrier_hz * plan.MODEL_BIN_SECONDS, dtype=np.float32),
                side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
            )
        else:
            prediction = np.empty((0, 2), dtype=np.float32)
        b3s, native, validity = _trial_capabilities(
            session=session, row=row, raw_neural=raw_neural, channel_ids=channel_ids,
        )
        if memory.system == plan.SYSTEM_ACTIVITY:
            memory.commit_activity_only(b3s)
        elif memory.system in (plan.SYSTEM_ORDINARY, plan.SYSTEM_PRECISION):
            causal_starts = np.asarray(row["causal_starts"], dtype=np.int64)
            if causal_starts.size:
                core.require(isinstance(validity, cdm.VelocityValidityEvidence),
                             "native M2 nonempty causal trial lacks validity evidence")
                causal_windows = _windows(neural, causal_starts)
                held = _held_predictions(
                    torch=torch, model=model, neural_windows=causal_windows, activity=activity,
                    carrier_hz=carrier_hz, groups=memory.ordinary.state.carrier.groups,
                    validity=validity, side_mean=side_mean, side_std=side_std,
                    device=device, batch_size=batch_size,
                )
            else:
                # The typed independent-activity state machine interprets an
                # empty velocity tuple as VELOCITY_SHAPE: activity still
                # advances, while the carrier remains exact.  No padding or
                # cross-trial history is invented for a <50-bin trial.
                core.require(validity is None, "native M2 empty causal trial synthesized validity")
                held = ()
            evidence = memory.observe_and_commit(
                b3s_trial_activity=b3s, native_counts=native, complementary_predictions=held,
            )
            transition_sha.append(core.object_sha256(evidence))
        if metric_starts.size:
            target = np.ascontiguousarray(
                behavior[metric_starts + plan.WINDOW_BINS - 1], dtype=np.float32,
            )
            core.require(target.shape == prediction.shape and np.isfinite(target).all(),
                         "native M2 target topology drift")
            predictions.append(prediction); targets.append(target); starts_joined.append(metric_starts)
    prediction = np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
    starts = np.ascontiguousarray(np.concatenate(starts_joined), dtype=np.int64)
    expected = _surface_starts(dataset, session)
    core.require(np.array_equal(starts, expected), "native M2 scored query differs from post30 authority")
    return {
        "window_count": int(starts.size), "query_starts_sha256": core.array_sha256(starts),
        "target_sha256": core.array_sha256(target), "prediction_sha256": core.array_sha256(prediction),
        "r2": core.variance_weighted_r2(target, prediction),
        "accepted_carrier_updates": memory.accepted_carrier_updates,
        "precision_rejections": memory.precision_rejections,
        "other_carrier_rejections": memory.other_carrier_rejections,
        "transition_evidence_sha256": core.object_sha256(transition_sha), "target_state_uses": 0,
    }


def _score_m30(
    *, torch: Any, model: Any, dataset: Any, session: str, raw_t4: np.ndarray,
    side_mean: np.ndarray, side_std: np.ndarray, device: Any, batch_size: int,
) -> dict[str, object]:
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    behavior = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    starts = _surface_starts(dataset, session)
    activity = np.asarray(dataset.calib_trialized_neural_features[session][:30], dtype=np.float32)
    prediction = shared_physical._predict(
        torch=torch, model=model, neural_windows=_windows(neural, starts), activity=activity,
        raw_t4=raw_t4, side_mean=side_mean, side_std=side_std,
        device=device, batch_size=batch_size,
    )
    target = np.ascontiguousarray(behavior[starts + plan.WINDOW_BINS - 1], dtype=np.float32)
    return {
        "window_count": int(starts.size), "query_starts_sha256": core.array_sha256(starts),
        "target_sha256": core.array_sha256(target), "prediction_sha256": core.array_sha256(prediction),
        "r2": core.variance_weighted_r2(target, prediction), "accepted_carrier_updates": 0,
        "precision_rejections": 0, "other_carrier_rejections": 0,
        "transition_evidence_sha256": core.object_sha256([]), "target_state_uses": 0,
    }


def _publish(root: Path, payload: Mapping[str, object]) -> str:
    core.require(not root.exists(), f"result root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        raw = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        score = temporary / "score.json"; score.write_bytes(raw); score.chmod(0o444)
        digest = hashlib.sha256(raw).hexdigest()
        sidecar = temporary / "score.json.sha256"
        sidecar.write_text(f"{digest}  score.json\n", encoding="ascii"); sidecar.chmod(0o444)
        temporary.rename(root); root.chmod(0o555)
        return digest
    except BaseException:
        if temporary.exists(): shutil.rmtree(temporary)
        raise


def execute(repo_root: Path, *, gpu_index: int = 1, batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, object]:
    shared_physical._validate_launch_environment(gpu_index=gpu_index)
    output_root = plan.result_root(repo_root)
    core.require(not output_root.exists(), f"result root already exists: {output_root}")
    import torch
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model, data_module, _task, metadata = load_frozen_model_and_data()
    model = model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    side_mean = np.asarray(metadata["normalization_mean"], dtype=np.float32)
    side_std = np.asarray(metadata["normalization_std"], dtype=np.float32)
    surfaces = {
        "within_post30": (data_module.train_dataset, data_module.train_calib_heldin_sessions),
        "external_post30_local": (
            data_module.val_heldout_dataset, data_module.val_calib_heldout_sessions,
        ),
    }
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    for surface, (dataset, raw_sessions) in surfaces.items():
        core.require(dataset is not None, f"{surface} dataset absent")
        sessions = sorted(dataset.calib_trialized_neural_features)
        core.require(len(sessions) == plan.EXPECTED_SESSIONS_BY_SURFACE[surface],
                     f"{surface} session count drift")
        for session in sessions:
            core.require(session in raw_sessions, f"{surface}/{session} raw-session authority absent")
            raw_neural, raw_starts, theta30, support_rates_hz30, activities = _native_trial_views(
                raw_sessions[session], session=session,
            )
            selected = {
                4: core.select_m4_support(theta30),
                10: _finite_m10_indices(theta30),
            }
            raw_m30_hz, valid_mask = _raw_fixed_ridge_m30(support_rates_hz30, theta30)
            raw_m30_model = np.ascontiguousarray(
                raw_m30_hz * plan.MODEL_BIN_SECONDS, dtype=np.float32,
            )
            query_rows = _query_trial_rows(
                dataset, session, raw_neural=raw_neural,
                raw_starts=raw_starts, activities=activities,
            )
            channels = np.arange(activities.shape[2], dtype=np.int64)
            channel_sha = cdm.channel_order_digest(channels)
            support_b3s = tuple(
                cdm.B3SInterpolatedSpikeCountTrial(
                    activity=np.ascontiguousarray(activities[index], dtype=np.float32),
                    session_id=session, trial_id=f"{session}:trial:{index}",
                    channel_order_sha256=channel_sha,
                )
                for index in range(30)
            )
            common = {
                "surface": surface, "session_id": session, "seed": plan.SEED,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "normalization_sha256": metadata["normalization_sha256"],
                "raw_m30_t4_hz_sha256": core.array_sha256(raw_m30_hz),
                "raw_m30_t4_model_count_per_bin_sha256": core.array_sha256(raw_m30_model),
                "raw_m30_valid_mask_sha256": core.array_sha256(valid_mask),
                "carrier_state_units": "hz", "model_t4_units": "counts_per_20ms_bin",
                "carrier_to_model_scale": plan.MODEL_BIN_SECONDS,
                "carrier_support_rate_domain": "complete_raw_rewarded_trial_mean_hz",
                "b3s_rate_domain": "per_raw_trial_calibration_visible_bins_interpolated_to_100",
                "short_query_trials_without_complete_50_bin_window": int(sum(
                    np.asarray(row["causal_starts"], dtype=np.int64).size == 0
                    for row in query_rows
                )),
                "short_trial_transition_rule": (
                    "activity_advances_carrier_rejects_velocity_shape_no_padding_no_cross_trial_history"
                ),
                "target_gradients": 0, "target_backward": 0, "parameter_updates": 0,
            }
            m30 = _score_m30(
                torch=torch, model=model, dataset=dataset, session=session, raw_t4=raw_m30_model,
                side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
            )
            for system in plan.SYSTEMS_BY_BUDGET[30]:
                rows.append({
                    **common, "budget": 30, "system": system, "cell": f"m30_{system}", **m30,
                    "m30_precision_literal_noop": system == plan.SYSTEM_PRECISION,
                    "reused_prediction_from_static": system == plan.SYSTEM_PRECISION,
                })
            for budget in plan.TRANSITION_BUDGETS:
                for system in plan.SYSTEMS_BY_BUDGET[budget]:
                    memory = _build_memory(
                        system=system, budget=budget, selected=selected[budget],
                        support_rates_hz30=support_rates_hz30, theta_first30=theta30,
                        support_b3s=support_b3s, channel_ids=channels,
                        valid_mask=valid_mask,
                    )
                    result = _score_system(
                        torch=torch, model=model, dataset=dataset, session=session, memory=memory,
                        query_rows=query_rows, side_mean=side_mean, side_std=side_std,
                        raw_neural=raw_neural, device=device, batch_size=batch_size,
                    )
                    rows.append({
                        **common, "budget": budget, "system": system, "cell": f"m{budget}_{system}",
                        "support_indices": selected[budget].tolist(),
                        "support_indices_sha256": core.array_sha256(selected[budget]), **result,
                    })
    core.require(len(rows) == plan.EXPECTED_ROWS, "native M2 screen row count drift")
    for key in sorted({(str(row["surface"]), str(row["session_id"])) for row in rows}):
        aligned = [row for row in rows if (row["surface"], row["session_id"]) == key]
        core.require(len({row["query_starts_sha256"] for row in aligned}) == 1
                     and len({row["target_sha256"] for row in aligned}) == 1,
                     f"native M2 {key} common-input drift")
        m30_rows = [row for row in aligned if row["budget"] == 30]
        core.require(m30_rows[0]["prediction_sha256"] == m30_rows[1]["prediction_sha256"]
                     and m30_rows[0]["r2"] == m30_rows[1]["r2"], f"native M2 {key} M30 no-op drift")
    summary = {
        surface: core.summarize_rows(
            [
                {"cell": row["cell"], "session_id": row["session_id"], "r2": row["r2"]}
                for row in rows if row["surface"] == surface
            ],
            expected_sessions=plan.EXPECTED_SESSIONS_BY_SURFACE[surface],
        )
        for surface in plan.SURFACES
    }
    torch.cuda.synchronize(device)
    payload: dict[str, object] = {
        "schema": plan.SCHEMA, "status": "TERMINAL", "scientific_role": "native_m2_local_trial_aware_engineering_screen",
        "result_root": str(output_root), "seed": plan.SEED,
        "protocol": {
            "surfaces": list(plan.SURFACES), "query": "post_first30_common_windows",
            "m4_support": "doptimal_four_from_first30", "m10_support": "first10_finite_direction_from_first30",
            "short_trial_policy": (
                "activity_advances_carrier_rejects_velocity_shape_no_padding_no_cross_trial_history"
            ),
            "official_submission_claimed": False, "target_state_uses": 0,
            "target_gradients": 0, "target_backward": 0, "parameter_updates": 0,
        },
        "device": {"cuda_visible_devices": str(gpu_index), "logical_device": "cuda:0",
                   "name": torch.cuda.get_device_name(0), "batch_size": batch_size,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                   "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "elapsed_seconds": float(time.monotonic() - started), "cell_order": list(plan.CELL_ORDER),
        "row_count": len(rows), "rows": rows, "summary": summary,
    }
    payload["score_sha256"] = _publish(output_root, payload)
    return payload
