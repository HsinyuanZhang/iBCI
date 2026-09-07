"""Frozen-weight pseudo-MUA Precision-CDM V2 engineering evaluator."""

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

from . import core, plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trial_id(session: str, row: Mapping[str, object]) -> str:
    return f"{session}:trial:{int(row['trial_index'])}"


def _interval(row: Mapping[str, object]) -> tuple[int, int]:
    start, stop = int(row["start"]), int(row["stop"])
    core.require(start >= 0 and stop - start >= plan.WINDOW_BINS, "rewarded trial interval drift")
    return start, stop


def _query_windows(neural: np.ndarray, *, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(start, stop - plan.WINDOW_BINS + 1, dtype=np.int64)
    core.require(starts.size > 0, "query trial has no complete window")
    indices = starts[:, None] + np.arange(plan.WINDOW_BINS, dtype=np.int64)[None, :]
    return np.ascontiguousarray(neural[indices], dtype=np.float32), starts


def _predict(
    *, torch: Any, model: Any, neural_windows: np.ndarray, activity: np.ndarray,
    raw_t4: np.ndarray, side_mean: np.ndarray, side_std: np.ndarray,
    device: Any, batch_size: int,
) -> np.ndarray:
    student = model.student
    core.require(getattr(student, "decoder_mode", None) == "coupled", "screen requires coupled decoder")
    core.require(getattr(student, "fixed_slot_router", None) is None, "screen forbids fixed-slot routing")
    core.require(not hasattr(student.id_encoder, "forward_batch_with_gate"),
                 "screen forbids hidden identity gate semantics")
    calibration = torch.from_numpy(np.ascontiguousarray(activity, dtype=np.float32)).unsqueeze(0).to(device)
    side_np = np.ascontiguousarray((raw_t4 - side_mean) / side_std, dtype=np.float32)
    side = torch.from_numpy(side_np).unsqueeze(0).to(device)
    values: list[np.ndarray] = []
    with torch.inference_mode():
        identity = student.compute_identity(calibration, side_features=side)
        core.require(tuple(identity.shape) == (1, neural_windows.shape[2], plan.WINDOW_BINS),
                     "screen identity shape drift")
        for offset in range(0, neural_windows.shape[0], batch_size):
            neural = torch.from_numpy(neural_windows[offset : offset + batch_size]).to(device)
            prediction = student.decode_with_identity(neural, identity)
            values.append(
                prediction[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False)
                / plan.BEHAVIOR_SCALE
            )
    result = np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)
    core.require(result.shape == (neural_windows.shape[0], 2) and np.isfinite(result).all(),
                 "screen prediction topology drift")
    return result


def _held_predictions(
    *, torch: Any, model: Any, neural_windows: np.ndarray, activity: np.ndarray,
    raw_t4: np.ndarray, groups: cdm.ComplementaryGroups, validity: cdm.VelocityValidityEvidence,
    side_mean: np.ndarray, side_std: np.ndarray, behavior_mean: np.ndarray,
    behavior_std: np.ndarray, device: Any, batch_size: int,
) -> tuple[cdm.CompletedVelocityPrediction, ...]:
    result: list[cdm.CompletedVelocityPrediction] = []
    for group in range(cdm.GROUP_COUNT):
        held = np.asarray(groups.held_mask(group), dtype=np.bool_)
        keep = ~held
        core.require(int(held.sum()) > 0 and int(keep.sum()) > 0, "complementary group is empty/full")
        normalized_prediction = _predict(
            torch=torch, model=model,
            neural_windows=np.ascontiguousarray(neural_windows[:, :, keep], dtype=np.float32),
            activity=np.ascontiguousarray(activity[:, :, keep], dtype=np.float32),
            raw_t4=np.ascontiguousarray(raw_t4[keep], dtype=np.float32),
            side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
        )
        physical = np.ascontiguousarray(
            normalized_prediction.astype(np.float64) * behavior_std[None, :] + behavior_mean[None, :],
            dtype=np.float64,
        )
        result.append(cdm.CompletedVelocityPrediction(physical, validity))
    return tuple(result)


def _support_rate_matrix(
    *, nwb_path: Path, trials: Sequence[Mapping[str, object]], unit_side: Any, multisession: Any,
) -> tuple[np.ndarray, np.ndarray]:
    from pynwb import NWBHDF5IO

    unit_rates, unit_count = unit_side._pool_trial_rate_matrix(
        nwb_path, list(trials[: plan.ACTIVITY_STACK_LIMIT])
    )
    core.require(unit_rates.shape == (unit_count, plan.ACTIVITY_STACK_LIMIT),
                 "pseudo-MUA source unit-rate matrix drift")
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        core.require(nwb.units is not None, "NWB units table missing")
        electrode_ids = multisession.electrode_ids_from_units(nwb.units.to_dataframe())
    pooled, channel_ids = unit_side.pool_trial_rates_by_electrode(unit_rates, electrode_ids)
    result = np.ascontiguousarray(pooled.T, dtype=np.float64)
    channels = np.ascontiguousarray(channel_ids, dtype=np.int64)
    core.require(result.shape == (plan.ACTIVITY_STACK_LIMIT, channels.size),
                 "pseudo-MUA pooled support rate topology drift")
    return result, channels


def _make_trial_capabilities(
    *, multisession: Any, session: str, trial_id: str, neural: np.ndarray,
    channel_ids: np.ndarray, start: int, stop: int, endpoint_count: int,
) -> tuple[cdm.B3SInterpolatedSpikeCountTrial, cdm.NativeRewardedTrialSpikeCounts, cdm.VelocityValidityEvidence]:
    activity = multisession._build_calib_trials(
        neural, [{"start": start, "stop": stop}], 1, 100, neural.shape[1], -1.0, True,
    )[0]
    channel_sha = cdm.channel_order_digest(channel_ids)
    b3s = cdm.B3SInterpolatedSpikeCountTrial(
        activity=np.ascontiguousarray(activity, dtype=np.float32), session_id=session,
        trial_id=trial_id, channel_order_sha256=channel_sha,
    )
    native = cdm.NativeRewardedTrialSpikeCounts(
        counts=np.ascontiguousarray(neural[start:stop], dtype=np.float32), session_id=session,
        trial_id=trial_id, channel_order_sha256=channel_sha,
        rewarded_interval_start_bin=start, rewarded_interval_stop_bin=stop,
    )
    validity = cdm.VelocityValidityEvidence(
        valid_mask=np.ones(endpoint_count, dtype=np.bool_), session_id=session, trial_id=trial_id,
        prediction_interval_start_bin=start + plan.WINDOW_BINS - 1,
        prediction_interval_stop_bin=stop,
    )
    return b3s, native, validity


def _build_memory(
    *, system: str, budget: int, support_indices: np.ndarray, support_rates30: np.ndarray,
    support_directions30: np.ndarray, support_b3s30: Sequence[cdm.B3SInterpolatedSpikeCountTrial],
    channel_ids: np.ndarray, valid_mask: np.ndarray,
) -> core.RuntimeMemory:
    rates = np.ascontiguousarray(support_rates30[support_indices], dtype=np.float64)
    directions = np.ascontiguousarray(support_directions30[support_indices], dtype=np.int64)
    if budget == 30:
        fitted = cdm.fit_carriers_from_trial_table(
            rates, directions, mode=cdm.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
            normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
        )
        config = cdm.CDMDConfig(
            support_budget_m=30, active_fit_mode=cdm.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        carrier = cdm.CarrierMemory.from_support_trials(
            initial_raw_t4=fitted, channel_ids=channel_ids, support_trial_rates=rates,
            support_direction_indices=directions, config=config, valid_mask=valid_mask,
        )
        return core.build_runtime_memory(
            system=plan.SYSTEM_STATIC, budget=30,
            support_trials=tuple(support_b3s30[index] for index in support_indices),
            carrier=carrier, posterior=None,
        )
    carrier, posterior = core.fit_initial_carrier(
        support_rates=rates, direction_indices=directions,
        channel_ids=channel_ids, valid_mask=valid_mask,
    )
    return core.build_runtime_memory(
        system=system, budget=budget,
        support_trials=tuple(support_b3s30[index] for index in support_indices),
        carrier=carrier, posterior=posterior,
    )


def _score_system(
    *, torch: Any, model: Any, record: Any, trials: Sequence[Mapping[str, object]],
    memory: core.RuntimeMemory, side_mean: np.ndarray, side_std: np.ndarray,
    behavior_mean: np.ndarray, behavior_std: np.ndarray, device: Any, batch_size: int,
    multisession: Any,
) -> dict[str, object]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    query_starts: list[np.ndarray] = []
    transition_sha: list[str] = []
    neural = np.ascontiguousarray(record.neural, dtype=np.float32)
    channel_ids = np.ascontiguousarray(record.channel_ids, dtype=np.int64)
    for row in trials[plan.ACTIVITY_STACK_LIMIT :]:
        start, stop = _interval(row)
        windows, starts = _query_windows(neural, start=start, stop=stop)
        trial_id = _trial_id(record.name, row)
        activity, active_t4 = memory.prediction_inputs()
        prediction = _predict(
            torch=torch, model=model, neural_windows=windows, activity=activity, raw_t4=active_t4,
            side_mean=side_mean, side_std=side_std, device=device, batch_size=batch_size,
        )
        b3s, native, validity = _make_trial_capabilities(
            multisession=multisession, session=record.name, trial_id=trial_id,
            neural=neural, channel_ids=channel_ids, start=start, stop=stop,
            endpoint_count=starts.size,
        )
        if memory.system == plan.SYSTEM_ACTIVITY:
            memory.commit_activity_only(b3s)
        elif memory.system in (plan.SYSTEM_ORDINARY, plan.SYSTEM_PRECISION):
            held = _held_predictions(
                torch=torch, model=model, neural_windows=windows, activity=activity,
                raw_t4=active_t4, groups=memory.ordinary.state.carrier.groups,
                validity=validity, side_mean=side_mean, side_std=side_std,
                behavior_mean=behavior_mean, behavior_std=behavior_std,
                device=device, batch_size=batch_size,
            )
            evidence = memory.observe_and_commit(
                b3s_trial_activity=b3s, native_counts=native,
                complementary_predictions=held,
            )
            transition_sha.append(core.object_sha256(evidence))
        # Behavior is read only after the state decision for this trial.
        target = np.ascontiguousarray(record.behavior[starts + plan.WINDOW_BINS - 1], dtype=np.float32)
        core.require(target.shape == prediction.shape and np.isfinite(target).all(),
                     "query target topology drift")
        predictions.append(prediction)
        targets.append(target)
        query_starts.append(starts)
    joined_prediction = np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
    joined_target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
    joined_starts = np.ascontiguousarray(np.concatenate(query_starts), dtype=np.int64)
    core.require(np.array_equal(joined_starts, np.asarray(record.valid_starts, dtype=np.int64)),
                 "screen query windows differ from exact post-first30 parser surface")
    return {
        "window_count": int(joined_starts.size),
        "query_starts_sha256": core.array_sha256(joined_starts),
        "target_sha256": core.array_sha256(joined_target),
        "prediction_sha256": core.array_sha256(joined_prediction),
        "r2": core.variance_weighted_r2(joined_target, joined_prediction),
        "accepted_carrier_updates": memory.accepted_carrier_updates,
        "precision_rejections": memory.precision_rejections,
        "other_carrier_rejections": memory.other_carrier_rejections,
        "transition_evidence_sha256": core.object_sha256(transition_sha),
        "target_state_uses": 0,
        "prediction": joined_prediction,
        "target": joined_target,
    }


def _publish(root: Path, payload: Mapping[str, object]) -> str:
    core.require(not root.exists(), f"result root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        raw = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        score = temporary / "score.json"
        score.write_bytes(raw)
        score.chmod(0o444)
        digest = hashlib.sha256(raw).hexdigest()
        sidecar = temporary / "score.json.sha256"
        sidecar.write_text(f"{digest}  score.json\n", encoding="ascii")
        sidecar.chmod(0o444)
        temporary.rename(root)
        root.chmod(0o555)
        return digest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_launch_environment(*, gpu_index: int) -> None:
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_index),
                 "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
                 "CUDA_DEVICE_ORDER mismatch")
    core.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
                 "deterministic CUBLAS_WORKSPACE_CONFIG mismatch")


def execute(repo_root: Path, *, gpu_index: int = 1, batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, object]:
    core.require(not isinstance(gpu_index, bool) and gpu_index >= 0, "gpu_index drift")
    core.require(not isinstance(batch_size, bool) and batch_size >= 1, "batch_size drift")
    _validate_launch_environment(gpu_index=gpu_index)
    output_root = plan.result_root(repo_root)
    core.require(not output_root.exists(), f"result root already exists: {output_root}")

    from tfpd_exploration.src.sua_paired_activity_budget_screen_v1 import physical as paired
    from sua_exploration.mc_maze import multisession_datamodule, unit_side_features
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity
    from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime

    contract, authority = paired._relocated_contract(repo_root)
    nwb_root = (Path(repo_root) / plan.NWB_ROOT_RELATIVE).resolve()
    paired._verify_selected_inputs(contract, nwb_root=nwb_root)
    owners = runtime._runtime_owners(Path(repo_root))
    torch = owners["torch"]
    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    model = runtime._load_model(contract.checkpoint("shared_t4", plan.SEED), contract, str(device), owners)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    behavior_mean, behavior_std = runtime._load_mean_std(
        contract.behavior_normalizer_paths[plan.VIEW], label=f"{plan.VIEW} behavior",
    )
    side_mean, side_std = runtime._load_mean_std(
        contract.side_normalizer_paths[plan.VIEW], label=f"{plan.VIEW} side",
    )
    behavior_mean = np.ascontiguousarray(behavior_mean, dtype=np.float64)
    behavior_std = np.ascontiguousarray(behavior_std, dtype=np.float64)
    side_mean = np.ascontiguousarray(side_mean, dtype=np.float32)
    side_std = np.ascontiguousarray(side_std, dtype=np.float32)

    started = time.monotonic()
    rows: list[dict[str, object]] = []
    for cohort in contract.cohort:
        nwb_path = nwb_root / cohort.frozen_path
        trials = tuple(owners["list_datamodule_rewarded_trials"](
            nwb_path, bin_size_ms=parity.BIN_SIZE_MS, window_size=parity.HISTORY_BINS,
            trial_result_filter="R",
        ))
        core.require(len(trials) > plan.ACTIVITY_STACK_LIMIT, f"{cohort.session_id} lacks query trials")
        record = owners["load_dandi688_session"](
            nwb_path, bin_size_ms=parity.BIN_SIZE_MS, window_size=parity.HISTORY_BINS,
            calibration_n_trials=plan.ACTIVITY_STACK_LIMIT, max_trial_length=100,
            pad_value=-1.0, interpolate_trials=True,
            behavior_mean=behavior_mean.astype(np.float32), behavior_std=behavior_std.astype(np.float32),
            trial_result_filter="R", exclude_calibration_trials_from_windows=True,
            cache_dir=None, signal_view=plan.VIEW,
        )
        core.require(record.name == cohort.session_id and record.signal_view == plan.VIEW,
                     "pseudo-MUA record identity drift")
        support_rates30, channel_ids = _support_rate_matrix(
            nwb_path=nwb_path, trials=trials, unit_side=unit_side_features,
            multisession=multisession_datamodule,
        )
        core.require(np.array_equal(channel_ids, np.asarray(record.channel_ids, dtype=np.int64)),
                     "pseudo-MUA channel order differs across parser/rate views")
        theta30 = np.ascontiguousarray(
            [float(row["target_dir"]) if row.get("target_dir") is not None else np.nan
             for row in trials[: plan.ACTIVITY_STACK_LIMIT]], dtype=np.float64,
        )
        core.require(np.isfinite(theta30).all(), "first30 support direction missing")
        directions30 = core.canonical_direction_indices(theta30)
        selected = {
            30: np.arange(30, dtype=np.int64),
            10: np.arange(10, dtype=np.int64),
            4: core.select_m4_support(theta30),
        }
        raw_m30, raw_metadata = unit_side_features.compute_unit_side_features_uncached(
            nwb_path, feature_group="t4", pool_size=30, bin_size_ms=20,
            window_size=50, trial_result_filter="R", signal_view=plan.VIEW,
        )
        raw_m30 = np.ascontiguousarray(raw_m30, dtype=np.float32)
        valid_mask = np.ascontiguousarray(raw_m30[:, 2] > float(unit_side_features.MODULATION_EPS), dtype=np.bool_)
        core.require(raw_m30.shape == (channel_ids.size, 4) and int(valid_mask.sum()) >= 4,
                     "pseudo-MUA raw M30 validity axis drift")
        support_b3s: list[cdm.B3SInterpolatedSpikeCountTrial] = []
        for row in trials[:30]:
            start, stop = _interval(row)
            b3s, _native, _validity = _make_trial_capabilities(
                multisession=multisession_datamodule, session=record.name,
                trial_id=_trial_id(record.name, row), neural=np.asarray(record.neural, dtype=np.float32),
                channel_ids=channel_ids, start=start, stop=stop,
                endpoint_count=stop - start - plan.WINDOW_BINS + 1,
            )
            support_b3s.append(b3s)
        core.require(
            np.array_equal(
                np.stack([item.activity for item in support_b3s]),
                np.asarray(record.calib_trials, dtype=np.float32),
            ),
            "pseudo-MUA support B3S reconstruction drift",
        )

        m30 = _build_memory(
            system=plan.SYSTEM_STATIC, budget=30, support_indices=selected[30],
            support_rates30=support_rates30, support_directions30=directions30,
            support_b3s30=support_b3s, channel_ids=channel_ids, valid_mask=valid_mask,
        )
        m30_result = _score_system(
            torch=torch, model=model, record=record, trials=trials, memory=m30,
            side_mean=side_mean, side_std=side_std, behavior_mean=behavior_mean,
            behavior_std=behavior_std, device=device, batch_size=batch_size,
            multisession=multisession_datamodule,
        )
        common = {
            "asset_id": cohort.asset_id, "session_id": cohort.session_id, "view": plan.VIEW,
            "seed": plan.SEED, "support_direction_sha256": core.array_sha256(theta30),
            "raw_m30_t4_sha256": core.array_sha256(raw_m30),
            "raw_m30_valid_mask_sha256": core.array_sha256(valid_mask),
            "raw_m30_metadata_pool_size": int(raw_metadata.pool_size),
            "target_gradients": 0, "target_backward": 0, "parameter_updates": 0,
        }
        for system in plan.SYSTEMS_BY_BUDGET[30]:
            rows.append({
                **common, "budget": 30, "system": system, "cell": f"m30_{system}",
                **{key: value for key, value in m30_result.items() if key not in ("prediction", "target")},
                "m30_precision_literal_noop": system == plan.SYSTEM_PRECISION,
                "reused_prediction_from_static": system == plan.SYSTEM_PRECISION,
            })
        for budget in plan.TRANSITION_BUDGETS:
            for system in plan.SYSTEMS_BY_BUDGET[budget]:
                memory = _build_memory(
                    system=system, budget=budget, support_indices=selected[budget],
                    support_rates30=support_rates30, support_directions30=directions30,
                    support_b3s30=support_b3s, channel_ids=channel_ids, valid_mask=valid_mask,
                )
                result = _score_system(
                    torch=torch, model=model, record=record, trials=trials, memory=memory,
                    side_mean=side_mean, side_std=side_std, behavior_mean=behavior_mean,
                    behavior_std=behavior_std, device=device, batch_size=batch_size,
                    multisession=multisession_datamodule,
                )
                rows.append({
                    **common, "budget": budget, "system": system, "cell": f"m{budget}_{system}",
                    "support_indices": selected[budget].tolist(),
                    "support_indices_sha256": core.array_sha256(selected[budget]),
                    **{key: value for key, value in result.items() if key not in ("prediction", "target")},
                })

    core.require(len(rows) == plan.EXPECTED_ROWS, "pseudo-MUA screen row cardinality drift")
    # Exact common-input proof inside each session.
    for session in sorted({str(row["session_id"]) for row in rows}):
        aligned = [row for row in rows if row["session_id"] == session]
        core.require(len({row["query_starts_sha256"] for row in aligned}) == 1
                     and len({row["target_sha256"] for row in aligned}) == 1,
                     f"{session} screen cells do not share exact query/target")
        m30_rows = [row for row in aligned if row["budget"] == 30]
        core.require(len(m30_rows) == 2 and m30_rows[0]["prediction_sha256"] == m30_rows[1]["prediction_sha256"]
                     and m30_rows[0]["r2"] == m30_rows[1]["r2"],
                     f"{session} M30 precision no-op drift")
    summary = core.summarize_rows(rows)
    torch.cuda.synchronize(device)
    payload: dict[str, object] = {
        "schema": plan.SCHEMA, "status": "TERMINAL", "scientific_role": "cross_view_engineering_replication",
        "result_root": str(output_root), "view": plan.VIEW, "seed": plan.SEED,
        "authority": authority,
        "protocol": {
            "query": "all_valid_windows_strictly_inside_rewarded_trials_after_first30",
            "support": {"m4": "doptimal_from_first30", "m10": "chronological_first10", "m30": "chronological_first30"},
            "target_state_uses": 0, "target_gradients": 0, "target_backward": 0,
            "parameter_updates": 0, "normalizer_refit": False,
        },
        "device": {
            "cuda_visible_devices": str(gpu_index), "logical_device": "cuda:0",
            "name": torch.cuda.get_device_name(0), "batch_size": batch_size,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "cell_order": list(plan.CELL_ORDER), "row_count": len(rows), "rows": rows,
        "summary": summary,
    }
    payload["score_sha256"] = _publish(output_root, payload)
    return payload
