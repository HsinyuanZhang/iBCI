"""Frozen-model paired sorted-SUA/pseudo-MUA activity-budget evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from . import core, plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o444


def _resolve_pinned_path(raw_path: str, *, repo_root: Path, label: str) -> Path:
    direct = Path(raw_path).expanduser()
    if direct.is_file() and not direct.is_symlink():
        return direct.resolve()
    try:
        index = direct.parts.index("sua_exploration")
    except ValueError as error:
        raise core.ScreenError(f"cannot relocate {label}: {raw_path}") from error
    candidate = (repo_root / Path(*direct.parts[index:])).resolve()
    core.require(candidate.is_file() and not candidate.is_symlink(),
                 f"missing relocated {label}: {candidate}")
    return candidate


def _relocated_contract(repo_root: Path) -> tuple[Any, dict[str, object]]:
    from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime

    manifest_path = repo_root / plan.V9_ROOT_RELATIVE / "run_manifest.json"
    core.require(_sha256_file(manifest_path) == plan.V9_MANIFEST_SHA256, "V9 manifest SHA drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core.require(manifest.get("schema") == runtime.RUNTIME_SCHEMA, "V9 manifest schema drift")
    core.require(manifest.get("status") == "PHASE_A_FORWARD_ONLY_NO_METRIC", "V9 status drift")
    core.require(manifest.get("contract_sha256") == plan.V9_CONTRACT_SHA256,
                 "V9 original contract SHA drift")
    value = manifest.get("contract")
    core.require(isinstance(value, Mapping), "V9 contract missing")
    normalizers = value["normalizers"]
    checkpoints = []
    relocation_rows: list[dict[str, object]] = []
    for item in value["models"]:
        if item["arm"] == "shared_t4":
            path = _resolve_pinned_path(str(item["path"]), repo_root=repo_root,
                                        label=f"{item['arm']}/seed{item['seed']} checkpoint")
            closure = _resolve_pinned_path(str(item["closure_path"]), repo_root=repo_root,
                                           label=f"{item['arm']}/seed{item['seed']} closure")
            core.require(path.stat().st_size == int(item["bytes"]) and _sha256_file(path) == item["sha256"],
                         f"checkpoint content drift {item['arm']}/seed{item['seed']}")
            core.require(closure.stat().st_size == int(item["closure_bytes"]) and _sha256_file(closure) == item["closure_sha256"],
                         f"closure content drift {item['arm']}/seed{item['seed']}")
            relocation_rows.append({
                "kind": "checkpoint", "arm": item["arm"], "seed": item["seed"],
                "sha256": item["sha256"], "local_path": str(path),
            })
        else:
            # RuntimeContract retains the original nine-slot manifest shape,
            # but this successor never resolves, verifies, or loads the six
            # zero/shuffle controls that are outside its declared cell matrix.
            path = Path(str(item["path"]))
            closure = Path(str(item["closure_path"]))
        checkpoints.append(runtime.CheckpointSpec(
            arm=str(item["arm"]), seed=int(item["seed"]), path=str(path),
            sha256=str(item["sha256"]), bytes=int(item["bytes"]),
            closure_path=str(closure), closure_sha256=str(item["closure_sha256"]),
            closure_bytes=int(item["closure_bytes"]),
        ))
    teacher_item = value["teacher"]
    teacher = _resolve_pinned_path(str(teacher_item["path"]), repo_root=repo_root, label="teacher")
    core.require(teacher.stat().st_size == int(teacher_item["bytes"]) and _sha256_file(teacher) == teacher_item["sha256"],
                 "teacher content drift")
    behavior_paths: dict[str, str] = {}
    side_paths: dict[str, str] = {}
    for view in plan.VIEWS:
        behavior = _resolve_pinned_path(str(normalizers[view]["behavior_path"]), repo_root=repo_root,
                                        label=f"{view} behavior normalizer")
        side = _resolve_pinned_path(str(normalizers[view]["side_path"]), repo_root=repo_root,
                                    label=f"{view} side normalizer")
        core.require(_sha256_file(behavior) == normalizers[view]["behavior_sha256"],
                     f"{view} behavior normalizer drift")
        core.require(_sha256_file(side) == normalizers[view]["side_sha256"],
                     f"{view} side normalizer drift")
        behavior_paths[view] = str(behavior)
        side_paths[view] = str(side)
    contract = runtime.RuntimeContract(
        cohort=tuple(runtime.CohortSession(**dict(row)) for row in value["cohort"]),
        checkpoints=tuple(checkpoints),
        teacher_path=str(teacher), teacher_sha256=str(teacher_item["sha256"]),
        teacher_bytes=int(teacher_item["bytes"]),
        behavior_normalizer_paths=behavior_paths,
        behavior_normalizer_sha256={view: str(normalizers[view]["behavior_sha256"]) for view in plan.VIEWS},
        side_normalizer_paths=side_paths,
        side_normalizer_sha256={view: str(normalizers[view]["side_sha256"]) for view in plan.VIEWS},
        cohort_receipt_sha256=str(value["preflight"]["receipt_sha256"]),
        scope_manifest_sha256=str(value["preflight"]["scope_manifest_sha256"]),
        query_map_sha256=str(value["preflight"]["query_map_sha256"]),
    )
    core.require(len(contract.cohort) == plan.EXPECTED_SESSIONS, "cohort count drift")
    return contract, {
        "v9_manifest_path": str(manifest_path),
        "v9_manifest_sha256": plan.V9_MANIFEST_SHA256,
        "original_contract_sha256": plan.V9_CONTRACT_SHA256,
        "relocated_contract_sha256": contract.sha256,
        "relocation_is_content_identity_only": True,
        "relocated_checkpoint_rows": relocation_rows,
        "loaded_model_arms": ["shared_t4"],
        "unused_v9_model_arms_not_resolved_or_loaded": ["shared_zero4", "shared_ts4"],
    }


def _verify_selected_inputs(contract: Any, *, nwb_root: Path) -> None:
    """Rehash exactly the inputs reachable by this successor."""
    for row in contract.cohort:
        path = nwb_root / row.frozen_path
        core.require(path.is_file() and not path.is_symlink(), f"missing NWB {row.asset_id}")
        core.require(path.stat().st_size == row.nwb_bytes and _sha256_file(path) == row.nwb_sha256,
                     f"NWB content drift {row.asset_id}")
    for seed in plan.SEEDS:
        item = contract.checkpoint("shared_t4", seed)
        checkpoint = Path(item.path)
        closure = Path(item.closure_path)
        core.require(checkpoint.stat().st_size == item.bytes and _sha256_file(checkpoint) == item.sha256,
                     f"shared_t4 checkpoint drift seed{seed}")
        core.require(closure.stat().st_size == item.closure_bytes and _sha256_file(closure) == item.closure_sha256,
                     f"shared_t4 closure drift seed{seed}")
    teacher = Path(contract.teacher_path)
    core.require(teacher.stat().st_size == contract.teacher_bytes
                 and _sha256_file(teacher) == contract.teacher_sha256, "teacher drift")
    for view in plan.VIEWS:
        core.require(_sha256_file(Path(contract.behavior_normalizer_paths[view]))
                     == contract.behavior_normalizer_sha256[view], f"{view} behavior normalizer drift")
        core.require(_sha256_file(Path(contract.side_normalizer_paths[view]))
                     == contract.side_normalizer_sha256[view], f"{view} side normalizer drift")


def _load_reference(
    repo_root: Path, *, asset_id: str, view: str, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    root = repo_root / plan.LABEL_BUDGET_ROOT_RELATIVE
    artifact = root / "artifacts" / asset_id / view / "m_10" / f"seed_{seed}" / "predictions_targets.npz"
    commit = root / "commits" / asset_id / view / "m_10" / f"seed_{seed}.json"
    core.require(_readonly(artifact) and _readonly(commit), f"unsafe M10 reference {asset_id}/{view}/s{seed}")
    artifact_sha = _sha256_file(artifact)
    evidence = json.loads(commit.read_text(encoding="utf-8"))
    core.require(evidence.get("artifact_sha256") == artifact_sha, "M10 reference artifact/commit drift")
    core.require(evidence.get("budget") == plan.M10 and evidence.get("seed") == seed
                 and evidence.get("view") == view and evidence.get("asset_id") == asset_id,
                 "M10 reference identity drift")
    with np.load(artifact, allow_pickle=False) as archive:
        prediction = np.ascontiguousarray(archive["predictions"], dtype=np.float32)
        target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    core.require(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 2,
                 "M10 reference shape drift")
    core.require(np.isfinite(prediction).all() and np.isfinite(target).all(), "M10 reference nonfinite")
    return prediction, target, {
        "artifact_path": str(artifact), "artifact_sha256": artifact_sha,
        "commit_path": str(commit), "commit_sha256": _sha256_file(commit),
        "checkpoint_sha256": evidence.get("checkpoint_sha256"),
        "descriptor": evidence.get("descriptor"),
        "base_protocol": evidence.get("base_protocol"),
    }


def _m10_side(
    *, nwb_path: Path, view: str, record: Any, side_mean: np.ndarray,
    side_std: np.ndarray, owners: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, object]]:
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity

    features, metadata = owners["load_unit_side_features"](
        nwb_path, feature_group="t4", pool_size=plan.M10,
        mean=side_mean, std=side_std, cache_dir=None, permutation_seed=None,
        bin_size_ms=parity.BIN_SIZE_MS, window_size=parity.HISTORY_BINS,
        trial_result_filter="R", signal_view=view,
    )
    side = np.ascontiguousarray(features, dtype=np.float32)
    core.require(side.shape == (record.neural.shape[1], 4) and np.isfinite(side).all(),
                 f"{record.name}/{view} M10 OLS side drift")
    return side, {
        "fit": "production_grouped_direction_ols", "carrier_budget": plan.M10,
        "normalized_t4_sha256": core.array_sha256(side),
        "loader_metadata_pool_size": metadata.get("pool_size") if isinstance(metadata, Mapping) else None,
    }


def _m4_sides(
    *, nwb_path: Path, trials: list[dict[str, Any]], selected: np.ndarray,
    side_stats: Mapping[str, tuple[np.ndarray, np.ndarray]], record_channels: Mapping[str, int]
) -> dict[str, tuple[np.ndarray, dict[str, object]]]:
    from pynwb import NWBHDF5IO
    from sua_exploration.mc_maze import multisession_datamodule, unit_side_features
    from tfpd_exploration.src.calibration_budget_comparators_v1 import fit_ridge_t4

    rates_units_trials, unit_count = unit_side_features._pool_trial_rate_matrix(
        nwb_path, trials[: plan.ACTIVITY_HORIZON]
    )
    core.require(rates_units_trials.shape == (unit_count, plan.ACTIVITY_HORIZON),
                 "SUA trial-rate matrix drift")
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        core.require(nwb.units is not None, "NWB units table missing")
        electrode_ids = multisession_datamodule.electrode_ids_from_units(nwb.units.to_dataframe())
    pooled, channel_ids = unit_side_features.pool_trial_rates_by_electrode(
        rates_units_trials, electrode_ids
    )
    theta = np.asarray([float(trials[index]["target_dir"]) for index in selected], dtype=np.float64)
    view_rates = {"sua": rates_units_trials, "pseudo_mua": pooled}
    output: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
    for view in plan.VIEWS:
        rates = np.ascontiguousarray(view_rates[view][:, selected].T, dtype=np.float64)
        raw, fit = fit_ridge_t4(rates, theta, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA)
        mean, std = side_stats[view]
        side = np.ascontiguousarray((raw - mean) / std, dtype=np.float32)
        core.require(side.shape == (record_channels[view], 4) and np.isfinite(side).all(),
                     f"{view} M4 ridge side drift")
        output[view] = (side, {
            **fit,
            "fit": "trialwise_fixed_ridge_0.1",
            "carrier_budget": plan.M4,
            "selected_first30_indices": selected.tolist(),
            "selected_indices_sha256": core.array_sha256(selected),
            "selected_theta_sha256": core.array_sha256(theta),
            "trial_rates_sha256": core.array_sha256(rates),
            "raw_t4_sha256_typed": core.array_sha256(raw),
            "normalized_t4_sha256": core.array_sha256(side),
            "channel_ids_sha256_or_null": (
                core.array_sha256(channel_ids) if view == "pseudo_mua" else None
            ),
        })
    return output


def _score_cached_identity(
    *, torch: Any, model: Any, record: Any, calibration: np.ndarray,
    side: np.ndarray, device: Any, batch_size: int, parity_required: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    student = model.student
    core.require(getattr(student, "decoder_mode", None) == "coupled", "cached route requires coupled decoder")
    core.require(getattr(student, "fixed_slot_router", None) is None, "cached route forbids fixed-slot router")
    core.require(not hasattr(student.id_encoder, "forward_batch_with_gate"),
                 "cached route forbids encoder gate semantics")
    calibration_tensor = torch.from_numpy(np.ascontiguousarray(calibration)).unsqueeze(0).to(device)
    side_tensor = torch.from_numpy(np.ascontiguousarray(side)).unsqueeze(0).to(device)
    starts = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
    core.require(starts.ndim == 1 and starts.size > 0 and np.all(np.diff(starts) > 0),
                 "query start order drift")
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    parity: dict[str, object] | None = None
    with torch.inference_mode():
        identity = student.compute_identity(calibration_tensor, side_features=side_tensor)
        core.require(tuple(identity.shape) == (1, record.neural.shape[1], plan.HISTORY_BINS),
                     "cached identity shape drift")
        for offset in range(0, starts.size, batch_size):
            chunk = starts[offset : offset + batch_size]
            indices = chunk[:, None] + np.arange(plan.HISTORY_BINS, dtype=np.int64)[None, :]
            neural_np = np.ascontiguousarray(record.neural[indices], dtype=np.float32)
            target_np = np.ascontiguousarray(record.behavior[chunk + plan.HISTORY_BINS - 1], dtype=np.float32)
            neural = torch.from_numpy(neural_np).to(device)
            raw_cached = student.decode_with_identity(neural, identity)
            if parity_required and parity is None:
                count = min(2, neural.shape[0])
                eager_calibration = calibration_tensor.expand(count, -1, -1, -1)
                eager_side = side_tensor.expand(count, -1, -1)
                raw_eager, eager_identity = student(
                    neural[:count], calib_trials=eager_calibration, side_features=eager_side,
                    decoder_key_features=model.decoder_key_features(eager_side), electrode_ids=None,
                )
                diff = float(torch.max(torch.abs(raw_eager - raw_cached[:count])).item())
                identity_diff = float(torch.max(torch.abs(eager_identity - identity.expand_as(eager_identity))).item())
                core.require(
                    diff <= plan.EAGER_CACHED_PARITY_ATOL
                    and identity_diff <= plan.EAGER_CACHED_PARITY_ATOL,
                             f"cached/eager parity failed: prediction={diff}, identity={identity_diff}")
                parity = {
                    "rows": count, "prediction_max_abs": diff,
                    "identity_max_abs": identity_diff,
                    "tolerance": plan.EAGER_CACHED_PARITY_ATOL,
                }
            prediction_np = (
                raw_cached[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False)
                / plan.BEHAVIOR_SCALE
            )
            predictions.append(prediction_np)
            targets.append(target_np)
    prediction = np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32)
    target = np.ascontiguousarray(np.concatenate(targets, axis=0), dtype=np.float32)
    core.require(prediction.shape == target.shape == (starts.size, 2), "cached score shape drift")
    core.require(np.isfinite(prediction).all() and np.isfinite(target).all(), "cached score nonfinite")
    return prediction, target, {
        "identity_sha256": core.array_sha256(identity.detach().cpu().numpy()),
        "activity_sha256": core.array_sha256(calibration),
        "query_starts_sha256": core.array_sha256(starts),
        "batches": int((starts.size + batch_size - 1) // batch_size),
        "cached_identity_once": True,
        "parity": parity,
    }


def _publish(root: Path, payload: Mapping[str, object]) -> str:
    core.require(not root.exists(), f"result root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        raw = (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        score = temporary / "score.json"
        score.write_bytes(raw)
        score.chmod(0o444)
        digest = hashlib.sha256(raw).hexdigest()
        sidecar = temporary / "score.json.sha256"
        sidecar.write_text(f"{digest}  score.json\n", encoding="ascii")
        sidecar.chmod(0o444)
        temporary.rename(root)
        root.chmod(0o555)
        core.require(_readonly(root / "score.json") and _readonly(root / "score.json.sha256"),
                     "published result is not immutable")
        return digest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def execute(repo_root: Path, *, gpu_index: int = 1, batch_size: int = plan.DEFAULT_BATCH_SIZE) -> dict[str, object]:
    if isinstance(gpu_index, bool) or gpu_index < 0:
        raise core.ScreenError("gpu_index must be a non-negative integer")
    if isinstance(batch_size, bool) or batch_size < 1:
        raise core.ScreenError("batch_size must be positive")
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_index), "CUDA_VISIBLE_DEVICES mismatch")
    core.require(os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER mismatch")
    core.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
                 "CUBLAS_WORKSPACE_CONFIG mismatch")
    output_root = plan.result_root(repo_root)
    core.require(not output_root.exists(), f"result root already exists: {output_root}")
    aggregate = repo_root / plan.LABEL_BUDGET_AGGREGATE_RELATIVE
    core.require(_sha256_file(aggregate) == plan.LABEL_BUDGET_AGGREGATE_SHA256,
                 "label-budget aggregate SHA drift")
    aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
    core.require(aggregate_payload.get("status") == "FULL_450_LOCAL_TORCHMETRICS_1_5_1_FINALIZED",
                 "label-budget aggregate is not finalized")

    contract, authority = _relocated_contract(repo_root)
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity
    from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime

    nwb_root = (repo_root / plan.NWB_ROOT_RELATIVE).resolve()
    _verify_selected_inputs(contract, nwb_root=nwb_root)
    owners = runtime._runtime_owners(repo_root)
    torch = owners["torch"]
    core.require(torch.cuda.is_available(), "CUDA unavailable")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    models = {
        seed: runtime._load_model(contract.checkpoint("shared_t4", seed), contract, str(device), owners)
        for seed in plan.SEEDS
    }
    for model in models.values():
        model.eval()
        for parameter in model.parameters():
            core.require(parameter.requires_grad is False, "frozen evaluator exposes trainable parameters")
    behavior_stats = {
        view: runtime._load_mean_std(contract.behavior_normalizer_paths[view], label=f"{view} behavior")
        for view in plan.VIEWS
    }
    side_stats = {
        view: runtime._load_mean_std(contract.side_normalizer_paths[view], label=f"{view} side")
        for view in plan.VIEWS
    }

    started = time.monotonic()
    rows: list[dict[str, object]] = []
    parity_remaining = {(view, seed, cell) for view in plan.VIEWS for seed in plan.SEEDS for cell in plan.NEW_CELLS}
    values: dict[str, dict[str, dict[str, dict[int, float]]]] = {
        view: {cell: {} for cell in plan.CELL_ORDER} for view in plan.VIEWS
    }
    for cohort in contract.cohort:
        nwb_path = nwb_root / cohort.frozen_path
        trials = owners["list_datamodule_rewarded_trials"](
            nwb_path, bin_size_ms=parity.BIN_SIZE_MS, window_size=parity.HISTORY_BINS,
            trial_result_filter="R",
        )
        core.require(len(trials) >= plan.ACTIVITY_HORIZON, f"{cohort.session_id} lacks first30 trials")
        theta_first30 = np.asarray(
            [float(row["target_dir"]) if row.get("target_dir") is not None else np.nan
             for row in trials[: plan.ACTIVITY_HORIZON]],
            dtype=np.float64,
        )
        selected_m4 = core.select_m4_support(theta_first30)
        records: dict[str, Any] = {}
        rebuilt_by_view: dict[str, np.ndarray] = {}
        for view in plan.VIEWS:
            behavior_mean, behavior_std = behavior_stats[view]
            record, rebuilt, _evidence = runtime._build_base(
                repo_root=repo_root, nwb_path=nwb_path, view=view,
                behavior_mean=behavior_mean, behavior_std=behavior_std, owners=owners,
            )
            core.require(record.name == cohort.session_id, "runtime session identity drift")
            core.require(int(record.valid_starts.size) == cohort.query_window_count,
                         "runtime query count drift")
            rebuilt = np.ascontiguousarray(rebuilt, dtype=np.float32)
            core.require(rebuilt.shape[0] == plan.ACTIVITY_HORIZON, "rebuilt activity count drift")
            records[view] = record
            rebuilt_by_view[view] = rebuilt
        m4_by_view = _m4_sides(
            nwb_path=nwb_path, trials=trials, selected=selected_m4,
            side_stats=side_stats,
            record_channels={view: int(records[view].neural.shape[1]) for view in plan.VIEWS},
        )
        for view in plan.VIEWS:
            record = records[view]
            rebuilt = rebuilt_by_view[view]
            m10_side, m10_evidence = _m10_side(
                nwb_path=nwb_path, view=view, record=record,
                side_mean=side_stats[view][0], side_std=side_stats[view][1], owners=owners,
            )
            m4_side, m4_evidence = m4_by_view[view]
            cell_inputs = {
                "ols_m10_activity10": (
                    core.select_activity(rebuilt, selected_support=np.arange(plan.M10), activity_budget=plan.M10),
                    m10_side, m10_evidence,
                ),
                "ridge_m4_activity4": (
                    core.select_activity(rebuilt, selected_support=selected_m4, activity_budget=plan.M4),
                    m4_side, m4_evidence,
                ),
                "ridge_m4_activity30": (
                    core.select_activity(rebuilt, selected_support=selected_m4, activity_budget=plan.ACTIVITY_HORIZON),
                    m4_side, m4_evidence,
                ),
            }
            for seed in plan.SEEDS:
                reference_prediction, reference_target, reference_evidence = _load_reference(
                    repo_root, asset_id=cohort.asset_id, view=view, seed=seed
                )
                core.require(reference_target.shape[0] == cohort.query_window_count,
                             "reference query count drift")
                reference_r2 = core.variance_weighted_r2(reference_target, reference_prediction)
                values[view][plan.REFERENCE_CELL].setdefault(cohort.session_id, {})[seed] = reference_r2
                rows.append({
                    "asset_id": cohort.asset_id, "session_id": cohort.session_id,
                    "view": view, "seed": seed, "cell": plan.REFERENCE_CELL,
                    "carrier_budget": plan.M10, "activity_budget": plan.ACTIVITY_HORIZON,
                    "query_window_count": int(reference_target.shape[0]),
                    "target_sha256": core.array_sha256(reference_target),
                    "prediction_sha256": core.array_sha256(reference_prediction),
                    "r2": reference_r2, "reused_immutable_reference": reference_evidence,
                })
                for cell in plan.NEW_CELLS:
                    activity, side, side_evidence = cell_inputs[cell]
                    parity_key = (view, seed, cell)
                    prediction, target, forward = _score_cached_identity(
                        torch=torch, model=models[seed], record=record,
                        calibration=activity, side=side, device=device, batch_size=batch_size,
                        parity_required=parity_key in parity_remaining,
                    )
                    parity_remaining.discard(parity_key)
                    core.require(np.array_equal(target, reference_target),
                                 f"new/reference target drift {cohort.session_id}/{view}/s{seed}/{cell}")
                    r2 = core.variance_weighted_r2(target, prediction)
                    values[view][cell].setdefault(cohort.session_id, {})[seed] = r2
                    rows.append({
                        "asset_id": cohort.asset_id, "session_id": cohort.session_id,
                        "view": view, "seed": seed, "cell": cell,
                        "carrier_budget": plan.M10 if cell.startswith("ols_m10") else plan.M4,
                        "activity_budget": (
                            plan.M10 if cell == "ols_m10_activity10"
                            else plan.M4 if cell == "ridge_m4_activity4"
                            else plan.ACTIVITY_HORIZON
                        ),
                        "query_window_count": int(target.shape[0]),
                        "selected_m4_first30_indices": selected_m4.tolist() if cell.startswith("ridge_m4") else None,
                        "target_sha256": core.array_sha256(target),
                        "prediction_sha256": core.array_sha256(prediction),
                        "r2": r2, "side_evidence": side_evidence, "forward_evidence": forward,
                    })

    core.require(not parity_remaining, "not every view/seed/cell received cached-eager parity")
    core.require(len(rows) == plan.EXPECTED_ALIGNED_ROWS, "aligned row count drift")
    summaries = {
        view: {cell: core.summarize_rows(values[view][cell]) for cell in plan.CELL_ORDER}
        for view in plan.VIEWS
    }
    contrasts = {
        view: {
            "activity30_minus_activity10_m10": core.paired_contrast(
                values[view][plan.REFERENCE_CELL], values[view]["ols_m10_activity10"]
            ),
            "activity30_minus_activity4_m4": core.paired_contrast(
                values[view]["ridge_m4_activity30"], values[view]["ridge_m4_activity4"]
            ),
        }
        for view in plan.VIEWS
    }
    granularity: dict[str, object] = {}
    for name in ("activity30_minus_activity10_m10", "activity30_minus_activity4_m4"):
        sua_delta = contrasts["sua"][name]["per_session_seed_delta"]
        pseudo_delta = contrasts["pseudo_mua"][name]["per_session_seed_delta"]
        granularity[f"sua_minus_pseudo_mua__{name}"] = core.paired_contrast(
            {session: {seed: float(sua_delta[session][str(seed)]) for seed in plan.SEEDS} for session in sua_delta},
            {session: {seed: float(pseudo_delta[session][str(seed)]) for seed in plan.SEEDS} for session in pseudo_delta},
        )
    torch.cuda.synchronize(device)
    payload: dict[str, object] = {
        "schema": plan.SCHEMA,
        "status": "TERMINAL",
        "scientific_role": "paired_frozen_weight_activity_axis_screen__not_online_carrier_cdm",
        "authority": {
            **authority,
            "label_budget_aggregate_path": str(aggregate),
            "label_budget_aggregate_sha256": plan.LABEL_BUDGET_AGGREGATE_SHA256,
        },
        "protocol": {
            "views": list(plan.VIEWS), "seeds": list(plan.SEEDS),
            "query_rule": "strictly_after_rewarded_trial_50__identical_within_each_paired_cell",
            "m10_support": "chronological_first10__production_grouped_direction_ols",
            "m4_support": "doptimal_four_from_first30_finite_cue_directions__fixed_ridge_0.1",
            "activity_arms": ["selected_support_only", "chronological_first30"],
            "target_gradients": 0, "target_backward": 0, "parameter_updates": 0,
        },
        "device": {
            "cuda_visible_devices": str(gpu_index), "logical_device": "cuda:0",
            "name": torch.cuda.get_device_name(0), "batch_size": batch_size,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "reference_row_count": plan.EXPECTED_REFERENCE_ROWS,
        "new_forward_row_count": plan.EXPECTED_NEW_ROWS,
        "aligned_row_count": len(rows),
        "cell_order": list(plan.CELL_ORDER),
        "rows": rows,
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "paired_granularity_contrasts": granularity,
    }
    payload["score_sha256"] = _publish(output_root, payload)
    return payload
