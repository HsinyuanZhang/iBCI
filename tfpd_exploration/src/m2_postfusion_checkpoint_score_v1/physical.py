"""Post-attempt CPU scoring primitives; no module-level Torch import."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import copy
import io
import hashlib
import numpy as np

from . import plan


class PhysicalError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise PhysicalError(message)


def _array_sha256(value: Any) -> str:
    """Shape/dtype-bound digest for ephemeral input-authority arrays."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(plan.canonical_json({"shape": list(array.shape), "dtype": str(array.dtype)}))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _governed_array_sha256(value: Any) -> str:
    """Digest arrays on the immutable POOLED/metric receipt contract.

    ``g_replay.rollout_g00m`` delegates its governed prediction, target, and
    start-array digests to the reviewed Precision-CDM helper.  The scorer also
    uses a route-local shape/dtype digest for internal activity authorities,
    but it must never compare that different framing to a sealed POOLED row.
    """
    from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import core as pseudo_core
    return str(pseudo_core.array_sha256(value))


def _window_indices_sha256(dataset: Any) -> str:
    windows = getattr(dataset, "window_indices", None)
    _require(isinstance(windows, (list, tuple)), "source dataset lacks ordered window authority")
    normalized = [[str(session), int(start)] for session, start in windows]
    return plan.sha256_bytes(plan.canonical_json(normalized))


def _source_authority_fingerprint(datamodule: Any) -> dict[str, object]:
    """Only source facts; deliberately excludes the append-only held-out arm."""
    dataset = getattr(datamodule, "train_dataset", None)
    _require(dataset is not None, "prepared PIT datamodule lacks source train dataset")
    normalization = getattr(datamodule, "native_t4_normalization", None)
    _require(isinstance(normalization, Mapping), "prepared PIT datamodule lacks native T4 authority")
    sessions = tuple(str(item) for item in getattr(datamodule, "train_session_names", ()))
    calibration = getattr(datamodule, "train_calib_heldin_sessions", None)
    _require(isinstance(calibration, Mapping) and set(calibration) == set(sessions),
             "prepared PIT source calibration roster drift")
    return {
        "datamodule_object_id": int(id(datamodule)),
        "train_dataset_object_id": int(id(dataset)),
        "train_sessions": list(sessions),
        "window_indices_sha256": _window_indices_sha256(dataset),
        "side_feature_mean_sha256": _array_sha256(dataset.side_feature_mean),
        "side_feature_std_sha256": _array_sha256(dataset.side_feature_std),
        "calibration_activity_sha256": {
            name: _array_sha256(dataset.calib_trialized_neural_features[name])
            for name in sorted(calibration)
        },
        "normalization": {
            "feature_group": str(normalization.get("feature_group")),
            "train_sessions": [str(item) for item in normalization.get("train_sessions", ())],
            "mean_sha256": _array_sha256(normalization.get("mean")),
            "std_sha256": _array_sha256(normalization.get("std")),
        },
    }


def _append_heldout_to_prepared_datamodule(*, datamodule: Any) -> dict[str, object]:
    """Append the test-only dataset without replaying ``DataModule.setup``.

    ``setup('test')`` reparses the entire source side and replaces
    ``train_dataset``.  The frozen DataModule already exposes the private,
    narrow held-out constructor used by that method.  We call that constructor
    once with the *existing* source statistics, then prove that every source
    authority byte remains unchanged.
    """
    _require(getattr(datamodule, "val_heldout_dataset", None) is None,
             "held-out dataset was already materialized before scorer authority")
    before = _source_authority_fingerprint(datamodule)
    calibration = datamodule.train_calib_heldin_sessions
    first_session = next(iter(calibration), None)
    _require(first_session is not None, "source calibration roster is empty")
    first = calibration[first_session]
    _require(isinstance(first, Mapping) and "covariates_mean" in first and "covariates_std" in first,
             "source covariate standardization authority is absent")
    normalization = datamodule.native_t4_normalization
    _require(str(normalization.get("feature_group")) in {"t4", "ts4"},
             "Post-Fusion checkpoint scorer requires native T4 source authority")
    train_dataset = datamodule.train_dataset
    _require(np.array_equal(np.asarray(normalization["mean"], dtype=np.float32),
                            np.asarray(train_dataset.side_feature_mean, dtype=np.float32))
             and np.array_equal(np.asarray(normalization["std"], dtype=np.float32),
                                np.asarray(train_dataset.side_feature_std, dtype=np.float32)),
             "source T4 normalizer disagrees with source dataset")
    from falcon_challenge.config import FalconConfig, FalconTask
    task_config = FalconConfig(task=FalconTask.__dict__["m2"])
    builder = getattr(datamodule, "_build_heldout_dataset", None)
    _require(callable(builder), "sealed DataModule lacks narrow held-out dataset constructor")
    builder(
        first["covariates_mean"], first["covariates_std"], task_config,
        side_feature_group=str(normalization["feature_group"]),
        side_feature_mean=np.asarray(normalization["mean"], dtype=np.float32),
        side_feature_std=np.asarray(normalization["std"], dtype=np.float32),
    )
    after = _source_authority_fingerprint(datamodule)
    _require(before == after, "appending held-out data rebuilt or drifted source authority")
    heldout = getattr(datamodule, "val_heldout_dataset", None)
    _require(heldout is not None, "narrow held-out constructor did not materialize a test dataset")
    return {"source_before": before, "source_after": after,
            "source_preserved_exact": True, "heldout_dataset_object_id": int(id(heldout)),
            "construction": "same_datamodule_private_build_heldout_dataset_once"}


def _record_from_session_material(*, surface: str, session: str, dataset: Any,
                                  raw_sessions: Mapping[str, Any], replay: Any) -> dict[str, Any]:
    _require(session in raw_sessions, f"{surface}/{session}: raw calibration authority absent")
    views = replay._g_session_views(raw_sessions, session=session)
    support = replay.g_support_material(session=session, views=views)
    selected = np.ascontiguousarray(np.asarray(support["selected"], dtype=np.int64))
    _require(selected.shape == (plan.BUDGET,) and np.all(selected >= 0) and np.all(selected < 30),
             f"{surface}/{session}: B30 D-opt-k4 support drift")
    query_rows = tuple(replay.g_query_rows(ds=dataset, session=session, views=views))
    metric_starts = np.ascontiguousarray(np.concatenate([
        np.asarray(row["metric_starts"], dtype=np.int64) for row in query_rows
    ]), dtype=np.int64)
    _require(metric_starts.size > 0 and np.all(np.diff(metric_starts) > 0),
             f"{surface}/{session}: governed query order drift")
    window_bins = int(getattr(dataset, "window_size", 50))
    _require(window_bins == 50, f"{surface}/{session}: sealed decoder window drift")
    targets = np.ascontiguousarray(
        np.asarray(dataset.covariate_data[session], dtype=np.float32)[metric_starts + window_bins - 1],
        dtype=np.float32,
    )
    activities = np.asarray(support["activities"], dtype=np.float32)
    raw_t4 = np.asarray(support["raw_m30_hz"], dtype=np.float32)
    # Carrier arrays are Hz; B3S side features consume expected counts/bin.
    normalized_t4 = np.ascontiguousarray(
        (raw_t4 * np.float32(0.020) - np.asarray(dataset.side_feature_mean, dtype=np.float32))
        / np.asarray(dataset.side_feature_std, dtype=np.float32), dtype=np.float32)
    _require(np.isfinite(normalized_t4).all(), "normalized T4 carrier is nonfinite")
    query_activities = np.ascontiguousarray(np.stack([
        np.asarray(row["activity"], dtype=np.float32) for row in query_rows
    ]), dtype=np.float32)
    trial_ids = [str(row["trial_id"]) for row in query_rows]
    eval_mask = np.asarray(dataset.eval_mask[session], dtype=np.bool_)
    endpoint_valid = np.ascontiguousarray(eval_mask[metric_starts + window_bins - 1], dtype=np.bool_)
    _require(bool(endpoint_valid.all()), f"{surface}/{session}: governed metric has invalid endpoint")
    return {
        "key": f"{surface}|{session}", "surface": surface, "session": session,
        "budget": int(plan.BUDGET), "pool_capacity": int(plan.POOL_CAPACITY),
        "support_indices": [int(item) for item in selected],
        "support_indices_sha256": _array_sha256(selected),
        "support_activity_sha256": _array_sha256(activities[selected]),
        "raw_t4_sha256": _array_sha256(raw_t4),
        "normalized_t4_sha256": _array_sha256(normalized_t4),
        # These three fields are cross-route metric links and therefore use
        # the exact reviewed core framing, not the local receipt framing.
        "query_starts_sha256": _governed_array_sha256(metric_starts),
        "target_sha256": _governed_array_sha256(targets), "window_count": int(metric_starts.size),
        "completed_query_trials": int(len(query_rows)),
        "query_trial_ids_sha256": plan.sha256_bytes(plan.canonical_json(trial_ids)),
        "query_activity_sha256": _array_sha256(query_activities),
        "dataset_eval_mask_sha256": _array_sha256(eval_mask),
        "query_endpoint_validity_sha256": _array_sha256(endpoint_valid),
        "completed_query_trace_sha256": _trial_sequence_digest(query_rows),
        # Runtime-only objects are kept out of receipt bodies.  One record is
        # shared by exactly six PF rows in the later scorer coordinator.
        "_runtime": {"dataset": dataset, "views": views, "support": support,
                     "query_rows": query_rows, "metric_starts": metric_starts, "targets": targets},
    }


def _bind_pooled_comparator(record: dict[str, Any], comparator: Mapping[str, Any]) -> None:
    """Bind an already sealed POOLED row to this exact input surface."""
    required = {"r2", "window_count", "query_starts_sha256", "target_sha256", "prediction_sha256"}
    _require(required <= set(comparator), "sealed POOLED comparator evidence is incomplete")
    _require(int(comparator["window_count"]) == int(record["window_count"]),
             f"{record['key']}: POOLED window count is not same-input")
    _require(str(comparator["query_starts_sha256"]) == str(record["query_starts_sha256"])
             and str(comparator["target_sha256"]) == str(record["target_sha256"]),
             f"{record['key']}: POOLED comparator surface drift")
    _require(np.isfinite(float(comparator["r2"])), f"{record['key']}: POOLED R2 is nonfinite")
    record["pooled_comparator"] = dict(comparator)


def _read_regular_nofollow(path: Any) -> bytes:
    """Read an immutable checkpoint body without following a final symlink."""
    import os
    import stat
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode), "sealed POOLED checkpoint is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _strict_sealed_pooled_clone(*, repo_root: Any, base_module: Any) -> tuple[Any, dict[str, Any]]:
    """Strict-load the selected 25d T4 checkpoint into a clone of PIT's model.

    PIT's prepared base is intentionally a teacher-initialized construction,
    not the selected T4 checkpoint.  The POOLED comparator must therefore use
    a separate strict clone, while still sharing the already prepared
    DataModule and never calling the broad exporter reconstruction helper.
    """
    import torch
    from pathlib import Path
    from tfpd_exploration.src.pit_m2_v1 import plan as pit_plan
    source = Path(repo_root).absolute() / pit_plan.SEALED_CHECKPOINT_RELATIVE
    body = _read_regular_nofollow(source)
    digest = plan.sha256_bytes(body)
    _require(digest == pit_plan.SEALED_CHECKPOINT_SHA256,
             "selected T4 checkpoint drifted before POOLED clone")
    # Unlike the three screen-owned state-only bodies, this historical
    # Lightning checkpoint contains an OmegaConf envelope.  PyTorch's
    # weights-only reader cannot decode that schema without an expanding set
    # of third-party allowlists.  The body was first read no-follow and bound
    # byte-for-byte to the immutable selected-T4 SHA, so this compatibility
    # load is safe only in that exact order; state extraction below remains
    # strict and no caller-provided path can reach this branch.
    payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping),
             "selected T4 checkpoint is not a full strict-load payload")
    model = copy.deepcopy(base_module)
    before = _student_state_sha(model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "selected POOLED clone remained trainable")
    after = _student_state_sha(model)
    return model, {"selected_t4_checkpoint_sha256": digest,
                   "student_state_before_load_sha256": before,
                   "student_state_after_load_sha256": after,
                   "strict_load": True, "eval": True,
                   "parameter_updates": 0, "optimizer_constructed": False}


def _score_sealed_pooled_comparator(*, record: Mapping[str, Any], model: Any,
                                    torch: Any, replay: Any, batch_size: int,
                                    model_evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Generate one native G00m POOLED comparator over a shared input record."""
    runtime = record["_runtime"]
    dataset = runtime["dataset"]
    before = _student_state_sha(model)
    result = replay.rollout_g00m(
        torch=torch, model=model, ds=dataset, session=str(record["session"]),
        views=runtime["views"], support=runtime["support"], query_rows=runtime["query_rows"],
        side_mean=np.asarray(dataset.side_feature_mean, dtype=np.float32),
        side_std=np.asarray(dataset.side_feature_std, dtype=np.float32),
        device=torch.device("cpu"), batch_size=int(batch_size),
    )
    after = _student_state_sha(model)
    _require(before == after == str(model_evidence["student_state_after_load_sha256"]),
             f"{record['key']}: POOLED scoring mutated the strict-loaded model")
    return {
        "policy": "POOLED", "source": "selected_t4_g_replay_rollout_g00m",
        "r2": float(result["r2"]), "window_count": int(result["window_count"]),
        "query_starts_sha256": str(result["query_starts_sha256"]),
        "target_sha256": str(result["target_sha256"]),
        "prediction_sha256": str(result["prediction_sha256"]),
        "model_state_before_sha256": before, "model_state_after_sha256": after,
        "selected_t4_checkpoint_sha256": str(model_evidence["selected_t4_checkpoint_sha256"]),
        "carrier_updates": 0, "parameter_updates": 0, "target_updates": 0,
    }


def materialize_13_inputs_and_pooled_comparators(*, prepared: Mapping[str, Any],
                                                  pooled_comparator: Mapping[str, Mapping[str, Any]]
                                                  | Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
                                                  replay_module: Any | None = None,
                                                  repo_root: Any | None = None) -> dict[str, Any]:
    """Append held-out data once and materialize the exact 13 shared inputs.

    This deliberately does not decode a Post-Fusion arm.  A caller may supply
    an already descriptor-validated POOLED row, or (only after attempt) this
    function strict-loads the selected 25d T4 checkpoint into a clone of the
    prepared PIT model and runs native ``rollout_g00m`` once per input record.
    """
    import os
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "input materialization is CPU-only")
    datamodule = prepared.get("datamodule")
    _require(datamodule is not None, "prepared shared PIT datamodule is absent")
    append_evidence = _append_heldout_to_prepared_datamodule(datamodule=datamodule)
    if replay_module is None:
        from tfpd_exploration.src.cdm_p1_m2_local_v1 import replay as g_replay
    else:
        g_replay = replay_module
    pooled_model = None
    pooled_model_evidence: Mapping[str, Any] | None = None
    if pooled_comparator is None:
        _require(repo_root is not None and prepared.get("base_module") is not None,
                 "native POOLED construction requires repo root and prepared base model")
        import torch
        _require(not torch.cuda.is_initialized(), "POOLED comparator must not initialize CUDA")
        pooled_model, pooled_model_evidence = _strict_sealed_pooled_clone(
            repo_root=repo_root, base_module=prepared["base_module"])
        pooled_batch_size = int(plan.CPU_DECODE_BATCH_SIZE)
        _require(pooled_batch_size == 1024, "CPU POOLED decode-batch law drift")
    surfaces = {
        "external_post30_local": (datamodule.val_heldout_dataset, datamodule.val_calib_heldout_sessions),
        "within_post30": (datamodule.train_dataset, datamodule.train_calib_heldin_sessions),
    }
    records: dict[str, dict[str, Any]] = {}
    for surface in plan.SURFACES:
        dataset, raw_sessions = surfaces[surface]
        _require(dataset is not None and isinstance(raw_sessions, Mapping), f"{surface}: materialization authority absent")
        sessions = tuple(sorted(dataset.calib_trialized_neural_features))
        _require(len(sessions) == plan.ROSTER_SIZES[surface], f"{surface}: roster size drift")
        for session in sessions:
            record = _record_from_session_material(
                surface=surface, session=session, dataset=dataset, raw_sessions=raw_sessions, replay=g_replay)
            if pooled_comparator is None:
                comparator = _score_sealed_pooled_comparator(
                    record=record, model=pooled_model, torch=torch, replay=g_replay,
                    batch_size=pooled_batch_size, model_evidence=pooled_model_evidence)
            else:
                comparator = (pooled_comparator(record) if callable(pooled_comparator)
                              else pooled_comparator.get(record["key"]))
            _require(isinstance(comparator, Mapping), f"{record['key']}: sealed POOLED comparator absent")
            _bind_pooled_comparator(record, comparator)
            records[record["key"]] = record
    _require(len(records) == 13, "shared materialization must produce 6 external plus 7 within records")
    return {"records": records, "append_heldout_evidence": append_evidence,
            "materializations": int(len(records)), "postfusion_arm_rollouts": 0,
            "pooled_comparators_bound": int(len(records)),
            "pooled_model_evidence": (dict(pooled_model_evidence)
                                      if pooled_model_evidence is not None else None)}


def _student_state_sha(module: Any) -> str:
    """Screen-compatible digest of the strictly loaded ``student`` state.

    The successful producer recorded this exact canonical tensor framing.  A
    raw-bytes-only digest would be a different claim and could not bind the
    held checkpoint to the screen receipt.
    """
    digest = hashlib.sha256()
    for name, value in sorted(module.student.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(plan.canonical_json({"name": name, "shape": list(array.shape),
                                           "dtype": str(array.dtype)}))
        digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_three_frozen_arms_from_held_screen(*, repo_root: Any, checkpoint_bytes: dict[str, bytes]):
    """Single source-only PIT construction then strict-load three PF clones."""
    import os
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only scorer requires empty CVD")
    _require(set(checkpoint_bytes) == set(plan.ARMS), "held checkpoint arm set drift")
    import torch
    _require(not torch.cuda.is_initialized(), "CUDA initialized before PIT source prepare")
    from tfpd_exploration.src.pit_m2_v1 import trainer
    from tfpd_exploration.src.m2_postfusion_variant_screen_v1 import variants
    runner = trainer.PitM2ArmedRunner(repo_root, "t0m", device="cpu")
    runner.prepare(attach_operator=False)
    base = runner._litmodule
    _require(base is not None and runner._datamodule is not None, "PIT source stack missing")
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in base.parameters()),
             "shared native POOLED base remained trainable")
    modules = {}; adapters = {}; evidence = {}
    for arm in plan.ARMS:
        module = copy.deepcopy(base)
        adapter = variants.install_variant(module.student, arm)
        before = _student_state_sha(module)
        payload = torch.load(io.BytesIO(checkpoint_bytes[arm]), map_location="cpu", weights_only=True)
        _require(isinstance(payload, dict) and isinstance(payload.get("model"), dict), f"{arm}: checkpoint payload drift")
        module.load_state_dict(payload["model"], strict=True)
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
        after = _student_state_sha(module)
        _require(all(not p.requires_grad for p in module.parameters()), f"{arm}: parameter unfrozen")
        modules[arm] = module; adapters[arm] = adapter
        expected = plan.LIVE_PRODUCER_LITERALS["checkpoints"][arm]["student_state_sha256"]
        _require(after == expected, f"{arm}: strict-loaded student state disagrees with held screen")
        evidence[arm] = {"student_state_before_load_sha256": before,
                         "student_state_after_load_sha256": after,
                         "held_screen_student_state_sha256": expected,
                         "strict_load": True, "eval": bool(not module.training),
                         "all_parameters_frozen": True, "parameter_updates": 0,
                         "optimizer_constructed": False}
    _require(not torch.cuda.is_initialized(), "PIT source prepare initialized CUDA")
    return {"datamodule": runner._datamodule, "base_module": base,
            "modules": modules, "adapters": adapters,
            "evidence": evidence, "pit_prepare_calls": 1}


def trained_trial_identity(*, adapter: Any, torch: Any, activity: np.ndarray, side_tensor: Any,
                           expected_channels: int = 96) -> Any:
    """Use the public trained PF adapter, never native streaming delegation."""
    item = np.ascontiguousarray(activity, dtype=np.float32)
    _require(item.shape == (100, int(expected_channels)), "activity row shape drift")
    with torch.inference_mode():
        identity = adapter.forward_batch(torch.from_numpy(item).unsqueeze(0).unsqueeze(0), side_features=side_tensor)
    _require(tuple(identity.shape) == (1, int(expected_channels), 50)
             and bool(torch.isfinite(identity).all().item()), "trained identity shape/finite drift")
    return identity


def _trial_sequence_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return plan.sha256_bytes(plan.canonical_json([
        {"trial_id": str(row["trial_id"]), "position": int(row["position"]),
         "activity_sha256": _array_sha256(row["activity"]),
         "metric_starts_sha256": _array_sha256(row["metric_starts"]),
         "causal_starts_sha256": _array_sha256(row["causal_starts"])}
        for row in rows
    ]))


def _validate_runtime_record(record: Mapping[str, Any]) -> None:
    runtime = record.get("_runtime")
    _require(isinstance(runtime, Mapping), "input record has no ephemeral runtime authority")
    rows = tuple(runtime.get("query_rows", ()))
    _require(rows and all(isinstance(row, Mapping) for row in rows), "query trial authority is absent")
    _require(str(record.get("query_trial_ids_sha256")) == plan.sha256_bytes(plan.canonical_json([
        str(row["trial_id"]) for row in rows
    ])), "query trial-order authority drift")
    _require(str(record.get("query_activity_sha256")) == _array_sha256(np.stack([
        np.asarray(row["activity"], dtype=np.float32) for row in rows
    ])), "query activity authority drift")
    _require(str(record.get("completed_query_trace_sha256")) == _trial_sequence_digest(rows),
             "query trial trace authority drift")
    dataset = runtime.get("dataset")
    session = str(record.get("session"))
    _require(dataset is not None and session in dataset.eval_mask, "dataset eval-mask authority is absent")
    _require(str(record.get("dataset_eval_mask_sha256")) == _array_sha256(dataset.eval_mask[session]),
             "dataset eval-mask authority drift")
    support = runtime.get("support")
    _require(isinstance(support, Mapping), "support authority is absent")
    selected = np.asarray(support["selected"], dtype=np.int64)
    _require(str(record.get("support_indices_sha256")) == _array_sha256(selected)
             and str(record.get("support_activity_sha256")) == _array_sha256(
                 np.asarray(support["activities"], dtype=np.float32)[selected])
             and str(record.get("raw_t4_sha256")) == _array_sha256(support["raw_m30_hz"])
             and str(record.get("normalized_t4_sha256")) == _array_sha256(
                 (np.asarray(support["raw_m30_hz"], dtype=np.float32) * np.float32(0.020)
                  - np.asarray(dataset.side_feature_mean, dtype=np.float32))
                 / np.asarray(dataset.side_feature_std, dtype=np.float32)),
             "support/T4 authority drift")
    starts = np.ascontiguousarray(np.concatenate([
        np.asarray(row["metric_starts"], dtype=np.int64) for row in rows
    ]), dtype=np.int64)
    targets = np.ascontiguousarray(np.asarray(dataset.covariate_data[session], dtype=np.float32)[starts + 49], dtype=np.float32)
    valid = np.ascontiguousarray(np.asarray(dataset.eval_mask[session], dtype=np.bool_)[starts + 49], dtype=np.bool_)
    _require(str(record.get("query_starts_sha256")) == _governed_array_sha256(starts)
             and str(record.get("target_sha256")) == _governed_array_sha256(targets)
             and str(record.get("query_endpoint_validity_sha256")) == _array_sha256(valid)
             and bool(valid.all()), "query/target/validity authority drift")


def _side_tensor_for_record(*, torch: Any, record: Mapping[str, Any]) -> Any:
    runtime = record["_runtime"]
    support = runtime["support"]; dataset = runtime["dataset"]
    # Raw carrier is Hz; the sealed B3S side interface is expected-counts/bin.
    raw = np.ascontiguousarray(np.asarray(support["raw_m30_hz"], dtype=np.float32) * np.float32(0.020))
    side = np.ascontiguousarray((raw - np.asarray(dataset.side_feature_mean, dtype=np.float32))
                                / np.asarray(dataset.side_feature_std, dtype=np.float32), dtype=np.float32)
    _require(np.isfinite(side).all() and side.ndim == 2, "record T4 side tensor drift")
    return torch.from_numpy(side).unsqueeze(0)


def _decode_identity_cpu(*, torch: Any, model: Any, neural_windows: np.ndarray,
                         identity: Any, batch_size: int) -> np.ndarray:
    """The frozen last-bin decode law for one already materialized window block."""
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, int(neural_windows.shape[0]), int(batch_size)):
            neural = torch.from_numpy(np.ascontiguousarray(neural_windows[offset:offset + batch_size], dtype=np.float32))
            output = model.student.decode_with_identity(neural, identity)
            _require(bool(torch.isfinite(output).all().item()), "PF decode produced nonfinite output")
            values.append(output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / np.float32(5.0))
    return np.ascontiguousarray(np.concatenate(values, axis=0), dtype=np.float32)


def _identity_sha256(identity: Any) -> str:
    return _array_sha256(identity.detach().cpu().contiguous().numpy())


def _assert_support_retained(*, pool: Any, support_identities: Sequence[Any]) -> list[str]:
    """Prove FIFO never replaces or mutates the support prefix.

    A count-only check cannot distinguish eviction of support followed by a
    completed-trial append.  The pool contract exposes ordered members, so
    every commit verifies reference identity *and* exact float content for
    every support item in its original arrival order.
    """
    members = pool.member_identities()
    _require(len(members) >= len(support_identities), "support prefix became shorter")
    digests: list[str] = []
    for index, expected in enumerate(support_identities):
        observed = members[index]
        _require(observed is expected, f"support identity object {index} was evicted/replaced")
        expected_digest = _identity_sha256(expected)
        _require(_identity_sha256(observed) == expected_digest,
                 f"support identity content {index} drifted")
        digests.append(expected_digest)
    return digests


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64); prediction = np.asarray(prediction, dtype=np.float64)
    _require(target.shape == prediction.shape and target.ndim == 2 and target.shape[0] > 1,
             "governed R2 topology drift")
    centered = target - target.mean(axis=0, keepdims=True)
    denom = float(np.sum(centered * centered)); _require(denom > 0.0, "governed target has zero variance")
    value = float(1.0 - np.sum((target - prediction) ** 2) / denom)
    _require(np.isfinite(value), "governed R2 nonfinite")
    return value


def score_78_rows_from_materialized(*, prepared: Mapping[str, Any],
                                    materialized: Mapping[str, Any],
                                    decode: Callable[[Any, np.ndarray, Any], np.ndarray] | None = None,
                                    expected_channels: int = 96,
                                    record_keys: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Pure CPU coordinator for all trained PF arms over the 13 shared inputs.

    Each query trial obtains one trained public-adapter identity per arm.  That
    one identity is committed to both memory laws only *after* both laws have
    decoded the trial's governed windows, preserving paired causal exposure.
    """
    import os
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "PF score coordinator is CPU-only")
    import torch
    _require(not torch.cuda.is_initialized(), "PF score coordinator cannot initialize CUDA")
    from tfpd_exploration.src.m2_postfusion_probe_v1 import memory
    modules = prepared.get("modules"); adapters = prepared.get("adapters")
    records = materialized.get("records")
    _require(isinstance(modules, Mapping) and set(modules) == set(plan.ARMS), "three frozen PF modules absent")
    _require(isinstance(adapters, Mapping) and set(adapters) == set(plan.ARMS), "three PF adapters absent")
    _require(isinstance(records, Mapping) and len(records) == 13, "13 materialized inputs absent")
    selected_keys = None if record_keys is None else tuple(str(item) for item in record_keys)
    if selected_keys is not None:
        _require(bool(selected_keys) and len(set(selected_keys)) == len(selected_keys)
                 and set(selected_keys) <= set(records), "per-record scorer selection drift")
    rows_out: list[dict[str, Any]] = []
    for surface in plan.SURFACES:
        selected_records = sorted((record for record in records.values() if record.get("surface") == surface),
                                  key=lambda item: str(item["session"]))
        _require(len(selected_records) == plan.ROSTER_SIZES[surface], "materialized surface roster drift")
        for record in selected_records:
            if selected_keys is not None and str(record["key"]) not in selected_keys:
                continue
            _validate_runtime_record(record)
            runtime = record["_runtime"]; support = runtime["support"]; query_rows = tuple(runtime["query_rows"])
            neural = np.asarray(runtime["dataset"].neural_data[record["session"]], dtype=np.float32)
            target_all = np.asarray(runtime["dataset"].covariate_data[record["session"]], dtype=np.float32)
            selected = np.asarray(support["selected"], dtype=np.int64)
            for arm in plan.ARMS:
                model = modules[arm]; adapter = adapters[arm]
                _require(adapter is model.student.id_encoder, f"{arm}: model does not expose the trained public adapter")
                _require(not model.training and all(not parameter.requires_grad for parameter in model.parameters()),
                         f"{arm}: scorer model is not frozen/eval")
                state_before = _student_state_sha(model); side = _side_tensor_for_record(torch=torch, record=record)
                support_identities = [trained_trial_identity(adapter=adapter, torch=torch,
                    activity=np.asarray(support["activities"][index], dtype=np.float32), side_tensor=side,
                    expected_channels=expected_channels)
                    for index in selected]
                pools = {"FIXED30": memory.PostFusionUniformIdentityPool(support_identities=support_identities, capacity=30),
                         "UNCAPPED": memory.PostFusionUniformIdentityPool(support_identities=support_identities, capacity=None)}
                state: dict[str, dict[str, Any]] = {law: {"pred": [], "target": [], "starts": [], "trace": []}
                                                     for law in plan.LAWS}
                support_identity_order_sha256 = plan.sha256_bytes(plan.canonical_json(
                    [_identity_sha256(item) for item in support_identities]))
                for law in plan.LAWS:
                    _require(_assert_support_retained(pool=pools[law], support_identities=support_identities)
                             == [_identity_sha256(item) for item in support_identities],
                             f"{law}: initial support order/content drift")
                query_identity_digests: list[str] = []
                for query in query_rows:
                    metric_starts = np.asarray(query["metric_starts"], dtype=np.int64)
                    if metric_starts.size:
                        indices = metric_starts[:, None] + np.arange(50, dtype=np.int64)[None, :]
                        windows = np.ascontiguousarray(neural[indices], dtype=np.float32)
                        targets = np.ascontiguousarray(target_all[metric_starts + 49], dtype=np.float32)
                        # Decode both laws before one shared trial-identity commit.
                        law_predictions: dict[str, np.ndarray] = {}
                        for law in plan.LAWS:
                            identity = pools[law].deployed()
                            value = (_decode_identity_cpu(torch=torch, model=model, neural_windows=windows,
                                                          identity=identity, batch_size=plan.CPU_DECODE_BATCH_SIZE)
                                     if decode is None else np.ascontiguousarray(decode(model, windows, identity), dtype=np.float32))
                            _require(value.shape == targets.shape and np.isfinite(value).all(), "PF decode topology drift")
                            law_predictions[law] = value
                            state[law]["pred"].append(value); state[law]["target"].append(targets); state[law]["starts"].append(metric_starts)
                        if pools["FIXED30"].evictions == 0:
                            _require(np.array_equal(law_predictions["FIXED30"], law_predictions["UNCAPPED"]),
                                     "laws diverged before first FIFO eviction")
                    identity = trained_trial_identity(adapter=adapter, torch=torch,
                        activity=np.asarray(query["activity"], dtype=np.float32), side_tensor=side,
                        expected_channels=expected_channels)
                    query_identity_digests.append(_array_sha256(identity.detach().cpu().numpy()))
                    for law in plan.LAWS:
                        pool = pools[law]; before_count = pool.pool_count; before_evictions = pool.evictions
                        pool.commit(identity)
                        state[law]["trace"].append({"trial_id": str(query["trial_id"]),
                            "position": int(query["position"]), "activity_sha256": _array_sha256(query["activity"]),
                            "metric_starts_sha256": _array_sha256(metric_starts), "members_before": int(before_count),
                            "members_after": int(pool.pool_count), "evictions_before": int(before_evictions),
                            "evictions_after": int(pool.evictions)})
                        retained = _assert_support_retained(pool=pool, support_identities=support_identities)
                        _require(plan.sha256_bytes(plan.canonical_json(retained)) == support_identity_order_sha256,
                                 "support member ordering/content receipt drift")
                for law in plan.LAWS:
                    pool = pools[law]; prediction = np.ascontiguousarray(np.concatenate(state[law]["pred"]), dtype=np.float32)
                    target = np.ascontiguousarray(np.concatenate(state[law]["target"]), dtype=np.float32)
                    starts = np.ascontiguousarray(np.concatenate(state[law]["starts"]), dtype=np.int64)
                    _require(str(record["query_starts_sha256"]) == _governed_array_sha256(starts)
                             and str(record["target_sha256"]) == _governed_array_sha256(target), "row escaped input authority")
                    after = _student_state_sha(model); _require(after == state_before, f"{arm}: scoring mutated model state")
                    trace = state[law]["trace"]
                    rows_out.append({"arm": arm, "memory_law": law, "surface": surface, "session": record["session"],
                        "budget": plan.BUDGET, "input_authority_key": record["key"], "r2": _r2(target, prediction),
                        "prediction_sha256": _governed_array_sha256(prediction), "target_sha256": _governed_array_sha256(target),
                        "window_count": int(starts.size), "initial_members": int(len(support_identities)),
                        "final_members": int(pool.pool_count), "commits": int(pool.completed_count),
                        "evictions": int(pool.evictions), "causal_trace_sha256": plan.sha256_bytes(plan.canonical_json(trace)),
                        "query_identity_trace_sha256": plan.sha256_bytes(plan.canonical_json(query_identity_digests)),
                        "support_never_evicted": True,
                        "support_retained_object_order_exact": True,
                        "support_identity_order_sha256": support_identity_order_sha256,
                        "model_state_before_sha256": state_before,
                        "model_state_after_sha256": after, "parameter_updates": 0, "target_updates": 0,
                        "pooled_comparator_key": record["key"]})
                fixed, uncapped = rows_out[-2], rows_out[-1]
                if fixed["evictions"] == 0:
                    _require(fixed["prediction_sha256"] == uncapped["prediction_sha256"]
                             and fixed["r2"] == uncapped["r2"], "no-eviction laws must have exact full-row parity")
    expected_rows = plan.EXPECTED_ROWS if selected_keys is None else len(selected_keys) * len(plan.ARMS) * len(plan.LAWS)
    _require(len(rows_out) == expected_rows, "PF coordinator row cardinality drift")
    return rows_out


def score_records_from_materialized(*, prepared: Mapping[str, Any], materialized: Mapping[str, Any],
                                    record_keys: Sequence[str], expected_channels: int = 96) -> list[dict[str, Any]]:
    """Route-owned per-record primitive used by the same-attempt smoke gate.

    This only partitions an already materialized canonical grid.  It does not
    reconstruct a model/data stack, reset an identity stream, or duplicate a
    scored input; the full coordinator remains the single implementation.
    """
    return score_78_rows_from_materialized(prepared=prepared, materialized=materialized,
                                           expected_channels=expected_channels, record_keys=record_keys)


def causal_rollout(*, pool: Any, support_identities: Sequence[Any], query_activities: Sequence[np.ndarray],
                   decode: Callable[[Any], np.ndarray], make_identity: Callable[[np.ndarray], Any]) -> dict[str, object]:
    """Shared FIXED30/UNCAPPED decode-before-commit law over causal rows."""
    del support_identities  # pool is already initialized with exactly these members
    predictions = []; trace = []
    for position, activity in enumerate(query_activities):
        before = pool.deployed(); predictions.append(np.asarray(decode(before), dtype=np.float32))
        member_before = int(pool.pool_count)
        pool.commit(make_identity(np.asarray(activity, dtype=np.float32)))
        trace.append((position, member_before, int(pool.pool_count), int(pool.evictions)))
    return {"predictions": predictions, "trace": trace, "commits": len(trace),
            "initial_members": int(trace[0][1]) if trace else int(pool.pool_count),
            "final_members": int(pool.pool_count), "evictions": int(pool.evictions)}


def coordinate_rows(*, input_records: dict[str, dict[str, Any]], arms: Sequence[str],
                    rollout: Callable[[str, str, dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """One materialized authority record feeds its six arm/law result rows."""
    _require(len(input_records) == 13, "exactly 13 shared input records required")
    rows: list[dict[str, Any]] = []
    for surface in plan.SURFACES:
        records = [value for value in input_records.values() if value.get("surface") == surface]
        _require(len(records) == plan.ROSTER_SIZES[surface], "surface input roster drift")
        for record in sorted(records, key=lambda item: str(item["session"])):
            for arm in arms:
                for law in plan.LAWS:
                    row = dict(rollout(arm, law, record))
                    required = {"prediction_sha256", "target_sha256", "window_count", "r2", "initial_members",
                                "final_members", "commits", "evictions", "causal_trace_sha256",
                                "model_state_before_sha256", "model_state_after_sha256"}
                    _require(required <= set(row), "governed row evidence missing")
                    _require(row["model_state_before_sha256"] == row["model_state_after_sha256"], "scoring changed model state")
                    row.update({"arm": arm, "memory_law": law, "surface": surface, "session": record["session"],
                                "budget": plan.BUDGET, "input_authority_key": record["key"], "parameter_updates": 0,
                                "target_updates": 0})
                    rows.append(row)
    return rows
