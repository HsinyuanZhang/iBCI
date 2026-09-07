#!/usr/bin/env python3
"""Run the reviewed A12 coupled-B3S descriptive attention forward on CPU.

This executable intentionally has a narrow, fail-closed surface.  Its only
real-data operation is a paired comparison of independently trained canonical
SUA B3S T4 and v10 Z4 checkpoints for one pre-declared seed/epoch.  It never
trains, runs backward, evaluates a formal-test session, zeroes an identity,
or replaces a carrier within a checkpoint.  The shared session loader does
load and standardize query behavior for its evaluator record, but A12 never
passes it to the model or uses it for attention metrics, selection, or
updates.  Attention is descriptive: the contrast is between frozen T4 and Z4
checkpoints, not a causal intervention.

Without ``--run-forward`` the command is metadata-only.  That dry run verifies
the immutable official A12 preflight and prints the exact paired checkpoint
plan without importing Torch, NWB, or the model/data loader stack.
"""
from __future__ import annotations

# These environment settings must precede every project import.  A real launch
# should additionally invoke the interpreter as ``PYTHONNOUSERSITE=1 python``;
# the runtime checker below rejects a process that did not do so.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = REPO_ROOT / "sua_exploration"
SCRIPTS_ROOT = SUA_ROOT / "scripts"
STREAMING_ROOT = REPO_ROOT / "streaming_calibration_exp"
if str(SUA_ROOT) not in sys.path:
    sys.path.insert(0, str(SUA_ROOT))

from mc_maze import a12_descriptive_attention_audit as core  # noqa: E402


FORWARD_STATUS = "COMPLETED_DESCRIPTIVE_CPU_FORWARD_ONLY"
CARRIER_CONTRAST = "paired_frozen_t4_vs_z4_checkpoints_no_within_forward_intervention"
FORWARD_REVIEW_ENV = "A12_ROOT_REVIEWED_CPU_FORWARD"
FORWARD_REVIEW_TOKEN = "I_AUTHORIZE_A12_DESCRIPTIVE_FORWARD"


class A12RunnerError(RuntimeError):
    """Raised when a real A12 forward cannot prove its provenance boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A12RunnerError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_sha256(payload: Mapping[str, Any]) -> str:
    return core.sha256_bytes(core.canonical_json_bytes(payload))


def _hash_ints(values: Sequence[int]) -> str:
    return _json_sha256({"values": [int(value) for value in values]})


def _sha256_numpy_array(value: Any) -> str:
    """Hash a NumPy array's dtype/shape/bytes without serializing a model."""

    normalized = value.astype(value.dtype, copy=False)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise A12RunnerError(f"{label} must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A12RunnerError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise A12RunnerError(f"{label} must be a JSON object: {path}")
    return payload


def load_official_metadata_preflight(path: Path) -> tuple[dict[str, Any], str]:
    """Read the immutable preflight and prove current implementation bindings."""

    resolved = path.expanduser().resolve()
    expected = (REPO_ROOT / core.CANONICAL_PREFLIGHT_RELATIVE_PATH).resolve()
    _require(
        resolved == expected,
        "A12 forward accepts only the canonical official metadata preflight path: "
        f"{expected}",
    )
    payload = core.load_verified_immutable_json(resolved, label="A12 official metadata preflight")
    core.validate_metadata_preflight_receipt(payload, require_operational_status=True)
    return payload, core.sha256_file(resolved)


def build_dry_run_plan(preflight: Mapping[str, Any], *, preflight_path: Path, seed: int, epoch: int) -> dict[str, Any]:
    """Build a no-Torch, no-NWB plan for one fixed paired forward receipt."""

    if seed not in core.SEEDS:
        raise A12RunnerError(f"seed outside canonical A12 matrix: {seed}")
    if epoch not in core.EPOCH_WINDOW:
        raise A12RunnerError(f"epoch outside canonical A12 window: {epoch}")
    pair = preflight["pairs"][str(seed)]
    sessions = tuple(pair["pairing"]["validation_sessions"])
    if sessions != core.DEFAULT_VALIDATION_SESSIONS:
        raise A12RunnerError("official preflight validation roster drift")
    checkpoints = {
        arm: core.expected_pair_checkpoint(preflight, arm=arm, seed=seed, epoch=epoch)
        for arm in ("t4", "z4")
    }
    return {
        "schema_version": core.SCHEMA_VERSION,
        "kind": "a12_descriptive_attention_forward_dry_run",
        "status": "DRY_RUN_ONLY__NO_TORCH_NWB_OR_FORWARD",
        "official_metadata_preflight_path": str(preflight_path.expanduser().resolve()),
        "official_metadata_preflight_sha256": core.sha256_file(preflight_path.expanduser().resolve()),
        "seed": seed,
        "epoch": epoch,
        "arms": {
            arm: {
                "checkpoint_path": checkpoints[arm]["checkpoint_path"],
                "checkpoint_sha256": checkpoints[arm]["checkpoint_sha256_observed"],
                "run_metadata_path": pair["arms"][arm]["metadata"]["metadata_path"],
                "run_metadata_sha256": pair["arms"][arm]["metadata"]["metadata_sha256"],
            }
            for arm in ("t4", "z4")
        },
        "sessions": list(sessions),
        "cpu_forward_batch_contract": {
            **core.cpu_forward_batch_contract(),
            "contract_sha256": core.cpu_forward_batch_contract_sha256(),
        },
        "scope": {
            "metadata_only": True,
            "torch_imported": False,
            "nwb_opened": False,
            "checkpoint_deserialized": False,
            "forward_run": False,
            "gpu_used": False,
            "shared_session_loader_invoked": False,
            "forward_query_behavior_contract": dict(core.QUERY_BEHAVIOR_FORWARD_CONTRACT),
        },
    }


def _prepare_actual_b3s_imports() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Import only the real streaming B3S and T4-aware data routes.

    This function is deliberately unreachable from dry-run mode.  It rejects
    any prior ``src`` import that did not originate in
    ``streaming_calibration_exp`` rather than silently accepting a similarly
    named legacy package such as ``SPINT-main``.
    """

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "A12 requires CUDA_VISIBLE_DEVICES=''")
    _require(os.environ.get("PYTHONNOUSERSITE") == "1", "A12 requires PYTHONNOUSERSITE=1")
    if str(STREAMING_ROOT) in sys.path:
        sys.path.remove(str(STREAMING_ROOT))
    sys.path.insert(0, str(STREAMING_ROOT))
    for location in (str(SCRIPTS_ROOT), str(SUA_ROOT)):
        if location not in sys.path:
            sys.path.insert(1, location)

    existing = sys.modules.get("src.models.components.spint")
    if existing is not None:
        existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        _require(_is_under(existing_file, STREAMING_ROOT), "A12 refuses a non-streaming_calibration_exp src.models.components.spint import")
    spint_module = importlib.import_module("src.models.components.spint")
    spint_path = Path(str(getattr(spint_module, "__file__", ""))).resolve()
    expected_spint = (STREAMING_ROOT / "src/models/components/spint.py").resolve()
    _require(spint_path == expected_spint, f"A12 loaded wrong decoder module: {spint_path}")

    import torch
    from torch.utils.data import DataLoader, Dataset

    from mc_maze import decoder_attention_diagnostic as diagnostic
    from mc_maze.multisession_datamodule import fit_behavior_stats
    from mc_maze.unit_side_features import side_feature_stats_sha256
    from eval_adaptation_dandi688 import (
        attach_side_features,
        load_session_with_trials,
        load_side_feature_stats_for_run_metadata,
        make_subset_dataset,
    )
    from select_gradient_free_protocol_dandi688 import load_frozen_model

    _require(_is_under(Path(torch.__file__).resolve(), Path(sys.prefix)), "A12 Torch import is outside the active interpreter prefix")
    _require(not torch.cuda.is_available(), "A12 refuses a CUDA-visible Torch runtime")
    return (
        torch,
        DataLoader,
        Dataset,
        diagnostic,
        fit_behavior_stats,
        side_feature_stats_sha256,
        (attach_side_features, load_session_with_trials, load_side_feature_stats_for_run_metadata, make_subset_dataset),
        load_frozen_model,
    )


def _inputs_only_dataset(base_dataset: Any, dataset_base: Any, torch: Any) -> Any:
    """Use the actual MCMaze dataset's B3S fields without querying behavior.

    The shared evaluator's normal dataset also returns behavior targets.  A12
    constructs its query windows through that same dataset after the shared
    session loader has loaded and standardized behavior.  This wrapper itself
    intentionally reads only neural, M30 calibration, and side tensors.  Thus
    no query velocity tensor is passed to or used by the diagnostic.
    """

    class InputsOnlyDataset(dataset_base):
        def __init__(self, dataset: Any) -> None:
            self.dataset = dataset
            if dataset.side_features is None:
                raise A12RunnerError("canonical B3S T4/Z4 dataset lacks [N,4] side features")
            if dataset.electrode_ids is not None:
                raise A12RunnerError("canonical T4/Z4 B3S audit does not accept electrode-id side paths")

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
            start = int(self.dataset.valid_starts[index])
            end = start + int(self.dataset.window_size)
            # Crucially no access to dataset.behavior occurs in this path.
            neural = torch.from_numpy(self.dataset.neural[start:end]).float()
            calibration = self.dataset.calib_trials
            side = self.dataset.side_features
            return neural, calibration, side

    return InputsOnlyDataset(base_dataset)


def _session_provenance_hash(rec: Mapping[str, Any], *, nwb_path: Path, expected_units: int) -> str:
    return _json_sha256(
        {
            "name": str(rec["name"]),
            "nwb_path": str(nwb_path.resolve()),
            "signal_view": str(rec["signal_view"]),
            "n_units": int(rec["n_units"]),
            "source_unit_count": int(rec["source_unit_count"]),
            "preflight_validation_unit_count": int(expected_units),
        }
    )


def _query_trial_hash(trials: Sequence[Mapping[str, Any]]) -> str:
    """Hash only trial identity/boundary fields, never target/velocity labels."""

    return _json_sha256(
        {
            "trials": [
                {
                    "start": int(trial["start"]),
                    "stop": int(trial["stop"]),
                    "trial_index": int(trial.get("trial_index", index)),
                }
                for index, trial in enumerate(trials)
            ]
        }
    )


def _input_digest_update(digest: Any, *, neural: Any, calibration: Any, side: Any, diagnostic: Any) -> None:
    for label, tensor in (("neural", neural), ("calibration", calibration), ("side", side)):
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(diagnostic.tensor_sha256(tensor).encode("ascii"))
        digest.update(b"\0")


def _normalizer_provenance(
    *,
    arm: str,
    run_metadata: Mapping[str, Any],
    source_train_files: Sequence[Path],
    cache_dir: Path | None,
    load_side_feature_stats_for_run_metadata: Any,
    side_feature_stats_sha256: Any,
    fit_behavior_stats: Any,
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[Any, Any]]:
    """Recover only train-fitted normalizers and reject target fitting/drift."""

    side = load_side_feature_stats_for_run_metadata(
        dict(run_metadata), list(source_train_files), cache_dir
    )
    _require(side is not None, f"{arm}: B3S run metadata did not resolve side-feature normalizer")
    group, waveform_group, pool_size, permutation_seed, side_mean, side_std = side
    _require(group == arm, f"{arm}: side feature group drift: {group!r}")
    _require(waveform_group == "t4", f"{arm}: normalizer must derive from ordinary t4")
    _require(int(pool_size) == core.M30, f"{arm}: side feature M30 drift")
    _require(permutation_seed is None, f"{arm}: canonical side feature permutation drift")
    if arm == "z4":
        descriptor_contract = (run_metadata.get("side_features") or {}).get("descriptor_contract")
        _require(
            descriptor_contract == core.EXPECTED_Z4_DESCRIPTOR_CONTRACT,
            "z4: descriptor must zero the carrier only after ordinary T4 standardization",
        )
    side_digest = side_feature_stats_sha256(side_mean, side_std)
    _require(
        side_digest == core.EXPECTED_NORMALIZER_SHA256,
        f"{arm}: train-only T4 side normalizer SHA drift",
    )
    behavior_mean, behavior_std = fit_behavior_stats(
        list(source_train_files), core.BIN_SIZE_MS, cache_dir=cache_dir
    )
    return (
        {
            "source_train_only": True,
            "target_or_validation_refit_performed": False,
            "normalization_base_feature_group": "t4",
            "side_feature_group": arm,
            "side_feature_normalizer_sha256": side_digest,
            "behavior_train_stats_sha256": _json_sha256(
                {
                    "mean_sha256": _sha256_numpy_array(behavior_mean),
                    "std_sha256": _sha256_numpy_array(behavior_std),
                }
            ),
            "support_direction_labels_used_for_t4_carrier": True,
            **core.QUERY_BEHAVIOR_FORWARD_CONTRACT,
        },
        (group, waveform_group, pool_size, permutation_seed, side_mean, side_std),
        (behavior_mean, behavior_std),
    )


def _run_arm_forward(
    *,
    arm: str,
    seed: int,
    epoch: int,
    preflight: Mapping[str, Any],
    batch_size: int,
    runtime: tuple[Any, ...],
) -> dict[str, Any]:
    """Run one frozen canonical arm and bind every source/query/window fact."""

    batch_size = core.require_cpu_forward_batch_size(
        batch_size,
        label=f"A12 {arm}/s{seed}/e{epoch} CPU forward batch size",
    )
    (
        torch,
        DataLoader,
        Dataset,
        diagnostic,
        fit_behavior_stats,
        side_feature_stats_sha256,
        data_helpers,
        load_frozen_model,
    ) = runtime
    attach_side_features, load_session_with_trials, load_side_feature_stats_for_run_metadata, make_subset_dataset = data_helpers
    pair = preflight["pairs"][str(seed)]
    arm_preflight = pair["arms"][arm]
    metadata_preflight = arm_preflight["metadata"]
    result_preflight = arm_preflight["result"]
    checkpoint_binding = core.expected_pair_checkpoint(preflight, arm=arm, seed=seed, epoch=epoch)
    checkpoint_path = Path(str(checkpoint_binding["checkpoint_path"])).resolve()
    checkpoint_sha = core.sha256_file(checkpoint_path)
    _require(
        checkpoint_sha == checkpoint_binding["checkpoint_sha256_declared"] == checkpoint_binding["checkpoint_sha256_observed"],
        f"{arm}/s{seed}/e{epoch}: checkpoint SHA drift before deserialization",
    )
    metadata_path = Path(str(metadata_preflight["metadata_path"])).resolve()
    _require(
        core.sha256_file(metadata_path) == metadata_preflight["metadata_sha256"],
        f"{arm}/s{seed}: run metadata SHA drift",
    )
    run_metadata = _load_json(metadata_path, label=f"{arm} run metadata")
    _require(run_metadata.get("variant") == "B3S", f"{arm}: only B3S is authorized")
    _require(run_metadata.get("signal_view") == "sua", f"{arm}: only sorted SUA is authorized")
    _require(run_metadata.get("seed") == seed, f"{arm}: metadata seed drift")
    _require(run_metadata.get("held_out_test_evaluated") is False, f"{arm}: held-out test provenance drift")
    _require((run_metadata.get("session_files") or {}).get("test") == [], f"{arm}: test files attached")

    train_files = [Path(value).resolve() for value in metadata_preflight["session_files"]["train"]]
    validation_files = [Path(value).resolve() for value in metadata_preflight["session_files"]["val"]]
    sessions = tuple(pair["pairing"]["validation_sessions"])
    diagnostic.assert_no_sealed_sessions(sessions)
    _require(sessions == core.DEFAULT_VALIDATION_SESSIONS, f"{arm}: validation roster drift")
    _require(
        tuple(path.name.replace("_behavior+ecephys.nwb", "") for path in validation_files) == sessions,
        f"{arm}: validation file ordering drift",
    )
    raw_cache_dir = run_metadata.get("cache_dir")
    cache_dir = Path(str(raw_cache_dir)).resolve() if isinstance(raw_cache_dir, str) and raw_cache_dir else None
    normalizer, side_args, behavior_stats = _normalizer_provenance(
        arm=arm,
        run_metadata=run_metadata,
        source_train_files=train_files,
        cache_dir=cache_dir,
        load_side_feature_stats_for_run_metadata=load_side_feature_stats_for_run_metadata,
        side_feature_stats_sha256=side_feature_stats_sha256,
        fit_behavior_stats=fit_behavior_stats,
    )
    group, waveform_group, pool_size, permutation_seed, side_mean, side_std = side_args
    behavior_mean, behavior_std = behavior_stats
    teacher_path = Path(str(preflight["teacher"]["path"])).resolve()
    _require(core.sha256_file(teacher_path) == preflight["teacher"]["sha256_observed"], "teacher SHA drift before model load")
    model = load_frozen_model(
        checkpoint_path,
        teacher_path,
        "B3S",
        torch.device("cpu"),
        identity_mode="calibrated",
    )
    model.eval()
    student = diagnostic.resolve_b3s_student(model)
    _require(not model.training and not student.training, f"{arm}: model/student must be eval")
    _require(not any(parameter.requires_grad for parameter in model.parameters()), f"{arm}: frozen checkpoint has trainable parameters")
    model_state_before = diagnostic.model_state_sha256(model)

    per_session: dict[str, dict[str, Any]] = {}
    expected_unit_counts = pair["pairing"]["validation_unit_counts"]
    for session, nwb_path in zip(sessions, validation_files, strict=True):
        _require(not nwb_path.is_symlink() and nwb_path.is_file(), f"{arm}/{session}: canonical NWB path missing")
        rec = load_session_with_trials(
            nwb_path,
            bin_size_ms=core.BIN_SIZE_MS,
            window_size=core.WINDOW_SIZE,
            calib_n=core.M30,
            max_trial_length=core.TRIAL_LENGTH,
            pad_value=-1.0,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            trial_result_filter="R",
            cache_dir=cache_dir,
            signal_view="sua",
        )
        _require(rec["name"] == session, f"{arm}/{session}: loader session name drift")
        _require(int(rec["n_units"]) == int(expected_unit_counts[session]), f"{arm}/{session}: N drift from paired metadata")
        _require(str(rec["signal_view"]) == "sua", f"{arm}/{session}: non-SUA loader result")
        _require(len(rec["trials"]) > core.M30, f"{arm}/{session}: no query trial after M30 support")
        _require(tuple(rec["calib_trials"].shape) == (core.M30, core.TRIAL_LENGTH, int(rec["n_units"])),
                 f"{arm}/{session}: M30 calibration tensor shape drift")
        rec = attach_side_features(
            rec,
            nwb_path,
            side_feature_group=group,
            waveform_feature_group=waveform_group,
            pool_size=int(pool_size),
            permutation_seed=permutation_seed,
            mean=side_mean,
            std=side_std,
            cache_dir=cache_dir,
        )
        _require(rec.get("side_features") is not None, f"{arm}/{session}: B3S side features missing")
        _require(tuple(rec["side_features"].shape) == (int(rec["n_units"]), 4), f"{arm}/{session}: side shape drift")
        # ``load_session_with_trials`` has loaded and standardized behavior
        # (including query velocity) for historic evaluator compatibility.
        # From this point A12 uses only trial boundaries, neural windows, M30
        # calibration spikes, and side features: query behavior is neither
        # accessed/passed to the decoder nor used by an attention metric or a
        # selection/update path.
        query_trials = list(rec["trials"][core.M30 :])
        query_dataset = make_subset_dataset(rec, query_trials, session)
        _require(len(query_dataset) > 0, f"{arm}/{session}: query trials have no W50 windows")
        query_starts = [int(value) for value in query_dataset.valid_starts.tolist()]
        inputs_only = _inputs_only_dataset(query_dataset, Dataset, torch)
        loader = DataLoader(inputs_only, batch_size=batch_size, shuffle=False, num_workers=0)
        # Keep only sufficient statistics for the descriptive metrics.  A
        # complete session can contain thousands of query windows; retaining
        # every Q/K/V and attention tensor here previously caused the CPU
        # process to grow until it was killed by the OOM killer before a
        # receipt could be written.
        summary_accumulator = diagnostic.AttentionSummaryAccumulator(
            expected_num_windows=len(query_dataset),
        )
        input_digest = hashlib.sha256()
        input_shapes: dict[str, int] = {}
        windows_seen = 0
        for neural, calibration, side in loader:
            diagnostic.assert_b3s_input_shapes(neural, calibration, side)
            _input_digest_update(input_digest, neural=neural, calibration=calibration, side=side, diagnostic=diagnostic)
            shape_key = json.dumps(
                {
                    "neural": list(neural.shape),
                    "calibration": list(calibration.shape),
                    "side": list(side.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            input_shapes[shape_key] = input_shapes.get(shape_key, 0) + 1
            plain_prediction, _ = diagnostic.plain_b3s_forward(model, neural, calibration, side)
            captured = diagnostic.forward_b3s_with_capture(model, neural, calibration, side)
            diagnostic.assert_output_parity(plain_prediction, captured.prediction)
            summary_accumulator.update(captured.layers)
            windows_seen += int(neural.shape[0])
        _require(windows_seen == len(query_dataset), f"{arm}/{session}: dataloader window accounting drift")
        summary = summary_accumulator.finalize()
        per_session[session] = {
            "input_shape_contract": diagnostic.INPUT_SHAPE_CONTRACT,
            "input_shapes": [
                {"shape": json.loads(shape), "batches": count}
                for shape, count in sorted(input_shapes.items())
            ],
            "session_provenance_sha256": _session_provenance_hash(
                rec, nwb_path=nwb_path, expected_units=int(expected_unit_counts[session])
            ),
            "support_trial_index_sha256": _query_trial_hash(rec["trials"][: core.M30]),
            "query_trial_index_sha256": _query_trial_hash(query_trials),
            "query_window_start_sha256": _hash_ints(query_starts),
            "model_input_sha256": input_digest.hexdigest(),
            "num_query_trials": len(query_trials),
            "num_query_windows": windows_seen,
            "qkv_order": ["Q", "K", "V"],
            "value_projection_used_for_head_contribution": True,
            "capture_output_parity_exact": True,
            "attention_summary": summary,
        }
    model_state_after = diagnostic.model_state_sha256(model)
    _require(model_state_before == model_state_after, f"{arm}: model state drifted during no-grad forward")
    return {
        "arm": arm,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": str(metadata_preflight["metadata_sha256"]),
        "normalizer": normalizer,
        "model_state_sha256_pre": model_state_before,
        "model_state_sha256_post": model_state_after,
        "model_state_unchanged": True,
        "qkv_order": ["Q", "K", "V"],
        "capture_output_parity_exact": True,
        "sessions": per_session,
        "preflight_result_sha256": result_preflight["result_sha256"],
    }


def run_pair_forward(
    *,
    preflight_path: Path,
    preflight: Mapping[str, Any],
    preflight_sha256: str,
    seed: int,
    epoch: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run both independently trained frozen arms and construct a receipt."""

    # Reject a programmatic caller's override before importing Torch or any
    # real loader/model helper.  The CLI has the same check through
    # ``choices=(512,)``; this closes the direct-call bypass as well.
    batch_size = core.require_cpu_forward_batch_size(batch_size)
    runtime = _prepare_actual_b3s_imports()
    torch = runtime[0]
    diagnostic = runtime[3]
    diagnostic.assert_cpu_only()
    pair = preflight["pairs"][str(seed)]
    arms = {
        arm: _run_arm_forward(
            arm=arm,
            seed=seed,
            epoch=epoch,
            preflight=preflight,
            batch_size=batch_size,
            runtime=runtime,
        )
        for arm in ("t4", "z4")
    }
    _require(arms["t4"]["checkpoint_sha256"] != arms["z4"]["checkpoint_sha256"],
             "A12 requires independently trained T4/Z4 checkpoint bytes")
    for session in core.DEFAULT_VALIDATION_SESSIONS:
        _require(
            arms["t4"]["sessions"][session]["session_provenance_sha256"]
            == arms["z4"]["sessions"][session]["session_provenance_sha256"],
            f"{session}: T4/Z4 same-unit provenance drift",
        )
        _require(
            arms["t4"]["sessions"][session]["support_trial_index_sha256"]
            == arms["z4"]["sessions"][session]["support_trial_index_sha256"],
            f"{session}: T4/Z4 M30 support trial provenance drift",
        )
        _require(
            arms["t4"]["sessions"][session]["query_trial_index_sha256"]
            == arms["z4"]["sessions"][session]["query_trial_index_sha256"],
            f"{session}: T4/Z4 query trial provenance drift",
        )
        _require(
            arms["t4"]["sessions"][session]["query_window_start_sha256"]
            == arms["z4"]["sessions"][session]["query_window_start_sha256"],
            f"{session}: T4/Z4 query window provenance drift",
        )
    receipt = {
        "schema_version": core.SCHEMA_VERSION,
        "kind": core.FORWARD_KIND,
        "status": FORWARD_STATUS,
        "generated_at_utc": _utc_now(),
        "official_metadata_preflight": {
            "path": str(preflight_path.expanduser().resolve()),
            "sha256": preflight_sha256,
            "receipt_body_sha256": preflight["receipt_body_sha256"],
        },
        "seed": seed,
        "epoch": epoch,
        "canonical_scope": {
            **core.FORWARD_CANONICAL_SCOPE_CONTRACT,
            "historical_t4_reference_qualification": preflight["canonical_scope"]["historical_t4_reference_qualification"],
        },
        "execution_scope": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "python_no_user_site": "1",
            "gpu_used": False,
            "training_performed": False,
            "backward_gradients": False,
            "decoder_weight_updates": False,
            "checkpoint_weight_updates": False,
            "sealed_formal_test_sessions_opened": False,
            **core.QUERY_BEHAVIOR_FORWARD_CONTRACT,
            "support_direction_labels_used_for_t4_carrier": True,
            "whole_identity_zeroing_performed": False,
            "carrier_forward_ablation_performed": False,
            "same_checkpoint_carrier_ablation_performed": False,
            "descriptive_not_causal": True,
            "output_capture_parity_checked": True,
            "cpu_forward_batch_size": batch_size,
            "cpu_forward_batch_contract_sha256": core.cpu_forward_batch_contract_sha256(),
            "torch_version": str(torch.__version__),
            "torch_path": str(Path(torch.__file__).resolve()),
        },
        "carrier_contrast": CARRIER_CONTRAST,
        "pairing": {
            "same_unit_paired": True,
            "sessions": list(core.DEFAULT_VALIDATION_SESSIONS),
            "validation_session_files": list(pair["pairing"]["validation_session_files"]),
            "validation_unit_counts": dict(pair["pairing"]["validation_unit_counts"]),
            "proof": pair["pairing"]["proof"],
            "support_query_trial_disjoint": True,
        },
        "arms": arms,
    }
    core.validate_pair_forward_receipt(receipt, preflight=preflight, preflight_path=preflight_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-metadata-preflight", type=Path, required=True)
    parser.add_argument("--seed", choices=core.SEEDS, type=int, required=True)
    parser.add_argument("--epoch", choices=core.EPOCH_WINDOW, type=int, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(core.CPU_FORWARD_BATCH_SIZE,),
        default=core.CPU_FORWARD_BATCH_SIZE,
        help=(
            "frozen CPU-forward DataLoader batch size; only 512 is accepted "
            "and the value is bound in the preflight and receipt"
        ),
    )
    parser.add_argument("--output", type=Path, help="write-once receipt path; required with --run-forward")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="metadata-only default; no Torch/NWB/model import")
    mode.add_argument("--run-forward", action="store_true", help="explicitly run the reviewed CPU-only real forward")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        preflight_path = args.official_metadata_preflight.expanduser().resolve()
        preflight, preflight_sha = load_official_metadata_preflight(preflight_path)
        if not args.run_forward:
            # Explicitly recompute the no-write plan after validation, so its
            # preflight SHA cannot be supplied by a caller-controlled string.
            plan = build_dry_run_plan(preflight, preflight_path=preflight_path, seed=args.seed, epoch=args.epoch)
            _require(plan["official_metadata_preflight_sha256"] == preflight_sha, "preflight SHA changed during dry run")
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        _require(args.output is not None, "--output is required with --run-forward")
        _require(
            os.environ.get(FORWARD_REVIEW_ENV) == FORWARD_REVIEW_TOKEN,
            "real A12 forward remains blocked until root review is explicitly recorded "
            f"via {FORWARD_REVIEW_ENV}={FORWARD_REVIEW_TOKEN}",
        )
        receipt = run_pair_forward(
            preflight_path=preflight_path,
            preflight=preflight,
            preflight_sha256=preflight_sha,
            seed=args.seed,
            epoch=args.epoch,
            batch_size=args.batch_size,
        )
        artifact = core.write_immutable_json(args.output, receipt, label="A12 paired descriptive forward receipt")
        print(json.dumps({"status": receipt["status"], "receipt": artifact}, indent=2, sort_keys=True))
    except (A12RunnerError, core.A12AuditError, FileExistsError, OSError, ValueError) as exc:
        print(f"FAIL_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
